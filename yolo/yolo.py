import cv2
import numpy as np
import collections
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from ultralytics import YOLO # 8.4.34


def get_affine_transform(cam_info_src, cam_info_dst):
    """
    Calculates a 2x3 Affine Transformation matrix to map the Source camera (Depth)
    to the Target camera (RGB) coordinate system.
    """
    k_src = np.array(cam_info_src.k).reshape(3, 3)
    k_dst = np.array(cam_info_dst.k).reshape(3, 3)

    fx_src, fy_src = k_src[0, 0], k_src[1, 1]
    cx_src, cy_src = k_src[0, 2], k_src[1, 2]

    fx_dst, fy_dst = k_dst[0, 0], k_dst[1, 1]
    cx_dst, cy_dst = k_dst[0, 2], k_dst[1, 2]

    scale_x = fx_dst / fx_src
    scale_y = fy_dst / fy_src

    shift_x = cx_dst - (cx_src * scale_x)
    shift_y = cy_dst - (cy_src * scale_y)

    matrix = np.float32([
        [scale_x, 0, shift_x],
        [0, scale_y, shift_y]
    ])
    return matrix


def run_semantic_depth_overlay(bag_dir_path, yolo_model_path, speed=0.5):
    """
    Visualizes aligned RGB-D data WITH YOLOv8 Semantic Segmentation & Depth querying.
    """
    RGB_TOPIC = '/camera/color/image_raw'
    RGB_INFO = '/camera/color/camera_info'
    DEPTH_TOPIC = '/camera/depth/image_rect_raw'
    DEPTH_INFO = '/camera/depth/camera_info'

    # Initialize YOLO model
    print(f"Loading YOLO semantic segmentation model: {yolo_model_path} ...")
    model = YOLO(yolo_model_path)

    # Generate distinct colors for each class
    num_classes = len(model.names)
    np.random.seed(42)
    class_colors = {i: tuple(map(int, np.random.randint(50, 255, 3))) for i in range(num_classes)}

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    rgb_queue = collections.deque()
    depth_queue = collections.deque()

    rgb_cam_info = None
    depth_cam_info = None
    affine_matrix = None
    last_render_time = None

    print(f"Launching Semantic RGB-D Overlay (Speed: {speed}x)... Press 'q' to exit")

    with AnyReader([Path(bag_dir_path)], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections
                       if x.topic in [RGB_TOPIC, RGB_INFO, DEPTH_TOPIC, DEPTH_INFO]]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == RGB_INFO and rgb_cam_info is None:
                rgb_cam_info = msg
            elif connection.topic == DEPTH_INFO and depth_cam_info is None:
                depth_cam_info = msg

            if rgb_cam_info and depth_cam_info and affine_matrix is None:
                affine_matrix = get_affine_transform(depth_cam_info, rgb_cam_info)
                print("Affine transformation matrix computed!")

            if connection.topic == RGB_TOPIC:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                rgb_queue.append({'t': timestamp, 'd': img})

            elif connection.topic == DEPTH_TOPIC:
                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                depth_queue.append({'t': timestamp, 'd': raw})

            if affine_matrix is not None and rgb_queue and depth_queue:
                while rgb_queue and depth_queue:
                    diff = rgb_queue[0]['t'] - depth_queue[0]['t']

                    if abs(diff) < 35000000:
                        current_time = rgb_queue[0]['t']
                        wait_ms = 1

                        if last_render_time is not None:
                            dt_ms = (current_time - last_render_time) / 1e6
                            if dt_ms > 0:
                                wait_ms = int(dt_ms / speed)

                        wait_ms = max(1, min(wait_ms, 500))
                        last_render_time = current_time

                        rgb_img = rgb_queue[0]['d']
                        depth_raw = depth_queue[0]['d']
                        h_rgb, w_rgb = rgb_img.shape[:2]

                        # Align depth map
                        depth_aligned = cv2.warpAffine(
                            depth_raw, affine_matrix, (w_rgb, h_rgb), flags=cv2.INTER_NEAREST
                        )

                        # Run YOLO inference
                        results = model.predict(source=rgb_img, conf=0.5, verbose=False)[0]

                        # Use original RGB image without depth overlay
                        display_img = rgb_img.copy()

                        # Draw polygon contours and distance info for each detection
                        if results.masks is not None:
                            masks = results.masks.data.cpu().numpy()
                            boxes = results.boxes.xyxy.cpu().numpy()
                            cls_ids = results.boxes.cls.cpu().numpy()
                            names = model.names

                            for i, mask in enumerate(masks):
                                # Resize mask to RGB size
                                mask_resized = cv2.resize(mask, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)

                                # Get median depth for the object
                                object_depths = depth_aligned[mask_resized > 0.5]
                                valid_depths = object_depths[(object_depths > 300) & (object_depths < 5000)]

                                if len(valid_depths) > 0:
                                    avg_depth_m = np.median(valid_depths) / 1000.0
                                    dist_str = f"{avg_depth_m:.2f}m"
                                else:
                                    dist_str = "N/A"

                                # Get class ID and corresponding color
                                cls_id = int(cls_ids[i])
                                color = class_colors[cls_id]

                                # Find contours from the mask for polygon drawing
                                mask_uint8 = (mask_resized * 255).astype(np.uint8)
                                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                if contours:
                                    # Get the largest contour (main object)
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    # Approximate contour with simpler polygon
                                    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                                    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                                    # Draw polygon contour
                                    cv2.polylines(display_img, [approx_contour], True, color, 2)

                                    # Get bounding rect for label placement
                                    x1, y1, x2, y2 = map(int, boxes[i])

                                    # Draw label with class name and distance
                                    label = f"{names[cls_id]} | {dist_str}"
                                    cv2.putText(display_img, label, (x1, max(15, y1 - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        cv2.putText(display_img, f"Speed: {speed}x ", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        cv2.imshow('Semantic Depth Overlay', display_img)

                        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                            exit(0)

                        rgb_queue.popleft()
                        depth_queue.popleft()
                        break

                    elif diff < -35000000:
                        rgb_queue.popleft()
                    else:
                        depth_queue.popleft()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace this with your actual best.pt path
    MODEL_PATH = 'best.pt'  # You can also use official pretrained weights for testing
    BAG_PATH = './'         # Path to your ROS bag directory containing the camera topics

    path = Path(MODEL_PATH)
    if not path.exists():
        print(f"Error: {MODEL_PATH} does not exist!")
        exit(1)
    if path.stat().st_size == 0:
        print(f"Error: {MODEL_PATH} is an empty file (0 bytes).")
        exit(1)

    run_semantic_depth_overlay(BAG_PATH, MODEL_PATH, speed=1)