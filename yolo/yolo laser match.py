import cv2
import numpy as np
import collections
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from ultralytics import YOLO


def run_column_mapping_semantic_lidar(bag_dir_path, yolo_model_path, speed=1.0):
    RGB_TOPIC = '/camera/color/image_raw'
    RGB_INFO = '/camera/color/camera_info'
    SCAN_TOPIC = '/scan'

    # --- 1. Initialize YOLO model ---
    print(f"Loading YOLO model: {yolo_model_path} ...")
    model = YOLO(yolo_model_path)
    names = model.names

    # Generate random class colors (BGR)
    np.random.seed(42)
    class_colors = {i: tuple(map(int, np.random.randint(50, 255, 3))) for i in range(len(names))}

    # List of classes to ignore (ignore road and other ground classifications)
    ignore_classes = ['road', 'street', 'ground', 'pavement']

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # We no longer need depth map, only synchronize RGB and LiDAR
    rgb_queue = collections.deque()
    scan_queue = collections.deque()

    rgb_info = None
    rgb_fx, rgb_cx = None, None
    last_render_time = None

    print(f"Starting 2D Position Mapping Semantic LiDAR (Speed: {speed}x)... Press 'q' to exit")

    with AnyReader([Path(bag_dir_path)], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if
                       x.topic in [RGB_TOPIC, RGB_INFO, SCAN_TOPIC]]

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)

            if conn.topic == RGB_INFO and rgb_info is None:
                rgb_info = msg
                # Extract RGB camera intrinsic parameters
                k_rgb = np.array(rgb_info.k).reshape(3, 3)
                rgb_fx = k_rgb[0, 0]
                rgb_cx = k_rgb[0, 2]
                print(f"Extracted camera intrinsic parameters: fx={rgb_fx:.2f}, cx={rgb_cx:.2f}")

            # Collect data into queues
            if conn.topic == RGB_TOPIC:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                rgb_queue.append({'t': timestamp, 'd': cv2.cvtColor(img, cv2.COLOR_RGB2BGR)})
            elif conn.topic == SCAN_TOPIC:
                scan_queue.append({'t': timestamp, 'd': msg})

            # Synchronize RGB and LiDAR data
            if rgb_fx is not None and rgb_queue and scan_queue:
                while rgb_queue and scan_queue:
                    diff = rgb_queue[0]['t'] - scan_queue[0]['t']

                    if abs(diff) < 50000000:  # 50ms time tolerance
                        curr_t = rgb_queue[0]['t']
                        wait_ms = max(1, int((curr_t - last_render_time) / 1e6 / speed)) if last_render_time else 1
                        last_render_time = curr_t

                        rgb_frame = rgb_queue[0]['d'].copy()
                        scan_msg = scan_queue[0]['d']
                        h, w = rgb_frame.shape[:2]

                        # --- 2. YOLO inference ---
                        results = model.predict(source=rgb_frame, conf=0.5, verbose=False)[0]

                        # Core optimization: Create 1D image column mapping array (length w)
                        # col_class records which class each column belongs to, initialized to -1
                        col_class = np.full(w, -1, dtype=np.int32)
                        # col_y2 records the lowest height (maximum y2) of objects in current column,
                        # used to resolve overlapping occlusion relationships
                        col_y2 = np.full(w, -1, dtype=np.float32)

                        if results.boxes is not None:
                            boxes = results.boxes.xyxy.cpu().numpy()
                            cls_ids = results.boxes.cls.cpu().numpy()

                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = map(int, box)
                                cls_id = int(cls_ids[i])
                                name = names[cls_id]

                                # Ignore road/ground classes
                                if name.lower() in ignore_classes:
                                    continue

                                color = class_colors[cls_id]

                                # Draw bounding box on RGB image for visualization
                                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(rgb_frame, name, (x1, max(15, y1 - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                # Clamp boundaries
                                x1 = max(0, min(w - 1, x1))
                                x2 = max(0, min(w - 1, x2))

                                # Register object class to the image columns it spans
                                for u in range(x1, x2 + 1):
                                    # If overlapping, prioritize the one with larger y2
                                    # (lower in image, meaning closer to camera)
                                    if y2 > col_y2[u]:
                                        col_y2[u] = y2
                                        col_class[u] = cls_id

                        # --- 3. Initialize LiDAR BEV canvas ---
                        lidar_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        lidar_canvas.fill(20)
                        lidar_cx, lidar_cy, scale = w // 2, h - 40, 60  # scale: pixels/meter

                        # --- 4. Project LiDAR and apply semantic coloring based on image column ---
                        s_angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
                        s_ranges = np.array(scan_msg.ranges)

                        v_mask = (s_ranges > scan_msg.range_min) & (s_ranges < scan_msg.range_max)
                        angles = s_angles[v_mask]
                        ranges = s_ranges[v_mask]

                        # Calculate LiDAR point coordinates on BEV canvas
                        l_xs = (lidar_cx - (ranges * np.sin(angles) * scale)).astype(int)
                        l_ys = (lidar_cy - (ranges * np.cos(angles) * scale)).astype(int)

                        # Projection: compute image column pixel 'u' corresponding to LiDAR point
                        us = (-rgb_fx * np.tan(angles) + rgb_cx).astype(int)

                        for i in range(len(ranges)):
                            lx, ly = l_xs[i], l_ys[i]
                            if not (0 <= lx < w and 0 <= ly < h):
                                continue

                            u = us[i]
                            color = (100, 100, 100)  # Default gray (no class or road)

                            # If LiDAR point's horizontal coordinate is within camera view
                            if 0 <= u < w:
                                # Query semantic class directly by column index
                                matched_cls = col_class[u]
                                if matched_cls != -1:
                                    color = class_colors[matched_cls]

                            # Draw colored LiDAR point
                            cv2.circle(lidar_canvas, (lx, ly), 3, color, -1)

                        cv2.putText(lidar_canvas, "Column Mapped LiDAR", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Render output
                        combined = np.hstack((rgb_frame, lidar_canvas))
                        cv2.imshow("Column Mapping RGB + LiDAR Analysis", combined)

                        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                            return

                        rgb_queue.popleft()
                        scan_queue.popleft()
                        break

                    elif diff < -50000000:
                        rgb_queue.popleft()
                    else:
                        scan_queue.popleft()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    MODEL_PATH = 'best.pt'  # YOLO model
    BAG_PATH = './'         # Path to ROS bag directory

    run_column_mapping_semantic_lidar(BAG_PATH, MODEL_PATH, speed=3.0)