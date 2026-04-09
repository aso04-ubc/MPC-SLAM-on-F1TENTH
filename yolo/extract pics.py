import cv2
import numpy as np
import random
import os
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def extract_random_frames(bag_dir_path, output_dir, n=100):
    CAMERA_TOPIC = '/camera/color/image_raw'
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    # Maintain a single reader session throughout processing
    with AnyReader([Path(bag_dir_path)], default_typestore=typestore) as reader:

        # Filter camera connections
        camera_connections = [x for x in reader.connections if x.topic == CAMERA_TOPIC]

        # Read total frame count from metadata
        total_frames = sum(conn.msgcount for conn in camera_connections)

        if total_frames == 0:
            print(f"No image data found on topic {CAMERA_TOPIC}")
            return

        # Generate random indices for extraction
        n = min(n, total_frames)
        random_indices = set(random.sample(range(total_frames), n))
        print(f"Found {total_frames} total frames, extracting {n} random frames...")

        # Iterate through messages and save selected frames
        current_idx = 0
        saved_count = 0

        for connection, timestamp, rawdata in reader.messages(connections=camera_connections):
            if current_idx in random_indices:
                msg = reader.deserialize(rawdata, connection.msgtype)

                # Convert image data to NumPy array
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

                # Convert RGB to BGR if necessary for OpenCV
                if hasattr(msg, 'encoding') and msg.encoding == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Construct filename with frame index and timestamp
                filename = os.path.join(output_dir, f"frame_{current_idx:05d}_{timestamp}.jpg")
                cv2.imwrite(filename, img)

                saved_count += 1
                print(f"Progress [{saved_count}/{n}] Saved: {filename}")

                # Exit early once target count is reached
                if saved_count >= n:
                    break

            current_idx += 1

    print("Frame extraction completed successfully")


if __name__ == "__main__":
    # Specify the bag file path and output directory
    run_bag_path = './'
    output_folder = './extracted_random_frames'

    extract_random_frames(run_bag_path, output_folder, n=100)