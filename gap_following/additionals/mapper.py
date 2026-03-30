import cv2
import numpy as np
import math
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import KDTree # Required for fast ICP matching


# --- Helper Functions ---
def get_yaw_from_quat(q):
    """Converts a quaternion to Euler Yaw (Z-axis rotation)."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def icp_correct_pose(local_points, map_points, current_pose, max_iterations=10):
    """
    2D ICP (Iterative Closest Point) using SVD.
    Matches current local laser scan to the existing global map to correct heading/position drift.
    """
    yaw = current_pose['yaw']
    tx, ty = current_pose['x'], current_pose['y']
    
    # 1. Project local laser points to global frame using the *uncorrected* pose guess
    R = np.array([[math.cos(yaw), -math.sin(yaw)],
                  [math.sin(yaw),  math.cos(yaw)]])
    global_pts = np.dot(local_points, R.T) + np.array([tx, ty])
    
    tree = KDTree(map_points)
    
    for _ in range(max_iterations):
        # 2. Find nearest neighbors in the existing map
        distances, indices = tree.query(global_pts)
        
        # Filter out points that are too far away (e.g., > 30cm) to ignore noise
        valid = distances < 0.3 
        if np.sum(valid) < 15: 
            break # Not enough overlapping points to make a confident match
        
        p_src = global_pts[valid]
        p_dst = map_points[indices[valid]]
        
        # 3. Calculate centroids
        c_src = np.mean(p_src, axis=0)
        c_dst = np.mean(p_dst, axis=0)
        
        # 4. SVD for Rotation correction
        H = np.dot((p_src - c_src).T, (p_dst - c_dst))
        U, S, Vt = np.linalg.svd(H)
        R_delta = np.dot(Vt.T, U.T)
        
        # Prevent reflection matrix
        if np.linalg.det(R_delta) < 0:
            Vt[1, :] *= -1
            R_delta = np.dot(Vt.T, U.T)
            
        t_delta = c_dst - np.dot(R_delta, c_src)
        
        # 5. Apply correction to the points for the next iteration
        global_pts = np.dot(global_pts, R_delta.T) + t_delta
        
        # 6. Apply correction to the robot's tracked pose
        pose_pt = np.dot(R_delta, np.array([tx, ty])) + t_delta
        tx, ty = pose_pt[0], pose_pt[1]
        yaw += math.atan2(R_delta[1, 0], R_delta[0, 0])
        
    return tx, ty, yaw


def run_imu_odom_scan_mapping(bag_dir_path, speed=1.0):
    # --- Topic Configuration ---
    IMU_TOPIC = '/sensors/imu/raw'
    ODOM_TOPIC = '/odom'
    SCAN_TOPIC = '/scan'

    # Visualization Parameters
    WINDOW_SIZE = 1000
    SCALE = 40
    CX, CY = WINDOW_SIZE // 2, WINDOW_SIZE // 2

    GYRO_SCALE = 1.0
    ODOM_SCALE = 1.0

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    odom_path = []
    fused_path = []
    last_icp_time = 0.0
    ICP_INTERVAL = 2.0

    last_odom_pos = None
    last_imu_time = None
    last_odom_time = None

    fused_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

    imu_bias_z = None
    bias_samples = []
    CALIBRATION_FRAMES = 100
    start_offset_screen = None

    # --- 🌟 HEADING RESET VARIABLES 🌟 ---
    start_anchor_points = None
    has_completed_lap = False
    total_dist_traveled = 0.0
    MIN_LAP_DISTANCE = 5.0  # Meters: Don't reset until we've driven at least this much
    RESET_THRESHOLD_RADIUS = 1.5  # Meters: How close to (0,0) to trigger reset

    map_canvas_float = np.full((WINDOW_SIZE, WINDOW_SIZE, 3), 127.0, dtype=np.float32)
    MAP_UPDATE_RATE = 0.05

    print(f"🚀 Initializing Probabilistic Occupancy Grid Mapping with ICP...")

    with AnyReader([Path(bag_dir_path)], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in [IMU_TOPIC, ODOM_TOPIC, SCAN_TOPIC]]

        for connection, _, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            msg_time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # --- 1. IMU Processing ---
            if connection.topic == IMU_TOPIC:
                gyro_z_deg = msg.angular_velocity.z
                gyro_z_rad = (gyro_z_deg * (math.pi / 180.0)) * GYRO_SCALE

                if imu_bias_z is None:
                    bias_samples.append(gyro_z_rad)
                    if len(bias_samples) >= CALIBRATION_FRAMES:
                        imu_bias_z = sum(bias_samples) / len(bias_samples)
                        print(f"✅ IMU 校准完成! Z轴零偏: {imu_bias_z:.6f} rad/s")
                    continue

                if last_imu_time is not None:
                    dt = msg_time_sec - last_imu_time
                    if 0 < dt < 0.5:
                        fused_pose['yaw'] += (gyro_z_rad - imu_bias_z) * dt
                last_imu_time = msg_time_sec

            # --- 2. Odometry Processing ---
            elif connection.topic == ODOM_TOPIC:
                px = msg.pose.pose.position.x
                py = msg.pose.pose.position.y

                if last_odom_pos is None:
                    last_odom_pos = {'x': px, 'y': py}
                    start_offset_screen = {'x': px, 'y': py}
                    last_odom_time = msg_time_sec
                    fused_pose['yaw'] = get_yaw_from_quat(msg.pose.pose.orientation)
                    continue

                screen_x_odom = int(CX + (py - start_offset_screen['y']) * SCALE)
                screen_y_odom = int(CY - (px - start_offset_screen['x']) * SCALE)
                if not odom_path or (abs(odom_path[-1][0] - screen_x_odom) + abs(odom_path[-1][1] - screen_y_odom) > 1):
                    odom_path.append((screen_x_odom, screen_y_odom))

                if last_odom_time is not None:
                    dt_odom = msg_time_sec - last_odom_time
                    if 0 < dt_odom < 0.5:
                        dist_step = msg.twist.twist.linear.x * dt_odom * ODOM_SCALE
                        fused_pose['x'] += dist_step * math.cos(fused_pose['yaw'])
                        fused_pose['y'] += dist_step * math.sin(fused_pose['yaw'])

                        screen_x_fused = int(CX + fused_pose['y'] * SCALE)
                        screen_y_fused = int(CY - fused_pose['x'] * SCALE)
                        total_dist_traveled += abs(dist_step)
                        if not fused_path or (abs(fused_path[-1][0] - screen_x_fused) + abs(fused_path[-1][1] - screen_y_fused) > 1):
                            fused_path.append((screen_x_fused, screen_y_fused))

                last_odom_pos = {'x': px, 'y': py}
                last_odom_time = msg_time_sec

            # --- 3. LiDAR Scan Processing & ICP ---
            elif connection.topic == SCAN_TOPIC:
                if imu_bias_z is None or last_odom_pos is None:
                    continue

                ranges = np.array(msg.ranges)
                angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

                TRIM_COUNT = 80
                if TRIM_COUNT > 0 and len(ranges) > 2 * TRIM_COUNT:
                    ranges = ranges[TRIM_COUNT:-TRIM_COUNT]
                    angles = angles[TRIM_COUNT:-TRIM_COUNT]

                valid = (ranges > msg.range_min) & (ranges < 1) & np.isfinite(ranges)
                ranges_valid = ranges[valid]
                angles_valid = angles[valid]

                # --- 🌟 ICP SCAN MATCHING INTEGRATION 🌟 ---
                obs_y_px, obs_x_px = np.where(map_canvas_float[:, :, 0] < 50)

                if len(obs_x_px) > 300 and (msg_time_sec - last_icp_time) >= ICP_INTERVAL:
            
                    step = max(1, len(obs_x_px) // 1500)
                    obs_x_px = obs_x_px[::step]
                    obs_y_px = obs_y_px[::step]

                    map_pts_x = (CY - obs_y_px) / SCALE
                    map_pts_y = (obs_x_px - CX) / SCALE
                    map_points = np.column_stack((map_pts_x, map_pts_y))

                    local_angles = angles_valid + math.pi
                    lx = ranges_valid * np.cos(local_angles)
                    ly = ranges_valid * np.sin(local_angles)
                    local_points = np.column_stack((lx, ly))

                    corrected_x, corrected_y, corrected_yaw = icp_correct_pose(
                        local_points, map_points, fused_pose
                    )

                    delta_dist = math.hypot(corrected_x - fused_pose['x'], corrected_y - fused_pose['y'])
                    delta_yaw  = abs(corrected_yaw - fused_pose['yaw'])

                    if delta_dist < 0.25 and delta_yaw < 0.08:
                        fused_pose['x']   = corrected_x
                        fused_pose['y']   = corrected_y
                        fused_pose['yaw'] = corrected_yaw

                    last_icp_time = msg_time_sec  # ✅ always update, even if correction was rejected
                # --- END ICP ---
                        
                # Calculate final map points using the (now corrected) pose
                global_angles = fused_pose['yaw'] + angles_valid + math.pi
                px_valid = fused_pose['x'] + ranges_valid * np.cos(global_angles)
                py_valid = fused_pose['y'] + ranges_valid * np.sin(global_angles)

                sx = (CX + py_valid * SCALE).astype(np.int32)
                sy = (CY - px_valid * SCALE).astype(np.int32)
                rx = int(CX + fused_pose['y'] * SCALE)
                ry = int(CY - fused_pose['x'] * SCALE)

                bounds_valid = (sx >= 0) & (sx < WINDOW_SIZE) & (sy >= 0) & (sy < WINDOW_SIZE)
                sx = sx[bounds_valid]
                sy = sy[bounds_valid]

                if len(sx) > 0:
                    pts = np.column_stack((sx, sy))
                    polygon_pts = np.vstack(([rx, ry], pts))

                    current_scan = np.full((WINDOW_SIZE, WINDOW_SIZE, 3), 127, dtype=np.uint8)
                    cv2.fillPoly(current_scan, [polygon_pts], (255, 255, 255))
                    current_scan[sy, sx] = [0, 0, 0]

                    active_mask = current_scan[:, :, 0] != 127
                    map_canvas_float[active_mask] = (
                            map_canvas_float[active_mask] * (1.0 - MAP_UPDATE_RATE) +
                            current_scan[active_mask] * MAP_UPDATE_RATE
                    )

            # --- 4. Real-time Rendering ---
            if connection.topic == ODOM_TOPIC and len(fused_path) % 5 == 0:
                canvas = map_canvas_float.astype(np.uint8)

                if len(odom_path) > 1:
                    cv2.polylines(canvas, [np.array(odom_path)], False, (0, 0, 150), 1)
                if len(fused_path) > 1:
                    cv2.polylines(canvas, [np.array(fused_path)], False, (0, 200, 0), 2)

                if len(fused_path) > 1:
                    curr = fused_path[-1]
                    prev = fused_path[-2]
                    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        tip_x = int(curr[0] + 20 * (dx / dist))
                        tip_y = int(curr[1] + 20 * (dy / dist))
                        cv2.arrowedLine(canvas, curr, (tip_x, tip_y), (0, 255, 255), 2)
                    cv2.circle(canvas, curr, 5, (0, 0, 255), -1)

                cv2.putText(canvas, "Grey: Unknown | White: Free | Black: Obstacle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.imshow('Probabilistic Grid Mapping', canvas)

                if cv2.waitKey(int(10 / speed)) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
    final_map = map_canvas_float.astype(np.uint8)
    cv2.imwrite('slam_map_final.png', final_map)
    print("💾 地图已保存至: slam_map_final.png")

if __name__ == "__main__":
    run_imu_odom_scan_mapping('./rosbag2_2026_03_26-15_48_05')