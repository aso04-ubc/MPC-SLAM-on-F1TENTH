#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import KDTree

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan

from planning_pkg.race_line_core import gray_image_to_occupancy_data


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def get_yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def icp_correct_pose(
    local_points: np.ndarray,
    map_points: np.ndarray,
    current_pose: dict,
    max_iterations: int = 10,
    match_distance: float = 0.30,
) -> Tuple[float, float, float]:
    yaw = float(current_pose['yaw'])
    tx = float(current_pose['x'])
    ty = float(current_pose['y'])

    rot = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    global_pts = np.dot(local_points, rot.T) + np.array([tx, ty], dtype=float)

    tree = KDTree(map_points)

    for _ in range(max_iterations):
        distances, indices = tree.query(global_pts)
        valid = distances < match_distance
        if int(np.sum(valid)) < 15:
            break

        p_src = global_pts[valid]
        p_dst = map_points[indices[valid]]

        c_src = np.mean(p_src, axis=0)
        c_dst = np.mean(p_dst, axis=0)

        h_mat = np.dot((p_src - c_src).T, (p_dst - c_dst))
        u_mat, _, vt_mat = np.linalg.svd(h_mat)
        r_delta = np.dot(vt_mat.T, u_mat.T)

        if np.linalg.det(r_delta) < 0:
            vt_mat[1, :] *= -1
            r_delta = np.dot(vt_mat.T, u_mat.T)

        t_delta = c_dst - np.dot(r_delta, c_src)

        global_pts = np.dot(global_pts, r_delta.T) + t_delta

        pose_pt = np.dot(r_delta, np.array([tx, ty], dtype=float)) + t_delta
        tx, ty = float(pose_pt[0]), float(pose_pt[1])
        yaw += math.atan2(r_delta[1, 0], r_delta[0, 0])

    return tx, ty, wrap_angle(yaw)


class LiveMapperNode(Node):
    def __init__(self) -> None:
        super().__init__('live_mapper_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('sim', True),
                ('imu_topic', '/sensors/imu/raw'),
                ('odom_topic', '/odom'),
                ('scan_topic', '/scan'),
                ('map_topic', '/mapping/occupancy_grid'),
                ('pose_topic', '/mapping/fused_pose'),
                ('map_frame_id', 'map'),
                ('map_publish_rate_hz', 2.0),
                ('pose_publish_rate_hz', 2.0),
                ('map_window_size', 1000),
                ('map_scale_px_per_m', 40.0),
                ('map_update_rate', 0.05),
                ('free_thresh', 200),
                ('occ_thresh', 90),
                ('scan_max_range_m', 1.0),
                ('scan_trim_count', 80),
                ('scan_angle_offset_rad', math.pi),
                ('imu_calibration_frames', 100),
                ('imu_gyro_in_degrees', True),
                ('gyro_scale', 1.0),
                ('odom_scale', 1.0),
                ('icp_enabled', False),
                ('icp_interval_s', 0.5),
                ('icp_max_iterations', 15),
                ('icp_match_distance_m', 0.30),
                ('icp_ref_min_points', 300),
                ('icp_accept_dist_max', 0.35),
                ('icp_accept_yaw_max', 0.60),
                ('show_opencv_debug', False),
            ],
        )

        self.sim = bool(self.get_parameter('sim').value)
        self.imu_topic = str(self.get_parameter('imu_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.map_frame_id = str(self.get_parameter('map_frame_id').value)

        self.map_publish_rate_hz = float(self.get_parameter('map_publish_rate_hz').value)
        self.pose_publish_rate_hz = float(self.get_parameter('pose_publish_rate_hz').value)

        self.window_size = int(self.get_parameter('map_window_size').value)
        self.scale_px_per_m = float(self.get_parameter('map_scale_px_per_m').value)
        self.map_update_rate = float(self.get_parameter('map_update_rate').value)
        self.free_thresh = int(self.get_parameter('free_thresh').value)
        self.occ_thresh = int(self.get_parameter('occ_thresh').value)

        self.scan_max_range_m = float(self.get_parameter('scan_max_range_m').value)
        self.scan_trim_count = int(self.get_parameter('scan_trim_count').value)
        self.scan_angle_offset_rad = float(self.get_parameter('scan_angle_offset_rad').value)

        self.imu_calibration_frames = int(self.get_parameter('imu_calibration_frames').value)
        self.imu_gyro_in_degrees = bool(self.get_parameter('imu_gyro_in_degrees').value)
        self.gyro_scale = float(self.get_parameter('gyro_scale').value)
        self.odom_scale = float(self.get_parameter('odom_scale').value)

        self.icp_enabled = bool(self.get_parameter('icp_enabled').value)
        self.icp_interval_s = float(self.get_parameter('icp_interval_s').value)
        self.icp_max_iterations = int(self.get_parameter('icp_max_iterations').value)
        self.icp_match_distance_m = float(self.get_parameter('icp_match_distance_m').value)
        self.icp_ref_min_points = int(self.get_parameter('icp_ref_min_points').value)
        self.icp_accept_dist_max = float(self.get_parameter('icp_accept_dist_max').value)
        self.icp_accept_yaw_max = float(self.get_parameter('icp_accept_yaw_max').value)

        self.show_opencv_debug = bool(self.get_parameter('show_opencv_debug').value) and self.sim

        self.cx = self.window_size // 2
        self.cy = self.window_size // 2

        self.map_canvas_float = np.full((self.window_size, self.window_size), 127.0, dtype=np.float32)

        self.fused_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.last_imu_time: Optional[float] = None
        self.last_odom_time: Optional[float] = None
        self.last_icp_time = 0.0

        self.imu_bias_z: Optional[float] = None
        self.imu_bias_samples: list[float] = []

        self.odom_ready = False

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(Imu, self.imu_topic, self.imu_callback, qos_sensor)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos_sensor)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_sensor)

        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, 1)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        self.create_timer(1.0 / max(0.1, self.map_publish_rate_hz), self.publish_map)
        self.create_timer(1.0 / max(0.1, self.pose_publish_rate_hz), self.publish_pose)

        if self.show_opencv_debug:
            cv2.namedWindow('Live Mapper Debug', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Live Mapper Debug', 1000, 1000)
            self.create_timer(0.1, self.show_debug)

        self.get_logger().info(
            f'Live mapper started: map_topic={self.map_topic}, pose_topic={self.pose_topic}, '
            f'scale={self.scale_px_per_m:.1f}px/m, size={self.window_size}'
        )

    def msg_time(self, sec: int, nsec: int) -> float:
        return float(sec) + float(nsec) * 1e-9

    def imu_callback(self, msg: Imu) -> None:
        ts = self.msg_time(msg.header.stamp.sec, msg.header.stamp.nanosec)

        gyro_z = float(msg.angular_velocity.z)
        if self.imu_gyro_in_degrees:
            gyro_z *= (math.pi / 180.0)
        gyro_z *= self.gyro_scale

        if self.imu_bias_z is None:
            self.imu_bias_samples.append(gyro_z)
            if len(self.imu_bias_samples) >= self.imu_calibration_frames:
                self.imu_bias_z = float(np.mean(self.imu_bias_samples))
                self.get_logger().info(f'IMU calibration completed. bias_z={self.imu_bias_z:.6f} rad/s')
            self.last_imu_time = ts
            return

        if self.last_imu_time is not None:
            dt = ts - self.last_imu_time
            if 0.0 < dt < 0.5:
                self.fused_pose['yaw'] = wrap_angle(
                    self.fused_pose['yaw'] + (gyro_z - self.imu_bias_z) * dt
                )

        self.last_imu_time = ts

    def odom_callback(self, msg: Odometry) -> None:
        ts = self.msg_time(msg.header.stamp.sec, msg.header.stamp.nanosec)

        if not self.odom_ready:
            self.odom_ready = True
            self.last_odom_time = ts
            self.fused_pose['yaw'] = get_yaw_from_quat(msg.pose.pose.orientation)
            return

        if self.last_odom_time is not None:
            dt = ts - self.last_odom_time
            if 0.0 < dt < 0.5:
                speed = float(msg.twist.twist.linear.x)
                dist_step = speed * dt * self.odom_scale
                self.fused_pose['x'] += dist_step * math.cos(self.fused_pose['yaw'])
                self.fused_pose['y'] += dist_step * math.sin(self.fused_pose['yaw'])

        self.last_odom_time = ts

    def scan_callback(self, msg: LaserScan) -> None:
        if not self.odom_ready or self.imu_bias_z is None:
            return

        ranges = np.array(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges), dtype=float) * msg.angle_increment

        trim = self.scan_trim_count
        if trim > 0 and len(ranges) > 2 * trim:
            ranges = ranges[trim:-trim]
            angles = angles[trim:-trim]

        valid = (
            (ranges > float(msg.range_min))
            & (ranges < min(float(msg.range_max), self.scan_max_range_m))
            & np.isfinite(ranges)
        )

        ranges_valid = ranges[valid]
        angles_valid = angles[valid]
        if len(ranges_valid) < 8:
            return

        local_angles = angles_valid + self.scan_angle_offset_rad
        lx = ranges_valid * np.cos(local_angles)
        ly = ranges_valid * np.sin(local_angles)
        local_points = np.column_stack((lx, ly))

        ts = self.msg_time(msg.header.stamp.sec, msg.header.stamp.nanosec)

        if self.icp_enabled and (ts - self.last_icp_time) >= self.icp_interval_s:
            obs_y, obs_x = np.where(self.map_canvas_float < 50.0)
            if len(obs_x) >= self.icp_ref_min_points:
                map_pts_x = (self.cy - obs_y.astype(float)) / self.scale_px_per_m
                map_pts_y = (obs_x.astype(float) - self.cx) / self.scale_px_per_m
                ref_points = np.column_stack((map_pts_x, map_pts_y))

                corrected_x, corrected_y, corrected_yaw = icp_correct_pose(
                    local_points=local_points,
                    map_points=ref_points,
                    current_pose=self.fused_pose,
                    max_iterations=self.icp_max_iterations,
                    match_distance=self.icp_match_distance_m,
                )

                delta_dist = math.hypot(corrected_x - self.fused_pose['x'], corrected_y - self.fused_pose['y'])
                delta_yaw = abs(wrap_angle(corrected_yaw - self.fused_pose['yaw']))

                if delta_dist < self.icp_accept_dist_max and delta_yaw < self.icp_accept_yaw_max:
                    self.fused_pose['x'] = corrected_x
                    self.fused_pose['y'] = corrected_y
                    self.fused_pose['yaw'] = corrected_yaw

            self.last_icp_time = ts

        global_angles = self.fused_pose['yaw'] + angles_valid + self.scan_angle_offset_rad
        px_valid = self.fused_pose['x'] + ranges_valid * np.cos(global_angles)
        py_valid = self.fused_pose['y'] + ranges_valid * np.sin(global_angles)

        sx = np.round(self.cx + py_valid * self.scale_px_per_m).astype(np.int32)
        sy = np.round(self.cy - px_valid * self.scale_px_per_m).astype(np.int32)

        rx = int(round(self.cx + self.fused_pose['y'] * self.scale_px_per_m))
        ry = int(round(self.cy - self.fused_pose['x'] * self.scale_px_per_m))

        in_bounds = (
            (sx >= 0) & (sx < self.window_size) &
            (sy >= 0) & (sy < self.window_size)
        )

        sx = sx[in_bounds]
        sy = sy[in_bounds]

        if len(sx) == 0:
            return

        pts = np.column_stack((sx, sy))
        polygon = np.vstack(([rx, ry], pts))

        current_scan = np.full((self.window_size, self.window_size), 127, dtype=np.uint8)
        cv2.fillPoly(current_scan, [polygon], 255)
        current_scan[sy, sx] = 0

        active = current_scan != 127
        self.map_canvas_float[active] = (
            self.map_canvas_float[active] * (1.0 - self.map_update_rate)
            + current_scan[active].astype(np.float32) * self.map_update_rate
        )

    def publish_pose(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame_id

        msg.pose.position.x = float(self.fused_pose['x'])
        msg.pose.position.y = float(self.fused_pose['y'])
        msg.pose.position.z = 0.0

        yaw = float(self.fused_pose['yaw'])
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = float(math.sin(0.5 * yaw))
        msg.pose.orientation.w = float(math.cos(0.5 * yaw))
        self.pose_pub.publish(msg)

    def publish_map(self) -> None:
        gray = self.map_canvas_float.astype(np.uint8)

        occ_data = gray_image_to_occupancy_data(
            gray,
            free_thresh=self.free_thresh,
            occ_thresh=self.occ_thresh,
        )

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame_id

        msg.info.resolution = float(1.0 / self.scale_px_per_m)
        msg.info.width = int(self.window_size)
        msg.info.height = int(self.window_size)
        msg.info.origin.position.x = float(-self.window_size / (2.0 * self.scale_px_per_m))
        msg.info.origin.position.y = float(-self.window_size / (2.0 * self.scale_px_per_m))
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = occ_data.tolist()
        self.map_pub.publish(msg)

    def show_debug(self) -> None:
        canvas = cv2.cvtColor(self.map_canvas_float.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        px = int(round(self.cx + self.fused_pose['y'] * self.scale_px_per_m))
        py = int(round(self.cy - self.fused_pose['x'] * self.scale_px_per_m))
        if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)

            tip_x = int(round(px + 20.0 * math.sin(self.fused_pose['yaw'])))
            tip_y = int(round(py - 20.0 * math.cos(self.fused_pose['yaw'])))
            cv2.arrowedLine(canvas, (px, py), (tip_x, tip_y), (0, 255, 255), 2)

        cv2.putText(
            canvas,
            f'x={self.fused_pose["x"]:.2f} y={self.fused_pose["y"]:.2f} yaw={self.fused_pose["yaw"]:.2f}',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 50, 255),
            2,
        )

        cv2.imshow('Live Mapper Debug', canvas)
        cv2.waitKey(1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LiveMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_opencv_debug:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
