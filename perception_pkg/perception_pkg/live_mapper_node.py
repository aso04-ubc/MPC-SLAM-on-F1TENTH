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
from sensor_msgs.msg import Image, Imu, LaserScan

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
                ('map_publish_rate_hz', 30.0),
                ('pose_publish_rate_hz', 60.0),
                ('map_window_size', 1600),
                ('map_scale_px_per_m', 20.0),
                ('map_update_rate', 0.05),
                ('map_update_rate_obstacle', 0.05),
                ('map_update_rate_free', 0.05),
                ('virtual_fill_free_boost', 0.25),
                ('virtual_wall_obstacle_boost', 0.35),
                ('virtual_fill_close_kernel_px', 7),
                ('free_thresh', 200),
                ('occ_thresh', 90),
                # Cap ranges used for mapping (m); was 1.0 and discarded almost all gym LiDAR.
                ('scan_max_range_m', 2.5),
                ('scan_trim_count', 80),
                ('scan_angle_offset_rad', 0.0),
                ('track_width_assumption_enabled', True),
                ('track_width_init_m', 1.0),
                ('track_width_alpha', 0.97),
                ('side_angle_center_deg', 90.0),
                ('side_angle_half_window_deg', 20.0),
                ('virtual_wall_span_deg', 24.0),
                ('virtual_wall_num_points', 17),
                ('virtual_wall_min_m', 0.20),
                ('virtual_wall_thickness_px', 3),
                ('virtual_wall_anchor_max_gap_deg', 18.0),
                ('virtual_wall_smooth_window', 5),
                ('virtual_wall_blend_alpha', 0.65),
                ('virtual_wall_max_step_m', 0.08),
                ('virtual_wall_max_segment_jump_px', 14),
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
                # When sim=False, publish the same debug canvas as sensor_msgs/Image.
                ('publish_debug_images', True),
                ('mapper_debug_image_topic', '/mapping/debug_image'),
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
        self.map_update_rate_obstacle = float(self.get_parameter('map_update_rate_obstacle').value)
        self.map_update_rate_free = float(self.get_parameter('map_update_rate_free').value)
        self.virtual_fill_free_boost = float(self.get_parameter('virtual_fill_free_boost').value)
        self.virtual_wall_obstacle_boost = float(self.get_parameter('virtual_wall_obstacle_boost').value)
        self.virtual_fill_close_kernel_px = int(self.get_parameter('virtual_fill_close_kernel_px').value)
        self.free_thresh = int(self.get_parameter('free_thresh').value)
        self.occ_thresh = int(self.get_parameter('occ_thresh').value)

        self.scan_max_range_m = float(self.get_parameter('scan_max_range_m').value)
        self.scan_trim_count = int(self.get_parameter('scan_trim_count').value)
        self.scan_angle_offset_rad = float(self.get_parameter('scan_angle_offset_rad').value)
        # Non-sim radar may use a different angle convention; force the original
        # mapping convention so that forward/back and left/right are consistent.
        if not self.sim:
            self.scan_angle_offset_rad = math.pi
        self.track_width_assumption_enabled = bool(self.get_parameter('track_width_assumption_enabled').value)
        self.track_width_est_m = float(self.get_parameter('track_width_init_m').value)
        self.track_width_alpha = float(self.get_parameter('track_width_alpha').value)
        self.side_angle_center_deg = float(self.get_parameter('side_angle_center_deg').value)
        self.side_angle_half_window_deg = float(self.get_parameter('side_angle_half_window_deg').value)
        self.virtual_wall_span_deg = float(self.get_parameter('virtual_wall_span_deg').value)
        self.virtual_wall_num_points = int(self.get_parameter('virtual_wall_num_points').value)
        self.virtual_wall_min_m = float(self.get_parameter('virtual_wall_min_m').value)
        self.virtual_wall_thickness_px = int(self.get_parameter('virtual_wall_thickness_px').value)
        self.virtual_wall_anchor_max_gap_deg = float(self.get_parameter('virtual_wall_anchor_max_gap_deg').value)
        self.virtual_wall_smooth_window = int(self.get_parameter('virtual_wall_smooth_window').value)
        self.virtual_wall_blend_alpha = float(self.get_parameter('virtual_wall_blend_alpha').value)
        self.virtual_wall_max_step_m = float(self.get_parameter('virtual_wall_max_step_m').value)
        self.virtual_wall_max_segment_jump_px = int(self.get_parameter('virtual_wall_max_segment_jump_px').value)

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
        self.publish_mapper_debug_images = (
            (not self.sim) and bool(self.get_parameter('publish_debug_images').value)
        )
        self.mapper_debug_image_topic = str(self.get_parameter('mapper_debug_image_topic').value)
        self.mapper_debug_image_pub = None
        if self.publish_mapper_debug_images:
            self.mapper_debug_image_pub = self.create_publisher(Image, self.mapper_debug_image_topic, 1)

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

        if self.sim:
            # f1tenth_gym_ros has no IMU; without bias the node would never fuse scans.
            self.imu_bias_z = 0.0
            self.get_logger().info(
                'sim=True: IMU calibration skipped; pose comes from odometry (gym-compatible).'
            )

        # Match gym_bridge default publishers (reliable, depth 10); BEST_EFFORT does not match.
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
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
        if self.show_opencv_debug or self.publish_mapper_debug_images:
            # In non-sim we only publish; we guard cv2.imshow inside show_debug.
            self.create_timer(0.1, self.show_debug)

        self.get_logger().info(
            f'Live mapper started: map_topic={self.map_topic}, pose_topic={self.pose_topic}, '
            f'scale={self.scale_px_per_m:.1f}px/m, size={self.window_size}'
        )

    def msg_time(self, sec: int, nsec: int) -> float:
        return float(sec) + float(nsec) * 1e-9

    def imu_callback(self, msg: Imu) -> None:
        if self.sim:
            return

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

        if self.sim:
            # Gym odometry is already a consistent map-frame pose; do not dead-reckon from speed.
            self.odom_ready = True
            self.fused_pose['x'] = float(msg.pose.pose.position.x)
            self.fused_pose['y'] = float(msg.pose.pose.position.y)
            self.fused_pose['yaw'] = get_yaw_from_quat(msg.pose.pose.orientation)
            self.last_odom_time = ts
            return

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

        synth_mask = np.zeros(len(ranges_valid), dtype=bool)
        if self.track_width_assumption_enabled:
            angles_valid, ranges_valid, synth_mask = self.apply_virtual_wall_completion(angles_valid, ranges_valid)

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
        synth_mask = synth_mask[in_bounds]

        if len(sx) == 0:
            return

        pts = np.column_stack((sx, sy))
        polygon = np.vstack(([rx, ry], pts))

        current_scan = np.full((self.window_size, self.window_size), 127, dtype=np.uint8)
        cv2.fillPoly(current_scan, [polygon], 255)
        real_pts_mask = ~synth_mask
        if np.any(real_pts_mask):
            current_scan[sy[real_pts_mask], sx[real_pts_mask]] = 0

        synth_line_mask = np.zeros((self.window_size, self.window_size), dtype=np.uint8)

        # Strengthen virtual boundary continuity by drawing a thick connected wall.
        if int(np.sum(synth_mask)) >= 2:
            synth_pts = pts[synth_mask]
            thick = max(1, self.virtual_wall_thickness_px)
            self.draw_segmented_polyline(
                image=current_scan,
                points=synth_pts,
                color=0,
                thickness=thick,
            )
            self.draw_segmented_polyline(
                image=synth_line_mask,
                points=synth_pts,
                color=255,
                thickness=thick,
            )

            close_k = max(3, self.virtual_fill_close_kernel_px)
            if (close_k % 2) == 0:
                close_k += 1
            close_kernel = np.ones((close_k, close_k), dtype=np.uint8)

            # Close small free-space holes around the inferred boundary so the lane interior becomes continuous.
            free_layer = np.where(current_scan == 255, 255, 0).astype(np.uint8)
            free_layer = cv2.morphologyEx(free_layer, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            current_scan[free_layer > 0] = 255
            current_scan[synth_line_mask > 0] = 0

        free_mask = current_scan == 255
        obs_mask = current_scan == 0

        free_rate = self.map_update_rate_free
        if int(np.sum(synth_mask)) >= 2:
            free_rate = max(free_rate, self.virtual_fill_free_boost)

        if np.any(free_mask):
            self.map_canvas_float[free_mask] = (
                self.map_canvas_float[free_mask] * (1.0 - free_rate)
                + 255.0 * free_rate
            )

        if np.any(obs_mask):
            obs_rate = max(self.map_update_rate_obstacle, self.map_update_rate)
            if int(np.sum(synth_mask)) >= 2:
                obs_rate = max(obs_rate, self.virtual_wall_obstacle_boost)
            self.map_canvas_float[obs_mask] = (
                self.map_canvas_float[obs_mask] * (1.0 - obs_rate)
                + 0.0 * obs_rate
            )

    def side_distance(self, angles: np.ndarray, ranges: np.ndarray, side_sign: int) -> Optional[float]:
        center = math.radians(self.side_angle_center_deg) * float(side_sign)
        half = math.radians(self.side_angle_half_window_deg)
        mask = np.abs(angles - center) <= half
        if int(np.sum(mask)) < 4:
            return None
        vals = ranges[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 4:
            return None
        return float(np.median(vals))

    def draw_segmented_polyline(
        self,
        image: np.ndarray,
        points: np.ndarray,
        color: int,
        thickness: int,
    ) -> None:
        if len(points) < 2:
            return

        jump_px = float(max(2, self.virtual_wall_max_segment_jump_px))
        start = 0
        for idx in range(len(points) - 1):
            step = float(np.linalg.norm(points[idx + 1].astype(float) - points[idx].astype(float)))
            if step > jump_px:
                segment = points[start:idx + 1]
                if len(segment) >= 2:
                    cv2.polylines(
                        image,
                        [segment.reshape(-1, 1, 2)],
                        isClosed=False,
                        color=color,
                        thickness=thickness,
                    )
                start = idx + 1

        tail = points[start:]
        if len(tail) >= 2:
            cv2.polylines(
                image,
                [tail.reshape(-1, 1, 2)],
                isClosed=False,
                color=color,
                thickness=thickness,
            )

    def apply_virtual_wall_completion(
        self,
        angles: np.ndarray,
        ranges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_mask = np.zeros(len(angles), dtype=bool)
        # 1) Update track width estimate when both side walls are observed.
        left_d = self.side_distance(angles, ranges, side_sign=1)
        right_d = self.side_distance(angles, ranges, side_sign=-1)
        observed_cutoff = 0.92 * self.scan_max_range_m

        left_obs = left_d is not None and left_d < observed_cutoff
        right_obs = right_d is not None and right_d < observed_cutoff

        if left_obs and right_obs:
            measured_w = float(np.clip(left_d + right_d, 0.6, 3.5))
            self.track_width_est_m = (
                self.track_width_alpha * self.track_width_est_m
                + (1.0 - self.track_width_alpha) * measured_w
            )
            return angles, ranges, base_mask

        # 2) If one wall is missing, synthesize the opposite wall from width assumption.
        missing_sign = 0
        known_d = None
        if left_obs and not right_obs:
            missing_sign = -1
            known_d = left_d
        elif right_obs and not left_obs:
            missing_sign = 1
            known_d = right_d

        if missing_sign == 0 or known_d is None:
            return angles, ranges, base_mask

        virtual_d = float(np.clip(
            self.track_width_est_m - known_d,
            self.virtual_wall_min_m,
            self.scan_max_range_m,
        ))

        center = math.radians(self.side_angle_center_deg) * float(missing_sign)
        span = math.radians(self.virtual_wall_span_deg)
        n_pts = max(5, self.virtual_wall_num_points)
        synthetic_angles = np.linspace(center - 0.5 * span, center + 0.5 * span, n_pts)
        synthetic_ranges = self.build_smooth_virtual_ranges(
            angles=angles,
            ranges=ranges,
            synthetic_angles=synthetic_angles,
            base_distance=virtual_d,
            observed_cutoff=observed_cutoff,
        )

        aug_angles = np.concatenate([angles, synthetic_angles])
        aug_ranges = np.concatenate([ranges, synthetic_ranges])
        aug_mask = np.concatenate([np.zeros(len(angles), dtype=bool), np.ones(n_pts, dtype=bool)])
        order = np.argsort(aug_angles)
        return aug_angles[order], aug_ranges[order], aug_mask[order]

    def nearest_anchor_distance(
        self,
        target_angle: float,
        angles: np.ndarray,
        ranges: np.ndarray,
        observed_cutoff: float,
    ) -> Optional[float]:
        if len(angles) == 0:
            return None
        idx = int(np.argmin(np.abs(angles - target_angle)))
        ang_err = abs(float(angles[idx]) - float(target_angle))
        max_gap = math.radians(self.virtual_wall_anchor_max_gap_deg)
        r_val = float(ranges[idx])
        if ang_err > max_gap or not np.isfinite(r_val) or r_val >= observed_cutoff:
            return None
        return r_val

    def build_smooth_virtual_ranges(
        self,
        angles: np.ndarray,
        ranges: np.ndarray,
        synthetic_angles: np.ndarray,
        base_distance: float,
        observed_cutoff: float,
    ) -> np.ndarray:
        n_pts = len(synthetic_angles)
        profile = np.full(n_pts, float(base_distance), dtype=float)
        if n_pts < 3:
            return profile

        left_anchor = self.nearest_anchor_distance(
            target_angle=float(synthetic_angles[0]),
            angles=angles,
            ranges=ranges,
            observed_cutoff=observed_cutoff,
        )
        right_anchor = self.nearest_anchor_distance(
            target_angle=float(synthetic_angles[-1]),
            angles=angles,
            ranges=ranges,
            observed_cutoff=observed_cutoff,
        )

        t = np.linspace(0.0, 1.0, n_pts, dtype=float)
        if left_anchor is not None:
            profile += (float(left_anchor) - float(base_distance)) * ((1.0 - t) ** 2)
        if right_anchor is not None:
            profile += (float(right_anchor) - float(base_distance)) * (t ** 2)

        # Blend toward any real observed points inside synthetic sector.
        sec_min = float(np.min(synthetic_angles))
        sec_max = float(np.max(synthetic_angles))
        in_sector = (
            (angles >= sec_min)
            & (angles <= sec_max)
            & np.isfinite(ranges)
            & (ranges < observed_cutoff)
        )
        if int(np.sum(in_sector)) >= 2:
            obs_ang = angles[in_sector]
            obs_rng = ranges[in_sector]
            order = np.argsort(obs_ang)
            obs_ang = obs_ang[order]
            obs_rng = obs_rng[order]
            interp_rng = np.interp(
                synthetic_angles,
                obs_ang,
                obs_rng,
                left=float(obs_rng[0]),
                right=float(obs_rng[-1]),
            )
            alpha = float(np.clip(self.virtual_wall_blend_alpha, 0.0, 1.0))
            profile = (1.0 - alpha) * profile + alpha * interp_rng

        smooth_w = max(1, int(self.virtual_wall_smooth_window))
        if smooth_w > 1 and n_pts >= smooth_w:
            if (smooth_w % 2) == 0:
                smooth_w += 1
            if smooth_w == 5:
                kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
            else:
                half = smooth_w // 2
                ramp = np.arange(1, half + 2, dtype=float)
                kernel = np.concatenate([ramp, ramp[-2::-1]])
            kernel /= np.sum(kernel)
            pad = smooth_w // 2
            padded = np.pad(profile, (pad, pad), mode='edge')
            profile = np.convolve(padded, kernel, mode='valid')

        max_step = max(0.01, float(self.virtual_wall_max_step_m))
        for i in range(1, n_pts):
            low = profile[i - 1] - max_step
            high = profile[i - 1] + max_step
            profile[i] = float(np.clip(profile[i], low, high))
        for i in range(n_pts - 2, -1, -1):
            low = profile[i + 1] - max_step
            high = profile[i + 1] + max_step
            profile[i] = float(np.clip(profile[i], low, high))

        profile = np.clip(profile, self.virtual_wall_min_m, self.scan_max_range_m)
        return profile

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

        if self.show_opencv_debug:
            cv2.imshow('Live Mapper Debug', canvas)
            cv2.waitKey(1)

        if self.publish_mapper_debug_images and self.mapper_debug_image_pub is not None:
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = ''
            msg.height = int(canvas.shape[0])
            msg.width = int(canvas.shape[1])
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = int(canvas.shape[1] * canvas.shape[2])
            msg.data = canvas.tobytes()
            self.mapper_debug_image_pub.publish(msg)


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
