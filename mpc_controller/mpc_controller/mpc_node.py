#!/usr/bin/env python3
"""
ROS 2 F1TENTH safer full MPC with:
- full LiDAR display window decoupled from planner horizon
- cluster-based corridor extraction from a broad forward FOV
- raw green corridor used for reference geometry
- orange bounds used only as safety constraints
- later turn-in / outside-bias reference shaping
- obstacle and wall aware MPC with slack
- OpenCV visualizer
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import osqp
import scipy.sparse as sp

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from dev_b7_interfaces.msg import DriveControlMessage
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, LaserScan

from mpc_controller.gap_utils import GapFollowAlgo


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float


class MPCNode(Node):
    def __init__(self) -> None:
        super().__init__('full_mpc_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('odom_topic', '/ego_racecar/odom'),
                ('scan_topic', '/scan'),
                ('race_line_topic', '/race_line/global_path'),
                ('control_rate_hz', 30.0),
                ('sim', True),
                ('use_race_line_planner', True),
                ('path_stale_timeout_s', 2.0),
                ('max_path_lateral_error_m', 2.0),

                ('dt', 0.05),
                ('horizon', 13),
                ('wheelbase', 0.50),

                ('max_speed', 3.5),
                ('min_speed', 0.0),
                ('straight_speed', 3.5),
                ('corner_speed_cap', 2.0),
                ('hard_stop_distance', 0.32),

                ('max_accel', 10.0),
                ('min_accel', -5.0),
                ('max_steer', 0.36),
                ('max_ddelta', 0.024),

                ('use_odom_speed', True),
                ('print_timing_every', 10),

                ('scan_y_sign', 1.0),
                ('steering_output_sign', 1.0),

                # FTG
                ('ftg_max_range', 3.5),
                ('ftg_min_safe_distance', 0.25),
                ('ftg_car_width', 0.35),
                ('ftg_disparity_threshold', 0.7),
                ('ftg_smoothing_window_size', 10),
                ('ftg_use_disparity_extender', False),

                # Goal filtering
                ('startup_straight_frames', 5),
                ('goal_min_distance', 0.45),
                ('goal_max_distance', 2.6),
                ('goal_filter_alpha_x', 0.72),
                ('goal_filter_alpha_y', 0.90),
                ('goal_max_step_x', 0.08),
                ('goal_max_step_y', 0.030),
                ('goal_lateral_deadband', 0.04),

                # Effective gap distance cap from front clearance
                ('effective_goal_base', 0.45),
                ('effective_goal_front_gain', 0.85),

                # Planner corridor
                ('path_x_max', 1.65),
                ('path_y_limit', 1.7),
                ('corridor_bin_half_width', 0.18),
                ('corridor_margin', 0.08),
                ('corridor_min_half_width', 0.18),
                ('corridor_smooth_passes', 2),
                # How much to smooth the planner reference path in the local frame.
                # 0 keeps the planner reference as-is.
                ('planner_ref_smooth_passes', 1),
                ('margin_speed_gain', 0.015),
                ('margin_steer_gain', 0.03),

                # Corridor extraction improvements
                ('corridor_dense_points', 50),
                ('corridor_front_fov_deg', 180.0),
                ('corridor_min_points_per_bin', 10),

                # Display-only limits
                ('display_x_max', 10.5),
                ('display_y_limit', 5.2),

                # Lookahead
                ('lookahead_base', 0.60),
                ('lookahead_front_gain', 0.55),
                ('lookahead_speed_gain', 0.05),

                # Gap guidance / late-apex shaping
                ('gap_influence_max', 0.14),
                ('gap_target_max_frac', 0.22),
                ('early_center_lock_steps', 3),
                ('gap_heading_front_cap_min', 0.10),
                ('gap_heading_front_cap_max', 0.28),
                ('outside_bias_gain', 0.18),
                ('outside_bias_max_frac', 0.42),
                ('terminal_goal_blend_max', 0.10),

                # Speed shaping
                ('speed_target_angle_gain', 1.65),
                ('speed_curvature_gain', 2.5),
                ('speed_front_clearance_gain', 1.3),
                ('speed_width_gain', 1.5),

                # State tracking cost
                ('q_x', 10.0),
                ('q_y', 100.0),
                ('q_psi', 45.0),
                ('q_v', 8.0),

                # Terminal state cost
                ('qf_x', 18.0),
                ('qf_y', 220.0),
                ('qf_psi', 110.0),
                ('qf_v', 10.0),

                # Input cost
                ('r_a', 0.35),
                ('r_delta', 32.0),

                # Input rate cost
                ('rd_a', 0.40),
                ('rd_delta', 190.0),

                # Slack
                ('slack_weight', 12000.0),

                # OpenCV debug
                ('show_opencv_debug', True),
                # When sim=False, publish the same debug canvas as sensor_msgs/Image.
                ('publish_debug_images', True),
                ('mpc_debug_image_topic', '/mpc/debug_image'),
                ('debug_canvas_width', 1500),
                ('debug_canvas_height', 980),
                ('debug_pixels_per_meter', 170.0),
                ('debug_window_name', 'Safer Gap Full MPC Debug'),
            ],
        )

        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.race_line_topic = self.get_parameter('race_line_topic').value
        self.sim = bool(self.get_parameter('sim').value)
        self.use_race_line_planner = bool(self.get_parameter('use_race_line_planner').value)
        self.path_stale_timeout_s = float(self.get_parameter('path_stale_timeout_s').value)
        self.max_path_lateral_error_m = float(self.get_parameter('max_path_lateral_error_m').value)

        self.dt = float(self.get_parameter('dt').value)
        self.N = int(self.get_parameter('horizon').value)
        self.L = float(self.get_parameter('wheelbase').value)
        self.invL = 1.0 / self.L

        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.straight_speed = float(self.get_parameter('straight_speed').value)
        self.corner_speed_cap = float(self.get_parameter('corner_speed_cap').value)
        self.hard_stop_distance = float(self.get_parameter('hard_stop_distance').value)

        self.max_accel = float(self.get_parameter('max_accel').value)
        self.min_accel = float(self.get_parameter('min_accel').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.max_ddelta = float(self.get_parameter('max_ddelta').value)

        self.use_odom_speed = bool(self.get_parameter('use_odom_speed').value)
        self.print_timing_every = int(self.get_parameter('print_timing_every').value)

        self.scan_y_sign = float(self.get_parameter('scan_y_sign').value)
        self.steering_output_sign = float(self.get_parameter('steering_output_sign').value)

        self.startup_straight_frames = int(self.get_parameter('startup_straight_frames').value)
        self.goal_min_distance = float(self.get_parameter('goal_min_distance').value)
        self.goal_max_distance = float(self.get_parameter('goal_max_distance').value)
        self.goal_filter_alpha_x = float(self.get_parameter('goal_filter_alpha_x').value)
        self.goal_filter_alpha_y = float(self.get_parameter('goal_filter_alpha_y').value)
        self.goal_max_step_x = float(self.get_parameter('goal_max_step_x').value)
        self.goal_max_step_y = float(self.get_parameter('goal_max_step_y').value)
        self.goal_lateral_deadband = float(self.get_parameter('goal_lateral_deadband').value)

        self.effective_goal_base = float(self.get_parameter('effective_goal_base').value)
        self.effective_goal_front_gain = float(self.get_parameter('effective_goal_front_gain').value)

        self.path_x_max = float(self.get_parameter('path_x_max').value)
        self.path_y_limit = float(self.get_parameter('path_y_limit').value)
        self.corridor_bin_half_width = float(self.get_parameter('corridor_bin_half_width').value)
        self.corridor_margin = float(self.get_parameter('corridor_margin').value)
        self.corridor_min_half_width = float(self.get_parameter('corridor_min_half_width').value)
        self.corridor_smooth_passes = int(self.get_parameter('corridor_smooth_passes').value)
        self.planner_ref_smooth_passes = int(self.get_parameter('planner_ref_smooth_passes').value)
        self.margin_speed_gain = float(self.get_parameter('margin_speed_gain').value)
        self.margin_steer_gain = float(self.get_parameter('margin_steer_gain').value)

        self.corridor_dense_points = int(self.get_parameter('corridor_dense_points').value)
        self.corridor_front_fov_deg = float(self.get_parameter('corridor_front_fov_deg').value)
        self.corridor_min_points_per_bin = int(self.get_parameter('corridor_min_points_per_bin').value)

        self.display_x_max = float(self.get_parameter('display_x_max').value)
        self.display_y_limit = float(self.get_parameter('display_y_limit').value)

        self.lookahead_base = float(self.get_parameter('lookahead_base').value)
        self.lookahead_front_gain = float(self.get_parameter('lookahead_front_gain').value)
        self.lookahead_speed_gain = float(self.get_parameter('lookahead_speed_gain').value)

        self.gap_influence_max = float(self.get_parameter('gap_influence_max').value)
        self.gap_target_max_frac = float(self.get_parameter('gap_target_max_frac').value)
        self.early_center_lock_steps = int(self.get_parameter('early_center_lock_steps').value)
        self.gap_heading_front_cap_min = float(self.get_parameter('gap_heading_front_cap_min').value)
        self.gap_heading_front_cap_max = float(self.get_parameter('gap_heading_front_cap_max').value)
        self.outside_bias_gain = float(self.get_parameter('outside_bias_gain').value)
        self.outside_bias_max_frac = float(self.get_parameter('outside_bias_max_frac').value)
        self.terminal_goal_blend_max = float(self.get_parameter('terminal_goal_blend_max').value)

        self.speed_target_angle_gain = float(self.get_parameter('speed_target_angle_gain').value)
        self.speed_curvature_gain = float(self.get_parameter('speed_curvature_gain').value)
        self.speed_front_clearance_gain = float(self.get_parameter('speed_front_clearance_gain').value)
        self.speed_width_gain = float(self.get_parameter('speed_width_gain').value)

        self.q_x = float(self.get_parameter('q_x').value)
        self.q_y = float(self.get_parameter('q_y').value)
        self.q_psi = float(self.get_parameter('q_psi').value)
        self.q_v = float(self.get_parameter('q_v').value)

        self.qf_x = float(self.get_parameter('qf_x').value)
        self.qf_y = float(self.get_parameter('qf_y').value)
        self.qf_psi = float(self.get_parameter('qf_psi').value)
        self.qf_v = float(self.get_parameter('qf_v').value)

        self.r_a = float(self.get_parameter('r_a').value)
        self.r_delta = float(self.get_parameter('r_delta').value)
        self.rd_a = float(self.get_parameter('rd_a').value)
        self.rd_delta = float(self.get_parameter('rd_delta').value)

        self.slack_weight = float(self.get_parameter('slack_weight').value)

        self.show_opencv_debug = bool(self.get_parameter('show_opencv_debug').value)
        self.show_opencv_debug = self.show_opencv_debug and self.sim
        self.publish_mpc_debug_images = (
            (not self.sim) and bool(self.get_parameter('publish_debug_images').value)
        )
        self.mpc_debug_image_topic = str(self.get_parameter('mpc_debug_image_topic').value)
        self.mpc_debug_image_pub = None
        if self.publish_mpc_debug_images:
            self.mpc_debug_image_pub = self.create_publisher(Image, self.mpc_debug_image_topic, 1)
        self.debug_canvas_width = int(self.get_parameter('debug_canvas_width').value)
        self.debug_canvas_height = int(self.get_parameter('debug_canvas_height').value)
        self.debug_pixels_per_meter = float(self.get_parameter('debug_pixels_per_meter').value)
        self.debug_window_name = str(self.get_parameter('debug_window_name').value)

        self.gap_algo = GapFollowAlgo(
            max_range=float(self.get_parameter('ftg_max_range').value),
            min_safe_distance=float(self.get_parameter('ftg_min_safe_distance').value),
            car_width=float(self.get_parameter('ftg_car_width').value),
            disparity_threshold=float(self.get_parameter('ftg_disparity_threshold').value),
            smoothing_window_size=int(self.get_parameter('ftg_smoothing_window_size').value),
            use_disparity_extender=bool(self.get_parameter('ftg_use_disparity_extender').value),
        )

        # f1tenth_gym_ros (and many sim bridges) use default reliable publishers; BEST_EFFORT
        # subscriptions do not match and never receive data.
        qos_sensors = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos_sensors)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_sensors)
        self.race_line_sub = self.create_subscription(Path, self.race_line_topic, self.race_line_callback, 1)
        self.drive_pub = self.create_publisher(
            DriveControlMessage,
            DriveControlMessage.BUILTIN_TOPIC_NAME_STRING,
            10,
        )
        self.timer = self.create_timer(
            1.0 / float(self.get_parameter('control_rate_hz').value),
            self.control_callback,
        )

        self.state: Optional[VehicleState] = None
        self.last_scan: Optional[LaserScan] = None
        self.race_path_xy: Optional[np.ndarray] = None
        self.race_path_s: Optional[np.ndarray] = None
        self.race_path_yaw: Optional[np.ndarray] = None
        self.race_path_total_s: float = 0.0
        self.race_path_stamp_s: float = 0.0

        self.frame_count = 0
        self.solve_count = 0
        self.solve_time_ms_last = 0.0
        self.solve_time_ms_ema = 0.0

        self.filtered_goal_x = 1.0
        self.filtered_goal_y = 0.0

        self.prev_a_cmd = 0.0
        self.prev_delta_cmd_internal = 0.0

        # debug / visualizer
        self.last_corridor = None
        self.last_ref = None
        self.last_pred = None
        self.last_u_pred = None
        self.last_goal_local = None
        self.last_gap_angle = 0.0
        self.last_gap_distance = 0.0
        self.last_front_min = 0.0
        self.last_min_width = 0.0
        self.last_gap_alpha = 0.0
        self.last_safe_margin = 0.0
        self.last_centerline = None
        self.last_base_ref = None
        self.last_gap_line = None
        self.last_safe_gap_angle = 0.0
        self.last_global_path_local = None
        self.last_path_horizon_local = None
        self.last_ref_source = 'ftg'

        if self.show_opencv_debug:
            cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.debug_window_name, self.debug_canvas_width, self.debug_canvas_height)

        self.get_logger().info('Safer gap-aware full MPC node started.')

    # ──────────────────────────────────────────────────────────────────
    # ROS callbacks
    # ──────────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        if self.use_odom_speed:
            speed = float(math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))
        else:
            speed = 0.0

        self.state = VehicleState(x=x, y=y, yaw=yaw, speed=speed)

    def scan_callback(self, msg: LaserScan) -> None:
        self.last_scan = msg

    def race_line_callback(self, msg: Path) -> None:
        n = len(msg.poses)
        if n < 8:
            return

        pts = np.array(
            [[float(p.pose.position.x), float(p.pose.position.y)] for p in msg.poses],
            dtype=float,
        )
        if pts.shape[0] < 8:
            return

        step = np.roll(pts, -1, axis=0) - pts
        ds = np.linalg.norm(step, axis=1)
        total = float(np.sum(ds))
        if total < 1.0:
            return

        s = np.zeros(n, dtype=float)
        if n > 1:
            s[1:] = np.cumsum(ds[:-1])

        yaw = np.arctan2(step[:, 1], step[:, 0])

        stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp_s <= 1e-9:
            stamp_s = self.now_seconds()

        self.race_path_xy = pts
        self.race_path_s = s
        self.race_path_yaw = yaw
        self.race_path_total_s = total
        self.race_path_stamp_s = stamp_s

    def now_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    # ──────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────

    def control_callback(self) -> None:
        if self.state is None or self.last_scan is None:
            return

        self.frame_count += 1

        ranges = np.array(self.last_scan.ranges, dtype=float)
        target_angle, _, target_distance = self.gap_algo.process_lidar_and_find_gap(
            ranges,
            float(self.last_scan.angle_min),
            float(self.last_scan.angle_increment),
        )
        target_angle = self.scan_y_sign * float(target_angle)
        target_distance = float(target_distance)

        front_min = float(getattr(self.gap_algo, 'last_front_min', target_distance))
        self.last_front_min = front_min

        goal_local = self.make_filtered_goal(target_angle, target_distance, front_min)
        self.last_goal_local = goal_local.copy()
        self.last_gap_angle = float(target_angle)
        self.last_gap_distance = float(target_distance)

        problem = None
        if self.use_race_line_planner:
            problem = self.build_race_path_problem(target_angle, target_distance, front_min)
            if problem is not None:
                self.last_ref_source = 'planner'

        if problem is None:
            problem = self.build_local_mpc_problem(goal_local, target_angle, target_distance, front_min)
            self.last_ref_source = 'ftg'

        if problem is None:
            v_cmd = 0.0
            delta_cmd = self.steering_output_sign * self.prev_delta_cmd_internal
            self.publish_drive(v_cmd, delta_cmd)
            if self.show_opencv_debug or self.publish_mpc_debug_images:
                self.draw_debug_canvas(v_cmd, delta_cmd)
            return

        v_cmd, delta_cmd, pred_states, solve_ms = self.solve_full_mpc(problem)
        self.last_pred = pred_states

        if front_min < self.hard_stop_distance:
            v_cmd = 0.0
        elif front_min < 0.42:
            v_cmd = min(v_cmd, 0.40)
        elif front_min < 0.55:
            v_cmd = min(v_cmd, 0.60)
        elif front_min < 0.70:
            v_cmd = min(v_cmd, 0.85)

        self.solve_time_ms_last = solve_ms
        self.solve_time_ms_ema = (
            solve_ms if self.solve_count == 0
            else 0.9 * self.solve_time_ms_ema + 0.1 * solve_ms
        )
        self.solve_count += 1

        if self.solve_count % max(1, self.print_timing_every) == 0:
            self.get_logger().info(
                f'full MPC solve: last={self.solve_time_ms_last:.2f} ms  '
                f'ema={self.solve_time_ms_ema:.2f} ms'
            )

        self.get_logger().info(
            f'cmd v={v_cmd:.3f} delta={delta_cmd:.3f}  '
            f'gap_ang={target_angle:.3f} gap_d={target_distance:.3f}  '
            f'front_min={front_min:.3f}  goal=({goal_local[0]:.3f},{goal_local[1]:.3f})  '
            f'min_width={self.last_min_width:.3f} gap_alpha={self.last_gap_alpha:.3f} '
            f'source={self.last_ref_source}'
        )

        self.publish_drive(v_cmd, delta_cmd)

        if self.show_opencv_debug or self.publish_mpc_debug_images:
            self.draw_debug_canvas(v_cmd, delta_cmd)

    # ──────────────────────────────────────────────────────────────────
    # Goal filtering
    # ──────────────────────────────────────────────────────────────────

    def make_filtered_goal(
        self,
        target_angle: float,
        target_distance: float,
        front_min: float,
    ) -> np.ndarray:
        d_cap = self.effective_goal_base + self.effective_goal_front_gain * max(0.0, front_min)
        d_cap = float(np.clip(d_cap, self.goal_min_distance, self.goal_max_distance))
        d = float(np.clip(min(target_distance, d_cap), self.goal_min_distance, self.goal_max_distance))

        raw_x = max(0.35, d * math.cos(target_angle))
        raw_y = d * math.sin(target_angle)

        if self.frame_count <= self.startup_straight_frames:
            self.filtered_goal_x = max(0.8, raw_x)
            self.filtered_goal_y = 0.0
            return np.array([self.filtered_goal_x, self.filtered_goal_y], dtype=float)

        desired_x = (
            self.goal_filter_alpha_x * self.filtered_goal_x
            + (1.0 - self.goal_filter_alpha_x) * raw_x
        )
        dx = float(np.clip(
            desired_x - self.filtered_goal_x,
            -self.goal_max_step_x,
            self.goal_max_step_x,
        ))
        self.filtered_goal_x = max(0.35, self.filtered_goal_x + dx)

        lateral_error = raw_y - self.filtered_goal_y
        if abs(lateral_error) >= self.goal_lateral_deadband:
            desired_y = (
                self.goal_filter_alpha_y * self.filtered_goal_y
                + (1.0 - self.goal_filter_alpha_y) * raw_y
            )
            dy = float(np.clip(
                desired_y - self.filtered_goal_y,
                -self.goal_max_step_y,
                self.goal_max_step_y,
            ))
            self.filtered_goal_y += dy

        if abs(self.filtered_goal_y) < 0.03:
            self.filtered_goal_y = 0.0

        return np.array([self.filtered_goal_x, self.filtered_goal_y], dtype=float)

    def has_fresh_race_path(self) -> bool:
        if self.race_path_xy is None or self.race_path_s is None or self.race_path_yaw is None:
            return False
        age = self.now_seconds() - self.race_path_stamp_s
        return age <= self.path_stale_timeout_s

    def interpolate_path_xy(self, s_query: np.ndarray) -> np.ndarray:
        if self.race_path_xy is None or self.race_path_s is None:
            raise ValueError('Race path is not available')

        s = self.race_path_s
        xy = self.race_path_xy
        total = self.race_path_total_s
        if total <= 1e-6:
            raise ValueError('Race path length is too small')

        s_wrapped = np.mod(s_query, total)
        s_ext = np.concatenate([s, [total]])
        x_ext = np.concatenate([xy[:, 0], [xy[0, 0]]])
        y_ext = np.concatenate([xy[:, 1], [xy[0, 1]]])

        xq = np.interp(s_wrapped, s_ext, x_ext)
        yq = np.interp(s_wrapped, s_ext, y_ext)
        return np.column_stack((xq, yq))

    def global_to_local_xy(self, points_xy: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise ValueError('Vehicle state unavailable')

        dx = points_xy[:, 0] - self.state.x
        dy = points_xy[:, 1] - self.state.y
        c = math.cos(self.state.yaw)
        s = math.sin(self.state.yaw)

        x_local = c * dx + s * dy
        y_local = -s * dx + c * dy
        return np.column_stack((x_local, y_local))

    def build_race_path_problem(
        self,
        target_angle: float,
        target_distance: float,
        front_min: float,
    ) -> Optional[dict]:
        if self.state is None or not self.has_fresh_race_path():
            return None
        if self.race_path_xy is None or self.race_path_s is None or self.race_path_yaw is None:
            return None
        if self.race_path_total_s <= 1.0:
            return None

        pts = self.race_path_xy
        dxy = pts - np.array([self.state.x, self.state.y], dtype=float)
        dist = np.hypot(dxy[:, 0], dxy[:, 1])
        idx = int(np.argmin(dist))
        if float(dist[idx]) > self.max_path_lateral_error_m:
            return None

        heading_err = abs(self.wrap_angle(float(self.race_path_yaw[idx]) - self.state.yaw))
        if heading_err > 1.8:
            return None

        lookahead_x = self.lookahead_base
        lookahead_x += self.lookahead_front_gain * max(0.0, front_min)
        lookahead_x += self.lookahead_speed_gain * max(0.0, self.state.speed)
        lookahead_x -= 0.28 * min(1.0, abs(target_angle) / 0.55)
        lookahead_x = float(np.clip(lookahead_x, 0.65, self.path_x_max))

        s0 = float(self.race_path_s[idx])
        s_targets = s0 + np.linspace(0.0, lookahead_x, self.N + 1)

        global_horizon = self.interpolate_path_xy(s_targets)
        local_horizon = self.global_to_local_xy(global_horizon)

        x_ref = local_horizon[:, 0]
        y_ref = local_horizon[:, 1]

        # Reject path segments that do not project mostly in front of the car.
        if int(np.sum(x_ref > -0.05)) < max(4, int(0.65 * len(x_ref))):
            return None

        x_ref = np.maximum.accumulate(x_ref)
        x_ref = x_ref - x_ref[0]
        x_ref = np.clip(x_ref, 0.0, self.path_x_max + 0.25)
        y_ref = self.smooth_1d(y_ref, passes=self.planner_ref_smooth_passes)

        psi_ref = self.compute_heading_from_xy(x_ref, y_ref)
        delta_ref = self.compute_delta_ref(x_ref, y_ref, psi_ref)

        v_target = self.compute_target_speed(
            target_angle=target_angle,
            target_distance=target_distance,
            front_min=front_min,
            min_width=2.0 * self.path_y_limit,
            delta_ref=delta_ref,
        )
        v_ref = np.full(self.N + 1, v_target, dtype=float)

        corridor = self.estimate_corridor(x_ref)
        if corridor is not None:
            y_left_ref, y_right_ref, dense = corridor
            safe_margin = self.corridor_margin
            safe_margin += self.margin_speed_gain * max(0.0, self.state.speed)
            safe_margin += self.margin_steer_gain * abs(self.prev_delta_cmd_internal)
            safe_margin = float(np.clip(safe_margin, self.corridor_margin, 0.20))

            lower = y_right_ref + safe_margin
            upper = y_left_ref - safe_margin
            lower, upper = self.make_bounds_feasible(lower, upper)

            self.last_corridor = {
                'x_ref': x_ref.copy(),
                'y_left_ref': y_left_ref.copy(),
                'y_right_ref': y_right_ref.copy(),
                'lower': lower.copy(),
                'upper': upper.copy(),
                'y_center': 0.5 * (y_left_ref + y_right_ref),
                'x_dense': dense['x_dense'].copy(),
                'y_left_dense': dense['y_left_dense'].copy(),
                'y_right_dense': dense['y_right_dense'].copy(),
                'x_sector_pts': dense['x_sector_pts'].copy(),
                'y_sector_pts': dense['y_sector_pts'].copy(),
                'cluster_centers': np.array(
                    dense.get('cluster_centers', np.zeros((0, 2), dtype=float)),
                    dtype=float,
                ).copy(),
                'left_cluster_center': None if dense.get('left_cluster_center') is None else np.array(
                    dense['left_cluster_center'],
                    dtype=float,
                ).copy(),
                'right_cluster_center': None if dense.get('right_cluster_center') is None else np.array(
                    dense['right_cluster_center'],
                    dtype=float,
                ).copy(),
            }
        else:
            half_span = min(self.path_y_limit - 0.05, 0.95)
            lower = np.clip(y_ref - half_span, -self.path_y_limit, self.path_y_limit)
            upper = np.clip(y_ref + half_span, -self.path_y_limit, self.path_y_limit)
            lower, upper = self.make_bounds_feasible(lower, upper)
            self.last_corridor = None

        z_ref = np.column_stack((x_ref, y_ref, psi_ref, v_ref))
        u_ref = np.column_stack((np.zeros(self.N, dtype=float), delta_ref))

        self.last_ref = z_ref.copy()
        self.last_path_horizon_local = np.column_stack((x_ref, y_ref))

        all_local = self.global_to_local_xy(self.race_path_xy)
        keep = (
            (all_local[:, 0] >= -1.0)
            & (all_local[:, 0] <= self.display_x_max)
            & (np.abs(all_local[:, 1]) <= self.display_y_limit)
        )
        self.last_global_path_local = all_local[keep]

        return {
            'lower': lower,
            'upper': upper,
            'z_ref': z_ref,
            'u_ref': u_ref,
        }

    # Corridor + nominal reference
    def build_local_mpc_problem(
        self,
        goal_local: np.ndarray,
        target_angle: float,
        target_distance: float,
        front_min: float,
    ) -> Optional[dict]:
        self.last_path_horizon_local = None

        if self.race_path_xy is not None and self.state is not None:
            all_local = self.global_to_local_xy(self.race_path_xy)
            keep = (
                (all_local[:, 0] >= -1.0)
                & (all_local[:, 0] <= self.display_x_max)
                & (np.abs(all_local[:, 1]) <= self.display_y_limit)
            )
            self.last_global_path_local = all_local[keep]
        else:
            self.last_global_path_local = None

        safe_gap_angle = self.compute_safe_gap_angle(target_angle, front_min)
        self.last_safe_gap_angle = safe_gap_angle

        lookahead_x = self.lookahead_base
        lookahead_x += self.lookahead_front_gain * max(0.0, front_min)
        lookahead_x += self.lookahead_speed_gain * max(0.0, self.state.speed)
        lookahead_x -= 0.28 * min(1.0, abs(target_angle) / 0.55)
        lookahead_x = min(lookahead_x, goal_local[0] + 0.15, self.path_x_max)
        lookahead_x = float(np.clip(lookahead_x, 0.65, self.path_x_max))

        x_ref = np.linspace(0.0, lookahead_x, self.N + 1)

        corridor = self.estimate_corridor(x_ref)
        if corridor is None:
            return None

        y_left_ref, y_right_ref, dense = corridor

        raw_width = y_left_ref - y_right_ref
        min_width_raw = float(np.min(raw_width))
        self.last_min_width = min_width_raw

        safe_margin = self.corridor_margin
        safe_margin += self.margin_speed_gain * max(0.0, self.state.speed)
        safe_margin += self.margin_steer_gain * abs(self.prev_delta_cmd_internal)
        safe_margin = float(np.clip(safe_margin, self.corridor_margin, 0.20))
        self.last_safe_margin = safe_margin

        lower = y_right_ref + safe_margin
        upper = y_left_ref - safe_margin
        lower, upper = self.make_bounds_feasible(lower, upper)

        y_center_raw = 0.5 * (y_left_ref + y_right_ref)
        self.last_centerline = np.column_stack((x_ref, y_center_raw))

        y_ref = self.build_guided_y_reference(
            x_ref=x_ref,
            lower=lower,
            upper=upper,
            y_center=y_center_raw,
            goal_local=goal_local,
            target_angle=target_angle,
            safe_gap_angle=safe_gap_angle,
            front_min=front_min,
            min_width=min_width_raw,
        )

        psi_ref = self.compute_heading_from_xy(x_ref, y_ref)
        delta_ref = self.compute_delta_ref(x_ref, y_ref, psi_ref)

        v_target = self.compute_target_speed(
            target_angle=target_angle,
            target_distance=target_distance,
            front_min=front_min,
            min_width=min_width_raw,
            delta_ref=delta_ref,
        )
        v_ref = np.full(self.N + 1, v_target, dtype=float)

        z_ref = np.column_stack((x_ref, y_ref, psi_ref, v_ref))
        u_ref = np.column_stack((np.zeros(self.N, dtype=float), delta_ref))

        self.last_corridor = {
            'x_ref': x_ref.copy(),
            'y_left_ref': y_left_ref.copy(),
            'y_right_ref': y_right_ref.copy(),
            'lower': lower.copy(),
            'upper': upper.copy(),
            'y_center': y_center_raw.copy(),
            'x_dense': dense['x_dense'].copy(),
            'y_left_dense': dense['y_left_dense'].copy(),
            'y_right_dense': dense['y_right_dense'].copy(),
            'x_sector_pts': dense['x_sector_pts'].copy(),
            'y_sector_pts': dense['y_sector_pts'].copy(),
            'cluster_centers': np.array(dense.get('cluster_centers', np.zeros((0, 2), dtype=float)), dtype=float).copy(),
            'left_cluster_center': None if dense.get('left_cluster_center') is None else np.array(dense['left_cluster_center'], dtype=float).copy(),
            'right_cluster_center': None if dense.get('right_cluster_center') is None else np.array(dense['right_cluster_center'], dtype=float).copy(),
        }
        self.last_ref = z_ref.copy()

        return {
            'lower': lower,
            'upper': upper,
            'z_ref': z_ref,
            'u_ref': u_ref,
        }

    def build_guided_y_reference(
        self,
        x_ref: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        y_center: np.ndarray,
        goal_local: np.ndarray,
        target_angle: float,
        safe_gap_angle: float,
        front_min: float,
        min_width: float,
    ) -> np.ndarray:
        width = upper - lower
        half_width = 0.5 * width
        progress = np.linspace(0.0, 1.0, len(x_ref))

        front_scale = float(np.clip((front_min - self.hard_stop_distance) / 1.1, 0.0, 1.0))
        width_scale = float(np.clip((min_width - 0.28) / 0.55, 0.0, 1.0))

        gap_line = np.tan(safe_gap_angle) * x_ref
        self.last_gap_line = np.column_stack((x_ref, gap_line))

        outside_dir = -np.sign(target_angle)
        outside_strength = self.outside_bias_gain * abs(target_angle) * (1.15 - 0.75 * front_scale)
        outside_strength = float(np.clip(outside_strength, 0.0, 0.35))
        outside_cap = self.outside_bias_max_frac * half_width
        outside_profile = outside_dir * np.minimum(outside_strength, outside_cap) * ((1.0 - progress) ** 2.3)

        base_ref = y_center + outside_profile
        base_ref = np.clip(base_ref, lower + 0.01, upper - 0.01)
        self.last_base_ref = np.column_stack((x_ref, base_ref))

        turn_in_start = 0.55
        turn_in_start += 0.15 * min(1.0, abs(target_angle) / 0.60)
        turn_in_start += 0.08 * (1.0 - front_scale)
        turn_in_start = float(np.clip(turn_in_start, 0.55, 0.80))

        ramp = np.clip((progress - turn_in_start) / max(1e-6, 1.0 - turn_in_start), 0.0, 1.0)

        gap_alpha = self.gap_influence_max * front_scale * width_scale * (ramp ** 2.6)

        lock_k = min(self.early_center_lock_steps, len(gap_alpha))
        gap_alpha[:lock_k] = 0.0
        self.last_gap_alpha = float(gap_alpha[-1])

        gap_shift_limit = self.gap_target_max_frac * half_width
        gap_target = np.clip(gap_line, base_ref - gap_shift_limit, base_ref + gap_shift_limit)

        y_ref = (1.0 - gap_alpha) * base_ref + gap_alpha * gap_target

        terminal_goal_blend = self.terminal_goal_blend_max * front_scale * width_scale
        terminal_cap = min(float(self.gap_target_max_frac * half_width[-1]), 0.14)
        terminal_goal = float(np.clip(
            goal_local[1],
            y_center[-1] - terminal_cap,
            y_center[-1] + terminal_cap,
        ))

        if len(y_ref) >= 2:
            y_ref[-2] = 0.97 * y_ref[-2] + 0.03 * terminal_goal
        y_ref[-1] = (1.0 - terminal_goal_blend) * y_ref[-1] + terminal_goal_blend * terminal_goal

        y_ref = self.smooth_1d(y_ref, passes=1)
        y_ref[:lock_k] = base_ref[:lock_k]

        y_ref = np.clip(y_ref, lower + 0.01, upper - 0.01)
        return y_ref

    def compute_safe_gap_angle(self, target_angle: float, front_min: float) -> float:
        front_scale = float(np.clip((front_min - self.hard_stop_distance) / 1.1, 0.0, 1.0))
        angle_cap = self.gap_heading_front_cap_min + (
            self.gap_heading_front_cap_max - self.gap_heading_front_cap_min
        ) * front_scale
        return float(np.clip(target_angle, -angle_cap, angle_cap))

    def split_points_into_clusters(
        self,
        x_pts: np.ndarray,
        y_pts: np.ndarray,
        dist_thresh: float,
        min_points: int,
    ):
        if len(x_pts) == 0:
            return []

        pts = np.column_stack((x_pts, y_pts))
        clusters: list[np.ndarray] = []
        start = 0

        for i in range(1, len(pts)):
            dx = float(pts[i, 0] - pts[i - 1, 0])
            dy = float(pts[i, 1] - pts[i - 1, 1])
            if math.hypot(dx, dy) > dist_thresh:
                if i - start >= min_points:
                    clusters.append(pts[start:i].copy())
                start = i

        if len(pts) - start >= min_points:
            clusters.append(pts[start:].copy())

        return clusters

    def dense_bound_from_cluster(
        self,
        cluster_pts: Optional[np.ndarray],
        x_dense: np.ndarray,
        default: float,
    ) -> np.ndarray:
        y_dense = np.full(len(x_dense), np.nan, dtype=float)

        if cluster_pts is None or len(cluster_pts) == 0:
            return np.full(len(x_dense), default, dtype=float)

        xs = cluster_pts[:, 0]
        ys = cluster_pts[:, 1]

        min_pts = max(2, self.corridor_min_points_per_bin // 3)

        for k, xk in enumerate(x_dense):
            mask = np.abs(xs - xk) <= self.corridor_bin_half_width
            if int(np.sum(mask)) < min_pts:
                mask = np.abs(xs - xk) <= 1.8 * self.corridor_bin_half_width

            if not np.any(mask):
                continue

            y_dense[k] = float(np.mean(ys[mask]))

        y_dense = self.fill_dense_bound(y_dense, default=default)
        y_dense = np.clip(y_dense, -self.path_y_limit, self.path_y_limit)
        return y_dense

    def estimate_corridor(self, x_ref: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
        if self.gap_algo.last_angles is None or self.gap_algo.last_extended is None:
            return None

        angles = np.array(self.gap_algo.last_angles, dtype=float)
        ranges = np.array(self.gap_algo.last_extended, dtype=float)

        x_pts = ranges * np.cos(angles)
        y_pts = self.scan_y_sign * ranges * np.sin(angles)

        valid = np.isfinite(x_pts) & np.isfinite(y_pts)
        valid &= (x_pts >= 0.0) & (x_pts <= self.path_x_max + 0.05)

        front_fov = math.radians(self.corridor_front_fov_deg)
        valid &= (np.abs(angles) <= 0.5 * front_fov)

        x_use = x_pts[valid]
        y_use = y_pts[valid]

        if len(x_use) < 12:
            return None

        cluster_dist = max(0.10, 1.15 * self.corridor_bin_half_width)
        cluster_min_points = max(6, self.corridor_min_points_per_bin // 2)

        clusters = self.split_points_into_clusters(
            x_use,
            y_use,
            dist_thresh=cluster_dist,
            min_points=cluster_min_points,
        )

        if len(clusters) == 0:
            return None

        cluster_info = []
        for pts in clusters:
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            xmin = float(np.min(pts[:, 0]))
            xmax = float(np.max(pts[:, 0]))
            span = xmax - xmin
            cluster_info.append({
                'pts': pts,
                'cx': cx,
                'cy': cy,
                'xmin': xmin,
                'xmax': xmax,
                'span': span,
                'size': len(pts),
            })

        def cluster_score(info: dict) -> float:
            return (
                0.045 * float(info['size'])
                + 1.20 * float(info['span'])
                + 0.80 * float(info['xmax'])
                - 0.10 * abs(float(info['cy']))
            )

        left_candidates = [c for c in cluster_info if c['cy'] > 0.05]
        right_candidates = [c for c in cluster_info if c['cy'] < -0.05]

        left_best = max(left_candidates, key=cluster_score) if left_candidates else None
        right_best = max(right_candidates, key=cluster_score) if right_candidates else None

        if left_best is None and right_best is None:
            return None

        x_dense = np.linspace(0.0, self.path_x_max, self.corridor_dense_points)

        y_left_dense = self.dense_bound_from_cluster(
            None if left_best is None else left_best['pts'],
            x_dense,
            default=self.path_y_limit,
        )
        y_right_dense = self.dense_bound_from_cluster(
            None if right_best is None else right_best['pts'],
            x_dense,
            default=-self.path_y_limit,
        )

        for _ in range(max(0, self.corridor_smooth_passes)):
            y_left_dense = self.smooth_1d(y_left_dense, passes=1)
            y_right_dense = self.smooth_1d(y_right_dense, passes=1)
                    
        y_left_dense = np.clip(y_left_dense, -self.path_y_limit, self.path_y_limit)
        y_right_dense = np.clip(y_right_dense, -self.path_y_limit, self.path_y_limit)

        for k in range(len(x_dense)):
            if y_left_dense[k] - y_right_dense[k] < 2.0 * self.corridor_min_half_width:
                mid = 0.5 * (y_left_dense[k] + y_right_dense[k])
                y_left_dense[k] = min(self.path_y_limit, mid + self.corridor_min_half_width)
                y_right_dense[k] = max(-self.path_y_limit, mid - self.corridor_min_half_width)

        y_left_ref = np.interp(x_ref, x_dense, y_left_dense)
        y_right_ref = np.interp(x_ref, x_dense, y_right_dense)

        dense = {
            'x_dense': x_dense,
            'y_left_dense': y_left_dense,
            'y_right_dense': y_right_dense,
            'x_sector_pts': x_use,
            'y_sector_pts': y_use,
            'cluster_centers': np.array([[c['cx'], c['cy']] for c in cluster_info], dtype=float),
            'left_cluster_center': None if left_best is None else np.array([left_best['cx'], left_best['cy']], dtype=float),
            'right_cluster_center': None if right_best is None else np.array([right_best['cx'], right_best['cy']], dtype=float),
        }

        return y_left_ref, y_right_ref, dense

    def fill_dense_bound(self, arr: np.ndarray, default: float) -> np.ndarray:
        out = arr.copy()
        idx = np.arange(len(out))
        valid = np.isfinite(out)

        if not np.any(valid):
            return np.full_like(out, default)

        out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])

        if default > 0.0:
            out = np.minimum(out, default)
        else:
            out = np.maximum(out, default)
        return out

    def compute_heading_from_xy(self, x_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        psi = np.zeros(len(x_ref), dtype=float)

        for i in range(len(x_ref) - 1):
            j = min(i + 2, len(x_ref) - 1)
            dx = x_ref[j] - x_ref[i]
            dy = y_ref[j] - y_ref[i]
            psi[i] = math.atan2(dy, max(dx, 1e-6))

        psi[-1] = psi[-2] if len(psi) >= 2 else 0.0

        prefix = min(max(self.early_center_lock_steps - 2, 0), len(psi))
        if prefix > 0:
            psi[:prefix] = np.clip(psi[:prefix], -0.06, 0.06)

        return psi

    def compute_delta_ref(self, x_ref: np.ndarray, y_ref: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        delta_ref = np.zeros(self.N, dtype=float)
        for k in range(self.N):
            dx = x_ref[min(k + 1, self.N)] - x_ref[k]
            dy = y_ref[min(k + 1, self.N)] - y_ref[k]
            ds = max(1e-3, math.hypot(dx, dy))
            psi_next = psi_ref[min(k + 1, self.N)]
            psi_now = psi_ref[k]
            kappa = self.wrap_angle(psi_next - psi_now) / ds
            delta_ref[k] = float(np.clip(math.atan(self.L * kappa), -self.max_steer, self.max_steer))
        return delta_ref

    def compute_target_speed(
        self,
        target_angle: float,
        target_distance: float,
        front_min: float,
        min_width: float,
        delta_ref: np.ndarray,
    ) -> float:
        curvature_metric = float(np.max(np.abs(delta_ref))) if len(delta_ref) > 0 else 0.0

        v = self.straight_speed
        v -= self.speed_target_angle_gain * abs(target_angle)
        v -= self.speed_curvature_gain * curvature_metric

        v = min(v, 0.40 + self.speed_front_clearance_gain * max(0.0, front_min - 0.20) + 0.45)
        v = min(v, 0.45 + self.speed_width_gain * max(0.0, min_width - 0.22) + 0.35)

        if front_min < self.hard_stop_distance:
            v = 0.0

        return float(np.clip(v, self.min_speed, self.max_speed))

    # ──────────────────────────────────────────────────────────────────
    # Full MPC
    # ──────────────────────────────────────────────────────────────────

    def solve_full_mpc(self, problem: dict) -> Tuple[float, float, np.ndarray, float]:
        t0 = time.perf_counter()

        lower = problem['lower']
        upper = problem['upper']
        z_ref = problem['z_ref']
        u_ref = problem['u_ref']

        n_state = 4
        n_acc = self.N
        n_del = self.N
        n_slack = self.N + 1
        nZ = (self.N + 1) * n_state
        nvar = nZ + n_acc + n_del + n_slack

        idx_a0 = nZ
        idx_d0 = nZ + n_acc
        idx_s0 = nZ + n_acc + n_del

        def z_index(k: int, i: int) -> int:
            return k * n_state + i

        def a_index(k: int) -> int:
            return idx_a0 + k

        def d_index(k: int) -> int:
            return idx_d0 + k

        def s_index(k: int) -> int:
            return idx_s0 + k

        P = sp.lil_matrix((nvar, nvar))
        q = np.zeros(nvar, dtype=float)

        Q = np.array([self.q_x, self.q_y, self.q_psi, self.q_v], dtype=float)
        Qf = np.array([self.qf_x, self.qf_y, self.qf_psi, self.qf_v], dtype=float)

        for k in range(self.N + 1):
            W = Qf if k == self.N else Q
            for i in range(n_state):
                idx = z_index(k, i)
                P[idx, idx] += 2.0 * W[i]
                q[idx] += -2.0 * W[i] * z_ref[k, i]

        for k in range(self.N):
            ia = a_index(k)
            idel = d_index(k)

            P[ia, ia] += 2.0 * self.r_a
            q[ia] += -2.0 * self.r_a * u_ref[k, 0]

            P[idel, idel] += 2.0 * self.r_delta
            q[idel] += -2.0 * self.r_delta * u_ref[k, 1]

        for k in range(self.N):
            ia = a_index(k)
            idel = d_index(k)

            if k == 0:
                P[ia, ia] += 2.0 * self.rd_a
                q[ia] += -2.0 * self.rd_a * self.prev_a_cmd

                P[idel, idel] += 2.0 * self.rd_delta
                q[idel] += -2.0 * self.rd_delta * self.prev_delta_cmd_internal
            else:
                ia_prev = a_index(k - 1)
                idel_prev = d_index(k - 1)

                P[ia, ia] += 2.0 * self.rd_a
                P[ia_prev, ia_prev] += 2.0 * self.rd_a
                P[ia, ia_prev] += -2.0 * self.rd_a
                P[ia_prev, ia] += -2.0 * self.rd_a

                P[idel, idel] += 2.0 * self.rd_delta
                P[idel_prev, idel_prev] += 2.0 * self.rd_delta
                P[idel, idel_prev] += -2.0 * self.rd_delta
                P[idel_prev, idel] += -2.0 * self.rd_delta

        for k in range(self.N + 1):
            islk = s_index(k)
            P[islk, islk] += 2.0 * self.slack_weight

        rows = []
        cols = []
        data = []
        l_eq = []
        u_eq = []

        z0 = np.array([0.0, 0.0, 0.0, float(np.clip(self.state.speed, 0.0, self.max_speed))], dtype=float)

        for i in range(n_state):
            rows.append(i)
            cols.append(z_index(0, i))
            data.append(1.0)
            l_eq.append(z0[i])
            u_eq.append(z0[i])

        row_base = n_state

        for k in range(self.N):
            zk_bar = z_ref[k]
            uk_bar = u_ref[k]
            A, B, g = self.linearize_bicycle_dynamics(zk_bar, uk_bar)

            for i in range(n_state):
                rows.append(row_base + i)
                cols.append(z_index(k + 1, i))
                data.append(1.0)

                for j in range(n_state):
                    rows.append(row_base + i)
                    cols.append(z_index(k, j))
                    data.append(-A[i, j])

                rows.append(row_base + i)
                cols.append(a_index(k))
                data.append(-B[i, 0])

                rows.append(row_base + i)
                cols.append(d_index(k))
                data.append(-B[i, 1])

                l_eq.append(g[i])
                u_eq.append(g[i])

            row_base += n_state

        Aeq = sp.csc_matrix((data, (rows, cols)), shape=(row_base, nvar))
        l_eq = np.array(l_eq, dtype=float)
        u_eq = np.array(u_eq, dtype=float)

        inf = 1e8
        rows = []
        cols = []
        data = []
        l_in = []
        u_in = []
        row = 0

        for k in range(self.N):
            rows.append(row)
            cols.append(a_index(k))
            data.append(1.0)
            l_in.append(self.min_accel)
            u_in.append(self.max_accel)
            row += 1

        for k in range(self.N):
            rows.append(row)
            cols.append(d_index(k))
            data.append(1.0)
            l_in.append(-self.max_steer)
            u_in.append(self.max_steer)
            row += 1

        for k in range(self.N + 1):
            rows.append(row)
            cols.append(z_index(k, 3))
            data.append(1.0)
            l_in.append(0.0)
            u_in.append(self.max_speed)
            row += 1

        for k in range(self.N + 1):
            rows.append(row)
            cols.append(z_index(k, 2))
            data.append(1.0)
            l_in.append(-1.0)
            u_in.append(1.0)
            row += 1

        for k in range(self.N + 1):
            rows.append(row)
            cols.append(z_index(k, 0))
            data.append(1.0)
            l_in.append(0.0)
            u_in.append(self.path_x_max + 0.30)
            row += 1

        for k in range(self.N + 1):
            rows.append(row)
            cols.append(z_index(k, 1))
            data.append(1.0)
            rows.append(row)
            cols.append(s_index(k))
            data.append(-1.0)
            l_in.append(-inf)
            u_in.append(upper[k])
            row += 1

        for k in range(self.N + 1):
            rows.append(row)
            cols.append(z_index(k, 1))
            data.append(1.0)
            rows.append(row)
            cols.append(s_index(k))
            data.append(1.0)
            l_in.append(lower[k])
            u_in.append(inf)
            row += 1

        for k in range(self.N + 1):
            rows.append(row)
            cols.append(s_index(k))
            data.append(1.0)
            l_in.append(0.0)
            u_in.append(inf)
            row += 1

        for k in range(self.N):
            if k == 0:
                rows.append(row)
                cols.append(d_index(k))
                data.append(1.0)
                l_in.append(self.prev_delta_cmd_internal - self.max_ddelta)
                u_in.append(self.prev_delta_cmd_internal + self.max_ddelta)
                row += 1
            else:
                rows.append(row)
                cols.append(d_index(k))
                data.append(1.0)
                rows.append(row)
                cols.append(d_index(k - 1))
                data.append(-1.0)
                l_in.append(-self.max_ddelta)
                u_in.append(self.max_ddelta)
                row += 1

        Aineq = sp.csc_matrix((data, (rows, cols)), shape=(row, nvar))
        l_in = np.array(l_in, dtype=float)
        u_in = np.array(u_in, dtype=float)

        Aqp = sp.vstack([Aeq, Aineq], format='csc')
        l_qp = np.concatenate([l_eq, l_in])
        u_qp = np.concatenate([u_eq, u_in])

        solver = osqp.OSQP()
        solver.setup(
            P=P.tocsc(),
            q=q,
            A=Aqp,
            l=l_qp,
            u=u_qp,
            verbose=False,
            warm_start=True,
            polish=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=450,
        )
        res = solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        if res.x is None or res.info.status_val not in (1, 2):
            v_cmd = 0.0
            delta_internal = float(np.clip(self.prev_delta_cmd_internal, -self.max_steer, self.max_steer))
            delta_cmd = self.steering_output_sign * delta_internal
            pred = np.zeros((self.N + 1, 4), dtype=float)
            self.last_u_pred = np.zeros((self.N, 2), dtype=float)
            return v_cmd, delta_cmd, pred, solve_ms

        sol = res.x
        z_seq = sol[:nZ].reshape(self.N + 1, n_state)
        a_seq = sol[nZ:nZ + n_acc]
        d_seq = sol[nZ + n_acc:nZ + n_acc + n_del]

        self.last_u_pred = np.column_stack((a_seq.copy(), d_seq.copy()))

        a_cmd = float(np.clip(a_seq[0], self.min_accel, self.max_accel))
        delta_internal = float(np.clip(d_seq[0], -self.max_steer, self.max_steer))

        v_cmd = float(np.clip(z_seq[1, 3], self.min_speed, self.max_speed))
        delta_cmd = float(self.steering_output_sign * delta_internal)

        self.prev_a_cmd = a_cmd
        self.prev_delta_cmd_internal = delta_internal

        return v_cmd, delta_cmd, z_seq, solve_ms

    # ──────────────────────────────────────────────────────────────────
    # Dynamics
    # ──────────────────────────────────────────────────────────────────

    def linearize_bicycle_dynamics(
        self,
        xbar: np.ndarray,
        ubar: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, psi, v = [float(val) for val in xbar]
        a, delta = [float(val) for val in ubar]

        delta = float(np.clip(delta, -0.95 * self.max_steer, 0.95 * self.max_steer))

        c = math.cos(psi)
        s = math.sin(psi)
        t = math.tan(delta)
        sec2 = 1.0 / (math.cos(delta) ** 2)

        f = np.array([
            x + self.dt * v * c,
            y + self.dt * v * s,
            psi + self.dt * v * self.invL * t,
            v + self.dt * a,
        ], dtype=float)

        A = np.eye(4, dtype=float)
        A[0, 2] = -self.dt * v * s
        A[0, 3] = self.dt * c
        A[1, 2] = self.dt * v * c
        A[1, 3] = self.dt * s
        A[2, 3] = self.dt * self.invL * t

        B = np.zeros((4, 2), dtype=float)
        B[2, 1] = self.dt * v * self.invL * sec2
        B[3, 0] = self.dt

        g = f - A @ np.array([x, y, psi, v], dtype=float) - B @ np.array([a, delta], dtype=float)
        return A, B, g

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────

    def make_bounds_feasible(self, lower: np.ndarray, upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lower = lower.copy()
        upper = upper.copy()
        for k in range(len(lower)):
            if upper[k] < lower[k] + 0.03:
                mid = 0.5 * (upper[k] + lower[k])
                lower[k] = mid - 0.015
                upper[k] = mid + 0.015
        return lower, upper

    def smooth_1d(self, arr: np.ndarray, passes: int = 1) -> np.ndarray:
        out = arr.copy()
        for _ in range(max(0, passes)):
            if len(out) < 3:
                return out
            tmp = out.copy()
            for i in range(1, len(out) - 1):
                tmp[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
            out = tmp
        return out

    # Visualizer

    def draw_debug_canvas(self, v_cmd: float, delta_cmd: float) -> None:
        W, H = self.debug_canvas_width, self.debug_canvas_height
        top_h = int(0.62 * H)
        bot_h = H - top_h
        scale = self.debug_pixels_per_meter

        canvas = np.full((H, W, 3), (18, 18, 18), dtype=np.uint8)
        origin = np.array([W // 2, top_h - 70], dtype=float)

        def to_px(p: np.ndarray) -> Tuple[int, int]:
            return (
                int(round(origin[0] - float(p[1]) * scale)),
                int(round(origin[1] - float(p[0]) * scale)),
            )

        for xm in np.arange(0.0, self.display_x_max + 0.6, 0.5):
            cv2.line(
                canvas,
                to_px(np.array([xm, -self.display_y_limit])),
                to_px(np.array([xm, self.display_y_limit])),
                (35, 35, 35),
                1,
            )
        for ym in np.arange(-self.display_y_limit, self.display_y_limit + 0.01, 0.5):
            cv2.line(
                canvas,
                to_px(np.array([0.0, ym])),
                to_px(np.array([self.display_x_max, ym])),
                (35, 35, 35),
                1,
            )
        cv2.line(
            canvas,
            to_px(np.array([0.0, -self.display_y_limit])),
            to_px(np.array([0.0, self.display_y_limit])),
            (70, 70, 70),
            2,
        )
        cv2.line(
            canvas,
            to_px(np.array([0.0, 0.0])),
            to_px(np.array([self.display_x_max, 0.0])),
            (70, 70, 70),
            2,
        )

        if self.last_global_path_local is not None and len(self.last_global_path_local) >= 2:
            pth = np.array(self.last_global_path_local, dtype=float)
            for i in range(len(pth) - 1):
                cv2.line(canvas, to_px(pth[i]), to_px(pth[i + 1]), (150, 150, 150), 1)

        if self.last_path_horizon_local is not None and len(self.last_path_horizon_local) >= 2:
            hor = np.array(self.last_path_horizon_local, dtype=float)
            for i in range(len(hor) - 1):
                cv2.line(canvas, to_px(hor[i]), to_px(hor[i + 1]), (255, 255, 80), 2)

        if self.gap_algo.last_angles is not None and self.gap_algo.last_extended is not None:
            angles = np.array(self.gap_algo.last_angles, dtype=float)
            ranges = np.array(self.gap_algo.last_extended, dtype=float)
            xs = ranges * np.cos(angles)
            ys = self.scan_y_sign * ranges * np.sin(angles)
            valid = np.isfinite(xs) & np.isfinite(ys)
            valid &= (xs >= 0.0) & (xs <= self.display_x_max)
            valid &= (np.abs(ys) <= self.display_y_limit)
            for x, y in zip(xs[valid], ys[valid]):
                cv2.circle(canvas, to_px(np.array([x, y])), 2, (110, 110, 110), -1)

        if self.last_corridor is not None:
            x_dense = self.last_corridor['x_dense']
            y_left_dense = self.last_corridor['y_left_dense']
            y_right_dense = self.last_corridor['y_right_dense']
            x_ref = self.last_corridor['x_ref']
            lower = self.last_corridor['lower']
            upper = self.last_corridor['upper']
            y_center = self.last_corridor['y_center']

            for i in range(len(x_dense) - 1):
                cv2.line(
                    canvas,
                    to_px(np.array([x_dense[i], y_left_dense[i]])),
                    to_px(np.array([x_dense[i + 1], y_left_dense[i + 1]])),
                    (0, 180, 0),
                    2,
                )
                cv2.line(
                    canvas,
                    to_px(np.array([x_dense[i], y_right_dense[i]])),
                    to_px(np.array([x_dense[i + 1], y_right_dense[i + 1]])),
                    (0, 180, 0),
                    2,
                )

            for i in range(len(x_ref) - 1):
                cv2.line(
                    canvas,
                    to_px(np.array([x_ref[i], upper[i]])),
                    to_px(np.array([x_ref[i + 1], upper[i + 1]])),
                    (0, 120, 255),
                    1,
                )
                cv2.line(
                    canvas,
                    to_px(np.array([x_ref[i], lower[i]])),
                    to_px(np.array([x_ref[i + 1], lower[i + 1]])),
                    (0, 120, 255),
                    1,
                )
                cv2.line(
                    canvas,
                    to_px(np.array([x_ref[i], y_center[i]])),
                    to_px(np.array([x_ref[i + 1], y_center[i + 1]])),
                    (255, 160, 0),
                    1,
                )

            cluster_centers = np.array(self.last_corridor.get('cluster_centers', np.zeros((0, 2), dtype=float)), dtype=float)
            for cxy in cluster_centers:
                cv2.circle(canvas, to_px(cxy), 5, (180, 180, 255), 1)

            left_cluster_center = self.last_corridor.get('left_cluster_center')
            if left_cluster_center is not None:
                cv2.circle(canvas, to_px(np.array(left_cluster_center, dtype=float)), 7, (0, 255, 255), 2)

            right_cluster_center = self.last_corridor.get('right_cluster_center')
            if right_cluster_center is not None:
                cv2.circle(canvas, to_px(np.array(right_cluster_center, dtype=float)), 7, (255, 255, 0), 2)

        if self.last_base_ref is not None:
            base_ref = np.array(self.last_base_ref, dtype=float)
            for i in range(len(base_ref) - 1):
                cv2.line(canvas, to_px(base_ref[i]), to_px(base_ref[i + 1]), (120, 200, 255), 1)

        if self.last_gap_line is not None:
            gap_line = np.array(self.last_gap_line, dtype=float)
            for i in range(len(gap_line) - 1):
                cv2.line(canvas, to_px(gap_line[i]), to_px(gap_line[i + 1]), (80, 80, 200), 1)

        if self.last_ref is not None:
            zref = np.array(self.last_ref, dtype=float)
            for i in range(len(zref) - 1):
                cv2.line(canvas, to_px(zref[i, 0:2]), to_px(zref[i + 1, 0:2]), (255, 0, 255), 2)
            for i in range(len(zref)):
                cv2.circle(canvas, to_px(zref[i, 0:2]), 4 if i > 0 else 6, (255, 0, 255), -1)

        if self.last_pred is not None:
            zpred = np.array(self.last_pred, dtype=float)
            for i in range(len(zpred) - 1):
                cv2.line(canvas, to_px(zpred[i, 0:2]), to_px(zpred[i + 1, 0:2]), (0, 165, 255), 2)

            car_w = 0.20
            car_l = 0.32
            body = np.array([
                [0.0, -car_w / 2],
                [0.0,  car_w / 2],
                [car_l,  car_w / 2],
                [car_l, -car_w / 2],
            ])
            for k in range(len(zpred)):
                p = zpred[k, 0:2]
                yaw = float(zpred[k, 2])
                c, s = math.cos(yaw), math.sin(yaw)
                R = np.array([[c, -s], [s, c]])
                poly = (R @ body.T).T + p
                poly_px = np.array([to_px(pt) for pt in poly], dtype=np.int32)
                cv2.polylines(canvas, [poly_px], True, (0, 120, 200), 1)
                cv2.circle(canvas, to_px(p), 4 if k > 0 else 6, (0, 165, 255), -1)
                cv2.putText(canvas, str(k), (to_px(p)[0] + 4, to_px(p)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        if self.last_goal_local is not None:
            cv2.circle(canvas, to_px(self.last_goal_local), 8, (0, 255, 0), -1)

        gap_d = float(np.clip(self.last_gap_distance, self.goal_min_distance, self.goal_max_distance))
        raw_gap = np.array([
            max(0.35, gap_d * math.cos(self.last_gap_angle)),
            gap_d * math.sin(self.last_gap_angle),
        ])
        cv2.circle(canvas, to_px(raw_gap), 7, (255, 255, 0), -1)

        ego_w = 0.20
        ego_l = 0.32
        ego_poly = np.array([
            [0.0, -ego_w / 2],
            [0.0,  ego_w / 2],
            [ego_l,  ego_w / 2],
            [ego_l, -ego_w / 2],
        ])
        ego_px = np.array([to_px(p) for p in ego_poly], dtype=np.int32)
        cv2.fillPoly(canvas, [ego_px], (210, 210, 210))

        left_x = 20
        left_y = top_h + 25
        left_w = int(0.57 * W)
        left_h = bot_h - 45
        cv2.rectangle(canvas, (left_x, left_y), (left_x + left_w, left_y + left_h), (70, 70, 70), 1)

        if self.last_u_pred is not None:
            u = np.array(self.last_u_pred, dtype=float)
            a_seq = u[:, 0]
            d_seq = u[:, 1]
            sub_h = left_h // 2

            def draw_series(series, ymin, ymax, color, title, x0, y0, w0, h0):
                cv2.rectangle(canvas, (x0, y0), (x0 + w0, y0 + h0), (45, 45, 45), 1)
                cv2.putText(canvas, title, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

                pts = []
                n = len(series)
                for i, val in enumerate(series):
                    px = int(round(x0 + 12 + i * (w0 - 24) / max(1, n - 1)))
                    t = (float(val) - ymin) / max(1e-6, ymax - ymin)
                    py = int(round(y0 + h0 - 10 - t * (h0 - 24)))
                    pts.append((px, py))
                    cv2.circle(canvas, (px, py), 3, color, -1)
                    cv2.putText(canvas, str(i), (px - 3, y0 + h0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)

                if len(pts) >= 2:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2)

            draw_series(a_seq, self.min_accel, self.max_accel, (255, 200, 0), 'accel horizon', left_x + 5, left_y + 5, left_w - 10, sub_h - 10)
            draw_series(d_seq, -self.max_steer, self.max_steer, (0, 255, 120), 'steer horizon', left_x + 5, left_y + sub_h + 5, left_w - 10, sub_h - 10)

        right_x = int(0.60 * W)
        right_y = top_h + 25
        right_w = W - right_x - 20
        right_h = bot_h - 45
        cv2.rectangle(canvas, (right_x, right_y), (right_x + right_w, right_y + right_h), (70, 70, 70), 1)

        if self.last_pred is not None:
            zpred = np.array(self.last_pred, dtype=float)
            v_seq = zpred[:, 3]
            psi_seq = zpred[:, 2]
            sub_h = right_h // 2

            def draw_state_series(series, ymin, ymax, color, title, x0, y0, w0, h0):
                cv2.rectangle(canvas, (x0, y0), (x0 + w0, y0 + h0), (45, 45, 45), 1)
                cv2.putText(canvas, title, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

                pts = []
                n = len(series)
                for i, val in enumerate(series):
                    px = int(round(x0 + 12 + i * (w0 - 24) / max(1, n - 1)))
                    t = (float(val) - ymin) / max(1e-6, ymax - ymin)
                    py = int(round(y0 + h0 - 10 - t * (h0 - 24)))
                    pts.append((px, py))
                    cv2.circle(canvas, (px, py), 3, color, -1)

                if len(pts) >= 2:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2)

            draw_state_series(v_seq, 0.0, max(self.max_speed, 0.5), (255, 100, 100), 'predicted speed', right_x + 5, right_y + 5, right_w - 10, sub_h - 10)
            draw_state_series(psi_seq, -1.0, 1.0, (100, 180, 255), 'predicted heading', right_x + 5, right_y + sub_h + 5, right_w - 10, sub_h - 10)

        lines = [
            f'v_cmd={v_cmd:.3f} m/s   delta_cmd={delta_cmd:.3f} rad',
            f'gap_angle={self.last_gap_angle:.3f} rad   safe_gap_angle={self.last_safe_gap_angle:.3f} rad',
            f'gap_distance={self.last_gap_distance:.3f} m   front_min={self.last_front_min:.3f} m',
            f'goal_x={0.0 if self.last_goal_local is None else self.last_goal_local[0]:.3f}   '
            f'goal_y={0.0 if self.last_goal_local is None else self.last_goal_local[1]:.3f}',
            f'min_width={self.last_min_width:.3f} m   gap_alpha={self.last_gap_alpha:.3f}   margin={self.last_safe_margin:.3f}',
            f'left_clear={float(getattr(self.gap_algo, "last_left_clear", 0.0)):.3f}   '
            f'right_clear={float(getattr(self.gap_algo, "last_right_clear", 0.0)):.3f}',
            f'display_x={self.display_x_max:.1f} planner_x={self.path_x_max:.1f} '
            f'solve={self.solve_time_ms_last:.2f} ms source={self.last_ref_source}',
        ]

        yy = 28
        for line in lines:
            cv2.putText(canvas, line, (18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
            yy += 24

        if self.show_opencv_debug:
            cv2.imshow(self.debug_window_name, canvas)
            cv2.waitKey(1)

        if self.publish_mpc_debug_images and self.mpc_debug_image_pub is not None:
            self.publish_debug_image(canvas)

    def publish_debug_image(self, canvas_bgr: np.ndarray) -> None:
        if self.mpc_debug_image_pub is None:
            return
        # canvas_bgr is a BGR uint8 image from OpenCV.
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.height = int(canvas_bgr.shape[0])
        msg.width = int(canvas_bgr.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(canvas_bgr.shape[1] * canvas_bgr.shape[2])
        msg.data = canvas_bgr.tobytes()
        self.mpc_debug_image_pub.publish(msg)

    # Output
    def publish_drive(self, speed: float, steering_angle: float) -> None:
        temp = AckermannDriveStamped()
        temp.header.stamp = self.get_clock().now().to_msg()
        temp.header.frame_id = 'base_link'
        temp.drive = AckermannDrive()
        temp.drive.speed = float(speed)
        temp.drive.steering_angle = float(steering_angle)

        msg = DriveControlMessage()
        msg.active = True
        msg.priority = 1004
        msg.drive = temp
        self.drive_pub.publish(msg)

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def wrap_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, 'show_opencv_debug', False):
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
