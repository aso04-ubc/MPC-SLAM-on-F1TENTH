#!/usr/bin/env python3
"""
ROS 2 F1TENTH FTG + lateral MPC node with richer OpenCV debug canvas.

This version keeps the same controller structure as the user's current node,
but improves visualization so you can see:
- the full MPC predicted horizon as numbered nodes
- ghost car footprints at each horizon step
- reference-to-prediction error bars
- horizon plots of e_y, e_psi, optimized delta, and reference delta
- the original FTG scan / extended scan / cost plot
"""

from __future__ import annotations

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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from gap_following.gap_utils import GapFollowAlgo


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float


class FTGMPCNode(Node):
    def __init__(self) -> None:
        super().__init__('ftg_mpc_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('odom_topic', '/ego_racecar/odom'),
                ('scan_topic', '/scan'),
                ('control_rate_hz', 40.0),
                ('dt', 0.025),
                ('horizon', 5),
                ('wheelbase', 0.33),
                ('max_speed', 3.0),
                ('min_speed', 0.4),
                ('max_steer', 0.4189),
                ('command_steer_limit', 0.30),
                ('ref_steer_limit', 0.16),
                ('max_dv', 0.10),
                ('max_ddelta', 0.015),
                ('use_odom_speed', True),
                ('print_timing_every', 10),
                ('scan_y_sign', 1.0),
                ('steering_output_sign', 1.0),
                ('startup_straight_frames', 6),
                # FTG params
                ('ftg_max_range', 3.5),
                ('ftg_min_safe_distance', 0.25),
                ('ftg_car_width', 0.45),
                ('ftg_disparity_threshold', 0.8),
                ('ftg_smoothing_window_size', 10),
                # Goal/path generation
                ('goal_min_distance', 0.5),
                ('goal_max_distance', 2.25),
                ('goal_filter_alpha', 0.6),
                ('goal_max_step_x', 0.08),
                ('goal_max_step_y', 0.09),
                ('goal_heading_gain', 0.55),
                ('goal_heading_limit', 0.22),
                ('path_ds', 0.22),
                ('path_y_limit', 0.45),
                ('wall_centering_gain', 0.5),
                ('wall_centering_max', 0.12),
                ('wall_centering_turn_scale', 1.1),
                ('wall_margin', 0.20),
                ('min_corridor_half_width', 0.10),
                # Speed policyls
                ('straight_speed', 3.0),
                ('corner_speed_cap', 1.2),
                ('speed_target_angle_gain', 2.0),
                ('speed_delta_gain', 0.65),
                ('speed_front_clearance_gain', 1.5),
                # MPC weights
                ('q_ey', 10.0),
                ('q_epsi', 12.0),
                ('qf_ey', 14.0),
                ('qf_epsi', 16.0),
                ('r_delta_err', 300.0),
                ('rd_delta_err', 400.0),
                # OpenCV debug
                ('show_opencv_debug', True),
                ('debug_canvas_width', 1400),
                ('debug_canvas_height', 980),
                ('debug_pixels_per_meter', 150.0),
                ('debug_window_name', 'FTG + MPC Debug'),
            ],
        )

        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.dt = float(self.get_parameter('dt').value)
        self.N = int(self.get_parameter('horizon').value)
        self.L = float(self.get_parameter('wheelbase').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.command_steer_limit = float(self.get_parameter('command_steer_limit').value)
        self.ref_steer_limit = float(self.get_parameter('ref_steer_limit').value)
        self.max_dv = float(self.get_parameter('max_dv').value)
        self.max_ddelta = float(self.get_parameter('max_ddelta').value)
        self.use_odom_speed = bool(self.get_parameter('use_odom_speed').value)
        self.print_timing_every = int(self.get_parameter('print_timing_every').value)
        self.scan_y_sign = float(self.get_parameter('scan_y_sign').value)
        self.steering_output_sign = float(self.get_parameter('steering_output_sign').value)
        self.startup_straight_frames = int(self.get_parameter('startup_straight_frames').value)

        self.goal_min_distance = float(self.get_parameter('goal_min_distance').value)
        self.goal_max_distance = float(self.get_parameter('goal_max_distance').value)
        self.goal_filter_alpha = float(self.get_parameter('goal_filter_alpha').value)
        self.goal_max_step_x = float(self.get_parameter('goal_max_step_x').value)
        self.goal_max_step_y = float(self.get_parameter('goal_max_step_y').value)
        self.goal_heading_gain = float(self.get_parameter('goal_heading_gain').value)
        self.goal_heading_limit = float(self.get_parameter('goal_heading_limit').value)
        self.path_ds = float(self.get_parameter('path_ds').value)
        self.path_y_limit = float(self.get_parameter('path_y_limit').value)
        self.wall_centering_gain = float(self.get_parameter('wall_centering_gain').value)
        self.wall_centering_max = float(self.get_parameter('wall_centering_max').value)
        self.wall_centering_turn_scale = float(self.get_parameter('wall_centering_turn_scale').value)
        self.wall_margin = float(self.get_parameter('wall_margin').value)
        self.min_corridor_half_width = float(self.get_parameter('min_corridor_half_width').value)

        self.straight_speed = float(self.get_parameter('straight_speed').value)
        self.corner_speed_cap = float(self.get_parameter('corner_speed_cap').value)
        self.speed_target_angle_gain = float(self.get_parameter('speed_target_angle_gain').value)
        self.speed_delta_gain = float(self.get_parameter('speed_delta_gain').value)
        self.speed_front_clearance_gain = float(self.get_parameter('speed_front_clearance_gain').value)

        self.q_ey = float(self.get_parameter('q_ey').value)
        self.q_epsi = float(self.get_parameter('q_epsi').value)
        self.qf_ey = float(self.get_parameter('qf_ey').value)
        self.qf_epsi = float(self.get_parameter('qf_epsi').value)
        self.r_delta_err = float(self.get_parameter('r_delta_err').value)
        self.rd_delta_err = float(self.get_parameter('rd_delta_err').value)

        self.show_opencv_debug = bool(self.get_parameter('show_opencv_debug').value)
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
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos)
        self.drive_pub = self.create_publisher(
            DriveControlMessage,
            DriveControlMessage.BUILTIN_TOPIC_NAME_STRING,
            10,
        )
        self.timer = self.create_timer(1.0 / float(self.get_parameter('control_rate_hz').value), self.control_callback)

        self.state: Optional[VehicleState] = None
        self.last_scan: Optional[LaserScan] = None
        self.last_u = np.array([0.6, 0.0], dtype=float)
        self.frame_count = 0
        self.solve_count = 0
        self.solve_time_ms_last = 0.0
        self.solve_time_ms_ema = 0.0

        self.filtered_goal = np.array([1.0, 0.0], dtype=float)
        self.last_ref_local: Optional[np.ndarray] = None
        self.last_pred_local: Optional[np.ndarray] = None
        self.last_mpc_debug: Optional[dict] = None
        self.last_scan_points: Optional[np.ndarray] = None
        self.last_corridor_samples: Optional[np.ndarray] = None

        if self.show_opencv_debug:
            cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.debug_window_name, self.debug_canvas_width, self.debug_canvas_height)

        self.get_logger().info('FTG + MPC node started.')

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
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            speed = float(math.hypot(vx, vy))
        else:
            speed = self.last_u[0]
        self.state = VehicleState(x=x, y=y, yaw=yaw, speed=speed)

    def scan_callback(self, msg: LaserScan) -> None:
        self.last_scan = msg

    def control_callback(self) -> None:
        if self.state is None or self.last_scan is None:
            return
        self.frame_count += 1

        ranges = np.array(self.last_scan.ranges, dtype=float)
        target_angle, best_idx, target_distance = self.gap_algo.process_lidar_and_find_gap(
            ranges,
            float(self.last_scan.angle_min),
            float(self.last_scan.angle_increment),
        )

        target_angle = self.scan_y_sign * target_angle
        goal_local = self.make_filtered_goal(target_angle, target_distance)
        ref_local = self.build_goal_path(goal_local, target_angle)
        self.last_ref_local = ref_local

        ref_speed, ref_delta = self.make_reference_profiles(ref_local, target_angle, target_distance)
        v_cmd, delta_cmd, solve_ms, pred_local, mpc_debug = self.solve_lateral_mpc(
            ref_local, ref_delta, ref_speed, self.last_u
        )
        self.last_pred_local = pred_local
        self.last_mpc_debug = mpc_debug

        self.solve_time_ms_last = solve_ms
        if self.solve_count == 0:
            self.solve_time_ms_ema = solve_ms
        else:
            self.solve_time_ms_ema = 0.9 * self.solve_time_ms_ema + 0.1 * solve_ms
        self.solve_count += 1
        if self.solve_count % max(1, self.print_timing_every) == 0:
            self.get_logger().info(
                f'MPC solve time: last={self.solve_time_ms_last:.2f} ms, ema={self.solve_time_ms_ema:.2f} ms'
            )

        self.get_logger().info(
            f'cmd: v={v_cmd:.3f}, delta={delta_cmd:.3f}, '
            f'gap_angle={target_angle:.3f}, gap_dist={target_distance:.3f}, '
            f'goal_x={goal_local[0]:.3f}, goal_y={goal_local[1]:.3f}'
        )

        self.last_u[:] = [v_cmd, delta_cmd]
        self.publish_drive(v_cmd, delta_cmd)

        if self.show_opencv_debug:
            self.draw_debug_canvas(
                ref_local, pred_local, target_angle, best_idx, target_distance,
                ref_delta, v_cmd, delta_cmd, self.last_mpc_debug
            )

    # ------------------------------------------------------------------
    # FTG goal -> local path
    # ------------------------------------------------------------------
    def make_filtered_goal(self, target_angle: float, target_distance: float) -> np.ndarray:
        d = float(np.clip(target_distance, self.goal_min_distance, self.goal_max_distance))
        goal = np.array([d * math.cos(target_angle), d * math.sin(target_angle)], dtype=float)
        goal[0] = max(goal[0], 0.4)

        # Push the goal away from the closer wall. Positive y is left.
        left_clear = float(getattr(self.gap_algo, 'last_left_clear', 0.0))
        right_clear = float(getattr(self.gap_algo, 'last_right_clear', 0.0))
        wall_balance = left_clear - right_clear

        turn_scale = 1.0 + self.wall_centering_turn_scale * min(1.0, abs(target_angle) / 0.35)
        wall_bias = np.clip(
            self.wall_centering_gain * turn_scale * wall_balance,
            -self.wall_centering_max,
            self.wall_centering_max,
        )
        goal[1] += wall_bias

        if abs(goal[1]) < 0.04:
            goal[1] = 0.0

        if self.frame_count <= self.startup_straight_frames:
            self.filtered_goal = np.array([max(1.0, goal[0]), 0.0], dtype=float)
            return self.filtered_goal.copy()

        alpha = float(np.clip(self.goal_filter_alpha, 0.0, 1.0))
        desired = alpha * goal + (1.0 - alpha) * self.filtered_goal
        dx = float(np.clip(desired[0] - self.filtered_goal[0], -self.goal_max_step_x, self.goal_max_step_x))
        dy = float(np.clip(desired[1] - self.filtered_goal[1], -self.goal_max_step_y, self.goal_max_step_y))
        self.filtered_goal[0] = max(0.4, self.filtered_goal[0] + dx)
        self.filtered_goal[1] = self.filtered_goal[1] + dy
        return self.filtered_goal.copy()

    def build_goal_path(self, goal_local: np.ndarray, target_angle: float) -> np.ndarray:
        goal_x = max(float(goal_local[0]), 0.4)
        goal_y = float(np.clip(goal_local[1], -self.path_y_limit, self.path_y_limit))

        # Use filtered goal direction instead of raw FTG angle.
        goal_heading = math.atan2(goal_y, goal_x)
        psi_goal = float(np.clip(
            self.goal_heading_gain * goal_heading,
            -self.goal_heading_limit,
            self.goal_heading_limit
        ))
        mT = math.tan(psi_goal)

        A = np.array(
            [[goal_x**3, goal_x**2], [3.0 * goal_x**2, 2.0 * goal_x]],
            dtype=float,
        )
        rhs = np.array([goal_y, mT], dtype=float)
        a3, a2 = np.linalg.solve(A, rhs)

        xs = np.arange(self.N + 1, dtype=float) * self.path_ds
        xs = np.clip(xs, 0.0, goal_x)
        ys = a3 * xs**3 + a2 * xs**2
        ys = np.clip(ys, -self.path_y_limit, self.path_y_limit)
        path = np.column_stack((xs, ys))
        path[0] = np.array([0.0, 0.0], dtype=float)
        return path

    def estimate_open_path_yaw(self, xy: np.ndarray) -> np.ndarray:
        yaw = np.zeros(len(xy), dtype=float)
        for i in range(len(xy) - 1):
            d = xy[i + 1] - xy[i]
            yaw[i] = math.atan2(d[1], d[0])
        yaw[-1] = yaw[-2] if len(xy) >= 2 else 0.0
        return yaw

    # ------------------------------------------------------------------
    # Speed / steering references
    # ------------------------------------------------------------------
    def compute_initial_ref_delta(self, ref_local: np.ndarray) -> float:
        if self.frame_count <= self.startup_straight_frames:
            return 0.0
        look_idx = min(len(ref_local) - 1, 3)
        xL = float(ref_local[look_idx, 0])
        yL = float(ref_local[look_idx, 1])
        Ld = max(1e-3, math.hypot(xL, yL))
        alpha = math.atan2(yL, xL)
        delta0 = math.atan2(2.0 * self.L * math.sin(alpha), Ld)
        return float(np.clip(delta0, -self.ref_steer_limit, self.ref_steer_limit))

    def make_reference_profiles(
        self,
        ref_local: np.ndarray,
        target_angle: float,
        target_distance: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        yaw_local = self.estimate_open_path_yaw(ref_local)
        ref_delta = np.zeros(self.N, dtype=float)
        ref_delta[0] = self.compute_initial_ref_delta(ref_local)

        for k in range(1, self.N):
            dyaw = self.wrap_angle(yaw_local[k + 1] - yaw_local[k])
            curvature = dyaw / max(1e-6, self.path_ds)
            delta_ref = math.atan(self.L * curvature)
            ref_delta[k] = float(np.clip(delta_ref, -self.ref_steer_limit, self.ref_steer_limit))

        front_clear = float(np.clip(target_distance, 0.0, self.goal_max_distance))
        v_ref0 = self.straight_speed
        v_ref0 -= self.speed_target_angle_gain * abs(target_angle)
        v_ref0 -= self.speed_delta_gain * abs(ref_delta[0])
        v_ref0 = min(v_ref0, self.corner_speed_cap + self.speed_front_clearance_gain * front_clear)
        v_ref0 = float(np.clip(v_ref0, self.min_speed, self.max_speed))
        ref_speed = np.full(self.N, v_ref0, dtype=float)
        return ref_speed, ref_delta

    # ------------------------------------------------------------------
    # Fast lateral MPC + debug extraction
    # ------------------------------------------------------------------
    def solve_lateral_mpc(
        self,
        ref_local: np.ndarray,
        ref_delta: np.ndarray,
        ref_speed: np.ndarray,
        last_u: np.ndarray,
    ) -> Tuple[float, float, float, Optional[np.ndarray], Optional[dict]]:
        t0 = time.perf_counter()

        v_nom = float(np.clip(ref_speed[0], self.min_speed, self.max_speed))
        A = np.array([[1.0, v_nom * self.dt], [0.0, 1.0]], dtype=float)
        B = np.array([[0.0], [v_nom * self.dt / self.L]], dtype=float)

        nx = 2
        nX = nx * (self.N + 1)
        nU = self.N
        nz = nX + nU

        yaw_local = self.estimate_open_path_yaw(ref_local)
        e0 = np.array([0.0, -float(yaw_local[1] if len(yaw_local) > 1 else 0.0)], dtype=float)

        Q = np.diag([self.q_ey, self.q_epsi])
        Qf = np.diag([self.qf_ey, self.qf_epsi])
        P_blocks = [2.0 * Q for _ in range(self.N)] + [2.0 * Qf]

        D = np.eye(self.N)
        for i in range(1, self.N):
            D[i, i - 1] = -1.0
        H_u = self.r_delta_err * np.eye(self.N) + self.rd_delta_err * (D.T @ D)
        P = sp.block_diag(P_blocks + [2.0 * H_u], format='csc')

        delta_err_prev = float(last_u[1] / max(1e-6, self.steering_output_sign) - ref_delta[0])
        d = np.zeros(self.N, dtype=float)
        d[0] = delta_err_prev
        q_u = -2.0 * self.rd_delta_err * (D.T @ d)
        q = np.zeros(nz, dtype=float)
        q[nX:] = q_u

        rows, cols, data = [], [], []
        l_eq, u_eq = [], []
        for i in range(nx):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            l_eq.append(e0[i])
            u_eq.append(e0[i])

        row_base = nx
        for k in range(self.N):
            for i in range(nx):
                rows.append(row_base + i)
                cols.append((k + 1) * nx + i)
                data.append(1.0)
            for i in range(nx):
                for j in range(nx):
                    rows.append(row_base + i)
                    cols.append(k * nx + j)
                    data.append(-A[i, j])
            for i in range(nx):
                rows.append(row_base + i)
                cols.append(nX + k)
                data.append(-B[i, 0])
            l_eq.extend([0.0, 0.0])
            u_eq.extend([0.0, 0.0])
            row_base += nx

        Aeq = sp.csc_matrix((data, (rows, cols)), shape=(nx * (self.N + 1), nz))

        # Steering input constraints
        Aineq = sp.hstack([sp.csc_matrix((self.N, nX)), sp.eye(self.N, format='csc')], format='csc')
        l_in = np.zeros(self.N, dtype=float)
        u_in = np.zeros(self.N, dtype=float)
        for k in range(self.N):
            l_in[k] = -self.command_steer_limit - ref_delta[k]
            u_in[k] = self.command_steer_limit - ref_delta[k]

        # Approximate corridor bounds on e_y using side clearances.
        left_clear = float(getattr(self.gap_algo, 'last_left_clear', 0.0))
        right_clear = float(getattr(self.gap_algo, 'last_right_clear', 0.0))

        ey_left_bound = max(self.min_corridor_half_width, left_clear - self.wall_margin)
        ey_right_bound = max(self.min_corridor_half_width, right_clear - self.wall_margin)

        Aey_rows, Aey_cols, Aey_data = [], [], []
        l_ey, u_ey = [], []
        row = 0
        for k in range(self.N + 1):
            Aey_rows.append(row)
            Aey_cols.append(k * nx + 0)  # e_y state
            Aey_data.append(1.0)
            l_ey.append(-ey_right_bound)
            u_ey.append(ey_left_bound)
            row += 1
        Aey = sp.csc_matrix((Aey_data, (Aey_rows, Aey_cols)), shape=(self.N + 1, nz))

        Aqp = sp.vstack([Aeq, Aineq, Aey], format='csc')
        l = np.concatenate([np.array(l_eq, dtype=float), l_in, np.array(l_ey, dtype=float)])
        u = np.concatenate([np.array(u_eq, dtype=float), u_in, np.array(u_ey, dtype=float)])

        solver = osqp.OSQP()
        solver.setup(
            P=P,
            q=q,
            A=Aqp,
            l=l,
            u=u,
            verbose=False,
            warm_start=True,
            polish=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=250,
        )
        res = solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        if res.x is None or res.info.status_val not in (1, 2):
            v_cmd = max(self.min_speed, min(self.max_speed, last_u[0] - self.max_dv))
            delta_cmd = float(np.clip(last_u[1], -self.command_steer_limit, self.command_steer_limit))
            return v_cmd, delta_cmd, solve_ms, None, None

        x_seq = res.x[:nX].reshape(self.N + 1, nx)
        u_seq = res.x[nX:nX + self.N]

        delta_err0 = float(u_seq[0])
        delta_cmd_internal = ref_delta[0] + delta_err0
        last_delta_internal = last_u[1] / max(1e-6, self.steering_output_sign)
        delta_cmd_internal = float(np.clip(
            delta_cmd_internal,
            last_delta_internal - self.max_ddelta,
            last_delta_internal + self.max_ddelta,
        ))
        delta_cmd_internal = float(np.clip(delta_cmd_internal, -self.command_steer_limit, self.command_steer_limit))
        delta_cmd = self.steering_output_sign * delta_cmd_internal

        v_cmd = float(np.clip(ref_speed[0], last_u[0] - self.max_dv, last_u[0] + self.max_dv))
        v_cmd = float(np.clip(v_cmd, self.min_speed, self.max_speed))

        pred_local, pred_yaw = self.reconstruct_predicted_path_debug(ref_local, yaw_local, x_seq)
        delta_seq_internal = ref_delta + u_seq
        delta_seq_internal = np.clip(delta_seq_internal, -self.command_steer_limit, self.command_steer_limit)

        mpc_debug = {
            'x_seq': x_seq.copy(),
            'u_seq': u_seq.copy(),
            'delta_seq': delta_seq_internal.copy(),
            'ref_delta': ref_delta.copy(),
            'pred_local': pred_local.copy(),
            'pred_yaw': pred_yaw.copy(),
            'yaw_local': yaw_local.copy(),
            'ey_left_bound': ey_left_bound,
            'ey_right_bound': ey_right_bound,
        }

        return v_cmd, delta_cmd, solve_ms, pred_local, mpc_debug

    def reconstruct_predicted_path_debug(
        self,
        ref_local: np.ndarray,
        yaw_local: np.ndarray,
        x_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pred = np.zeros_like(ref_local)
        pred_yaw = np.zeros(len(ref_local), dtype=float)
        for k in range(len(ref_local)):
            yaw_ref = yaw_local[min(k, len(yaw_local) - 1)]
            ey = float(x_seq[k, 0])
            epsi = float(x_seq[k, 1])
            normal = np.array([-math.sin(yaw_ref), math.cos(yaw_ref)], dtype=float)
            pred[k] = ref_local[k] + ey * normal
            pred_yaw[k] = self.wrap_angle(yaw_ref + epsi)
        return pred, pred_yaw

    def reconstruct_predicted_path(
        self,
        ref_local: np.ndarray,
        yaw_local: np.ndarray,
        e0: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        u_seq: np.ndarray,
    ) -> np.ndarray:
        x = e0.copy()
        pred = np.zeros_like(ref_local)
        for k in range(self.N + 1):
            yaw_ref = yaw_local[min(k, len(yaw_local) - 1)]
            normal = np.array([-math.sin(yaw_ref), math.cos(yaw_ref)], dtype=float)
            pred[k] = ref_local[k] + x[0] * normal
            if k < self.N:
                x = A @ x + B.flatten() * float(u_seq[k])
        return pred

    # ------------------------------------------------------------------
    # Debug canvas
    # ------------------------------------------------------------------
    def draw_debug_canvas(
        self,
        ref_local: np.ndarray,
        pred_local: Optional[np.ndarray],
        target_angle: float,
        best_idx: int,
        target_distance: float,
        ref_delta: np.ndarray,
        v_cmd: float,
        delta_cmd: float,
        mpc_debug: Optional[dict],
    ) -> None:
        W = self.debug_canvas_width
        H = self.debug_canvas_height
        top_h = int(0.60 * H)
        bot_h = H - top_h
        scale = self.debug_pixels_per_meter

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)

        origin = np.array([W // 2, top_h - 60], dtype=float)

        def to_px(p: np.ndarray) -> Tuple[int, int]:
            x, y = float(p[0]), float(p[1])
            px = int(round(origin[0] - y * scale))
            py = int(round(origin[1] - x * scale))
            return px, py

        # --------------------------------------------------------------
        # Top-down plan view
        # --------------------------------------------------------------
        for m in np.arange(0.0, 4.6, 0.5):
            cv2.line(canvas, to_px(np.array([m, -2.0])), to_px(np.array([m, 2.0])), (35, 35, 35), 1)
        for m in np.arange(-2.0, 2.1, 0.5):
            cv2.line(canvas, to_px(np.array([0.0, m])), to_px(np.array([4.5, m])), (35, 35, 35), 1)
        cv2.line(canvas, to_px(np.array([0.0, -2.0])), to_px(np.array([0.0, 2.0])), (70, 70, 70), 2)
        cv2.line(canvas, to_px(np.array([0.0, 0.0])), to_px(np.array([4.5, 0.0])), (70, 70, 70), 2)

        if self.gap_algo.last_angles is not None and self.gap_algo.last_extended is not None:
            xs = self.gap_algo.last_extended * np.cos(self.gap_algo.last_angles)
            ys = self.gap_algo.last_extended * np.sin(self.gap_algo.last_angles)
            for x, y in zip(xs, ys):
                cv2.circle(canvas, to_px(np.array([x, y])), 2, (110, 110, 110), -1)

        goal_x = float(self.filtered_goal[0])
        goal_y = float(self.filtered_goal[1])
        cv2.circle(canvas, to_px(np.array([goal_x, goal_y])), 8, (0, 255, 0), -1)

        raw_goal_x = float(np.clip(target_distance, self.goal_min_distance, self.goal_max_distance) * math.cos(target_angle))
        raw_goal_y = float(np.clip(target_distance, self.goal_min_distance, self.goal_max_distance) * math.sin(target_angle))
        raw_goal_x = max(raw_goal_x, 0.4)
        cv2.circle(canvas, to_px(np.array([raw_goal_x, raw_goal_y])), 7, (0, 255, 255), -1)

        for i in range(len(ref_local) - 1):
            cv2.line(canvas, to_px(ref_local[i]), to_px(ref_local[i + 1]), (255, 0, 0), 2)

        if mpc_debug is not None:
            pred = mpc_debug['pred_local']
            pred_yaw = mpc_debug['pred_yaw']
            x_seq = mpc_debug['x_seq']

            # error bars from reference to predicted state
            for k in range(len(ref_local)):
                cv2.line(canvas, to_px(ref_local[k]), to_px(pred[k]), (80, 80, 180), 1)

            # predicted polyline
            for i in range(len(pred) - 1):
                cv2.line(canvas, to_px(pred[i]), to_px(pred[i + 1]), (0, 165, 255), 2)

            # predicted nodes + labels
            for k in range(len(pred)):
                p = to_px(pred[k])
                cv2.circle(canvas, p, 4 if k > 0 else 6, (0, 165, 255), -1)
                cv2.putText(canvas, str(k), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

            # ghost cars along horizon
            car_w = 0.20
            car_l = 0.32
            body = np.array([
                [0.0, -car_w / 2],
                [0.0,  car_w / 2],
                [car_l, car_w / 2],
                [car_l, -car_w / 2],
            ], dtype=float)
            for k in range(len(pred)):
                c = math.cos(pred_yaw[k])
                s = math.sin(pred_yaw[k])
                R = np.array([[c, -s], [s, c]], dtype=float)
                poly = (R @ body.T).T + pred[k]
                poly_px = np.array([to_px(pt) for pt in poly], dtype=np.int32)
                cv2.polylines(canvas, [poly_px], True, (0, 110, 180), 1)

        car_w = 0.20
        car_l = 0.32
        car_poly = np.array(
            [[0.0, -car_w / 2], [0.0, car_w / 2], [car_l, car_w / 2], [car_l, -car_w / 2]],
            dtype=float,
        )
        car_pts = np.array([to_px(p) for p in car_poly], dtype=np.int32)
        cv2.fillPoly(canvas, [car_pts], (200, 200, 200))

        # --------------------------------------------------------------
        # Bottom-left FTG profile plot
        # --------------------------------------------------------------
        x0 = 20
        y0 = top_h + 30
        plot_w = int(0.58 * W)
        plot_h = bot_h - 60
        cv2.rectangle(canvas, (x0, y0), (x0 + plot_w, y0 + plot_h), (70, 70, 70), 1)

        if self.gap_algo.last_processed is not None and self.gap_algo.last_extended is not None:
            proc = self.gap_algo.last_processed
            ext = self.gap_algo.last_extended
            costs = self.gap_algo.last_costs
            n = len(proc)
            if n > 1:
                vmax = max(self.gap_algo.max_range, float(np.max(ext)))

                def draw_series(series: np.ndarray, color: Tuple[int, int, int], normalize_to_vmax: bool = True) -> None:
                    pts = []
                    smax = vmax if normalize_to_vmax else max(1e-6, float(np.max(series)))
                    for i, v in enumerate(series):
                        px = int(round(x0 + i * (plot_w - 1) / max(1, n - 1)))
                        py = int(round(y0 + plot_h - 1 - (float(v) / max(1e-6, smax)) * (plot_h - 1)))
                        pts.append((px, py))
                    if len(pts) >= 2:
                        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2)

                draw_series(proc, (120, 120, 120), True)
                draw_series(ext, (255, 255, 0), True)
                if costs is not None:
                    draw_series(costs, (255, 0, 255), False)

                if self.gap_algo.last_best_idx is not None:
                    bx = int(round(x0 + self.gap_algo.last_best_idx * (plot_w - 1) / max(1, n - 1)))
                    cv2.line(canvas, (bx, y0), (bx, y0 + plot_h), (0, 255, 0), 2)

        # --------------------------------------------------------------
        # Bottom-right MPC horizon plots
        # --------------------------------------------------------------
        panel_x = int(0.62 * W)
        panel_y = top_h + 30
        panel_w = W - panel_x - 20
        panel_h = bot_h - 60
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (70, 70, 70), 1)

        if mpc_debug is not None:
            x_seq = mpc_debug['x_seq']
            delta_seq = mpc_debug['delta_seq']
            ref_delta_seq = mpc_debug['ref_delta']
            ey = x_seq[:, 0]
            epsi = x_seq[:, 1]
            sub_h = panel_h // 3

            def draw_horizon_series(series, y_min, y_max, color, title, x_start, y_start, w, h):
                cv2.rectangle(canvas, (x_start, y_start), (x_start + w, y_start + h), (50, 50, 50), 1)
                cv2.putText(canvas, title, (x_start + 8, y_start + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

                pts = []
                n = len(series)
                for i, v in enumerate(series):
                    px = int(round(x_start + 10 + i * (w - 20) / max(1, n - 1)))
                    norm = (float(v) - y_min) / max(1e-6, (y_max - y_min))
                    py = int(round(y_start + h - 10 - norm * (h - 25)))
                    pts.append((px, py))
                    cv2.circle(canvas, (px, py), 3, color, -1)
                    cv2.putText(canvas, str(i), (px - 3, y_start + h - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

                if len(pts) >= 2:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2)

            draw_horizon_series(
                ey, -0.6, 0.6, (255, 200, 0), 'e_y horizon',
                panel_x + 5, panel_y + 5, panel_w - 10, sub_h - 10
            )
            draw_horizon_series(
                epsi, -0.8, 0.8, (0, 220, 255), 'e_psi horizon',
                panel_x + 5, panel_y + sub_h + 5, panel_w - 10, sub_h - 10
            )
            draw_horizon_series(
                delta_seq, -self.command_steer_limit, self.command_steer_limit, (0, 255, 120), 'delta horizon',
                panel_x + 5, panel_y + 2 * sub_h + 5, panel_w - 10, sub_h - 10
            )

            x_start = panel_x + 5
            y_start = panel_y + 2 * sub_h + 5
            w = panel_w - 10
            h = sub_h - 10
            pts_ref = []
            for i, v in enumerate(ref_delta_seq):
                px = int(round(x_start + 10 + i * (w - 20) / max(1, len(ref_delta_seq) - 1)))
                norm = (float(v) + self.command_steer_limit) / max(1e-6, 2.0 * self.command_steer_limit)
                py = int(round(y_start + h - 10 - norm * (h - 25)))
                pts_ref.append((px, py))
            if len(pts_ref) >= 2:
                cv2.polylines(canvas, [np.array(pts_ref, dtype=np.int32)], False, (255, 255, 255), 1)

        # --------------------------------------------------------------
        # Text overlay
        # --------------------------------------------------------------
        lines = [
            f'v_cmd={v_cmd:.3f} m/s',
            f'delta_cmd={delta_cmd:.3f} rad',
            f'ref_delta0={ref_delta[0]:.3f} rad',
            f'gap_angle={target_angle:.3f} rad',
            f'gap_dist={target_distance:.3f} m',
            f'goal=({goal_x:.3f}, {goal_y:.3f})',
            f'front_min={self.gap_algo.last_front_min:.3f} m',
            f'left_clear={self.gap_algo.last_left_clear:.3f}, right_clear={self.gap_algo.last_right_clear:.3f}',
            f'solve={self.solve_time_ms_last:.2f} ms',
        ]
        yy = 28
        for line in lines:
            cv2.putText(canvas, line, (18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
            yy += 24

        cv2.imshow(self.debug_window_name, canvas)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish_drive(self, speed: float, steering_angle: float) -> None:
        temp_msg = AckermannDriveStamped()
        temp_msg.header.stamp = self.get_clock().now().to_msg()
        temp_msg.header.frame_id = 'base_link'
        temp_msg.drive = AckermannDrive()
        temp_msg.drive.speed = float(speed)
        temp_msg.drive.steering_angle = float(steering_angle)

        full_msg = DriveControlMessage()
        full_msg.active = True
        full_msg.priority = 1004
        full_msg.drive = temp_msg
        self.drive_pub.publish(full_msg)

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def wrap_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FTGMPCNode()
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
