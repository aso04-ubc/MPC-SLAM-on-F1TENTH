"""ROS2 node for the F1TENTH MPC controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Sequence

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

try:
    from dev_b7_interfaces.msg import DriveControlMessage
except Exception:  # pragma: no cover - optional outside a built ROS workspace
    DriveControlMessage = None

from mpc_controller.mpc_core.path_loader import load_path_reference_from_csv
from mpc_controller.mpc_core.reference import (
    PathReference,
    ReferenceSegment,
    build_constant_speed_reference,
    extract_local_reference,
)
from mpc_controller.mpc_core.tracking_problem import (
    build_tracking_problem,
    estimate_steering_from_yaw_rate,
    integrate_control_step,
)
from mpc_controller.mpc_core.vehicle_model import VehicleParams, VehicleState, clamp
from mpc_controller.mpc_solvers.qp_cvxpy_osqp import QPMPCController


@dataclass(frozen=True)
class ControllerCommand:
    """Drive command ready to publish."""

    speed: float
    steering_angle: float
    acceleration: float = 0.0
    steering_angle_velocity: float = 0.0


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion to yaw."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_reference_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in ('constant_speed', 'constant_speed_placeholder'):
        return 'constant_speed'
    if normalized in ('path', 'path_csv'):
        return 'path_csv'
    if normalized == 'auto':
        return 'auto'
    return normalized


class MPCControllerNode(Node):
    """ROS2 node that closes the loop around the QP MPC controller."""

    def __init__(self) -> None:
        super().__init__('mpc_controller_node')

        self._declare_parameters()
        self._load_parameters()

        self.current_state: Optional[VehicleState] = None
        self.last_odom_stamp: Optional[Time] = None
        self.last_stop_reason: Optional[str] = None
        self.last_solver_status: Optional[str] = None
        self.last_control: List[float] = [0.0, 0.0]
        self.estimated_delta = 0.0
        self.previous_reference_index: Optional[int] = None
        self.has_logged_first_odom = False
        self.path_reference: Optional[PathReference] = None
        self.path_load_error: Optional[str] = None

        self.solver = QPMPCController(
            nx=5,
            nu=2,
            N=self.horizon_N,
            dt=self.dt,
            constraints=self.constraints,
            weights=self.weights,
        )

        self._load_path_reference()

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.raw_drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/mpc/drive',
            10,
        )

        self.drive_control_pub = None
        if self.publish_drive_control:
            if DriveControlMessage is None:
                self.get_logger().warn(
                    'dev_b7_interfaces is unavailable; /drive_control publishing is disabled'
                )
            else:
                self.drive_control_pub = self.create_publisher(
                    DriveControlMessage,
                    self.drive_control_topic,
                    10,
                )

        self.control_timer = self.create_timer(
            self.dt,
            self.control_loop,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

        self.get_logger().info(
            'mpc_controller_node started with '
            f'reference_mode={self.reference_mode}, '
            f'publish_drive_control={self.publish_drive_control}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('horizon_N', 15)
        self.declare_parameter('wheelbase_L', 0.33)
        self.declare_parameter('v_min', 0.0)
        self.declare_parameter('v_max', 4.0)
        self.declare_parameter('delta_max', 0.4)
        self.declare_parameter('a_min', -3.0)
        self.declare_parameter('a_max', 3.0)
        self.declare_parameter('ddelta_max', 1.0)
        self.declare_parameter('Q', [20.0, 10.0, 2.0, 1.0])
        self.declare_parameter('Q_N', [40.0, 20.0, 4.0, 2.0])
        self.declare_parameter('R', [1.0, 10.0])
        self.declare_parameter('R_delta', [0.1, 5.0])
        self.declare_parameter('solver_time_budget_ms', 10)
        self.declare_parameter('solver_max_iter', 4000)
        self.declare_parameter('reference_mode', 'auto')
        self.declare_parameter('path_csv', '')
        self.declare_parameter('target_speed', 1.0)
        self.declare_parameter('search_window', 15)
        self.declare_parameter('odom_timeout_sec', 0.25)
        self.declare_parameter('stop_on_invalid_reference', True)
        self.declare_parameter('stop_on_solver_failure', True)
        self.declare_parameter('publish_drive_control', True)
        self.declare_parameter(
            'drive_control_topic',
            '/drive_control' if DriveControlMessage is None else DriveControlMessage.BUILTIN_TOPIC_NAME_STRING,
        )
        self.declare_parameter('command_priority', 700)
        self.declare_parameter('command_active', True)
        self.declare_parameter('min_speed_for_steering_estimate', 0.2)

    def _load_parameters(self) -> None:
        self.dt = float(self.get_parameter('dt').value)
        self.horizon_N = int(self.get_parameter('horizon_N').value)
        self.reference_mode = _normalize_reference_mode(self.get_parameter('reference_mode').value)
        self.path_csv = str(self.get_parameter('path_csv').value).strip()
        self.target_speed = float(self.get_parameter('target_speed').value)
        self.search_window = int(self.get_parameter('search_window').value)
        self.odom_timeout_sec = float(self.get_parameter('odom_timeout_sec').value)
        self.stop_on_invalid_reference = bool(self.get_parameter('stop_on_invalid_reference').value)
        self.stop_on_solver_failure = bool(self.get_parameter('stop_on_solver_failure').value)
        self.publish_drive_control = bool(self.get_parameter('publish_drive_control').value)
        self.drive_control_topic = str(self.get_parameter('drive_control_topic').value)
        self.command_priority = int(self.get_parameter('command_priority').value)
        self.command_active = bool(self.get_parameter('command_active').value)
        self.min_speed_for_steering_estimate = float(
            self.get_parameter('min_speed_for_steering_estimate').value
        )

        self.vehicle_params = VehicleParams(
            wheelbase_L=float(self.get_parameter('wheelbase_L').value),
            delta_max=float(self.get_parameter('delta_max').value),
            delta_dot_max=float(self.get_parameter('ddelta_max').value),
            v_min=float(self.get_parameter('v_min').value),
            v_max=float(self.get_parameter('v_max').value),
            a_min=float(self.get_parameter('a_min').value),
            a_max=float(self.get_parameter('a_max').value),
        )

        self.weights = {
            'Q': self._float_sequence_from_parameter('Q'),
            'Q_N': self._float_sequence_from_parameter('Q_N'),
            'R': self._float_sequence_from_parameter('R'),
            'R_delta': self._float_sequence_from_parameter('R_delta'),
        }
        self.constraints = {
            'a_min': self.vehicle_params.a_min,
            'a_max': self.vehicle_params.a_max,
            'ddelta_min': -self.vehicle_params.delta_dot_max,
            'ddelta_max': self.vehicle_params.delta_dot_max,
            'delta_max': self.vehicle_params.delta_max,
            'v_min': self.vehicle_params.v_min,
            'v_max': self.vehicle_params.v_max,
            'max_iter': int(self.get_parameter('solver_max_iter').value),
        }
        self.solver_time_budget_ms = float(self.get_parameter('solver_time_budget_ms').value)

    def _float_sequence_from_parameter(self, name: str) -> List[float]:
        return [float(value) for value in self.get_parameter(name).value]

    def _load_path_reference(self) -> None:
        self.path_reference = None
        self.path_load_error = None
        self.previous_reference_index = None
        if not self.path_csv:
            return

        try:
            self.path_reference = load_path_reference_from_csv(self.path_csv, self.target_speed)
            resolved_path = Path(self.path_csv).expanduser()
            if not resolved_path.is_absolute():
                resolved_path = resolved_path.resolve()
            self.get_logger().info(
                f'Loaded path CSV with {len(self.path_reference.x)} points from {resolved_path}'
            )
        except Exception as exc:
            self.path_load_error = f'{type(exc).__name__}: {exc}'
            self.get_logger().warn(f'Failed to load path CSV: {self.path_load_error}')

    def odom_callback(self, msg: Odometry) -> None:
        """Update vehicle state from odometry."""
        pose = msg.pose.pose
        twist = msg.twist.twist

        yaw = quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        speed = float(twist.linear.x)
        self.estimated_delta = estimate_steering_from_yaw_rate(
            speed=speed,
            yaw_rate=float(twist.angular.z),
            params=self.vehicle_params,
            fallback_delta=self.estimated_delta,
            min_speed=self.min_speed_for_steering_estimate,
        )

        self.current_state = VehicleState(
            x=float(pose.position.x),
            y=float(pose.position.y),
            psi=yaw,
            v=speed,
            delta=self.estimated_delta,
        )
        if msg.header.stamp.sec or msg.header.stamp.nanosec:
            self.last_odom_stamp = Time.from_msg(msg.header.stamp)
        else:
            self.last_odom_stamp = self.get_clock().now()

        if not self.has_logged_first_odom:
            self.get_logger().info(
                'Received first odom sample: '
                f'x={self.current_state.x:.3f}, '
                f'y={self.current_state.y:.3f}, '
                f'v={self.current_state.v:.3f}'
            )
            self.has_logged_first_odom = True

    def control_loop(self) -> None:
        """Run one MPC iteration."""
        if self.current_state is None:
            self.publish_stop('waiting_for_odom')
            return

        if self.odom_is_stale():
            self.publish_stop('stale_odom')
            return

        reference = self.build_reference()
        if reference is None or not reference.is_valid():
            if self.stop_on_invalid_reference:
                self.publish_stop('invalid_reference')
                return

        command = self.compute_mpc_command(reference)
        if command is None:
            return

        self.publish_command(command)
        self.last_stop_reason = None

    def odom_is_stale(self) -> bool:
        """Check whether odometry is too old."""
        if self.last_odom_stamp is None:
            return True
        age = (self.get_clock().now() - self.last_odom_stamp).nanoseconds / 1e9
        return age > self.odom_timeout_sec

    def build_reference(self) -> Optional[ReferenceSegment]:
        """Build the current local reference."""
        if self.current_state is None:
            return None

        if self.reference_mode == 'auto':
            if self.path_reference is not None and self.path_reference.is_valid():
                return self._build_path_reference()
            return self._build_constant_speed_reference()

        if self.reference_mode == 'path_csv':
            return self._build_path_reference()

        if self.reference_mode == 'constant_speed':
            return self._build_constant_speed_reference()

        self.get_logger().warn(f'Unsupported reference_mode={self.reference_mode!r}')
        return None

    def _build_constant_speed_reference(self) -> ReferenceSegment:
        return build_constant_speed_reference(
            current_state=self.current_state,
            horizon_N=self.horizon_N,
            dt=self.dt,
            target_speed=clamp(self.target_speed, self.vehicle_params.v_min, self.vehicle_params.v_max),
        )

    def _build_path_reference(self) -> Optional[ReferenceSegment]:
        if self.current_state is None:
            return None
        if self.path_reference is None or not self.path_reference.is_valid():
            if self.path_load_error is not None:
                self.get_logger().warn(f'Path reference unavailable: {self.path_load_error}')
            return None

        segment, nearest = extract_local_reference(
            path=self.path_reference,
            current_state=self.current_state,
            horizon_N=self.horizon_N,
            dt=self.dt,
            previous_index=self.previous_reference_index,
            search_window=self.search_window,
        )
        self.previous_reference_index = nearest.index
        return segment

    def compute_mpc_command(self, reference: Optional[ReferenceSegment]) -> Optional[ControllerCommand]:
        """Solve the tracking problem and convert the result into a drive command."""
        if reference is None or self.current_state is None or not reference.is_valid():
            return ControllerCommand(speed=0.0, steering_angle=self.estimated_delta)

        try:
            problem = build_tracking_problem(
                current_state=self.current_state,
                reference=reference,
                dt=self.dt,
                params=self.vehicle_params,
            )
        except Exception as exc:
            self.publish_stop(f'tracking_problem_error:{type(exc).__name__}')
            return None

        u0, info = self.solver.solve(
            lin_sys=problem.lin_sys,
            ref=problem.ref,
            x0=self.current_state.as_vector(),
            u_prev=self.last_control,
            time_budget_ms=self.solver_time_budget_ms,
        )

        solver_status = str(info.get('status', 'ERROR'))
        has_solution = bool(info.get('has_solution', False))
        if solver_status != self.last_solver_status:
            self.get_logger().info(
                'MPC solve status='
                f'{solver_status}, '
                f'has_solution={has_solution}, '
                f'solve_time_ms={float(info.get("solve_time_ms", 0.0)):.2f}'
            )
            self.last_solver_status = solver_status

        if self.stop_on_solver_failure and not has_solution:
            self.publish_stop(f'solver_{solver_status.lower()}')
            return None

        accel = clamp(float(u0[0]), self.vehicle_params.a_min, self.vehicle_params.a_max)
        delta_dot = clamp(
            float(u0[1]),
            -self.vehicle_params.delta_dot_max,
            self.vehicle_params.delta_dot_max,
        )
        commanded_speed, commanded_delta = integrate_control_step(
            current_state=self.current_state,
            control=[accel, delta_dot],
            dt=self.dt,
            params=self.vehicle_params,
        )

        self.last_control = [accel, delta_dot]
        self.estimated_delta = commanded_delta
        if self.current_state is not None:
            self.current_state = VehicleState(
                x=self.current_state.x,
                y=self.current_state.y,
                psi=self.current_state.psi,
                v=commanded_speed,
                delta=commanded_delta,
            )

        return ControllerCommand(
            speed=commanded_speed,
            steering_angle=commanded_delta,
            acceleration=accel,
            steering_angle_velocity=delta_dot,
        )

    def publish_command(self, command: ControllerCommand) -> None:
        """Publish the command to raw and muxed control topics."""
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = float(command.speed)
        drive_msg.drive.steering_angle = float(command.steering_angle)
        drive_msg.drive.acceleration = float(command.acceleration)
        drive_msg.drive.steering_angle_velocity = float(command.steering_angle_velocity)
        self.raw_drive_pub.publish(drive_msg)

        if self.drive_control_pub is not None and DriveControlMessage is not None:
            mux_msg = DriveControlMessage()
            mux_msg.priority = int(self.command_priority)
            mux_msg.active = bool(self.command_active)
            mux_msg.drive = drive_msg
            self.drive_control_pub.publish(mux_msg)

    def publish_stop(self, reason: str) -> None:
        """Publish a safe stop command."""
        self.last_control = [0.0, 0.0]
        if self.last_stop_reason != reason:
            self.get_logger().warn(f'Publishing safe stop: {reason}')
            self.last_stop_reason = reason
        self.publish_command(
            ControllerCommand(
                speed=0.0,
                steering_angle=self.estimated_delta,
                acceleration=0.0,
                steering_angle_velocity=0.0,
            )
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Start the node."""
    rclpy.init(args=args)
    node = MPCControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
