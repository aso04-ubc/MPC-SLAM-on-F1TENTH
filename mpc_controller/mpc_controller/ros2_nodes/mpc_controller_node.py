"""Simple MPC ROS2 node."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from mpc_controller.mpc_core.reference import (
    ReferenceSegment,
    build_constant_speed_reference,
)
from mpc_controller.mpc_core.vehicle_model import VehicleParams, VehicleState


@dataclass(frozen=True)
class ControllerCommand:
    """Simple drive command."""

    speed: float
    steering_angle: float
    acceleration: float = 0.0
    steering_angle_velocity: float = 0.0


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion to yaw."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class MPCControllerNode(Node):
    """ROS2 node for the MPC controller."""

    def __init__(self) -> None:
        super().__init__('mpc_controller_node')

        self.declare_parameter('dt', 0.02)
        self.declare_parameter('horizon_N', 15)
        self.declare_parameter('wheelbase_L', 0.33)
        self.declare_parameter('v_min', 0.0)
        self.declare_parameter('v_max', 4.0)
        self.declare_parameter('delta_max', 0.4)
        self.declare_parameter('a_min', -3.0)
        self.declare_parameter('a_max', 3.0)
        self.declare_parameter('ddelta_max', 1.0)
        self.declare_parameter('solver_time_budget_ms', 10)
        self.declare_parameter('reference_mode', 'constant_speed_placeholder')
        self.declare_parameter('target_speed', 0.0)
        self.declare_parameter('odom_timeout_sec', 0.25)
        self.declare_parameter('stop_on_invalid_reference', True)

        self.dt = float(self.get_parameter('dt').value)
        self.horizon_N = int(self.get_parameter('horizon_N').value)
        self.reference_mode = str(self.get_parameter('reference_mode').value)
        self.target_speed = float(self.get_parameter('target_speed').value)
        self.odom_timeout_sec = float(self.get_parameter('odom_timeout_sec').value)
        self.stop_on_invalid_reference = bool(self.get_parameter('stop_on_invalid_reference').value)

        self.vehicle_params = VehicleParams(
            wheelbase_L=float(self.get_parameter('wheelbase_L').value),
            delta_max=float(self.get_parameter('delta_max').value),
            delta_dot_max=float(self.get_parameter('ddelta_max').value),
            v_min=float(self.get_parameter('v_min').value),
            v_max=float(self.get_parameter('v_max').value),
            a_min=float(self.get_parameter('a_min').value),
            a_max=float(self.get_parameter('a_max').value),
        )

        self.current_state: Optional[VehicleState] = None
        self.last_odom_stamp: Optional[Time] = None
        self.last_stop_reason: Optional[str] = None

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/mpc/drive',
            10,
        )
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            'mpc_controller_node started: subscribing to /odom and publishing to /mpc/drive'
        )

    def odom_callback(self, msg: Odometry) -> None:
        """Update state from odometry."""
        pose = msg.pose.pose
        twist = msg.twist.twist

        yaw = quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        self.current_state = VehicleState(
            x=pose.position.x,
            y=pose.position.y,
            psi=yaw,
            v=twist.linear.x,
            delta=0.0,
        )
        self.last_odom_stamp = Time.from_msg(msg.header.stamp) if msg.header.stamp.sec or msg.header.stamp.nanosec else self.get_clock().now()

    def control_loop(self) -> None:
        """Run one control step."""
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

        command = self.compute_placeholder_command(reference)
        self.publish_command(command)
        self.last_stop_reason = None

    def odom_is_stale(self) -> bool:
        """Check if odometry is too old."""
        if self.last_odom_stamp is None:
            return True
        age = (self.get_clock().now() - self.last_odom_stamp).nanoseconds / 1e9
        return age > self.odom_timeout_sec

    def build_reference(self) -> Optional[ReferenceSegment]:
        """Build the current local reference."""
        if self.current_state is None:
            return None
        if self.reference_mode != 'constant_speed_placeholder':
            return None
        return build_constant_speed_reference(
            current_state=self.current_state,
            horizon_N=self.horizon_N,
            dt=self.dt,
            target_speed=max(self.vehicle_params.v_min, min(self.target_speed, self.vehicle_params.v_max)),
        )

    def compute_placeholder_command(self, reference: Optional[ReferenceSegment]) -> ControllerCommand:
        """Build a simple placeholder control command."""
        if reference is None or not reference.is_valid():
            return ControllerCommand(speed=0.0, steering_angle=0.0)

        target_speed = max(self.vehicle_params.v_min, min(reference.v_ref[0], self.vehicle_params.v_max))
        return ControllerCommand(
            speed=target_speed,
            steering_angle=0.0,
            acceleration=0.0,
            steering_angle_velocity=0.0,
        )

    def publish_command(self, command: ControllerCommand) -> None:
        """Publish one drive command."""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(command.speed)
        msg.drive.steering_angle = float(command.steering_angle)
        msg.drive.acceleration = float(command.acceleration)
        msg.drive.steering_angle_velocity = float(command.steering_angle_velocity)
        self.drive_pub.publish(msg)

    def publish_stop(self, reason: str) -> None:
        """Publish a stop command."""
        if self.last_stop_reason != reason:
            self.get_logger().warn(f'Publishing safe stop: {reason}')
            self.last_stop_reason = reason
        self.publish_command(ControllerCommand(speed=0.0, steering_angle=0.0))


def main(args: Optional[List[str]] = None) -> None:
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
