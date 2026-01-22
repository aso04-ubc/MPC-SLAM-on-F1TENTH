"""
- Use LIDAR (/scan) to find the closest obstacle straight ahead (d_min).
- Use odometry (/ego_racecar/odom) to get the car forward speed v.
- Brake if:
  1) d_min < hard_stop_dist   (too close)
  2) v > min_speed_for_ttc and d_min / v < ttc_threshold   (TTC too small)
- When braking, publish speed=0 to /drive repeatedly (override) at publish_hz.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyNode(Node):
    """
    Subscribes:
      - /scan (LaserScan)
      - /ego_racecar/odom (Odometry): uses twist.twist.linear.x as forward speed

    Publishes:
      - /drive (AckermannDriveStamped): speed=0 when emergency braking
    """

    def __init__(self) -> None:
        super().__init__("safety_node")

        # Parameters
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")

        self.declare_parameter("forward_fov_deg", 40.0)     # look in front cone
        self.declare_parameter("hard_stop_dist", 0.35)      # meters
        self.declare_parameter("ttc_threshold", 0.8)        # seconds
        self.declare_parameter("min_speed_for_ttc", 0.05)   # m/s
        self.declare_parameter("publish_hz", 50.0)          # publish stop repeatedly

        # If scan is missing/invalid, braking is safer for the demo.
        self.declare_parameter("brake_on_invalid_lidar", True)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)

        self.forward_fov_deg = float(self.get_parameter("forward_fov_deg").value)
        self.hard_stop_dist = float(self.get_parameter("hard_stop_dist").value)
        self.ttc_threshold = float(self.get_parameter("ttc_threshold").value)
        self.min_speed_for_ttc = float(self.get_parameter("min_speed_for_ttc").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.brake_on_invalid_lidar = bool(self.get_parameter("brake_on_invalid_lidar").value)

        # State
        self.latest_scan: Optional[LaserScan] = None
        self.speed_mps: float = 0.0
        self.emergency: bool = False
        self.last_d_min: float = float("inf")
        self.reason: str = ""

        # ROS I/O 
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)

        period = 1.0 / max(self.publish_hz, 1e-6)
        self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"safety_node started. scan={self.scan_topic}, odom={self.odom_topic}, drive={self.drive_topic}"
        )

    def on_scan(self, msg: LaserScan) -> None:
        """Save the latest LIDAR scan."""
        self.latest_scan = msg

    def on_odom(self, msg: Odometry) -> None:
        """Save the latest forward speed (m/s)."""
        self.speed_mps = float(msg.twist.twist.linear.x)

    def on_timer(self) -> None:
        """
        Run AEB check periodically.
        If emergency is True, publish stop to /drive (speed=0).
        """
        self.update_emergency()

        if self.emergency:
            self.publish_stop()


    def update_emergency(self) -> None:
        """
        Decide whether we should brake.
        Sets: self.emergency, self.reason, self.last_d_min
        """
        if self.latest_scan is None:
            self.emergency = False
            self.reason = ""
            return

        d_min = self.forward_min_distance(self.latest_scan)
        self.last_d_min = d_min

        # If scan has no usable points
        if not math.isfinite(d_min):
            if self.brake_on_invalid_lidar:
                self.emergency = True
                self.reason = "INVALID_LIDAR"
            else:
                self.emergency = False
                self.reason = ""
            return

        # Rule 1: hard stop
        if d_min < self.hard_stop_dist:
            self.emergency = True
            self.reason = f"HARD_STOP (<{self.hard_stop_dist:.2f}m)"
            return

        # Rule 2: TTC stop
        v = self.speed_mps
        if v > self.min_speed_for_ttc:
            ttc = d_min / max(v, 1e-6)
            if ttc < self.ttc_threshold:
                self.emergency = True
                self.reason = f"TTC_STOP (<{self.ttc_threshold:.2f}s)"
                return

        # Otherwise: safe
        self.emergency = False
        self.reason = ""

    def forward_min_distance(self, scan: LaserScan) -> float:
        """
        Get the minimum valid distance in a forward cone around 0 rad.
        We discard invalid values:
        - NaN/Inf
        - <= 0.0
        - < range_min or > range_max
        Returns:
            Minimum distance (meters) in the cone, or +inf if none.
        """
        if scan.angle_increment == 0.0 or len(scan.ranges) == 0:
            return float("inf")

        half = math.radians(self.forward_fov_deg) / 2.0
        n = len(scan.ranges)

        # Convert angles to index range
        i_start = int(math.floor(((-half) - scan.angle_min) / scan.angle_increment))
        i_end = int(math.ceil(((+half) - scan.angle_min) / scan.angle_increment))

        i_start = max(0, min(n - 1, i_start))
        i_end = max(0, min(n - 1, i_end))
        if i_start > i_end:
            i_start, i_end = i_end, i_start

        d_min = float("inf")
        for r in scan.ranges[i_start : i_end + 1]:
            if not math.isfinite(r):
                continue
            if r <= 0.0:
                continue
            if r < scan.range_min or r > scan.range_max:
                continue
            if r < d_min:
                d_min = r

        return d_min

    def publish_stop(self) -> None:

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()

        out.drive.speed = 0.0
        out.drive.steering_angle = 0.0
        out.drive.steering_angle_velocity = 0.0
        out.drive.acceleration = 0.0
        out.drive.jerk = 0.0

        self.drive_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()