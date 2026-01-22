from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
import rclpy
import numpy as np

# TTC threshold in seconds - if TTC is below this, emergency braking is triggered
g_TTC_Threshold = 0.8

# Minimum speed threshold to avoid division by zero and false positives when stationary
g_MinimumSpeed = 0.1

class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_node")

        self.current_speed = 0.0

        self.scan_subscription = self.create_subscription(
                LaserScan, "/scan",
                self.OnProcessLaserScanInfo, 10)
        
        self.odom_subscription = self.create_subscription(
                Odometry, "/ego_racecar/odom",
                self.OnProcessOdometry, 10)

        self.controll_publisher = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)

    def OnProcessOdometry(self, msg: Odometry):
        self.current_speed = msg.twist.twist.linear.x

    def OnProcessLaserScanInfo(self, info: LaserScan):
        if self.current_speed < g_MinimumSpeed:
            return

        min_ttc = float('inf')

        for i, distance in enumerate(info.ranges):
            if np.isnan(distance) or np.isinf(distance):
                continue
            if distance < info.range_min or distance > info.range_max:
                continue

            angle = info.angle_min + i * info.angle_increment

            range_rate = self.current_speed * np.cos(angle)

            if range_rate > 0:
                ttc = distance / range_rate

                if ttc < min_ttc:
                    min_ttc = ttc

        if min_ttc != float('inf'):
            print(f"Min TTC: {min_ttc:.3f}s at speed {self.current_speed:.2f}m/s")

        if min_ttc < g_TTC_Threshold:
            msg = AckermannDriveStamped()
            msg.drive.speed = 0.0
            print(f"EMERGENCY BRAKING! TTC: {min_ttc:.3f}s")
            self.controll_publisher.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    print("Safety node running")
    main()