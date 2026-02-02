import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import math
import numpy as np

class GapFollowing(Node):

    def __init__(self):
        super().__init__('gap_following')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.current_velocity = 0.0

        self.minimum_distance = 3.0

    def odom_callback(self, msg):
        self.current_velocity = msg.twist.twist.linear.x

    def scan_callback(self, msg):
        pass

def main(args=None):
    rclpy.init(args=args)
    gap_following = GapFollowing()
    rclpy.spin(gap_following)
    
    # Good practice to destroy node explicitly
    gap_following.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()