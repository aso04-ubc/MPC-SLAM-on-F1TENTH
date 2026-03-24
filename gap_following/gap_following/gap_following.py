import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from dev_b7_interfaces.msg import DriveControlMessage
from std_msgs.msg import Float64 # For your debug messages
import math
import numpy as np
from f1tenth_utils.drive_utils import wrap_drive_message




def main(args=None):
    rclpy.init(args=args)
    gap_following = GapFollowing()
    rclpy.spin(gap_following)

    # Good practice to destroy node explicitly
    gap_following.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
