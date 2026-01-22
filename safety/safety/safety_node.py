from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import array
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from time import time

import numpy as np


def list_list_minus( list1, list2):
    result = [x - y for x, y in zip(list1, list2)]
    return result

def list_num_mul (list1, num):
    return [x * num for x in list1]


class SafetyNode(Node):
    pre_frame : array
    
    def __init__(self):
        super().__init__("safety_node")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/ego_racecar/odom",
            self.OnReceiveOdomInfo,
            sensor_qos
        )

        self.laser_receiver = self.create_subscription(
            LaserScan,
            "scan",
            self.OnReceiveLaserInfo,
            sensor_qos
        )

        self.control_info_pusher = self.create_publisher(
            AckermannDriveStamped,
            'drive',
            10
        )
        
        self.current_speed = 0.0
        self.ttc_thresholds = 0.5
        self.ttc_threshold_distance = 0.2

        self.pre_frame = None
        self.last_stamp = None

        self.get_logger().info("Safety Node Online")


    def OnReceiveOdomInfo(self, msg: Odometry):
        """
        Updates the current speed of the car from the odometry topic.
        We focus on linear.x (forward velocity).
        """
        self.current_speed = msg.twist.twist.linear.x


    def OnReceiveLaserInfo(self, laser_info : LaserScan):
        """
        Callback function to process incoming laser scan data and 
        determine if a braking action is necessary based on 
        Time-To-Collision (TTC) calculations.
        """

        ranges = np.array(laser_info.ranges)
        
        ranges[np.isinf(ranges)] = np.inf
        ranges[np.isnan(ranges)] = np.inf

        angles = np.arange(
            laser_info.angle_min, 
            laser_info.angle_max, 
            laser_info.angle_increment
        )
        
        if len(angles) > len(ranges):
            angles = angles[:len(ranges)]
        elif len(angles) < len(ranges):
            ranges = ranges[:len(angles)]

        range_rates = self.current_speed * np.cos(angles)
        range_rates = np.maximum(range_rates, 0.001)

        ttc_values = ranges / range_rates

        min_ttc = np.min(ttc_values)

        if min_ttc < self.ttc_thresholds:
            self.get_logger().info("Sent break to speed {} due to ttc: {}".format(0, min_ttc))
            self.SentBreak()

        if np.min(ranges) < self.ttc_threshold_distance:
            self.get_logger().info("Sent break due to min distance")
            self.SentBreak()

    def SentBreak(self, speed = 0.0):
        """
        Sends a braking command to the vehicle by publishing
        """
        new_pack = AckermannDriveStamped()
        new_pack.drive.speed = speed
        self.control_info_pusher.publish(new_pack)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = SafetyNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()