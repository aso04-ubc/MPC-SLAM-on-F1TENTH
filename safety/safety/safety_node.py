from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import array
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

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

        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )   

        self.laser_receiver = self.create_subscription(
            LaserScan,
            "scan",
            self.OnReceiveLaserInfo,
            qos_policy
        )

        self.control_info_pusher = self.create_publisher(
            AckermannDriveStamped,
            'drive',
            10
        )

        self.pre_frame = None
        self.last_stamp = None


    def OnReceiveLaserInfo(self, laser_info : LaserScan):

        # get_current_sec
        current_stamp = laser_info.header.stamp.sec + laser_info.header.stamp.nanosec * 1e-9
        
        if self.last_stamp is None:
            dt = 0.1
        else:
            dt = current_stamp - self.last_stamp

        if dt < 1/40:
            return 

        self.last_stamp = current_stamp
        
        scan_time = laser_info.scan_time
        if scan_time == 0.0:
            scan_time = dt

        # self.get_logger().info(str(scan_time))

        # deal with ttc
        current_ranges = np.array(laser_info.ranges)
        
        current_ranges[np.isinf(current_ranges)] = 0
        current_ranges[np.isnan(current_ranges)] = 0
        current_ranges[current_ranges == 0] = 0

        if self.pre_frame is not None:
            delta_dist = current_ranges - self.pre_frame
            velocity = delta_dist / dt

            danger_mask = (velocity < -0.5) & (current_ranges < 5.0)
            
            if np.any(danger_mask):
                dists = current_ranges[danger_mask]
                vels = velocity[danger_mask]
                
                ttc_values = dists / np.abs(vels)
                
                min_ttc = np.min(ttc_values)
                self.get_logger().info(f"Min TTC: {min_ttc:.2f} s")

                if min_ttc < 0.3 or np.min(current_ranges) < 0.2:
                    self.SentBreak()
            
            elif np.min(current_ranges) < 0.2:
                self.SentBreak()

        self.pre_frame = current_ranges
    
    def SentBreak(self):
        self.get_logger().info("Break Sent")
        new_pack = AckermannDriveStamped()
        new_pack.drive.speed = 0.0
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