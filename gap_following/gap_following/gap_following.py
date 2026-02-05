import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from dev_b7_interfaces.msg import DriveControlMessage
from std_msgs.msg import Float64 # For your debug messages
import math
import numpy as np


class GapFollowing(Node):

    def __init__(self):
        super().__init__('gap_following')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # self.drive_pub = self.create_publisher(DriveControlMessage, DriveControlMessage.BUILTIN_TOPIC_NAME_STRING, 10) 
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.current_velocity = 0.0

        self.minimum_distance = 3.0
        self.minimum_gap_width = 5.0

        self.car_width = 0.3
        self.steering_limit = 0.7

        self.kp = 0.8
        self.kd = 0.2
        self.ki = 0.0

    def odom_callback(self, msg):
        self.current_velocity = msg.twist.twist.linear.x

    """
    Gap Following
        1. smooth data using a moving average
        2. create a bubble around the obstacles for safe driving
        3. look for distances greater than self.minimum distance to find gaps
        4. calculate the width of the gap and follow the widest
        5. use np.diff to quickly find where a gap appears
        6. use PID to steer the car towards the center of the gap
    """

    def scan_callback(self, msg):

        range_np = np.array(msg.ranges)

        # filter out noisy range data
        # range_np = np.clip(range_np, 0, 5.0)
        window = 5
        smooth_ranges = np.convolve(range_np, np.ones(window) / window, mode='same')

        bubble_ranges = self.bubble(msg, smooth_ranges)
        disparity_extended_ranges = self.extend_disparity(msg, smooth_ranges)

        largest_gap_start, largest_gap_end = self.find_largest_gap(disparity_extended_ranges)

        gap_start_angle = 0
        gap_end_angle = 0

        if not largest_gap_start or not largest_gap_end:  # straighten the wheel and stop if there is no gap found, the car is stuck
            self.current_velocity = 0
        else :
            gap_start_angle = msg.angle_min + largest_gap_start * msg.angle_increment
            gap_end_angle = msg.angle_min + largest_gap_end * msg.angle_increment

        desired_angle = (gap_end_angle + gap_start_angle) / 2.0 # angle heading of the gap relative to the car, trying to make this 0

        steer_angle = max(-self.steering_limit, min(self.steering_limit, desired_angle))

        # drive_msg = AckermannDriveStamped()
        # drive_msg.drive = AckermannDrive()

        # control_msg = DriveControlMessage()
        # control_msg.active = True
        # control_msg.priority = 1000 # Subject to change
        # control_msg.drive = drive_msg

        # drive_msg.drive.steering_angle = float(steer_angle * self.kp)
        # drive_msg.drive.speed = float(2.0-abs(steer_angle))

        # self.drive_pub.publish(control_msg)

        drive = AckermannDriveStamped()

        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = "base_link"

        drive.drive.steering_angle = float(steer_angle * self.kp)
        drive.drive.speed = float(2.0 - abs(steer_angle)) 

        self.drive_pub.publish(drive)


    """
    Gap finding
    
    uses the data in ranges[] to look for gaps and returns the start and end index of the largest gap
        
        Arguments
            - @msg: contains the range data for the lidar
        Returns
            - the start and end indices of the largest gap
    """

    def find_largest_gap(self, ranges_np):

        gaps = ranges_np > self.minimum_distance

        gaps_padded = np.concatenate(([0.0], gaps.astype(int), [0.0]))
        gaps_marked = np.diff(gaps_padded)

        gap_starts = np.where(gaps_marked == 1)[0]
        gap_ends = np.where(gaps_marked == -1)[0]

        gap_widths = gap_ends - gap_starts

        usable_gaps = gap_widths > self.minimum_gap_width

        widest_start, widest_end = None, None

        if usable_gaps.any():
            widest = np.argmax(usable_gaps * gap_widths)
            widest_start = gap_starts[widest]
            widest_end = gap_ends[widest]

        return widest_start, widest_end

    """
    Apply a bubble around obstacles
        Arguments
            - @self: data associated with GapFollowing
            - @msg: LiDAR data
            - @np_ranges: numpy array of range data
        Return
            - new range data with bubble applied
    """

    def bubble(self, msg, np_ranges):
        nearest_index = np.argmin(np_ranges)
        nearest_distance = np_ranges[nearest_index]

        mask_angle = math.atan(self.car_width / nearest_distance)
        mask_lasers = int(mask_angle / msg.angle_increment)

        bubbled_ranges = np_ranges.copy()
        bubbled_ranges[min(0, nearest_index - mask_lasers):max(nearest_index + mask_lasers, len(np_ranges))]

        return bubbled_ranges

    def extend_disparity(self, msg, np_ranges, gap_threshold=0.3): 
            
            finder = np.abs(np.diff(np_ranges))
            disparities_idx = np.where(finder > gap_threshold)[0] 

            extended = np.copy(np_ranges)

            if len(disparities_idx) == 0:
                return np_ranges

            for index in disparities_idx:
                if index + 1 >= len(np_ranges):
                    continue

                distance_to_obstacle = min(np_ranges[index], np_ranges[index+1])
                
                # protect against divide by zero
                if distance_to_obstacle < 0.1:
                    distance_to_obstacle = 0.1
                    
                extend_angle = np.arctan((self.car_width) / distance_to_obstacle)
                
                mask_width = int(np.ceil(extend_angle / msg.angle_increment))

                begin = int(max(0, index - mask_width))
                end = int(min(len(np_ranges), index + mask_width + 1))

                extended[begin:end] = np.minimum(extended[begin:end], distance_to_obstacle)

            return extended

def main(args=None):
    rclpy.init(args=args)
    gap_following = GapFollowing()
    rclpy.spin(gap_following)

    # Good practice to destroy node explicitly
    gap_following.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
