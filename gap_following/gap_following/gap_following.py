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
        self.minimum_gap_width = 5.0

        self.car_width = 0.5
        self.steering_limit = 0.7

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
        window = 5
        smooth_ranges = np.convolve(range_np, np.ones(window)/window, mode='same')

        bubble_ranges = self.bubble(msg, smooth_ranges)

        largest_gap_start, largest_gap_end = self.find_largest_gap(bubble_ranges)

        if not largest_gap_start or not largest_gap_end: # stop if there is no gap found, the car is stuck
            self.current_velocity = 0

        gap_start_angle = msg.angle_min + largest_gap_start * msg.angle_increment
        gap_end_angle = msg.angle_min + largest_gap_end * msg.angle_increment

        desired_angle = (gap_end_angle - gap_start_angle)/2.0

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

        gaps_padded = np.concatenate([0], gaps.astype(int), [0])
        gaps_marked = np.diff(gaps_padded)

        gap_starts = np.where(gaps_marked == 1)
        gap_ends = np.where(gaps_marked == -1)

        gap_widths = gap_ends - gap_starts

        usable_gaps = gap_widths > self.minimum_gap_width

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
        nearest_distance = np_ranges(nearest_index)

        mask_angle = math.atan(self.car_width/nearest_distance)
        mask_lasers = int(mask_angle/msg.angle_increment)

        bubbled_ranges = np_ranges.copy()
        bubbled_ranges[min(0, nearest_index-mask_lasers):max(nearest_index+mask_lasers, len(np_ranges))]

        return bubbled_ranges



def main(args=None):
    rclpy.init(args=args)
    gap_following = GapFollowing()
    rclpy.spin(gap_following)
    
    # Good practice to destroy node explicitly
    gap_following.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()