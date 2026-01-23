from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64

import numpy as np
from time import time

from wall_follow.config import DEBUG
from wall_follow.KalmanFilter import SimpleKalmanFilter
from wall_follow.PID_control import PIDControl

class DataProcess(Node):

    PID_control : PIDControl

    def __init__(self):
        super().__init__("wall_follow")

        self.windows_size = 100 # larger better 

        self.kf_left_angle = SimpleKalmanFilter(10.0, 0.1)
        self.kf_right_angle = SimpleKalmanFilter(10.0, 0.1)
        self.kf_left_dist = SimpleKalmanFilter(10.0, 0.1)
        self.kf_right_dist = SimpleKalmanFilter(10.0, 0.1)

        ## set up PID controller
        self.PID_control = PIDControl()

        # make sure to get most recent messages
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


        # TBD 
        # self.control_info_pusher = self.create_publisher(
        #     AckermannDriveStamped,
        #     'drive',
        #     10
        # )

        if DEBUG:
            self.pub_debug_left_dist   = self.create_publisher(Float64, "debug/left_dist", 10)
            self.pub_debug_left_angle  = self.create_publisher(Float64, "debug/left_angle", 10)
            self.pub_debug_right_dist  = self.create_publisher(Float64, "debug/right_dist", 10)
            self.pub_debug_right_angle = self.create_publisher(Float64, "debug/right_angle", 10)


    def OnReceiveOdomInfo(self, odom_data : Odometry):
        # Process the odometry data
        pass


    # process usually take less than 1 ms on my machine, using windows size of 100.
    def OnReceiveLaserInfo(self, lidar_data):
        # get basic info from the input data
        ranges = np.array(lidar_data.ranges)
        self.angle_increment = lidar_data.angle_increment
        self.angle_min = lidar_data.angle_min
        self.angle_max = lidar_data.angle_max
        
        all_angles = self.angle_min + np.arange(len(ranges)) * self.angle_increment
        
        # filter out invalid data
        ranges = np.nan_to_num(ranges, 
                               nan=lidar_data.range_min, 
                               posinf=lidar_data.range_max, 
                               neginf=lidar_data.range_min)

        # get vectors on each side
        idx_0 = self.get_target_index(0.0)
        idx_90 = -1
        idx_neg_90 = 0

        # get oriented distances and angles
        left_dist, left_tangent = self.process_lidar_one_side(
            ranges[idx_0 : idx_90], 
            all_angles[idx_0 : idx_90]
        )

        right_dist, right_tangent = self.process_lidar_one_side(
            ranges[idx_neg_90 : idx_0], 
            all_angles[idx_neg_90 : idx_0]
        )

        # Apply Kalman Filter to smooth the results
        left_tangent = self.kf_left_angle.update(left_tangent)
        left_dist = self.kf_left_dist.update(left_dist)
        right_tangent = self.kf_right_angle.update(right_tangent)
        right_dist = self.kf_right_dist.update(right_dist)


        if DEBUG:
            msg_ld = Float64(); msg_ld.data = float(left_dist)
            msg_la = Float64(); msg_la.data = float(left_tangent)
            msg_rd = Float64(); msg_rd.data = float(right_dist)
            msg_ra = Float64(); msg_ra.data = float(right_tangent)

            self.pub_debug_left_dist.publish(msg_ld)
            self.pub_debug_left_angle.publish(msg_la)
            self.pub_debug_right_dist.publish(msg_rd)
            self.pub_debug_right_angle.publish(msg_ra)
            # self.get_logger().info(f"Left dist: {left_dist:.2f}, angle: {left_tangent:.2f} | Right dist: {right_dist:.2f}, angle: {right_tangent:.2f}")
        

    def process_lidar_one_side(self, local_ranges, local_angles):

        # Find the index of the minimum distance, which should be close to the wall
        min_idx = np.argmin(local_ranges)

        start_idx = max(0, min_idx - self.windows_size)
        end_idx = min(len(local_ranges) - 1, min_idx + self.windows_size)

        # Get the angle and distance data for fitting
        fit_ranges = local_ranges[start_idx:end_idx]
        fit_angles = local_angles[start_idx:end_idx] 

        # Get x and y coordinates
        x = fit_ranges * np.cos(fit_angles)
        y = fit_ranges * np.sin(fit_angles)

        if len(x) < 3:
            return fit_ranges[0], 0.0

        # Perform linear regression (least squares) to find the line parameters
        A_matrix = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A_matrix, y, rcond=None)[0]

        # Calculate the distance from the origin to the line (Ax + By + C = 0)
        dist = abs(c) / np.sqrt(m**2 + 1)
        tangent_angle = np.arctan(m)

        return dist, tangent_angle

    def get_target_index(self,  target_angle : float):
        raw_index = (target_angle - self.angle_min) / self.angle_increment
        target_index = int(np.clip(raw_index, 0, (self.angle_max - self.angle_min) / self.angle_increment - 1))
        return target_index
        

        

