from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from dev_b7_interfaces.msg import DriveControlMessage

import numpy as np

from wall_follow.config import DEBUG
from wall_follow.KalmanFilter import SimpleKalmanFilter
from wall_follow.PID_control import PIDControl

class DataProcess(Node):

    PID_control : PIDControl

    def __init__(self):
        super().__init__("wall_follow")

        self.window_size = 50

        # Kalman filter parameters (configurable via ROS parameters)
        # Angle filters
        self.declare_parameter('kalman_angle_R', 300.0)
        self.declare_parameter('kalman_angle_Q', 0.1)
        kalman_angle_R = self.get_parameter('kalman_angle_R').get_parameter_value().double_value
        kalman_angle_Q = self.get_parameter('kalman_angle_Q').get_parameter_value().double_value

        # Distance filters
        self.declare_parameter('kalman_distance_R', 100.0)
        self.declare_parameter('kalman_distance_Q', 0.1)
        kalman_distance_R = self.get_parameter('kalman_distance_R').get_parameter_value().double_value
        kalman_distance_Q = self.get_parameter('kalman_distance_Q').get_parameter_value().double_value

        # steering filters
        self.declare_parameter('kalman_steering_R', 10.0)
        self.declare_parameter('kalman_steering_Q', 0.1)
        kalman_steering_R = self.get_parameter('kalman_steering_R').get_parameter_value().double_value
        kalman_steering_Q = self.get_parameter('kalman_steering_Q').get_parameter_value().double_value

        self.kf_left_angle = SimpleKalmanFilter(kalman_angle_R, kalman_angle_Q)
        self.kf_right_angle = SimpleKalmanFilter(kalman_angle_R, kalman_angle_Q)
        self.kf_left_dist = SimpleKalmanFilter(kalman_distance_R, kalman_distance_Q)
        self.kf_right_dist = SimpleKalmanFilter(kalman_distance_R, kalman_distance_Q)
        self.kf_steering = SimpleKalmanFilter(kalman_steering_R, kalman_steering_Q)

        # Desired distance from wall (meters)
        self.desired_distance = 1.0
        
        # set up PID controller
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

        self.control_info_pusher = self.create_publisher(
            DriveControlMessage,
            '/drive_control',
            10
        )

        self.get_logger().info("Data Process Node Initialized.")
        if DEBUG:
            self.get_logger().info("Debug mode is ON.")
            self.pub_debug_left_dist   = self.create_publisher(Float64, "debug/left_dist", 10)
            self.pub_debug_left_angle  = self.create_publisher(Float64, "debug/left_angle", 10)
            self.pub_debug_right_dist  = self.create_publisher(Float64, "debug/right_dist", 10)
            self.pub_debug_right_angle = self.create_publisher(Float64, "debug/right_angle", 10)
            self.pub_debug_dip_steering = self.create_publisher(Float64, "debug/dip_steering", 10)


    def OnReceiveOdomInfo(self, odom_data : Odometry):
        # Process the odometry data
        pass


    # process usually take less than 1 ms on my machine, using window size of 100.
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
        idx_middle = self.get_target_index(0.0)
        idx_leftmost = self.get_target_index(np.pi / 2)
        idx_rightmost = self.get_target_index(-np.pi / 2)

        # get oriented distances and angles
        left_dist, left_tangent = self.process_lidar_one_side(
            ranges[idx_middle : idx_leftmost], 
            all_angles[idx_middle : idx_leftmost]
        )

        right_dist, right_tangent = self.process_lidar_one_side(
            ranges[idx_rightmost : idx_middle], 
            all_angles[idx_rightmost : idx_middle]
        )

        # Apply Kalman Filter to smooth the results
        left_tangent = self.kf_left_angle.update(left_tangent)
        left_dist = self.kf_left_dist.update(left_dist)
        right_tangent = self.kf_right_angle.update(right_tangent)
        right_dist = self.kf_right_dist.update(right_dist)

        # Calculate distance error (positive = too far, negative = too close)
        distance_error = right_dist - self.desired_distance
        
        # Run PID controller (follow right wall)
        pid_command = self.PID_control.run(self, right_tangent, distance_error)

        pid_command = self.kf_steering.update(pid_command)
        
        temp_msg = AckermannDriveStamped()
        temp_msg.drive = AckermannDrive()
        temp_msg.drive.steering_angle = pid_command
        temp_msg.drive.speed = 3.0  

        full_msg = DriveControlMessage()
        full_msg.active = True
        full_msg.priority = 1000 # Subject to change
        full_msg.drive = temp_msg
        
        self.control_info_pusher.publish(full_msg)
        
        if DEBUG:
            self.get_logger().info(f"Right: dist={right_dist:.3f}, angle={right_tangent:.3f}, error={distance_error:.3f}, steering={pid_command:.3f}")


        if DEBUG:
            msg_ld = Float64(); msg_ld.data = float(left_dist)
            msg_la = Float64(); msg_la.data = float(left_tangent)
            msg_rd = Float64(); msg_rd.data = float(right_dist)
            msg_ra = Float64(); msg_ra.data = float(right_tangent)
            msg_sd = Float64(); msg_sd.data = float(pid_command)
            

            self.pub_debug_left_dist.publish(msg_ld)
            self.pub_debug_left_angle.publish(msg_la)
            self.pub_debug_right_dist.publish(msg_rd)
            self.pub_debug_right_angle.publish(msg_ra)
            self.pub_debug_dip_steering.publish(msg_sd)
            # self.get_logger().info(f"Left dist: {left_dist:.2f}, angle: {left_tangent:.2f} | Right dist: {right_dist:.2f}, angle: {right_tangent:.2f}")
        

    def process_lidar_one_side(self, local_ranges, local_angles):

        # Find the index of the minimum distance, which should be close to the wall
        min_idx = np.argmin(local_ranges)

        start_idx = max(0, min_idx - self.window_size)
        end_idx = min(len(local_ranges) - 1, min_idx + self.window_size)

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

        coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, y, rcond=None)

        # If the matrix is rank-deficient, the fit is unreliable; fall back to a safe default
        if rank < A_matrix.shape[1]:
            return fit_ranges[0], 0.0
        
        m, c = coeffs

        # Calculate the distance from the origin to the line (Ax + By + C = 0)
        denom = np.sqrt(m**2 + 1.0)

        if not np.isfinite(m) or not np.isfinite(c) or denom == 0.0 or not np.isfinite(denom):
            return fit_ranges[0], 0.0
        
        dist = abs(c) / denom

        tangent_angle = np.arctan(m)

        return dist, tangent_angle

    def get_target_index(self,  target_angle : float):
        raw_index = (target_angle - self.angle_min) / self.angle_increment
        target_index = int(np.clip(raw_index, 0, (self.angle_max - self.angle_min) / self.angle_increment - 1))
        return target_index
        

        
