import time
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from dev_b7_interfaces.msg import DriveControlMessage

from wall_follow.config import DEBUG
from wall_follow.KalmanFilter import SimpleKalmanFilter
from wall_follow.PID_control import PIDControl


class DataProcess(Node):
    """Left wall follow node using dual-beam measurement and PID control."""

    # Two beam angles for dual-beam measurement (relative to the vehicle's left 90-degree direction)
    BEAM_ANGLE_A = np.deg2rad(45)   # front-left 45 degrees
    BEAM_ANGLE_B = np.deg2rad(90)   # pure left 90 degrees

    # Speed and steering relationship
    SPEED = 1.0      # forward speed
    

    def __init__(self):
        super().__init__("wall_follow")
        self.start_time = time.time()
        self.last_time = 0
        
        # Target follow distance
        self.declare_parameter('target_distance', 0.5)
        self.target_distance = self.get_parameter('target_distance').get_parameter_value().double_value

        # Kalman filter parameters
        self.declare_parameter('kalman_R', 5.0)    # measurement noise
        self.declare_parameter('kalman_Q', 0.5)    # process noise
        kalman_R = self.get_parameter('kalman_R').get_parameter_value().double_value
        kalman_Q = self.get_parameter('kalman_Q').get_parameter_value().double_value

        # Initialize Kalman filters
        self.kf_distance = SimpleKalmanFilter(kalman_R, kalman_Q)
        self.kf_angle = SimpleKalmanFilter(kalman_R, kalman_Q)
        self.kf_control = SimpleKalmanFilter(0.1, 0.5)
        
        # PID controller
        self.pid_controller = PIDControl(
            kp=0.6,
            ki=0.0,
            kd=0.2,
            lookahead_L=0.05,
            steering_limit=0.6,
            d_filter_alpha = 0.2
        )

        # QoS configuration
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to LaserScan data
        self.laser_receiver = self.create_subscription(
            LaserScan, "scan", self.on_scan_received, sensor_qos
        )

        # Publish control commands
        self.control_publisher = self.create_publisher(
            DriveControlMessage,
            DriveControlMessage.BUILTIN_TOPIC_NAME_STRING,
            10
        )

        self.get_logger().info(f"Wall Follow Node initialized. Target distance: {self.target_distance}m")
        
        # Debug topics
        if DEBUG:
            self.pub_debug_distance = self.create_publisher(Float64, "debug/left_dist", 10)
            self.pub_debug_angle = self.create_publisher(Float64, "debug/left_angle", 10)
            self.pub_debug_steering = self.create_publisher(Float64, "debug/steering", 10)
            self.pub_debug_speed = self.create_publisher(Float64, "debug/speed", 10)





    def on_scan_received(self, scan: LaserScan):
        """Process LaserScan data and perform left wall following."""
        
        # 40Hz frequency limit
        current_time = time.time_ns()
        if current_time - self.last_time < 25_000_000:  # 25ms = 40Hz
            return
        self.last_time = current_time

        # Use dual-beam method to compute left wall distance and angle
        distance, angle = self.dual_beam_measurement(scan)
        
        if distance is None:
            return

        # Kalman filter smoothing
        distance = self.kf_distance.update(distance)
        angle = self.kf_angle.update(angle)

        # Compute distance error
        distance_error = distance - self.target_distance

        # PID control
        steering = self.pid_controller.run(self, angle, distance_error)

        # Deadband processing
        if abs(steering) < 0.02:
            steering = 0.0


        speed = self.SPEED

        steering = self.kf_control.update(steering)

        # Publish drive command
        self.publish_drive_command(steering, speed)

        # Debug output
        if DEBUG:
            self.publish_debug_info(distance, angle, steering, speed)

    def dual_beam_measurement(self, scan: LaserScan):
        """
        Dual-beam measurement: use two laser beams to compute the perpendicular distance
        to the left wall and the angle of the vehicle relative to the wall.

        Beam A: front-left 45 degrees (BEAM_ANGLE_A)
        Beam B: pure left 90 degrees (BEAM_ANGLE_B)

        Returns:
            (distance, alpha): perpendicular distance to the wall and the vehicle-wall angle
        """
        ranges = np.array(scan.ranges)
        
        # Get indices for the two beams
        idx_a = self.angle_to_index(scan, self.BEAM_ANGLE_A)
        idx_b = self.angle_to_index(scan, self.BEAM_ANGLE_B)
        
        if idx_a is None or idx_b is None:
            return None, None
        
        # Get ranges for the two beams, use a local window median filter
        window = 3
        a = self.get_filtered_range(ranges, idx_a, window, scan.range_max)
        b = self.get_filtered_range(ranges, idx_b, window, scan.range_max)
        
        if a <= 0 or b <= 0 or a > scan.range_max or b > scan.range_max:
            return None, None
        
        # Angle between the two beams
        theta = self.BEAM_ANGLE_B - self.BEAM_ANGLE_A  # 45 degrees
        
        # Compute the angle alpha between the vehicle and the wall
        # alpha = atan((a*cos(theta) - b) / (a*sin(theta)))
        numerator = a * np.cos(theta) - b
        denominator = a * np.sin(theta)
        
        if abs(denominator) < 1e-6:
            alpha = 0.0
        else:
            alpha = np.arctan2(numerator, denominator)
        
        # Compute the perpendicular distance to the wall
        # D = b * cos(alpha)
        distance = b * np.cos(alpha)
        
        # Validity check
        if not np.isfinite(distance) or not np.isfinite(alpha):
            return None, None
        
        return distance, alpha

    def angle_to_index(self, scan: LaserScan, angle: float):
        """Convert an angle to a LaserScan array index."""
        if angle < scan.angle_min or angle > scan.angle_max:
            return None
        index = int((angle - scan.angle_min) / scan.angle_increment)
        return max(0, min(index, len(scan.ranges) - 1))

    def get_filtered_range(self, ranges, idx, window, max_range):
        """Get the median-filtered range value around a specified index."""
        start = max(0, idx - window)
        end = min(len(ranges), idx + window + 1)
        local = ranges[start:end]
        
        # Filter invalid values
        valid = local[(local > 0) & (local < max_range) & np.isfinite(local)]
        
        if len(valid) == 0:
            return max_range
        
        return np.median(valid)

    def publish_drive_command(self, steering: float, speed: float):
        """Publish drive control command."""
        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive()
        drive_msg.drive.steering_angle = float(steering)
        drive_msg.drive.speed = float(speed)

        msg = DriveControlMessage()
        msg.active = True
        msg.priority = 1000
        msg.drive = drive_msg

        self.control_publisher.publish(msg)

    def publish_debug_info(self, distance: float, angle: float, steering: float, speed: float):
        """Publish debug information."""
        msg_d = Float64(); msg_d.data = float(distance)
        msg_a = Float64(); msg_a.data = float(angle)
        msg_s = Float64(); msg_s.data = float(steering)
        msg_v = Float64(); msg_v.data = float(speed)
        
        self.pub_debug_distance.publish(msg_d)
        self.pub_debug_angle.publish(msg_a)
        self.pub_debug_steering.publish(msg_s)
        self.pub_debug_speed.publish(msg_v)
        
        self.get_logger().debug(
            f"Distance: {distance:.2f}m, Angle: {np.rad2deg(angle):.1f}deg, "
            f"Steering: {steering:.3f}rad, Speed: {speed:.1f}m/s"
        )

        

