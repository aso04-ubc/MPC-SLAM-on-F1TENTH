"""
Main ROS 2 node for gap following with visualization.

This node implements a gap following algorithm for autonomous navigation using LiDAR data.
It processes laser scan data to identify safe driving gaps, computes steering commands using
PID control, and publishes drive commands integrated with the safety system.

Features:
- Real-time LiDAR processing with obstacle inflation
- Cost-based gap selection with forward bias
- PID steering control with stabilization
- Speed adaptation based on steering angle and distance
- Side wall clearance detection and correction
- Live parameter tuning via config.json hot-reload
- Visualization HUD for debugging and monitoring

start ros2 bridge using 
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
"""

import time

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import math
import json
import os
from pathlib import Path


from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from dev_b7_interfaces.msg import DriveControlMessage

from gap_following.gap_utils import GapFollowAlgo
from gap_following.PID_control import PIDControl

SIM = False

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory
    )
except Exception:  # pragma: no cover - optional during non-ROS linting
    PackageNotFoundError = Exception
    get_package_share_directory = None


class GapFollowingNode(Node):
    """
    ROS 2 node that implements gap following algorithm with visualization.

    This node subscribes to LiDAR scan data and odometry, processes the data to find
    safe driving gaps, computes steering and speed commands, and publishes drive
    commands wrapped in DriveControlMessage for integration with the safety system.
    """

    def __init__(self):
        super().__init__('gap_following_node')

        # Declare ROS parameters
        self.declare_parameter('sim', False)
        self.sim = self.get_parameter('sim').get_parameter_value().bool_value

        self.get_logger().info(f"Simulation mode: {self.sim}")

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Subscribe to odometry for speed-based adjustments
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom' if self.sim else '/odom',
            self.odom_callback,
            10
        )

        # Publishers
        self.drive_pub = self.create_publisher(
            DriveControlMessage,
            DriveControlMessage.BUILTIN_TOPIC_NAME_STRING,
            10
        )

        # # HUD visualization publishers
        self.pub_lidar_hud = None
        if not SIM:
            self.pub_lidar_hud = self.create_publisher(
                Image, '/gap_following/lidar_hud', 10
            )
        
        self.create_timer(1/30, self.visualization_timer_callback)

        # Variables to store latest LiDAR info for HUD
        self._latest_ranges = None
        self._latest_angle_min = None
        self._latest_angle_increment = None
        self._latest_target_angle = 0.0
        self._latest_mixed_steering = 0.0
        self._latest_final_steering = 0.0
        self._latest_speed = 0.0
        
        self.canvas_h = 480 
        self.canvas_w = 640
        self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        self.last_mod_time = -1.0
        self._warned_missing_config = False
        self.tune = {
            'max_speed': 2.0,
            'min_speed': 0.5,
            'MAX_STEERING_ANGLE': 1.4,
            'kp': 0.6,
            'ki': 0.0,
            'kd': 0.1,
            'smoothing_alpha': 0.5,
            'cost_diff': 30,
            'steering_deadzone': 0.05,
            'steering_gain': 0.7,
            'speed_gain': 20.0,
            'max_range': 4.0,
            'min_safe_distance': 0.25,
            'car_width': 0.6,
            'disparity_threshold': 1.4,
            'smoothing_window_size': 10,
            'info_print_frequency': 30,
            'max_accel': 1.5,
            'max_decel': 2.5,
            'distance_percentile': 0.9,
            'distance_slowdown_threshold': 1.5,
            'side_distance_threshold': 0.35,
            'side_steering_nudge': 0.06,
            'side_sector_start_deg': 55.0,
            'side_sector_end_deg': 100.0,
            'side_distance_percentile': 20.0,
        }

        # State variables
        self.prev_steering_angle = 0.0
        self.current_speed = 0.0
        self.have_odom = False
        self._last_scan_time_ns = None
        self._last_commanded_speed = 0.0

        self._apply_tuning_from_dict(self.tune)
        self._rebuild_controllers()

        default_config_path = "/home/jetson/f1tenth_ws/src/MS5_B7/gap_following/config.json"
        self.declare_parameter('config_path', default_config_path)
        configured_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.config_path = configured_path if configured_path else default_config_path

        # Load config from JSON (first load)
        self.hot_reload_config()
        # Periodic reload timer (every 1 second)
        self.create_timer(1.0, self.hot_reload_config)

        self.get_logger().info('Gap Following Node initialized')

        if SIM:
            self.last_scan_received_time = 0

    def _resolve_default_config_path(self):
        """
        Resolve the default configuration file path.

        Prefer installed share path, then source-tree fallback.
        This allows the package to work both in development and installed environments.

        Returns:
            str: Path to the default config.json file
        """
        if get_package_share_directory is not None:
            try:
                installed = Path(get_package_share_directory('gap_following')) / 'config.json'
                if installed.exists():
                    return str(installed)
            except PackageNotFoundError:
                pass

        # .../gap_following/gap_following/main_node.py -> .../gap_following/config.json
        source_tree = Path(__file__).resolve().parents[1] / 'config.json'
        return str(source_tree)

    def _apply_tuning_from_dict(self, tune):
        """
        Validate and apply runtime tuning values from configuration dictionary.

        This method takes a dictionary of tuning parameters and applies them to the
        node's control parameters with validation and clamping to safe ranges.

        Args:
            tune (dict): Dictionary containing tuning parameters with keys matching
                        the parameter names used in the node.
        """
        self.max_speed = max(0.0, float(tune['max_speed']))
        self.min_speed = max(0.0, min(float(tune['min_speed']), self.max_speed))
        self.MAX_STEERING_ANGLE = max(1e-3, float(tune['MAX_STEERING_ANGLE']))

        self.kp = float(tune['kp'])
        self.ki = float(tune['ki'])
        self.kd = float(tune['kd'])

        self.smoothing_alpha = float(np.clip(float(tune['smoothing_alpha']), 0.0, 1.0))
        self.cost_diff = max(0.0, float(tune['cost_diff']))

        self.steering_deadzone = max(0.0, float(tune['steering_deadzone']))
        self.steering_gain = float(tune['steering_gain'])
        self.speed_gain = max(0.0, float(tune['speed_gain']))

        self.max_range = max(0.1, float(tune['max_range']))
        self.min_safe_distance = max(0.0, float(tune['min_safe_distance']))
        self.car_width = max(0.01, float(tune['car_width']))
        self.disparity_threshold = max(0.0, float(tune['disparity_threshold']))
        self.smoothing_window_size = max(1, int(tune['smoothing_window_size']))
        self.info_print_frequency = max(1, int(tune['info_print_frequency']))

        self.max_accel = max(0.01, float(tune.get('max_accel', 1.5)))
        self.max_decel = max(0.01, float(tune.get('max_decel', 2.5)))

        self.distance_percentile = float(np.clip(float(tune.get('distance_percentile', 0.9)), 0.0, 1.0))
        self.distance_slowdown_threshold = max(0.01, float(tune.get('distance_slowdown_threshold', 1.5)))

        self.side_distance_threshold = max(0.01, float(tune.get('side_distance_threshold', 0.35)))
        self.side_steering_nudge = max(0.0, float(tune.get('side_steering_nudge', 0.06)))
        self.side_sector_start_deg = float(np.clip(float(tune.get('side_sector_start_deg', 55.0)), 0.0, 180.0))
        self.side_sector_end_deg = float(np.clip(float(tune.get('side_sector_end_deg', 100.0)), 0.0, 180.0))
        if self.side_sector_end_deg < self.side_sector_start_deg:
            self.side_sector_start_deg, self.side_sector_end_deg = self.side_sector_end_deg, self.side_sector_start_deg
        self.side_distance_percentile = float(
            np.clip(float(tune.get('side_distance_percentile', 20.0)), 0.0, 100.0)
        )

    def _rebuild_controllers(self):
        """
        Recreate control helpers when gains/limits change.

        This method reinitializes the PID controller and gap following algorithm
        with the current parameter values. Called when parameters are updated
        via config file hot-reload.
        """
        self.pid = PIDControl(
            self.kp,
            self.ki,
            self.kd,
            steering_limit=self.MAX_STEERING_ANGLE
        )
        self.gap_algo = GapFollowAlgo(
            max_range=self.max_range,
            min_safe_distance=self.min_safe_distance,
            car_width=self.car_width,
            disparity_threshold=self.disparity_threshold,
            smoothing_window_size=self.smoothing_window_size,
        )

    def odom_callback(self, msg):
        """
        Process odometry data and extract current speed.

        Updates the current vehicle speed from odometry messages, which is used
        for speed-based control adjustments and safety monitoring.

        Args:
            msg: Odometry message from /ego_racecar/odom topic containing
                 twist.linear.x with current speed in m/s
        """
        # Use absolute speed so reverse motion does not invert speed limiting.
        self.current_speed = abs(msg.twist.twist.linear.x)
        self.have_odom = True

    def hot_reload_config(self):
        """
        Check if config.json was modified and reload parameters if so.

        This method implements hot-reloading of configuration parameters by monitoring
        the config file for changes. When changes are detected, it reloads the JSON
        file and updates the control parameters without restarting the node.
        """
        try:
            if not os.path.exists(self.config_path):
                if not self._warned_missing_config:
                    self.get_logger().warning(
                        f"Config file not found: {self.config_path}. Using in-code defaults."
                    )
                    self._warned_missing_config = True
                return

            current_mod_time = os.path.getmtime(self.config_path)
            if current_mod_time <= self.last_mod_time:
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict):
                raise ValueError("Top-level config must be a JSON object")

            self.tune.update(loaded)
            self._apply_tuning_from_dict(self.tune)
            self._rebuild_controllers()
            self.last_mod_time = current_mod_time
            self._warned_missing_config = False

            self.get_logger().info(f"Config reloaded from: {self.config_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to reload config: {e}")

    def _limit_speed_change(self, desired_speed):
        """
        Apply acceleration and deceleration limits using odom speed when available.

        This method enforces realistic acceleration/deceleration constraints based on
        the time elapsed since the last command and the current vehicle speed.

        Args:
            desired_speed (float): The target speed in m/s

        Returns:
            float: The bounded speed respecting acceleration limits
        """
        now_ns = self.get_clock().now().nanoseconds
        dt = 0.0 if self._last_scan_time_ns is None else (now_ns - self._last_scan_time_ns) * 1e-9
        self._last_scan_time_ns = now_ns

        if dt <= 0.0:
            bounded_speed = desired_speed
        else:
            ref_speed = self.current_speed if self.have_odom else self._last_commanded_speed
            min_allowed = max(0.0, ref_speed - self.max_decel * dt)
            max_allowed = ref_speed + self.max_accel * dt
            bounded_speed = float(np.clip(desired_speed, min_allowed, max_allowed))

        bounded_speed = float(np.clip(bounded_speed, 0.0, self.max_speed))
        self._last_commanded_speed = bounded_speed
        return bounded_speed

    def _compute_side_clearance_bias(
        self,
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max
    ):
        """
        Return a small steering bias away from the closer side wall.

        Analyzes left and right side sectors to detect if the vehicle is too close
        to a wall on one side. If so, applies a small steering correction to move
        away from the wall.

        Args:
            ranges: Array of LiDAR range measurements
            angle_min: Starting angle of the scan in radians
            angle_increment: Angular increment between measurements in radians
            range_min: Minimum valid range value
            range_max: Maximum valid range value

        Returns:
            tuple: (steering_bias, left_distance, right_distance)
                   steering_bias: Small steering correction in radians
                   left_distance: Average distance to left side obstacles
                   right_distance: Average distance to right side obstacles
        """
        if len(ranges) == 0:
            return 0.0, np.nan, np.nan

        angles = angle_min + np.arange(len(ranges)) * angle_increment
        valid = np.isfinite(ranges)
        if range_min is not None:
            valid &= (ranges >= range_min)
        if range_max is not None and range_max > 0.0:
            valid &= (ranges <= range_max)

        side_start = np.deg2rad(self.side_sector_start_deg)
        side_end = np.deg2rad(self.side_sector_end_deg)

        left_mask = valid & (angles >= side_start) & (angles <= side_end)
        right_mask = valid & (angles <= -side_start) & (angles >= -side_end)

        if np.any(left_mask):
            left_dist = float(np.percentile(ranges[left_mask], self.side_distance_percentile))
        else:
            left_dist = np.nan

        if np.any(right_mask):
            right_dist = float(np.percentile(ranges[right_mask], self.side_distance_percentile))
        else:
            right_dist = np.nan

        side_bias = 0.0
        left_close = np.isfinite(left_dist) and left_dist < self.side_distance_threshold
        right_close = np.isfinite(right_dist) and right_dist < self.side_distance_threshold

        if left_close and (not right_close or left_dist < right_dist):
            # Too close on left -> nudge right (negative steering).
            side_bias = -self.side_steering_nudge
        elif right_close and (not left_close or right_dist < left_dist):
            # Too close on right -> nudge left (positive steering).
            side_bias = self.side_steering_nudge

        return side_bias, left_dist, right_dist

    def scan_callback(self, msg):
        """
        Process LiDAR scan data and publish drive commands with visualization.

        This is the main callback that processes incoming LiDAR data through the
        gap following pipeline:
        1. Preprocess LiDAR ranges (smoothing, filtering)
        2. Apply disparity extender for obstacle inflation
        3. Find optimal driving gap using cost-based selection
        4. Apply steering stabilization and side wall corrections
        5. Compute speed based on steering angle and distance
        6. Publish drive commands via DriveControlMessage

        Args:
            msg: LaserScan message from /scan topic containing range measurements
        """

        if SIM:
            if time.time() - self.last_scan_received_time < 0.05:  # 20 Hz limit
                return
            self.last_scan_received_time = time.time() 

        # Convert scan message to numpy array
        ranges = np.array(msg.ranges)

        # Use GapFollowAlgo to find target steering angle with dynamic bubble radius
        target_steering_angle, best_idx = self.gap_algo.process_lidar_and_find_gap(
            ranges,
            msg.angle_min,
            msg.angle_increment
        )

        # Apply steering smoothing to reduce oscillation
        smoothed_steering_angle = target_steering_angle

        # Apply stabilization: deadzone and gain dampening
        stabilized_steering_angle = self.apply_steering_stabilization(smoothed_steering_angle)

        # Side-wall safety nudge: if one side is too close, add a tiny opposite correction.
        side_bias, left_dist, right_dist = self._compute_side_clearance_bias(
            ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max
        )

        mixed_steering_angle = stabilized_steering_angle + side_bias
        final_steering_angle = mixed_steering_angle
        final_steering_angle = float(
            np.clip(final_steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        )

        if side_bias != 0.0:
            self.get_logger().info(
                f"Side nudge {side_bias:+.3f} (L={left_dist:.2f}m, R={right_dist:.2f}m, thr={self.side_distance_threshold:.2f}m)"
            )

        steering_factor = abs(final_steering_angle / self.MAX_STEERING_ANGLE)
        speed_factor = 1.0 - steering_factor       # quadratic scaling works well
        speed_factor = np.clip(speed_factor, 0.1, 1.0) # avoid going too slow or negative

        # Distance-aware damping: use the 90th percentile (configurable) of the focused region
        window_size = max(3, int(np.deg2rad(20) / msg.angle_increment))
        start_idx = max(0, best_idx - window_size)
        end_idx = min(len(ranges), best_idx + window_size + 1)
        focused_ranges = ranges[start_idx:end_idx]
        gap_percentile_dist = float(np.percentile(focused_ranges, self.distance_percentile * 100.0))
        distance_factor = np.clip(
            gap_percentile_dist / self.distance_slowdown_threshold**1.6,
            0,
            1.0
        )
        speed_factor *= distance_factor
        
        self.get_logger().info(
            f"Speed factor: {speed_factor:.3f} (steer {steering_factor:.3f}, dist {distance_factor:.3f}, p{self.distance_percentile*100:.0f}={gap_percentile_dist:.2f}m)"
        )


        target_speed = self.min_speed + speed_factor * (self.max_speed - self.min_speed)
        # current_speed = self._limit_speed_change(target_speed)
        current_speed = target_speed
                
        current_steering_angle = self.pid.run(self, final_steering_angle)
        current_steering_angle = float(
            np.clip(current_steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        )
        # Publish drive command with final processed steering
        self.publish_drive(current_speed, current_steering_angle)

        # Store data for HUD
        self._latest_ranges = ranges
        self._latest_angle_min = msg.angle_min
        self._latest_angle_increment = msg.angle_increment
        self._latest_target_angle = target_steering_angle
        self._latest_mixed_steering = mixed_steering_angle
        self._latest_final_steering = current_steering_angle
        self._latest_speed = current_speed
        
        
    def visualization_timer_callback(self):
        """
        Timer callback for updating the LiDAR visualization HUD.

        Renders the current LiDAR data, gap selection, and steering commands
        as a visual overlay for debugging and monitoring. Publishes the
        visualization image to the appropriate topic.
        """
        if self._latest_ranges is None:
            return  # nothing yet

        # Call your HUD function with the stored latest data
        self._render_lidar_hud(
            msg=None,  # you can modify your HUD to accept ranges directly if needed
            target_steering_angle=self._latest_target_angle,
            mixed_steering_angle=self._latest_mixed_steering,
            final_steering_angle=self._latest_final_steering,
            current_speed=self._latest_speed,
            ranges=self._latest_ranges,
            angle_min=self._latest_angle_min,
            angle_increment=self._latest_angle_increment
        )
        if SIM:
            cv2.imshow('LiDAR HUD', self.canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        elif self.pub_lidar_hud is not None:
            try:
                self.pub_lidar_hud.publish(self._cv2_to_ros_image(self.canvas))
            except Exception as e:
                self.get_logger().error(f"Failed to publish LiDAR HUD: {e}")

    def smooth_steering(self, new_angle):
        """
        Apply Exponential Moving Average (EMA) filter to steering angle.

        Smooths steering commands to reduce oscillations and provide more stable
        control. Uses the configured smoothing_alpha parameter.

        Args:
            new_angle: new target steering angle in radians

        Returns:
            float: smoothed steering angle in radians
        """
        # Exponential Moving Average formula
        smoothed_angle = self.smoothing_alpha * new_angle + \
                        (1.0 - self.smoothing_alpha) * self.prev_steering_angle
        
        # Update previous angle for next iteration
        self.prev_steering_angle = smoothed_angle
        
        return smoothed_angle
    
    def apply_steering_stabilization(self, steering_angle):
        """
        Apply deadzone and gain dampening to stabilize steering on straights.

        Implements two stabilization techniques:
        1. Deadzone: Ignores small steering corrections to prevent noise reaction
        2. Gain dampening: Reduces steering gain to prevent overcorrection

        Args:
            steering_angle: smoothed steering angle in radians

        Returns:
            float: stabilized steering angle in radians
        """
        # Apply deadzone - ignore tiny corrections to prevent noise reaction
        if abs(steering_angle) < self.steering_deadzone:
            return 0.0
        
        # Apply gain dampening to prevent overcorrection
        stabilized_angle = steering_angle * self.steering_gain
        
        return stabilized_angle
    
    def publish_drive(self, speed, steering_angle):
        """
        Publish drive command using DriveControlMessage for safety node integration.

        Wraps the computed speed and steering commands in a DriveControlMessage
        and publishes to the safety system's drive control topic. The message
        includes priority information for arbitration with other control sources.

        Args:
            speed: desired speed in m/s
            steering_angle: desired steering angle in radians
        """
        # Create internal Ackermann drive message
        temp_msg = AckermannDriveStamped()
        temp_msg.header.stamp = self.get_clock().now().to_msg()
        temp_msg.header.frame_id = 'base_link'
        temp_msg.drive = AckermannDrive()
        temp_msg.drive.speed = float(speed)
        temp_msg.drive.steering_angle = float(steering_angle)
        
        # Wrap in DriveControlMessage for safety node
        full_msg = DriveControlMessage()
        full_msg.active = True
        full_msg.priority = 1004  # Medium priority (adjust as needed)
        full_msg.drive = temp_msg
        
        self.drive_pub.publish(full_msg)

    def _cv2_to_ros_image(self, cv_image, encoding='bgr8'):
        """
        Convert a CV2 image to a ROS Image message without cv_bridge.

        Creates a ROS Image message from an OpenCV image array by copying
        the image data directly. This avoids the cv_bridge dependency for
        simple image publishing.

        Args:
            cv_image: OpenCV image array (numpy array)
            encoding: ROS image encoding string (default: 'bgr8')

        Returns:
            Image: ROS Image message ready for publishing
        """
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = cv_image.shape[:2]
        msg.encoding = encoding
        if len(cv_image.shape) == 2:
            msg.step = msg.width
        else:
            msg.step = msg.width * cv_image.shape[2]
        msg.data = cv_image.tobytes()
        return msg

    def _render_lidar_hud(self, msg, target_steering_angle, mixed_steering_angle, final_steering_angle, current_speed,
                        ranges=None, angle_min=None, angle_increment=None):
        """
        Draw LiDAR scatter visualization and publish to topic.

        Renders a top-down view of the LiDAR data showing:
        - Raw LiDAR points (white)
        - Processed/inflated obstacles (red)
        - Selected gap region (green overlay)
        - Target and final steering directions (blue/cyan arrows)
        - Vehicle position and status text

        Args:
            msg: LaserScan message (can be None if ranges provided directly)
            target_steering_angle: Raw steering angle from gap selection
            mixed_steering_angle: Steering after side wall corrections
            final_steering_angle: Final steering after PID control
            current_speed: Current commanded speed
            ranges: LiDAR range array (extracted from msg if not provided)
            angle_min: Scan start angle (extracted from msg if not provided)
            angle_increment: Angular increment (extracted from msg if not provided)
        """
        self.canvas.fill(10)
        cx, cy, scale = 320, 440, 70

        if msg is not None:
            ranges = np.array(msg.ranges)
            angle_min = msg.angle_min
            angle_increment = msg.angle_increment

        if ranges is None or angle_min is None or angle_increment is None:
            return  # nothing to draw

        n_points = len(ranges)
        angles = angle_min + np.arange(n_points) * angle_increment

        # --- Raw LiDAR points (white) ---
        xs = (cx - (ranges * np.sin(angles) * scale)).astype(int)
        ys = (cy - (ranges * np.cos(angles) * scale)).astype(int)
        for x, y in zip(xs, ys):
            if 0 <= x < self.canvas_w and 0 <= y < self.canvas_h:
                cv2.circle(self.canvas, (x, y), 2, (255, 255, 255), -1)

        # --- Processed / disparity-extended points (red) ---
        processed = self.gap_algo._preprocess_ranges(ranges, angle_min, angle_increment)
        disparity = self.gap_algo._apply_disparity_extender(processed, angle_increment)
        
        fov = np.deg2rad(180)
        mask = (angles >= -fov/2) & (angles <= fov/2)
        angles_processed  = angles[mask]

        xs_d = (cx - (disparity * np.sin(angles_processed) * scale)).astype(int)
        ys_d = (cy - (disparity * np.cos(angles_processed) * scale)).astype(int)
        inflated_mask = disparity < processed

        for x, y, inflated in zip(xs_d, ys_d, inflated_mask):
            if inflated and 0 <= x < self.canvas_w and 0 <= y < self.canvas_h:
                cv2.circle(self.canvas, (x, y), 3, (0, 0, 255), -1)

        # --- Show the gap selected by the algorithm (green overlay) ---
        target_angle, best_idx = self.gap_algo.process_lidar_and_find_gap(ranges, angle_min, angle_increment)

        # Reconstruct the "best region" around the median
        # We'll draw a span of +/- window_size beams around best_idx
        window_size = int(np.deg2rad(10) / angle_increment)
        window_size = max(window_size, 3)
        start_idx = max(0, best_idx - window_size)
        end_idx = min(n_points - 1, best_idx + window_size-1)

        for i in range(start_idx, end_idx):
            xg = int(cx - (disparity[i] * np.sin(angles_processed[i]) * scale))
            yg = int(cy - (disparity[i] * np.cos(angles_processed[i]) * scale))
            if 0 <= xg < self.canvas_w and 0 <= yg < self.canvas_h:
                cv2.circle(self.canvas, (xg, yg), 4, (0, 255, 0), 1)  # green = gap

        # --- Ego vehicle marker ---
        cv2.circle(self.canvas, (cx, cy), 6, (0, 200, 0), -1)

        # --- Target steering arrow (blue) ---
        arrow_len = 120
        x_t = int(cx - math.sin(target_steering_angle) * arrow_len)
        y_t = int(cy - math.cos(target_steering_angle) * arrow_len)
        cv2.arrowedLine(self.canvas, (cx, cy), (x_t, y_t), (255, 0, 0), 2, tipLength=0.2)

        # --- Final steering arrow (cyan) ---
        x_f = int(cx - math.sin(final_steering_angle) * arrow_len)
        y_f = int(cy - math.cos(final_steering_angle) * arrow_len)
        cv2.arrowedLine(self.canvas, (cx, cy), (x_f, y_f), (255, 255, 0), 2, tipLength=0.2)

        # --- Text info ---
        cv2.putText(self.canvas, f"LiDAR Steer: {target_steering_angle:.3f} rad", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(self.canvas, f"Mixed Steer: {mixed_steering_angle:.3f} rad", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(self.canvas, f"Final Steer: {final_steering_angle:.3f} rad", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(self.canvas, f"Speed: {current_speed:.2f} m/s", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Legend
        cv2.putText(self.canvas, "White=Raw  Red=Inflated  Green=Gap  Blue=Target  Cyan=Final", 
                    (10, self.canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
def main(args=None):
    """
    Main entry point for the gap following node.

    Initializes the ROS 2 node, creates the GapFollowingNode instance,
    and runs the ROS 2 event loop until shutdown.

    Args:
        args: Command line arguments (passed to rclpy.init)
    """
    rclpy.init(args=args)
    
    node = GapFollowingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
