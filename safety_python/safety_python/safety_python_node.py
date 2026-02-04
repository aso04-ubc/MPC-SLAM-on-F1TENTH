#!/usr/bin/env python3
"""
Safety Node with Autonomous Emergency Braking (AEB)

This node provides collision avoidance through:
1. Priority-based drive command arbitration
2. Autonomous Emergency Braking (AEB) based on laser scan data
3. Time-To-Collision (TTC) and minimum distance thresholds

Subscribed Topics:
    /drive_control (dev_b7_interfaces/DriveControlMessage): Drive commands with priority
    /scan (sensor_msgs/LaserScan): Lidar data for obstacle detection  
    /ego_racecar/odom (nav_msgs/Odometry): Vehicle odometry for speed

Published Topics:
    /drive (ackermann_msgs/AckermannDriveStamped): Final drive commands
"""

import math
import rclpy
from rclpy.node import Node

# Message imports
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from dev_b7_interfaces.msg import DriveControlMessage


class SafetyNode(Node):
    """
    Safety node implementing AEB and priority-based command arbitration.
    
    The node monitors laser scan data to detect potential collisions and
    automatically applies emergency braking when needed. It also arbitrates
    between multiple drive command sources based on priority levels.
    """

    def __init__(self):
        super().__init__('safety_node')
        
        # Declare and get parameters
        self.declare_parameter('ttc_threshold', 0.5)
        self.declare_parameter('distance_threshold', 0.25)
        
        self.ttc_threshold = self.get_parameter('ttc_threshold').value
        self.distance_threshold = self.get_parameter('distance_threshold').value
        
        self.get_logger().info(
            f'Safety Node initialized with TTC threshold: {self.ttc_threshold}s, '
            f'Distance threshold: {self.distance_threshold}m'
        )
        
        # State variables
        self.current_speed = 0.0  # Current vehicle speed (m/s)
        self.is_aeb_active = False  # Whether AEB is currently engaged
        self.last_drive_msg = AckermannDriveStamped()  # Last drive command
        
        # Priority map: {priority: DriveControlMessage}
        # Higher priority values take precedence
        self.priority_map = {}
        
        # Publishers
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.drive_control_sub = self.create_subscription(
            DriveControlMessage,
            DriveControlMessage.BUILTIN_TOPIC_NAME_STRING,
            self.drive_control_callback,
            10
        )
        
        # Timer for publishing AEB commands when active
        # Runs at 200 Hz (5ms) to ensure quick response
        self.aeb_timer = self.create_timer(0.005, self.aeb_timer_callback)
        
        self.get_logger().info('Safety Node started successfully')
    
    def odom_callback(self, msg: Odometry):
        """
        Update current vehicle speed from odometry.
        
        Args:
            msg: Odometry message containing vehicle velocity
        """
        self.current_speed = abs(msg.twist.twist.linear.x)
    
    def scan_callback(self, msg: LaserScan):
        """
        Process laser scan data to detect collisions and engage/disengage AEB.
        
        Algorithm:
        1. For each valid scan point, calculate angle and distance
        2. Compute range_rate = vehicle_speed * cos(angle)
        3. Calculate TTC = distance / range_rate (if approaching)
        4. Engage AEB if TTC < threshold OR distance < threshold
        
        Args:
            msg: LaserScan message with range measurements
        """
        v = self.current_speed
        
        # Initialize tracking variables
        min_ttc = float('inf')
        min_distance = float('inf')
        
        # Only check the front 180 degrees (skip side/rear scans)
        # This avoids triggering on obstacles to the side
        quarter = len(msg.ranges) // 4
        
        for i in range(quarter, len(msg.ranges) - quarter):
            r = msg.ranges[i]
            
            # Skip invalid measurements
            if math.isnan(r) or math.isinf(r) or r < msg.range_min or r > msg.range_max:
                continue
            
            # Track closest obstacle
            min_distance = min(min_distance, r)
            
            # Calculate angle of this scan point
            angle = msg.angle_min + i * msg.angle_increment
            
            # Calculate range rate (closing speed toward obstacle)
            # Positive range_rate means we're approaching the obstacle
            range_rate = v * math.cos(angle)
            
            # Calculate TTC only if approaching
            if range_rate > 0:
                ttc = r / range_rate
                min_ttc = min(min_ttc, ttc)
        
        # Check if AEB should be engaged
        if min_ttc < self.ttc_threshold or min_distance < self.distance_threshold:
            if not self.is_aeb_active:
                self.get_logger().error(
                    f'AEB TRIGGERED! TTC: {min_ttc:.3f}s, Distance: {min_distance:.3f}m'
                )
                self.is_aeb_active = True
        else:    
            # if self.is_aeb_active:
            #     self.is_aeb_active = False
            #     self.get_logger().info('AEB Released')
            pass
    
    def drive_control_callback(self, msg: DriveControlMessage):
        """
        Handle drive control commands with priority-based arbitration.
        
        Priority Arbitration:
        - If command is active: Add/update in priority map
        - If command is inactive: Remove from priority map
        - Always publish the highest priority active command
        
        Args:
            msg: Drive control message with priority and drive command
        """
        # Store the message for AEB to preserve steering angle
        self.last_drive_msg = msg.drive
        
        if msg.active:
            # Command is active: add/update in priority map
            self.priority_map[msg.priority] = msg
            
            # Get highest priority command
            if self.priority_map:
                highest_priority = max(self.priority_map.keys())
                highest_msg = self.priority_map[highest_priority]
                
                # Only publish if AEB is not active
                if not self.is_aeb_active:
                    self.drive_publisher.publish(highest_msg.drive)
        else:
            # Command is inactive: remove from priority map
            if msg.priority in self.priority_map:
                del self.priority_map[msg.priority]
    
    def aeb_timer_callback(self):
        """
        Timer callback that publishes emergency stop commands when AEB is active.
        
        Runs at 200 Hz to ensure quick response. Preserves steering angle
        from the last drive command but sets speed to 0.
        """
        if self.is_aeb_active:
            # Create stop message preserving steering angle
            stop_msg = AckermannDriveStamped()
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_link'
            
            # Preserve steering from last command
            stop_msg.drive.steering_angle = self.last_drive_msg.drive.steering_angle
            stop_msg.drive.steering_angle_velocity = self.last_drive_msg.drive.steering_angle_velocity
            
            # Set speed to 0 (emergency stop)
            stop_msg.drive.speed = 0.0
            stop_msg.drive.acceleration = 0.0
            
            self.drive_publisher.publish(stop_msg)


def main(args=None):
    """Main entry point for the safety node."""
    rclpy.init(args=args)
    
    try:
        node = SafetyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
