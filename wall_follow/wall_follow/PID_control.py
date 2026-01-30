# wall_follow/PID_control.py
# Fengwei

import math
from time import time
from typing import Optional


class PIDControl:
    """
    PID controller for left-wall-following.

    This controller uses data from the left wall to maintain
    a target distance from it.

    Input:
      - left_angle  : angle of left wall relative to car heading (rad)
      - left_dist   : distance to left wall (m)
      - right_angle : angle of right wall relative to car heading (rad) [unused]
      - right_dist  : distance to right wall (m) [unused]

    Error definition:
      - Distance error: (left_dist - target_dist)
        Positive = car is too far from left wall, should steer left
        Negative = car is too close to left wall, should steer right
      - Heading error: left wall angle, indicates if car is 
        angled relative to the wall

    Output:
      - steering command (rad)
    """

    def __init__(
        self,
        kp: float = 0.3,
        ki: float = 0.0,
        kd: float = 0.0,
        kp_heading: float = 0.8,
        lookahead_L: float = 0.0,
        steering_limit: float = 0.7,
        integral_limit: float = 1.0,
        d_filter_alpha: float = 0,      # alpha -> closer to 1 = smoother D
        target_dist: float = 1.0        # target distance from left wall (m)
    ):
        # PID gains for distance error
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Separate gain for heading correction
        self.kp_heading = kp_heading

        # Lookahead distance for combining heading into distance error
        self.L = lookahead_L
        
        # Target distance from left wall
        self.target_dist = target_dist

        # Output and state limits
        self.steering_limit = abs(steering_limit)
        self.integral_limit = abs(integral_limit)

        # Derivative low-pass filter parameter (0 = no filter, 1 = full filter)
        self.d_filter_alpha = d_filter_alpha

        # Internal states
        self.prev_error: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.integral = 0.0
        self.d_filtered = 0.0

    def reset(self):
        """Reset PID internal state (useful when restarting the controller)."""
        self.prev_error = None
        self.prev_time = None
        self.integral = 0.0
        self.d_filtered = 0.0

    def _get_time(self, sender):
        """
        Get current time in seconds.
        Prefer ROS clock if sender is a Node; otherwise fall back to system time.
        """
        try:
            return sender.get_clock().now().nanoseconds * 1e-9
        except Exception:
            return time()

    def run(
        self, 
        sender, 
        left_angle: float, 
        left_dist: float,
        right_angle: float,
        right_dist: float
    ) -> float:
        """
        Run one PID control step using left wall only.

        Parameters
        ----------
        sender : Node
            ROS2 node (used only for clock).
        left_angle : float
            Angle of left wall relative to car heading (rad).
        left_dist : float
            Distance to left wall (m).
        right_angle : float
            Angle of right wall relative to car heading (rad). [unused]
        right_dist : float
            Distance to right wall (m). [unused]

        Returns
        -------
        float
            Steering command (rad). Positive = steer left, Negative = steer right.
        """
        # Note: right_angle and right_dist are kept in the API for compatibility
        # but are not used in left-wall-only following mode
        _ = right_angle, right_dist
        
        # Distance Error (left wall only)
        # Positive error means car is too far from left wall -> need to steer left (positive steering)
        # Negative error means car is too close to left wall -> need to steer right (negative steering)
        dist_error = self.target_dist - left_dist
        
        # Heading Error (left wall only)
        # Uses left wall angle to determine heading relative to the wall
        # Positive angle means car is angled away from wall
        # Negative angle means car is angled toward wall
        heading_error = left_angle
        
        # Combined Error with Lookahead
        # Project where the car will be based on heading error
        # This helps anticipate turns and reduces oscillation
        error = dist_error + self.L * math.sin(heading_error)
        
        # Time Delta
        current_time = self._get_time(sender)

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time

        # Reject unreasonable dt (startup / pause / clock jump)
        if dt <= 1e-6 or dt > 0.5:
            dt = 0.0

        # Integral Term (with anti-windup)
        if dt > 0.0 and self.ki != 0.0:
            self.integral += error * dt
            self.integral = max(
                -self.integral_limit,
                min(self.integral, self.integral_limit)
            )

        # Derivative Term (with low-pass filter)
        if dt > 0.0 and self.prev_error is not None:
            d_raw = (error - self.prev_error) / dt
        else:
            d_raw = 0.0

        # First-order low-pass filter: alpha closer to 1 = smoother
        alpha = self.d_filter_alpha
        self.d_filtered = alpha * self.d_filtered + (1.0 - alpha) * d_raw

        # PID Control Law
        # Distance-based PID
        steering = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.d_filtered
        )
        
        # Add direct heading correction for faster response
        steering += self.kp_heading * heading_error

        # Output Saturation
        steering = max(
            -self.steering_limit,
            min(steering, self.steering_limit)
        )

        # Update state for next iteration
        self.prev_error = error
        self.prev_time = current_time

        return float(steering)