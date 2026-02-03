# wall_follow/PID_control.py
# Fengwei

import math
from time import time
from typing import Optional


class PIDControl:
    """
    PID controller for wall-following using both walls.

    This controller uses data from both left and right walls to keep
    the vehicle centered between them.

    Input:
      - left_angle  : angle of left wall relative to car heading (rad)
      - left_dist   : distance to left wall (m)
      - right_angle : angle of right wall relative to car heading (rad)
      - right_dist  : distance to right wall (m)

    Error definition:
      - Distance error: (right_dist - left_dist) / 2
        Positive = car is closer to left wall, should steer right
        Negative = car is closer to right wall, should steer left
      - Heading error: average of wall angles, indicates if car is 
        angled relative to the corridor

    Output:
      - steering command (rad)
    """

    def __init__(
        self,
        kp: float = 0.3,
        ki: float = 0.05,
        kd: float = 0.0,
        kp_heading: float = 0.8,
        lookahead_L: float = 0.3,
        steering_limit: float = 0.7,
        integral_limit: float = 1.0,
        d_filter_alpha: float = 0       # alpha -> closer to 1 = smoother D
    ):
        # PID gains for distance error
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Separate gain for heading correction
        self.kp_heading = kp_heading

        # Lookahead distance for combining heading into distance error
        self.L = lookahead_L

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
        Run one PID control step using both walls.

        Parameters
        ----------
        sender : Node
            ROS2 node (used only for clock).
        left_angle : float
            Angle of left wall relative to car heading (rad).
        left_dist : float
            Distance to left wall (m).
        right_angle : float
            Angle of right wall relative to car heading (rad).
        right_dist : float
            Distance to right wall (m).

        Returns
        -------
        float
            Steering command (rad). Positive = steer left, Negative = steer right.
        """
        
        # Distance Error
        # Positive error means car is closer to left wall -> need to steer right (negative steering)
        # Negative error means car is closer to right wall -> need to steer left (positive steering)
        dist_error = (left_dist - right_dist) / 2.0
        
        # Heading Error
        # Average of both wall angles gives the heading error relative to corridor
        # If both walls have positive angle, car is angled to the right
        # If both walls have negative angle, car is angled to the left
        # Note: left wall angle is typically negative when parallel, right wall is positive
        heading_error = (left_angle + right_angle) / 2.0
        
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