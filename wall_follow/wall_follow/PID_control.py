# wall_follow/PID_control.py
# PID controller for left wall following

import math
from time import time
from typing import Optional


class PIDControl:
    """
    PID controller for left wall following.
        The wall is on the left side of the vehicle. The goal is to maintain a specified distance
        and keep the vehicle parallel to the wall.

        Input:
            - theta: angle between vehicle heading and wall (rad)
                             Positive = vehicle heading toward the wall (requires right turn)
                             Negative = vehicle heading away from the wall (requires left turn)
            - distance_error: distance error (m) = current_distance - target_distance
                             Positive = too far from the wall (requires left turn)
                             Negative = too close to the wall (requires right turn)

        Output:
            - steering command (rad), Positive = steer right, Negative = steer left
    """

    def __init__(
        self,
        kp: float = 1.5,
        ki: float = 0.0,
        kd: float = 0.5,
        lookahead_L: float = 0.8,
        steering_limit: float = 0.5,
        integral_limit: float = 0.3,
        d_filter_alpha: float = 0.2
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.L = lookahead_L
        self.steering_limit = abs(steering_limit)
        self.integral_limit = abs(integral_limit)
        self.d_filter_alpha = d_filter_alpha

        # Internal state
        self.prev_error: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.integral = 0.0
        self.d_filtered = 0.0

    def reset(self):
        """Reset PID internal state."""
        self.prev_error = None
        self.prev_time = None
        self.integral = 0.0
        self.d_filtered = 0.0

    def _get_time(self, sender):
        """Get current time from ROS clock or system time."""
        try:
            return sender.get_clock().now().nanoseconds * 1e-9
        except Exception:
            return time()

    def run(self, sender, theta: float, distance_error: float) -> float:
        """
        Run one PID step.

        Parameters
        ----------
        sender : Node
            ROS2 node for clock access.
        theta : float
            Angle between car heading and wall (rad).
            Positive = car pointing toward wall.
        distance_error : float
            Distance error (m) = current_dist - target_dist.
            Positive = too far, Negative = too close.

        Returns
        -------
        float
            Steering command (rad). Positive = steer right (away from left wall).
        """
        # Error calculation:
        # - distance_error negative (too close) -> need to steer right -> positive output
        # - theta positive (heading toward wall) -> need to steer right -> positive output
        # So: error = -distance_error + L * theta
        #      = -(current - target) + L * theta
        #      = target - current + L * theta
        
        error = float(distance_error) - self.L * math.sin(float(theta))

        # Time delta
        current_time = self._get_time(sender)
        dt = 0.0
        if self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt <= 1e-6 or dt > 0.5:
                dt = 0.0

        # Integral with anti-windup
        if dt > 0.0:
            self.integral += error * dt
            self.integral = max(-self.integral_limit, min(self.integral, self.integral_limit))

        # Derivative with low-pass filter
        d_raw = 0.0
        if dt > 0.0 and self.prev_error is not None:
            d_raw = (error - self.prev_error) / dt
        self.d_filtered = self.d_filter_alpha * self.d_filtered + (1.0 - self.d_filter_alpha) * d_raw

        # PID output
        steering = self.kp * error + self.ki * self.integral + self.kd * self.d_filtered

        # Saturate
        steering = max(-self.steering_limit, min(steering, self.steering_limit))

        self.prev_error = error
        self.prev_time = current_time

        return float(steering)