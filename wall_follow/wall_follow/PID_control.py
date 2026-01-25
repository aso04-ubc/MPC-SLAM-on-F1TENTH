# wall_follow/PID_control.py
# Fengwei

import math
from time import time
from typing import Optional


class PIDControl:
    """
    Simple and stable PID controller for wall-following.

    Input:
      - theta : heading error relative to wall direction (rad)
      - y     : lateral distance error (m)

    Error definition (from lecture slides):
        e(t) = -( y + L * sin(theta) )

    Output:
      - steering command (rad)

    Note:
      This class is a PURE controller:
      - no ROS publisher
      - no topic names
      - no speed planning
    """

    def __init__(
        self,
        kp: float = 0.3,
        ki: float = 0.02,
        kd: float = 0.3,
        lookahead_L: float = 0.3,
        steering_limit: float = 0.6,
        integral_limit: float = 1.0,
        d_filter_alpha: float = 0       # alpha -> closer to 1 = smoother D
    ):
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # lookahead distance L
        self.L = lookahead_L

        # output / state limits
        self.steering_limit = abs(steering_limit)
        self.integral_limit = abs(integral_limit)

        # derivative low-pass filter parameter
        self.d_filter_alpha = d_filter_alpha

        # internal states
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

    def run(self, sender, theta: float, y: float) -> float:
        """
        Run one PID control step.

        Parameters
        ----------
        sender : Node
            ROS2 node (used only for clock).
        theta : float
            Angle between car heading and wall direction (rad).
        y : float
            Lateral distance error (m).

        Returns
        -------
        float
            Steering command (rad).
        """

        # Compute error (lookahead wall-following) 
        error = -(float(y) + self.L * math.sin(float(theta)))

        # Compute dt with safety bounds
        current_time = self._get_time(sender)

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time

        # reject unreasonable dt (startup / pause / clock jump)
        if dt <= 1e-6 or dt > 0.5:
            dt = 0.0

        # Integral term (anti-windup) 
        if dt > 0.0 and self.ki != 0.0:
            self.integral += error * dt
            self.integral = max(
                -self.integral_limit,
                min(self.integral, self.integral_limit)
            )

        # Derivative term with low-pass filter 
        if dt > 0.0 and self.prev_error is not None:
            d_raw = (error - self.prev_error) / dt
        else:
            d_raw = 0.0

        # standard first-order low-pass:
        # alpha closer to 1 -> smoother 
        alpha = self.d_filter_alpha
        self.d_filtered = alpha * self.d_filtered + (1.0 - alpha) * d_raw

        # PID control law 
        steering = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.d_filtered
        )

        # Saturate steering output
        steering = max(
            -self.steering_limit,
            min(steering, self.steering_limit)
        )

        self.prev_error = error
        self.prev_time = current_time

        return float(steering)