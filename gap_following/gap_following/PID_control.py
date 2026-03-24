
from time import time
from typing import Optional
from gap_following.LowPassFilter import LowPassFilter


class PIDControl:
    """
    Standard PID controller.

    Input:
      - error (e.g. left_angle)

    Output:
      - steering command (rad)
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.1,
        steering_limit: float = 0.7,
        integral_limit: float = 1.0,
        d_filter_alpha: float = 0.2   # closer to 1 = smoother derivative
    ):
        """Initialize PID controller with given parameters."""
        self.kp : float = kp
        self.ki : float = ki
        self.kd : float = kd

        self.filter = LowPassFilter(d_filter_alpha)
        
        self.steering_limit : float = abs(steering_limit)
        self.integral_limit : float = abs(integral_limit)
        

        self.prev_error: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.integral : float = 0.0

    def reset(self):
        """Reset the PID controller state (e.g. when starting a new run)."""
        self.prev_error = None
        self.prev_time = None
        self.integral = 0.0
        self.d_filtered = 0.0

    def _get_time(self, sender):
        try:
            return sender.get_clock().now().nanoseconds * 1e-9
        except Exception:
            return time()

    def run(self, sender, error: float) -> float:
        """
        Run one PID step.

        Parameters
        ----------
        error : float
            Control error (e.g. left wall angle)

        Returns
        -------
        float
            Steering command (rad)
        """

        current_time = self._get_time(sender)

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time

        # Protect against bad dt
        if dt <= 1e-6 or dt > 0.5:
            dt = 0.0

        # Integral term
        if dt > 0.0 and self.ki != 0.0:
            self.integral += error * dt
            self.integral = max(
                -self.integral_limit,
                min(self.integral, self.integral_limit)
            )

        # Derivative term
        if dt > 0.0 and self.prev_error is not None:
            d_raw = (error - self.prev_error) / dt
        else:
            d_raw = 0.0

        # Low-pass filter on D
        
        self.d_filtered = self.filter.first_order_filter(d_raw)
        
        # PID output
        steering = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.d_filtered
        )

        # Saturation
        steering = max(
            -self.steering_limit,
            min(steering, self.steering_limit)
        )

        self.prev_error = error
        self.prev_time = current_time

        return float(steering)