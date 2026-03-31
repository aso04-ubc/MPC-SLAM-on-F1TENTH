
"""
PID Control Module for Steering Control.

This module implements a standard PID (Proportional-Integral-Derivative) controller
specifically designed for steering control in autonomous vehicles. The controller
includes features like integral windup protection, derivative filtering, and
configurable saturation limits.
"""

from time import time
from typing import Optional
from gap_following.LowPassFilter import LowPassFilter


class PIDControl:
    """
    Standard PID controller for steering control.

    This controller computes steering commands based on the error between desired
    and current steering angles. It includes:
    - Proportional term for immediate response
    - Integral term for steady-state error elimination
    - Derivative term with low-pass filtering for stability
    - Anti-windup protection on integral term
    - Configurable output saturation

    Input: Steering angle error (desired - current)
    Output: Steering command in radians
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
        """
        Initialize PID controller with given parameters.

        Args:
            kp: Proportional gain - affects immediate response to error
            ki: Integral gain - affects steady-state error correction
            kd: Derivative gain - affects response to rate of change
            steering_limit: Maximum absolute steering output in radians
            integral_limit: Maximum absolute integral term accumulation
            d_filter_alpha: Low-pass filter alpha for derivative smoothing (0-1)
        """
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
        """
        Reset the PID controller state (e.g. when starting a new run).

        Clears all internal state variables including previous error, time,
        integral accumulation, and filtered derivative. This should be called
        when the controller needs to start fresh, such as when switching control modes.
        """
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
        Run one PID step and compute steering command.

        Executes one iteration of the PID control algorithm. This method should
        be called at regular intervals with the current steering error.

        The algorithm computes:
        - Proportional term: kp * error
        - Integral term: ki * integral (with anti-windup protection)
        - Derivative term: kd * filtered_derivative
        - Output: sum of terms, clamped to steering limits

        Parameters
        ----------
        sender : object
            ROS node or object with get_clock() method for timing
        error : float
            Control error (desired_steering - current_steering) in radians

        Returns
        -------
        float
            Steering command in radians, clamped to [-steering_limit, +steering_limit]
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