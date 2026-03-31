"""
Low Pass Filter Module.

This module provides a simple first-order low-pass filter implementation
using exponential moving average (EMA) for smoothing noisy signals.
"""

class LowPassFilter:
    """
    First-order low-pass filter using exponential moving average.

    This filter smooths input signals by computing a weighted average between
    the current input and the previous filtered output. The alpha parameter
    controls the smoothing strength:

    - alpha = 0.0: No filtering (output = input)
    - alpha = 1.0: Maximum smoothing (output = previous_output)
    - alpha = 0.1-0.3: Typical range for responsive smoothing

    The filter has one pole at (1-alpha) in the z-domain.
    """

    def __init__(self, alpha : float):
        """
        Initialize the low-pass filter.

        Args:
            alpha: Smoothing factor between 0.0 and 1.0
                  Higher values = more smoothing, lower values = more responsive
        """
        self.alpha: float  = alpha
        self.prev :float = 0.0


    def first_order_filter(self, data: float):
        """
        Apply first-order low-pass filter to input data.

        Computes the exponential moving average using the formula:
        filtered = alpha * data + (1 - alpha) * previous_filtered

        This is equivalent to: filtered = previous_filtered + alpha * (data - previous_filtered)

        Parameters:
            data (float): Input data value to be filtered

        Returns:
            float: Filtered output value using exponential moving average
        """
        prev = self.prev
        self.prev = (prev * (1 - self.alpha) + data * self.alpha)
        return self.prev
