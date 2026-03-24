class LowPassFilter:
    def __init__(self, alpha : float):
        """init simple low pass filter"""
        self.alpha: float  = alpha
        self.prev :float = 0.0


    def first_order_filter(self, data: float):
        """
        First order low pass filter.

        Parameters:
            data (float): data in which we are applying the low pass filter on.

        Returns:
            float: a low pass filtered version of the input data based on an EMA (exponential moving average filter).
        """
        prev = self.prev
        self.prev = (prev * (1 - self.alpha) + data * self.alpha)
        return self.prev
