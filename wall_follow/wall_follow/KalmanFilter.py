# From website
# Kalman Filter implementation for 1D data smoothing

class SimpleKalmanFilter:
    def __init__(self, R, Q, initial_value=0.0):
        """
        Initialize the Kalman Filter.

        :param R: Measurement Noise Covariance -> Larger values mean less trust in the sensor 
                  (result is smoother, but with more lag).
        :param Q: Process Noise Covariance -> Larger values mean more trust in the sensor/measurement 
                  (result reacts faster, but incorporates more noise).
        :param initial_value: The initial state estimate.
        """
        # A: State Transition Matrix (1x1 for simple 1D)
        self.A = 1.0 
        # B: Control Matrix (None here)
        self.B = 0.0
        # H: Measurement Matrix (1x1)
        self.H = 1.0
        
        # X: State Estimate Vector (State)
        self.current_state_estimate = initial_value
        # P: Estimate Covariance (Prediction Covariance)
        self.current_prob_estimate = 1.0
        
        # Q: Process Noise
        self.Q = Q
        # R: Measurement Noise
        self.R = R

    def update(self, measurement):
        # --- 1. Predict ---
        # Predict the state at the next moment (x = A * x)
        prior_state_estimate = self.A * self.current_state_estimate
        
        # Predict the covariance (P = A * P * A^T + Q)
        prior_prob_estimate = self.A * self.current_prob_estimate * self.A + self.Q

        # --- 2. Update / Correct ---
        # Calculate Kalman Gain (K = P * H^T / (H * P * H^T + R))
        kalman_gain = prior_prob_estimate * self.H / (self.H * prior_prob_estimate * self.H + self.R)
        
        # Update state estimate (x = x_pred + K * (measurement - H * x_pred))
        self.current_state_estimate = prior_state_estimate + kalman_gain * (measurement - (self.H * prior_state_estimate))
        
        # Update covariance (P = (I - K * H) * P)
        self.current_prob_estimate = (1.0 - kalman_gain * self.H) * prior_prob_estimate
        
        return self.current_state_estimate