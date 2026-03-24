import numpy as np
import matplotlib.pyplot as plt


class GapFollowAlgo:
    """
    Gap following algorithm that processes LiDAR data to find safe driving gaps.
    Designed for smooth driving and handling U-turns effectively.
    """

    def __init__(self, max_range=3.5, min_safe_distance=0.25, car_width=0.5, disparity_threshold=1.2, smoothing_window_size=10):
        """
        Initialize the gap following algorithm with default parameters.

        - max_range: distances larger than this are saturated; used to detect open space / U‑turns.
        - min_safe_distance: conceptual threshold for how close we want to get to obstacles.
        - car_width: physical width of the vehicle (used to inflate obstacles laterally).
        - disparity_threshold: minimum jump in consecutive ranges to treat as an obstacle edge.
        - smoothing_window_size: size of moving average kernel for LiDAR smoothing.
        """
        self.max_range = max_range           # meters; upper bound for valid LiDAR measurements
        self.min_safe_distance = min_safe_distance   # meters; not used directly here but carried for callers
        self.car_width = car_width           # meters; used to compute obstacle inflation angle
        self.disparity_threshold = disparity_threshold  # range jump to classify a disparity edge


        self.prev_angle = 0.0
        self.prev_idx = None

        # Smoothing parameters
        # Larger window => smoother ranges but more lag / loss of detail.
        self.smoothing_window_size = smoothing_window_size  # window size for moving average filter
        
        # plt.ion()
        # self.fig, self.ax = plt.subplots()
        # self.line_processed, = self.ax.plot([], [], label="processed")
        # self.line_extended, = self.ax.plot([], [], label="extended")
        # self.line_costs, = self.ax.plot([], [], label="costs")
        # self.best_idx_line = self.ax.axvline(0, linestyle="--", label="best_idx")
        # self.ax.set_title("Gap Follow Debug")
        # self.ax.set_xlabel("Beam Index")
        # self.ax.set_ylabel("Value")
        # self.ax.set_ylim(0.0, self.max_range + 0.5)
        # self.ax.legend()
        # self._plot_counter = 0
        
        
    # def _update_debug_plot(self, processed, extended, costs, best_idx):
    #     self._plot_counter += 1

    #     # plot at 10 Hz if callback is 40 Hz
    #     if self._plot_counter % 4 != 0:
    #         return

    #     x = np.arange(len(processed))

    #     self.line_processed.set_data(x, processed)
    #     self.line_extended.set_data(x, extended)
    #     self.line_costs.set_data(x, costs)
    #     self.best_idx_line.set_xdata([best_idx, best_idx])

    #     self.ax.set_xlim(0, len(processed) - 1)

    #     ymax = max(
    #         self.max_range + 0.5,
    #         float(np.max(processed)) if len(processed) else 0.0,
    #         float(np.max(extended)) if len(extended) else 0.0,
    #         float(np.max(costs)) if len(costs) else 0.0,
    #     )
    #     self.ax.set_ylim(0.0, ymax + 0.2)

    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()
    #     plt.pause(0.001)


    def process_lidar_and_find_gap(self, ranges, angle_min, angle_increment):

        # --- 1) preprocess ---
        processed = self._preprocess_ranges(ranges, angle_min, angle_increment)
        extended = self._apply_disparity_extender(processed, angle_increment)

        n = len(extended)
        angles = np.deg2rad(-90) + np.arange(n) * angle_increment
        center_idx = n // 2

        # ---------- 1) emergency near-wall recovery ----------
        front_half_width = max(1, int(np.deg2rad(15) / angle_increment))
        front_start = max(0, center_idx - front_half_width)
        front_end = min(n, center_idx + front_half_width + 1)
        front_slice = extended[front_start:front_end]
        front_min = float(np.min(front_slice)) if len(front_slice) else self.max_range

        # side sectors used for wall escape
        side_half_width = max(1, int(np.deg2rad(25) / angle_increment))

        right_start = min(n, center_idx + side_half_width)
        right_end = min(n, center_idx + 3 * side_half_width)
        left_start = max(0, center_idx - 3 * side_half_width)
        left_end = max(0, center_idx - side_half_width)

        left_clear = float(np.mean(extended[left_start:left_end])) if left_end > left_start else 0.0
        right_clear = float(np.mean(extended[right_start:right_end])) if right_end > right_start else 0.0

        # If front is dangerously close, override normal planner
        if front_min < 0.35:
            if left_clear > right_clear:
                target_angle = 0.7
                best_idx = min(n - 1, center_idx + side_half_width)
            else:
                target_angle = -0.7
                best_idx = max(0, center_idx - side_half_width)

            alpha = 0.3
            target_angle = alpha * self.prev_angle + (1.0 - alpha) * target_angle
            self.prev_angle = target_angle
            self.prev_idx = best_idx
            return float(target_angle), int(best_idx)

        # ---------- 2) normal cost-based planner ----------
        window_size = int(np.deg2rad(10) / angle_increment)
        window_size = max(window_size, 3)

        costs = np.convolve(extended, np.ones(window_size), mode='same') / window_size

        # forward bias
        costs *= (0.5 + 0.5 * np.cos(angles))

        # bias away from nearby side wall
        # positive bias -> more room on left, negative -> more room on right
        wall_bias = np.clip((right_clear - left_clear) / max(self.max_range, 1e-6), -0.4, 0.4)
        costs *= (1.0 + wall_bias * np.sin(angles))

        # smooth costs
        #costs = np.convolve(costs, np.ones(5) / 5, mode='same')

        max_cost = np.max(costs)
        mask = costs > 0.90 * max_cost
        indices = np.where(mask)[0]

        if len(indices) == 0:
            target_angle = 0.0
            best_idx = center_idx
        else:
            splits = np.where(np.diff(indices) != 1)[0] + 1
            regions = np.split(indices, splits)
            best_region = max(regions, key=len)
            best_idx = int(np.median(best_region))

            if self.prev_idx is not None and abs(best_idx - self.prev_idx) < 5:
                best_idx = self.prev_idx

            target_angle = np.deg2rad(-90) + best_idx * angle_increment

        # ---------- 3) adaptive smoothing ----------
        # react faster when front clearance is getting smaller
        if front_min < 0.6:
            alpha = 0.4
        else:
            alpha = 0.7

        target_angle = alpha * self.prev_angle + (1.0 - alpha) * target_angle

        self.prev_angle = target_angle
        self.prev_idx = best_idx
        
        #self._update_debug_plot(processed, extended, costs, best_idx)

        return float(target_angle), int(best_idx)

    def _preprocess_ranges(self, ranges, angle_min, angle_increment):
        """
        Preprocess incoming LiDAR ranges before gap search.

        - Replaces NaNs/Infs with finite values.
        - Applies a moving-average style smoothing that does not increase ranges.
        - Clips values to [0, max_range].
        - Zeros out points outside the desired field of view and very small ranges.
        """

        # Work on a copy so we never modify the caller's array in‑place.
        processed = ranges.copy()

        # Replace NaNs and Infs with bounded numeric values.
        processed = np.nan_to_num(
            processed,
            nan=0.0,
            posinf=self.max_range,
            neginf=0.0
        )

        # Moving‑average window used to smooth the scan.
        window = np.ones(self.smoothing_window_size) / self.smoothing_window_size
        # Use the minimum between original and smoothed so we never artificially
        # increase ranges (which could "invent" free space near obstacles).
        processed = np.minimum(processed, np.convolve(processed, window, mode='same'))

        # Enforce physical limits on the distance values.
        processed = np.clip(processed, 0.0, self.max_range)

        # Compute angle for each beam so we can restrict the field of view.
        n = len(processed)
        angles = angle_min + np.arange(n) * angle_increment

        # Ignore extremely small ranges (likely noise or self‑hits).
        processed[processed < 0.1] = 0.0

        # Restrict processing to a symmetric front 180° field of view.
        fov = np.deg2rad(180)
        mask = (angles >= -fov/2) & (angles <= fov/2)
        
        processed = processed[mask]

        return processed

    def _apply_disparity_extender(self, ranges, angle_increment):
        """
        Inflate obstacles around disparity edges to respect vehicle width.

        For each large jump in consecutive ranges (a "disparity"), we assume an
        obstacle edge. We then extend the closer distance sideways over a number
        of neighbouring beams that correspond geometrically to half the car width.
        This car‑shaped "bubble" prevents the planner from threading gaps that
        are too narrow for the vehicle.
        """
        bubble_ranges = ranges.copy()
        n = len(bubble_ranges)

        # First derivative of ranges to find sudden jumps.
        edges = np.diff(bubble_ranges)
        disparities = np.where(np.abs(edges) >= self.disparity_threshold)[0]

        for i in disparities:
            left = bubble_ranges[i]
            right = bubble_ranges[i + 1]

            # Choose closer obstacle side as the limiting distance.
            if left < right:
                closer_idx = i
                closer_dist = left
                direction = 1
            else:
                closer_idx = i + 1
                closer_dist = right
                direction = -1

            # cap inflation so nearby walls do not erase the whole scan
            theta = np.arctan((self.car_width / 2) / max(closer_dist, 0.20))
            theta = min(theta, np.deg2rad(20))
            
            extend = int(np.ceil(theta / angle_increment))

            if direction == 1:
                # Extend bubble to the right side of the edge.
                end = min(n, closer_idx + extend)
                bubble_ranges[closer_idx:end] = closer_dist
            else:
                # Extend bubble to the left side of the edge.
                start = max(0, closer_idx - extend)
                bubble_ranges[start:closer_idx] = closer_dist

        return bubble_ranges
