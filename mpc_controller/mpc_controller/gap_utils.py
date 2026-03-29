import numpy as np


class GapFollowAlgo:
    """
    Gap following algorithm that processes LiDAR data to find safe driving gaps.

    Updated version:
    - keeps actual cropped beam angles instead of reconstructing from hardcoded -90 deg
    - returns target_distance in addition to target_angle and best_idx
    - stores debug arrays for the OpenCV canvas
    - increases hysteresis / smoothing to reduce post-turn oscillation
    """

    def __init__(
        self,
        max_range=3.5,
        min_safe_distance=0.25,
        car_width=0.5,
        disparity_threshold=1.2,
        smoothing_window_size=10,
    ):
        self.max_range = max_range
        self.min_safe_distance = min_safe_distance
        self.car_width = car_width
        self.disparity_threshold = disparity_threshold
        self.smoothing_window_size = smoothing_window_size

        self.prev_angle = 0.0
        self.prev_idx = None

        # Debug state used by main_node.py
        self.last_processed = None
        self.last_extended = None
        self.last_costs = None
        self.last_angles = None
        self.last_best_idx = None
        self.last_target_angle = 0.0
        self.last_target_distance = 0.0
        self.last_front_min = 0.0
        self.last_left_clear = 0.0
        self.last_right_clear = 0.0

    def process_lidar_and_find_gap(self, ranges, angle_min, angle_increment):
        """
        Returns:
            target_angle, best_idx, target_distance
        """
        processed, cropped_angles = self._preprocess_ranges(ranges, angle_min, angle_increment)
        extended = self._apply_disparity_extender(processed, angle_increment)

        n = len(extended)
        if n == 0:
            self._store_debug(
                processed=processed,
                extended=extended,
                costs=None,
                angles=cropped_angles,
                best_idx=0,
                target_angle=0.0,
                target_distance=0.0,
                front_min=0.0,
                left_clear=0.0,
                right_clear=0.0,
            )
            return 0.0, 0, 0.0

        center_idx = n // 2

        # Emergency near-wall recovery
        front_half_width = max(1, int(np.deg2rad(15.0) / angle_increment))
        front_start = max(0, center_idx - front_half_width)
        front_end = min(n, center_idx + front_half_width + 1)
        front_slice = extended[front_start:front_end]
        front_min = float(np.min(front_slice)) if len(front_slice) else self.max_range

        side_half_width = max(1, int(np.deg2rad(25.0) / angle_increment))
        right_start = min(n, center_idx + side_half_width)
        right_end = min(n, center_idx + 3 * side_half_width)
        left_start = max(0, center_idx - 3 * side_half_width)
        left_end = max(0, center_idx - side_half_width)

        left_clear = float(np.mean(extended[left_start:left_end])) if left_end > left_start else 0.0
        right_clear = float(np.mean(extended[right_start:right_end])) if right_end > right_start else 0.0

        if front_min < 0.35:
            if left_clear > right_clear:
                target_angle = 0.7
                best_idx = min(n - 1, center_idx + side_half_width)
            else:
                target_angle = -0.7
                best_idx = max(0, center_idx - side_half_width)

            # Keep emergency angle magnitude, but add more memory than before
            alpha = 0.5
            target_angle = alpha * self.prev_angle + (1.0 - alpha) * target_angle
            self.prev_angle = target_angle
            self.prev_idx = best_idx

            target_distance = float(front_min)

            self._store_debug(
                processed=processed,
                extended=extended,
                costs=None,
                angles=cropped_angles,
                best_idx=best_idx,
                target_angle=target_angle,
                target_distance=target_distance,
                front_min=front_min,
                left_clear=left_clear,
                right_clear=right_clear,
            )
            return float(target_angle), int(best_idx), target_distance

        # Normal cost-based planner
        window_size = int(np.deg2rad(10.0) / angle_increment)
        window_size = max(window_size, 3)

        costs = np.convolve(extended, np.ones(window_size), mode='same') / window_size

        # Forward bias
        costs *= (0.5 + 0.5 * np.cos(cropped_angles))

        # Bias away from nearby side wall
        wall_bias = np.clip((right_clear - left_clear) / max(self.max_range, 1e-6), -0.4, 0.4)
        costs *= (1.0 + wall_bias * np.sin(cropped_angles))

        max_cost = float(np.max(costs)) if len(costs) else 0.0
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

            # Stronger beam hysteresis than before
            if self.prev_idx is not None and abs(best_idx - self.prev_idx) < 8:
                best_idx = int(self.prev_idx)

            target_angle = float(cropped_angles[best_idx])

        # More smoothing than before to reduce post-turn oscillation
        if front_min < 0.6:
            alpha = 0.6
        else:
            alpha = 0.85

        target_angle = alpha * self.prev_angle + (1.0 - alpha) * target_angle

        self.prev_angle = target_angle
        self.prev_idx = best_idx

        target_distance = float(extended[best_idx]) if 0 <= best_idx < len(extended) else 0.0

        self._store_debug(
            processed=processed,
            extended=extended,
            costs=costs,
            angles=cropped_angles,
            best_idx=best_idx,
            target_angle=target_angle,
            target_distance=target_distance,
            front_min=front_min,
            left_clear=left_clear,
            right_clear=right_clear,
        )

        return float(target_angle), int(best_idx), target_distance

    def _preprocess_ranges(self, ranges, angle_min, angle_increment):
        """
        Returns:
            processed_ranges, cropped_angles
        """
        processed = np.array(ranges, dtype=float).copy()

        processed = np.nan_to_num(
            processed,
            nan=0.0,
            posinf=self.max_range,
            neginf=0.0,
        )

        window = np.ones(self.smoothing_window_size, dtype=float) / float(self.smoothing_window_size)
        processed = np.minimum(processed, np.convolve(processed, window, mode='same'))

        processed = np.clip(processed, 0.0, self.max_range)

        n = len(processed)
        angles = angle_min + np.arange(n, dtype=float) * angle_increment

        processed[processed < 0.1] = 0.0

        fov = np.deg2rad(180.0)
        mask = (angles >= -fov / 2.0) & (angles <= fov / 2.0)

        return processed[mask], angles[mask]

    def _apply_disparity_extender(self, ranges, angle_increment):
        bubble_ranges = ranges.copy()
        n = len(bubble_ranges)

        if n < 2:
            return bubble_ranges

        edges = np.diff(bubble_ranges)
        disparities = np.where(np.abs(edges) >= self.disparity_threshold)[0]

        for i in disparities:
            left = bubble_ranges[i]
            right = bubble_ranges[i + 1]

            if left < right:
                closer_idx = i
                closer_dist = left
                direction = 1
            else:
                closer_idx = i + 1
                closer_dist = right
                direction = -1

            theta = np.arctan((self.car_width / 2.0) / max(closer_dist, 0.20))
            theta = min(theta, np.deg2rad(20.0))
            extend = int(np.ceil(theta / angle_increment))

            if direction == 1:
                end = min(n, closer_idx + extend)
                bubble_ranges[closer_idx:end] = closer_dist
            else:
                start = max(0, closer_idx - extend)
                bubble_ranges[start:closer_idx] = closer_dist

        return bubble_ranges

    def _store_debug(
        self,
        processed,
        extended,
        costs,
        angles,
        best_idx,
        target_angle,
        target_distance,
        front_min,
        left_clear,
        right_clear,
    ):
        self.last_processed = processed.copy()
        self.last_extended = extended.copy()
        self.last_costs = None if costs is None else costs.copy()
        self.last_angles = angles.copy()
        self.last_best_idx = int(best_idx)
        self.last_target_angle = float(target_angle)
        self.last_target_distance = float(target_distance)
        self.last_front_min = float(front_min)
        self.last_left_clear = float(left_clear)
        self.last_right_clear = float(right_clear)