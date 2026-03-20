"""Speed profile helpers."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional


def build_constant_speed_profile(point_count: int, target_speed: float) -> List[float]:
    """Make a constant speed profile."""
    if point_count < 0:
        raise ValueError('point_count must be non-negative')
    return [float(target_speed)] * point_count


def apply_curvature_speed_limit(
    target_speeds: Iterable[float],
    curvature: Iterable[float],
    min_speed: float = 0.0,
    max_speed: Optional[float] = None,
    curvature_gain: float = 1.0,
) -> List[float]:
    """Slow the profile down a bit on sharper curves."""
    speeds: List[float] = []
    for base_speed, kappa in zip(target_speeds, curvature):
        denom = 1.0 + curvature_gain * abs(float(kappa))
        limited = float(base_speed) / denom
        limited = max(min_speed, limited)
        if max_speed is not None:
            limited = min(max_speed, limited)
        speeds.append(limited)
    return speeds


def estimate_path_curvature(x_values: Iterable[float], y_values: Iterable[float]) -> List[float]:
    """Estimate curvature from nearby path points."""
    x_list = [float(value) for value in x_values]
    y_list = [float(value) for value in y_values]
    point_count = len(x_list)
    if point_count == 0:
        return []
    if point_count < 3:
        return [0.0] * point_count

    curvature = [0.0] * point_count
    for index in range(1, point_count - 1):
        ax = x_list[index] - x_list[index - 1]
        ay = y_list[index] - y_list[index - 1]
        bx = x_list[index + 1] - x_list[index]
        by = y_list[index + 1] - y_list[index]
        cross = ax * by - ay * bx
        norm_a = math.hypot(ax, ay)
        norm_b = math.hypot(bx, by)
        chord = math.hypot(x_list[index + 1] - x_list[index - 1], y_list[index + 1] - y_list[index - 1])
        if norm_a <= 1e-9 or norm_b <= 1e-9 or chord <= 1e-9:
            curvature[index] = 0.0
            continue
        curvature[index] = 2.0 * cross / (norm_a * norm_b * chord)

    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature
