"""Nearest-point helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class NearestPointResult:
    """Nearest point projected onto the path."""

    index: int
    next_index: int
    distance: float
    x: float
    y: float
    s: float
    t: float


def _squared_distance(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _project_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> Tuple[float, float, float, float]:
    """Project a point onto one segment."""
    vx = bx - ax
    vy = by - ay
    seg_norm_sq = vx * vx + vy * vy
    if seg_norm_sq <= 1e-12:
        proj_x = ax
        proj_y = ay
        t = 0.0
    else:
        t = ((px - ax) * vx + (py - ay) * vy) / seg_norm_sq
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * vx
        proj_y = ay + t * vy

    distance = math.hypot(px - proj_x, py - proj_y)
    return proj_x, proj_y, t, distance


def find_nearest_point(
    path_x: Sequence[float],
    path_y: Sequence[float],
    path_s: Sequence[float],
    query_x: float,
    query_y: float,
    previous_index: Optional[int] = None,
    search_window: int = 15,
) -> NearestPointResult:
    """Find a stable nearest point using windowed search and projection."""
    if not (len(path_x) == len(path_y) == len(path_s)):
        raise ValueError('path_x, path_y, and path_s must have the same length')
    if len(path_x) < 2:
        raise ValueError('reference path must contain at least two points')

    point_count = len(path_x)
    if previous_index is None:
        candidate_indices = range(point_count)
    else:
        clamped_prev = max(0, min(previous_index, point_count - 1))
        start = max(0, clamped_prev - max(search_window, 1))
        stop = min(point_count, clamped_prev + max(search_window, 1) + 1)
        candidate_indices = range(start, stop)

    best_index = min(
        candidate_indices,
        key=lambda idx: _squared_distance(query_x, query_y, path_x[idx], path_y[idx]),
    )

    if previous_index is not None:
        global_best_index = min(
            range(point_count),
            key=lambda idx: _squared_distance(query_x, query_y, path_x[idx], path_y[idx]),
        )
        window_distance = _squared_distance(query_x, query_y, path_x[best_index], path_y[best_index])
        global_distance = _squared_distance(
            query_x, query_y, path_x[global_best_index], path_y[global_best_index]
        )
        if global_distance + 1e-9 < window_distance:
            best_index = global_best_index

    segment_candidates = []
    if best_index > 0:
        segment_candidates.append(best_index - 1)
    if best_index < point_count - 1:
        segment_candidates.append(best_index)
    if not segment_candidates:
        segment_candidates.append(0)

    best_result: Optional[NearestPointResult] = None
    for seg_index in segment_candidates:
        proj_x, proj_y, t, distance = _project_to_segment(
            query_x,
            query_y,
            path_x[seg_index],
            path_y[seg_index],
            path_x[seg_index + 1],
            path_y[seg_index + 1],
        )
        s_value = path_s[seg_index] + t * (path_s[seg_index + 1] - path_s[seg_index])
        candidate = NearestPointResult(
            index=seg_index,
            next_index=seg_index + 1,
            distance=distance,
            x=proj_x,
            y=proj_y,
            s=s_value,
            t=t,
        )
        if best_result is None or candidate.distance < best_result.distance:
            best_result = candidate

    if best_result is None:
        raise RuntimeError('nearest-point search failed to produce a result')

    return best_result
