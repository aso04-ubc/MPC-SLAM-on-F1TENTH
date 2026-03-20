"""Reference path helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence

from .nearest_point import NearestPointResult, find_nearest_point
from .speed_profile import build_constant_speed_profile, estimate_path_curvature
from .vehicle_model import VehicleState, wrap_angle


@dataclass(frozen=True)
class ReferencePoint:
    """One path point."""

    s: float
    x: float
    y: float
    yaw: float
    kappa: float
    v_ref: float


@dataclass(frozen=True)
class ReferenceSegment:
    """Local reference of length N + 1."""

    s: List[float]
    x: List[float]
    y: List[float]
    yaw: List[float]
    kappa: List[float]
    v_ref: List[float]

    def __len__(self) -> int:
        """Return the segment length."""
        return len(self.x)

    def is_valid(self) -> bool:
        """Check that all fields have matching lengths."""
        size = len(self.x)
        if size == 0:
            return False
        return all(
            len(field) == size
            for field in (self.s, self.y, self.yaw, self.kappa, self.v_ref)
        )


@dataclass(frozen=True)
class PathReference:
    """Stored global path."""

    s: List[float]
    x: List[float]
    y: List[float]
    yaw: List[float]
    kappa: List[float]
    v_ref: List[float]

    def __len__(self) -> int:
        """Return the path length."""
        return len(self.x)

    def is_valid(self) -> bool:
        """Check that the path data is usable."""
        size = len(self.x)
        return size >= 2 and all(
            len(field) == size
            for field in (self.s, self.y, self.yaw, self.kappa, self.v_ref)
        )


def unwrap_yaw_sequence(yaw_values: Iterable[float]) -> List[float]:
    """Unwrap yaw so it stays continuous."""
    unwrapped: List[float] = []
    previous = None
    for yaw in yaw_values:
        current = wrap_angle(float(yaw))
        if previous is None:
            unwrapped.append(current)
            previous = current
            continue

        delta = wrap_angle(current - previous)
        previous = previous + delta
        unwrapped.append(previous)
    return unwrapped


def build_constant_speed_reference(
    current_state: VehicleState,
    horizon_N: int,
    dt: float,
    target_speed: float,
) -> ReferenceSegment:
    """Build a simple straight reference from the current state."""
    sample_count = max(horizon_N, 0) + 1
    s_values: List[float] = []
    x_values: List[float] = []
    y_values: List[float] = []
    yaw_values: List[float] = []
    kappa_values: List[float] = []
    v_values: List[float] = []

    cos_psi = math.cos(current_state.psi)
    sin_psi = math.sin(current_state.psi)

    for index in range(sample_count):
        step_distance = target_speed * dt * index
        s_values.append(step_distance)
        x_values.append(current_state.x + step_distance * cos_psi)
        y_values.append(current_state.y + step_distance * sin_psi)
        yaw_values.append(current_state.psi)
        kappa_values.append(0.0)
        v_values.append(target_speed)

    return ReferenceSegment(
        s=s_values,
        x=x_values,
        y=y_values,
        yaw=unwrap_yaw_sequence(yaw_values),
        kappa=kappa_values,
        v_ref=v_values,
    )


def build_path_reference(
    x_values: Sequence[float],
    y_values: Sequence[float],
    target_speed: float,
) -> PathReference:
    """Build a path object from x and y lists."""
    if len(x_values) != len(y_values):
        raise ValueError('x_values and y_values must have the same length')
    if len(x_values) < 2:
        raise ValueError('path requires at least two points')

    s_values: List[float] = [0.0]
    yaw_values: List[float] = []
    for index in range(len(x_values) - 1):
        dx = float(x_values[index + 1]) - float(x_values[index])
        dy = float(y_values[index + 1]) - float(y_values[index])
        s_values.append(s_values[-1] + math.hypot(dx, dy))
        yaw_values.append(math.atan2(dy, dx))
    yaw_values.append(yaw_values[-1])

    kappa_values = estimate_path_curvature(x_values, y_values)
    speed_values = build_constant_speed_profile(len(x_values), target_speed)

    return PathReference(
        s=s_values,
        x=[float(value) for value in x_values],
        y=[float(value) for value in y_values],
        yaw=unwrap_yaw_sequence(yaw_values),
        kappa=kappa_values,
        v_ref=speed_values,
    )


def _interp_scalar(
    s_query: float,
    s_values: Sequence[float],
    field_values: Sequence[float],
    start_index: int,
) -> float:
    """Interpolate one value along s."""
    index = max(0, min(start_index, len(s_values) - 2))
    while index < len(s_values) - 2 and s_values[index + 1] < s_query:
        index += 1

    s0 = s_values[index]
    s1 = s_values[index + 1]
    if abs(s1 - s0) <= 1e-9:
        return float(field_values[index])

    ratio = (s_query - s0) / (s1 - s0)
    ratio = max(0.0, min(1.0, ratio))
    return float(field_values[index]) + ratio * (float(field_values[index + 1]) - float(field_values[index]))


def extract_local_reference(
    path: PathReference,
    current_state: VehicleState,
    horizon_N: int,
    dt: float,
    previous_index: Optional[int] = None,
    search_window: int = 15,
) -> tuple[ReferenceSegment, NearestPointResult]:
    """Extract a local reference of length N + 1."""
    if not path.is_valid():
        raise ValueError('path reference is invalid')

    nearest = find_nearest_point(
        path_x=path.x,
        path_y=path.y,
        path_s=path.s,
        query_x=current_state.x,
        query_y=current_state.y,
        previous_index=previous_index,
        search_window=search_window,
    )

    sample_count = max(horizon_N, 0) + 1
    s_values: List[float] = []
    x_values: List[float] = []
    y_values: List[float] = []
    yaw_values: List[float] = []
    kappa_values: List[float] = []
    v_values: List[float] = []

    path_end_s = path.s[-1]
    current_s = nearest.s
    interp_start_index = nearest.index

    for step in range(sample_count):
        if step == 0:
            sample_s = current_s
        else:
            previous_speed = max(0.0, v_values[-1])
            sample_s = min(path_end_s, s_values[-1] + previous_speed * dt)
        s_values.append(sample_s)

        x_values.append(_interp_scalar(sample_s, path.s, path.x, interp_start_index))
        y_values.append(_interp_scalar(sample_s, path.s, path.y, interp_start_index))
        yaw_values.append(_interp_scalar(sample_s, path.s, path.yaw, interp_start_index))
        kappa_values.append(_interp_scalar(sample_s, path.s, path.kappa, interp_start_index))
        v_values.append(_interp_scalar(sample_s, path.s, path.v_ref, interp_start_index))

        while interp_start_index < len(path.s) - 2 and path.s[interp_start_index + 1] < sample_s:
            interp_start_index += 1

    return (
        ReferenceSegment(
            s=s_values,
            x=x_values,
            y=y_values,
            yaw=unwrap_yaw_sequence(yaw_values),
            kappa=kappa_values,
            v_ref=v_values,
        ),
        nearest,
    )
