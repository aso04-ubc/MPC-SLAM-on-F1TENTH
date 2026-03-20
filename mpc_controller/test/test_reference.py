import math

from mpc_controller.mpc_core.nearest_point import find_nearest_point
from mpc_controller.mpc_core.reference import (
    build_path_reference,
    extract_local_reference,
    unwrap_yaw_sequence,
)
from mpc_controller.mpc_core.vehicle_model import VehicleState


def test_nearest_point_continuity_with_windowed_search():
    path = build_path_reference(
        x_values=[float(index) for index in range(20)],
        y_values=[0.0] * 20,
        target_speed=2.0,
    )

    previous_index = None
    projected_indices = []
    for step in range(8):
        query_x = 2.2 + 0.7 * step
        result = find_nearest_point(
            path_x=path.x,
            path_y=path.y,
            path_s=path.s,
            query_x=query_x,
            query_y=0.2,
            previous_index=previous_index,
            search_window=3,
        )
        projected_indices.append(result.index)
        previous_index = result.index

    assert projected_indices == sorted(projected_indices)
    assert max(b - a for a, b in zip(projected_indices[:-1], projected_indices[1:])) <= 2


def test_yaw_unwrap_preserves_continuity():
    wrapped = [3.05, 3.12, -3.10, -3.02, -2.95]
    unwrapped = unwrap_yaw_sequence(wrapped)

    assert len(unwrapped) == len(wrapped)
    assert max(abs(b - a) for a, b in zip(unwrapped[:-1], unwrapped[1:])) < 0.3


def test_local_reference_extraction_returns_n_plus_one_samples():
    path = build_path_reference(
        x_values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        y_values=[0.0, 0.2, 0.8, 1.8, 3.2, 5.0],
        target_speed=1.5,
    )
    state = VehicleState(x=1.1, y=0.4, psi=0.0, v=1.0, delta=0.0)

    segment, nearest = extract_local_reference(
        path=path,
        current_state=state,
        horizon_N=5,
        dt=0.1,
        previous_index=1,
        search_window=3,
    )

    assert nearest.index in (1, 2)
    assert len(segment) == 6
    assert segment.is_valid()
    assert all(b >= a for a, b in zip(segment.s[:-1], segment.s[1:]))
    assert segment.s[0] >= path.s[nearest.index]
    assert max(abs(b - a) for a, b in zip(segment.yaw[:-1], segment.yaw[1:])) < math.pi / 2.0


def test_nearest_point_projection_reduces_discrete_waypoint_jump():
    path = build_path_reference(
        x_values=[0.0, 1.0, 2.0, 3.0],
        y_values=[0.0, 0.0, 0.0, 0.0],
        target_speed=1.0,
    )

    result = find_nearest_point(
        path_x=path.x,
        path_y=path.y,
        path_s=path.s,
        query_x=1.49,
        query_y=0.3,
        previous_index=1,
        search_window=1,
    )

    assert result.index == 1
    assert result.next_index == 2
    assert abs(result.x - 1.49) < 1e-6
    assert abs(result.y) < 1e-6
    assert abs(result.s - 1.49) < 1e-6
