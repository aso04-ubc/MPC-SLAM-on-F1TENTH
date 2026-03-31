from mpc_controller.mpc_core.reference import build_path_reference, extract_local_reference
from mpc_controller.mpc_core.tracking_problem import (
    build_tracking_problem,
    curvature_to_steering_angle,
    estimate_steering_from_yaw_rate,
    integrate_control_step,
)
from mpc_controller.mpc_core.vehicle_model import VehicleParams, VehicleState


def test_curvature_to_steering_angle_stays_bounded():
    params = VehicleParams(wheelbase_L=0.33, delta_max=0.4)

    steer = curvature_to_steering_angle(5.0, params)

    assert abs(steer - 0.4) < 1e-9


def test_tracking_problem_builds_horizon_shapes():
    params = VehicleParams()
    state = VehicleState(x=0.1, y=0.0, psi=0.0, v=1.0, delta=0.0)
    path = build_path_reference(
        x_values=[0.0, 1.0, 2.0, 3.0, 4.0],
        y_values=[0.0, 0.0, 0.2, 0.5, 0.9],
        target_speed=1.2,
    )
    segment, _ = extract_local_reference(
        path=path,
        current_state=state,
        horizon_N=4,
        dt=0.1,
    )

    problem = build_tracking_problem(
        current_state=state,
        reference=segment,
        dt=0.1,
        params=params,
    )

    assert len(problem.ref['x_ref']) == 5
    assert len(problem.ref['u_ref']) == 4
    assert len(problem.lin_sys['A']) == 4
    assert len(problem.lin_sys['B']) == 4
    assert len(problem.lin_sys['c']) == 4
    assert len(problem.x_bar) == 5


def test_estimate_steering_from_yaw_rate_falls_back_at_low_speed():
    params = VehicleParams(wheelbase_L=0.33, delta_max=0.4)

    low_speed_delta = estimate_steering_from_yaw_rate(
        speed=0.05,
        yaw_rate=1.0,
        params=params,
        fallback_delta=0.12,
        min_speed=0.2,
    )
    moving_delta = estimate_steering_from_yaw_rate(
        speed=2.0,
        yaw_rate=0.5,
        params=params,
        fallback_delta=0.0,
        min_speed=0.2,
    )

    assert abs(low_speed_delta - 0.12) < 1e-9
    assert moving_delta > 0.0


def test_integrate_control_step_applies_limits():
    params = VehicleParams(v_max=2.0, delta_max=0.4, a_max=3.0, delta_dot_max=1.0)
    state = VehicleState(x=0.0, y=0.0, psi=0.0, v=1.9, delta=0.39)

    speed, steer = integrate_control_step(
        current_state=state,
        control=[3.0, 1.0],
        dt=0.2,
        params=params,
    )

    assert abs(speed - 2.0) < 1e-9
    assert abs(steer - 0.4) < 1e-9
