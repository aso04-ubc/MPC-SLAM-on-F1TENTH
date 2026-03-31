"""Build the tracking problem that feeds the MPC solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .discretization import euler_step
from .linearization import linearize_discrete_dynamics
from .reference import ReferenceSegment
from .vehicle_model import VehicleControl, VehicleParams, VehicleState, clamp


Vector = List[float]
Matrix = List[List[float]]


@dataclass(frozen=True)
class TrackingProblem:
    """QP-ready tracking problem."""

    lin_sys: Dict[str, List[object]]
    ref: Dict[str, List[Vector]]
    x_bar: List[Vector]
    u_bar: List[Vector]


def curvature_to_steering_angle(curvature: float, params: VehicleParams) -> float:
    """Map path curvature into a steering angle."""
    return clamp(
        math.atan(params.wheelbase_L * float(curvature)),
        -params.delta_max,
        params.delta_max,
    )


def build_state_reference(
    reference: ReferenceSegment,
    params: VehicleParams,
) -> List[Vector]:
    """Convert a reference segment into solver state references."""
    states: List[Vector] = []
    for x_value, y_value, yaw_value, speed_value, curvature_value in zip(
        reference.x,
        reference.y,
        reference.yaw,
        reference.v_ref,
        reference.kappa,
    ):
        states.append(
            [
                float(x_value),
                float(y_value),
                float(yaw_value),
                clamp(float(speed_value), params.v_min, params.v_max),
                curvature_to_steering_angle(curvature_value, params),
            ]
        )
    return states


def build_input_reference(
    x_ref: Sequence[Vector],
    current_state: VehicleState,
    dt: float,
    params: VehicleParams,
) -> List[Vector]:
    """Approximate feedforward controls from successive reference states."""
    if len(x_ref) < 2:
        return []

    controls: List[Vector] = []
    previous_speed = float(current_state.v)
    previous_delta = float(current_state.delta)
    for stage in range(len(x_ref) - 1):
        next_speed = float(x_ref[stage + 1][3])
        next_delta = float(x_ref[stage + 1][4])
        accel = clamp((next_speed - previous_speed) / dt, params.a_min, params.a_max)
        delta_dot = clamp(
            (next_delta - previous_delta) / dt,
            -params.delta_dot_max,
            params.delta_dot_max,
        )
        controls.append([accel, delta_dot])
        previous_speed = next_speed
        previous_delta = next_delta
    return controls


def rollout_linearization_trajectory(
    current_state: VehicleState,
    u_ref: Sequence[Vector],
    dt: float,
    params: VehicleParams,
) -> List[Vector]:
    """Roll out the nominal state trajectory used for linearization."""
    trajectory: List[Vector] = [current_state.as_vector()]
    nominal_state = current_state
    for accel, delta_dot in u_ref:
        nominal_state = euler_step(
            nominal_state,
            VehicleControl(a=float(accel), delta_dot=float(delta_dot)),
            dt,
            params,
        )
        trajectory.append(nominal_state.as_vector())
    return trajectory


def build_tracking_problem(
    current_state: VehicleState,
    reference: ReferenceSegment,
    dt: float,
    params: VehicleParams,
) -> TrackingProblem:
    """Build the solver inputs for one MPC iteration."""
    x_ref = build_state_reference(reference, params)
    if len(x_ref) < 2:
        raise ValueError('reference must contain at least two points')

    u_ref = build_input_reference(x_ref, current_state, dt, params)
    x_bar = rollout_linearization_trajectory(current_state, u_ref, dt, params)

    a_horizon: List[Matrix] = []
    b_horizon: List[Matrix] = []
    c_horizon: List[Vector] = []
    for stage, control in enumerate(u_ref):
        a_matrix, b_matrix, c_vector = linearize_discrete_dynamics(
            x_bar[stage],
            control,
            dt,
            params,
        )
        a_horizon.append(a_matrix)
        b_horizon.append(b_matrix)
        c_horizon.append(c_vector)

    return TrackingProblem(
        lin_sys={'A': a_horizon, 'B': b_horizon, 'c': c_horizon},
        ref={'x_ref': [list(stage) for stage in x_ref], 'u_ref': [list(stage) for stage in u_ref]},
        x_bar=[list(stage) for stage in x_bar],
        u_bar=[list(stage) for stage in u_ref],
    )


def estimate_steering_from_yaw_rate(
    speed: float,
    yaw_rate: float,
    params: VehicleParams,
    fallback_delta: float,
    min_speed: float = 0.2,
) -> float:
    """Estimate steering from odometry yaw rate."""
    if abs(speed) < min_speed:
        return clamp(float(fallback_delta), -params.delta_max, params.delta_max)
    return clamp(
        math.atan(params.wheelbase_L * float(yaw_rate) / float(speed)),
        -params.delta_max,
        params.delta_max,
    )


def integrate_control_step(
    current_state: VehicleState,
    control: Iterable[float],
    dt: float,
    params: VehicleParams,
) -> Tuple[float, float]:
    """Integrate one control input into a speed and steering command."""
    accel, delta_dot = list(control)
    next_speed = clamp(current_state.v + float(accel) * dt, params.v_min, params.v_max)
    next_delta = clamp(
        current_state.delta + float(delta_dot) * dt,
        -params.delta_max,
        params.delta_max,
    )
    return next_speed, next_delta
