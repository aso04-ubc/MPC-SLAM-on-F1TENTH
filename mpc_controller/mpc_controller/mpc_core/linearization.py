"""Linearization helpers."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple, Union

from .discretization import ControlLike, StateLike, euler_step
from .vehicle_model import (
    VehicleControl,
    VehicleParams,
    VehicleState,
    clamp_control,
    clamp_state,
    control_from_iterable,
    state_from_iterable,
)


Matrix = List[List[float]]
Vector = List[float]


def _coerce_state(state: StateLike) -> VehicleState:
    """Turn input into a VehicleState."""
    if isinstance(state, VehicleState):
        return state
    return state_from_iterable(state)


def _coerce_control(control: ControlLike) -> VehicleControl:
    """Turn input into a VehicleControl."""
    if isinstance(control, VehicleControl):
        return control
    return control_from_iterable(control)


def _mat_vec_mul(matrix: Matrix, vector: Vector) -> Vector:
    """Multiply a matrix by a vector."""
    return [sum(row[col] * vector[col] for col in range(len(vector))) for row in matrix]


def linearize_discrete_dynamics(
    x_bar: Union[VehicleState, Iterable[float]],
    u_bar: Union[VehicleControl, Iterable[float]],
    dt: float,
    params: VehicleParams,
) -> Tuple[Matrix, Matrix, Vector]:
    """Linearize the Euler model and return A, B, c."""
    state = clamp_state(_coerce_state(x_bar), params)
    control = clamp_control(_coerce_control(u_bar), params)

    cos_psi = math.cos(state.psi)
    sin_psi = math.sin(state.psi)
    cos_delta = math.cos(state.delta)
    sec_delta_sq = 1.0 / max(cos_delta * cos_delta, 1e-12)

    a_matrix: Matrix = [
        [1.0, 0.0, -dt * state.v * sin_psi, dt * cos_psi, 0.0],
        [0.0, 1.0, dt * state.v * cos_psi, dt * sin_psi, 0.0],
        [0.0, 0.0, 1.0, dt * math.tan(state.delta) / params.wheelbase_L, dt * state.v * sec_delta_sq / params.wheelbase_L],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    b_matrix: Matrix = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [dt, 0.0],
        [0.0, dt],
    ]

    x_next = euler_step(state, control, dt, params).as_vector()
    x_vec = state.as_vector()
    u_vec = control.as_vector()
    ax = _mat_vec_mul(a_matrix, x_vec)
    bu = _mat_vec_mul(b_matrix, u_vec)
    c_vector = [x_next[row] - ax[row] - bu[row] for row in range(len(x_next))]
    return a_matrix, b_matrix, c_vector
