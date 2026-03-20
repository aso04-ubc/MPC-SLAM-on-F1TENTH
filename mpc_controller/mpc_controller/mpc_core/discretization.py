"""Discrete model helpers."""

from __future__ import annotations

from typing import Iterable, Union

from .vehicle_model import (
    VehicleControl,
    VehicleParams,
    VehicleState,
    clamp_state,
    continuous_dynamics,
    control_from_iterable,
    state_from_iterable,
    wrap_angle,
)


StateLike = Union[VehicleState, Iterable[float]]
ControlLike = Union[VehicleControl, Iterable[float]]


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


def euler_step(
    x: StateLike,
    u: ControlLike,
    dt: float,
    params: VehicleParams,
) -> VehicleState:
    """Step the model forward with Euler."""
    state = _coerce_state(x)
    control = _coerce_control(u)
    derivative = continuous_dynamics(state, control, params)

    next_state = VehicleState(
        x=state.x + dt * derivative.x,
        y=state.y + dt * derivative.y,
        psi=wrap_angle(state.psi + dt * derivative.psi),
        v=state.v + dt * derivative.v,
        delta=state.delta + dt * derivative.delta,
    )
    return clamp_state(next_state, params)


def rk4_step(
    x: StateLike,
    u: ControlLike,
    dt: float,
    params: VehicleParams,
) -> VehicleState:
    """Placeholder RK4 interface. For now it uses Euler."""
    return euler_step(x=x, u=u, dt=dt, params=params)
