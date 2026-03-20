"""Vehicle model helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class VehicleParams:
    """Basic vehicle parameters."""

    wheelbase_L: float = 0.33
    delta_max: float = 0.4
    delta_dot_max: float = 1.0
    v_min: float = 0.0
    v_max: float = 4.0
    a_min: float = -3.0
    a_max: float = 3.0


@dataclass(frozen=True)
class VehicleState:
    """Vehicle state."""

    x: float
    y: float
    psi: float
    v: float
    delta: float

    def as_vector(self) -> List[float]:
        """Return the state as a list."""
        return [self.x, self.y, self.psi, self.v, self.delta]


@dataclass(frozen=True)
class VehicleControl:
    """Vehicle control input."""

    a: float
    delta_dot: float

    def as_vector(self) -> List[float]:
        """Return the control as a list."""
        return [self.a, self.delta_dot]


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value into a range."""
    return max(lower, min(upper, value))


def clamp_control(control: VehicleControl, params: VehicleParams) -> VehicleControl:
    """Clamp the control input."""
    return VehicleControl(
        a=clamp(control.a, params.a_min, params.a_max),
        delta_dot=clamp(control.delta_dot, -params.delta_dot_max, params.delta_dot_max),
    )


def clamp_state(state: VehicleState, params: VehicleParams) -> VehicleState:
    """Clamp state values that have limits."""
    return VehicleState(
        x=state.x,
        y=state.y,
        psi=wrap_angle(state.psi),
        v=clamp(state.v, params.v_min, params.v_max),
        delta=clamp(state.delta, -params.delta_max, params.delta_max),
    )


def continuous_dynamics(
    state: VehicleState,
    control: VehicleControl,
    params: VehicleParams,
) -> VehicleState:
    """Compute the continuous model derivative."""
    bounded_state = clamp_state(state, params)
    bounded_control = clamp_control(control, params)

    yaw_rate = 0.0
    if abs(params.wheelbase_L) > 1e-9:
        yaw_rate = bounded_state.v * math.tan(bounded_state.delta) / params.wheelbase_L

    return VehicleState(
        x=bounded_state.v * math.cos(bounded_state.psi),
        y=bounded_state.v * math.sin(bounded_state.psi),
        psi=yaw_rate,
        v=bounded_control.a,
        delta=bounded_control.delta_dot,
    )


def params_from_mapping(values: Dict[str, float]) -> VehicleParams:
    """Build params from a dict."""
    return VehicleParams(
        wheelbase_L=float(values.get('wheelbase_L', VehicleParams.wheelbase_L)),
        delta_max=float(values.get('delta_max', VehicleParams.delta_max)),
        delta_dot_max=float(values.get('ddelta_max', VehicleParams.delta_dot_max)),
        v_min=float(values.get('v_min', VehicleParams.v_min)),
        v_max=float(values.get('v_max', VehicleParams.v_max)),
        a_min=float(values.get('a_min', VehicleParams.a_min)),
        a_max=float(values.get('a_max', VehicleParams.a_max)),
    )


def state_from_iterable(values: Iterable[float]) -> VehicleState:
    """Build a state from 5 values."""
    x, y, psi, v, delta = values
    return VehicleState(x=x, y=y, psi=psi, v=v, delta=delta)


def control_from_iterable(values: Iterable[float]) -> VehicleControl:
    """Build a control from 2 values."""
    a, delta_dot = values
    return VehicleControl(a=a, delta_dot=delta_dot)
