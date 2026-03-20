import math

from mpc_controller.mpc_core.discretization import euler_step, rk4_step
from mpc_controller.mpc_core.linearization import linearize_discrete_dynamics
from mpc_controller.mpc_core.vehicle_model import (
    VehicleControl,
    VehicleParams,
    VehicleState,
)


def _vec_add(base, perturbation):
    return [base[index] + perturbation[index] for index in range(len(base))]


def _vec_sub(a, b):
    return [a[index] - b[index] for index in range(len(a))]


def _mat_vec_mul(matrix, vector):
    return [sum(row[col] * vector[col] for col in range(len(vector))) for row in matrix]


def _finite_difference_state_column(state_vec, control_vec, dt, params, index, eps):
    perturb = [0.0] * len(state_vec)
    perturb[index] = eps
    plus = euler_step(_vec_add(state_vec, perturb), control_vec, dt, params).as_vector()
    minus = euler_step(_vec_sub(state_vec, perturb), control_vec, dt, params).as_vector()
    return [(plus[row] - minus[row]) / (2.0 * eps) for row in range(len(plus))]


def _finite_difference_control_column(state_vec, control_vec, dt, params, index, eps):
    perturb = [0.0] * len(control_vec)
    perturb[index] = eps
    plus = euler_step(state_vec, _vec_add(control_vec, perturb), dt, params).as_vector()
    minus = euler_step(state_vec, _vec_sub(control_vec, perturb), dt, params).as_vector()
    return [(plus[row] - minus[row]) / (2.0 * eps) for row in range(len(plus))]


def test_euler_step_straight_line_motion():
    params = VehicleParams()
    state = VehicleState(x=0.0, y=0.0, psi=0.0, v=2.0, delta=0.0)
    control = VehicleControl(a=0.0, delta_dot=0.0)

    next_state = euler_step(state, control, 0.1, params)

    assert abs(next_state.x - 0.2) < 1e-9
    assert abs(next_state.y) < 1e-9
    assert abs(next_state.psi) < 1e-9
    assert abs(next_state.v - 2.0) < 1e-9
    assert abs(next_state.delta) < 1e-9


def test_euler_step_constant_steering_turning_trend():
    params = VehicleParams(wheelbase_L=0.33)
    state = VehicleState(x=0.0, y=0.0, psi=0.2, v=1.5, delta=0.15)
    control = VehicleControl(a=0.0, delta_dot=0.0)

    next_state = euler_step(state, control, 0.1, params)

    assert next_state.x > state.x
    assert next_state.y > state.y
    assert next_state.psi > state.psi


def test_euler_step_low_speed_is_numerically_reasonable():
    params = VehicleParams()
    state = VehicleState(x=1.0, y=-2.0, psi=1.2, v=1e-6, delta=0.3)
    control = VehicleControl(a=0.0, delta_dot=0.0)

    next_state = euler_step(state, control, 0.05, params)

    assert math.isfinite(next_state.x)
    assert math.isfinite(next_state.y)
    assert math.isfinite(next_state.psi)
    assert abs(next_state.x - state.x) < 1e-4
    assert abs(next_state.y - state.y) < 1e-4
    assert abs(next_state.psi - state.psi) < 1e-4


def test_rk4_step_interface_matches_euler_placeholder():
    params = VehicleParams()
    state = VehicleState(x=0.4, y=0.1, psi=-0.3, v=1.2, delta=0.05)
    control = VehicleControl(a=0.2, delta_dot=0.1)

    euler_state = euler_step(state, control, 0.02, params)
    rk4_state = rk4_step(state, control, 0.02, params)

    assert rk4_state.as_vector() == euler_state.as_vector()


def test_linearization_matches_numerical_jacobians():
    params = VehicleParams(wheelbase_L=0.33, delta_max=0.4)
    state = VehicleState(x=1.2, y=-0.5, psi=0.4, v=2.0, delta=0.12)
    control = VehicleControl(a=0.3, delta_dot=-0.2)
    dt = 0.05
    eps = 1e-6

    a_matrix, b_matrix, c_vector = linearize_discrete_dynamics(state, control, dt, params)
    state_vec = state.as_vector()
    control_vec = control.as_vector()

    for col in range(len(state_vec)):
        numerical_col = _finite_difference_state_column(state_vec, control_vec, dt, params, col, eps)
        analytic_col = [a_matrix[row][col] for row in range(len(a_matrix))]
        for analytic_value, numerical_value in zip(analytic_col, numerical_col):
            assert abs(analytic_value - numerical_value) < 1e-5

    for col in range(len(control_vec)):
        numerical_col = _finite_difference_control_column(state_vec, control_vec, dt, params, col, eps)
        analytic_col = [b_matrix[row][col] for row in range(len(b_matrix))]
        for analytic_value, numerical_value in zip(analytic_col, numerical_col):
            assert abs(analytic_value - numerical_value) < 1e-6

    predicted_next = _vec_add(
        _vec_add(_mat_vec_mul(a_matrix, state_vec), _mat_vec_mul(b_matrix, control_vec)),
        c_vector,
    )
    actual_next = euler_step(state, control, dt, params).as_vector()
    for predicted_value, actual_value in zip(predicted_next, actual_next):
        assert abs(predicted_value - actual_value) < 1e-8
