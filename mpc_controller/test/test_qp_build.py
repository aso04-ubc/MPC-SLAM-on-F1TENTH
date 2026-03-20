from mpc_controller.mpc_core.linearization import linearize_discrete_dynamics
from mpc_controller.mpc_core.vehicle_model import VehicleControl, VehicleParams, VehicleState
from mpc_controller.mpc_solvers.qp_cvxpy_osqp import QPMPCController
from mpc_controller.mpc_solvers.qp_matrices import (
    SparseMatrixData,
    build_input_rate_matrix,
    build_qp_tracking_matrices,
    canonicalize_constraints,
    canonicalize_weights,
)
from mpc_controller.mpc_solvers.warmstart import (
    initial_control_guess,
    shift_warmstart_controls,
)


def _shape_of(matrix_like):
    if hasattr(matrix_like, 'shape'):
        return tuple(matrix_like.shape)
    return matrix_like.shape


def test_weight_and_constraint_canonicalization():
    weights = canonicalize_weights({'Q': [20.0, 10.0, 2.0, 1.0], 'R': [1.0, 10.0], 'R_delta': [0.1, 5.0]}, 5, 2)
    constraints = canonicalize_constraints({'a_max': 3.0, 'ddelta_max': 1.0, 'delta_max': 0.4, 'v_min': 0.0, 'v_max': 4.0})

    assert len(weights['Q']) == 5
    assert weights['Q'][0][0] == 20.0
    assert weights['Q'][1][1] == 20.0
    assert weights['R'][1][1] == 10.0
    assert constraints['a_min'] == -3.0
    assert constraints['ddelta_min'] == -1.0


def test_qp_tracking_matrix_shapes_and_sparse_pattern():
    matrices = build_qp_tracking_matrices(
        nx=5,
        nu=2,
        N=4,
        weights={'Q': [20.0, 10.0, 2.0, 1.0], 'R': [1.0, 10.0], 'R_delta': [0.1, 5.0]},
    )

    assert _shape_of(matrices['Q_bar']) == (25, 25)
    assert _shape_of(matrices['R_bar']) == (8, 8)
    assert _shape_of(matrices['R_delta_bar']) == (8, 8)
    assert _shape_of(matrices['D']) == (8, 8)

    if isinstance(matrices['D'], SparseMatrixData):
        assert matrices['D'].nnz == 14


def test_warmstart_helpers_shift_and_repeat_controls():
    initial = initial_control_guess([0.2, -0.1], nu=2, N=3)
    shifted = shift_warmstart_controls(
        previous_solution=[[0.2, -0.1], [0.1, -0.05], [0.0, 0.0]],
        fallback_control=[-0.2, 0.1],
        nu=2,
        N=3,
    )

    assert initial == [[0.2, -0.1], [0.2, -0.1], [0.2, -0.1]]
    assert shifted == [[0.1, -0.05], [0.0, 0.0], [-0.2, 0.1]]


def test_solver_returns_backend_error_fallback_when_cvxpy_missing():
    controller = QPMPCController(
        nx=5,
        nu=2,
        N=3,
        dt=0.1,
        constraints={'a_max': 3.0, 'ddelta_max': 1.0, 'delta_max': 0.4, 'v_min': 0.0, 'v_max': 4.0},
        weights={'Q': [20.0, 10.0, 2.0, 1.0], 'R': [1.0, 10.0], 'R_delta': [0.1, 5.0]},
    )
    params = VehicleParams()
    x_bar = VehicleState(x=0.0, y=0.0, psi=0.0, v=1.0, delta=0.0)
    u_bar = VehicleControl(a=0.0, delta_dot=0.0)
    a_matrix, b_matrix, c_vector = linearize_discrete_dynamics(x_bar, u_bar, 0.1, params)

    lin_sys = {'A': a_matrix, 'B': b_matrix, 'c': c_vector}
    ref = {
        'x_ref': [[0.0, 0.0, 0.0, 1.0, 0.0] for _ in range(4)],
        'u_ref': [[0.0, 0.0] for _ in range(3)],
    }
    u0, info = controller.solve(lin_sys=lin_sys, ref=ref, x0=x_bar.as_vector(), u_prev=[0.1, -0.1], time_budget_ms=10.0)

    assert u0 == [0.1, -0.1]
    assert info['status'] == 'ERROR'
    assert info['fallback_recommended'] is True
    assert 'backend_error' in info


def test_solver_nominal_solve_when_backend_available():
    try:
        import cvxpy  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return

    controller = QPMPCController(
        nx=5,
        nu=2,
        N=3,
        dt=0.1,
        constraints={'a_max': 3.0, 'ddelta_max': 1.0, 'delta_max': 0.4, 'v_min': 0.0, 'v_max': 4.0},
        weights={'Q': [20.0, 10.0, 2.0, 1.0], 'R': [1.0, 10.0], 'R_delta': [0.1, 5.0]},
    )
    params = VehicleParams()
    x_bar = VehicleState(x=0.0, y=0.0, psi=0.0, v=1.0, delta=0.0)
    u_bar = VehicleControl(a=0.0, delta_dot=0.0)
    a_matrix, b_matrix, c_vector = linearize_discrete_dynamics(x_bar, u_bar, 0.1, params)

    lin_sys = {'A': a_matrix, 'B': b_matrix, 'c': c_vector}
    ref = {
        'x_ref': [
            [0.0, 0.0, 0.0, 1.2, 0.0],
            [0.12, 0.0, 0.0, 1.2, 0.0],
            [0.24, 0.0, 0.0, 1.2, 0.0],
            [0.36, 0.0, 0.0, 1.2, 0.0],
        ],
        'u_ref': [[0.0, 0.0] for _ in range(3)],
    }
    u0, info = controller.solve(lin_sys=lin_sys, ref=ref, x0=x_bar.as_vector(), u_prev=[0.0, 0.0], time_budget_ms=50.0)

    assert len(u0) == 2
    assert info['status'] in ('SOLVED', 'SOLVED_INACCURATE')
    assert info['iteration_count'] >= 0
