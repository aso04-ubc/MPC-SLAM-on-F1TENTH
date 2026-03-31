"""CVXPY + OSQP MPC solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .qp_matrices import canonicalize_constraints, canonicalize_weights
from .warmstart import initial_control_guess, shift_warmstart_controls


Matrix = List[List[float]]
Vector = List[float]


@dataclass(frozen=True)
class LinearSystemHorizon:
    """Linearized dynamics over the horizon."""

    A: List[Matrix]
    B: List[Matrix]
    c: List[Vector]


def _mat_shape(matrix: Sequence[Sequence[float]]) -> Tuple[int, int]:
    """Return matrix shape."""
    return len(matrix), len(matrix[0]) if matrix else 0


def _canonicalize_linear_system(
    lin_sys: Mapping[str, object],
    nx: int,
    nu: int,
    N: int,
) -> LinearSystemHorizon:
    """Turn linear system input into horizon lists."""
    a_value = lin_sys['A']
    b_value = lin_sys['B']
    c_value = lin_sys['c']

    if len(a_value) == N and isinstance(a_value[0], Sequence) and a_value and isinstance(a_value[0][0], Sequence):
        a_list = [
            [[float(entry) for entry in row] for row in stage_matrix]
            for stage_matrix in a_value
        ]
    else:
        a_list = [
            [[float(entry) for entry in row] for row in a_value]
            for _ in range(N)
        ]

    if len(b_value) == N and isinstance(b_value[0], Sequence) and b_value and isinstance(b_value[0][0], Sequence):
        b_list = [
            [[float(entry) for entry in row] for row in stage_matrix]
            for stage_matrix in b_value
        ]
    else:
        b_list = [
            [[float(entry) for entry in row] for row in b_value]
            for _ in range(N)
        ]

    if len(c_value) == N and isinstance(c_value[0], Sequence):
        c_list = [[float(entry) for entry in stage_vector] for stage_vector in c_value]
    else:
        c_list = [[float(entry) for entry in c_value] for _ in range(N)]

    for stage in range(N):
        a_rows, a_cols = _mat_shape(a_list[stage])
        b_rows, b_cols = _mat_shape(b_list[stage])
        if a_rows != nx or a_cols != nx:
            raise ValueError('A stage matrix has incorrect shape')
        if b_rows != nx or b_cols != nu:
            raise ValueError('B stage matrix has incorrect shape')
        if len(c_list[stage]) != nx:
            raise ValueError('c stage vector has incorrect length')

    return LinearSystemHorizon(A=a_list, B=b_list, c=c_list)


def _zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _canonicalize_reference(
    ref: Mapping[str, object],
    nx: int,
    nu: int,
    N: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Turn the reference into x_ref and u_ref lists."""
    x_ref_raw = ref.get('x_ref')
    u_ref_raw = ref.get('u_ref')
    if x_ref_raw is None:
        raise ValueError('ref must provide x_ref')

    x_ref = [[float(value) for value in row] for row in x_ref_raw]
    if len(x_ref) == nx and len(x_ref[0]) == N + 1:
        x_ref = [[x_ref[row][stage] for row in range(nx)] for stage in range(N + 1)]
    elif len(x_ref) == N + 1 and len(x_ref[0]) == nx:
        pass
    else:
        raise ValueError('x_ref must have shape (N+1, nx) or (nx, N+1)')

    if u_ref_raw is None:
        u_ref = _zeros(N, nu)
    else:
        u_ref = [[float(value) for value in row] for row in u_ref_raw]
        if len(u_ref) == nu and len(u_ref[0]) == N:
            u_ref = [[u_ref[row][stage] for row in range(nu)] for stage in range(N)]
        elif len(u_ref) == N and len(u_ref[0]) == nu:
            pass
        else:
            raise ValueError('u_ref must have shape (N, nu) or (nu, N)')

    return x_ref, u_ref


def _map_status(raw_status: str) -> str:
    """Map backend status to our solver status."""
    status = raw_status.lower()
    if status in ('optimal', 'solved'):
        return 'SOLVED'
    if 'optimal_inaccurate' in status or 'solved inaccurate' in status:
        return 'SOLVED_INACCURATE'
    if 'time_limit' in status or 'run time limit' in status:
        return 'TIME_LIMIT_REACHED'
    if 'maximum iterations' in status or 'max_iter' in status or 'user_limit' in status:
        return 'MAX_ITER_REACHED'
    if 'infeasible' in status:
        return 'PRIMAL_INFEASIBLE'
    if 'unbounded' in status:
        return 'DUAL_INFEASIBLE'
    return 'ERROR'


def _fallback_recommended(status: str) -> bool:
    """Check if fallback should be used."""
    return status not in ('SOLVED', 'SOLVED_INACCURATE')


class QPMPCController:
    """QP MPC controller."""

    def __init__(self, nx, nu, N, dt, constraints, weights):
        self.nx = int(nx)
        self.nu = int(nu)
        self.N = int(N)
        self.dt = float(dt)
        self.constraints = canonicalize_constraints(constraints)
        self.weights = canonicalize_weights(weights, self.nx, self.nu)
        self.last_solution: Optional[List[List[float]]] = None

    def solve(self, lin_sys, ref, x0, u_prev, time_budget_ms):
        """Solve the QP and return u0 and info."""
        x0_vec = [float(value) for value in x0]
        u_prev_vec = [float(value) for value in u_prev]
        if len(x0_vec) != self.nx:
            raise ValueError('x0 length must equal nx')
        if len(u_prev_vec) != self.nu:
            raise ValueError('u_prev length must equal nu')

        system = _canonicalize_linear_system(lin_sys, self.nx, self.nu, self.N)
        x_ref, u_ref = _canonicalize_reference(ref, self.nx, self.nu, self.N)

        try:
            import cvxpy as cp  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            fallback_u = list(u_prev_vec)
            info = {
                'status': 'ERROR',
                'solve_time_ms': 0.0,
                'iteration_count': 0,
                'has_solution': False,
                'fallback_recommended': True,
                'backend_error': f'{type(exc).__name__}: {exc}',
            }
            return fallback_u, info

        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))

        q = np.array(self.weights['Q'], dtype=float)
        q_n = np.array(self.weights['Q_N'], dtype=float)
        r = np.array(self.weights['R'], dtype=float)
        r_delta = np.array(self.weights['R_delta'], dtype=float)

        objective = 0.0
        constraints = [x_var[:, 0] == np.array(x0_vec, dtype=float)]

        for stage in range(self.N):
            a_k = np.array(system.A[stage], dtype=float)
            b_k = np.array(system.B[stage], dtype=float)
            c_k = np.array(system.c[stage], dtype=float)
            x_ref_k = np.array(x_ref[stage], dtype=float)
            u_ref_k = np.array(u_ref[stage], dtype=float)

            objective += cp.quad_form(x_var[:, stage] - x_ref_k, q)
            objective += cp.quad_form(u_var[:, stage] - u_ref_k, r)

            if stage == 0:
                delta_u = u_var[:, stage] - np.array(u_prev_vec, dtype=float)
            else:
                delta_u = u_var[:, stage] - u_var[:, stage - 1]
            objective += cp.quad_form(delta_u, r_delta)

            constraints.append(x_var[:, stage + 1] == a_k @ x_var[:, stage] + b_k @ u_var[:, stage] + c_k)
            constraints.append(u_var[0, stage] >= self.constraints['a_min'])
            constraints.append(u_var[0, stage] <= self.constraints['a_max'])
            constraints.append(u_var[1, stage] >= self.constraints['ddelta_min'])
            constraints.append(u_var[1, stage] <= self.constraints['ddelta_max'])

        x_ref_terminal = np.array(x_ref[self.N], dtype=float)
        objective += cp.quad_form(x_var[:, self.N] - x_ref_terminal, q_n)

        for stage in range(self.N + 1):
            constraints.append(x_var[3, stage] >= self.constraints['v_min'])
            constraints.append(x_var[3, stage] <= self.constraints['v_max'])
            constraints.append(x_var[4, stage] >= -self.constraints['delta_max'])
            constraints.append(x_var[4, stage] <= self.constraints['delta_max'])

        problem = cp.Problem(cp.Minimize(objective), constraints)

        warmstart_guess = (
            shift_warmstart_controls(self.last_solution, u_prev_vec, self.nu, self.N)
            if self.last_solution is not None
            else initial_control_guess(u_prev_vec, self.nu, self.N)
        )
        for stage in range(self.N):
            u_var[:, stage].value = warmstart_guess[stage]

        solve_start = perf_counter()
        raw_status = 'ERROR'
        iteration_count = 0
        try:
            problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=self.constraints['max_iter'],
                time_limit=max(float(time_budget_ms) / 1000.0, 1e-4),
            )
            solve_time_ms = (perf_counter() - solve_start) * 1000.0
            raw_status = str(problem.status)
            if problem.solver_stats is not None and problem.solver_stats.num_iters is not None:
                iteration_count = int(problem.solver_stats.num_iters)
        except Exception as exc:
            solve_time_ms = (perf_counter() - solve_start) * 1000.0
            fallback_u = list(u_prev_vec)
            info = {
                'status': 'ERROR',
                'solve_time_ms': solve_time_ms,
                'iteration_count': 0,
                'has_solution': False,
                'fallback_recommended': True,
                'backend_error': f'{type(exc).__name__}: {exc}',
            }
            return fallback_u, info

        status = _map_status(raw_status)
        fallback_recommended = _fallback_recommended(status)
        has_solution = u_var.value is not None and status not in ('PRIMAL_INFEASIBLE', 'DUAL_INFEASIBLE', 'ERROR')

        if has_solution:
            solution = [
                [float(u_var.value[input_index, stage]) for input_index in range(self.nu)]
                for stage in range(self.N)
            ]
            self.last_solution = solution
            u0 = list(solution[0])
        elif self.last_solution is not None:
            shifted = shift_warmstart_controls(self.last_solution, u_prev_vec, self.nu, self.N)
            self.last_solution = shifted
            u0 = list(shifted[0])
        else:
            u0 = list(u_prev_vec)

        info = {
            'status': status,
            'solve_time_ms': solve_time_ms,
            'iteration_count': iteration_count,
            'has_solution': has_solution,
            'fallback_recommended': fallback_recommended,
        }
        return u0, info
