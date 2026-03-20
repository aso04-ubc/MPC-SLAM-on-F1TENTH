"""QP matrix helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Matrix = List[List[float]]


@dataclass(frozen=True)
class SparseMatrixData:
    """Simple sparse matrix data."""

    row_indices: List[int]
    col_indices: List[int]
    data: List[float]
    shape: Tuple[int, int]

    @property
    def nnz(self) -> int:
        """Return the number of stored entries."""
        return len(self.data)

    def to_dense(self) -> Matrix:
        """Convert to a dense matrix."""
        rows, cols = self.shape
        dense = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for row, col, value in zip(self.row_indices, self.col_indices, self.data):
            dense[row][col] = value
        return dense


def _is_matrix(value: object) -> bool:
    """Check if the input looks like a matrix."""
    return isinstance(value, Sequence) and bool(value) and isinstance(value[0], Sequence)


def _diag_matrix(diagonal: Sequence[float]) -> Matrix:
    """Build a diagonal matrix."""
    size = len(diagonal)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for index, value in enumerate(diagonal):
        matrix[index][index] = float(value)
    return matrix


def _ensure_square_matrix(values: Sequence[Sequence[float]], size: int, name: str) -> Matrix:
    """Check and copy a square matrix."""
    if len(values) != size or any(len(row) != size for row in values):
        raise ValueError(f'{name} must be a {size}x{size} matrix')
    return [[float(entry) for entry in row] for row in values]


def _expand_state_weight(values: Sequence[float], nx: int, name: str) -> Matrix:
    """Expand state weights into an nx by nx matrix."""
    if len(values) == nx:
        return _diag_matrix(values)
    if nx == 5 and len(values) == 4:
        expanded = [values[0], values[0], values[1], values[2], values[3]]
        return _diag_matrix(expanded)
    raise ValueError(f'{name} must have length {nx} or 4 when nx==5')


def _expand_input_weight(values: Sequence[float], nu: int, name: str) -> Matrix:
    """Expand input weights into an nu by nu matrix."""
    if len(values) != nu:
        raise ValueError(f'{name} must have length {nu}')
    return _diag_matrix(values)


def canonicalize_weights(weights: Mapping[str, object], nx: int, nu: int) -> Dict[str, Matrix]:
    """Build Q, Q_N, R, and R_delta matrices."""
    q_value = weights.get('Q', [20.0, 10.0, 2.0, 1.0])
    q_n_value = weights.get('Q_N')
    r_value = weights.get('R', [1.0, 10.0])
    r_delta_value = weights.get('R_delta', [0.1, 5.0])

    if _is_matrix(q_value):
        q_matrix = _ensure_square_matrix(q_value, nx, 'Q')
    else:
        q_matrix = _expand_state_weight(list(q_value), nx, 'Q')

    if q_n_value is None:
        q_n_matrix = [[2.0 * entry for entry in row] for row in q_matrix]
    elif _is_matrix(q_n_value):
        q_n_matrix = _ensure_square_matrix(q_n_value, nx, 'Q_N')
    else:
        q_n_matrix = _expand_state_weight(list(q_n_value), nx, 'Q_N')

    if _is_matrix(r_value):
        r_matrix = _ensure_square_matrix(r_value, nu, 'R')
    else:
        r_matrix = _expand_input_weight(list(r_value), nu, 'R')

    if _is_matrix(r_delta_value):
        r_delta_matrix = _ensure_square_matrix(r_delta_value, nu, 'R_delta')
    else:
        r_delta_matrix = _expand_input_weight(list(r_delta_value), nu, 'R_delta')

    return {
        'Q': q_matrix,
        'Q_N': q_n_matrix,
        'R': r_matrix,
        'R_delta': r_delta_matrix,
    }


def canonicalize_constraints(constraints: Mapping[str, float]) -> Dict[str, float]:
    """Fill in missing solver bounds."""
    if 'a_min' in constraints:
        a_min = float(constraints['a_min'])
    else:
        a_min = -float(constraints.get('a_max', 3.0))

    if 'ddelta_min' in constraints:
        ddelta_min = float(constraints['ddelta_min'])
    else:
        ddelta_min = -float(constraints.get('ddelta_max', 1.0))

    return {
        'a_min': a_min,
        'a_max': float(constraints.get('a_max', 3.0)),
        'ddelta_min': ddelta_min,
        'ddelta_max': float(constraints.get('ddelta_max', 1.0)),
        'delta_max': float(constraints.get('delta_max', 0.4)),
        'v_min': float(constraints.get('v_min', 0.0)),
        'v_max': float(constraints.get('v_max', 4.0)),
        'max_iter': int(constraints.get('max_iter', 4000)),
    }


def build_block_diagonal(blocks: Sequence[Matrix], sparse: bool = True):
    """Build a block diagonal matrix."""
    if sparse:
        try:
            from scipy import sparse as scipy_sparse  # type: ignore

            return scipy_sparse.block_diag(blocks, format='csc')
        except Exception:
            pass

    row_indices: List[int] = []
    col_indices: List[int] = []
    data: List[float] = []
    row_offset = 0
    col_offset = 0
    for block in blocks:
        for row_index, row in enumerate(block):
            for col_index, value in enumerate(row):
                if abs(value) <= 1e-12:
                    continue
                row_indices.append(row_offset + row_index)
                col_indices.append(col_offset + col_index)
                data.append(float(value))
        row_offset += len(block)
        col_offset += len(block[0]) if block else 0
    return SparseMatrixData(row_indices, col_indices, data, (row_offset, col_offset))


def build_input_rate_matrix(nu: int, N: int, sparse: bool = True):
    """Build the difference matrix for control changes."""
    row_count = nu * N
    col_count = nu * N
    row_indices: List[int] = []
    col_indices: List[int] = []
    data: List[float] = []

    for stage in range(N):
        for input_index in range(nu):
            row = stage * nu + input_index
            col = stage * nu + input_index
            row_indices.append(row)
            col_indices.append(col)
            data.append(1.0)
            if stage > 0:
                row_indices.append(row)
                col_indices.append((stage - 1) * nu + input_index)
                data.append(-1.0)

    if sparse:
        try:
            from scipy import sparse as scipy_sparse  # type: ignore

            return scipy_sparse.csc_matrix((data, (row_indices, col_indices)), shape=(row_count, col_count))
        except Exception:
            pass

    return SparseMatrixData(row_indices, col_indices, data, (row_count, col_count))


def build_qp_tracking_matrices(nx: int, nu: int, N: int, weights: Mapping[str, object]) -> Dict[str, object]:
    """Build the stacked cost matrices."""
    canonical_weights = canonicalize_weights(weights, nx, nu)
    q_bar = build_block_diagonal([canonical_weights['Q']] * N + [canonical_weights['Q_N']])
    r_bar = build_block_diagonal([canonical_weights['R']] * N)
    r_delta_bar = build_block_diagonal([canonical_weights['R_delta']] * N)
    delta_operator = build_input_rate_matrix(nu, N)

    return {
        'Q': canonical_weights['Q'],
        'Q_N': canonical_weights['Q_N'],
        'R': canonical_weights['R'],
        'R_delta': canonical_weights['R_delta'],
        'Q_bar': q_bar,
        'R_bar': r_bar,
        'R_delta_bar': r_delta_bar,
        'D': delta_operator,
    }
