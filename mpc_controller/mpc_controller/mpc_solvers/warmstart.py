"""Warm-start helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def reshape_controls(control_values: Sequence[float], nu: int, N: int) -> List[List[float]]:
    """Turn a flat control vector into stage controls."""
    if len(control_values) != nu * N:
        raise ValueError('control_values length must equal nu * N')
    return [
        [float(control_values[stage * nu + input_index]) for input_index in range(nu)]
        for stage in range(N)
    ]


def flatten_controls(stage_controls: Sequence[Sequence[float]]) -> List[float]:
    """Flatten stage controls."""
    flattened: List[float] = []
    for stage_control in stage_controls:
        flattened.extend(float(value) for value in stage_control)
    return flattened


def initial_control_guess(u_prev: Iterable[float], nu: int, N: int) -> List[List[float]]:
    """Copy the last control across the horizon."""
    base = [float(value) for value in u_prev]
    if len(base) != nu:
        raise ValueError('u_prev length must equal nu')
    return [list(base) for _ in range(N)]


def shift_warmstart_controls(
    previous_solution: Sequence[Sequence[float]],
    fallback_control: Iterable[float],
    nu: int,
    N: int,
) -> List[List[float]]:
    """Shift the old solution forward by one step."""
    fallback = [float(value) for value in fallback_control]
    if len(fallback) != nu:
        raise ValueError('fallback_control length must equal nu')
    if len(previous_solution) != N:
        raise ValueError('previous_solution must contain N stage controls')

    shifted: List[List[float]] = []
    for stage in range(max(N - 1, 0)):
        current = [float(value) for value in previous_solution[stage + 1]]
        if len(current) != nu:
            raise ValueError('each stage control must have length nu')
        shifted.append(current)
    if N > 0:
        shifted.append(list(fallback))
    return shifted
