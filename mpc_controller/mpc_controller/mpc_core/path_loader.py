"""Load path references from CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence

from .reference import PathReference, build_path_reference


_X_KEYS = ('x', 'px', 'pos_x')
_Y_KEYS = ('y', 'py', 'pos_y')
_SPEED_KEYS = ('v_ref', 'speed', 'velocity', 'target_speed')


def _normalize_key(name: str) -> str:
    return name.strip().lower().replace(' ', '_')


def _find_key(fieldnames: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalized = {_normalize_key(name): name for name in fieldnames}
    for candidate in candidates:
        original = normalized.get(candidate)
        if original is not None:
            return original
    return None


def _parse_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'failed to parse {field_name!r} value {value!r} as float') from exc


def _override_speed_profile(path: PathReference, speeds: Sequence[float]) -> PathReference:
    if len(speeds) != len(path.x):
        raise ValueError('speed column must have the same number of rows as x/y')
    return PathReference(
        s=list(path.s),
        x=list(path.x),
        y=list(path.y),
        yaw=list(path.yaw),
        kappa=list(path.kappa),
        v_ref=[float(value) for value in speeds],
    )


def load_path_reference_from_csv(csv_path: str, default_speed: float) -> PathReference:
    """Load a path from CSV.

    Supported formats:
    - Headered CSV with at least ``x`` and ``y`` columns.
    - Headerless CSV with two or more numeric columns, where the first two columns are x and y.
    - Optional speed column: ``v_ref`` / ``speed`` / ``velocity`` / ``target_speed``.
    """

    path = Path(csv_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f'path CSV not found: {path}')

    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        sample = handle.read(2048)
        if sample == '':
            raise ValueError(f'path CSV is empty: {path}')
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False

        if has_header:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f'CSV header is missing in {path}')
            x_key = _find_key(reader.fieldnames, _X_KEYS)
            y_key = _find_key(reader.fieldnames, _Y_KEYS)
            speed_key = _find_key(reader.fieldnames, _SPEED_KEYS)
            if x_key is None or y_key is None:
                raise ValueError('CSV header must contain x/y columns')

            x_values: List[float] = []
            y_values: List[float] = []
            speed_values: List[float] = []

            for row in reader:
                if row is None:
                    continue
                if not any((value or '').strip() for value in row.values()):
                    continue
                x_values.append(_parse_float(row.get(x_key), x_key))
                y_values.append(_parse_float(row.get(y_key), y_key))
                if speed_key is not None and (row.get(speed_key) or '').strip():
                    speed_values.append(_parse_float(row.get(speed_key), speed_key))

            path_reference = build_path_reference(x_values=x_values, y_values=y_values, target_speed=default_speed)
            if speed_key is not None and speed_values:
                return _override_speed_profile(path_reference, speed_values)
            return path_reference

        reader = csv.reader(handle)
        x_values = []
        y_values = []
        speed_values = []
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 2:
                raise ValueError('headerless CSV rows must contain at least x and y')
            x_values.append(_parse_float(row[0], 'x'))
            y_values.append(_parse_float(row[1], 'y'))
            if len(row) >= 3 and row[2].strip():
                speed_values.append(_parse_float(row[2], 'v_ref'))

    path_reference = build_path_reference(x_values=x_values, y_values=y_values, target_speed=default_speed)
    if speed_values:
        return _override_speed_profile(path_reference, speed_values)
    return path_reference
