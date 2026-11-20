"""Optional native operations preserve NumPy's statistics and mode/tie rules."""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["native", "decode", "decode_compact_into", "center", "append", "centering_mass"]

try:
    from phenoradar import _aa_kernels as native  # type: ignore[attr-defined]
except ImportError:
    native = None


def decode(data: np.ndarray, nrows: int, width: int, rows: np.ndarray,
           columns: np.ndarray) -> np.ndarray | None:
    if native is None or not np.little_endian:
        return None
    result = np.empty((len(rows), len(columns)), dtype=np.float32)
    native.decode(np.ascontiguousarray(data), nrows, width,
                  np.ascontiguousarray(rows, dtype=np.int64),
                  np.ascontiguousarray(columns, dtype=np.int64), result)
    return result


def decode_compact_into(
    data: np.ndarray, nrows: int, width: int, coverage_rows: np.ndarray,
    state_rows: np.ndarray, samples: np.ndarray, targets: np.ndarray,
    selected: np.ndarray, features: np.ndarray, values: np.ndarray,
    observed: np.ndarray, coverage: np.ndarray | None, *, packed: bool,
) -> bool:
    """Copy exact float32 bits straight into the cache caller's output axes."""
    kernel = getattr(native, "decode_compact", None)
    if kernel is None or not np.little_endian:
        return False
    axes = tuple(np.ascontiguousarray(axis, dtype=np.int64) for axis in
                 [coverage_rows, state_rows, samples, targets, selected, features])
    kernel(np.ascontiguousarray(data), nrows, width, axes,
           (values, observed, coverage), packed)
    return True


def center(values: np.ndarray, selected: np.ndarray, sites: np.ndarray,
           means: np.ndarray, mass: np.ndarray) -> np.ndarray:
    result = np.empty((len(values), len(selected)))
    native.center(np.ascontiguousarray(values), np.ascontiguousarray(selected, dtype=np.int64),
                  np.ascontiguousarray(sites, dtype=np.int64), np.ascontiguousarray(means),
                  np.ascontiguousarray(mass), result, len(values), values.shape[1], mass.shape[1])
    return result


def packed_csc(ones: np.ndarray, mass: np.ndarray, sites: np.ndarray, rows: np.ndarray,
               categories: np.ndarray, width: int,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    kernel = getattr(native, "packed_csc", None)
    capacity = len(rows) * len(sites)
    if kernel is None or max(capacity, len(rows), len(sites)) > np.iinfo(np.int32).max:
        return None
    data = np.empty(capacity, dtype=np.float64)
    indices = np.empty(capacity, dtype=np.int32)
    indptr = np.empty(len(sites) + 1, dtype=np.int32)
    count = kernel(np.ascontiguousarray(ones), np.ascontiguousarray(mass),
                   np.ascontiguousarray(sites, dtype=np.int64),
                   np.ascontiguousarray(rows, dtype=np.int64),
                   np.ascontiguousarray(categories.T), data, indices, indptr, width, len(mass))
    return data[:count], indices[:count], indptr


def append(encoding: Any, values: np.ndarray) -> None:
    from phenoradar.amino_acid_sparse import modal_shift

    if not values.shape[1]:
        return
    shift = modal_shift(encoding._selected(values, encoding.training))
    encoding.shifts.append(shift)
    for i, rows in enumerate(encoding.rows):
        capacity = len(rows) * values.shape[1]
        data = np.empty(capacity)
        indices = np.empty(capacity, dtype=np.int32)
        ends = np.empty(values.shape[1], dtype=np.int64)
        count = native.csc(np.ascontiguousarray(values), np.ascontiguousarray(shift),
                           np.ascontiguousarray(rows, dtype=np.int64), data, indices, ends,
                           len(values), values.shape[1])
        # _append_csc writes or buffers the arrays synchronously.
        encoding._append_csc(i, data[:count], indices[:count], ends)
    encoding.width += values.shape[1]


def centering_mass(values: np.ndarray, sites: np.ndarray, residues: np.ndarray,
                   eligible: np.ndarray, known: np.ndarray) -> np.ndarray | None:
    """Sum only provably exact sites; preserve the original reduction elsewhere."""
    kernel = getattr(native, "centering_mass", None)
    if kernel is None or values.dtype != np.float32:
        return None
    steps = np.diff(sites)
    if (np.any((residues < 0) | (residues >= 20))
            or not np.all((steps > 0) | ((steps == 0) & (np.diff(residues) > 0)))):
        return None
    local = np.cumsum(eligible) - 1
    keep = eligible[sites]
    columns = np.flatnonzero(keep & known[sites, residues])
    nsites = int(eligible.sum())
    mass = np.empty((len(values), nsites))
    fallback = np.empty(nsites, dtype=bool)
    kernel(np.ascontiguousarray(values), np.ascontiguousarray(columns, dtype=np.int64),
           np.ascontiguousarray(local[sites[columns]], dtype=np.int64), mass, fallback,
           len(values), values.shape[1], nsites)
    if fallback.any():
        from phenoradar.amino_acid_encoding import _compact_mass

        positions = np.flatnonzero(keep)
        columns = positions[fallback[local[sites[positions]]]]
        raw = np.nan_to_num(values[:, columns], copy=False)
        raw *= known[sites[columns], residues[columns]][None, :]
        local_fallback = np.cumsum(fallback) - 1
        mass[:, fallback] = _compact_mass(
            raw, local_fallback[local[sites[columns]]], residues[columns], int(fallback.sum()))
    return mass
