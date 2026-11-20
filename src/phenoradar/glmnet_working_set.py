"""Fit oversized sparse logistic models using glmnet and full KKT checks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import expit

from phenoradar.amino_acid_packed import PackedAAMatrix
from phenoradar.glmnet import GlmnetError, fit_glmnet_path

_SCAN_COLUMNS = 65_536
_ADD_COLUMNS = 1_024
_MAX_ROUNDS = 100


def _column_ranges(x: Any) -> Iterator[tuple[int, int]]:
    start = 0
    while start < x.shape[1]:
        stop = min(start + _SCAN_COLUMNS, x.shape[1])
        # Keep even unusually tall or dense blocks within the local index range.
        while x.indptr[stop] - x.indptr[start] >= np.iinfo(np.int32).max:
            stop = (start + stop) // 2
        if stop == start:
            raise GlmnetError("A single feature exceeds glmnet's sparse index limit")
        yield start, stop
        start = stop


def _scan_gradients(x: Any, residual: np.ndarray, gradients: np.ndarray,
                    ranges: list[tuple[int, int]]) -> None:
    if isinstance(x, PackedAAMatrix):
        x.gradient(residual, gradients)
        return
    for start, stop in ranges:
        left, right = x.indptr[start], x.indptr[stop]
        pointers = x.indptr[start:stop + 1] - left
        block = sparse.csc_array((x.data[left:right], x.indices[left:right], pointers),
                                 shape=(x.shape[0], stop - start), copy=False)
        np.abs(block.T @ residual, out=gradients[start:stop])


def _largest_violations(columns: np.ndarray, violation: np.ndarray,
                        count: int) -> tuple[np.ndarray, np.ndarray]:
    if len(columns) <= count:
        return columns, violation
    cutoff = np.partition(violation, -count)[-count]
    stronger = np.flatnonzero(violation > cutoff)
    ties = np.flatnonzero(violation == cutoff)
    # Use original feature order to make equal violations deterministic.
    n_ties = count - len(stronger)
    first = np.argpartition(columns[ties], n_ties - 1)[:n_ties]
    keep = np.r_[stronger, ties[first]]
    return columns[keep], violation[keep]


def _violators(gradients: np.ndarray, penalty: np.ndarray, working: np.ndarray,
               tolerance: float, ranges: list[tuple[int, int]],
               ) -> tuple[np.ndarray, float]:
    candidates = np.empty(0, dtype=np.int64)
    violations = np.empty(0)
    total = 0.0
    for start, stop in ranges:
        violation = gradients[start:stop] - penalty[start:stop]
        if not np.isfinite(violation).all():
            raise GlmnetError("Nonfinite gradients in the glmnet working-set check")
        a, b = np.searchsorted(working, (start, stop))
        violation[working[a:b] - start] = 0
        total += float(np.maximum(violation, 0).sum())
        found = np.flatnonzero(violation > tolerance)
        local_columns, local_violations = _largest_violations(
            found + start, violation[found], _ADD_COLUMNS,
        )
        candidates, violations = _largest_violations(
            np.r_[candidates, local_columns], np.r_[violations, local_violations], _ADD_COLUMNS,
        )
    return candidates, total


def solve_working_set(
    x: Any, y: np.ndarray, weights: np.ndarray, factors: np.ndarray, path: np.ndarray,
    *, alpha: float, thresh: float, maxit: int,
) -> Iterator[tuple[float, np.ndarray, np.ndarray, int, int, float]]:
    """Yield each requested solution after checking all its KKT conditions.

    Column batches only schedule candidates for the next solve. Every omitted
    feature is checked at every candidate solution before it can be accepted.
    """
    if alpha <= 0 or path[-1] <= 0:
        raise GlmnetError("Oversized sparse models require alpha > 0 and positive lambda")
    # Computing each class mass separately avoids losing a tiny class in 1 - p.
    class_weight = np.bincount(y.astype(int), weights=weights, minlength=2)
    if not np.all(class_weight > 0):
        raise GlmnetError("Both classes need positive normalized weight in the glmnet working set")
    intercept = float(np.log(class_weight[1]) - np.log(class_weight[0]))
    active = np.empty(0, dtype=np.int64)
    residual = weights * (expit(intercept) - y)
    min_factor = float(factors.min())
    ranges = list(_column_ranges(x))
    gradients = np.empty(x.shape[1])
    for step, strength in enumerate(path):
        l1 = strength * alpha * factors
        if not np.isfinite(l1).all() or np.any(l1 <= 0):
            raise GlmnetError("The glmnet working set requires finite, positive L1 penalties")
        # glmnet stops on changes in its objective, while the check below
        # measures gradients. Use a separate absolute gradient tolerance.
        tolerance = max(1e-9, min(1e-7, strength * alpha * min_factor * .01))
        working = active
        coef = np.empty(0)
        passes = fits = 0
        # The final scan at the preceding lambda used the same residual.
        # Reapply the penalty and working-set mask to those unmasked gradients.
        if step == 0:
            _scan_gradients(x, residual, gradients, ranges)
        extra, omitted_kkt = _violators(gradients, l1, working, tolerance, ranges)
        working = np.union1d(working, extra)
        for _ in range(_MAX_ROUNDS):
            local_kkt = np.empty(0)
            if working.size:
                size = (x.indptr[working + 1] - x.indptr[working]).sum(dtype=np.int64)
                if size >= np.iinfo(np.int32).max:
                    raise GlmnetError("The glmnet working set exceeds the sparse index limit")
                local = x[:, working]
                # Replaying the preceding lambda gives glmnet its own warm start,
                # including the intermediate points on the native path.
                warm_path = path[max(step - 1, 0):step + 1]
                for precision in (min(thresh, 1e-12), min(thresh, 1e-15)):
                    model = fit_glmnet_path(
                        local, y, warm_path.tolist(), alpha=alpha, thresh=precision, maxit=maxit,
                        sample_weight=weights, penalty_factor=factors[working],
                        compute_kkt=False, compact_coefficients=True,
                    )[-1]
                    passes += model.n_iter_
                    fits += 1
                    coef = model.coef_
                    intercept = model.intercept_
                    residual = weights * (expit(local @ coef + intercept) - y)
                    gradient = (local.T @ residual
                                + strength * (1 - alpha) * factors[working] * coef)
                    nonzero = coef != 0
                    local_kkt = np.maximum(np.abs(gradient) - l1[working], 0)
                    local_kkt[nonzero] = np.abs(
                        gradient[nonzero] + l1[working[nonzero]] * np.sign(coef[nonzero])
                    )
                    max_error = float(np.max(np.r_[abs(residual.sum()), local_kkt]))
                    if np.isfinite(max_error) and max_error <= tolerance:
                        break
                else:
                    raise GlmnetError(
                        f"glmnet working set failed its KKT check at lambda={strength:g} "
                        f"(residual {max_error:.3g}, tolerance {tolerance:.3g})"
                    )
                del local
                _scan_gradients(x, residual, gradients, ranges)
                extra, omitted_kkt = _violators(gradients, l1, working, tolerance, ranges)
            if not extra.size:
                nonzero = coef != 0
                active = working[nonzero]
                yield (intercept, active, coef[nonzero], passes, fits,
                       float(abs(residual.sum()) + local_kkt.sum() + omitted_kkt))
                break
            working = np.union1d(working, extra)
        else:
            raise GlmnetError(
                f"glmnet working set did not satisfy the full KKT check at lambda={strength:g}"
            )
