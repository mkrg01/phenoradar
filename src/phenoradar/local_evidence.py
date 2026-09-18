"""Bounded workspaces for ensemble-local linear contributions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from phenoradar.cv import FeatureScaler, apply_feature_scaling

_CONTRIBUTION_BLOCK_CELLS = 262_144


@dataclass(frozen=True)
class LocalContribution:
    feature: str
    mean: float
    mean_abs: float
    minimum: float
    maximum: float


def _top_indices(scores: np.ndarray, feature_rank: np.ndarray, limit: int) -> np.ndarray:
    eligible = np.flatnonzero(scores > 1e-12)
    if eligible.size > limit:
        cutoff = np.partition(scores[eligible], -limit)[-limit]
        above = eligible[scores[eligible] > cutoff]
        tied = eligible[scores[eligible] == cutoff]
        remaining = limit - above.size
        if tied.size > remaining:
            tied = tied[np.argpartition(feature_rank[tied], remaining - 1)[:remaining]]
        eligible = np.concatenate((above, tied))
    return eligible[np.lexsort((feature_rank[eligible], -scores[eligible]))]


def iter_top_contributions(
    matrix: np.ndarray,
    feature_names: list[str],
    *,
    rows: np.ndarray,
    model_features: Sequence[list[str]],
    scalers: Sequence[FeatureScaler],
    coefficients: Sequence[np.ndarray],
    scaling_method: str,
    top_features: int,
) -> Iterator[tuple[int, list[LocalContribution]]]:
    """Yield top features for each requested row, including absent-model zeros.

    Only one row block and one model's contributions are held at a time. The
    fitted scalers still see their complete, ordered feature schema. A single
    very wide row can exceed the cell target; no model x species x feature
    cube or full-population contribution summaries are allocated.
    """
    if top_features < 1:
        raise ValueError("Local evidence top_features must be >= 1")
    count = len(model_features)
    if count == 0 or len(scalers) != count or len(coefficients) != count:
        raise ValueError("Local evidence requires matching models, scalers, and coefficients")
    index = {name: i for i, name in enumerate(feature_names)}
    union = sorted({name for names in model_features for name in names})
    union_index = {name: i for i, name in enumerate(union)}
    if not union:
        for row in range(len(rows)):
            yield row, []
        return
    source_columns = []
    target_columns = []
    for names, coef in zip(model_features, coefficients, strict=True):
        if coef.shape != (len(names),):
            raise ValueError("Local evidence coefficient width does not match its feature schema")
        source_columns.append(np.array([index[name] for name in names], dtype=int))
        target_columns.append(np.array([union_index[name] for name in names], dtype=int))
    feature_rank = np.arange(len(union))  # Union is already in lexical order.
    block_rows = max(1, _CONTRIBUTION_BLOCK_CELLS // len(union))
    for start in range(0, len(rows), block_rows):
        selected_rows = rows[start : start + block_rows]
        shape = (len(selected_rows), len(union))
        totals = np.zeros(shape)
        magnitudes = np.zeros(shape)
        minima = np.full(shape, np.inf)
        maxima = np.full(shape, -np.inf)
        contribution = np.zeros(shape)
        for source, target, scaler, coef in zip(
            source_columns, target_columns, scalers, coefficients, strict=True
        ):
            values = apply_feature_scaling(
                matrix[np.ix_(selected_rows, source)], scaler, scaling_method
            )
            contribution.fill(0.0)
            contribution[:, target] = values * coef
            totals += contribution
            np.minimum(minima, contribution, out=minima)
            np.maximum(maxima, contribution, out=maxima)
            np.abs(contribution, out=contribution)
            magnitudes += contribution
        totals /= count
        magnitudes /= count
        for local_row in range(len(selected_rows)):
            indices = _top_indices(magnitudes[local_row], feature_rank, top_features)
            yield (
                start + local_row,
                [
                    LocalContribution(
                        feature=union[i],
                        mean=float(totals[local_row, i]),
                        mean_abs=float(magnitudes[local_row, i]),
                        minimum=float(minima[local_row, i]),
                        maximum=float(maxima[local_row, i]),
                    )
                    for i in indices
                ],
            )
