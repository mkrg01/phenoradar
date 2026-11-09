"""Training-local relative lambda paths, streamed scoring and bounded extensions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import expit

from phenoradar.config import AppConfig
from phenoradar.glmnet import (
    DEFAULT_ALPHA,
    DEFAULT_MAXIT,
    DEFAULT_THRESH,
    GlmnetError,
    GlmnetLogisticRegression,
)
from phenoradar.model_selection import Candidate, auto_lambda_spec

AUTO_COLUMNS = (
    "lambda_fraction",
    "lambda_max",
    "lambda_actual",
    "lambda_extension_round",
    "nonzero_expression",
    "nonzero_amino_acid",
    "lambda_path_stop_reason",
    "lambda_path_min_fraction",
    "lambda_path_attempted_rounds",
)


def normalized_weights(y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float).copy()
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("lambda_max requires finite nonnegative weights with positive total")
    w /= w.max()
    return np.asarray(w / w.sum(), dtype=float)


def null_gradients(x: Any, y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """One design scan; affine sparse shifts cancel because residuals sum to zero."""
    w = normalized_weights(y, weights)
    return np.asarray(np.abs(np.asarray(x.T @ (w * (y - w @ y))).reshape(-1)), dtype=float)


def lambda_max(
    x: Any,
    y: np.ndarray,
    weights: np.ndarray | None,
    *,
    alpha: float,
    penalty_factor: np.ndarray | None = None,
) -> float:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("lambda_max requires alpha > 0")
    gradient = null_gradients(x, y, weights)
    if penalty_factor is not None:
        factors = np.asarray(penalty_factor, dtype=float)
        if (
            factors.shape != gradient.shape
            or not np.isfinite(factors).all()
            or np.any(factors <= 0)
        ):
            raise ValueError("lambda_max requires one positive finite penalty per feature")
        gradient /= factors
    return float(np.max(gradient, initial=0) / alpha)


def fit_null_model(
    estimator: GlmnetLogisticRegression, x: Any, y: np.ndarray, weights: np.ndarray | None
) -> None:
    prevalence = float(normalized_weights(y, weights) @ y)
    if not 0 < prevalence < 1:
        raise ValueError("Both classes must have positive sample weight")
    estimator.coef_ = np.zeros(x.shape[1])
    estimator.intercept_ = float(np.log(prevalence / (1 - prevalence)))
    estimator.classes_ = np.array([0, 1])
    estimator.n_features_in_ = x.shape[1]
    estimator.n_iter_ = 0
    estimator.path_length_ = 1
    estimator.native_path_length_ = 0
    estimator.converged_ = True
    estimator.kkt_residual_ = 0.0


def realized_params(params: dict[str, Any], model: Any) -> dict[str, Any]:
    result = dict(params)
    if "lambda_fraction" in params:
        result.update({"lambda": float(model.lambda_), "lambda_max": float(model.lambda_max_)})
    return result


@dataclass
class AutoSearch:
    candidates: list[Candidate]
    scores: list[tuple[float, list[dict[str, Any]]]]


class _FoldCache:
    """Keep only small metadata in RAM; CSC buffers live in scope-local temp files."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.items: list[tuple[Any, dict[str, Any], tuple[float, float]]] = []

    @staticmethod
    def _save(path: Path, x: Any) -> None:
        if sparse.issparse(x):
            for name in ["data", "indices", "indptr"]:
                np.save(str(path) + "_" + name, getattr(x, name), allow_pickle=False)
        else:
            np.save(path, x, allow_pickle=False)

    @staticmethod
    def _load(path: Path, shape: tuple[int, int], is_sparse: bool) -> Any:
        if is_sparse:
            arrays = [
                np.load(str(path) + "_" + k + ".npy", mmap_mode="r")
                for k in ["data", "indices", "indptr"]
            ]
            return sparse.csc_matrix(tuple(arrays), shape=shape, copy=False)
        return np.load(str(path) + ".npy", mmap_mode="r")

    def add(self, fold: Any, gradients: tuple[float, float]) -> None:
        i = len(self.items)
        for name in ["x_train", "x_valid"]:
            self._save(self.directory / f"{i}_{name}", getattr(fold, name))
        metadata = {
            k: getattr(fold, k)
            for k in [
                "inner_fold_id",
                "y_train",
                "y_valid",
                "sample_weight",
                "n_expression_features",
            ]
        }
        metadata["shapes"] = [fold.x_train.shape, fold.x_valid.shape]
        metadata["sparse"] = [sparse.issparse(fold.x_train), sparse.issparse(fold.x_valid)]
        self.items.append((type(fold), metadata, gradients))

    def __iter__(self) -> Any:
        for i, (cls, metadata, gradients) in enumerate(self.items):
            values = {k: v for k, v in metadata.items() if k not in {"shapes", "sparse"}}
            for j, name in enumerate(["x_train", "x_valid"]):
                values[name] = self._load(
                    self.directory / f"{i}_{name}", metadata["shapes"][j], metadata["sparse"][j]
                )
            yield cls(**values), gradients


def score_auto_paths(
    config: AppConfig,
    candidates: list[Candidate],
    folds: Iterable[Any],
    *,
    fit_path: Callable[..., Any],
    metric: Callable[[np.ndarray, np.ndarray], float],
    diagnostic: Callable[[Any], Any],
    warnings: list[str],
    progress: Callable[[str], None] | None = None,
) -> AutoSearch:
    """Tune common fractions; all scales use only the corresponding training fold.

    Extension decisions see aggregated inner-CV scores only. Native path prefixes
    are refitted on cached designs; no cross-call solver warm state is assumed.
    """
    spec = auto_lambda_spec(config)
    assert spec is not None
    combined = config.data.feature_mode == "combined"
    default_alpha = 1.0 if combined else DEFAULT_ALPHA
    collected: dict[int, list[dict[str, Any]]] = defaultdict(list)
    complete: list[Candidate] = []
    pending = candidates
    current_min = spec.min_ratio
    attempted = 0
    reason = "extension_disabled"

    def path_key(candidate: Candidate) -> tuple[float, float, int, float]:
        p = candidate.params
        return (
            float(p.get("alpha", default_alpha)),
            float(p.get("thresh", DEFAULT_THRESH)),
            int(p.get("maxit", DEFAULT_MAXIT)),
            float(p.get("amino_acid_penalty_ratio", 1)) if combined else 1.0,
        )

    def score_fold(
        fold: Any, maxima: tuple[float, float], round_id: int
    ) -> dict[int, list[dict[str, Any]]]:
        rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        paths: dict[tuple[float, float, int, float], list[Candidate]] = defaultdict(list)
        for candidate in pending:
            paths[path_key(candidate)].append(candidate)
        n = (
            fold.n_expression_features
            if combined
            else (fold.x_train.shape[1] if config.data.feature_mode == "expression" else 0)
        )
        assert n is not None
        for (alpha, thresh, maxit, ratio), path in paths.items():
            maximum = max(maxima[0], maxima[1] / ratio) / alpha
            factors = (
                np.r_[np.ones(n), np.full(fold.x_train.shape[1] - n, ratio)] if combined else None
            )

            def record(
                candidate: Candidate, probability: np.ndarray, model: Any, maximum: float = maximum
            ) -> None:
                fraction = float(candidate.params["lambda_fraction"])
                if model is not None:
                    model.lambda_fraction_ = fraction
                    model.lambda_max_ = maximum
                if model is None:
                    nonzero_expression = nonzero_amino_acid = 0
                elif hasattr(model, "coefficient_indices_"):
                    nonzero_expression = int(np.count_nonzero(model.coefficient_indices_ < n))
                    nonzero_amino_acid = len(model.coefficient_indices_) - nonzero_expression
                else:
                    nonzero_expression = int(np.count_nonzero(model.coef_[:n]))
                    nonzero_amino_acid = int(np.count_nonzero(model.coef_[n:]))
                rows[candidate.candidate_index].append(
                    {
                        "candidate_index": candidate.candidate_index,
                        "inner_fold_id": fold.inner_fold_id,
                        "metric_name": config.model_selection.selection_metric,
                        "metric_value": metric(fold.y_valid, probability),
                        "params_json": json.dumps(candidate.params, sort_keys=True),
                        "fit_diagnostic": None if model is None else diagnostic(model),
                        "lambda_fraction": fraction,
                        "lambda_max": maximum,
                        "lambda_actual": maximum * fraction,
                        "lambda_extension_round": round_id,
                        "nonzero_expression": nonzero_expression,
                        "nonzero_amino_acid": nonzero_amino_acid,
                    }
                )

            requested = []
            for candidate in path:
                if maximum == 0 or candidate.params["lambda_fraction"] == 1:
                    w = normalized_weights(fold.y_train, fold.sample_weight)
                    record(candidate, np.full(len(fold.y_valid), w @ fold.y_train), None)
                else:
                    requested.append(candidate)
            if not requested:
                continue
            warmup = [
                maximum * float(c.params["lambda_fraction"])
                for c in [*complete, *pending]
                if path_key(c) == (alpha, thresh, maxit, ratio)
            ]

            def consume(
                index: int,
                model: Any,
                targets: list[Candidate] = requested,
                emit: Callable[..., None] = record,
            ) -> None:
                # Prepared validation arrays are finite; project exact nonzeros
                # before multiplication, avoiding repeated scans of millions of zeros.
                if hasattr(model, "coefficient_indices_"):
                    active, values = model.coefficient_indices_, model.coefficient_values_
                else:
                    active = np.flatnonzero(model.coef_ != 0)
                    values = model.coef_[active]
                probability = expit(
                    np.asarray(fold.x_valid[:, active] @ values).reshape(-1)
                    + model.intercept_
                )
                emit(targets[index], probability, model)

            fit_path(
                fold.x_train,
                fold.y_train,
                [maximum * float(c.params["lambda_fraction"]) for c in requested],
                alpha=alpha,
                thresh=thresh,
                maxit=maxit,
                penalty_factor=factors,
                sample_weight=fold.sample_weight,
                compute_kkt=False,
                compact_coefficients=config.data.feature_mode in {"amino_acid", "combined"},
                warmup_lambdas=warmup,
                on_model=consume,
            )
        return rows

    with TemporaryDirectory(prefix="phenoradar_lambda_folds_") as directory:
        cache = _FoldCache(Path(directory))
        for round_id in range(spec.max_extensions + 1):
            round_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
            attempted = round_id
            if progress:
                progress(
                    f"Automatic lambda round {round_id}: {len(pending)} new candidates; "
                    f"minimum fraction {current_min:.6g}."
                )

            def initial() -> Any:
                for fold in folds:
                    gradient = null_gradients(fold.x_train, fold.y_train, fold.sample_weight)
                    n = (
                        fold.n_expression_features
                        if combined
                        else (len(gradient) if config.data.feature_mode == "expression" else 0)
                    )
                    assert n is not None
                    maxima = (
                        float(np.max(gradient[:n], initial=0)),
                        float(np.max(gradient[n:], initial=0)),
                    )
                    del gradient
                    if spec.extend_lower and spec.max_extensions:
                        cache.add(fold, maxima)
                    yield fold, maxima

            try:
                for fold, maxima in initial() if round_id == 0 else cache:
                    if progress:
                        progress(
                            f"Automatic lambda round {round_id}: score inner fold "
                            f"{fold.inner_fold_id}."
                        )
                    result = score_fold(fold, maxima, round_id)
                    for key, values in result.items():
                        round_rows[key].extend(values)
                    del result, fold
            except (GlmnetError, OSError, MemoryError) as error:
                if round_id == 0:
                    raise
                reason = "extension_failed"
                warnings.append(
                    f"Automatic lambda extension failed; retaining completed inner "
                    f"CV candidates: {error}"
                )
                break
            for key, values in round_rows.items():
                collected[key].extend(values)
            complete.extend(pending)
            means = {
                c.candidate_index: float(
                    np.nanmean([row["metric_value"] for row in collected[c.candidate_index]])
                )
                for c in complete
            }
            finite = [c for c in complete if np.isfinite(means[c.candidate_index])]
            if not finite:
                raise ValueError("No finite automatic lambda inner-CV scores")
            sign = 1 if config.model_selection.selection_metric == "log_loss" else -1
            best_value = min(sign * means[c.candidate_index] for c in finite)
            tied = [c for c in finite if abs(sign * means[c.candidate_index] - best_value) <= 1e-10]
            best_fraction = max(float(c.params["lambda_fraction"]) for c in tied)
            if best_fraction > current_min * (1 + 1e-10):
                reason = "interior_or_null_optimum"
                break
            if not spec.extend_lower:
                reason = "extension_disabled_at_boundary"
                break
            if round_id == spec.max_extensions:
                reason = "extension_limit_at_boundary"
                warnings.append(
                    "Automatic lambda optimum remains at the lower boundary after "
                    f"{round_id} extensions (fraction={current_min:.6g})."
                )
                break
            bases = {
                json.dumps(
                    {k: v for k, v in c.params.items() if k != "lambda_fraction"}, sort_keys=True
                )
                for c in complete
            }
            added = max(1, int(np.ceil((spec.n_points - 1) / -np.log10(spec.min_ratio))))
            fractions = np.geomspace(current_min, current_min / 10, added + 1)[1:]
            pending = []
            for encoded in sorted(bases):
                for fraction in fractions:
                    pending.append(
                        Candidate(
                            len(complete) + len(pending),
                            {**json.loads(encoded), "lambda_fraction": float(fraction)},
                        )
                    )
            current_min /= 10
    for rows in collected.values():
        for row in rows:
            row.update(
                lambda_path_stop_reason=reason,
                lambda_path_min_fraction=min(float(c.params["lambda_fraction"]) for c in complete),
                lambda_path_attempted_rounds=attempted,
            )
    scores = [
        (
            float(np.nanmean([r["metric_value"] for r in collected[c.candidate_index]])),
            collected[c.candidate_index],
        )
        for c in complete
    ]
    return AutoSearch(complete, scores)
