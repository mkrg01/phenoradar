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


def score_auto_fold(
    config: AppConfig,
    pending: list[Candidate],
    complete: list[Candidate],
    fold: Any,
    maxima: tuple[float, float],
    round_id: int,
    *,
    fit_path: Callable[..., Any],
    metric: Callable[[np.ndarray, np.ndarray], float],
    diagnostic: Callable[[Any], Any],
) -> dict[int, list[dict[str, Any]]]:
    combined = config.data.feature_mode == "combined"
    default_alpha = 1.0 if combined else DEFAULT_ALPHA

    def path_key(candidate: Candidate) -> tuple[float, float, int, float]:
        p = candidate.params
        return (
            float(p.get("alpha", default_alpha)),
            float(p.get("thresh", DEFAULT_THRESH)),
            int(p.get("maxit", DEFAULT_MAXIT)),
            float(p.get("amino_acid_penalty_ratio", 1)) if combined else 1.0,
        )

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



def fold_maxima(config: AppConfig, fold: Any) -> tuple[float, float]:
    """Scan this training design once, shared by every penalty family."""
    gradient = null_gradients(fold.x_train, fold.y_train, fold.sample_weight)
    n = (fold.n_expression_features if config.data.feature_mode == "combined"
         else len(gradient) if config.data.feature_mode == "expression" else 0)
    assert n is not None
    return (float(np.max(gradient[:n], initial=0)),
            float(np.max(gradient[n:], initial=0)))


class AutoPathState:
    """Commit complete rounds in fold order; never decide from partial scores."""

    def __init__(self, config: AppConfig, candidates: list[Candidate]) -> None:
        spec = auto_lambda_spec(config)
        assert spec is not None
        self.config, self.spec = config, spec
        self.pending = candidates
        self.complete: list[Candidate] = []
        self.collected: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.current_min = spec.min_ratio
        self.round_id = 0
        self.reason = "extension_disabled"
        self.warnings: list[str] = []
        self.finished = False

    def commit_round(self, results: Iterable[dict[int, list[dict[str, Any]]]]) -> bool:
        """Return whether another round is needed, after every fold completed."""
        assert not self.finished
        round_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            for key, values in result.items():
                round_rows[key].extend(values)
        for key, values in round_rows.items():
            self.collected[key].extend(values)
        self.complete.extend(self.pending)
        means = {
            c.candidate_index: float(np.nanmean(
                [row["metric_value"] for row in self.collected[c.candidate_index]]))
            for c in self.complete
        }
        finite = [c for c in self.complete if np.isfinite(means[c.candidate_index])]
        if not finite:
            raise ValueError("No finite automatic lambda inner-CV scores")
        sign = 1 if self.config.model_selection.selection_metric == "log_loss" else -1
        best_value = min(sign * means[c.candidate_index] for c in finite)
        tied = [c for c in finite if abs(sign * means[c.candidate_index] - best_value) <= 1e-10]
        best_fraction = max(float(c.params["lambda_fraction"]) for c in tied)
        if best_fraction > self.current_min * (1 + 1e-10):
            self.reason = "interior_or_null_optimum"
        elif not self.spec.extend_lower:
            self.reason = "extension_disabled_at_boundary"
        elif self.round_id == self.spec.max_extensions:
            self.reason = "extension_limit_at_boundary"
            self.warnings.append(
                "Automatic lambda optimum remains at the lower boundary after "
                f"{self.round_id} extensions (fraction={self.current_min:.6g}).")
        else:
            bases = {
                json.dumps({k: v for k, v in c.params.items() if k != "lambda_fraction"},
                           sort_keys=True)
                for c in self.complete
            }
            added = max(1, int(np.ceil(
                (self.spec.n_points - 1) / -np.log10(self.spec.min_ratio))))
            fractions = np.geomspace(self.current_min, self.current_min / 10, added + 1)[1:]
            self.pending = []
            for encoded in sorted(bases):
                for fraction in fractions:
                    self.pending.append(Candidate(
                        len(self.complete) + len(self.pending),
                        {**json.loads(encoded), "lambda_fraction": float(fraction)}))
            self.current_min /= 10
            self.round_id += 1
            return True
        self.finished = True
        return False

    def fail_extension(self, error: Exception) -> None:
        assert self.round_id > 0
        self.reason = "extension_failed"
        self.finished = True
        self.warnings.append(
            "Automatic lambda extension failed; retaining completed inner "
            f"CV candidates: {error}")

    def result(self) -> AutoSearch:
        assert self.finished
        minimum = min(float(c.params["lambda_fraction"]) for c in self.complete)
        for rows in self.collected.values():
            for row in rows:
                row.update(lambda_path_stop_reason=self.reason,
                           lambda_path_min_fraction=minimum,
                           lambda_path_attempted_rounds=self.round_id)
        return AutoSearch(self.complete, [
            (float(np.nanmean([r["metric_value"] for r in self.collected[c.candidate_index]])),
             self.collected[c.candidate_index]) for c in self.complete])


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
    """Serial driver for training-local paths and complete-round extensions."""
    state = AutoPathState(config, candidates)
    with TemporaryDirectory(prefix="phenoradar_lambda_folds_") as directory:
        cache = _FoldCache(Path(directory))

        def initial() -> Any:
            for fold in folds:
                maxima = fold_maxima(config, fold)
                if state.spec.extend_lower and state.spec.max_extensions:
                    cache.add(fold, maxima)
                yield fold, maxima

        while not state.finished:
            if progress:
                progress(f"Automatic lambda round {state.round_id}: "
                         f"{len(state.pending)} new candidates; "
                         f"minimum fraction {state.current_min:.6g}.")
            try:
                results = []
                for fold, maxima in initial() if state.round_id == 0 else cache:
                    if progress:
                        progress(f"Automatic lambda round {state.round_id}: score inner fold "
                                 f"{fold.inner_fold_id}.")
                    results.append(score_auto_fold(
                        config, state.pending, state.complete, fold, maxima, state.round_id,
                        fit_path=fit_path, metric=metric, diagnostic=diagnostic))
                    del fold
            except (GlmnetError, OSError, MemoryError) as error:
                if state.round_id == 0:
                    raise
                state.fail_extension(error)
                break
            state.commit_round(results)
    warnings.extend(state.warnings)
    return state.result()
