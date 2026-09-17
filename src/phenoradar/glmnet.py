"""Binary glmnet paths with exact lambda selection and portable fitted models.

Only fitting calls the native core. Prediction and serialization use ordinary
NumPy coefficients, so no R runtime, subprocess, or lambda interpolation is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from glmnet._glmnet import lognet, lsolns, splognet
from scipy import sparse
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

DEFAULT_LAMBDA = 0.01
DEFAULT_ALPHA = 0.5
DEFAULT_THRESH = 1e-14
DEFAULT_MAXIT = 2_000_000

# Nearby solutions keep warm starts and the native strong screening rule useful.
# A half-decade candidate grid is expanded to ten intervals between candidates.
_MAX_LOG_LAMBDA_STEP = 0.05 * np.log(10.0)


class GlmnetError(ValueError):
    """Raised when glmnet cannot return the complete requested path."""


def _native_lambda_path(targets: np.ndarray) -> np.ndarray:
    """Bridge descending positive targets without rounding or inventing targets.

    Zero stays an exact unregularized endpoint; no logarithmic bridge to zero
    is defined. Single targets are also left alone.
    """
    values: list[float] = []
    for high, low in zip(targets[:-1], targets[1:], strict=True):
        values.append(float(high))
        if low == 0:
            continue
        log_high, log_low = np.log(high), np.log(low)
        # Avoid an extra interval caused solely by roundoff at an exact step.
        intervals = max(1, int(np.ceil((log_high - log_low) / _MAX_LOG_LAMBDA_STEP - 1e-12)))
        interior = np.exp(np.linspace(log_high, log_low, intervals + 1)[1:-1])
        values.extend(float(value) for value in interior if low < value < high)
    values.append(float(targets[-1]))
    # Subnormal floats can round multiple interior points to the same value.
    return np.unique(values)[::-1].copy()


class GlmnetLogisticRegression(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """One binary elastic-net model, selected exactly from a glmnet path.

    ``lambda_`` is regularization strength; ``alpha`` is the L1 fraction.
    Input features are already preprocessed: internal standardization is disabled.
    ``n_iter_`` records coordinate passes including internal warm-start points.
    ``path_length_`` counts distinct requested lambdas; ``native_path_length_``
    counts native fits, including those intermediate points (zero for a
    constant design that needs no native fit).
    """

    def __init__(
        self,
        *,
        lambda_: float = DEFAULT_LAMBDA,
        alpha: float = DEFAULT_ALPHA,
        thresh: float = DEFAULT_THRESH,
        maxit: int = DEFAULT_MAXIT,
    ) -> None:
        self.lambda_ = lambda_
        self.alpha = alpha
        self.thresh = thresh
        self.maxit = maxit

    def fit(
        self, X: Any, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> GlmnetLogisticRegression:
        model = fit_glmnet_path(
            X,
            y,
            [self.lambda_],
            alpha=self.alpha,
            thresh=self.thresh,
            maxit=self.maxit,
            sample_weight=sample_weight,
        )[0]
        self.__dict__.update(model.__dict__)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_", "n_features_in_"])
        x = check_array(X, accept_sparse="csc", dtype=np.float64)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}")
        return np.asarray(x @ self.coef_ + self.intercept_, dtype=float).reshape(-1)

    def predict_proba(self, X: Any) -> np.ndarray:
        positive = expit(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.decision_function(X) >= 0, dtype=int)


def fit_glmnet_path(
    x: Any,
    y: np.ndarray,
    lambdas: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    thresh: float = DEFAULT_THRESH,
    maxit: int = DEFAULT_MAXIT,
    sample_weight: np.ndarray | None = None,
) -> list[GlmnetLogisticRegression]:
    """Fit one descending path; return independent models in requested order.

    The low-level entry points avoid the Python wrapper's interpolation and
    first-lambda extrapolation, and preserve the native convergence status.
    Explicit lambdas disable automatic path truncation in the glmnet core.
    Intermediate positive lambdas improve warm starts on coarse candidate grids;
    only exact requested models are extracted, scored, or returned.
    """
    requested = np.asarray(lambdas, dtype=float)
    if (
        requested.ndim != 1
        or requested.size == 0
        or not np.isfinite(requested).all()
        or np.any(requested < 0)
    ):
        raise GlmnetError("lambda must contain finite, nonnegative values")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise GlmnetError("alpha must be a finite number in [0, 1]")
    if not np.isfinite(thresh) or thresh <= 0:
        raise GlmnetError("thresh must be a finite number > 0")
    if isinstance(maxit, bool) or not isinstance(maxit, (int, np.integer)) or maxit < 1:
        raise GlmnetError("maxit must be a positive integer")
    x, y = check_X_y(x, y, accept_sparse="csc", dtype=np.float64, ensure_min_samples=2)
    if not np.array_equal(np.unique(y), [0, 1]):
        raise GlmnetError("glmnet requires both binary labels 0 and 1")
    weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if (
        weights.shape != y.shape
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise GlmnetError("sample_weight must be finite, nonnegative, and have positive total")
    positive = weights > 0
    x, y, weights = x[positive], y[positive], weights[positive]
    if not np.array_equal(np.unique(y), [0, 1]):
        raise GlmnetError("Both classes must have positive sample weight")
    # Normalizing first prevents overflow when forming the weighted responses.
    weights = weights / weights.max()
    weights /= weights.sum()
    n, p = x.shape
    path = np.unique(requested)[::-1].copy()
    if sparse.issparse(x):
        x = sparse.csc_matrix(x, copy=True)
        x.sum_duplicates()
        x.sort_indices()
        variable = np.any(x.max(axis=0).toarray() != x.min(axis=0).toarray())
    else:
        variable = np.any(np.ptp(x, axis=0) > 0)

    if not variable:
        # The native solver rejects all-constant designs; the exact optimum is
        # an unpenalized intercept with all coefficients zero.
        prevalence = float(weights @ y)
        intercepts = np.full(len(path), np.log(prevalence / (1 - prevalence)))
        coefficients = np.zeros((p, len(path)))
        passes = 0
        native_path_length = 0
    else:
        native_path = _native_lambda_path(path)
        native_path_length = len(native_path)
        response = np.array(np.column_stack((y, 1 - y)) * weights[:, None], order="F")
        offset = np.zeros((n, 2), order="F")
        excluded = np.array([0], dtype=np.int32)
        penalties = np.ones(p)
        bounds = np.array([np.full(p, -np.inf), np.full(p, np.inf)], order="F")
        options = dict(ne=p + 1, nlam=native_path_length, isd=0, intr=1, maxit=int(maxit), kopt=0)
        if sparse.issparse(x):
            result = splognet(
                alpha,
                n,
                p,
                1,
                x.data.copy(),
                x.indptr + 1,
                x.indices + 1,
                response,
                offset,
                excluded,
                penalties,
                bounds,
                p + 1,
                p,
                1.0,
                native_path,
                thresh,
                nlam=native_path_length,
                isd=0,
                intr=1,
                maxit=int(maxit),
                kopt=0,
            )
        else:
            result = lognet(
                alpha,
                1,
                np.array(x, dtype=float, order="F", copy=True),
                response,
                offset,
                excluded,
                penalties,
                bounds,
                p,
                1.0,
                native_path,
                thresh,
                **options,
            )
        count, intercepts_raw, compressed, indices, sizes, _, _, returned, passes, error = result
        if error != 0 or count != native_path_length:
            detail = "Increase maxit." if error < 0 else "Check the training data and weights."
            raise GlmnetError(
                f"glmnet did not complete the requested lambda path "
                f"({count}/{native_path_length} native models, including warm-start points; "
                f"error={error}). {detail}"
            )
        if not np.allclose(returned[:count], native_path, rtol=1e-12, atol=0):
            raise GlmnetError("glmnet returned unexpected lambda values")
        target_indices = np.searchsorted(-native_path, -path)
        # Discard auxiliary solutions before expanding compressed coefficients.
        coefficients = np.asarray(
            lsolns(p, compressed[:, :, target_indices], indices, sizes[target_indices])[:, 0, :]
        )
        intercepts = np.asarray(intercepts_raw[0, target_indices])
    if not np.isfinite(coefficients).all() or not np.isfinite(intercepts).all():
        raise GlmnetError("glmnet returned nonfinite coefficients")

    models = []
    for strength in requested:
        index = int(np.flatnonzero(path == strength)[0])
        model = GlmnetLogisticRegression(
            lambda_=float(strength), alpha=alpha, thresh=thresh, maxit=int(maxit)
        )
        model.coef_ = coefficients[:, index].copy()
        model.intercept_ = float(intercepts[index])
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = p
        model.n_iter_ = int(passes)
        model.path_length_ = len(path)
        model.native_path_length_ = native_path_length
        model.converged_ = True
        residual = weights * (expit(x @ model.coef_ + model.intercept_) - y)
        gradient = np.asarray(x.T @ residual).reshape(-1) + strength * (1 - alpha) * model.coef_
        active = model.coef_ != 0
        subgradient = np.maximum(np.abs(gradient) - strength * alpha, 0)
        subgradient[active] = np.abs(
            gradient[active] + strength * alpha * np.sign(model.coef_[active])
        )
        model.kkt_residual_ = float(subgradient.sum() + abs(residual.sum()))
        models.append(model)
    return models
