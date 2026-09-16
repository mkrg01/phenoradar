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


class GlmnetError(ValueError):
    """Raised when glmnet cannot return the complete requested path."""


class GlmnetLogisticRegression(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """One binary elastic-net model, selected exactly from a glmnet path.

    ``lambda_`` is regularization strength; ``alpha`` is the L1 fraction.
    Input features are already preprocessed: internal standardization is disabled.
    ``n_iter_`` records coordinate passes for the entire originating path.
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
    else:
        response = np.array(np.column_stack((y, 1 - y)) * weights[:, None], order="F")
        offset = np.zeros((n, 2), order="F")
        excluded = np.array([0], dtype=np.int32)
        penalties = np.ones(p)
        bounds = np.array([np.full(p, -np.inf), np.full(p, np.inf)], order="F")
        options = dict(ne=p + 1, nlam=len(path), isd=0, intr=1, maxit=int(maxit), kopt=0)
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
                path,
                thresh,
                nlam=len(path),
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
                path,
                thresh,
                **options,
            )
        count, intercepts_raw, compressed, indices, sizes, _, _, returned, passes, error = result
        if error != 0 or count != len(path):
            detail = "Increase maxit." if error < 0 else "Check the training data and weights."
            raise GlmnetError(
                f"glmnet did not complete the requested lambda path "
                f"({count}/{len(path)} models; error={error}). {detail}"
            )
        if not np.allclose(returned[:count], path, rtol=1e-12, atol=0):
            raise GlmnetError("glmnet returned unexpected lambda values")
        coefficients = np.asarray(lsolns(p, compressed, indices, sizes)[:, 0, :])
        intercepts = np.asarray(intercepts_raw[0])
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
