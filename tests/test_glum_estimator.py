"""Check the native binomial backend against its weighted logistic objective."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from glum import GeneralizedLinearRegressor
from scipy.special import expit

from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.cv import CVError, _build_estimator, _fit_estimator, _predict_positive_probability
from phenoradar.model_selection import generate_candidates


def _weighted_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(841)
    x = rng.normal(size=(60, 5))
    probability = expit(0.7 + x @ np.array([1.2, -0.8, 0.0, 0.0, 0.4]))
    y = (rng.uniform(size=x.shape[0]) < probability).astype(int)
    weights = rng.integers(1, 5, size=x.shape[0]).astype(float)
    return x, y, weights


def _estimator(y: np.ndarray, *, alpha: float, l1_ratio: float) -> GeneralizedLinearRegressor:
    estimator = _build_estimator(
        load_and_resolve_config([], allow_empty=True),
        model_seed=42,
        y_train=y,
        model_params={
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": 100,
            "gradient_tol": 1e-9,
        },
    )
    assert isinstance(estimator, GeneralizedLinearRegressor)
    return estimator


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_binomial_fit_satisfies_weighted_mean_loss_optimality(l1_ratio: float) -> None:
    x, y, weights = _weighted_problem()
    alpha = 0.03
    estimator = _estimator(y, alpha=alpha, l1_ratio=l1_ratio)

    diagnostic = _fit_estimator(estimator, x, y, weights)

    assert diagnostic.converged is True
    assert estimator.coef_.shape == (x.shape[1],)
    probability = expit(x @ estimator.coef_ + estimator.intercept_)
    residual = weights * (probability - y) / weights.sum()
    smooth_gradient = x.T @ residual + alpha * (1 - l1_ratio) * estimator.coef_
    active = np.abs(estimator.coef_) > 1e-10
    l1_threshold = alpha * l1_ratio
    np.testing.assert_allclose(
        smooth_gradient[active] + l1_threshold * np.sign(estimator.coef_[active]),
        0.0,
        atol=1e-7,
    )
    assert np.all(np.abs(smooth_gradient[~active]) <= l1_threshold + 1e-7)
    # The intercept remains unpenalized, including with unequal sample weights.
    assert abs(residual.sum()) < 1e-7
    np.testing.assert_allclose(_predict_positive_probability(estimator, x), probability, atol=1e-12)


def test_alpha_has_same_mean_loss_semantics_when_weights_or_rows_are_rescaled() -> None:
    x, y, weights = _weighted_problem()
    baseline = _estimator(y, alpha=0.03, l1_ratio=0.5)
    rescaled = _estimator(y, alpha=0.03, l1_ratio=0.5)
    replicated = _estimator(y, alpha=0.03, l1_ratio=0.5)
    _fit_estimator(baseline, x, y, weights)
    _fit_estimator(rescaled, x, y, weights * 7)
    _fit_estimator(
        replicated,
        np.repeat(x, weights.astype(int), axis=0),
        np.repeat(y, weights.astype(int)),
        sample_weight=None,
    )

    for estimator in (rescaled, replicated):
        np.testing.assert_allclose(estimator.coef_, baseline.coef_, atol=1e-7)
        assert estimator.intercept_ == pytest.approx(baseline.intercept_, abs=1e-7)
        np.testing.assert_allclose(
            _predict_positive_probability(estimator, x),
            _predict_positive_probability(baseline, x),
            atol=1e-8,
        )


def test_larger_alpha_increases_l1_regularization_without_penalizing_intercept() -> None:
    x, y, weights = _weighted_problem()
    weak = _estimator(y, alpha=0.01, l1_ratio=1.0)
    strong = _estimator(y, alpha=10.0, l1_ratio=1.0)
    _fit_estimator(weak, x, y, weights)
    _fit_estimator(strong, x, y, weights)

    assert np.linalg.norm(weak.coef_, ord=1) > 0
    np.testing.assert_array_equal(strong.coef_, np.zeros(x.shape[1]))
    np.testing.assert_allclose(
        _predict_positive_probability(strong, x), np.average(y, weights=weights), atol=1e-8
    )


@pytest.mark.parametrize("name", ["alpha", "l1_ratio", "gradient_tol", "max_iter"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, False, "1", None])
def test_glum_rejects_nonfinite_or_nonnumeric_search_parameters(name: str, value: Any) -> None:
    with pytest.raises(CVError, match=rf"search_space\.{name}.*must be"):
        _build_estimator(
            AppConfig(), model_seed=42, y_train=np.array([0, 1]), model_params={name: value}
        )


@pytest.mark.parametrize(
    "name, value",
    [
        ("alpha", -0.1),
        ("l1_ratio", -0.1),
        ("l1_ratio", 1.1),
        ("gradient_tol", 0),
        ("gradient_tol", -1e-6),
        ("max_iter", 0),
        ("max_iter", -1),
        ("max_iter", 1.9),
    ],
)
def test_glum_rejects_out_of_range_search_parameters(name: str, value: float) -> None:
    with pytest.raises(CVError, match=rf"search_space\.{name}.*must be"):
        _build_estimator(
            AppConfig(), model_seed=42, y_train=np.array([0, 1]), model_params={name: value}
        )


@pytest.mark.parametrize("l1_ratio", [0, 1])
def test_glum_accepts_unregularized_alpha_and_elastic_net_boundaries(l1_ratio: int) -> None:
    estimator = _build_estimator(
        AppConfig(),
        model_seed=42,
        y_train=np.array([0, 1]),
        model_params={"alpha": np.float64(0), "l1_ratio": l1_ratio, "max_iter": np.int64(1)},
    )

    assert isinstance(estimator, GeneralizedLinearRegressor)
    assert estimator.alpha == 0
    assert estimator.l1_ratio == l1_ratio
    assert estimator.max_iter == 1


def test_glum_accepts_integral_max_iter_values_generated_by_float_ranges() -> None:
    config = AppConfig.model_validate(
        {
            "model_selection": {
                "search_space": {
                    "max_iter": {"type": "range", "start": 10, "end": 30, "step": 10}
                }
            }
        }
    )
    candidates = generate_candidates(
        config=config, training_scope_id="validation", source_sample_set_id=0, warnings=[]
    )

    assert [candidate.params["max_iter"] for candidate in candidates] == [10.0, 20.0]
    for candidate in candidates:
        estimator = _build_estimator(
            config, model_seed=42, y_train=np.array([0, 1]), model_params=candidate.params
        )
        assert estimator.max_iter == candidate.params["max_iter"]
        assert isinstance(estimator.max_iter, int)


def test_binomial_convergence_on_last_allowed_iteration_is_not_a_failure() -> None:
    x, y, weights = _weighted_problem()
    reference = _estimator(y, alpha=0.03, l1_ratio=0.5)
    assert _fit_estimator(reference, x, y, weights).converged is True
    assert reference.n_iter_ > 0
    limited = _estimator(y, alpha=0.03, l1_ratio=0.5)
    limited.set_params(max_iter=int(reference.n_iter_))

    diagnostic = _fit_estimator(limited, x, y, weights)

    assert diagnostic.n_iter_values == (limited.max_iter,)
    assert diagnostic.converged is True
    assert diagnostic.convergence_warning_count == 0
    assert diagnostic.convergence_warning_messages == ()
    np.testing.assert_allclose(limited.coef_, reference.coef_, atol=1e-10)
