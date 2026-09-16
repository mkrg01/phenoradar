"""Check the native binomial backend against its weighted logistic objective."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.cv import CVError, _build_estimator, _fit_estimator, _predict_positive_probability
from phenoradar.glmnet import GlmnetError, GlmnetLogisticRegression, fit_glmnet_path
from phenoradar.model_selection import generate_candidates


def _weighted_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(841)
    x = rng.normal(size=(60, 5))
    probability = expit(0.7 + x @ np.array([1.2, -0.8, 0.0, 0.0, 0.4]))
    y = (rng.uniform(size=x.shape[0]) < probability).astype(int)
    weights = rng.integers(1, 5, size=x.shape[0]).astype(float)
    return x, y, weights


def _estimator(y: np.ndarray, *, lambda_: float, alpha: float) -> GlmnetLogisticRegression:
    estimator = _build_estimator(
        load_and_resolve_config([], allow_empty=True),
        model_seed=42,
        y_train=y,
        model_params={
            "lambda": lambda_,
            "alpha": alpha,
            "maxit": 2000000,
            "thresh": 1e-14,
        },
    )
    assert isinstance(estimator, GlmnetLogisticRegression)
    return estimator


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_binomial_fit_satisfies_weighted_mean_loss_optimality(alpha: float) -> None:
    x, y, weights = _weighted_problem()
    lambda_ = 0.03
    estimator = _estimator(y, lambda_=lambda_, alpha=alpha)

    diagnostic = _fit_estimator(estimator, x, y, weights)

    assert diagnostic.converged is True
    assert estimator.coef_.shape == (x.shape[1],)
    probability = expit(x @ estimator.coef_ + estimator.intercept_)
    residual = weights * (probability - y) / weights.sum()
    smooth_gradient = x.T @ residual + lambda_ * (1 - alpha) * estimator.coef_
    active = np.abs(estimator.coef_) > 1e-10
    l1_threshold = lambda_ * alpha
    np.testing.assert_allclose(
        smooth_gradient[active] + l1_threshold * np.sign(estimator.coef_[active]),
        0.0,
        atol=1e-7,
    )
    assert np.all(np.abs(smooth_gradient[~active]) <= l1_threshold + 1e-7)
    # The intercept remains unpenalized, including with unequal sample weights.
    assert abs(residual.sum()) < 1e-7
    np.testing.assert_allclose(_predict_positive_probability(estimator, x), probability, atol=1e-12)


def test_lambda_has_same_mean_loss_semantics_when_weights_or_rows_are_rescaled() -> None:
    x, y, weights = _weighted_problem()
    baseline = _estimator(y, lambda_=0.03, alpha=0.5)
    rescaled = _estimator(y, lambda_=0.03, alpha=0.5)
    replicated = _estimator(y, lambda_=0.03, alpha=0.5)
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


def test_larger_lambda_increases_l1_regularization_without_penalizing_intercept() -> None:
    x, y, weights = _weighted_problem()
    weak = _estimator(y, lambda_=0.01, alpha=1.0)
    strong = _estimator(y, lambda_=10.0, alpha=1.0)
    _fit_estimator(weak, x, y, weights)
    _fit_estimator(strong, x, y, weights)

    assert np.linalg.norm(weak.coef_, ord=1) > 0
    np.testing.assert_array_equal(strong.coef_, np.zeros(x.shape[1]))
    np.testing.assert_allclose(
        _predict_positive_probability(strong, x), np.average(y, weights=weights), atol=1e-8
    )


@pytest.mark.parametrize("name", ["lambda", "alpha", "thresh", "maxit"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, False, "1", None])
def test_glmnet_rejects_nonfinite_or_nonnumeric_search_parameters(name: str, value: Any) -> None:
    with pytest.raises(CVError, match=rf"search_space\.{name}.*must be"):
        _build_estimator(
            AppConfig(), model_seed=42, y_train=np.array([0, 1]), model_params={name: value}
        )


@pytest.mark.parametrize(
    "name, value",
    [
        ("lambda", -0.1),
        ("alpha", -0.1),
        ("alpha", 1.1),
        ("thresh", 0),
        ("thresh", -1e-6),
        ("maxit", 0),
        ("maxit", -1),
        ("maxit", 1.9),
    ],
)
def test_glmnet_rejects_out_of_range_search_parameters(name: str, value: float) -> None:
    with pytest.raises(CVError, match=rf"search_space\.{name}.*must be"):
        _build_estimator(
            AppConfig(), model_seed=42, y_train=np.array([0, 1]), model_params={name: value}
        )


@pytest.mark.parametrize("alpha", [0, 1])
def test_glmnet_accepts_unregularized_lambda_and_elastic_net_boundaries(alpha: int) -> None:
    estimator = _build_estimator(
        AppConfig(),
        model_seed=42,
        y_train=np.array([0, 1]),
        model_params={"lambda": np.float64(0), "alpha": alpha, "maxit": np.int64(1)},
    )

    assert isinstance(estimator, GlmnetLogisticRegression)
    assert estimator.lambda_ == 0
    assert estimator.alpha == alpha
    assert estimator.maxit == 1


def test_glmnet_accepts_integral_maxit_values_generated_by_float_ranges() -> None:
    config = AppConfig.model_validate(
        {
            "model_selection": {
                "search_space": {"maxit": {"type": "range", "start": 10, "end": 30, "step": 10}}
            }
        }
    )
    candidates = generate_candidates(
        config=config, training_scope_id="validation", source_sample_set_id=0, warnings=[]
    )

    assert [candidate.params["maxit"] for candidate in candidates] == [10.0, 20.0]
    for candidate in candidates:
        estimator = _build_estimator(
            config, model_seed=42, y_train=np.array([0, 1]), model_params=candidate.params
        )
        assert estimator.maxit == candidate.params["maxit"]
        assert isinstance(estimator.maxit, int)


def test_binomial_convergence_on_last_allowed_iteration_is_not_a_failure() -> None:
    x, y, weights = _weighted_problem()
    reference = _estimator(y, lambda_=0.03, alpha=0.5)
    assert _fit_estimator(reference, x, y, weights).converged is True
    assert reference.n_iter_ > 0
    limited = _estimator(y, lambda_=0.03, alpha=0.5)
    limited.set_params(maxit=int(reference.n_iter_))

    diagnostic = _fit_estimator(limited, x, y, weights)

    assert diagnostic.n_iter_values == (limited.maxit,)
    assert diagnostic.converged is True
    assert diagnostic.convergence_warning_count == 0
    assert diagnostic.convergence_warning_messages == ()
    np.testing.assert_allclose(limited.coef_, reference.coef_, atol=1e-10)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_explicit_irregular_path_matches_independent_fits_without_mutating_inputs(
    sparse_input: bool,
) -> None:
    from scipy.sparse import csc_matrix

    x, y, weights = _weighted_problem()
    x[np.abs(x) < 0.6] = 0
    matrix = csc_matrix(x) if sparse_input else x.copy()
    original_weights = weights.copy()
    # Neither geometric spacing nor request order may change the fitted lambda.
    lambdas = [0.017, 0.1, 0.043, 0.017]
    models = fit_glmnet_path(matrix, y, lambdas, alpha=0.7, sample_weight=weights)
    assert [model.lambda_ for model in models] == lambdas
    assert len({id(model) for model in models}) == len(lambdas)
    assert all(model.path_length_ == 3 for model in models)
    assert len({model.n_iter_ for model in models}) == 1
    np.testing.assert_array_equal(matrix.toarray() if sparse_input else matrix, x)
    np.testing.assert_array_equal(weights, original_weights)
    for model in models:
        independent = GlmnetLogisticRegression(lambda_=model.lambda_, alpha=0.7).fit(
            x, y, sample_weight=weights
        )
        np.testing.assert_allclose(
            model.predict_proba(matrix), independent.predict_proba(x), atol=2e-7
        )
        np.testing.assert_allclose(model.coef_, independent.coef_, atol=1e-6)
        assert model.kkt_residual_ < 1e-6
    models[0].coef_[:] = 123
    assert not np.array_equal(models[0].coef_, models[-1].coef_)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_all_constant_design_returns_weighted_intercept(sparse_input: bool) -> None:
    from scipy.sparse import csc_matrix

    x = np.tile([0.0, 3.0], (6, 1))
    matrix = csc_matrix(x) if sparse_input else x
    y = np.array([0, 0, 0, 1, 1, 1])
    weights = np.array([1, 1, 1, 2, 2, 2])
    models = fit_glmnet_path(matrix, y, [0, 0.01, 10], sample_weight=weights)
    for model in models:
        np.testing.assert_array_equal(model.coef_, [0, 0])
        np.testing.assert_allclose(model.predict_proba(matrix)[:, 1], 2 / 3)
        assert model.n_iter_ == 0


def test_zero_weight_rows_are_excluded_from_native_fit() -> None:
    x, y, weights = _weighted_problem()
    weights[:10] = 0
    model = GlmnetLogisticRegression().fit(x, y, sample_weight=weights)
    reference = GlmnetLogisticRegression().fit(x[10:], y[10:], sample_weight=weights[10:])
    np.testing.assert_allclose(model.predict_proba(x), reference.predict_proba(x), atol=1e-12)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [1, np.inf], [1, np.nan], [1]])
def test_invalid_weights_are_rejected(weights: list[float]) -> None:
    with pytest.raises(GlmnetError, match="sample_weight"):
        fit_glmnet_path(
            np.array([[0], [1]]), np.array([0, 1]), [0.01], sample_weight=np.array(weights)
        )


def test_incomplete_native_path_is_never_returned() -> None:
    x, y, weights = _weighted_problem()
    with pytest.raises(GlmnetError, match="did not complete.*Increase maxit"):
        fit_glmnet_path(x, y, [0.1, 0.01, 0.001], maxit=1, sample_weight=weights)


def test_both_classes_must_have_positive_weight() -> None:
    with pytest.raises(GlmnetError, match="Both classes"):
        fit_glmnet_path(
            np.array([[0], [1]]), np.array([0, 1]), [0.01], sample_weight=np.array([0, 1])
        )
