"""Selected models warm start within their own training data and parameter path."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import phenoradar.cv as cv
from phenoradar.config import AppConfig
from phenoradar.glmnet import (
    DEFAULT_ALPHA,
    DEFAULT_MAXIT,
    DEFAULT_THRESH,
    GlmnetError,
    GlmnetLogisticRegression,
    fit_glmnet_path,
)
from phenoradar.model_selection import Candidate


def _source(
    selected: list[dict[str, Any]], trials: list[dict[str, Any]] | None = None
) -> cv.SourceSelectionResult:
    trial_params = [] if trials is None else trials
    return cv.SourceSelectionResult(
        selected_candidates=[
            cv.SelectedCandidate(
                candidate=Candidate(candidate_index=index, params=params), score=0.5
            )
            for index, params in enumerate(selected)
        ],
        n_available_candidates=max(len(selected), len(trial_params)),
        n_scored_candidates=len(trial_params),
        selected_candidate_count_requested=len(selected),
        selected_candidate_count_effective=len(selected),
        trial_rows=[
            {
                "candidate_index": index,
                "inner_fold_id": str(fold),
                "params_json": json.dumps(params),
            }
            for index, params in enumerate(trial_params)
            for fold in range(2)
        ],
    )


def test_selected_path_uses_only_stronger_matching_candidates_and_exact_target() -> None:
    estimator = GlmnetLogisticRegression(lambda_=0.017, alpha=0.7, thresh=1e-12, maxit=123456)
    matching = {"alpha": 0.7, "thresh": 1e-12, "maxit": 123456}
    source = _source(
        [{**matching, "lambda": 0.017}, {**matching, "lambda": 0.2}],
        [
            {**matching, "lambda": 0.043},
            {**matching, "lambda": 0.1},
            {**matching, "lambda": 0.043},
            {**matching, "lambda": 0.001},
            {**matching, "lambda": 0.9, "alpha": 0.8},
            {**matching, "lambda": 0.8, "thresh": 1e-10},
            {**matching, "lambda": 0.7, "maxit": 123457},
        ],
    )

    assert cv._selected_glmnet_lambda_path(estimator, source) == [0.2, 0.1, 0.043, 0.017]


def test_selected_path_resolves_omitted_defaults_like_the_estimator() -> None:
    estimator = GlmnetLogisticRegression(lambda_=0.003)
    source = _source(
        [{"lambda": 0.003}],
        [
            {"lambda": 0.1},
            {
                "lambda": 0.03,
                "alpha": DEFAULT_ALPHA,
                "thresh": DEFAULT_THRESH,
                "maxit": float(DEFAULT_MAXIT),
            },
            {},  # The default lambda is also part of the candidate space.
        ],
    )

    assert cv._selected_glmnet_lambda_path(estimator, source) == [0.1, 0.03, 0.01, 0.003]


@pytest.mark.parametrize("lambda_", [0.0, 0.017, 0.2])
def test_selected_candidates_define_path_when_no_inner_cv_was_needed(lambda_: float) -> None:
    source = _source([{"lambda": 0.2}, {"lambda": 0.017}, {"lambda": 0.0}])
    estimator = GlmnetLogisticRegression(lambda_=lambda_)

    assert cv._selected_glmnet_lambda_path(estimator, source) == [
        strength for strength in [0.2, 0.017, 0.0] if strength >= lambda_
    ]


@pytest.mark.parametrize("params", [{}, {"lambda": 0.0}, {"lambda": 0.23, "alpha": 0.9}])
def test_single_candidate_does_not_invent_additional_lambdas(params: dict[str, Any]) -> None:
    estimator = cv._build_estimator(
        AppConfig(), model_seed=42, y_train=np.array([0, 1]), model_params=params
    )
    assert isinstance(estimator, GlmnetLogisticRegression)

    assert cv._selected_glmnet_lambda_path(estimator, _source([params])) == [estimator.lambda_]


def _weighted_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(649)
    y = np.tile([0, 1], 40)
    x = rng.normal(size=(80, 6))
    x[:, 0] += y
    return x, y, rng.uniform(0.5, 3.0, size=len(y))


@pytest.mark.parametrize("weighted", [False, True])
def test_selected_fit_matches_exact_reference_path_and_preserves_inputs(weighted: bool) -> None:
    x, y, weights = _weighted_problem()
    sample_weight = weights if weighted else None
    originals = [array.copy() for array in (x, y, weights)]
    estimator = GlmnetLogisticRegression(lambda_=0.017, alpha=0.7)
    params_before = estimator.get_params()
    source = _source(
        [{"lambda": 0.017, "alpha": 0.7}],
        [{"lambda": strength, "alpha": 0.7} for strength in [0.001, 0.043, 0.2]],
    )
    reference = fit_glmnet_path(x, y, [0.2, 0.043, 0.017], alpha=0.7, sample_weight=sample_weight)[
        -1
    ]

    diagnostic = cv._fit_selected_estimator(estimator, x, y, sample_weight, source)

    assert estimator.get_params() == params_before
    assert estimator.path_length_ == 3
    assert diagnostic.converged is True
    assert diagnostic.n_iter_values == (reference.n_iter_,)
    assert estimator.kkt_residual_ < 1e-6
    np.testing.assert_array_equal(estimator.coef_, reference.coef_)
    assert estimator.intercept_ == reference.intercept_
    np.testing.assert_array_equal(estimator.predict_proba(x), reference.predict_proba(x))
    for array, original in zip((x, y, weights), originals, strict=True):
        np.testing.assert_array_equal(array, original)


def test_selected_path_failure_is_translated_without_returning_a_partial_model() -> None:
    x, y, weights = _weighted_problem()
    estimator = GlmnetLogisticRegression(lambda_=0.001, maxit=1)
    source = _source([{"lambda": 0.001, "maxit": 1}], [{"lambda": 0.2, "maxit": 1}])

    with pytest.raises(cv.CVError, match="did not complete.*Increase maxit") as error:
        cv._fit_selected_estimator(estimator, x, y, weights, source)

    assert isinstance(error.value.__cause__, GlmnetError)
    assert not hasattr(estimator, "coef_")


def test_non_glmnet_selected_fit_preserves_sample_weight_behavior() -> None:
    x, y, weights = _weighted_problem()
    estimator = RandomForestClassifier(n_estimators=5, random_state=19)
    reference = RandomForestClassifier(n_estimators=5, random_state=19)
    reference.fit(x, y, sample_weight=weights)

    diagnostic = cv._fit_selected_estimator(estimator, x, y, weights, _source([{}]))

    assert diagnostic.estimator_class == "RandomForestClassifier"
    np.testing.assert_array_equal(estimator.predict_proba(x), reference.predict_proba(x))


@pytest.mark.parametrize("scope", ["outer_fold", "final_refit"])
def test_sample_set_refits_rebuild_paths_using_own_preprocessing_and_weights(
    scope: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig.model_validate(
        {"runtime": {"n_jobs": 1}, "sampling": {"weighting": "group_label_inverse"}}
    )
    x, y, _ = _weighted_problem()
    x = np.exp(x)
    x[40:, 0] = 0.0  # Different sample sets retain different feature schemas.
    original_x = x.copy()
    groups = np.array([f"group_{index // 7}" for index in range(len(y))])
    feature_names = [f"OG{index}" for index in range(x.shape[1])]
    targets = x[[3, 43]].copy()
    source = _source(
        [{"lambda": 0.017, "alpha": 0.7}],
        [{"lambda": strength, "alpha": 0.7} for strength in [0.2, 0.043, 0.001]],
    )
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]] = []

    def tracked_fit(
        train: np.ndarray, labels: np.ndarray, lambdas: list[float], **kwargs: Any
    ) -> list[GlmnetLogisticRegression]:
        calls.append((train.copy(), labels.copy(), kwargs["sample_weight"].copy(), list(lambdas)))
        return fit_glmnet_path(train, labels, lambdas, **kwargs)

    monkeypatch.setattr(cv, "fit_glmnet_path", tracked_fit)
    models = []
    for sample_set_id, indices in enumerate((np.arange(40), np.arange(40, 80))):
        common = dict(
            config=config,
            sample_set_id=sample_set_id,
            sampled_idx=indices,
            source_result=source,
            selection_source_sample_set_id=0,
            base_model_index=sample_set_id,
            x_train_raw=x,
            y_train=y,
            groups_train=groups,
            feature_names=feature_names,
        )
        if scope == "outer_fold":
            result = cv._fit_outer_sample_set(
                **common,
                fold_id="1",
                contrast_groups_train=None,
                x_valid_raw=targets,
                valid_species=["sp_a", "sp_b"],
                x_inference_matrix=np.empty((0, len(feature_names))),
                inference_transform_applied=False,
            )
            estimator = result.fold_models[0]
        else:
            result = cv._fit_final_refit_sample_set(
                **common, x_target_raw=targets, target_count=len(targets)
            )
            estimator = result.fitted_models[0]
        models.append(estimator)
        expected_x, expected_features, scaler, _ = cv._preprocess_train_only_with_counts(
            config, x[indices], feature_names, y_train=y[indices], warnings=[]
        )
        expected_weight = cv._fit_sample_weights(config, y[indices], groups[indices])
        expected_target = cv._transform_target_for_selected_features(
            config, targets, feature_names, expected_features, scaler
        )
        reference = fit_glmnet_path(
            expected_x,
            y[indices],
            [0.2, 0.043, 0.017],
            alpha=0.7,
            sample_weight=expected_weight,
        )[-1]
        assert len(calls) == sample_set_id + 1
        fitted_x, fitted_y, fitted_weight, fitted_path = calls[-1]
        np.testing.assert_array_equal(fitted_x, expected_x)
        np.testing.assert_array_equal(fitted_y, y[indices])
        np.testing.assert_array_equal(fitted_weight, expected_weight)
        assert fitted_path == [0.2, 0.043, 0.017]
        assert result.selected_features == expected_features
        np.testing.assert_array_equal(estimator.coef_, reference.coef_)
        np.testing.assert_array_equal(
            result.model_probs[0], reference.predict_proba(expected_target)[:, 1]
        )
        row = result.convergence_rows[0]
        assert row["training_scope"] == scope
        assert row["selection_source_sample_set_id"] == 0
        assert json.loads(row["params_json"]) == {"lambda": 0.017, "alpha": 0.7}

    assert [model.n_features_in_ for model in models] == [6, 5]
    assert models[0] is not models[1]
    assert not np.shares_memory(models[0].coef_, models[1].coef_)
    np.testing.assert_array_equal(x, original_x)
