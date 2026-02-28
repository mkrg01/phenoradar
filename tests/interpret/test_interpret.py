from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import phenoradar.interpret as interpret_mod
from phenoradar.interpret import (
    InterpretationError,
    ModelFeatureEntry,
    build_interpretation_tables,
)


def _fit_linear_model() -> LogisticRegression:
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 0], dtype=int)
    model = LogisticRegression(solver="liblinear", random_state=0)
    model.fit(x, y)
    return model


def test_build_interpretation_tables_requires_non_empty_entries() -> None:
    with pytest.raises(InterpretationError, match="No fitted models"):
        build_interpretation_tables([])


def test_build_interpretation_tables_for_linear_model_outputs_coefficients() -> None:
    model = _fit_linear_model()
    entries = [ModelFeatureEntry(feature_names=["OG2", "OG1"], model=model)]

    artifacts = build_interpretation_tables(entries)

    assert artifacts.feature_importance.height == 2
    assert set(artifacts.feature_importance.select("method").to_series().to_list()) == {
        "coef_abs_l1_norm"
    }
    assert set(artifacts.coefficients.select("method").to_series().to_list()) == {"coef_signed"}
    assert set(artifacts.coefficients.select("reason").to_series().to_list()) == {"NA"}


def test_build_interpretation_tables_for_random_forest_marks_coefficients_unsupported() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 0], dtype=int)
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(x, y)
    entries = [ModelFeatureEntry(feature_names=["OG1", "OG2"], model=model)]

    artifacts = build_interpretation_tables(entries)

    assert set(artifacts.feature_importance.select("method").to_series().to_list()) == {
        "feature_importances_l1_norm"
    }
    assert set(artifacts.coefficients.select("reason").to_series().to_list()) == {
        "unsupported_model_non_linear"
    }


def test_build_interpretation_tables_rejects_mixed_model_families() -> None:
    linear_model = _fit_linear_model()
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 0], dtype=int)
    forest_model = RandomForestClassifier(n_estimators=5, random_state=0)
    forest_model.fit(x, y)

    with pytest.raises(InterpretationError, match="Mixed model families"):
        build_interpretation_tables(
            [
                ModelFeatureEntry(feature_names=["OG1", "OG2"], model=linear_model),
                ModelFeatureEntry(feature_names=["OG1", "OG2"], model=forest_model),
            ]
        )


def test_build_interpretation_tables_warns_when_raw_importance_sum_is_zero() -> None:
    model = _fit_linear_model()
    model.coef_ = np.zeros_like(model.coef_)
    artifacts = build_interpretation_tables(
        [ModelFeatureEntry(feature_names=["OG1", "OG2"], model=model)]
    )

    assert any("zero total raw importance" in warning for warning in artifacts.warnings)


def test_build_interpretation_tables_rejects_importance_width_mismatch() -> None:
    model = _fit_linear_model()
    model.coef_ = np.array([[0.1, 0.2, 0.3]], dtype=float)

    with pytest.raises(InterpretationError, match="Importance vector width"):
        build_interpretation_tables([ModelFeatureEntry(feature_names=["OG1", "OG2"], model=model)])


def test_build_interpretation_tables_rejects_entries_with_no_features() -> None:
    model = _fit_linear_model()

    with pytest.raises(InterpretationError, match="No features were available"):
        build_interpretation_tables([ModelFeatureEntry(feature_names=[], model=model)])


def test_raw_importance_rejects_unsupported_model_type() -> None:
    with pytest.raises(InterpretationError, match="Unsupported fitted model"):
        interpret_mod._raw_importance(object())  # type: ignore[arg-type]


def test_linear_coefficients_returns_none_for_invalid_logistic_coef_shape() -> None:
    model = _fit_linear_model()
    model.coef_ = np.array([0.1, 0.2], dtype=float)

    assert interpret_mod._linear_coefficients(model) is None


def test_linear_coefficients_returns_none_for_unfitted_calibrated_model() -> None:
    model = CalibratedClassifierCV(estimator=LinearSVC(), method="sigmoid", cv=2)

    assert interpret_mod._linear_coefficients(model) is None


def test_linear_coefficients_returns_none_when_calibrated_estimator_has_no_coef() -> None:
    class _Wrapper:
        estimator = object()

    model = CalibratedClassifierCV(estimator=LinearSVC(), method="sigmoid", cv=2)
    model.calibrated_classifiers_ = [_Wrapper()]

    assert interpret_mod._linear_coefficients(model) is None


def test_linear_coefficients_returns_none_when_calibrated_coef_shape_is_invalid() -> None:
    class _EstimatorWithBadCoef:
        coef_ = np.array([0.1, 0.2], dtype=float)

    class _Wrapper:
        estimator = _EstimatorWithBadCoef()

    model = CalibratedClassifierCV(estimator=LinearSVC(), method="sigmoid", cv=2)
    model.calibrated_classifiers_ = [_Wrapper()]

    assert interpret_mod._linear_coefficients(model) is None


def test_build_interpretation_tables_rejects_coefficient_width_mismatch_via_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fit_linear_model()
    calls = {"n": 0}

    def _linear_coefficients_side_effect(_model: object) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] == 1:
            return np.array([0.1, 0.2], dtype=float)
        return np.array([0.1, 0.2, 0.3], dtype=float)

    monkeypatch.setattr(interpret_mod, "_linear_coefficients", _linear_coefficients_side_effect)

    with pytest.raises(InterpretationError, match="Coefficient vector width"):
        build_interpretation_tables([ModelFeatureEntry(feature_names=["OG1", "OG2"], model=model)])
