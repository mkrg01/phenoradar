"""Model interpretation table builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

_METHOD_COEF_ABS_L1 = "coef_abs_l1_norm"
_METHOD_IMPORTANCES_L1 = "feature_importances_l1_norm"


class InterpretationError(ValueError):
    """Raised when interpretation table construction fails."""


@dataclass(frozen=True)
class ModelFeatureEntry:
    """One fitted model with its fold-local selected feature schema."""

    feature_names: list[str]
    model: LogisticRegression | CalibratedClassifierCV | RandomForestClassifier


@dataclass(frozen=True)
class InterpretationArtifacts:
    """Interpretation tables generated from fitted fold models."""

    feature_importance: pl.DataFrame
    coefficients: pl.DataFrame
    warnings: list[str]


def _linear_coefficients(
    model: LogisticRegression | CalibratedClassifierCV | RandomForestClassifier,
) -> np.ndarray | None:
    if isinstance(model, LogisticRegression):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim != 2 or coef.shape[0] < 1:
            return None
        return np.asarray(coef[0], dtype=float)

    if isinstance(model, CalibratedClassifierCV):
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if not isinstance(calibrated, list) or not calibrated:
            return None
        coef_list: list[np.ndarray] = []
        for calibrated_model in calibrated:
            estimator = getattr(calibrated_model, "estimator", None)
            if estimator is None or not hasattr(estimator, "coef_"):
                return None
            coef = np.asarray(estimator.coef_, dtype=float)
            if coef.ndim != 2 or coef.shape[0] < 1:
                return None
            coef_list.append(np.asarray(coef[0], dtype=float))
        return np.asarray(np.mean(np.vstack(coef_list), axis=0), dtype=float)

    return None


def _raw_importance(
    model: LogisticRegression | CalibratedClassifierCV | RandomForestClassifier,
) -> np.ndarray:
    linear_coef = _linear_coefficients(model)
    if linear_coef is not None:
        return np.asarray(np.abs(linear_coef), dtype=float)
    if isinstance(model, RandomForestClassifier):
        return np.asarray(model.feature_importances_, dtype=float)
    raise InterpretationError("Unsupported fitted model for importance extraction")


def _importance_method(
    model: LogisticRegression | CalibratedClassifierCV | RandomForestClassifier,
) -> str:
    if isinstance(model, RandomForestClassifier):
        return _METHOD_IMPORTANCES_L1
    return _METHOD_COEF_ABS_L1


def build_interpretation_tables(entries: list[ModelFeatureEntry]) -> InterpretationArtifacts:
    """Build feature importance and coefficients tables."""
    if not entries:
        raise InterpretationError("No fitted models were provided for interpretation")

    all_features = sorted(
        {
            feature
            for entry in entries
            for feature in entry.feature_names
        }
    )
    if not all_features:
        raise InterpretationError("No features were available for interpretation")

    feature_index = {feature: idx for idx, feature in enumerate(all_features)}
    n_models = len(entries)
    importance_matrix = np.zeros((n_models, len(all_features)), dtype=float)
    coefficient_matrix = np.zeros((n_models, len(all_features)), dtype=float)
    coefficient_supported = True
    warnings: list[str] = []
    methods: set[str] = set()

    for model_idx, entry in enumerate(entries):
        raw_importance = _raw_importance(entry.model)
        if raw_importance.shape[0] != len(entry.feature_names):
            raise InterpretationError("Importance vector width does not match feature schema")
        methods.add(_importance_method(entry.model))

        norm_denominator = float(np.sum(raw_importance))
        if np.isclose(norm_denominator, 0.0):
            normalized_importance = np.zeros_like(raw_importance, dtype=float)
            warnings.append(
                f"Model index {model_idx} produced zero total raw importance; "
                "emitted zero normalized importances"
            )
        else:
            normalized_importance = np.asarray(raw_importance / norm_denominator, dtype=float)

        for feature_name, importance_value in zip(
            entry.feature_names,
            normalized_importance.tolist(),
            strict=True,
        ):
            importance_matrix[model_idx, feature_index[feature_name]] = float(importance_value)

        linear_coefficients = _linear_coefficients(entry.model)
        if linear_coefficients is None:
            coefficient_supported = False
            continue
        if linear_coefficients.shape[0] != len(entry.feature_names):
            raise InterpretationError("Coefficient vector width does not match feature schema")
        for feature_name, coefficient_value in zip(
            entry.feature_names,
            linear_coefficients.tolist(),
            strict=True,
        ):
            coefficient_matrix[model_idx, feature_index[feature_name]] = float(coefficient_value)

    if len(methods) != 1:
        raise InterpretationError(
            "Mixed model families are not supported in one interpretation set"
        )
    importance_method = next(iter(methods))

    importance_mean = np.mean(importance_matrix, axis=0)
    importance_std = np.std(importance_matrix, axis=0)
    feature_importance = pl.DataFrame(
        {
            "feature": all_features,
            "importance_mean": importance_mean.astype(float, copy=False).tolist(),
            "importance_std": importance_std.astype(float, copy=False).tolist(),
            "n_models": [n_models] * len(all_features),
            "method": [importance_method] * len(all_features),
        }
    ).sort("feature")

    coefficients_df: pl.DataFrame
    if coefficient_supported:
        coefficients_df = pl.DataFrame(
            {
                "feature": all_features,
                "coef_mean": np.mean(coefficient_matrix, axis=0).astype(float, copy=False).tolist(),
                "coef_std": np.std(coefficient_matrix, axis=0).astype(float, copy=False).tolist(),
                "n_models": [n_models] * len(all_features),
                "method": ["coef_signed"] * len(all_features),
                "reason": ["NA"] * len(all_features),
            }
        ).sort("feature")
    else:
        coefficients_df = pl.DataFrame(
            {
                "feature": all_features,
                "coef_mean": [None] * len(all_features),
                "coef_std": [None] * len(all_features),
                "n_models": [n_models] * len(all_features),
                "method": ["NA"] * len(all_features),
                "reason": ["unsupported_model_non_linear"] * len(all_features),
            }
        ).sort("feature")

    return InterpretationArtifacts(
        feature_importance=feature_importance,
        coefficients=coefficients_df,
        warnings=warnings,
    )
