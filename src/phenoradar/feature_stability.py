"""Post-hoc feature-stability summaries for outer cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import fmean, median
from typing import Any

import numpy as np
import polars as pl

DEFAULT_NONZERO_TOLERANCE = 1e-12

_BY_FEATURE_SCHEMA = {
    "feature": pl.String,
    "n_outer_folds": pl.Int64,
    "n_retained_folds": pl.Int64,
    "retained_frequency": pl.Float64,
    "n_nonzero_folds": pl.Int64,
    "selection_frequency": pl.Float64,
    "selection_frequency_when_retained": pl.Float64,
    "n_positive_folds": pl.Int64,
    "n_negative_folds": pl.Int64,
    "dominant_sign": pl.String,
    "sign_agreement_rate": pl.Float64,
    "sign_reason": pl.String,
    "coef_mean_nonzero": pl.Float64,
    "coef_median_nonzero": pl.Float64,
    "selection_method": pl.String,
    "nonzero_tolerance": pl.Float64,
}

_BY_FOLD_PAIR_SCHEMA = {
    "fold_id_a": pl.String,
    "fold_id_b": pl.String,
    "n_nonzero_a": pl.Int64,
    "n_nonzero_b": pl.Int64,
    "n_intersection": pl.Int64,
    "n_union": pl.Int64,
    "jaccard": pl.Float64,
    "reason": pl.String,
    "selection_method": pl.String,
    "nonzero_tolerance": pl.Float64,
}

_SUMMARY_SCHEMA = {
    "n_outer_folds": pl.Int64,
    "n_fold_pairs": pl.Int64,
    "n_valid_jaccard_pairs": pl.Int64,
    "n_features_evaluated": pl.Int64,
    "n_features_ever_retained": pl.Int64,
    "n_features_ever_selected": pl.Int64,
    "n_features_selected_in_half_folds": pl.Int64,
    "n_features_selected_in_all_folds": pl.Int64,
    "jaccard_mean": pl.Float64,
    "jaccard_median": pl.Float64,
    "jaccard_min": pl.Float64,
    "jaccard_max": pl.Float64,
    "n_features_with_sign_agreement": pl.Int64,
    "sign_agreement_mean": pl.Float64,
    "selection_method": pl.String,
    "sign_available": pl.Boolean,
    "nonzero_tolerance": pl.Float64,
}


class FeatureStabilityError(ValueError):
    """Raised when fold-level interpretation inputs cannot be summarized."""


@dataclass(frozen=True)
class FeatureStabilityArtifacts:
    """Feature-level, fold-pair, and overall outer-CV stability tables."""

    by_feature: pl.DataFrame
    by_fold_pair: pl.DataFrame
    summary: pl.DataFrame


def _fold_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _require_columns(frame: pl.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FeatureStabilityError(f"{context} is missing required columns: {', '.join(missing)}")


def _finite_float(value: Any, *, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureStabilityError(f"{context} contains a non-numeric value") from exc
    if not np.isfinite(numeric):
        raise FeatureStabilityError(f"{context} contains a non-finite value")
    return numeric


def _fold_feature_values(
    frame: pl.DataFrame,
    *,
    value_col: str,
    context: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for row in frame.select("fold_id", "feature", value_col).iter_rows(named=True):
        fold_id = str(row["fold_id"])
        feature = str(row["feature"])
        numeric = _finite_float(row[value_col], context=f"{context}.{value_col}")
        fold_values = values.setdefault(fold_id, {})
        if feature in fold_values:
            raise FeatureStabilityError(
                f"{context} contains duplicate fold/feature rows: {fold_id}/{feature}"
            )
        fold_values[feature] = numeric
    return values


def _retained_sets(retained_features: pl.DataFrame) -> dict[str, set[str]]:
    required = {"scope", "fold_id", "feature"}
    _require_columns(retained_features, required, "retained_features")
    retained: dict[str, set[str]] = {}
    outer_rows = retained_features.filter(pl.col("scope") == "outer_fold")
    for row in outer_rows.select("fold_id", "feature").unique().iter_rows(named=True):
        retained.setdefault(str(row["fold_id"]), set()).add(str(row["feature"]))
    return retained


def build_feature_stability_tables(
    *,
    feature_importance_by_fold: pl.DataFrame,
    coefficients_by_fold: pl.DataFrame,
    retained_features: pl.DataFrame,
    nonzero_tolerance: float = DEFAULT_NONZERO_TOLERANCE,
) -> FeatureStabilityArtifacts:
    """Summarize outer-fold feature-set and coefficient-direction stability."""

    tolerance = float(nonzero_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise FeatureStabilityError("nonzero_tolerance must be a finite value >= 0")

    _require_columns(
        feature_importance_by_fold,
        {"fold_id", "feature", "importance_mean"},
        "feature_importance_by_fold",
    )
    _require_columns(
        coefficients_by_fold,
        {"fold_id", "feature", "coef_mean", "method", "reason"},
        "coefficients_by_fold",
    )
    if feature_importance_by_fold.height == 0:
        raise FeatureStabilityError("feature_importance_by_fold is empty")

    importance_values = _fold_feature_values(
        feature_importance_by_fold,
        value_col="importance_mean",
        context="feature_importance_by_fold",
    )
    signed_rows = coefficients_by_fold.filter(
        (pl.col("method") == "coef_signed") & pl.col("coef_mean").is_not_null()
    )
    sign_available = (
        coefficients_by_fold.height > 0 and signed_rows.height == coefficients_by_fold.height
    )
    coefficient_values: dict[str, dict[str, float]] = {}
    if sign_available:
        coefficient_values = _fold_feature_values(
            signed_rows,
            value_col="coef_mean",
            context="coefficients_by_fold",
        )

    retained_by_fold = _retained_sets(retained_features)
    fold_ids = sorted(
        set(importance_values) | set(coefficient_values) | set(retained_by_fold),
        key=_fold_sort_key,
    )
    if not fold_ids:
        raise FeatureStabilityError("No outer folds are available for feature stability")

    features = sorted(
        {feature for fold_values in importance_values.values() for feature in fold_values}
        | {feature for fold_features in retained_by_fold.values() for feature in fold_features}
    )
    if not features:
        raise FeatureStabilityError("No features are available for feature stability")

    selection_method = "coef_abs_gt_tol" if sign_available else "feature_importance_gt_tol"
    selected_by_fold: dict[str, set[str]] = {}
    signed_selected_values: dict[str, list[float]] = {feature: [] for feature in features}
    for fold_id in fold_ids:
        source_values = (
            coefficient_values.get(fold_id, {})
            if sign_available
            else importance_values.get(fold_id, {})
        )
        selected = {feature for feature, value in source_values.items() if abs(value) > tolerance}
        selected_by_fold[fold_id] = selected
        if sign_available:
            for feature in selected:
                signed_selected_values[feature].append(source_values[feature])

    n_folds = len(fold_ids)
    by_feature_rows: list[dict[str, Any]] = []
    for feature in features:
        n_retained = sum(feature in retained_by_fold.get(fold_id, set()) for fold_id in fold_ids)
        n_nonzero = sum(feature in selected_by_fold[fold_id] for fold_id in fold_ids)
        signed_values = signed_selected_values[feature] if sign_available else []
        n_positive = sum(value > tolerance for value in signed_values)
        n_negative = sum(value < -tolerance for value in signed_values)

        dominant_sign: str | None = None
        sign_agreement: float | None = None
        coef_mean_nonzero: float | None = None
        coef_median_nonzero: float | None = None
        if not sign_available:
            sign_reason = "signed_coefficients_unavailable"
        elif n_nonzero == 0:
            sign_reason = "never_selected"
        elif n_nonzero < 2:
            sign_reason = "selected_in_fewer_than_two_folds"
            dominant_sign = "positive" if n_positive > 0 else "negative"
            coef_mean_nonzero = float(fmean(signed_values))
            coef_median_nonzero = float(median(signed_values))
        else:
            sign_reason = "ok"
            if n_positive > n_negative:
                dominant_sign = "positive"
            elif n_negative > n_positive:
                dominant_sign = "negative"
            else:
                dominant_sign = "tie"
            sign_agreement = max(n_positive, n_negative) / n_nonzero
            coef_mean_nonzero = float(fmean(signed_values))
            coef_median_nonzero = float(median(signed_values))

        by_feature_rows.append(
            {
                "feature": feature,
                "n_outer_folds": n_folds,
                "n_retained_folds": n_retained,
                "retained_frequency": n_retained / n_folds,
                "n_nonzero_folds": n_nonzero,
                "selection_frequency": n_nonzero / n_folds,
                "selection_frequency_when_retained": (
                    n_nonzero / n_retained if n_retained > 0 else None
                ),
                "n_positive_folds": n_positive,
                "n_negative_folds": n_negative,
                "dominant_sign": dominant_sign,
                "sign_agreement_rate": sign_agreement,
                "sign_reason": sign_reason,
                "coef_mean_nonzero": coef_mean_nonzero,
                "coef_median_nonzero": coef_median_nonzero,
                "selection_method": selection_method,
                "nonzero_tolerance": tolerance,
            }
        )

    by_feature = pl.DataFrame(by_feature_rows, schema=_BY_FEATURE_SCHEMA).sort(
        ["selection_frequency", "retained_frequency", "feature"],
        descending=[True, True, False],
    )

    pair_rows: list[dict[str, Any]] = []
    for fold_a, fold_b in combinations(fold_ids, 2):
        selected_a = selected_by_fold[fold_a]
        selected_b = selected_by_fold[fold_b]
        intersection_count = len(selected_a & selected_b)
        union_count = len(selected_a | selected_b)
        pair_rows.append(
            {
                "fold_id_a": fold_a,
                "fold_id_b": fold_b,
                "n_nonzero_a": len(selected_a),
                "n_nonzero_b": len(selected_b),
                "n_intersection": intersection_count,
                "n_union": union_count,
                "jaccard": intersection_count / union_count if union_count > 0 else None,
                "reason": "ok" if union_count > 0 else "both_feature_sets_empty",
                "selection_method": selection_method,
                "nonzero_tolerance": tolerance,
            }
        )
    by_fold_pair = (
        pl.DataFrame(pair_rows, schema=_BY_FOLD_PAIR_SCHEMA)
        if pair_rows
        else pl.DataFrame(schema=_BY_FOLD_PAIR_SCHEMA)
    )

    jaccard_values = [
        float(value) for value in by_fold_pair.get_column("jaccard").drop_nulls().to_list()
    ]
    sign_values = [
        float(value)
        for value in by_feature.get_column("sign_agreement_rate").drop_nulls().to_list()
    ]
    nonzero_counts = by_feature.get_column("n_nonzero_folds").to_list()
    retained_counts = by_feature.get_column("n_retained_folds").to_list()
    summary = pl.DataFrame(
        [
            {
                "n_outer_folds": n_folds,
                "n_fold_pairs": len(pair_rows),
                "n_valid_jaccard_pairs": len(jaccard_values),
                "n_features_evaluated": len(features),
                "n_features_ever_retained": sum(int(value) > 0 for value in retained_counts),
                "n_features_ever_selected": sum(int(value) > 0 for value in nonzero_counts),
                "n_features_selected_in_half_folds": sum(
                    2 * int(value) >= n_folds for value in nonzero_counts
                ),
                "n_features_selected_in_all_folds": sum(
                    int(value) == n_folds for value in nonzero_counts
                ),
                "jaccard_mean": float(fmean(jaccard_values)) if jaccard_values else None,
                "jaccard_median": float(median(jaccard_values)) if jaccard_values else None,
                "jaccard_min": min(jaccard_values) if jaccard_values else None,
                "jaccard_max": max(jaccard_values) if jaccard_values else None,
                "n_features_with_sign_agreement": len(sign_values),
                "sign_agreement_mean": float(fmean(sign_values)) if sign_values else None,
                "selection_method": selection_method,
                "sign_available": sign_available,
                "nonzero_tolerance": tolerance,
            }
        ],
        schema=_SUMMARY_SCHEMA,
    )
    return FeatureStabilityArtifacts(
        by_feature=by_feature,
        by_fold_pair=by_fold_pair,
        summary=summary,
    )
