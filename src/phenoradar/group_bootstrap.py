"""Group-cluster bootstrap confidence intervals for out-of-fold predictions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import polars as pl

from phenoradar.metrics import (
    FIXED_PROBABILITY_THRESHOLD_VALUE,
    binary_log_loss,
    binary_probability_metrics,
)

_BOOTSTRAP_METRICS = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "mcc",
    "brier",
    "log_loss",
)
_BOOTSTRAP_METHOD = "percentile_group"
_LOW_GROUP_COUNT_WARNING_THRESHOLD = 10


class GroupBootstrapError(ValueError):
    """Raised when OOF group bootstrap cannot be computed."""


@dataclass(frozen=True)
class GroupBootstrapArtifacts:
    """Summary, replicate values, and warnings for one OOF group bootstrap."""

    summary: pl.DataFrame
    replicates: pl.DataFrame
    warnings: list[str]
    seed: int
    n_groups: int


def _bootstrap_seed(runtime_seed: int) -> int:
    digest = sha256(f"{runtime_seed}|oof_group_bootstrap".encode()).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _metric_values(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    metrics = binary_probability_metrics(
        y_true,
        probability,
        threshold=FIXED_PROBABILITY_THRESHOLD_VALUE,
    )
    metrics["log_loss"] = binary_log_loss(y_true, probability)
    return metrics


def _validated_oof_groups(
    *,
    oof_predictions: pl.DataFrame,
    split_manifest: pl.DataFrame,
) -> pl.DataFrame:
    prediction_required = {"species", "label", "prob"}
    missing_prediction = sorted(prediction_required - set(oof_predictions.columns))
    if missing_prediction:
        raise GroupBootstrapError(
            "OOF predictions are missing required columns: " + ", ".join(missing_prediction)
        )
    manifest_required = {"species", "pool", "group_id"}
    missing_manifest = sorted(manifest_required - set(split_manifest.columns))
    if missing_manifest:
        raise GroupBootstrapError(
            "Split manifest is missing required columns: " + ", ".join(missing_manifest)
        )
    if oof_predictions.height == 0:
        raise GroupBootstrapError("OOF predictions are empty")
    if oof_predictions.select(pl.col("species").n_unique()).item() != oof_predictions.height:
        raise GroupBootstrapError("OOF predictions must contain one row per species")

    validation_groups = split_manifest.filter(pl.col("pool") == "validation").select(
        pl.col("species").cast(pl.String, strict=False).alias("species"),
        pl.col("group_id").cast(pl.String, strict=False).alias("group_id"),
    )
    if validation_groups.height == 0:
        raise GroupBootstrapError("Split manifest contains no validation rows")
    if validation_groups.select(pl.col("species").n_unique()).item() != validation_groups.height:
        raise GroupBootstrapError(
            "Split manifest validation rows must contain one row per species"
        )

    predictions = oof_predictions.select(
        pl.col("species").cast(pl.String, strict=False).alias("species"),
        pl.col("label").cast(pl.Int64, strict=False).alias("label"),
        pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
    )
    prediction_species = set(predictions.select("species").to_series().to_list())
    validation_species = set(validation_groups.select("species").to_series().to_list())
    if prediction_species != validation_species:
        missing_groups = sorted(str(value) for value in prediction_species - validation_species)
        missing_predictions = sorted(
            str(value) for value in validation_species - prediction_species
        )
        details: list[str] = []
        if missing_groups:
            details.append("missing group assignments=" + ", ".join(missing_groups[:10]))
        if missing_predictions:
            details.append("missing OOF predictions=" + ", ".join(missing_predictions[:10]))
        raise GroupBootstrapError("OOF/split species mismatch: " + "; ".join(details))

    joined = predictions.join(validation_groups, on="species", how="inner").sort("species")
    invalid_group_count = joined.filter(
        pl.col("group_id").is_null() | (pl.col("group_id").str.strip_chars() == "")
    ).height
    if invalid_group_count > 0:
        raise GroupBootstrapError(
            f"OOF group assignments contain {invalid_group_count} null/empty values"
        )
    invalid_label_count = joined.filter(
        pl.col("label").is_null() | ~pl.col("label").is_in([0, 1])
    ).height
    if invalid_label_count > 0:
        raise GroupBootstrapError(
            f"OOF predictions contain {invalid_label_count} invalid labels"
        )
    invalid_probability_count = joined.filter(
        pl.col("prob").is_null()
        | ~pl.col("prob").is_finite()
        | (pl.col("prob") < 0.0)
        | (pl.col("prob") > 1.0)
    ).height
    if invalid_probability_count > 0:
        raise GroupBootstrapError(
            f"OOF predictions contain {invalid_probability_count} invalid probabilities"
        )
    return joined


def run_oof_group_bootstrap(
    *,
    oof_predictions: pl.DataFrame,
    split_manifest: pl.DataFrame,
    group_col: str,
    n_resamples: int,
    confidence_level: float,
    runtime_seed: int,
) -> GroupBootstrapArtifacts:
    """Bootstrap pooled OOF metrics by resampling intact split groups."""
    if n_resamples < 1:
        raise GroupBootstrapError("n_resamples must be >= 1")
    if not 0.0 < confidence_level < 1.0:
        raise GroupBootstrapError("confidence_level must be in (0, 1)")
    if not group_col.strip():
        raise GroupBootstrapError("group_col must be non-empty")

    joined = _validated_oof_groups(
        oof_predictions=oof_predictions,
        split_manifest=split_manifest,
    )
    group_values = np.asarray(joined.select("group_id").to_series().to_list(), dtype=str)
    y_true = np.asarray(joined.select("label").to_series().to_list(), dtype=int)
    probability = np.asarray(joined.select("prob").to_series().to_list(), dtype=float)
    group_ids = sorted(set(group_values.tolist()))
    n_groups = len(group_ids)
    if n_groups < 2:
        raise GroupBootstrapError("OOF group bootstrap requires at least two groups")

    row_indices_by_group = [np.flatnonzero(group_values == group_id) for group_id in group_ids]
    seed = _bootstrap_seed(runtime_seed)
    rng = np.random.default_rng(seed)
    point_metrics = _metric_values(y_true, probability)
    metric_values: dict[str, list[float]] = {name: [] for name in _BOOTSTRAP_METRICS}
    replicate_rows: list[dict[str, int | float | str | None]] = []

    for resample_id in range(1, n_resamples + 1):
        sampled_group_indices = rng.integers(0, n_groups, size=n_groups)
        sampled_row_indices = np.concatenate(
            [row_indices_by_group[index] for index in sampled_group_indices]
        )
        sampled_y = y_true[sampled_row_indices]
        sampled_probability = probability[sampled_row_indices]
        sampled_metrics = _metric_values(sampled_y, sampled_probability)
        n_unique_sampled_groups = int(np.unique(sampled_group_indices).size)
        n_pos = int(np.sum(sampled_y == 1))
        n_neg = int(np.sum(sampled_y == 0))
        for metric_name in _BOOTSTRAP_METRICS:
            metric_value = float(sampled_metrics[metric_name])
            metric_values[metric_name].append(metric_value)
            replicate_rows.append(
                {
                    "resample_id": resample_id,
                    "metric": metric_name,
                    "metric_value": _finite_or_none(metric_value),
                    "n_sampled_groups": n_groups,
                    "n_unique_sampled_groups": n_unique_sampled_groups,
                    "n_species_with_multiplicity": int(sampled_row_indices.size),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
            )

    alpha = (1.0 - confidence_level) / 2.0
    summary_rows: list[dict[str, int | float | str | None]] = []
    invalid_metrics: list[str] = []
    for metric_name in _BOOTSTRAP_METRICS:
        values = np.asarray(metric_values[metric_name], dtype=float)
        finite_values = values[np.isfinite(values)]
        n_valid = int(finite_values.size)
        if n_valid < n_resamples:
            invalid_metrics.append(f"{metric_name}={n_resamples - n_valid}")
        if n_valid == 0:
            ci_lower: float | None = None
            ci_upper: float | None = None
        else:
            ci_lower, ci_upper = (
                float(value)
                for value in np.quantile(finite_values, [alpha, 1.0 - alpha])
            )
        summary_rows.append(
            {
                "metric": metric_name,
                "point_estimate": _finite_or_none(point_metrics[metric_name]),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence_level": confidence_level,
                "n_resamples": n_resamples,
                "n_valid_resamples": n_valid,
                "valid_resample_fraction": n_valid / n_resamples,
                "n_groups": n_groups,
                "group_col": group_col,
                "bootstrap_method": _BOOTSTRAP_METHOD,
                "seed": seed,
            }
        )

    warnings: list[str] = []
    if n_groups < _LOW_GROUP_COUNT_WARNING_THRESHOLD:
        warnings.append(
            "OOF group-bootstrap confidence intervals may be unstable because "
            f"only {n_groups} groups were available"
        )
    if invalid_metrics:
        warnings.append(
            "OOF group bootstrap produced single-label/undefined metric replicates "
            f"(invalid counts: {', '.join(invalid_metrics)})"
        )

    summary = pl.DataFrame(
        summary_rows,
        schema={
            "metric": pl.String,
            "point_estimate": pl.Float64,
            "ci_lower": pl.Float64,
            "ci_upper": pl.Float64,
            "confidence_level": pl.Float64,
            "n_resamples": pl.Int64,
            "n_valid_resamples": pl.Int64,
            "valid_resample_fraction": pl.Float64,
            "n_groups": pl.Int64,
            "group_col": pl.String,
            "bootstrap_method": pl.String,
            "seed": pl.Int64,
        },
    )
    replicates = pl.DataFrame(
        replicate_rows,
        schema={
            "resample_id": pl.Int64,
            "metric": pl.String,
            "metric_value": pl.Float64,
            "n_sampled_groups": pl.Int64,
            "n_unique_sampled_groups": pl.Int64,
            "n_species_with_multiplicity": pl.Int64,
            "n_pos": pl.Int64,
            "n_neg": pl.Int64,
        },
    )
    return GroupBootstrapArtifacts(
        summary=summary,
        replicates=replicates,
        warnings=warnings,
        seed=seed,
        n_groups=n_groups,
    )
