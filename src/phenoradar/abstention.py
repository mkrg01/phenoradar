"""Fixed, label-independent information coverage and selective decisions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from phenoradar.metrics import binary_probability_metrics
from phenoradar.missing_expression import mask_expression

ABSTENTION_COLUMNS = [
    "information_coverage",
    "abstention_threshold",
    "pred_label_selective",
    "decision_status",
    "abstention_reason",
    "missing_features_json",
]


def prediction_label_expr(frame: pl.DataFrame) -> pl.Expr:
    """Use the selective decision when present; never turn abstention into zero."""
    if "pred_label_selective" in frame.columns:
        return pl.col("pred_label_selective").cast(pl.Int64)
    if "pred_label_fixed_threshold" in frame.columns:
        return pl.col("pred_label_fixed_threshold").cast(pl.Int64)
    return (pl.col("prob") >= 0.5).cast(pl.Int64)


def annotate_abstention(
    predictions: pl.DataFrame,
    *,
    species: Sequence[str],
    matrix: np.ndarray,
    feature_names: Sequence[str],
    model_coefficients: Sequence[tuple[Sequence[str], np.ndarray]],
    zero_as_missing: bool,
    threshold: float,
    top_features: int = 30,
) -> pl.DataFrame:
    """Attach coverage from the original observation mask, aligned by species.

    Normalize absolute coefficients within each model *before* averaging their
    weights. This is exactly mean model coverage, even for disjoint feature sets
    and opposite signs. Intercept-only models contribute zero coverage.
    """
    values = mask_expression(matrix, zero_as_missing=zero_as_missing)
    if values.shape != (len(species), len(feature_names)):
        raise ValueError("Abstention matrix does not match species/feature schema")
    if len(set(species)) != len(species) or len(set(feature_names)) != len(feature_names):
        raise ValueError("Abstention schema contains duplicate identifiers")
    if not 0 < threshold <= 1 or not model_coefficients:
        raise ValueError("Abstention requires models and a threshold in (0, 1]")
    if np.isinf(values).any() or np.any(values < 0):
        raise ValueError("Abstention requires nonnegative raw TPM or missing values")
    index = {name: i for i, name in enumerate(feature_names)}
    weights = np.zeros(len(feature_names), dtype=float)
    for names, coefficients in model_coefficients:
        coef = np.abs(np.asarray(coefficients, dtype=float).reshape(-1))
        if coef.shape != (len(names),) or not np.isfinite(coef).all():
            raise ValueError("Abstention coefficient schema is invalid")
        # Scaling first also avoids overflow for otherwise finite coefficients.
        maximum = float(coef.max()) if coef.size else 0.0
        if maximum == 0.0:
            continue
        normalized = coef / maximum
        normalized /= normalized.sum()
        indices = [index[name] for name in names]
        weights[indices] += normalized / len(model_coefficients)
    observed = np.isfinite(values)
    coverage = np.clip(observed @ weights, 0.0, 1.0)
    has_coefficients = bool(np.any(weights > 0))
    # Include numerical equality at the boundary without using rounded output.
    accepted = (coverage >= threshold) | np.isclose(coverage, threshold, rtol=0.0, atol=1e-12)
    accepted &= has_coefficients
    order = sorted(
        np.flatnonzero(weights > 0).tolist(), key=lambda j: (-weights[j], feature_names[j])
    )
    missing_details = []
    for row in range(len(species)):
        details = (
            [
                {"feature": feature_names[j], "coefficient_fraction": float(weights[j])}
                for j in order
                if not observed[row, j]
            ][:top_features]
            if not accepted[row]
            else []
        )
        missing_details.append(json.dumps(details, ensure_ascii=False, separators=(",", ":")))
    annotations = pl.DataFrame(
        {
            "species": pl.Series(species, dtype=pl.String),
            "information_coverage": pl.Series(coverage, dtype=pl.Float64),
            "abstention_threshold": pl.Series([threshold] * len(species), dtype=pl.Float64),
            "decision_status": pl.Series(
                np.where(accepted, "accepted", "abstained"), dtype=pl.String
            ),
            "abstention_reason": pl.Series(
                [
                    None
                    if ok
                    else "insufficient_information"
                    if has_coefficients
                    else "no_informative_coefficients"
                    for ok in accepted
                ],
                dtype=pl.String,
            ),
            "missing_features_json": pl.Series(missing_details, dtype=pl.String),
        }
    )
    if set(predictions["species"].to_list()) != set(species):
        raise ValueError("Abstention predictions do not match matrix species")
    result = predictions.with_columns(pl.col("species").cast(pl.String)).join(
        annotations, on="species", how="left", validate="1:1"
    )
    return result.with_columns(
        pl.when(pl.col("decision_status") == "accepted")
        .then(prediction_label_expr(predictions))
        .otherwise(None)
        .cast(pl.Int64)
        .alias("pred_label_selective")
    )


def abstention_summary(predictions: pl.DataFrame) -> pl.DataFrame:
    """Report denominators and selective performance, including all-abstained data."""
    rows: list[dict[str, Any]] = []
    label_col = "true_label" if "true_label" in predictions.columns else "label"
    subsets = [("overall", "all", predictions)]
    for column in (label_col, "group_id", "fold_id"):
        if column not in predictions.columns:
            continue
        for partition_key, subset in predictions.partition_by(column, as_dict=True).items():
            if partition_key[0] is not None:
                subsets.append((column, str(partition_key[0]), subset))
    for scope, key, frame in subsets:
        kept = frame.filter(pl.col("decision_status") == "accepted")
        row: dict[str, Any] = {
            "scope": scope,
            "value": key,
            "n_species": frame.height,
            "n_accepted": kept.height,
            "n_abstained": frame.height - kept.height,
            "decision_rate": kept.height / frame.height if frame.height else None,
            "n_pred_positive": kept.filter(pl.col("pred_label_selective") == 1).height,
            "n_pred_negative": kept.filter(pl.col("pred_label_selective") == 0).height,
        }
        labeled = (
            kept.filter(pl.col(label_col).is_in([0, 1]))
            if label_col in kept.columns
            else kept.head(0)
        )
        row["n_evaluated"] = labeled.height
        row.update(
            dict.fromkeys(
                [
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                    "error_rate",
                    "positive_error_rate",
                    "negative_error_rate",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "mcc",
                    "brier",
                ]
            )
        )
        if labeled.height:
            y = labeled[label_col].to_numpy()
            pred = labeled["pred_label_selective"].to_numpy()
            tp = int(np.sum((y == 1) & (pred == 1)))
            tn = int(np.sum((y == 0) & (pred == 0)))
            fp = int(np.sum((y == 0) & (pred == 1)))
            fn = int(np.sum((y == 1) & (pred == 0)))
            row.update(
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
                error_rate=(fp + fn) / len(y),
                positive_error_rate=fp / (tp + fp) if tp + fp else None,
                negative_error_rate=fn / (tn + fn) if tn + fn else None,
            )
            metrics = binary_probability_metrics(y, labeled["prob"].to_numpy(), threshold=0.5)
            row.update(
                {
                    key: float(value) if np.isfinite(value) else None
                    for key, value in metrics.items()
                }
            )
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def write_abstention_artifacts(predictions: pl.DataFrame, directory: Path) -> None:
    """Write gate evaluation and a readable long table of missing evidence."""
    if "decision_status" not in predictions.columns:
        return
    directory.mkdir(parents=True, exist_ok=True)
    abstention_summary(predictions).write_csv(
        directory / "abstention_summary.tsv", separator="\t", null_value="NA"
    )
    rows = []
    for row in predictions.iter_rows(named=True):
        for rank, feature in enumerate(json.loads(row.get("missing_features_json", "[]")), 1):
            rows.append({"species": row["species"], "rank": rank, **feature})
    pl.DataFrame(
        rows,
        schema={
            "species": pl.String,
            "rank": pl.Int64,
            "feature": pl.String,
            "coefficient_fraction": pl.Float64,
        },
    ).write_csv(directory / "abstention_features.tsv", separator="\t", null_value="NA")
