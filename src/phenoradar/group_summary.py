"""Group-level prediction summaries from metadata annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


class GroupSummaryError(ValueError):
    """Raised when group summaries cannot be generated."""


@dataclass(frozen=True)
class GroupSummaryArtifacts:
    """Group summary table and joined per-species predictions."""

    summary: pl.DataFrame
    predictions: pl.DataFrame
    suffix: str
    group_label: str


def group_summary_suffix(group_col: str) -> str:
    """Return a stable artifact-name suffix from a metadata group column."""
    text = group_col.strip()
    if text.endswith("_id"):
        text = text[:-3]
    normalized = "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "group"


def build_group_summary_artifacts(
    *,
    predictions: pl.DataFrame,
    metadata_path: Path,
    species_col: str,
    group_col: str,
    group_name_col: str | None,
    source_table_name: str,
) -> GroupSummaryArtifacts:
    """Join prediction rows to metadata and summarize probabilities by group."""
    if "species" not in predictions.columns or "prob" not in predictions.columns:
        raise GroupSummaryError(f"{source_table_name} schema is invalid for group summary")
    if predictions.height == 0:
        raise GroupSummaryError(f"{source_table_name} is empty; cannot write group summary")

    try:
        metadata = pl.read_csv(metadata_path, separator="\t")
    except FileNotFoundError as exc:
        raise GroupSummaryError(
            f"Metadata file not found for group summary: {metadata_path}"
        ) from exc
    except Exception as exc:
        raise GroupSummaryError(
            f"Failed to read metadata TSV for group summary: {metadata_path}"
        ) from exc

    missing = sorted({species_col, group_col} - set(metadata.columns))
    if missing:
        raise GroupSummaryError(
            "Skipped group summary because metadata is missing required column(s): "
            + ", ".join(missing)
        )

    group_name_expr: pl.Expr
    effective_group_name_col = group_name_col
    if effective_group_name_col is not None and effective_group_name_col not in metadata.columns:
        effective_group_name_col = None
    if effective_group_name_col is None:
        group_name_expr = pl.col(group_col).cast(pl.String, strict=False).alias("__group_name")
    else:
        group_name_expr = (
            pl.col(effective_group_name_col).cast(pl.String, strict=False).alias("__group_name")
        )

    group_metadata = (
        metadata.select(
            [
                pl.col(species_col)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("species"),
                pl.col(group_col)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("__group_id"),
                group_name_expr,
            ]
        )
        .unique("species")
        .with_columns(
            pl.when(pl.col("__group_id").is_null() | (pl.col("__group_id") == ""))
            .then(pl.lit("unassigned"))
            .otherwise(pl.col("__group_id"))
            .alias("group_id"),
            pl.when(pl.col("__group_name").is_null() | (pl.col("__group_name") == ""))
            .then(pl.col("__group_id"))
            .otherwise(pl.col("__group_name"))
            .alias("__group_name_filled"),
        )
        .with_columns(
            pl.when(pl.col("__group_name_filled").is_null() | (pl.col("__group_name_filled") == ""))
            .then(pl.lit("unassigned"))
            .otherwise(pl.col("__group_name_filled"))
            .alias("group_name")
        )
        .select(["species", "group_id", "group_name"])
    )

    normalized = predictions.select(
        [
            pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
            pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
            _label_expr(predictions),
            _pred_label_expr(predictions),
            _uncertainty_expr(predictions),
        ]
    ).filter(pl.col("species").is_not_null() & (pl.col("species") != ""))

    joined = (
        normalized.join(group_metadata, on="species", how="left")
        .with_columns(
            pl.when(pl.col("group_id").is_null() | (pl.col("group_id") == ""))
            .then(pl.lit("unassigned"))
            .otherwise(pl.col("group_id"))
            .alias("group_id"),
            pl.when(pl.col("group_name").is_null() | (pl.col("group_name") == ""))
            .then(pl.lit("unassigned"))
            .otherwise(pl.col("group_name"))
            .alias("group_name"),
        )
        .filter(pl.col("prob").is_not_null() & pl.col("prob").is_finite())
    )
    if joined.height == 0:
        raise GroupSummaryError(
            f"{source_table_name} has no finite probabilities for group summary"
        )

    summary = _summarize_joined_predictions(joined, group_col=group_col)
    suffix = group_summary_suffix(group_col)
    return GroupSummaryArtifacts(
        summary=summary,
        predictions=joined.sort(["group_name", "species"]),
        suffix=suffix,
        group_label=group_col[:-3] if group_col.endswith("_id") else group_col,
    )


def _label_expr(predictions: pl.DataFrame) -> pl.Expr:
    if "true_label" in predictions.columns:
        return pl.col("true_label").cast(pl.Int64, strict=False).alias("true_label")
    if "label" in predictions.columns:
        return pl.col("label").cast(pl.Int64, strict=False).alias("true_label")
    return pl.lit(None, dtype=pl.Int64).alias("true_label")


def _pred_label_expr(predictions: pl.DataFrame) -> pl.Expr:
    if "pred_label_fixed_threshold" in predictions.columns:
        return (
            pl.col("pred_label_fixed_threshold")
            .cast(pl.Int64, strict=False)
            .alias("pred_label_fixed_threshold")
        )
    return (
        pl.when(pl.col("prob").cast(pl.Float64, strict=False) >= 0.5)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int64)
        .alias("pred_label_fixed_threshold")
    )


def _uncertainty_expr(predictions: pl.DataFrame) -> pl.Expr:
    if "uncertainty_std" in predictions.columns:
        return pl.col("uncertainty_std").cast(pl.Float64, strict=False).alias("uncertainty_std")
    return pl.lit(None, dtype=pl.Float64).alias("uncertainty_std")


def _summarize_joined_predictions(joined: pl.DataFrame, *, group_col: str) -> pl.DataFrame:
    top_species = (
        joined.sort(["group_id", "prob", "species"], descending=[False, True, False])
        .group_by("group_id")
        .agg(
            pl.col("species").first().alias("top_species"),
            pl.col("prob").first().alias("top_prob"),
        )
    )
    summary = (
        joined.group_by(["group_id", "group_name"])
        .agg(
            pl.len().alias("n_species"),
            (pl.col("true_label") == 1).sum().alias("n_true_positive"),
            (pl.col("true_label") == 0).sum().alias("n_true_negative"),
            (pl.col("pred_label_fixed_threshold") == 1).sum().alias("n_pred_positive"),
            pl.col("prob").min().alias("prob_min"),
            pl.col("prob").quantile(0.25, interpolation="linear").alias("prob_q1"),
            pl.col("prob").median().alias("prob_median"),
            pl.col("prob").mean().alias("prob_mean"),
            pl.col("prob").quantile(0.75, interpolation="linear").alias("prob_q3"),
            pl.col("prob").max().alias("prob_max"),
            pl.col("uncertainty_std").mean().alias("uncertainty_mean"),
        )
        .with_columns(
            (pl.col("n_pred_positive") / pl.col("n_species")).alias("pred_positive_rate"),
            pl.lit(group_col).alias("group_col"),
        )
        .join(top_species, on="group_id", how="left")
        .select(
            [
                "group_col",
                "group_id",
                "group_name",
                "n_species",
                "n_true_positive",
                "n_true_negative",
                "n_pred_positive",
                "pred_positive_rate",
                "prob_min",
                "prob_q1",
                "prob_median",
                "prob_mean",
                "prob_q3",
                "prob_max",
                "uncertainty_mean",
                "top_species",
                "top_prob",
            ]
        )
        .sort(["prob_mean", "prob_max", "group_name"], descending=[True, True, False])
    )
    if joined.select(pl.col("true_label").is_not_null().any()).item() is False:
        summary = summary.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("n_true_positive"),
            pl.lit(None, dtype=pl.Int64).alias("n_true_negative"),
        )
    if joined.select(pl.col("uncertainty_std").is_not_null().any()).item() is False:
        summary = summary.with_columns(pl.lit(None, dtype=pl.Float64).alias("uncertainty_mean"))
    return summary


def finite_group_probabilities(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return plotting-ready grouped probabilities."""
    required = {"species", "prob", "group_id", "group_name", "pred_label_fixed_threshold"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise GroupSummaryError(
            "Grouped prediction table is missing required column(s): " + ", ".join(missing)
        )
    return (
        predictions.select(
            [
                pl.col("species").cast(pl.String, strict=False).alias("species"),
                pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
                pl.col("group_id").cast(pl.String, strict=False).alias("group_id"),
                pl.col("group_name").cast(pl.String, strict=False).alias("group_name"),
                pl.col("pred_label_fixed_threshold")
                .cast(pl.Int64, strict=False)
                .alias("pred_label_fixed_threshold"),
            ]
        )
        .filter(
            pl.col("species").is_not_null()
            & pl.col("prob").is_not_null()
            & pl.col("prob").is_finite()
            & pl.col("group_id").is_not_null()
            & pl.col("group_name").is_not_null()
        )
        .with_columns(
            pl.when(pl.col("group_name") == "")
            .then(pl.col("group_id"))
            .otherwise(pl.col("group_name"))
            .alias("group_name")
        )
    )
