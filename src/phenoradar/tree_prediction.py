"""Phylogenetic tree prediction annotations and optional SVG figures."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib
import matplotlib.colors
import polars as pl

from phenoradar.colors import (
    CONFUSION_GROUP_COLORS,
    CONFUSION_GROUP_LABELS,
    CONFUSION_GROUP_ORDER,
)
from phenoradar.metrics import (
    FIXED_PROBABILITY_THRESHOLD_NAME,
    FIXED_PROBABILITY_THRESHOLD_VALUE,
)


class TreePredictionError(ValueError):
    """Raised when tree prediction artifacts cannot be generated."""


_MISSING_COLOR = "#eeeeee"
_TEXT_COLOR = "#000000"
_FEATURE_HEATMAP_LIMIT = 30
_SVG_NS = "http://www.w3.org/2000/svg"
_SVG_BACKGROUND_ID = "phenoradar-svg-background"
_SVG_BACKGROUND_FILL = "#ffffff"
_EXPRESSION_SOURCE_LINE_COL = "__phenoradar_tree_source_line"
_INVALID_TPM_REASON_LABELS = (
    (1, "missing"),
    (2, "non-numeric"),
    (4, "non-finite"),
    (8, "negative"),
    (16, "non-finite-after-sum"),
)
type _TreeSvgJob = tuple[str, Callable[..., list[str]], dict[str, Any]]


def _stage_figures_dir(run_dir: Path, stage: str) -> Path:
    figures_dir = run_dir / stage / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _stage_tables_dir(run_dir: Path, stage: str) -> Path:
    tables_dir = run_dir / stage / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def _execute_tree_svg_job(
    name: str,
    func: Callable[..., list[str]],
    kwargs: dict[str, Any],
) -> tuple[str, list[str]]:
    return name, func(**kwargs)


def _run_tree_svg_jobs(
    jobs: list[_TreeSvgJob],
    *,
    parallel_workers: int,
) -> list[list[str]]:
    if not jobs:
        return []
    worker_count = max(1, min(int(parallel_workers), len(jobs)))
    if worker_count == 1:
        return [_execute_tree_svg_job(name, func, kwargs)[1] for name, func, kwargs in jobs]

    warnings_by_index: dict[int, list[str]] = {}
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    ) as executor:
        future_to_index = {
            executor.submit(_execute_tree_svg_job, name, func, kwargs): index
            for index, (name, func, kwargs) in enumerate(jobs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            _job_name, job_warnings = future.result()
            warnings_by_index[index] = job_warnings
    return [warnings_by_index.get(index, []) for index in range(len(jobs))]


def write_run_tree_prediction_artifacts(
    *,
    run_dir: Path,
    tree_path: Path | None,
    metadata_path: Path,
    tpm_path: Path,
    species_col: str,
    feature_col: str,
    value_col: str,
    trait_col: str,
    group_col: str,
    oof_predictions: pl.DataFrame,
    thresholds: pl.DataFrame,
    feature_importance: pl.DataFrame,
    coefficients: pl.DataFrame,
    pred_external_test: pl.DataFrame | None,
    top_feature_expression: pl.DataFrame | None = None,
    feature_limit: int = _FEATURE_HEATMAP_LIMIT,
    orthogroup_annotations: pl.DataFrame | None = None,
    parallel_workers: int = 1,
) -> list[str]:
    """Write run-level tree annotation TSVs and optional Toytree SVG figures."""
    if tree_path is None:
        return []
    _require_tree(tree_path)
    metadata = _load_metadata(
        metadata_path,
        species_col=species_col,
        trait_col=trait_col,
        group_col=group_col,
    )
    ordered_steps: list[tuple[str, int]] = []
    warning_steps: list[list[str]] = []
    svg_jobs: list[_TreeSvgJob] = []

    def add_warning(message: str) -> None:
        warning_steps.append([message])
        ordered_steps.append(("warning", len(warning_steps) - 1))

    def add_svg_job(
        name: str,
        func: Callable[..., list[str]],
        kwargs: dict[str, Any],
    ) -> None:
        svg_jobs.append((name, func, kwargs))
        ordered_steps.append(("job", len(svg_jobs) - 1))

    contrast_annotation = build_contrast_pair_tree_annotation(
        metadata=metadata,
        group_col=group_col,
    )
    if contrast_annotation.height > 0:
        cv_figures_dir = _stage_figures_dir(run_dir, "cv")
        cv_tables_dir = _stage_tables_dir(run_dir, "cv")
        contrast_annotation.write_csv(
            cv_tables_dir / "tree_contrast_pairs_annotation.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        add_svg_job(
            "tree_group",
            _write_tree_prediction_svg,
            {
                "tree_path": tree_path,
                "annotation": contrast_annotation,
                "out_path": cv_figures_dir / "tree_group.svg",
                "title": "",
                "tracks": ["true_label", "group_id"],
            },
        )
    else:
        add_warning("Skipped tree_group.svg: metadata contains no non-empty split group.")

    feature_annotation = build_tree_feature_heatmap_annotation(
        metadata=metadata,
        tpm_path=tpm_path,
        species_col=species_col,
        feature_col=feature_col,
        value_col=value_col,
        group_col=group_col,
        oof_predictions=oof_predictions,
        feature_importance=feature_importance,
        coefficients=coefficients,
        feature_limit=feature_limit,
        orthogroup_annotations=orthogroup_annotations,
        top_feature_expression=top_feature_expression,
    )
    if feature_annotation.height > 0:
        cv_figures_dir = _stage_figures_dir(run_dir, "cv")
        cv_tables_dir = _stage_tables_dir(run_dir, "cv")
        feature_annotation.write_csv(
            cv_tables_dir / "tree_feature_heatmap_annotation.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        add_svg_job(
            "tree_feature_heatmap_zscore",
            _write_tree_feature_heatmap_svg,
            {
                "tree_path": tree_path,
                "annotation": feature_annotation,
                "value_col": "z_score_log2_tpm",
                "out_path": cv_figures_dir / "tree_feature_heatmap_zscore.svg",
                "title": "",
                "cmap_name": "coolwarm",
                "annotate_features": orthogroup_annotations is not None,
            },
        )
        add_svg_job(
            "tree_feature_heatmap_log2_tpm",
            _write_tree_feature_heatmap_svg,
            {
                "tree_path": tree_path,
                "annotation": feature_annotation,
                "value_col": "log2_tpm_plus1",
                "out_path": cv_figures_dir / "tree_feature_heatmap_log2_tpm.svg",
                "title": "",
                "cmap_name": "viridis",
                "annotate_features": orthogroup_annotations is not None,
            },
        )
    else:
        add_warning("Skipped tree_feature_heatmap.svg: no top features were available.")

    cv_annotation = build_cv_tree_prediction_annotation(
        metadata=metadata,
        oof_predictions=oof_predictions,
        thresholds=thresholds,
        group_col=group_col,
    )
    if cv_annotation.height > 0:
        cv_figures_dir = _stage_figures_dir(run_dir, "cv")
        cv_tables_dir = _stage_tables_dir(run_dir, "cv")
        cv_annotation.write_csv(
            cv_tables_dir / "tree_prediction_cv_annotation.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        add_svg_job(
            "tree_prediction_cv",
            _write_tree_prediction_svg,
            {
                "tree_path": tree_path,
                "annotation": cv_annotation,
                "out_path": cv_figures_dir / "tree_prediction_cv.svg",
                "title": "",
                "tracks": [
                    "true_label",
                    "prob",
                    "pred_label",
                    "uncertainty_std",
                    "group_id",
                    "fold_id",
                ],
            },
        )
    else:
        add_warning("Skipped tree_prediction_cv.svg: no CV predictions with non-empty split group.")

    if pred_external_test is not None and pred_external_test.height > 0:
        external_test_figures_dir = _stage_figures_dir(run_dir, "external_test")
        external_test_tables_dir = _stage_tables_dir(run_dir, "external_test")
        external_annotation = build_external_tree_prediction_annotation(
            metadata=metadata,
            pred_external_test=pred_external_test,
            group_col=group_col,
        )
        external_annotation.write_csv(
            external_test_tables_dir / "tree_prediction_external_annotation.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        add_svg_job(
            "tree_prediction_external",
            _write_tree_prediction_svg,
            {
                "tree_path": tree_path,
                "annotation": external_annotation,
                "out_path": external_test_figures_dir / "tree_prediction_external.svg",
                "title": "External Test Tree Prediction",
                "tracks": [
                    "true_label",
                    "prob",
                    "pred_label",
                    "uncertainty_std",
                    "group_id",
                ],
            },
        )
    svg_warnings = _run_tree_svg_jobs(svg_jobs, parallel_workers=parallel_workers)
    warnings: list[str] = []
    for step_type, index in ordered_steps:
        if step_type == "warning":
            warnings.extend(warning_steps[index])
        else:
            warnings.extend(svg_warnings[index])
    return warnings


def write_predict_tree_prediction_artifacts(
    *,
    run_dir: Path,
    tree_path: Path | None,
    metadata_path: Path,
    species_col: str,
    trait_col: str,
    group_col: str,
    pred_predict: pl.DataFrame,
) -> list[str]:
    """Write predict-level tree annotation TSV and optional Toytree SVG figure."""
    if tree_path is None:
        return []
    _require_tree(tree_path)
    metadata = _load_metadata(
        metadata_path,
        species_col=species_col,
        trait_col=trait_col,
        group_col=group_col,
        require_trait=False,
    )
    annotation = build_predict_tree_prediction_annotation(
        metadata=metadata,
        pred_predict=pred_predict,
        group_col=group_col,
    )
    annotation.write_csv(
        _stage_tables_dir(run_dir, "inference") / "tree_prediction_predict_annotation.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    inference_figures_dir = _stage_figures_dir(run_dir, "inference")
    return _write_tree_prediction_svg(
        tree_path=tree_path,
        annotation=annotation,
        out_path=inference_figures_dir / "tree_prediction_predict.svg",
        title="Prediction Tree",
        tracks=[
            "true_label",
            "prob",
            "pred_label_fixed_threshold",
            "uncertainty_std",
            "group_id",
        ],
    )


def build_contrast_pair_tree_annotation(
    *,
    metadata: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Build ggtree-friendly metadata annotation for grouped species."""
    _require_columns(metadata, {"species", "true_label", group_col}, "metadata TSV")
    group_lookup = _metadata_group_lookup(metadata, group_col=group_col)
    return (
        group_lookup.join(metadata.select(["species", "true_label"]), on="species", how="left")
        .filter(pl.col("group_id").is_not_null() & (pl.col("group_id") != ""))
        .with_columns(
            pl.col("species").alias("label"),
            pl.col("true_label").cast(pl.Int8, strict=False).alias("true_label"),
        )
        .select(["label", "species", "true_label", "group_id", "group_name"])
        .sort(["group_id", "species"])
    )


def build_tree_feature_heatmap_annotation(
    *,
    metadata: pl.DataFrame,
    tpm_path: Path,
    species_col: str,
    feature_col: str,
    value_col: str,
    group_col: str,
    feature_importance: pl.DataFrame,
    coefficients: pl.DataFrame,
    oof_predictions: pl.DataFrame | None = None,
    feature_limit: int = _FEATURE_HEATMAP_LIMIT,
    orthogroup_annotations: pl.DataFrame | None = None,
    top_feature_expression: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build long-form feature heatmap values for grouped species and top features."""
    _require_columns(metadata, {"species", "true_label", group_col}, "metadata TSV")
    _require_columns(feature_importance, {"feature", "importance_mean"}, "feature_importance.tsv")
    if feature_limit < 1:
        raise TreePredictionError("feature heatmap limit must be >= 1")

    species_meta = (
        _metadata_group_lookup(metadata, group_col=group_col)
        .join(metadata.select(["species", "true_label"]), on="species", how="left")
        .filter(pl.col("group_id").is_not_null() & (pl.col("group_id") != ""))
        .select(
            [
                "species",
                "true_label",
                "group_id",
                "group_name",
            ]
        )
        .unique("species")
        .sort(["group_id", "species"])
    )
    if species_meta.height == 0:
        return _empty_feature_heatmap_annotation()
    if oof_predictions is None:
        species_meta = species_meta.with_columns(pl.lit(None, dtype=pl.Float64).alias("prob"))
    else:
        _require_columns(oof_predictions, {"species", "prob"}, "prediction_cv.tsv")
        prediction_probs = (
            oof_predictions.with_columns(
                pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
                pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
            )
            .drop_nulls(["species"])
            .group_by("species")
            .agg(pl.col("prob").mean().alias("prob"))
        )
        species_meta = species_meta.join(prediction_probs, on="species", how="left")

    top_features = (
        feature_importance.drop_nulls(["feature", "importance_mean"])
        .with_columns(
            pl.col("feature").cast(pl.String, strict=False).str.strip_chars().alias("feature"),
            pl.col("importance_mean").cast(pl.Float64, strict=False).alias("importance_mean"),
        )
        .filter(pl.col("feature").is_not_null() & (pl.col("feature") != ""))
        .sort(["importance_mean", "feature"], descending=[True, False])
        .head(feature_limit)
        .with_row_index("feature_rank", offset=1)
        .select(["feature_rank", "feature", "importance_mean"])
    )
    if top_features.height == 0:
        return _empty_feature_heatmap_annotation()
    top_features = _join_orthogroup_annotations(
        top_features,
        orthogroup_annotations=orthogroup_annotations,
    )

    coef_lookup = _coefficient_lookup(coefficients)
    grid = species_meta.join(top_features, how="cross")
    requested_species = list(species_meta.select("species").to_series().to_list())
    requested_features = list(top_features.select("feature").to_series().to_list())
    expression = _cached_expression_for_heatmap(
        top_feature_expression,
        species=requested_species,
        features=requested_features,
    )
    if expression is None:
        expression = _load_expression_for_heatmap(
            tpm_path=tpm_path,
            species=requested_species,
            features=requested_features,
            species_col=species_col,
            feature_col=feature_col,
            value_col=value_col,
        )
    annotated = (
        grid.join(expression, on=["species", "feature"], how="left")
        .with_columns(pl.col("tpm").fill_null(0.0))
        .join(coef_lookup, on="feature", how="left")
        .with_columns((pl.col("tpm") + 1.0).log(base=2.0).alias("log2_tpm_plus1"))
    )
    stats = annotated.group_by("feature").agg(
        pl.col("log2_tpm_plus1").mean().alias("__feature_mean"),
        pl.col("log2_tpm_plus1").std(ddof=0).alias("__feature_std"),
    )
    return (
        annotated.join(stats, on="feature", how="left")
        .with_columns(
            pl.when(pl.col("__feature_std").is_null() | (pl.col("__feature_std") <= 0.0))
            .then(0.0)
            .otherwise(
                (pl.col("log2_tpm_plus1") - pl.col("__feature_mean"))
                / pl.col("__feature_std")
            )
            .alias("z_score_log2_tpm"),
            pl.col("species").alias("label"),
        )
        .select(
            [
                "label",
                "species",
                "true_label",
                "prob",
                "group_id",
                "group_name",
                "feature_rank",
                "feature",
                "orthogroup_annotation_taxid",
                "orthogroup_annotation",
                "importance_mean",
                "coef_mean",
                "tpm",
                "log2_tpm_plus1",
                "z_score_log2_tpm",
            ]
        )
        .sort(["feature_rank", "group_id", "species"])
    )


def build_cv_tree_prediction_annotation(
    *,
    metadata: pl.DataFrame,
    oof_predictions: pl.DataFrame,
    thresholds: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Build ggtree-friendly CV annotation for grouped validation species."""
    _require_columns(oof_predictions, {"fold_id", "species", "label", "prob"}, "prediction_cv.tsv")
    threshold = _fixed_probability_threshold(thresholds)
    predictions = oof_predictions.with_columns(
        pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
        pl.col("fold_id").cast(pl.String, strict=False).alias("fold_id"),
        pl.col("label").cast(pl.Int8, strict=False).alias("true_label"),
        pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
    )
    if "uncertainty_std" not in predictions.columns:
        predictions = predictions.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("uncertainty_std")
        )
    joined = predictions.join(
        _metadata_group_lookup(metadata, group_col=group_col), on="species", how="left"
    )
    return (
        joined.filter(pl.col("group_id").is_not_null() & (pl.col("group_id") != ""))
        .with_columns(
            pl.col("species").alias("label"),
            (pl.col("prob") >= threshold).cast(pl.Int8).alias("pred_label"),
        )
        .select(
            [
                "label",
                "species",
                "true_label",
                "prob",
                "pred_label",
                "uncertainty_std",
                "group_id",
                "group_name",
                "fold_id",
            ]
        )
        .sort(["group_id", "fold_id", "species"])
    )


def build_external_tree_prediction_annotation(
    *,
    metadata: pl.DataFrame,
    pred_external_test: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Build ggtree-friendly external-test annotation."""
    _require_columns(
        pred_external_test,
        {"species", "true_label", "prob", "pred_label_fixed_threshold"},
        "prediction_external_test.tsv",
    )
    predictions = pred_external_test.with_columns(
        pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
        pl.col("true_label").cast(pl.Int8, strict=False).alias("true_label"),
        pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
        pl.col("pred_label_fixed_threshold")
        .cast(pl.Int8, strict=False)
        .alias("pred_label"),
    )
    if "uncertainty_std" not in predictions.columns:
        predictions = predictions.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("uncertainty_std")
        )
    return (
        predictions.join(
            _metadata_group_lookup(metadata, group_col=group_col), on="species", how="left"
        )
        .with_columns(
            pl.col("species").alias("label"),
        )
        .select(
            [
                "label",
                "species",
                "true_label",
                "prob",
                "pred_label",
                "uncertainty_std",
                "group_id",
                "group_name",
            ]
        )
        .sort("species")
    )


def build_predict_tree_prediction_annotation(
    *,
    metadata: pl.DataFrame,
    pred_predict: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Build ggtree-friendly prediction annotation."""
    _require_columns(
        pred_predict,
        {
            "species",
            "true_label",
            "prob",
            "pred_label_fixed_threshold",
        },
        "prediction_inference.tsv",
    )
    predictions = pred_predict.with_columns(
        pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
        pl.col("true_label").cast(pl.Int8, strict=False).alias("true_label"),
        pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
        pl.col("pred_label_fixed_threshold")
        .cast(pl.Int8, strict=False)
        .alias("pred_label_fixed_threshold"),
    )
    if "uncertainty_std" not in predictions.columns:
        predictions = predictions.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("uncertainty_std")
        )
    return (
        predictions.join(
            _metadata_group_lookup(metadata, group_col=group_col), on="species", how="left"
        )
        .with_columns(
            pl.col("species").alias("label"),
        )
        .select(
            [
                "label",
                "species",
                "true_label",
                "prob",
                "pred_label_fixed_threshold",
                "uncertainty_std",
                "group_id",
                "group_name",
            ]
        )
        .sort("species")
    )


def _load_metadata(
    path: Path,
    *,
    species_col: str,
    trait_col: str,
    group_col: str,
    require_trait: bool = True,
) -> pl.DataFrame:
    try:
        metadata = pl.read_csv(path, separator="\t")
    except FileNotFoundError as exc:
        raise TreePredictionError(f"Metadata file not found: {path}") from exc
    except Exception as exc:
        raise TreePredictionError(f"Failed to read metadata TSV: {path}") from exc

    required = {species_col, group_col}
    if require_trait:
        required.add(trait_col)
    _require_columns(metadata, required, "metadata TSV")
    normalized = metadata.with_columns(
        pl.col(species_col).cast(pl.String, strict=False).str.strip_chars().alias("species"),
        pl.col(group_col).cast(pl.String, strict=False).str.strip_chars().alias(group_col),
    )
    if trait_col in normalized.columns:
        normalized = normalized.with_columns(
            pl.col(trait_col).cast(pl.Int8, strict=False).alias("true_label")
        )
    return normalized


def _metadata_group_lookup(metadata: pl.DataFrame, *, group_col: str) -> pl.DataFrame:
    group_name_col = _group_name_column(group_col, metadata.columns)
    group_name_expr: pl.Expr
    if group_name_col is None:
        group_name_expr = pl.lit(None, dtype=pl.String).alias("group_name")
    else:
        group_name_expr = (
            pl.col(group_name_col)
            .cast(pl.String, strict=False)
            .str.strip_chars()
            .alias("group_name")
        )
    return (
        metadata.select(
            [
                pl.col("species"),
                pl.col(group_col)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("group_id"),
                group_name_expr,
            ]
        )
        .unique("species")
        .sort("species")
    )


def _group_name_column(group_col: str, columns: Iterable[str]) -> str | None:
    column_set = set(columns)
    candidates: list[str] = []
    if group_col.endswith("_id"):
        candidates.append(f"{group_col[:-3]}_name")
    candidates.append(f"{group_col}_name")
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _cached_expression_for_heatmap(
    expression: pl.DataFrame | None,
    *,
    species: list[str],
    features: list[str],
) -> pl.DataFrame | None:
    if expression is None:
        return None
    required = {"species", "feature", "tpm"}
    if not required.issubset(expression.columns):
        raise TreePredictionError("Cached feature expression table schema is invalid")
    requested_species = set(species)
    requested_features = set(features)
    normalized = (
        expression.select(
            pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
            pl.col("feature").cast(pl.String, strict=False).str.strip_chars().alias("feature"),
            pl.col("tpm").cast(pl.Float64, strict=False).alias("tpm"),
        )
        .filter(
            pl.col("species").is_in(species)
            & pl.col("feature").is_in(features)
        )
    )
    cached_species = set(normalized.get_column("species").drop_nulls().to_list())
    cached_features = set(normalized.get_column("feature").drop_nulls().to_list())
    if not requested_species.issubset(cached_species) or not requested_features.issubset(
        cached_features
    ):
        return None
    if normalized.filter(
        pl.col("tpm").is_null() | ~pl.col("tpm").is_finite() | (pl.col("tpm") < 0.0)
    ).height:
        raise TreePredictionError(
            "Cached feature expression table contains invalid TPM values"
        )
    return normalized.group_by(["species", "feature"]).agg(pl.col("tpm").sum())


def _load_expression_for_heatmap(
    *,
    tpm_path: Path,
    species: list[str],
    features: list[str],
    species_col: str,
    feature_col: str,
    value_col: str,
) -> pl.DataFrame:
    try:
        schema_scan = pl.scan_csv(tpm_path, separator="\t")
        schema_columns = set(schema_scan.collect_schema().names())
    except FileNotFoundError as exc:
        raise TreePredictionError(f"Expression file not found: {tpm_path}") from exc
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise TreePredictionError(f"Failed to read expression TSV: {tpm_path}") from exc
    required = {species_col, feature_col, value_col}
    missing = sorted(required - schema_columns)
    if missing:
        raise TreePredictionError(
            f"Missing required columns in expression TSV: {', '.join(missing)}"
        )
    if _EXPRESSION_SOURCE_LINE_COL in schema_columns:
        raise TreePredictionError(
            "Expression TSV uses a reserved internal column name: "
            f"{_EXPRESSION_SOURCE_LINE_COL}"
        )

    try:
        expression = pl.scan_csv(
            tpm_path,
            separator="\t",
            schema_overrides={value_col: pl.String},
            row_index_name=_EXPRESSION_SOURCE_LINE_COL,
            row_index_offset=2,
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise TreePredictionError(f"Failed to read expression TSV: {tpm_path}") from exc

    species_value = pl.col(species_col).cast(pl.String, strict=False).str.strip_chars()
    feature_value = pl.col(feature_col).cast(pl.String, strict=False).str.strip_chars()
    raw_value = pl.col(value_col).cast(pl.String, strict=False).str.strip_chars()
    parsed_value = raw_value.cast(pl.Float64, strict=False)
    invalid_reason_mask = (
        pl.when(raw_value.is_null() | (raw_value == ""))
        .then(pl.lit(1, dtype=pl.UInt8))
        .when(parsed_value.is_null())
        .then(pl.lit(2, dtype=pl.UInt8))
        .when(~parsed_value.is_finite())
        .then(pl.lit(4, dtype=pl.UInt8))
        .when(parsed_value < 0.0)
        .then(pl.lit(8, dtype=pl.UInt8))
        .otherwise(pl.lit(0, dtype=pl.UInt8))
    )
    normalized = (
        expression.select(
            species_value.alias("species"),
            feature_value.alias("feature"),
            parsed_value.alias("__parsed_value"),
            invalid_reason_mask.alias("__invalid_reason_mask"),
            pl.col(_EXPRESSION_SOURCE_LINE_COL).alias("__source_line"),
        )
        .filter(pl.col("species").is_in(species) & pl.col("feature").is_in(features))
        .with_columns(
            pl.when(pl.col("__invalid_reason_mask") == 0)
            .then(pl.col("__parsed_value"))
            .otherwise(pl.lit(float("nan")))
            .alias("tpm")
        )
    )
    invalid = pl.col("__invalid_reason_mask") != 0
    data_scan = (
        normalized
        .group_by(["species", "feature"])
        .agg(
            pl.col("tpm").sum(),
            invalid.sum().alias("__invalid_count"),
            pl.col("__source_line").min().alias("__group_first_line"),
            pl.col("__source_line").filter(invalid).min().alias("__invalid_line"),
            pl.col("__invalid_reason_mask").max().alias("__invalid_reason_mask"),
        )
    )
    aggregate_overflow = (pl.col("__invalid_count") == 0) & ~pl.col("tpm").is_finite()
    data_scan = (
        data_scan.with_columns(
            pl.when(aggregate_overflow)
            .then(pl.lit(1, dtype=pl.UInt32))
            .otherwise(pl.col("__invalid_count"))
            .alias("__invalid_count"),
            pl.when(aggregate_overflow)
            .then(pl.col("__group_first_line"))
            .otherwise(pl.col("__invalid_line"))
            .alias("__invalid_line"),
            pl.when(aggregate_overflow)
            .then(pl.lit(16, dtype=pl.UInt8))
            .otherwise(pl.col("__invalid_reason_mask"))
            .alias("__invalid_reason_mask"),
        ).drop("__group_first_line")
    )
    try:
        data = data_scan.collect()
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise TreePredictionError(f"Failed to read expression TSV: {tpm_path}: {exc}") from exc

    invalid_rows = (
        data.filter(pl.col("__invalid_count") > 0).sort("__invalid_line").head(10)
    )
    if invalid_rows.height > 0:
        total = int(data.select(pl.col("__invalid_count").sum()).item())
        examples: list[str] = []
        for row in invalid_rows.iter_rows(named=True):
            reason_mask = int(row["__invalid_reason_mask"])
            reasons = ",".join(
                label for flag, label in _INVALID_TPM_REASON_LABELS if reason_mask & flag
            )
            examples.append(
                f"first_invalid_line={row['__invalid_line']}: species={row['species']!r}, "
                f"feature={row['feature']!r}, example_reasons_in_coordinate=({reasons})"
            )
        raise TreePredictionError(
            "Invalid expression TSV: consumed TPM values must be non-negative finite numbers; "
            f"invalid_rows={total}; examples: {'; '.join(examples)}"
        )
    return data.select(["species", "feature", "tpm"])


def _coefficient_lookup(coefficients: pl.DataFrame) -> pl.DataFrame:
    if not {"feature", "coef_mean"}.issubset(coefficients.columns):
        return pl.DataFrame(
            {"feature": [], "coef_mean": []},
            schema={"feature": pl.String, "coef_mean": pl.Float64},
        )
    data = coefficients.with_columns(
        pl.col("feature").cast(pl.String, strict=False).str.strip_chars().alias("feature"),
        pl.col("coef_mean").cast(pl.Float64, strict=False).alias("coef_mean"),
    )
    if "method" in data.columns:
        data = data.filter(pl.col("method") == "coef_signed")
    return data.select(["feature", "coef_mean"]).unique("feature")


def _empty_feature_heatmap_annotation() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "label": pl.String,
            "species": pl.String,
            "true_label": pl.Int8,
            "prob": pl.Float64,
            "group_id": pl.String,
            "group_name": pl.String,
            "feature_rank": pl.UInt32,
            "feature": pl.String,
            "orthogroup_annotation_taxid": pl.String,
            "orthogroup_annotation": pl.String,
            "importance_mean": pl.Float64,
            "coef_mean": pl.Float64,
            "tpm": pl.Float64,
            "log2_tpm_plus1": pl.Float64,
            "z_score_log2_tpm": pl.Float64,
        }
    )


def _join_orthogroup_annotations(
    top_features: pl.DataFrame,
    *,
    orthogroup_annotations: pl.DataFrame | None,
) -> pl.DataFrame:
    if orthogroup_annotations is None:
        return top_features.with_columns(
            [
                pl.lit(None, dtype=pl.String).alias("orthogroup_annotation_taxid"),
                pl.lit(None, dtype=pl.String).alias("orthogroup_annotation"),
            ]
        )
    required = {"feature", "orthogroup_annotation_taxid", "orthogroup_annotation"}
    if not required.issubset(orthogroup_annotations.columns):
        raise TreePredictionError("orthogroup annotation table schema is invalid")
    annotations = (
        orthogroup_annotations.select(
            [
                pl.col("feature").cast(pl.String, strict=False).str.strip_chars().alias("feature"),
                pl.col("orthogroup_annotation_taxid")
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("orthogroup_annotation_taxid"),
                pl.col("orthogroup_annotation")
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("orthogroup_annotation"),
            ]
        )
        .filter(pl.col("feature").is_not_null() & (pl.col("feature") != ""))
        .unique("feature")
    )
    return top_features.join(annotations, on="feature", how="left")


def _require_tree(tree_path: Path) -> None:
    if not tree_path.exists():
        raise TreePredictionError(f"Input tree not found: {tree_path}")
    if not tree_path.is_file():
        raise TreePredictionError(f"Input tree is not a file: {tree_path}")


def _require_columns(frame: pl.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TreePredictionError(f"Missing required columns in {context}: {', '.join(missing)}")


def _fixed_probability_threshold(thresholds: pl.DataFrame) -> float:
    _require_columns(thresholds, {"threshold_name", "threshold_value"}, "thresholds.tsv")
    row = thresholds.filter(
        pl.col("threshold_name") == FIXED_PROBABILITY_THRESHOLD_NAME
    )
    if row.height == 0:
        raise TreePredictionError(
            f"thresholds.tsv must contain {FIXED_PROBABILITY_THRESHOLD_NAME}"
        )
    raw = row.select("threshold_value").to_series().to_list()[0]
    if raw is None:
        raise TreePredictionError("Selected threshold_value is null")
    return float(raw)


def _write_tree_prediction_svg(
    *,
    tree_path: Path,
    annotation: pl.DataFrame,
    out_path: Path,
    title: str,
    tracks: list[str],
) -> list[str]:
    if annotation.height == 0:
        return [f"Skipped {out_path.name}: annotation table is empty."]
    try:
        toytree = importlib.import_module("toytree")
    except ImportError:
        return [
            f"Skipped {out_path.name}: Toytree is unavailable. Reinstall phenoradar or "
            "install toytree manually to enable SVG output."
        ]

    try:
        tree = toytree.tree(str(tree_path))
    except Exception as exc:
        raise TreePredictionError(f"Failed to read tree with Toytree: {tree_path}") from exc

    tip_labels = [str(v) for v in tree.get_tip_labels()]
    tip_set = set(tip_labels)
    requested_species = [
        str(v)
        for v in annotation.select("species").to_series().to_list()
        if v is not None and str(v) in tip_set
    ]
    requested_species = _unique_preserve_order(requested_species)
    missing_count = annotation.height - len(requested_species)
    warnings: list[str] = []
    if missing_count > 0:
        warnings.append(
            f"{out_path.name}: skipped {missing_count} annotation row(s) absent from the tree."
        )
    if len(requested_species) < 2:
        warnings.append(
            f"Skipped {out_path.name}: fewer than two annotated species are in the tree."
        )
        return warnings

    if len(requested_species) < len(tip_labels):
        try:
            tree = tree.mod.prune(*requested_species)
        except Exception as exc:
            raise TreePredictionError(f"Failed to prune tree for {out_path.name}") from exc
    with suppress(Exception):
        tree = tree.ladderize()
    _draw_toytree_heatmap(
        tree=tree,
        annotation=annotation,
        out_path=out_path,
        title=title,
        tracks=tracks,
        toytree_module=toytree,
    )
    return warnings


def _write_tree_feature_heatmap_svg(
    *,
    tree_path: Path,
    annotation: pl.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    cmap_name: str,
    annotate_features: bool = False,
) -> list[str]:
    if annotation.height == 0:
        return [f"Skipped {out_path.name}: annotation table is empty."]
    try:
        toytree = importlib.import_module("toytree")
    except ImportError:
        return [
            f"Skipped {out_path.name}: Toytree is unavailable. Reinstall phenoradar or "
            "install toytree manually to enable SVG output."
        ]

    try:
        tree = toytree.tree(str(tree_path))
    except Exception as exc:
        raise TreePredictionError(f"Failed to read tree with Toytree: {tree_path}") from exc

    tip_labels = [str(v) for v in tree.get_tip_labels()]
    tip_set = set(tip_labels)
    requested_species = [
        str(v)
        for v in annotation.select("species").unique().to_series().to_list()
        if v is not None and str(v) in tip_set
    ]
    requested_species = _unique_preserve_order(requested_species)
    total_species = annotation.select("species").unique().height
    missing_count = total_species - len(requested_species)
    warnings: list[str] = []
    if missing_count > 0:
        warnings.append(
            f"{out_path.name}: skipped {missing_count} species absent from the tree."
        )
    if len(requested_species) < 2:
        warnings.append(
            f"Skipped {out_path.name}: fewer than two annotated species are in the tree."
        )
        return warnings

    if len(requested_species) < len(tip_labels):
        try:
            tree = tree.mod.prune(*requested_species)
        except Exception as exc:
            raise TreePredictionError(f"Failed to prune tree for {out_path.name}") from exc
    with suppress(Exception):
        tree = tree.ladderize()
    _draw_toytree_feature_heatmap(
        tree=tree,
        annotation=annotation,
        value_col=value_col,
        out_path=out_path,
        title=title,
        cmap_name=cmap_name,
        annotate_features=annotate_features,
        toytree_module=toytree,
    )
    return warnings


def _draw_toytree_heatmap(
    *,
    tree: Any,
    annotation: pl.DataFrame,
    out_path: Path,
    title: str,
    tracks: list[str],
    toytree_module: Any,
) -> None:
    tip_labels = [str(v) for v in tree.get_tip_labels()]
    track_count = len(tracks)
    by_species = {str(row["species"]): row for row in annotation.iter_rows(named=True)}
    cell_text_by_track: dict[str, list[str]] = {}
    for track in tracks:
        values: list[str] = []
        for species in tip_labels:
            row = by_species.get(species)
            value = _track_display_value(track, row)
            values.append(_format_cell_value(track, value))
        cell_text_by_track[track] = values
    track_widths = [
        _text_column_width_px([_track_label(track), *cell_text_by_track[track]])
        for track in tracks
    ]
    annotation_width = sum(track_widths)
    species_label_width = _text_column_width_px(tip_labels, min_width=120, padding=28)
    height = max(360, 34 + 18 * len(tip_labels))
    width = max(900, 560 + annotation_width + species_label_width)
    canvas, axes, _mark = tree.draw(
        width=width,
        height=height,
        layout="r",
        tip_labels=False,
        node_sizes=0,
        scale_bar=False,
    )
    axes.show = False
    axes.x.domain.max = max(track_count + 2.8, (annotation_width + species_label_width) / 68.0)

    column_start_px = 18
    column_scale_px = 88.0
    column_cursor_px = column_start_px
    for track_index, track in enumerate(tracks):
        track_width = track_widths[track_index]
        x = (column_cursor_px + track_width / 2.0) / column_scale_px
        titles: list[str] = []
        values = cell_text_by_track[track]
        for species in tip_labels:
            row = by_species.get(species)
            value = _track_title_value(track, row)
            titles.append(f"{species} {_track_label(track)}={_format_value(value)}")
        axes.text(
            [x] * len(tip_labels),
            list(range(len(tip_labels))),
            values,
            color=_TEXT_COLOR,
            title=titles,
            style={"font-size": "8px", "text-anchor": "middle"},
        )
        axes.text(
            x,
            len(tip_labels) + 0.35,
            _track_label(track),
            angle=90,
            color=_TEXT_COLOR,
            style={"font-size": "9px", "text-anchor": "start"},
        )
        column_cursor_px += track_width
    species_x = (column_cursor_px + 16) / column_scale_px
    axes.text(
        [species_x] * len(tip_labels),
        list(range(len(tip_labels))),
        tip_labels,
        color=_TEXT_COLOR,
        title=tip_labels,
        style={"font-size": "9px", "text-anchor": "start"},
    )
    if title:
        axes.text(
            -0.05,
            len(tip_labels) + 1.1,
            title,
            color=_TEXT_COLOR,
            style={"font-size": "15px", "font-weight": "bold", "text-anchor": "start"},
        )
    _save_toytree_svg(canvas=canvas, out_path=out_path, toytree_module=toytree_module)


def _feature_heatmap_label(feature: object, annotation: object) -> str:
    feature_text = str(feature)
    if not _has_text(annotation):
        return feature_text
    return f"{feature_text}: {' '.join(str(annotation).split())}"


def _feature_heatmap_title(feature: object, annotation: object) -> str:
    feature_text = str(feature)
    if not _has_text(annotation):
        return feature_text
    return f"{feature_text}: {' '.join(str(annotation).split())}"


def _feature_label_depth_px(labels: list[str]) -> int:
    longest_line = max(
        (len(line) for label in labels for line in label.splitlines()),
        default=0,
    )
    return max(0, 7 * longest_line + 10)


def _draw_toytree_feature_heatmap(
    *,
    tree: Any,
    annotation: pl.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    cmap_name: str,
    annotate_features: bool,
    toytree_module: Any,
) -> None:
    tip_labels = [str(v) for v in tree.get_tip_labels()]
    feature_rows = (
        annotation.select(["feature_rank", "feature", "orthogroup_annotation"])
        .unique("feature")
        .sort("feature_rank")
        .iter_rows(named=True)
    )
    feature_records = list(feature_rows)
    features = [str(row["feature"]) for row in feature_records]
    if annotate_features:
        feature_labels = [
            _feature_heatmap_label(row["feature"], row.get("orthogroup_annotation"))
            for row in feature_records
        ]
        feature_titles = [
            _feature_heatmap_title(row["feature"], row.get("orthogroup_annotation"))
            for row in feature_records
        ]
    else:
        feature_labels = features
        feature_titles = features
    feature_step = 0.48
    trait_x = 0.45
    prob_x = trait_x + feature_step
    feature_start_x = prob_x + feature_step
    feature_xs = [
        feature_start_x + feature_index * feature_step
        for feature_index in range(len(feature_labels))
    ]
    heatmap_end_x = feature_xs[-1] + feature_step / 2.0
    species_x = heatmap_end_x + 0.28
    feature_label_depth_px = _feature_label_depth_px(feature_labels)
    height = max(420, 88 + feature_label_depth_px + 18 * len(tip_labels))
    width = max(1040, 620 + 28 * len(feature_labels))
    canvas, axes, _mark = tree.draw(
        width=width,
        height=height,
        layout="r",
        tip_labels=False,
        node_sizes=0,
        scale_bar=False,
    )
    axes.show = False
    axes.x.domain.max = max(len(feature_labels) + 3.0, 4.0)

    value_lookup: dict[tuple[str, str], object] = {}
    trait_lookup: dict[str, object] = {}
    prob_lookup: dict[str, object] = {}
    for row in annotation.iter_rows(named=True):
        species = str(row["species"])
        value_lookup[(species, str(row["feature"]))] = row.get(value_col)
        trait_lookup.setdefault(species, row.get("true_label"))
        prob_lookup.setdefault(species, row.get("prob"))
    confusion_groups = [
        _oof_confusion_group(trait_lookup.get(species), prob_lookup.get(species))
        for species in tip_labels
    ]
    finite_values = _finite_values(annotation, value_col)
    vmin, vmax = _heatmap_domain(value_col, finite_values)
    trait_values = [
        _format_cell_value("true_label", trait_lookup.get(species)) for species in tip_labels
    ]
    axes.text(
        [trait_x] * len(tip_labels),
        list(range(len(tip_labels))),
        trait_values,
        color=_TEXT_COLOR,
        title=[
            f"{species} trait={_format_value(trait_lookup.get(species))}" for species in tip_labels
        ],
        style={"font-size": "8px", "text-anchor": "middle"},
    )
    axes.text(
        trait_x,
        len(tip_labels) + 0.35,
        "trait",
        angle=90,
        color=_TEXT_COLOR,
        style={"font-size": "7px", "text-anchor": "start"},
    )
    prob_values = [_format_heatmap_prob_value(prob_lookup.get(species)) for species in tip_labels]
    axes.text(
        [prob_x] * len(tip_labels),
        list(range(len(tip_labels))),
        prob_values,
        color=_TEXT_COLOR,
        title=[
            f"{species} prob={_format_value(prob_lookup.get(species))}" for species in tip_labels
        ],
        style={"font-size": "8px", "text-anchor": "middle"},
    )
    axes.text(
        prob_x,
        len(tip_labels) + 0.35,
        "prob",
        angle=90,
        color=_TEXT_COLOR,
        style={"font-size": "7px", "text-anchor": "start"},
    )
    for x, feature, feature_label, feature_title in zip(
        feature_xs,
        features,
        feature_labels,
        feature_titles,
        strict=True,
    ):
        colors: list[str] = []
        titles: list[str] = []
        for species in tip_labels:
            value = value_lookup.get((species, feature))
            colors.append(_continuous_color(value, vmin=vmin, vmax=vmax, cmap_name=cmap_name))
            titles.append(f"{species} {feature_title} {value_col}={_format_value(value)}")
        axes.scatterplot(
            [x] * len(tip_labels),
            list(range(len(tip_labels))),
            marker="s",
            size=9,
            color=colors,
            title=titles,
        )
        axes.text(
            x,
            len(tip_labels) + 0.35,
            feature_label,
            angle=90,
            color=_TEXT_COLOR,
            style={"font-size": "7px", "text-anchor": "start"},
        )
    axes.text(
        [species_x] * len(tip_labels),
        list(range(len(tip_labels))),
        tip_labels,
        color=[
            CONFUSION_GROUP_COLORS[group] if group is not None else _TEXT_COLOR
            for group in confusion_groups
        ],
        title=[
            (
                f"{species} OOF confusion group={group}"
                if group is not None
                else species
            )
            for species, group in zip(tip_labels, confusion_groups, strict=True)
        ],
        style={"font-size": "9px", "text-anchor": "start"},
    )
    _draw_heatmap_legend(
        canvas=canvas,
        width=width,
        value_col=value_col,
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
    )
    _draw_confusion_group_legend(canvas=canvas, width=width)
    if title:
        axes.text(
            -0.05,
            len(tip_labels) + 1.45,
            title,
            color=_TEXT_COLOR,
            style={"font-size": "15px", "font-weight": "bold", "text-anchor": "start"},
        )
    _save_toytree_svg(canvas=canvas, out_path=out_path, toytree_module=toytree_module)


def _save_toytree_svg(*, canvas: Any, out_path: Path, toytree_module: Any) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    toytree_module.save(canvas, str(out_path))
    _ensure_svg_white_background(out_path)


def _ensure_svg_white_background(svg_path: Path) -> None:
    ET.register_namespace("", _SVG_NS)
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as exc:
        raise TreePredictionError(f"Failed to parse SVG output: {svg_path}") from exc
    except OSError as exc:
        raise TreePredictionError(f"Failed to read SVG output: {svg_path}") from exc

    root = tree.getroot()
    if _xml_local_name(root.tag) != "svg":
        raise TreePredictionError(f"SVG output root is not <svg>: {svg_path}")

    namespace = _xml_namespace(root.tag)
    rect_tag = f"{{{namespace}}}rect" if namespace else "rect"
    for child in list(root):
        if child.get("id") == _SVG_BACKGROUND_ID:
            root.remove(child)
    root.insert(
        0,
        ET.Element(
            rect_tag,
            {
                "id": _SVG_BACKGROUND_ID,
                "x": "0",
                "y": "0",
                "width": "100%",
                "height": "100%",
                "fill": _SVG_BACKGROUND_FILL,
            },
        ),
    )
    try:
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    except OSError as exc:
        raise TreePredictionError(f"Failed to write SVG output: {svg_path}") from exc


def _xml_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag[1:].partition("}")[0]
    return ""


def _xml_local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.partition("}")[2]
    return tag


def _draw_heatmap_legend(
    *,
    canvas: Any,
    width: int,
    value_col: str,
    vmin: float,
    vmax: float,
    cmap_name: str,
) -> None:
    toyplot_locator = importlib.import_module("toyplot.locator")
    tick_locations = [vmin] if vmin == vmax else [vmin, vmax]
    tick_labels = [_format_legend_value(value) for value in tick_locations]
    canvas.color_scale(
        _toyplot_linear_colormap(cmap_name=cmap_name, vmin=vmin, vmax=vmax),
        x1=width - 270,
        y1=82,
        x2=width - 105,
        y2=82,
        width=12,
        label=_heatmap_legend_label(value_col),
        min=vmin,
        max=vmax,
        ticklocator=toyplot_locator.Explicit(tick_locations, tick_labels),
    )
    missing_axes = canvas.cartesian(
        bounds=(width - 92, width - 35, 50, 112),
        show=False,
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1,
    )
    missing_axes.show = False
    missing_axes.rectangle(
        [0.06],
        [0.28],
        [0.42],
        [0.58],
        color=[_MISSING_COLOR],
        title=["Missing value"],
        style={"stroke": "none"},
    )
    missing_axes.text(
        0.36,
        0.5,
        "NA",
        color=_TEXT_COLOR,
        style={"font-size": "7px", "text-anchor": "start"},
    )


def _draw_confusion_group_legend(*, canvas: Any, width: int) -> None:
    legend_axes = canvas.cartesian(
        bounds=(width - 270, width - 35, 150, 266),
        show=False,
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1,
    )
    legend_axes.show = False
    legend_axes.text(
        0.02,
        0.92,
        "OOF confusion group",
        color=_TEXT_COLOR,
        style={"font-size": "8px", "font-weight": "bold", "text-anchor": "start"},
    )
    y_positions = [0.70, 0.51, 0.32, 0.13]
    for group, y in zip(CONFUSION_GROUP_ORDER, y_positions, strict=True):
        color = CONFUSION_GROUP_COLORS[group]
        legend_axes.rectangle(
            [0.03],
            [0.09],
            [y - 0.055],
            [y + 0.055],
            color=[color],
            title=[f"{group}: {CONFUSION_GROUP_LABELS[group]}"],
            style={"stroke": "none"},
        )
        legend_axes.text(
            0.13,
            y,
            f"{group}: {CONFUSION_GROUP_LABELS[group]}",
            color=color,
            style={"font-size": "7px", "text-anchor": "start"},
        )


def _oof_confusion_group(true_label: object, probability: object) -> str | None:
    try:
        label_value = float(true_label)  # type: ignore[arg-type]
        probability_value = float(probability)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if label_value not in {0.0, 1.0} or not math.isfinite(probability_value):
        return None
    predicted_label = int(probability_value >= FIXED_PROBABILITY_THRESHOLD_VALUE)
    return {
        (1, 1): "TP",
        (1, 0): "FN",
        (0, 0): "TN",
        (0, 1): "FP",
    }[(int(label_value), predicted_label)]


def _heatmap_legend_label(value_col: str) -> str:
    return {
        "log2_tpm_plus1": "log2(TPM + 1)",
        "z_score_log2_tpm": "within-feature z-score",
    }.get(value_col, value_col)


def _toyplot_linear_colormap(*, cmap_name: str, vmin: float, vmax: float) -> Any:
    toyplot_color = importlib.import_module("toyplot.color")
    cmap = matplotlib.colormaps[cmap_name]
    colors = [matplotlib.colors.to_hex(cmap(index / 255.0)) for index in range(256)]
    palette = toyplot_color.Palette(colors)
    return toyplot_color.LinearMap(
        palette=palette,
        domain_min=vmin,
        domain_max=vmax,
    )


def _continuous_color(value: object, *, vmin: float, vmax: float, cmap_name: str) -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _MISSING_COLOR
    if numeric != numeric:
        return _MISSING_COLOR
    ratio = 0.0 if vmax <= vmin else min(max((numeric - vmin) / (vmax - vmin), 0.0), 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    return matplotlib.colors.to_hex(cmap(ratio))


def _heatmap_domain(value_col: str, values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    if value_col == "z_score_log2_tpm":
        max_abs = max(abs(value) for value in values)
        return -max_abs, max_abs
    return min(values), max(values)


def _finite_values(annotation: pl.DataFrame, column: str) -> list[float]:
    if column not in annotation.columns:
        return []
    values: list[float] = []
    for value in annotation.select(column).to_series().to_list():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == numeric:
            values.append(numeric)
    return values


def _format_legend_value(value: float) -> str:
    if abs(value) < 100:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _format_heatmap_prob_value(value: object) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "NA"
    if numeric != numeric:
        return "NA"
    return f"{numeric:.2f}"


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _track_display_value(track: str, row: dict[str, Any] | None) -> object:
    if row is None:
        return None
    if track == "group_id" and _has_text(row.get("group_name")):
        return row.get("group_name")
    return row.get(track)


def _track_title_value(track: str, row: dict[str, Any] | None) -> object:
    if row is None:
        return None
    value = _track_display_value(track, row)
    if track != "group_id":
        return value
    group_id = row.get("group_id")
    if _has_text(value) and _has_text(group_id) and str(value) != str(group_id):
        return f"{value} (id={group_id})"
    return value


def _has_text(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def _format_cell_value(track: str, value: object) -> str:
    if value is None:
        return "NA"
    if track == "true_label" or track.startswith("pred_label"):
        label_value = _int_or_none(value)
        return "NA" if label_value is None else str(label_value)
    if track in {"prob", "uncertainty_std"}:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "NA"
        if numeric != numeric:
            return "NA"
        return f"{numeric:.3f}"
    return str(value)


def _text_column_width_px(
    values: Iterable[str], *, min_width: int = 42, padding: int = 18
) -> int:
    max_chars = max((len(value) for value in values), default=0)
    return max(min_width, padding + 7 * max_chars)


def _track_label(track: str) -> str:
    return {
        "true_label": "trait",
        "prob": "prob",
        "pred_label": "pred",
        "pred_label_fixed_threshold": "pred",
        "uncertainty_std": "uncert",
        "group_id": "group",
        "fold_id": "fold",
    }.get(track, track)


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
