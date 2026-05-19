"""Phylogenetic tree prediction annotations and optional SVG figures."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors
import polars as pl


class TreePredictionError(ValueError):
    """Raised when tree prediction artifacts cannot be generated."""


_MISSING_COLOR = "#eeeeee"
_LABEL_COLORS = {0: "#d62728", 1: "#1f77b4"}
_PRED_COLORS = {0: "#f4a3a3", 1: "#8ecae6"}
_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]
_FEATURE_HEATMAP_LIMIT = 30


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
    warnings: list[str] = []
    contrast_annotation = build_contrast_pair_tree_annotation(
        metadata=metadata,
        group_col=group_col,
    )
    if contrast_annotation.height > 0:
        contrast_annotation.write_csv(
            run_dir / "tree_contrast_pairs_annotation.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        warnings.extend(
            _write_tree_prediction_svg(
                tree_path=tree_path,
                annotation=contrast_annotation,
                out_path=run_dir / "figures" / "tree_contrast_pairs.svg",
                title="Tree Contrast Pairs",
                tracks=["true_label", "contrast_pair_id"],
            )
        )
    else:
        warnings.append(
            "Skipped tree_contrast_pairs.svg: metadata contains no non-empty contrast_pair_id."
        )

    feature_annotation = build_tree_feature_heatmap_annotation(
        metadata=metadata,
        tpm_path=tpm_path,
        species_col=species_col,
        feature_col=feature_col,
        value_col=value_col,
        group_col=group_col,
        feature_importance=feature_importance,
        coefficients=coefficients,
    )
    if feature_annotation.height > 0:
        feature_annotation.write_csv(
            run_dir / "tree_feature_heatmap_annotation.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        warnings.extend(
            _write_tree_feature_heatmap_svg(
                tree_path=tree_path,
                annotation=feature_annotation,
                value_col="z_score_log2_tpm",
                out_path=run_dir / "figures" / "tree_feature_heatmap_zscore.svg",
                title="Tree Feature Heatmap (z-score)",
                cmap_name="coolwarm",
            )
        )
        warnings.extend(
            _write_tree_feature_heatmap_svg(
                tree_path=tree_path,
                annotation=feature_annotation,
                value_col="log2_tpm_plus1",
                out_path=run_dir / "figures" / "tree_feature_heatmap_log2_tpm.svg",
                title="Tree Feature Heatmap (log2 TPM + 1)",
                cmap_name="viridis",
            )
        )
    else:
        warnings.append("Skipped tree_feature_heatmap.svg: no top features were available.")

    cv_annotation = build_cv_tree_prediction_annotation(
        metadata=metadata,
        oof_predictions=oof_predictions,
        thresholds=thresholds,
        group_col=group_col,
    )
    if cv_annotation.height > 0:
        cv_annotation.write_csv(
            run_dir / "tree_prediction_cv_annotation.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        warnings.extend(
            _write_tree_prediction_svg(
                tree_path=tree_path,
                annotation=cv_annotation,
                out_path=run_dir / "figures" / "tree_prediction_cv.svg",
                title="CV Tree Prediction",
                tracks=[
                    "true_label",
                    "prob",
                    "pred_label",
                    "uncertainty_std",
                    "contrast_pair_id",
                    "fold_id",
                ],
            )
        )
    else:
        warnings.append(
            "Skipped tree_prediction_cv.svg: no CV predictions with non-empty contrast_pair_id."
        )

    if pred_external_test is not None and pred_external_test.height > 0:
        external_annotation = build_external_tree_prediction_annotation(
            metadata=metadata,
            pred_external_test=pred_external_test,
            group_col=group_col,
        )
        external_annotation.write_csv(
            run_dir / "tree_prediction_external_annotation.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        warnings.extend(
            _write_tree_prediction_svg(
                tree_path=tree_path,
                annotation=external_annotation,
                out_path=run_dir / "figures" / "tree_prediction_external.svg",
                title="External Test Tree Prediction",
                tracks=[
                    "true_label",
                    "prob",
                    "pred_label",
                    "uncertainty_std",
                    "contrast_pair_id",
                ],
            )
        )
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
        run_dir / "tree_prediction_predict_annotation.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    return _write_tree_prediction_svg(
        tree_path=tree_path,
        annotation=annotation,
        out_path=run_dir / "figures" / "tree_prediction_predict.svg",
        title="Prediction Tree",
        tracks=[
            "true_label",
            "prob",
            "pred_label_cv_derived_threshold",
            "uncertainty_std",
            "contrast_pair_id",
        ],
    )


def build_contrast_pair_tree_annotation(
    *,
    metadata: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Build ggtree-friendly metadata annotation for contrast-pair species."""
    _require_columns(metadata, {"species", "true_label", group_col}, "metadata TSV")
    return (
        metadata.filter(pl.col(group_col).is_not_null() & (pl.col(group_col) != ""))
        .with_columns(
            pl.col("species").alias("label"),
            pl.col("true_label").cast(pl.Int8, strict=False).alias("true_label"),
            pl.col(group_col).cast(pl.String, strict=False).alias("contrast_pair_id"),
        )
        .select(["label", "species", "true_label", "contrast_pair_id"])
        .sort(["contrast_pair_id", "species"])
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
    feature_limit: int = _FEATURE_HEATMAP_LIMIT,
) -> pl.DataFrame:
    """Build long-form feature heatmap values for grouped species and top features."""
    _require_columns(metadata, {"species", "true_label", group_col}, "metadata TSV")
    _require_columns(feature_importance, {"feature", "importance_mean"}, "feature_importance.tsv")
    if feature_limit < 1:
        raise TreePredictionError("feature heatmap limit must be >= 1")

    species_meta = (
        metadata.filter(pl.col(group_col).is_not_null() & (pl.col(group_col) != ""))
        .select(
            [
                "species",
                "true_label",
                pl.col(group_col).cast(pl.String, strict=False).alias("contrast_pair_id"),
            ]
        )
        .unique("species")
        .sort(["contrast_pair_id", "species"])
    )
    if species_meta.height == 0:
        return _empty_feature_heatmap_annotation()

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

    coef_lookup = _coefficient_lookup(coefficients)
    grid = species_meta.join(top_features, how="cross")
    expression = _load_expression_for_heatmap(
        tpm_path=tpm_path,
        species=list(species_meta.select("species").to_series().to_list()),
        features=list(top_features.select("feature").to_series().to_list()),
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
                "contrast_pair_id",
                "feature_rank",
                "feature",
                "importance_mean",
                "coef_mean",
                "tpm",
                "log2_tpm_plus1",
                "z_score_log2_tpm",
            ]
        )
        .sort(["feature_rank", "contrast_pair_id", "species"])
    )


def build_cv_tree_prediction_annotation(
    *,
    metadata: pl.DataFrame,
    oof_predictions: pl.DataFrame,
    thresholds: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Build ggtree-friendly CV annotation for contrast-pair validation species."""
    _require_columns(oof_predictions, {"fold_id", "species", "label", "prob"}, "prediction_cv.tsv")
    threshold = _cv_derived_threshold(thresholds)
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
        metadata.select(["species", group_col]),
        on="species",
        how="left",
    )
    return (
        joined.filter(pl.col(group_col).is_not_null() & (pl.col(group_col) != ""))
        .with_columns(
            pl.col("species").alias("label"),
            (pl.col("prob") >= threshold).cast(pl.Int8).alias("pred_label"),
            pl.col(group_col).cast(pl.String, strict=False).alias("contrast_pair_id"),
        )
        .select(
            [
                "label",
                "species",
                "true_label",
                "prob",
                "pred_label",
                "uncertainty_std",
                "contrast_pair_id",
                "fold_id",
            ]
        )
        .sort(["contrast_pair_id", "fold_id", "species"])
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
        {"species", "true_label", "prob", "pred_label_cv_derived_threshold"},
        "prediction_external_test.tsv",
    )
    predictions = pred_external_test.with_columns(
        pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
        pl.col("true_label").cast(pl.Int8, strict=False).alias("true_label"),
        pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
        pl.col("pred_label_cv_derived_threshold")
        .cast(pl.Int8, strict=False)
        .alias("pred_label"),
    )
    if "uncertainty_std" not in predictions.columns:
        predictions = predictions.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("uncertainty_std")
        )
    return (
        predictions.join(metadata.select(["species", group_col]), on="species", how="left")
        .with_columns(
            pl.col("species").alias("label"),
            pl.col(group_col).cast(pl.String, strict=False).alias("contrast_pair_id"),
        )
        .select(
            [
                "label",
                "species",
                "true_label",
                "prob",
                "pred_label",
                "uncertainty_std",
                "contrast_pair_id",
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
            "pred_label_cv_derived_threshold",
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
        pl.col("pred_label_cv_derived_threshold")
        .cast(pl.Int8, strict=False)
        .alias("pred_label_cv_derived_threshold"),
    )
    if "uncertainty_std" not in predictions.columns:
        predictions = predictions.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("uncertainty_std")
        )
    return (
        predictions.join(metadata.select(["species", group_col]), on="species", how="left")
        .with_columns(
            pl.col("species").alias("label"),
            pl.col(group_col).cast(pl.String, strict=False).alias("contrast_pair_id"),
        )
        .select(
            [
                "label",
                "species",
                "true_label",
                "prob",
                "pred_label_fixed_threshold",
                "pred_label_cv_derived_threshold",
                "uncertainty_std",
                "contrast_pair_id",
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
        expression = pl.scan_csv(tpm_path, separator="\t")
    except FileNotFoundError as exc:
        raise TreePredictionError(f"Expression file not found: {tpm_path}") from exc
    try:
        schema_columns = set(expression.collect_schema().names())
    except Exception as exc:
        raise TreePredictionError(f"Failed to read expression TSV: {tpm_path}") from exc
    required = {species_col, feature_col, value_col}
    missing = sorted(required - schema_columns)
    if missing:
        raise TreePredictionError(
            f"Missing required columns in expression TSV: {', '.join(missing)}"
        )

    data = (
        expression.select(
            pl.col(species_col).cast(pl.String, strict=False).str.strip_chars().alias("species"),
            pl.col(feature_col).cast(pl.String, strict=False).str.strip_chars().alias("feature"),
            pl.col(value_col).cast(pl.Float64, strict=False).fill_null(0.0).alias("tpm"),
        )
        .filter(pl.col("species").is_in(species) & pl.col("feature").is_in(features))
        .group_by(["species", "feature"])
        .agg(pl.col("tpm").sum())
        .collect()
    )
    if data.filter(pl.col("tpm") < 0.0).height > 0:
        raise TreePredictionError("Expression TSV contains negative TPM values")
    return data


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
            "contrast_pair_id": pl.String,
            "feature_rank": pl.UInt32,
            "feature": pl.String,
            "importance_mean": pl.Float64,
            "coef_mean": pl.Float64,
            "tpm": pl.Float64,
            "log2_tpm_plus1": pl.Float64,
            "z_score_log2_tpm": pl.Float64,
        }
    )


def _require_tree(tree_path: Path) -> None:
    if not tree_path.exists():
        raise TreePredictionError(f"Input tree not found: {tree_path}")
    if not tree_path.is_file():
        raise TreePredictionError(f"Input tree is not a file: {tree_path}")


def _require_columns(frame: pl.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TreePredictionError(f"Missing required columns in {context}: {', '.join(missing)}")


def _cv_derived_threshold(thresholds: pl.DataFrame) -> float:
    _require_columns(thresholds, {"threshold_name", "threshold_value"}, "thresholds.tsv")
    row = thresholds.filter(pl.col("threshold_name") == "cv_derived_threshold")
    if row.height == 0:
        row = thresholds.filter(pl.col("threshold_name") == "fixed_probability_threshold")
    if row.height == 0:
        raise TreePredictionError(
            "thresholds.tsv must contain cv_derived_threshold or fixed_probability_threshold"
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
    height = max(360, 34 + 18 * len(tip_labels))
    width = max(900, 560 + 58 * track_count)
    label_shift = 56 + 44 * track_count
    canvas, axes, _mark = tree.draw(
        width=width,
        height=height,
        layout="r",
        tip_labels=True,
        tip_labels_align=True,
        tip_labels_style={"font-size": "9px", "-toyplot-anchor-shift": f"{label_shift}px"},
        node_sizes=0,
        scale_bar=False,
    )
    axes.show = False
    axes.x.domain.max = max(track_count + 2.0, float(track_count) + 1.2)

    by_species = {str(row["species"]): row for row in annotation.iter_rows(named=True)}
    for track_index, track in enumerate(tracks):
        x = 0.45 + track_index * 0.52
        colors: list[str] = []
        titles: list[str] = []
        for species in tip_labels:
            row = by_species.get(species)
            value = None if row is None else row.get(track)
            colors.append(_track_color(track, value, annotation))
            titles.append(f"{species} {track}={_format_value(value)}")
        axes.scatterplot(
            [x] * len(tip_labels),
            list(range(len(tip_labels))),
            marker="s",
            size=10,
            color=colors,
            title=titles,
        )
        axes.text(
            x,
            len(tip_labels) + 0.35,
            _track_label(track),
            angle=-45,
            style={"font-size": "9px", "text-anchor": "end"},
        )
    axes.text(
        -0.05,
        len(tip_labels) + 1.1,
        title,
        style={"font-size": "15px", "font-weight": "bold", "text-anchor": "start"},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    toytree_module.save(canvas, str(out_path))


def _draw_toytree_feature_heatmap(
    *,
    tree: Any,
    annotation: pl.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    cmap_name: str,
    toytree_module: Any,
) -> None:
    tip_labels = [str(v) for v in tree.get_tip_labels()]
    features = (
        annotation.select(["feature_rank", "feature"])
        .unique("feature")
        .sort("feature_rank")
        .select("feature")
        .to_series()
        .to_list()
    )
    feature_labels = [str(v) for v in features]
    height = max(420, 58 + 18 * len(tip_labels))
    width = max(1040, 620 + 28 * len(feature_labels))
    label_shift = 70 + 24 * len(feature_labels)
    canvas, axes, _mark = tree.draw(
        width=width,
        height=height,
        layout="r",
        tip_labels=True,
        tip_labels_align=True,
        tip_labels_style={"font-size": "9px", "-toyplot-anchor-shift": f"{label_shift}px"},
        node_sizes=0,
        scale_bar=False,
    )
    axes.show = False
    axes.x.domain.max = max(len(feature_labels) + 3.0, 4.0)

    value_lookup: dict[tuple[str, str], object] = {}
    for row in annotation.iter_rows(named=True):
        value_lookup[(str(row["species"]), str(row["feature"]))] = row.get(value_col)
    finite_values = _finite_values(annotation, value_col)
    vmin, vmax = _heatmap_domain(value_col, finite_values)
    for feature_index, feature in enumerate(feature_labels):
        x = 0.45 + feature_index * 0.42
        colors: list[str] = []
        titles: list[str] = []
        for species in tip_labels:
            value = value_lookup.get((species, feature))
            colors.append(_continuous_color(value, vmin=vmin, vmax=vmax, cmap_name=cmap_name))
            titles.append(f"{species} {feature} {value_col}={_format_value(value)}")
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
            feature,
            angle=-65,
            style={"font-size": "7px", "text-anchor": "end"},
        )
    axes.text(
        -0.05,
        len(tip_labels) + 1.1,
        title,
        style={"font-size": "15px", "font-weight": "bold", "text-anchor": "start"},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    toytree_module.save(canvas, str(out_path))


def _track_color(track: str, value: object, annotation: pl.DataFrame) -> str:
    if value is None:
        return _MISSING_COLOR
    if track == "true_label":
        label_value = _int_or_none(value)
        return (
            _MISSING_COLOR
            if label_value is None
            else _LABEL_COLORS.get(label_value, _MISSING_COLOR)
        )
    if track.startswith("pred_label"):
        pred_value = _int_or_none(value)
        return (
            _MISSING_COLOR
            if pred_value is None
            else _PRED_COLORS.get(pred_value, _MISSING_COLOR)
        )
    if track == "prob":
        return _continuous_color(value, vmin=0.0, vmax=1.0, cmap_name="viridis")
    if track == "uncertainty_std":
        numeric_values = _finite_values(annotation, track)
        vmax = max(numeric_values) if numeric_values else 1.0
        return _continuous_color(value, vmin=0.0, vmax=max(vmax, 1e-12), cmap_name="Greys")
    if track in {"contrast_pair_id", "fold_id"}:
        category_values: list[str] = sorted(
            str(v) for v in annotation.select(track).drop_nulls().to_series().to_list()
        )
        unique_values = list(dict.fromkeys(category_values))
        mapping: dict[str, str] = {
            v: _PALETTE[idx % len(_PALETTE)] for idx, v in enumerate(unique_values)
        }
        return mapping.get(str(value), _MISSING_COLOR)
    return _MISSING_COLOR


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


def _track_label(track: str) -> str:
    return {
        "true_label": "true",
        "prob": "prob",
        "pred_label": "pred",
        "pred_label_cv_derived_threshold": "pred_cv",
        "uncertainty_std": "uncert",
        "contrast_pair_id": "contrast",
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
