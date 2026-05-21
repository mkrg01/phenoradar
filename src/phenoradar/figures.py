"""Deterministic SVG figure generation for run/predict/report artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.hashsalt"] = "phenoradar"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["axes.labelsize"] = 7
matplotlib.rcParams["axes.titlesize"] = 7
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.edgecolor"] = "#222222"
matplotlib.rcParams["axes.linewidth"] = 0.6
matplotlib.rcParams["xtick.labelsize"] = 6
matplotlib.rcParams["ytick.labelsize"] = 6
matplotlib.rcParams["xtick.major.width"] = 0.6
matplotlib.rcParams["ytick.major.width"] = 0.6
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["legend.title_fontsize"] = 7
matplotlib.rcParams["grid.color"] = "#e6e6e6"
matplotlib.rcParams["grid.linewidth"] = 0.5


class FigureError(ValueError):
    """Raised when figure generation cannot proceed."""


_FIG_DPI = 100
_NATURE_SINGLE_COLUMN_WIDTH_PX = 350
_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX = 535
_NATURE_DOUBLE_COLUMN_WIDTH_PX = 720
_TITLE_FONTSIZE = 7
_SUBTITLE_FONTSIZE = 6
_LABEL_FONTSIZE = 7
_TICK_FONTSIZE = 6
_ANNOTATION_FONTSIZE = 6
_MONO_FONTSIZE = 6
_GRID_COLOR = "#e6e6e6"
_AXIS_COLOR = "#222222"
_MUTED_TEXT_COLOR = "#666666"
_COLOR_BLUE = "#0072B2"
_COLOR_SKY = "#56B4E9"
_COLOR_GREEN = "#009E73"
_COLOR_ORANGE = "#E69F00"
_COLOR_PURPLE = "#CC79A7"
_TRAIT_NEGATIVE_COLOR = "#d62728"
_TRAIT_POSITIVE_COLOR = "#1f77b4"
_MODEL_SELECTION_SAMPLE_SET_LIMIT = 1
_RETAINED_FEATURE_LIMIT = 40
_DEFAULT_TOP_FEATURES = 30
_FEATURE_IMPORTANCE_TOP_WIDTH_PX = _NATURE_DOUBLE_COLUMN_WIDTH_PX
_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE = _LABEL_FONTSIZE
_FEATURE_IMPORTANCE_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "phenoradar_feature_importance_blues",
    ["#ffffff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"],
)
_COEFFICIENTS_TOP_WIDTH_PX = _NATURE_DOUBLE_COLUMN_WIDTH_PX
_COEFFICIENTS_AXIS_LABEL_FONTSIZE = _LABEL_FONTSIZE
_FEATURE_FILTER_FIGURE_DEFAULT_STAGE_ORDER = (
    "n_features_before",
    "n_features_after_low_prevalence",
    "n_features_after_low_variance",
    "n_features_after_pair_aware",
    "n_features_after_correlation",
    "n_features_after_all",
)
_FEATURE_FILTER_FIGURE_STAGE_LABELS = {
    "n_features_before": "Input",
    "n_features_after_low_prevalence": "Low prevalence",
    "n_features_after_low_variance": "Low variance",
    "n_features_after_pair_aware": "Pair aware",
    "n_features_after_correlation": "Correlation",
    "n_features_after_all": "Final",
}


def _figure_size_inches(width_px: int, height_px: int) -> tuple[float, float]:
    return width_px / _FIG_DPI, height_px / _FIG_DPI


def _fold_axis_width_px(fold_count: int, *, base_px: int = 140, per_fold_px: int = 44) -> int:
    return min(
        _NATURE_DOUBLE_COLUMN_WIDTH_PX,
        max(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, base_px + fold_count * per_fold_px),
    )


def _label_left_margin(labels: list[str], *, width_px: int, fontsize_px: int) -> float:
    label_px = max((len(label) for label in labels), default=0) * fontsize_px * 0.62 + 32
    return min(0.34, max(0.16, label_px / width_px))


def _compact_bottom_margin(height_px: int) -> float:
    return min(0.14, max(0.065, 42 / height_px))


def _save_svg_figure(fig: Figure, out_path: Path) -> None:
    fig.savefig(
        out_path,
        format="svg",
        dpi=_FIG_DPI,
        metadata={"Date": None},
    )
    plt.close(fig)


def _write_message_figure(
    *,
    title: str,
    message: str,
    out_path: Path,
    width_px: int,
    height_px: int,
) -> None:
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, x=0.01, ha="left", fontsize=_TITLE_FONTSIZE)
    ax.axis("off")
    ax.text(
        0.01,
        0.60,
        message,
        transform=ax.transAxes,
        fontsize=_MONO_FONTSIZE,
        fontfamily="monospace",
    )
    _save_svg_figure(fig, out_path)


def _plot_horizontal_values(
    *,
    title: str,
    subtitle: str | None,
    labels: list[str],
    values: list[float],
    out_path: Path,
    color: str,
    width_px: int,
    min_height_px: int,
    row_height_px: int,
    base_height_px: int,
    left_margin: float,
    right_margin: float,
    x_label: str,
    y_tick_fontsize: int,
    value_formatter: Callable[[float], str] | None = None,
) -> None:
    if value_formatter is None:
        def value_formatter(value: float) -> str:
            return f"{value:.8f}"

    max_value = max(values) if values else 1.0
    if np.isclose(max_value, 0.0):
        max_value = 1.0

    height_px = max(min_height_px, base_height_px + len(labels) * row_height_px)
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(labels), dtype=float)
    bars = ax.barh(y_pos, values, color=color, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=y_tick_fontsize, fontfamily="monospace")
    ax.invert_yaxis()

    right_limit = max_value * 1.15
    ax.set_xlim(0.0, right_limit)
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(x_label, fontsize=_LABEL_FONTSIZE)

    value_offset = right_limit * 0.01
    for bar, value in zip(bars, values, strict=True):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            float(value) + value_offset,
            y,
            value_formatter(float(value)),
            va="center",
            ha="left",
            fontsize=_MONO_FONTSIZE,
            fontfamily="monospace",
        )

    fig.subplots_adjust(left=left_margin, right=right_margin, top=0.96, bottom=0.12)
    _save_svg_figure(fig, out_path)


def _format_float(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    return f"{value:.8f}"


def _format_feature_count_label(value: float) -> str:
    rounded = round(value)
    if np.isclose(value, rounded):
        return f"{int(rounded):,}"
    return f"{value:,.1f}"


def _format_n_records_label(records: list[int]) -> str:
    if not records:
        return "n=NA"
    minimum = min(records)
    maximum = max(records)
    if minimum == maximum:
        return f"n={minimum}"
    return f"n={minimum}-{maximum}"


def _feature_filter_figure_stage_order(stage_order: Sequence[str] | None) -> list[str]:
    requested = _FEATURE_FILTER_FIGURE_DEFAULT_STAGE_ORDER if stage_order is None else stage_order
    normalized: list[str] = []
    seen: set[str] = set()
    for stage in requested:
        if stage not in _FEATURE_FILTER_FIGURE_STAGE_LABELS or stage in seen:
            continue
        normalized.append(stage)
        seen.add(stage)
    if normalized:
        return normalized
    return list(_FEATURE_FILTER_FIGURE_DEFAULT_STAGE_ORDER)


def _ellipsize_label(label: str, *, max_chars: int) -> str:
    if len(label) <= max_chars:
        return label
    return f"{label[: max_chars - 3]}..."


def _padded_domain(
    values: list[float], *, include_zero: bool, min_pad: float = 0.05
) -> tuple[float, float]:
    finite = [float(v) for v in values if not np.isnan(v)]
    if not finite:
        return -1.0, 1.0
    lower = min(finite)
    upper = max(finite)
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.1, min_pad)
        return lower - pad, upper + pad
    pad = max((upper - lower) * 0.08, min_pad)
    return lower - pad, upper + pad


def _metric_score(
    y_true: np.ndarray, prob: np.ndarray, threshold: float, metric: str
) -> float:
    pred = (prob >= threshold).astype(int)
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, pred))
    return float(balanced_accuracy_score(y_true, pred))


def _cv_metrics_overview(metrics_cv: pl.DataFrame, out_path: Path) -> None:
    required_columns = {"aggregate_scope", "fold_id", "metric", "metric_value"}
    if not required_columns.issubset(metrics_cv.columns):
        raise FigureError("metrics_cv.tsv schema is invalid for cv_metrics_overview.svg")

    aggregate = metrics_cv.filter(pl.col("aggregate_scope").is_in(["macro", "micro"]))
    aggregate = aggregate.filter(pl.col("fold_id") == "NA")
    if aggregate.height == 0:
        raise FigureError("metrics_cv.tsv does not contain macro/micro aggregate rows")

    macro_lookup: dict[str, float] = {}
    micro_lookup: dict[str, float] = {}
    for row in aggregate.iter_rows(named=True):
        metric = str(row["metric"])
        value_raw = row["metric_value"]
        value = None if value_raw is None else float(value_raw)
        if str(row["aggregate_scope"]) == "macro":
            macro_lookup[metric] = np.nan if value is None else value
        elif str(row["aggregate_scope"]) == "micro":
            micro_lookup[metric] = np.nan if value is None else value

    metric_names = set(macro_lookup.keys()) | set(micro_lookup.keys())
    preferred_order = ["mcc", "balanced_accuracy", "roc_auc", "pr_auc", "brier"]
    metrics = [name for name in preferred_order if name in metric_names] + sorted(
        metric_names - set(preferred_order)
    )
    if not metrics:
        raise FigureError("metrics_cv.tsv does not contain macro/micro aggregate rows")

    values: list[float] = []
    for metric_name in metrics:
        for score in (macro_lookup.get(metric_name, np.nan), micro_lookup.get(metric_name, np.nan)):
            if not np.isnan(score):
                values.append(float(score))
    y_min, y_max = _padded_domain(values, include_zero=True)

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_DOUBLE_COLUMN_WIDTH_PX, 360),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    x_pos = np.arange(len(metrics), dtype=float)
    bar_width = 0.35
    macro_values = np.array([macro_lookup.get(name, np.nan) for name in metrics], dtype=float)
    micro_values = np.array([micro_lookup.get(name, np.nan) for name in metrics], dtype=float)

    macro_mask = np.isfinite(macro_values)
    micro_mask = np.isfinite(micro_values)
    ax.bar(
        x_pos[macro_mask] - bar_width / 2,
        macro_values[macro_mask],
        width=bar_width,
        color=_COLOR_BLUE,
        label="macro",
    )
    ax.bar(
        x_pos[micro_mask] + bar_width / 2,
        micro_values[micro_mask],
        width=bar_width,
        color=_COLOR_ORANGE,
        label="micro",
    )

    for idx, score in enumerate(macro_values):
        if np.isnan(score):
            ax.text(
                x_pos[idx] - bar_width / 2,
                0.0,
                "NA",
                ha="center",
                va="bottom",
                fontsize=_ANNOTATION_FONTSIZE,
                color=_MUTED_TEXT_COLOR,
            )
    for idx, score in enumerate(micro_values):
        if np.isnan(score):
            ax.text(
                x_pos[idx] + bar_width / 2,
                0.0,
                "NA",
                ha="center",
                va="bottom",
                fontsize=_ANNOTATION_FONTSIZE,
                color=_MUTED_TEXT_COLOR,
            )

    ax.set_xlim(-0.75, len(metrics) - 0.25)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=_TICK_FONTSIZE)
    ax.set_xlabel("Metric", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Score", fontsize=_LABEL_FONTSIZE)
    ax.axhline(0.0, color=_AXIS_COLOR, linewidth=0.8)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.18)
    _save_svg_figure(fig, out_path)


def _cv_loss_by_split(loss_by_split_cv: pl.DataFrame, out_path: Path) -> None:
    required_columns = {"fold_id", "split", "metric", "metric_value"}
    if not required_columns.issubset(loss_by_split_cv.columns):
        raise FigureError("loss_by_split_cv.tsv schema is invalid for cv_loss_by_split.svg")

    data = (
        loss_by_split_cv.select(
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("split").cast(pl.String, strict=False).alias("__split"),
            pl.col("metric").cast(pl.String, strict=False).alias("__metric"),
            pl.col("metric_value").cast(pl.Float64, strict=False).alias("__metric_value"),
        )
        .filter(
            pl.col("__fold_id").is_not_null()
            & (pl.col("__fold_id") != "")
            & pl.col("__split").is_not_null()
            & (pl.col("__split") != "")
            & (pl.col("__metric") == "log_loss")
            & pl.col("__metric_value").is_not_null()
            & pl.col("__metric_value").is_finite()
        )
        .sort(["__fold_id", "__split"])
    )
    if data.height == 0:
        raise FigureError("loss_by_split_cv.tsv is empty; cannot draw cv_loss_by_split.svg")

    fold_ids = [str(v) for v in data.select("__fold_id").unique().to_series().to_list()]
    fold_ids = sorted(
        fold_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    split_values = [str(v) for v in data.select("__split").unique().to_series().to_list()]
    split_order = [value for value in ["train", "validation"] if value in split_values]
    split_order.extend(sorted(set(split_values) - set(split_order)))
    if not fold_ids or not split_order:
        raise FigureError("loss_by_split_cv.tsv is empty; cannot draw cv_loss_by_split.svg")

    split_to_color = {
        "train": _COLOR_BLUE,
        "validation": _TRAIT_NEGATIVE_COLOR,
    }
    x_positions = np.arange(len(fold_ids), dtype=float)
    y_values: list[float] = []

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_fold_axis_width_px(len(fold_ids)), 360),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    for split in split_order:
        values: list[float] = []
        for fold_id in fold_ids:
            subset = data.filter((pl.col("__fold_id") == fold_id) & (pl.col("__split") == split))
            if subset.height == 0:
                values.append(np.nan)
                continue
            value = subset.select("__metric_value").to_series().to_list()[0]
            values.append(float(value))
        series = np.array(values, dtype=float)
        mask = np.isfinite(series)
        if np.any(mask):
            ax.plot(
                x_positions[mask],
                series[mask],
                linewidth=1.0,
                marker="o",
                markersize=3.0,
                color=split_to_color.get(split, "#444444"),
                label=split,
            )
            y_values.extend(series[mask].tolist())

    if not y_values:
        raise FigureError("loss_by_split_cv.tsv has no finite rows for cv_loss_by_split.svg")

    y_min, y_max = _padded_domain(y_values, include_zero=False, min_pad=0.01)
    ax.set_xlim(-0.4, len(fold_ids) - 0.6)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(fold_id) for fold_id in fold_ids], fontsize=_TICK_FONTSIZE)
    ax.set_xlabel("CV fold", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("log_loss", fontsize=_LABEL_FONTSIZE)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.16)
    _save_svg_figure(fig, out_path)


def _final_refit_loss_by_split(loss_by_split_final_refit: pl.DataFrame, out_path: Path) -> None:
    required_columns = {"split", "metric", "metric_value"}
    if not required_columns.issubset(loss_by_split_final_refit.columns):
        raise FigureError(
            "loss_by_split_final_refit.tsv schema is invalid for final_refit_loss_by_split.svg"
        )

    data = (
        loss_by_split_final_refit.select(
            pl.col("split").cast(pl.String, strict=False).alias("__split"),
            pl.col("metric").cast(pl.String, strict=False).alias("__metric"),
            pl.col("metric_value").cast(pl.Float64, strict=False).alias("__metric_value"),
        )
        .filter(
            pl.col("__split").is_not_null()
            & (pl.col("__split") != "")
            & (pl.col("__metric") == "log_loss")
            & pl.col("__metric_value").is_not_null()
            & pl.col("__metric_value").is_finite()
        )
        .group_by("__split")
        .agg(pl.col("__metric_value").mean().alias("__metric_value"))
    )
    if data.height == 0:
        raise FigureError(
            "loss_by_split_final_refit.tsv is empty; cannot draw final_refit_loss_by_split.svg"
        )

    split_values = [str(v) for v in data.select("__split").to_series().to_list()]
    split_order = [value for value in ["train", "external_test"] if value in split_values]
    split_order.extend(sorted(set(split_values) - set(split_order)))

    labels: list[str] = []
    values: list[float] = []
    for split in split_order:
        subset = data.filter(pl.col("__split") == split)
        if subset.height == 0:
            continue
        labels.append(split)
        values.append(float(subset.select("__metric_value").to_series().to_list()[0]))

    if not values:
        raise FigureError(
            "loss_by_split_final_refit.tsv is empty; cannot draw final_refit_loss_by_split.svg"
        )

    _plot_horizontal_values(
        title="Final Refit Loss by Split",
        subtitle="Final log_loss on refit train and external_test",
        labels=labels,
        values=values,
        out_path=out_path,
        color=_COLOR_GREEN,
        width_px=_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX,
        min_height_px=170,
        row_height_px=26,
        base_height_px=64,
        left_margin=_label_left_margin(
            labels,
            width_px=_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX,
            fontsize_px=_TICK_FONTSIZE,
        ),
        right_margin=0.96,
        x_label="log_loss",
        y_tick_fontsize=_TICK_FONTSIZE,
    )


def _threshold_selection_curve(
    oof_predictions: pl.DataFrame,
    thresholds: pl.DataFrame,
    selection_metric: str,
    out_path: Path,
) -> None:
    required_oof = {"label", "prob"}
    if not required_oof.issubset(oof_predictions.columns):
        raise FigureError("prediction_cv.tsv schema is invalid for threshold_selection_curve.svg")

    y_true = np.array(oof_predictions.select("label").to_series().to_list(), dtype=int)
    prob = np.array(oof_predictions.select("prob").to_series().to_list(), dtype=float)
    if y_true.size == 0:
        raise FigureError("prediction_cv.tsv is empty; cannot plot threshold selection curve")

    finite_prob = prob[np.isfinite(prob)]
    candidate_thresholds = np.unique(np.concatenate([np.array([0.0, 1.0]), finite_prob]))
    score_points: list[tuple[float, float]] = []
    for threshold in candidate_thresholds.tolist():
        score = _metric_score(y_true, prob, float(threshold), selection_metric)
        if np.isnan(score):
            continue
        score_points.append((float(threshold), float(score)))

    cv_threshold_values = (
        thresholds.filter(pl.col("threshold_name") == "cv_derived_threshold")
        .select("threshold_value")
        .to_series()
        .to_list()
    )
    selected_threshold = float(cv_threshold_values[0]) if len(cv_threshold_values) == 1 else 0.5

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_DOUBLE_COLUMN_WIDTH_PX, 360),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    if score_points:
        score_points = sorted(score_points, key=lambda item: item[0])
        thresholds_plot = np.array([point[0] for point in score_points], dtype=float)
        scores_plot = np.array([point[1] for point in score_points], dtype=float)
        score_min, score_max = _padded_domain(scores_plot.tolist(), include_zero=True)

        ax.plot(thresholds_plot, scores_plot, color=_COLOR_BLUE, linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(score_min, score_max)
        ax.set_xlabel("Threshold", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(f"{selection_metric} score", fontsize=_LABEL_FONTSIZE)
        ax.grid(color=_GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color=_MUTED_TEXT_COLOR, linewidth=0.8)

        if np.isfinite(selected_threshold):
            ax.axvline(
                selected_threshold,
                color=_TRAIT_NEGATIVE_COLOR,
                linewidth=0.8,
                linestyle=(0, (5, 4)),
            )

        selected_score = _metric_score(y_true, prob, selected_threshold, selection_metric)
        if np.isfinite(selected_score) and np.isfinite(selected_threshold):
            ax.scatter(
                [selected_threshold],
                [selected_score],
                color=_TRAIT_NEGATIVE_COLOR,
                s=18,
                zorder=3,
            )
    else:
        ax.axis("off")
        ax.text(
            0.02,
            0.50,
            "No valid threshold scores (all NaN)",
            transform=ax.transAxes,
            fontsize=_LABEL_FONTSIZE,
        )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.15)
    _save_svg_figure(fig, out_path)


def _feature_importance_top(
    feature_importance: pl.DataFrame,
    out_path: Path,
    feature_importance_by_fold: pl.DataFrame | None = None,
    top_features: int = _DEFAULT_TOP_FEATURES,
) -> None:
    required = {"feature", "importance_mean"}
    if not required.issubset(feature_importance.columns):
        raise FigureError("feature_importance.tsv schema is invalid for feature_importance_top.svg")
    if top_features < 1:
        raise FigureError("figures.top_features must be >= 1")

    top = feature_importance.sort(
        by=["importance_mean", "feature"],
        descending=[True, False],
    ).head(top_features)
    if top.height == 0:
        raise FigureError("feature_importance.tsv is empty; cannot draw feature_importance_top.svg")

    features = [str(v) for v in top.select("feature").to_series().to_list()]
    values = [float(v) for v in top.select("importance_mean").to_series().to_list()]
    if feature_importance_by_fold is not None:
        fold_required = {"fold_id", "feature", "importance_mean"}
        if not fold_required.issubset(feature_importance_by_fold.columns):
            raise FigureError(
                "feature_importance_by_fold.tsv schema is invalid for feature_importance_top.svg"
            )
        fold_data = (
            feature_importance_by_fold.select(
                pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
                pl.col("feature").cast(pl.String, strict=False).alias("__feature"),
                pl.col("importance_mean").cast(pl.Float64, strict=False).alias("__value"),
            )
            .filter(
                pl.col("__feature").is_in(features)
                & pl.col("__value").is_not_null()
                & pl.col("__value").is_finite()
            )
        )
        if fold_data.height > 0:
            feature_to_values = {
                feature: np.array(
                    fold_data.filter(pl.col("__feature") == feature)
                    .sort("__fold_id")
                    .select("__value")
                    .to_series()
                    .to_list(),
                    dtype=float,
                )
                for feature in features
            }
            max_value = max(
                [max(values) if values else 0.0]
                + [
                    float(np.max(feature_values))
                    for feature_values in feature_to_values.values()
                    if feature_values.size > 0
                ]
            )
            if np.isclose(max_value, 0.0):
                max_value = 1.0

            height_px = max(240, 70 + len(features) * 18)
            fig, ax = plt.subplots(
                figsize=_figure_size_inches(_FEATURE_IMPORTANCE_TOP_WIDTH_PX, height_px),
                dpi=_FIG_DPI,
            )
            fig.patch.set_facecolor("white")

            y_pos = np.arange(len(features), dtype=float)
            plot_values = [
                feature_to_values[feature]
                if feature_to_values[feature].size > 0
                else np.array([0.0], dtype=float)
                for feature in features
            ]
            ax.boxplot(
                plot_values,
                orientation="horizontal",
                positions=y_pos,
                widths=0.58,
                patch_artist=True,
                showmeans=True,
                boxprops={"facecolor": "#eeeeee", "edgecolor": _MUTED_TEXT_COLOR, "linewidth": 0.8},
                whiskerprops={"color": _MUTED_TEXT_COLOR, "linewidth": 0.8},
                capprops={"color": _MUTED_TEXT_COLOR, "linewidth": 0.8},
                medianprops={"color": "#111111", "linewidth": 0.9},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "#111111",
                    "markeredgecolor": "#111111",
                    "markersize": 3.0,
                    "zorder": 5,
                },
                flierprops={"marker": ""},
            )
            for y, feature in zip(y_pos, features, strict=True):
                feature_values = feature_to_values[feature]
                if feature_values.size == 0:
                    continue
                offsets = np.linspace(-0.16, 0.16, feature_values.size)
                ax.scatter(
                    feature_values,
                    y + offsets,
                    s=12,
                    facecolors="white",
                    edgecolors=_MUTED_TEXT_COLOR,
                    linewidths=0.5,
                    alpha=0.78,
                    zorder=4,
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
            ax.set_ylabel(
                "Orthogroup ID",
                fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE,
            )
            ax.invert_yaxis()
            ax.set_xlim(0.0, max_value * 1.15)
            ax.set_xlabel(
                "Mean feature importance per fold",
                fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE,
            )
            ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
            ax.set_axisbelow(True)
            fig.subplots_adjust(
                left=_label_left_margin(
                    features,
                    width_px=_FEATURE_IMPORTANCE_TOP_WIDTH_PX,
                    fontsize_px=_MONO_FONTSIZE,
                ),
                right=0.985,
                top=0.985,
                bottom=_compact_bottom_margin(height_px),
            )
            _save_svg_figure(fig, out_path)
            return

    max_value = max(values) if values else 1.0
    if np.isclose(max_value, 0.0):
        max_value = 1.0

    height_px = max(220, 60 + len(features) * 18)
    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_FEATURE_IMPORTANCE_TOP_WIDTH_PX, height_px),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(features), dtype=float)
    bars = ax.barh(y_pos, values, color=_MUTED_TEXT_COLOR, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.set_ylabel("Orthogroup ID", fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE)
    ax.invert_yaxis()

    right_limit = max_value * 1.15
    ax.set_xlim(0.0, right_limit)
    ax.set_xlabel(
        "Mean feature importance per fold",
        fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE,
    )
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

    value_offset = right_limit * 0.01
    for bar, value in zip(bars, values, strict=True):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            value + value_offset,
            y,
            f"{value:.8f}",
            va="center",
            ha="left",
            fontsize=_MONO_FONTSIZE,
            fontfamily="monospace",
        )

    fig.subplots_adjust(
        left=_label_left_margin(
            features,
            width_px=_FEATURE_IMPORTANCE_TOP_WIDTH_PX,
            fontsize_px=_MONO_FONTSIZE,
        ),
        right=0.98,
        top=0.985,
        bottom=_compact_bottom_margin(height_px),
    )
    _save_svg_figure(fig, out_path)


def _feature_importance_by_fold_heatmap(
    feature_importance: pl.DataFrame,
    feature_importance_by_fold: pl.DataFrame,
    out_path: Path,
    top_features: int = _DEFAULT_TOP_FEATURES,
) -> None:
    required = {"feature", "importance_mean"}
    if not required.issubset(feature_importance.columns):
        raise FigureError(
            "feature_importance.tsv schema is invalid for feature_importance_by_fold_heatmap.svg"
        )
    fold_required = {"fold_id", "feature", "importance_mean"}
    if not fold_required.issubset(feature_importance_by_fold.columns):
        raise FigureError(
            "feature_importance_by_fold.tsv schema is invalid for "
            "feature_importance_by_fold_heatmap.svg"
        )
    if top_features < 1:
        raise FigureError("figures.top_features must be >= 1")

    top = feature_importance.sort(
        by=["importance_mean", "feature"],
        descending=[True, False],
    ).head(top_features)
    if top.height == 0:
        raise FigureError(
            "feature_importance.tsv is empty; cannot draw feature_importance_by_fold_heatmap.svg"
        )

    features = [str(v) for v in top.select("feature").to_series().to_list()]
    data = (
        feature_importance_by_fold.select(
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("feature").cast(pl.String, strict=False).alias("__feature"),
            pl.col("importance_mean").cast(pl.Float64, strict=False).alias("__value"),
        )
        .filter(
            pl.col("__feature").is_in(features)
            & pl.col("__fold_id").is_not_null()
            & (pl.col("__fold_id") != "")
            & pl.col("__feature").is_not_null()
            & (pl.col("__feature") != "")
            & pl.col("__value").is_not_null()
            & pl.col("__value").is_finite()
        )
        .group_by(["__feature", "__fold_id"])
        .agg(pl.col("__value").mean().alias("__value"))
    )
    if data.height == 0:
        _write_message_figure(
            title="Feature Importance by Fold Heatmap",
            message="No fold-level feature importance rows match the selected top features.",
            out_path=out_path,
            width_px=_FEATURE_IMPORTANCE_TOP_WIDTH_PX,
            height_px=360,
        )
        return

    fold_ids = [str(v) for v in data.select("__fold_id").unique().to_series().to_list()]
    fold_ids = sorted(
        fold_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if not fold_ids:
        _write_message_figure(
            title="Feature Importance by Fold Heatmap",
            message="No CV folds are available in feature_importance_by_fold.tsv.",
            out_path=out_path,
            width_px=_FEATURE_IMPORTANCE_TOP_WIDTH_PX,
            height_px=360,
        )
        return

    feature_index = {feature: idx for idx, feature in enumerate(features)}
    fold_index = {fold_id: idx for idx, fold_id in enumerate(fold_ids)}
    importance_matrix = np.full((len(features), len(fold_ids)), np.nan, dtype=float)
    for row in data.iter_rows(named=True):
        feature_name = str(row["__feature"])
        fold_id = str(row["__fold_id"])
        importance_matrix[feature_index[feature_name], fold_index[fold_id]] = float(
            row["__value"]
        )

    finite_values = importance_matrix[np.isfinite(importance_matrix)]
    max_value = float(np.max(finite_values)) if finite_values.size > 0 else 1.0
    if np.isclose(max_value, 0.0):
        max_value = 1.0

    height_px = max(260, 95 + len(features) * 18)
    width_px = _fold_axis_width_px(len(fold_ids), base_px=260, per_fold_px=44)
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")

    cmap = _FEATURE_IMPORTANCE_HEATMAP_CMAP.copy()
    cmap.set_bad("#ffffff")
    image = ax.imshow(
        np.ma.masked_invalid(importance_matrix),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=max_value,
    )
    ax.set_xticks(np.arange(len(fold_ids), dtype=float))
    ax.set_xticklabels([str(fold_id) for fold_id in fold_ids], fontsize=_TICK_FONTSIZE)
    ax.set_yticks(np.arange(len(features), dtype=float))
    ax.set_yticklabels(features, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.set_xlabel("CV fold", fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Orthogroup ID", fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(np.arange(-0.5, len(fold_ids), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(features), 1.0), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if len(features) <= 12 and len(fold_ids) <= 8:
        for row_index in range(len(features)):
            for col_index in range(len(fold_ids)):
                value = importance_matrix[row_index, col_index]
                if not np.isfinite(value):
                    continue
                color = "#ffffff" if value > max_value * 0.55 else _AXIS_COLOR
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.3g}",
                    ha="center",
                    va="center",
                    fontsize=_MONO_FONTSIZE,
                    color=color,
                    fontfamily="monospace",
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(
        "Mean feature importance per fold",
        rotation=90,
        fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE,
    )
    colorbar.ax.tick_params(labelsize=_TICK_FONTSIZE)
    fig.subplots_adjust(
        left=_label_left_margin(features, width_px=width_px, fontsize_px=_MONO_FONTSIZE),
        right=0.94,
        top=0.98,
        bottom=_compact_bottom_margin(height_px),
    )
    _save_svg_figure(fig, out_path)


def _coefficients_signed_top(
    coefficients: pl.DataFrame,
    out_path: Path,
    coefficients_by_fold: pl.DataFrame | None = None,
    top_features: int = _DEFAULT_TOP_FEATURES,
) -> None:
    required = {"feature", "coef_mean", "method"}
    if not required.issubset(coefficients.columns):
        raise FigureError("coefficients.tsv schema is invalid for coefficients_signed_top.svg")
    if top_features < 1:
        raise FigureError("figures.top_features must be >= 1")

    linear = coefficients.filter(pl.col("method") == "coef_signed").drop_nulls("coef_mean")
    if linear.height == 0:
        return

    top = linear.with_columns(pl.col("coef_mean").abs().alias("__abs_coef")).sort(
        by=["__abs_coef", "feature"],
        descending=[True, False],
    ).head(top_features)

    features = [str(v) for v in top.select("feature").to_series().to_list()]
    values = [float(v) for v in top.select("coef_mean").to_series().to_list()]
    if coefficients_by_fold is not None:
        fold_required = {"fold_id", "feature", "coef_mean", "method"}
        if not fold_required.issubset(coefficients_by_fold.columns):
            raise FigureError(
                "coefficients_by_fold.tsv schema is invalid for coefficients_signed_top.svg"
            )
        fold_data = (
            coefficients_by_fold.filter(pl.col("method") == "coef_signed")
            .select(
                pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
                pl.col("feature").cast(pl.String, strict=False).alias("__feature"),
                pl.col("coef_mean").cast(pl.Float64, strict=False).alias("__value"),
            )
            .filter(
                pl.col("__feature").is_in(features)
                & pl.col("__value").is_not_null()
                & pl.col("__value").is_finite()
            )
        )
        if fold_data.height > 0:
            feature_to_values = {
                feature: np.array(
                    fold_data.filter(pl.col("__feature") == feature)
                    .sort("__fold_id")
                    .select("__value")
                    .to_series()
                    .to_list(),
                    dtype=float,
                )
                for feature in features
            }
            max_abs = max(
                [max(abs(value) for value in values) if values else 0.0]
                + [
                    float(np.max(np.abs(feature_values)))
                    for feature_values in feature_to_values.values()
                    if feature_values.size > 0
                ]
            )
            if np.isclose(max_abs, 0.0):
                max_abs = 1.0

            height_px = max(240, 70 + len(features) * 18)
            fig, ax = plt.subplots(
                figsize=_figure_size_inches(_COEFFICIENTS_TOP_WIDTH_PX, height_px),
                dpi=_FIG_DPI,
            )
            fig.patch.set_facecolor("white")

            y_pos = np.arange(len(features), dtype=float)
            plot_values = [
                feature_to_values[feature]
                if feature_to_values[feature].size > 0
                else np.array([0.0], dtype=float)
                for feature in features
            ]
            ax.boxplot(
                plot_values,
                orientation="horizontal",
                positions=y_pos,
                widths=0.58,
                patch_artist=True,
                showmeans=True,
                boxprops={"facecolor": "#eeeeee", "edgecolor": _MUTED_TEXT_COLOR, "linewidth": 0.8},
                whiskerprops={"color": _MUTED_TEXT_COLOR, "linewidth": 0.8},
                capprops={"color": _MUTED_TEXT_COLOR, "linewidth": 0.8},
                medianprops={"color": "#111111", "linewidth": 0.9},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "#111111",
                    "markeredgecolor": "#111111",
                    "markersize": 3.0,
                    "zorder": 5,
                },
                flierprops={"marker": ""},
            )
            for y, feature in zip(y_pos, features, strict=True):
                feature_values = feature_to_values[feature]
                if feature_values.size == 0:
                    continue
                offsets = np.linspace(-0.16, 0.16, feature_values.size)
                ax.scatter(
                    feature_values,
                    y + offsets,
                    s=12,
                    facecolors="white",
                    edgecolors=_MUTED_TEXT_COLOR,
                    linewidths=0.5,
                    alpha=0.78,
                    zorder=4,
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
            ax.set_ylabel("Orthogroup ID", fontsize=_COEFFICIENTS_AXIS_LABEL_FONTSIZE)
            ax.invert_yaxis()
            limit = max_abs * 1.15
            ax.set_xlim(-limit, limit)
            ax.set_xlabel(
                "Mean signed coefficient per fold",
                fontsize=_COEFFICIENTS_AXIS_LABEL_FONTSIZE,
            )
            ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.axvline(0.0, color=_MUTED_TEXT_COLOR, linewidth=0.8)
            fig.subplots_adjust(
                left=_label_left_margin(
                    features,
                    width_px=_COEFFICIENTS_TOP_WIDTH_PX,
                    fontsize_px=_MONO_FONTSIZE,
                ),
                right=0.985,
                top=0.985,
                bottom=_compact_bottom_margin(height_px),
            )
            _save_svg_figure(fig, out_path)
            return

    max_abs = max(abs(v) for v in values) if values else 1.0
    if np.isclose(max_abs, 0.0):
        max_abs = 1.0

    height_px = max(220, 60 + len(features) * 18)
    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_COEFFICIENTS_TOP_WIDTH_PX, height_px),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(features), dtype=float)
    bars = ax.barh(y_pos, values, color=_MUTED_TEXT_COLOR, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.set_ylabel("Orthogroup ID", fontsize=_COEFFICIENTS_AXIS_LABEL_FONTSIZE)
    ax.invert_yaxis()

    limit = max_abs * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(
        "Mean signed coefficient per fold",
        fontsize=_COEFFICIENTS_AXIS_LABEL_FONTSIZE,
    )
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color=_MUTED_TEXT_COLOR, linewidth=0.8)
    value_offset = limit * 0.03
    for bar, value in zip(bars, values, strict=True):
        y = bar.get_y() + bar.get_height() / 2
        if value >= 0:
            text_x = value + value_offset
            align = "left"
        else:
            text_x = value - value_offset
            align = "right"
        ax.text(
            text_x,
            y,
            f"{value:.8f}",
            va="center",
            ha=align,
            fontsize=_MONO_FONTSIZE,
            fontfamily="monospace",
        )

    fig.subplots_adjust(
        left=_label_left_margin(
            features,
            width_px=_COEFFICIENTS_TOP_WIDTH_PX,
            fontsize_px=_MONO_FONTSIZE,
        ),
        right=0.98,
        top=0.985,
        bottom=_compact_bottom_margin(height_px),
    )
    _save_svg_figure(fig, out_path)


def _predict_probability_distribution(pred_predict: pl.DataFrame, out_path: Path) -> None:
    if "prob" not in pred_predict.columns:
        raise FigureError(
            "prediction_inference.tsv schema is invalid for predict_probability_distribution.svg"
        )

    probs = np.array(pred_predict.select("prob").to_series().to_list(), dtype=float)
    if probs.size == 0:
        raise FigureError(
            "prediction_inference.tsv is empty; cannot draw predict_probability_distribution.svg"
        )

    counts, _bins = np.histogram(probs, bins=10, range=(0.0, 1.0))
    max_count = int(counts.max()) if counts.size > 0 else 1
    if max_count < 1:
        max_count = 1

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 320),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    bin_starts = np.arange(10, dtype=float) / 10.0
    bars = ax.bar(bin_starts, counts.tolist(), width=0.08, align="edge", color=_COLOR_SKY)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max_count * 1.15)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_xlabel("Probability", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Count", fontsize=_LABEL_FONTSIZE)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

    for bar, count in zip(bars, counts.tolist(), strict=True):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(
            x,
            y + max_count * 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontsize=_MONO_FONTSIZE,
            fontfamily="monospace",
        )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.16)
    _save_svg_figure(fig, out_path)


def _predict_uncertainty(pred_predict: pl.DataFrame, out_path: Path, *, required: bool) -> None:
    if "uncertainty_std" not in pred_predict.columns:
        if required:
            raise FigureError(
                "prediction_inference.tsv is missing required column uncertainty_std "
                "for predict_uncertainty.svg"
            )
        return

    data = pred_predict.select("species", "uncertainty_std").sort(
        by=["uncertainty_std", "species"],
        descending=[True, False],
    ).head(30)
    if data.height == 0:
        if required:
            raise FigureError(
                "prediction_inference.tsv is empty; cannot draw predict_uncertainty.svg"
            )
        return

    species = [str(v) for v in data.select("species").to_series().to_list()]
    values = [float(v) for v in data.select("uncertainty_std").to_series().to_list()]
    max_val = max(values) if values else 1.0
    if np.isclose(max_val, 0.0):
        max_val = 1.0

    height_px = max(220, 70 + len(species) * 18)
    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_DOUBLE_COLUMN_WIDTH_PX, height_px),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(species), dtype=float)
    bars = ax.barh(y_pos, values, color=_COLOR_PURPLE, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(species, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.invert_yaxis()

    right_limit = max_val * 1.15
    ax.set_xlim(0.0, right_limit)
    ax.set_xlabel("uncertainty_std", fontsize=_LABEL_FONTSIZE)
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

    value_offset = right_limit * 0.01
    for bar, value in zip(bars, values, strict=True):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            value + value_offset,
            y,
            f"{value:.8f}",
            va="center",
            ha="left",
            fontsize=_MONO_FONTSIZE,
            fontfamily="monospace",
        )

    fig.subplots_adjust(
        left=_label_left_margin(
            species,
            width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
            fontsize_px=_MONO_FONTSIZE,
        ),
        right=0.98,
        top=0.98,
        bottom=0.12,
    )
    _save_svg_figure(fig, out_path)


def _deterministic_offsets(count: int, spread: float) -> np.ndarray:
    if count <= 1:
        return np.array([0.0], dtype=float)
    return np.linspace(-spread, spread, num=count, dtype=float)


def _binary_trait_color_map(
    traits: list[int], *, source_table_name: str, figure_name: str
) -> dict[int, str]:
    non_binary = sorted({int(value) for value in traits if int(value) not in (0, 1)})
    if non_binary:
        values = ", ".join(str(value) for value in non_binary)
        raise FigureError(
            f"{source_table_name} contains non-binary trait values for {figure_name}: {values}"
        )
    return {
        0: _TRAIT_NEGATIVE_COLOR,
        1: _TRAIT_POSITIVE_COLOR,
    }


def _species_probability_by_trait(
    *,
    predictions: pl.DataFrame,
    trait_col: str,
    trait_name: str = "trait",
    out_path: Path,
    title: str,
    subtitle: str,
    source_table_name: str,
    figure_name: str,
) -> None:
    required = {"species", trait_col, "prob"}
    if not required.issubset(predictions.columns):
        raise FigureError(f"{source_table_name} schema is invalid for {figure_name}")

    data = (
        predictions.select(
            pl.col("species").cast(pl.String, strict=False).alias("__species"),
            pl.col(trait_col).cast(pl.Int64, strict=False).alias("__trait"),
            pl.col("prob").cast(pl.Float64, strict=False).alias("__prob"),
        )
        .filter(
            pl.col("__species").is_not_null()
            & (pl.col("__species") != "")
            & pl.col("__trait").is_not_null()
            & pl.col("__prob").is_not_null()
            & pl.col("__prob").is_finite()
        )
        .sort(["__trait", "__species"])
    )
    if data.height == 0:
        raise FigureError(f"{source_table_name} is empty; cannot draw {figure_name}")

    traits = [int(v) for v in data.select("__trait").unique().sort("__trait").to_series().to_list()]
    if not traits:
        raise FigureError(f"{source_table_name} is empty; cannot draw {figure_name}")

    trait_to_color = _binary_trait_color_map(
        traits,
        source_table_name=source_table_name,
        figure_name=figure_name,
    )

    group_probs: list[list[float]] = []
    group_counts: list[int] = []
    x_labels: list[str] = []
    positions = np.arange(1, len(traits) + 1, dtype=float)
    for trait in traits:
        trait_df = data.filter(pl.col("__trait") == trait).sort(["__prob", "__species"])
        probs = [float(v) for v in trait_df.select("__prob").to_series().to_list()]
        if not probs:
            continue
        group_probs.append(probs)
        group_counts.append(len(probs))
        x_labels.append(str(trait))

    if not group_probs:
        raise FigureError(f"{source_table_name} is empty; cannot draw {figure_name}")

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 390),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    box = ax.boxplot(
        group_probs,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        manage_ticks=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": _AXIS_COLOR,
            "markeredgecolor": _AXIS_COLOR,
            "markersize": 3.0,
        },
        medianprops={"linewidth": 0.9, "color": _AXIS_COLOR},
        whiskerprops={"linewidth": 0.8, "color": _MUTED_TEXT_COLOR},
        capprops={"linewidth": 0.8, "color": _MUTED_TEXT_COLOR},
    )
    for idx, patch in enumerate(box["boxes"]):
        color = trait_to_color[traits[idx]]
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)

    for idx, trait in enumerate(traits):
        trait_df = data.filter(pl.col("__trait") == trait).sort(["__prob", "__species"])
        probs_array = np.array(trait_df.select("__prob").to_series().to_list(), dtype=float)
        offsets = _deterministic_offsets(probs_array.size, 0.17)
        x_values = np.full(probs_array.shape[0], positions[idx], dtype=float) + offsets
        ax.scatter(
            x_values,
            probs_array,
            s=18,
            color=trait_to_color[trait],
            edgecolors="white",
            linewidths=0.4,
            alpha=0.78,
            zorder=3,
        )

    for idx, count in enumerate(group_counts):
        ax.text(
            positions[idx],
            1.02,
            f"n={count}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=_ANNOTATION_FONTSIZE,
            color=_MUTED_TEXT_COLOR,
            clip_on=False,
        )

    ax.set_xlim(0.5, len(traits) + 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=_TICK_FONTSIZE)
    ax.set_xlabel(trait_name, fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted probability", fontsize=_LABEL_FONTSIZE)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.16)
    _save_svg_figure(fig, out_path)


def _cv_fold_trait_probability(
    oof_predictions: pl.DataFrame, out_path: Path, *, trait_name: str = "trait"
) -> None:
    required = {"fold_id", "label", "prob"}
    if not required.issubset(oof_predictions.columns):
        raise FigureError("prediction_cv.tsv schema is invalid for cv_fold_trait_probability.svg")

    data = (
        oof_predictions.select(
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("label").cast(pl.Int64, strict=False).alias("__trait"),
            pl.col("prob").cast(pl.Float64, strict=False).alias("__prob"),
        )
        .filter(
            pl.col("__fold_id").is_not_null()
            & (pl.col("__fold_id") != "")
            & pl.col("__trait").is_not_null()
            & pl.col("__prob").is_not_null()
            & pl.col("__prob").is_finite()
        )
        .sort(["__fold_id", "__trait", "__prob"])
    )
    if data.height == 0:
        raise FigureError("prediction_cv.tsv is empty; cannot draw cv_fold_trait_probability.svg")

    fold_ids = [str(v) for v in data.select("__fold_id").unique().to_series().to_list()]
    fold_ids = sorted(
        fold_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    traits = [int(v) for v in data.select("__trait").unique().sort("__trait").to_series().to_list()]
    if not fold_ids or not traits:
        raise FigureError("prediction_cv.tsv is empty; cannot draw cv_fold_trait_probability.svg")

    fold_centers = np.arange(1, len(fold_ids) + 1, dtype=float)
    if len(traits) == 1:
        trait_offsets = np.array([0.0], dtype=float)
    else:
        trait_offsets = np.linspace(-0.25, 0.25, num=len(traits), dtype=float)
    box_width = min(0.36, 0.72 / max(1, len(traits)))

    trait_to_color = _binary_trait_color_map(
        traits,
        source_table_name="prediction_cv.tsv",
        figure_name="cv_fold_trait_probability.svg",
    )

    width_px = _fold_axis_width_px(len(fold_ids), base_px=140, per_fold_px=44)
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, 390), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")

    for fold_idx, center in enumerate(fold_centers):
        if fold_idx % 2 == 0:
            ax.axvspan(center - 0.48, center + 0.48, color="#f7f7f7", zorder=0)

    for boundary in np.arange(1.5, len(fold_ids), 1.0):
        ax.axvline(boundary, color="#d9d9d9", linewidth=0.5, zorder=1)

    for trait_idx, trait in enumerate(traits):
        values_for_box: list[list[float]] = []
        positions_for_box: list[float] = []
        for fold_idx, fold_id in enumerate(fold_ids):
            subset = data.filter((pl.col("__fold_id") == fold_id) & (pl.col("__trait") == trait))
            probs = np.array(subset.select("__prob").to_series().to_list(), dtype=float)
            if probs.size == 0:
                continue
            x_position = fold_centers[fold_idx] + trait_offsets[trait_idx]
            values_for_box.append(probs.tolist())
            positions_for_box.append(float(x_position))

            offsets = _deterministic_offsets(probs.size, min(0.08, box_width * 0.42))
            ax.scatter(
                np.full(probs.shape[0], x_position, dtype=float) + offsets,
                probs,
                s=14,
                color=trait_to_color[trait],
                edgecolors="white",
                linewidths=0.4,
                alpha=0.75,
                zorder=3,
            )

        if not values_for_box:
            continue

        box = ax.boxplot(
            values_for_box,
            positions=positions_for_box,
            widths=box_width,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            manage_ticks=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": _AXIS_COLOR,
                "markeredgecolor": _AXIS_COLOR,
                "markersize": 3.0,
            },
            medianprops={"linewidth": 0.9, "color": _AXIS_COLOR},
            whiskerprops={"linewidth": 0.8, "color": _MUTED_TEXT_COLOR},
            capprops={"linewidth": 0.8, "color": _MUTED_TEXT_COLOR},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(trait_to_color[trait])
            patch.set_alpha(0.28)
            patch.set_edgecolor(trait_to_color[trait])
            patch.set_linewidth(0.8)

    ax.set_xlim(0.52, len(fold_ids) + 0.48)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(fold_centers)
    ax.set_xticklabels([str(fold_id) for fold_id in fold_ids], fontsize=_TICK_FONTSIZE)
    ax.set_xlabel("CV fold", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted probability", fontsize=_LABEL_FONTSIZE)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    legend_handles = [
        Patch(
            facecolor=to_rgba(trait_to_color[trait], 0.28),
            edgecolor=trait_to_color[trait],
            linewidth=0.8,
            label=str(trait),
        )
        for trait in traits
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        borderaxespad=0.0,
        ncol=min(len(legend_handles), 4),
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#dddddd",
        title=trait_name,
        handlelength=1.1,
        handleheight=0.6,
        columnspacing=0.9,
        borderpad=0.25,
        labelspacing=0.2,
    )

    fig.subplots_adjust(left=0.095, right=0.995, top=0.89, bottom=0.14)
    _save_svg_figure(fig, out_path)


def _roc_pr_curves_cv(oof_predictions: pl.DataFrame, out_path: Path) -> None:
    required = {"fold_id", "label", "prob"}
    if not required.issubset(oof_predictions.columns):
        raise FigureError("prediction_cv.tsv schema is invalid for roc_pr_curves_cv.svg")
    if oof_predictions.height == 0:
        raise FigureError("prediction_cv.tsv is empty; cannot draw roc_pr_curves_cv.svg")

    y_true = np.array(oof_predictions.select("label").to_series().to_list(), dtype=int)
    prob = np.array(oof_predictions.select("prob").to_series().to_list(), dtype=float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        raise FigureError("roc_pr_curves_cv.svg could not be drawn (no folds with both labels)")

    fpr, tpr, _ = roc_curve(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    recall_order = np.argsort(recall)
    recall_plot = np.asarray(recall[recall_order], dtype=float)
    precision_plot = np.asarray(precision[recall_order], dtype=float)
    roc_auc = float(roc_auc_score(y_true, prob))
    pr_auc = float(average_precision_score(y_true, prob))
    prevalence = float(np.mean(y_true))

    fig, (ax_roc, ax_pr) = plt.subplots(
        1,
        2,
        figsize=_figure_size_inches(_NATURE_DOUBLE_COLUMN_WIDTH_PX, 340),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    ax_roc.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.7, linestyle=(0, (4, 4)))
    ax_roc.plot(fpr, tpr, color=_COLOR_BLUE, linewidth=1.0)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("False Positive Rate", fontsize=_LABEL_FONTSIZE)
    ax_roc.set_ylabel("True Positive Rate", fontsize=_LABEL_FONTSIZE)
    ax_roc.grid(color=_GRID_COLOR, linewidth=0.5)
    ax_roc.set_axisbelow(True)
    ax_roc.set_title(f"ROC AUC={roc_auc:.6f}", fontsize=_LABEL_FONTSIZE)

    ax_pr.axhline(prevalence, color="#999999", linewidth=0.7, linestyle=(0, (4, 4)))
    ax_pr.plot(recall_plot, precision_plot, color=_COLOR_ORANGE, linewidth=1.0)
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_xlabel("Recall", fontsize=_LABEL_FONTSIZE)
    ax_pr.set_ylabel("Precision", fontsize=_LABEL_FONTSIZE)
    ax_pr.grid(color=_GRID_COLOR, linewidth=0.5)
    ax_pr.set_axisbelow(True)
    ax_pr.set_title(
        f"PR AUC={pr_auc:.6f}, positive_rate={prevalence:.6f}",
        fontsize=_LABEL_FONTSIZE,
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14, wspace=0.28)
    _save_svg_figure(fig, out_path)


def _report_metric_ranking(report_ranking: pl.DataFrame, out_path: Path) -> None:
    required = {"rank", "run_id", "metric_value"}
    if not required.issubset(report_ranking.columns):
        raise FigureError("report_ranking.tsv schema is invalid for report_metric_ranking.svg")

    if report_ranking.height == 0:
        _write_message_figure(
            title="Report Metric Ranking",
            message="No ranked runs",
            out_path=out_path,
            width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
            height_px=320,
        )
        return

    top = report_ranking.sort("rank").head(30)
    run_ids = [str(v) for v in top.select("run_id").to_series().to_list()]
    values = [float(v) for v in top.select("metric_value").to_series().to_list()]

    _plot_horizontal_values(
        title="Report Metric Ranking",
        subtitle=None,
        labels=run_ids,
        values=values,
        out_path=out_path,
        color=_COLOR_BLUE,
        width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
        min_height_px=260,
        row_height_px=24,
        base_height_px=80,
        left_margin=_label_left_margin(
            run_ids,
            width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
            fontsize_px=_MONO_FONTSIZE,
        ),
        right_margin=0.96,
        x_label="metric_value",
        y_tick_fontsize=_MONO_FONTSIZE,
    )


def _report_metric_comparison(report_runs: pl.DataFrame, out_path: Path) -> None:
    required = {"run_id", "metric_value", "start_time"}
    if not required.issubset(report_runs.columns):
        raise FigureError("report_runs.tsv schema is invalid for report_metric_comparison.svg")

    comparable = report_runs.drop_nulls("metric_value").sort(
        by=["metric_value", "start_time", "run_id"],
        descending=[True, False, False],
    ).head(30)
    if comparable.height == 0:
        _write_message_figure(
            title="Report Metric Comparison",
            message="No comparable runs with metric values",
            out_path=out_path,
            width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
            height_px=320,
        )
        return

    run_ids = [str(v) for v in comparable.select("run_id").to_series().to_list()]
    values = [float(v) for v in comparable.select("metric_value").to_series().to_list()]

    _plot_horizontal_values(
        title="Report Metric Comparison",
        subtitle=None,
        labels=run_ids,
        values=values,
        out_path=out_path,
        color=_COLOR_ORANGE,
        width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
        min_height_px=260,
        row_height_px=24,
        base_height_px=80,
        left_margin=_label_left_margin(
            run_ids,
            width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
            fontsize_px=_MONO_FONTSIZE,
        ),
        right_margin=0.96,
        x_label="metric_value",
        y_tick_fontsize=_MONO_FONTSIZE,
    )


def _report_stage_breakdown(report_runs: pl.DataFrame, out_path: Path) -> None:
    if "execution_stage" not in report_runs.columns:
        raise FigureError("report_runs.tsv schema is invalid for report_stage_breakdown.svg")

    counts = (
        report_runs.group_by("execution_stage")
        .len()
        .sort(by=["len", "execution_stage"], descending=[True, False])
    )
    if counts.height <= 1:
        return

    stages = [str(v) for v in counts.select("execution_stage").to_series().to_list()]
    values = [float(v) for v in counts.select("len").to_series().to_list()]

    def _as_int(value: float) -> str:
        return str(int(round(value)))

    _plot_horizontal_values(
        title="Report Stage Breakdown",
        subtitle=None,
        labels=stages,
        values=values,
        out_path=out_path,
        color=_COLOR_GREEN,
        width_px=_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX,
        min_height_px=220,
        row_height_px=40,
        base_height_px=80,
        left_margin=_label_left_margin(
            stages,
            width_px=_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX,
            fontsize_px=_TICK_FONTSIZE,
        ),
        right_margin=0.96,
        x_label="count",
        y_tick_fontsize=_TICK_FONTSIZE,
        value_formatter=_as_int,
    )


def _summarize_model_selection_trials_for_figure(
    model_selection_trials: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "metric_name",
        "metric_value",
    }
    if not required.issubset(model_selection_trials.columns):
        raise FigureError(
            "model_selection_trials.tsv schema is invalid for model_selection_trials.svg"
        )

    scored = model_selection_trials
    if "params_json" not in scored.columns:
        scored = scored.with_columns(pl.lit("{}").alias("params_json"))

    scored = scored.with_columns(
        pl.col("metric_value").cast(pl.Float64, strict=False).alias("_metric_value_raw")
    ).with_columns(
        pl.when(pl.col("_metric_value_raw").is_nan())
        .then(None)
        .otherwise(pl.col("_metric_value_raw"))
        .alias("_metric_value_valid")
    )
    return scored.group_by(
        ["fold_id", "sample_set_id", "candidate_index", "metric_name", "params_json"]
    ).agg(
        [
            pl.len().alias("n_inner_folds"),
            pl.col("_metric_value_valid").count().alias("n_valid_inner_folds"),
            pl.col("_metric_value_valid").mean().alias("metric_value_mean"),
            pl.col("_metric_value_valid").std(ddof=0).alias("metric_value_std"),
        ]
    ).sort(["fold_id", "sample_set_id", "candidate_index"])


def _params_dict(params_json: str | None) -> dict[str, Any] | None:
    if params_json is None:
        return None
    raw = params_json.strip()
    if raw == "":
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _varying_param_keys(params_json_values: list[str | None]) -> set[str]:
    dicts = [_params_dict(value) for value in params_json_values]
    all_keys = sorted({key for item in dicts if item is not None for key in item})
    varying: set[str] = set()
    for key in all_keys:
        observed: set[str] = set()
        for item in dicts:
            if item is None or key not in item:
                observed.add("__MISSING__")
                continue
            value = item[key]
            observed.add(
                json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            )
        if len(observed) > 1:
            varying.add(key)
    return varying


def _compact_params_label(
    params_json: str | None,
    *,
    include_keys: set[str] | None = None,
) -> str:
    if params_json is None:
        return "{}"
    raw = params_json.strip()
    if raw == "":
        return "{}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict):
        if include_keys is not None:
            parsed = {
                key: value
                for key, value in sorted(parsed.items())
                if key in include_keys
            }
        if not parsed:
            return "{}"
        return json.dumps(parsed, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if parsed is None:
        return "null"
    return json.dumps(parsed, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _model_selection_trials_summary_panels(
    model_selection_trials_summary: pl.DataFrame,
    out_path: Path,
    *,
    max_sample_sets_per_fold: int,
) -> None:
    required = {
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "metric_name",
        "metric_value_mean",
        "metric_value_std",
    }
    if not required.issubset(model_selection_trials_summary.columns):
        raise FigureError(
            "model_selection_trials_summary.tsv schema is invalid for model_selection_trials.svg"
        )

    summary = model_selection_trials_summary
    if "params_json" not in summary.columns:
        summary = summary.with_columns(pl.lit("{}").alias("params_json"))

    data = (
        summary.select(
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("sample_set_id").cast(pl.Int64, strict=False).alias("__sample_set_id"),
            pl.col("candidate_index").cast(pl.Int64, strict=False).alias("__candidate_index"),
            pl.col("metric_name").cast(pl.String, strict=False).alias("__metric_name"),
            pl.col("metric_value_mean").cast(pl.Float64, strict=False).alias("__mean"),
            pl.col("metric_value_std").cast(pl.Float64, strict=False).alias("__std"),
            pl.col("params_json").cast(pl.String, strict=False).alias("__params_json"),
        )
        .with_columns(
            pl.when(
                pl.col("__std").is_null() | pl.col("__std").is_nan() | (pl.col("__std") < 0.0)
            )
            .then(0.0)
            .otherwise(pl.col("__std"))
            .alias("__std_plot")
        )
        .filter(
            pl.col("__fold_id").is_not_null()
            & (pl.col("__fold_id") != "")
            & pl.col("__sample_set_id").is_not_null()
            & pl.col("__candidate_index").is_not_null()
            & pl.col("__metric_name").is_not_null()
            & (pl.col("__metric_name") != "")
            & pl.col("__mean").is_not_null()
            & pl.col("__mean").is_finite()
        )
    )
    if data.height == 0:
        return

    if max_sample_sets_per_fold < 1:
        raise FigureError("max_sample_sets_per_fold must be >= 1")

    fold_ids = [str(v) for v in data.select("__fold_id").unique().to_series().to_list()]
    fold_ids = sorted(
        fold_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )

    sample_sets_by_fold: dict[str, list[int]] = {}
    per_fold_total: dict[str, int] = {}
    n_rows = 0
    for fold_id in fold_ids:
        sample_set_ids = sorted(
            int(v)
            for v in data.filter(pl.col("__fold_id") == fold_id)
            .select("__sample_set_id")
            .unique()
            .to_series()
            .to_list()
        )
        per_fold_total[fold_id] = len(sample_set_ids)
        selected_sample_set_ids = sample_set_ids[:max_sample_sets_per_fold]
        sample_sets_by_fold[fold_id] = selected_sample_set_ids
        n_rows = max(n_rows, len(selected_sample_set_ids))
    if n_rows == 0:
        return

    panels: list[dict[str, Any]] = []
    x_values: list[float] = []
    max_candidates = 1
    max_label_length = 1
    for fold_id in fold_ids:
        selected_sample_set_ids = sample_sets_by_fold[fold_id]
        if not selected_sample_set_ids:
            continue
        sample_set_id = selected_sample_set_ids[0]
        panel_data = data.filter(
            (pl.col("__fold_id") == fold_id) & (pl.col("__sample_set_id") == sample_set_id)
        ).sort("__candidate_index")
        if panel_data.height == 0:
            continue

        candidates = [int(v) for v in panel_data.select("__candidate_index").to_series().to_list()]
        means = np.array(panel_data.select("__mean").to_series().to_list(), dtype=float)
        stds = np.array(panel_data.select("__std_plot").to_series().to_list(), dtype=float)
        params_json_values = [
            None if value is None else str(value)
            for value in panel_data.select("__params_json").to_series().to_list()
        ]
        varying_keys = _varying_param_keys(params_json_values)
        params_labels = [
            _compact_params_label(value, include_keys=varying_keys)
            for value in params_json_values
        ]
        y_labels = [
            _ellipsize_label(f"{candidate}: {params_label}", max_chars=72)
            for candidate, params_label in zip(candidates, params_labels, strict=True)
        ]

        max_candidates = max(max_candidates, len(candidates))
        max_label_length = max(max_label_length, max(len(label) for label in y_labels))
        for mean_value, std_value in zip(means.tolist(), stds.tolist(), strict=True):
            x_values.extend([mean_value - std_value, mean_value + std_value])

        panels.append(
            {
                "fold_id": fold_id,
                "sample_set_id": sample_set_id,
                "candidates": candidates,
                "means": means,
                "stds": stds,
                "y_labels": y_labels,
            }
        )

    if not panels:
        return

    x_min, x_max = _padded_domain(x_values, include_zero=True)

    n_panels = len(panels)
    max_cols = min(5, n_panels)
    n_cols = min(
        range(1, max_cols + 1),
        key=lambda cols: (
            abs((cols / int(np.ceil(n_panels / cols))) - 1.6)
            + (int(np.ceil(n_panels / cols)) * cols - n_panels) * 0.15,
            int(np.ceil(n_panels / cols)) * cols - n_panels,
        ),
    )
    n_rows = int(np.ceil(n_panels / n_cols))

    panel_width_px = 340
    left_label_px = min(640, max(190, 40 + int(max_label_length * 4)))
    right_pad_px = 24
    fig_width_px = left_label_px + panel_width_px * n_cols + right_pad_px

    panel_height_px = max(210, 86 + max_candidates * 18)
    header_px = 26
    footer_px = 38
    fig_height_px = header_px + panel_height_px * n_rows + footer_px

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_figure_size_inches(fig_width_px, fig_height_px),
        dpi=_FIG_DPI,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    metric_names = sorted(
        {str(v) for v in data.select("__metric_name").unique().to_series().to_list()}
    )
    if len(metric_names) == 1:
        metric_axis_label = {
            "mcc": "MCC",
            "balanced_accuracy": "Balanced Accuracy",
            "log_loss": "Log Loss",
        }.get(metric_names[0], metric_names[0].replace("_", " ").title())
    else:
        metric_axis_label = "Score"
    for panel_index, panel in enumerate(panels):
        row_index, col_index = divmod(panel_index, n_cols)
        ax = axes[row_index][col_index]
        y_pos = np.arange(len(panel["candidates"]), dtype=float)
        ax.errorbar(
            panel["means"],
            y_pos,
            xerr=panel["stds"],
            fmt="o",
            color=_COLOR_BLUE,
            ecolor=_COLOR_SKY,
            elinewidth=0.8,
            capsize=2.5,
            markersize=3.0,
            markeredgecolor=_COLOR_BLUE,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(panel["y_labels"], fontsize=6, fontfamily="monospace")
        ax.tick_params(axis="y", pad=1.5)
        ax.invert_yaxis()
        ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.axvline(0.0, color=_MUTED_TEXT_COLOR, linewidth=0.8)
        ax.set_title(str(panel["fold_id"]), fontsize=_LABEL_FONTSIZE, pad=5.0)
        if col_index == 0:
            ax.set_ylabel("candidate_index:params", fontsize=_LABEL_FONTSIZE)
        ax.set_xlabel(metric_axis_label, fontsize=_LABEL_FONTSIZE, labelpad=3.0)

    for panel_index in range(n_panels, n_rows * n_cols):
        row_index, col_index = divmod(panel_index, n_cols)
        axes[row_index][col_index].axis("off")

    left_margin = left_label_px / fig_width_px
    right_margin = 1.0 - (right_pad_px / fig_width_px)
    top_margin = min(0.98, 1.0 - (header_px / fig_height_px) + 0.01)
    bottom_margin = footer_px / fig_height_px
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.32,
        hspace=0.55,
    )
    _save_svg_figure(fig, out_path)


def _feature_filter_funnel(
    feature_filter_counts_summary: pl.DataFrame,
    out_path: Path,
    *,
    stage_order: Sequence[str] | None = None,
) -> None:
    required = {
        "scope",
        "stage",
        "n_records",
        "n_features_min",
        "n_features_q1",
        "n_features_median",
        "n_features_q3",
        "n_features_max",
    }
    if not required.issubset(feature_filter_counts_summary.columns):
        raise FigureError(
            "feature_filter_counts_summary.tsv schema is invalid for feature_filter_funnel.svg"
        )
    stage_order = _feature_filter_figure_stage_order(stage_order)
    data = feature_filter_counts_summary.filter(pl.col("stage").is_in(stage_order))
    if data.height == 0:
        _write_message_figure(
            title="Feature Filter Funnel",
            message="No feature-filter summary rows are available.",
            out_path=out_path,
            width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
            height_px=320,
        )
        return

    scopes = sorted(str(v) for v in data.select("scope").unique().to_series().to_list())
    x_positions = np.arange(len(stage_order), dtype=float)
    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_DOUBLE_COLUMN_WIDTH_PX, 340),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    colors = [_COLOR_BLUE, _COLOR_ORANGE, _COLOR_GREEN, _COLOR_PURPLE]
    n_records_labels: list[str] = []
    for scope_index, scope in enumerate(scopes):
        scope_rows = data.filter(pl.col("scope") == scope)
        stage_to_stats: dict[str, tuple[float, float, float, float, float, int]] = {}
        for row in scope_rows.iter_rows(named=True):
            stage = str(row["stage"])
            n_records = int(row["n_records"])
            minimum = float(row["n_features_min"])
            q1_raw = row["n_features_q1"]
            median_raw = row["n_features_median"]
            q3_raw = row["n_features_q3"]
            maximum = float(row["n_features_max"])
            q1 = np.nan if q1_raw is None else float(q1_raw)
            median = np.nan if median_raw is None else float(median_raw)
            q3 = np.nan if q3_raw is None else float(q3_raw)
            stage_to_stats[stage] = (minimum, q1, median, q3, maximum, n_records)
        y_min: list[float] = []
        y_q1: list[float] = []
        y_median: list[float] = []
        y_q3: list[float] = []
        y_max: list[float] = []
        n_records_by_stage: list[int] = []
        for stage in stage_order:
            stats = stage_to_stats.get(stage)
            if stats is None:
                y_min.append(np.nan)
                y_q1.append(np.nan)
                y_median.append(np.nan)
                y_q3.append(np.nan)
                y_max.append(np.nan)
                continue
            y_min.append(stats[0])
            y_q1.append(stats[1])
            y_median.append(stats[2])
            y_q3.append(stats[3])
            y_max.append(stats[4])
            n_records_by_stage.append(stats[5])
        color = colors[scope_index % len(colors)]
        y_min_array = np.array(y_min, dtype=float)
        y_q1_array = np.array(y_q1, dtype=float)
        y_median_array = np.array(y_median, dtype=float)
        y_q3_array = np.array(y_q3, dtype=float)
        y_max_array = np.array(y_max, dtype=float)
        finite_mask = np.isfinite(y_median_array)
        if not np.any(finite_mask):
            continue
        iqr_mask = finite_mask & np.isfinite(y_q1_array) & np.isfinite(y_q3_array)
        if np.any(iqr_mask):
            ax.fill_between(
                x_positions[iqr_mask],
                y_q1_array[iqr_mask],
                y_q3_array[iqr_mask],
                color=color,
                alpha=0.22,
                edgecolor="none",
                linewidth=0.0,
            )
        ax.plot(
            x_positions[finite_mask],
            y_min_array[finite_mask],
            linestyle="--",
            linewidth=0.7,
            color=color,
            alpha=0.9,
        )
        ax.plot(
            x_positions[finite_mask],
            y_max_array[finite_mask],
            linestyle="--",
            linewidth=0.7,
            color=color,
            alpha=0.9,
        )
        ax.plot(
            x_positions[finite_mask],
            y_median_array[finite_mask],
            marker="o",
            linewidth=1.0,
            color=color,
        )
        n_records_label = _format_n_records_label(n_records_by_stage)
        if len(scopes) == 1:
            n_records_labels.append(n_records_label)
        else:
            n_records_labels.append(f"{scope}: {n_records_label}")
        for x_value, y_value in zip(
            x_positions[finite_mask],
            y_median_array[finite_mask],
            strict=True,
        ):
            ax.annotate(
                _format_feature_count_label(float(y_value)),
                xy=(x_value, y_value),
                xytext=(0, 6 + (scope_index % len(colors)) * 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=_MONO_FONTSIZE,
                fontfamily="monospace",
                color=color,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_FEATURE_FILTER_FIGURE_STAGE_LABELS[stage] for stage in stage_order],
        fontsize=_TICK_FONTSIZE,
    )
    ax.set_xlabel("Feature selection step", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Number of features", fontsize=_LABEL_FONTSIZE)
    ax.margins(y=0.15)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    if n_records_labels:
        ax.text(
            0.01,
            0.98,
            "\n".join(n_records_labels),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_MONO_FONTSIZE,
            fontfamily="monospace",
            color=_MUTED_TEXT_COLOR,
        )
    legend_color = colors[0]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=legend_color,
            marker="o",
            linewidth=1.0,
            markersize=3.0,
            label="median",
        ),
        Patch(
            facecolor=to_rgba(legend_color, 0.22),
            edgecolor="none",
            label="IQR (25-75%)",
        ),
        Line2D(
            [0],
            [0],
            color=legend_color,
            linestyle="--",
            linewidth=0.7,
            label="min-max",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.16)
    _save_svg_figure(fig, out_path)


def _selected_features_by_fold_after_preprocessing(
    retained_features_summary: pl.DataFrame, out_path: Path
) -> None:
    required = {"scope", "fold_id", "feature", "retained_count", "n_sample_sets", "retained_rate"}
    if not required.issubset(retained_features_summary.columns):
        raise FigureError(
            "retained_features_summary.tsv schema is invalid for "
            "selected_features_by_fold_after_preprocessing.svg"
        )
    data = (
        retained_features_summary.select(
            pl.col("scope").cast(pl.String, strict=False).alias("__scope"),
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("feature").cast(pl.String, strict=False).alias("__feature"),
            pl.col("retained_count").cast(pl.Int64, strict=False).alias("__retained_count"),
            pl.col("n_sample_sets").cast(pl.Int64, strict=False).alias("__n_sample_sets"),
            pl.col("retained_rate").cast(pl.Float64, strict=False).alias("__retained_rate"),
        )
        .filter(
            (pl.col("__scope") == "outer_fold")
            & pl.col("__fold_id").is_not_null()
            & (pl.col("__fold_id") != "")
            & pl.col("__feature").is_not_null()
            & (pl.col("__feature") != "")
            & pl.col("__retained_count").is_not_null()
            & pl.col("__n_sample_sets").is_not_null()
            & (pl.col("__n_sample_sets") > 0)
            & pl.col("__retained_rate").is_not_null()
            & pl.col("__retained_rate").is_finite()
        )
    )
    if data.height == 0:
        _write_message_figure(
            title="Selected Features by Fold after Preprocessing",
            message="No outer-fold selected-feature summary rows are available.",
            out_path=out_path,
            width_px=980,
            height_px=520,
        )
        return

    fold_ids = [str(v) for v in data.select("__fold_id").unique().to_series().to_list()]
    fold_ids = sorted(
        fold_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    ranked_features = (
        data.group_by("__feature")
        .agg(
            pl.col("__retained_rate").max().alias("__retained_rate_max"),
            pl.col("__retained_rate").mean().alias("__retained_rate_mean"),
            pl.len().alias("__n_folds_retained"),
        )
        .sort(
            [
                "__retained_rate_max",
                "__retained_rate_mean",
                "__n_folds_retained",
                "__feature",
            ],
            descending=[True, True, True, False],
        )
    )
    shown_feature_table = ranked_features.head(_RETAINED_FEATURE_LIMIT)
    features = [str(v) for v in shown_feature_table.select("__feature").to_series().to_list()]
    if not fold_ids or not features:
        _write_message_figure(
            title="Selected Features by Fold after Preprocessing",
            message="No selected features are available after filtering.",
            out_path=out_path,
            width_px=980,
            height_px=520,
        )
        return

    feature_index = {feature: idx for idx, feature in enumerate(features)}
    fold_index = {fold_id: idx for idx, fold_id in enumerate(fold_ids)}
    rate_matrix = np.zeros((len(features), len(fold_ids)), dtype=float)
    count_labels: dict[tuple[int, int], str] = {}
    visible = data.filter(pl.col("__feature").is_in(features))
    for row in visible.iter_rows(named=True):
        feature_name = str(row["__feature"])
        fold_id = str(row["__fold_id"])
        row_index = feature_index[feature_name]
        col_index = fold_index[fold_id]
        rate_matrix[row_index, col_index] = float(row["__retained_rate"])
        retained_count = int(row["__retained_count"])
        n_sample_sets = int(row["__n_sample_sets"])
        count_labels[(row_index, col_index)] = f"{retained_count}/{n_sample_sets}"

    height_px = max(320, 110 + len(features) * 14)
    width_px = _fold_axis_width_px(len(fold_ids), base_px=210, per_fold_px=36)
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")

    image = ax.imshow(
        rate_matrix,
        aspect="auto",
        cmap="Blues",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(np.arange(len(fold_ids), dtype=float))
    ax.set_xticklabels([str(fold_id) for fold_id in fold_ids], fontsize=_TICK_FONTSIZE)
    ax.set_yticks(np.arange(len(features), dtype=float))
    ax.set_yticklabels(features, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.set_xlabel("CV fold", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Selected features after preprocessing", fontsize=_LABEL_FONTSIZE)
    ax.set_xticks(np.arange(-0.5, len(fold_ids), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(features), 1.0), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if len(features) <= 12 and len(fold_ids) <= 8:
        for (row_index, col_index), label in count_labels.items():
            color = _AXIS_COLOR if rate_matrix[row_index, col_index] < 0.6 else "#ffffff"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=_MONO_FONTSIZE,
                color=color,
                fontfamily="monospace",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("selection_rate", rotation=90, fontsize=_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=_TICK_FONTSIZE)
    fig.subplots_adjust(
        left=_label_left_margin(features, width_px=width_px, fontsize_px=_MONO_FONTSIZE),
        right=0.94,
        top=0.98,
        bottom=0.12,
    )
    _save_svg_figure(fig, out_path)


def _non_zero_feature_count_by_fold(model_sparsity: pl.DataFrame, out_path: Path) -> None:
    required = {"scope", "fold_id", "n_nonzero_features"}
    if not required.issubset(model_sparsity.columns):
        raise FigureError(
            "model_sparsity.tsv schema is invalid for non_zero_feature_count_by_fold.svg"
        )

    data = (
        model_sparsity.select(
            pl.col("scope").cast(pl.String, strict=False).alias("__scope"),
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("n_nonzero_features").cast(pl.Float64, strict=False).alias("__count"),
        )
        .filter(
            (pl.col("__scope") == "outer_fold")
            & pl.col("__fold_id").is_not_null()
            & (pl.col("__fold_id") != "")
            & pl.col("__count").is_not_null()
            & pl.col("__count").is_finite()
        )
        .sort(["__fold_id", "__count"])
    )
    if data.height == 0:
        _write_message_figure(
            title="Non-zero Feature Count by Fold",
            message="No outer-fold non-zero feature counts are available in model_sparsity.tsv.",
            out_path=out_path,
            width_px=_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX,
            height_px=320,
        )
        return

    fold_ids = [str(v) for v in data.select("__fold_id").unique().to_series().to_list()]
    fold_ids = sorted(
        fold_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if not fold_ids:
        _write_message_figure(
            title="Non-zero Feature Count by Fold",
            message="No outer-fold non-zero feature counts are available in model_sparsity.tsv.",
            out_path=out_path,
            width_px=_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX,
            height_px=320,
        )
        return

    fold_centers = np.arange(1, len(fold_ids) + 1, dtype=float)
    values_by_fold: list[np.ndarray] = []
    all_values: list[float] = []
    for fold_id in fold_ids:
        subset = data.filter(pl.col("__fold_id") == fold_id)
        values = np.array(subset.select("__count").to_series().to_list(), dtype=float)
        values_by_fold.append(values)
        all_values.extend(values.tolist())

    y_max = max(all_values) if all_values else 1.0
    if np.isclose(y_max, 0.0):
        y_max = 1.0

    width_px = _fold_axis_width_px(len(fold_ids), base_px=140, per_fold_px=44)
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, 340), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")

    for fold_idx, center in enumerate(fold_centers):
        if fold_idx % 2 == 0:
            ax.axvspan(center - 0.48, center + 0.48, color="#f7f7f7", zorder=0)
    for boundary in np.arange(1.5, len(fold_ids), 1.0):
        ax.axvline(boundary, color="#d9d9d9", linewidth=0.5, zorder=1)

    box_values: list[list[float]] = []
    box_positions: list[float] = []
    for center, values in zip(fold_centers, values_by_fold, strict=True):
        if values.size > 1:
            box_values.append(values.tolist())
            box_positions.append(float(center))
    if box_values:
        box = ax.boxplot(
            box_values,
            positions=box_positions,
            widths=0.42,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            manage_ticks=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": _AXIS_COLOR,
                "markeredgecolor": _AXIS_COLOR,
                "markersize": 3.0,
            },
            medianprops={"linewidth": 0.9, "color": _AXIS_COLOR},
            whiskerprops={"linewidth": 0.8, "color": _MUTED_TEXT_COLOR},
            capprops={"linewidth": 0.8, "color": _MUTED_TEXT_COLOR},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(_COLOR_BLUE)
            patch.set_alpha(0.22)
            patch.set_edgecolor(_COLOR_BLUE)
            patch.set_linewidth(0.8)

    for center, values in zip(fold_centers, values_by_fold, strict=True):
        if values.size == 0:
            continue
        offsets = _deterministic_offsets(values.size, 0.15)
        ax.scatter(
            np.full(values.shape[0], center, dtype=float) + offsets,
            values,
            s=18,
            color=_COLOR_BLUE,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.78,
            zorder=3,
        )

    for center, values in zip(fold_centers, values_by_fold, strict=True):
        if values.size == 0:
            continue
        ax.text(
            center,
            1.02,
            f"n={values.size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=_ANNOTATION_FONTSIZE,
            color=_MUTED_TEXT_COLOR,
            clip_on=False,
        )

    ax.set_xlim(0.52, len(fold_ids) + 0.48)
    ax.set_ylim(0.0, y_max * 1.15)
    ax.set_xticks(fold_centers)
    ax.set_xticklabels([str(fold_id) for fold_id in fold_ids], fontsize=_TICK_FONTSIZE)
    ax.set_xlabel("CV fold", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Number of non-zero features per model", fontsize=_LABEL_FONTSIZE)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.12, right=0.995, top=0.90, bottom=0.15)
    _save_svg_figure(fig, out_path)


def write_run_figures(
    *,
    run_dir: Path,
    metrics_cv: pl.DataFrame,
    oof_predictions: pl.DataFrame,
    thresholds: pl.DataFrame,
    feature_importance: pl.DataFrame,
    coefficients: pl.DataFrame,
    ensemble_model_probs: pl.DataFrame | None,
    model_selection_trials: pl.DataFrame | None,
    auto_threshold_metric: Literal["mcc", "balanced_accuracy"],
    feature_importance_by_fold: pl.DataFrame | None = None,
    coefficients_by_fold: pl.DataFrame | None = None,
    loss_by_split_cv: pl.DataFrame | None = None,
    loss_by_split_final_refit: pl.DataFrame | None = None,
    pred_external_test: pl.DataFrame | None = None,
    trait_name: str = "trait",
    model_selection_trials_summary: pl.DataFrame | None = None,
    feature_filter_counts_summary: pl.DataFrame | None = None,
    retained_features_summary: pl.DataFrame | None = None,
    model_sparsity: pl.DataFrame | None = None,
    model_sparsity_summary: pl.DataFrame | None = None,
    feature_filter_funnel_stage_order: Sequence[str] | None = None,
    top_features: int = _DEFAULT_TOP_FEATURES,
) -> list[str]:
    """Write run-level SVG figures under <run_dir>/figures."""
    warnings: list[str] = []
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)
    _cv_metrics_overview(metrics_cv, figures_dir / "cv_metrics_overview.svg")
    if loss_by_split_cv is not None:
        _cv_loss_by_split(loss_by_split_cv, figures_dir / "cv_loss_by_split.svg")
    if loss_by_split_final_refit is not None:
        _final_refit_loss_by_split(
            loss_by_split_final_refit, figures_dir / "final_refit_loss_by_split.svg"
        )
    _threshold_selection_curve(
        oof_predictions,
        thresholds,
        selection_metric=auto_threshold_metric,
        out_path=figures_dir / "threshold_selection_curve.svg",
    )
    _feature_importance_top(
        feature_importance,
        figures_dir / "feature_importance_top.svg",
        feature_importance_by_fold=feature_importance_by_fold,
        top_features=top_features,
    )
    if feature_importance_by_fold is not None:
        _feature_importance_by_fold_heatmap(
            feature_importance,
            feature_importance_by_fold,
            figures_dir / "feature_importance_by_fold_heatmap.svg",
            top_features=top_features,
        )
    _coefficients_signed_top(
        coefficients,
        figures_dir / "coefficients_signed_top.svg",
        coefficients_by_fold=coefficients_by_fold,
        top_features=top_features,
    )
    _species_probability_by_trait(
        predictions=oof_predictions,
        trait_col="label",
        trait_name=trait_name,
        out_path=figures_dir / "cv_species_probability_by_trait.svg",
        title="CV Species Probability by Trait",
        subtitle="Out-of-fold probabilities grouped by observed trait labels",
        source_table_name="prediction_cv.tsv",
        figure_name="cv_species_probability_by_trait.svg",
    )
    _cv_fold_trait_probability(
        oof_predictions,
        figures_dir / "cv_fold_trait_probability.svg",
        trait_name=trait_name,
    )
    try:
        _roc_pr_curves_cv(oof_predictions, figures_dir / "roc_pr_curves_cv.svg")
    except FigureError as exc:
        warnings.append(str(exc))
    selection_summary = model_selection_trials_summary
    if selection_summary is None and model_selection_trials is not None:
        selection_summary = _summarize_model_selection_trials_for_figure(model_selection_trials)
    if selection_summary is not None:
        _model_selection_trials_summary_panels(
            selection_summary,
            figures_dir / "model_selection_trials.svg",
            max_sample_sets_per_fold=_MODEL_SELECTION_SAMPLE_SET_LIMIT,
        )
    if feature_filter_counts_summary is not None:
        _feature_filter_funnel(
            feature_filter_counts_summary,
            figures_dir / "feature_filter_funnel.svg",
            stage_order=feature_filter_funnel_stage_order,
        )
    if retained_features_summary is not None:
        _selected_features_by_fold_after_preprocessing(
            retained_features_summary,
            figures_dir / "selected_features_by_fold_after_preprocessing.svg",
        )
    if model_sparsity is not None:
        _non_zero_feature_count_by_fold(
            model_sparsity,
            figures_dir / "non_zero_feature_count_by_fold.svg",
        )
    if pred_external_test is not None:
        try:
            _species_probability_by_trait(
                predictions=pred_external_test,
                trait_col="true_label",
                trait_name=trait_name,
                out_path=figures_dir / "external_species_probability_by_trait.svg",
                title="External Test Species Probability by Trait",
                subtitle="Final-refit probabilities grouped by external-test true labels",
                source_table_name="prediction_external_test.tsv",
                figure_name="external_species_probability_by_trait.svg",
            )
        except FigureError as exc:
            warnings.append(str(exc))
    return warnings


def write_predict_figures(
    *,
    run_dir: Path,
    pred_predict: pl.DataFrame,
    require_uncertainty: bool = False,
) -> None:
    """Write predict-level SVG figures under <run_dir>/figures."""
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)
    _predict_probability_distribution(
        pred_predict,
        figures_dir / "predict_probability_distribution.svg",
    )
    _predict_uncertainty(
        pred_predict,
        figures_dir / "predict_uncertainty.svg",
        required=require_uncertainty,
    )


def write_report_figures(
    *,
    report_dir: Path,
    report_runs: pl.DataFrame,
    report_ranking: pl.DataFrame,
) -> None:
    """Write report-level SVG figures under <report_dir>/figures."""
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _report_metric_ranking(report_ranking, figures_dir / "report_metric_ranking.svg")
    _report_metric_comparison(report_runs, figures_dir / "report_metric_comparison.svg")
    _report_stage_breakdown(report_runs, figures_dir / "report_stage_breakdown.svg")
