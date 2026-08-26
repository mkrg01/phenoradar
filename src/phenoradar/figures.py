"""Deterministic SVG figure generation for run/predict/report artifacts."""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from phenoradar.colors import CONFUSION_GROUP_COLORS, CONFUSION_GROUP_ORDER
from phenoradar.group_summary import GroupSummaryError, finite_group_probabilities
from phenoradar.metrics import (
    FIXED_PROBABILITY_THRESHOLD_NAME,
    FIXED_PROBABILITY_THRESHOLD_VALUE,
    metric_contract,
    metric_direction,
    metric_higher_is_better,
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
_CURVE_PANEL_SIZE_PX = _NATURE_SINGLE_COLUMN_WIDTH_PX
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
_UNANNOTATED_COLOR = "#6f6f6f"
_PROBABILITY_THRESHOLD_COLOR = "#999999"
_MODEL_SELECTION_SAMPLE_SET_LIMIT = 1
_DEFAULT_TOP_FEATURES = 30
_CONFUSION_GROUP_ORDER = CONFUSION_GROUP_ORDER
_CONFUSION_GROUP_COLORS = CONFUSION_GROUP_COLORS
_FEATURE_IMPORTANCE_TOP_WIDTH_PX = _NATURE_DOUBLE_COLUMN_WIDTH_PX
_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE = _LABEL_FONTSIZE
_FEATURE_ANNOTATION_LABEL_PADDING_PX = 180
_FEATURE_IMPORTANCE_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "phenoradar_feature_importance_blues",
    ["#ffffff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"],
)
_CONFUSION_MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "phenoradar_confusion_matrix_blues",
    ["#ffffff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"],
)
_COEFFICIENTS_TOP_WIDTH_PX = _NATURE_DOUBLE_COLUMN_WIDTH_PX
_COEFFICIENTS_AXIS_LABEL_FONTSIZE = _LABEL_FONTSIZE
_FEATURE_FILTER_FIGURE_DEFAULT_STAGE_ORDER = (
    "n_features_before",
    "n_features_after_sparse_feature_filter",
    "n_features_after_low_variance",
    "n_features_after_ranked_feature_filter",
    "n_features_after_correlation",
    "n_features_after_all",
)
_FEATURE_FILTER_FIGURE_STAGE_LABELS = {
    "n_features_before": "Input",
    "n_features_after_sparse_feature_filter": "Sparse feature",
    "n_features_after_low_variance": "Low variance",
    "n_features_after_ranked_feature_filter": "Ranked filter",
    "n_features_after_correlation": "Correlation",
    "n_features_after_all": "Final",
}
_RUN_FIGURE_STAGES = ("cv", "external_test", "inference")
_CV_EXTERNAL_METRIC_ORDER = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("mcc", "MCC"),
)
type _FigureJob = tuple[
    str,
    Callable[..., None],
    tuple[Any, ...],
    dict[str, Any],
    bool,
]


def _figure_size_inches(width_px: int, height_px: int) -> tuple[float, float]:
    return width_px / _FIG_DPI, height_px / _FIG_DPI


def _fold_axis_width_px(fold_count: int, *, base_px: int = 140, per_fold_px: int = 44) -> int:
    return min(
        _NATURE_DOUBLE_COLUMN_WIDTH_PX,
        max(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, base_px + fold_count * per_fold_px),
    )


def _label_text_width_px(labels: list[str], *, fontsize_px: int, padding: int = 32) -> float:
    max_chars = max(
        (len(line) for label in labels for line in str(label).splitlines()),
        default=0,
    )
    return max_chars * fontsize_px * 0.62 + padding


def _label_left_margin(labels: list[str], *, width_px: int, fontsize_px: int) -> float:
    label_px = _label_text_width_px(labels, fontsize_px=fontsize_px)
    return min(0.34, max(0.16, label_px / width_px))


def _feature_label_row_height_px(labels: list[str]) -> int:
    max_lines = max((len(label.splitlines()) for label in labels), default=1)
    return max(18, 10 + max_lines * 9)


def _feature_label_axis_title(_labels: list[str]) -> str:
    return "Orthogroup"


def _orthogroup_annotation_lookup(
    orthogroup_annotations: pl.DataFrame | None,
) -> dict[str, str]:
    if orthogroup_annotations is None:
        return {}
    required = {"feature", "orthogroup_annotation"}
    if not required.issubset(orthogroup_annotations.columns):
        raise FigureError("orthogroup annotation table schema is invalid for feature labels")
    lookup: dict[str, str] = {}
    for row in orthogroup_annotations.select(
        [
            pl.col("feature").cast(pl.String, strict=False).alias("feature"),
            pl.col("orthogroup_annotation")
            .cast(pl.String, strict=False)
            .alias("orthogroup_annotation"),
        ]
    ).iter_rows(named=True):
        feature = row["feature"]
        annotation = row["orthogroup_annotation"]
        if feature is None or annotation is None:
            continue
        feature_text = str(feature).strip()
        annotation_text = str(annotation).strip()
        if feature_text and annotation_text:
            lookup[feature_text] = annotation_text
    return lookup


def _feature_axis_labels(
    features: list[str],
    orthogroup_annotations: pl.DataFrame | None,
) -> list[str]:
    annotation_lookup = _orthogroup_annotation_lookup(orthogroup_annotations)
    labels: list[str] = []
    for feature in features:
        annotation = annotation_lookup.get(feature)
        if annotation is None:
            labels.append(feature)
            continue
        labels.append(f"{' '.join(annotation.split())} ({feature})")
    return labels


def _feature_axis_layout(
    labels: list[str],
    *,
    base_width_px: int,
    base_left: float,
    base_right: float,
    fontsize_px: int,
) -> tuple[int, float, float]:
    base_left_px = base_left * base_width_px
    base_right_px = base_right * base_width_px
    label_width_px = _label_text_width_px(
        labels,
        fontsize_px=fontsize_px,
        padding=_FEATURE_ANNOTATION_LABEL_PADDING_PX,
    )
    extra_left_px = max(0, int(np.ceil(label_width_px - base_left_px)))
    width_px = base_width_px + extra_left_px
    left = (base_left_px + extra_left_px) / width_px
    right = (base_right_px + extra_left_px) / width_px
    return width_px, left, right


def _compact_bottom_margin(height_px: int) -> float:
    return min(0.24, max(0.065, 42 / height_px))


def _save_svg_figure(fig: Figure, out_path: Path) -> None:
    fig.savefig(
        out_path,
        format="svg",
        dpi=_FIG_DPI,
        metadata={"Date": None},
    )
    plt.close(fig)


def _save_pdf_figure(fig: Figure, out_path: Path, *, title: str) -> None:
    fig.savefig(
        out_path,
        format="pdf",
        dpi=_FIG_DPI,
        metadata={
            "Title": title,
            "Creator": "PhenoRadar",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def _stage_figure_dirs(run_dir: Path) -> dict[str, Path]:
    figure_dirs = {stage: run_dir / stage / "figures" for stage in _RUN_FIGURE_STAGES}
    for figures_dir in figure_dirs.values():
        figures_dir.mkdir(parents=True, exist_ok=True)
    return figure_dirs


def _top_feature_importance_features(
    feature_importance: pl.DataFrame,
    *,
    top_features: int,
) -> list[str]:
    required = {"feature", "importance_mean"}
    if top_features < 1 or not required.issubset(feature_importance.columns):
        return []
    top = feature_importance.sort(
        by=["importance_mean", "feature"],
        descending=[True, False],
    ).head(top_features)
    return [str(value) for value in top.select("feature").to_series().to_list()]


def _top_coefficient_features(
    coefficients: pl.DataFrame,
    *,
    top_features: int,
) -> list[str]:
    required = {"feature", "coef_mean", "method"}
    if top_features < 1 or not required.issubset(coefficients.columns):
        return []
    linear = coefficients.filter(pl.col("method") == "coef_signed").drop_nulls("coef_mean")
    if linear.height == 0:
        return []
    top = (
        linear.with_columns(pl.col("coef_mean").abs().alias("__abs_coef"))
        .sort(
            by=["__abs_coef", "feature"],
            descending=[True, False],
        )
        .head(top_features)
    )
    return [str(value) for value in top.select("feature").to_series().to_list()]


def _feature_label_annotations_subset(
    orthogroup_annotations: pl.DataFrame | None,
    *,
    feature_importance: pl.DataFrame,
    coefficients: pl.DataFrame,
    top_features: int,
) -> pl.DataFrame | None:
    if orthogroup_annotations is None:
        return None
    required = {"feature", "orthogroup_annotation"}
    if not required.issubset(orthogroup_annotations.columns):
        return orthogroup_annotations

    features = figure_annotation_features(
        feature_importance=feature_importance,
        coefficients=coefficients,
        top_features=top_features,
    )
    if not features:
        return orthogroup_annotations
    return orthogroup_annotations.select(
        [
            pl.col("feature").cast(pl.String, strict=False).str.strip_chars().alias("feature"),
            pl.col("orthogroup_annotation")
            .cast(pl.String, strict=False)
            .alias("orthogroup_annotation"),
        ]
    ).filter(pl.col("feature").is_in(features))


def figure_annotation_features(
    *,
    feature_importance: pl.DataFrame,
    coefficients: pl.DataFrame,
    top_features: int,
) -> list[str]:
    """Return sorted feature names whose labels can appear in run figures."""
    return sorted(
        {
            *[
                feature.strip()
                for feature in _top_feature_importance_features(
                    feature_importance,
                    top_features=top_features,
                )
                if feature.strip()
            ],
            *[
                feature.strip()
                for feature in _top_coefficient_features(
                    coefficients,
                    top_features=top_features,
                )
                if feature.strip()
            ],
        }
    )


def _execute_figure_job(
    name: str,
    func: Callable[..., None],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    catch_figure_error: bool,
) -> tuple[str, list[str]]:
    try:
        func(*args, **kwargs)
    except FigureError as exc:
        if catch_figure_error:
            return name, [str(exc)]
        raise
    return name, []


def _run_figure_jobs(jobs: list[_FigureJob], *, parallel_workers: int) -> list[str]:
    if not jobs:
        return []
    worker_count = max(1, min(int(parallel_workers), len(jobs)))
    if worker_count == 1:
        sequential_warnings: list[str] = []
        for name, func, args, kwargs, catch_figure_error in jobs:
            _job_name, job_warnings = _execute_figure_job(
                name,
                func,
                args,
                kwargs,
                catch_figure_error,
            )
            sequential_warnings.extend(job_warnings)
        return sequential_warnings

    warnings_by_index: dict[int, list[str]] = {}
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    ) as executor:
        future_to_index = {
            executor.submit(
                _execute_figure_job,
                name,
                func,
                args,
                kwargs,
                catch_figure_error,
            ): index
            for index, (name, func, args, kwargs, catch_figure_error) in enumerate(jobs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            _job_name, job_warnings = future.result()
            warnings_by_index[index] = job_warnings

    ordered_warnings: list[str] = []
    for index in range(len(jobs)):
        ordered_warnings.extend(warnings_by_index.get(index, []))
    return ordered_warnings


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

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=0.96,
        bottom=_compact_bottom_margin(height_px),
    )
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


def _score_domain_with_zero_floor(values: np.ndarray) -> tuple[float, float]:
    lower, upper = _padded_domain(values.tolist(), include_zero=True)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return lower, upper
    if float(np.min(finite)) >= 0.0:
        lower = 0.0
    return lower, upper


def _place_x_axis_at_zero(ax: Any) -> None:
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["bottom"].set_color(_AXIS_COLOR)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")


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
    _place_x_axis_at_zero(ax)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.18)
    _save_svg_figure(fig, out_path)


def _group_bootstrap_metrics_figure(
    group_bootstrap_metrics: pl.DataFrame,
    out_path: Path,
) -> None:
    required_columns = {
        "metric",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "n_resamples",
        "n_valid_resamples",
        "n_groups",
        "group_col",
    }
    if not required_columns.issubset(group_bootstrap_metrics.columns):
        raise FigureError(
            "group_bootstrap_metrics.tsv schema is invalid for group_bootstrap_metrics.svg"
        )
    data = group_bootstrap_metrics.filter(
        pl.col("point_estimate").is_not_null()
        & pl.col("point_estimate").is_finite()
        & pl.col("ci_lower").is_not_null()
        & pl.col("ci_lower").is_finite()
        & pl.col("ci_upper").is_not_null()
        & pl.col("ci_upper").is_finite()
    )
    if data.height == 0:
        raise FigureError("group_bootstrap_metrics.tsv contains no finite confidence intervals")

    preferred_order = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "brier",
        "log_loss",
    ]
    rows_by_metric = {str(row["metric"]): row for row in data.iter_rows(named=True)}
    metric_names = [name for name in preferred_order if name in rows_by_metric] + sorted(
        set(rows_by_metric) - set(preferred_order)
    )
    rows = [rows_by_metric[name] for name in metric_names]
    point_estimates = np.asarray([float(row["point_estimate"]) for row in rows])
    ci_lower = np.asarray([float(row["ci_lower"]) for row in rows])
    ci_upper = np.asarray([float(row["ci_upper"]) for row in rows])
    labels: list[str] = []
    for metric_name in metric_names:
        try:
            labels.append(str(metric_contract(metric_name)["display_name"]))
        except ValueError:
            labels.append(metric_name)

    y_pos = np.arange(len(rows), dtype=float)
    width_px = _NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX
    height_px = max(280, 120 + len(rows) * 34)
    fig, ax = plt.subplots(
        figsize=_figure_size_inches(width_px, height_px),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")
    ax.hlines(y_pos, ci_lower, ci_upper, color=_COLOR_SKY, linewidth=1.4)
    ax.scatter(point_estimates, y_pos, color=_COLOR_BLUE, s=18, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=_TICK_FONTSIZE)
    ax.invert_yaxis()

    x_min, x_max = _padded_domain(
        [*ci_lower.tolist(), *ci_upper.tolist(), *point_estimates.tolist()],
        include_zero=True,
    )
    ax.set_xlim(x_min, x_max)
    if x_min <= 0.0 <= x_max:
        ax.axvline(0.0, color=_MUTED_TEXT_COLOR, linewidth=0.7)
    confidence_level = float(rows[0]["confidence_level"])
    confidence_percent = confidence_level * 100.0
    ax.set_xlabel(
        f"Metric value ({confidence_percent:g}% group-bootstrap CI)",
        fontsize=_LABEL_FONTSIZE,
    )
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    group_col = str(rows[0]["group_col"])
    n_groups = int(rows[0]["n_groups"])
    n_resamples = int(rows[0]["n_resamples"])
    min_valid_resamples = min(int(row["n_valid_resamples"]) for row in rows)
    ax.set_title("OOF group-bootstrap confidence intervals", pad=8.0)
    fig.text(
        0.99,
        0.01,
        (
            f"group_col={group_col}; groups={n_groups}; resamples={n_resamples}; "
            f"minimum valid resamples={min_valid_resamples}"
        ),
        ha="right",
        va="bottom",
        fontsize=_ANNOTATION_FONTSIZE,
        color=_MUTED_TEXT_COLOR,
    )
    fig.subplots_adjust(left=0.31, right=0.985, top=0.89, bottom=0.16)
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
    ax.set_ylabel("Log loss", fontsize=_LABEL_FONTSIZE)
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
        x_label="Log Loss",
        y_tick_fontsize=_TICK_FONTSIZE,
    )


def _feature_importance_top(
    feature_importance: pl.DataFrame,
    out_path: Path,
    feature_importance_by_fold: pl.DataFrame | None = None,
    top_features: int = _DEFAULT_TOP_FEATURES,
    orthogroup_annotations: pl.DataFrame | None = None,
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
    feature_labels = _feature_axis_labels(features, orthogroup_annotations)
    feature_axis_title = _feature_label_axis_title(feature_labels)
    row_height_px = _feature_label_row_height_px(feature_labels)
    base_width_px = _FEATURE_IMPORTANCE_TOP_WIDTH_PX
    base_left = _label_left_margin(
        features,
        width_px=base_width_px,
        fontsize_px=_MONO_FONTSIZE,
    )
    values = [float(v) for v in top.select("importance_mean").to_series().to_list()]
    if feature_importance_by_fold is not None:
        fold_required = {"fold_id", "feature", "importance_mean"}
        if not fold_required.issubset(feature_importance_by_fold.columns):
            raise FigureError(
                "feature_importance_by_fold.tsv schema is invalid for feature_importance_top.svg"
            )
        fold_data = feature_importance_by_fold.select(
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("feature").cast(pl.String, strict=False).alias("__feature"),
            pl.col("importance_mean").cast(pl.Float64, strict=False).alias("__value"),
        ).filter(
            pl.col("__feature").is_in(features)
            & pl.col("__value").is_not_null()
            & pl.col("__value").is_finite()
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

            height_px = max(240, 70 + len(features) * row_height_px)
            width_px, left_margin, right_margin = _feature_axis_layout(
                feature_labels,
                base_width_px=base_width_px,
                base_left=base_left,
                base_right=0.985,
                fontsize_px=_MONO_FONTSIZE,
            )
            fig, ax = plt.subplots(
                figsize=_figure_size_inches(width_px, height_px),
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
            ax.set_yticklabels(feature_labels, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
            ax.set_ylabel(
                feature_axis_title,
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
                left=left_margin,
                right=right_margin,
                top=0.985,
                bottom=_compact_bottom_margin(height_px),
            )
            _save_svg_figure(fig, out_path)
            return

    max_value = max(values) if values else 1.0
    if np.isclose(max_value, 0.0):
        max_value = 1.0

    height_px = max(220, 60 + len(features) * row_height_px)
    width_px, left_margin, right_margin = _feature_axis_layout(
        feature_labels,
        base_width_px=base_width_px,
        base_left=base_left,
        base_right=0.98,
        fontsize_px=_MONO_FONTSIZE,
    )
    fig, ax = plt.subplots(
        figsize=_figure_size_inches(width_px, height_px),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(features), dtype=float)
    bars = ax.barh(y_pos, values, color=_MUTED_TEXT_COLOR, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.set_ylabel(feature_axis_title, fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE)
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
        left=left_margin,
        right=right_margin,
        top=0.985,
        bottom=_compact_bottom_margin(height_px),
    )
    _save_svg_figure(fig, out_path)


def _feature_importance_by_fold_heatmap(
    feature_importance: pl.DataFrame,
    feature_importance_by_fold: pl.DataFrame,
    out_path: Path,
    top_features: int = _DEFAULT_TOP_FEATURES,
    orthogroup_annotations: pl.DataFrame | None = None,
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
    feature_labels = _feature_axis_labels(features, orthogroup_annotations)
    feature_axis_title = _feature_label_axis_title(feature_labels)
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
        importance_matrix[feature_index[feature_name], fold_index[fold_id]] = float(row["__value"])

    finite_values = importance_matrix[np.isfinite(importance_matrix)]
    max_value = float(np.max(finite_values)) if finite_values.size > 0 else 1.0
    if np.isclose(max_value, 0.0):
        max_value = 1.0

    row_height_px = _feature_label_row_height_px(feature_labels)
    height_px = max(260, 95 + len(features) * row_height_px)
    base_width_px = _fold_axis_width_px(len(fold_ids), base_px=260, per_fold_px=44)
    base_left = _label_left_margin(
        features,
        width_px=base_width_px,
        fontsize_px=_MONO_FONTSIZE,
    )
    width_px, left_margin, right_margin = _feature_axis_layout(
        feature_labels,
        base_width_px=base_width_px,
        base_left=base_left,
        base_right=0.94,
        fontsize_px=_MONO_FONTSIZE,
    )
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
    ax.set_yticklabels(feature_labels, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.set_xlabel("CV fold", fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(feature_axis_title, fontsize=_FEATURE_IMPORTANCE_AXIS_LABEL_FONTSIZE)
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
        left=left_margin,
        right=right_margin,
        top=0.98,
        bottom=_compact_bottom_margin(height_px),
    )
    _save_svg_figure(fig, out_path)


def _coefficients_signed_top(
    coefficients: pl.DataFrame,
    out_path: Path,
    coefficients_by_fold: pl.DataFrame | None = None,
    top_features: int = _DEFAULT_TOP_FEATURES,
    orthogroup_annotations: pl.DataFrame | None = None,
) -> None:
    required = {"feature", "coef_mean", "method"}
    if not required.issubset(coefficients.columns):
        raise FigureError("coefficients.tsv schema is invalid for coefficients_signed_top.svg")
    if top_features < 1:
        raise FigureError("figures.top_features must be >= 1")

    linear = coefficients.filter(pl.col("method") == "coef_signed").drop_nulls("coef_mean")
    if linear.height == 0:
        return

    top = (
        linear.with_columns(pl.col("coef_mean").abs().alias("__abs_coef"))
        .sort(
            by=["__abs_coef", "feature"],
            descending=[True, False],
        )
        .head(top_features)
    )

    features = [str(v) for v in top.select("feature").to_series().to_list()]
    feature_labels = _feature_axis_labels(features, orthogroup_annotations)
    feature_axis_title = _feature_label_axis_title(feature_labels)
    row_height_px = _feature_label_row_height_px(feature_labels)
    base_width_px = _COEFFICIENTS_TOP_WIDTH_PX
    base_left = _label_left_margin(
        features,
        width_px=base_width_px,
        fontsize_px=_MONO_FONTSIZE,
    )
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

            height_px = max(240, 70 + len(features) * row_height_px)
            width_px, left_margin, right_margin = _feature_axis_layout(
                feature_labels,
                base_width_px=base_width_px,
                base_left=base_left,
                base_right=0.985,
                fontsize_px=_MONO_FONTSIZE,
            )
            fig, ax = plt.subplots(
                figsize=_figure_size_inches(width_px, height_px),
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
            ax.set_yticklabels(feature_labels, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
            ax.set_ylabel(feature_axis_title, fontsize=_COEFFICIENTS_AXIS_LABEL_FONTSIZE)
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
                left=left_margin,
                right=right_margin,
                top=0.985,
                bottom=_compact_bottom_margin(height_px),
            )
            _save_svg_figure(fig, out_path)
            return

    max_abs = max(abs(v) for v in values) if values else 1.0
    if np.isclose(max_abs, 0.0):
        max_abs = 1.0

    height_px = max(220, 60 + len(features) * row_height_px)
    width_px, left_margin, right_margin = _feature_axis_layout(
        feature_labels,
        base_width_px=base_width_px,
        base_left=base_left,
        base_right=0.98,
        fontsize_px=_MONO_FONTSIZE,
    )
    fig, ax = plt.subplots(
        figsize=_figure_size_inches(width_px, height_px),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(features), dtype=float)
    bars = ax.barh(y_pos, values, color=_MUTED_TEXT_COLOR, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.set_ylabel(feature_axis_title, fontsize=_COEFFICIENTS_AXIS_LABEL_FONTSIZE)
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
        left=left_margin,
        right=right_margin,
        top=0.985,
        bottom=_compact_bottom_margin(height_px),
    )
    _save_svg_figure(fig, out_path)


def _top_feature_expression_by_confusion(
    *,
    oof_predictions: pl.DataFrame,
    top_feature_expression: pl.DataFrame,
    feature_importance: pl.DataFrame,
    coefficients: pl.DataFrame,
    out_path: Path,
    top_features: int = _DEFAULT_TOP_FEATURES,
    orthogroup_annotations: pl.DataFrame | None = None,
) -> None:
    prediction_required = {"species", "label", "prob"}
    expression_required = {"species", "feature", "tpm"}
    importance_required = {"feature", "importance_mean"}
    coefficient_required = {"feature", "coef_mean", "method"}
    if not prediction_required.issubset(oof_predictions.columns):
        raise FigureError(
            "prediction_cv.tsv schema is invalid for top_feature_expression_by_confusion.svg"
        )
    if not expression_required.issubset(top_feature_expression.columns):
        raise FigureError(
            "top-feature expression schema is invalid for top_feature_expression_by_confusion.svg"
        )
    if not importance_required.issubset(feature_importance.columns):
        raise FigureError(
            "feature_importance.tsv schema is invalid for top_feature_expression_by_confusion.svg"
        )
    if not coefficient_required.issubset(coefficients.columns):
        raise FigureError(
            "coefficients.tsv schema is invalid for top_feature_expression_by_confusion.svg"
        )
    if top_features < 1:
        raise FigureError("figures.top_features must be >= 1")

    predictions = (
        oof_predictions.select(
            pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("__species"),
            pl.col("label").cast(pl.Int8, strict=False).alias("__label"),
            pl.col("prob").cast(pl.Float64, strict=False).alias("__prob"),
        )
        .filter(
            pl.col("__species").is_not_null()
            & (pl.col("__species") != "")
            & pl.col("__label").is_not_null()
            & pl.col("__prob").is_not_null()
            & pl.col("__prob").is_finite()
        )
        .sort("__species")
    )
    if predictions.height == 0:
        raise FigureError(
            "prediction_cv.tsv is empty; cannot draw top_feature_expression_by_confusion.svg"
        )
    labels = [int(value) for value in predictions.get_column("__label").to_list()]
    _binary_trait_color_map(
        labels,
        source_table_name="prediction_cv.tsv",
        figure_name="top_feature_expression_by_confusion.svg",
    )
    if predictions.get_column("__species").n_unique() != predictions.height:
        raise FigureError(
            "prediction_cv.tsv must contain one row per species for "
            "top_feature_expression_by_confusion.svg"
        )
    predictions = predictions.with_columns(
        (pl.col("__prob") >= FIXED_PROBABILITY_THRESHOLD_VALUE).alias("__pred_label")
    ).with_columns(
        pl.when((pl.col("__label") == 1) & pl.col("__pred_label"))
        .then(pl.lit("TP"))
        .when((pl.col("__label") == 1) & ~pl.col("__pred_label"))
        .then(pl.lit("FN"))
        .when((pl.col("__label") == 0) & ~pl.col("__pred_label"))
        .then(pl.lit("TN"))
        .otherwise(pl.lit("FP"))
        .alias("__confusion_group")
    )

    top = (
        feature_importance.select(
            pl.col("feature").cast(pl.String, strict=False).str.strip_chars().alias("__feature"),
            pl.col("importance_mean").cast(pl.Float64, strict=False).alias("__importance"),
        )
        .filter(
            pl.col("__feature").is_not_null()
            & (pl.col("__feature") != "")
            & pl.col("__importance").is_not_null()
            & pl.col("__importance").is_finite()
        )
        .sort(["__importance", "__feature"], descending=[True, False])
        .head(top_features)
    )
    features = [str(value) for value in top.get_column("__feature").to_list()]
    if not features:
        raise FigureError(
            "feature_importance.tsv is empty; cannot draw top_feature_expression_by_confusion.svg"
        )
    importance_lookup = {
        str(row["__feature"]): float(row["__importance"]) for row in top.iter_rows(named=True)
    }
    coefficient_lookup = {
        str(row["__feature"]): float(row["__coef"])
        for row in coefficients.filter(pl.col("method") == "coef_signed")
        .select(
            pl.col("feature").cast(pl.String, strict=False).str.strip_chars().alias("__feature"),
            pl.col("coef_mean").cast(pl.Float64, strict=False).alias("__coef"),
        )
        .filter(
            pl.col("__feature").is_not_null()
            & (pl.col("__feature") != "")
            & pl.col("__coef").is_not_null()
            & pl.col("__coef").is_finite()
        )
        .iter_rows(named=True)
    }
    annotation_lookup = _orthogroup_annotation_lookup(orthogroup_annotations)
    expression = top_feature_expression.select(
        pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("__species"),
        pl.col("feature").cast(pl.String, strict=False).str.strip_chars().alias("__feature"),
        pl.col("tpm").cast(pl.Float64, strict=False).alias("__tpm"),
    ).filter(
        pl.col("__species").is_not_null()
        & (pl.col("__species") != "")
        & pl.col("__feature").is_in(features)
        & pl.col("__tpm").is_not_null()
        & pl.col("__tpm").is_finite()
    )
    if expression.filter(pl.col("__tpm") < 0.0).height > 0:
        raise FigureError(
            "top-feature expression contains negative TPM values for "
            "top_feature_expression_by_confusion.svg"
        )
    data = (
        expression.join(
            predictions.select(["__species", "__confusion_group"]),
            on="__species",
            how="inner",
        )
        .with_columns((pl.col("__tpm") + 1.0).log(base=2.0).alias("__log2_tpm"))
        .sort(["__feature", "__confusion_group", "__species"])
    )
    available_features = set(data.get_column("__feature").unique().to_list())
    features = [feature for feature in features if feature in available_features]
    if not features:
        raise FigureError(
            "No top-feature expression rows overlap CV species for "
            "top_feature_expression_by_confusion.svg"
        )

    title_lines_by_feature: dict[str, list[str]] = {}
    for feature in features:
        annotation = annotation_lookup.get(feature)
        title_lines = (
            textwrap.wrap(
                " ".join(annotation.split()),
                width=36,
                break_long_words=False,
            )
            if annotation is not None
            else []
        )
        title_lines.extend([f"({feature})", f"importance={importance_lookup[feature]:.3g}"])
        coefficient = coefficient_lookup.get(feature)
        if coefficient is not None:
            if np.isclose(coefficient, 0.0):
                title_lines[-1] += " | β=0"
            else:
                title_lines[-1] += f" | β={coefficient:+.3g}"
        title_lines_by_feature[feature] = title_lines

    n_columns = min(5, len(features))
    n_rows = int(np.ceil(len(features) / n_columns))
    width_px = max(_NATURE_DOUBLE_COLUMN_WIDTH_PX, 210 * n_columns)
    max_title_lines = max(len(lines) for lines in title_lines_by_feature.values())
    top_padding_px = max(42, 12 + 9 * max_title_lines)
    bottom_padding_px = 60
    height_px = max(
        430,
        145 + 205 * n_rows + max(0, top_padding_px - 57),
    )
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=_figure_size_inches(width_px, height_px),
        dpi=_FIG_DPI,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    positions = np.arange(1, len(_CONFUSION_GROUP_ORDER) + 1, dtype=float)

    for axis_index, feature in enumerate(features):
        ax = flat_axes[axis_index]
        feature_data = data.filter(pl.col("__feature") == feature)
        values_by_group: dict[str, np.ndarray] = {}
        for group in _CONFUSION_GROUP_ORDER:
            group_data = feature_data.filter(pl.col("__confusion_group") == group).sort("__species")
            values_by_group[group] = np.asarray(
                group_data.get_column("__log2_tpm").to_list(), dtype=float
            )

        nonempty_groups = [
            group for group in _CONFUSION_GROUP_ORDER if values_by_group[group].size > 0
        ]
        if nonempty_groups:
            box = ax.boxplot(
                [values_by_group[group].tolist() for group in nonempty_groups],
                positions=[
                    float(_CONFUSION_GROUP_ORDER.index(group) + 1) for group in nonempty_groups
                ],
                widths=0.56,
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                manage_ticks=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": _AXIS_COLOR,
                    "markeredgecolor": _AXIS_COLOR,
                    "markersize": 2.4,
                },
                medianprops={"linewidth": 0.8, "color": _AXIS_COLOR},
                whiskerprops={"linewidth": 0.7, "color": _MUTED_TEXT_COLOR},
                capprops={"linewidth": 0.7, "color": _MUTED_TEXT_COLOR},
            )
            for patch, group in zip(box["boxes"], nonempty_groups, strict=True):
                color = _CONFUSION_GROUP_COLORS[group]
                patch.set_facecolor(color)
                patch.set_alpha(0.22)
                patch.set_edgecolor(color)
                patch.set_linewidth(0.8)

        for group_index, group in enumerate(_CONFUSION_GROUP_ORDER):
            values = values_by_group[group]
            if values.size == 0:
                continue
            offsets = _deterministic_offsets(values.size, 0.18)
            x_values = np.full(values.size, positions[group_index], dtype=float) + offsets
            scatter_kwargs: dict[str, Any] = {
                "s": 8,
                "color": _CONFUSION_GROUP_COLORS[group],
                "linewidths": 0.45,
                "alpha": 0.64,
                "zorder": 3,
            }
            if group in {"FN", "FP"}:
                scatter_kwargs["marker"] = "x"
            else:
                scatter_kwargs["marker"] = "o"
                scatter_kwargs["edgecolors"] = "white"
            ax.scatter(x_values, values, **scatter_kwargs)

        ax.set_title(
            "\n".join(title_lines_by_feature[feature]),
            fontsize=_SUBTITLE_FONTSIZE,
            pad=3.0,
        )
        ax.axvline(2.5, color="#d0d0d0", linewidth=0.6, linestyle=(0, (3, 3)))
        ax.set_xlim(0.55, 4.45)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{group}\nn={values_by_group[group].size}" for group in _CONFUSION_GROUP_ORDER],
            fontsize=_TICK_FONTSIZE,
        )
        ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)

    for axis_index in range(len(features), flat_axes.size):
        flat_axes[axis_index].set_visible(False)

    fig.supxlabel(
        "OOF confusion group",
        fontsize=_LABEL_FONTSIZE,
        y=10 / height_px,
    )
    fig.supylabel("log2(TPM + 1)", fontsize=_LABEL_FONTSIZE, x=0.008)
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=1.0 - top_padding_px / height_px,
        bottom=bottom_padding_px / height_px,
        wspace=0.32,
        hspace=0.88,
    )
    _save_svg_figure(fig, out_path)


def _predict_probability_distribution(
    pred_predict: pl.DataFrame,
    out_path: Path,
    *,
    figure_name: str = "predict_probability_distribution.svg",
) -> None:
    if "prob" not in pred_predict.columns:
        raise FigureError(f"prediction_inference.tsv schema is invalid for {figure_name}")

    probs = np.array(pred_predict.select("prob").to_series().to_list(), dtype=float)
    if probs.size == 0:
        raise FigureError(f"prediction_inference.tsv is empty; cannot draw {figure_name}")

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 320),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    bins = np.linspace(0.0, 1.0, 11).tolist()
    counts_raw, _bins, bars_raw = ax.hist(
        probs[np.isfinite(probs)],
        bins=bins,
        range=(0.0, 1.0),
        color=_COLOR_SKY,
        edgecolor=_COLOR_SKY,
        linewidth=0.0,
    )
    counts = cast(np.ndarray, counts_raw)
    bars = list(cast(Any, bars_raw))
    max_count = int(counts.max()) if counts.size > 0 else 1
    if max_count < 1:
        max_count = 1

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max_count * 1.15)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_xlabel("Predicted probability", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Number of species", fontsize=_LABEL_FONTSIZE)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

    for bar, count in zip(bars, counts.astype(int).tolist(), strict=True):
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

    data = (
        pred_predict.select("species", "uncertainty_std")
        .sort(
            by=["uncertainty_std", "species"],
            descending=[True, False],
        )
        .head(30)
    )
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
    ax.axhline(
        0.5,
        color=_PROBABILITY_THRESHOLD_COLOR,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        zorder=1,
    )
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.16)
    _save_svg_figure(fig, out_path)


def _species_probability_cv_and_inference(
    *,
    oof_predictions: pl.DataFrame,
    pred_inference: pl.DataFrame,
    trait_name: str,
    out_path: Path,
    figure_name: str = "species_probability_cv_and_inference.svg",
) -> None:
    cv_required = {"species", "label", "prob"}
    if not cv_required.issubset(oof_predictions.columns):
        raise FigureError(f"prediction_cv.tsv schema is invalid for {figure_name}")
    inference_required = {"species", "prob"}
    if not inference_required.issubset(pred_inference.columns):
        raise FigureError(f"prediction_inference.tsv schema is invalid for {figure_name}")

    cv_data = (
        oof_predictions.select(
            pl.col("species").cast(pl.String, strict=False).alias("__species"),
            pl.col("label").cast(pl.Int64, strict=False).alias("__trait"),
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
    if cv_data.height == 0:
        raise FigureError(f"prediction_cv.tsv is empty; cannot draw {figure_name}")

    inference_data = (
        pred_inference.select(
            pl.col("species").cast(pl.String, strict=False).alias("__species"),
            pl.col("prob").cast(pl.Float64, strict=False).alias("__prob"),
        )
        .filter(
            pl.col("__species").is_not_null()
            & (pl.col("__species") != "")
            & pl.col("__prob").is_not_null()
            & pl.col("__prob").is_finite()
        )
        .sort(["__prob", "__species"])
    )
    if inference_data.height == 0:
        raise FigureError(f"prediction_inference.tsv is empty; cannot draw {figure_name}")

    traits = [
        int(v) for v in cv_data.select("__trait").unique().sort("__trait").to_series().to_list()
    ]
    _binary_trait_color_map(
        traits,
        source_table_name="prediction_cv.tsv",
        figure_name=figure_name,
    )

    groups: list[tuple[str, np.ndarray, str]] = []
    for trait in (0, 1):
        probs = np.array(
            cv_data.filter(pl.col("__trait") == trait)
            .sort(["__prob", "__species"])
            .select("__prob")
            .to_series()
            .to_list(),
            dtype=float,
        )
        color = _TRAIT_NEGATIVE_COLOR if trait == 0 else _TRAIT_POSITIVE_COLOR
        groups.append((str(trait), probs, color))
    inference_probs = np.array(
        inference_data.select("__prob").to_series().to_list(),
        dtype=float,
    )
    groups.append(("unannotated", inference_probs, _UNANNOTATED_COLOR))

    positions = np.arange(1, len(groups) + 1, dtype=float)
    box_probs: list[list[float]] = []
    box_positions: list[float] = []
    box_colors: list[str] = []
    for position, (_label, probs, color) in zip(positions, groups, strict=True):
        if probs.size == 0:
            continue
        box_probs.append(probs.tolist())
        box_positions.append(float(position))
        box_colors.append(color)

    if not box_probs:
        raise FigureError(
            f"prediction_cv.tsv and prediction_inference.tsv are empty for {figure_name}"
        )

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 390),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    box = ax.boxplot(
        box_probs,
        positions=box_positions,
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
    for patch, color in zip(box["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)

    for position, (_label, probs, color) in zip(positions, groups, strict=True):
        if probs.size == 0:
            continue
        offsets = _deterministic_offsets(probs.size, 0.17)
        x_values = np.full(probs.shape[0], position, dtype=float) + offsets
        ax.scatter(
            x_values,
            probs,
            s=18,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.78,
            zorder=3,
        )

    for position, (_label, probs, _color) in zip(positions, groups, strict=True):
        ax.text(
            position,
            1.02,
            f"n={probs.size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=_ANNOTATION_FONTSIZE,
            color=_MUTED_TEXT_COLOR,
            clip_on=False,
        )

    ax.set_xlim(0.5, len(groups) + 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [label for label, _probs, _color in groups],
        fontsize=_TICK_FONTSIZE,
    )
    ax.set_xlabel(trait_name, fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted probability", fontsize=_LABEL_FONTSIZE)
    ax.axhline(
        0.5,
        color=_PROBABILITY_THRESHOLD_COLOR,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        zorder=1,
    )
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.16)
    _save_svg_figure(fig, out_path)


def write_group_probability_figure(
    *,
    grouped_predictions: pl.DataFrame,
    out_path: Path,
    group_label: str,
    source_table_name: str,
    figure_name: str,
) -> None:
    """Write a grouped probability distribution figure from joined predictions."""
    try:
        data = finite_group_probabilities(grouped_predictions)
    except GroupSummaryError as exc:
        raise FigureError(str(exc)) from exc
    if data.height == 0:
        raise FigureError(f"{source_table_name} is empty; cannot draw {figure_name}")

    group_order = (
        data.group_by(["group_id", "group_name"])
        .agg(
            pl.col("prob").max().alias("__prob_max"),
            pl.col("prob").mean().alias("__prob_mean"),
            pl.len().alias("__n_species"),
        )
        .sort(["__prob_max", "__prob_mean", "group_name"], descending=[True, True, False])
    )
    if group_order.height == 0:
        raise FigureError(f"{source_table_name} has no groups for {figure_name}")

    groups = [
        (str(row["group_id"]), str(row["group_name"]), int(row["__n_species"]))
        for row in group_order.iter_rows(named=True)
    ]
    group_ids = [group_id for group_id, _group_name, _n in groups]
    data = data.filter(pl.col("group_id").is_in(group_ids))

    labels = [
        f"{_ellipsize_label(group_name, max_chars=34)} (n={n_species})"
        for _group_id, group_name, n_species in groups
    ]
    values_by_group: list[np.ndarray] = []
    pred_labels_by_group: list[np.ndarray] = []
    for group_id, _group_name, _n_species in groups:
        subset = data.filter(pl.col("group_id") == group_id).sort(["prob", "species"])
        values_by_group.append(np.array(subset.select("prob").to_series().to_list(), dtype=float))
        pred_labels_by_group.append(
            np.array(
                subset.select("pred_label_fixed_threshold").to_series().to_list(),
                dtype=float,
            )
        )

    height_px = max(260, 90 + len(groups) * 24)
    width_px = _NATURE_DOUBLE_COLUMN_WIDTH_PX
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")

    y_positions = np.arange(len(groups), dtype=float)
    ax.boxplot(
        [values.tolist() for values in values_by_group],
        orientation="horizontal",
        positions=y_positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        manage_ticks=False,
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

    for y_value, probs, pred_labels in zip(
        y_positions,
        values_by_group,
        pred_labels_by_group,
        strict=True,
    ):
        offsets = _deterministic_offsets(probs.size, 0.18)
        colors = [
            _TRAIT_POSITIVE_COLOR if int(label) == 1 else _TRAIT_NEGATIVE_COLOR
            for label in pred_labels
        ]
        ax.scatter(
            probs,
            np.full(probs.shape[0], y_value, dtype=float) + offsets,
            s=16,
            color=colors,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.78,
            zorder=4,
        )

    ax.axvline(0.5, color="#999999", linewidth=0.8, linestyle=(0, (4, 4)))
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Predicted probability", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel(group_label, fontsize=_LABEL_FONTSIZE)
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_TRAIT_NEGATIVE_COLOR,
            markeredgecolor="white",
            markersize=4,
            label="pred 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_TRAIT_POSITIVE_COLOR,
            markeredgecolor="white",
            markersize=4,
            label="pred 1",
        ),
        Line2D([0], [0], color="#999999", linewidth=0.8, linestyle=(0, (4, 4)), label="0.5"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#dddddd",
        borderpad=0.25,
        handlelength=1.2,
        columnspacing=0.9,
    )

    fig.subplots_adjust(
        left=_label_left_margin(labels, width_px=width_px, fontsize_px=_MONO_FONTSIZE),
        right=0.985,
        top=0.90,
        bottom=_compact_bottom_margin(height_px),
    )
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


def _cv_curve_inputs(oof_predictions: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"fold_id", "label", "prob"}
    if not required.issubset(oof_predictions.columns):
        raise FigureError("prediction_cv.tsv schema is invalid for ROC/PR curve figures")
    if oof_predictions.height == 0:
        raise FigureError("prediction_cv.tsv is empty; cannot draw ROC/PR curve figures")

    y_true = np.array(oof_predictions.select("label").to_series().to_list(), dtype=int)
    prob = np.array(oof_predictions.select("prob").to_series().to_list(), dtype=float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        raise FigureError("ROC/PR curve figures could not be drawn (no folds with both labels)")
    return y_true, prob


def _binary_prediction_curve_inputs(
    predictions: pl.DataFrame,
    *,
    label_col: str,
    source_table_name: str,
    figure_name: str,
    empty_message: str,
    degenerate_message: str,
) -> tuple[np.ndarray, np.ndarray]:
    required = {label_col, "prob"}
    if not required.issubset(predictions.columns):
        raise FigureError(f"{source_table_name} schema is invalid for {figure_name}")
    if predictions.height == 0:
        raise FigureError(empty_message)

    data = predictions.select(
        pl.col(label_col).cast(pl.Int64, strict=False).alias("__label"),
        pl.col("prob").cast(pl.Float64, strict=False).alias("__prob"),
    ).filter(
        pl.col("__label").is_not_null()
        & pl.col("__prob").is_not_null()
        & pl.col("__prob").is_finite()
    )
    if data.height == 0:
        raise FigureError(empty_message)

    labels = [int(v) for v in data.select("__label").to_series().to_list()]
    non_binary = sorted({value for value in labels if value not in (0, 1)})
    if non_binary:
        values = ", ".join(str(value) for value in non_binary)
        raise FigureError(
            f"{source_table_name} contains non-binary labels for {figure_name}: {values}"
        )

    y_true = np.array(labels, dtype=int)
    prob = np.array(data.select("__prob").to_series().to_list(), dtype=float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        raise FigureError(degenerate_message)
    return y_true, prob


def _roc_curve_cv(y_true: np.ndarray, prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = float(roc_auc_score(y_true, prob))

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_CURVE_PANEL_SIZE_PX, _CURVE_PANEL_SIZE_PX),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.7, linestyle=(0, (4, 4)))
    ax.plot(fpr, tpr, color=_COLOR_BLUE, linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False Positive Rate", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=_LABEL_FONTSIZE)
    ax.grid(color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"ROC AUC={roc_auc:.6f}", fontsize=_LABEL_FONTSIZE)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.14)
    _save_svg_figure(fig, out_path)


def _pr_curve_cv(y_true: np.ndarray, prob: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    # Keep sklearn's threshold order; sorting recall can reorder tied-recall steps.
    recall_plot = np.asarray(recall, dtype=float)
    precision_plot = np.asarray(precision, dtype=float)
    average_precision = float(average_precision_score(y_true, prob))
    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_CURVE_PANEL_SIZE_PX, _CURVE_PANEL_SIZE_PX),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    ax.axhline(prevalence, color="#999999", linewidth=0.7, linestyle=(0, (4, 4)))
    ax.plot(
        recall_plot,
        precision_plot,
        color=_COLOR_ORANGE,
        linewidth=1.0,
        drawstyle="steps-post",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Recall", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=_LABEL_FONTSIZE)
    ax.grid(color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Average Precision={average_precision:.6f}, positive_rate={prevalence:.6f}",
        fontsize=_LABEL_FONTSIZE,
    )

    fig.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.14)
    _save_svg_figure(fig, out_path)


def _roc_pr_curves_cv(
    oof_predictions: pl.DataFrame,
    *,
    roc_out_path: Path,
    pr_out_path: Path,
) -> None:
    y_true, prob = _cv_curve_inputs(oof_predictions)
    _roc_curve_cv(y_true, prob, roc_out_path)
    _pr_curve_cv(y_true, prob, pr_out_path)


def _roc_curve_external(y_true: np.ndarray, prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = float(roc_auc_score(y_true, prob))

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 340),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.7, linestyle=(0, (4, 4)))
    ax.plot(fpr, tpr, color=_COLOR_BLUE, linewidth=1.15)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("True positive rate", fontsize=_LABEL_FONTSIZE)
    ax.grid(color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(
        0.97,
        0.05,
        f"ROC AUC = {roc_auc:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=_ANNOTATION_FONTSIZE,
        bbox={
            "boxstyle": "square,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "linewidth": 0.5,
        },
    )

    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.14)
    _save_svg_figure(fig, out_path)


def _pr_curve_external(y_true: np.ndarray, prob: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    average_precision = float(average_precision_score(y_true, prob))
    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 340),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    ax.axhline(prevalence, color="#999999", linewidth=0.7, linestyle=(0, (4, 4)))
    ax.plot(
        np.asarray(recall, dtype=float),
        np.asarray(precision, dtype=float),
        color=_COLOR_ORANGE,
        linewidth=1.15,
        drawstyle="steps-post",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Recall", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=_LABEL_FONTSIZE)
    ax.grid(color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.05,
        f"AP = {average_precision:.3f}\nPositive rate = {prevalence:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=_ANNOTATION_FONTSIZE,
        bbox={
            "boxstyle": "square,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "linewidth": 0.5,
        },
    )

    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.14)
    _save_svg_figure(fig, out_path)


def _external_roc_pr_curves(
    pred_external_test: pl.DataFrame,
    *,
    roc_out_path: Path,
    pr_out_path: Path,
) -> None:
    y_true, prob = _binary_prediction_curve_inputs(
        pred_external_test,
        label_col="true_label",
        source_table_name="prediction_external_test.tsv",
        figure_name="external ROC/PR curve figures",
        empty_message=(
            "prediction_external_test.tsv is empty; cannot draw external ROC/PR curve figures"
        ),
        degenerate_message=(
            "External ROC/PR curve figures could not be drawn (external_test requires both labels)"
        ),
    )
    _roc_curve_external(y_true, prob, roc_out_path)
    _pr_curve_external(y_true, prob, pr_out_path)


def _format_metric_for_confusion(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "NA"
    return f"{value:.3f}"


def _binary_metric_summary_from_counts(
    *, tn: int, fp: int, fn: int, tp: int
) -> dict[str, float | None]:
    n_total = tn + fp + fn + tp
    if n_total == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
            "mcc": None,
        }

    accuracy = float((tp + tn) / n_total)
    precision = None if (tp + fp) == 0 else float(tp / (tp + fp))
    recall = None if (tp + fn) == 0 else float(tp / (tp + fn))
    specificity = None if (tn + fp) == 0 else float(tn / (tn + fp))
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    mcc_denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float((tp * tn - fp * fn) / mcc_denom) if mcc_denom > 0.0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
    }


def _external_confusion_matrix(pred_external_test: pl.DataFrame, out_path: Path) -> None:
    required = {"true_label", "pred_label_fixed_threshold"}
    if not required.issubset(pred_external_test.columns):
        raise FigureError(
            "prediction_external_test.tsv schema is invalid for external_confusion_matrix.svg"
        )

    data = pred_external_test.select(
        pl.col("true_label").cast(pl.Int64, strict=False).alias("__true_label"),
        pl.col("pred_label_fixed_threshold").cast(pl.Int64, strict=False).alias("__pred_label"),
    ).filter(pl.col("__true_label").is_not_null() & pl.col("__pred_label").is_not_null())
    if data.height == 0:
        raise FigureError(
            "prediction_external_test.tsv is empty; cannot draw external_confusion_matrix.svg"
        )

    true_labels = [int(v) for v in data.select("__true_label").to_series().to_list()]
    pred_labels = [int(v) for v in data.select("__pred_label").to_series().to_list()]
    non_binary = sorted({value for value in [*true_labels, *pred_labels] if value not in (0, 1)})
    if non_binary:
        values = ", ".join(str(value) for value in non_binary)
        raise FigureError(
            "prediction_external_test.tsv contains non-binary labels for "
            f"external_confusion_matrix.svg: {values}"
        )

    matrix = np.zeros((2, 2), dtype=int)
    for true_label, pred_label in zip(true_labels, pred_labels, strict=True):
        matrix[true_label, pred_label] += 1
    tn = int(matrix[0, 0])
    fp = int(matrix[0, 1])
    fn = int(matrix[1, 0])
    tp = int(matrix[1, 1])
    metrics = _binary_metric_summary_from_counts(tn=tn, fp=fp, fn=fn, tp=tp)

    max_count = max(int(matrix.max()), 1)
    fig = plt.figure(
        figsize=_figure_size_inches(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 330),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.34)
    ax = fig.add_subplot(grid[0, 0])
    summary_ax = fig.add_subplot(grid[0, 1])

    image = ax.imshow(matrix, cmap=_CONFUSION_MATRIX_CMAP, vmin=0, vmax=max_count)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=_TICK_FONTSIZE)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=_TICK_FONTSIZE)
    ax.set_xlabel("Predicted label", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("True label", fontsize=_LABEL_FONTSIZE)
    ax.set_xticks(np.arange(-0.5, 2.0, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, 2.0, 1.0), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    row_totals = matrix.sum(axis=1)
    for row_idx in range(2):
        for col_idx in range(2):
            count = int(matrix[row_idx, col_idx])
            denominator = int(row_totals[row_idx])
            percent = None if denominator == 0 else count / denominator
            text = f"{count}"
            if percent is not None:
                text = f"{text}\n{percent:.1%}"
            text_color = "#ffffff" if count > max_count * 0.55 else _AXIS_COLOR
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=_LABEL_FONTSIZE,
                fontfamily="monospace",
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Count", rotation=90, fontsize=_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    summary_ax.axis("off")
    summary_lines = [
        f"n = {data.height}",
        f"Accuracy = {_format_metric_for_confusion(metrics['accuracy'])}",
        f"Precision = {_format_metric_for_confusion(metrics['precision'])}",
        f"Recall = {_format_metric_for_confusion(metrics['recall'])}",
        f"Specificity = {_format_metric_for_confusion(metrics['specificity'])}",
        f"F1 = {_format_metric_for_confusion(metrics['f1'])}",
        f"MCC = {_format_metric_for_confusion(metrics['mcc'])}",
    ]
    summary_ax.text(
        0.0,
        0.95,
        "\n".join(summary_lines),
        transform=summary_ax.transAxes,
        ha="left",
        va="top",
        fontsize=_MONO_FONTSIZE,
        fontfamily="monospace",
        linespacing=1.45,
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.16)
    _save_svg_figure(fig, out_path)


def _cv_external_metric_comparison(
    classification_summary: pl.DataFrame,
    out_path: Path,
) -> None:
    metric_columns = {metric for metric, _label in _CV_EXTERNAL_METRIC_ORDER}
    required = {"pool", "fold_id", "threshold_name", *metric_columns}
    if not required.issubset(classification_summary.columns):
        raise FigureError(
            "classification_summary.tsv schema is invalid for cv_external_metric_comparison.svg"
        )

    data = classification_summary.select(
        [
            pl.col("pool").cast(pl.String, strict=False).alias("__pool"),
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("threshold_name").cast(pl.String, strict=False).alias("__threshold_name"),
            *[
                pl.col(metric).cast(pl.Float64, strict=False).alias(metric)
                for metric in metric_columns
            ],
        ]
    ).filter(
        pl.col("__pool").is_in(["validation_oof", "external_test"])
        & (pl.col("__fold_id").is_null() | (pl.col("__fold_id") == "NA"))
    )
    if data.height == 0:
        raise FigureError(
            "classification_summary.tsv has no pooled validation/external rows for "
            "cv_external_metric_comparison.svg"
        )

    fixed_threshold = data.filter(pl.col("__threshold_name") == FIXED_PROBABILITY_THRESHOLD_NAME)
    if fixed_threshold.height > 0:
        data = fixed_threshold
    else:
        threshold_names = [
            str(value)
            for value in data.select("__threshold_name").drop_nulls().unique().to_series().to_list()
        ]
        if threshold_names:
            selected_threshold = sorted(threshold_names)[0]
            data = data.filter(pl.col("__threshold_name") == selected_threshold)

    pool_order = ["validation_oof", "external_test"]
    pool_labels = {
        "validation_oof": "Validation OOF",
        "external_test": "External test",
    }
    pool_values: dict[str, list[float]] = {}
    for pool in pool_order:
        subset = data.filter(pl.col("__pool") == pool)
        if subset.height == 0:
            raise FigureError(
                "classification_summary.tsv must contain pooled validation_oof and "
                "external_test rows for cv_external_metric_comparison.svg"
            )
        row = subset.row(0, named=True)
        values: list[float] = []
        for metric, _label in _CV_EXTERNAL_METRIC_ORDER:
            raw = row[metric]
            values.append(np.nan if raw is None else float(raw))
        pool_values[pool] = values

    finite_values = [
        value for values in pool_values.values() for value in values if np.isfinite(value)
    ]
    if not finite_values:
        _write_message_figure(
            title="CV External Metric Comparison",
            message="No finite metric values are available.",
            out_path=out_path,
            width_px=_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX,
            height_px=320,
        )
        return

    has_negative = min(finite_values) < 0.0
    if has_negative:
        y_min = -1.12
        y_max = 1.12
        y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    else:
        y_min = 0.0
        y_max = 1.12
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    metric_labels = [label for _metric, label in _CV_EXTERNAL_METRIC_ORDER]
    x_positions = np.arange(len(metric_labels), dtype=float)
    bar_width = 0.34
    colors = {
        "validation_oof": _COLOR_BLUE,
        "external_test": _COLOR_ORANGE,
    }
    offsets = {
        "validation_oof": -bar_width / 2,
        "external_test": bar_width / 2,
    }

    fig, ax = plt.subplots(
        figsize=_figure_size_inches(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 340),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")

    for pool in pool_order:
        value_array = np.array(pool_values[pool], dtype=float)
        positions = x_positions + offsets[pool]
        finite_mask = np.isfinite(value_array)
        ax.bar(
            positions[finite_mask],
            value_array[finite_mask],
            width=bar_width,
            color=colors[pool],
            label=pool_labels[pool],
        )
        for x_value, value in zip(positions, value_array, strict=True):
            if np.isfinite(value):
                if value >= 0.0:
                    text_y = float(value) + 0.035
                    va = "bottom"
                else:
                    text_y = float(value) - 0.035
                    va = "top"
                ax.text(
                    x_value,
                    text_y,
                    f"{float(value):.3f}",
                    ha="center",
                    va=va,
                    fontsize=_MONO_FONTSIZE,
                    fontfamily="monospace",
                    rotation=90,
                )
            else:
                ax.text(
                    x_value,
                    0.02 if not has_negative else 0.04,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=_MONO_FONTSIZE,
                    fontfamily="monospace",
                    color=_MUTED_TEXT_COLOR,
                    rotation=90,
                )

    ax.axhline(0.0, color=_AXIS_COLOR, linewidth=0.7)
    ax.set_xlim(-0.55, len(metric_labels) - 0.45)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_labels, fontsize=_TICK_FONTSIZE)
    ax.set_ylabel("Score", fontsize=_LABEL_FONTSIZE)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#dddddd",
        borderpad=0.25,
        handlelength=1.2,
        columnspacing=1.0,
    )

    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.16)
    _save_svg_figure(fig, out_path)


def _single_report_metric_name(
    data: pl.DataFrame,
    *,
    column: str,
    artifact_name: str,
) -> str:
    metric_names = [
        str(value) for value in data.select(column).drop_nulls().unique().to_series().to_list()
    ]
    if len(metric_names) != 1:
        raise FigureError(f"{artifact_name} must contain exactly one metric name")
    metric_name = metric_names[0]
    try:
        metric_direction(metric_name)
    except ValueError as exc:
        raise FigureError(str(exc)) from exc
    return metric_name


def _report_metric_axis_label(metric_name: str, metric_display_name: str | None = None) -> str:
    direction = metric_direction(metric_name)
    direction_label = "higher is better" if direction == "maximize" else "lower is better"
    metric_label = (
        metric_name if metric_display_name is None else f"{metric_display_name} [{metric_name}]"
    )
    return f"{metric_label} ({direction_label})"


def _report_metric_display_name(frame: pl.DataFrame) -> str | None:
    if "metric_display_name" not in frame.columns:
        return None
    values = frame.select("metric_display_name").drop_nulls().unique().to_series().to_list()
    if len(values) != 1:
        return None
    return str(values[0])


def _report_metric_ranking(report_ranking: pl.DataFrame, out_path: Path) -> None:
    required = {"rank", "run_id", "metric_name", "metric_value"}
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
    metric_name = _single_report_metric_name(
        top,
        column="metric_name",
        artifact_name="report_ranking.tsv",
    )
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
        x_label=_report_metric_axis_label(metric_name, _report_metric_display_name(top)),
        y_tick_fontsize=_MONO_FONTSIZE,
    )


def _sorted_report_metric_rows(report_runs: pl.DataFrame) -> pl.DataFrame:
    required = {"run_id", "metric_value", "start_time", "primary_metric"}
    if not required.issubset(report_runs.columns):
        raise FigureError("report_runs.tsv schema is invalid for report_metric_comparison.svg")

    comparable = report_runs.drop_nulls("metric_value")
    if comparable.height == 0:
        return comparable
    metric_name = _single_report_metric_name(
        comparable,
        column="primary_metric",
        artifact_name="report_runs.tsv",
    )
    return comparable.sort(
        by=["metric_value", "start_time", "run_id"],
        descending=[metric_higher_is_better(metric_name), False, False],
    ).head(30)


def _report_metric_comparison(report_runs: pl.DataFrame, out_path: Path) -> None:
    comparable = _sorted_report_metric_rows(report_runs)
    if comparable.height == 0:
        _write_message_figure(
            title="Report Metric Comparison",
            message="No comparable runs with metric values",
            out_path=out_path,
            width_px=_NATURE_DOUBLE_COLUMN_WIDTH_PX,
            height_px=320,
        )
        return

    metric_name = _single_report_metric_name(
        comparable,
        column="primary_metric",
        artifact_name="report_runs.tsv",
    )
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
        x_label=_report_metric_axis_label(metric_name, _report_metric_display_name(comparable)),
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
    summary = scored.group_by(
        ["fold_id", "sample_set_id", "candidate_index", "metric_name", "params_json"]
    ).agg(
        [
            pl.len().alias("n_inner_folds"),
            pl.col("_metric_value_valid").count().alias("n_valid_inner_folds"),
            pl.col("_metric_value_valid").mean().alias("metric_value_mean"),
            pl.col("_metric_value_valid").std(ddof=0).alias("metric_value_std"),
        ]
    )
    return summary.with_columns(
        pl.when(pl.col("n_valid_inner_folds") > 0)
        .then(pl.col("metric_value_std") / pl.col("n_valid_inner_folds").cast(pl.Float64).sqrt())
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("metric_value_se")
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
            parsed = {key: value for key, value in sorted(parsed.items()) if key in include_keys}
        if not parsed:
            return "{}"
        return json.dumps(parsed, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if parsed is None:
        return "null"
    return json.dumps(parsed, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _model_selection_metric_axis_label(metric_names: Sequence[str]) -> str:
    unique = sorted({str(value) for value in metric_names})
    if len(unique) != 1:
        return "Score"
    return {
        "mcc": "MCC",
        "balanced_accuracy": "Balanced Accuracy",
        "log_loss": "Log Loss",
    }.get(unique[0], unique[0].replace("_", " ").title())


def _model_selection_higher_is_better(metric_name: str) -> bool:
    return metric_higher_is_better(metric_name)


def _numeric_param_from_json(params_json: str | None, key: str) -> float | None:
    params = _params_dict(params_json)
    if params is None or key not in params:
        return None
    try:
        value = float(params[key])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _model_selection_summary_with_se(
    model_selection_trials_summary: pl.DataFrame,
) -> pl.DataFrame:
    summary = model_selection_trials_summary
    if "params_json" not in summary.columns:
        summary = summary.with_columns(pl.lit("{}").alias("params_json"))
    if "metric_value_std" not in summary.columns:
        summary = summary.with_columns(pl.lit(None, dtype=pl.Float64).alias("metric_value_std"))
    if "metric_value_se" not in summary.columns:
        if "n_valid_inner_folds" in summary.columns:
            summary = summary.with_columns(
                pl.when(pl.col("n_valid_inner_folds").cast(pl.Float64, strict=False) > 0)
                .then(
                    pl.col("metric_value_std").cast(pl.Float64, strict=False)
                    / pl.col("n_valid_inner_folds").cast(pl.Float64, strict=False).sqrt()
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("metric_value_se")
            )
        else:
            summary = summary.with_columns(pl.col("metric_value_std").alias("metric_value_se"))
    return summary


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

    summary = _model_selection_summary_with_se(model_selection_trials_summary)

    data = (
        summary.select(
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("sample_set_id").cast(pl.Int64, strict=False).alias("__sample_set_id"),
            pl.col("candidate_index").cast(pl.Int64, strict=False).alias("__candidate_index"),
            pl.col("metric_name").cast(pl.String, strict=False).alias("__metric_name"),
            pl.col("metric_value_mean").cast(pl.Float64, strict=False).alias("__mean"),
            pl.col("metric_value_std").cast(pl.Float64, strict=False).alias("__std"),
            pl.col("metric_value_se").cast(pl.Float64, strict=False).alias("__se"),
            pl.col("params_json").cast(pl.String, strict=False).alias("__params_json"),
        )
        .with_columns(
            pl.when(pl.col("__se").is_null() | pl.col("__se").is_nan() | (pl.col("__se") < 0.0))
            .then(0.0)
            .otherwise(pl.col("__se"))
            .alias("__se_plot")
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
        ses = np.array(panel_data.select("__se_plot").to_series().to_list(), dtype=float)
        params_json_values = [
            None if value is None else str(value)
            for value in panel_data.select("__params_json").to_series().to_list()
        ]
        varying_keys = _varying_param_keys(params_json_values)
        params_labels = [
            _compact_params_label(value, include_keys=varying_keys) for value in params_json_values
        ]
        y_labels = [
            _ellipsize_label(f"{candidate}: {params_label}", max_chars=72)
            for candidate, params_label in zip(candidates, params_labels, strict=True)
        ]

        max_candidates = max(max_candidates, len(candidates))
        max_label_length = max(max_label_length, max(len(label) for label in y_labels))
        for mean_value, se_value in zip(means.tolist(), ses.tolist(), strict=True):
            x_values.extend([mean_value - se_value, mean_value + se_value])

        panels.append(
            {
                "fold_id": fold_id,
                "sample_set_id": sample_set_id,
                "candidates": candidates,
                "means": means,
                "ses": ses,
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

    metric_names = [str(v) for v in data.select("__metric_name").unique().to_series().to_list()]
    metric_axis_label = f"{_model_selection_metric_axis_label(metric_names)} mean +/- SE"
    for panel_index, panel in enumerate(panels):
        row_index, col_index = divmod(panel_index, n_cols)
        ax = axes[row_index][col_index]
        y_pos = np.arange(len(panel["candidates"]), dtype=float)
        ax.errorbar(
            panel["means"],
            y_pos,
            xerr=panel["ses"],
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


def _model_selection_one_se_curve(
    model_selection_trials_summary: pl.DataFrame,
    model_selection_selected: pl.DataFrame | None,
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
    }
    if not required.issubset(model_selection_trials_summary.columns):
        raise FigureError(
            "model_selection_trials_summary.tsv schema is invalid for "
            "model_selection_one_se_curve.svg"
        )

    summary = _model_selection_summary_with_se(model_selection_trials_summary)
    data = (
        summary.select(
            pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
            pl.col("sample_set_id").cast(pl.Int64, strict=False).alias("__sample_set_id"),
            pl.col("candidate_index").cast(pl.Int64, strict=False).alias("__candidate_index"),
            pl.col("metric_name").cast(pl.String, strict=False).alias("__metric_name"),
            pl.col("metric_value_mean").cast(pl.Float64, strict=False).alias("__mean"),
            pl.col("metric_value_se").cast(pl.Float64, strict=False).alias("__se"),
            pl.col("params_json").cast(pl.String, strict=False).alias("__params_json"),
        )
        .with_columns(
            pl.when(pl.col("__se").is_null() | pl.col("__se").is_nan() | (pl.col("__se") < 0.0))
            .then(0.0)
            .otherwise(pl.col("__se"))
            .alias("__se_plot")
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

    selected_by_key: dict[tuple[str, int], int] = {}
    if model_selection_selected is not None:
        selected_required = {
            "selection_scope",
            "fold_id",
            "sample_set_id",
            "candidate_index",
            "rank",
        }
        if not selected_required.issubset(model_selection_selected.columns):
            raise FigureError(
                "model_selection_selected.tsv schema is invalid for "
                "model_selection_one_se_curve.svg"
            )
        selected_rows = (
            model_selection_selected.select(
                pl.col("selection_scope").cast(pl.String, strict=False).alias("__scope"),
                pl.col("fold_id").cast(pl.String, strict=False).alias("__fold_id"),
                pl.col("sample_set_id").cast(pl.Int64, strict=False).alias("__sample_set_id"),
                pl.col("candidate_index").cast(pl.Int64, strict=False).alias("__candidate_index"),
                pl.col("rank").cast(pl.Int64, strict=False).alias("__rank"),
            )
            .filter(
                (pl.col("__scope") == "outer_fold")
                & (pl.col("__rank") == 1)
                & pl.col("__fold_id").is_not_null()
                & pl.col("__sample_set_id").is_not_null()
                & pl.col("__candidate_index").is_not_null()
            )
            .to_dicts()
        )
        for row in selected_rows:
            selected_by_key[(str(row["__fold_id"]), int(row["__sample_set_id"]))] = int(
                row["__candidate_index"]
            )

    fold_ids = [str(v) for v in data.select("__fold_id").unique().to_series().to_list()]
    fold_ids = sorted(
        fold_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    panels: list[dict[str, Any]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    use_log_c_values: list[bool] = []

    for fold_id in fold_ids:
        sample_set_ids = sorted(
            int(v)
            for v in data.filter(pl.col("__fold_id") == fold_id)
            .select("__sample_set_id")
            .unique()
            .to_series()
            .to_list()
        )[:max_sample_sets_per_fold]
        for sample_set_id in sample_set_ids:
            panel_data = data.filter(
                (pl.col("__fold_id") == fold_id) & (pl.col("__sample_set_id") == sample_set_id)
            ).sort("__candidate_index")
            if panel_data.height == 0:
                continue

            rows = panel_data.to_dicts()
            c_values = [
                _numeric_param_from_json(
                    None if row["__params_json"] is None else str(row["__params_json"]), "C"
                )
                for row in rows
            ]
            use_log_c = all(value is not None and value > 0.0 for value in c_values)
            if use_log_c:
                positive_c_values = [float(value) for value in c_values if value is not None]
                x_values = np.array([np.log10(value) for value in positive_c_values], dtype=float)
                x_label = "log10(C)"
            else:
                x_values = np.array([int(row["__candidate_index"]) for row in rows], dtype=float)
                x_label = "candidate_index"
            means = np.array([float(row["__mean"]) for row in rows], dtype=float)
            ses = np.array([float(row["__se_plot"]) for row in rows], dtype=float)
            candidate_indices = [int(row["__candidate_index"]) for row in rows]
            metric_names = [str(row["__metric_name"]) for row in rows]
            metric_name = metric_names[0]
            higher_is_better = _model_selection_higher_is_better(metric_name)
            best_offset = int(np.argmax(means) if higher_is_better else np.argmin(means))
            best_mean = float(means[best_offset])
            best_se = float(ses[best_offset])
            threshold = best_mean - best_se if higher_is_better else best_mean + best_se
            eligible = means >= threshold if higher_is_better else means <= threshold
            selected_candidate = selected_by_key.get((fold_id, sample_set_id))

            order = np.argsort(x_values)
            panels.append(
                {
                    "fold_id": fold_id,
                    "sample_set_id": sample_set_id,
                    "x": x_values[order],
                    "means": means[order],
                    "ses": ses[order],
                    "eligible": eligible[order],
                    "candidate_indices": [candidate_indices[int(index)] for index in order],
                    "metric_names": metric_names,
                    "metric_name": metric_name,
                    "threshold": threshold,
                    "best_candidate": candidate_indices[best_offset],
                    "selected_candidate": selected_candidate,
                    "x_label": x_label,
                }
            )
            all_x.extend(x_values.tolist())
            for mean, se in zip(means.tolist(), ses.tolist(), strict=True):
                all_y.extend([mean - se, mean + se])
            all_y.append(threshold)
            use_log_c_values.append(use_log_c)

    if not panels:
        return

    x_label = "log10(C)" if all(use_log_c_values) else "candidate_index"
    metric_axis_label = _model_selection_metric_axis_label(
        [panel["metric_name"] for panel in panels]
    )
    if metric_axis_label == "Log Loss":
        metric_axis_label = "Log loss"
    x_min, x_max = _padded_domain(all_x, include_zero=False)
    y_min, y_max = _padded_domain(all_y, include_zero=False)

    n_panels = len(panels)
    max_cols = min(4, n_panels)
    n_cols = min(
        range(1, max_cols + 1),
        key=lambda cols: (
            abs((cols / int(np.ceil(n_panels / cols))) - 1.4)
            + (int(np.ceil(n_panels / cols)) * cols - n_panels) * 0.15,
            int(np.ceil(n_panels / cols)) * cols - n_panels,
        ),
    )
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_width_px = max(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, 230 * n_cols + 80)
    fig_height_px = max(250, 200 * n_rows + 66)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_figure_size_inches(fig_width_px, fig_height_px),
        dpi=_FIG_DPI,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for panel_index, panel in enumerate(panels):
        row_index, col_index = divmod(panel_index, n_cols)
        ax = axes[row_index][col_index]
        x_values = np.asarray(panel["x"], dtype=float)
        means = np.asarray(panel["means"], dtype=float)
        ses = np.asarray(panel["ses"], dtype=float)
        eligible = np.asarray(panel["eligible"], dtype=bool)
        candidate_indices = [int(value) for value in panel["candidate_indices"]]
        ax.plot(x_values, means, color=_COLOR_BLUE, linewidth=0.8, alpha=0.75)
        ax.errorbar(
            x_values,
            means,
            yerr=ses,
            fmt="none",
            ecolor=_COLOR_SKY,
            elinewidth=0.8,
            capsize=2.2,
            zorder=1,
        )
        ax.scatter(
            x_values[~eligible],
            means[~eligible],
            s=18,
            color=_COLOR_BLUE,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
            label="Candidate",
        )
        ax.scatter(
            x_values[eligible],
            means[eligible],
            s=24,
            color=_COLOR_GREEN,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            label="Within one-SE",
        )
        ax.axhline(
            float(panel["threshold"]),
            color=_COLOR_ORANGE,
            linewidth=0.9,
            linestyle="--",
            label="one-SE threshold",
        )

        best_candidate = int(panel["best_candidate"])
        if best_candidate in candidate_indices:
            best_offset = candidate_indices.index(best_candidate)
            ax.scatter(
                [x_values[best_offset]],
                [means[best_offset]],
                marker="D",
                s=28,
                color=_COLOR_PURPLE,
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
                label="Best mean",
            )
        selected_candidate = panel["selected_candidate"]
        if selected_candidate is not None and int(selected_candidate) in candidate_indices:
            selected_offset = candidate_indices.index(int(selected_candidate))
            ax.scatter(
                [x_values[selected_offset]],
                [means[selected_offset]],
                marker="*",
                s=72,
                color=_COLOR_ORANGE,
                edgecolor=_AXIS_COLOR,
                linewidth=0.4,
                zorder=5,
                label="Selected candidate",
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(color=_GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        title = f"fold={panel['fold_id']}"
        if int(panel["sample_set_id"]) != 0:
            title += f", sample_set={panel['sample_set_id']}"
        ax.set_title(title, fontsize=_LABEL_FONTSIZE, pad=5.0)
        if col_index == 0:
            ax.set_ylabel(f"{metric_axis_label} mean", fontsize=_LABEL_FONTSIZE)
        ax.set_xlabel(x_label, fontsize=_LABEL_FONTSIZE)
        if x_label == "candidate_index":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for panel_index in range(n_panels, n_rows * n_cols):
        row_index, col_index = divmod(panel_index, n_cols)
        axes[row_index][col_index].axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_COLOR_GREEN,
            label="Within one-SE",
        ),
        Line2D([0], [0], color=_COLOR_ORANGE, linestyle="--", label="one-SE threshold"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=_COLOR_PURPLE,
            label="Best mean",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=_COLOR_ORANGE,
            markeredgecolor=_AXIS_COLOR,
            label="Selected candidate",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.93, bottom=0.18, wspace=0.30, hspace=0.48)
    _save_svg_figure(fig, out_path)


def _feature_filter_funnel(
    feature_filter_counts_summary: pl.DataFrame,
    out_path: Path,
    *,
    stage_order: Sequence[str] | None = None,
    scopes: Sequence[str] | None = None,
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
    if scopes is not None:
        scope_values = [str(scope) for scope in scopes]
        data = data.filter(pl.col("scope").is_in(scope_values))
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
    for scope_index, scope in enumerate(scopes):
        scope_rows = data.filter(pl.col("scope") == scope)
        stage_to_stats: dict[str, tuple[float, float, float, float, float]] = {}
        for row in scope_rows.iter_rows(named=True):
            stage = str(row["stage"])
            minimum = float(row["n_features_min"])
            q1_raw = row["n_features_q1"]
            median_raw = row["n_features_median"]
            q3_raw = row["n_features_q3"]
            maximum = float(row["n_features_max"])
            q1 = np.nan if q1_raw is None else float(q1_raw)
            median = np.nan if median_raw is None else float(median_raw)
            q3 = np.nan if q3_raw is None else float(q3_raw)
            stage_to_stats[stage] = (minimum, q1, median, q3, maximum)
        y_min: list[float] = []
        y_q1: list[float] = []
        y_median: list[float] = []
        y_q3: list[float] = []
        y_max: list[float] = []
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
    ax.set_ylabel("Number of selected features", fontsize=_LABEL_FONTSIZE)
    ax.margins(y=0.15)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
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


def _feature_stability_top(
    feature_stability: pl.DataFrame,
    out_path: Path,
    *,
    top_features: int = _DEFAULT_TOP_FEATURES,
    orthogroup_annotations: pl.DataFrame | None = None,
) -> None:
    required = {
        "feature",
        "retained_frequency",
        "selection_frequency",
        "dominant_sign",
    }
    if not required.issubset(feature_stability.columns):
        raise FigureError(
            "feature_stability_by_feature.tsv schema is invalid for feature_stability_top.svg"
        )
    if top_features < 1:
        raise FigureError("figures.top_features must be >= 1")

    top = (
        feature_stability.select(
            pl.col("feature").cast(pl.String, strict=False).alias("feature"),
            pl.col("retained_frequency").cast(pl.Float64, strict=False).alias("retained_frequency"),
            pl.col("selection_frequency")
            .cast(pl.Float64, strict=False)
            .alias("selection_frequency"),
            pl.col("dominant_sign").cast(pl.String, strict=False).alias("dominant_sign"),
        )
        .filter(
            pl.col("feature").is_not_null()
            & (pl.col("feature") != "")
            & pl.col("retained_frequency").is_not_null()
            & pl.col("retained_frequency").is_finite()
            & pl.col("selection_frequency").is_not_null()
            & pl.col("selection_frequency").is_finite()
        )
        .sort(
            ["selection_frequency", "retained_frequency", "feature"],
            descending=[True, True, False],
        )
        .head(top_features)
    )
    if top.height == 0:
        _write_message_figure(
            title="Feature Stability",
            message="No finite feature-stability rows are available.",
            out_path=out_path,
            width_px=_FEATURE_IMPORTANCE_TOP_WIDTH_PX,
            height_px=300,
        )
        return

    features = [str(value) for value in top.get_column("feature").to_list()]
    feature_labels = _feature_axis_labels(features, orthogroup_annotations)
    retained = np.asarray(top.get_column("retained_frequency").to_list(), dtype=float)
    selected = np.asarray(top.get_column("selection_frequency").to_list(), dtype=float)
    signs = [None if value is None else str(value) for value in top["dominant_sign"].to_list()]
    sign_colors = {
        "positive": _TRAIT_POSITIVE_COLOR,
        "negative": _TRAIT_NEGATIVE_COLOR,
        "tie": _COLOR_PURPLE,
    }
    colors = [
        sign_colors.get(sign, _COLOR_BLUE) if sign is not None else _COLOR_BLUE for sign in signs
    ]

    row_height_px = _feature_label_row_height_px(feature_labels)
    height_px = max(260, 85 + len(features) * row_height_px)
    base_width_px = _FEATURE_IMPORTANCE_TOP_WIDTH_PX
    base_left = _label_left_margin(
        features,
        width_px=base_width_px,
        fontsize_px=_MONO_FONTSIZE,
    )
    width_px, left_margin, right_margin = _feature_axis_layout(
        feature_labels,
        base_width_px=base_width_px,
        base_left=base_left,
        base_right=0.985,
        fontsize_px=_MONO_FONTSIZE,
    )
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    y_pos = np.arange(len(features), dtype=float)
    ax.barh(y_pos, retained, color="#dddddd", height=0.68, label="Retained")
    ax.barh(y_pos, selected, color=colors, height=0.42, label="Non-zero")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=_MONO_FONTSIZE, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Outer-fold frequency", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel(_feature_label_axis_title(feature_labels), fontsize=_LABEL_FONTSIZE)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    legend_handles = [
        Patch(facecolor="#dddddd", label="Retained after preprocessing"),
        Patch(facecolor=_TRAIT_POSITIVE_COLOR, label="Non-zero; positive coefficient"),
        Patch(facecolor=_TRAIT_NEGATIVE_COLOR, label="Non-zero; negative coefficient"),
        Patch(facecolor=_COLOR_PURPLE, label="Non-zero; tied coefficient sign"),
        Patch(facecolor=_COLOR_BLUE, label="Non-zero; sign unavailable"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=_TICK_FONTSIZE)
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=0.98,
        bottom=_compact_bottom_margin(height_px),
    )
    _save_svg_figure(fig, out_path)


def _feature_set_jaccard_heatmap(
    feature_stability_by_fold_pair: pl.DataFrame,
    out_path: Path,
) -> None:
    required = {"fold_id_a", "fold_id_b", "jaccard"}
    if not required.issubset(feature_stability_by_fold_pair.columns):
        raise FigureError(
            "feature_stability_by_fold_pair.tsv schema is invalid for "
            "feature_set_jaccard_heatmap.svg"
        )
    if feature_stability_by_fold_pair.height == 0:
        _write_message_figure(
            title="Feature-set Jaccard by Fold",
            message="At least two outer folds are required for pairwise stability.",
            out_path=out_path,
            width_px=_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX,
            height_px=300,
        )
        return

    pair_data = feature_stability_by_fold_pair.select(
        pl.col("fold_id_a").cast(pl.String, strict=False).alias("fold_id_a"),
        pl.col("fold_id_b").cast(pl.String, strict=False).alias("fold_id_b"),
        pl.col("jaccard").cast(pl.Float64, strict=False).alias("jaccard"),
    ).filter(
        pl.col("fold_id_a").is_not_null()
        & pl.col("fold_id_b").is_not_null()
        & (pl.col("fold_id_a") != "")
        & (pl.col("fold_id_b") != "")
    )
    fold_ids = sorted(
        set(str(value) for value in pair_data["fold_id_a"].to_list())
        | set(str(value) for value in pair_data["fold_id_b"].to_list()),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if not fold_ids:
        raise FigureError("No fold identifiers are available for Jaccard heatmap")

    fold_index = {fold_id: index for index, fold_id in enumerate(fold_ids)}
    matrix = np.full((len(fold_ids), len(fold_ids)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    for row in pair_data.iter_rows(named=True):
        value = row["jaccard"]
        if value is None or not np.isfinite(float(value)):
            continue
        index_a = fold_index[str(row["fold_id_a"])]
        index_b = fold_index[str(row["fold_id_b"])]
        matrix[index_a, index_b] = float(value)
        matrix[index_b, index_a] = float(value)

    width_px = _fold_axis_width_px(len(fold_ids), base_px=260, per_fold_px=36)
    width_px = max(_NATURE_ONE_AND_HALF_COLUMN_WIDTH_PX, width_px)
    height_px = max(420, 145 + len(fold_ids) * 24)
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    cmap = _FEATURE_IMPORTANCE_HEATMAP_CMAP.copy()
    cmap.set_bad("#ffffff")
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="equal",
        cmap=cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    positions = np.arange(len(fold_ids), dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels(fold_ids, rotation=90, fontsize=_TICK_FONTSIZE)
    ax.set_yticks(positions)
    ax.set_yticklabels(fold_ids, fontsize=_TICK_FONTSIZE)
    ax.set_xlabel("Outer CV fold", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Outer CV fold", fontsize=_LABEL_FONTSIZE)
    ax.set_xticks(np.arange(-0.5, len(fold_ids), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(fold_ids), 1.0), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    if len(fold_ids) <= 10:
        for row_index in range(len(fold_ids)):
            for col_index in range(len(fold_ids)):
                value = matrix[row_index, col_index]
                if not np.isfinite(value):
                    continue
                color = "#ffffff" if value > 0.55 else _AXIS_COLOR
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=_MONO_FONTSIZE,
                    color=color,
                    fontfamily="monospace",
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Jaccard similarity", fontsize=_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=_TICK_FONTSIZE)
    fig.subplots_adjust(left=0.12, right=0.92, top=0.98, bottom=0.16)
    _save_svg_figure(fig, out_path)


def write_run_figures(
    *,
    run_dir: Path,
    metrics_cv: pl.DataFrame,
    oof_predictions: pl.DataFrame,
    feature_importance: pl.DataFrame,
    coefficients: pl.DataFrame,
    ensemble_model_probs: pl.DataFrame | None,
    model_selection_trials: pl.DataFrame | None,
    feature_importance_by_fold: pl.DataFrame | None = None,
    coefficients_by_fold: pl.DataFrame | None = None,
    feature_stability_by_feature: pl.DataFrame | None = None,
    feature_stability_by_fold_pair: pl.DataFrame | None = None,
    loss_by_split_cv: pl.DataFrame | None = None,
    loss_by_split_final_refit: pl.DataFrame | None = None,
    pred_external_test: pl.DataFrame | None = None,
    pred_inference: pl.DataFrame | None = None,
    classification_summary: pl.DataFrame | None = None,
    trait_name: str = "trait",
    model_selection_trials_summary: pl.DataFrame | None = None,
    model_selection_selected: pl.DataFrame | None = None,
    feature_filter_counts_summary: pl.DataFrame | None = None,
    model_sparsity: pl.DataFrame | None = None,
    model_sparsity_summary: pl.DataFrame | None = None,
    top_feature_expression: pl.DataFrame | None = None,
    feature_filter_funnel_stage_order: Sequence[str] | None = None,
    top_features: int = _DEFAULT_TOP_FEATURES,
    orthogroup_annotations: pl.DataFrame | None = None,
    parallel_workers: int = 1,
    group_bootstrap_metrics: pl.DataFrame | None = None,
) -> list[str]:
    """Write run-level SVG figures under <run_dir>/<stage>/figures."""
    stage_dirs = _stage_figure_dirs(run_dir)
    cv_dir = stage_dirs["cv"]
    external_test_dir = stage_dirs["external_test"]
    inference_dir = stage_dirs["inference"]
    feature_label_annotations = _feature_label_annotations_subset(
        orthogroup_annotations,
        feature_importance=feature_importance,
        coefficients=coefficients,
        top_features=top_features,
    )
    jobs: list[_FigureJob] = []

    def add_job(
        name: str,
        func: Callable[..., None],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        catch_figure_error: bool = False,
    ) -> None:
        jobs.append((name, func, args, {} if kwargs is None else kwargs, catch_figure_error))

    add_job(
        "cv_metrics_overview",
        _cv_metrics_overview,
        (metrics_cv, cv_dir / "cv_metrics_overview.svg"),
    )
    if group_bootstrap_metrics is not None:
        add_job(
            "group_bootstrap_metrics",
            _group_bootstrap_metrics_figure,
            (group_bootstrap_metrics, cv_dir / "group_bootstrap_metrics.svg"),
            catch_figure_error=True,
        )
    if loss_by_split_cv is not None:
        add_job(
            "cv_loss_by_split",
            _cv_loss_by_split,
            (loss_by_split_cv, cv_dir / "cv_loss_by_split.svg"),
        )
    if loss_by_split_final_refit is not None:
        add_job(
            "final_refit_loss_by_split",
            _final_refit_loss_by_split,
            (loss_by_split_final_refit, external_test_dir / "final_refit_loss_by_split.svg"),
        )
    add_job(
        "feature_importance_top",
        _feature_importance_top,
        (feature_importance, cv_dir / "feature_importance_top.svg"),
        {
            "feature_importance_by_fold": feature_importance_by_fold,
            "top_features": top_features,
            "orthogroup_annotations": feature_label_annotations,
        },
    )
    if feature_importance_by_fold is not None:
        add_job(
            "feature_importance_by_fold_heatmap",
            _feature_importance_by_fold_heatmap,
            (
                feature_importance,
                feature_importance_by_fold,
                cv_dir / "feature_importance_by_fold_heatmap.svg",
            ),
            {
                "top_features": top_features,
                "orthogroup_annotations": feature_label_annotations,
            },
        )
    add_job(
        "coefficients_signed_top",
        _coefficients_signed_top,
        (coefficients, cv_dir / "coefficients_signed_top.svg"),
        {
            "coefficients_by_fold": coefficients_by_fold,
            "top_features": top_features,
            "orthogroup_annotations": feature_label_annotations,
        },
    )
    if top_feature_expression is not None:
        add_job(
            "top_feature_expression_by_confusion",
            _top_feature_expression_by_confusion,
            kwargs={
                "oof_predictions": oof_predictions,
                "top_feature_expression": top_feature_expression,
                "feature_importance": feature_importance,
                "coefficients": coefficients,
                "out_path": cv_dir / "top_feature_expression_by_confusion.svg",
                "top_features": top_features,
                "orthogroup_annotations": feature_label_annotations,
            },
            catch_figure_error=True,
        )
    if feature_stability_by_feature is not None:
        add_job(
            "feature_stability_top",
            _feature_stability_top,
            (feature_stability_by_feature, cv_dir / "feature_stability_top.svg"),
            {
                "top_features": top_features,
                "orthogroup_annotations": orthogroup_annotations,
            },
        )
    if feature_stability_by_fold_pair is not None:
        add_job(
            "feature_set_jaccard_heatmap",
            _feature_set_jaccard_heatmap,
            (
                feature_stability_by_fold_pair,
                cv_dir / "feature_set_jaccard_heatmap.svg",
            ),
        )
    add_job(
        "cv_species_probability_by_trait",
        _species_probability_by_trait,
        kwargs={
            "predictions": oof_predictions,
            "trait_col": "label",
            "trait_name": trait_name,
            "out_path": cv_dir / "cv_species_probability_by_trait.svg",
            "title": "CV Species Probability by Trait",
            "subtitle": "Out-of-fold probabilities grouped by observed trait labels",
            "source_table_name": "prediction_cv.tsv",
            "figure_name": "cv_species_probability_by_trait.svg",
        },
    )
    add_job(
        "cv_fold_trait_probability",
        _cv_fold_trait_probability,
        (oof_predictions, cv_dir / "cv_fold_trait_probability.svg"),
        {"trait_name": trait_name},
    )
    add_job(
        "roc_pr_curves_cv",
        _roc_pr_curves_cv,
        (oof_predictions,),
        {
            "roc_out_path": cv_dir / "roc_curve_cv.svg",
            "pr_out_path": cv_dir / "pr_curve_cv.svg",
        },
        catch_figure_error=True,
    )
    selection_summary = model_selection_trials_summary
    if selection_summary is None and model_selection_trials is not None:
        selection_summary = _summarize_model_selection_trials_for_figure(model_selection_trials)
    if selection_summary is not None:
        add_job(
            "model_selection_trials",
            _model_selection_trials_summary_panels,
            (selection_summary, cv_dir / "model_selection_trials.svg"),
            {"max_sample_sets_per_fold": _MODEL_SELECTION_SAMPLE_SET_LIMIT},
        )
        add_job(
            "model_selection_one_se_curve",
            _model_selection_one_se_curve,
            (
                selection_summary,
                model_selection_selected,
                cv_dir / "model_selection_one_se_curve.svg",
            ),
            {"max_sample_sets_per_fold": _MODEL_SELECTION_SAMPLE_SET_LIMIT},
        )
    if feature_filter_counts_summary is not None:
        add_job(
            "cv_feature_filter_funnel",
            _feature_filter_funnel,
            (feature_filter_counts_summary, cv_dir / "feature_filter_funnel.svg"),
            {
                "stage_order": feature_filter_funnel_stage_order,
                "scopes": ("outer_fold",),
            },
        )
        final_refit_feature_filter = feature_filter_counts_summary.filter(
            pl.col("scope") == "final_refit"
        )
        if final_refit_feature_filter.height > 0:
            add_job(
                "final_refit_feature_filter_funnel",
                _feature_filter_funnel,
                (feature_filter_counts_summary, external_test_dir / "feature_filter_funnel.svg"),
                {
                    "stage_order": feature_filter_funnel_stage_order,
                    "scopes": ("final_refit",),
                },
            )
    if model_sparsity is not None:
        add_job(
            "non_zero_feature_count_by_fold",
            _non_zero_feature_count_by_fold,
            (model_sparsity, cv_dir / "non_zero_feature_count_by_fold.svg"),
        )
    if pred_external_test is not None:
        if classification_summary is not None:
            add_job(
                "cv_external_metric_comparison",
                _cv_external_metric_comparison,
                (classification_summary, external_test_dir / "cv_external_metric_comparison.svg"),
                catch_figure_error=True,
            )
        add_job(
            "external_confusion_matrix",
            _external_confusion_matrix,
            (pred_external_test, external_test_dir / "external_confusion_matrix.svg"),
            catch_figure_error=True,
        )
        add_job(
            "external_species_probability_by_trait",
            _species_probability_by_trait,
            kwargs={
                "predictions": pred_external_test,
                "trait_col": "true_label",
                "trait_name": trait_name,
                "out_path": external_test_dir / "external_species_probability_by_trait.svg",
                "title": "External Test Species Probability by Trait",
                "subtitle": "Final-refit probabilities grouped by external-test true labels",
                "source_table_name": "prediction_external_test.tsv",
                "figure_name": "external_species_probability_by_trait.svg",
            },
            catch_figure_error=True,
        )
        add_job(
            "external_roc_pr_curves",
            _external_roc_pr_curves,
            (pred_external_test,),
            {
                "roc_out_path": external_test_dir / "external_roc_curve.svg",
                "pr_out_path": external_test_dir / "external_pr_curve.svg",
            },
            catch_figure_error=True,
        )
    if pred_inference is not None:
        add_job(
            "inference_probability_distribution",
            _predict_probability_distribution,
            (pred_inference, inference_dir / "inference_probability_distribution.svg"),
            {"figure_name": "inference_probability_distribution.svg"},
            catch_figure_error=True,
        )
        add_job(
            "species_probability_cv_and_inference",
            _species_probability_cv_and_inference,
            kwargs={
                "oof_predictions": oof_predictions,
                "pred_inference": pred_inference,
                "trait_name": trait_name,
                "out_path": inference_dir / "species_probability_cv_and_inference.svg",
            },
            catch_figure_error=True,
        )
    return _run_figure_jobs(jobs, parallel_workers=parallel_workers)


_CANDIDATE_MANIFEST_SCHEMA = {
    "species": pl.String,
    "family_id": pl.String,
    "family_name": pl.String,
    "prob": pl.Float64,
    "probability_bin": pl.String,
    "n_cross_fold_predictions": pl.Int64,
    "cross_fold_prob_min": pl.Float64,
    "cross_fold_prob_q1": pl.Float64,
    "cross_fold_prob_median": pl.Float64,
    "cross_fold_prob_q3": pl.Float64,
    "cross_fold_prob_max": pl.Float64,
    "n_features": pl.Int64,
    "figure_path": pl.String,
}


def _candidate_probability_bin(probability: float) -> str:
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise FigureError("Candidate probability must be finite and within [0, 1]")
    if probability >= 0.95:
        return "p_095_100"
    if probability >= 0.90:
        return "p_090_095"
    if probability >= 0.85:
        return "p_085_090"
    if probability >= 0.80:
        return "p_080_085"
    return "p_050_080"


def _candidate_filename(species: str, *, used_names: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", species.strip()).strip("._-")
    if not stem:
        stem = "candidate"
    filename = f"{stem}.pdf"
    if filename in used_names:
        digest = sha256(species.encode("utf-8")).hexdigest()[:8]
        filename = f"{stem}_{digest}.pdf"
    used_names.add(filename)
    return filename


def _candidate_feature_labels(
    features: list[str],
    orthogroup_annotations: pl.DataFrame | None,
) -> list[str]:
    annotation_lookup: dict[str, str] = {}
    if orthogroup_annotations is not None and {
        "feature",
        "orthogroup_annotation",
    }.issubset(orthogroup_annotations.columns):
        for row in orthogroup_annotations.select("feature", "orthogroup_annotation").iter_rows(
            named=True
        ):
            feature = str(row["feature"]).strip()
            annotation = row["orthogroup_annotation"]
            if annotation is None:
                continue
            annotation_text = str(annotation).strip()
            if annotation_text:
                annotation_lookup[feature] = annotation_text

    labels: list[str] = []
    for feature in features:
        annotation = annotation_lookup.get(feature, "Unannotated orthogroup")
        shortened = textwrap.shorten(annotation, width=58, placeholder="…")
        wrapped = textwrap.fill(shortened, width=31)
        labels.append(f"{wrapped}\n({feature})")
    return labels


def _finite_distribution_summary(values: np.ndarray) -> tuple[float, float, float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise FigureError("Candidate evidence distribution contains no finite values")
    q10, q25, median, q75, q90 = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90])
    return float(q10), float(q25), float(median), float(q75), float(q90)


def _candidate_evidence_figure(
    *,
    candidate: dict[str, Any],
    candidate_features: pl.DataFrame,
    reference_expression: pl.DataFrame,
    cross_fold_predictions: pl.DataFrame,
    orthogroup_annotations: pl.DataFrame | None,
    trait_name: str,
    out_path: Path,
) -> None:
    required_features = {
        "feature",
        "local_rank",
        "contribution_mean",
        "candidate_log2_tpm_plus1",
    }
    missing_features = sorted(required_features - set(candidate_features.columns))
    if missing_features:
        raise FigureError(
            "Candidate feature evidence schema is invalid: " + ", ".join(missing_features)
        )
    if candidate_features.height == 0:
        raise FigureError("Candidate feature evidence is empty")
    required_reference = {"feature", "label", "log2_tpm_plus1"}
    missing_reference = sorted(required_reference - set(reference_expression.columns))
    if missing_reference:
        raise FigureError(
            "Candidate reference expression schema is invalid: " + ", ".join(missing_reference)
        )

    ordered = candidate_features.sort("local_rank")
    features = [str(value) for value in ordered.get_column("feature").to_list()]
    labels = _candidate_feature_labels(features, orthogroup_annotations)
    contributions = np.asarray(ordered.get_column("contribution_mean").to_list(), dtype=float)
    candidate_expression = np.asarray(
        ordered.get_column("candidate_log2_tpm_plus1").to_list(), dtype=float
    )
    species = str(candidate["species"])
    probability = float(candidate["prob"])
    family_id = str(candidate.get("family_id") or "unassigned")
    family_name = str(candidate.get("family_name") or family_id)
    family_text = family_name
    if family_id not in {"", "unassigned", family_name}:
        family_text = f"{family_name} ({family_id})"

    feature_count = len(features)
    figure_height = max(4.8, 1.85 + 0.39 * feature_count)
    fig = plt.figure(
        figsize=(_NATURE_DOUBLE_COLUMN_WIDTH_PX / _FIG_DPI, figure_height),
        dpi=_FIG_DPI,
    )
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[0.75, max(2.4, 0.32 * feature_count)],
        width_ratios=[0.43, 0.57],
        left=0.285,
        right=0.98,
        top=0.81,
        bottom=max(0.08, 0.36 / figure_height),
        hspace=0.54,
        wspace=0.22,
    )
    ax_stability = fig.add_subplot(grid[0, :])
    ax_contribution = fig.add_subplot(grid[1, 0])
    ax_expression = fig.add_subplot(grid[1, 1], sharey=ax_contribution)

    fig.text(
        0.025,
        0.955,
        species,
        ha="left",
        va="top",
        fontsize=10,
        fontstyle="italic",
        color=_AXIS_COLOR,
    )
    fig.text(
        0.025,
        0.905,
        f"Family: {family_text}",
        ha="left",
        va="top",
        fontsize=7,
        color=_MUTED_TEXT_COLOR,
    )
    fig.text(
        0.98,
        0.947,
        f"P({trait_name} = 1) = {probability:.3f}",
        ha="right",
        va="top",
        fontsize=9,
        color=_AXIS_COLOR,
    )

    ax_stability.set_title(
        "A   Prediction stability across outer-CV models",
        loc="left",
        fontsize=8,
        pad=6,
    )
    fold_values = np.asarray(
        cross_fold_predictions.get_column("prob").to_list()
        if "prob" in cross_fold_predictions.columns
        else [],
        dtype=float,
    )
    fold_values = fold_values[np.isfinite(fold_values)]
    if fold_values.size > 0:
        q1, median, q3 = np.quantile(fold_values, [0.25, 0.50, 0.75])
        ax_stability.hlines(
            0.0,
            float(np.min(fold_values)),
            float(np.max(fold_values)),
            color="#7a7a7a",
            linewidth=0.9,
            zorder=1,
        )
        ax_stability.hlines(
            0.0,
            float(q1),
            float(q3),
            color="#4d4d4d",
            linewidth=4.0,
            zorder=2,
        )
        jitter = np.linspace(-0.045, 0.045, fold_values.size)
        ax_stability.scatter(
            fold_values,
            jitter,
            s=13,
            color="#777777",
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )
        ax_stability.scatter(
            [float(median)],
            [0.0],
            s=24,
            color="#222222",
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        summary_text = (
            f"Outer-CV median {float(median):.3f} "
            f"(range {float(np.min(fold_values)):.3f}–{float(np.max(fold_values)):.3f})"
        )
        ax_stability.text(
            0.995,
            0.96,
            summary_text,
            transform=ax_stability.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            color=_MUTED_TEXT_COLOR,
        )
    ax_stability.scatter(
        [probability],
        [0.12],
        marker="D",
        s=31,
        color="#111111",
        edgecolors="white",
        linewidths=0.55,
        zorder=5,
        label="Final refit",
    )
    ax_stability.set_xlim(0.0, 1.0)
    ax_stability.set_ylim(-0.14, 0.22)
    ax_stability.set_yticks([])
    ax_stability.set_xlabel(f"Predicted probability of {trait_name} = 1", labelpad=2)
    ax_stability.grid(axis="x", alpha=0.7)
    ax_stability.spines["left"].set_visible(False)
    ax_stability.legend(loc="upper left", frameon=False, fontsize=6, handletextpad=0.3)

    y_positions = np.arange(feature_count, dtype=float)
    bar_colors = np.where(
        contributions >= 0.0,
        _TRAIT_POSITIVE_COLOR,
        "#D55E00",
    )
    ax_contribution.barh(
        y_positions,
        contributions,
        height=0.58,
        color=bar_colors.tolist(),
        edgecolor="none",
    )
    contribution_limit = max(float(np.max(np.abs(contributions))), 1e-9) * 1.12
    ax_contribution.set_xlim(-contribution_limit, contribution_limit)
    ax_contribution.axvline(0.0, color=_AXIS_COLOR, linewidth=0.65)
    ax_contribution.set_yticks(y_positions, labels=labels)
    ax_contribution.tick_params(axis="y", length=0, pad=5, labelsize=6)
    ax_contribution.invert_yaxis()
    ax_contribution.set_xlabel(f"Contribution to {trait_name} = 1 linear score")
    ax_contribution.set_title("B   Local contribution", loc="left", fontsize=8, pad=7)
    ax_contribution.grid(axis="x", alpha=0.55)
    ax_contribution.spines["left"].set_visible(False)

    reference = reference_expression.filter(pl.col("feature").is_in(features))
    reference_values: list[float] = []
    label_counts: dict[int, int] = {}
    for label in (0, 1):
        label_counts[label] = (
            reference.filter(pl.col("label") == label).get_column("species").n_unique()
        )
    expression_colors = {0: "#D55E00", 1: _TRAIT_POSITIVE_COLOR}
    offsets = {0: -0.14, 1: 0.14}
    for feature_idx, feature in enumerate(features):
        for label in (0, 1):
            values = np.asarray(
                reference.filter((pl.col("feature") == feature) & (pl.col("label") == label))
                .get_column("log2_tpm_plus1")
                .to_list(),
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            reference_values.extend(values.tolist())
            y_value = float(feature_idx) + offsets[label]
            ax_expression.scatter(
                values,
                np.full(values.size, y_value),
                s=8,
                color=expression_colors[label],
                alpha=0.25,
                linewidths=0,
                zorder=1,
            )
            q10, q25, median, q75, q90 = _finite_distribution_summary(values)
            ax_expression.hlines(
                y_value,
                q10,
                q90,
                color=expression_colors[label],
                linewidth=0.75,
                zorder=2,
            )
            ax_expression.hlines(
                y_value,
                q25,
                q75,
                color=expression_colors[label],
                linewidth=2.7,
                zorder=3,
            )
            ax_expression.scatter(
                [median],
                [y_value],
                s=13,
                color=expression_colors[label],
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
            )
        ax_expression.scatter(
            [candidate_expression[feature_idx]],
            [float(feature_idx)],
            marker="D",
            s=24,
            color="#111111",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    all_expression = np.asarray([*reference_values, *candidate_expression.tolist()], dtype=float)
    finite_expression = all_expression[np.isfinite(all_expression)]
    expression_max = max(float(np.max(finite_expression)), 1.0) if finite_expression.size else 1.0
    ax_expression.set_xlim(0.0, expression_max * 1.04)
    ax_expression.set_ylim(ax_contribution.get_ylim())
    ax_expression.tick_params(axis="y", left=False, labelleft=False)
    ax_expression.set_xlabel(r"Expression, $\log_2(\mathrm{TPM} + 1)$")
    ax_expression.set_title("C   Expression evidence", loc="left", fontsize=8, pad=7)
    ax_expression.grid(axis="x", alpha=0.55)
    ax_expression.spines["left"].set_visible(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=expression_colors[0],
            markersize=3.5,
            linewidth=1.5,
            label=f"Known 0 (n={label_counts[0]})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=expression_colors[1],
            markersize=3.5,
            linewidth=1.5,
            label=f"Known 1 (n={label_counts[1]})",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color="#111111",
            markersize=4,
            label="Candidate",
        ),
    ]
    ax_expression.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.10),
        frameon=False,
        ncol=3,
        fontsize=5.8,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_pdf_figure(fig, out_path, title=f"Candidate evidence: {species}")


def write_candidate_evidence_figures(
    *,
    run_dir: Path,
    candidates: pl.DataFrame,
    features: pl.DataFrame,
    reference_expression: pl.DataFrame,
    cross_fold_predictions: pl.DataFrame,
    trait_name: str,
    orthogroup_annotations: pl.DataFrame | None = None,
    parallel_workers: int = 1,
) -> tuple[pl.DataFrame, list[str]]:
    """Write one publication-oriented candidate-evidence PDF per positive species."""
    if candidates.height == 0 or features.height == 0:
        return pl.DataFrame(schema=_CANDIDATE_MANIFEST_SCHEMA), []
    required_candidates = {"species", "prob", "family_id", "family_name"}
    missing_candidates = sorted(required_candidates - set(candidates.columns))
    if missing_candidates:
        raise FigureError(
            "Candidate evidence candidates schema is invalid: " + ", ".join(missing_candidates)
        )

    output_root = run_dir / "inference" / "figures" / "candidate_evidence"
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[_FigureJob] = []
    manifest_rows: list[dict[str, Any]] = []
    used_names_by_bin: dict[str, set[str]] = {}
    ordered_candidates = candidates.sort(["prob", "species"], descending=[True, False])
    for candidate in ordered_candidates.iter_rows(named=True):
        species = str(candidate["species"])
        probability = float(candidate["prob"])
        bin_name = _candidate_probability_bin(probability)
        used_names = used_names_by_bin.setdefault(bin_name, set())
        filename = _candidate_filename(species, used_names=used_names)
        relative_path = Path(bin_name) / filename
        out_path = output_root / relative_path
        candidate_features = features.filter(pl.col("species") == species).sort("local_rank")
        if candidate_features.height == 0:
            continue
        feature_names = candidate_features.get_column("feature").to_list()
        reference_subset = reference_expression.filter(pl.col("feature").is_in(feature_names))
        cross_fold_subset = cross_fold_predictions.filter(pl.col("species") == species)
        jobs.append(
            (
                f"candidate_evidence_{species}",
                _candidate_evidence_figure,
                (),
                {
                    "candidate": candidate,
                    "candidate_features": candidate_features,
                    "reference_expression": reference_subset,
                    "cross_fold_predictions": cross_fold_subset,
                    "orthogroup_annotations": orthogroup_annotations,
                    "trait_name": trait_name,
                    "out_path": out_path,
                },
                False,
            )
        )
        fold_values = np.asarray(
            cross_fold_subset.get_column("prob").to_list()
            if "prob" in cross_fold_subset.columns
            else [],
            dtype=float,
        )
        fold_values = fold_values[np.isfinite(fold_values)]
        if fold_values.size:
            q1, median, q3 = np.quantile(fold_values, [0.25, 0.50, 0.75])
            fold_min: float | None = float(np.min(fold_values))
            fold_max: float | None = float(np.max(fold_values))
            fold_q1: float | None = float(q1)
            fold_median: float | None = float(median)
            fold_q3: float | None = float(q3)
        else:
            fold_min = fold_q1 = fold_median = fold_q3 = fold_max = None
        manifest_rows.append(
            {
                "species": species,
                "family_id": str(candidate["family_id"]),
                "family_name": str(candidate["family_name"]),
                "prob": probability,
                "probability_bin": bin_name,
                "n_cross_fold_predictions": int(fold_values.size),
                "cross_fold_prob_min": fold_min,
                "cross_fold_prob_q1": fold_q1,
                "cross_fold_prob_median": fold_median,
                "cross_fold_prob_q3": fold_q3,
                "cross_fold_prob_max": fold_max,
                "n_features": candidate_features.height,
                "figure_path": relative_path.as_posix(),
            }
        )

    warnings = _run_figure_jobs(jobs, parallel_workers=parallel_workers)
    manifest = pl.DataFrame(manifest_rows, schema=_CANDIDATE_MANIFEST_SCHEMA).sort(
        ["prob", "species"], descending=[True, False]
    )
    manifest.write_csv(
        output_root / "candidate_manifest.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    return manifest, warnings


def write_predict_figures(
    *,
    run_dir: Path,
    pred_predict: pl.DataFrame,
    require_uncertainty: bool = False,
) -> None:
    """Write predict-level SVG figures under <run_dir>/inference/figures."""
    inference_dir = _stage_figure_dirs(run_dir)["inference"]
    _predict_probability_distribution(
        pred_predict,
        inference_dir / "predict_probability_distribution.svg",
    )
    _predict_uncertainty(
        pred_predict,
        inference_dir / "predict_uncertainty.svg",
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
