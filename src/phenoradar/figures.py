"""Deterministic SVG figure generation for run/predict/report artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import polars as pl
from matplotlib.figure import Figure
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
matplotlib.rcParams["svg.hashsalt"] = "phenoradar"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.edgecolor"] = "#222222"
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["grid.color"] = "#ececec"
matplotlib.rcParams["grid.linewidth"] = 0.8


class FigureError(ValueError):
    """Raised when figure generation cannot proceed."""


_FIG_DPI = 100


def _figure_size_inches(width_px: int, height_px: int) -> tuple[float, float]:
    return width_px / _FIG_DPI, height_px / _FIG_DPI


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
    fig.suptitle(title, x=0.01, ha="left", fontsize=16)
    ax.axis("off")
    ax.text(
        0.01,
        0.60,
        message,
        transform=ax.transAxes,
        fontsize=11,
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
    fig.suptitle(title, x=0.01, ha="left", fontsize=16)
    if subtitle is not None:
        fig.text(0.01, 0.90, subtitle, fontsize=10)

    y_pos = np.arange(len(labels), dtype=float)
    bars = ax.barh(y_pos, values, color=color, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=y_tick_fontsize, fontfamily="monospace")
    ax.invert_yaxis()

    right_limit = max_value * 1.15
    ax.set_xlim(0.0, right_limit)
    ax.grid(axis="x", color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(x_label)

    value_offset = right_limit * 0.01
    for bar, value in zip(bars, values, strict=True):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            float(value) + value_offset,
            y,
            value_formatter(float(value)),
            va="center",
            ha="left",
            fontsize=9,
            fontfamily="monospace",
        )

    fig.subplots_adjust(left=left_margin, right=right_margin, top=0.82, bottom=0.10)
    _save_svg_figure(fig, out_path)


def _format_float(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    return f"{value:.8f}"


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

    fold_count = (
        metrics_cv.filter(pl.col("aggregate_scope") == "NA")
        .select("fold_id")
        .unique()
        .height
    )

    fig, ax = plt.subplots(figsize=_figure_size_inches(1240, 540), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("CV Metrics Overview", x=0.01, ha="left", fontsize=16)
    fig.text(0.01, 0.90, f"macro/micro aggregate metrics, fold_count={fold_count}", fontsize=10)

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
        color="#1f77b4",
        label="macro",
    )
    ax.bar(
        x_pos[micro_mask] + bar_width / 2,
        micro_values[micro_mask],
        width=bar_width,
        color="#ff7f0e",
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
                fontsize=9,
                color="#666666",
            )
    for idx, score in enumerate(micro_values):
        if np.isnan(score):
            ax.text(
                x_pos[idx] + bar_width / 2,
                0.0,
                "NA",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#666666",
            )

    ax.set_xlim(-0.75, len(metrics) - 0.25)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.axhline(0.0, color="#333333", linewidth=1.2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.20)
    _save_svg_figure(fig, out_path)


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

    fig, ax = plt.subplots(figsize=_figure_size_inches(1240, 520), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("Threshold Selection Curve", x=0.01, ha="left", fontsize=16)
    fig.text(0.01, 0.90, f"selection_metric={selection_metric}", fontsize=10)

    if score_points:
        score_points = sorted(score_points, key=lambda item: item[0])
        thresholds_plot = np.array([point[0] for point in score_points], dtype=float)
        scores_plot = np.array([point[1] for point in score_points], dtype=float)
        score_min, score_max = _padded_domain(scores_plot.tolist(), include_zero=True)

        ax.plot(thresholds_plot, scores_plot, color="#1f77b4", linewidth=2.2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(score_min, score_max)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(f"{selection_metric} score")
        ax.grid(color="#ececec", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color="#555555", linewidth=1.2)

        if np.isfinite(selected_threshold):
            ax.axvline(
                selected_threshold,
                color="#d62728",
                linewidth=1.5,
                linestyle=(0, (5, 4)),
            )

        selected_score = _metric_score(y_true, prob, selected_threshold, selection_metric)
        if np.isfinite(selected_score) and np.isfinite(selected_threshold):
            ax.scatter([selected_threshold], [selected_score], color="#d62728", s=30, zorder=3)
            selected_text = (
                f"selected_threshold={selected_threshold:.8f}, "
                f"score={selected_score:.8f}"
            )
        else:
            selected_text = f"selected_threshold={selected_threshold:.8f}, score=NaN"
        fig.text(0.01, 0.86, selected_text, fontsize=10)
    else:
        ax.axis("off")
        ax.text(
            0.02,
            0.50,
            "No valid threshold scores (all NaN)",
            transform=ax.transAxes,
            fontsize=12,
        )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16)
    _save_svg_figure(fig, out_path)


def _feature_importance_top(feature_importance: pl.DataFrame, out_path: Path) -> None:
    required = {"feature", "importance_mean"}
    if not required.issubset(feature_importance.columns):
        raise FigureError("feature_importance.tsv schema is invalid for feature_importance_top.svg")

    top = feature_importance.sort(
        by=["importance_mean", "feature"],
        descending=[True, False],
    ).head(30)
    if top.height == 0:
        raise FigureError("feature_importance.tsv is empty; cannot draw feature_importance_top.svg")

    features = [str(v) for v in top.select("feature").to_series().to_list()]
    values = [float(v) for v in top.select("importance_mean").to_series().to_list()]
    max_value = max(values) if values else 1.0
    if np.isclose(max_value, 0.0):
        max_value = 1.0

    height_px = max(320, 140 + len(features) * 24)
    fig, ax = plt.subplots(figsize=_figure_size_inches(1300, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("Feature Importance Top", x=0.01, ha="left", fontsize=16)
    fig.text(0.01, 0.90, "Top 30 by importance_mean", fontsize=10)

    y_pos = np.arange(len(features), dtype=float)
    bars = ax.barh(y_pos, values, color="#2ca02c", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()

    right_limit = max_value * 1.15
    ax.set_xlim(0.0, right_limit)
    ax.set_xlabel("importance_mean")
    ax.grid(axis="x", color="#ececec", linewidth=0.8)
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
            fontsize=9,
            fontfamily="monospace",
        )

    fig.subplots_adjust(left=0.33, right=0.94, top=0.84, bottom=0.12)
    _save_svg_figure(fig, out_path)


def _coefficients_signed_top(coefficients: pl.DataFrame, out_path: Path) -> None:
    required = {"feature", "coef_mean", "method"}
    if not required.issubset(coefficients.columns):
        raise FigureError("coefficients.tsv schema is invalid for coefficients_signed_top.svg")

    linear = coefficients.filter(pl.col("method") == "coef_signed").drop_nulls("coef_mean")
    if linear.height == 0:
        return

    top = linear.with_columns(pl.col("coef_mean").abs().alias("__abs_coef")).sort(
        by=["__abs_coef", "feature"],
        descending=[True, False],
    ).head(30)

    features = [str(v) for v in top.select("feature").to_series().to_list()]
    values = [float(v) for v in top.select("coef_mean").to_series().to_list()]
    max_abs = max(abs(v) for v in values) if values else 1.0
    if np.isclose(max_abs, 0.0):
        max_abs = 1.0

    height_px = max(320, 150 + len(features) * 24)
    fig, ax = plt.subplots(figsize=_figure_size_inches(1340, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("Coefficients Signed Top", x=0.01, ha="left", fontsize=16)
    fig.text(0.01, 0.90, "Top 30 by |coef_mean|", fontsize=10)

    y_pos = np.arange(len(features), dtype=float)
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in values]
    bars = ax.barh(y_pos, values, color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()

    limit = max_abs * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("coef_mean (signed)")
    ax.grid(axis="x", color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color="#444444", linewidth=1.3)
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
            fontsize=9,
            fontfamily="monospace",
        )

    fig.subplots_adjust(left=0.31, right=0.94, top=0.84, bottom=0.12)
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

    fig, ax = plt.subplots(figsize=_figure_size_inches(960, 460), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("Predict Probability Distribution", x=0.01, ha="left", fontsize=16)

    bin_starts = np.arange(10, dtype=float) / 10.0
    bars = ax.bar(bin_starts, counts.tolist(), width=0.08, align="edge", color="#17becf")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max_count * 1.15)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#ececec", linewidth=0.8)
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
            fontsize=9,
            fontfamily="monospace",
        )

    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.16)
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

    height_px = max(260, 80 + len(species) * 24)
    fig, ax = plt.subplots(figsize=_figure_size_inches(1200, height_px), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("Predict Uncertainty", x=0.01, ha="left", fontsize=16)

    y_pos = np.arange(len(species), dtype=float)
    bars = ax.barh(y_pos, values, color="#8c564b", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(species, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()

    right_limit = max_val * 1.15
    ax.set_xlim(0.0, right_limit)
    ax.set_xlabel("uncertainty_std")
    ax.grid(axis="x", color="#ececec", linewidth=0.8)
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
            fontsize=9,
            fontfamily="monospace",
        )

    fig.subplots_adjust(left=0.30, right=0.94, top=0.84, bottom=0.12)
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
        0: "#d62728",
        1: "#1f77b4",
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
    x_labels: list[str] = []
    positions = np.arange(1, len(traits) + 1, dtype=float)
    for trait in traits:
        trait_df = data.filter(pl.col("__trait") == trait).sort(["__prob", "__species"])
        probs = [float(v) for v in trait_df.select("__prob").to_series().to_list()]
        if not probs:
            continue
        group_probs.append(probs)
        x_labels.append(f"{trait_name}={trait}\nn={len(probs)}, mean={float(np.mean(probs)):.3f}")

    if not group_probs:
        raise FigureError(f"{source_table_name} is empty; cannot draw {figure_name}")

    fig, ax = plt.subplots(figsize=_figure_size_inches(980, 560), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, x=0.01, ha="left", fontsize=16)
    fig.text(
        0.01,
        0.90,
        (
            f"{subtitle}; trait={trait_name}; "
            "box=IQR/median, marker=mean, points=species "
            f"(n={data.height})"
        ),
        fontsize=10,
    )

    box = ax.boxplot(
        group_probs,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        manage_ticks=False,
        meanprops={"marker": "D", "markerfacecolor": "#222222", "markeredgecolor": "#222222"},
        medianprops={"linewidth": 1.6, "color": "#222222"},
        whiskerprops={"linewidth": 1.2, "color": "#444444"},
        capprops={"linewidth": 1.2, "color": "#444444"},
    )
    for idx, patch in enumerate(box["boxes"]):
        color = trait_to_color[traits[idx]]
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    for idx, trait in enumerate(traits):
        trait_df = data.filter(pl.col("__trait") == trait).sort(["__prob", "__species"])
        probs = np.array(trait_df.select("__prob").to_series().to_list(), dtype=float)
        offsets = _deterministic_offsets(probs.size, 0.17)
        x_values = np.full(probs.shape[0], positions[idx], dtype=float) + offsets
        ax.scatter(
            x_values,
            probs,
            s=32,
            color=trait_to_color[trait],
            edgecolors="white",
            linewidths=0.5,
            alpha=0.78,
            zorder=3,
        )

    ax.set_xlim(0.5, len(traits) + 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Trait")
    ax.set_ylabel("Predicted probability")
    ax.grid(axis="y", color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.24)
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

    width_px = max(980, 260 + len(fold_ids) * 150)
    fig, ax = plt.subplots(figsize=_figure_size_inches(width_px, 560), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("CV Fold Trait Probability", x=0.01, ha="left", fontsize=16)
    fig.text(
        0.01,
        0.90,
        (
            f"Fold-wise probability distribution by {trait_name} "
            "(box=IQR/median, marker=mean, points=species)"
        ),
        fontsize=10,
    )

    for trait_idx, trait in enumerate(traits):
        label_set = False
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
                s=25,
                color=trait_to_color[trait],
                edgecolors="white",
                linewidths=0.5,
                alpha=0.75,
                zorder=3,
                label=f"{trait_name}={trait}" if not label_set else "_nolegend_",
            )
            label_set = True

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
            meanprops={"marker": "D", "markerfacecolor": "#222222", "markeredgecolor": "#222222"},
            medianprops={"linewidth": 1.6, "color": "#222222"},
            whiskerprops={"linewidth": 1.2, "color": "#444444"},
            capprops={"linewidth": 1.2, "color": "#444444"},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(trait_to_color[trait])
            patch.set_alpha(0.28)
            patch.set_edgecolor(trait_to_color[trait])
            patch.set_linewidth(1.2)

    ax.set_xlim(0.4, len(fold_ids) + 0.6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(fold_centers)
    ax.set_xticklabels([f"fold={fold_id}" for fold_id in fold_ids], fontsize=10)
    ax.set_xlabel("CV fold")
    ax.set_ylabel("Predicted probability")
    ax.grid(axis="y", color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, title=trait_name)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.18)
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

    fold_count = oof_predictions.select("fold_id").unique().height

    fpr, tpr, _ = roc_curve(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    recall_order = np.argsort(recall)
    recall_plot = np.asarray(recall[recall_order], dtype=float)
    precision_plot = np.asarray(precision[recall_order], dtype=float)
    roc_auc = float(roc_auc_score(y_true, prob))
    pr_auc = float(average_precision_score(y_true, prob))
    prevalence = float(np.mean(y_true))

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=_figure_size_inches(1180, 500), dpi=_FIG_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("ROC/PR Curves (CV)", x=0.01, ha="left", fontsize=16)
    fig.text(
        0.01,
        0.90,
        f"OOF pooled curves across folds, n={y_true.size}, fold_count={fold_count}",
        fontsize=10,
    )

    ax_roc.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=1.0, linestyle=(0, (4, 4)))
    ax_roc.plot(fpr, tpr, color="#1f77b4", linewidth=2.4)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(color="#efefef", linewidth=0.8)
    ax_roc.set_axisbelow(True)
    ax_roc.set_title(f"ROC AUC={roc_auc:.6f}", fontsize=11)

    ax_pr.axhline(prevalence, color="#999999", linewidth=1.0, linestyle=(0, (4, 4)))
    ax_pr.plot(recall_plot, precision_plot, color="#ff7f0e", linewidth=2.4)
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(color="#efefef", linewidth=0.8)
    ax_pr.set_axisbelow(True)
    ax_pr.set_title(f"PR AUC={pr_auc:.6f}, positive_rate={prevalence:.6f}", fontsize=11)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14, wspace=0.28)
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
            width_px=1200,
            height_px=500,
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
        color="#1f77b4",
        width_px=1200,
        min_height_px=260,
        row_height_px=24,
        base_height_px=80,
        left_margin=0.30,
        right_margin=0.94,
        x_label="metric_value",
        y_tick_fontsize=9,
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
            width_px=1200,
            height_px=500,
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
        color="#ff7f0e",
        width_px=1200,
        min_height_px=260,
        row_height_px=24,
        base_height_px=80,
        left_margin=0.30,
        right_margin=0.94,
        x_label="metric_value",
        y_tick_fontsize=9,
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
        color="#2ca02c",
        width_px=900,
        min_height_px=220,
        row_height_px=40,
        base_height_px=80,
        left_margin=0.28,
        right_margin=0.92,
        x_label="count",
        y_tick_fontsize=11,
        value_formatter=_as_int,
    )


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
    pred_external_test: pl.DataFrame | None = None,
    trait_name: str = "trait",
) -> list[str]:
    """Write run-level SVG figures under <run_dir>/figures."""
    warnings: list[str] = []
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)
    _cv_metrics_overview(metrics_cv, figures_dir / "cv_metrics_overview.svg")
    _threshold_selection_curve(
        oof_predictions,
        thresholds,
        selection_metric=auto_threshold_metric,
        out_path=figures_dir / "threshold_selection_curve.svg",
    )
    _feature_importance_top(feature_importance, figures_dir / "feature_importance_top.svg")
    _coefficients_signed_top(coefficients, figures_dir / "coefficients_signed_top.svg")
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
