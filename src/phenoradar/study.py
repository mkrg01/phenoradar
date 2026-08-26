"""Multi-condition study manifests, paired comparisons, and figures."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

from phenoradar.config import ConfigConditionSet
from phenoradar.metrics import metric_higher_is_better

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class StudyError(ValueError):
    """Raised when a multi-condition study cannot be resumed or summarized."""


_METRIC_ORDER = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "mcc",
    "brier",
    "log_loss",
)
_METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "pr_auc": "Average Precision",
    "balanced_accuracy": "Balanced Accuracy",
    "mcc": "MCC",
    "brier": "Brier Score",
    "log_loss": "Log Loss",
}
_COMPLETED_RUN_STATUSES = {"cv_completed", "full_run_completed"}
_RANKED_METHOD_PATH = "preprocess.ranked_feature_filter.method"
_RANKED_MAX_FEATURES_PATH = "preprocess.ranked_feature_filter.max_features"
_TRAINING_GROUP_COUNT_PATH = "sampling.training_group_count"
_GROUP_SUBSAMPLE_REPEAT_PATH = "sampling.group_subsample_repeat"


@dataclass(frozen=True)
class StudyReportArtifacts:
    """Tabular artifacts generated across completed study conditions."""

    condition_metrics: pl.DataFrame
    pairwise_comparisons: pl.DataFrame
    training_group_sensitivity: pl.DataFrame | None
    figure_paths: tuple[Path, ...]


def _condition_dir_name(index: int, condition_id: str, attempt: int) -> str:
    base = f"condition_{index:03d}_{condition_id}"
    return base if attempt == 1 else f"{base}_attempt_{attempt}"


def new_condition_manifest(
    condition_set: ConfigConditionSet,
    *,
    study_dir: Path,
) -> list[dict[str, Any]]:
    """Create ordered pending manifest rows for a new study."""
    rows: list[dict[str, Any]] = []
    for condition in condition_set.conditions:
        attempt = 1
        run_dir = study_dir / "conditions" / _condition_dir_name(
            condition.index,
            condition.condition_id,
            attempt,
        )
        rows.append(
            {
                "condition_index": condition.index,
                "condition_id": condition.condition_id,
                "condition_label": condition.label,
                "varying_parameters_json": json.dumps(
                    dict(condition.values),
                    ensure_ascii=False,
                    sort_keys=False,
                    separators=(",", ":"),
                ),
                "status": "pending",
                "attempt": attempt,
                "run_dir": str(run_dir),
                "resolved_config_path": str(run_dir / "resolved_config.yml"),
                "started_at": None,
                "ended_at": None,
                "error": None,
            }
        )
    return rows


def write_condition_manifest(study_dir: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Atomically write the ordered condition manifest."""
    output_path = study_dir / "condition_manifest.tsv"
    temporary_path = study_dir / ".condition_manifest.tsv.tmp"
    frame = pl.DataFrame(
        list(rows),
        schema={
            "condition_index": pl.Int64,
            "condition_id": pl.String,
            "condition_label": pl.String,
            "varying_parameters_json": pl.String,
            "status": pl.String,
            "attempt": pl.Int64,
            "run_dir": pl.String,
            "resolved_config_path": pl.String,
            "started_at": pl.String,
            "ended_at": pl.String,
            "error": pl.String,
        },
        orient="row",
    ).sort("condition_index")
    frame.write_csv(temporary_path, separator="\t", null_value="NA")
    temporary_path.replace(output_path)
    return output_path


def write_config_differences(
    study_dir: Path,
    condition_set: ConfigConditionSet,
) -> Path:
    """Write one ordered, wide table containing only condition-varying fields."""
    output_path = study_dir / "config_differences.tsv"
    rows: list[dict[str, Any]] = []
    for condition in condition_set.conditions:
        values = dict(condition.values)
        row: dict[str, Any] = {
            "condition_index": condition.index,
            "condition_id": condition.condition_id,
            "condition_label": condition.label,
        }
        for dimension in condition_set.dimensions:
            value = values[dimension.dotted_path]
            row[dimension.dotted_path] = (
                value
                if value is None or isinstance(value, str | int | float | bool)
                else json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        rows.append(row)
    pl.DataFrame(rows).sort("condition_index").write_csv(
        output_path,
        separator="\t",
        null_value="NA",
    )
    return output_path


def load_condition_manifest(study_dir: Path) -> list[dict[str, Any]]:
    """Load a previously written condition manifest."""
    path = study_dir / "condition_manifest.tsv"
    if not path.exists():
        raise StudyError(f"Study condition manifest was not found: {path}")
    return pl.read_csv(path, separator="\t", null_values="NA").sort(
        "condition_index"
    ).to_dicts()


def validate_resume_manifest(
    rows: Sequence[dict[str, Any]],
    condition_set: ConfigConditionSet,
) -> None:
    """Require the resumed config to generate the same ordered conditions."""
    observed = [str(row["condition_id"]) for row in rows]
    expected = [condition.condition_id for condition in condition_set.conditions]
    if observed != expected:
        raise StudyError(
            "The current config does not generate the same ordered conditions as the study"
        )


def condition_run_is_complete(row: dict[str, Any]) -> bool:
    """Check the material artifacts required to safely skip a condition."""
    run_dir = Path(str(row["run_dir"]))
    metadata_path = run_dir / "run_metadata.json"
    required_paths = (
        run_dir / "resolved_config.yml",
        run_dir / "split" / "tables" / "split_manifest.tsv",
        run_dir / "cv" / "tables" / "metrics_cv.tsv",
        run_dir / "cv" / "tables" / "prediction_cv.tsv",
    )
    if not metadata_path.exists() or not all(path.exists() for path in required_paths):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("status") in _COMPLETED_RUN_STATUSES


def prepare_condition_attempt(row: dict[str, Any], *, study_dir: Path) -> Path:
    """Return a non-overwriting output directory for the next attempt."""
    if str(row["status"]) == "pending" and not Path(str(row["run_dir"])).exists():
        return Path(str(row["run_dir"]))
    attempt = int(row["attempt"]) + 1
    run_dir = study_dir / "conditions" / _condition_dir_name(
        int(row["condition_index"]),
        str(row["condition_id"]),
        attempt,
    )
    while run_dir.exists():
        attempt += 1
        run_dir = study_dir / "conditions" / _condition_dir_name(
            int(row["condition_index"]),
            str(row["condition_id"]),
            attempt,
        )
    row["attempt"] = attempt
    row["run_dir"] = str(run_dir)
    row["resolved_config_path"] = str(run_dir / "resolved_config.yml")
    return run_dir


def _condition_metric_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(str(row["run_dir"]))
    bootstrap_path = run_dir / "cv" / "tables" / "group_bootstrap_metrics.tsv"
    metrics_path = run_dir / "cv" / "tables" / "metrics_cv.tsv"
    if not metrics_path.exists():
        raise StudyError(f"Condition metrics were not found: {metrics_path}")

    point_by_metric: dict[str, float | None] = {}
    metrics = pl.read_csv(metrics_path, separator="\t", null_values="NA")
    micro = metrics.filter(pl.col("aggregate_scope") == "micro")
    for metric_row in micro.select("metric", "metric_value").iter_rows(named=True):
        value = metric_row["metric_value"]
        point_by_metric[str(metric_row["metric"])] = None if value is None else float(value)

    interval_by_metric: dict[str, tuple[float | None, float | None, float | None]] = {}
    if bootstrap_path.exists():
        bootstrap = pl.read_csv(bootstrap_path, separator="\t", null_values="NA")
        for bootstrap_row in bootstrap.select(
            "metric", "point_estimate", "ci_lower", "ci_upper", "confidence_level"
        ).iter_rows(named=True):
            metric = str(bootstrap_row["metric"])
            point = bootstrap_row["point_estimate"]
            if point is not None:
                point_by_metric[metric] = float(point)
            interval_by_metric[metric] = (
                None if bootstrap_row["ci_lower"] is None else float(bootstrap_row["ci_lower"]),
                None if bootstrap_row["ci_upper"] is None else float(bootstrap_row["ci_upper"]),
                (
                    None
                    if bootstrap_row["confidence_level"] is None
                    else float(bootstrap_row["confidence_level"])
                ),
            )

    output: list[dict[str, Any]] = []
    for metric in _METRIC_ORDER:
        lower, upper, confidence_level = interval_by_metric.get(metric, (None, None, None))
        output.append(
            {
                "condition_index": int(row["condition_index"]),
                "condition_id": str(row["condition_id"]),
                "condition_label": str(row["condition_label"]),
                "metric": metric,
                "point_estimate": point_by_metric.get(metric),
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence_level,
            }
        )
    return output


def _replicates_by_condition(
    rows: Sequence[dict[str, Any]],
) -> dict[str, pl.DataFrame]:
    replicates: dict[str, pl.DataFrame] = {}
    for row in rows:
        path = Path(str(row["run_dir"])) / "cv" / "tables" / "group_bootstrap_replicates.tsv"
        if not path.exists():
            return {}
        replicates[str(row["condition_id"])] = pl.read_csv(
            path,
            separator="\t",
            null_values="NA",
        ).select("resample_id", "metric", "metric_value")
    return replicates


def _finite_quantile(values: np.ndarray, probability: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, probability))


def _pairwise_rows(
    manifest_rows: Sequence[dict[str, Any]],
    condition_metrics: pl.DataFrame,
) -> list[dict[str, Any]]:
    replicates = _replicates_by_condition(manifest_rows)
    points = {
        (str(row["condition_id"]), str(row["metric"])): row["point_estimate"]
        for row in condition_metrics.iter_rows(named=True)
    }
    output: list[dict[str, Any]] = []
    for row_a, row_b in combinations(manifest_rows, 2):
        condition_a = str(row_a["condition_id"])
        condition_b = str(row_b["condition_id"])
        for metric in _METRIC_ORDER:
            point_a = points.get((condition_a, metric))
            point_b = points.get((condition_b, metric))
            raw_delta = None
            improvement = None
            if point_a is not None and point_b is not None:
                raw_delta = float(point_a) - float(point_b)
                improvement = raw_delta if metric_higher_is_better(metric) else -raw_delta

            ci_lower: float | None = None
            ci_upper: float | None = None
            probability_a_better: float | None = None
            n_valid_resamples = 0
            if replicates:
                joined = replicates[condition_a].filter(pl.col("metric") == metric).join(
                    replicates[condition_b].filter(pl.col("metric") == metric),
                    on=["resample_id", "metric"],
                    how="inner",
                    suffix="_b",
                )
                deltas = (
                    joined.get_column("metric_value").to_numpy()
                    - joined.get_column("metric_value_b").to_numpy()
                ).astype(float)
                if not metric_higher_is_better(metric):
                    deltas = -deltas
                finite = deltas[np.isfinite(deltas)]
                n_valid_resamples = int(finite.size)
                if finite.size:
                    confidence_values = condition_metrics.filter(
                        (pl.col("condition_id") == condition_a)
                        & (pl.col("metric") == metric)
                    ).get_column("confidence_level").drop_nulls()
                    confidence_level = (
                        float(confidence_values.item(0)) if confidence_values.len() else 0.95
                    )
                    alpha = (1.0 - confidence_level) / 2.0
                    ci_lower = _finite_quantile(finite, alpha)
                    ci_upper = _finite_quantile(finite, 1.0 - alpha)
                    probability_a_better = float(np.mean(finite > 0.0))

            output.append(
                {
                    "condition_a_index": int(row_a["condition_index"]),
                    "condition_a_id": condition_a,
                    "condition_a_label": str(row_a["condition_label"]),
                    "condition_b_index": int(row_b["condition_index"]),
                    "condition_b_id": condition_b,
                    "condition_b_label": str(row_b["condition_label"]),
                    "metric": metric,
                    "raw_delta_a_minus_b": raw_delta,
                    "improvement_a_over_b": improvement,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "probability_a_better": probability_a_better,
                    "n_valid_resamples": n_valid_resamples,
                }
            )
    return output


def _save_figure_formats(fig: Any, base_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for extension in ("svg", "pdf", "png"):
        path = base_path.with_suffix(f".{extension}")
        metadata = {"Date": None} if extension == "svg" else None
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return tuple(paths)


def _condition_figure_label(raw_label: str) -> str:
    assignments = raw_label.split("; ")
    parsed: list[tuple[str, str]] = []
    for assignment in assignments:
        path, separator, raw_value = assignment.partition("=")
        if not separator:
            return raw_label
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            display_value = raw_value
        else:
            display_value = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
        parsed.append((path.rsplit(".", maxsplit=1)[-1], str(display_value)))
    if len(parsed) == 1:
        return parsed[0][1]
    return "\n".join(f"{name}={value}" for name, value in parsed)


def _condition_metric_figure(
    condition_metrics: pl.DataFrame,
    *,
    output_dir: Path,
) -> tuple[Path, ...]:
    condition_rows = condition_metrics.select(
        "condition_index", "condition_id", "condition_label"
    ).unique(maintain_order=True).sort("condition_index")
    indices = condition_rows.get_column("condition_index").to_list()
    labels = [
        _condition_figure_label(str(value))
        for value in condition_rows.get_column("condition_label")
    ]
    figure_height = max(5.5, 2.4 + 0.28 * len(indices) * 2)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, figure_height), squeeze=False)
    colors = plt.get_cmap("tab20")(np.linspace(0.05, 0.95, max(1, len(indices))))
    y = np.arange(len(indices), dtype=float)
    for axis, metric in zip(axes.flat, _METRIC_ORDER, strict=True):
        metric_rows = condition_metrics.filter(pl.col("metric") == metric).sort(
            "condition_index"
        )
        points = np.asarray(metric_rows.get_column("point_estimate").to_list(), dtype=float)
        lower = np.asarray(metric_rows.get_column("ci_lower").to_list(), dtype=float)
        upper = np.asarray(metric_rows.get_column("ci_upper").to_list(), dtype=float)
        for position, point in enumerate(points):
            if not np.isfinite(point):
                continue
            if np.isfinite(lower[position]) and np.isfinite(upper[position]):
                axis.errorbar(
                    point,
                    y[position],
                    xerr=np.asarray(
                        [[point - lower[position]], [upper[position] - point]], dtype=float
                    ),
                    fmt="o",
                    color=colors[position],
                    capsize=2,
                    markersize=4,
                    linewidth=1,
                )
            else:
                axis.scatter(point, y[position], color=colors[position], s=18)
        axis.set_title(_METRIC_LABELS[metric])
        axis.set_yticks(y, labels=labels)
        axis.set_ylabel("Condition")
        axis.grid(axis="x", alpha=0.3)
        axis.invert_yaxis()
    fig.suptitle("Multi-condition OOF performance")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure_formats(fig, output_dir / "condition_metrics")


def _ranked_sensitivity_metadata(
    manifest_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for row in manifest_rows:
        raw_values = row.get("varying_parameters_json")
        if not isinstance(raw_values, str):
            return []
        try:
            values = json.loads(raw_values)
        except json.JSONDecodeError:
            return []
        if not isinstance(values, dict):
            return []
        method = values.get(_RANKED_METHOD_PATH)
        max_features = values.get(_RANKED_MAX_FEATURES_PATH)
        if not isinstance(method, str) or not isinstance(max_features, int):
            return []
        metadata.append(
            {
                "condition_id": str(row["condition_id"]),
                "method": method,
                "max_features": max_features,
            }
        )
    methods = list(dict.fromkeys(str(row["method"]) for row in metadata))
    feature_counts = {int(row["max_features"]) for row in metadata}
    if len(methods) != 2 or len(feature_counts) < 2:
        return []
    expected = {(method, count) for method in methods for count in feature_counts}
    observed = {
        (str(row["method"]), int(row["max_features"])) for row in metadata
    }
    return metadata if observed == expected else []


def _ranked_feature_sensitivity_figures(
    condition_metrics: pl.DataFrame,
    pairwise: pl.DataFrame,
    manifest_rows: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
) -> tuple[Path, ...]:
    metadata_rows = _ranked_sensitivity_metadata(manifest_rows)
    if not metadata_rows:
        return ()
    metadata = pl.DataFrame(metadata_rows)
    methods = metadata.get_column("method").unique(maintain_order=True).to_list()
    feature_counts = sorted(metadata.get_column("max_features").unique().to_list())
    joined = condition_metrics.join(metadata, on="condition_id", how="inner")
    colors = plt.get_cmap("tab10")(np.linspace(0.05, 0.55, len(methods)))

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), squeeze=False)
    for axis, metric in zip(axes.flat, _METRIC_ORDER, strict=True):
        for color, method in zip(colors, methods, strict=True):
            rows = joined.filter(
                (pl.col("metric") == metric) & (pl.col("method") == method)
            ).sort("max_features")
            x = np.asarray(rows.get_column("max_features").to_list(), dtype=float)
            points = np.asarray(rows.get_column("point_estimate").to_list(), dtype=float)
            lower = np.asarray(rows.get_column("ci_lower").to_list(), dtype=float)
            upper = np.asarray(rows.get_column("ci_upper").to_list(), dtype=float)
            axis.plot(x, points, marker="o", markersize=4, linewidth=1.2, color=color)
            finite_interval = np.isfinite(lower) & np.isfinite(upper)
            if np.any(finite_interval):
                axis.fill_between(
                    x[finite_interval],
                    lower[finite_interval],
                    upper[finite_interval],
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        axis.set_title(_METRIC_LABELS[metric])
        axis.set_xscale("log")
        axis.set_xticks(feature_counts, labels=[str(value) for value in feature_counts])
        axis.set_xlabel("Maximum selected features")
        axis.grid(alpha=0.25)
    handles = [
        Line2D([], [], color=color, marker="o", linewidth=1.2, label=str(method))
        for color, method in zip(colors, methods, strict=True)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), frameon=False)
    fig.suptitle("Ranked-feature sensitivity")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths = list(
        _save_figure_formats(fig, output_dir / "ranked_feature_sensitivity")
    )

    method_a, method_b = (str(method) for method in methods)
    condition_ids = {
        (str(row["method"]), int(row["max_features"])): str(row["condition_id"])
        for row in metadata_rows
    }
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), squeeze=False)
    for axis, metric in zip(axes.flat, _METRIC_ORDER, strict=True):
        x_values: list[int] = []
        improvements: list[float] = []
        lower_values: list[float] = []
        upper_values: list[float] = []
        for max_features in feature_counts:
            condition_a = condition_ids[(method_a, int(max_features))]
            condition_b = condition_ids[(method_b, int(max_features))]
            matches = pairwise.filter(
                (pl.col("metric") == metric)
                & (
                    (
                        (pl.col("condition_a_id") == condition_a)
                        & (pl.col("condition_b_id") == condition_b)
                    )
                    | (
                        (pl.col("condition_a_id") == condition_b)
                        & (pl.col("condition_b_id") == condition_a)
                    )
                )
            )
            if matches.height != 1:
                continue
            row = matches.row(0, named=True)
            value = row["improvement_a_over_b"]
            if value is None:
                continue
            stored_a = str(row["condition_a_id"])
            orientation = -1.0 if stored_a == condition_a else 1.0
            lower = row["ci_lower"]
            upper = row["ci_upper"]
            x_values.append(int(max_features))
            improvements.append(orientation * float(value))
            if lower is None or upper is None:
                lower_values.append(np.nan)
                upper_values.append(np.nan)
            elif orientation > 0:
                lower_values.append(float(lower))
                upper_values.append(float(upper))
            else:
                lower_values.append(-float(upper))
                upper_values.append(-float(lower))
        x = np.asarray(x_values, dtype=float)
        points = np.asarray(improvements, dtype=float)
        lower = np.asarray(lower_values, dtype=float)
        upper = np.asarray(upper_values, dtype=float)
        yerr = np.vstack(
            (
                np.maximum(0.0, points - lower),
                np.maximum(0.0, upper - points),
            )
        )
        yerr[:, ~(np.isfinite(lower) & np.isfinite(upper))] = 0.0
        axis.errorbar(
            x,
            points,
            yerr=yerr,
            color="#4C72B0",
            marker="o",
            capsize=2,
            markersize=4,
            linewidth=1.2,
        )
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(_METRIC_LABELS[metric])
        axis.set_xscale("log")
        axis.set_xticks(feature_counts, labels=[str(value) for value in feature_counts])
        axis.set_xlabel("Maximum selected features")
        axis.set_ylabel("Improvement")
        axis.grid(alpha=0.25)
    fig.suptitle(f"{method_b} improvement over {method_a}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths.extend(
        _save_figure_formats(
            fig,
            output_dir / "ranked_feature_method_difference",
        )
    )
    return tuple(paths)


def _training_group_sensitivity_metadata(
    manifest_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load condition axes and effective fold-local group counts for a pure group sweep."""
    allowed_paths = {_TRAINING_GROUP_COUNT_PATH, _GROUP_SUBSAMPLE_REPEAT_PATH}
    metadata: list[dict[str, Any]] = []
    for row in manifest_rows:
        raw_values = row.get("varying_parameters_json")
        if not isinstance(raw_values, str):
            return []
        try:
            values = json.loads(raw_values)
        except json.JSONDecodeError:
            return []
        if (
            not isinstance(values, dict)
            or _TRAINING_GROUP_COUNT_PATH not in values
            or not set(values).issubset(allowed_paths)
        ):
            return []
        requested = values[_TRAINING_GROUP_COUNT_PATH]
        configured_repeat = values.get(_GROUP_SUBSAMPLE_REPEAT_PATH)
        if (requested is not None and not isinstance(requested, int)) or (
            configured_repeat is not None and not isinstance(configured_repeat, int)
        ):
            return []

        audit_path = (
            Path(str(row["run_dir"]))
            / "model"
            / "tables"
            / "training_group_subsets.tsv"
        )
        if not audit_path.exists():
            return []
        audit = pl.read_csv(audit_path, separator="\t", null_values="NA").filter(
            pl.col("scope") == "outer_fold"
        )
        if audit.is_empty():
            return []
        audit_repeats = audit.get_column("group_subsample_repeat").unique().to_list()
        if len(audit_repeats) != 1:
            return []
        repeat = int(audit_repeats[0])
        if configured_repeat is not None and repeat != configured_repeat:
            return []
        per_fold = audit.select(
            "fold_id", "n_training_groups_selected"
        ).unique()
        effective_counts = np.asarray(
            per_fold.get_column("n_training_groups_selected").to_list(),
            dtype=float,
        )
        if requested is not None and np.any(effective_counts != requested):
            return []
        metadata.append(
            {
                "condition_id": str(row["condition_id"]),
                "condition_index": int(row["condition_index"]),
                "training_group_count": requested,
                "full_training_set": requested is None,
                "group_subsample_repeat": int(repeat),
                "effective_training_groups_mean": float(np.mean(effective_counts)),
                "effective_training_groups_min": int(np.min(effective_counts)),
                "effective_training_groups_max": int(np.max(effective_counts)),
            }
        )

    group_settings = {
        (bool(row["full_training_set"]), row["training_group_count"])
        for row in metadata
    }
    return metadata if len(group_settings) >= 2 else []


def _build_training_group_sensitivity(
    condition_metrics: pl.DataFrame,
    manifest_rows: Sequence[dict[str, Any]],
) -> pl.DataFrame | None:
    metadata_rows = _training_group_sensitivity_metadata(manifest_rows)
    if not metadata_rows:
        return None

    grouped_metadata: dict[tuple[bool, int | None], list[dict[str, Any]]] = {}
    for row in metadata_rows:
        key = (bool(row["full_training_set"]), row["training_group_count"])
        grouped_metadata.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (full_training_set, requested), group_rows in grouped_metadata.items():
        condition_ids = [str(row["condition_id"]) for row in group_rows]
        effective_means = np.asarray(
            [float(row["effective_training_groups_mean"]) for row in group_rows],
            dtype=float,
        )
        effective_min = min(int(row["effective_training_groups_min"]) for row in group_rows)
        effective_max = max(int(row["effective_training_groups_max"]) for row in group_rows)
        repeats = {int(row["group_subsample_repeat"]) for row in group_rows}
        for metric in _METRIC_ORDER:
            values = np.asarray(
                condition_metrics.filter(
                    pl.col("condition_id").is_in(condition_ids)
                    & (pl.col("metric") == metric)
                )
                .sort("condition_index")
                .get_column("point_estimate")
                .to_list(),
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            output.append(
                {
                    "training_group_count": requested,
                    "training_group_count_label": (
                        "All available" if full_training_set else str(requested)
                    ),
                    "full_training_set": full_training_set,
                    "effective_training_groups_mean": float(np.mean(effective_means)),
                    "effective_training_groups_min": effective_min,
                    "effective_training_groups_max": effective_max,
                    "n_subset_repeats": len(repeats),
                    "n_conditions": len(group_rows),
                    "metric": metric,
                    "n_valid_conditions": int(finite.size),
                    "point_estimate_mean": (
                        None if finite.size == 0 else float(np.mean(finite))
                    ),
                    "point_estimate_std": (
                        None if finite.size < 2 else float(np.std(finite, ddof=1))
                    ),
                    "point_estimate_min": (
                        None if finite.size == 0 else float(np.min(finite))
                    ),
                    "point_estimate_q1": (
                        None if finite.size == 0 else float(np.quantile(finite, 0.25))
                    ),
                    "point_estimate_median": (
                        None if finite.size == 0 else float(np.median(finite))
                    ),
                    "point_estimate_q3": (
                        None if finite.size == 0 else float(np.quantile(finite, 0.75))
                    ),
                    "point_estimate_max": (
                        None if finite.size == 0 else float(np.max(finite))
                    ),
                }
            )
    return pl.DataFrame(output).sort(
        ["effective_training_groups_mean", "full_training_set", "metric"]
    )


def _training_group_sensitivity_figure(
    sensitivity: pl.DataFrame,
    *,
    output_dir: Path,
) -> tuple[Path, ...]:
    settings = (
        sensitivity.select(
            "effective_training_groups_mean", "training_group_count_label"
        )
        .unique()
        .sort("effective_training_groups_mean")
    )
    x_ticks = np.asarray(
        settings.get_column("effective_training_groups_mean").to_list(), dtype=float
    )
    x_labels = [str(value) for value in settings.get_column("training_group_count_label")]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), squeeze=False)
    for axis, metric in zip(axes.flat, _METRIC_ORDER, strict=True):
        rows = sensitivity.filter(pl.col("metric") == metric).sort(
            "effective_training_groups_mean"
        )
        x = np.asarray(rows.get_column("effective_training_groups_mean").to_list(), dtype=float)
        points = np.asarray(rows.get_column("point_estimate_mean").to_list(), dtype=float)
        lower = np.asarray(rows.get_column("point_estimate_q1").to_list(), dtype=float)
        upper = np.asarray(rows.get_column("point_estimate_q3").to_list(), dtype=float)
        minimum = np.asarray(rows.get_column("point_estimate_min").to_list(), dtype=float)
        maximum = np.asarray(rows.get_column("point_estimate_max").to_list(), dtype=float)
        finite = np.isfinite(points)
        if np.any(finite):
            range_lower = np.maximum(0.0, points - minimum)
            range_upper = np.maximum(0.0, maximum - points)
            range_error = np.vstack((range_lower, range_upper))
            range_error[:, ~(np.isfinite(minimum) & np.isfinite(maximum))] = 0.0
            axis.errorbar(
                x[finite],
                points[finite],
                yerr=range_error[:, finite],
                marker="o",
                markersize=4,
                linewidth=1.2,
                capsize=2,
                color="#4C72B0",
            )
            interval = finite & np.isfinite(lower) & np.isfinite(upper)
            if np.any(interval):
                axis.fill_between(
                    x[interval],
                    lower[interval],
                    upper[interval],
                    color="#4C72B0",
                    alpha=0.18,
                    linewidth=0,
                )
        axis.set_xticks(x_ticks, labels=x_labels)
        axis.set_xlabel("Number of training groups")
        axis.set_ylabel(_METRIC_LABELS[metric])
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return _save_figure_formats(fig, output_dir / "training_group_sensitivity")


def generate_study_report(
    study_dir: Path,
    manifest_rows: Sequence[dict[str, Any]],
) -> StudyReportArtifacts:
    """Aggregate all completed conditions and write symmetric study comparisons."""
    ordered_rows = sorted(manifest_rows, key=lambda row: int(row["condition_index"]))
    if len(ordered_rows) < 2:
        raise StudyError("A study report requires at least two completed conditions")
    incomplete = [
        str(row["condition_id"])
        for row in ordered_rows
        if not condition_run_is_complete(row)
    ]
    if incomplete:
        raise StudyError("Study contains incomplete conditions: " + ", ".join(incomplete))

    metadata_rows: list[dict[str, Any]] = []
    for row in ordered_rows:
        metadata_path = Path(str(row["run_dir"])) / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_rows.append(metadata)
    split_fingerprints = {metadata.get("split_fingerprint") for metadata in metadata_rows}
    dataset_fingerprints = {metadata.get("dataset_fingerprint") for metadata in metadata_rows}
    if len(split_fingerprints) != 1 or len(dataset_fingerprints) != 1:
        raise StudyError("Study conditions do not share one dataset and split fingerprint")

    metric_rows = [metric for row in ordered_rows for metric in _condition_metric_rows(row)]
    condition_metrics = pl.DataFrame(metric_rows).sort(["condition_index", "metric"])
    pairwise_rows = _pairwise_rows(ordered_rows, condition_metrics)
    pairwise = pl.DataFrame(pairwise_rows).sort(
        ["condition_a_index", "condition_b_index", "metric"]
    )

    tables_dir = study_dir / "tables"
    figures_dir = study_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    condition_metrics.write_csv(
        tables_dir / "condition_metrics.tsv",
        separator="\t",
        float_precision=10,
        null_value="NA",
    )
    pairwise.write_csv(
        tables_dir / "pairwise_comparisons.tsv",
        separator="\t",
        float_precision=10,
        null_value="NA",
    )
    training_group_sensitivity = _build_training_group_sensitivity(
        condition_metrics,
        ordered_rows,
    )
    training_group_figure_paths: tuple[Path, ...] = ()
    if training_group_sensitivity is not None:
        training_group_sensitivity.write_csv(
            tables_dir / "training_group_sensitivity.tsv",
            separator="\t",
            float_precision=10,
            null_value="NA",
        )
        training_group_figure_paths = _training_group_sensitivity_figure(
            training_group_sensitivity,
            output_dir=figures_dir,
        )
    figure_paths = (
        *_condition_metric_figure(condition_metrics, output_dir=figures_dir),
        *_ranked_feature_sensitivity_figures(
            condition_metrics,
            pairwise,
            ordered_rows,
            output_dir=figures_dir,
        ),
        *training_group_figure_paths,
    )
    return StudyReportArtifacts(
        condition_metrics=condition_metrics,
        pairwise_comparisons=pairwise,
        training_group_sensitivity=training_group_sensitivity,
        figure_paths=figure_paths,
    )
