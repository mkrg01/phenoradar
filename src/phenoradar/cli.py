"""CLI entry points for PhenoRadar."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import polars as pl
import typer

from phenoradar import __version__
from phenoradar.bundle import (
    BundleError,
    export_model_bundle,
    load_model_bundle,
    predict_with_bundle,
)
from phenoradar.config import (
    ConfigError,
    ExecutionStage,
    load_and_resolve_config,
    write_resolved_config,
)
from phenoradar.cv import CVError, run_final_refit, run_outer_cv
from phenoradar.figures import FigureError, write_predict_figures, write_run_figures
from phenoradar.provenance import (
    ProvenanceError,
    bundle_payload_sha256,
    collect_input_files,
    git_snapshot,
    runtime_environment_snapshot,
)
from phenoradar.reporting import (
    AggregateScope,
    IncludeStage,
    OutputFormat,
    PrimaryMetric,
    ReportError,
    ReportOptions,
    generate_report,
)
from phenoradar.split import SplitError, build_split_artifacts
from phenoradar.testdata import (
    DEFAULT_C4_TINY_BASE_URL,
    TestDataError,
    fetch_c4_tiny_test_data,
)

app = typer.Typer(
    help="PhenoRadar: orthogroup TPM-based phenotype prediction CLI",
    context_settings={"help_option_names": ["-h", "--help"]},
)


LogVerbosity = Literal["quiet", "normal", "verbose"]

ConfigPathsArg = Annotated[
    list[Path],
    typer.Option(
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="YAML config file.",
    ),
]
OptionalConfigPathsArg = Annotated[
    list[Path] | None,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="YAML config file. If omitted, built-in defaults are used.",
    ),
]
ExecutionStageArg = Annotated[
    ExecutionStage | None,
    typer.Option(
        "--execution-stage",
        help="Execution stage override for run command.",
    ),
]
ModelBundleArg = Annotated[
    Path,
    typer.Option(
        "--model-bundle",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to exported model bundle directory.",
    ),
]
ReportRunDirArg = Annotated[
    list[Path] | None,
    typer.Option(
        "--run-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Explicit run directory. Can be specified multiple times.",
    ),
]
ReportRunsRootArg = Annotated[
    Path | None,
    typer.Option(
        "--runs-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Root directory to scan for run directories.",
    ),
]
VerboseArg = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Emit detailed stage-level progress logs.",
    ),
]
QuietArg = Annotated[
    bool,
    typer.Option(
        "--quiet",
        "-q",
        help="Suppress progress logs and print only final summaries/warnings.",
    ),
]


def _version_callback(value: bool) -> None:
    if not value:
        return
    typer.echo(f"phenoradar {__version__}")
    raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """PhenoRadar CLI root options."""


def _build_run_dir(prefix: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    shortid = uuid4().hex[:8]
    run_dir = Path("runs") / f"{timestamp}_{prefix}_{shortid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_report_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    shortid = uuid4().hex[:8]
    return Path("reports") / f"{timestamp}_report_{shortid}"


def _write_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _utc_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_log_verbosity(*, verbose: bool, quiet: bool) -> LogVerbosity:
    if verbose and quiet:
        raise typer.BadParameter("`--verbose` and `--quiet` cannot be used together.")
    if verbose:
        return "verbose"
    if quiet:
        return "quiet"
    return "normal"


def _should_emit_progress(log_verbosity: LogVerbosity, *, detail: bool) -> bool:
    if log_verbosity == "quiet":
        return False
    return not (log_verbosity == "normal" and detail)


def _progress_log(
    command: str,
    message: str,
    *,
    start_time: datetime | None = None,
    log_verbosity: LogVerbosity = "normal",
    detail: bool = False,
) -> None:
    if not _should_emit_progress(log_verbosity, detail=detail):
        return
    now = datetime.now(UTC)
    log_line = f"[{_utc_iso(now)}] [{command}] {message}"
    if start_time is not None:
        elapsed_sec = (now - start_time).total_seconds()
        log_line += f" (elapsed={elapsed_sec:.1f}s)"
    typer.echo(log_line)


def _warning_log(command: str, message: str, *, start_time: datetime | None = None) -> None:
    now = datetime.now(UTC)
    log_line = f"[{_utc_iso(now)}] [{command}] WARNING: {message}"
    if start_time is not None:
        elapsed_sec = (now - start_time).total_seconds()
        log_line += f" (elapsed={elapsed_sec:.1f}s)"
    typer.echo(log_line)


def _emit_warning_summary(
    command: str,
    warnings: list[str],
    *,
    metadata_path: Path,
    start_time: datetime | None = None,
    max_items: int = 5,
) -> None:
    count = len(warnings)
    if count == 0:
        return
    _warning_log(
        command,
        f"Recorded {count} warning(s). See {metadata_path} for the full list.",
        start_time=start_time,
    )
    for index, warning in enumerate(warnings[:max_items], start=1):
        _warning_log(command, f"{index}/{count}: {warning}", start_time=start_time)
    if count > max_items:
        _warning_log(
            command,
            f"... and {count - max_items} more warning(s).",
            start_time=start_time,
        )


def _extract_macro_metric(metrics_cv: pl.DataFrame, metric: str) -> float | None:
    values = (
        metrics_cv.filter(
            (pl.col("aggregate_scope") == "macro")
            & (pl.col("fold_id") == "NA")
            & (pl.col("metric") == metric)
        )
        .select("metric_value")
        .to_series()
        .to_list()
    )
    if len(values) != 1:
        return None
    value = values[0]
    if not isinstance(value, (int, float)):
        return None
    metric_value = float(value)
    if metric_value != metric_value:
        return None
    return metric_value


def _emit_run_metric_summary(
    metrics_cv: pl.DataFrame,
    *,
    start_time: datetime,
    log_verbosity: LogVerbosity,
) -> None:
    metric_names = ("mcc", "balanced_accuracy", "roc_auc", "pr_auc", "brier")
    parts: list[str] = []
    for metric_name in metric_names:
        metric_value = _extract_macro_metric(metrics_cv, metric_name)
        if metric_value is not None:
            parts.append(f"{metric_name}={metric_value:.4f}")
    if parts:
        _progress_log(
            "run",
            f"CV metric summary (macro): {', '.join(parts)}.",
            start_time=start_time,
            log_verbosity=log_verbosity,
        )


def _emit_predict_summary(
    pred_predict: pl.DataFrame,
    *,
    start_time: datetime,
    log_verbosity: LogVerbosity,
) -> None:
    n_species = pred_predict.height
    n_positive = None
    n_positive_cv = None
    if "pred_label_fixed_threshold" in pred_predict.columns:
        n_positive = int(
            pred_predict.filter(pl.col("pred_label_fixed_threshold") == 1).height
        )
    if "pred_label_cv_derived_threshold" in pred_predict.columns:
        n_positive_cv = int(
            pred_predict.filter(pl.col("pred_label_cv_derived_threshold") == 1).height
        )
    message = f"Prediction summary (n_species={n_species}"
    if n_positive is not None:
        message += f", n_pred_positive={n_positive}"
    if n_positive_cv is not None:
        message += f", n_pred_positive_cv={n_positive_cv}"
    message += ")."
    _progress_log("predict", message, start_time=start_time, log_verbosity=log_verbosity)


def _emit_report_summary(
    *,
    output_dir: Path,
    start_time: datetime,
    log_verbosity: LogVerbosity,
) -> None:
    manifest_path = output_dir / "report_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except json.JSONDecodeError:
        return
    if not isinstance(manifest, dict):
        return
    included = manifest.get("included_runs")
    ranked = manifest.get("ranked_run_count")
    included_count = len(included) if isinstance(included, list) else None
    ranked_count = int(ranked) if isinstance(ranked, int) else None
    parts: list[str] = []
    if included_count is not None:
        parts.append(f"included_runs={included_count}")
    if ranked_count is not None:
        parts.append(f"ranked_runs={ranked_count}")
    if parts:
        _progress_log(
            "report",
            f"Report summary ({', '.join(parts)}).",
            start_time=start_time,
            log_verbosity=log_verbosity,
        )


def _emit_report_warning_summary(
    *,
    output_dir: Path,
    start_time: datetime,
    max_types: int = 8,
) -> None:
    warning_path = output_dir / "report_warnings.tsv"
    if not warning_path.exists():
        return
    try:
        warning_rows = pl.read_csv(warning_path, separator="\t")
    except Exception as exc:  # noqa: BLE001
        _warning_log(
            "report",
            f"Failed to read report warnings from {warning_path}: {exc}",
            start_time=start_time,
        )
        return
    count = warning_rows.height
    if count == 0:
        return
    _warning_log(
        "report",
        f"Recorded {count} warning row(s). See {warning_path} for the full list.",
        start_time=start_time,
    )
    if "warning_type" not in warning_rows.columns:
        _warning_log(
            "report",
            "report_warnings.tsv does not contain warning_type; skipped type summary.",
            start_time=start_time,
        )
        return

    if "run_id" in warning_rows.columns:
        type_counts = warning_rows.group_by("warning_type").agg(
            pl.len().alias("row_count"),
            pl.col("run_id").n_unique().alias("run_count"),
        )
    else:
        type_counts = warning_rows.group_by("warning_type").agg(pl.len().alias("row_count"))

    type_counts = type_counts.sort(["row_count", "warning_type"], descending=[True, False])
    type_total = type_counts.height
    _warning_log(
        "report",
        f"Warning type summary ({type_total} type(s)).",
        start_time=start_time,
    )
    for index, row in enumerate(type_counts.iter_rows(named=True), start=1):
        if index > max_types:
            break
        warning_type = str(row.get("warning_type", "warning"))
        row_count = int(row.get("row_count", 0))
        if "run_count" in row:
            run_count = int(row["run_count"])
            _warning_log(
                "report",
                f"{index}/{type_total}: type={warning_type}, rows={row_count}, runs={run_count}",
                start_time=start_time,
            )
        else:
            _warning_log(
                "report",
                f"{index}/{type_total}: type={warning_type}, rows={row_count}",
                start_time=start_time,
            )
    if type_total > max_types:
        _warning_log(
            "report",
            f"... and {type_total - max_types} more warning type(s).",
            start_time=start_time,
        )


def _normalize_config_paths(config: Sequence[Path] | None) -> list[Path]:
    config_paths = [] if config is None else list(config)
    if len(config_paths) > 1:
        raise typer.BadParameter("`--config` / `-c` can be specified at most once.")
    return config_paths


def _threshold_lookup(thresholds: pl.DataFrame) -> dict[str, float]:
    required = {"threshold_name", "threshold_value"}
    if not required.issubset(thresholds.columns):
        raise typer.BadParameter("thresholds table does not contain required columns")

    values: dict[str, float] = {}
    for row in thresholds.iter_rows(named=True):
        name = str(row["threshold_name"])
        raw_value = row["threshold_value"]
        if raw_value is None:
            continue
        values[name] = float(raw_value)
    for required_name in ("fixed_probability_threshold", "cv_derived_threshold"):
        if required_name not in values:
            raise typer.BadParameter(f"{required_name} was not found in thresholds table")
    return values


def _append_classification_summary_rows(
    rows: list[dict[str, str | float | int | None]],
    *,
    pool: str,
    fold_id: str,
    label: pl.Series,
    prob: pl.Series,
    thresholds: dict[str, float],
) -> None:
    if label.len() != prob.len():
        raise typer.BadParameter("Prediction summary received mismatched label/prob lengths")
    if label.len() == 0:
        return

    labels = label.cast(pl.Int8, strict=True)
    probs = prob.cast(pl.Float64, strict=True)
    n_total = int(labels.len())
    n_positive = int(labels.sum())

    for threshold_name, threshold_value in sorted(thresholds.items()):
        pred = (probs >= threshold_value).cast(pl.Int8)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        n_pred_positive = int(tp + fp)
        n_correct = int(tp + tn)

        accuracy = float(n_correct / n_total)
        precision: float | None = None
        if n_pred_positive > 0:
            precision = float(tp / n_pred_positive)
        recall: float | None = None
        if n_positive > 0:
            recall = float(tp / n_positive)
        f1: float | None = None
        if precision is not None and recall is not None and (precision + recall) > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        mcc_denom = sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = float((tp * tn - fp * fn) / mcc_denom) if mcc_denom > 0.0 else 0.0
        rows.append(
            {
                "pool": pool,
                "fold_id": fold_id,
                "threshold_name": threshold_name,
                "threshold_value": float(threshold_value),
                "n_total": n_total,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc,
            }
        )


def _classification_summary(
    *,
    oof_predictions: pl.DataFrame,
    thresholds: pl.DataFrame,
    pred_external_test: pl.DataFrame | None,
) -> pl.DataFrame:
    required_oof = {"fold_id", "label", "prob"}
    if not required_oof.issubset(oof_predictions.columns):
        raise typer.BadParameter(
            "prediction_cv.tsv schema is invalid for classification_summary.tsv"
        )

    threshold_values = _threshold_lookup(thresholds)
    rows: list[dict[str, str | float | int | None]] = []

    # Validation pooled summary from out-of-fold predictions.
    _append_classification_summary_rows(
        rows,
        pool="validation_oof",
        fold_id="NA",
        label=oof_predictions.get_column("label"),
        prob=oof_predictions.get_column("prob"),
        thresholds=threshold_values,
    )

    # Validation per-fold summary.
    fold_values = oof_predictions.select("fold_id").unique().to_series().to_list()

    def _fold_sort_key(value: str) -> tuple[int, str]:
        text = str(value)
        try:
            return (0, f"{int(text):010d}")
        except ValueError:
            return (1, text)

    for fold_id in sorted((str(v) for v in fold_values), key=_fold_sort_key):
        fold_df = oof_predictions.filter(pl.col("fold_id") == fold_id)
        _append_classification_summary_rows(
            rows,
            pool="validation_oof",
            fold_id=fold_id,
            label=fold_df.get_column("label"),
            prob=fold_df.get_column("prob"),
            thresholds=threshold_values,
        )

    # External test pooled summary (full_run only).
    if pred_external_test is not None:
        required_external_pred = {"prob", "true_label"}
        if not required_external_pred.issubset(pred_external_test.columns):
            raise typer.BadParameter(
                "prediction_external_test.tsv schema is invalid for "
                "classification_summary.tsv"
            )
        missing_label_count = pred_external_test.filter(
            pl.col("true_label").is_null()
        ).height
        if missing_label_count > 0:
            raise typer.BadParameter(
                "prediction_external_test.tsv contains species with missing external_test labels"
            )
        _append_classification_summary_rows(
            rows,
            pool="external_test",
            fold_id="NA",
            label=pred_external_test.get_column("true_label"),
            prob=pred_external_test.get_column("prob"),
            thresholds=threshold_values,
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "pool": pl.String,
                "fold_id": pl.String,
                "threshold_name": pl.String,
                "threshold_value": pl.Float64,
                "n_total": pl.Int64,
                "tp": pl.Int64,
                "fp": pl.Int64,
                "tn": pl.Int64,
                "fn": pl.Int64,
                "accuracy": pl.Float64,
                "precision": pl.Float64,
                "recall": pl.Float64,
                "f1": pl.Float64,
                "mcc": pl.Float64,
            }
        )

    return pl.DataFrame(rows).sort(["pool", "fold_id", "threshold_name"])


@app.command()
def run(
    config: ConfigPathsArg,
    execution_stage: ExecutionStageArg = None,
    verbose: VerboseArg = False,
    quiet: QuietArg = False,
) -> None:
    """Run training/evaluation pipeline."""
    start_time = datetime.now(UTC)
    log_verbosity = _resolve_log_verbosity(verbose=verbose, quiet=quiet)
    config_paths = _normalize_config_paths(config)

    def _log(message: str, *, detail: bool = False) -> None:
        _progress_log(
            "run",
            message,
            start_time=start_time,
            log_verbosity=log_verbosity,
            detail=detail,
        )

    _log("Start training/evaluation pipeline.")
    _log("Load and resolve configuration.")
    try:
        resolved = load_and_resolve_config(
            config_paths,
            execution_stage_override=execution_stage,
            allow_empty=False,
        )
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _log(f"Configuration resolved (execution_stage={resolved.runtime.execution_stage}).")

    _log("Build split artifacts.")
    try:
        split_artifacts = build_split_artifacts(resolved)
    except SplitError as exc:
        raise typer.BadParameter(str(exc)) from exc
    fold_count = getattr(split_artifacts, "fold_count", "unknown")
    excluded_rows = getattr(split_artifacts, "expression_rows_excluded", "unknown")
    _log(
        "Split artifacts ready "
        f"(fold_count={fold_count}, excluded_expression_rows={excluded_rows})."
    )

    _log("Run outer cross-validation.")

    def _outer_cv_progress(message: str) -> None:
        _log(message, detail=message.startswith("Outer CV fold stage"))

    try:
        cv_artifacts = run_outer_cv(
            resolved,
            split_artifacts.split_manifest,
            progress_callback=_outer_cv_progress,
        )
    except CVError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _log("Outer cross-validation completed.")

    _log("Derive CV threshold from thresholds table.")
    cv_threshold_values = (
        cv_artifacts.thresholds.filter(pl.col("threshold_name") == "cv_derived_threshold")
        .select("threshold_value")
        .to_series()
        .to_list()
    )
    if len(cv_threshold_values) != 1:
        raise typer.BadParameter("cv_derived_threshold was not found in thresholds table")
    cv_threshold = float(cv_threshold_values[0])
    _log(f"Derived cv_derived_threshold={cv_threshold:.8f}.")
    _emit_run_metric_summary(
        cv_artifacts.metrics_cv,
        start_time=start_time,
        log_verbosity=log_verbosity,
    )

    warnings = list(cv_artifacts.warnings)
    status = "cv_completed"
    final_refit_artifacts = None
    if resolved.runtime.execution_stage == "full_run":
        _log("Run final refit stage.")
        try:
            final_refit_artifacts = run_final_refit(
                config=resolved,
                split_manifest=split_artifacts.split_manifest,
                cv_threshold=cv_threshold,
            )
        except CVError as exc:
            raise typer.BadParameter(str(exc)) from exc
        warnings.extend(final_refit_artifacts.warnings)
        status = "full_run_completed"
        _log("Final refit completed.")
    else:
        _log("Skip final refit stage (execution_stage=cv_only).")

    _log("Create run directory and write core tabular artifacts.")
    run_dir = _build_run_dir("run")
    write_resolved_config(resolved, run_dir / "resolved_config.yml")
    split_artifacts.split_manifest.write_csv(run_dir / "split_manifest.tsv", separator="\t")
    cv_artifacts.metrics_cv.write_csv(
        run_dir / "metrics_cv.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    cv_artifacts.loss_by_split_cv.write_csv(
        run_dir / "loss_by_split_cv.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    cv_artifacts.thresholds.write_csv(
        run_dir / "thresholds.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    cv_artifacts.feature_importance.write_csv(
        run_dir / "feature_importance.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    cv_artifacts.coefficients.write_csv(
        run_dir / "coefficients.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    cv_artifacts.oof_predictions.write_csv(
        run_dir / "prediction_cv.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    if cv_artifacts.ensemble_model_probs is not None:
        cv_artifacts.ensemble_model_probs.write_csv(
            run_dir / "ensemble_model_probs.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
    if cv_artifacts.model_selection_trials is not None:
        cv_artifacts.model_selection_trials.write_csv(
            run_dir / "model_selection_trials.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
    if cv_artifacts.model_selection_trials_summary is not None:
        cv_artifacts.model_selection_trials_summary.write_csv(
            run_dir / "model_selection_trials_summary.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
    bundle_export_result = None
    selected_tables: list[pl.DataFrame] = []
    if cv_artifacts.model_selection_selected is not None:
        selected_tables.append(cv_artifacts.model_selection_selected)
    if final_refit_artifacts is not None:
        final_refit_artifacts.pred_external_test.write_csv(
            run_dir / "prediction_external_test.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        final_refit_artifacts.pred_inference.write_csv(
            run_dir / "prediction_inference.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        final_refit_artifacts.loss_by_split_final_refit.write_csv(
            run_dir / "loss_by_split_final_refit.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        if final_refit_artifacts.model_selection_selected is not None:
            selected_tables.append(final_refit_artifacts.model_selection_selected)
        _log("Export model bundle.")
        try:
            bundle_export_result = export_model_bundle(
                run_dir=run_dir,
                resolved_config_path=run_dir / "resolved_config.yml",
                config=resolved,
                final_refit_artifacts=final_refit_artifacts,
                thresholds=cv_artifacts.thresholds,
            )
        except BundleError as exc:
            raise typer.BadParameter(str(exc)) from exc
        _log(f"Model bundle exported: {bundle_export_result.bundle_dir}.")
    if selected_tables:
        pl.concat(selected_tables, how="vertical_relaxed").sort(
            ["selection_scope", "fold_id", "sample_set_id", "rank", "candidate_index"]
        ).write_csv(
            run_dir / "model_selection_selected.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    feature_filter_counts_tables: list[pl.DataFrame] = []
    cv_feature_filter_counts = getattr(cv_artifacts, "feature_filter_counts", None)
    if isinstance(cv_feature_filter_counts, pl.DataFrame):
        feature_filter_counts_tables.append(cv_feature_filter_counts)
    final_feature_filter_counts = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "feature_filter_counts", None)
    )
    if isinstance(final_feature_filter_counts, pl.DataFrame):
        feature_filter_counts_tables.append(final_feature_filter_counts)
    feature_filter_counts_table: pl.DataFrame | None = None
    if feature_filter_counts_tables:
        feature_filter_counts_table = pl.concat(
            feature_filter_counts_tables, how="vertical_relaxed"
        ).sort(["scope", "fold_id", "sample_set_id"])
        feature_filter_counts_table.write_csv(
            run_dir / "feature_filter_counts.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    feature_filter_summary_tables: list[pl.DataFrame] = []
    cv_feature_filter_summary = getattr(cv_artifacts, "feature_filter_counts_summary", None)
    if isinstance(cv_feature_filter_summary, pl.DataFrame):
        feature_filter_summary_tables.append(cv_feature_filter_summary)
    final_feature_filter_summary = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "feature_filter_counts_summary", None)
    )
    if isinstance(final_feature_filter_summary, pl.DataFrame):
        feature_filter_summary_tables.append(final_feature_filter_summary)
    feature_filter_counts_summary_table: pl.DataFrame | None = None
    if feature_filter_summary_tables:
        feature_filter_counts_summary_table = pl.concat(
            feature_filter_summary_tables, how="vertical_relaxed"
        ).sort(["scope", "stage"])
        feature_filter_counts_summary_table.write_csv(
            run_dir / "feature_filter_counts_summary.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    model_sparsity_tables: list[pl.DataFrame] = []
    cv_model_sparsity = getattr(cv_artifacts, "model_sparsity", None)
    if isinstance(cv_model_sparsity, pl.DataFrame):
        model_sparsity_tables.append(cv_model_sparsity)
    final_model_sparsity = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "model_sparsity", None)
    )
    if isinstance(final_model_sparsity, pl.DataFrame):
        model_sparsity_tables.append(final_model_sparsity)
    model_sparsity_table: pl.DataFrame | None = None
    if model_sparsity_tables:
        model_sparsity_table = pl.concat(model_sparsity_tables, how="vertical_relaxed").sort(
            ["scope", "fold_id", "sample_set_id", "model_index"]
        )
        model_sparsity_table.write_csv(
            run_dir / "model_sparsity.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    model_sparsity_summary_tables: list[pl.DataFrame] = []
    cv_model_sparsity_summary = getattr(cv_artifacts, "model_sparsity_summary", None)
    if isinstance(cv_model_sparsity_summary, pl.DataFrame):
        model_sparsity_summary_tables.append(cv_model_sparsity_summary)
    final_model_sparsity_summary = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "model_sparsity_summary", None)
    )
    if isinstance(final_model_sparsity_summary, pl.DataFrame):
        model_sparsity_summary_tables.append(final_model_sparsity_summary)
    model_sparsity_summary_table: pl.DataFrame | None = None
    if model_sparsity_summary_tables:
        model_sparsity_summary_table = pl.concat(
            model_sparsity_summary_tables, how="vertical_relaxed"
        ).sort(["scope", "model_name"])
        model_sparsity_summary_table.write_csv(
            run_dir / "model_sparsity_summary.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    classification_summary = _classification_summary(
        oof_predictions=cv_artifacts.oof_predictions,
        thresholds=cv_artifacts.thresholds,
        pred_external_test=(
            None if final_refit_artifacts is None else final_refit_artifacts.pred_external_test
        ),
    )
    classification_summary.write_csv(
        run_dir / "classification_summary.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    figure_warnings: list[str] = []
    _log("Generate run figures.")
    try:
        figure_warnings = write_run_figures(
            run_dir=run_dir,
            metrics_cv=cv_artifacts.metrics_cv,
            oof_predictions=cv_artifacts.oof_predictions,
            thresholds=cv_artifacts.thresholds,
            feature_importance=cv_artifacts.feature_importance,
            coefficients=cv_artifacts.coefficients,
            ensemble_model_probs=cv_artifacts.ensemble_model_probs,
            model_selection_trials=cv_artifacts.model_selection_trials,
            model_selection_trials_summary=cv_artifacts.model_selection_trials_summary,
            auto_threshold_metric=resolved.report.auto_threshold_selection_metric,
            loss_by_split_cv=cv_artifacts.loss_by_split_cv,
            loss_by_split_final_refit=(
                None
                if final_refit_artifacts is None
                else final_refit_artifacts.loss_by_split_final_refit
            ),
            pred_external_test=(
                None if final_refit_artifacts is None else final_refit_artifacts.pred_external_test
            ),
            trait_name=resolved.data.trait_col,
            feature_filter_counts_summary=feature_filter_counts_summary_table,
            model_sparsity=model_sparsity_table,
            model_sparsity_summary=model_sparsity_summary_table,
        )
    except FigureError as exc:
        raise typer.BadParameter(str(exc)) from exc
    warnings.extend(figure_warnings)
    _log(f"Run figures generated (figure_warnings={len(figure_warnings)}).")

    end_time = datetime.now(UTC)
    _log("Collect provenance metadata.")
    try:
        input_files = collect_input_files(
            [
                *config_paths,
                Path(resolved.data.metadata_path),
                Path(resolved.data.tpm_path),
            ]
        )
    except ProvenanceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    git_meta = git_snapshot(Path.cwd())
    environment = runtime_environment_snapshot()
    metadata_payload: dict[str, Any] = {
        "command": "run",
        "execution_stage": resolved.runtime.execution_stage,
        "status": status,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_sec": (end_time - start_time).total_seconds(),
        "seed_policy": {
            "runtime_seed": resolved.runtime.seed,
            "ensemble_seed_formula": (
                "int(sha256('{runtime.seed}|{training_scope_id}|{model_index}')[:16],16)%(2**31-1)"
            ),
        },
        "fold_count": split_artifacts.fold_count,
        "pool_counts": split_artifacts.pool_counts,
        "expression_rows_excluded_from_metadata": split_artifacts.expression_rows_excluded,
        "input_files": input_files,
        "environment": environment,
        "warnings": warnings,
        **git_meta,
    }
    if final_refit_artifacts is not None:
        metadata_payload["final_refit_ensemble_size"] = final_refit_artifacts.ensemble_size
    if bundle_export_result is not None:
        metadata_payload["model_bundle_dir"] = str(bundle_export_result.bundle_dir)
        metadata_payload["model_bundle_manifest_sha256"] = bundle_export_result.manifest_sha256

    metadata_path = run_dir / "run_metadata.json"
    _write_metadata(run_dir, payload=metadata_payload)
    _log(f"Metadata written: {metadata_path}.")
    _emit_warning_summary(
        "run",
        warnings,
        metadata_path=metadata_path,
        start_time=start_time,
    )
    full_run_suffix = ""
    if final_refit_artifacts is not None:
        full_run_suffix = (
            "; full_run outputs: "
            "prediction_external_test.tsv, "
            "prediction_inference.tsv, "
            "loss_by_split_final_refit.tsv, "
            "model_bundle/"
        )
    typer.echo(
        f"Wrote run artifacts at {run_dir} "
        "(resolved_config.yml, split_manifest.tsv, metrics_cv.tsv, loss_by_split_cv.tsv, "
        "thresholds.tsv, "
        "feature_importance.tsv, coefficients.tsv, prediction_cv.tsv, "
        "feature_filter_counts.tsv, feature_filter_counts_summary.tsv, "
        "model_sparsity.tsv, model_sparsity_summary.tsv, "
        "classification_summary.tsv, "
        "run_metadata.json"
        f"{full_run_suffix}; warnings={len(warnings)}).",
    )
    _log("Completed.")


@app.command("config")
def config_command(
    out: Annotated[
        Path,
        typer.Option("--out", file_okay=True, dir_okay=False),
    ] = Path("config.yml"),
    config: OptionalConfigPathsArg = None,
    verbose: VerboseArg = False,
    quiet: QuietArg = False,
) -> None:
    """Resolve YAML config (or defaults) into a validated config."""
    start_time = datetime.now(UTC)
    log_verbosity = _resolve_log_verbosity(verbose=verbose, quiet=quiet)
    _progress_log(
        "config",
        "Load and resolve configuration.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )
    config_paths = _normalize_config_paths(config)
    try:
        resolved = load_and_resolve_config(config_paths, allow_empty=True)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _progress_log(
        "config",
        "Write resolved config YAML.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )
    write_resolved_config(resolved, out)
    typer.echo(f"Wrote resolved config: {out}")
    _progress_log(
        "config",
        "Completed.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )


@app.command("dataset")
def dataset(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            help="Output directory for compact C4 test data.",
        ),
    ] = Path("testdata/c4_tiny"),
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help=(
                "Base URL containing species_metadata.tsv and tpm.tsv. "
                "Defaults to GitHub raw content for this repository; "
                "can also be set via PHENORADAR_TESTDATA_BASE_URL."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing files when checksum differs from expected values.",
        ),
    ] = False,
    verbose: VerboseArg = False,
    quiet: QuietArg = False,
) -> None:
    """Fetch compact bundled test data from GitHub (or a custom base URL)."""
    start_time = datetime.now(UTC)
    log_verbosity = _resolve_log_verbosity(verbose=verbose, quiet=quiet)
    _progress_log(
        "dataset",
        "Fetch compact test data files.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )
    try:
        written_paths = fetch_c4_tiny_test_data(out, base_url=base_url, overwrite=force)
    except TestDataError as exc:
        raise typer.BadParameter(str(exc)) from exc

    file_names = ", ".join(path.name for path in written_paths)
    resolved_source = base_url
    if resolved_source is None:
        resolved_source = os.environ.get(
            "PHENORADAR_TESTDATA_BASE_URL",
            DEFAULT_C4_TINY_BASE_URL,
        )
    typer.echo(f"Fetched test data into {out} from {resolved_source} ({file_names}).")
    _progress_log(
        "dataset",
        "Completed.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )


@app.command()
def predict(
    model_bundle: ModelBundleArg,
    config: ConfigPathsArg,
    verbose: VerboseArg = False,
    quiet: QuietArg = False,
) -> None:
    """Predict using an exported model bundle."""
    start_time = datetime.now(UTC)
    log_verbosity = _resolve_log_verbosity(verbose=verbose, quiet=quiet)

    def _log(message: str, *, detail: bool = False) -> None:
        _progress_log(
            "predict",
            message,
            start_time=start_time,
            log_verbosity=log_verbosity,
            detail=detail,
        )

    config_paths = _normalize_config_paths(config)
    if not config_paths:
        raise typer.BadParameter("`--config` / `-c` is required.")
    config_path = config_paths[0]

    _log("Start prediction pipeline.")
    _log("Load and resolve configuration.")
    try:
        resolved = load_and_resolve_config([config_path])
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _log("Load model bundle and run predictions.")
    try:
        bundle = load_model_bundle(model_bundle)
        pred_predict, predict_warnings = predict_with_bundle(resolved, bundle)
    except BundleError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if "true_label" not in pred_predict.columns:
        pred_predict = pred_predict.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("true_label")
        )
    pred_predict = pred_predict.select(
        [
            name
            for name in [
                "species",
                "true_label",
                "prob",
                "pred_label_fixed_threshold",
                "pred_label_cv_derived_threshold",
                "uncertainty_std",
            ]
            if name in pred_predict.columns
        ]
    )
    _emit_predict_summary(pred_predict, start_time=start_time, log_verbosity=log_verbosity)

    _log("Write prediction artifacts.")
    run_dir = _build_run_dir("predict")
    write_resolved_config(resolved, run_dir / "resolved_config.yml")
    pred_predict.write_csv(
        run_dir / "prediction_inference.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    _log("Generate prediction figures.")
    try:
        write_predict_figures(
            run_dir=run_dir,
            pred_predict=pred_predict,
            require_uncertainty=len(bundle.models) > 1,
        )
    except FigureError as exc:
        raise typer.BadParameter(str(exc)) from exc
    end_time = datetime.now(UTC)
    _log("Collect provenance metadata.")
    try:
        input_files = collect_input_files(
            [
                config_path,
                Path(resolved.data.metadata_path),
                Path(resolved.data.tpm_path),
                model_bundle / "bundle_manifest.json",
            ]
        )
        payload_sha = bundle_payload_sha256(model_bundle)
    except ProvenanceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    git_meta = git_snapshot(Path.cwd())
    environment = runtime_environment_snapshot()
    metadata_path = run_dir / "run_metadata.json"
    _write_metadata(
        run_dir,
        payload={
            "command": "predict",
            "execution_stage": "predict",
            "status": "predict_completed",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_sec": (end_time - start_time).total_seconds(),
            "seed_policy": {
                "runtime_seed": resolved.runtime.seed,
            },
            "model_bundle_path": str(model_bundle),
            "model_bundle_manifest_sha256": bundle.manifest_sha256,
            "model_bundle_payload_sha256": payload_sha,
            "bundle_source_run_id": bundle.source_run_id,
            "bundle_source_run_dir": str(bundle.manifest.get("source_run_dir", "unknown")),
            "input_files": input_files,
            "environment": environment,
            "warnings": predict_warnings,
            **git_meta,
        },
    )
    _log(f"Metadata written: {metadata_path}.")
    _emit_warning_summary(
        "predict",
        predict_warnings,
        metadata_path=metadata_path,
        start_time=start_time,
    )
    typer.echo(
        f"Wrote predict artifacts at {run_dir} "
        "(resolved_config.yml, prediction_inference.tsv, "
        f"run_metadata.json; warnings={len(predict_warnings)}).",
    )
    _log("Completed.")


@app.command()
def report(
    run_dir: ReportRunDirArg = None,
    runs_root: ReportRunsRootArg = None,
    glob_pattern: Annotated[
        str,
        typer.Option(
            "--glob",
            help="Glob pattern when scanning --runs-root.",
        ),
    ] = "*",
    latest: Annotated[
        int | None,
        typer.Option(
            "--latest",
            min=1,
            help="Include only latest N run directories after glob expansion.",
        ),
    ] = None,
    primary_metric: Annotated[
        PrimaryMetric,
        typer.Option(
            "--primary-metric",
            help="Metric used for ranking.",
        ),
    ] = "mcc",
    aggregate_scope: Annotated[
        AggregateScope,
        typer.Option(
            "--aggregate-scope",
            help="Aggregate scope in metrics_cv.tsv used for ranking.",
        ),
    ] = "macro",
    include_stage: Annotated[
        IncludeStage,
        typer.Option(
            "--include-stage",
            help="Stage filter for selected runs.",
        ),
    ] = "all",
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--output-format",
            help="Report output format.",
        ),
    ] = "tsv",
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Fail on missing/invalid run artifacts instead of warn-and-skip.",
        ),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            help="Output directory for report artifacts.",
        ),
    ] = None,
    verbose: VerboseArg = False,
    quiet: QuietArg = False,
) -> None:
    """Generate cross-run comparison reports."""
    start_time = datetime.now(UTC)
    log_verbosity = _resolve_log_verbosity(verbose=verbose, quiet=quiet)
    _progress_log(
        "report",
        "Start report generation.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )
    try:
        run_dir_values = [] if run_dir is None else run_dir
        output_dir = out if out is not None else _build_report_dir()
        _progress_log(
            "report",
            f"Generate report artifacts into {output_dir}.",
            start_time=start_time,
            log_verbosity=log_verbosity,
        )
        generate_report(
            run_dirs=run_dir_values,
            runs_root=runs_root,
            run_glob=glob_pattern,
            latest=latest,
            options=ReportOptions(
                primary_metric=primary_metric,
                aggregate_scope=aggregate_scope,
                include_stage=include_stage,
                output_format=output_format,
                strict=strict,
                run_glob=glob_pattern,
                latest=latest,
            ),
            output_dir=output_dir,
        )
    except ReportError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit_report_summary(output_dir=output_dir, start_time=start_time, log_verbosity=log_verbosity)
    _emit_report_warning_summary(output_dir=output_dir, start_time=start_time)
    typer.echo(
        f"Wrote report artifacts at {output_dir} "
        "(report_manifest.json, report_runs.tsv, report_ranking.tsv, report_warnings.tsv)."
    )
    _progress_log(
        "report",
        "Completed.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )


if __name__ == "__main__":
    app()
