"""CLI entry points for PhenoRadar."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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
    AppConfig,
    ConfigConditionSet,
    ConfigError,
    ExecutionStage,
    has_condition_dimensions,
    load_and_resolve_config,
    load_config_conditions,
    write_resolved_config,
)
from phenoradar.cv import CVError, run_final_refit, run_outer_cv
from phenoradar.figures import (
    FigureError,
    figure_annotation_features,
    write_group_probability_figure,
    write_predict_figures,
    write_run_figures,
)
from phenoradar.group_bootstrap import (
    GroupBootstrapArtifacts,
    GroupBootstrapError,
    run_oof_group_bootstrap,
)
from phenoradar.group_summary import (
    GroupSummaryError,
    build_group_summary_artifacts,
)
from phenoradar.metadata import (
    MetadataError,
    build_species_metadata_from_skim,
    build_species_taxid_tsv,
    fetch_ncbi_tree,
)
from phenoradar.metrics import (
    FIXED_PROBABILITY_THRESHOLD_NAME,
    evaluation_metric_contract,
    evaluation_metric_contract_rows,
)
from phenoradar.orthogroup_annotation import (
    OrthogroupAnnotationError,
    load_orthogroup_annotations,
)
from phenoradar.provenance import (
    FINGERPRINT_SCHEMA_VERSION,
    ProvenanceError,
    bundle_payload_sha256,
    collect_input_files,
    dataset_fingerprint,
    experiment_fingerprint,
    input_hashes_by_role,
    phenoradar_build_snapshot,
    runtime_environment_snapshot,
    split_fingerprint,
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
from phenoradar.split import SplitArtifacts, SplitError, build_split_artifacts
from phenoradar.study import (
    StudyError,
    condition_run_is_complete,
    generate_study_report,
    load_condition_manifest,
    new_condition_manifest,
    prepare_condition_attempt,
    validate_resume_manifest,
    write_condition_manifest,
    write_config_differences,
)
from phenoradar.testdata import (
    BUNDLED_C4_TINY_SOURCE,
    TestDataError,
    fetch_c4_tiny_test_data,
    resolve_c4_tiny_base_url,
)
from phenoradar.timing import TimingRecorder, run_timing_summary
from phenoradar.tree_prediction import (
    TreePredictionError,
    write_predict_tree_prediction_artifacts,
    write_run_tree_prediction_artifacts,
)

app = typer.Typer(
    help="PhenoRadar: orthogroup TPM-based phenotype prediction CLI",
    context_settings={"help_option_names": ["-h", "--help"]},
)


LogVerbosity = Literal["quiet", "normal", "verbose"]
_ARTIFACT_PARALLEL_WORKER_CAP = 4
_EVALUATION_CONTRACT_VERSION = 1


def _build_run_fingerprint_metadata(
    *,
    config: AppConfig,
    split_manifest: pl.DataFrame,
    input_files: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_sha256 = dataset_fingerprint(
        input_hashes_by_role(
            input_files,
            {
                "metadata": Path(config.data.metadata_path),
                "tpm": Path(config.data.tpm_path),
            },
        )
    )
    split_sha256 = split_fingerprint(split_manifest)
    evaluation_contract = {
        "evaluation_contract_version": _EVALUATION_CONTRACT_VERSION,
        "label_unit": "species",
        "trait_col": config.data.trait_col,
        "group_col": config.split.group_col,
        "metric_contract": evaluation_metric_contract(),
    }
    experiment_sha256 = experiment_fingerprint(
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        evaluation_contract=evaluation_contract,
    )
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_sha256,
        "split_fingerprint": split_sha256,
        "experiment_fingerprint": experiment_sha256,
        "evaluation_contract": evaluation_contract,
    }


def _artifact_parallel_workers(config: AppConfig) -> int:
    runtime_n_jobs = int(getattr(config.runtime, "n_jobs", 1))
    return max(1, min(runtime_n_jobs, _ARTIFACT_PARALLEL_WORKER_CAP))


def _prepare_run_inputs(
    *,
    config_paths: list[Path],
    config: AppConfig,
    timing_recorder: TimingRecorder,
    split_artifacts_override: SplitArtifacts | None = None,
) -> tuple[list[dict[str, Any]], SplitArtifacts]:
    """Hash run inputs while independently constructing split artifacts."""

    tree_path = getattr(config.data, "tree_path", None)
    orthogroup_annotation_path = getattr(config.data, "orthogroup_annotation_path", None)
    provenance_paths = [
        *config_paths,
        Path(config.data.metadata_path),
        Path(config.data.tpm_path),
        *([] if tree_path is None else [Path(tree_path)]),
        *(
            []
            if orthogroup_annotation_path is None
            else [Path(orthogroup_annotation_path)]
        ),
    ]

    def _collect_provenance() -> list[dict[str, Any]]:
        started = timing_recorder.start()
        try:
            return collect_input_files(provenance_paths)
        finally:
            timing_recorder.record_since(
                started,
                scope="run",
                stage="input_provenance",
            )

    def _build_splits() -> SplitArtifacts:
        started = timing_recorder.start()
        try:
            if split_artifacts_override is not None:
                return split_artifacts_override
            return build_split_artifacts(config)
        finally:
            timing_recorder.record_since(
                started,
                scope="run",
                stage="split_construction",
            )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="phenoradar-input") as executor:
        provenance_future = executor.submit(_collect_provenance)
        split_future = executor.submit(_build_splits)
        return provenance_future.result(), split_future.result()


def _feature_filter_funnel_stage_order(config: AppConfig) -> list[str]:
    stages = ["n_features_before"]
    if config.preprocess.sparse_feature_filter.enabled:
        stages.append("n_features_after_sparse_feature_filter")
    if config.preprocess.low_variance_filter.enabled:
        stages.append("n_features_after_low_variance")
    if config.preprocess.ranked_feature_filter.method != "none":
        stages.append("n_features_after_ranked_feature_filter")
    if config.preprocess.correlation_filter.enabled:
        stages.append("n_features_after_correlation")
    return stages


def _stage_tables_dir(run_dir: Path, stage: str) -> Path:
    tables_dir = run_dir / stage / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def _stage_figures_dir(run_dir: Path, stage: str) -> Path:
    figures_dir = run_dir / stage / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _write_group_summary_artifacts(
    *,
    run_dir: Path,
    stage: str,
    predictions: pl.DataFrame,
    source_table_name: str,
    config: AppConfig,
) -> list[str]:
    try:
        artifacts = build_group_summary_artifacts(
            predictions=predictions,
            metadata_path=Path(config.data.metadata_path),
            species_col=config.data.species_col,
            group_col=config.summary.group_col,
            group_name_col=config.summary.group_name_col,
            source_table_name=source_table_name,
        )
    except GroupSummaryError as exc:
        return [str(exc)]

    tables_dir = _stage_tables_dir(run_dir, stage)
    figures_dir = _stage_figures_dir(run_dir, stage)
    artifacts.summary.write_csv(
        tables_dir / f"group_summary_{artifacts.suffix}.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    try:
        write_group_probability_figure(
            grouped_predictions=artifacts.predictions,
            out_path=figures_dir / f"probability_by_{artifacts.suffix}.svg",
            group_label=artifacts.group_label,
            source_table_name=source_table_name,
            figure_name=f"probability_by_{artifacts.suffix}.svg",
        )
    except FigureError as exc:
        return [str(exc)]
    return []


def _run_table_dirs(run_dir: Path) -> dict[str, Path]:
    table_dirs = {
        "split": run_dir / "split" / "tables",
        "cv": run_dir / "cv" / "tables",
        "external_test": run_dir / "external_test" / "tables",
        "inference": run_dir / "inference" / "tables",
        "model": run_dir / "model" / "tables",
        "summary": run_dir / "summary" / "tables",
        "runtime": run_dir / "runtime" / "tables",
    }
    for tables_dir in table_dirs.values():
        tables_dir.mkdir(parents=True, exist_ok=True)
    return table_dirs


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
ResumeStudyArg = Annotated[
    Path | None,
    typer.Option(
        "--resume",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=True,
        help="Resume a multi-condition study directory.",
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
    if "pred_label_fixed_threshold" in pred_predict.columns:
        n_positive = int(pred_predict.filter(pl.col("pred_label_fixed_threshold") == 1).height)
    message = f"Prediction summary (n_species={n_species}"
    if n_positive is not None:
        message += f", n_pred_positive={n_positive}"
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
    for required_name in (FIXED_PROBABILITY_THRESHOLD_NAME,):
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
                "prediction_external_test.tsv schema is invalid for classification_summary.tsv"
            )
        missing_label_count = pred_external_test.filter(pl.col("true_label").is_null()).height
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


def _run_single(
    config: ConfigPathsArg,
    execution_stage: ExecutionStageArg = None,
    verbose: VerboseArg = False,
    quiet: QuietArg = False,
    *,
    resolved_override: AppConfig | None = None,
    split_artifacts_override: SplitArtifacts | None = None,
    run_dir_override: Path | None = None,
    study_context: dict[str, Any] | None = None,
) -> Path:
    """Run training/evaluation pipeline."""
    start_time = datetime.now(UTC)
    timing_recorder = TimingRecorder()
    run_total_started = timing_recorder.start()
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
    config_started = timing_recorder.start()
    if resolved_override is None:
        try:
            resolved = load_and_resolve_config(
                config_paths,
                execution_stage_override=execution_stage,
                allow_empty=False,
            )
        except ConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        resolved = resolved_override
    timing_recorder.record_since(
        config_started,
        scope="run",
        stage="config_resolution",
    )
    _log(f"Configuration resolved (execution_stage={resolved.runtime.execution_stage}).")

    tree_path = getattr(resolved.data, "tree_path", None)
    orthogroup_annotation_path = getattr(resolved.data, "orthogroup_annotation_path", None)
    _log("Collect input file identities and build split artifacts.")
    try:
        input_files, split_artifacts = _prepare_run_inputs(
            config_paths=config_paths,
            config=resolved,
            timing_recorder=timing_recorder,
            split_artifacts_override=split_artifacts_override,
        )
    except (ProvenanceError, SplitError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    fold_count = getattr(split_artifacts, "fold_count", "unknown")
    excluded_rows = getattr(split_artifacts, "expression_rows_excluded", "unknown")
    _log(
        "Split artifacts ready "
        f"(fold_count={fold_count}, excluded_expression_rows={excluded_rows})."
    )
    single_label_validation_folds = split_artifacts.fold_diagnostics.filter(
        ~pl.col("two_class_validation_metrics_defined")
    ).height
    single_label_validation_groups = split_artifacts.fold_validation_groups.filter(
        pl.col("validation_label_profile") != "both"
    ).height
    _log(
        "Split diagnostics ready "
        f"(single_label_validation_folds={single_label_validation_folds}, "
        f"single_label_validation_groups={single_label_validation_groups})."
    )
    fingerprint_started = timing_recorder.start()
    try:
        fingerprint_metadata = _build_run_fingerprint_metadata(
            config=resolved,
            split_manifest=split_artifacts.split_manifest,
            input_files=input_files,
        )
    except ProvenanceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    timing_recorder.record_since(
        fingerprint_started,
        scope="run",
        stage="fingerprint_generation",
    )

    _log("Run outer cross-validation.")

    def _outer_cv_progress(message: str) -> None:
        _log(message, detail=message.startswith("Outer CV fold stage"))

    outer_cv_started = timing_recorder.start()
    try:
        cv_artifacts = run_outer_cv(
            resolved,
            split_artifacts.split_manifest,
            progress_callback=_outer_cv_progress,
            timing_recorder=timing_recorder,
        )
    except CVError as exc:
        raise typer.BadParameter(str(exc)) from exc
    timing_recorder.record_since(
        outer_cv_started,
        scope="run",
        stage="outer_cv",
    )
    _log("Outer cross-validation completed.")

    _emit_run_metric_summary(
        cv_artifacts.metrics_cv,
        start_time=start_time,
        log_verbosity=log_verbosity,
    )

    warnings = list(cv_artifacts.warnings)
    group_bootstrap_artifacts: GroupBootstrapArtifacts | None = None
    group_bootstrap_config = resolved.evaluation.group_bootstrap
    if group_bootstrap_config.enabled:
        _log(
            "Run OOF group bootstrap "
            f"(group_col={resolved.split.group_col}, "
            f"n_resamples={group_bootstrap_config.n_resamples})."
        )
        group_bootstrap_started = timing_recorder.start()
        try:
            group_bootstrap_artifacts = run_oof_group_bootstrap(
                oof_predictions=cv_artifacts.oof_predictions,
                split_manifest=split_artifacts.split_manifest,
                group_col=resolved.split.group_col,
                n_resamples=int(group_bootstrap_config.n_resamples),
                confidence_level=float(group_bootstrap_config.confidence_level),
                runtime_seed=int(resolved.runtime.seed),
            )
        except GroupBootstrapError as exc:
            raise typer.BadParameter(str(exc)) from exc
        timing_recorder.record_since(
            group_bootstrap_started,
            scope="run",
            stage="group_bootstrap",
        )
        warnings.extend(group_bootstrap_artifacts.warnings)
        _log(
            "OOF group bootstrap completed "
            f"(n_groups={group_bootstrap_artifacts.n_groups}, "
            f"seed={group_bootstrap_artifacts.seed})."
        )
    else:
        _log("Skip OOF group bootstrap (evaluation.group_bootstrap.enabled=false).")

    status = "cv_completed"
    final_refit_artifacts = None
    if resolved.runtime.execution_stage == "full_run":
        _log("Run final refit stage.")
        final_refit_started = timing_recorder.start()
        try:
            final_refit_artifacts = run_final_refit(
                config=resolved,
                split_manifest=split_artifacts.split_manifest,
                timing_recorder=timing_recorder,
            )
        except CVError as exc:
            raise typer.BadParameter(str(exc)) from exc
        timing_recorder.record_since(
            final_refit_started,
            scope="run",
            stage="final_refit",
        )
        warnings.extend(final_refit_artifacts.warnings)
        status = "full_run_completed"
        _log("Final refit completed.")
    else:
        _log("Skip final refit stage (execution_stage=cv_only).")

    _log("Create run directory and write core tabular artifacts.")
    artifact_writing_started = timing_recorder.start()
    if run_dir_override is None:
        run_dir = _build_run_dir("run")
    else:
        run_dir = run_dir_override
        run_dir.mkdir(parents=True, exist_ok=False)
    table_dirs = _run_table_dirs(run_dir)
    split_tables_dir = table_dirs["split"]
    cv_tables_dir = table_dirs["cv"]
    external_test_tables_dir = table_dirs["external_test"]
    inference_tables_dir = table_dirs["inference"]
    model_tables_dir = table_dirs["model"]
    summary_tables_dir = table_dirs["summary"]
    runtime_tables_dir = table_dirs["runtime"]
    write_resolved_config(resolved, run_dir / "resolved_config.yml")
    split_artifacts.split_manifest.write_csv(
        split_tables_dir / "split_manifest.tsv", separator="\t"
    )
    split_artifacts.fold_validation_groups.write_csv(
        split_tables_dir / "fold_validation_groups.tsv", separator="\t"
    )
    split_artifacts.fold_diagnostics.write_csv(
        split_tables_dir / "fold_diagnostics.tsv", separator="\t"
    )
    cv_artifacts.metrics_cv.write_csv(
        cv_tables_dir / "metrics_cv.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    cv_artifacts.loss_by_split_cv.write_csv(
        cv_tables_dir / "loss_by_split_cv.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    if group_bootstrap_artifacts is not None:
        group_bootstrap_artifacts.summary.write_csv(
            cv_tables_dir / "group_bootstrap_metrics.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        group_bootstrap_artifacts.replicates.write_csv(
            cv_tables_dir / "group_bootstrap_replicates.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
    cv_artifacts.thresholds.write_csv(
        model_tables_dir / "thresholds.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    pl.DataFrame(evaluation_metric_contract_rows()).sort("metric_name").write_csv(
        model_tables_dir / "evaluation_contract.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    cv_artifacts.feature_importance.write_csv(
        cv_tables_dir / "feature_importance.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    cv_artifacts.feature_importance_by_fold.write_csv(
        cv_tables_dir / "feature_importance_by_fold.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    cv_artifacts.coefficients.write_csv(
        cv_tables_dir / "coefficients.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    cv_artifacts.coefficients_by_fold.write_csv(
        cv_tables_dir / "coefficients_by_fold.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    cv_artifacts.feature_stability_by_feature.write_csv(
        cv_tables_dir / "feature_stability_by_feature.tsv",
        separator="\t",
        float_precision=12,
        null_value="NA",
    )
    cv_artifacts.feature_stability_by_fold_pair.write_csv(
        cv_tables_dir / "feature_stability_by_fold_pair.tsv",
        separator="\t",
        float_precision=12,
        null_value="NA",
    )
    cv_artifacts.feature_stability_summary.write_csv(
        cv_tables_dir / "feature_stability_summary.tsv",
        separator="\t",
        float_precision=12,
        null_value="NA",
    )
    cv_artifacts.oof_predictions.write_csv(
        cv_tables_dir / "prediction_cv.tsv", separator="\t", float_precision=8, null_value="NA"
    )
    if cv_artifacts.ensemble_model_probs is not None:
        cv_artifacts.ensemble_model_probs.write_csv(
            cv_tables_dir / "ensemble_model_probs.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
    if cv_artifacts.model_selection_trials is not None:
        cv_artifacts.model_selection_trials.write_csv(
            cv_tables_dir / "model_selection_trials.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
    if cv_artifacts.model_selection_trials_summary is not None:
        cv_artifacts.model_selection_trials_summary.write_csv(
            cv_tables_dir / "model_selection_trials_summary.tsv",
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
            external_test_tables_dir / "prediction_external_test.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        final_refit_artifacts.pred_inference.write_csv(
            inference_tables_dir / "prediction_inference.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )
        final_refit_artifacts.loss_by_split_final_refit.write_csv(
            external_test_tables_dir / "loss_by_split_final_refit.tsv",
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
    model_selection_selected_table: pl.DataFrame | None = None
    if selected_tables:
        model_selection_selected_table = pl.concat(selected_tables, how="vertical_relaxed").sort(
            ["selection_scope", "fold_id", "sample_set_id", "rank", "candidate_index"]
        )
        model_selection_selected_table.write_csv(
            model_tables_dir / "model_selection_selected.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    training_group_subset_tables: list[pl.DataFrame] = []
    cv_training_group_subsets = getattr(cv_artifacts, "training_group_subsets", None)
    if isinstance(cv_training_group_subsets, pl.DataFrame):
        training_group_subset_tables.append(cv_training_group_subsets)
    final_training_group_subsets = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "training_group_subsets", None)
    )
    if isinstance(final_training_group_subsets, pl.DataFrame):
        training_group_subset_tables.append(final_training_group_subsets)
    if training_group_subset_tables:
        pl.concat(training_group_subset_tables, how="vertical").sort(
            ["scope", "fold_id", "group_rank", "group_id"]
        ).write_csv(
            model_tables_dir / "training_group_subsets.tsv",
            separator="\t",
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
            model_tables_dir / "feature_filter_counts.tsv",
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
            model_tables_dir / "feature_filter_counts_summary.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    ranked_feature_score_tables: list[pl.DataFrame] = []
    cv_ranked_feature_scores = getattr(cv_artifacts, "ranked_feature_scores", None)
    if isinstance(cv_ranked_feature_scores, pl.DataFrame):
        ranked_feature_score_tables.append(cv_ranked_feature_scores)
    final_ranked_feature_scores = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "ranked_feature_scores", None)
    )
    if isinstance(final_ranked_feature_scores, pl.DataFrame):
        ranked_feature_score_tables.append(final_ranked_feature_scores)
    if ranked_feature_score_tables:
        pl.concat(ranked_feature_score_tables, how="vertical_relaxed").sort(
            ["scope", "fold_id", "sample_set_id", "rank", "feature"],
            nulls_last=True,
        ).write_csv(
            model_tables_dir / "ranked_feature_scores.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    retained_features_tables: list[pl.DataFrame] = []
    cv_retained_features = getattr(cv_artifacts, "retained_features", None)
    if isinstance(cv_retained_features, pl.DataFrame):
        retained_features_tables.append(cv_retained_features)
    final_retained_features = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "retained_features", None)
    )
    if isinstance(final_retained_features, pl.DataFrame):
        retained_features_tables.append(final_retained_features)
    retained_features_table: pl.DataFrame | None = None
    if retained_features_tables:
        retained_features_table = pl.concat(retained_features_tables, how="vertical_relaxed").sort(
            ["scope", "fold_id", "sample_set_id", "feature"]
        )
        retained_features_table.write_csv(
            model_tables_dir / "retained_features.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    retained_feature_summary_tables: list[pl.DataFrame] = []
    cv_retained_features_summary = getattr(cv_artifacts, "retained_features_summary", None)
    if isinstance(cv_retained_features_summary, pl.DataFrame):
        retained_feature_summary_tables.append(cv_retained_features_summary)
    final_retained_features_summary = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "retained_features_summary", None)
    )
    if isinstance(final_retained_features_summary, pl.DataFrame):
        retained_feature_summary_tables.append(final_retained_features_summary)
    retained_features_summary_table: pl.DataFrame | None = None
    if retained_feature_summary_tables:
        retained_features_summary_table = pl.concat(
            retained_feature_summary_tables, how="vertical_relaxed"
        ).sort(
            ["scope", "fold_id", "retained_rate", "retained_count", "feature"],
            descending=[False, False, True, True, False],
        )
        retained_features_summary_table.write_csv(
            model_tables_dir / "retained_features_summary.tsv",
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
            model_tables_dir / "model_sparsity.tsv",
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
            model_tables_dir / "model_sparsity_summary.tsv",
            separator="\t",
            float_precision=8,
            null_value="NA",
        )

    convergence_tables: list[pl.DataFrame] = []
    cv_convergence_diagnostics = getattr(cv_artifacts, "convergence_diagnostics", None)
    if isinstance(cv_convergence_diagnostics, pl.DataFrame):
        convergence_tables.append(cv_convergence_diagnostics)
    final_convergence_diagnostics = (
        None
        if final_refit_artifacts is None
        else getattr(final_refit_artifacts, "convergence_diagnostics", None)
    )
    if isinstance(final_convergence_diagnostics, pl.DataFrame):
        convergence_tables.append(final_convergence_diagnostics)
    if convergence_tables:
        convergence_diagnostics = pl.concat(convergence_tables, how="vertical").sort(
            [
                "training_scope",
                "fold_id",
                "fit_scope",
                "sample_set_id",
                "candidate_index",
                "inner_fold_id",
                "model_index",
            ],
            nulls_last=True,
        )
        convergence_diagnostics.write_csv(
            model_tables_dir / "convergence_diagnostics.tsv",
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
        summary_tables_dir / "classification_summary.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )

    group_summary_warnings: list[str] = []
    group_summary_warnings.extend(
        _write_group_summary_artifacts(
            run_dir=run_dir,
            stage="cv",
            predictions=cv_artifacts.oof_predictions,
            source_table_name="prediction_cv.tsv",
            config=resolved,
        )
    )
    if final_refit_artifacts is not None:
        if final_refit_artifacts.pred_external_test.height > 0:
            group_summary_warnings.extend(
                _write_group_summary_artifacts(
                    run_dir=run_dir,
                    stage="external_test",
                    predictions=final_refit_artifacts.pred_external_test,
                    source_table_name="prediction_external_test.tsv",
                    config=resolved,
                )
            )
        if final_refit_artifacts.pred_inference.height > 0:
            group_summary_warnings.extend(
                _write_group_summary_artifacts(
                    run_dir=run_dir,
                    stage="inference",
                    predictions=final_refit_artifacts.pred_inference,
                    source_table_name="prediction_inference.tsv",
                    config=resolved,
                )
            )
    warnings.extend(group_summary_warnings)
    timing_recorder.record_since(
        artifact_writing_started,
        scope="run",
        stage="artifact_writing",
    )

    figure_generation_started = timing_recorder.start()
    figure_warnings: list[str] = []
    try:
        annotation_features = figure_annotation_features(
            feature_importance=cv_artifacts.feature_importance,
            coefficients=cv_artifacts.coefficients,
            top_features=resolved.figures.top_features,
        )
        orthogroup_annotations = load_orthogroup_annotations(
            None if orthogroup_annotation_path is None else Path(orthogroup_annotation_path),
            feature_names=annotation_features,
        )
    except OrthogroupAnnotationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _log("Generate run figures.")
    try:
        figure_warnings = write_run_figures(
            run_dir=run_dir,
            metrics_cv=cv_artifacts.metrics_cv,
            oof_predictions=cv_artifacts.oof_predictions,
            feature_importance=cv_artifacts.feature_importance,
            coefficients=cv_artifacts.coefficients,
            feature_importance_by_fold=cv_artifacts.feature_importance_by_fold,
            coefficients_by_fold=cv_artifacts.coefficients_by_fold,
            feature_stability_by_feature=cv_artifacts.feature_stability_by_feature,
            feature_stability_by_fold_pair=cv_artifacts.feature_stability_by_fold_pair,
            ensemble_model_probs=cv_artifacts.ensemble_model_probs,
            model_selection_trials=cv_artifacts.model_selection_trials,
            model_selection_trials_summary=cv_artifacts.model_selection_trials_summary,
            model_selection_selected=model_selection_selected_table,
            loss_by_split_cv=cv_artifacts.loss_by_split_cv,
            loss_by_split_final_refit=(
                None
                if final_refit_artifacts is None
                else final_refit_artifacts.loss_by_split_final_refit
            ),
            pred_external_test=(
                None if final_refit_artifacts is None else final_refit_artifacts.pred_external_test
            ),
            pred_inference=(
                None if final_refit_artifacts is None else final_refit_artifacts.pred_inference
            ),
            classification_summary=classification_summary,
            trait_name=resolved.data.trait_col,
            feature_filter_counts_summary=feature_filter_counts_summary_table,
            feature_filter_funnel_stage_order=_feature_filter_funnel_stage_order(resolved),
            model_sparsity=model_sparsity_table,
            model_sparsity_summary=model_sparsity_summary_table,
            top_feature_expression=cv_artifacts.top_feature_expression,
            top_features=resolved.figures.top_features,
            orthogroup_annotations=orthogroup_annotations,
            parallel_workers=_artifact_parallel_workers(resolved),
            group_bootstrap_metrics=(
                None
                if group_bootstrap_artifacts is None
                else group_bootstrap_artifacts.summary
            ),
        )
    except FigureError as exc:
        raise typer.BadParameter(str(exc)) from exc
    contrast_pair_col = resolved.data.contrast_pair_col
    if tree_path is not None and contrast_pair_col is not None:
        try:
            tree_warnings = write_run_tree_prediction_artifacts(
                run_dir=run_dir,
                tree_path=Path(tree_path),
                metadata_path=Path(resolved.data.metadata_path),
                tpm_path=Path(resolved.data.tpm_path),
                species_col=resolved.data.species_col,
                feature_col=resolved.data.feature_col,
                value_col=resolved.data.value_col,
                trait_col=resolved.data.trait_col,
                group_col=contrast_pair_col,
                oof_predictions=cv_artifacts.oof_predictions,
                thresholds=cv_artifacts.thresholds,
                feature_importance=cv_artifacts.feature_importance,
                coefficients=cv_artifacts.coefficients,
                pred_external_test=(
                    None
                    if final_refit_artifacts is None
                    else final_refit_artifacts.pred_external_test
                ),
                top_feature_expression=getattr(
                    cv_artifacts, "top_feature_expression", None
                ),
                feature_limit=resolved.figures.top_features,
                orthogroup_annotations=orthogroup_annotations,
                parallel_workers=_artifact_parallel_workers(resolved),
            )
        except TreePredictionError as exc:
            raise typer.BadParameter(str(exc)) from exc
        figure_warnings.extend(tree_warnings)
    elif tree_path is not None:
        figure_warnings.append(
            "Skipped tree prediction artifacts because data.contrast_pair_col is null."
        )
    warnings.extend(figure_warnings)
    timing_recorder.record_since(
        figure_generation_started,
        scope="run",
        stage="figure_generation",
    )
    _log(f"Run figures generated (figure_warnings={len(figure_warnings)}).")

    _log("Collect runtime provenance metadata.")
    runtime_provenance_started = timing_recorder.start()
    build_meta = phenoradar_build_snapshot()
    environment = runtime_environment_snapshot()
    timing_recorder.record_since(
        runtime_provenance_started,
        scope="run",
        stage="runtime_provenance",
    )
    timing_recorder.record_since(
        run_total_started,
        scope="run",
        stage="total",
    )
    timing_table = timing_recorder.to_frame()
    timing_table.write_csv(
        runtime_tables_dir / "timing.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    end_time = datetime.now(UTC)
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
        "timing": {
            "artifact_path": "runtime/tables/timing.tsv",
            "clock": "time.perf_counter",
            "parallel_intervals_may_overlap": True,
            "stage_duration_sec": run_timing_summary(timing_table),
        },
        **fingerprint_metadata,
        **build_meta,
    }
    if study_context is not None:
        metadata_payload["study"] = study_context
    if final_refit_artifacts is not None:
        metadata_payload["final_refit_ensemble_size"] = final_refit_artifacts.ensemble_size
    if bundle_export_result is not None:
        metadata_payload["model_bundle_dir"] = str(bundle_export_result.bundle_dir)
        metadata_payload["model_bundle_manifest_sha256"] = bundle_export_result.manifest_sha256
    if group_bootstrap_artifacts is not None:
        metadata_payload["group_bootstrap"] = {
            "group_col": resolved.split.group_col,
            "n_groups": group_bootstrap_artifacts.n_groups,
            "n_resamples": int(group_bootstrap_config.n_resamples),
            "confidence_level": float(group_bootstrap_config.confidence_level),
            "bootstrap_method": "percentile_group",
            "seed": group_bootstrap_artifacts.seed,
        }

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
            "external_test/tables/, "
            "external_test/figures/, "
            "inference/tables/, "
            "inference/figures/, "
            "model_bundle/"
        )
    typer.echo(
        f"Wrote run artifacts at {run_dir} "
        "(resolved_config.yml, split/tables/, cv/tables/, cv/figures/, "
        "model/tables/, summary/tables/, runtime/tables/, "
        "run_metadata.json"
        f"{full_run_suffix}; warnings={len(warnings)}).",
    )
    _log("Completed.")
    return run_dir


def _write_study_metadata(study_dir: Path, payload: dict[str, Any]) -> Path:
    metadata_path = study_dir / "study_metadata.json"
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _write_study_split_artifacts(study_dir: Path, split_artifacts: SplitArtifacts) -> None:
    tables_dir = study_dir / "split" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    split_artifacts.split_manifest.write_csv(
        tables_dir / "split_manifest.tsv",
        separator="\t",
    )
    split_artifacts.fold_validation_groups.write_csv(
        tables_dir / "fold_validation_groups.tsv",
        separator="\t",
    )
    split_artifacts.fold_diagnostics.write_csv(
        tables_dir / "fold_diagnostics.tsv",
        separator="\t",
    )


def _study_split_artifacts(
    *,
    study_dir: Path,
    condition_set: ConfigConditionSet,
    resume: bool,
) -> tuple[SplitArtifacts, str]:
    try:
        split_artifacts = build_split_artifacts(condition_set.conditions[0].config)
        split_sha256 = split_fingerprint(split_artifacts.split_manifest)
    except (ProvenanceError, SplitError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    shared_manifest_path = study_dir / "split" / "tables" / "split_manifest.tsv"
    if resume:
        if not shared_manifest_path.exists():
            raise typer.BadParameter(
                f"Study split manifest was not found: {shared_manifest_path}"
            )
    else:
        _write_study_split_artifacts(study_dir, split_artifacts)
    return split_artifacts, split_sha256


def _run_condition_study(
    *,
    config_paths: list[Path],
    condition_set: ConfigConditionSet,
    execution_stage: ExecutionStage | None,
    verbose: bool,
    quiet: bool,
    resume_dir: Path | None,
) -> Path:
    log_verbosity = _resolve_log_verbosity(verbose=verbose, quiet=quiet)
    session_start = datetime.now(UTC)
    is_resume = resume_dir is not None
    study_dir = resume_dir if resume_dir is not None else _build_run_dir("study")
    assert study_dir is not None
    study_dir = study_dir.resolve()

    if is_resume:
        try:
            manifest_rows = load_condition_manifest(study_dir)
            validate_resume_manifest(manifest_rows, condition_set)
        except StudyError as exc:
            raise typer.BadParameter(str(exc)) from exc
        metadata_path = study_dir / "study_metadata.json"
        try:
            study_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(f"Failed to read study metadata: {metadata_path}") from exc
        study_metadata["status"] = "running"
        study_metadata["resume_time"] = session_start.isoformat()
    else:
        (study_dir / "conditions").mkdir(parents=True, exist_ok=True)
        source_config_path = study_dir / "source_config.yml"
        source_config_path.write_text(
            config_paths[0].read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        manifest_rows = new_condition_manifest(condition_set, study_dir=study_dir)
        write_condition_manifest(study_dir, manifest_rows)
        study_metadata = {
            "command": "run",
            "mode": "multi_condition",
            "status": "running",
            "start_time": session_start.isoformat(),
            "source_config": str(config_paths[0].resolve()),
            "archived_source_config": str(source_config_path),
            "condition_count": len(condition_set.conditions),
            "condition_dimensions": [
                dimension.dotted_path for dimension in condition_set.dimensions
            ],
        }
    write_config_differences(study_dir, condition_set)

    split_artifacts, split_sha256 = _study_split_artifacts(
        study_dir=study_dir,
        condition_set=condition_set,
        resume=is_resume,
    )
    stored_split_sha256 = study_metadata.get("split_fingerprint")
    if stored_split_sha256 is not None and stored_split_sha256 != split_sha256:
        raise typer.BadParameter(
            "The current config/data split fingerprint differs from the resumed study"
        )
    study_metadata["split_fingerprint"] = split_sha256
    _write_study_metadata(study_dir, study_metadata)

    condition_by_id = {
        condition.condition_id: condition for condition in condition_set.conditions
    }
    _progress_log(
        "run",
        f"Run multi-condition study ({len(manifest_rows)} conditions, study_dir={study_dir}).",
        start_time=session_start,
        log_verbosity=log_verbosity,
    )
    for row in manifest_rows:
        condition_id = str(row["condition_id"])
        condition = condition_by_id[condition_id]
        if condition_run_is_complete(row):
            row["status"] = "completed"
            write_condition_manifest(study_dir, manifest_rows)
            _progress_log(
                "run",
                f"Skip completed condition {condition.index}/{len(manifest_rows)} "
                f"({condition_id}).",
                start_time=session_start,
                log_verbosity=log_verbosity,
            )
            continue

        run_dir = prepare_condition_attempt(row, study_dir=study_dir)
        row["status"] = "running"
        row["started_at"] = datetime.now(UTC).isoformat()
        row["ended_at"] = None
        row["error"] = None
        write_condition_manifest(study_dir, manifest_rows)
        _progress_log(
            "run",
            f"Start condition {condition.index}/{len(manifest_rows)}: {condition.label}.",
            start_time=session_start,
            log_verbosity=log_verbosity,
        )
        try:
            completed_dir = _run_single(
                config_paths,
                execution_stage=execution_stage,
                verbose=verbose,
                quiet=quiet,
                resolved_override=condition.config,
                split_artifacts_override=split_artifacts,
                run_dir_override=run_dir,
                study_context={
                    "study_dir": str(study_dir),
                    "condition_index": condition.index,
                    "condition_id": condition.condition_id,
                    "condition_label": condition.label,
                },
            )
            metadata = json.loads(
                (completed_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            if metadata.get("split_fingerprint") != split_sha256:
                raise StudyError(
                    f"Condition {condition_id} did not use the shared study split"
                )
        except Exception as exc:
            row["status"] = "failed"
            row["ended_at"] = datetime.now(UTC).isoformat()
            row["error"] = str(exc)
            write_condition_manifest(study_dir, manifest_rows)
            study_metadata["status"] = "failed"
            study_metadata["end_time"] = datetime.now(UTC).isoformat()
            study_metadata["failed_condition_id"] = condition_id
            _write_study_metadata(study_dir, study_metadata)
            raise
        row["status"] = "completed"
        row["ended_at"] = datetime.now(UTC).isoformat()
        write_condition_manifest(study_dir, manifest_rows)

    try:
        report_artifacts = generate_study_report(study_dir, manifest_rows)
    except Exception as exc:
        study_metadata["status"] = "report_failed"
        study_metadata["end_time"] = datetime.now(UTC).isoformat()
        study_metadata["report_error"] = str(exc)
        _write_study_metadata(study_dir, study_metadata)
        if isinstance(exc, StudyError):
            raise typer.BadParameter(str(exc)) from exc
        raise

    end_time = datetime.now(UTC)
    study_metadata["status"] = "completed"
    study_metadata.pop("failed_condition_id", None)
    study_metadata.pop("report_error", None)
    study_metadata["end_time"] = end_time.isoformat()
    study_metadata["last_session_duration_sec"] = (end_time - session_start).total_seconds()
    study_metadata["condition_metrics_path"] = "tables/condition_metrics.tsv"
    study_metadata["pairwise_comparisons_path"] = "tables/pairwise_comparisons.tsv"
    if report_artifacts.training_group_sensitivity is not None:
        study_metadata["training_group_sensitivity_path"] = (
            "tables/training_group_sensitivity.tsv"
        )
    else:
        study_metadata.pop("training_group_sensitivity_path", None)
    study_metadata["config_differences_path"] = "config_differences.tsv"
    study_metadata["figure_paths"] = [
        str(path.relative_to(study_dir)) for path in report_artifacts.figure_paths
    ]
    _write_study_metadata(study_dir, study_metadata)
    typer.echo(
        f"Wrote multi-condition study at {study_dir} "
        "(condition_manifest.tsv, config_differences.tsv, split/tables/, "
        "conditions/, tables/, figures/)."
    )
    return study_dir


@app.command()
def run(
    config: ConfigPathsArg,
    execution_stage: ExecutionStageArg = None,
    verbose: VerboseArg = False,
    quiet: QuietArg = False,
    resume: ResumeStudyArg = None,
) -> None:
    """Run training/evaluation pipeline."""
    config_paths = _normalize_config_paths(config)
    try:
        resolved = load_and_resolve_config(
            config_paths,
            execution_stage_override=execution_stage,
            allow_empty=False,
        )
    except ConfigError as single_config_error:
        try:
            contains_conditions = has_condition_dimensions(
                config_paths,
                execution_stage_override=execution_stage,
            )
        except ConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if not contains_conditions:
            raise typer.BadParameter(str(single_config_error)) from single_config_error
        try:
            condition_set = load_config_conditions(
                config_paths,
                execution_stage_override=execution_stage,
            )
        except ConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        if resume is not None:
            raise typer.BadParameter("--resume requires a config with multiple conditions")
        _run_single(
            config_paths,
            execution_stage=execution_stage,
            verbose=verbose,
            quiet=quiet,
            resolved_override=resolved,
        )
        return

    if len(condition_set.conditions) == 1:
        if resume is not None:
            raise typer.BadParameter("--resume requires at least two generated conditions")
        _run_single(
            config_paths,
            execution_stage=execution_stage,
            verbose=verbose,
            quiet=quiet,
            resolved_override=condition_set.conditions[0].config,
        )
        return

    _run_condition_study(
        config_paths=config_paths,
        condition_set=condition_set,
        execution_stage=execution_stage,
        verbose=verbose,
        quiet=quiet,
        resume_dir=resume,
    )


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


@app.command("metadata")
def metadata_command(
    species_trait: Annotated[
        Path,
        typer.Option(
            "--species-trait",
            file_okay=True,
            dir_okay=False,
            help="Input TSV containing species and binary trait columns.",
        ),
    ] = Path("species_trait.tsv"),
    species_taxid: Annotated[
        Path | None,
        typer.Option(
            "--species-taxid",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Optional TSV containing species and NCBI taxid columns for tree retrieval.",
        ),
    ] = None,
    species_taxid_out: Annotated[
        Path | None,
        typer.Option(
            "--species-taxid-out",
            file_okay=True,
            dir_okay=False,
            help=(
                "Output generated species/taxid TSV when --species-taxid is omitted. "
                "Defaults to species_taxid.tsv next to --out when taxon rank annotations "
                "or blocks need it."
            ),
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=True,
            dir_okay=False,
            help="Output PhenoRadar metadata TSV with contrast_pair_id assignments.",
        ),
    ] = Path("species_metadata.tsv"),
    tree_in: Annotated[
        Path | None,
        typer.Option(
            "--tree-in",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Existing Newick tree to use for group assignment. Skips NCBI tree retrieval.",
        ),
    ] = None,
    tree_out: Annotated[
        Path,
        typer.Option(
            "--tree-out",
            file_okay=True,
            dir_okay=False,
            help="Output Newick tree path generated from NCBI Taxonomy via nwkit.",
        ),
    ] = Path("ncbi_tree.nwk"),
    species_col: Annotated[
        str,
        typer.Option(
            "--species-col",
            help="Species column name in species_trait.tsv and species_taxid.tsv.",
        ),
    ] = "species",
    taxid_col: Annotated[
        str,
        typer.Option(
            "--taxid-col",
            help="Taxid column name in species_taxid.tsv.",
        ),
    ] = "taxid",
    trait_col: Annotated[
        str,
        typer.Option(
            "--trait-col",
            help="Binary trait column name in species_trait.tsv and output metadata.",
        ),
    ] = "C4",
    contrast_pair_col: Annotated[
        str,
        typer.Option(
            "--contrast-pair-col",
            help="Output contrast-pair column name.",
        ),
    ] = "contrast_pair_id",
    contrast_pair_test_holdout_col: Annotated[
        str,
        typer.Option(
            "--contrast-pair-test-holdout-col",
            help="Output column marking labeled species held out from contrast-pair CV.",
        ),
    ] = "contrast_pair_test_holdout",
    taxon_annotation_rank: Annotated[
        list[str] | None,
        typer.Option(
            "--taxon-annotation-rank",
            help=(
                "NCBI taxonomy rank to emit as annotation columns. Repeat for multiple "
                "ranks. Defaults to order and family."
            ),
        ),
    ] = None,
    taxon_block_rank: Annotated[
        list[str] | None,
        typer.Option(
            "--taxon-block-rank",
            help=(
                "NCBI taxonomy rank to emit as a split block. Repeat for multiple "
                "ranks, for example --taxon-block-rank family --taxon-block-rank order."
            ),
        ),
    ] = None,
    taxon_block_min_species_per_label: Annotated[
        int,
        typer.Option(
            "--taxon-block-min-species-per-label",
            min=1,
            help="Minimum labeled species per trait value required for a taxon block to enter CV.",
        ),
    ] = 1,
    taxon_block_mixed_test_fraction: Annotated[
        float,
        typer.Option(
            "--taxon-block-mixed-test-fraction",
            min=0.0,
            max=0.999999,
            help="Fraction of mixed-label taxon blocks to reserve as external test blocks.",
        ),
    ] = 0.0,
    taxon_block_mixed_test_seed: Annotated[
        int,
        typer.Option(
            "--taxon-block-mixed-test-seed",
            help="Random seed for selecting mixed-label taxon blocks held out for test.",
        ),
    ] = 42,
    ncbi_taxonomy_db: Annotated[
        Path | None,
        typer.Option(
            "--ncbi-taxonomy-db",
            file_okay=True,
            dir_okay=False,
            help="Optional ete4 NCBI taxonomy SQLite database path.",
        ),
    ] = None,
    nwkit_bin: Annotated[
        str,
        typer.Option(
            "--nwkit-bin",
            help="nwkit executable path.",
        ),
    ] = "nwkit",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing tree or metadata outputs.",
        ),
    ] = False,
    write_metadata: Annotated[
        bool,
        typer.Option(
            "--write-metadata/--tree-only",
            help="Write species_metadata.tsv after tree retrieval.",
        ),
    ] = True,
    verbose: VerboseArg = False,
    quiet: QuietArg = False,
) -> None:
    """Fetch an NCBI tree and assign contrast-pair metadata from species_trait.tsv."""
    start_time = datetime.now(UTC)
    log_verbosity = _resolve_log_verbosity(verbose=verbose, quiet=quiet)
    if tree_in is not None and not write_metadata:
        raise typer.BadParameter("`--tree-in` with `--tree-only` has no work to do.")
    if species_taxid is not None and species_taxid_out is not None:
        raise typer.BadParameter("`--species-taxid-out` cannot be used with `--species-taxid`.")

    resolved_species_taxid = species_taxid
    generated_taxid_result = None
    taxon_annotation_rank_values = (
        ["order", "family"] if taxon_annotation_rank is None else list(taxon_annotation_rank)
    )
    should_generate_taxid = species_taxid is None and (
        species_taxid_out is not None
        or (write_metadata and bool(taxon_annotation_rank_values or taxon_block_rank))
    )

    try:
        if should_generate_taxid:
            taxid_out = species_taxid_out or out.with_name("species_taxid.tsv")
            _progress_log(
                "metadata",
                "Resolve NCBI taxids for species metadata.",
                start_time=start_time,
                log_verbosity=log_verbosity,
            )
            generated_taxid_result = build_species_taxid_tsv(
                species_trait,
                taxid_out,
                species_col=species_col,
                taxid_col=taxid_col,
                ncbi_taxonomy_db=ncbi_taxonomy_db,
                overwrite=force,
            )
            resolved_species_taxid = generated_taxid_result.taxid_path
            typer.echo(
                f"Wrote species taxid TSV: {generated_taxid_result.taxid_path} "
                f"(resolved_species={generated_taxid_result.resolved_species_count}, "
                f"unresolved_species={generated_taxid_result.unresolved_species_count})."
            )
    except MetadataError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _progress_log(
        "metadata",
        "Resolve tree for metadata preparation.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )
    try:
        if tree_in is None:
            tree_result = fetch_ncbi_tree(
                species_trait,
                tree_out,
                species_taxid_path=resolved_species_taxid,
                species_col=species_col,
                taxid_col=taxid_col,
                nwkit_bin=nwkit_bin,
                overwrite=force,
            )
            tree_path = tree_result.tree_path
            typer.echo(
                f"Wrote NCBI taxonomy tree: {tree_result.tree_path} "
                f"(species={tree_result.species_count})."
            )
        else:
            tree_path = tree_in
            typer.echo(f"Using existing tree: {tree_path}")

        if write_metadata:
            _progress_log(
                "metadata",
                "Assign contrast-pair metadata with nwkit skim.",
                start_time=start_time,
                log_verbosity=log_verbosity,
            )
            metadata_result = build_species_metadata_from_skim(
                species_trait,
                tree_path,
                out,
                species_taxid_path=resolved_species_taxid,
                species_col=species_col,
                taxid_col=taxid_col,
                trait_col=trait_col,
                contrast_pair_col=contrast_pair_col,
                contrast_pair_test_holdout_col=contrast_pair_test_holdout_col,
                taxon_annotation_ranks=taxon_annotation_rank_values,
                taxon_block_ranks=taxon_block_rank,
                taxon_block_min_species_per_label=taxon_block_min_species_per_label,
                taxon_block_mixed_test_fraction=taxon_block_mixed_test_fraction,
                taxon_block_mixed_test_seed=taxon_block_mixed_test_seed,
                ncbi_taxonomy_db=ncbi_taxonomy_db,
                nwkit_bin=nwkit_bin,
                overwrite=force,
            )
    except MetadataError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if write_metadata:
        taxon_summary = ""
        if metadata_result.taxon_block_counts:
            taxon_summary = ", taxon_blocks=["
            taxon_summary += "; ".join(
                f"{rank}:blocks={metadata_result.taxon_block_counts[rank]},"
                f"test_holdouts={metadata_result.taxon_block_test_holdout_counts[rank]},"
                f"excluded={metadata_result.taxon_block_exclude_counts[rank]}"
                for rank in sorted(metadata_result.taxon_block_counts)
            )
            taxon_summary += "]"
        typer.echo(
            f"Wrote species metadata: {metadata_result.metadata_path} "
            f"(species={metadata_result.species_count}, "
            f"grouped_species={metadata_result.grouped_species_count}, "
            f"contrast_pairs={metadata_result.contrast_pair_count}, "
            f"contrast_pair_test_holdouts="
            f"{metadata_result.contrast_pair_test_holdout_count}, "
            f"tree_missing_species={metadata_result.tree_missing_species_count}"
            f"{taxon_summary})."
        )
    _progress_log(
        "metadata",
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
                "Optional base URL containing c4_tiny dataset files. "
                "By default the dataset bundled with PhenoRadar is copied; "
                "an external source can also be set via PHENORADAR_TESTDATA_BASE_URL."
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
    """Install compact bundled test data, optionally from a custom base URL."""
    start_time = datetime.now(UTC)
    log_verbosity = _resolve_log_verbosity(verbose=verbose, quiet=quiet)
    _progress_log(
        "dataset",
        "Install compact test data files.",
        start_time=start_time,
        log_verbosity=log_verbosity,
    )
    try:
        resolved_base_url = resolve_c4_tiny_base_url(base_url)
        written_paths = fetch_c4_tiny_test_data(
            out,
            base_url=resolved_base_url,
            overwrite=force,
        )
    except TestDataError as exc:
        raise typer.BadParameter(str(exc)) from exc

    file_names = ", ".join(path.name for path in written_paths)
    resolved_source = resolved_base_url or BUNDLED_C4_TINY_SOURCE
    typer.echo(f"Installed test data into {out} from {resolved_source} ({file_names}).")
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
        pred_predict = pred_predict.with_columns(pl.lit(None, dtype=pl.Int64).alias("true_label"))
    pred_predict = pred_predict.select(
        [
            name
            for name in [
                "species",
                "true_label",
                "prob",
                "pred_label_fixed_threshold",
                "uncertainty_std",
            ]
            if name in pred_predict.columns
        ]
    )
    _emit_predict_summary(pred_predict, start_time=start_time, log_verbosity=log_verbosity)

    _log("Write prediction artifacts.")
    run_dir = _build_run_dir("predict")
    inference_tables_dir = _stage_tables_dir(run_dir, "inference")
    write_resolved_config(resolved, run_dir / "resolved_config.yml")
    pred_predict.write_csv(
        inference_tables_dir / "prediction_inference.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    if pred_predict.height > 0:
        predict_warnings.extend(
            _write_group_summary_artifacts(
                run_dir=run_dir,
                stage="inference",
                predictions=pred_predict,
                source_table_name="prediction_inference.tsv",
                config=resolved,
            )
        )
    _log("Generate prediction figures.")
    tree_path = getattr(resolved.data, "tree_path", None)
    try:
        write_predict_figures(
            run_dir=run_dir,
            pred_predict=pred_predict,
            require_uncertainty=len(bundle.models) > 1,
        )
    except FigureError as exc:
        raise typer.BadParameter(str(exc)) from exc
    contrast_pair_col = resolved.data.contrast_pair_col
    if tree_path is not None and contrast_pair_col is not None:
        try:
            tree_warnings = write_predict_tree_prediction_artifacts(
                run_dir=run_dir,
                tree_path=Path(tree_path),
                metadata_path=Path(resolved.data.metadata_path),
                species_col=resolved.data.species_col,
                trait_col=resolved.data.trait_col,
                group_col=contrast_pair_col,
                pred_predict=pred_predict,
            )
        except TreePredictionError as exc:
            raise typer.BadParameter(str(exc)) from exc
        predict_warnings.extend(tree_warnings)
    elif tree_path is not None:
        predict_warnings.append(
            "Skipped tree prediction artifacts because data.contrast_pair_col is null."
        )
    end_time = datetime.now(UTC)
    _log("Collect provenance metadata.")
    try:
        input_files = collect_input_files(
            [
                config_path,
                Path(resolved.data.metadata_path),
                Path(resolved.data.tpm_path),
                *([] if tree_path is None else [Path(tree_path)]),
                model_bundle / "bundle_manifest.json",
            ]
        )
        payload_sha = bundle_payload_sha256(model_bundle)
    except ProvenanceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    build_meta = phenoradar_build_snapshot()
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
            "bundle_source_provenance_schema_version": bundle.manifest.get(
                "source_provenance_schema_version"
            ),
            "bundle_source_phenoradar_version": bundle.manifest.get(
                "source_phenoradar_version"
            ),
            "bundle_source_git_commit": bundle.manifest.get("source_git_commit"),
            "input_files": input_files,
            "environment": environment,
            "warnings": predict_warnings,
            **build_meta,
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
        "(resolved_config.yml, inference/tables/prediction_inference.tsv, "
        "inference/figures/, "
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
    allow_mixed_experiments: Annotated[
        bool,
        typer.Option(
            "--allow-mixed-experiments",
            help="Allow ranking runs with different or unknown experiment fingerprints.",
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
                allow_mixed_experiments=allow_mixed_experiments,
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
