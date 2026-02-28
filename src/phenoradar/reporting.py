"""Cross-run report aggregation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml

from phenoradar.figures import FigureError, write_report_figures

PrimaryMetric = Literal["mcc", "balanced_accuracy", "roc_auc", "pr_auc", "brier"]
AggregateScope = Literal["macro", "micro"]
IncludeStage = Literal["cv_only", "full_run", "predict", "all"]
OutputFormat = Literal["tsv", "md", "html", "json"]


class ReportError(ValueError):
    """Raised when report aggregation cannot proceed."""


@dataclass(frozen=True)
class ReportOptions:
    """Options that control report aggregation and ranking."""

    primary_metric: PrimaryMetric
    aggregate_scope: AggregateScope
    include_stage: IncludeStage
    output_format: OutputFormat
    strict: bool
    run_glob: str
    latest: int | None


def _json_load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ReportError(f"JSON root must be an object: {path}")
    return payload


def _yaml_load(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"Required file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ReportError(f"Invalid YAML file: {path}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ReportError(f"YAML root must be a mapping: {path}")
    return payload


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _collect_run_dirs(
    *,
    run_dirs: list[Path],
    runs_root: Path | None,
    run_glob: str,
    latest: int | None,
) -> list[Path]:
    if run_dirs:
        resolved = sorted({path for path in run_dirs if path.is_dir()})
    else:
        if runs_root is None:
            raise ReportError("Either --run-dir or --runs-root must be provided")
        if not runs_root.exists():
            raise ReportError(f"runs-root does not exist: {runs_root}")
        resolved = sorted(path for path in runs_root.glob(run_glob) if path.is_dir())
    if not resolved:
        raise ReportError("No run directories were found for report aggregation")
    if latest is not None:
        if latest < 1:
            raise ReportError("--latest must be >= 1 when specified")
        resolved = resolved[-latest:]
    return resolved


def _load_metric_value(
    *,
    metrics_path: Path,
    aggregate_scope: AggregateScope,
    primary_metric: PrimaryMetric,
) -> float | None:
    metrics = pl.read_csv(metrics_path, separator="\t")
    if not {"aggregate_scope", "fold_id", "metric", "metric_value"}.issubset(metrics.columns):
        raise ReportError(f"metrics_cv.tsv has invalid schema: {metrics_path}")
    values = (
        metrics.filter(
            (pl.col("aggregate_scope") == aggregate_scope)
            & (pl.col("fold_id") == "NA")
            & (pl.col("metric") == primary_metric)
        )
        .select("metric_value")
        .to_series()
        .to_list()
    )
    if len(values) != 1:
        return None
    metric = _float_or_none(values[0])
    if metric is None or metric != metric:
        return None
    return metric


def _record_warning(
    warning_rows: list[dict[str, str]],
    *,
    run_id: str,
    run_dir: Path,
    warning_type: str,
    message: str,
) -> None:
    warning_rows.append(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "warning_type": warning_type,
            "message": message,
        }
    )


def _write_narrative(
    *,
    output_dir: Path,
    output_format: OutputFormat,
    options: ReportOptions,
    selected_run_count: int,
    ranked_run_count: int,
) -> None:
    if output_format == "md":
        text = (
            "# PhenoRadar Report\n\n"
            f"- Primary metric: `{options.primary_metric}` ({options.aggregate_scope})\n"
            f"- Selected runs: `{selected_run_count}`\n"
            f"- Ranked runs: `{ranked_run_count}`\n"
        )
        (output_dir / "report.md").write_text(text, encoding="utf-8")
    elif output_format == "html":
        html = (
            "<!doctype html>\n"
            "<html><head><meta charset=\"utf-8\"><title>PhenoRadar Report</title></head><body>\n"
            "<h1>PhenoRadar Report</h1>\n"
            f"<p>Primary metric: <code>{options.primary_metric}</code> "
            f"({options.aggregate_scope})</p>\n"
            f"<p>Selected runs: <code>{selected_run_count}</code></p>\n"
            f"<p>Ranked runs: <code>{ranked_run_count}</code></p>\n"
            "</body></html>\n"
        )
        (output_dir / "report.html").write_text(html, encoding="utf-8")


def generate_report(
    *,
    run_dirs: list[Path],
    runs_root: Path | None,
    run_glob: str,
    latest: int | None,
    options: ReportOptions,
    output_dir: Path,
) -> None:
    """Aggregate selected runs and write report artifacts."""
    selected_dirs = _collect_run_dirs(
        run_dirs=run_dirs,
        runs_root=runs_root,
        run_glob=run_glob,
        latest=latest,
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    run_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, str]] = []
    skipped_runs: list[dict[str, str]] = []

    for run_dir in selected_dirs:
        run_id = run_dir.name
        metadata_path = run_dir / "run_metadata.json"
        config_path = run_dir / "resolved_config.yml"

        if not metadata_path.exists():
            message = f"Missing required artifact: {metadata_path.name}"
            if options.strict:
                raise ReportError(f"{run_id}: {message}")
            skipped_runs.append({"run_id": run_id, "run_dir": str(run_dir), "reason": message})
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=run_dir,
                warning_type="missing_artifact",
                message=message,
            )
            continue

        if not config_path.exists():
            message = f"Missing required artifact: {config_path.name}"
            if options.strict:
                raise ReportError(f"{run_id}: {message}")
            skipped_runs.append({"run_id": run_id, "run_dir": str(run_dir), "reason": message})
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=run_dir,
                warning_type="missing_artifact",
                message=message,
            )
            continue

        try:
            metadata = _json_load(metadata_path)
        except ReportError as exc:
            if options.strict:
                raise ReportError(f"{run_id}: {exc}") from exc
            skipped_runs.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "reason": str(exc),
                }
            )
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=run_dir,
                warning_type="invalid_artifact",
                message=str(exc),
            )
            continue
        try:
            _yaml_load(config_path)
        except ReportError as exc:
            if options.strict:
                raise ReportError(f"{run_id}: {exc}") from exc
            skipped_runs.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "reason": str(exc),
                }
            )
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=run_dir,
                warning_type="invalid_artifact",
                message=str(exc),
            )
            continue

        stage = str(metadata.get("execution_stage", "unknown"))
        if options.include_stage != "all" and stage != options.include_stage:
            skipped_runs.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "reason": f"Filtered by include-stage ({options.include_stage})",
                }
            )
            continue

        command = str(metadata.get("command", "unknown"))
        status = str(metadata.get("status", "unknown"))
        start_time = str(metadata.get("start_time", ""))
        end_time = str(metadata.get("end_time", ""))
        duration_sec = _float_or_none(metadata.get("duration_sec"))

        metric_value: float | None = None
        metrics_path = run_dir / "metrics_cv.tsv"
        if metrics_path.exists():
            try:
                metric_value = _load_metric_value(
                    metrics_path=metrics_path,
                    aggregate_scope=options.aggregate_scope,
                    primary_metric=options.primary_metric,
                )
            except ReportError as exc:
                if options.strict:
                    raise
                _record_warning(
                    warning_rows,
                    run_id=run_id,
                    run_dir=run_dir,
                    warning_type="invalid_metrics",
                    message=str(exc),
                )
        elif stage != "predict":
            message = "metrics_cv.tsv is missing for a non-predict run"
            if options.strict:
                raise ReportError(f"{run_id}: {message}")
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=run_dir,
                warning_type="missing_metrics",
                message=message,
            )

        raw_warnings = metadata.get("warnings")
        if isinstance(raw_warnings, list):
            for item in raw_warnings:
                _record_warning(
                    warning_rows,
                    run_id=run_id,
                    run_dir=run_dir,
                    warning_type="run_warning",
                    message=str(item),
                )

        run_rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "command": command,
                "execution_stage": stage,
                "status": status,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration_sec,
                "primary_metric": options.primary_metric,
                "aggregate_scope": options.aggregate_scope,
                "metric_value": metric_value,
            }
        )
        if metric_value is not None:
            ranking_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "execution_stage": stage,
                    "start_time": start_time,
                    "metric_name": options.primary_metric,
                    "aggregate_scope": options.aggregate_scope,
                    "metric_value": metric_value,
                }
            )

    run_rows_sorted = sorted(
        run_rows,
        key=lambda row: (
            str(row["run_id"]),
        ),
    )
    ranking_sorted = sorted(
        ranking_rows,
        key=lambda row: (
            -float(row["metric_value"]),
            str(row["start_time"]),
            str(row["run_id"]),
        ),
    )
    for rank_idx, row in enumerate(ranking_sorted, start=1):
        row["rank"] = rank_idx

    report_runs = (
        pl.DataFrame(run_rows_sorted).sort("run_id")
        if run_rows_sorted
        else pl.DataFrame(
            schema={
                "run_id": pl.String,
                "run_dir": pl.String,
                "command": pl.String,
                "execution_stage": pl.String,
                "status": pl.String,
                "start_time": pl.String,
                "end_time": pl.String,
                "duration_sec": pl.Float64,
                "primary_metric": pl.String,
                "aggregate_scope": pl.String,
                "metric_value": pl.Float64,
            }
        )
    )
    report_ranking = (
        pl.DataFrame(ranking_sorted).sort("rank")
        if ranking_sorted
        else pl.DataFrame(
            schema={
                "rank": pl.Int64,
                "run_id": pl.String,
                "run_dir": pl.String,
                "execution_stage": pl.String,
                "start_time": pl.String,
                "metric_name": pl.String,
                "aggregate_scope": pl.String,
                "metric_value": pl.Float64,
            }
        )
    )
    report_warnings = (
        pl.DataFrame(warning_rows).sort(["run_id", "warning_type", "message"])
        if warning_rows
        else pl.DataFrame(
            schema={
                "run_id": pl.String,
                "run_dir": pl.String,
                "warning_type": pl.String,
                "message": pl.String,
            }
        )
    )

    report_runs.write_csv(output_dir / "report_runs.tsv", separator="\t", float_precision=8)
    report_ranking.write_csv(output_dir / "report_ranking.tsv", separator="\t", float_precision=8)
    report_warnings.write_csv(output_dir / "report_warnings.tsv", separator="\t")

    manifest = {
        "report_options": {
            "primary_metric": options.primary_metric,
            "aggregate_scope": options.aggregate_scope,
            "include_stage": options.include_stage,
            "output_format": options.output_format,
            "strict": options.strict,
            "glob": options.run_glob,
            "latest": options.latest,
        },
        "selected_run_dirs": [str(path) for path in selected_dirs],
        "included_runs": [row["run_id"] for row in run_rows_sorted],
        "skipped_runs": skipped_runs,
        "ranked_run_count": len(ranking_sorted),
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if options.output_format == "json":
        report_json = {
            "manifest": manifest,
            "runs": run_rows_sorted,
            "ranking": ranking_sorted,
            "warnings": warning_rows,
        }
        (output_dir / "report.json").write_text(
            json.dumps(report_json, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    try:
        write_report_figures(
            report_dir=output_dir,
            report_runs=report_runs,
            report_ranking=report_ranking,
        )
    except FigureError as exc:
        raise ReportError(str(exc)) from exc

    _write_narrative(
        output_dir=output_dir,
        output_format=options.output_format,
        options=options,
        selected_run_count=len(run_rows_sorted),
        ranked_run_count=len(ranking_sorted),
    )
