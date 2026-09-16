"""Cross-run report aggregation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml

from phenoradar.figures import FigureError, write_report_figures
from phenoradar.metrics import metric_contract, metric_direction, metric_sort_value
from phenoradar.provenance import phenoradar_build_snapshot

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
    allow_mixed_experiments: bool = False


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


def _run_metrics_path(run_dir: Path) -> Path:
    staged_path = run_dir / "cv" / "tables" / "metrics_cv.tsv"
    if staged_path.exists():
        return staged_path
    return run_dir / "metrics_cv.tsv"


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
    metrics = pl.read_csv(
        metrics_path,
        separator="\t",
        schema_overrides={"fold_id": pl.String},
    )
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
    if metric is None or not isfinite(metric):
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


def _fingerprint_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        return None
    return normalized


def _fingerprint_schema_version_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return int(value)


def _nonempty_string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _run_metric_contract_or_none(
    metadata: dict[str, Any], metric_name: str
) -> dict[str, Any] | None:
    evaluation_contract = metadata.get("evaluation_contract")
    if not isinstance(evaluation_contract, dict):
        return None
    metric_registry = evaluation_contract.get("metric_contract")
    if not isinstance(metric_registry, dict):
        return None
    version = metric_registry.get("metric_contract_version")
    metrics = metric_registry.get("metrics")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or not isinstance(metrics, dict)
    ):
        return None
    contract = metrics.get(metric_name)
    if not isinstance(contract, dict):
        return None
    display_name = contract.get("display_name")
    implementation = contract.get("implementation")
    threshold_name = contract.get("threshold_name")
    threshold_value = contract.get("threshold_value")
    if not isinstance(display_name, str) or not isinstance(implementation, str):
        return None
    if threshold_name is not None and not isinstance(threshold_name, str):
        return None
    if isinstance(threshold_value, bool) or (
        threshold_value is not None and not isinstance(threshold_value, (int, float))
    ):
        return None
    return {
        "metric_contract_version": version,
        "display_name": display_name,
        "implementation": implementation,
        "threshold_name": threshold_name,
        "threshold_value": None if threshold_value is None else float(threshold_value),
    }


def _validate_experiment_compatibility(
    *,
    run_rows: list[dict[str, Any]],
    warning_rows: list[dict[str, str]],
    options: ReportOptions,
) -> dict[str, Any]:
    comparable_rows = [row for row in run_rows if row["metric_value"] is not None]
    if not comparable_rows:
        return {
            "status": "not_applicable",
            "mixed": False,
            "experiment_fingerprints": [],
            "unknown_run_ids": [],
        }

    known_fingerprints: set[str] = set()
    unknown_run_ids: list[str] = []
    for row in comparable_rows:
        required_values = (
            row["fingerprint_schema_version"],
            row["dataset_fingerprint"],
            row["split_fingerprint"],
            row["experiment_fingerprint"],
        )
        if any(value is None for value in required_values):
            run_id = str(row["run_id"])
            if options.strict:
                raise ReportError(
                    f"{run_id}: missing or invalid experiment fingerprint metadata"
                )
            unknown_run_ids.append(run_id)
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=Path(str(row["run_dir"])),
                warning_type="missing_experiment_fingerprint",
                message=(
                    "Run predates experiment fingerprints or contains invalid fingerprint "
                    "metadata; comparability cannot be verified."
                ),
            )
            continue
        known_fingerprints.add(str(row["experiment_fingerprint"]))

    known_sorted = sorted(known_fingerprints)
    unknown_sorted = sorted(unknown_run_ids)
    mixed = len(known_sorted) > 1 or (bool(known_sorted) and bool(unknown_sorted))
    if mixed and not options.allow_mixed_experiments:
        raise ReportError(
            "Ranked runs do not share one verifiable experiment fingerprint. "
            "Use --allow-mixed-experiments to override this comparison guard."
        )
    if mixed:
        message = (
            "Report combines runs with different or unknown experiment fingerprints because "
            "--allow-mixed-experiments was specified."
        )
        for row in comparable_rows:
            _record_warning(
                warning_rows,
                run_id=str(row["run_id"]),
                run_dir=Path(str(row["run_dir"])),
                warning_type="mixed_experiments",
                message=message,
            )

    if mixed:
        status = "mixed_override"
    elif unknown_sorted:
        status = "legacy_unverified"
    else:
        status = "verified"
    return {
        "status": status,
        "mixed": mixed,
        "experiment_fingerprints": known_sorted,
        "unknown_run_ids": unknown_sorted,
    }


def _validate_software_compatibility(
    *,
    run_rows: list[dict[str, Any]],
    warning_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not run_rows:
        return {
            "status": "not_applicable",
            "mixed": False,
            "phenoradar_versions": [],
            "unknown_run_ids": [],
            "dirty_run_ids": [],
        }

    known_versions: set[str] = set()
    unknown_run_ids: list[str] = []
    dirty_run_ids: list[str] = []
    for row in run_rows:
        run_id = str(row["run_id"])
        run_dir = Path(str(row["run_dir"]))
        version = row["phenoradar_version"]
        if version is None:
            unknown_run_ids.append(run_id)
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=run_dir,
                warning_type="missing_phenoradar_version",
                message=(
                    "Run predates PhenoRadar software provenance or contains an invalid "
                    "phenoradar_version; software compatibility cannot be verified."
                ),
            )
        else:
            known_versions.add(str(version))

        if row["git_dirty"] is True:
            dirty_run_ids.append(run_id)
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=run_dir,
                warning_type="dirty_phenoradar_build",
                message=(
                    "Run used a PhenoRadar source checkout with uncommitted or untracked "
                    "changes; inspect its Git commit and worktree patch checksum."
                ),
            )

    known_sorted = sorted(known_versions)
    unknown_sorted = sorted(unknown_run_ids)
    dirty_sorted = sorted(dirty_run_ids)
    mixed = len(known_sorted) > 1
    if mixed:
        versions_label = ", ".join(known_sorted)
        message = f"Report combines PhenoRadar versions: {versions_label}."
        for row in run_rows:
            _record_warning(
                warning_rows,
                run_id=str(row["run_id"]),
                run_dir=Path(str(row["run_dir"])),
                warning_type="mixed_phenoradar_versions",
                message=message,
            )

    if mixed:
        status = "mixed"
    elif unknown_sorted:
        status = "legacy_unverified"
    else:
        status = "verified"
    return {
        "status": status,
        "mixed": mixed,
        "phenoradar_versions": known_sorted,
        "unknown_run_ids": unknown_sorted,
        "dirty_run_ids": dirty_sorted,
    }


def _write_narrative(
    *,
    output_dir: Path,
    output_format: OutputFormat,
    options: ReportOptions,
    selected_run_count: int,
    ranked_run_count: int,
) -> None:
    primary_contract = metric_contract(options.primary_metric)
    direction_label = (
        "higher is better"
        if metric_direction(options.primary_metric) == "maximize"
        else "lower is better"
    )
    if output_format == "md":
        text = (
            "# PhenoRadar Report\n\n"
            f"- Primary metric: `{options.primary_metric}` ({options.aggregate_scope})\n"
            f"- Metric definition: `{primary_contract['display_name']}` via "
            f"`{primary_contract['implementation']}`\n"
            f"- Metric direction: `{direction_label}`\n"
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
            f"<p>Metric definition: <code>{primary_contract['display_name']}</code> via "
            f"<code>{primary_contract['implementation']}</code></p>\n"
            f"<p>Metric direction: <code>{direction_label}</code></p>\n"
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
    primary_contract = metric_contract(options.primary_metric)

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
        provenance_schema_version = _fingerprint_schema_version_or_none(
            metadata.get("provenance_schema_version")
        )
        phenoradar_version = _nonempty_string_or_none(metadata.get("phenoradar_version"))
        phenoradar_install_type = _nonempty_string_or_none(
            metadata.get("phenoradar_install_type")
        )
        git_source = _nonempty_string_or_none(metadata.get("git_source"))
        git_commit = _nonempty_string_or_none(metadata.get("git_commit"))
        git_dirty = _bool_or_none(metadata.get("git_dirty"))
        git_worktree_patch_sha256 = _fingerprint_or_none(
            metadata.get("git_worktree_patch_sha256")
        )
        fingerprint_schema_version = _fingerprint_schema_version_or_none(
            metadata.get("fingerprint_schema_version")
        )
        dataset_sha256 = _fingerprint_or_none(metadata.get("dataset_fingerprint"))
        split_sha256 = _fingerprint_or_none(metadata.get("split_fingerprint"))
        experiment_sha256 = _fingerprint_or_none(metadata.get("experiment_fingerprint"))
        run_primary_contract = _run_metric_contract_or_none(metadata, options.primary_metric)

        metric_value: float | None = None
        metrics_path = _run_metrics_path(run_dir)
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

        if metric_value is not None and run_primary_contract is None:
            message = (
                f"Missing or invalid evaluation contract for metric {options.primary_metric}"
            )
            if options.strict:
                raise ReportError(f"{run_id}: {message}")
            _record_warning(
                warning_rows,
                run_id=run_id,
                run_dir=run_dir,
                warning_type="missing_metric_contract",
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
                "provenance_schema_version": provenance_schema_version,
                "phenoradar_version": phenoradar_version,
                "phenoradar_install_type": phenoradar_install_type,
                "git_source": git_source,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "git_worktree_patch_sha256": git_worktree_patch_sha256,
                "primary_metric": options.primary_metric,
                "metric_contract_version": (
                    None
                    if run_primary_contract is None
                    else run_primary_contract["metric_contract_version"]
                ),
                "metric_display_name": (
                    None if run_primary_contract is None else run_primary_contract["display_name"]
                ),
                "metric_implementation": (
                    None
                    if run_primary_contract is None
                    else run_primary_contract["implementation"]
                ),
                "metric_threshold_name": (
                    None
                    if run_primary_contract is None
                    else run_primary_contract["threshold_name"]
                ),
                "metric_threshold_value": (
                    None
                    if run_primary_contract is None
                    else run_primary_contract["threshold_value"]
                ),
                "aggregate_scope": options.aggregate_scope,
                "metric_value": metric_value,
                "fingerprint_schema_version": fingerprint_schema_version,
                "dataset_fingerprint": dataset_sha256,
                "split_fingerprint": split_sha256,
                "experiment_fingerprint": experiment_sha256,
            }
        )
        if metric_value is not None:
            ranking_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "execution_stage": stage,
                    "start_time": start_time,
                    "provenance_schema_version": provenance_schema_version,
                    "phenoradar_version": phenoradar_version,
                    "phenoradar_install_type": phenoradar_install_type,
                    "git_source": git_source,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "git_worktree_patch_sha256": git_worktree_patch_sha256,
                    "metric_name": options.primary_metric,
                    "metric_contract_version": (
                        None
                        if run_primary_contract is None
                        else run_primary_contract["metric_contract_version"]
                    ),
                    "metric_display_name": (
                        None
                        if run_primary_contract is None
                        else run_primary_contract["display_name"]
                    ),
                    "metric_implementation": (
                        None
                        if run_primary_contract is None
                        else run_primary_contract["implementation"]
                    ),
                    "metric_threshold_name": (
                        None
                        if run_primary_contract is None
                        else run_primary_contract["threshold_name"]
                    ),
                    "metric_threshold_value": (
                        None
                        if run_primary_contract is None
                        else run_primary_contract["threshold_value"]
                    ),
                    "aggregate_scope": options.aggregate_scope,
                    "metric_value": metric_value,
                    "fingerprint_schema_version": fingerprint_schema_version,
                    "dataset_fingerprint": dataset_sha256,
                    "split_fingerprint": split_sha256,
                    "experiment_fingerprint": experiment_sha256,
                }
            )

    compatibility = _validate_experiment_compatibility(
        run_rows=run_rows,
        warning_rows=warning_rows,
        options=options,
    )
    software_compatibility = _validate_software_compatibility(
        run_rows=run_rows,
        warning_rows=warning_rows,
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
            metric_sort_value(options.primary_metric, float(row["metric_value"])),
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
                "provenance_schema_version": pl.Int64,
                "phenoradar_version": pl.String,
                "phenoradar_install_type": pl.String,
                "git_source": pl.String,
                "git_commit": pl.String,
                "git_dirty": pl.Boolean,
                "git_worktree_patch_sha256": pl.String,
                "primary_metric": pl.String,
                "metric_contract_version": pl.Int64,
                "metric_display_name": pl.String,
                "metric_implementation": pl.String,
                "metric_threshold_name": pl.String,
                "metric_threshold_value": pl.Float64,
                "aggregate_scope": pl.String,
                "metric_value": pl.Float64,
                "fingerprint_schema_version": pl.Int64,
                "dataset_fingerprint": pl.String,
                "split_fingerprint": pl.String,
                "experiment_fingerprint": pl.String,
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
                "provenance_schema_version": pl.Int64,
                "phenoradar_version": pl.String,
                "phenoradar_install_type": pl.String,
                "git_source": pl.String,
                "git_commit": pl.String,
                "git_dirty": pl.Boolean,
                "git_worktree_patch_sha256": pl.String,
                "metric_name": pl.String,
                "metric_contract_version": pl.Int64,
                "metric_display_name": pl.String,
                "metric_implementation": pl.String,
                "metric_threshold_name": pl.String,
                "metric_threshold_value": pl.Float64,
                "aggregate_scope": pl.String,
                "metric_value": pl.Float64,
                "fingerprint_schema_version": pl.Int64,
                "dataset_fingerprint": pl.String,
                "split_fingerprint": pl.String,
                "experiment_fingerprint": pl.String,
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
            "metric_display_name": primary_contract["display_name"],
            "metric_implementation": primary_contract["implementation"],
            "metric_threshold_name": primary_contract["threshold_name"],
            "metric_threshold_value": primary_contract["threshold_value"],
            "metric_direction": metric_direction(options.primary_metric),
            "aggregate_scope": options.aggregate_scope,
            "include_stage": options.include_stage,
            "output_format": options.output_format,
            "strict": options.strict,
            "allow_mixed_experiments": options.allow_mixed_experiments,
            "glob": options.run_glob,
            "latest": options.latest,
        },
        "generated_by": phenoradar_build_snapshot(),
        "experiment_compatibility": compatibility,
        "software_compatibility": software_compatibility,
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
