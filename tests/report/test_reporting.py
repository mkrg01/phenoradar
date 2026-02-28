from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import phenoradar.reporting as reporting_mod
from phenoradar.figures import FigureError
from phenoradar.reporting import ReportError, ReportOptions, generate_report


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_run_dir(
    runs_root: Path,
    *,
    run_id: str,
    stage: str,
    start_time: str,
    metric_value: float | None,
    include_metrics: bool = True,
    metrics_valid_schema: bool = True,
) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write(
        run_dir / "run_metadata.json",
        json.dumps(
            {
                "command": "predict" if stage == "predict" else "run",
                "execution_stage": stage,
                "status": "ok",
                "start_time": start_time,
                "end_time": start_time,
                "duration_sec": 1.0,
                "warnings": [],
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )
    _write(
        run_dir / "resolved_config.yml",
        "runtime:\n  seed: 42\n",
    )
    if include_metrics:
        if metrics_valid_schema:
            pl.DataFrame(
                {
                    "aggregate_scope": ["macro"],
                    "fold_id": ["NA"],
                    "metric": ["mcc"],
                    "metric_value": [metric_value if metric_value is not None else float("nan")],
                }
            ).write_csv(run_dir / "metrics_cv.tsv", separator="\t")
        else:
            pl.DataFrame(
                {
                    "aggregate_scope": ["macro"],
                    "fold_id": ["NA"],
                    "metric": ["mcc"],
                }
            ).write_csv(run_dir / "metrics_cv.tsv", separator="\t")
    return run_dir


def _options(
    *,
    include_stage: str = "all",
    strict: bool = True,
) -> ReportOptions:
    return ReportOptions(
        primary_metric="mcc",
        aggregate_scope="macro",
        include_stage=include_stage,
        output_format="tsv",
        strict=strict,
        run_glob="*",
        latest=None,
    )


def test_json_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="Required file not found"):
        reporting_mod._json_load(tmp_path / "missing.json")


def test_yaml_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="Required file not found"):
        reporting_mod._yaml_load(tmp_path / "missing.yml")


def test_yaml_load_returns_empty_mapping_for_empty_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.yml", "")
    assert reporting_mod._yaml_load(path) == {}


def test_float_or_none_returns_none_for_non_numeric_values() -> None:
    assert reporting_mod._float_or_none("1.0") is None


def test_collect_run_dirs_rejects_nonexistent_runs_root(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="runs-root does not exist"):
        reporting_mod._collect_run_dirs(
            run_dirs=[],
            runs_root=tmp_path / "missing-runs-root",
            run_glob="*",
            latest=None,
        )


def test_load_metric_value_returns_none_when_multiple_rows_match(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics_cv.tsv"
    pl.DataFrame(
        {
            "aggregate_scope": ["macro", "macro"],
            "fold_id": ["NA", "NA"],
            "metric": ["mcc", "mcc"],
            "metric_value": [0.1, 0.2],
        }
    ).write_csv(metrics_path, separator="\t")

    value = reporting_mod._load_metric_value(
        metrics_path=metrics_path,
        aggregate_scope="macro",
        primary_metric="mcc",
    )

    assert value is None


def test_load_metric_value_returns_none_for_nan_metric(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics_cv.tsv"
    pl.DataFrame(
        {
            "aggregate_scope": ["macro"],
            "fold_id": ["NA"],
            "metric": ["mcc"],
            "metric_value": [float("nan")],
        }
    ).write_csv(metrics_path, separator="\t")

    value = reporting_mod._load_metric_value(
        metrics_path=metrics_path,
        aggregate_scope="macro",
        primary_metric="mcc",
    )

    assert value is None


def test_generate_report_ranking_tie_breaks_by_start_time_then_run_id(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000003Z_run_z",
        stage="full_run",
        start_time="2026-01-02T00:00:00+00:00",
        metric_value=0.8,
    )
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_a",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.8,
    )
    _write_run_dir(
        runs_root,
        run_id="20260101T000002Z_run_m",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.8,
    )
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=_options(include_stage="all", strict=True),
        output_dir=out_dir,
    )

    ranking = pl.read_csv(out_dir / "report_ranking.tsv", separator="\t")
    assert ranking.select("run_id").to_series().to_list() == [
        "20260101T000001Z_run_a",
        "20260101T000002Z_run_m",
        "20260101T000003Z_run_z",
    ]
    assert ranking.select("rank").to_series().to_list() == [1, 2, 3]


def test_generate_report_include_stage_predict_filters_and_leaves_empty_ranking(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_full",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.9,
    )
    _write_run_dir(
        runs_root,
        run_id="20260101T000002Z_predict_only",
        stage="predict",
        start_time="2026-01-02T00:00:00+00:00",
        metric_value=None,
        include_metrics=False,
    )
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=_options(include_stage="predict", strict=True),
        output_dir=out_dir,
    )

    report_runs = pl.read_csv(out_dir / "report_runs.tsv", separator="\t")
    ranking = pl.read_csv(out_dir / "report_ranking.tsv", separator="\t")
    assert report_runs.height == 1
    assert report_runs.select("execution_stage").to_series().to_list() == ["predict"]
    assert ranking.height == 0


def test_generate_report_non_strict_warns_when_metrics_missing_for_non_predict(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_missing_metrics",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=None,
        include_metrics=False,
    )
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=_options(include_stage="all", strict=False),
        output_dir=out_dir,
    )

    warnings = pl.read_csv(out_dir / "report_warnings.tsv", separator="\t")
    assert warnings.filter(pl.col("warning_type") == "missing_metrics").height == 1


def test_generate_report_latest_limits_selected_run_dirs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_old",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.7,
    )
    _write_run_dir(
        runs_root,
        run_id="20260101T000002Z_run_new",
        stage="full_run",
        start_time="2026-01-02T00:00:00+00:00",
        metric_value=0.8,
    )
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=1,
        options=_options(include_stage="all", strict=True),
        output_dir=out_dir,
    )

    manifest = json.loads((out_dir / "report_manifest.json").read_text(encoding="utf-8"))
    selected = manifest["selected_run_dirs"]
    assert len(selected) == 1
    assert selected[0].endswith("20260101T000002Z_run_new")


def test_generate_report_strict_fails_on_invalid_metrics_schema(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_bad_metrics",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.9,
        include_metrics=True,
        metrics_valid_schema=False,
    )
    out_dir = tmp_path / "report_out"

    with pytest.raises(ReportError, match="invalid schema"):
        generate_report(
            run_dirs=[],
            runs_root=runs_root,
            run_glob="*",
            latest=None,
            options=_options(include_stage="all", strict=True),
            output_dir=out_dir,
        )


def test_generate_report_requires_run_dirs_or_runs_root(tmp_path: Path) -> None:
    out_dir = tmp_path / "report_out"
    with pytest.raises(ReportError, match="Either --run-dir or --runs-root must be provided"):
        generate_report(
            run_dirs=[],
            runs_root=None,
            run_glob="*",
            latest=None,
            options=_options(include_stage="all", strict=True),
            output_dir=out_dir,
        )


def test_generate_report_fails_when_no_run_directories_found(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    out_dir = tmp_path / "report_out"
    with pytest.raises(ReportError, match="No run directories were found"):
        generate_report(
            run_dirs=[],
            runs_root=runs_root,
            run_glob="*",
            latest=None,
            options=_options(include_stage="all", strict=True),
            output_dir=out_dir,
        )


def test_generate_report_rejects_latest_less_than_one(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_one",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.9,
    )
    out_dir = tmp_path / "report_out"
    with pytest.raises(ReportError, match="--latest must be >= 1"):
        generate_report(
            run_dirs=[],
            runs_root=runs_root,
            run_glob="*",
            latest=0,
            options=_options(include_stage="all", strict=True),
            output_dir=out_dir,
        )


def test_generate_report_non_strict_warns_on_invalid_metadata_root_type(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101T000001Z_run_bad_meta_root"
    run_dir.mkdir(parents=True)
    _write(run_dir / "run_metadata.json", "[1, 2, 3]\n")
    _write(run_dir / "resolved_config.yml", "runtime:\n  seed: 42\n")
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=_options(include_stage="all", strict=False),
        output_dir=out_dir,
    )

    warnings = pl.read_csv(out_dir / "report_warnings.tsv", separator="\t")
    assert warnings.filter(pl.col("warning_type") == "invalid_artifact").height == 1


def test_generate_report_non_strict_warns_on_invalid_resolved_config_root_type(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101T000001Z_run_bad_yaml_root"
    run_dir.mkdir(parents=True)
    _write(
        run_dir / "run_metadata.json",
        json.dumps(
            {
                "command": "run",
                "execution_stage": "cv_only",
                "status": "ok",
                "start_time": "2026-01-01T00:00:00+00:00",
                "end_time": "2026-01-01T00:00:00+00:00",
                "duration_sec": 1.0,
            },
            ensure_ascii=True,
        )
        + "\n",
    )
    _write(run_dir / "resolved_config.yml", "- not-a-mapping\n")
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=_options(include_stage="all", strict=False),
        output_dir=out_dir,
    )

    warnings = pl.read_csv(out_dir / "report_warnings.tsv", separator="\t")
    assert warnings.filter(pl.col("warning_type") == "invalid_artifact").height == 1


@pytest.mark.parametrize(
    ("output_format", "expected_artifact"),
    [
        ("md", "report.md"),
        ("html", "report.html"),
        ("json", "report.json"),
    ],
)
def test_generate_report_output_format_artifacts(
    tmp_path: Path, output_format: str, expected_artifact: str
) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_for_format",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.9,
    )
    out_dir = tmp_path / "report_out"
    options = ReportOptions(
        primary_metric="mcc",
        aggregate_scope="macro",
        include_stage="all",
        output_format=output_format,  # type: ignore[arg-type]
        strict=True,
        run_glob="*",
        latest=None,
    )

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=options,
        output_dir=out_dir,
    )

    assert (out_dir / expected_artifact).exists()


def test_generate_report_uses_explicit_run_dirs_without_runs_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_explicit",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.9,
    )
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[run_dir],
        runs_root=None,
        run_glob="*",
        latest=None,
        options=_options(include_stage="all", strict=True),
        output_dir=out_dir,
    )

    report_runs = pl.read_csv(out_dir / "report_runs.tsv", separator="\t")
    assert report_runs.height == 1
    assert report_runs.select("run_id").to_series().to_list() == [run_dir.name]


def test_generate_report_strict_fails_when_non_predict_metrics_are_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_no_metrics",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=None,
        include_metrics=False,
    )
    out_dir = tmp_path / "report_out"

    with pytest.raises(ReportError, match="metrics_cv.tsv is missing for a non-predict run"):
        generate_report(
            run_dirs=[],
            runs_root=runs_root,
            run_glob="*",
            latest=None,
            options=_options(include_stage="all", strict=True),
            output_dir=out_dir,
        )


def test_generate_report_non_strict_warns_when_metrics_schema_invalid(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_bad_metrics",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.9,
        include_metrics=True,
        metrics_valid_schema=False,
    )
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=_options(include_stage="all", strict=False),
        output_dir=out_dir,
    )

    warnings = pl.read_csv(out_dir / "report_warnings.tsv", separator="\t")
    assert warnings.filter(pl.col("warning_type") == "invalid_metrics").height == 1


def test_generate_report_strict_fails_when_resolved_config_is_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101T000001Z_run_missing_config"
    run_dir.mkdir(parents=True)
    _write(
        run_dir / "run_metadata.json",
        json.dumps(
            {
                "command": "run",
                "execution_stage": "full_run",
                "status": "ok",
                "start_time": "2026-01-01T00:00:00+00:00",
                "end_time": "2026-01-01T00:00:00+00:00",
                "duration_sec": 1.0,
            },
            ensure_ascii=True,
        )
        + "\n",
    )
    out_dir = tmp_path / "report_out"

    with pytest.raises(ReportError, match="Missing required artifact: resolved_config.yml"):
        generate_report(
            run_dirs=[],
            runs_root=runs_root,
            run_glob="*",
            latest=None,
            options=_options(include_stage="all", strict=True),
            output_dir=out_dir,
        )


def test_generate_report_non_strict_warns_when_resolved_config_is_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101T000001Z_run_missing_config"
    run_dir.mkdir(parents=True)
    _write(
        run_dir / "run_metadata.json",
        json.dumps(
            {
                "command": "run",
                "execution_stage": "full_run",
                "status": "ok",
                "start_time": "2026-01-01T00:00:00+00:00",
                "end_time": "2026-01-01T00:00:00+00:00",
                "duration_sec": 1.0,
            },
            ensure_ascii=True,
        )
        + "\n",
    )
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=_options(include_stage="all", strict=False),
        output_dir=out_dir,
    )

    warnings = pl.read_csv(out_dir / "report_warnings.tsv", separator="\t")
    assert warnings.filter(pl.col("warning_type") == "missing_artifact").height == 1


def test_generate_report_strict_fails_on_invalid_resolved_config_yaml(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101T000001Z_run_invalid_yaml"
    run_dir.mkdir(parents=True)
    _write(
        run_dir / "run_metadata.json",
        json.dumps(
            {
                "command": "run",
                "execution_stage": "full_run",
                "status": "ok",
                "start_time": "2026-01-01T00:00:00+00:00",
                "end_time": "2026-01-01T00:00:00+00:00",
                "duration_sec": 1.0,
            },
            ensure_ascii=True,
        )
        + "\n",
    )
    _write(run_dir / "resolved_config.yml", "runtime: [1, 2\n")
    out_dir = tmp_path / "report_out"

    with pytest.raises(ReportError, match="Invalid YAML file"):
        generate_report(
            run_dirs=[],
            runs_root=runs_root,
            run_glob="*",
            latest=None,
            options=_options(include_stage="all", strict=True),
            output_dir=out_dir,
        )


def test_generate_report_collects_run_warning_entries_from_metadata(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101T000001Z_run_with_warning"
    run_dir.mkdir(parents=True)
    _write(
        run_dir / "run_metadata.json",
        json.dumps(
            {
                "command": "run",
                "execution_stage": "full_run",
                "status": "ok",
                "start_time": "2026-01-01T00:00:00+00:00",
                "end_time": "2026-01-01T00:00:00+00:00",
                "duration_sec": 1.0,
                "warnings": ["warning-a", "warning-b"],
            },
            ensure_ascii=True,
        )
        + "\n",
    )
    _write(run_dir / "resolved_config.yml", "runtime:\n  seed: 42\n")
    pl.DataFrame(
        {
            "aggregate_scope": ["macro"],
            "fold_id": ["NA"],
            "metric": ["mcc"],
            "metric_value": [0.5],
        }
    ).write_csv(run_dir / "metrics_cv.tsv", separator="\t")
    out_dir = tmp_path / "report_out"

    generate_report(
        run_dirs=[],
        runs_root=runs_root,
        run_glob="*",
        latest=None,
        options=_options(include_stage="all", strict=True),
        output_dir=out_dir,
    )

    warnings = pl.read_csv(out_dir / "report_warnings.tsv", separator="\t")
    run_warnings = warnings.filter(pl.col("warning_type") == "run_warning")
    assert run_warnings.height == 2


def test_generate_report_wraps_figure_error_as_report_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    _write_run_dir(
        runs_root,
        run_id="20260101T000001Z_run_for_figure_failure",
        stage="full_run",
        start_time="2026-01-01T00:00:00+00:00",
        metric_value=0.9,
    )
    monkeypatch.setattr(
        reporting_mod,
        "write_report_figures",
        lambda **_kwargs: (_ for _ in ()).throw(FigureError("figure generation failed")),
    )
    out_dir = tmp_path / "report_out"

    with pytest.raises(ReportError, match="figure generation failed"):
        generate_report(
            run_dirs=[],
            runs_root=runs_root,
            run_glob="*",
            latest=None,
            options=_options(include_stage="all", strict=True),
            output_dir=out_dir,
        )
