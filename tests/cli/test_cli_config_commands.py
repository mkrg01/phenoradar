from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from phenoradar import __version__
from phenoradar.bundle import BundleError
from phenoradar.cli import app
from phenoradar.config import ConfigError
from phenoradar.cv import CVError
from phenoradar.figures import FigureError
from phenoradar.provenance import ProvenanceError
from phenoradar.reporting import ReportError
from phenoradar.split import SplitError

_ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _plain_output(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _c4_tiny_source_uri() -> str:
    return (Path(__file__).resolve().parents[2] / "testdata" / "c4_tiny").resolve().as_uri()


def _write_split_fixture(tmp_path: Path) -> tuple[Path, Path]:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t0\tg1",
                "sp3\t1\tg2",
                "sp4\t0\tg2",
                "sp5\t1\t",
                "sp6\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
                "sp6\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    return metadata, tpm


def _stub_resolved_config(*, execution_stage: str) -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(execution_stage=execution_stage, seed=42),
        report=SimpleNamespace(auto_threshold_selection_metric="mcc"),
        model_selection=SimpleNamespace(),
        data=SimpleNamespace(metadata_path="metadata.tsv", tpm_path="tpm.tsv"),
    )


def _stub_split_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        split_manifest=pl.DataFrame({"fold_id": ["0"], "species": ["sp1"], "pool": ["validation"]}),
        fold_count=1,
        pool_counts={"validation": 1},
        expression_rows_excluded=0,
    )


def _stub_cv_artifacts(
    *,
    ensemble_model_probs: pl.DataFrame | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        thresholds=pl.DataFrame(
            {
                "threshold_name": ["fixed_probability_threshold", "cv_derived_threshold"],
                "threshold_value": [0.5, 0.4],
                "source": ["config", "oof_predictions"],
                "selection_metric": ["NA", "mcc"],
                "selection_scope": ["NA", "outer_cv"],
            }
        ),
        warnings=[],
        metrics_cv=pl.DataFrame(
            {
                "aggregate_scope": ["macro"],
                "fold_id": ["NA"],
                "metric": ["mcc"],
                "metric_value": [0.5],
            }
        ),
        feature_importance=pl.DataFrame(
            {
                "feature": ["OG1"],
                "importance_mean": [1.0],
                "importance_std": [0.0],
                "n_models": [1],
                "method": ["coef_abs_l1_norm"],
            }
        ),
        coefficients=pl.DataFrame(
            {
                "feature": ["OG1"],
                "coef_mean": [0.2],
                "coef_std": [0.0],
                "n_models": [1],
                "method": ["coef_signed"],
                "reason": ["NA"],
            }
        ),
        oof_predictions=pl.DataFrame(
            {
                "fold_id": ["0", "0"],
                "species": ["sp1", "sp2"],
                "label": [0, 1],
                "prob": [0.2, 0.8],
            }
        ),
        ensemble_model_probs=ensemble_model_probs,
        model_selection_trials=None,
        model_selection_trials_summary=None,
        model_selection_selected=None,
    )


def _stub_final_refit_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        pred_external_test=pl.DataFrame(
            {
                "species": ["sp1"],
                "true_label": [1],
                "prob": [0.7],
                "pred_label_fixed_threshold": [1],
                "pred_label_cv_derived_threshold": [1],
            }
        ),
        pred_inference=pl.DataFrame(
            {
                "species": ["sp2"],
                "true_label": [None],
                "prob": [0.6],
                "pred_label_fixed_threshold": [1],
                "pred_label_cv_derived_threshold": [1],
            }
        ),
        warnings=[],
        model_selection_selected=None,
        ensemble_size=1,
    )


def test_config_writes_resolved_yaml(tmp_path: Path) -> None:
    runner = CliRunner()
    config = _write(
        tmp_path / "config.yml",
        """
runtime:
  seed: 123
sampling:
  weighting: group_label_inverse
""".strip()
        + "\n",
    )
    out = tmp_path / "resolved.yml"

    result = runner.invoke(
        app,
        [
            "config",
            "-c",
            str(config),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert payload["runtime"]["seed"] == 123
    assert "search_seed" not in payload["model_selection"]
    assert payload["sampling"]["weighting"] == "group_label_inverse"


def test_config_without_config_writes_default_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "config.yml"

    result = runner.invoke(
        app,
        ["config"],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert payload["runtime"]["seed"] == 42
    assert payload["runtime"]["execution_stage"] == "cv_only"
    assert payload["data"]["metadata_path"] == "testdata/c4_tiny/species_metadata.tsv"
    assert payload["data"]["tpm_path"] == "testdata/c4_tiny/tpm.tsv"


def test_run_rejects_multiple_config_options(tmp_path: Path) -> None:
    runner = CliRunner()
    config_a = _write(tmp_path / "config-a.yml", "{}\n")
    config_b = _write(tmp_path / "config-b.yml", "{}\n")

    result = runner.invoke(
        app,
        [
            "run",
            "-c",
            str(config_a),
            "-c",
            str(config_b),
        ],
    )

    assert result.exit_code != 0
    assert "can be specified at most once" in result.output


def test_config_rejects_multiple_config_options(tmp_path: Path) -> None:
    runner = CliRunner()
    config_a = _write(tmp_path / "config-a.yml", "{}\n")
    config_b = _write(tmp_path / "config-b.yml", "{}\n")

    result = runner.invoke(
        app,
        [
            "config",
            "--out",
            str(tmp_path / "resolved.yml"),
            "-c",
            str(config_a),
            "-c",
            str(config_b),
        ],
    )

    assert result.exit_code != 0
    assert "can be specified at most once" in result.output


def test_compose_config_alias_is_not_available() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["compose-config"])

    assert result.exit_code != 0
    assert "No such command" in result.output
    assert "compose-config" in result.output


def test_predict_rejects_multiple_config_options(tmp_path: Path) -> None:
    runner = CliRunner()
    config_a = _write(tmp_path / "config-a.yml", "{}\n")
    config_b = _write(tmp_path / "config-b.yml", "{}\n")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(bundle_dir),
            "-c",
            str(config_a),
            "-c",
            str(config_b),
        ],
    )

    assert result.exit_code != 0
    assert "can be specified at most once" in result.output


def test_cli_accepts_short_help_option() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["-h"])
    output = _plain_output(result.output)

    assert result.exit_code == 0, output
    assert "Usage: " in output
    assert "PhenoRadar: orthogroup TPM-based phenotype prediction CLI" in output
    assert "--version" in output
    assert "-V" in output

    run_help = runner.invoke(app, ["run", "-h"])
    run_help_output = _plain_output(run_help.output)
    assert run_help.exit_code == 0, run_help_output
    assert "Run training/evaluation pipeline." in run_help_output


def test_cli_accepts_global_version_option() -> None:
    runner = CliRunner()

    version_result = runner.invoke(app, ["--version"])
    assert version_result.exit_code == 0, version_result.output
    assert version_result.output.strip() == f"phenoradar {__version__}"

    short_result = runner.invoke(app, ["-V"])
    assert short_result.exit_code == 0, short_result.output
    assert short_result.output.strip() == f"phenoradar {__version__}"


def test_run_respects_execution_stage_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata, tpm = _write_split_fixture(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )

    result = runner.invoke(
        app,
        [
            "run",
            "-c",
            str(config),
            "--execution-stage",
            "full_run",
        ],
    )

    assert result.exit_code == 0, result.output
    runs_root = tmp_path / "runs"
    run_dirs = sorted(runs_root.glob("*_run_*"))
    assert len(run_dirs) == 1
    resolved_path = run_dirs[0] / "resolved_config.yml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["runtime"]["execution_stage"] == "full_run"
    assert (run_dirs[0] / "split_manifest.tsv").exists()
    assert (run_dirs[0] / "metrics_cv.tsv").exists()
    assert (run_dirs[0] / "thresholds.tsv").exists()
    assert (run_dirs[0] / "feature_importance.tsv").exists()
    assert (run_dirs[0] / "coefficients.tsv").exists()
    assert (run_dirs[0] / "prediction_external_test.tsv").exists()
    assert (run_dirs[0] / "prediction_inference.tsv").exists()
    assert (run_dirs[0] / "classification_summary.tsv").exists()
    assert (run_dirs[0] / "model_bundle").exists()
    assert (run_dirs[0] / "figures" / "cv_metrics_overview.svg").exists()
    assert (run_dirs[0] / "figures" / "threshold_selection_curve.svg").exists()
    assert (run_dirs[0] / "figures" / "feature_importance_top.svg").exists()
    assert (run_dirs[0] / "figures" / "coefficients_signed_top.svg").exists()
    assert (run_dirs[0] / "figures" / "roc_pr_curves_cv.svg").exists()

    metrics = pl.read_csv(run_dirs[0] / "metrics_cv.tsv", separator="\t")
    assert {"aggregate_scope", "fold_id", "metric", "metric_value"}.issubset(metrics.columns)
    thresholds = pl.read_csv(run_dirs[0] / "thresholds.tsv", separator="\t")
    assert set(thresholds.select("threshold_name").to_series().to_list()) == {
        "fixed_probability_threshold",
        "cv_derived_threshold",
    }
    pred_external = pl.read_csv(run_dirs[0] / "prediction_external_test.tsv", separator="\t")
    pred_inference = pl.read_csv(run_dirs[0] / "prediction_inference.tsv", separator="\t")
    classification_summary = pl.read_csv(
        run_dirs[0] / "classification_summary.tsv",
        separator="\t",
    )
    assert {
        "species",
        "true_label",
        "prob",
        "pred_label_fixed_threshold",
        "pred_label_cv_derived_threshold",
    }.issubset(pred_external.columns)
    assert {
        "species",
        "true_label",
        "prob",
        "pred_label_fixed_threshold",
        "pred_label_cv_derived_threshold",
    }.issubset(pred_inference.columns)
    assert "uncertainty_std" not in pred_external.columns
    assert "uncertainty_std" not in pred_inference.columns
    assert pred_external.height == 1
    assert pred_inference.height == 1
    assert pred_inference.get_column("true_label").to_list() == ["NA"]
    assert {
        "pool",
        "fold_id",
        "threshold_name",
        "threshold_value",
        "n_total",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
    }.issubset(classification_summary.columns)
    assert classification_summary.filter((pl.col("mcc") < -1.0) | (pl.col("mcc") > 1.0)).height == 0
    assert classification_summary.height == 8
    assert set(classification_summary.select("pool").to_series().to_list()) == {
        "validation_oof",
        "external_test",
    }
    assert set(classification_summary.select("threshold_name").to_series().to_list()) == {
        "fixed_probability_threshold",
        "cv_derived_threshold",
    }
    run_metadata = yaml.safe_load((run_dirs[0] / "run_metadata.json").read_text(encoding="utf-8"))
    assert "git_commit" in run_metadata
    assert "git_dirty" in run_metadata
    assert "git_worktree_patch_sha256" in run_metadata
    assert "seed_policy" in run_metadata
    assert run_metadata["seed_policy"]["runtime_seed"] == 42
    assert "environment" in run_metadata
    assert "input_files" in run_metadata
    assert isinstance(run_metadata["input_files"], list)


def test_run_cv_only_does_not_emit_final_prediction_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata, tpm = _write_split_fixture(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )

    result = runner.invoke(app, ["run", "-c", str(config), "--verbose"])
    assert result.exit_code == 0, result.output
    assert "Outer CV fold execution started" in result.output
    assert "Outer CV fold stage (fold_id=" in result.output
    assert "Outer CV fold completed" in result.output
    assert "progress=1/2" in result.output
    assert "progress=2/2" in result.output
    assert "features_before=" in result.output
    assert "features_after_low_prevalence=" in result.output
    assert "features_after_low_variance=" in result.output
    assert "features_after_correlation=" in result.output
    assert "features_after=" in result.output

    run_dirs = sorted((tmp_path / "runs").glob("*_run_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "feature_importance.tsv").exists()
    assert (run_dirs[0] / "coefficients.tsv").exists()
    assert (run_dirs[0] / "figures" / "cv_metrics_overview.svg").exists()
    assert (run_dirs[0] / "figures" / "threshold_selection_curve.svg").exists()
    assert (run_dirs[0] / "figures" / "feature_importance_top.svg").exists()
    assert (run_dirs[0] / "figures" / "coefficients_signed_top.svg").exists()
    assert (run_dirs[0] / "figures" / "roc_pr_curves_cv.svg").exists()
    assert (run_dirs[0] / "classification_summary.tsv").exists()
    assert not (run_dirs[0] / "prediction_external_test.tsv").exists()
    assert not (run_dirs[0] / "prediction_inference.tsv").exists()
    assert not (run_dirs[0] / "model_bundle").exists()
    classification_summary = pl.read_csv(
        run_dirs[0] / "classification_summary.tsv",
        separator="\t",
    )
    assert classification_summary.height == 6
    assert set(classification_summary.select("pool").to_series().to_list()) == {"validation_oof"}


def test_run_default_progress_is_compact_without_stage_level_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata, tpm = _write_split_fixture(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert "Outer CV fold execution started" in result.output
    assert "Outer CV fold completed" in result.output
    assert "Outer CV fold stage (fold_id=" not in result.output
    assert "features_before=" not in result.output


def test_run_requires_config_option() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run"])
    output = _plain_output(result.output)

    assert result.exit_code != 0
    assert "Missing option" in output
    assert "--config" in output
    assert "-c" in output


def test_run_emits_model_selection_artifacts_when_selection_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "g1_pos\t1\tg1",
                "g1_neg\t0\tg1",
                "g2_pos\t1\tg2",
                "g2_neg\t0\tg2",
                "g3_pos\t1\tg3",
                "g3_neg\t0\tg3",
                "g4_pos\t1\tg4",
                "g4_neg\t0\tg4",
                "ext1\t1\t",
                "inf1\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "g1_pos\tOG1\t5.0",
                "g1_pos\tOG2\t1.5",
                "g1_neg\tOG1\t1.0",
                "g1_neg\tOG2\t0.2",
                "g2_pos\tOG1\t4.8",
                "g2_pos\tOG2\t1.7",
                "g2_neg\tOG1\t0.8",
                "g2_neg\tOG2\t0.4",
                "g3_pos\tOG1\t5.2",
                "g3_pos\tOG2\t1.8",
                "g3_neg\tOG1\t1.1",
                "g3_neg\tOG2\t0.3",
                "g4_pos\tOG1\t5.1",
                "g4_pos\tOG2\t1.6",
                "g4_neg\tOG1\t0.9",
                "g4_neg\tOG2\t0.1",
                "ext1\tOG1\t3.3",
                "ext1\tOG2\t1.1",
                "inf1\tOG1\t2.0",
                "inf1\tOG2\t0.6",
            ]
        )
        + "\n",
    )
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
split:
  outer_cv_strategy: group_kfold
  outer_cv_n_splits: 2
model_selection:
  search_strategy: grid
  search_space:
    C: [0.5, 1.0]
  selected_candidate_count: 1
  inner_cv_strategy: logo
""".strip()
        + "\n",
    )

    result = runner.invoke(
        app, ["run", "-c", str(config), "--execution-stage", "full_run", "--verbose"]
    )
    assert result.exit_code == 0, result.output
    assert "stage=selection_start" in result.output
    assert "stage=selection_source_done" in result.output
    assert "stage=selection_candidate_done" in result.output
    assert "source_progress=" in result.output
    assert "candidate_progress=" in result.output

    run_dirs = sorted((tmp_path / "runs").glob("*_run_*"))
    assert len(run_dirs) == 1
    selected_path = run_dirs[0] / "model_selection_selected.tsv"
    trials_path = run_dirs[0] / "model_selection_trials.tsv"
    trials_summary_path = run_dirs[0] / "model_selection_trials_summary.tsv"
    assert selected_path.exists()
    assert trials_path.exists()
    assert trials_summary_path.exists()
    assert not (run_dirs[0] / "figures" / "model_selection_trials.svg").exists()

    selected_df = pl.read_csv(selected_path, separator="\t")
    scopes = set(selected_df.select("selection_scope").to_series().to_list())
    assert {"outer_fold", "final_refit"}.issubset(scopes)

    trials_df = pl.read_csv(trials_path, separator="\t")
    assert {"fold_id", "sample_set_id", "candidate_index", "inner_fold_id"}.issubset(
        trials_df.columns
    )
    trials_summary_df = pl.read_csv(trials_summary_path, separator="\t")
    assert {
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "metric_name",
        "params_json",
        "n_inner_folds",
        "n_valid_inner_folds",
        "metric_value_mean",
        "metric_value_std",
    }.issubset(trials_summary_df.columns)


def test_run_emits_warning_summary_and_quiet_mode_suppresses_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")
    cv_artifacts = _stub_cv_artifacts()
    cv_artifacts.warnings = ["warn-a", "warn-b", "warn-c", "warn-d", "warn-e", "warn-f"]

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="cv_only"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: _stub_split_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: cv_artifacts,
    )
    monkeypatch.setattr("phenoradar.cli.write_resolved_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("phenoradar.cli.write_run_figures", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("phenoradar.cli.collect_input_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("phenoradar.cli.git_snapshot", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "phenoradar.cli.runtime_environment_snapshot",
        lambda *_args, **_kwargs: {"python": "test"},
    )

    result = runner.invoke(app, ["run", "-c", str(config), "--quiet"])

    assert result.exit_code == 0, result.output
    assert "[run] Start training/evaluation pipeline." not in result.output
    assert "WARNING: Recorded 6 warning(s)." in result.output
    assert "WARNING: 1/6: warn-a" in result.output
    assert "WARNING: ... and 1 more warning(s)." in result.output
    assert "Wrote run artifacts at" in result.output
    assert "warnings=6" in result.output


def test_run_rejects_verbose_and_quiet_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata, tpm = _write_split_fixture(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )

    result = runner.invoke(app, ["run", "-c", str(config), "--verbose", "--quiet"])
    output = _plain_output(result.output)

    assert result.exit_code != 0
    assert "--verbose" in output
    assert "--quiet" in output


def test_predict_uses_model_bundle_and_emits_predict_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata, tpm = _write_split_fixture(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )

    run_result = runner.invoke(
        app,
        [
            "run",
            "-c",
            str(config),
            "--execution-stage",
            "full_run",
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    run_dirs = sorted((tmp_path / "runs").glob("*_run_*"))
    assert len(run_dirs) == 1
    bundle_dir = run_dirs[0] / "model_bundle"
    assert bundle_dir.exists()

    predict_result = runner.invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(bundle_dir),
            "-c",
            str(config),
        ],
    )
    assert predict_result.exit_code == 0, predict_result.output

    predict_dirs = sorted((tmp_path / "runs").glob("*_predict_*"))
    assert len(predict_dirs) == 1
    assert (predict_dirs[0] / "prediction_inference.tsv").exists()
    assert not (predict_dirs[0] / "pred_predict.tsv").exists()
    assert (predict_dirs[0] / "resolved_config.yml").exists()
    assert (predict_dirs[0] / "run_metadata.json").exists()
    assert (predict_dirs[0] / "figures" / "predict_probability_distribution.svg").exists()
    assert not (predict_dirs[0] / "metrics_cv.tsv").exists()

    pred_inference = pl.read_csv(predict_dirs[0] / "prediction_inference.tsv", separator="\t")
    assert {
        "species",
        "true_label",
        "prob",
        "pred_label_fixed_threshold",
        "pred_label_cv_derived_threshold",
    }.issubset(pred_inference.columns)
    assert pred_inference.height == 6
    assert pred_inference.get_column("true_label").to_list() == ["NA"] * pred_inference.height
    predict_metadata = yaml.safe_load(
        (predict_dirs[0] / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert "git_commit" in predict_metadata
    assert "git_dirty" in predict_metadata
    assert "git_worktree_patch_sha256" in predict_metadata
    assert "environment" in predict_metadata
    assert "input_files" in predict_metadata
    assert "model_bundle_manifest_sha256" in predict_metadata
    assert "model_bundle_payload_sha256" in predict_metadata
    assert "bundle_source_run_dir" in predict_metadata
    assert "seed_policy" in predict_metadata


def test_report_aggregates_run_and_predict_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata, tpm = _write_split_fixture(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )

    full_run_result = runner.invoke(
        app,
        [
            "run",
            "-c",
            str(config),
            "--execution-stage",
            "full_run",
        ],
    )
    assert full_run_result.exit_code == 0, full_run_result.output
    run_dirs = sorted((tmp_path / "runs").glob("*_run_*"))
    assert len(run_dirs) == 1

    predict_result = runner.invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(run_dirs[0] / "model_bundle"),
            "-c",
            str(config),
        ],
    )
    assert predict_result.exit_code == 0, predict_result.output

    report_result = runner.invoke(
        app,
        [
            "report",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )
    assert report_result.exit_code == 0, report_result.output

    report_dirs = sorted((tmp_path / "reports").glob("*_report_*"))
    assert len(report_dirs) == 1
    report_dir = report_dirs[0]
    assert (report_dir / "report_manifest.json").exists()
    assert (report_dir / "report_runs.tsv").exists()
    assert (report_dir / "report_ranking.tsv").exists()
    assert (report_dir / "report_warnings.tsv").exists()
    assert (report_dir / "figures").exists()
    assert (report_dir / "figures" / "report_metric_ranking.svg").exists()
    assert (report_dir / "figures" / "report_metric_comparison.svg").exists()
    assert (report_dir / "figures" / "report_stage_breakdown.svg").exists()

    report_runs = pl.read_csv(report_dir / "report_runs.tsv", separator="\t")
    assert {"run_id", "execution_stage", "metric_value"}.issubset(report_runs.columns)
    stages = set(report_runs.select("execution_stage").to_series().to_list())
    assert {"full_run", "predict"}.issubset(stages)

    report_ranking = pl.read_csv(report_dir / "report_ranking.tsv", separator="\t")
    assert {"rank", "run_id", "execution_stage", "metric_value"}.issubset(report_ranking.columns)
    if report_ranking.height > 0:
        ranking_stages = set(report_ranking.select("execution_stage").to_series().to_list())
        assert "predict" not in ranking_stages


def test_report_strict_fails_on_missing_required_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    broken_run_dir = tmp_path / "runs" / "20260101T000000Z_run_broken"
    broken_run_dir.mkdir(parents=True)

    non_strict_result = runner.invoke(
        app,
        [
            "report",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )
    assert non_strict_result.exit_code == 0, non_strict_result.output
    report_dirs = sorted((tmp_path / "reports").glob("*_report_*"))
    assert len(report_dirs) == 1
    warnings_df = pl.read_csv(report_dirs[0] / "report_warnings.tsv", separator="\t")
    assert warnings_df.height >= 1

    strict_result = runner.invoke(
        app,
        [
            "report",
            "--runs-root",
            str(tmp_path / "runs"),
            "--strict",
        ],
    )
    assert strict_result.exit_code != 0


def test_report_non_strict_skips_invalid_metadata_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    broken_run_dir = tmp_path / "runs" / "20260101T000000Z_run_invalid_meta"
    broken_run_dir.mkdir(parents=True)
    _write(
        broken_run_dir / "run_metadata.json",
        "{invalid-json}\n",
    )
    _write(
        broken_run_dir / "resolved_config.yml",
        "runtime:\n  seed: 42\n",
    )

    non_strict_result = runner.invoke(
        app,
        [
            "report",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )
    assert non_strict_result.exit_code == 0, non_strict_result.output
    report_dirs = sorted((tmp_path / "reports").glob("*_report_*"))
    assert len(report_dirs) == 1
    warnings_df = pl.read_csv(report_dirs[0] / "report_warnings.tsv", separator="\t")
    assert warnings_df.filter(pl.col("warning_type") == "invalid_artifact").height >= 1

    strict_result = runner.invoke(
        app,
        [
            "report",
            "--runs-root",
            str(tmp_path / "runs"),
            "--strict",
        ],
    )
    assert strict_result.exit_code != 0


def test_report_quiet_emits_warning_type_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)

    missing_meta = runs_root / "20260101T000000Z_run_missing_meta"
    missing_meta.mkdir(parents=True)

    invalid_meta = runs_root / "20260101T000001Z_run_invalid_meta"
    invalid_meta.mkdir(parents=True)
    _write(invalid_meta / "run_metadata.json", "{invalid-json}\n")
    _write(invalid_meta / "resolved_config.yml", "runtime:\n  seed: 42\n")

    result = runner.invoke(
        app,
        [
            "report",
            "--runs-root",
            str(runs_root),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "WARNING: Recorded 2 warning row(s)." in result.output
    assert "WARNING: Warning type summary (2 type(s))." in result.output
    assert "type=invalid_artifact, rows=1, runs=1" in result.output
    assert "type=missing_artifact, rows=1, runs=1" in result.output


def test_report_requires_run_selection_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["report"])
    output = _plain_output(result.output)

    assert result.exit_code != 0
    assert "Either --run-dir or --runs-root must be provided" in output


def test_config_fails_for_invalid_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    invalid = _write(tmp_path / "invalid.yml", "runtime: [1, 2\n")

    result = runner.invoke(
        app,
        [
            "config",
            "-c",
            str(invalid),
            "--out",
            str(tmp_path / "resolved.yml"),
        ],
    )

    assert result.exit_code != 0
    assert "Invalid YAML in config file" in result.output


def test_run_fails_when_cv_threshold_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            runtime=SimpleNamespace(execution_stage="cv_only")
        ),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(split_manifest=pl.DataFrame()),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: SimpleNamespace(
            thresholds=pl.DataFrame(
                {
                    "threshold_name": ["fixed_probability_threshold"],
                    "threshold_value": [0.5],
                }
            )
        ),
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code != 0
    assert "cv_derived_threshold was not found in thresholds table" in result.output


def test_predict_fails_when_predict_figure_generation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata, tpm = _write_split_fixture(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    monkeypatch.setattr(
        "phenoradar.cli.load_model_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            models=[object()],
            manifest_sha256="manifest-sha",
            source_run_id="source-run",
            manifest={},
        ),
    )
    monkeypatch.setattr(
        "phenoradar.cli.predict_with_bundle",
        lambda *_args, **_kwargs: (
            pl.DataFrame(
                {
                    "species": ["sp1"],
                    "prob": [0.5],
                    "pred_label_fixed_threshold": [1],
                    "pred_label_cv_derived_threshold": [1],
                }
            ),
            [],
        ),
    )
    monkeypatch.setattr(
        "phenoradar.cli.write_predict_figures",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FigureError("figure write failed")),
    )

    result = runner.invoke(app, ["predict", "--model-bundle", str(bundle_dir), "-c", str(config)])

    assert result.exit_code != 0
    assert "figure write failed" in result.output


def test_report_fails_when_generate_report_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    monkeypatch.setattr(
        "phenoradar.cli.generate_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ReportError("report failure")),
    )

    result = runner.invoke(app, ["report", "--runs-root", str(runs_root)])

    assert result.exit_code != 0
    assert "report failure" in result.output


def test_run_fails_when_config_resolution_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConfigError("config failure")),
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code != 0
    assert "config failure" in result.output


def test_run_fails_when_split_artifact_build_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="cv_only"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SplitError("split failure")),
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code != 0
    assert "split failure" in result.output


def test_run_fails_when_outer_cv_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="cv_only"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: _stub_split_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CVError("outer cv failure")),
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code != 0
    assert "outer cv failure" in result.output


def test_run_fails_when_final_refit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="full_run"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: _stub_split_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: _stub_cv_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_final_refit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CVError("final refit failure")),
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code != 0
    assert "final refit failure" in result.output


def test_run_fails_when_bundle_export_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="full_run"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: _stub_split_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: _stub_cv_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_final_refit",
        lambda *_args, **_kwargs: _stub_final_refit_artifacts(),
    )
    monkeypatch.setattr("phenoradar.cli.write_resolved_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "phenoradar.cli.export_model_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(BundleError("bundle export failure")),
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code != 0
    assert "bundle export failure" in result.output


def test_run_fails_when_run_figure_generation_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="cv_only"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: _stub_split_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: _stub_cv_artifacts(),
    )
    monkeypatch.setattr("phenoradar.cli.write_resolved_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "phenoradar.cli.write_run_figures",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FigureError("run figure failure")),
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code != 0
    assert "run figure failure" in result.output


def test_run_fails_when_input_provenance_collection_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="cv_only"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: _stub_split_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: _stub_cv_artifacts(),
    )
    monkeypatch.setattr("phenoradar.cli.write_resolved_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("phenoradar.cli.write_run_figures", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "phenoradar.cli.collect_input_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProvenanceError("provenance failure")),
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code != 0
    assert "provenance failure" in result.output


def test_run_writes_ensemble_tables_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")
    ensemble_model_probs = pl.DataFrame(
        {
            "fold_id": ["0", "0"],
            "model_index": [0, 1],
            "species": ["sp1", "sp1"],
            "prob": [0.2, 0.3],
        }
    )
    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="cv_only"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: _stub_split_artifacts(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: _stub_cv_artifacts(
            ensemble_model_probs=ensemble_model_probs,
        ),
    )
    monkeypatch.setattr("phenoradar.cli.write_resolved_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("phenoradar.cli.write_run_figures", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("phenoradar.cli.collect_input_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("phenoradar.cli.git_snapshot", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "phenoradar.cli.runtime_environment_snapshot",
        lambda *_args, **_kwargs: {"python": "test"},
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code == 0, result.output
    run_dirs = sorted((tmp_path / "runs").glob("*_run_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "ensemble_model_probs.tsv").exists()


def test_predict_fails_when_config_resolution_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConfigError("predict config failure")),
    )

    result = runner.invoke(app, ["predict", "--model-bundle", str(bundle_dir), "-c", str(config)])

    assert result.exit_code != 0
    assert "predict config failure" in result.output


def test_predict_fails_when_bundle_loading_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="cv_only"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.load_model_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(BundleError("bundle load failure")),
    )

    result = runner.invoke(app, ["predict", "--model-bundle", str(bundle_dir), "-c", str(config)])

    assert result.exit_code != 0
    assert "bundle load failure" in result.output


def test_predict_fails_when_input_provenance_collection_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(execution_stage="cv_only"),
    )
    monkeypatch.setattr(
        "phenoradar.cli.load_model_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            models=[object()],
            manifest_sha256="manifest-sha",
            source_run_id="source-run",
            manifest={},
        ),
    )
    monkeypatch.setattr(
        "phenoradar.cli.predict_with_bundle",
        lambda *_args, **_kwargs: (
            pl.DataFrame(
                {
                    "species": ["sp1"],
                    "prob": [0.5],
                    "pred_label_fixed_threshold": [1],
                    "pred_label_cv_derived_threshold": [1],
                }
            ),
            [],
        ),
    )
    monkeypatch.setattr("phenoradar.cli.write_resolved_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("phenoradar.cli.write_predict_figures", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "phenoradar.cli.collect_input_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProvenanceError("predict provenance failure")
        ),
    )

    result = runner.invoke(app, ["predict", "--model-bundle", str(bundle_dir), "-c", str(config)])

    assert result.exit_code != 0
    assert "predict provenance failure" in result.output


def test_predict_reuses_bundle_without_invoking_training_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    metadata, tpm = _write_split_fixture(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )

    run_result = runner.invoke(app, ["run", "-c", str(config), "--execution-stage", "full_run"])
    assert run_result.exit_code == 0, run_result.output
    run_dirs = sorted((tmp_path / "runs").glob("*_run_*"))
    assert len(run_dirs) == 1
    bundle_dir = run_dirs[0] / "model_bundle"
    assert bundle_dir.exists()

    monkeypatch.setattr(
        "phenoradar.cli.build_split_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("training path called")),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_outer_cv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("training path called")),
    )
    monkeypatch.setattr(
        "phenoradar.cli.run_final_refit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("training path called")),
    )

    predict_result = runner.invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(bundle_dir),
            "-c",
            str(config),
        ],
    )
    assert predict_result.exit_code == 0, predict_result.output


def test_dataset_downloads_compact_dataset(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "c4_dataset"

    result = runner.invoke(
        app,
        [
            "dataset",
            "--base-url",
            _c4_tiny_source_uri(),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "species_metadata.tsv").exists()
    assert (out_dir / "tpm.tsv").exists()


def test_dataset_requires_force_for_checksum_mismatch(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "c4_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "species_metadata.tsv").write_text("broken\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "dataset",
            "--base-url",
            _c4_tiny_source_uri(),
            "--out",
            str(out_dir),
        ],
    )
    output = _plain_output(result.output)
    assert result.exit_code != 0
    assert "checksum" in output
    assert "use --force" in output

    force_result = runner.invoke(
        app,
        [
            "dataset",
            "--base-url",
            _c4_tiny_source_uri(),
            "--out",
            str(out_dir),
            "--force",
        ],
    )
    assert force_result.exit_code == 0, force_result.output
