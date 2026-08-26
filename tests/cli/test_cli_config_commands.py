from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

import phenoradar.cli as cli_mod
from phenoradar import __version__
from phenoradar.bundle import BundleError
from phenoradar.cli import app
from phenoradar.config import ConfigError
from phenoradar.cv import CVError
from phenoradar.figures import FigureError
from phenoradar.provenance import ProvenanceError
from phenoradar.reporting import ReportError
from phenoradar.split import SplitError
from phenoradar.timing import TimingRecorder

_ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _plain_output(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _c4_tiny_source_uri() -> str:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "phenoradar"
        / "data"
        / "c4_tiny"
    ).resolve().as_uri()


def _write_split_fixture(tmp_path: Path) -> tuple[Path, Path]:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout\tfamily_id\tfamily_name",
                "sp1\t1\tg1\tno\tf1\tFamily 1",
                "sp2\t0\tg1\tno\tf1\tFamily 1",
                "sp3\t1\tg2\tno\tf2\tFamily 2",
                "sp4\t0\tg2\tno\tf2\tFamily 2",
                "sp5\t1\t\tyes\tf3\tFamily 3",
                "sp6\t\t\tno\tf3\tFamily 3",
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


def _stub_resolved_config(
    *,
    execution_stage: str,
    top_features: int = 30,
    tree_path: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(execution_stage=execution_stage, seed=42),
        split=SimpleNamespace(group_col="contrast_pair_id"),
        evaluation=SimpleNamespace(
            group_bootstrap=SimpleNamespace(
                enabled=False,
                n_resamples=2000,
                confidence_level=0.95,
            )
        ),
        report=SimpleNamespace(),
        summary=SimpleNamespace(group_col="family_id", group_name_col="family_name"),
        figures=SimpleNamespace(top_features=top_features),
        model_selection=SimpleNamespace(),
        preprocess=SimpleNamespace(
            sparse_feature_filter=SimpleNamespace(enabled=True),
            low_variance_filter=SimpleNamespace(enabled=True),
            ranked_feature_filter=SimpleNamespace(method="none"),
            correlation_filter=SimpleNamespace(enabled=False),
        ),
        data=SimpleNamespace(
            metadata_path="metadata.tsv",
            tpm_path="tpm.tsv",
            tree_path=tree_path,
            species_col="species",
            feature_col="orthogroup",
            value_col="tpm",
            trait_col="C4",
            contrast_pair_col="contrast_pair_id",
        ),
    )


def _stub_split_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        split_manifest=pl.DataFrame({"fold_id": ["0"], "species": ["sp1"], "pool": ["validation"]}),
        fold_validation_groups=pl.DataFrame(
            {
                "fold_id": ["0"],
                "group_id": ["g1"],
                "n_validation_species": [1],
                "n_validation_pos": [1],
                "n_validation_neg": [0],
                "validation_label_profile": ["positive_only"],
            }
        ),
        fold_diagnostics=pl.DataFrame(
            {
                "fold_id": ["0"],
                "n_train_groups": [1],
                "n_validation_groups": [1],
                "n_train_species": [2],
                "n_train_pos": [1],
                "n_train_neg": [1],
                "n_validation_species": [1],
                "n_validation_pos": [1],
                "n_validation_neg": [0],
                "train_label_profile": ["both"],
                "validation_label_profile": ["positive_only"],
                "two_class_validation_metrics_defined": [False],
            }
        ),
        fold_count=1,
        pool_counts={"validation": 1},
        expression_rows_excluded=0,
    )


def _stub_fingerprint_metadata() -> dict[str, object]:
    return {
        "fingerprint_schema_version": 1,
        "dataset_fingerprint": "a" * 64,
        "split_fingerprint": "b" * 64,
        "experiment_fingerprint": "c" * 64,
        "evaluation_contract": {"evaluation_contract_version": 1},
    }


def _stub_run_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("phenoradar.cli.collect_input_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "phenoradar.cli._build_run_fingerprint_metadata",
        lambda **_kwargs: _stub_fingerprint_metadata(),
    )


def test_prepare_run_inputs_hashes_and_builds_splits_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance_started = Event()
    split_started = Event()
    expected_splits = _stub_split_artifacts()

    def _collect(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        provenance_started.set()
        assert split_started.wait(timeout=5.0)
        return []

    def _split(*_args: object, **_kwargs: object) -> SimpleNamespace:
        split_started.set()
        assert provenance_started.wait(timeout=5.0)
        return expected_splits

    monkeypatch.setattr(cli_mod, "collect_input_files", _collect)
    monkeypatch.setattr(cli_mod, "build_split_artifacts", _split)
    timing_recorder = TimingRecorder()

    input_files, split_artifacts = cli_mod._prepare_run_inputs(
        config_paths=[Path("config.yml")],
        config=_stub_resolved_config(execution_stage="cv_only"),  # type: ignore[arg-type]
        timing_recorder=timing_recorder,
    )

    assert input_files == []
    assert split_artifacts is expected_splits
    assert set(timing_recorder.to_frame().get_column("stage")) == {
        "input_provenance",
        "split_construction",
    }


def _stub_cv_artifacts(
    *,
    ensemble_model_probs: pl.DataFrame | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        thresholds=pl.DataFrame(
            {
                "threshold_name": ["fixed_probability_threshold"],
                "threshold_value": [0.5],
                "source": ["constant"],
                "selection_metric": ["NA"],
                "selection_scope": ["NA"],
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
        loss_by_split_cv=pl.DataFrame(
            {
                "fold_id": ["0", "0"],
                "split": ["train", "validation"],
                "metric": ["log_loss", "log_loss"],
                "metric_value": [0.42, 0.56],
            }
        ),
        feature_importance=pl.DataFrame(
            {
                "feature": ["OG1"],
                "importance_mean": [1.0],
                "importance_std": [0.0],
                "n_models": [1],
                "n_folds": [1],
                "method": ["coef_abs_l1_norm"],
            }
        ),
        feature_importance_by_fold=pl.DataFrame(
            {
                "fold_id": ["0"],
                "feature": ["OG1"],
                "importance_mean": [1.0],
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
                "n_folds": [1],
                "method": ["coef_signed"],
                "reason": ["NA"],
            }
        ),
        coefficients_by_fold=pl.DataFrame(
            {
                "fold_id": ["0"],
                "feature": ["OG1"],
                "coef_mean": [0.2],
                "n_models": [1],
                "method": ["coef_signed"],
                "reason": ["NA"],
            }
        ),
        feature_stability_by_feature=pl.DataFrame(
            {
                "feature": ["OG1"],
                "retained_frequency": [1.0],
                "selection_frequency": [1.0],
                "dominant_sign": ["positive"],
            }
        ),
        feature_stability_by_fold_pair=pl.DataFrame(
            schema={
                "fold_id_a": pl.String,
                "fold_id_b": pl.String,
                "jaccard": pl.Float64,
            }
        ),
        feature_stability_summary=pl.DataFrame(
            {
                "n_outer_folds": [1],
                "n_fold_pairs": [0],
                "jaccard_mean": [None],
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
        top_feature_expression=pl.DataFrame(
            {
                "species": ["sp1", "sp2"],
                "feature": ["OG1", "OG1"],
                "tpm": [1.0, 4.0],
            }
        ),
        ensemble_model_probs=ensemble_model_probs,
        model_selection_trials=None,
        model_selection_trials_summary=None,
        model_selection_selected=None,
        retained_features=pl.DataFrame(
            {
                "scope": ["outer_fold"],
                "fold_id": ["0"],
                "sample_set_id": [0],
                "feature": ["OG1"],
            }
        ),
        retained_features_summary=pl.DataFrame(
            {
                "scope": ["outer_fold"],
                "fold_id": ["0"],
                "feature": ["OG1"],
                "retained_count": [1],
                "n_sample_sets": [1],
                "retained_rate": [1.0],
            }
        ),
    )


def _stub_final_refit_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        pred_external_test=pl.DataFrame(
            {
                "species": ["sp1"],
                "true_label": [1],
                "prob": [0.7],
                "pred_label_fixed_threshold": [1],
            }
        ),
        pred_inference=pl.DataFrame(
            {
                "species": ["sp2"],
                "true_label": [None],
                "prob": [0.6],
                "pred_label_fixed_threshold": [1],
            }
        ),
        loss_by_split_final_refit=pl.DataFrame(
            {
                "split": ["train", "external_test"],
                "metric": ["log_loss", "log_loss"],
                "metric_value": [0.33, 0.55],
            }
        ),
        warnings=[],
        model_selection_selected=None,
        retained_features=pl.DataFrame(
            {
                "scope": ["final_refit"],
                "fold_id": ["NA"],
                "sample_set_id": [0],
                "feature": ["OG1"],
            }
        ),
        retained_features_summary=pl.DataFrame(
            {
                "scope": ["final_refit"],
                "fold_id": ["NA"],
                "feature": ["OG1"],
                "retained_count": [1],
                "n_sample_sets": [1],
                "retained_rate": [1.0],
            }
        ),
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
    assert payload["data"]["tree_path"] is None
    assert payload["data"]["orthogroup_annotation_path"] is None
    assert payload["figures"]["top_features"] == 30


def test_run_passes_top_features_to_run_and_tree_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")
    captured_run_kwargs: dict[str, object] = {}
    captured_tree_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        "phenoradar.cli.load_and_resolve_config",
        lambda *_args, **_kwargs: _stub_resolved_config(
            execution_stage="cv_only",
            top_features=7,
            tree_path="tree.nwk",
        ),
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

    def _capture_run_figures(*_args: object, **kwargs: object) -> list[str]:
        captured_run_kwargs.update(kwargs)
        return []

    def _capture_tree_figures(*_args: object, **kwargs: object) -> list[str]:
        captured_tree_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("phenoradar.cli.write_run_figures", _capture_run_figures)
    monkeypatch.setattr(
        "phenoradar.cli.write_run_tree_prediction_artifacts",
        _capture_tree_figures,
    )
    monkeypatch.setattr("phenoradar.cli.collect_input_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "phenoradar.cli._build_run_fingerprint_metadata",
        lambda **_kwargs: _stub_fingerprint_metadata(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.phenoradar_build_snapshot", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        "phenoradar.cli.runtime_environment_snapshot",
        lambda *_args, **_kwargs: {"python": "test"},
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert captured_run_kwargs["top_features"] == 7
    assert captured_run_kwargs["top_feature_expression"] is not None
    assert captured_run_kwargs["feature_stability_by_feature"] is not None
    assert captured_run_kwargs["feature_stability_by_fold_pair"] is not None
    assert captured_tree_kwargs["feature_limit"] == 7


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
evaluation:
  group_bootstrap:
    enabled: true
    n_resamples: 20
    confidence_level: 0.9
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
    assert (run_dirs[0] / "split" / "tables" / "split_manifest.tsv").exists()
    assert (run_dirs[0] / "split" / "tables" / "fold_validation_groups.tsv").exists()
    assert (run_dirs[0] / "split" / "tables" / "fold_diagnostics.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "metrics_cv.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "group_bootstrap_metrics.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "group_bootstrap_replicates.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "loss_by_split_cv.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "thresholds.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "evaluation_contract.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "feature_importance.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "feature_importance_by_fold.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "coefficients.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "coefficients_by_fold.tsv").exists()
    stability_feature_path = run_dirs[0] / "cv" / "tables" / "feature_stability_by_feature.tsv"
    stability_pair_path = run_dirs[0] / "cv" / "tables" / "feature_stability_by_fold_pair.tsv"
    stability_summary_path = run_dirs[0] / "cv" / "tables" / "feature_stability_summary.tsv"
    assert stability_feature_path.exists()
    assert stability_pair_path.exists()
    assert stability_summary_path.exists()
    assert {
        "feature",
        "retained_frequency",
        "selection_frequency",
        "sign_agreement_rate",
    }.issubset(pl.read_csv(stability_feature_path, separator="\t").columns)
    assert {
        "fold_id_a",
        "fold_id_b",
        "jaccard",
    }.issubset(pl.read_csv(stability_pair_path, separator="\t").columns)
    stability_summary = pl.read_csv(stability_summary_path, separator="\t")
    assert {
        "n_outer_folds",
        "n_fold_pairs",
        "jaccard_mean",
    }.issubset(stability_summary.columns)
    assert stability_summary.get_column("nonzero_tolerance").item() > 0.0
    assert (run_dirs[0] / "model" / "tables" / "feature_filter_counts.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "feature_filter_counts_summary.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "ranked_feature_scores.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "retained_features.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "retained_features_summary.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "model_sparsity.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "model_sparsity_summary.tsv").exists()
    convergence_path = run_dirs[0] / "model" / "tables" / "convergence_diagnostics.tsv"
    assert convergence_path.exists()
    assert {
        "training_scope",
        "fit_scope",
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "model_index",
        "convergence_applicable",
        "converged",
        "n_iter_max",
        "max_iter",
        "convergence_warning_count",
    }.issubset(pl.read_csv(convergence_path, separator="\t").columns)
    assert (run_dirs[0] / "external_test" / "tables" / "prediction_external_test.tsv").exists()
    assert (run_dirs[0] / "inference" / "tables" / "prediction_inference.tsv").exists()
    assert (run_dirs[0] / "external_test" / "tables" / "loss_by_split_final_refit.tsv").exists()
    assert (run_dirs[0] / "summary" / "tables" / "classification_summary.tsv").exists()
    assert (run_dirs[0] / "runtime" / "tables" / "timing.tsv").exists()
    assert (run_dirs[0] / "model_bundle").exists()
    assert (run_dirs[0] / "cv" / "figures" / "cv_metrics_overview.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "group_bootstrap_metrics.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "cv_loss_by_split.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "feature_importance_top.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "top_feature_expression_by_confusion.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "feature_importance_by_fold_heatmap.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "coefficients_signed_top.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "feature_stability_top.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "feature_set_jaccard_heatmap.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "cv_species_probability_by_trait.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "cv_fold_trait_probability.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "roc_curve_cv.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "pr_curve_cv.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "feature_filter_funnel.svg").exists()
    assert (run_dirs[0] / "cv" / "tables" / "group_summary_family.tsv").exists()
    assert (run_dirs[0] / "cv" / "figures" / "probability_by_family.svg").exists()
    funnel_svg = (run_dirs[0] / "cv" / "figures" / "feature_filter_funnel.svg").read_text(
        encoding="utf-8"
    )
    assert "Input" in funnel_svg
    assert "Sparse feature" in funnel_svg
    assert "sparse_feature" not in funnel_svg
    assert "Low variance" not in funnel_svg
    assert "Pair aware" not in funnel_svg
    assert "Correlation" not in funnel_svg
    assert ">Final<" not in funnel_svg
    assert not (
        run_dirs[0] / "cv" / "figures" / "selected_features_by_fold_after_preprocessing.svg"
    ).exists()
    assert not (
        run_dirs[0] / "cv" / "figures" / "selected_features_after_preprocessing.svg"
    ).exists()
    assert not (run_dirs[0] / "cv" / "figures" / "selected_features_by_fold.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "non_zero_feature_count_by_fold.svg").exists()
    assert not (run_dirs[0] / "cv" / "figures" / "selected_feature_count_by_fold.svg").exists()
    assert not (run_dirs[0] / "cv" / "figures" / "model_sparsity_scatter.svg").exists()
    assert (run_dirs[0] / "external_test" / "figures" / "final_refit_loss_by_split.svg").exists()
    assert (run_dirs[0] / "external_test" / "figures" / "feature_filter_funnel.svg").exists()
    assert (
        run_dirs[0] / "external_test" / "figures" / "external_species_probability_by_trait.svg"
    ).exists()
    assert (
        run_dirs[0] / "external_test" / "figures" / "external_confusion_matrix.svg"
    ).exists()
    assert (
        run_dirs[0] / "external_test" / "figures" / "cv_external_metric_comparison.svg"
    ).exists()
    assert (
        run_dirs[0] / "inference" / "figures" / "inference_probability_distribution.svg"
    ).exists()
    assert (
        run_dirs[0] / "inference" / "figures" / "species_probability_cv_and_inference.svg"
    ).exists()
    assert (run_dirs[0] / "external_test" / "tables" / "group_summary_family.tsv").exists()
    assert (
        run_dirs[0] / "external_test" / "figures" / "probability_by_family.svg"
    ).exists()
    assert (run_dirs[0] / "inference" / "tables" / "group_summary_family.tsv").exists()
    assert (run_dirs[0] / "inference" / "figures" / "probability_by_family.svg").exists()
    cv_trait_svg = (
        run_dirs[0] / "cv" / "figures" / "cv_species_probability_by_trait.svg"
    ).read_text(encoding="utf-8")
    assert "C4" in cv_trait_svg
    assert "C4=0" not in cv_trait_svg
    assert "C4=1" not in cv_trait_svg
    pr_curve_svg = (run_dirs[0] / "cv" / "figures" / "pr_curve_cv.svg").read_text(
        encoding="utf-8"
    )
    assert "Average Precision=" in pr_curve_svg
    assert "PR AUC=" not in pr_curve_svg

    metrics = pl.read_csv(run_dirs[0] / "cv" / "tables" / "metrics_cv.tsv", separator="\t")
    assert {"aggregate_scope", "fold_id", "metric", "metric_value"}.issubset(metrics.columns)
    bootstrap_metrics = pl.read_csv(
        run_dirs[0] / "cv" / "tables" / "group_bootstrap_metrics.tsv",
        separator="\t",
        null_values="NA",
    )
    assert set(bootstrap_metrics.get_column("metric")) == {
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "brier",
        "log_loss",
    }
    assert bootstrap_metrics.get_column("group_col").unique().to_list() == [
        "contrast_pair_id"
    ]
    assert bootstrap_metrics.get_column("n_groups").unique().to_list() == [2]
    assert bootstrap_metrics.get_column("n_resamples").unique().to_list() == [20]
    assert bootstrap_metrics.get_column("confidence_level").unique().to_list() == [0.9]
    bootstrap_replicates = pl.read_csv(
        run_dirs[0] / "cv" / "tables" / "group_bootstrap_replicates.tsv",
        separator="\t",
        null_values="NA",
    )
    assert bootstrap_replicates.height == 20 * 6
    assert bootstrap_replicates.get_column("resample_id").n_unique() == 20
    timing = pl.read_csv(
        run_dirs[0] / "runtime" / "tables" / "timing.tsv",
        separator="\t",
        null_values="NA",
    )
    assert {
        "scope",
        "stage",
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "started_at_sec",
        "ended_at_sec",
        "duration_sec",
    } == set(timing.columns)
    run_timing_stages = set(
        timing.filter(pl.col("scope") == "run").get_column("stage")
    )
    assert {
        "config_resolution",
        "split_construction",
        "outer_cv",
        "group_bootstrap",
        "final_refit",
        "artifact_writing",
        "figure_generation",
        "total",
    }.issubset(run_timing_stages)
    assert timing.filter(pl.col("duration_sec") < 0.0).height == 0
    fold_validation_groups = pl.read_csv(
        run_dirs[0] / "split" / "tables" / "fold_validation_groups.tsv",
        separator="\t",
    )
    assert {
        "fold_id",
        "group_id",
        "n_validation_species",
        "n_validation_pos",
        "n_validation_neg",
        "validation_label_profile",
    }.issubset(fold_validation_groups.columns)
    fold_diagnostics = pl.read_csv(
        run_dirs[0] / "split" / "tables" / "fold_diagnostics.tsv",
        separator="\t",
    )
    assert {
        "fold_id",
        "n_train_groups",
        "n_validation_groups",
        "n_train_species",
        "n_train_pos",
        "n_train_neg",
        "n_validation_species",
        "n_validation_pos",
        "n_validation_neg",
        "train_label_profile",
        "validation_label_profile",
        "two_class_validation_metrics_defined",
    }.issubset(fold_diagnostics.columns)
    thresholds = pl.read_csv(run_dirs[0] / "model" / "tables" / "thresholds.tsv", separator="\t")
    assert set(thresholds.select("threshold_name").to_series().to_list()) == {
        "fixed_probability_threshold",
    }
    assert thresholds.row(0, named=True)["policy"] == "fixed_constant"
    assert thresholds.row(0, named=True)["derived_from_cv"] is False
    training_group_subsets = pl.read_csv(
        run_dirs[0] / "model" / "tables" / "training_group_subsets.tsv",
        separator="\t",
        null_values="NA",
    )
    assert set(training_group_subsets.get_column("scope")) == {
        "outer_fold",
        "final_refit",
    }
    assert training_group_subsets.get_column("group_col").unique().to_list() == [
        "contrast_pair_id"
    ]
    evaluation_contract = pl.read_csv(
        run_dirs[0] / "model" / "tables" / "evaluation_contract.tsv",
        separator="\t",
        null_values="NA",
    )
    pr_auc_contract = evaluation_contract.filter(pl.col("metric_name") == "pr_auc").row(
        0, named=True
    )
    assert pr_auc_contract["display_name"] == "Average Precision"
    assert pr_auc_contract["implementation"] == "sklearn.metrics.average_precision_score"
    assert pr_auc_contract["threshold_name"] is None
    pred_external = pl.read_csv(
        run_dirs[0] / "external_test" / "tables" / "prediction_external_test.tsv",
        separator="\t",
    )
    pred_inference = pl.read_csv(
        run_dirs[0] / "inference" / "tables" / "prediction_inference.tsv",
        separator="\t",
    )
    classification_summary = pl.read_csv(
        run_dirs[0] / "summary" / "tables" / "classification_summary.tsv",
        separator="\t",
    )
    assert {
        "species",
        "true_label",
        "prob",
        "pred_label_fixed_threshold",
    }.issubset(pred_external.columns)
    assert {
        "species",
        "true_label",
        "prob",
        "pred_label_fixed_threshold",
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
    assert classification_summary.height == 4
    assert set(classification_summary.select("pool").to_series().to_list()) == {
        "validation_oof",
        "external_test",
    }
    assert set(classification_summary.select("threshold_name").to_series().to_list()) == {
        "fixed_probability_threshold",
    }
    run_metadata = yaml.safe_load((run_dirs[0] / "run_metadata.json").read_text(encoding="utf-8"))
    assert "git_commit" in run_metadata
    assert "git_dirty" in run_metadata
    assert "git_worktree_patch_sha256" in run_metadata
    assert run_metadata["provenance_schema_version"] == 1
    assert isinstance(run_metadata["phenoradar_version"], str)
    assert run_metadata["git_source"] in {"phenoradar_source", "unavailable"}
    assert "seed_policy" in run_metadata
    assert run_metadata["seed_policy"]["runtime_seed"] == 42
    assert run_metadata["group_bootstrap"] == {
        "group_col": "contrast_pair_id",
        "n_groups": 2,
        "n_resamples": 20,
        "confidence_level": 0.9,
        "bootstrap_method": "percentile_group",
        "seed": bootstrap_metrics.get_column("seed").item(0),
    }
    assert run_metadata["timing"]["artifact_path"] == "runtime/tables/timing.tsv"
    assert run_metadata["timing"]["clock"] == "time.perf_counter"
    assert run_metadata["timing"]["parallel_intervals_may_overlap"] is True
    assert set(run_metadata["timing"]["stage_duration_sec"]) == run_timing_stages
    assert "environment" in run_metadata
    assert "input_files" in run_metadata
    assert isinstance(run_metadata["input_files"], list)
    assert run_metadata["fingerprint_schema_version"] == 1
    assert len(run_metadata["dataset_fingerprint"]) == 64
    assert len(run_metadata["split_fingerprint"]) == 64
    assert len(run_metadata["experiment_fingerprint"]) == 64
    assert run_metadata["evaluation_contract"]["trait_col"] == "C4"
    metric_contract = run_metadata["evaluation_contract"]["metric_contract"]
    assert metric_contract["classification_threshold"]["threshold_value"] == 0.5
    assert metric_contract["classification_threshold"]["derived_from_cv"] is False
    assert metric_contract["metrics"]["pr_auc"]["display_name"] == "Average Precision"


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
    assert "features_after_sparse_feature_filter=" in result.output
    assert "features_after_low_variance=" in result.output
    assert "features_after_correlation=" in result.output
    assert "features_after=" in result.output

    run_dirs = sorted((tmp_path / "runs").glob("*_run_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "cv" / "tables" / "feature_importance.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "feature_importance_by_fold.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "coefficients.tsv").exists()
    assert (run_dirs[0] / "cv" / "tables" / "coefficients_by_fold.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "feature_filter_counts.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "feature_filter_counts_summary.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "ranked_feature_scores.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "retained_features.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "retained_features_summary.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "model_sparsity.tsv").exists()
    assert (run_dirs[0] / "model" / "tables" / "model_sparsity_summary.tsv").exists()
    assert (run_dirs[0] / "cv" / "figures" / "cv_metrics_overview.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "cv_loss_by_split.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "feature_importance_top.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "feature_importance_by_fold_heatmap.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "coefficients_signed_top.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "cv_species_probability_by_trait.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "cv_fold_trait_probability.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "roc_curve_cv.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "pr_curve_cv.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "feature_filter_funnel.svg").exists()
    assert (run_dirs[0] / "cv" / "tables" / "group_summary_family.tsv").exists()
    assert (run_dirs[0] / "cv" / "figures" / "probability_by_family.svg").exists()
    assert not (
        run_dirs[0] / "cv" / "figures" / "selected_features_by_fold_after_preprocessing.svg"
    ).exists()
    assert not (
        run_dirs[0] / "cv" / "figures" / "selected_features_after_preprocessing.svg"
    ).exists()
    assert not (run_dirs[0] / "cv" / "figures" / "selected_features_by_fold.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "non_zero_feature_count_by_fold.svg").exists()
    assert not (run_dirs[0] / "cv" / "figures" / "selected_feature_count_by_fold.svg").exists()
    assert not (run_dirs[0] / "cv" / "figures" / "model_sparsity_scatter.svg").exists()
    assert not (
        run_dirs[0] / "external_test" / "figures" / "final_refit_loss_by_split.svg"
    ).exists()
    assert not (
        run_dirs[0] / "external_test" / "figures" / "feature_filter_funnel.svg"
    ).exists()
    assert not (
        run_dirs[0] / "external_test" / "figures" / "external_species_probability_by_trait.svg"
    ).exists()
    assert not (
        run_dirs[0] / "external_test" / "figures" / "cv_external_metric_comparison.svg"
    ).exists()
    assert (run_dirs[0] / "summary" / "tables" / "classification_summary.tsv").exists()
    assert not (run_dirs[0] / "external_test" / "tables" / "prediction_external_test.tsv").exists()
    assert not (run_dirs[0] / "inference" / "tables" / "prediction_inference.tsv").exists()
    assert not (run_dirs[0] / "model_bundle").exists()
    classification_summary = pl.read_csv(
        run_dirs[0] / "summary" / "tables" / "classification_summary.tsv",
        separator="\t",
    )
    assert classification_summary.height == 3
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
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout\tfamily_id\tfamily_name",
                "g1_pos\t1\tg1\tno\tf1\tFamily 1",
                "g1_neg\t0\tg1\tno\tf1\tFamily 1",
                "g2_pos\t1\tg2\tno\tf2\tFamily 2",
                "g2_neg\t0\tg2\tno\tf2\tFamily 2",
                "g3_pos\t1\tg3\tno\tf3\tFamily 3",
                "g3_neg\t0\tg3\tno\tf3\tFamily 3",
                "g4_pos\t1\tg4\tno\tf4\tFamily 4",
                "g4_neg\t0\tg4\tno\tf4\tFamily 4",
                "ext1\t1\t\tyes\tf5\tFamily 5",
                "inf1\t\t\tno\tf5\tFamily 5",
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
    selected_path = run_dirs[0] / "model" / "tables" / "model_selection_selected.tsv"
    trials_path = run_dirs[0] / "cv" / "tables" / "model_selection_trials.tsv"
    trials_summary_path = run_dirs[0] / "cv" / "tables" / "model_selection_trials_summary.tsv"
    assert selected_path.exists()
    assert trials_path.exists()
    assert trials_summary_path.exists()
    assert (run_dirs[0] / "cv" / "figures" / "model_selection_trials.svg").exists()
    assert (run_dirs[0] / "cv" / "figures" / "model_selection_one_se_curve.svg").exists()
    assert not (run_dirs[0] / "cv" / "figures" / "selected_hyperparameter_stability.svg").exists()

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
        "metric_value_se",
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
    monkeypatch.setattr(
        "phenoradar.cli._write_group_summary_artifacts",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr("phenoradar.cli.collect_input_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "phenoradar.cli._build_run_fingerprint_metadata",
        lambda **_kwargs: _stub_fingerprint_metadata(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.phenoradar_build_snapshot", lambda *_args, **_kwargs: {}
    )
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
    assert (predict_dirs[0] / "inference" / "tables" / "prediction_inference.tsv").exists()
    assert (predict_dirs[0] / "inference" / "tables" / "group_summary_family.tsv").exists()
    assert (predict_dirs[0] / "inference" / "figures" / "probability_by_family.svg").exists()
    assert not (predict_dirs[0] / "pred_predict.tsv").exists()
    assert (predict_dirs[0] / "resolved_config.yml").exists()
    assert (predict_dirs[0] / "run_metadata.json").exists()
    assert (
        predict_dirs[0] / "inference" / "figures" / "predict_probability_distribution.svg"
    ).exists()
    assert not (predict_dirs[0] / "cv" / "tables" / "metrics_cv.tsv").exists()

    pred_inference = pl.read_csv(
        predict_dirs[0] / "inference" / "tables" / "prediction_inference.tsv",
        separator="\t",
    )
    assert {
        "species",
        "true_label",
        "prob",
        "pred_label_fixed_threshold",
    }.issubset(pred_inference.columns)
    assert pred_inference.height == 6
    assert pred_inference.get_column("true_label").to_list() == ["NA"] * pred_inference.height
    predict_metadata = yaml.safe_load(
        (predict_dirs[0] / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert "git_commit" in predict_metadata
    assert "git_dirty" in predict_metadata
    assert "git_worktree_patch_sha256" in predict_metadata
    assert predict_metadata["provenance_schema_version"] == 1
    assert isinstance(predict_metadata["phenoradar_version"], str)
    assert predict_metadata["git_source"] in {"phenoradar_source", "unavailable"}
    assert "environment" in predict_metadata
    assert "input_files" in predict_metadata
    assert "model_bundle_manifest_sha256" in predict_metadata
    assert "model_bundle_payload_sha256" in predict_metadata
    assert "bundle_source_run_dir" in predict_metadata
    assert predict_metadata["bundle_source_provenance_schema_version"] == 1
    assert isinstance(predict_metadata["bundle_source_phenoradar_version"], str)
    assert isinstance(predict_metadata["bundle_source_git_commit"], str)
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


def test_report_cli_ranks_brier_in_ascending_order(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    for run_id, start_time, brier in [
        ("20260101T000001Z_run_worst", "2026-01-01T00:00:00+00:00", 0.40),
        ("20260101T000002Z_run_best", "2026-01-02T00:00:00+00:00", 0.05),
    ]:
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True)
        _write(
            run_dir / "run_metadata.json",
            json.dumps(
                {
                    "command": "run",
                    "execution_stage": "full_run",
                    "status": "ok",
                    "start_time": start_time,
                    "end_time": start_time,
                    "duration_sec": 1.0,
                    "warnings": [],
                }
            )
            + "\n",
        )
        _write(run_dir / "resolved_config.yml", "runtime:\n  seed: 42\n")
        pl.DataFrame(
            {
                "aggregate_scope": ["macro"],
                "fold_id": ["NA"],
                "metric": ["brier"],
                "metric_value": [brier],
            }
        ).write_csv(run_dir / "metrics_cv.tsv", separator="\t")

    report_dir = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "report",
            "--runs-root",
            str(runs_root),
            "--primary-metric",
            "brier",
            "--out",
            str(report_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    ranking = pl.read_csv(report_dir / "report_ranking.tsv", separator="\t")
    assert ranking.select("run_id").to_series().to_list() == [
        "20260101T000002Z_run_best",
        "20260101T000001Z_run_worst",
    ]
    assert ranking.select("metric_value").to_series().to_list() == [0.05, 0.40]


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


def test_run_expands_scalar_lists_into_ordered_study_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(
        tmp_path / "config.yml",
        """
preprocess:
  ranked_feature_filter:
    method: [none, pair_aware]
    max_features: 100
sampling:
  strategy: all_samples
  max_samples_per_label_per_group: null
  sampled_set_count: 1
""".lstrip(),
    )
    split_artifacts = _stub_split_artifacts()
    split_artifacts.split_manifest = pl.DataFrame(
        {
            "species": ["sp1"],
            "pool": ["validation"],
            "fold_id": ["0"],
            "group_id": ["g1"],
            "contrast_group_id": ["g1"],
            "label": [1],
        }
    )
    split_call_count = 0
    cv_call_count = 0

    def _split(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal split_call_count
        split_call_count += 1
        return split_artifacts

    def _cv(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal cv_call_count
        cv_call_count += 1
        return _stub_cv_artifacts()

    def _fingerprints(**kwargs: object) -> dict[str, object]:
        payload = _stub_fingerprint_metadata()
        split_manifest = kwargs["split_manifest"]
        assert isinstance(split_manifest, pl.DataFrame)
        payload["split_fingerprint"] = cli_mod.split_fingerprint(split_manifest)
        return payload

    monkeypatch.setattr("phenoradar.cli.build_split_artifacts", _split)
    monkeypatch.setattr("phenoradar.cli.run_outer_cv", _cv)
    monkeypatch.setattr("phenoradar.cli.collect_input_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("phenoradar.cli._build_run_fingerprint_metadata", _fingerprints)
    monkeypatch.setattr("phenoradar.cli.write_run_figures", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "phenoradar.cli._write_group_summary_artifacts",
        lambda *_args, **_kwargs: [],
    )

    result = runner.invoke(app, ["run", "-c", str(config), "--quiet"])

    assert result.exit_code == 0, result.output
    study_dirs = sorted((tmp_path / "runs").glob("*_study_*"))
    assert len(study_dirs) == 1
    study_dir = study_dirs[0]
    manifest = pl.read_csv(
        study_dir / "condition_manifest.tsv",
        separator="\t",
        null_values="NA",
    ).sort("condition_index")
    assert manifest.get_column("status").to_list() == ["completed", "completed"]
    assert manifest.get_column("condition_index").to_list() == [1, 2]
    assert [
        json.loads(value)["preprocess.ranked_feature_filter.method"]
        for value in manifest.get_column("varying_parameters_json")
    ] == ["none", "pair_aware"]
    assert split_call_count == 1
    assert cv_call_count == 2
    assert (study_dir / "tables" / "condition_metrics.tsv").exists()
    assert (study_dir / "tables" / "pairwise_comparisons.tsv").exists()
    differences = pl.read_csv(study_dir / "config_differences.tsv", separator="\t")
    assert differences.get_column("preprocess.ranked_feature_filter.method").to_list() == [
        "none",
        "pair_aware",
    ]
    for extension in ("svg", "pdf", "png"):
        assert (study_dir / "figures" / f"condition_metrics.{extension}").exists()
        assert not (study_dir / "figures" / f"pairwise_improvement.{extension}").exists()

    resumed = runner.invoke(
        app,
        ["run", "-c", str(config), "--resume", str(study_dir), "--quiet"],
    )

    assert resumed.exit_code == 0, resumed.output
    assert split_call_count == 2
    assert cv_call_count == 2


def test_run_fails_when_split_artifact_build_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yml", "{}\n")
    _stub_run_provenance(monkeypatch)

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
    _stub_run_provenance(monkeypatch)

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
    _stub_run_provenance(monkeypatch)

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
    _stub_run_provenance(monkeypatch)

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
    _stub_run_provenance(monkeypatch)

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
    monkeypatch.setattr(
        "phenoradar.cli._build_run_fingerprint_metadata",
        lambda **_kwargs: _stub_fingerprint_metadata(),
    )
    monkeypatch.setattr(
        "phenoradar.cli.phenoradar_build_snapshot", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        "phenoradar.cli.runtime_environment_snapshot",
        lambda *_args, **_kwargs: {"python": "test"},
    )

    result = runner.invoke(app, ["run", "-c", str(config)])

    assert result.exit_code == 0, result.output
    run_dirs = sorted((tmp_path / "runs").glob("*_run_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "cv" / "tables" / "ensemble_model_probs.tsv").exists()


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


def test_dataset_copies_bundled_compact_dataset_by_default(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "c4_dataset"

    result = runner.invoke(
        app,
        [
            "dataset",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "bundled package data" in result.output
    assert (out_dir / "species_metadata.tsv").exists()
    assert (out_dir / "species_trait.tsv").exists()
    assert (out_dir / "ncbi_tree.nwk").exists()
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
