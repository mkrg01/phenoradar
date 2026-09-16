from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from phenoradar.bundle import (
    BundleError,
    export_model_bundle,
    load_model_bundle,
    predict_with_bundle,
)
from phenoradar.candidate_evidence import build_candidate_evidence_artifacts
from phenoradar.cli import app
from phenoradar.config import AppConfig, write_resolved_config
from phenoradar.cv import run_final_refit, run_outer_cv
from phenoradar.split import build_split_artifacts


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("predict-bundle")
    metadata = root / "metadata.tsv"
    metadata.write_text(
        "species\tC4\tcontrast_pair_id\nsp1\t0\tg1\nsp2\t1\tg1\nsp3\t0\tg2\nsp4\t1\tg2\nnovel\t\t\n"
    )
    tpm = root / "tpm.tsv"
    tpm.write_text(
        "species\torthogroup\ttpm\n"
        "sp1\tOG1\t1\nsp2\tOG1\t10\nsp3\tOG1\t2\nsp4\tOG1\t20\n"
        "novel\tOG1\t5\nnovel\tOG1\t1\n"
    )
    config = AppConfig.model_validate(
        {"data": {"metadata_path": str(metadata), "tpm_path": str(tpm)}}
    )
    split = build_split_artifacts(config)
    cv = run_outer_cv(config, split.split_manifest)
    refit = run_final_refit(config, split.split_manifest)
    resolved = root / "resolved_config.yml"
    write_resolved_config(config, resolved)
    evidence = build_candidate_evidence_artifacts(
        config=config, split_manifest=split.split_manifest, final_refit=refit,
        cross_fold_predictions=None, top_features=30, include_model_reference=True,
    )
    exported = export_model_bundle(
        run_dir=root,
        resolved_config_path=resolved,
        config=config,
        final_refit_artifacts=refit,
        thresholds=cv.thresholds,
        reference_expression=evidence.reference_expression,
    )
    return exported.bundle_dir, tpm, metadata


@pytest.mark.parametrize("use_config", [False, True])
def test_predict_all_tpm_species_without_metadata_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_bundle: tuple[Path, Path, Path],
    use_config: bool,
) -> None:
    bundle_dir, tpm, metadata = trained_bundle
    expected, _ = predict_with_bundle(
        AppConfig.model_validate({"data": {"metadata_path": str(metadata), "tpm_path": str(tpm)}}),
        load_model_bundle(bundle_dir),
    )
    monkeypatch.chdir(tmp_path)

    def no_training(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Prediction must not train")

    for name in ["run_outer_cv", "run_final_refit", "build_split_artifacts"]:
        monkeypatch.setattr(f"phenoradar.cli.{name}", no_training)
    args = ["predict", "--model-bundle", str(bundle_dir), "--n-jobs", "2"]
    if use_config:
        config = tmp_path / "predict.yml"
        config.write_text(yaml.safe_dump({"data": {"tpm_path": str(tpm)}}))
        args += ["-c", str(config)]
    else:
        args += ["--tpm-path", str(tpm)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").glob("*_predict_*"))
    actual = pl.read_csv(run / "inference/tables/prediction_inference.tsv", separator="\t")
    assert actual["species"].to_list() == expected["species"].to_list()
    np.testing.assert_allclose(actual["prob"], expected["prob"], atol=1e-8)
    resolved = yaml.safe_load((run / "resolved_config.yml").read_text())
    assert resolved["data"]["metadata_path"] is None
    assert resolved["runtime"] == {"n_jobs": 2}
    assert "model" not in resolved and "split" not in resolved
    assert resolved["preprocess"] == {"max_pivot_cells": 50_000_000}
    provenance = json.loads((run / "run_metadata.json").read_text())
    assert not any("metadata.tsv" in str(item) for item in provenance["input_files"])
    assert not any("group summary" in message for message in provenance["warnings"])


def test_predict_generates_species_evidence_from_portable_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trained_bundle: tuple[Path, Path, Path],
) -> None:
    bundle_dir = tmp_path / "portable_bundle"
    shutil.copytree(trained_bundle[0], bundle_dir)
    monkeypatch.chdir(tmp_path)
    tpm = tmp_path / "new.tsv"
    tpm.write_text("species\torthogroup\ttpm\nnew_species\tOG1\t100\n")
    result = CliRunner().invoke(app, [
        "predict", "--model-bundle", str(bundle_dir), "--tpm-path", str(tpm),
    ])
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").glob("*_predict_*"))
    figures = run / "inference/figures/candidate_evidence"
    assert len(list(figures.glob("*/*.pdf"))) == 1
    manifest = pl.read_csv(figures / "candidate_manifest.tsv", separator="\t")
    assert manifest["species"].to_list() == ["new_species"]
    assert manifest["n_bundle_models"].to_list() == [1]
    reference = pl.read_csv(
        run / "inference/tables/candidate_reference_expression.tsv", separator="\t"
    )
    assert reference["species"].n_unique() == 4
    assert "novel" not in reference["species"].to_list()
    assert (run / "inference/tables/candidate_model_probabilities.tsv").exists()


def test_bundle_verifies_optional_interpretation_snapshot(
    tmp_path: Path, trained_bundle: tuple[Path, Path, Path],
) -> None:
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(trained_bundle[0], bundle_dir)
    assert load_model_bundle(bundle_dir).reference_expression is not None
    with (bundle_dir / "reference_expression.parquet").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(BundleError, match="integrity check failed"):
        load_model_bundle(bundle_dir)


def test_predict_cli_overrides_config_and_selects_metadata_species(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_bundle: tuple[Path, Path, Path],
) -> None:
    bundle_dir, tpm, _ = trained_bundle
    monkeypatch.chdir(tmp_path)
    metadata = tmp_path / "subset.tsv"
    metadata.write_text("species\nnovel\n")
    config = tmp_path / "predict.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {"tpm_path": "missing.tsv", "metadata_path": "missing-metadata.tsv"},
                "runtime": {"n_jobs": 0},
                # Training settings have no effect and need not satisfy training validation.
                "preprocess": {"missing_expression": {"method": "neutral"}},
                "model": {"name": "random_forest"},
            }
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(bundle_dir),
            "-c",
            str(config),
            "--tpm-path",
            str(tpm),
            "--metadata-path",
            str(metadata),
            "--n-jobs",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").glob("*_predict_*"))
    prediction = pl.read_csv(run / "inference/tables/prediction_inference.tsv", separator="\t")
    assert prediction["species"].to_list() == ["novel"]
    resolved = yaml.safe_load((run / "resolved_config.yml").read_text())
    assert resolved["data"]["tpm_path"] == str(tpm)
    assert resolved["data"]["metadata_path"] == str(metadata)
    assert resolved["runtime"]["n_jobs"] == 2


def test_predict_accepts_custom_columns_with_explicit_null_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_bundle: tuple[Path, Path, Path],
) -> None:
    monkeypatch.chdir(tmp_path)
    tpm = tmp_path / "custom.tsv"
    tpm.write_text("taxon\tog\texpression\nnovel\tOG1\t6\n")
    config = tmp_path / "predict.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "tpm_path": str(tpm),
                    "metadata_path": None,
                    "species_col": "taxon",
                    "feature_col": "og",
                    "value_col": "expression",
                }
            }
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(trained_bundle[0]),
            "-c",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").glob("*_predict_*"))
    prediction = pl.read_csv(run / "inference/tables/prediction_inference.tsv", separator="\t")
    assert prediction["species"].to_list() == ["novel"]


@pytest.mark.parametrize(
    "contents, error",
    [
        ("species\torthogroup\ttpm\n", "zero valid species"),
        ("species\torthogroup\ttpm\n\tOG1\t2\n", "empty species names"),
        ("sample\torthogroup\ttpm\nsp1\tOG1\t2\n", "missing species column"),
        ("species\torthogroup\ttpm\nsp1\tOG1\t-1\n", "negative"),
    ],
)
def test_predict_rejects_invalid_tpm_without_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_bundle: tuple[Path, Path, Path],
    contents: str,
    error: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    tpm = tmp_path / "invalid.tsv"
    tpm.write_text(contents)
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(trained_bundle[0]),
            "--tpm-path",
            str(tpm),
        ],
    )
    assert result.exit_code != 0
    assert error in result.output
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "config_text, error",
    [
        (None, "Provide --tpm-path"),
        ("{}\n", "Provide --tpm-path"),
        ("data:\n  tpm_paths: file.tsv\n", "Unknown data field"),
        ("runtime:\n  n_jobs: 0\n", "greater than 0"),
    ],
)
def test_predict_requires_explicit_tpm_and_valid_predict_settings(
    tmp_path: Path,
    trained_bundle: tuple[Path, Path, Path],
    config_text: str | None,
    error: str,
) -> None:
    args = ["predict", "--model-bundle", str(trained_bundle[0])]
    if config_text is not None:
        config = tmp_path / "config.yml"
        config.write_text(config_text)
        args += ["-c", str(config)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert error in result.output


@pytest.mark.parametrize("with_metadata", [False, True])
def test_predict_tree_accepts_missing_metadata_and_species_only_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_bundle: tuple[Path, Path, Path],
    with_metadata: bool,
) -> None:
    bundle_dir, tpm, _ = trained_bundle
    monkeypatch.chdir(tmp_path)
    tree = tmp_path / "tree.nwk"
    tree.write_text("((sp1:1,sp2:1):1,(sp3:1,sp4:1):1,novel:1);")
    data = {"tpm_path": str(tpm), "tree_path": str(tree)}
    if with_metadata:
        metadata = tmp_path / "metadata.tsv"
        metadata.write_text("species\nnovel\n")
        data["metadata_path"] = str(metadata)
    config = tmp_path / "predict.yml"
    config.write_text(yaml.safe_dump({"data": data}))
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(bundle_dir),
            "-c",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").glob("*_predict_*"))
    annotations = pl.read_csv(
        run / "inference/tables/tree_prediction_predict_annotation.tsv",
        separator="\t",
    )
    assert annotations.height == (1 if with_metadata else 5)
