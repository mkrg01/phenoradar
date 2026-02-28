from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import phenoradar.bundle as bundle_mod
from phenoradar.bundle import (
    BundleError,
    export_model_bundle,
    load_model_bundle,
    predict_with_bundle,
)
from phenoradar.config import load_and_resolve_config, write_resolved_config
from phenoradar.cv import run_final_refit, run_outer_cv
from phenoradar.split import build_split_artifacts


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _fixture_data(tmp_path: Path) -> tuple[Path, Path]:
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
                "sp1\tOG2\t0.5",
                "sp2\tOG1\t2.0",
                "sp2\tOG2\t0.3",
                "sp3\tOG1\t3.0",
                "sp3\tOG2\t2.0",
                "sp4\tOG1\t4.0",
                "sp4\tOG2\t0.1",
                "sp5\tOG1\t5.0",
                "sp5\tOG2\t0.9",
                "sp6\tOG1\t6.0",
                "sp6\tOG2\t0.2",
            ]
        )
        + "\n",
    )
    return metadata, tpm


def _config_path(tmp_path: Path, metadata: Path, tpm: Path) -> Path:
    return _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )


def _export_and_load_bundle(tmp_path: Path, metadata: Path, tpm: Path) -> tuple[object, object]:
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)
    cv_threshold = (
        cv_artifacts.thresholds.filter(pl.col("threshold_name") == "cv_derived_threshold")
        .select("threshold_value")
        .to_series()
        .item()
    )
    refit = run_final_refit(config, split_artifacts.split_manifest, float(cv_threshold))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved_config_path = run_dir / "resolved_config.yml"
    write_resolved_config(config, resolved_config_path)
    export_result = export_model_bundle(
        run_dir=run_dir,
        resolved_config_path=resolved_config_path,
        config=config,
        final_refit_artifacts=refit,
        thresholds=cv_artifacts.thresholds,
    )
    return config, load_model_bundle(export_result.bundle_dir)


def _prepare_export_inputs(
    tmp_path: Path,
) -> tuple[object, object, object, Path, Path]:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)
    cv_threshold = (
        cv_artifacts.thresholds.filter(pl.col("threshold_name") == "cv_derived_threshold")
        .select("threshold_value")
        .to_series()
        .item()
    )
    refit = run_final_refit(config, split_artifacts.split_manifest, float(cv_threshold))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved_config_path = run_dir / "resolved_config.yml"
    write_resolved_config(config, resolved_config_path)
    return config, cv_artifacts, refit, run_dir, resolved_config_path


def _manifest_payload(bundle_dir: Path) -> dict[str, object]:
    return json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))


def _manifest_payload_without_self_inventory(bundle_dir: Path) -> dict[str, object]:
    manifest = _manifest_payload(bundle_dir)
    files = dict(manifest["files"])  # type: ignore[index]
    files.pop("bundle_manifest.json", None)
    manifest["files"] = files
    return manifest


def test_render_manifest_payload_requires_files_inventory() -> None:
    with pytest.raises(BundleError, match="missing 'files' inventory"):
        bundle_mod._render_manifest_payload({}, manifest_sha="0" * 64, manifest_size=0)


def test_resolve_manifest_size_raises_when_not_converged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _never_converges(
        _manifest_base: dict[str, object], *, manifest_sha: str, manifest_size: int
    ) -> str:
        _ = manifest_sha, manifest_size
        calls["n"] += 1
        return "x" * calls["n"]

    monkeypatch.setattr(bundle_mod, "_render_manifest_payload", _never_converges)

    with pytest.raises(BundleError, match="Failed to resolve deterministic"):
        bundle_mod._resolve_manifest_size({"files": {}})


def test_run_git_returns_none_on_file_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise_file_not_found(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(bundle_mod.subprocess, "run", _raise_file_not_found)
    assert bundle_mod._run_git(["git", "status"], cwd=tmp_path) is None


def test_run_git_returns_none_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        bundle_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert bundle_mod._run_git(["git", "status"], cwd=tmp_path) is None


def test_run_git_returns_stripped_stdout_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        bundle_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="  abc123\n"),
    )
    assert bundle_mod._run_git(["git", "rev-parse", "HEAD"], cwd=tmp_path) == "abc123"


def test_threshold_value_raises_when_not_exactly_one_row() -> None:
    thresholds = pl.DataFrame(
        {
            "threshold_name": ["fixed_probability_threshold", "fixed_probability_threshold"],
            "threshold_value": [0.5, 0.6],
        }
    )
    with pytest.raises(BundleError, match="Expected exactly one threshold value"):
        bundle_mod._threshold_value(thresholds, "fixed_probability_threshold")


def test_load_feature_schema_raises_when_no_features(tmp_path: Path) -> None:
    schema_path = tmp_path / "feature_schema.tsv"
    pl.DataFrame(
        schema={
            "feature": pl.String,
            "feature_index": pl.Int64,
        }
    ).write_csv(schema_path, separator="\t")

    with pytest.raises(BundleError, match="does not contain any features"):
        bundle_mod._load_feature_schema(schema_path)


def test_predict_probability_raises_for_invalid_shape() -> None:
    class _BadEstimator:
        def predict_proba(self, _x: np.ndarray) -> np.ndarray:
            return np.array([0.1, 0.9], dtype=float)

    with pytest.raises(BundleError, match="invalid probability shape"):
        bundle_mod._predict_probability(_BadEstimator(), np.array([[0.0], [1.0]], dtype=float))


def test_aggregate_probabilities_mean_and_median() -> None:
    probs = [np.array([0.1, 0.9], dtype=float), np.array([0.3, 0.7], dtype=float)]

    mean_values = bundle_mod._aggregate_probabilities(probs, "mean")
    median_values = bundle_mod._aggregate_probabilities(probs, "median")

    assert mean_values.tolist() == pytest.approx([0.2, 0.8])
    assert median_values.tolist() == pytest.approx([0.2, 0.8])


def test_predict_with_bundle_emits_uncertainty_column_when_multiple_models(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    multi_model_bundle = replace(bundle, models=[bundle.models[0], bundle.models[0]])

    pred_df, _warnings = predict_with_bundle(config, multi_model_bundle)

    assert "uncertainty_std" in pred_df.columns


def test_export_model_bundle_rejects_manifest_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cv_artifacts, refit, run_dir, resolved_config_path = _prepare_export_inputs(tmp_path)
    monkeypatch.setattr(bundle_mod, "_resolve_manifest_size", lambda _payload: 0)

    with pytest.raises(BundleError, match="size did not match deterministic self inventory entry"):
        export_model_bundle(
            run_dir=run_dir,
            resolved_config_path=resolved_config_path,
            config=config,  # type: ignore[arg-type]
            final_refit_artifacts=refit,  # type: ignore[arg-type]
            thresholds=cv_artifacts.thresholds,  # type: ignore[attr-defined]
        )


def test_verify_file_inventory_rejects_non_mapping_file_entry(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    manifest = _manifest_payload_without_self_inventory(bundle.bundle_dir)  # type: ignore[attr-defined]
    files = dict(manifest["files"])  # type: ignore[index]
    files["thresholds.tsv"] = ["invalid"]
    manifest["files"] = files

    with pytest.raises(BundleError, match="Invalid file inventory entry for: thresholds.tsv"):
        bundle_mod._verify_file_inventory(bundle.bundle_dir, manifest)  # type: ignore[arg-type]


def test_verify_file_inventory_rejects_missing_file_listed_in_manifest(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    manifest = _manifest_payload_without_self_inventory(bundle.bundle_dir)  # type: ignore[attr-defined]
    files = dict(manifest["files"])  # type: ignore[index]
    files["missing.tsv"] = {"sha256": "0" * 64, "size": 1}
    manifest["files"] = files

    with pytest.raises(BundleError, match="Bundle file listed in manifest is missing: missing.tsv"):
        bundle_mod._verify_file_inventory(bundle.bundle_dir, manifest)  # type: ignore[arg-type]


def test_verify_file_inventory_rejects_non_string_sha_entry(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    manifest = _manifest_payload_without_self_inventory(bundle.bundle_dir)  # type: ignore[attr-defined]
    files = dict(manifest["files"])  # type: ignore[index]
    threshold_size = (bundle.bundle_dir / "thresholds.tsv").stat().st_size
    files["thresholds.tsv"] = {"sha256": 123, "size": threshold_size}
    manifest["files"] = files

    with pytest.raises(
        BundleError,
        match="Invalid sha256 entry in manifest for file: thresholds.tsv",
    ):
        bundle_mod._verify_file_inventory(bundle.bundle_dir, manifest)  # type: ignore[arg-type]


def test_verify_file_inventory_rejects_non_integer_size_entry(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    manifest = _manifest_payload_without_self_inventory(bundle.bundle_dir)  # type: ignore[attr-defined]
    files = dict(manifest["files"])  # type: ignore[index]
    threshold_sha = bundle_mod._sha256_file(bundle.bundle_dir / "thresholds.tsv")
    files["thresholds.tsv"] = {"sha256": threshold_sha, "size": "bad-size"}
    manifest["files"] = files

    with pytest.raises(
        BundleError,
        match="Invalid size entry in manifest for file: thresholds.tsv",
    ):
        bundle_mod._verify_file_inventory(bundle.bundle_dir, manifest)  # type: ignore[arg-type]


def test_verify_file_inventory_rejects_manifest_self_sha_mismatch(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    manifest = _manifest_payload(bundle.bundle_dir)  # type: ignore[attr-defined]
    files = dict(manifest["files"])  # type: ignore[index]
    bundle_manifest_entry = dict(files["bundle_manifest.json"])  # type: ignore[index]
    bundle_manifest_entry["sha256"] = "0" * 64
    files["bundle_manifest.json"] = bundle_manifest_entry
    manifest["files"] = files

    with pytest.raises(BundleError, match="sha256 mismatch.*bundle_manifest.json"):
        bundle_mod._verify_file_inventory(bundle.bundle_dir, manifest)  # type: ignore[arg-type]


def test_verify_file_inventory_rejects_regular_file_sha_mismatch(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    manifest = _manifest_payload_without_self_inventory(bundle.bundle_dir)  # type: ignore[attr-defined]
    files = dict(manifest["files"])  # type: ignore[index]
    threshold_entry = dict(files["thresholds.tsv"])  # type: ignore[index]
    threshold_entry["sha256"] = "f" * 64
    files["thresholds.tsv"] = threshold_entry
    manifest["files"] = files

    with pytest.raises(BundleError, match="sha256 mismatch.*thresholds.tsv"):
        bundle_mod._verify_file_inventory(bundle.bundle_dir, manifest)  # type: ignore[arg-type]
