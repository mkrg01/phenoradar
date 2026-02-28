from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

from phenoradar.bundle import (
    BundleError,
    LoadedBundle,
    ModelPreprocessEntry,
    export_model_bundle,
    load_model_bundle,
    predict_with_bundle,
)
from phenoradar.config import AppConfig, load_and_resolve_config, write_resolved_config
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


def _config(tmp_path: Path, metadata: Path, tpm: Path) -> Path:
    return _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )


def _export_and_load_bundle(
    tmp_path: Path, metadata: Path, tpm: Path
) -> tuple[AppConfig, LoadedBundle]:
    config = load_and_resolve_config([_config(tmp_path, metadata, tpm)])
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


def _rewrite_manifest_to_current_files(
    bundle_dir: Path,
    *,
    mutate_files: dict[str, dict[str, int | str] | None] | None = None,
    manifest_mutator: Callable[[dict[str, object]], None] | None = None,
) -> None:
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "bundle_manifest.json":
            continue
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
    if mutate_files is not None:
        for filename, override in mutate_files.items():
            if override is None:
                files.pop(filename, None)
            else:
                files[filename] = override
    manifest["files"] = files

    def _render(base_manifest: dict[str, object], manifest_sha: str, manifest_size: int) -> str:
        payload = dict(base_manifest)
        payload_files = dict(payload["files"])  # type: ignore[index]
        payload_files["bundle_manifest.json"] = {"sha256": manifest_sha, "size": manifest_size}
        payload["files"] = payload_files
        return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"

    placeholder = "0" * 64
    size = 0
    for _ in range(32):
        rendered = _render(manifest, placeholder, size)
        rendered_size = len(rendered.encode("utf-8"))
        if rendered_size == size:
            break
        size = rendered_size
    canonical = _render(manifest, placeholder, size)
    self_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path.write_text(_render(manifest, self_sha, size), encoding="utf-8")


def test_bundle_export_load_and_predict(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_config(tmp_path, metadata, tpm)])
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
    bundle = load_model_bundle(export_result.bundle_dir)
    threshold_fixed = (
        cv_artifacts.thresholds.filter(pl.col("threshold_name") == "fixed_probability_threshold")
        .select("threshold_value")
        .to_series()
        .item()
    )
    if refit.model_entries:
        refit_model_preprocess = [
            ModelPreprocessEntry(feature_names=entry.feature_names, scaler=entry.scaler)
            for entry in refit.model_entries
        ]
    else:
        refit_model_preprocess = [
            ModelPreprocessEntry(feature_names=refit.feature_names, scaler=refit.scaler)
            for _ in refit.models
        ]
    refit_feature_schema = sorted(
        {feature for entry in refit_model_preprocess for feature in entry.feature_names}
    )
    refit_bundle = LoadedBundle(
        bundle_dir=export_result.bundle_dir,
        manifest={},
        manifest_sha256="",
        feature_names=refit_feature_schema,
        scaler=refit.scaler,
        model_preprocess=refit_model_preprocess,
        models=refit.models,
        probability_aggregation=config.ensemble.probability_aggregation,
        threshold_fixed=float(threshold_fixed),
        threshold_cv_derived=float(cv_threshold),
        source_run_id=run_dir.name,
    )
    refit_pred_df, refit_warnings = predict_with_bundle(config, refit_bundle)
    pred_df, warnings = predict_with_bundle(config, bundle)
    manifest = json.loads((export_result.bundle_dir / "bundle_manifest.json").read_text("utf-8"))
    assert "files" in manifest
    assert "bundle_manifest.json" in manifest["files"]
    assert isinstance(manifest["files"]["bundle_manifest.json"]["sha256"], str)
    assert isinstance(manifest["files"]["bundle_manifest.json"]["size"], int)

    assert pred_df.get_column("species").to_list() == refit_pred_df.get_column("species").to_list()
    np.testing.assert_allclose(
        pred_df.get_column("prob").to_numpy(),
        refit_pred_df.get_column("prob").to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    assert (
        pred_df.select("pred_label_fixed_threshold", "pred_label_cv_derived_threshold").to_dicts()
        == refit_pred_df.select("pred_label_fixed_threshold", "pred_label_cv_derived_threshold")
        .to_dicts()
    )
    assert warnings == refit_warnings
    assert pred_df.height == 6
    assert {
        "species",
        "prob",
        "pred_label_fixed_threshold",
        "pred_label_cv_derived_threshold",
    }.issubset(pred_df.columns)
    assert bundle.source_run_id == run_dir.name
    assert isinstance(warnings, list)


def test_bundle_integrity_failure_on_tampered_file(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_config(tmp_path, metadata, tpm)])
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

    model_state_path = export_result.bundle_dir / "model_state.joblib"
    model_state_path.write_bytes(model_state_path.read_bytes() + b"tamper")

    with pytest.raises(BundleError):
        load_model_bundle(export_result.bundle_dir)


def test_predict_with_bundle_uses_thresholds_from_loaded_bundle(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    bundle_low = replace(bundle, threshold_fixed=0.0, threshold_cv_derived=0.0)
    bundle_high = replace(bundle, threshold_fixed=1.0, threshold_cv_derived=1.0)

    pred_low, _ = predict_with_bundle(config, bundle_low)
    pred_high, _ = predict_with_bundle(config, bundle_high)

    low_labels = set(pred_low.select("pred_label_fixed_threshold").to_series().to_list())
    low_cv_labels = set(pred_low.select("pred_label_cv_derived_threshold").to_series().to_list())
    high_labels = set(pred_high.select("pred_label_fixed_threshold").to_series().to_list())
    high_cv_labels = set(pred_high.select("pred_label_cv_derived_threshold").to_series().to_list())
    assert low_labels == {1}
    assert low_cv_labels == {1}
    assert high_labels == {0}
    assert high_cv_labels == {0}


def test_predict_with_bundle_feature_alignment_missing_and_extra(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)

    predict_tpm = _write(
        tmp_path / "predict_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp1\tOGX\t0.2",
                "sp2\tOG1\t2.0",
                "sp2\tOGX\t0.2",
                "sp3\tOG1\t3.0",
                "sp3\tOGX\t0.2",
                "sp4\tOG1\t4.0",
                "sp4\tOGX\t0.2",
                "sp5\tOG1\t5.0",
                "sp5\tOGX\t0.2",
                "sp6\tOG1\t6.0",
                "sp6\tOGX\t0.2",
            ]
        )
        + "\n",
    )
    predict_config = load_and_resolve_config([_config(tmp_path, metadata, predict_tpm)])

    pred_df, warnings = predict_with_bundle(predict_config, bundle)

    assert pred_df.height == 6
    assert any("missing bundle features" in warning for warning in warnings)
    assert any("extra features" in warning for warning in warnings)


def test_predict_with_bundle_fails_when_feature_overlap_is_zero(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)

    predict_tpm = _write(
        tmp_path / "predict_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOGX\t1.0",
                "sp2\tOGX\t2.0",
                "sp3\tOGX\t3.0",
                "sp4\tOGX\t4.0",
                "sp5\tOGX\t5.0",
                "sp6\tOGX\t6.0",
            ]
        )
        + "\n",
    )
    predict_config = load_and_resolve_config([_config(tmp_path, metadata, predict_tpm)])

    with pytest.raises(
        BundleError, match="No bundle features were available in prediction input after alignment"
    ):
        predict_with_bundle(predict_config, bundle)


def test_predict_with_bundle_rejects_negative_tpm_values(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)

    predict_tpm = _write(
        tmp_path / "predict_tpm_negative.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t-0.1",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
                "sp6\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    predict_config = load_and_resolve_config([_config(tmp_path, metadata, predict_tpm)])

    with pytest.raises(BundleError, match="TPM values must be non-negative"):
        predict_with_bundle(predict_config, bundle)


def test_predict_with_bundle_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)

    pred_first, warnings_first = predict_with_bundle(config, bundle)
    pred_second, warnings_second = predict_with_bundle(config, bundle)

    assert pred_first.to_dicts() == pred_second.to_dicts()
    assert warnings_first == warnings_second


@pytest.mark.parametrize(
    ("model_name", "expected_calibration"),
    [
        ("linear_svm", "sigmoid"),
        ("random_forest", "none"),
    ],
)
def test_bundle_manifest_calibration_policy(
    tmp_path: Path, model_name: str, expected_calibration: str
) -> None:
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
                "ext1\tOG1\t3.0",
                "ext1\tOG2\t1.0",
                "inf1\tOG1\t2.0",
                "inf1\tOG2\t0.7",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _write(
                tmp_path / "config.yml",
                f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
model:
  name: {model_name}
""".strip()
                + "\n",
            )
        ]
    )
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
    manifest = json.loads((export_result.bundle_dir / "bundle_manifest.json").read_text("utf-8"))

    assert manifest["model_name"] == model_name
    assert manifest["calibration"] == expected_calibration


def test_load_model_bundle_rejects_missing_manifest(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    with pytest.raises(BundleError, match="Bundle manifest not found"):
        load_model_bundle(bundle_dir)


def test_load_model_bundle_rejects_invalid_manifest_json(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    (bundle.bundle_dir / "bundle_manifest.json").write_text("{invalid-json}\n", encoding="utf-8")

    with pytest.raises(BundleError, match="Invalid bundle_manifest.json"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_unsupported_version(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    manifest_path = bundle.bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_format_version"] = "999"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleError, match="Unsupported bundle_format_version"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_missing_files_inventory(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    manifest_path = bundle.bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("files", None)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleError, match="missing 'files' inventory"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_missing_required_file(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    (bundle.bundle_dir / "thresholds.tsv").unlink()

    with pytest.raises(BundleError, match="Bundle is missing required file: thresholds.tsv"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_missing_required_file_entry(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    _rewrite_manifest_to_current_files(
        bundle.bundle_dir,
        mutate_files={"thresholds.tsv": None},
    )

    with pytest.raises(BundleError, match="missing required entry: thresholds.tsv"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_feature_schema_missing_required_columns(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    pl.DataFrame({"feature": ["OG1"]}).write_csv(
        bundle.bundle_dir / "feature_schema.tsv",
        separator="\t",
    )
    _rewrite_manifest_to_current_files(bundle.bundle_dir)

    with pytest.raises(BundleError, match="feature_schema.tsv must contain"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_non_contiguous_feature_index(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    pl.DataFrame(
        {
            "feature": ["OG1", "OG2"],
            "feature_index": [0, 2],
        }
    ).write_csv(bundle.bundle_dir / "feature_schema.tsv", separator="\t")
    _rewrite_manifest_to_current_files(bundle.bundle_dir)

    with pytest.raises(BundleError, match="non-contiguous or unordered"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_invalid_scaler_type(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    schema = pl.read_csv(bundle.bundle_dir / "feature_schema.tsv", separator="\t").sort(
        "feature_index"
    )
    feature_names = [str(v) for v in schema.select("feature").to_series().to_list()]
    joblib.dump(
        {
            "feature_names": feature_names,
            "scaler": "not-a-scaler",
            "transform": "log1p_then_standard_scaler",
        },
        bundle.bundle_dir / "preprocess_state.joblib",
    )
    _rewrite_manifest_to_current_files(bundle.bundle_dir)

    with pytest.raises(BundleError, match="missing a valid StandardScaler"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_mismatched_preprocess_feature_names(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    joblib.dump(
        {
            "feature_names": ["OTHER"],
            "scaler": bundle.scaler,
            "transform": "log1p_then_standard_scaler",
        },
        bundle.bundle_dir / "preprocess_state.joblib",
    )
    _rewrite_manifest_to_current_files(bundle.bundle_dir)

    with pytest.raises(BundleError, match="feature_names do not match"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_empty_models_list(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    model_state = joblib.load(bundle.bundle_dir / "model_state.joblib")
    model_state["models"] = []
    joblib.dump(model_state, bundle.bundle_dir / "model_state.joblib")
    _rewrite_manifest_to_current_files(bundle.bundle_dir)

    with pytest.raises(BundleError, match="non-empty model list"):
        load_model_bundle(bundle.bundle_dir)


def test_load_model_bundle_rejects_invalid_probability_aggregation(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    model_state = joblib.load(bundle.bundle_dir / "model_state.joblib")
    model_state["probability_aggregation"] = "invalid"
    joblib.dump(model_state, bundle.bundle_dir / "model_state.joblib")
    _rewrite_manifest_to_current_files(bundle.bundle_dir)

    with pytest.raises(BundleError, match="invalid probability_aggregation"):
        load_model_bundle(bundle.bundle_dir)


def test_predict_with_bundle_rejects_missing_species_column(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _resolved, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    bad_metadata = _write(
        tmp_path / "bad_metadata.tsv",
        "\n".join(
            [
                "sp\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
            ]
        )
        + "\n",
    )
    predict_config = load_and_resolve_config([_config(tmp_path, bad_metadata, tpm)])

    with pytest.raises(BundleError, match="Metadata is missing species column"):
        predict_with_bundle(predict_config, bundle)


def test_predict_with_bundle_rejects_empty_species_after_normalization(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    _resolved, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    bad_metadata = _write(
        tmp_path / "empty_species.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "\t1\tg1",
                "  \t0\tg1",
            ]
        )
        + "\n",
    )
    predict_config = load_and_resolve_config([_config(tmp_path, bad_metadata, tpm)])

    with pytest.raises(BundleError, match="zero valid species"):
        predict_with_bundle(predict_config, bundle)


def test_predict_with_bundle_rejects_invalid_model_probability_shape(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)

    class _BadModel:
        def predict_proba(self, x: object) -> np.ndarray:
            _ = x
            return np.array([0.1, 0.9], dtype=float)

    bad_bundle = replace(bundle, models=[_BadModel()])

    with pytest.raises(BundleError, match="invalid probability shape"):
        predict_with_bundle(config, bad_bundle)


def test_predict_with_bundle_rejects_invalid_probability_aggregation(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config, bundle = _export_and_load_bundle(tmp_path, metadata, tpm)
    bad_bundle = replace(bundle, probability_aggregation="invalid")

    with pytest.raises(BundleError, match="Unsupported probability aggregation"):
        predict_with_bundle(config, bad_bundle)
