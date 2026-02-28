"""Model bundle export, loading, and bundle-based prediction."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from phenoradar.config import AppConfig
from phenoradar.cv import ExpressionMatrixBuilder, FinalRefitArtifacts

BUNDLE_FORMAT_VERSION = "1"
_BUNDLE_DIRNAME = "model_bundle"
_SELF_SHA256_PLACEHOLDER = "0" * 64
_REQUIRED_FILES = [
    "bundle_manifest.json",
    "feature_schema.tsv",
    "preprocess_state.joblib",
    "model_state.joblib",
    "thresholds.tsv",
    "resolved_config.yml",
]


class BundleError(ValueError):
    """Raised when bundle export/load/predict cannot proceed."""


@dataclass(frozen=True)
class BundleExportResult:
    """Metadata for an exported model bundle."""

    bundle_dir: Path
    manifest_sha256: str


@dataclass(frozen=True)
class LoadedBundle:
    """Loaded and integrity-verified model bundle payload."""

    bundle_dir: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    feature_names: list[str]
    scaler: StandardScaler
    model_preprocess: list[ModelPreprocessEntry]
    models: list[Any]
    probability_aggregation: str
    threshold_fixed: float
    threshold_cv_derived: float
    source_run_id: str


@dataclass(frozen=True)
class ModelPreprocessEntry:
    """One model-local preprocessing state from bundle payload."""

    feature_names: list[str]
    scaler: StandardScaler


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path) -> dict[str, int | str]:
    return {
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _render_manifest_payload(
    manifest_base: dict[str, Any], *, manifest_sha: str, manifest_size: int
) -> str:
    files_raw = manifest_base.get("files")
    if not isinstance(files_raw, dict):
        raise BundleError("bundle manifest base payload is missing 'files' inventory")
    files = dict(files_raw)
    files["bundle_manifest.json"] = {
        "sha256": manifest_sha,
        "size": manifest_size,
    }
    payload = dict(manifest_base)
    payload["files"] = files
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _resolve_manifest_size(manifest_base: dict[str, Any]) -> int:
    size = 0
    for _ in range(32):
        rendered = _render_manifest_payload(
            manifest_base,
            manifest_sha=_SELF_SHA256_PLACEHOLDER,
            manifest_size=size,
        )
        rendered_size = len(rendered.encode("utf-8"))
        if rendered_size == size:
            return size
        size = rendered_size
    raise BundleError("Failed to resolve deterministic bundle_manifest.json self size")


def _manifest_canonical_self_sha(manifest_base: dict[str, Any], manifest_size: int) -> str:
    rendered = _render_manifest_payload(
        manifest_base,
        manifest_sha=_SELF_SHA256_PLACEHOLDER,
        manifest_size=manifest_size,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_metadata(cwd: Path) -> tuple[str, bool, str]:
    commit = _run_git(["git", "rev-parse", "HEAD"], cwd=cwd) or "unknown"
    status = _run_git(["git", "status", "--porcelain"], cwd=cwd)
    dirty = bool(status) if status is not None else False
    patch = _run_git(["git", "diff", "HEAD"], cwd=cwd)
    patch_sha = sha256((patch or "").encode("utf-8")).hexdigest()
    return commit, dirty, patch_sha


def _calibration_for_model(model_name: str) -> str:
    if model_name == "linear_svm":
        return "sigmoid"
    return "none"


def _threshold_value(thresholds: pl.DataFrame, threshold_name: str) -> float:
    values = (
        thresholds.filter(pl.col("threshold_name") == threshold_name)
        .select("threshold_value")
        .to_series()
        .to_list()
    )
    if len(values) != 1:
        raise BundleError(f"Expected exactly one threshold value for {threshold_name}")
    return float(values[0])


def export_model_bundle(
    *,
    run_dir: Path,
    resolved_config_path: Path,
    config: AppConfig,
    final_refit_artifacts: FinalRefitArtifacts,
    thresholds: pl.DataFrame,
) -> BundleExportResult:
    """Export reusable model bundle from final-refit artifacts."""
    bundle_dir = run_dir / _BUNDLE_DIRNAME
    bundle_dir.mkdir(parents=True, exist_ok=False)

    feature_schema_path = bundle_dir / "feature_schema.tsv"
    preprocess_state_path = bundle_dir / "preprocess_state.joblib"
    model_state_path = bundle_dir / "model_state.joblib"
    thresholds_path = bundle_dir / "thresholds.tsv"
    resolved_copy_path = bundle_dir / "resolved_config.yml"
    manifest_path = bundle_dir / "bundle_manifest.json"

    if final_refit_artifacts.model_entries:
        model_entries = [
            ModelPreprocessEntry(
                feature_names=entry.feature_names,
                scaler=entry.scaler,
            )
            for entry in final_refit_artifacts.model_entries
        ]
    else:
        model_entries = [
            ModelPreprocessEntry(
                feature_names=final_refit_artifacts.feature_names,
                scaler=final_refit_artifacts.scaler,
            )
            for _ in final_refit_artifacts.models
        ]
    if not model_entries:
        raise BundleError("Final refit artifacts produced zero model preprocessing entries")
    if len(model_entries) != len(final_refit_artifacts.models):
        raise BundleError("Model preprocessing entries count does not match model count")

    feature_schema = sorted(
        {
            feature
            for entry in model_entries
            for feature in entry.feature_names
        }
    )
    if not feature_schema:
        raise BundleError("Final refit artifacts produced zero bundle features")

    feature_schema_df = pl.DataFrame(
        {
            "feature": feature_schema,
            "feature_index": list(range(len(feature_schema))),
        }
    )
    feature_schema_df.write_csv(feature_schema_path, separator="\t")

    joblib.dump(
        {
            "feature_names": feature_schema,
            "scaler": final_refit_artifacts.scaler,
            "transform": "log1p_then_standard_scaler",
            "model_preprocess": [
                {
                    "feature_names": entry.feature_names,
                    "scaler": entry.scaler,
                }
                for entry in model_entries
            ],
        },
        preprocess_state_path,
    )
    joblib.dump(
        {
            "model_name": config.model.name,
            "probability_aggregation": config.ensemble.probability_aggregation,
            "models": final_refit_artifacts.models,
        },
        model_state_path,
    )
    thresholds.write_csv(
        thresholds_path,
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    shutil.copy2(resolved_config_path, resolved_copy_path)

    commit, dirty, patch_sha = _git_metadata(run_dir)
    files_info: dict[str, dict[str, int | str]] = {}
    for filename in _REQUIRED_FILES:
        if filename == "bundle_manifest.json":
            continue
        files_info[filename] = _file_info(bundle_dir / filename)

    threshold_fixed = _threshold_value(thresholds, "fixed_probability_threshold")
    threshold_cv = _threshold_value(thresholds, "cv_derived_threshold")
    manifest_base = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "source_run_dir": str(run_dir),
        "source_run_id": run_dir.name,
        "source_git_commit": commit,
        "source_git_dirty": dirty,
        "source_git_worktree_patch_sha256": patch_sha,
        "model_name": config.model.name,
        "calibration": _calibration_for_model(config.model.name),
        "ensemble_size": final_refit_artifacts.ensemble_size,
        "ensemble_probability_aggregation": config.ensemble.probability_aggregation,
        "threshold_fixed": threshold_fixed,
        "threshold_cv_derived": threshold_cv,
        "python_version": platform.python_version(),
        "library_versions": {
            "polars": package_version("polars"),
            "scikit-learn": package_version("scikit-learn"),
            "pydantic": package_version("pydantic"),
            "typer": package_version("typer"),
        },
        "files": files_info,
    }
    manifest_size = _resolve_manifest_size(manifest_base)
    manifest_self_sha = _manifest_canonical_self_sha(manifest_base, manifest_size)
    manifest_path.write_text(
        _render_manifest_payload(
            manifest_base,
            manifest_sha=manifest_self_sha,
            manifest_size=manifest_size,
        ),
        encoding="utf-8",
    )
    actual_manifest_size = manifest_path.stat().st_size
    if actual_manifest_size != manifest_size:
        raise BundleError(
            "bundle_manifest.json size did not match deterministic self inventory entry"
        )
    manifest_sha = _sha256_file(manifest_path)
    return BundleExportResult(bundle_dir=bundle_dir, manifest_sha256=manifest_sha)


def _verify_file_inventory(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise BundleError("bundle_manifest.json is missing 'files' inventory")

    for filename in _REQUIRED_FILES:
        path = bundle_dir / filename
        if not path.exists():
            raise BundleError(f"Bundle is missing required file: {filename}")
        if filename != "bundle_manifest.json" and filename not in files:
            raise BundleError(f"Manifest file inventory is missing required entry: {filename}")

    for filename, expected in files.items():
        if not isinstance(expected, dict):
            raise BundleError(f"Invalid file inventory entry for: {filename}")
        path = bundle_dir / filename
        if not path.exists():
            raise BundleError(f"Bundle file listed in manifest is missing: {filename}")
        expected_sha = expected.get("sha256")
        expected_size = expected.get("size")
        if not isinstance(expected_sha, str):
            raise BundleError(f"Invalid sha256 entry in manifest for file: {filename}")
        if not isinstance(expected_size, int):
            raise BundleError(f"Invalid size entry in manifest for file: {filename}")
        actual_size = path.stat().st_size
        if expected_size != actual_size:
            raise BundleError(
                f"Bundle integrity check failed (size mismatch) for file: {filename}"
            )
        if filename == "bundle_manifest.json":
            manifest_without_self = dict(manifest)
            manifest_files_without_self = dict(files)
            manifest_files_without_self.pop("bundle_manifest.json", None)
            manifest_without_self["files"] = manifest_files_without_self
            canonical_sha = _manifest_canonical_self_sha(
                manifest_without_self,
                manifest_size=expected_size,
            )
            if expected_sha != canonical_sha:
                raise BundleError(
                    "Bundle integrity check failed (sha256 mismatch) for file: "
                    "bundle_manifest.json"
                )
            continue
        actual_sha = _sha256_file(path)
        if expected_sha != actual_sha:
            raise BundleError(
                f"Bundle integrity check failed (sha256 mismatch) for file: {filename}"
            )


def _load_feature_schema(path: Path) -> list[str]:
    schema = pl.read_csv(path, separator="\t")
    if not {"feature", "feature_index"}.issubset(schema.columns):
        raise BundleError("feature_schema.tsv must contain 'feature' and 'feature_index' columns")
    sorted_schema = schema.sort("feature_index")
    expected_indices = list(range(sorted_schema.height))
    actual_indices = [int(v) for v in sorted_schema.select("feature_index").to_series().to_list()]
    if actual_indices != expected_indices:
        raise BundleError("feature_schema.tsv has non-contiguous or unordered feature_index values")
    features = [str(v) for v in sorted_schema.select("feature").to_series().to_list()]
    if not features:
        raise BundleError("feature_schema.tsv does not contain any features")
    return features


def load_model_bundle(bundle_dir: Path) -> LoadedBundle:
    """Load and verify model bundle."""
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise BundleError(f"Bundle manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"Invalid bundle_manifest.json: {manifest_path}") from exc

    version_value = manifest.get("bundle_format_version")
    if version_value != BUNDLE_FORMAT_VERSION:
        raise BundleError(
            f"Unsupported bundle_format_version: {version_value} "
            f"(expected {BUNDLE_FORMAT_VERSION})"
        )

    _verify_file_inventory(bundle_dir, manifest)

    feature_names = _load_feature_schema(bundle_dir / "feature_schema.tsv")
    preprocess_state = joblib.load(bundle_dir / "preprocess_state.joblib")
    model_state = joblib.load(bundle_dir / "model_state.joblib")

    scaler = preprocess_state.get("scaler")
    state_features = preprocess_state.get("feature_names")
    if not isinstance(scaler, StandardScaler):
        raise BundleError("preprocess_state.joblib is missing a valid StandardScaler")
    if state_features != feature_names:
        raise BundleError("preprocess_state feature_names do not match feature_schema.tsv")

    models = model_state.get("models")
    aggregation = model_state.get("probability_aggregation")
    if not isinstance(models, list) or not models:
        raise BundleError("model_state.joblib must contain a non-empty model list")
    if aggregation not in {"mean", "median"}:
        raise BundleError("model_state.joblib has invalid probability_aggregation")

    model_preprocess_raw = preprocess_state.get("model_preprocess")
    model_preprocess: list[ModelPreprocessEntry] = []
    if model_preprocess_raw is None:
        model_preprocess = [ModelPreprocessEntry(feature_names=feature_names, scaler=scaler)] * len(
            models
        )
    else:
        if not isinstance(model_preprocess_raw, list) or not model_preprocess_raw:
            raise BundleError(
                "preprocess_state.joblib model_preprocess must be a non-empty list"
            )
        if len(model_preprocess_raw) != len(models):
            raise BundleError(
                "preprocess_state.joblib model_preprocess count does not match model count"
            )
        schema_feature_set = set(feature_names)
        for entry in model_preprocess_raw:
            if not isinstance(entry, dict):
                raise BundleError("preprocess_state.joblib model_preprocess contains invalid entry")
            entry_features = entry.get("feature_names")
            entry_scaler = entry.get("scaler")
            if not isinstance(entry_scaler, StandardScaler):
                raise BundleError(
                    "preprocess_state.joblib model_preprocess entry is missing a valid scaler"
                )
            if not isinstance(entry_features, list) or not entry_features:
                raise BundleError(
                    "preprocess_state.joblib model_preprocess entry feature_names is invalid"
                )
            normalized_features = [str(value) for value in entry_features]
            if any(feature not in schema_feature_set for feature in normalized_features):
                raise BundleError(
                    "preprocess_state.joblib model_preprocess feature_names are not "
                    "contained in feature_schema.tsv"
                )
            model_preprocess.append(
                ModelPreprocessEntry(
                    feature_names=normalized_features,
                    scaler=entry_scaler,
                )
            )

    thresholds = pl.read_csv(bundle_dir / "thresholds.tsv", separator="\t")
    threshold_fixed = _threshold_value(thresholds, "fixed_probability_threshold")
    threshold_cv = _threshold_value(thresholds, "cv_derived_threshold")

    source_run_id_raw = manifest.get("source_run_id")
    source_run_id = str(source_run_id_raw) if source_run_id_raw is not None else "unknown"

    return LoadedBundle(
        bundle_dir=bundle_dir,
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
        feature_names=feature_names,
        scaler=scaler,
        model_preprocess=model_preprocess,
        models=models,
        probability_aggregation=aggregation,
        threshold_fixed=threshold_fixed,
        threshold_cv_derived=threshold_cv,
        source_run_id=source_run_id,
    )


def _predict_probability(estimator: Any, x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise BundleError("Loaded model returned invalid probability shape")
    return probabilities[:, 1]


def _aggregate_probabilities(probs: list[np.ndarray], aggregation: str) -> np.ndarray:
    stacked = np.vstack(probs)
    if aggregation == "mean":
        return np.asarray(np.mean(stacked, axis=0), dtype=float)
    if aggregation == "median":
        return np.asarray(np.median(stacked, axis=0), dtype=float)
    raise BundleError(f"Unsupported probability aggregation: {aggregation}")


def predict_with_bundle(
    config: AppConfig, bundle: LoadedBundle
) -> tuple[pl.DataFrame, list[str]]:
    """Run deterministic inference using a loaded model bundle."""
    metadata = pl.read_csv(config.data.metadata_path, separator="\t")
    if config.data.species_col not in metadata.columns:
        raise BundleError(f"Metadata is missing species column: {config.data.species_col}")

    species = (
        metadata.select(
            pl.col(config.data.species_col)
            .cast(pl.String, strict=False)
            .str.strip_chars()
            .alias("species")
        )
        .filter(pl.col("species").is_not_null() & (pl.col("species") != ""))
        .unique()
        .sort("species")
        .select("species")
        .to_series()
        .to_list()
    )
    species_list = [str(v) for v in species]
    if not species_list:
        raise BundleError("Predict metadata produced zero valid species")

    matrix_builder = ExpressionMatrixBuilder(config)
    x_raw, input_features = matrix_builder.build_matrix(species_list)
    if np.any(x_raw < 0):
        raise BundleError("TPM values must be non-negative before log1p transform")
    input_index = {feature: idx for idx, feature in enumerate(input_features)}
    bundle_features = bundle.feature_names

    aligned = np.zeros((len(species_list), len(bundle_features)), dtype=float)
    overlap_count = 0
    for feature_idx, feature_name in enumerate(bundle_features):
        input_idx = input_index.get(feature_name)
        if input_idx is None:
            continue
        aligned[:, feature_idx] = x_raw[:, input_idx]
        overlap_count += 1

    if overlap_count == 0:
        raise BundleError("No bundle features were available in prediction input after alignment")

    missing_count = len(bundle_features) - overlap_count
    extra_count = len(set(input_features) - set(bundle_features))
    warnings: list[str] = []
    if missing_count > 0:
        warnings.append(
            "Prediction input is missing bundle features; "
            f"filled with 0 for {missing_count} features"
        )
    if extra_count > 0:
        warnings.append(
            "Prediction input contains extra features not in bundle; "
            f"ignored {extra_count} features"
        )

    x_log = np.log1p(aligned)
    schema_index = {feature: idx for idx, feature in enumerate(bundle_features)}

    model_probs: list[np.ndarray] = []
    if len(bundle.model_preprocess) == len(bundle.models):
        preprocess_entries = bundle.model_preprocess
    elif len(bundle.model_preprocess) == 1 and len(bundle.models) > 1:
        preprocess_entries = bundle.model_preprocess * len(bundle.models)
    else:
        raise BundleError("Bundle model/preprocess count mismatch")

    for model, preprocess in zip(bundle.models, preprocess_entries, strict=True):
        selected_indices = np.array(
            [schema_index[feature] for feature in preprocess.feature_names], dtype=int
        )
        x_model_log = x_log[:, selected_indices]
        x_model_scaled = np.asarray(preprocess.scaler.transform(x_model_log), dtype=float)
        model_probs.append(_predict_probability(model, x_model_scaled))

    prob = _aggregate_probabilities(model_probs, bundle.probability_aggregation)
    uncertainty_std = np.std(np.vstack(model_probs), axis=0) if len(model_probs) > 1 else None

    pred_label = (prob >= bundle.threshold_fixed).astype(int)
    payload: dict[str, list[Any]] = {
        "species": species_list,
        "prob": prob.astype(float, copy=False).tolist(),
        "pred_label_fixed_threshold": pred_label.astype(int, copy=False).tolist(),
        "pred_label_cv_derived_threshold": (
            prob >= bundle.threshold_cv_derived
        ).astype(int).tolist(),
    }
    if uncertainty_std is not None:
        payload["uncertainty_std"] = uncertainty_std.astype(float, copy=False).tolist()

    pred_df = pl.DataFrame(payload).sort("species")
    return pred_df, warnings
