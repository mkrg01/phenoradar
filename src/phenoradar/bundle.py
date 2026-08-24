"""Model bundle export, loading, and bundle-based prediction."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from phenoradar.config import AppConfig
from phenoradar.cv import (
    CVError,
    ExpressionMatrixBuilder,
    FeatureScaler,
    FinalRefitArtifacts,
    apply_expression_transform,
    apply_feature_scaling,
)
from phenoradar.metrics import (
    FIXED_PROBABILITY_THRESHOLD_DERIVED_FROM_CV,
    FIXED_PROBABILITY_THRESHOLD_NAME,
    FIXED_PROBABILITY_THRESHOLD_POLICY,
)
from phenoradar.provenance import phenoradar_build_snapshot, runtime_environment_snapshot

BUNDLE_FORMAT_VERSION = "2"
_LEGACY_BUNDLE_FORMAT_VERSION = "1"
_SUPPORTED_BUNDLE_FORMAT_VERSIONS = {
    _LEGACY_BUNDLE_FORMAT_VERSION,
    BUNDLE_FORMAT_VERSION,
}
_BUNDLE_DIRNAME = "model_bundle"
_SELF_SHA256_PLACEHOLDER = "0" * 64
_CONTEXTUAL_EXPRESSION_TRANSFORMS = {"sample_rank", "sample_percentile_rank"}
_REQUIRED_FILES = [
    "bundle_manifest.json",
    "feature_schema.tsv",
    "transform_feature_schema.tsv",
    "preprocess_state.joblib",
    "model_state.joblib",
    "thresholds.tsv",
    "resolved_config.yml",
]
_LEGACY_REQUIRED_FILES = [
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
    transform_feature_names: list[str]
    scaler: FeatureScaler
    model_preprocess: list[ModelPreprocessEntry]
    models: list[Any]
    probability_aggregation: str
    threshold_fixed: float
    source_run_id: str
    expression_transform: str
    feature_scaling: str


@dataclass(frozen=True)
class ModelPreprocessEntry:
    """One model-local preprocessing state from bundle payload."""

    feature_names: list[str]
    scaler: FeatureScaler


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


def _preprocess_methods(preprocess_state: dict[str, Any]) -> tuple[str, str]:
    expression_transform = preprocess_state.get("expression_transform")
    feature_scaling = preprocess_state.get("feature_scaling")

    if expression_transform is None and feature_scaling is None:
        legacy_transform = preprocess_state.get("transform")
        if legacy_transform == "log1p_then_standard_scaler":
            return "log1p", "standard"
        raise BundleError("preprocess_state.joblib is missing preprocessing method metadata")

    if not isinstance(expression_transform, str):
        raise BundleError("preprocess_state.joblib has invalid expression_transform")
    if expression_transform not in {"none", "log1p", "sample_rank", "sample_percentile_rank"}:
        raise BundleError(
            f"preprocess_state.joblib has unsupported expression_transform: "
            f"{expression_transform}"
        )
    if not isinstance(feature_scaling, str):
        raise BundleError("preprocess_state.joblib has invalid feature_scaling")
    if feature_scaling not in {"none", "standard"}:
        raise BundleError(
            f"preprocess_state.joblib has unsupported feature_scaling: {feature_scaling}"
        )
    return expression_transform, feature_scaling


def _validate_scaler_state(
    scaler: Any,
    *,
    feature_scaling: str,
    context: str,
    standard_message: str,
) -> FeatureScaler:
    if feature_scaling == "none":
        if scaler is not None:
            raise BundleError(f"{context} scaler must be null when feature_scaling=none")
        return None
    if not isinstance(scaler, StandardScaler):
        raise BundleError(standard_message)
    return scaler


def _preprocess_transform_label(config: AppConfig) -> str:
    expression_transform = config.preprocess.expression_transform.method
    feature_scaling = config.preprocess.feature_scaling.method
    if expression_transform == "log1p" and feature_scaling == "standard":
        return "log1p_then_standard_scaler"
    return f"{expression_transform}_then_{feature_scaling}"


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
    transform_feature_schema_path = bundle_dir / "transform_feature_schema.tsv"
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

    transform_feature_schema = [
        str(feature) for feature in final_refit_artifacts.transform_feature_names
    ]
    if not transform_feature_schema:
        raise BundleError("Final refit artifacts produced zero transform input features")
    if any(not feature.strip() for feature in transform_feature_schema):
        raise BundleError("Final refit artifacts contain an empty transform input feature")
    if len(set(transform_feature_schema)) != len(transform_feature_schema):
        raise BundleError("Final refit artifacts contain duplicate transform input features")
    transform_feature_set = set(transform_feature_schema)
    if any(feature not in transform_feature_set for feature in feature_schema):
        raise BundleError(
            "Final refit model features are not contained in the transform input schema"
        )

    feature_schema_df = pl.DataFrame(
        {
            "feature": feature_schema,
            "feature_index": list(range(len(feature_schema))),
        }
    )
    feature_schema_df.write_csv(feature_schema_path, separator="\t")
    pl.DataFrame(
        {
            "feature": transform_feature_schema,
            "feature_index": list(range(len(transform_feature_schema))),
        }
    ).write_csv(transform_feature_schema_path, separator="\t")

    joblib.dump(
        {
            "feature_names": feature_schema,
            "transform_feature_names": transform_feature_schema,
            "scaler": final_refit_artifacts.scaler,
            "transform": _preprocess_transform_label(config),
            "expression_transform": config.preprocess.expression_transform.method,
            "feature_scaling": config.preprocess.feature_scaling.method,
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

    build = phenoradar_build_snapshot()
    environment = runtime_environment_snapshot()
    files_info: dict[str, dict[str, int | str]] = {}
    for filename in _REQUIRED_FILES:
        if filename == "bundle_manifest.json":
            continue
        files_info[filename] = _file_info(bundle_dir / filename)

    threshold_fixed = _threshold_value(thresholds, FIXED_PROBABILITY_THRESHOLD_NAME)
    manifest_base = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "source_run_dir": str(run_dir),
        "source_run_id": run_dir.name,
        "source_provenance_schema_version": build["provenance_schema_version"],
        "source_phenoradar_version": build["phenoradar_version"],
        "source_phenoradar_install_type": build["phenoradar_install_type"],
        "source_git_source": build["git_source"],
        "source_git_commit": build["git_commit"],
        "source_git_dirty": build["git_dirty"],
        "source_git_worktree_patch_sha256": build["git_worktree_patch_sha256"],
        "model_name": config.model.name,
        "calibration": _calibration_for_model(config.model.name),
        "ensemble_size": final_refit_artifacts.ensemble_size,
        "ensemble_probability_aggregation": config.ensemble.probability_aggregation,
        "threshold_fixed": threshold_fixed,
        "threshold_name": FIXED_PROBABILITY_THRESHOLD_NAME,
        "threshold_policy": FIXED_PROBABILITY_THRESHOLD_POLICY,
        "threshold_derived_from_cv": FIXED_PROBABILITY_THRESHOLD_DERIVED_FROM_CV,
        "python_version": environment["python_version"],
        "library_versions": environment["library_versions"],
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


def _verify_file_inventory(
    bundle_dir: Path,
    manifest: dict[str, Any],
    *,
    required_files: list[str] | None = None,
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise BundleError("bundle_manifest.json is missing 'files' inventory")

    resolved_required_files = _REQUIRED_FILES if required_files is None else required_files
    for filename in resolved_required_files:
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
    schema_name = path.name
    schema = pl.read_csv(path, separator="\t")
    if not {"feature", "feature_index"}.issubset(schema.columns):
        raise BundleError(
            f"{schema_name} must contain 'feature' and 'feature_index' columns"
        )
    sorted_schema = schema.sort("feature_index")
    expected_indices = list(range(sorted_schema.height))
    actual_indices = [int(v) for v in sorted_schema.select("feature_index").to_series().to_list()]
    if actual_indices != expected_indices:
        raise BundleError(
            f"{schema_name} has non-contiguous or unordered feature_index values"
        )
    features = [str(v) for v in sorted_schema.select("feature").to_series().to_list()]
    if not features:
        raise BundleError(f"{schema_name} does not contain any features")
    if any(not feature.strip() for feature in features):
        raise BundleError(f"{schema_name} contains an empty feature identifier")
    if len(set(features)) != len(features):
        raise BundleError(f"{schema_name} contains duplicate feature identifiers")
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
    if (
        not isinstance(version_value, str)
        or version_value not in _SUPPORTED_BUNDLE_FORMAT_VERSIONS
    ):
        raise BundleError(
            f"Unsupported bundle_format_version: {version_value} "
            f"(supported: {', '.join(sorted(_SUPPORTED_BUNDLE_FORMAT_VERSIONS))})"
        )

    required_files = (
        _LEGACY_REQUIRED_FILES
        if version_value == _LEGACY_BUNDLE_FORMAT_VERSION
        else _REQUIRED_FILES
    )
    _verify_file_inventory(bundle_dir, manifest, required_files=required_files)

    feature_names = _load_feature_schema(bundle_dir / "feature_schema.tsv")
    preprocess_state = joblib.load(bundle_dir / "preprocess_state.joblib")
    model_state = joblib.load(bundle_dir / "model_state.joblib")
    if not isinstance(preprocess_state, dict):
        raise BundleError("preprocess_state.joblib must contain a mapping")
    if not isinstance(model_state, dict):
        raise BundleError("model_state.joblib must contain a mapping")

    expression_transform, feature_scaling = _preprocess_methods(preprocess_state)
    if version_value == _LEGACY_BUNDLE_FORMAT_VERSION:
        if expression_transform in _CONTEXTUAL_EXPRESSION_TRANSFORMS:
            raise BundleError(
                "Bundle format version 1 does not preserve the complete transform input "
                f"schema required for {expression_transform}; rerun full_run and export the "
                "model bundle with the current PhenoRadar version"
            )
        transform_feature_names = feature_names
    else:
        transform_feature_names = _load_feature_schema(
            bundle_dir / "transform_feature_schema.tsv"
        )
    scaler = preprocess_state.get("scaler")
    state_features = preprocess_state.get("feature_names")
    state_transform_features = preprocess_state.get("transform_feature_names")
    scaler = _validate_scaler_state(
        scaler,
        feature_scaling=feature_scaling,
        context="preprocess_state.joblib",
        standard_message="preprocess_state.joblib is missing a valid StandardScaler",
    )
    if state_features != feature_names:
        raise BundleError("preprocess_state feature_names do not match feature_schema.tsv")
    if (
        version_value == BUNDLE_FORMAT_VERSION
        and state_transform_features != transform_feature_names
    ):
        raise BundleError(
            "preprocess_state transform_feature_names do not match "
            "transform_feature_schema.tsv"
        )
    transform_feature_set = set(transform_feature_names)
    if any(feature not in transform_feature_set for feature in feature_names):
        raise BundleError(
            "feature_schema.tsv features are not contained in "
            "transform_feature_schema.tsv"
        )

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
            entry_scaler = _validate_scaler_state(
                entry_scaler,
                feature_scaling=feature_scaling,
                context="preprocess_state.joblib model_preprocess entry",
                standard_message=(
                    "preprocess_state.joblib model_preprocess entry is missing a valid scaler"
                ),
            )
            if not isinstance(entry_features, list) or not entry_features:
                raise BundleError(
                    "preprocess_state.joblib model_preprocess entry feature_names is invalid"
                )
            normalized_features = [str(value) for value in entry_features]
            if len(set(normalized_features)) != len(normalized_features):
                raise BundleError(
                    "preprocess_state.joblib model_preprocess feature_names contain duplicates"
                )
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
    threshold_fixed = _threshold_value(thresholds, FIXED_PROBABILITY_THRESHOLD_NAME)

    source_run_id_raw = manifest.get("source_run_id")
    source_run_id = str(source_run_id_raw) if source_run_id_raw is not None else "unknown"

    return LoadedBundle(
        bundle_dir=bundle_dir,
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
        feature_names=feature_names,
        transform_feature_names=transform_feature_names,
        scaler=scaler,
        model_preprocess=model_preprocess,
        models=models,
        probability_aggregation=aggregation,
        threshold_fixed=threshold_fixed,
        source_run_id=source_run_id,
        expression_transform=expression_transform,
        feature_scaling=feature_scaling,
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

    try:
        matrix_builder = ExpressionMatrixBuilder(config)
        x_raw, input_features = matrix_builder.build_matrix(species_list)
    except CVError as exc:
        raise BundleError(str(exc)) from exc
    input_index = {feature: idx for idx, feature in enumerate(input_features)}
    bundle_features = bundle.feature_names
    input_feature_set = set(input_features)
    model_overlap_count = len(input_feature_set.intersection(bundle_features))
    if model_overlap_count == 0:
        raise BundleError("No bundle features were available in prediction input after alignment")

    if bundle.expression_transform in _CONTEXTUAL_EXPRESSION_TRANSFORMS:
        alignment_features = bundle.transform_feature_names
    else:
        # Feature-wise transforms commute with feature selection, so aligning only
        # the model-feature union avoids materializing unused input columns.
        alignment_features = bundle_features
    alignment_feature_set = set(alignment_features)

    aligned_raw = np.zeros((len(species_list), len(alignment_features)), dtype=float)
    alignment_overlap_count = 0
    for feature_idx, feature_name in enumerate(alignment_features):
        input_idx = input_index.get(feature_name)
        if input_idx is None:
            continue
        aligned_raw[:, feature_idx] = x_raw[:, input_idx]
        alignment_overlap_count += 1

    try:
        transformed = apply_expression_transform(
            aligned_raw,
            bundle.expression_transform,
        )
    except CVError as exc:
        raise BundleError(str(exc)) from exc

    missing_count = len(alignment_features) - alignment_overlap_count
    extra_count = len(input_feature_set - alignment_feature_set)
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

    schema_index = {feature: idx for idx, feature in enumerate(alignment_features)}

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
        x_model = transformed[:, selected_indices]
        try:
            x_model_scaled = apply_feature_scaling(
                x_model,
                preprocess.scaler,
                bundle.feature_scaling,
            )
        except CVError as exc:
            raise BundleError(str(exc)) from exc
        model_probs.append(_predict_probability(model, x_model_scaled))

    prob = _aggregate_probabilities(model_probs, bundle.probability_aggregation)
    uncertainty_std = np.std(np.vstack(model_probs), axis=0) if len(model_probs) > 1 else None

    pred_label = (prob >= bundle.threshold_fixed).astype(int)
    payload: dict[str, list[Any]] = {
        "species": species_list,
        "prob": prob.astype(float, copy=False).tolist(),
        "pred_label_fixed_threshold": pred_label.astype(int, copy=False).tolist(),
    }
    if uncertainty_std is not None:
        payload["uncertainty_std"] = uncertainty_std.astype(float, copy=False).tolist()

    pred_df = pl.DataFrame(payload).sort("species")
    return pred_df, warnings
