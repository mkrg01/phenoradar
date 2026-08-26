"""Candidate-level evidence tables for predicted-positive inference species."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from phenoradar.config import AppConfig
from phenoradar.cv import (
    CVError,
    ExpressionMatrixBuilder,
    FinalModelEntry,
    FinalRefitArtifacts,
    apply_expression_transform,
    apply_feature_scaling,
)
from phenoradar.interpret import _linear_coefficients


class CandidateEvidenceError(ValueError):
    """Raised when candidate-evidence artifacts cannot be constructed."""


@dataclass(frozen=True)
class CandidateEvidenceArtifacts:
    """Tables required to render one evidence figure per positive candidate."""

    candidates: pl.DataFrame
    features: pl.DataFrame
    reference_expression: pl.DataFrame
    cross_fold_predictions: pl.DataFrame
    warnings: list[str]


_CANDIDATE_SCHEMA = {
    "species": pl.String,
    "prob": pl.Float64,
    "family_id": pl.String,
    "family_name": pl.String,
}
_FEATURE_SCHEMA = {
    "species": pl.String,
    "feature": pl.String,
    "local_rank": pl.Int64,
    "contribution_mean": pl.Float64,
    "contribution_mean_abs": pl.Float64,
    "contribution_min": pl.Float64,
    "contribution_max": pl.Float64,
    "candidate_tpm": pl.Float64,
    "candidate_log2_tpm_plus1": pl.Float64,
    "n_models": pl.Int64,
}
_REFERENCE_SCHEMA = {
    "species": pl.String,
    "label": pl.Int64,
    "feature": pl.String,
    "tpm": pl.Float64,
    "log2_tpm_plus1": pl.Float64,
}
_CROSS_FOLD_SCHEMA = {
    "fold_id": pl.String,
    "species": pl.String,
    "prob": pl.Float64,
}
_NONZERO_TOLERANCE = 1e-12


def _empty_artifacts(*, warnings: list[str]) -> CandidateEvidenceArtifacts:
    return CandidateEvidenceArtifacts(
        candidates=pl.DataFrame(schema=_CANDIDATE_SCHEMA),
        features=pl.DataFrame(schema=_FEATURE_SCHEMA),
        reference_expression=pl.DataFrame(schema=_REFERENCE_SCHEMA),
        cross_fold_predictions=pl.DataFrame(schema=_CROSS_FOLD_SCHEMA),
        warnings=warnings,
    )


def _require_columns(frame: pl.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CandidateEvidenceError(
            f"{context} is missing required column(s): {', '.join(missing)}"
        )


def _positive_candidates(pred_inference: pl.DataFrame) -> pl.DataFrame:
    _require_columns(
        pred_inference,
        {"species", "prob", "pred_label_fixed_threshold"},
        "prediction_inference.tsv",
    )
    return (
        pred_inference.select(
            pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
            pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
            pl.col("pred_label_fixed_threshold")
            .cast(pl.Int64, strict=False)
            .alias("pred_label_fixed_threshold"),
        )
        .filter(
            pl.col("species").is_not_null()
            & (pl.col("species") != "")
            & pl.col("prob").is_not_null()
            & pl.col("prob").is_finite()
            & (pl.col("pred_label_fixed_threshold") == 1)
        )
        .unique("species")
        .sort(["prob", "species"], descending=[True, False])
    )


def _known_species(split_manifest: pl.DataFrame) -> pl.DataFrame:
    _require_columns(split_manifest, {"species", "pool", "label"}, "split_manifest.tsv")
    known = (
        split_manifest.filter(pl.col("pool").is_in(["train", "validation"]))
        .select(
            pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
            pl.col("label").cast(pl.Int64, strict=False).alias("label"),
        )
        .drop_nulls(["species", "label"])
        .unique()
        .sort("species")
    )
    inconsistent = known.group_by("species").agg(pl.col("label").n_unique().alias("n_labels"))
    if inconsistent.filter(pl.col("n_labels") != 1).height > 0:
        raise CandidateEvidenceError(
            "split_manifest.tsv assigns inconsistent labels to a reference species"
        )
    return known.unique("species").sort("species")


def _final_model_entries(final_refit: FinalRefitArtifacts) -> list[FinalModelEntry]:
    entries = list(getattr(final_refit, "model_entries", []))
    if entries:
        return entries
    models = list(getattr(final_refit, "models", []))
    feature_names = list(getattr(final_refit, "feature_names", []))
    if not models or not feature_names:
        return []
    return [
        FinalModelEntry(
            feature_names=feature_names,
            scaler=getattr(final_refit, "scaler", None),
            model=model,
        )
        for model in models
    ]


def _candidate_family_metadata(
    config: AppConfig,
    candidates: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    warnings: list[str] = []
    try:
        metadata = pl.read_csv(Path(config.data.metadata_path), separator="\t")
    except FileNotFoundError as exc:
        raise CandidateEvidenceError(
            f"Metadata file not found for candidate evidence: {config.data.metadata_path}"
        ) from exc
    except Exception as exc:
        raise CandidateEvidenceError(
            f"Failed to read metadata TSV for candidate evidence: {config.data.metadata_path}"
        ) from exc

    species_col = config.data.species_col
    _require_columns(metadata, {species_col}, "metadata TSV")
    family_id_col: str | None = "family_id" if "family_id" in metadata.columns else None
    family_name_col: str | None = "family_name" if "family_name" in metadata.columns else None
    if family_id_col is None and config.summary.group_col in metadata.columns:
        family_id_col = config.summary.group_col
        configured_name = config.summary.group_name_col
        if configured_name is not None and configured_name in metadata.columns:
            family_name_col = configured_name
    if family_id_col is None:
        warnings.append(
            "Candidate evidence metadata has no family_id or configured summary group; "
            "family labels were emitted as unassigned"
        )
        family_id_expr = pl.lit("unassigned", dtype=pl.String).alias("family_id")
    else:
        family_id_expr = (
            pl.col(family_id_col).cast(pl.String, strict=False).str.strip_chars().alias("family_id")
        )
    if family_name_col is None:
        family_name_expr = family_id_expr.alias("family_name")
    else:
        family_name_expr = (
            pl.col(family_name_col)
            .cast(pl.String, strict=False)
            .str.strip_chars()
            .alias("family_name")
        )

    family = (
        metadata.select(
            pl.col(species_col).cast(pl.String, strict=False).str.strip_chars().alias("species"),
            family_id_expr,
            family_name_expr,
        )
        .unique("species")
        .with_columns(
            pl.when(pl.col("family_id").is_null() | (pl.col("family_id") == ""))
            .then(pl.lit("unassigned"))
            .otherwise(pl.col("family_id"))
            .alias("family_id"),
            pl.when(pl.col("family_name").is_null() | (pl.col("family_name") == ""))
            .then(pl.col("family_id"))
            .otherwise(pl.col("family_name"))
            .alias("family_name"),
        )
    )
    return (
        candidates.select("species", "prob")
        .join(family, on="species", how="left")
        .with_columns(
            pl.col("family_id").fill_null("unassigned"),
            pl.col("family_name").fill_null("unassigned"),
        )
        .sort(["prob", "species"], descending=[True, False]),
        warnings,
    )


def _normalized_cross_fold_predictions(
    cross_fold_predictions: pl.DataFrame | None,
    *,
    candidate_species: list[str],
) -> pl.DataFrame:
    if cross_fold_predictions is None or cross_fold_predictions.height == 0:
        return pl.DataFrame(schema=_CROSS_FOLD_SCHEMA)
    _require_columns(
        cross_fold_predictions,
        {"fold_id", "species", "prob"},
        "prediction_inference_by_fold.tsv",
    )
    return (
        cross_fold_predictions.select(
            pl.col("fold_id").cast(pl.String, strict=False).alias("fold_id"),
            pl.col("species").cast(pl.String, strict=False).str.strip_chars().alias("species"),
            pl.col("prob").cast(pl.Float64, strict=False).alias("prob"),
        )
        .filter(
            pl.col("species").is_in(candidate_species)
            & pl.col("prob").is_not_null()
            & pl.col("prob").is_finite()
        )
        .sort(["species", "fold_id"])
    )


def build_candidate_evidence_artifacts(
    *,
    config: AppConfig,
    split_manifest: pl.DataFrame,
    final_refit: FinalRefitArtifacts,
    cross_fold_predictions: pl.DataFrame | None,
    top_features: int,
) -> CandidateEvidenceArtifacts:
    """Build candidate-local contributions and known-trait expression references."""
    if top_features < 1:
        raise CandidateEvidenceError("candidate evidence top_features must be >= 1")

    warnings: list[str] = []
    positive = _positive_candidates(final_refit.pred_inference)
    if positive.height == 0:
        warnings.append("Skipped candidate evidence figures: no inference species predicted as 1")
        return _empty_artifacts(warnings=warnings)

    entries = _final_model_entries(final_refit)
    if not entries:
        warnings.append("Skipped candidate evidence figures: final refit has no fitted models")
        return _empty_artifacts(warnings=warnings)
    coefficients: list[np.ndarray] = []
    for entry in entries:
        coef = _linear_coefficients(entry.model)
        if coef is None:
            warnings.append(
                "Skipped candidate evidence figures: local contribution is unavailable "
                "for the final model family"
            )
            return _empty_artifacts(warnings=warnings)
        if coef.shape[0] != len(entry.feature_names):
            raise CandidateEvidenceError(
                "Final model coefficient width does not match its feature schema"
            )
        coefficients.append(np.asarray(coef, dtype=float))

    candidate_species = [str(value) for value in positive.get_column("species").to_list()]
    known = _known_species(split_manifest)
    known_species = [str(value) for value in known.get_column("species").to_list()]
    transform_features = [str(value) for value in final_refit.transform_feature_names]
    if not transform_features:
        raise CandidateEvidenceError("Final refit transform feature schema is empty")
    transform_index = {feature: idx for idx, feature in enumerate(transform_features)}
    if len(transform_index) != len(transform_features):
        raise CandidateEvidenceError("Final refit transform feature schema contains duplicates")

    model_features = sorted({feature for entry in entries for feature in entry.feature_names})
    missing_model_features = sorted(set(model_features) - set(transform_features))
    if missing_model_features:
        raise CandidateEvidenceError(
            "Final model features are absent from the transform schema: "
            + ", ".join(missing_model_features[:10])
        )

    try:
        matrix_builder = ExpressionMatrixBuilder(config)
        matrix_builder.cache_species([*known_species, *candidate_species])
        candidate_raw, _ = matrix_builder.build_matrix(
            candidate_species,
            feature_order=transform_features,
        )
        candidate_transformed = apply_expression_transform(
            candidate_raw,
            config.preprocess.expression_transform.method,
        )
    except CVError as exc:
        raise CandidateEvidenceError(str(exc)) from exc

    feature_union_index = {feature: idx for idx, feature in enumerate(model_features)}
    contribution_cube = np.zeros(
        (len(entries), len(candidate_species), len(model_features)), dtype=float
    )
    for model_idx, (entry, coef) in enumerate(zip(entries, coefficients, strict=True)):
        transform_indices = np.array(
            [transform_index[feature] for feature in entry.feature_names], dtype=int
        )
        selected = candidate_transformed[:, transform_indices]
        try:
            scaled = apply_feature_scaling(
                selected,
                entry.scaler,
                config.preprocess.feature_scaling.method,
            )
        except CVError as exc:
            raise CandidateEvidenceError(str(exc)) from exc
        model_contributions = scaled * coef[np.newaxis, :]
        union_indices = np.array(
            [feature_union_index[feature] for feature in entry.feature_names], dtype=int
        )
        contribution_cube[model_idx][:, union_indices] = model_contributions

    contribution_mean = np.mean(contribution_cube, axis=0)
    contribution_mean_abs = np.mean(np.abs(contribution_cube), axis=0)
    contribution_min = np.min(contribution_cube, axis=0)
    contribution_max = np.max(contribution_cube, axis=0)
    feature_rows: list[dict[str, Any]] = []
    selected_features: set[str] = set()
    for candidate_idx, species in enumerate(candidate_species):
        order = sorted(
            range(len(model_features)),
            key=lambda idx: (
                -float(contribution_mean_abs[candidate_idx, idx]),
                model_features[idx],
            ),
        )
        nonzero_order = [
            idx
            for idx in order
            if float(contribution_mean_abs[candidate_idx, idx]) > _NONZERO_TOLERANCE
        ][:top_features]
        if not nonzero_order:
            warnings.append(
                f"Skipped candidate evidence figure for {species}: "
                "all final-model local contributions are zero"
            )
            continue
        for local_rank, feature_idx in enumerate(nonzero_order, start=1):
            feature = model_features[feature_idx]
            raw_value = float(candidate_raw[candidate_idx, transform_index[feature]])
            selected_features.add(feature)
            feature_rows.append(
                {
                    "species": species,
                    "feature": feature,
                    "local_rank": local_rank,
                    "contribution_mean": float(contribution_mean[candidate_idx, feature_idx]),
                    "contribution_mean_abs": float(
                        contribution_mean_abs[candidate_idx, feature_idx]
                    ),
                    "contribution_min": float(contribution_min[candidate_idx, feature_idx]),
                    "contribution_max": float(contribution_max[candidate_idx, feature_idx]),
                    "candidate_tpm": raw_value,
                    "candidate_log2_tpm_plus1": float(np.log2(raw_value + 1.0)),
                    "n_models": len(entries),
                }
            )

    if not feature_rows:
        warnings.append("Skipped candidate evidence figures: no non-zero local features")
        return _empty_artifacts(warnings=warnings)
    features = pl.DataFrame(feature_rows, schema=_FEATURE_SCHEMA).sort(["species", "local_rank"])
    retained_candidate_species = features.get_column("species").unique().to_list()
    candidates, metadata_warnings = _candidate_family_metadata(
        config,
        positive.filter(pl.col("species").is_in(retained_candidate_species)),
    )
    warnings.extend(metadata_warnings)

    reference_features = sorted(selected_features)
    try:
        reference_raw, _ = matrix_builder.build_matrix(
            known_species,
            feature_order=reference_features,
        )
    except CVError as exc:
        raise CandidateEvidenceError(str(exc)) from exc
    known_labels = [int(value) for value in known.get_column("label").to_list()]
    reference_rows: list[dict[str, Any]] = []
    for species_idx, (species, label) in enumerate(zip(known_species, known_labels, strict=True)):
        for feature_idx, feature in enumerate(reference_features):
            raw_value = float(reference_raw[species_idx, feature_idx])
            reference_rows.append(
                {
                    "species": species,
                    "label": label,
                    "feature": feature,
                    "tpm": raw_value,
                    "log2_tpm_plus1": float(np.log2(raw_value + 1.0)),
                }
            )
    reference_expression = pl.DataFrame(reference_rows, schema=_REFERENCE_SCHEMA).sort(
        ["feature", "label", "species"]
    )
    normalized_cross_fold = _normalized_cross_fold_predictions(
        cross_fold_predictions,
        candidate_species=[str(value) for value in retained_candidate_species],
    )
    if normalized_cross_fold.height == 0:
        warnings.append(
            "Candidate evidence figures have no outer-CV model predictions; "
            "the prediction-stability panel will show final refit only"
        )

    return CandidateEvidenceArtifacts(
        candidates=candidates,
        features=features,
        reference_expression=reference_expression,
        cross_fold_predictions=normalized_cross_fold,
        warnings=warnings,
    )
