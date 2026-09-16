"""Local interpretation of bundle predictions without training or CV refits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from phenoradar.abstention import ABSTENTION_COLUMNS
from phenoradar.bundle import (
    LoadedBundle,
    _predict_with_jobs,
    bundle_preprocess_entries,
    prepare_bundle_input,
)
from phenoradar.candidate_evidence import (
    _FEATURE_SCHEMA,
    _REFERENCE_SCHEMA,
    _positive_candidates,
)
from phenoradar.config import PredictConfig
from phenoradar.cv import apply_feature_scaling
from phenoradar.interpret import _linear_coefficients
from phenoradar.orthogroup_annotation import load_orthogroup_annotations


@dataclass
class PredictEvidenceArtifacts:
    candidates: pl.DataFrame = field(default_factory=pl.DataFrame)
    features: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=_FEATURE_SCHEMA))
    reference_expression: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=_REFERENCE_SCHEMA)
    )
    model_predictions: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={"species": pl.String, "model_index": pl.Int64, "prob": pl.Float64}
        )
    )
    annotations: pl.DataFrame | None = None
    trait_name: str = "trait"
    input_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _source_run(bundle: LoadedBundle) -> Path | None:
    for path in [bundle.bundle_dir.parent, Path(str(bundle.manifest.get("source_run_dir", "")))]:
        if path.name == bundle.source_run_id and path.is_dir():
            return path
    return None


def _training_annotation_path(raw_path: str, bundle: LoadedBundle) -> Path | None:
    path = Path(raw_path)
    if path.is_absolute():
        return path if path.is_file() else None
    for root in [Path.cwd(), *bundle.bundle_dir.resolve().parents]:
        if (root / path).is_file():
            return root / path
    return None


def _load_interpretation_context(
    artifacts: PredictEvidenceArtifacts,
    config: PredictConfig,
    bundle: LoadedBundle,
) -> None:
    saved = yaml.safe_load((bundle.bundle_dir / "resolved_config.yml").read_text()) or {}
    data = saved.get("data", {})
    artifacts.trait_name = str(bundle.manifest.get("trait_name") or data.get("trait_col", "trait"))
    features = artifacts.features["feature"].unique().to_list()
    reference = bundle.reference_expression
    source = _source_run(bundle)
    if reference is None and source is not None:
        path = source / "inference/tables/candidate_reference_expression.tsv"
        if path.is_file():
            reference = pl.read_csv(
                path,
                separator="\t",
                null_values="NA",
                schema_overrides=_REFERENCE_SCHEMA,
            )
            artifacts.input_paths.append(path)
    if reference is not None:
        artifacts.reference_expression = reference.filter(pl.col("feature").is_in(features))
    available = set(artifacts.reference_expression["feature"].to_list())
    missing = sorted(set(features) - available)
    if missing:
        artifacts.warnings.append(
            f"Candidate evidence has no known-trait reference for {len(missing)} feature(s); "
            "expression panels mark unavailable references."
        )

    annotations = bundle.orthogroup_annotations
    annotation_path = (
        Path(config.data.orthogroup_annotation_path)
        if config.data.orthogroup_annotation_path is not None
        else None
    )
    if annotation_path is None and annotations is None and data.get("orthogroup_annotation_path"):
        annotation_path = _training_annotation_path(str(data["orthogroup_annotation_path"]), bundle)
    if annotation_path is not None:
        annotations = load_orthogroup_annotations(annotation_path, feature_names=features)
        artifacts.input_paths.append(annotation_path)
    artifacts.annotations = annotations


def build_predict_evidence_artifacts(
    *,
    config: PredictConfig,
    bundle: LoadedBundle,
    predictions: pl.DataFrame,
) -> PredictEvidenceArtifacts:
    """Explain accepted positive candidates using exactly the bundled transforms."""
    artifacts = PredictEvidenceArtifacts()
    positive = _positive_candidates(predictions)
    if positive.height == 0:
        return artifacts
    coefficients = [_linear_coefficients(model) for model in bundle.models]
    if not coefficients or any(coef is None for coef in coefficients):
        artifacts.warnings.append(
            "Skipped candidate evidence figures: local linear contributions are unavailable "
            "for this bundle's model family."
        )
        return artifacts
    species = [str(value) for value in positive["species"].to_list()]
    raw, transformed, alignment, _ = prepare_bundle_input(config, bundle, species)
    index = {name: i for i, name in enumerate(alignment)}
    model_index = {name: i for i, name in enumerate(bundle.feature_names)}
    entries = bundle_preprocess_entries(bundle)
    cube = np.zeros((len(bundle.models), len(species), len(bundle.feature_names)))
    probability_rows: list[dict[str, Any]] = []
    for number, (model, entry, coef) in enumerate(
        zip(
            bundle.models,
            entries,
            coefficients,
            strict=True,
        )
    ):
        assert coef is not None
        if coef.shape != (len(entry.feature_names),):
            raise ValueError("Bundle coefficient width does not match its feature schema")
        selected = transformed[:, [index[name] for name in entry.feature_names]]
        scaled = apply_feature_scaling(selected, entry.scaler, bundle.feature_scaling)
        cube[number][:, [model_index[name] for name in entry.feature_names]] = scaled * coef
        probabilities = _predict_with_jobs(model, scaled, config.runtime.n_jobs)
        probability_rows.extend(
            {"species": name, "model_index": number, "prob": float(probability)}
            for name, probability in zip(species, probabilities, strict=True)
        )
    means, magnitudes = cube.mean(axis=0), np.abs(cube).mean(axis=0)
    minima, maxima = cube.min(axis=0), cube.max(axis=0)
    rows: list[dict[str, Any]] = []
    for row, name in enumerate(species):
        order = sorted(
            range(len(bundle.feature_names)),
            key=lambda i: (
                -float(magnitudes[row, i]),
                bundle.feature_names[i],
            ),
        )
        selected_features = [i for i in order if magnitudes[row, i] > 1e-12][
            : config.figures.top_features
        ]
        if not selected_features:
            artifacts.warnings.append(
                f"Skipped candidate evidence figure for {name}: all local contributions are zero"
            )
        for rank, i in enumerate(selected_features, 1):
            feature = bundle.feature_names[i]
            tpm = float(raw[row, index[feature]])
            rows.append(
                {
                    "species": name,
                    "feature": feature,
                    "local_rank": rank,
                    "contribution_mean": float(means[row, i]),
                    "contribution_mean_abs": float(magnitudes[row, i]),
                    "contribution_min": float(minima[row, i]),
                    "contribution_max": float(maxima[row, i]),
                    "candidate_tpm": tpm,
                    "candidate_log2_tpm_plus1": float(np.log2(tpm + 1)),
                    "n_models": len(bundle.models),
                }
            )
    if not rows:
        return artifacts
    artifacts.features = pl.DataFrame(rows, schema=_FEATURE_SCHEMA).sort(["species", "local_rank"])
    retained = artifacts.features["species"].unique().to_list()
    candidates = positive.filter(pl.col("species").is_in(retained)).drop(
        "pred_label_fixed_threshold"
    )
    if config.data.metadata_path is not None:
        metadata = pl.read_csv(config.data.metadata_path, separator="\t")
        if "family" in metadata.columns:
            candidates = candidates.join(
                metadata.select(
                    pl.col(config.data.species_col)
                    .cast(pl.String)
                    .str.strip_chars()
                    .alias("species"),
                    pl.col("family").cast(pl.String),
                ).unique("species"),
                on="species",
                how="left",
            )
    if "family" not in candidates.columns:
        candidates = candidates.with_columns(pl.lit("unassigned").alias("family"))
    else:
        candidates = candidates.with_columns(pl.col("family").fill_null("unassigned"))
    coverage = [name for name in ABSTENTION_COLUMNS if name in predictions.columns]
    artifacts.candidates = candidates.join(
        predictions.select("species", *coverage), on="species", how="left"
    ).sort(["prob", "species"], descending=[True, False])
    artifacts.model_predictions = (
        pl.DataFrame(probability_rows)
        .filter(pl.col("species").is_in(retained))
        .sort(["species", "model_index"])
    )
    _load_interpretation_context(artifacts, config, bundle)
    return artifacts
