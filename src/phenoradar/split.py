"""Dataset splitting utilities for run-time CV planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from phenoradar.config import AppConfig

_POOL_TRAINING_VALIDATION = "training_validation"
_POOL_EXTERNAL_TEST = "external_test"
_POOL_DISCOVERY_INFERENCE = "discovery_inference"


class SplitError(ValueError):
    """Raised when split construction or preflight checks fail."""


@dataclass(frozen=True)
class SplitArtifacts:
    """Generated split metadata artifacts for one run."""

    split_manifest: pl.DataFrame
    pool_counts: dict[str, int]
    fold_count: int
    expression_rows_excluded: int


def _load_tsv(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path, separator="\t")
    except FileNotFoundError as exc:
        raise SplitError(f"Input file not found: {path}") from exc


def _require_columns(frame: pl.DataFrame, required: list[str], context: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        missing_str = ", ".join(missing)
        raise SplitError(f"Missing required columns in {context}: {missing_str}")


def _normalize_metadata(config: AppConfig) -> pl.DataFrame:
    metadata = _load_tsv(Path(config.data.metadata_path))
    species_col = config.data.species_col
    trait_col = config.data.trait_col
    group_col = config.data.group_col

    _require_columns(metadata, [species_col, trait_col, group_col], "metadata")

    metadata = metadata.with_columns(
        pl.col(species_col).cast(pl.String, strict=False).str.strip_chars().alias("__species"),
        pl.col(trait_col).cast(pl.String, strict=False).str.strip_chars().alias("__trait_raw"),
        pl.col(group_col).cast(pl.String, strict=False).str.strip_chars().alias("__group_raw"),
    )

    invalid_species = metadata.filter(
        pl.col("__species").is_null() | (pl.col("__species") == "")
    ).height
    if invalid_species > 0:
        raise SplitError(f"Metadata contains {invalid_species} rows with empty species identifiers")

    duplicated_species = (
        metadata.group_by("__species")
        .len()
        .filter(pl.col("len") > 1)
        .select("__species")
        .to_series()
        .to_list()
    )
    if duplicated_species:
        duplicates = ", ".join(str(v) for v in sorted(duplicated_species)[:10])
        raise SplitError(f"Metadata species identifiers must be unique; duplicates: {duplicates}")

    invalid_trait_values = (
        metadata.filter(
            pl.col("__trait_raw").is_not_null()
            & (pl.col("__trait_raw") != "")
            & ~pl.col("__trait_raw").is_in(["0", "1"])
        )
        .select("__trait_raw")
        .unique()
        .sort("__trait_raw")
        .to_series()
        .to_list()
    )
    if invalid_trait_values:
        invalid = ", ".join(str(v) for v in invalid_trait_values)
        raise SplitError(
            "Trait column must contain only 0/1 or null/empty values; "
            f"offending values: {invalid}"
        )

    return metadata.with_columns(
        pl.when(pl.col("__trait_raw").is_null() | (pl.col("__trait_raw") == ""))
        .then(None)
        .otherwise(pl.col("__trait_raw").cast(pl.Int8))
        .alias("__label"),
        pl.when(pl.col("__group_raw").is_null() | (pl.col("__group_raw") == ""))
        .then(None)
        .otherwise(pl.col("__group_raw"))
        .alias("__group"),
    ).with_columns(
        pl.when(pl.col("__label").is_not_null() & pl.col("__group").is_not_null())
        .then(pl.lit(_POOL_TRAINING_VALIDATION))
        .when(pl.col("__label").is_not_null() & pl.col("__group").is_null())
        .then(pl.lit(_POOL_EXTERNAL_TEST))
        .otherwise(pl.lit(_POOL_DISCOVERY_INFERENCE))
        .alias("__pool")
    )


def _expression_species_and_excluded_rows(
    config: AppConfig, metadata_species: list[str]
) -> tuple[set[str], int]:
    expression_path = Path(config.data.tpm_path)
    species_col = config.data.species_col
    try:
        scan = pl.scan_csv(expression_path, separator="\t")
    except FileNotFoundError as exc:
        raise SplitError(f"Input file not found: {expression_path}") from exc

    columns = scan.collect_schema().names()
    if species_col not in columns:
        raise SplitError(f"Missing required column in expression data: {species_col}")

    species_expr = (
        pl.col(species_col).cast(pl.String, strict=False).str.strip_chars().alias("__species")
    )
    expression_species = (
        scan.select(species_expr)
        .filter(pl.col("__species").is_not_null() & (pl.col("__species") != ""))
        .unique()
        .collect()
        .to_series()
        .to_list()
    )
    excluded_rows = (
        scan.select(species_expr)
        .filter(~pl.col("__species").is_in(metadata_species))
        .select(pl.len().alias("n"))
        .collect()
        .item()
    )
    if not isinstance(excluded_rows, int):
        raise SplitError("Failed to compute expression rows excluded from metadata")
    return set(str(v) for v in expression_species), excluded_rows


def _validate_expression_coverage(metadata: pl.DataFrame, expression_species: set[str]) -> None:
    required_species = metadata.select("__species").to_series().to_list()
    missing = sorted(set(str(v) for v in required_species) - expression_species)
    if missing:
        missing_preview = ", ".join(missing[:10])
        raise SplitError(
            "Every species in metadata pools must exist in expression data; "
            f"missing species include: {missing_preview}"
        )


def _preflight_group_labels(training_df: pl.DataFrame) -> None:
    if training_df.height == 0:
        raise SplitError("No species available in training and validation pool")

    invalid_groups = (
        training_df.group_by("__group")
        .agg(pl.col("__label").n_unique().alias("n_labels"))
        .filter(pl.col("n_labels") < 2)
        .select("__group")
        .to_series()
        .to_list()
    )
    if invalid_groups:
        groups_str = ", ".join(str(v) for v in sorted(invalid_groups))
        raise SplitError(
            "Each training group must contain both labels before CV; "
            f"offending groups: {groups_str}"
        )


def _build_fold_indices(
    config: AppConfig, training_df: pl.DataFrame
) -> list[tuple[list[int], list[int]]]:
    groups = training_df.select("__group").to_series().to_list()
    labels = training_df.select("__label").to_series().to_list()
    indices = list(range(training_df.height))

    split_iter: Any
    if config.split.outer_cv_strategy == "logo":
        split_iter = LeaveOneGroupOut().split(indices, labels, groups)
    else:
        n_splits = config.split.outer_cv_n_splits
        if n_splits is None:
            raise SplitError("split.outer_cv_n_splits must be set for group_kfold")
        split_iter = GroupKFold(n_splits=n_splits).split(indices, labels, groups)

    folds: list[tuple[list[int], list[int]]] = []
    try:
        for train_idx, valid_idx in split_iter:
            folds.append((list(train_idx), list(valid_idx)))
    except ValueError as exc:
        raise SplitError(str(exc)) from exc

    if not folds:
        raise SplitError("Outer CV produced zero folds")
    return folds


def _append_training_rows(
    rows: list[dict[str, Any]],
    training_df: pl.DataFrame,
    fold_id: int,
    row_indices: list[int],
    pool: str,
) -> None:
    species = training_df.select("__species").to_series().to_list()
    groups = training_df.select("__group").to_series().to_list()
    labels = training_df.select("__label").to_series().to_list()

    for idx in row_indices:
        rows.append(
            {
                "species": str(species[idx]),
                "pool": pool,
                "fold_id": str(fold_id),
                "group_id": str(groups[idx]),
                "label": int(labels[idx]),
            }
        )


def _append_pool_rows(rows: list[dict[str, Any]], pool_df: pl.DataFrame, pool_name: str) -> None:
    species = pool_df.select("__species").to_series().to_list()
    labels = pool_df.select("__label").to_series().to_list()
    for idx in range(pool_df.height):
        label_value = labels[idx]
        rows.append(
            {
                "species": str(species[idx]),
                "pool": pool_name,
                "fold_id": "NA",
                "group_id": None,
                "label": None if label_value is None else int(label_value),
            }
        )


def _build_split_manifest(
    training_df: pl.DataFrame,
    external_df: pl.DataFrame,
    inference_df: pl.DataFrame,
    folds: list[tuple[list[int], list[int]]],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        _append_training_rows(
            rows, training_df, fold_id=fold_id, row_indices=train_idx, pool="train"
        )
        _append_training_rows(
            rows,
            training_df,
            fold_id=fold_id,
            row_indices=valid_idx,
            pool="validation",
        )

    _append_pool_rows(rows, external_df, pool_name="external_test")
    _append_pool_rows(rows, inference_df, pool_name="discovery_inference")

    if not rows:
        raise SplitError("Split manifest is empty")

    manifest = pl.DataFrame(
        rows,
        schema={
            "species": pl.String,
            "pool": pl.String,
            "fold_id": pl.String,
            "group_id": pl.String,
            "label": pl.Int8,
        },
    )
    return manifest.sort(by=["pool", "fold_id", "group_id", "species"], nulls_last=False)


def build_split_artifacts(config: AppConfig) -> SplitArtifacts:
    """Build split manifest and related counts from input config."""
    metadata = _normalize_metadata(config)
    metadata_species = metadata.select("__species").to_series().to_list()
    expression_species, excluded_rows = _expression_species_and_excluded_rows(
        config, metadata_species
    )
    _validate_expression_coverage(metadata, expression_species)

    training_df = metadata.filter(pl.col("__pool") == _POOL_TRAINING_VALIDATION)
    external_df = metadata.filter(pl.col("__pool") == _POOL_EXTERNAL_TEST)
    inference_df = metadata.filter(pl.col("__pool") == _POOL_DISCOVERY_INFERENCE)

    _preflight_group_labels(training_df)
    folds = _build_fold_indices(config, training_df)
    manifest = _build_split_manifest(training_df, external_df, inference_df, folds)

    pool_counts = {
        _POOL_TRAINING_VALIDATION: training_df.height,
        _POOL_EXTERNAL_TEST: external_df.height,
        _POOL_DISCOVERY_INFERENCE: inference_df.height,
    }

    return SplitArtifacts(
        split_manifest=manifest,
        pool_counts=pool_counts,
        fold_count=len(folds),
        expression_rows_excluded=excluded_rows,
    )
