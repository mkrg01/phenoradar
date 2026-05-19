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
_POOL_EXCLUDED = "excluded"
_TRUE_FLAG_VALUES = {"1", "true", "yes"}
_FALSE_FLAG_VALUES = {"0", "false", "no", ""}


class SplitError(ValueError):
    """Raised when split construction or preflight checks fail."""


@dataclass(frozen=True)
class SplitArtifacts:
    """Generated split metadata artifacts for one run."""

    split_manifest: pl.DataFrame
    fold_validation_groups: pl.DataFrame
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


def _normalized_string_expr(column: str, alias: str) -> pl.Expr:
    return pl.col(column).cast(pl.String, strict=False).str.strip_chars().alias(alias)


def _normalize_metadata(config: AppConfig) -> pl.DataFrame:
    metadata = _load_tsv(Path(config.data.metadata_path))
    species_col = config.data.species_col
    trait_col = config.data.trait_col
    split_group_col = config.split.group_col
    contrast_pair_col = config.data.contrast_pair_col
    test_holdout_col = config.split.test_holdout_col
    exclude_col = config.split.exclude_col

    required = [species_col, trait_col, split_group_col]
    if contrast_pair_col is not None:
        required.append(contrast_pair_col)
    if test_holdout_col is not None:
        required.append(test_holdout_col)
    if exclude_col is not None:
        required.append(exclude_col)
    _require_columns(metadata, required, "metadata")

    columns = [
        _normalized_string_expr(species_col, "__species"),
        _normalized_string_expr(trait_col, "__trait_raw"),
        _normalized_string_expr(split_group_col, "__group_raw"),
    ]
    if contrast_pair_col is None:
        columns.append(pl.lit(None, dtype=pl.String).alias("__contrast_group_raw"))
    else:
        columns.append(_normalized_string_expr(contrast_pair_col, "__contrast_group_raw"))
    if test_holdout_col is None:
        columns.append(pl.lit("no", dtype=pl.String).alias("__test_holdout_raw"))
    else:
        columns.append(_normalized_string_expr(test_holdout_col, "__test_holdout_raw"))
    if exclude_col is None:
        columns.append(pl.lit("no", dtype=pl.String).alias("__exclude_raw"))
    else:
        columns.append(_normalized_string_expr(exclude_col, "__exclude_raw"))

    metadata = metadata.with_columns(columns)

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

    normalized = metadata.with_columns(
        pl.col("__test_holdout_raw").str.to_lowercase().alias("__test_holdout_norm"),
        pl.col("__exclude_raw").str.to_lowercase().alias("__exclude_norm"),
    )
    invalid_holdout_values = (
        normalized.filter(
            pl.col("__test_holdout_norm").is_not_null()
            & ~pl.col("__test_holdout_norm").is_in(
                sorted(_TRUE_FLAG_VALUES | _FALSE_FLAG_VALUES)
            )
        )
        .select("__test_holdout_raw")
        .unique()
        .sort("__test_holdout_raw")
        .to_series()
        .to_list()
    )
    if invalid_holdout_values:
        invalid = ", ".join(str(v) for v in invalid_holdout_values)
        holdout_name = "split.test_holdout_col" if test_holdout_col is None else test_holdout_col
        raise SplitError(
            f"Holdout column {holdout_name} must contain yes/no, true/false, 1/0, "
            f"or null/empty values; offending values: {invalid}"
        )
    invalid_exclude_values = (
        normalized.filter(
            pl.col("__exclude_norm").is_not_null()
            & ~pl.col("__exclude_norm").is_in(sorted(_TRUE_FLAG_VALUES | _FALSE_FLAG_VALUES))
        )
        .select("__exclude_raw")
        .unique()
        .sort("__exclude_raw")
        .to_series()
        .to_list()
    )
    if invalid_exclude_values:
        invalid = ", ".join(str(v) for v in invalid_exclude_values)
        exclude_name = "split.exclude_col" if exclude_col is None else exclude_col
        raise SplitError(
            f"Exclude column {exclude_name} must contain yes/no, true/false, 1/0, "
            f"or null/empty values; offending values: {invalid}"
        )

    normalized = normalized.with_columns(
        pl.when(pl.col("__trait_raw").is_null() | (pl.col("__trait_raw") == ""))
        .then(None)
        .otherwise(pl.col("__trait_raw").cast(pl.Int8))
        .alias("__label"),
        pl.when(pl.col("__group_raw").is_null() | (pl.col("__group_raw") == ""))
        .then(None)
        .otherwise(pl.col("__group_raw"))
        .alias("__group"),
        pl.when(
            pl.col("__contrast_group_raw").is_null() | (pl.col("__contrast_group_raw") == "")
        )
        .then(None)
        .otherwise(pl.col("__contrast_group_raw"))
        .alias("__contrast_group"),
        pl.col("__test_holdout_norm")
        .is_in(sorted(_TRUE_FLAG_VALUES))
        .fill_null(False)
        .alias("__test_holdout"),
        pl.col("__exclude_norm")
        .is_in(sorted(_TRUE_FLAG_VALUES))
        .fill_null(False)
        .alias("__exclude"),
    )

    unlabeled_holdouts = normalized.filter(
        pl.col("__label").is_null() & pl.col("__test_holdout") & ~pl.col("__exclude")
    ).height
    if unlabeled_holdouts > 0:
        raise SplitError(
            f"Holdout column marks {unlabeled_holdouts} trait-missing species as test holdout"
        )

    mixed_holdout_groups = (
        normalized.filter(
            pl.col("__group").is_not_null()
            & pl.col("__label").is_not_null()
            & ~pl.col("__exclude")
        )
        .group_by("__group")
        .agg(pl.col("__test_holdout").n_unique().alias("n_holdout_values"))
        .filter(pl.col("n_holdout_values") > 1)
        .select("__group")
        .to_series()
        .to_list()
    )
    if mixed_holdout_groups:
        groups_str = ", ".join(str(v) for v in sorted(mixed_holdout_groups)[:10])
        raise SplitError(
            "Each split group must have a consistent test-holdout assignment; "
            f"offending groups: {groups_str}"
        )

    missing_training_groups = (
        normalized.filter(
            pl.col("__label").is_not_null()
            & ~pl.col("__test_holdout")
            & ~pl.col("__exclude")
            & pl.col("__group").is_null()
        )
        .select("__species")
        .to_series()
        .to_list()
    )
    if missing_training_groups:
        species_str = ", ".join(str(v) for v in sorted(missing_training_groups)[:10])
        raise SplitError(
            "Labeled non-holdout species must have a non-empty split group; "
            f"offending species: {species_str}"
        )

    return normalized.with_columns(
        pl.when(
            pl.col("__label").is_not_null()
            & ~pl.col("__test_holdout")
            & ~pl.col("__exclude")
            & pl.col("__group").is_not_null()
        )
        .then(pl.lit(_POOL_TRAINING_VALIDATION))
        .when(pl.col("__label").is_not_null() & pl.col("__test_holdout") & ~pl.col("__exclude"))
        .then(pl.lit(_POOL_EXTERNAL_TEST))
        .when(pl.col("__exclude"))
        .then(pl.lit(_POOL_EXCLUDED))
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


def _preflight_training_pool(config: AppConfig, training_df: pl.DataFrame) -> None:
    if training_df.height == 0:
        raise SplitError("No species available in training and validation pool")

    if training_df.select(pl.col("__label").n_unique()).item() < 2:
        raise SplitError("Training and validation pool must contain both labels before CV")

    if config.sampling.strategy == "group_balanced":
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
                "group_balanced sampling requires both labels in each split group before CV; "
                f"offending groups: {groups_str}"
            )

    if config.preprocess.pair_aware_filter.enabled:
        missing_contrast_groups = training_df.filter(pl.col("__contrast_group").is_null()).height
        if missing_contrast_groups > 0:
            raise SplitError(
                "pair_aware_filter requires non-empty data.contrast_pair_col values for all "
                f"training species; missing={missing_contrast_groups}"
            )
        invalid_contrast_groups = (
            training_df.group_by("__contrast_group")
            .agg(pl.col("__label").n_unique().alias("n_labels"))
            .filter(pl.col("n_labels") < 2)
            .select("__contrast_group")
            .to_series()
            .to_list()
        )
        if invalid_contrast_groups:
            groups_str = ", ".join(str(v) for v in sorted(invalid_contrast_groups))
            raise SplitError(
                "pair_aware_filter requires both labels in each contrast pair before CV; "
                f"offending contrast groups: {groups_str}"
            )


def _validate_fold_labels(
    training_df: pl.DataFrame, folds: list[tuple[list[int], list[int]]]
) -> None:
    labels = training_df.select("__label").to_series().to_list()
    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        train_labels = {int(labels[idx]) for idx in train_idx}
        valid_labels = {int(labels[idx]) for idx in valid_idx}
        if len(train_labels) < 2:
            raise SplitError(f"Fold {fold_id} training split contains fewer than two labels")
        if len(valid_labels) < 2:
            raise SplitError(f"Fold {fold_id} validation split contains fewer than two labels")


def _validate_pair_aware_fold_contrasts(
    config: AppConfig,
    training_df: pl.DataFrame,
    folds: list[tuple[list[int], list[int]]],
) -> None:
    if not config.preprocess.pair_aware_filter.enabled:
        return
    labels = training_df.select("__label").to_series().to_list()
    contrast_groups = training_df.select("__contrast_group").to_series().to_list()
    for fold_id, (train_idx, _valid_idx) in enumerate(folds, start=1):
        labels_by_group: dict[str, set[int]] = {}
        for idx in train_idx:
            group = str(contrast_groups[idx])
            labels_by_group.setdefault(group, set()).add(int(labels[idx]))
        invalid = sorted(
            group for group, group_labels in labels_by_group.items() if len(group_labels) < 2
        )
        if invalid:
            groups_str = ", ".join(invalid[:10])
            raise SplitError(
                "pair_aware_filter requires both labels in each training contrast pair "
                f"within every fold; fold_id={fold_id}, offending contrast groups: {groups_str}"
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
    contrast_groups = training_df.select("__contrast_group").to_series().to_list()
    labels = training_df.select("__label").to_series().to_list()

    for idx in row_indices:
        rows.append(
            {
                "species": str(species[idx]),
                "pool": pool,
                "fold_id": str(fold_id),
                "group_id": str(groups[idx]),
                "contrast_group_id": None
                if contrast_groups[idx] is None
                else str(contrast_groups[idx]),
                "label": int(labels[idx]),
            }
        )


def _append_pool_rows(rows: list[dict[str, Any]], pool_df: pl.DataFrame, pool_name: str) -> None:
    species = pool_df.select("__species").to_series().to_list()
    labels = pool_df.select("__label").to_series().to_list()
    contrast_groups = pool_df.select("__contrast_group").to_series().to_list()
    for idx in range(pool_df.height):
        label_value = labels[idx]
        rows.append(
            {
                "species": str(species[idx]),
                "pool": pool_name,
                "fold_id": "NA",
                "group_id": None,
                "contrast_group_id": None
                if contrast_groups[idx] is None
                else str(contrast_groups[idx]),
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
    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
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
            "contrast_group_id": pl.String,
            "label": pl.Int8,
        },
    )
    return manifest.sort(by=["pool", "fold_id", "group_id", "species"], nulls_last=False)


def _build_fold_validation_groups(
    training_df: pl.DataFrame,
    folds: list[tuple[list[int], list[int]]],
) -> pl.DataFrame:
    groups = training_df.select("__group").to_series().to_list()
    labels = training_df.select("__label").to_series().to_list()

    rows: list[dict[str, Any]] = []
    for fold_id, (_train_idx, valid_idx) in enumerate(folds, start=1):
        for idx in valid_idx:
            rows.append(
                {
                    "fold_id": str(fold_id),
                    "group_id": str(groups[idx]),
                    "label": int(labels[idx]),
                }
            )

    if not rows:
        raise SplitError("Fold validation-group manifest is empty")

    manifest = pl.DataFrame(rows).group_by(["fold_id", "group_id"]).agg(
        pl.len().alias("n_validation_species"),
        pl.col("label").sum().alias("n_validation_pos"),
    )
    return (
        manifest.with_columns(
            (
                pl.col("n_validation_species") - pl.col("n_validation_pos")
            ).alias("n_validation_neg"),
            pl.col("fold_id").cast(pl.Int64).alias("__fold_order"),
        )
        .select(
            "fold_id",
            "group_id",
            "n_validation_species",
            "n_validation_pos",
            "n_validation_neg",
            "__fold_order",
        )
        .sort(["__fold_order", "group_id"])
        .drop("__fold_order")
    )


def build_split_artifacts(config: AppConfig) -> SplitArtifacts:
    """Build split manifest and related counts from input config."""
    metadata = _normalize_metadata(config)
    coverage_metadata = metadata.filter(pl.col("__pool") != _POOL_EXCLUDED)
    metadata_species = coverage_metadata.select("__species").to_series().to_list()
    expression_species, excluded_rows = _expression_species_and_excluded_rows(
        config, metadata_species
    )
    _validate_expression_coverage(coverage_metadata, expression_species)

    training_df = metadata.filter(pl.col("__pool") == _POOL_TRAINING_VALIDATION)
    external_df = metadata.filter(pl.col("__pool") == _POOL_EXTERNAL_TEST)
    inference_df = metadata.filter(pl.col("__pool") == _POOL_DISCOVERY_INFERENCE)
    excluded_df = metadata.filter(pl.col("__pool") == _POOL_EXCLUDED)

    _preflight_training_pool(config, training_df)
    folds = _build_fold_indices(config, training_df)
    _validate_fold_labels(training_df, folds)
    _validate_pair_aware_fold_contrasts(config, training_df, folds)
    manifest = _build_split_manifest(training_df, external_df, inference_df, folds)
    fold_validation_groups = _build_fold_validation_groups(training_df, folds)

    pool_counts = {
        _POOL_TRAINING_VALIDATION: training_df.height,
        _POOL_EXTERNAL_TEST: external_df.height,
        _POOL_DISCOVERY_INFERENCE: inference_df.height,
        _POOL_EXCLUDED: excluded_df.height,
    }

    return SplitArtifacts(
        split_manifest=manifest,
        fold_validation_groups=fold_validation_groups,
        pool_counts=pool_counts,
        fold_count=len(folds),
        expression_rows_excluded=excluded_rows,
    )
