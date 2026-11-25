"""Group and fold identities shared by prediction figures."""

from pathlib import Path

import polars as pl

from phenoradar.contrast_metadata import canonical_metadata


def with_cv_groups(
    predictions: pl.DataFrame, *, metadata_path: Path, species_col: str, group_col: str,
) -> pl.DataFrame:
    """Attach the actual split groups without changing prediction or fold values."""
    metadata = canonical_metadata(pl.read_csv(metadata_path, separator="\t", infer_schema=False))
    groups = metadata.select(
        pl.col(species_col).cast(pl.String).str.strip_chars().alias("species"),
        pl.col(group_col).cast(pl.String).str.strip_chars().alias("group_id"),
    ).unique()
    return predictions.drop("group_id", strict=False).join(groups, on="species", how="left",
                                                         validate="m:1")


def with_prediction_context(
    predictions: pl.DataFrame, *, metadata_path: Path | None, species_col: str,
    group_col: str, trait_col: str,
) -> pl.DataFrame:
    """Attach optional observed metadata; prediction targets have no CV-fold identity."""
    result = predictions.drop("fold_id", strict=False)
    metadata = (canonical_metadata(pl.read_csv(metadata_path, separator="\t", infer_schema=False))
                if metadata_path is not None and metadata_path.is_file() else None)
    columns = {"group_id": (group_col, pl.String), "true_label": (trait_col, pl.Int8)}
    for name, (source, dtype) in columns.items():
        if name in result.columns:
            result = result.with_columns(pl.col(name).cast(dtype, strict=False))
            continue
        if metadata is not None and source in metadata.columns:
            values = metadata.select(
                pl.col(species_col).cast(pl.String).str.strip_chars().alias("species"),
                pl.col(source).cast(dtype, strict=False).alias(name),
            ).unique()
            result = result.join(values, on="species", how="left", validate="m:1")
        else:
            result = result.with_columns(pl.lit(None, dtype=dtype).alias(name))
    return result


def fold_group_labels(frame: pl.DataFrame) -> dict[str, str] | None:
    """Return fold-to-group labels only for a complete one-to-one correspondence."""
    if not {"group_id", "fold_id"}.issubset(frame.columns):
        return None
    pairs = frame.select(
        pl.col(name).cast(pl.String).str.strip_chars() for name in ["fold_id", "group_id"]
    )
    if pairs.is_empty() or pairs.select(
        pl.any_horizontal(pl.all().is_null() | (pl.all() == "")).any()
    ).item():
        return None
    pairs = pairs.unique()
    if pairs.height != pairs["fold_id"].n_unique() or pairs.height != pairs["group_id"].n_unique():
        return None
    return dict(pairs.iter_rows())


def tree_context_tracks(frame: pl.DataFrame) -> list[str]:
    """Keep both identities when groups and folds represent different partitions."""
    tracks = [name for name in ["group_id", "fold_id"] if name in frame.columns
              and frame[name].drop_nulls().cast(pl.String).str.strip_chars().ne("").any()]
    if "group_id" in tracks and "fold_id" in tracks and fold_group_labels(frame) is not None:
        tracks.remove("fold_id")
    return tracks
