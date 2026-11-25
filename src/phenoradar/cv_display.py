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
