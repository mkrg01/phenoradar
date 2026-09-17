"""Observed binary traits used only for post-prediction phylogenetic interpretation."""

from pathlib import Path

import polars as pl

TRAIT_SCHEMA = {"species": pl.String, "label": pl.Int8}


def read_observed_traits(
    metadata_path: str | None,
    *,
    species_col: str,
    trait_col: str,
    exclude_col: str | None = None,
) -> pl.DataFrame:
    """Read known traits without requiring expression or contrast-pair membership."""
    if metadata_path is None:
        return pl.DataFrame(schema=TRAIT_SCHEMA)
    frame = pl.read_csv(Path(metadata_path), separator="\t", infer_schema=False)
    if species_col not in frame.columns:
        raise ValueError(f"Metadata is missing species column: {species_col}")
    if trait_col not in frame.columns:
        return pl.DataFrame(schema=TRAIT_SCHEMA)
    if exclude_col is not None:
        if exclude_col not in frame.columns:
            raise ValueError(f"Metadata is missing exclusion column: {exclude_col}")
        frame = frame.filter(
            ~pl.col(exclude_col)
            .str.strip_chars()
            .str.to_lowercase()
            .is_in(["1", "true", "yes"])
            .fill_null(False)
        )
    frame = frame.select(
        pl.col(species_col).str.strip_chars().alias("species"),
        pl.col(trait_col).str.strip_chars().alias("label"),
    )
    return validate_observed_traits(frame)


def validate_observed_traits(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep observed states, preserving literal species IDs and rejecting conflicts."""
    if not set(TRAIT_SCHEMA).issubset(frame.columns):
        raise ValueError("Observed trait reference requires species and label columns")
    frame = frame.select(
        pl.col("species").cast(pl.String).str.strip_chars(),
        pl.col("label").cast(pl.String).str.strip_chars(),
    )
    if frame.filter(pl.col("species").is_null() | (pl.col("species") == "")).height:
        raise ValueError("Observed trait reference contains empty species IDs")
    invalid = frame.filter(pl.col("label").is_not_null() & ~pl.col("label").is_in(["", "0", "1"]))
    if invalid.height:
        raise ValueError("Observed traits must be 0, 1, or empty/null")
    frame = frame.filter(pl.col("label").is_in(["0", "1"]))
    conflicts = (
        frame.group_by("species").agg(pl.col("label").n_unique()).filter(pl.col("label") > 1)
    )
    if conflicts.height:
        raise ValueError(
            "Conflicting observed traits for: " + ", ".join(conflicts["species"].to_list()[:10])
        )
    return frame.unique("species").with_columns(pl.col("label").cast(pl.Int8)).sort("species")


def merge_observed_traits(*frames: pl.DataFrame | None) -> pl.DataFrame:
    present = [validate_observed_traits(frame) for frame in frames if frame is not None]
    if not present:
        return pl.DataFrame(schema=TRAIT_SCHEMA)
    return validate_observed_traits(pl.concat(present))
