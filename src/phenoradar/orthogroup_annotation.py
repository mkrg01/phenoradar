"""Orthogroup annotation table loading helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl


class OrthogroupAnnotationError(ValueError):
    """Raised when orthogroup annotation input cannot be loaded."""


ORTHOGROUP_ANNOTATION_COLUMNS = [
    "feature",
    "orthogroup_annotation_taxid",
    "orthogroup_annotation",
]


def load_orthogroup_annotations(path: Path | None) -> pl.DataFrame | None:
    """Load an optional headerless OrthoDB orthogroup annotation TSV."""
    if path is None:
        return None
    try:
        annotations = pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=ORTHOGROUP_ANNOTATION_COLUMNS,
        )
    except FileNotFoundError as exc:
        raise OrthogroupAnnotationError(f"Orthogroup annotation file not found: {path}") from exc
    except Exception as exc:
        raise OrthogroupAnnotationError(
            f"Failed to read orthogroup annotation TSV: {path}"
        ) from exc

    required = set(ORTHOGROUP_ANNOTATION_COLUMNS)
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise OrthogroupAnnotationError(
            "Missing required columns in orthogroup annotation TSV: "
            + ", ".join(missing)
        )

    return (
        annotations.select(
            [
                pl.col("feature")
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("feature"),
                pl.col("orthogroup_annotation_taxid")
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("orthogroup_annotation_taxid"),
                pl.col("orthogroup_annotation")
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("orthogroup_annotation"),
            ]
        )
        .filter(pl.col("feature").is_not_null() & (pl.col("feature") != ""))
        .group_by("feature")
        .agg(
            [
                pl.col("orthogroup_annotation_taxid").drop_nulls().first(),
                pl.col("orthogroup_annotation").drop_nulls().first(),
            ]
        )
        .sort("feature")
    )
