"""Canonical contrast-group names and compatibility with earlier inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

BLOCK_COLUMN = "phylogenetic_block_id"
LEGACY_BLOCK_COLUMN = "expanded_pair_clade_id"


def canonical_column(name: str) -> str:
    if name == LEGACY_BLOCK_COLUMN or name.startswith(LEGACY_BLOCK_COLUMN + "_"):
        return BLOCK_COLUMN + name[len(LEGACY_BLOCK_COLUMN) :]
    return name


def _column_values(value: Any) -> Any:
    if isinstance(value, str):
        return canonical_column(value)
    if isinstance(value, list):
        return [_column_values(item) for item in value]
    return value


def canonical_config_fields(value: Any) -> Any:
    """Resolve aliases before validation or merging configuration layers."""
    if not isinstance(value, Mapping):
        return value
    result = dict(value)
    for old, new in [
        ("contrast_pair_col", "contrast_group_col"),
        ("min_contrast_pairs", "min_contrast_groups"),
    ]:
        if old not in result:
            continue
        legacy = result.pop(old)
        current = result.get(new, legacy)
        if new == "contrast_group_col":
            legacy = _column_values(legacy)
            current = _column_values(current)
        if current != legacy:
            raise ValueError(f"conflicting configuration aliases: {old} and {new}")
        result[new] = current
    return result


def canonical_config_tree(value: Any) -> Any:
    """Normalize keys before inspecting scalar/list condition dimensions."""
    if isinstance(value, Mapping):
        return {
            key: _column_values(canonical_config_tree(item))
            if key in {"contrast_group_col", "group_col"}
            else canonical_config_tree(item)
            for key, item in canonical_config_fields(value).items()
        }
    if isinstance(value, list):
        return [canonical_config_tree(item) for item in value]
    return value


def canonical_metadata(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize legacy block columns, including tree suffixes, without changing IDs."""
    import polars as pl

    for old in frame.columns:
        new = canonical_column(old)
        if old == new:
            continue
        if new in frame.columns:
            legacy = pl.col(old).cast(pl.String, strict=False).str.strip_chars().fill_null("")
            current = pl.col(new).cast(pl.String, strict=False).str.strip_chars().fill_null("")
            if frame.select((legacy != current).any()).item():
                raise ValueError(f"conflicting metadata aliases: {old} and {new}")
            frame = frame.drop(old)
        else:
            frame = frame.rename({old: new})
    return frame
