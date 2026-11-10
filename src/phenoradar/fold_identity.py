"""Separate human-readable fold IDs from reproducible execution/seed indices."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from phenoradar.config import AppConfig


def validation_fold_ids(
    groups: Sequence[object], validation_rows: Sequence[Sequence[int]],
) -> list[str]:
    """Reuse integer group IDs only for a one-group-to-one-fold partition."""
    sequential = [str(i) for i in range(1, len(validation_rows) + 1)]
    labels = []
    for rows in validation_rows:
        held_out = {str(groups[i]) for i in rows}
        if len(held_out) != 1:
            return sequential
        labels.append(held_out.pop())
    if len(set(labels)) != len(labels) or set(labels) != {str(g) for g in groups}:
        return sequential
    if not all(re.fullmatch(r"0|[1-9][0-9]*", label) and int(label) <= 2**63 - 1
               for label in labels):
        return sequential
    return labels


def configure_fold_seed_ids(config: AppConfig, manifest: pl.DataFrame) -> None:
    """Restore legacy seed indices even when processing a saved split manifest."""
    config._fold_seed_ids = {}
    if "fold_index" not in manifest.columns:
        return
    rows = manifest.filter(pl.col("pool") == "validation").select(
        "fold_id", "fold_index"
    ).unique()
    if rows["fold_id"].n_unique() != rows.height or rows["fold_index"].null_count():
        raise ValueError("Each validation fold must have one non-null fold_index")
    config._fold_seed_ids = {
        str(fold): str(index) for fold, index in rows.iter_rows() if str(fold) != str(index)
    }


def seed_scope(config: AppConfig, scope: str) -> str:
    """Translate IDs only at random-seed construction, leaving logs/outputs intact."""
    for prefix in ("outer_fold_", "fold_"):
        if scope.startswith(prefix):
            fold, separator, remainder = scope[len(prefix):].partition("|")
            return prefix + config._fold_seed_ids.get(fold, fold) + separator + remainder
    return scope
