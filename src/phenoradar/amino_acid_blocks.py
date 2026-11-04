"""Bounded columnar views of fitted sites and their stable feature identifiers."""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import polars as pl

from phenoradar.amino_acid import ALPHABET, AASite, AASiteTable


def site_blocks(sites: Sequence[AASite]) -> Iterator[dict[str, Any]]:
    fields = ["column", "sample_fraction", "minor_samples", "minor_groups", "score",
              "mean", "known", "active", "post_aggregation_sample_fraction"]
    for start in range(0, len(sites), 4096):
        stop = start + 4096
        if isinstance(sites, AASiteTable):
            yield {"orthogroup": sites.orthogroups[start:stop],
                   **{name: getattr(sites, name)[start:stop] for name in fields}}
        else:
            block = sites[start:stop]
            yield {"orthogroup": np.array([s.orthogroup for s in block], dtype=object),
                   **{name: np.array([getattr(s, name) for s in block]) for name in fields}}


def feature_axes(block: dict[str, Any], mask: np.ndarray, *, names: bool = True) -> pl.DataFrame:
    rows, states = np.nonzero(mask)
    ogs = block["orthogroup"][rows]
    frame = pl.DataFrame({
        "orthogroup": ogs, "alignment_pos_1based": block["column"][rows] + 1,
        "residue": np.array(list(ALPHABET))[states],
    })
    if names:
        # JSON escaping is performed once per OG, including non-ASCII names.
        prefixes = {og: "[" + json.dumps(str(og)) + "," for og in set(ogs)}
        frame = frame.with_columns(pl.Series("_prefix", [prefixes[og] for og in ogs]))
        frame = frame.with_columns(pl.concat_str([
            "_prefix", pl.col("alignment_pos_1based").cast(pl.String), pl.lit(',"'),
            "residue", pl.lit('"]'),
        ]).alias("feature")).drop("_prefix")
    return frame


def coefficient_blocks(
    sites: Sequence[AASite], coefficients: np.ndarray, offset: int = 0,
) -> Iterator[tuple[dict[str, Any], np.ndarray, int]]:
    for block in site_blocks(sites):
        active = block["active"]
        count = int(active.sum())
        beta = np.zeros(active.shape)
        beta[active] = coefficients[offset:offset + count]
        yield block, beta, offset
        offset += count
