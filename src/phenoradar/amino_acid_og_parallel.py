"""Independent OG preprocessing with bounded buffers and ordered assembly."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from typing import Any

import numpy as np

from phenoradar.amino_acid import (
    AASiteTable,
    AATransform,
    AlignmentError,
    AlignmentOG,
    AlignmentStore,
    fit_amino_acid,
)
from phenoradar.amino_acid_encoding import FoldEncoding
from phenoradar.config.schema import AminoAcidConfig
from phenoradar.resource_budget import memory_budget

OG_BUFFER_BYTES = 64 * 1024**2


def fit_orthogroups(
    store: AlignmentStore, species: list[str], settings: AminoAcidConfig, *,
    n_jobs: int, encoding: FoldEncoding | None, **kwargs: Any,
) -> AATransform:
    # Each task owns its numeric arrays and one OG's frequency-cache entry.
    # Completed encoded OGs spill to disk rather than growing the worker queue.
    workers = min(n_jobs, len(store.ogs), 4,
                  max(1, memory_budget() // (4 * OG_BUFFER_BYTES)))
    if workers == 1:
        return fit_amino_acid(store, species, settings, _encoding=encoding, **kwargs)
    buffer = min(OG_BUFFER_BYTES, max(1, memory_budget() // (workers * 8)))

    def fit(og: AlignmentOG) -> tuple[AATransform, np.ndarray | None]:
        local = copy(store)
        local.ogs = {og.orthogroup: og}
        encoder = FoldEncoding([encoding.species]) if encoding is not None else None
        if encoder is not None:
            encoder.memory_limit = buffer
        try:
            fitted = fit_amino_acid(local, species, settings, _encoding=encoder,
                                    _allow_empty=True, **kwargs)
            matrix = encoder.matrices()[0] if encoder is not None else None
            return fitted, matrix
        finally:
            if encoder is not None:
                encoder.close()

    blocks = []
    qc = []
    representatives: dict[str, np.ndarray] | None = (
        {} if settings.aggregation == "best_covered" else None
    )
    orthogroups = iter(store.ogs.values())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Future[tuple[AATransform, np.ndarray | None]]] = deque()

        def submit() -> bool:
            og = next(orthogroups, None)
            if og is None:
                return False
            if store.cache_frequencies:
                # Allocate unique cache paths on the caller thread, before any write.
                store._frequency_cache(og, settings.aggregation == "tpm_weighted_frequency")
            pending.append(executor.submit(fit, og))
            return True

        for _ in range(workers):
            if not submit():
                break
        while pending:
            fitted, matrix = pending.popleft().result()
            qc.extend(fitted.qc)
            if isinstance(fitted.sites, AASiteTable):
                sites = fitted.sites
                blocks.append({"orthogroup": str(sites.orthogroups[0]), **{
                    name: getattr(sites, name) for name in [
                        "column", "sample_fraction", "minor_samples", "minor_groups", "score",
                        "mean", "known", "active", "post_aggregation_sample_fraction",
                    ]
                }})
            if representatives is not None:
                representatives.update(fitted.representative_columns or {})
            if encoding is not None and matrix is not None:
                for start in range(0, matrix.shape[1], 4096):
                    encoding.append(matrix[:, start:start + 4096])
            del fitted, matrix
            submit()
    if not blocks:
        raise AlignmentError(
            "AA preprocessing removed all sites; inspect training coverage/filters"
        )
    sites = AASiteTable(blocks)
    if not np.any(sites.active):
        raise AlignmentError("AA aggregation produced no variable features in this training fold")
    return AATransform(settings.model_copy(deep=True), sites, qc, representatives)
