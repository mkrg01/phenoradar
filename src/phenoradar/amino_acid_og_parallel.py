"""Independent OG preprocessing with bounded buffers and ordered assembly."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
from phenoradar.amino_acid_encoding import EncodedBlock, FoldEncoding
from phenoradar.config.schema import AminoAcidConfig
from phenoradar.resource_budget import memory_budget

OG_BUFFER_BYTES = 64 * 1024**2


def _result_bytes(fitted: AATransform, block: EncodedBlock | None) -> int:
    size = sum(value.nbytes for value in vars(fitted.sites).values()
               if isinstance(value, np.ndarray)) if isinstance(fitted.sites, AASiteTable) else 0
    if block is not None:
        for matrix in block.matrices:
            if hasattr(matrix, "nbytes"):
                size += matrix.nbytes
            else:
                size += sum(getattr(matrix, name).nbytes for name in ["data", "indices", "indptr"])
        if block.shift is not None:
            size += block.shift.nbytes
    return int(size)


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

    def fit(og: AlignmentOG) -> tuple[AATransform, EncodedBlock | None]:
        # Keep cache restoration behind the training-only coverage gates in
        # fit_amino_acid. Each OG still owns its shared dictionary entry and
        # private append path; retained OGs restore concurrently in these tasks.
        local = copy(store)
        local.ogs = {og.orthogroup: og}
        encoder = encoding.fork() if encoding is not None else None
        if encoder is not None:
            encoder.memory_limit = buffer
        try:
            fitted = fit_amino_acid(local, species, settings, _encoding=encoder,
                                    _allow_empty=True, **kwargs)
            block = encoder.block() if encoder is not None else None
            return fitted, block
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
        pending: dict[Future[tuple[AATransform, EncodedBlock | None]], int] = {}
        ready: dict[int, tuple[AATransform, EncodedBlock | None, int]] = {}
        submitted = assembled = buffered = 0
        exhausted = False
        # Refill on completion, while assembling in the original OG order. Bound
        # both lookahead and completed bytes; already-running OGs may finish
        # above the byte limit, but no further work is then dispatched.
        while pending or ready or not exhausted:
            while (not exhausted and len(pending) < workers
                   and len(pending) + len(ready) < 2 * workers
                   and buffered < workers * buffer):
                og = next(orthogroups, None)
                if og is None:
                    exhausted = True
                    break
                pending[executor.submit(fit, og)] = submitted
                submitted += 1
            if assembled not in ready:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    fitted, block = future.result()
                    size = _result_bytes(fitted, block)
                    ready[index] = fitted, block, size
                    buffered += size
                if assembled not in ready:
                    continue
            fitted, block, size = ready.pop(assembled)
            buffered -= size
            assembled += 1
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
            if encoding is not None and block is not None:
                encoding.extend(block)
            del fitted, block
    if not blocks:
        raise AlignmentError(
            "AA preprocessing removed all sites; inspect training coverage/filters"
        )
    sites = AASiteTable(blocks)
    if not np.any(sites.active):
        raise AlignmentError("AA aggregation produced no variable features in this training fold")
    return AATransform(settings.model_copy(deep=True), sites, qc, representatives)
