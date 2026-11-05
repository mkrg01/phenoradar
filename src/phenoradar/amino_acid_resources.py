"""Internal sparse-matrix budgets; no dataset or model-selection decisions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phenoradar.resource_budget import memory_budget

if TYPE_CHECKING:
    from phenoradar.amino_acid import AATransform, AlignmentStore

MAX_MATRIX_BUFFER = 8 * 1024**3


@dataclass
class SparseUsage:
    matrix_bytes: int = 0
    metadata_bytes: int = 0
    peak_rss_bytes: int = 0


def process_peak_rss() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def matrix_buffer_limit(store: AlignmentStore) -> int:
    budget: int | None = getattr(store, "_matrix_buffer_bytes", None)
    return min(MAX_MATRIX_BUFFER, max(0, memory_budget() // 4 if budget is None else budget))


def record_sparse_usage(store: AlignmentStore, fitted: AATransform,
                        matrices: list[Any]) -> None:
    usage: SparseUsage = getattr(store, "_sparse_usage", SparseUsage())
    usage.matrix_bytes = max(usage.matrix_bytes, sum(
        array.nbytes for matrix in matrices
        for array in [matrix.data, matrix.indices, matrix.indptr]
    ))
    # Means, known/active states and the columnar site audit (including axes).
    usage.metadata_bytes = max(usage.metadata_bytes, len(fitted.sites) * 264)
    usage.peak_rss_bytes = max(usage.peak_rss_bytes, process_peak_rss())
    store._sparse_usage = usage


def worker_bytes(usage: SparseUsage, n_species: int, n_features: int,
                 retained_matrices: int = 1) -> int:
    matrix = usage.matrix_bytes or n_species * n_features * 8
    # Buffered chunks and their concatenation, one-based sparse indices, native
    # workspaces, coefficient vectors, site metadata and headroom. The observed
    # pilot peak also covers larger lambda grids and ensembles.
    estimated = (matrix * (retained_matrices + 1) + matrix // 3
                 + n_features * 192 + 2 * usage.metadata_bytes + 1024**3)
    return max(1, int(1.25 * max(estimated, usage.peak_rss_bytes)))
