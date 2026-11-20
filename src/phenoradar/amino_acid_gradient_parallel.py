"""Bound packed gradient threads; each column retains its original reduction order."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import pairwise
from threading import Lock
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np

if TYPE_CHECKING:
    from phenoradar.amino_acid_packed import PackedAAMatrix


class _Scanner:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.executor: ThreadPoolExecutor | None = None
        self.lock = Lock()
        self.groups: WeakKeyDictionary[PackedAAMatrix, list[tuple[int, int]]] = WeakKeyDictionary()

    def scan(self, matrix: PackedAAMatrix, residual: np.ndarray, out: np.ndarray) -> None:
        with self.lock:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.workers, thread_name_prefix="phenoradar-aa-gradient"
                )
            groups = self.groups.get(matrix)
            if groups is None:
                weights = np.array(
                    [max(1, int(b.indptr[-1])) for b in matrix.blocks], dtype=np.int64
                )
                cumulative = np.r_[0, np.cumsum(weights)]
                targets = np.linspace(0, cumulative[-1], min(len(weights), self.workers * 4) + 1)
                bounds = np.unique(
                    np.r_[0, np.searchsorted(cumulative, targets[1:-1]), len(weights)]
                )
                groups = [(int(a), int(b)) for a, b in pairwise(bounds) if b > a]
                self.groups[matrix] = groups
            executor = self.executor

        def scan_group(bounds: tuple[int, int]) -> None:
            for i in range(*bounds):
                left, right = matrix.boundaries[i : i + 2]
                matrix.blocks[i].gradient(residual, out[left:right], fused=matrix.fused)

        list(executor.map(scan_group, groups))

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


_SCANNER: ContextVar[_Scanner | None] = ContextVar("aa_gradient_scanner", default=None)


@contextmanager
def gradient_thread_limit(n_jobs: int) -> Iterator[None]:
    if n_jobs < 1:
        raise ValueError("Packed gradient threads must be positive")
    scanner = _Scanner(min(4, n_jobs)) if n_jobs > 1 else None
    token = _SCANNER.set(scanner)
    try:
        yield
    finally:
        try:
            if scanner is not None:
                scanner.close()
        finally:
            _SCANNER.reset(token)


def scan_blocks(matrix: PackedAAMatrix, residual: np.ndarray, out: np.ndarray) -> bool:
    scanner = _SCANNER.get()
    if scanner is None or len(matrix.blocks) < 2 or matrix.shape[1] < 65_536:
        return False
    scanner.scan(matrix, residual, out)
    return True
