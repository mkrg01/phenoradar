"""Scope figure formats and reuse workers across groups of rendering jobs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from multiprocessing import get_context

from phenoradar.config.schema import FigureFormat
from phenoradar.resource_budget import memory_budget

figure_formats: ContextVar[tuple[FigureFormat, ...] | None] = ContextVar(
    "figure_formats", default=None,
)


@contextmanager
def use_figure_formats(formats: Sequence[FigureFormat] | None) -> Iterator[None]:
    """Apply a format selection within one run or a spawned rendering job."""
    token = figure_formats.set(None if formats is None else tuple(formats))
    try:
        yield
    finally:
        figure_formats.reset(token)


# Include headroom over the measured memory footprint of a PDF process.
_CANDIDATE_PDF_WORKER_BYTES = 1536 * 1024**2


def candidate_pdf_worker_count(parallel_workers: int, pdf_count: int) -> int:
    """Limit candidate PDF rendering by CPUs, jobs, and available memory."""
    limit = min(max(1, int(parallel_workers)), pdf_count)
    if limit <= 1:
        return limit
    memory_workers = memory_budget() * 4 // (5 * _CANDIDATE_PDF_WORKER_BYTES)
    return max(1, min(limit, memory_workers))


class FigureWorkers:
    """Lazily start one bounded pool and release it at the end of figure generation."""

    def __init__(self, max_workers: int) -> None:
        self._max_workers = max(1, int(max_workers))
        self._executor: ProcessPoolExecutor | None = None

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers, mp_context=get_context("spawn")
            )
        return self._executor

    def __enter__(self) -> FigureWorkers:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the pool so another rendering stage can use the slots."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
