"""Reuse figure workers across sequential groups of rendering jobs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context


class FigureWorkers:
    """Lazily start one bounded pool and release it at the end of figure generation."""

    def __init__(self, max_workers: int) -> None:
        self._max_workers = max(1, int(max_workers))
        self._executor: ProcessPoolExecutor | None = None

    def executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers, mp_context=get_context("spawn")
            )
        return self._executor

    def __enter__(self) -> FigureWorkers:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
