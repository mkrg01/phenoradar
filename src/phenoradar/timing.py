"""Thread-safe monotonic wall-clock timing records."""

from __future__ import annotations

from threading import Lock
from time import perf_counter

import polars as pl

_TIMING_SCHEMA = {
    "scope": pl.String,
    "stage": pl.String,
    "fold_id": pl.String,
    "sample_set_id": pl.Int64,
    "candidate_index": pl.Int64,
    "started_at_sec": pl.Float64,
    "ended_at_sec": pl.Float64,
    "duration_sec": pl.Float64,
}


class TimingRecorder:
    """Collect nested and concurrent timings relative to one monotonic origin."""

    def __init__(self) -> None:
        self._origin = perf_counter()
        self._rows: list[dict[str, str | int | float | None]] = []
        self._lock = Lock()

    def start(self) -> float:
        """Return a monotonic timestamp for a later ``record_since`` call."""
        return perf_counter()

    @property
    def record_count(self) -> int:
        """Return the current record count safely."""
        with self._lock:
            return len(self._rows)

    def record_since(
        self,
        started_at: float,
        *,
        scope: str,
        stage: str,
        fold_id: str | None = None,
        sample_set_id: int | None = None,
        candidate_index: int | None = None,
    ) -> None:
        """Record a completed interval using the monotonic clock."""
        if not scope.strip():
            raise ValueError("timing scope must be non-empty")
        if not stage.strip():
            raise ValueError("timing stage must be non-empty")
        ended_at = perf_counter()
        duration = max(0.0, ended_at - started_at)
        row: dict[str, str | int | float | None] = {
            "scope": scope,
            "stage": stage,
            "fold_id": fold_id,
            "sample_set_id": sample_set_id,
            "candidate_index": candidate_index,
            "started_at_sec": max(0.0, started_at - self._origin),
            "ended_at_sec": max(0.0, ended_at - self._origin),
            "duration_sec": duration,
        }
        with self._lock:
            self._rows.append(row)

    def to_frame(self, *, start_index: int = 0) -> pl.DataFrame:
        """Return a detached, chronologically sorted timing table."""
        if start_index < 0:
            raise ValueError("start_index must be >= 0")
        with self._lock:
            rows = [dict(row) for row in self._rows[start_index:]]
        if not rows:
            return pl.DataFrame(schema=_TIMING_SCHEMA)
        return pl.DataFrame(rows, schema=_TIMING_SCHEMA).sort(
            [
                "started_at_sec",
                "ended_at_sec",
                "scope",
                "stage",
                "fold_id",
                "sample_set_id",
                "candidate_index",
            ],
            nulls_last=True,
        )


def run_timing_summary(timing: pl.DataFrame) -> dict[str, float]:
    """Return top-level run-stage durations for run metadata."""
    required = {"scope", "stage", "duration_sec"}
    if not required.issubset(timing.columns):
        raise ValueError("timing table is missing run-summary columns")
    run_rows = timing.filter(pl.col("scope") == "run")
    summary: dict[str, float] = {}
    for row in run_rows.select("stage", "duration_sec").iter_rows(named=True):
        summary[str(row["stage"])] = float(row["duration_sec"])
    return summary
