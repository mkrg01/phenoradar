from __future__ import annotations

import os
from pathlib import Path

import pytest

from phenoradar.figure_runtime import FigureWorkers
from phenoradar.figures import _run_figure_jobs
from phenoradar.timing import TimingRecorder


def _write_pid(path: Path) -> None:
    path.write_text(str(os.getpid()))


def test_shared_figure_workers_reuse_process_and_close(tmp_path: Path) -> None:
    recorder = TimingRecorder()
    # One actual worker makes process reuse observable without depending on
    # how a larger executor schedules short jobs across its idle processes.
    with FigureWorkers(1) as workers:
        executor = workers.executor()
        for group in ["first", "second"]:
            jobs = [
                (f"{group}_{i}", _write_pid, (tmp_path / f"{group}_{i}",), {}, False)
                for i in range(2)
            ]
            assert not _run_figure_jobs(
                jobs, parallel_workers=2, workers=workers, timing_recorder=recorder
            )
    pids = {path.read_text() for path in tmp_path.iterdir()}
    assert len(pids) == 1 and str(os.getpid()) not in pids
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(os.getpid)
    frame = recorder.to_frame()
    assert frame.height == 4
    assert set(frame["stage"]) == {
        f"{group}_{i}" for group in ["first", "second"] for i in range(2)
    }
    assert (frame["duration_sec"] >= 0).all()
    assert (frame["ended_at_sec"] >= frame["started_at_sec"]).all()


def test_unused_figure_workers_do_not_start_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Empty figure groups must not start a process pool")

    monkeypatch.setattr("phenoradar.figure_runtime.ProcessPoolExecutor", fail)
    with FigureWorkers(4) as workers:
        assert _run_figure_jobs([], parallel_workers=4, workers=workers) == []
