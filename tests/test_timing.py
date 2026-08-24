from __future__ import annotations

import pytest

import phenoradar.timing as timing_mod
from phenoradar.timing import TimingRecorder, run_timing_summary


def test_timing_recorder_uses_monotonic_offsets_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter([100.0, 101.0, 104.0])
    monkeypatch.setattr(timing_mod, "perf_counter", lambda: next(values))
    recorder = TimingRecorder()

    started_at = recorder.start()
    recorder.record_since(
        started_at,
        scope="outer_fold",
        stage="candidate_score",
        fold_id="2",
        sample_set_id=3,
        candidate_index=4,
    )

    timing = recorder.to_frame()
    assert timing.columns == [
        "scope",
        "stage",
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "started_at_sec",
        "ended_at_sec",
        "duration_sec",
    ]
    assert timing.row(0, named=True) == {
        "scope": "outer_fold",
        "stage": "candidate_score",
        "fold_id": "2",
        "sample_set_id": 3,
        "candidate_index": 4,
        "started_at_sec": 1.0,
        "ended_at_sec": 4.0,
        "duration_sec": 3.0,
    }


def test_timing_recorder_can_return_records_since_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter([10.0, 11.0, 12.0, 13.0, 15.0])
    monkeypatch.setattr(timing_mod, "perf_counter", lambda: next(values))
    recorder = TimingRecorder()
    first = recorder.start()
    recorder.record_since(first, scope="run", stage="split_construction")
    start_index = recorder.record_count
    second = recorder.start()
    recorder.record_since(second, scope="run", stage="outer_cv")

    timing = recorder.to_frame(start_index=start_index)
    assert timing.height == 1
    assert timing.get_column("stage").to_list() == ["outer_cv"]


def test_run_timing_summary_selects_top_level_rows() -> None:
    recorder = TimingRecorder()
    started = recorder.start()
    recorder.record_since(started, scope="outer_fold", stage="total", fold_id="1")
    started = recorder.start()
    recorder.record_since(started, scope="run", stage="outer_cv")

    summary = run_timing_summary(recorder.to_frame())

    assert set(summary) == {"outer_cv"}
    assert summary["outer_cv"] >= 0.0
