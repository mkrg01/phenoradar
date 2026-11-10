from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy import sparse
from test_amino_acid import fixture_config

from phenoradar.amino_acid_resources import SparseUsage
from phenoradar.amino_acid_run import run_amino_acid
from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.glmnet import _sparse_has_variation


@pytest.mark.parametrize("width", [1, 64, 65, 1000])
@pytest.mark.parametrize("case", ["zero", "constant", "explicit_zero", "early", "late"])
def test_sparse_variation_matches_full_scan(width, case):
    values = np.zeros((8, width))
    if case == "constant":
        values[:, ::3] = 2
    elif case in {"early", "late"}:
        values[3, 0 if case == "early" else -1] = .3
    matrix = sparse.csc_matrix(values)
    if case == "explicit_zero":
        matrix = sparse.csc_matrix((np.zeros(width), np.zeros(width, dtype=np.int32),
                                    np.arange(width + 1, dtype=np.int32)), shape=values.shape)
    expected = bool(np.any(matrix.max(axis=0).toarray() != matrix.min(axis=0).toarray()))
    assert _sparse_has_variation(matrix) == expected


def test_grid_queue_bounds_both_pools_and_falls_back_when_memory_is_small(monkeypatch):
    import phenoradar.amino_acid_queue as queue

    config = AppConfig()
    config.runtime.n_jobs = 19
    monkeypatch.setattr(queue, "memory_budget", lambda: 100 * 1024**3)
    scoring, fitting, buffer = queue.queue_workers(config, SparseUsage(2 * 1024**3),
                                                  116, 8_000_000, 20)
    assert scoring > 0 and fitting == 2 and scoring + fitting <= 19
    assert 0 < buffer <= 8 * 1024**3
    monkeypatch.setattr(queue, "memory_budget", lambda: 1024)
    assert queue.queue_workers(config, SparseUsage(), 116, 8_000_000, 20)[1] == 0
    assert not queue.enabled(config)
    config.model_selection.selected_candidate_count = 1
    assert queue.enabled(config)  # Automatic paths now synchronize complete rounds.
    config.model_selection.search_space = {"lambda": [.1, .01]}
    assert queue.enabled(config)
    config.model_selection.search_strategy = "tpe"
    assert not queue.enabled(config)


@pytest.mark.parametrize("policy", ["per_sample_set", "reuse_first_sample_set"])
def test_queued_sample_ensembles_match_serial_all_public_tables(
    tmp_path: Path, monkeypatch, policy,
):
    monkeypatch.chdir(tmp_path)
    config = load_and_resolve_config([fixture_config(tmp_path, "aa_frequency")])
    config.sampling.strategy = "group_balanced"
    config.sampling.max_samples_per_label_per_group = 1
    config.sampling.sampled_set_count = 2
    config.model_selection.selected_candidate_count = 2
    config.model_selection.candidate_source_policy = policy
    config.preprocess.amino_acid.ranked_site_filter.method = "none"
    config.figures.top_features = 3
    serial = run_amino_acid(config, tmp_path / "serial")
    config.runtime.n_jobs = 4
    events = []
    queued = run_amino_acid(config, tmp_path / "queued", progress=events.append)
    assert any("AA grid queue:" in event for event in events)
    for path in serial.rglob("*.tsv"):
        relative = path.relative_to(serial)
        # Provenance paths and timing are intentionally not model outputs.
        if path.name in {"alignment_inputs.tsv", "sequence_correspondence.tsv"}:
            continue
        actual = queued / relative
        assert actual.exists(), relative
        left = pl.read_csv(path, separator="\t", infer_schema_length=None)
        right = pl.read_csv(actual, separator="\t", schema_overrides=left.schema,
                            infer_schema_length=None)
        assert_frame_equal(left.sort(left.columns), right.sort(right.columns),
                           check_exact=False, abs_tol=1e-14, rel_tol=1e-14)
    assert sorted(p.relative_to(serial) for p in serial.rglob("*.svg")) == sorted(
        p.relative_to(queued) for p in queued.rglob("*.svg")
    )


@pytest.mark.parametrize("fail", [False, True])
def test_queue_refits_before_later_scores_finish_and_keeps_output_order(
    tmp_path: Path, monkeypatch, fail: bool,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    import phenoradar.amino_acid_queue as queue
    import phenoradar.amino_acid_run as run
    from phenoradar.split import build_split_artifacts

    config = load_and_resolve_config([fixture_config(tmp_path, "aa_frequency")])
    config.runtime.n_jobs = 4
    manifest = build_split_artifacts(config).split_manifest
    event = Event()

    class Executor(ThreadPoolExecutor):
        def __init__(self, max_workers, **kwargs):
            super().__init__(max_workers=max_workers)

    def task(config, scope, *args):
        if scope == "outer_fold_2":
            assert event.wait(5), "Refit must overlap later independent scoring tasks"
            if fail:
                raise RuntimeError("test score failure")
        return [(1.0, [])], []

    def refit(config, plan, *args):
        if plan.name == "outer_fold_1":
            event.set()
        plan.directory.mkdir()
        return plan.directory

    store = SimpleNamespace(_sparse_usage=SparseUsage(), flush_frequency_caches=lambda: None)
    monkeypatch.setattr(run, "_prepare_inner_fold", lambda *args: SimpleNamespace(
        y_train=np.array([0, 1]), x_train=np.zeros((2, 2))))
    monkeypatch.setattr(queue, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(queue, "queue_workers", lambda *args: (2, 1, 1024))
    monkeypatch.setattr(queue, "snapshot", lambda *args: {})
    monkeypatch.setattr(queue, "_task", task)
    monkeypatch.setattr(queue, "_score", lambda *args: [(1.0, [])])
    monkeypatch.setattr(queue, "_selections", lambda *args: {})
    monkeypatch.setattr(queue, "_refit", refit)
    iterator = queue.grid_scopes(config, store, None, manifest, ["1", "2", "3", "4"], None)
    if fail:
        with pytest.raises(RuntimeError, match="test score failure"):
            list(iterator)
    else:
        results = list(iterator)
        assert [scope for scope, _ in results] == ["1", "2", "3", "4", "final_refit"]
    assert event.is_set()
