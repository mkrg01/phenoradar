from pathlib import Path

import numpy as np
import pytest

from phenoradar.amino_acid_encoding import _compact_mass
from phenoradar.amino_acid_resources import SparseUsage, worker_bytes


@pytest.mark.parametrize("n", [1, 2, 7, 64, 116])
def test_compact_mass_preserves_original_twenty_slot_reduction(n: int) -> None:
    rng = np.random.default_rng(773)
    raw = np.zeros((n, 109, 20), dtype=np.float32)
    for j in range(109):
        states = rng.choice(20, 1 + j % 20, replace=False)
        raw[:, j, states] = rng.dirichlet(np.full(len(states), .1), size=n)
    raw[rng.random(raw.shape[:2]) < .1] = 0
    raw[:, 0, :2] = [np.nextafter(np.float32(1), np.float32(0)), 0]
    raw[:, 1, :2] = [np.float32(.25), np.float32(.75 - 1e-7)]
    raw[:, 2, :2] = [np.float32(.25), np.float32(.75 + 1e-7)]
    known = rng.random((109, 20)) > .15
    raw *= known
    sites, aa = np.nonzero(np.any(raw != 0, axis=0))
    expected = raw.sum(axis=2, dtype=np.float64)
    expected = np.where(np.isclose(expected, 1, rtol=0, atol=1e-7), 1, expected)
    actual = _compact_mass(raw[:, sites, aa], sites, aa, raw.shape[1])
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("values", [[-1, 2], [1, 1], [-0., -0.], [.25, .75]])
def test_compact_mass_falls_back_for_unordered_or_repeated_axes(values) -> None:
    q = np.array([values], dtype=np.float32)
    sites, aa = np.array([0, 0]), np.array([2, 1])
    dense = np.zeros((1, 1, 20), dtype=np.float32)
    dense[:, sites, aa] = q
    expected = dense.sum(axis=2, dtype=np.float64)
    expected = np.where(np.isclose(expected, 1, rtol=0, atol=1e-7), 1, expected)
    np.testing.assert_array_equal(_compact_mass(q, sites, aa, 1), expected)


def test_sparse_worker_estimate_uses_csc_bytes_and_respects_pilot_peak(monkeypatch) -> None:
    import phenoradar.amino_acid_parallel as parallel
    from phenoradar.config import AppConfig

    gib = 1024**3
    config = AppConfig()
    config.runtime.n_jobs = 19
    usage = SparseUsage(matrix_bytes=int(2.4*gib), metadata_bytes=gib//2,
                        peak_rss_bytes=8*gib)
    monkeypatch.setattr(parallel, "memory_budget", lambda: 140*gib)
    workers = parallel.fold_workers(config, 116, 8_500_000, 18, usage)
    assert 7 <= workers <= 10
    larger_peak = SparseUsage(usage.matrix_bytes, usage.metadata_bytes, 30*gib)
    assert parallel.fold_workers(config, 116, 8_500_000, 18, larger_peak) < workers
    assert worker_bytes(usage, 116, 8_500_000, 5) > worker_bytes(usage, 116, 8_500_000)
    monkeypatch.setattr(parallel, "memory_budget", lambda: gib)
    assert parallel.fold_workers(config, 116, 8_500_000, 18, usage) == 1


@pytest.mark.parametrize("budget", [0, 1, 10**8])
def test_default_sparse_fold_buffer_is_bounded_and_preserves_design(
    tmp_path: Path, monkeypatch, budget: int,
) -> None:
    from test_amino_acid import alignment, settings

    from phenoradar.amino_acid import AlignmentStore
    from phenoradar.amino_acid_sparse import fit_sparse_encoded

    path = alignment(tmp_path, [("a_x", "AVAA"), ("b_x", "VAVA"), ("held_x", "WAVV")])
    with AlignmentStore(path, ["a", "b", "held"], cache_frequencies=True) as store:
        store._matrix_buffer_bytes = budget
        fitted, matrices, shift = fit_sparse_encoded(
            store, ["a", "b"], settings(), [["a", "b"], ["held"]],
            y=np.array([0, 1]), groups=np.array(["g0", "g1"]),
        )
        for samples, matrix in zip([["a", "b"], ["held"]], matrices, strict=True):
            np.testing.assert_array_equal(matrix.toarray() - shift,
                                          fitted.matrix(store, samples))
        assert store._sparse_usage.matrix_bytes > 0


def _spawned_polars_threads() -> int:
    import polars as pl
    return pl.thread_pool_size()


@pytest.mark.parametrize("previous", [None, "19"])
def test_spawned_polars_pool_uses_worker_cpu_budget_and_restores_parent(
    monkeypatch, previous,
) -> None:
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    from phenoradar.amino_acid_parallel import _worker_threads

    if previous is None:
        monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    else:
        monkeypatch.setenv("POLARS_MAX_THREADS", previous)
    with _worker_threads(2), ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        assert executor.submit(_spawned_polars_threads).result() == 2
    assert os.environ.get("POLARS_MAX_THREADS") == previous
