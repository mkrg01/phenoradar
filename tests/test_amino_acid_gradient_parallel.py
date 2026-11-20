import gc
import threading
import weakref

import numpy as np
import pytest
from scipy import sparse

from phenoradar import amino_acid_gradient_parallel as parallel
from phenoradar.amino_acid_packed import PackedAAMatrix, PackedColumns, packed_arithmetic
from phenoradar.cv import _with_native_thread_limit


def matrix():
    rng = np.random.default_rng(6)
    blocks = [
        PackedColumns.from_csc(
            sparse.csc_matrix(rng.choice([0.0, 0.13, -0.21, 0.5], size=(17, 8192)))
        )
        for _ in range(8)
    ]
    return PackedAAMatrix(17, blocks, fused=packed_arithmetic())


@pytest.mark.parametrize("threads", [1, 2, 4, 12])
def test_native_budget_preserves_every_gradient_bit_and_closes_threads(threads):
    x = matrix()
    rng = np.random.default_rng(7)
    residuals = [rng.normal(size=17), np.zeros(17), rng.normal(size=17) * 1e-100]
    expected = []
    for residual in residuals:
        out = np.empty(x.shape[1])
        x.gradient(residual, out)
        expected.append(out.tobytes())

    def scan():
        observed = []
        for residual in residuals:
            out = np.empty(x.shape[1])
            x.gradient(residual, out)
            observed.append(out.tobytes())
        scanner = parallel._SCANNER.get()
        assert (scanner is None) == (threads == 1)
        if scanner is not None:
            assert scanner.workers == min(threads, 4) and scanner.executor is not None
        return observed

    assert _with_native_thread_limit(threads, scan) == expected
    assert parallel._SCANNER.get() is None
    assert not any(t.name.startswith("phenoradar-aa-gradient") for t in threading.enumerate())


def test_context_restores_budget_after_worker_failure_and_nested_serial(monkeypatch):
    x = matrix()
    with parallel.gradient_thread_limit(4):
        outer = parallel._SCANNER.get()
        with parallel.gradient_thread_limit(1):
            assert parallel._SCANNER.get() is None
        assert parallel._SCANNER.get() is outer

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate worker failure")

    monkeypatch.setattr(x.blocks[3], "gradient", fail)
    with pytest.raises(RuntimeError, match="deliberate"), parallel.gradient_thread_limit(4):
        x.gradient(np.ones(17), np.empty(x.shape[1]))
    assert parallel._SCANNER.get() is None
    assert not any(t.name.startswith("phenoradar-aa-gradient") for t in threading.enumerate())


def test_parallel_pool_does_not_retain_consumed_matrices():
    with parallel.gradient_thread_limit(4):
        x = matrix()
        observed = weakref.ref(x)
        x.gradient(np.ones(17), np.empty(x.shape[1]))
        del x
        gc.collect()
        assert observed() is None
        assert not parallel._SCANNER.get().groups


def test_small_matrices_never_start_thread_pool(monkeypatch):
    x = PackedAAMatrix(
        2, [PackedColumns.from_csc(sparse.csc_matrix([[0.0, 1.0], [1.0, 0.0]]))], fused=False
    )

    def fail(*args, **kwargs):
        raise RuntimeError("unnecessary thread pool")

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", fail)
    with parallel.gradient_thread_limit(4):
        x.gradient(np.ones(2), np.empty(2))
        assert parallel._SCANNER.get().executor is None


def test_native_thread_budget_preserves_working_set_models_and_predictions(monkeypatch):
    from phenoradar import glmnet

    x = matrix()
    y = np.tile([0.0, 1.0], 9)[:17]
    monkeypatch.setattr(glmnet, "SPARSE_INDEX_LIMIT", 30000)
    expected = None
    for threads in [1, 2, 4]:
        models = _with_native_thread_limit(
            threads,
            glmnet.fit_glmnet_path,
            x,
            y,
            [0.1, 0.05],
            alpha=0.7,
            compute_kkt=False,
            compact_coefficients=True,
        )
        assert all(m.used_working_set_ for m in models)
        observed = [
            (
                m.coefficient_indices_.tobytes(),
                m.coefficient_values_.tobytes(),
                m.intercept_.hex(),
                m.predict_proba(x).tobytes(),
            )
            for m in models
        ]
        if expected is None:
            expected = observed
        assert observed == expected
