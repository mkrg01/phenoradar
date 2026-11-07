from types import SimpleNamespace

import numpy as np

from phenoradar.amino_acid_inner import inner_workers
from phenoradar.amino_acid_resources import SparseUsage
from phenoradar.config import AppConfig
from phenoradar.cv import InnerCvPreprocessedFold


def test_inner_workers_use_child_memory_and_prevent_nested_processes(monkeypatch):
    import phenoradar.amino_acid_inner as inner

    config = AppConfig()
    config.runtime.n_jobs = 19
    x = np.zeros((10, 30))
    first = InnerCvPreprocessedFold("1", x, x, np.zeros(10), np.zeros(10), np.ones(10))
    # Earlier reporting HWM in the parent is not a fresh child's working set.
    store = SimpleNamespace(_sparse_usage=SparseUsage(1024, 1024, 100 * 1024**3))
    monkeypatch.setattr(inner, "memory_budget", lambda: 12 * 1024**3)
    assert inner_workers(config, store, first, 4) == 4
    store._inner_parallel_disabled = True
    assert inner_workers(config, store, first, 4) == 0
    store._inner_parallel_disabled = False
    monkeypatch.setattr(inner, "memory_budget", lambda: 1024)
    assert inner_workers(config, store, first, 4) == 0
