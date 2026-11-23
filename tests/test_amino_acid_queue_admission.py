"""Admission preserves both refits and falls back without reliable packed bounds."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from phenoradar.amino_acid_packed import PackedAAMatrix
from phenoradar.amino_acid_queue import _packed_refit_path, queue_workers
from phenoradar.amino_acid_resources import SparseUsage
from phenoradar.config import AppConfig
from phenoradar.model_selection import Candidate

GIB = 1024**3


@pytest.fixture
def measured(monkeypatch):
    import phenoradar.amino_acid_queue as queue

    monkeypatch.setattr(queue, "memory_budget", lambda: 384 * GIB)
    config = AppConfig()
    config.data.feature_mode = "amino_acid"
    config.runtime.amino_acid_matrix = "packed"
    config.runtime.n_jobs = 24
    config.sampling.sampled_set_count = 1
    config.model_selection.selected_candidate_count = 1
    usage = SparseUsage(18_060_696_328, 595_635_480, 28_685_914_112)
    return config, usage


def test_measured_packed_pilot_admits_three_scores_without_reducing_refits(measured):
    config, usage = measured
    baseline = queue_workers(config, usage, 665, 16_584_806, 20, 1038)
    actual = queue_workers(config, usage, 665, 16_584_806, 20, 1038, True)
    assert baseline[:2] == (2, 2)
    assert actual == (3, 2, baseline[2])


@pytest.mark.parametrize("case", [
    "unmeasured", "large_peak", "more_refit_rows", "ensemble", "sampled_ensemble",
])
def test_packed_admission_falls_back_when_headroom_is_not_proven(measured, case):
    config, usage = measured
    refit_rows = 1038
    if case == "unmeasured":
        usage.peak_rss_bytes = 0
    elif case == "large_peak":
        usage.peak_rss_bytes = 100 * GIB
    elif case == "more_refit_rows":
        refit_rows = 1500
    elif case == "ensemble":
        config.model_selection.selected_candidate_count = 2
    else:
        config.sampling.sampled_set_count = 2
    assert queue_workers(config, usage, 665, 16_584_806, 20, refit_rows, True) == (
        queue_workers(config, usage, 665, 16_584_806, 20, refit_rows)
    )


@pytest.mark.parametrize("jobs,budget", [(4, 384), (24, 256), (24, 1)])
def test_packed_admission_preserves_slot_memory_and_serial_fallbacks(measured, monkeypatch,
                                                                 jobs, budget):
    import phenoradar.amino_acid_queue as queue

    config, usage = measured
    config.runtime.n_jobs = jobs
    monkeypatch.setattr(queue, "memory_budget", lambda: budget * GIB)
    assert queue_workers(config, usage, 665, 16_584_806, 20, 1038, True) == (
        queue_workers(config, usage, 665, 16_584_806, 20, 1038)
    )


def test_small_packed_inputs_keep_existing_higher_concurrency(measured):
    config, _ = measured
    usage = SparseUsage(64 * 1024**2, 1024**2, GIB)
    actual = queue_workers(config, usage, 100, 1000, 20, 150, True)
    assert actual[0] > 3
    assert actual == queue_workers(config, usage, 100, 1000, 20, 150)


@pytest.mark.parametrize("alpha,strength", [(1, 1e-4), (0, 1e-4), (1, 0)])
def test_only_positive_penalty_packed_working_set_paths_qualify(monkeypatch, alpha, strength):
    import phenoradar.glmnet as glmnet

    packed = PackedAAMatrix(2, [], fused=False)
    source = SimpleNamespace(candidates=[Candidate(0, {"alpha": alpha, "lambda": strength})])
    plans = [SimpleNamespace(sources={0: source})]
    assert not _packed_refit_path(packed, plans)  # Small matrices expand to CSC.
    monkeypatch.setattr(glmnet, "_PACKED_WORKING_SET_CSC_BYTES", 0)
    assert _packed_refit_path(packed, plans) == (alpha > 0 and strength > 0)
    assert not _packed_refit_path(sparse.csc_matrix(np.eye(2)), plans)
