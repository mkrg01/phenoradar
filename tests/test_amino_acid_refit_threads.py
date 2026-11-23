"""Refits use otherwise idle CPU slots without changing model/Polars concurrency."""
import pytest

from phenoradar.amino_acid_queue import preprocessing_threads, refit_preprocessing_threads
from phenoradar.config import AppConfig


@pytest.mark.parametrize('jobs', range(3, 33))
def test_all_refit_admissions_respect_live_cpu_reservations(jobs):
    config = AppConfig()
    config.runtime.n_jobs = jobs
    for scoring in range(1, jobs):
        refits = min(2, jobs - scoring)
        threads = preprocessing_threads(config, scoring, refits)
        running = []
        for _ in range(refits):
            added = refit_preprocessing_threads(config, scoring, threads, refits, running)
            if not added:
                break
            assert 1 <= added <= 4
            running.append(added)
            assert scoring * threads + sum(running) <= jobs


def test_phase_change_accounts_for_existing_multithreaded_refits():
    config = AppConfig()
    config.runtime.n_jobs = 24
    assert refit_preprocessing_threads(config, 3, 4, 2, []) == 4
    assert refit_preprocessing_threads(config, 3, 4, 2, [4]) == 4
    running = [4, 4]
    while True:
        added = refit_preprocessing_threads(config, 0, 4, 24, running)
        if not added:
            break
        running.append(added)
        assert sum(running) <= 24
    assert sum(running) == 24
    assert len(running) < 24  # Earlier four-thread tasks still occupy their slots.
