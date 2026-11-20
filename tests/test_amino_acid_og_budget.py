import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from test_amino_acid import alignment, settings, tpm_table

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_queue import preprocessing_threads
from phenoradar.amino_acid_sparse import fit_sparse_encoded
from phenoradar.config import AppConfig


@pytest.mark.parametrize("slots,scoring,refits,expected", [
    (24, 2, 2, 4), (19, 2, 2, 4), (4, 2, 1, 1),
    (3, 1, 1, 2), (12, 4, 2, 2), (18, 12, 2, 1),
])
def test_preprocessing_budget_keeps_all_active_pools_within_cpu_slots(
    slots, scoring, refits, expected,
):
    config = AppConfig()
    config.runtime.n_jobs = slots
    threads = preprocessing_threads(config, scoring, refits)
    assert threads == expected
    assert scoring * threads + refits <= slots


@pytest.mark.parametrize("score_jobs", [1, 3])
@pytest.mark.parametrize("preparation_jobs", [None, 1, 2, 4])
def test_task_shares_its_cpu_budget_between_preparation_and_model_fit(
    monkeypatch, score_jobs, preparation_jobs,
):
    from phenoradar import amino_acid_inner as inner
    from phenoradar import amino_acid_parallel as parallel
    from phenoradar import amino_acid_run as run

    config = AppConfig()
    config.runtime.n_jobs = score_jobs
    prepared = SimpleNamespace(identity="unchanged")
    observed = []

    def prepare(current, *args):
        observed.append(("prepare", current.runtime.n_jobs))
        args[-1].append("kept")
        return prepared

    def score(current, scope, sample, candidates, fold):
        observed.append(("score", current.runtime.n_jobs))
        assert fold is prepared
        assert current.model_dump(exclude={"runtime"}) == config.model_dump(exclude={"runtime"})
        return []

    monkeypatch.setattr(parallel, "_WORKER", (SimpleNamespace(), None))
    monkeypatch.setattr(run, "_prepare_inner_fold", prepare)
    monkeypatch.setattr(inner, "_score", score)
    result, warnings = inner._task(config, "outer_fold_1", 0, [], ["a"], np.array([0]),
                                   np.array(["g"]), np.array(["p"]),
                                   (np.array([0]), np.array([], dtype=int), "1"),
                                   preparation_jobs)
    assert observed == [("prepare", preparation_jobs or score_jobs),
                        ("score", preparation_jobs or score_jobs)]
    assert config.runtime.n_jobs == score_jobs
    assert result == [] and warnings == ["kept"]


def matrix_digest(matrix):
    digest = hashlib.sha256()
    for block in matrix.blocks:
        for array in (block.indptr, *block._buffers()):
            digest.update(memoryview(np.ascontiguousarray(array).reshape(-1)).cast("B"))
    return digest.hexdigest()


def matrix_arithmetic(matrix):
    weights = np.arange(matrix.shape[1], dtype=float)
    residual = np.arange(matrix.shape[0], dtype=float)
    gradient = np.empty(matrix.shape[1])
    matrix.gradient(residual, gradient)
    csc = matrix.to_csc()
    return gradient.tobytes(), (csc @ weights).tobytes(), (csc.T @ residual).tobytes()


def test_packed_og_threads_preserve_buffers_statistics_and_matrix_arithmetic(tmp_path):
    records = [("a_1", "AC-DVAX"), ("a_2", "ADDDVAX"), ("b_1", "ADCEVAX"),
               ("c_1", "ADCDAAX"), ("d_1", "ACCEAAX"), ("held_1", "W-XWWVX")]
    path = alignment(tmp_path / "aa", records)
    (path / "OG2.faa").write_text((path / "OG1.faa").read_text())
    (path / "OG_gap.faa").write_text("".join(f">{s}\n-------\n" for s, _ in records))
    tpm = tpm_table(tmp_path / "tpm.tsv", [
        (name.split("_")[0], name, i + 1) for i, (name, _) in enumerate(records)
    ])
    training = ["d", "b", "a", "c", "a"]
    targets = [training, ["held", "absent", "c"]]
    kwargs = dict(y=np.array([0, 1, 0, 1, 0]),
                  groups=np.array(["1", "1", "2", "2", "2"]), audit=False)
    reference = None
    with AlignmentStore(path, ["a", "b", "c", "d", "held", "absent"],
                        cache_frequencies=True, sequence_tpm_path=str(tpm)) as store:
        for threads in [1, 2, 4]:
            fitted, matrices, shift = fit_sparse_encoded(
                store, training, settings("tpm_weighted_frequency"), targets,
                packed=True, n_jobs=threads, **kwargs,
            )
            identity = (
                fitted.features, fitted.qc,
                [getattr(fitted.sites, name).tobytes() for name in
                 ["column", "mean", "known", "active", "sample_fraction"]],
                shift.tobytes(), [matrix_digest(m) for m in matrices],
                [matrix_arithmetic(m) for m in matrices],
            )
            if reference is None:
                reference = identity
            assert identity == reference


def test_cli_worker_limit_caps_shared_cpu_budget(monkeypatch):
    from phenoradar import amino_acid_queue as queue
    from phenoradar.amino_acid_parallel import fold_workers
    from phenoradar.amino_acid_resources import SparseUsage

    config = AppConfig()
    config.runtime.n_jobs = 12
    config._worker_limit = 5
    monkeypatch.setattr(queue, 'memory_budget', lambda: 100 * 1024**3)
    monkeypatch.setattr(queue, 'worker_bytes', lambda *args, **kwargs: 1024**3)
    monkeypatch.setattr(queue, '_refit_bytes', lambda *args, **kwargs: 1024**3)
    scoring, refits, _ = queue.queue_workers(config, SparseUsage(), 1000, 100000, 19)
    assert scoring + refits <= 5
    assert scoring * preprocessing_threads(config, scoring, refits) + refits <= 5
    assert fold_workers(config, 10, 10, 19) <= 5


def test_preparation_honors_cli_worker_reservation(monkeypatch):
    from phenoradar import amino_acid_run as run

    config = AppConfig()
    config.data.feature_mode = 'amino_acid'
    config.runtime.n_jobs = 12
    config._worker_limit = 5
    observed = []
    def fit(*args, **kwargs):
        observed.append(kwargs['n_jobs'])
        raise RuntimeError('captured budget')
    monkeypatch.setattr(run, 'fit_sparse_encoded', fit)
    with pytest.raises(RuntimeError, match='captured budget'):
        run._prepare_inner_fold(config, SimpleNamespace(), None, ['a', 'b', 'c', 'd'],
                                np.array([0, 1, 0, 1]), np.array(['g', 'g', 'h', 'h']),
                                np.array(['g', 'g', 'h', 'h']),
                                (np.array([0, 1]), np.array([2, 3]), '5'), [])
    assert observed == [5] and config.runtime.n_jobs == 12
