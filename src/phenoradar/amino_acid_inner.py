"""Bound independent grid-CV folds; only small score records cross processes."""
from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_parallel import _worker_threads, initialize, snapshot
from phenoradar.amino_acid_resources import MAX_MATRIX_BUFFER, SparseUsage, worker_bytes
from phenoradar.combined import ExpressionInput
from phenoradar.config import AppConfig
from phenoradar.cv import InnerCvPreprocessedFold, _score_glmnet_paths
from phenoradar.model_selection import Candidate
from phenoradar.resource_budget import memory_budget

Score = list[tuple[float, list[dict[str, Any]]]]


def inner_workers(config: AppConfig, store: AlignmentStore, first: InnerCvPreprocessedFold,
                  remaining: int) -> int:
    if getattr(store, "_inner_parallel_disabled", False) or config.runtime.n_jobs < 3:
        return 0
    usage = getattr(store, "_sparse_usage", SparseUsage())
    # A fresh child does not inherit the parent's earlier reporting arenas/HWM.
    per_worker = worker_bytes(replace(usage, peak_rss_bytes=0),
                              len(first.y_train), first.x_train.shape[1])
    return min(remaining, config.runtime.n_jobs - 1, int(memory_budget() * .6) // per_worker)


def _score(config: AppConfig, scope: str, sample_set: int, candidates: list[Candidate],
           fold: InnerCvPreprocessedFold) -> Score:
    return _score_glmnet_paths(
        config=config, training_scope_id=scope, source_sample_set_id=sample_set,
        candidates=candidates, preprocessed_folds=[fold], progress_callback=None,
        timing_recorder=None, consume_training_matrix=True, max_path_workers=1,
    )


def _task(config: AppConfig, scope: str, sample_set: int, candidates: list[Candidate],
          species: list[str], y: np.ndarray, groups: np.ndarray, pairs: np.ndarray,
          split: tuple[np.ndarray, np.ndarray, str]) -> tuple[Score, list[str]]:
    from phenoradar import amino_acid_parallel
    from phenoradar.amino_acid_run import _prepare_inner_fold

    assert amino_acid_parallel._WORKER is not None
    store, expression = amino_acid_parallel._WORKER
    warnings: list[str] = []
    fold = _prepare_inner_fold(config, store, expression, species, y, groups, pairs,
                               split, warnings)
    return _score(config, scope, sample_set, candidates, fold), warnings


def score_parallel(config: AppConfig, store: AlignmentStore, expression: ExpressionInput | None,
                   scope: str, sample_set: int, candidates: list[Candidate],
                   species: list[str], y: np.ndarray, groups: np.ndarray, pairs: np.ndarray,
                   first: InnerCvPreprocessedFold,
                   splits: list[tuple[np.ndarray, np.ndarray, str]], workers: int,
                   warnings: list[str], progress: Callable[[str], None] | None,
                   ) -> Iterator[Score]:
    child = config.model_copy(deep=True)
    child.runtime.n_jobs = max(1, (config.runtime.n_jobs - 1) // workers)
    buffer = min(MAX_MATRIX_BUFFER, max(1, int(memory_budget() * .6) // (3 * workers)))
    if progress:
        progress(f"{scope}: {workers} inner-fold workers; first fold scores concurrently; "
                 f"matrix buffer {buffer / 1024**3:.2f} GiB/worker.")
    with _worker_threads(child.runtime.n_jobs), ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(snapshot(store, expression), child.runtime.n_jobs, buffer),
    ) as executor:
        futures = [executor.submit(_task, child, scope, sample_set, candidates,
                                   species, y, groups, pairs, split) for split in splits]
        try:
            yield _score(config, scope, sample_set, candidates, first)
            del first
            for split, future in zip(splits, futures, strict=True):
                result, messages = future.result()
                warnings.extend(messages)
                if progress:
                    progress(f"{scope}: inner fold {split[2]} scored.")
                yield result
        finally:
            for future in futures:
                future.cancel()
