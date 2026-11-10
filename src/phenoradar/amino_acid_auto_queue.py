"""Bound automatic-path fold jobs, with a complete-round barrier per source."""

from __future__ import annotations

import multiprocessing
import shutil
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import polars as pl

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_parallel import _worker_threads, initialize, snapshot
from phenoradar.amino_acid_queue import (
    Selection,
    _plan,
    _refit,
    _refit_bytes,
    _Source,
    queue_workers,
)
from phenoradar.amino_acid_resources import MAX_MATRIX_BUFFER
from phenoradar.auto_lambda import AutoPathState, _FoldCache, fold_maxima, score_auto_fold
from phenoradar.combined import ExpressionInput
from phenoradar.config import AppConfig
from phenoradar.cv import (
    SelectedCandidate,
    _glmnet_fit_diagnostic,
    _score_std_error_from_trial_rows,
    _selection_metric_from_probability,
)
from phenoradar.glmnet import GlmnetError, fit_glmnet_path
from phenoradar.model_selection import Candidate, auto_lambda_spec
from phenoradar.resource_budget import memory_budget


@dataclass(frozen=True)
class _Task:
    scope: int
    source: int
    fold: int


@dataclass
class _Result:
    rows: dict[int, list[dict[str, Any]]]
    warnings: list[str]
    error: Exception | None = None


def _score(config: AppConfig, pending: list[Candidate], complete: list[Candidate],
           fold: Any, maxima: tuple[float, float], round_id: int,
           ) -> dict[int, list[dict[str, Any]]]:
    return score_auto_fold(
        config, pending, complete, fold, maxima, round_id, fit_path=fit_glmnet_path,
        metric=lambda y, p: _selection_metric_from_probability(config, y, p),
        diagnostic=_glmnet_fit_diagnostic,
    )


def _initial(config: AppConfig, pending: list[Candidate], fold: Any,
             directory: Path, warnings: list[str]) -> _Result:
    maxima = fold_maxima(config, fold)
    spec = auto_lambda_spec(config)
    assert spec is not None
    if spec.extend_lower and spec.max_extensions:
        directory.mkdir(parents=True)
        cache = _FoldCache(directory)
        cache.add(fold, maxima)
        # Only labels, shapes and the fold class cross jobs; CSC buffers stay on disk.
        joblib.dump(cache.items, directory / "fold.joblib")
    return _Result(_score(config, pending, [], fold, maxima, 0), warnings)


def _task(config: AppConfig, source: _Source, split_index: int, directory: Path,
          pending: list[Candidate], complete: list[Candidate], round_id: int) -> _Result:
    from phenoradar import amino_acid_parallel
    from phenoradar.amino_acid_run import _prepare_inner_fold

    assert amino_acid_parallel._WORKER is not None
    store, expression = amino_acid_parallel._WORKER
    if round_id == 0:
        warnings: list[str] = []
        fold = _prepare_inner_fold(config, store, expression, source.species, source.y,
                                   source.groups, source.pairs, source.splits[split_index],
                                   warnings)
        return _initial(config, pending, fold, directory, warnings)
    try:
        cache = _FoldCache(directory)
        cache.items = joblib.load(directory / "fold.joblib")
        fold, maxima = next(iter(cache))
        return _Result(_score(config, pending, complete, fold, maxima, round_id), [])
    except (GlmnetError, OSError, MemoryError) as error:
        return _Result({}, [], error)


def _selection(config: AppConfig, state: AutoPathState, source: _Source,
               fold_messages: dict[int, list[str]]) -> Selection:
    from phenoradar.amino_acid_run import _selected_from_scores

    result = state.result()
    trials: list[dict[str, Any]] = []
    scored = []
    for candidate, (mean, detail) in zip(result.candidates, result.scores, strict=True):
        trials.extend(detail)
        scored.append(SelectedCandidate(candidate, mean, _score_std_error_from_trial_rows(detail),
                                        config.model_selection.selection_rule))
    warnings = source.warnings.copy()
    for index in range(len(source.splits)):
        warnings.extend(fold_messages[index])
    warnings.extend(state.warnings)
    return _selected_from_scores(config, scored), trials, warnings


def automatic_scopes(config: AppConfig, store: AlignmentStore, expression: ExpressionInput | None,
                     manifest: pl.DataFrame, folds: list[str],
                     progress: Callable[[str], None] | None) -> Iterator[tuple[str, Path]]:
    """Schedule fold work globally; extensions observe all folds in fixed order."""
    from phenoradar.amino_acid_parallel import outer_folds
    from phenoradar.amino_acid_run import _prepare_inner_fold

    with TemporaryDirectory(prefix="phenoradar_aa_auto_queue_") as temporary:
        root = Path(temporary)
        plans = [_plan(config, manifest.filter(
            (pl.col("fold_id") == fold) & (pl.col("pool") == "train")
        ).sort("species"), f"outer_fold_{fold}", root / fold, i * 1_000_000)
                 for i, fold in enumerate(folds)]
        if config.runtime.execution_stage == "full_run":
            rows = manifest.filter(pl.col("pool").is_in(["train", "validation"]))
            plans.append(_plan(config, rows.unique(subset="species").sort("species"),
                               "final_refit", root / "final", len(folds) * 1_000_000))
        source_id, source = next(iter(plans[0].sources.items()))
        if progress:
            progress(f"{plans[0].name}: prepare automatic queue pilot inner fold.")
        messages: list[str] = []
        pilot = _prepare_inner_fold(config, store, expression, source.species, source.y,
                                    source.groups, source.pairs, source.splits[0], messages)
        refit_n = max(len(s.species) for p in plans for s in p.sources.values())
        workers, refits, buffer = queue_workers(config, store._sparse_usage,
                                                len(pilot.y_train), pilot.x_train.shape[1],
                                                len(plans), refit_n)
        refit_bytes = _refit_bytes(store._sparse_usage, len(pilot.y_train),
                                   pilot.x_train.shape[1], refit_n)
        if not refits:
            del pilot
            yield from zip(folds, outer_folds(config, store, expression, manifest,
                                              folds, progress), strict=True)
            return
        states = {(index, sid): AutoPathState(config, src.candidates)
                  for index, plan in enumerate(plans) for sid, src in plan.sources.items()}
        results: dict[tuple[int, int], dict[int, _Result]] = {key: {} for key in states}
        fold_messages: dict[tuple[int, int], dict[int, list[str]]] = {key: {} for key in states}
        selections: dict[int, dict[int, Selection]] = {index: {} for index in range(len(plans))}
        tasks: deque[_Task] = deque()
        order = ([0, len(plans) - 1, *range(1, len(plans) - 1)]
                 if plans[-1].name == "final_refit" else list(range(len(plans))))
        for index in order:
            for sid, src in plans[index].sources.items():
                tasks.extend(_Task(index, sid, i) for i in range(len(src.splits))
                             if (index, sid, i) != (0, source_id, 0))

        def directory(task: _Task) -> Path:
            return root / "designs" / str(task.scope) / str(task.source) / str(task.fold)

        def receive(task: _Task, result: _Result) -> None:
            key = task.scope, task.source
            state = states[key]
            src = plans[task.scope].sources[task.source]
            results[key][task.fold] = result
            if state.round_id == 0:
                fold_messages[key][task.fold] = result.warnings
            if len(results[key]) != len(src.splits):
                return
            ordered = [results[key][i] for i in range(len(src.splits))]
            failure = next((r.error for r in ordered if r.error is not None), None)
            if failure is not None:
                state.fail_extension(failure)
            else:
                state.commit_round(r.rows for r in ordered)
            results[key].clear()
            if not state.finished:
                # Prioritize extensions so completed source designs are released
                # promptly, rather than retaining every scope's CSC files.
                for i in reversed(range(len(src.splits))):
                    tasks.appendleft(_Task(task.scope, task.source, i))
                if progress:
                    progress(f"{plans[task.scope].name}: automatic lambda round {state.round_id}, "
                             f"{len(state.pending)} new candidates after all inner folds.")
                return
            selections[task.scope][task.source] = _selection(
                config, state, src, fold_messages[key])
            cached = directory(task).parent
            if cached.exists():
                shutil.rmtree(cached)

        receive(_Task(0, source_id, 0), _initial(config, source.candidates, pilot,
                                               directory(_Task(0, source_id, 0)), messages))
        del pilot
        store.flush_frequency_caches()
        child = config.model_copy(deep=True)
        child.runtime.n_jobs = 1
        if progress:
            progress(f"AA automatic queue: {workers} scoring workers, "
                     f"{refits} short-lived refit workers; "
                     f"matrix buffer {buffer / 1024**3:.2f} GiB/scoring worker.")
        payload = snapshot(store, expression)
        with _worker_threads(1), ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=(payload, 1, buffer),
        ) as scoring, ProcessPoolExecutor(
            max_workers=config.runtime.n_jobs, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=(payload, 1, min(MAX_MATRIX_BUFFER, buffer * 2)),
            max_tasks_per_child=1,
        ) as fitting:
            scoring_jobs: dict[Future[_Result], _Task] = {}
            fitting_jobs: dict[Future[Path], int] = {}
            submitted: set[int] = set()
            ready: dict[int, Path] = {}
            next_output = 0
            scoring_finished = False
            future: Future[Any]
            try:
                while next_output < len(plans):
                    while tasks and len(scoring_jobs) < workers:
                        task = tasks.popleft()
                        src = plans[task.scope].sources[task.source]
                        state = states[task.scope, task.source]
                        future = scoring.submit(_task, child, src, task.fold, directory(task),
                                                state.pending, state.complete, state.round_id)
                        scoring_jobs[future] = task
                    if not scoring_finished and not tasks and not scoring_jobs:
                        scoring.shutdown(wait=True)
                        scoring_finished = True
                        refits = min(config.runtime.n_jobs, max(refits,
                                     int(memory_budget() * .8) // refit_bytes))
                        if progress:
                            progress(f"AA automatic queue: scoring complete; up to {refits} "
                                     "refit workers after releasing scoring memory.")
                    for index, plan in enumerate(plans):
                        if len(fitting_jobs) >= refits:
                            break
                        if index not in submitted and len(selections[index]) == len(plan.sources):
                            fitting_jobs[fitting.submit(_refit, child, replace(plan, sources={}),
                                                        selections[index], manifest)] = index
                            submitted.add(index)
                    pending: list[Future[Any]] = [*scoring_jobs, *fitting_jobs]
                    if not pending:
                        raise RuntimeError("Automatic lambda queue made no progress")
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        if future in scoring_jobs:
                            task = scoring_jobs.pop(future)
                            receive(task, future.result())
                        else:
                            index = fitting_jobs.pop(future)
                            ready[index] = future.result()
                            if progress:
                                progress(f"{plans[index].name}: automatic queued refit completed.")
                    while next_output in ready:
                        plan = plans[next_output]
                        yield plan.name.removeprefix("outer_fold_"), ready.pop(next_output)
                        next_output += 1
            finally:
                outstanding: list[Future[Any]] = [*scoring_jobs, *fitting_jobs]
                for future in outstanding:
                    future.cancel()
