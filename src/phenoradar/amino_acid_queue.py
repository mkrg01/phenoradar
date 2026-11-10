"""One bounded grid-CV queue, with short-lived refit/audit processes."""
from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import numpy as np
import polars as pl

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_inner import Score, _score, _task
from phenoradar.amino_acid_parallel import _worker_threads, initialize, run_fold, snapshot
from phenoradar.amino_acid_resources import MAX_MATRIX_BUFFER, SparseUsage, worker_bytes
from phenoradar.combined import ExpressionInput
from phenoradar.config import AppConfig
from phenoradar.cv import (
    SelectedCandidate,
    _inner_cv_splits,
    _sample_training_sets,
    _score_std_error_from_trial_rows,
    _select_training_groups,
    _selection_is_active,
    _selection_source_sample_set_id,
)
from phenoradar.model_selection import Candidate, auto_lambda_spec, generate_candidates
from phenoradar.resource_budget import memory_budget

Selection = tuple[list[SelectedCandidate], list[dict[str, Any]], list[str]]


@dataclass
class _Source:
    candidates: list[Candidate]
    species: list[str]
    y: np.ndarray
    groups: np.ndarray
    pairs: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray, str]]
    warnings: list[str]
    results: dict[int, tuple[Score, list[str]]] = field(default_factory=dict)


@dataclass
class _Scope:
    name: str
    rows: pl.DataFrame
    sources: dict[int, _Source]
    directory: Path
    model_base: int


def enabled(config: AppConfig) -> bool:
    return (config.runtime.n_jobs >= 3 and _selection_is_active(config)
            and config.model_selection.search_strategy == "grid")


def _plan(config: AppConfig, rows: pl.DataFrame, scope: str,
          directory: Path, model_base: int) -> _Scope:
    from phenoradar.amino_acid_run import _arrays

    _, y, groups, _ = _arrays(rows)
    subset = _select_training_groups(config, y, groups, scope=scope, fold_id=scope)
    sampled = _sample_training_sets(config, y, groups, scope, [], subset.group_ids)
    sources = {}
    for sample in range(len(sampled)):
        source_id = _selection_source_sample_set_id(config, sample)
        if source_id in sources:
            continue
        local = rows[sampled[source_id].tolist()]
        names, labels, group_ids, pairs = _arrays(local)
        warnings: list[str] = []
        candidates = generate_candidates(config=config, training_scope_id=scope,
                                         source_sample_set_id=source_id, warnings=warnings)
        sources[source_id] = _Source(candidates, names, labels, group_ids, pairs,
                                    list(_inner_cv_splits(config, labels, group_ids)), warnings)
    return _Scope(scope, rows, sources, directory, model_base)


def _selections(config: AppConfig, plan: _Scope) -> dict[int, Selection]:
    from phenoradar.amino_acid_run import _selected_from_scores

    selections = {}
    for source_id, source in plan.sources.items():
        collected: list[list[dict[str, Any]]] = [[] for _ in source.candidates]
        messages = source.warnings.copy()
        for i in range(len(source.splits)):
            scores, warnings = source.results[i]
            messages.extend(warnings)
            for rows, (_, trials) in zip(collected, scores, strict=True):
                rows.extend(trials)
        scored, trials = [], []
        for candidate, rows in zip(source.candidates, collected, strict=True):
            values = [r["metric_value"] for r in rows if not np.isnan(r["metric_value"])]
            mean = float(np.mean(values)) if values else float("nan")
            trials.extend(rows)
            scored.append(SelectedCandidate(candidate, mean, _score_std_error_from_trial_rows(rows),
                                            config.model_selection.selection_rule))
        selections[source_id] = _selected_from_scores(config, scored), trials, messages
    return selections


def queue_workers(config: AppConfig, usage: SparseUsage, n: int,
                  p: int, scopes: int, refit_n: int | None = None) -> tuple[int, int, int]:
    budget = int(memory_budget() * .8)
    # Inner paths retain compressed coefficients, not dense refit/report models.
    inner = worker_bytes(replace(usage, peak_rss_bytes=0), n, p, retained_matrices=0)
    refit = _refit_bytes(usage, n, p, refit_n or n)
    refits = min(2, scopes, max(1, config.runtime.n_jobs // 4))
    while refits and budget - refits * refit < inner:
        refits -= 1
    workers = min(config.runtime.n_jobs - refits, (budget - refits * refit) // inner)
    return max(1, workers), refits, min(MAX_MATRIX_BUFFER, max(1, inner // 3))


def _refit_bytes(usage: SparseUsage, n: int, p: int, refit_n: int) -> int:
    return worker_bytes(replace(usage, matrix_bytes=int(
        usage.matrix_bytes * 1.25 * max(1, refit_n / n)),
                                metadata_bytes=int(usage.metadata_bytes * 1.25), peak_rss_bytes=0),
                        refit_n, int(p * 1.25))


def _refit(config: AppConfig, plan: _Scope, selections: dict[int, Selection],
           manifest: pl.DataFrame) -> Path:
    from phenoradar import amino_acid_parallel
    from phenoradar.amino_acid_outputs import write_site_scores
    from phenoradar.amino_acid_run import fit_scope

    assert amino_acid_parallel._WORKER is not None
    store, expression = amino_acid_parallel._WORKER
    if plan.name != "final_refit":
        return run_fold(config, store, expression, manifest,
                        plan.name.removeprefix("outer_fold_"), plan.directory,
                        plan.model_base, selections=selections)
    plan.directory.mkdir(parents=True)
    tables = plan.directory / "tables"
    tables.mkdir()
    warnings: list[str] = []
    members = fit_scope(config, store, plan.rows, plan.name, tables, warnings, expression,
                        selections=selections)
    write_site_scores(tables, members)
    joblib.dump((members, warnings), plan.directory / "members.joblib")
    store.flush_frequency_caches()
    return plan.directory


def grid_scopes(config: AppConfig, store: AlignmentStore, expression: ExpressionInput | None,
                manifest: pl.DataFrame, folds: list[str], progress: Callable[[str], None] | None,
                ) -> Iterator[tuple[str, Path]]:
    """Preserve fold/candidate ordering while scheduling independent fits globally."""
    if auto_lambda_spec(config) is not None:
        from phenoradar.amino_acid_auto_queue import automatic_scopes

        yield from automatic_scopes(config, store, expression, manifest, folds, progress)
        return
    from phenoradar.amino_acid_parallel import outer_folds
    from phenoradar.amino_acid_run import _prepare_inner_fold

    with TemporaryDirectory(prefix="phenoradar_aa_queue_") as temporary:
        root = Path(temporary)
        plans = [_plan(config, manifest.filter(
            (pl.col("fold_id") == fold) & (pl.col("pool") == "train")
        ).sort("species"), f"outer_fold_{fold}", root / fold, i * 1_000_000)
                 for i, fold in enumerate(folds)]
        if config.runtime.execution_stage == "full_run":
            rows = manifest.filter(pl.col("pool").is_in(["train", "validation"]))
            plans.append(_plan(config, rows.unique(subset="species").sort("species"),
                               "final_refit", root / "final", len(folds) * 1_000_000))
        first = plans[0]
        source_id, source = next(iter(first.sources.items()))
        if progress:
            progress(f"{first.name}: score {len(source.candidates)} candidates; "
                     "prepare queue pilot inner fold.")
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
            # Insufficient memory for concurrent scoring and refitting. Retain
            # the established serial/pilot path instead of exceeding the budget.
            yield from zip(folds, outer_folds(config, store, expression, manifest,
                                              folds, progress), strict=True)
            return
        source.results[0] = (
            _score(config, first.name, source_id, source.candidates, pilot), messages,
        )
        del pilot
        store.flush_frequency_caches()
        child = config.model_copy(deep=True)
        child.runtime.n_jobs = 1
        if progress:
            progress(f"AA grid queue: {workers} scoring workers, "
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
            pending: dict[Future[Any], tuple[int, int | None, int | None]] = {}
            future: Future[Any]
            # Start final selection early too; its labels never enter outer-CV
            # preprocessing or model selection, and outputs still merge last.
            order = ([0, len(plans) - 1, *range(1, len(plans) - 1)]
                     if plans[-1].name == "final_refit" else list(range(len(plans))))
            for index in order:
                plan = plans[index]
                for sid, src in plan.sources.items():
                    for fold_index, split in enumerate(src.splits):
                        if fold_index in src.results:
                            continue
                        future = scoring.submit(_task, child, plan.name, sid, src.candidates,
                                                src.species, src.y, src.groups, src.pairs, split)
                        pending[future] = index, sid, fold_index
            submitted: set[int] = set()
            prepared: dict[int, dict[int, Selection]] = {}
            ready: dict[int, Path] = {}
            next_output = 0
            scoring_finished = False
            try:
                while pending or len(submitted) < len(plans):
                    for index, plan in enumerate(plans):
                        if index not in submitted and index not in prepared and all(
                            len(s.results) == len(s.splits) for s in plan.sources.values()
                        ):
                            prepared[index] = _selections(config, plan)
                    running = sum(sid is None for _, sid, _ in pending.values())
                    if not scoring_finished and not any(sid is not None
                                                        for _, sid, _ in pending.values()):
                        scoring.shutdown(wait=True)
                        scoring_finished = True
                        refits = min(config.runtime.n_jobs, max(refits, running + int(
                            memory_budget() * .8) // refit_bytes))
                        if progress:
                            progress(f"AA grid queue: scoring complete; "
                                     f"up to {refits} refit workers "
                                     "after releasing scoring memory.")
                    for index in list(prepared)[:max(0, refits - running)]:
                        plan = plans[index]
                        # Pass selection records, never matrices or fitted inner-fold state.
                        task = replace(plan, sources={})
                        future = fitting.submit(_refit, child, task,
                                                prepared.pop(index), manifest)
                        pending[future] = index, None, None
                        submitted.add(index)
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, source_key, split_index = pending.pop(future)
                        plan = plans[index]
                        if source_key is None:
                            ready[index] = future.result()
                            if progress:
                                progress(f"{plan.name}: queued refit completed.")
                        else:
                            assert split_index is not None
                            plan.sources[source_key].results[split_index] = future.result()
                            if progress:
                                label = plan.sources[source_key].splits[split_index][2]
                                progress(f"{plan.name}: inner fold "
                                         f"{label} scored.")
                    while next_output in ready:
                        plan = plans[next_output]
                        yield plan.name.removeprefix("outer_fold_"), ready.pop(next_output)
                        next_output += 1
            finally:
                for future in pending:
                    future.cancel()
