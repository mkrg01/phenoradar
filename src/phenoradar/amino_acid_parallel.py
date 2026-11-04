"""Memory-bounded outer folds with private caches and deterministic table merging."""
from __future__ import annotations

import multiprocessing
import os
import shutil
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import polars as pl

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_reporting import AAReportWriter
from phenoradar.combined import ExpressionInput
from phenoradar.config import AppConfig
from phenoradar.model_selection import active_search_space
from phenoradar.resource_budget import memory_budget


def fold_workers(config: AppConfig, n_species: int, n_features: int, n_folds: int) -> int:
    # One streamed training/validation pair, native workspaces and refit/metadata,
    # plus headroom for fold-specific schema variation and shared file cache.
    one_path = all(isinstance(value, list) and len(value) == 1
                   for name, value in active_search_space(config).items() if name != "lambda")
    factor = 2.5 if config.data.feature_mode == "amino_acid" and one_path else 3.5
    if config.model_selection.search_strategy == "tpe":
        factor = 8.0
    per_worker = max(1, int(n_species * n_features * 8 * factor) + 1024**3)
    return max(1, min(config.runtime.n_jobs, n_folds, int(memory_budget() * .7) // per_worker))


def snapshot(store: AlignmentStore, expression: ExpressionInput | None) -> dict[str, Any]:
    return {
        "store": {name: getattr(store, name) for name in
                  ["ogs", "species", "retained_species", "cache_frequencies",
                   "_frequency_caches", "tpm_identity"]},
        "expression": None if expression is None else
        {name: getattr(expression, name) for name in ["features", "caches", "present"]},
    }


def run_fold(config: AppConfig, store: AlignmentStore, expression: ExpressionInput | None,
             manifest: pl.DataFrame, fold: str, directory: Path, model_base: int,
             progress: Callable[[str], None] | None = None) -> Path:
    from phenoradar.amino_acid_prediction import prediction_members
    from phenoradar.amino_acid_run import fit_scope, predict_members

    directory.mkdir(parents=True)
    tables = directory / "tables"
    tables.mkdir()
    train = manifest.filter((pl.col("fold_id") == fold) & (pl.col("pool") == "train"))
    valid = manifest.filter((pl.col("fold_id") == fold) & (pl.col("pool") == "validation"))
    warnings: list[str] = []
    members = fit_scope(config, store, train.sort("species"), f"outer_fold_{fold}", tables,
                        warnings, expression, progress=progress)
    features = max(len(m.model.coef_) for m in members)
    with AAReportWriter(tables, model_base=model_base) as report:
        report.add(members)
        report.export(directory / "reports")
    compact = prediction_members(members, padding_sites=config.figures.top_features)
    prediction = predict_members(config, store, valid["species"].to_list(), compact,
                                 tables=tables, expression=expression)
    prediction = prediction.join(valid.select("species", "label", "group_id"), on="species")
    prediction = prediction.with_columns(pl.lit(fold).alias("fold_id"))
    joblib.dump((compact, prediction, warnings, features), directory / "result.joblib")
    return directory


_WORKER: tuple[AlignmentStore, ExpressionInput | None] | None = None


def initialize(payload: dict[str, Any], n_jobs: int) -> None:
    global _WORKER
    # Bound Polars as well as BLAS; each process has its own native thread limits.
    os.environ["POLARS_MAX_THREADS"] = str(n_jobs)
    store = object.__new__(AlignmentStore)
    store.__dict__.update(payload["store"])
    store._temp = TemporaryDirectory(prefix="phenoradar_aa_worker_")
    caches = {}
    for index, (key, original) in enumerate(store._frequency_caches.items()):
        cached = copy(original)
        # Copy-on-write detachment is per OG and only when a new site is needed.
        destination = Path(store._temp.name) / f"{index}_frequency.bin"
        os.symlink(cached.path, destination)
        cached.path, cached.shared = destination, True
        caches[key] = cached
    store._frequency_caches = caches
    expression = None
    if payload["expression"] is not None:
        expression = object.__new__(ExpressionInput)
        expression.__dict__.update(payload["expression"])
    _WORKER = store, expression


def worker(config: AppConfig, manifest: pl.DataFrame, fold: str,
           directory: Path, model_base: int) -> Path:
    assert _WORKER is not None
    return run_fold(config, *_WORKER, manifest, fold, directory, model_base)


def outer_folds(config: AppConfig, store: AlignmentStore, expression: ExpressionInput | None,
                manifest: pl.DataFrame, folds: list[str],
                progress: Callable[[str], None] | None) -> Iterator[Path]:
    """Run a pilot fold, then size independent workers from its actual feature count."""
    with TemporaryDirectory(prefix="phenoradar_aa_folds_") as temporary:
        root = Path(temporary)
        first = run_fold(
            config, store, expression, manifest, folds[0], root / folds[0], 0, progress
        )
        yield first
        if len(folds) == 1:
            return
        _, _, _, features = joblib.load(first / "result.joblib")
        n = manifest.filter(pl.col("pool").is_in(["train", "validation"]))["species"].n_unique()
        workers = fold_workers(config, n, features, len(folds) - 1)
        if progress:
            progress(f"AA outer folds: {workers} workers; pilot had {features:,} features.")
        child = config.model_copy(deep=True)
        child.runtime.n_jobs = max(1, config.runtime.n_jobs // workers)
        if workers == 1:
            for i, fold in enumerate(folds[1:], 1):
                yield run_fold(child, store, expression, manifest, fold, root / fold,
                               i * 1_000_000, progress)
            return
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=(snapshot(store, expression), child.runtime.n_jobs),
        ) as executor:
            # At most one queued task per worker. Consume in the original fold order.
            pending = {}
            for i, fold in enumerate(folds[1:], 1):
                pending[i] = executor.submit(worker, child, manifest, fold, root / fold,
                                             i * 1_000_000)
                if len(pending) == workers:
                    index = min(pending)
                    yield pending.pop(index).result()
            for index in sorted(pending):
                yield pending[index].result()


def merge_tables(source: Path, destination: Path) -> None:
    for path in sorted(source.glob("*.tsv")):
        target = destination / path.name
        with target.open("ab") as output, path.open("rb") as input_file:
            if output.tell():
                input_file.readline()
            shutil.copyfileobj(input_file, output)
