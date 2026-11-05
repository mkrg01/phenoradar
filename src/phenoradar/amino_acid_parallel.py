"""Memory-bounded outer folds with private caches and deterministic table merging."""
from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import polars as pl

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_presence import SpeciesPresenceCache
from phenoradar.amino_acid_reporting import AAReportWriter
from phenoradar.amino_acid_resources import (
    MAX_MATRIX_BUFFER,
    SparseUsage,
    process_peak_rss,
    worker_bytes,
)
from phenoradar.combined import ExpressionInput
from phenoradar.config import AppConfig
from phenoradar.resource_budget import memory_budget


@contextmanager
def _worker_threads(n_jobs: int) -> Iterator[None]:
    # Spawned imports may initialize Polars before the pool initializer runs.
    previous = os.environ.get("POLARS_MAX_THREADS")
    os.environ["POLARS_MAX_THREADS"] = str(n_jobs)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("POLARS_MAX_THREADS", None)
        else:
            os.environ["POLARS_MAX_THREADS"] = previous


def fold_workers(config: AppConfig, n_species: int, n_features: int, n_folds: int,
                 usage: SparseUsage | None = None) -> int:
    retained = ((config.model_selection.inner_cv_n_splits or n_species)
                if config.model_selection.search_strategy == "tpe" else 1)
    per_worker = worker_bytes(usage or SparseUsage(), n_species, n_features, retained)
    return max(1, min(config.runtime.n_jobs, n_folds, int(memory_budget() * .7) // per_worker))


def snapshot(store: AlignmentStore, expression: ExpressionInput | None) -> dict[str, Any]:
    state = {name: getattr(store, name) for name in
             ["ogs", "species", "retained_species", "cache_frequencies",
              "_frequency_caches", "tpm_identity"]}
    state["_site_presence"] = store._site_presence.entries
    return {
        "store": state,
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
    usage = getattr(store, "_sparse_usage", SparseUsage())
    usage.peak_rss_bytes = max(usage.peak_rss_bytes, process_peak_rss())
    (directory / "resources.json").write_text(json.dumps(asdict(usage)))
    joblib.dump((compact, prediction, warnings, features), directory / "result.joblib")
    return directory


_WORKER: tuple[AlignmentStore, ExpressionInput | None] | None = None


def initialize(payload: dict[str, Any], n_jobs: int,
               matrix_buffer_bytes: int | None = None) -> None:
    global _WORKER
    # Bound Polars as well as BLAS; each process has its own native thread limits.
    os.environ["POLARS_MAX_THREADS"] = str(n_jobs)
    store = object.__new__(AlignmentStore)
    store.__dict__.update(payload["store"])
    store._temp = TemporaryDirectory(prefix="phenoradar_aa_worker_")
    store._sparse_usage = SparseUsage()
    if matrix_buffer_bytes is not None:
        store._matrix_buffer_bytes = matrix_buffer_bytes
    store._site_presence = SpeciesPresenceCache(
        Path(store._temp.name) / "species_presence.bin", payload["store"]["_site_presence"],
    )
    caches = {}
    for index, (key, original) in enumerate(store._frequency_caches.items()):
        # Existing extents remain immutable in their original files. Each worker
        # appends only new columns to its private file; no OG data are copied.
        destination = Path(store._temp.name) / f"{index}_frequency.bin"
        caches[key] = original.fork(destination)
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
        usage_path = first / "resources.json"
        usage = SparseUsage(**json.loads(usage_path.read_text())) if usage_path.exists() else None
        workers = fold_workers(config, n, features, len(folds) - 1, usage)
        matrix_buffer = min(MAX_MATRIX_BUFFER, max(1,
            int(memory_budget() * .7) // (3 * workers)))
        if progress:
            progress(f"AA outer folds: {workers} workers; pilot had {features:,} features; "
                     f"CSC pair {(usage.matrix_bytes if usage else 0) / 1024**3:.2f} GiB; "
                     f"matrix buffer {matrix_buffer / 1024**3:.2f} GiB/worker.")
        child = config.model_copy(deep=True)
        child.runtime.n_jobs = max(1, config.runtime.n_jobs // workers)
        if workers == 1:
            for i, fold in enumerate(folds[1:], 1):
                yield run_fold(child, store, expression, manifest, fold, root / fold,
                               i * 1_000_000, progress)
            return
        with _worker_threads(child.runtime.n_jobs), ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(snapshot(store, expression), child.runtime.n_jobs, matrix_buffer),
        ) as executor:
            # Queue the remaining small tasks so a free worker never waits for
            # an earlier fold's result. Only paths cross IPC; merge in fold order.
            pending = [
                executor.submit(worker, child, manifest, fold, root / fold, i * 1_000_000)
                for i, fold in enumerate(folds[1:], 1)
            ]
            try:
                for future in pending:
                    yield future.result()
            finally:
                for future in pending:
                    future.cancel()


def merge_tables(source: Path, destination: Path) -> None:
    for path in sorted(source.glob("*.tsv")):
        target = destination / path.name
        with target.open("ab") as output, path.open("rb") as input_file:
            if output.tell():
                input_file.readline()
            shutil.copyfileobj(input_file, output)
