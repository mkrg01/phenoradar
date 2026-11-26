"""Optional structure postprocessing for a completed sequence-mode run."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Literal

import polars as pl

from phenoradar.amino_acid import ALPHABET, AlignmentOG, AlignmentStore
from phenoradar.config.schema import StructureReportConfig
from phenoradar.site_pdb_search import prepare_pdb_catalog
from phenoradar.site_reference import (
    ReferencePolicy,
    _check_annotation_tools,
    _checked_alignment,
    _run_inputs,
    _run_orthodb_release,
    annotate_sites,
    selected_sites,
)
from phenoradar.site_reference_resources import ResourceCache, SiteAnnotationError, write_json


def _representative_query(og: AlignmentOG) -> tuple[str, str]:
    rows = []
    for sequence_id, row in zip(og.sequence_ids, og.matrix(), strict=True):
        sequence = row.tobytes().decode("ascii").replace("-", "").replace(".", "")
        sequence = sequence.removesuffix("*").replace("*", "X")
        if any(residue in ALPHABET for residue in sequence):
            rows.append((sequence_id, sequence))
    if not rows:
        raise SiteAnnotationError(
            f"No training sequence with standard residues in {og.orthogroup}."
        )
    middle = median(len(sequence) for _, sequence in rows)
    return min(rows, key=lambda row: (abs(len(row[1]) - middle), row[0]))


def run_site_annotation_report(
    run: Path,
    options: StructureReportConfig,
    scope: Literal["final", "cv"],
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Save a separate status so an optional annotation cannot invalidate model results."""
    started = perf_counter()
    inputs = run / "site_annotation_inputs"
    status: dict[str, Any] = {"scope": scope, "render": options.render}
    try:
        top = (
            selected_sites(run, scope).group_by("orthogroup").agg(pl.col("pss").max())
            .sort(["pss", "orthogroup"], descending=[True, False]).head(options.top_orthogroups)
        )
        names: list[str] = top["orthogroup"].to_list()
        status["orthogroups"] = names
        if not names:
            status.update(status="skipped", reason="no_selected_sites")
            return
        if progress:
            progress(f"Annotate protein structures for {len(names):,} selected orthogroups.")
        _check_annotation_tools(options.mafft_exe, options.pymol_exe, options.render)
        config, species, training = _run_inputs(run)
        release = _run_orthodb_release(config, options.orthodb_release, required=False)
        inventory = pl.read_csv(run / "model/tables/alignment_inputs.tsv", separator="\t")
        expected = {str(row["orthogroup"]): row for row in inventory.iter_rows(named=True)}
        with AlignmentStore(
            Path(config["data"]["alignment_dir"]), species, cache_species=training,
            only_orthogroups=set(names),
        ) as store:
            rows = []
            sequence_ids = {}
            for name in names:
                og = store.ogs[name]
                _checked_alignment(og, expected.get(name))
                sequence_id, sequence = _representative_query(og)
                rows.append(f">{name} {sequence_id}\n{sequence}\n")
                sequence_ids[name] = sequence_id
        inputs.mkdir(exist_ok=True)
        queries = inputs / "queries.fa"
        queries.write_text("".join(rows), encoding="utf-8")
        catalog = inputs / "pdb_candidates.json"
        cache = ResourceCache(Path(options.cache_dir))
        prepare_pdb_catalog(
            orthogroups=names, release=release, queries=queries, out=catalog, cache=cache,
            alignment_sequence_ids=sequence_ids,
        )
        annotate_sites(
            run=run, catalog=catalog, out=run / "site_annotations", scope=scope,
            orthogroups=names, policy=ReferencePolicy(), cache=cache, mafft=options.mafft_exe,
            pymol=options.pymol_exe, render=options.render, orthodb_release=release,
        )
        status.update(status="completed", manifest="site_annotations/manifest.json")
    except Exception as exc:
        status.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        if progress:
            progress(f"Warning: protein structure annotation failed: {exc}")
    finally:
        status["duration_sec"] = perf_counter() - started
        write_json(inputs / "status.json", status)
