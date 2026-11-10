"""Categorical tree heatmaps of representative sequence residues, for display only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from phenoradar.amino_acid import ALPHABET, AlignmentStore
from phenoradar.figures import _orthogroup_annotation_lookup
from phenoradar.tree_prediction import _write_tree_feature_heatmap_svg


def build_residue_annotation(
    store: AlignmentStore, sites: pl.DataFrame, predictions: pl.DataFrame,
    *, aggregation: str, annotations: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Choose one gene per species/OG, independently of the displayed site.

    Frequencies have no unique representative: use maximum TPM in weighted mode,
    otherwise maximum canonical coverage over the input MSA. This display never
    replaces a model's training-frozen best-covered representative or frequency.
    """
    lookup = _orthogroup_annotation_lookup(annotations)
    species = predictions.sort("species").to_dicts()
    ranked = sites.with_row_index("feature_rank", offset=1)
    records = []
    canonical = np.frombuffer(ALPHABET.encode("ascii"), dtype=np.uint8)
    for og_name in ranked["orthogroup"].unique(maintain_order=True):
        og = store.ogs[og_name]
        matrix = og.matrix()
        weighted = aggregation == "tpm_weighted_frequency"
        if weighted and og.tpm is None:
            raise ValueError("Representative residue heatmap requires sequence TPM")
        coverage = None if weighted else np.isin(matrix, canonical).sum(axis=1)
        for meta in species:
            sample = str(meta["species"])
            candidates = og.species_rows.get(sample, [])
            row = None
            status = "missing_sequence"
            if candidates:
                scores = og.tpm if weighted else coverage
                assert scores is not None
                row = min(candidates, key=lambda r: (-float(scores[r]), og.sequence_ids[r]))
                status = "observed"
                if float(scores[row]) <= 0:
                    row = None
                    status = "zero_tpm" if weighted else "no_canonical_residues"
            # Only selected sites are materialized; no all-species frequency tensor.
            for site in ranked.filter(pl.col("orthogroup") == og_name).iter_rows(named=True):
                position = int(site["alignment_pos_1based"])
                if not 1 <= position <= og.width:
                    raise ValueError(f"Site outside input alignment: {og_name}:{position}")
                residue = None if row is None else chr(matrix[row, position - 1])
                call_status = status
                sequence_position = None
                if row is not None:
                    if residue in {"-", "."}:
                        call_status = "gap"
                    else:
                        sequence_position = int(np.count_nonzero(
                            ~np.isin(matrix[row, :position], [ord("-"), ord(".")])
                        ))
                        if residue is not None and residue not in ALPHABET:
                            call_status = "unknown_residue"
                records.append({
                    **site, "feature": json.dumps([og_name, position], separators=(",", ":")),
                    "orthogroup_annotation": lookup.get(og_name),
                    "species": sample, "true_label": meta["label"], "prob": meta["prob"],
                    **({"decision_status": meta["decision_status"]}
                       if "decision_status" in meta else {}),
                    "residue": residue, "residue_status": call_status,
                    "representative_sequence_id": None if row is None else og.sequence_ids[row],
                    "representative_tpm": (None if row is None or og.tpm is None
                                           else float(og.tpm[row])),
                    "representative_selection": ("max_tpm" if weighted
                                                 else "max_input_msa_coverage"),
                    "sequence_pos_1based": sequence_position,
                    "color_scheme": "Zappo",
                })
    return pl.DataFrame(records, schema_overrides={
        "orthogroup_annotation": pl.String, "residue": pl.String,
        "representative_sequence_id": pl.String, "representative_tpm": pl.Float64,
        "sequence_pos_1based": pl.Int64,
    }) if records else pl.DataFrame()


def write_residue_tree(
    run_dir: Path, tree_path: Path | None, store: AlignmentStore, sites: pl.DataFrame,
    predictions: pl.DataFrame, *, aggregation: str, annotations: pl.DataFrame | None = None,
) -> list[str]:
    if tree_path is None:
        return []
    if sites.is_empty():
        return ["Skipped tree_feature_heatmap_amino_acid.svg: no positive-PSS sites."]
    table = build_residue_annotation(
        store, sites, predictions, aggregation=aggregation, annotations=annotations,
    )
    table_path = run_dir / "cv/tables/tree_feature_heatmap_amino_acid_annotation.tsv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.sort("feature_rank", "species").write_csv(table_path, separator="\t")
    return _write_tree_feature_heatmap_svg(
        tree_path=tree_path, annotation=table, value_col="residue", cmap_name="",
        out_path=run_dir / "cv/figures/tree_feature_heatmap_amino_acid.svg",
        title="Representative residues (sites ranked by mean OOF PSS)",
        annotate_features=True,
    )
