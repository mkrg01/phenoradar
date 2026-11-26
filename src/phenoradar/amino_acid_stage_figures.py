"""Interpret final-model predictions with the same residue displays used for CV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_blocks import coefficient_blocks
from phenoradar.amino_acid_composition import write_group_trait_composition
from phenoradar.amino_acid_diagnostics import write_frequency_by_confusion
from phenoradar.amino_acid_outputs import write_frequency_table
from phenoradar.amino_acid_prediction import prediction_members
from phenoradar.amino_acid_tree import write_residue_tree
from phenoradar.config import AppConfig
from phenoradar.cv_display import with_prediction_context
from phenoradar.orthogroup_annotation import load_orthogroup_annotations

if TYPE_CHECKING:
    from phenoradar.amino_acid_run import AAMember
    from phenoradar.combined import ExpressionInput


def final_model_sites(members: list[AAMember], limit: int) -> pl.DataFrame:
    """Rank by mean per-model PSS, including zeros from models omitting a site."""
    scores: dict[tuple[str, int], float] = {}
    for member in prediction_members(members):
        for block, beta, _ in coefficient_blocks(member.transform.sites, member.model.coef_,
                                                member.expression_width):
            pss = (np.abs(beta) * block["known"]).sum(axis=1)
            for i in np.flatnonzero(pss > 0):
                key = str(block["orthogroup"][i]), int(block["column"][i]) + 1
                scores[key] = scores.get(key, 0.0) + float(pss[i]) / len(members)
    return pl.DataFrame([
        {"orthogroup": og, "alignment_pos_1based": position, "pss": score}
        for (og, position), score in sorted(scores.items(), key=lambda item: (
            -item[1], item[0]))[:limit]
    ], schema={"orthogroup": pl.String, "alignment_pos_1based": pl.Int64, "pss": pl.Float64})


def write_prediction_interpretation(
    run_dir: Path, stage: str, config: AppConfig, store: AlignmentStore,
    members: list[AAMember], predictions: pl.DataFrame,
    *, expression: ExpressionInput | None = None, metadata_path: Path | None = None,
) -> list[str]:
    from phenoradar.amino_acid_figures import _final_features

    if predictions.is_empty():
        return []
    context = with_prediction_context(
        predictions, metadata_path=metadata_path or Path(config.data.metadata_path),
        species_col=config.data.species_col, group_col=config.split.group_col,
        trait_col=config.data.trait_col,
    ).with_columns(pl.col("true_label").alias("label"))
    limit = config.figures.top_features
    sites = final_model_sites(members, limit)
    importance, coefficients = _final_features(members)
    annotation_path = (Path(config.data.orthogroup_annotation_path)
                       if config.data.orthogroup_annotation_path else None)
    annotations = load_orthogroup_annotations(
        annotation_path if annotation_path is not None and annotation_path.is_file() else None,
        feature_names=sorted({site.orthogroup for member in prediction_members(members)
                              for site in member.transform.sites} |
                             {og for member in members if member.expression is not None
                              for og in member.expression.orthogroups}),
    )
    tree_path = Path(config.data.tree_path) if config.data.tree_path else None
    warnings = write_residue_tree(
        run_dir, tree_path, store, sites, context, stage=stage,
        aggregation=config.preprocess.amino_acid.aggregation, annotations=annotations,
        trait_name=config.data.trait_col,
    )
    tables = run_dir / stage / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    sites.write_csv(tables / "amino_acid_site_pss_top_summary.tsv", separator="\t")
    # A final-model feature set is fixed before examining held-out or unknown species.
    aa_importance = importance.filter(
        pl.col("feature").cast(pl.String).str.json_path_match("$[2]").is_not_null())
    if aa_importance.height:
        write_frequency_table(tables, store, pl.DataFrame(), members, aa_importance, limit,
                              prediction_species=context["species"].to_list())
        if context["label"].is_not_null().any():
            write_frequency_by_confusion(run_dir, context, limit, annotations, stage=stage)
    warnings.extend(write_group_trait_composition(
        run_dir, store, sites, context, stage=stage,
        aggregation=config.preprocess.amino_acid.aggregation, annotations=annotations,
        prediction_labels=stage in {"inference", "predict"},
        trait_name=config.data.trait_col,
    ))
    if expression is not None:
        from phenoradar import figures
        from phenoradar.cv import _build_top_feature_expression_from_matrix
        from phenoradar.missing_expression import mark_expression_observations
        from phenoradar.tree_prediction import write_expression_tree_heatmaps

        top = importance.filter(pl.col("feature").cast(pl.String).str.json_path_match("$[0]") ==
                                "expression").sort(["importance_mean", "feature"],
                                                   descending=[True, False]).head(limit)
        if top.height:
            names = top["feature"].to_list()
            raw = _build_top_feature_expression_from_matrix(
                species_order=context["species"].to_list(), feature_names=names,
                matrix=expression.matrix(context["species"].to_list(),
                                         [json.loads(name)[1] for name in names]),
                feature_importance=top, feature_limit=limit,
            )
            missing = config.preprocess.expression.missing_expression
            if missing.method == "neutral":
                raw = mark_expression_observations(raw, zero_as_missing=missing.zero_as_missing)
            raw.write_csv(tables / "top_feature_expression.tsv", separator="\t")
            if context["label"].is_not_null().any():
                figures._top_feature_expression_by_confusion(
                    oof_predictions=context, top_feature_expression=raw, feature_importance=top,
                    coefficients=coefficients, out_path=run_dir / stage / "figures" /
                    "top_feature_expression_by_confusion.svg", top_features=limit,
                    orthogroup_annotations=annotations,
                )
            warnings.extend(write_expression_tree_heatmaps(
                run_dir=run_dir, stage=stage, tree_path=tree_path, predictions=context,
                feature_importance=top, coefficients=coefficients,
                tpm_path=Path(config.data.tpm_path),
                species_col=config.data.species_col, feature_col=config.data.feature_col,
                value_col=config.data.value_col, top_feature_expression=raw,
                trait_name=config.data.trait_col,
                feature_limit=limit, orthogroup_annotations=annotations,
                preserve_missing=missing.method == "neutral",
                zero_as_missing=config.preprocess.expression.missing_expression.zero_as_missing,
            ))
    return warnings
