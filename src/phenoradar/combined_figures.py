"""Joint-model diagnostics using held-out contributions and common feature plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from phenoradar import figures
from phenoradar.cv import _summarize_feature_filter_counts
from phenoradar.glmnet import DEFAULT_LAMBDA


def write_combined_figures(
    run_dir: Path, top_features: int, *,
    final_importance: pl.DataFrame | None = None,
    final_coefficients: pl.DataFrame | None = None,
) -> None:
    """Keep each modality's coefficient/importance scale separate."""
    cv = run_dir / "cv/tables"
    for modality in ["expression", "amino_acid"]:
        for name, column, draw in [
            ("feature_importance", "importance_mean", figures._feature_importance_top),
            ("coefficients", "coef_mean", figures._coefficients_signed_top),
        ]:
            suffix = "importance_top" if name == "feature_importance" else "coefficients_signed_top"
            frame = pl.scan_csv(cv / f"{modality}_{name}.tsv", separator="\t").with_columns(
                pl.col(column).abs().alias("_order")
            ).sort(["_order", "feature"], descending=[True, False]).head(top_features).drop(
                "_order"
            ).collect(engine="streaming")
            if not frame.is_empty():
                draw(frame, run_dir / "cv/figures" / f"{modality}_{suffix}.svg",
                     top_features=top_features,
                     x_label=(f"Mean {modality} importance (normalized within modality)"
                              if name == "feature_importance" else
                              f"Mean {modality} coefficient (model units)"))
            final = final_importance if name == "feature_importance" else final_coefficients
            if final is not None:
                mask = [
                    (len(axis := json.loads(feature)) == 2 and axis[0] == "expression")
                    == (modality == "expression") for feature in final["feature"]
                ]
                selected = final.filter(pl.Series(mask, dtype=pl.Boolean))
                if not selected.is_empty():
                    draw(selected, run_dir / "model/figures" / f"{modality}_{suffix}.svg",
                         top_features=top_features,
                         x_label=f"Final {modality} " + (
                             "importance (normalized within modality)"
                             if name == "feature_importance" else "coefficient (model units)"))
    _modality_contributions(run_dir)
    _penalty_selection(run_dir)
    _expression_filter_funnel(run_dir)


def _expression_filter_funnel(run_dir: Path) -> None:
    cv = run_dir / "cv/tables"
    counts = pl.read_csv(cv / "expression_feature_filter_counts.tsv", separator="\t").with_columns(
        pl.col("scope").str.replace("^outer_fold_", "").alias("fold_id"),
        pl.lit("outer_fold").alias("scope"),
    )
    summary = _summarize_feature_filter_counts(counts)
    summary.write_csv(cv / "expression_feature_filter_counts_summary.tsv", separator="\t")
    figures._feature_filter_funnel(
        summary, run_dir / "cv/figures/expression_feature_filter_funnel.svg",
    )


def _modality_contributions(run_dir: Path) -> None:
    cv = run_dir / "cv/tables"
    columns = ["expression_contribution", "amino_acid_contribution", "intercept", "logit"]
    data = pl.scan_csv(cv / "combined_modality_contributions.tsv", separator="\t").group_by(
        "species"
    ).agg(*(pl.col(column).mean() for column in columns), pl.len().alias("n_models")).collect(
        engine="streaming"
    ).join(pl.read_csv(cv / "prediction_cv.tsv", separator="\t",
                      schema_overrides={"fold_id": pl.String}).select(
                          "species", "label", "prob", "fold_id"), on="species", how="inner"
    ).sort("species")
    data.write_csv(cv / "combined_modality_contributions_summary.tsv", separator="\t")
    colors = figures._binary_trait_color_map(
        [0, 1], source_table_name="prediction_cv.tsv",
        figure_name="combined_modality_contributions.svg",
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    for label in [0, 1]:
        rows = data.filter(pl.col("label") == label)
        axis.scatter(rows["expression_contribution"], rows["amino_acid_contribution"],
                     color=colors[label], alpha=.7, label=f"Trait {label}")
    axis.axhline(0, color="gray", linewidth=.6)
    axis.axvline(0, color="gray", linewidth=.6)
    axis.set(xlabel="Mean expression contribution (log-odds)",
             ylabel="Mean amino-acid contribution (log-odds)",
             title="Held-out modality contributions")
    axis.legend()
    figure.text(.5, .01,
                "Mean model contributions; ensemble probabilities aggregate after sigmoid.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, 1))
    figures._save_svg_figure(figure, run_dir / "cv/figures/combined_modality_contributions.svg")


def _penalty_selection(run_dir: Path) -> None:
    rows = []
    for stage in ["cv", "model"]:
        path = run_dir / stage / "tables/model_selection_selected.tsv"
        if not path.exists():
            continue
        for row in pl.read_csv(path, separator="\t").iter_rows(named=True):
            params = json.loads(row["params_json"])
            rows.append({key: row[key] for key in ["scope", "sample_set_id", "candidate_index"]}
                        | {"lambda": params.get("lambda", DEFAULT_LAMBDA),
                           "amino_acid_penalty_ratio": params.get("amino_acid_penalty_ratio", 1.)})
    if not rows:
        return
    data = pl.DataFrame(rows)
    scopes = sorted(set(data["scope"]), key=lambda s: (
        s == "final_refit", int(s.removeprefix("outer_fold_")) if s != "final_refit" else 0
    ))
    data = data.with_columns(pl.Series("_order", [scopes.index(s) for s in data["scope"]])).sort(
        "_order", "sample_set_id", "candidate_index"
    )
    data.drop("_order").write_csv(run_dir / "cv/tables/combined_penalty_selection.tsv",
                                  separator="\t")
    figure, axes = plt.subplots(2, 1, figsize=(max(6, len(scopes) * .3), 5), sharex=True)
    for axis, column, label in zip(
        axes, ["lambda", "amino_acid_penalty_ratio"],
        ["Selected lambda", "AA / expression penalty ratio"], strict=True,
    ):
        axis.scatter(data["_order"], data[column], color="#4477aa", alpha=.7)
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.2)
    axes[1].set_xticks(np.arange(len(scopes)),
                      [s.removeprefix("outer_fold_") for s in scopes], rotation=45)
    axes[1].set_xlabel("Outer fold / final refit (inner-CV selection)")
    figure.tight_layout()
    figures._save_svg_figure(figure, run_dir / "cv/figures/combined_penalty_selection.svg")
