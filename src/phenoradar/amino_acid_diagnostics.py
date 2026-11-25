"""Site stability and held-out prediction diagnostics for sequence modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from phenoradar import figures

_SITE_KEYS = ["orthogroup", "alignment_pos_1based"]


def site_score_summary(
    source: pl.LazyFrame,
    selections: pl.DataFrame,
    metric: str,
    limit: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Average models within folds, then folds, including all zero/absent sites."""
    counts = (
        selections.filter(pl.col("scope").str.starts_with("outer_fold_"))
        .group_by("scope")
        .len(name="n_models")
    )
    if counts.is_empty():
        raise figures.FigureError("Site summaries require selected outer-fold models")
    scores = (
        source.filter(pl.col(metric).is_finite() & (pl.col(metric) > 0))
        .group_by("scope", *_SITE_KEYS)
        .agg(pl.col(metric).sum())
        .join(counts.lazy(), on="scope")
        .with_columns((pl.col(metric) / pl.col("n_models")).alias(metric))
    )
    top = (
        scores.group_by(_SITE_KEYS)
        .agg((pl.col(metric).sum() / counts.height).alias(metric))
        .sort([metric, *_SITE_KEYS], descending=[True, False, False])
        .head(limit)
        .collect(engine="streaming")
    )
    folds = (
        top.select(_SITE_KEYS)
        .lazy()
        .join(counts.lazy(), how="cross")
        .join(scores, on=["scope", *_SITE_KEYS, "n_models"], how="left")
        .with_columns(pl.col(metric).fill_null(0))
        .collect(engine="streaming")
    )
    return top, folds


def _site_labels(frame: pl.DataFrame, annotations: pl.DataFrame | None) -> list[str]:
    features = [json.dumps([og, position]) for og, position in frame.select(_SITE_KEYS).iter_rows()]
    return figures._feature_detail_labels(features, annotations)


def write_site_scores(
    run_dir: Path,
    top: pl.DataFrame,
    folds: pl.DataFrame,
    metric: str,
    annotations: pl.DataFrame | None,
    *, unit_label: str = "Fold",
) -> None:
    stem = f"amino_acid_site_{metric}_top"
    top.write_csv(run_dir / "cv/tables" / f"{stem}_summary.tsv", separator="\t")
    folds.sort([*_SITE_KEYS, "scope"]).write_csv(
        run_dir / "cv/tables" / f"{stem}_by_fold.tsv", separator="\t"
    )
    path = run_dir / "cv/figures" / f"{stem}.svg"
    if top.is_empty():
        figures._write_message_figure(
            title="Site scores",
            message="No nonzero site scores.",
            out_path=path,
            width_px=720,
            height_px=220,
        )
        return
    labels = _site_labels(top, annotations)
    row_height = max(0.32, 0.12 * max(label.count("\n") + 1 for label in labels))
    fig, ax = plt.subplots(figsize=(7.2, max(2.8, len(labels) * row_height + 0.8)))
    for index, (og, position) in enumerate(top.select(_SITE_KEYS).iter_rows()):
        values = (
            folds.filter(
                (pl.col("orthogroup") == og) & (pl.col("alignment_pos_1based") == position)
            )
            .sort("scope")[metric]
            .to_numpy()
        )
        ax.scatter(
            values,
            index + figures._deterministic_offsets(len(values), 0.16),
            s=9,
            color=figures._COLOR_BLUE,
            alpha=0.5,
            linewidths=0,
            zorder=2,
        )
    ax.scatter(top[metric], np.arange(top.height), marker="D", s=20, color="#222222", zorder=3)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.grid(axis="x", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Site PSS")
    ax.set_ylabel("Orthogroup / MSA position")
    figures._legend_above(
        ax,
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=figures._COLOR_BLUE,
                markersize=3,
                label=f"{unit_label} mean",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                color="#222222",
                markersize=3,
                label="Overall mean",
            ),
        ],
        ncol=2,
    )
    figures._save_svg_figure(fig, path)


def write_site_stability(
    run_dir: Path,
    top: pl.DataFrame,
    annotations: pl.DataFrame | None,
    *, unit_label: str = "Fold",
) -> None:
    source = run_dir / "cv/tables/amino_acid_site_stability.tsv"
    if not source.exists() or top.is_empty():
        return
    table = top.join(
        pl.scan_csv(source, separator="\t")
        .join(top.select(_SITE_KEYS).lazy(), on=_SITE_KEYS, how="semi")
        .collect(engine="streaming"),
        on=_SITE_KEYS,
        how="left",
        validate="1:1",
    )
    table.write_csv(run_dir / "cv/tables/amino_acid_site_stability_top.tsv", separator="\t")
    labels = _site_labels(table, annotations)
    row_height = max(0.32, 0.12 * max(label.count("\n") + 1 for label in labels))
    fig, ax = plt.subplots(figsize=(7.2, max(2.8, len(labels) * row_height + 0.8)))
    ys = np.arange(table.height)
    eligible = (table["n_eligible_folds"] / table["n_outer_folds"]).to_numpy()
    selected = table["selection_frequency"].to_numpy()
    ax.barh(ys, eligible, height=0.65, color="#dceaf3", label="Eligible")
    ax.barh(ys, selected, height=0.35, color=figures._COLOR_BLUE, label="Selected")
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(
        "Fraction of all groups" if unit_label == "Group" else "Fraction of all outer folds"
    )
    ax.set_ylabel("Orthogroup / MSA position")
    ax.grid(axis="x", linewidth=0.5)
    ax.set_axisbelow(True)
    figures._legend_above(ax, ncol=2)
    figures._save_svg_figure(fig, run_dir / "cv/figures/amino_acid_site_stability_top.svg")


def write_frequency_by_confusion(
    run_dir: Path,
    oof: pl.DataFrame,
    limit: int,
    annotations: pl.DataFrame | None = None,
    *, stage: str = "cv",
) -> None:
    source = run_dir / stage / "tables/top_amino_acid_frequencies.tsv"
    if not source.exists():
        return
    data = pl.read_csv(source, separator="\t")
    names = data["feature"].unique(maintain_order=True).head(limit).to_list()
    if names:
        figures._write_feature_values_by_confusion(
            values=data.rename({"frequency": "value"}),
            predictions=oof,
            features=names,
            out_path=run_dir / stage / "figures/amino_acid_frequency_by_confusion.svg",
            value_label="Residue frequency",
            value_limits=(-0.05, 1.05),
            orthogroup_annotations=annotations,
            context="OOF" if stage == "cv" else "Final-model predictions",
        )


def write_site_filter_funnel(
    run_dir: Path,
    *,
    retained_counts: pl.DataFrame | None = None,
    ranking_enabled: bool = False,
) -> None:
    cv = run_dir / "cv/tables"
    raw = (
        pl.scan_csv(cv / "amino_acid_og_qc.tsv", separator="\t")
        .filter(pl.col("scope").str.starts_with("outer_fold_"))
        .group_by("scope", "sample_set_id")
        .agg(
            pl.col("n_columns").sum(),
            pl.col("n_after_call_filter").sum(),
            pl.col("n_after_variant_filter").sum(),
        )
        .collect(engine="streaming")
    )
    columns = {
        "alignment_sites": "n_columns",
        "observed_sites": "n_after_call_filter",
        "variable_sites": "n_after_variant_filter",
    }
    qc = cv / "amino_acid_site_qc.tsv"
    if retained_counts is not None or qc.exists():
        retained = (
            retained_counts
            if retained_counts is not None
            else (
                pl.scan_csv(qc, separator="\t")
                .filter(pl.col("n_active_states") > 0)
                .group_by("scope", "sample_set_id")
                .len(name="n_retained_sites")
                .collect(engine="streaming")
            )
        )
        raw = raw.join(retained, on=["scope", "sample_set_id"], how="left").with_columns(
            pl.col("n_retained_sites").fill_null(0)
        )
        columns["retained_sites"] = "n_retained_sites"
    if raw.is_empty():
        return
    rows = []
    for stage, column in columns.items():
        quantiles = np.quantile(raw[column].to_numpy(), [0, 0.25, 0.5, 0.75, 1])
        rows.append(
            {
                "scope": "outer_fold",
                "stage": stage,
                "n_records": raw.height,
                **dict(
                    zip(
                        [
                            "n_features_min",
                            "n_features_q1",
                            "n_features_median",
                            "n_features_q3",
                            "n_features_max",
                        ],
                        quantiles,
                        strict=True,
                    )
                ),
            }
        )
    summary = pl.DataFrame(rows)
    summary.write_csv(cv / "amino_acid_site_filter_counts_summary.tsv", separator="\t")
    stages = list(columns)
    if "retained_sites" in columns and not ranking_enabled:
        stages.remove("retained_sites")
    figures._feature_filter_funnel(
        summary, run_dir / "cv/figures/amino_acid_site_filter_funnel.svg", stage_order=stages
    )
