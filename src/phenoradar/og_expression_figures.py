"""Raw orthogroup expression by CV group and observed trait, shared across modes."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from phenoradar.colors import BINARY_TRAIT_COLORS
from phenoradar.config import AppConfig
from phenoradar.cv import CVError, ExpressionMatrixBuilder
from phenoradar.cv_display import with_cv_groups
from phenoradar.figure_population import (
    population_figure_path,
    prediction_figure_populations,
    write_population_message_svg,
)
from phenoradar.figures import (
    _feature_address,
    _orthogroup_annotation_lookup,
    _top_feature_importance_features,
)
from phenoradar.tree_prediction import _cell_text_color

_DISTRIBUTION = "og_expression_by_group_trait"
_HEATMAP = "og_expression_group_trait_heatmap"


def expression_orthogroups(importance: pl.DataFrame, limit: int) -> list[str]:
    """Keep the rank of the selected features while collapsing repeated OGs."""
    return list(dict.fromkeys(_feature_address(name)[0] for name in
                             _top_feature_importance_features(importance, top_features=limit)))


def read_og_expression(
    config: AppConfig, species: list[str], orthogroups: list[str],
) -> pl.DataFrame:
    """Read only selected coordinates; absent rows stay absent, explicit zeros stay zero.

    Reuse the training reader's normalization, duplicate summation and validation,
    filtering OGs before aggregation so memory is bounded by the selected subset.
    No fitted preprocessing or absent-feature fill is applied to this diagnostic.
    """
    builder = ExpressionMatrixBuilder(config, absent_feature_fill="nan")
    try:
        rows = builder._normalized_rows_for_species(species).filter(
            pl.col("__feature").is_in(orthogroups))
        values = builder._aggregate_normalized_rows(rows).collect(engine="streaming")
        builder._validate_tpm_frame(values)
        return values.select(pl.col("__species").alias("species"),
                             pl.col("__feature").alias("orthogroup"),
                             pl.col("__value").alias("tpm"))
    finally:
        builder.close()


def expression_tables(
    values: pl.DataFrame, predictions: pl.DataFrame, orthogroups: list[str],
    *, annotations: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize species equally; heatmap medians are on the displayed log scale."""
    if predictions["species"].n_unique() != predictions.height:
        raise ValueError("OG expression requires one CV row per species")
    if predictions["group_id"].null_count() or predictions.filter(
        ~pl.col("label").is_in([0, 1]) | pl.col("label").is_null()
    ).height:
        raise ValueError("OG expression requires groups and observed binary labels")
    predictions = predictions.with_columns(pl.col("group_id").cast(pl.String))
    groups = sorted(predictions["group_id"].unique().to_list(),
                    key=lambda value: (0, int(value), value) if value.isdigit() else (1, 0, value))
    lookup = _orthogroup_annotation_lookup(annotations)
    observed = values.join(predictions.select("species", "group_id", "label"), on="species",
                           how="inner", validate="m:1").with_columns(
        (pl.col("tpm") + 1).log(base=2).alias("log2_tpm_plus1"))
    if observed.select("species", "orthogroup").is_duplicated().any():
        raise ValueError("OG expression requires aggregated species/OG coordinates")
    if observed.filter(~pl.col("tpm").is_finite() | (pl.col("tpm") < 0) |
                       pl.col("tpm").is_null()).height:
        raise ValueError("OG expression requires finite, non-negative observed TPM")
    stats: list[dict[str, Any]] = []
    points: list[pl.DataFrame] = []
    for rank, og in enumerate(dict.fromkeys(orthogroups), 1):
        for group_index, group in enumerate(groups):
            for trait, offset in [(0, -.19), (1, .19)]:
                members = predictions.filter((pl.col("group_id") == group) &
                                             (pl.col("label") == trait))
                part = observed.filter((pl.col("orthogroup") == og) &
                                       (pl.col("group_id") == group) &
                                       (pl.col("label") == trait)).sort(
                                           ["log2_tpm_plus1", "species"])
                n = part.height
                quantiles = np.percentile(part["log2_tpm_plus1"], [25, 50, 75]) if n else [None]*3
                stats.append({"orthogroup": og, "og_rank": rank,
                    "orthogroup_annotation": lookup.get(og), "group_id": group,
                    "group_index": group_index, "trait": trait,
                    "n_total": members.height, "n_observed": n, "n_missing": members.height-n,
                    "n_zero": int((part["tpm"] == 0).sum()),
                    "median_tpm": part["tpm"].median(), "mean_tpm": part["tpm"].mean(),
                    "q25_log2_tpm_plus1": quantiles[0], "median_log2_tpm_plus1": quantiles[1],
                    "q75_log2_tpm_plus1": quantiles[2]})
                if n:
                    dx = np.linspace(-.115, .115, n) if n > 1 else np.zeros(n)
                    points.append(part.with_row_index("within_stratum_rank", offset=1).with_columns(
                        pl.lit(rank).alias("og_rank"), pl.lit(group_index).alias("group_index"),
                        pl.Series("plot_x", group_index + offset + dx)))
    summary = pl.DataFrame(stats, schema_overrides={
        "orthogroup_annotation": pl.String,
        **dict.fromkeys(["median_tpm", "mean_tpm", "q25_log2_tpm_plus1",
                         "median_log2_tpm_plus1", "q75_log2_tpm_plus1"], pl.Float64)})
    point_table = pl.concat(points) if points else observed.head(0).with_columns(
        pl.lit(None, dtype=pl.UInt32).alias("within_stratum_rank"),
        pl.lit(None, dtype=pl.Int32).alias("og_rank"),
        pl.lit(None, dtype=pl.Int32).alias("group_index"),
        pl.lit(None, dtype=pl.Float64).alias("plot_x"))
    return summary, point_table


def _distribution_page(summary: pl.DataFrame, points: pl.DataFrame, ranks: list[int]) -> Figure:
    groups = summary.select("group_index", "group_id").unique().sort("group_index")["group_id"]
    width = max(7, len(groups) * .44 + .85)
    titles = []
    for rank in ranks:
        og = summary.filter(pl.col("og_rank") == rank).row(0, named=True)
        annotation = " ".join((og["orthogroup_annotation"] or "").split())
        title = f"{rank}.  {annotation} ({og['orthogroup']})" if annotation else og["orthogroup"]
        titles.append(textwrap.fill(title, width=int(width*12), break_long_words=False))
    title_heights = [.20 + .16*len(title.splitlines()) for title in titles]
    plot_height = 1.35
    height = .4 + sum(title_heights) + plot_height*len(ranks) + .35*(len(ranks)-1) + .92
    fig = plt.figure(figsize=(width, height))
    axes = []
    cursor = height-.4
    for title_height in title_heights:
        bottom = cursor-title_height-plot_height
        axes.append(fig.add_axes((.09, bottom/height, .895, plot_height/height)))
        cursor = bottom-.35
    fig.supylabel("log2(TPM + 1)", x=.012, fontsize=8.5)
    handles = [Line2D([], [], color=BINARY_TRAIT_COLORS[t], marker="o", markersize=4,
                      lw=0, label=f"Trait {t}") for t in [0, 1]]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="upper right",
               bbox_to_anchor=(.985, .995), fontsize=8, handletextpad=.4, columnspacing=1)
    for row_index, (ax, rank) in enumerate(zip(axes, ranks, strict=True)):
        og = summary.filter(pl.col("og_rank") == rank).row(0, named=True)
        values = (points.filter(pl.col("orthogroup") == og["orthogroup"])
                  if points.height else points)
        maximum = float(values["log2_tpm_plus1"].to_numpy().max()) if values.height else 0
        limit = max(3, math.ceil(maximum)) + .45
        for group_index, group in enumerate(groups):
            if group_index % 2 == 0:
                ax.axvspan(group_index-.5, group_index+.5, color="#f5f5f5", zorder=0)
            for trait, offset in [(0, -.19), (1, .19)]:
                part = values.filter((pl.col("group_id") == group) & (pl.col("label") == trait)
                                     ) if values.height else values
                y = part["log2_tpm_plus1"].to_numpy() if part.height else np.array([])
                color = BINARY_TRAIT_COLORS[trait]
                if len(y) >= 3:
                    ax.boxplot([y], positions=[group_index+offset], widths=.28,
                        patch_artist=True, showfliers=False,
                        boxprops={"facecolor": color, "alpha": .16,
                                  "edgecolor": color, "linewidth": .6},
                        medianprops={"color": "#222222", "linewidth": .85},
                        whiskerprops={"color": color, "linewidth": .6},
                        capprops={"color": color, "linewidth": .6})
                if len(y):
                    ax.scatter(part["plot_x"], y, s=5, color=color, alpha=.58,
                               edgecolors="none", linewidths=0, zorder=3)
                ax.text(group_index+offset, -.105, str(len(y)), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=6.8, color=color)
        ax.text(-.045, -.105, "n", transform=ax.transAxes, ha="right", va="top", fontsize=6.8)
        ax.set(xlim=(-.5, len(groups)-.5), ylim=(-.35, limit), xticks=[])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.tick_params(axis="y", labelsize=7)
        ax.set_yticks([v for v in ax.get_yticks() if 0 <= v <= limit])
        ax.grid(axis="y", color="#d7d7d7", lw=.45, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(titles[row_index], loc="left", fontsize=8.5, pad=9)
        if row_index == len(ranks)-1:
            for position, group in enumerate(groups):
                ax.text(position, -.29, group, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7.5)
            ax.text(.5, -.53, "Group", transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    return fig


def _heatmap_figure(summary: pl.DataFrame) -> Figure:
    ogs = summary.unique("og_rank", maintain_order=True).sort("og_rank")
    context = summary.select("group_index", "group_id", "trait", "n_total").unique().sort(
        "group_index", "trait")
    labels = [(textwrap.fill(" ".join((r["orthogroup_annotation"] or "").split()), 32,
                            break_long_words=False, break_on_hyphens=False) + "\n"
               if r["orthogroup_annotation"] else "") + f"({r['orthogroup']})"
              for r in ogs.iter_rows(named=True)]
    fig = plt.figure()
    canvas = cast(Any, FigureCanvasAgg(fig))
    renderer = canvas.get_renderer()
    probes = [fig.text(0, 0, label, fontsize=7.2) for label in labels]
    header = max(t.get_window_extent(renderer).width for t in probes)/fig.dpi + .4
    column = max(.54, max(t.get_window_extent(renderer).height for t in probes)/fig.dpi + .13)
    for t in probes:
        t.remove()
    probes = [fig.text(0, 0, g, fontsize=8) for g in context["group_id"].unique()]
    group_width = max(t.get_window_extent(renderer).width for t in probes)/fig.dpi
    for t in probes:
        t.remove()
    metadata_width, matrix_width = max(1.6, group_width+1.25), max(4.5, column*ogs.height)
    height = header + max(.7, context.height*.22) + 1.15
    width = metadata_width + matrix_width + .5
    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(1, 2, width_ratios=[metadata_width, matrix_width], wspace=.015,
                           left=.025, right=.99, top=1-header/height, bottom=1.1/height)
    meta, ax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    for pane in [meta, ax]:
        pane.set_ylim(context.height, 0)
        pane.set_yticks([])
        for spine in pane.spines.values():
            spine.set_visible(False)
    meta.set(xlim=(0, metadata_width), xticks=[])
    matrix = summary.sort("group_index", "trait", "og_rank")[
        "median_log2_tpm_plus1"].to_numpy().reshape(context.height, ogs.height)
    cmap = plt.get_cmap("viridis").with_extremes(bad="#dddddd")
    maximum = max(1, math.ceil(np.nanmax(matrix))) if np.isfinite(matrix).any() else 1
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=maximum,
        extent=(0, ogs.height, context.height, 0), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(ogs.height)+.5, labels, rotation=90, ha="left", va="center",
                  rotation_mode="anchor", fontsize=7.2)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False,
                   length=0, pad=7)
    for j in range(1, ogs.height):
        ax.axvline(j, color="white", lw=.25)
    for r in range(1, context.height):
        ax.axhline(r, color="white", lw=.25)
    trait_x, n_x = metadata_width-.76, metadata_width-.3
    for r, row in enumerate(context.iter_rows(named=True)):
        if row["trait"] == 0:
            if row["group_index"] % 2 == 0:
                meta.axhspan(r, r+2, color="#f5f5f5", zorder=0)
            meta.text(.02, r+1, row["group_id"], va="center", fontsize=8)
            if r:
                for pane in [meta, ax]:
                    pane.axhline(r, color="#777777", lw=.65)
        color = BINARY_TRAIT_COLORS[row["trait"]]
        meta.add_patch(Rectangle((trait_x-.15, r+.15), .3, .7, facecolor=color, lw=0))
        meta.text(trait_x, r+.5, str(row["trait"]), ha="center", va="center",
                  color=_cell_text_color(color), fontsize=7.5)
        meta.text(n_x, r+.5, str(row["n_total"]), ha="center", va="center", fontsize=7.5)
        for missing_index in np.flatnonzero(~np.isfinite(matrix[r])):
            ax.text(missing_index+.5, r+.5, "NA", ha="center", va="center",
                    fontsize=5.5, color="#444444")
    for x, label in [(.02, "Group"), (trait_x, "Trait"), (n_x, "n total")]:
        meta.text(x, -.25, label, ha="left" if x == .02 else "center", va="bottom", fontsize=8)
    cax = fig.add_axes((.30, .40/height, .55, .12/height))
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7, length=2, pad=2)
    cbar.set_label("Median log2(TPM + 1)", fontsize=8, labelpad=3)
    fig.text(.03, .33/height, "Gray: no observed TPM", fontsize=7)
    return fig


def write_group_trait_expression(
    run_dir: Path, config: AppConfig, predictions: pl.DataFrame, orthogroups: list[str],
    *, annotations: pl.DataFrame | None = None,
) -> list[str]:
    """Write CV diagnostics when optional OG-level TPM is available in AA mode."""
    if not orthogroups or predictions.is_empty() or not Path(config.data.tpm_path).is_file():
        return []
    if "group_id" not in predictions.columns:
        predictions = with_cv_groups(predictions, metadata_path=Path(config.data.metadata_path),
            species_col=config.data.species_col, group_col=config.split.group_col)
    try:
        values = read_og_expression(config, predictions["species"].to_list(), orthogroups)
    except (CVError, OSError, pl.exceptions.PolarsError) as exc:
        return [f"OG expression figures skipped: {exc}"]
    if values.is_empty():
        return ["OG expression figures skipped: "
                "no selected species/OG coordinates in data.tpm_path."]
    for suffix, population in prediction_figure_populations({"cv": predictions}):
        figure_dir = population_figure_path(run_dir / "cv/figures" / "placeholder", suffix).parent
        table_dir = population_figure_path(run_dir / "cv/tables" / "placeholder", suffix).parent
        figure_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        og_dir = figure_dir / _DISTRIBUTION
        og_dir.mkdir(parents=True, exist_ok=True)
        # Remove retired page SVGs and individual OGs no longer selected.
        for path in [*figure_dir.glob(f"{_DISTRIBUTION}_*.svg"),
                     *og_dir.glob("*.svg"),
                     figure_dir / f"{_DISTRIBUTION}.pdf", figure_dir / f"{_HEATMAP}.pdf"]:
            path.unlink(missing_ok=True)
        if population["cv"].is_empty():
            for path in [og_dir / "no_accepted_species.svg", figure_dir / f"{_HEATMAP}.svg"]:
                write_population_message_svg(path, "No accepted species.")
            for name in ["og_expression_group_trait_summary", "og_expression_species"]:
                (table_dir / f"{name}.tsv").unlink(missing_ok=True)
            continue
        summary, points = expression_tables(values, population["cv"], orthogroups,
                                            annotations=annotations)
        summary.write_csv(table_dir / "og_expression_group_trait_summary.tsv", separator="\t")
        points.write_csv(table_dir / "og_expression_species.tsv", separator="\t")
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none",
                             "pdf.fonttype": 42, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.linewidth": .65}):
            ranks = summary["og_rank"].unique(maintain_order=True).to_list()
            for og in summary.unique("og_rank", maintain_order=True).iter_rows(named=True):
                fig = _distribution_page(summary, points, [og["og_rank"]])
                fig.savefig(og_dir / f"{quote(og['orthogroup'], safe='')}.svg",
                            metadata={"Date": None})
                plt.close(fig)
            with PdfPages(figure_dir / f"{_DISTRIBUTION}.pdf", metadata={"CreationDate": None,
                                                                 "ModDate": None}) as pdf:
                for start in range(0, len(ranks), 4):
                    fig = _distribution_page(summary, points, ranks[start:start+4])
                    pdf.savefig(fig)
                    plt.close(fig)
            fig = _heatmap_figure(summary)
            fig.savefig(figure_dir / f"{_HEATMAP}.svg", metadata={"Date": None})
            fig.savefig(figure_dir / f"{_HEATMAP}.pdf",
                        metadata={"CreationDate": None, "ModDate": None})
            plt.close(fig)
    return []
