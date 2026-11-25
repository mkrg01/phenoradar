"""Equal-species residue compositions for the PSS-ranked sites, by group and trait."""

from __future__ import annotations

import importlib
import textwrap
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from phenoradar.amino_acid import ALPHABET, PREDICTION_BATCH_SIZE, AlignmentStore
from phenoradar.colors import BINARY_TRAIT_COLORS
from phenoradar.figure_population import (
    population_figure_path,
    prediction_figure_populations,
    write_population_message_svg,
)
from phenoradar.figures import _orthogroup_annotation_lookup
from phenoradar.residue_colors import ZAPPO_GROUPS, residue_color
from phenoradar.tree_prediction import _PROBABILITY_CMAP, _cell_text_color

_STEM = "amino_acid_group_trait_composition"


def _group_order(predictions: pl.DataFrame, tree_path: Path | None) -> list[str]:
    groups = sorted(predictions["group_id"].unique().to_list())
    if tree_path is None:
        return groups
    toytree = importlib.import_module("toytree")

    tree = toytree.tree(str(tree_path))
    lookup = dict(predictions.select("species", "group_id").iter_rows())
    present = [name for name in tree.get_tip_labels() if name in lookup]
    if len(present) < 2:
        return groups
    if len(present) < tree.ntips:
        tree = tree.mod.prune(*present)
    ordered = list(dict.fromkeys(lookup[name] for name in reversed(
        tree.ladderize().get_tip_labels())))
    return ordered + [group for group in groups if group not in ordered]


def build_group_trait_composition(
    store: AlignmentStore, sites: pl.DataFrame, predictions: pl.DataFrame,
    *, aggregation: str, annotations: pl.DataFrame | None = None,
    tree_path: Path | None = None,
) -> pl.DataFrame:
    """Aggregate raw observations in species batches; missing species get no weight.

    Expression weights apply only within each species. No centering, imputation,
    coefficients or predicted labels enter the residue proportions.
    """
    if sites.is_empty() or predictions.is_empty():
        return pl.DataFrame()
    if predictions["species"].n_unique() != predictions.height:
        raise ValueError("Residue composition requires one OOF row per species")
    if "group_id" not in predictions.columns or predictions["group_id"].null_count():
        raise ValueError("Residue composition requires a split group for every species")
    if aggregation not in {"aa_frequency", "tpm_weighted_frequency", "best_covered"}:
        raise ValueError(f"Unknown amino-acid aggregation: {aggregation}")
    predictions = predictions.with_columns(pl.col("group_id").cast(pl.String))
    rows = []
    row_species = []
    for group in _group_order(predictions, tree_path):
        for trait in [0, 1]:
            subset = predictions.filter((pl.col("group_id") == group) & (pl.col("label") == trait))
            if subset.is_empty():
                continue
            rows.append({"group_id": group, "trait": trait, "n_species": subset.height,
                         "mean_oof_probability": subset["prob"].mean()})
            row_species.append(subset["species"].sort().to_list())
    if not rows:
        return pl.DataFrame()
    totals = np.zeros((len(rows), sites.height, len(ALPHABET)), dtype=np.float64)
    observed = np.zeros((len(rows), sites.height), dtype=np.int64)
    ranked = sites.with_row_index("feature_rank", offset=1)
    for name in sites["orthogroup"].unique(maintain_order=True):
        og = store.ogs[name]
        selected = ranked.filter(pl.col("orthogroup") == name)
        indices = selected["feature_rank"].to_numpy() - 1
        columns = selected["alignment_pos_1based"].to_numpy() - 1
        if np.any((columns < 0) | (columns >= og.width)):
            raise ValueError(f"Site outside input alignment: {name}")
        for r, species in enumerate(row_species):
            for start in range(0, len(species), PREDICTION_BATCH_SIZE):
                frequencies, _, _ = store.frequencies(
                    og, species[start:start + PREDICTION_BATCH_SIZE], columns,
                    weighted=aggregation == "tpm_weighted_frequency",
                    representative=aggregation == "best_covered",
                    scoring_columns=np.arange(og.width) if aggregation == "best_covered" else None,
                )
                valid = np.isfinite(frequencies).all(axis=2)
                totals[r, indices] += np.where(valid[:, :, None], frequencies, 0).sum(
                    axis=0, dtype=np.float64)
                observed[r, indices] += valid.sum(axis=0)
    means = np.divide(totals, observed[:, :, None], out=np.full_like(totals, np.nan),
                      where=observed[:, :, None] > 0)
    lookup = _orthogroup_annotation_lookup(annotations)
    return pl.DataFrame([
        {**row, **site, "orthogroup_annotation": lookup.get(site["orthogroup"]),
         "n_observed_species": int(observed[r, j]),
         "observed_species_fraction": float(observed[r, j] / row["n_species"]),
         **{aa: float(means[r, j, a]) for a, aa in enumerate(ALPHABET)}}
        for r, row in enumerate(rows) for j, site in enumerate(ranked.iter_rows(named=True))
    ], schema_overrides={"orthogroup_annotation": pl.String})


def _site_label(site: dict[str, Any]) -> str:
    annotation = " ".join((site["orthogroup_annotation"] or "").split())
    name = textwrap.fill(annotation, width=32, break_long_words=False, break_on_hyphens=False)
    address = f"({site['orthogroup']}):{site['alignment_pos_1based']}"
    return f"{name}\n{address}" if name else address


def _composition_figure(table: pl.DataFrame) -> Figure:
    sites = table.unique("feature_rank", maintain_order=True).sort("feature_rank")
    context = table.select("group_id", "trait", "n_species", "mean_oof_probability").unique(
        maintain_order=True)
    labels = [_site_label(site) for site in sites.iter_rows(named=True)]
    # Measure full labels before reserving the header; long names are never truncated.
    fig = plt.figure()
    canvas = cast(Any, FigureCanvasAgg(fig))
    renderer = canvas.get_renderer()
    probes = [fig.text(0, 0, label, fontsize=7.4) for label in labels]
    header = max(probe.get_window_extent(renderer).width for probe in probes) / fig.dpi + .3
    column = max(.525, max(probe.get_window_extent(renderer).height for probe in probes)
                 / fig.dpi + .12)
    for probe in probes:
        probe.remove()
    probes = [fig.text(0, 0, group, fontsize=8) for group in context["group_id"].unique()]
    group_width = max(probe.get_window_extent(renderer).width for probe in probes) / fig.dpi
    for probe in probes:
        probe.remove()
    meta_width = max(1.85, group_width + 1.55)
    matrix_width = max(column * sites.height, 4.5)
    matrix_height = max(.65, context.height * .2215)
    width, height = meta_width + matrix_width + .55, header + matrix_height + 1.2
    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(1, 2, width_ratios=[meta_width, matrix_width], wspace=.015,
                           left=.025, right=.99, top=1 - header / height, bottom=1.1 / height)
    meta, ax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    for pane in [meta, ax]:
        pane.set_ylim(context.height, 0)
        pane.set_yticks([])
        for spine in pane.spines.values():
            spine.set_visible(False)
    meta.set_xlim(0, meta_width)
    meta.set_xticks([])
    ax.set_xlim(0, sites.height)
    ax.set_xticks(np.arange(sites.height) + .5, labels, rotation=90, ha="left", va="center",
                  rotation_mode="anchor", fontsize=7.4)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False,
                   length=0, pad=7)
    trait_x, n_x, prob_x = meta_width - 1.20, meta_width - .81, meta_width - .30
    for title, x in [("Group", .03), ("Trait", trait_x), ("n", n_x), ("Mean OOF p", prob_x)]:
        meta.text(x, -.3, title, ha="left" if title == "Group" else "center", va="bottom",
                  fontsize=8)
    group_ids = context["group_id"].to_list()
    for g, group in enumerate(dict.fromkeys(group_ids)):
        indices = [r for r, value in enumerate(group_ids) if value == group]
        start, end = indices[0], indices[-1] + 1
        if g % 2 == 0:
            for pane in [meta, ax]:
                pane.axhspan(start, end, color="#f7f7f7", zorder=0)
        meta.text(.03, (start + end) / 2, group, va="center", ha="left", fontsize=8)
        if start:
            for pane in [meta, ax]:
                pane.axhline(start, color="#909090", linewidth=.65)
    for r, row in enumerate(context.iter_rows(named=True)):
        color = BINARY_TRAIT_COLORS[row["trait"]]
        meta.add_patch(Rectangle((trait_x - .15, r + .14), .30, .61, color=color, linewidth=0))
        meta.text(trait_x, r + .445, str(row["trait"]), ha="center", va="center", fontsize=8,
                  color=_cell_text_color(color))
        meta.text(n_x, r + .445, str(row["n_species"]), ha="center", va="center", fontsize=8)
        p = row["mean_oof_probability"]
        color = to_hex(_PROBABILITY_CMAP(p))
        meta.add_patch(Rectangle((prob_x - .20, r + .14), .40, .61, color=color, linewidth=0))
        meta.text(prob_x, r + .445, f"{p:.2f}", ha="center", va="center", fontsize=7,
                  color=_cell_text_color(color))
        cells = table.filter((pl.col("group_id") == row["group_id"]) &
                             (pl.col("trait") == row["trait"])).sort("feature_rank")
        for j, cell in enumerate(cells.iter_rows(named=True)):
            x = j + .065
            if not cell["n_observed_species"]:
                ax.add_patch(Rectangle((x, r + .14), .87, .61, color="#e2e2e2", linewidth=0))
                ax.text(j + .5, r + .445, "NA", ha="center", va="center", fontsize=6)
            else:
                for aa in ALPHABET:
                    fraction = cell[aa]
                    if fraction <= 0:
                        continue
                    color = residue_color(aa)
                    ax.add_patch(Rectangle((x, r + .14), .87 * fraction, .61, facecolor=color,
                                           edgecolor="white", linewidth=.25))
                    if fraction >= .14:
                        ax.text(x + .87 * fraction / 2, r + .445, aa, ha="center", va="center",
                                fontsize=7, color=_cell_text_color(color))
                    x += .87 * fraction
            ax.add_patch(Rectangle((j + .065, r + .83), .87, .07, color="#dedede", linewidth=0))
            ax.add_patch(Rectangle((j + .065, r + .83), .87 * cell["observed_species_fraction"],
                                   .07, color="#737373", linewidth=0))
    handles = [Patch(facecolor=color, edgecolor="#999999", linewidth=.3, label=" ".join(residues))
               for residues, _, color in ZAPPO_GROUPS]
    fig.legend(handles=handles, ncol=7 if width >= 10 else 4, loc="lower center",
               bbox_to_anchor=(.60 if width >= 10 else .69, .35 / height),
               frameon=False, fontsize=7.4, handlelength=.85, handletextpad=.35, columnspacing=.8)
    availability = fig.add_axes((.025, .28 / height, 2.55 / width, .55 / height))
    availability.set_xlim(0, 3)
    availability.set_ylim(0, 1)
    availability.axis("off")
    availability.text(0, 1, "Observed species", ha="left", va="top", fontsize=7)
    for i, fraction in enumerate([0, .5, 1]):
        for length, color in [(.72, "#dedede"), (.72 * fraction, "#737373")]:
            availability.add_patch(Rectangle((i + .03, .35), length, .09, color=color, linewidth=0))
        availability.text(i + .39, .25, f"{fraction:.0%}", ha="center", va="top", fontsize=6.5)
    return fig


def write_group_trait_composition(
    run_dir: Path, store: AlignmentStore, sites: pl.DataFrame, predictions: pl.DataFrame,
    *, aggregation: str, annotations: pl.DataFrame | None = None,
    tree_path: Path | None = None,
) -> list[str]:
    """Write only the composition summary, including existing accepted-only populations."""
    for suffix, population in prediction_figure_populations({"oof": predictions}):
        table = build_group_trait_composition(
            store, sites, population["oof"], aggregation=aggregation,
            annotations=annotations, tree_path=tree_path,
        )
        figure_path = population_figure_path(run_dir / "cv/figures" / f"{_STEM}.svg", suffix)
        table_path = population_figure_path(run_dir / "cv/tables" / f"{_STEM}.tsv", suffix)
        if table.is_empty():
            table_path.unlink(missing_ok=True)
            write_population_message_svg(figure_path, "No observed groups or positive-PSS sites.")
            continue
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(table_path, separator="\t")
        fig = _composition_figure(table)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        # Explicit axes use measured full headers; automatic layout would move the matrix cells.
        fig.savefig(figure_path, format="svg", metadata={"Date": None})
        plt.close(fig)
    return []
