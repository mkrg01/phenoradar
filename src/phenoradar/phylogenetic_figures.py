"""Observed and imputed traits alongside expression predictions on the same tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from phenoradar.figures import _save_svg_figure
from phenoradar.phylogenetic_imputation import PhylogeneticImputationArtifacts


def _comparison_scatter(artifacts: PhylogeneticImputationArtifacts, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
    rows = [
        row
        for row in artifacts.comparison.iter_rows(named=True)
        if row["phylo_prob"] is not None and row["prob"] is not None
    ]
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linewidth=0.8, zorder=0)
    for abstained, marker, label in [(False, "o", "Accepted"), (True, "x", "Abstained")]:
        selected = [row for row in rows if (row.get("decision_status") == "abstained") == abstained]
        if selected:
            ax.scatter(
                [row["phylo_prob"] for row in selected],
                [row["prob"] for row in selected],
                c=[row["prob_difference"] for row in selected],
                vmin=-1,
                vmax=1,
                cmap="RdBu_r",
                marker=marker,
                s=30,
                label=label,
                zorder=2,
            )
    for index, row in enumerate(rows[:10]):
        horizontal = -5 if row["phylo_prob"] > 0.65 else 5
        vertical = -(10 + (index % 3) * 9) if row["prob"] > 0.85 else 5 + (index % 3) * 9
        ax.annotate(
            row["species"],
            (row["phylo_prob"], row["prob"]),
            xytext=(horizontal, vertical),
            textcoords="offset points",
            ha="right" if horizontal < 0 else "left",
            fontsize=7,
        )
    if not rows:
        ax.text(0.5, 0.5, "No comparable phylogenetic estimates", ha="center", va="center")
    else:
        ax.legend(loc="lower right", fontsize=8)
    ax.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xlabel="Phylogenetic P(trait = 1)",
        ylabel="Expression P(trait = 1)",
    )
    fig.colorbar(
        ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r"),
        ax=ax,
        label="Expression minus phylogenetic probability",
        shrink=0.7,
    )
    _save_svg_figure(fig, path)


def _tree_coordinates(tree: Any) -> tuple[list[Any], dict[Any, float], dict[Any, float]]:
    tips = list(tree.leaves())
    y = {node: float(index) for index, node in enumerate(tips)}
    for node in tree.traverse("postorder"):
        if not node.is_leaf:
            y[node] = float(np.mean([y[child] for child in node.children]))
    x = {tree: 0.0}
    for node in tree.traverse("preorder"):
        if not node.is_root:
            x[node] = x[node.up] + node.dist
    return tips, x, y


def _tree_comparison(
    artifacts: PhylogeneticImputationArtifacts,
    path: Path,
    branch_length_mode: str,
) -> None:
    tree = artifacts.tree.copy()
    by_species = {row["species"]: row for row in artifacts.annotation.iter_rows(named=True)}
    retained = [node.name for node in tree.leaves() if node.name in by_species]
    if not retained:
        return
    # Prune for display only, retaining the fitted lengths and original node IDs.
    tree.prune(retained, preserve_branch_length=True)
    tips, x, y = _tree_coordinates(tree)
    names_width = max(1.8, min(7, max(len(str(tip.name)) for tip in tips) * 0.065))
    fig, axes = plt.subplots(
        1,
        7,
        figsize=(8.5 + names_width, max(3.8, len(tips) * 0.19 + 1.8)),
        gridspec_kw={"width_ratios": [3.2, names_width, 0.75, 0.85, 0.85, 0.85, 1.8]},
        layout="constrained",
    )
    tree_ax, names_ax, *tracks = axes
    posterior = {row["branch_id"]: row["p_1"] for row in artifacts.nodes.iter_rows(named=True)}
    for node in tree.traverse():
        if not node.is_root:
            tree_ax.plot([x[node.up], x[node]], [y[node], y[node]], color="#777777", lw=0.7)
        if not node.is_leaf:
            children_y = [y[child] for child in node.children]
            tree_ax.plot(
                [x[node], x[node]], [min(children_y), max(children_y)], color="#777777", lw=0.7
            )
            probability = posterior.get(node.props["phylo_id"])
            if probability is not None:
                tree_ax.scatter(
                    [x[node]],
                    [y[node]],
                    c=[probability],
                    vmin=0,
                    vmax=1,
                    cmap="viridis",
                    s=14,
                    zorder=3,
                )
    tree_ax.set_xlabel("Branch count" if branch_length_mode == "unit" else "Input branch length")
    depth = max(max(x.values()), 0.1)
    tree_ax.set_xlim(-0.02 * depth, depth * 1.04)
    tree_ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in ["top", "left", "right"]:
        tree_ax.spines[spine].set_visible(False)
    names_ax.set_title("Species", fontsize=9)
    for tip in tips:
        row = by_species[tip.name]
        names_ax.text(
            0,
            y[tip],
            tip.name,
            va="center",
            fontsize=8,
            fontweight="bold" if row["is_prediction_target"] else "normal",
        )
    names_ax.set_axis_off()
    for ax, column, title, domain, cmap_name in zip(
        tracks[:4],
        ["observed_trait", "prob", "phylo_prob", "prob_difference"],
        ["Observed", "Expression", "Phylogenetic", "Difference"],
        [(0, 1), (0, 1), (0, 1), (-1, 1)],
        ["viridis", "viridis", "viridis", "RdBu_r"],
        strict=True,
    ):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#eeeeee")
        values = np.array(
            [
                np.nan if by_species[tip.name].get(column) is None else by_species[tip.name][column]
                for tip in tips
            ],
            dtype=float,
        ).reshape(-1, 1)
        ax.imshow(
            values, vmin=domain[0], vmax=domain[1], cmap=cmap, aspect="auto", interpolation="none"
        )
        for index, value in enumerate(values[:, 0]):
            text = (
                "NA"
                if np.isnan(value)
                else (str(int(value)) if column == "observed_trait" else f"{value:.2f}")
            )
            rgba = cmap(0 if np.isnan(value) else (value - domain[0]) / (domain[1] - domain[0]))
            luminance = sum(
                weight * channel
                for weight, channel in zip(
                    [0.2126, 0.7152, 0.0722],
                    rgba[:3],
                    strict=True,
                )
            )
            color = "black" if np.isnan(value) or luminance > 0.48 else "white"
            ax.text(0, index, text, ha="center", va="center", color=color, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    status_ax = tracks[4]
    status_ax.set_title("Status", fontsize=8)
    status_ax.set_axis_off()
    for tip in tips:
        row = by_species[tip.name]
        status = "reference" if not row["is_prediction_target"] else row["phylo_status"]
        if row.get("decision_status") == "abstained":
            status += "; abstained"
        status_ax.text(0, y[tip], status, fontsize=7, va="center")
    for ax in axes:
        ax.set_ylim(len(tips) - 0.5, -0.5)
    fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
        ax=list(axes[:5]),
        location="bottom",
        shrink=0.4,
        label="Trait 1 / probability (including ancestral nodes)",
        pad=0.04,
    )
    fig.colorbar(
        ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r"),
        ax=list(axes[5:]),
        location="bottom",
        shrink=0.7,
        label="Expression minus phylogenetic",
        pad=0.04,
    )
    tree_ax.legend(
        handles=[Patch(facecolor="#eeeeee", label="Missing / unavailable")],
        loc="upper left",
        bbox_to_anchor=(0, 1.10),
        frameon=False,
        fontsize=7,
    )
    _save_svg_figure(fig, path)


def write_phylogenetic_figures(
    artifacts: PhylogeneticImputationArtifacts,
    *,
    run_dir: Path,
    branch_length_mode: str,
) -> None:
    if artifacts.comparison.height == 0:
        return
    directory = run_dir / "inference" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    _comparison_scatter(artifacts, directory / "phylogenetic_comparison.svg")
    if artifacts.tree is not None:
        _tree_comparison(
            artifacts, directory / "tree_phylogenetic_imputation.svg", branch_length_mode
        )
