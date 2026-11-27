"""Cell-type expression share of selected final-model orthogroups."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from textwrap import fill
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from phenoradar.figures import _save_figure

if TYPE_CHECKING:
    from phenoradar.amino_acid_run import AAMember


N_DRAWS = 50_000


def expression_candidates(coefficients: pl.DataFrame) -> pl.DataFrame:
    """Union the final model axes and keep features active in any member."""
    return (
        coefficients.filter(pl.col("method") == "coef_signed")
        .group_by(orthogroup=pl.col("feature"))
        .agg((pl.col("coefficient") != 0).any().alias("selected"))
        .sort("orthogroup")
    )


def amino_acid_candidates(members: list[AAMember]) -> pl.DataFrame:
    """Count distinct retained sites and mark OGs with any selected site."""
    from phenoradar.amino_acid_blocks import coefficient_blocks

    retained: dict[str, set[int]] = defaultdict(set)
    selected: set[str] = set()
    for member in members:
        for block, beta, _ in coefficient_blocks(
            member.transform.sites, member.model.coef_, member.expression_width,
        ):
            valid = block["active"].any(axis=1)
            for og, column in zip(block["orthogroup"][valid], block["column"][valid], strict=True):
                retained[og].add(column)
            selected.update(block["orthogroup"][np.any(beta != 0, axis=1)])
    return pl.DataFrame(
        [(og, len(retained[og]), og in selected) for og in sorted(retained)],
        schema={"orthogroup": pl.String, "candidate_sites": pl.UInt32, "selected": pl.Boolean},
        orient="row",
    )


def _null_means(
    shares: NDArray[np.float64], size: int, weights: NDArray[np.float64] | None,
    seed: int, n_draws: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng([seed % 2**64, int(weights is not None)])
    n_candidates, n_types = shares.shape
    null = np.empty((n_draws, n_types))
    batch_size = max(1, min(256, 2_000_000 // max(n_candidates, size * n_types)))
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        shape = (stop - start, n_candidates)
        # Exponential clocks implement successive sampling proportional to remaining weights.
        keys = (
            rng.random(shape) if weights is None else rng.standard_exponential(shape) / weights
        )
        indices = np.argpartition(keys, size - 1, axis=1)[:, :size]
        null[start:stop] = shares[indices].mean(axis=1)
    return null


def _statistics(
    shares: NDArray[np.float64], selected: NDArray[np.bool_], cell_types: list[str],
    weights: NDArray[np.float64] | None, seed: int, n_draws: int,
) -> tuple[pl.DataFrame, NDArray[np.float64]]:
    size = int(selected.sum())
    observed = shares[selected].mean(axis=0)
    null = _null_means(shares, size, weights, seed, n_draws)
    expected = shares.mean(axis=0) if weights is None else null.mean(axis=0)
    exceedances = (null >= observed - 1e-10).sum(axis=0)
    p = (exceedances + 1) / (n_draws + 1)
    order = np.argsort(p, kind="stable")
    holm = np.empty_like(p)
    holm[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    low, high = np.quantile(null, (0.025, 0.975), axis=0)
    return pl.DataFrame({
        "cell_type": cell_types,
        "n_selected_analyzed": size,
        "n_eligible_analyzed": len(shares),
        "n_draws": n_draws,
        "observed_pct": observed,
        "expected_pct": expected,
        "difference_pp": observed - expected,
        "null_low_pct": low,
        "null_high_pct": high,
        "n_null_ge_observed": exceedances,
        "p_enrichment": p,
        "p_holm": holm,
    }), null


def _plot_null(null: NDArray[np.float64], table: pl.DataFrame, path: Path) -> None:
    observed = table["observed_pct"].to_numpy()
    low = np.floor(min(null.min(), observed.min()))
    high = np.ceil(max(null.max(), observed.max()))
    if low == high:
        low, high = low - 0.5, high + 0.5
    bins = max(5, min(200, int(5 * (high - low))))
    edges = np.linspace(low, high, bins + 1)
    density = [np.histogram(values, bins=edges, density=True)[0] for values in null.T]
    y_max = max(values.max() for values in density) * 1.46
    n_cols = min(table.height, 3)
    n_rows = (table.height + n_cols - 1) // n_cols
    width, height = 2.55 * n_cols + 0.8, 1.68 * n_rows + 0.64
    fill_color, observed_color = "#CFD6DC", "#11678D"
    with matplotlib.rc_context({"font.size": 8.5, "axes.linewidth": 0.65}):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(width, height), sharex=True, sharey=True, squeeze=False,
        )
        left, right = 0.76 / width, 1 - 0.16 / width
        bottom, top = 0.58 / height, 1 - 0.55 / height
        fig.subplots_adjust(
            left=left, right=right, bottom=bottom, top=top, wspace=0.16, hspace=0.39,
        )
        fig.legend(
            handles=[
                Patch(facecolor=fill_color, edgecolor="none", label="Null distribution"),
                Line2D([], [], color=observed_color, lw=1.55, label="Observed"),
            ],
            loc="upper right", bbox_to_anchor=(right, 1 - 0.055 / height),
            frameon=False, ncols=2, fontsize=8.25,
        )
        for i, row in enumerate(table.iter_rows(named=True)):
            axis = axes.flat[i]
            axis.stairs(density[i], edges, baseline=0, fill=True, color=fill_color, lw=0)
            axis.axvline(row["observed_pct"], ymax=0.72, color=observed_color, lw=1.55)
            name = row["cell_type"].replace("_", " ")
            axis.set_title(
                fill(name[:1].upper() + name[1:], 26), loc="left", fontsize=9.5, weight="semibold",
            )
            axis.text(
                0.963, 0.951, f"$p$ = {row['p_enrichment']:.3g}\n"
                f"$p_{{\\mathrm{{Holm}}}}$ = {row['p_holm']:.3g}",
                ha="right", va="top", transform=axis.transAxes, fontsize=8, linespacing=1.38,
            )
            axis.set(xlim=(low, high), ylim=(0, y_max))
            axis.spines[["top", "right"]].set_visible(False)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            axis.tick_params(length=2.5, pad=3, labelsize=8, labelbottom=i + n_cols >= table.height)
        for axis in axes.flat[table.height:]:
            axis.set_visible(False)
        fig.text(0.28 / width, (bottom + top) / 2, "Density", rotation=90, ha="center", va="center")
        fig.text((left + right) / 2, 0.185 / height, "Mean share of OG expression (%)", ha="center")
        _save_figure(fig, path, layout=False)


def write_cell_type_enrichment(
    run_dir: Path, candidates: pl.DataFrame, cell_types: list[str],
    cpm: NDArray[np.float64], *, seed: int,
) -> list[str]:
    """Compare selected OGs with the retained OGs having atlas expression."""
    if candidates.is_empty():
        return ["Cell-type enrichment: no final-model OGs with signed coefficients."]
    valid_types: NDArray[np.bool_] = np.asarray(np.isfinite(cpm).any(axis=0))
    if valid_types.sum() < 2:
        return ["Cell-type enrichment: at least two cell types with atlas data are required."]
    valid_cpm = cpm[:, valid_types]
    totals = valid_cpm.sum(axis=1)
    eligible = totals > 0
    selected = candidates["selected"].to_numpy()
    n_selected = int(selected[eligible].sum())
    if not 0 < n_selected < eligible.sum():
        return ["Cell-type enrichment: requires selected and unselected OGs with atlas expression."]
    warnings = []
    if n_selected < selected.sum():
        warnings.append(f"Cell-type enrichment: {n_selected} of {selected.sum()} selected OGs "
                        "have atlas expression.")

    shares = 100 * valid_cpm[eligible] / totals[eligible, None]
    weights = (
        candidates["candidate_sites"].to_numpy()[eligible].astype(float)
        if "candidate_sites" in candidates.columns else None
    )
    labels = [label for label, use in zip(cell_types, valid_types, strict=True) if use]
    table, null = _statistics(shares, selected[eligible], labels, weights, seed, N_DRAWS)
    table.write_csv(run_dir / "model/tables/cell_type_enrichment.tsv", separator="\t")
    _plot_null(null, table, run_dir / "model/figures/cell_type_enrichment_null.svg")
    return warnings
