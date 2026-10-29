"""Site-level stability and held-out amino-acid composition figures."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from phenoradar.amino_acid import ALPHABET, PREDICTION_BATCH_SIZE, AlignmentStore

if TYPE_CHECKING:
    from phenoradar.amino_acid_run import AAMember


def write_site_scores(destination: Path, members: list[AAMember]) -> None:
    """Export PSS and effect range separately, before any cross-model averaging."""
    from phenoradar.amino_acid_run import _effect_range, _site_coefficients

    records: dict[str, list[dict[str, Any]]] = {"pss": [], "effect_range": []}
    context = ["scope", "sample_set_id", "candidate_index"]
    for member in members:
        betas = _site_coefficients(member)
        scores = {
            "pss": np.array(
                [
                    np.abs(beta[site.known]).sum()
                    for site, beta in zip(member.transform.sites, betas, strict=True)
                ]
            ),
            "effect_range": np.array(
                [
                    _effect_range(site.known, beta)
                    for site, beta in zip(member.transform.sites, betas, strict=True)
                ]
            ),
        }
        for metric, values in scores.items():
            total = float(values.sum())
            for site, beta, value in zip(member.transform.sites, betas, values, strict=True):
                records[metric].append(
                    {
                        "scope": member.scope,
                        "sample_set_id": member.sample_set_id,
                        "candidate_index": member.candidate.candidate_index,
                        "orthogroup": site.orthogroup,
                        "alignment_pos_1based": site.column + 1,
                        "n_known_residue_states": int(np.count_nonzero(site.known)),
                        "n_nonzero_coefficients": int(np.count_nonzero(beta[site.known])),
                        metric: float(value),
                        "normalized_score": float(value / total) if total > 0 else 0.0,
                    }
                )
    destination.mkdir(parents=True, exist_ok=True)
    for metric, rows in records.items():
        frame = pl.DataFrame(
            rows,
            schema={
                "scope": pl.String,
                "sample_set_id": pl.Int64,
                "candidate_index": pl.Int64,
                "orthogroup": pl.String,
                "alignment_pos_1based": pl.Int64,
                "n_known_residue_states": pl.Int64,
                "n_nonzero_coefficients": pl.Int64,
                metric: pl.Float64,
                "normalized_score": pl.Float64,
            },
        ).with_columns(
            pl.when(pl.col(metric) > 0)
            .then(pl.col(metric).rank(method="min", descending=True).over(context))
            .otherwise(None)
            .alias("rank")
        )
        frame.sort(
            [*context, metric, "orthogroup", "alignment_pos_1based"],
            descending=[False, False, False, True, False, False],
        ).write_csv(destination / f"amino_acid_site_{metric}.tsv", separator="\t")


def write_site_stability(destination: Path, members: list[AAMember]) -> None:
    from phenoradar.amino_acid_run import _effect_range, _site_coefficients

    eligible: dict[tuple[str, int], set[str]] = defaultdict(set)
    selected: dict[tuple[str, int], set[str]] = defaultdict(set)
    folds = {member.scope for member in members}
    og_eligible: dict[str, set[str]] = defaultdict(set)
    og_selected: dict[str, set[str]] = defaultdict(set)
    for member in members:
        for site, beta in zip(member.transform.sites, _site_coefficients(member), strict=True):
            key = site.orthogroup, site.column + 1
            if site.active.any():
                eligible[key].add(member.scope)
                og_eligible[site.orthogroup].add(member.scope)
            if _effect_range(site.known, beta) > 1e-12:
                selected[key].add(member.scope)
                og_selected[site.orthogroup].add(member.scope)
    rows = []
    for (og, position), eligible_folds in sorted(eligible.items()):
        selected_folds = selected[og, position]
        rows.append(
            {
                "orthogroup": og,
                "alignment_pos_1based": position,
                "n_outer_folds": len(folds),
                "n_eligible_folds": len(eligible_folds),
                "n_selected_folds": len(selected_folds),
                "selection_frequency": len(selected_folds) / len(folds),
                "selection_frequency_when_eligible": len(selected_folds) / len(eligible_folds),
            }
        )
    pl.DataFrame(rows).write_csv(destination / "amino_acid_site_stability.tsv", separator="\t")
    pl.DataFrame(
        [
            {
                "orthogroup": og,
                "n_outer_folds": len(folds),
                "n_eligible_folds": len(eligible_folds),
                "n_selected_folds": len(og_selected[og]),
                "selection_frequency": len(og_selected[og]) / len(folds),
                "selection_frequency_when_eligible": len(og_selected[og]) / len(eligible_folds),
            }
            for og, eligible_folds in sorted(og_eligible.items())
        ]
    ).write_csv(destination / "amino_acid_og_stability.tsv", separator="\t")


def write_frequency_figure(
    destination: Path,
    store: AlignmentStore,
    manifest: pl.DataFrame,
    members: list[AAMember],
    importance: pl.DataFrame,
    limit: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = (
        importance.sort(["importance_mean", "feature"], descending=[True, False])
        .head(limit)["feature"]
        .to_list()
    )
    species = sorted(manifest.filter(pl.col("pool") == "validation")["species"].unique().to_list())
    index = {sample: i for i, sample in enumerate(species)}
    totals = np.zeros((len(species), len(features)))
    counts = np.zeros_like(totals)
    parsed = [json.loads(feature) for feature in features]
    seen = set()
    for member in members:
        if id(member.transform) in seen:
            continue
        seen.add(id(member.transform))
        fold = member.scope.removeprefix("outer_fold_")
        targets = manifest.filter((pl.col("pool") == "validation") & (pl.col("fold_id") == fold))[
            "species"
        ].to_list()
        site_index = {
            (site.orthogroup, site.column + 1): j for j, site in enumerate(member.transform.sites)
        }
        batch = PREDICTION_BATCH_SIZE
        for start in range(0, len(targets), batch):
            samples = targets[start : start + batch]
            raw, _, _ = member.transform.raw(store, samples)
            rows = np.array([index[sample] for sample in samples])
            for k, (og, position, aa) in enumerate(parsed):
                j = site_index.get((og, position))
                if j is None:
                    continue
                values = raw[:, j, ALPHABET.index(aa)]
                valid = np.isfinite(values)
                totals[rows[valid], k] += values[valid]
                counts[rows[valid], k] += 1
    values = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    figure, axis = plt.subplots(
        figsize=(max(6, len(features) * 0.3), min(16, max(3, len(species) * 0.15)))
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    artist = axis.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
    axis.set_xticks(
        range(len(features)),
        [f"{og}:{pos}:{aa}" for og, pos, aa in parsed],
        rotation=90,
        fontsize=7,
    )
    if len(species) <= 80:
        axis.set_yticks(range(len(species)), species, fontsize=6)
    else:
        axis.set_yticks([])
    axis.set_ylabel("Species (OOF preprocessing)")
    axis.set_title("Amino-acid frequency; grey = unavailable")
    figure.colorbar(artist, ax=axis, label="Residue frequency")
    figure.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / "amino_acid_frequency_heatmap.svg")
    plt.close(figure)
