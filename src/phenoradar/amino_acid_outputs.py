"""Site-level stability and held-out amino-acid composition figures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from phenoradar.amino_acid import (
    ALPHABET,
    PREDICTION_BATCH_SIZE,
    AASiteTable,
    AATransform,
    AlignmentStore,
)

if TYPE_CHECKING:
    from phenoradar.amino_acid_run import AAMember
    from phenoradar.combined import ExpressionTransform


def _audit_blocks(fitted: AATransform) -> Iterator[dict[str, Any]]:
    fields = ["column", "sample_fraction", "minor_samples", "minor_groups", "score",
              "mean", "active", "post_aggregation_sample_fraction"]
    sites = fitted.sites
    for start in range(0, len(sites), 512):
        stop = start + 512
        if isinstance(sites, AASiteTable):
            yield {"orthogroup": sites.orthogroups[start:stop].tolist(),
                   **{name: getattr(sites, name)[start:stop] for name in fields}}
        else:
            block = list(sites[start:stop])
            yield {"orthogroup": [s.orthogroup for s in block],
                   **{name: np.array([getattr(s, name) for s in block]) for name in fields}}


def _append_frame(path: Path, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        frame.write_csv(handle, separator="\t", include_header=header, line_terminator="\r\n")


def write_preprocessing_audits(
    destination: Path,
    fitted: AATransform,
    context: dict[str, Any],
    expression: ExpressionTransform | None = None,
) -> None:
    """Write every eligible site and feature without per-residue Python dictionaries."""
    keys = ["feature", "modality", "orthogroup", "alignment_pos_1based", "residue",
            "aggregation", "training_mean"]
    schema_path = destination / "feature_schema.tsv"
    if expression is not None:
        _append_frame(schema_path, pl.DataFrame([
            {**context, **{key: row.get(key) for key in keys}} for row in expression.schema()
        ]))
    alphabet = np.array(list(ALPHABET))
    for block in _audit_blocks(fitted):
        ogs, columns, active = block["orthogroup"], block["column"] + 1, block["active"]
        qc = pl.DataFrame({
            **{key: [value] * len(columns) for key, value in context.items()},
            "orthogroup": ogs, "alignment_pos_1based": columns,
            "sample_fraction": block["sample_fraction"],
            "post_aggregation_sample_fraction": block["post_aggregation_sample_fraction"],
            "minor_samples": block["minor_samples"], "minor_groups": block["minor_groups"],
            "score": block["score"], "n_active_states": np.count_nonzero(active, axis=1),
        })
        _append_frame(destination / "amino_acid_site_qc.tsv", qc)
        rows, states = np.nonzero(active)
        if not len(rows):
            continue
        # Encode each OG once per block; residue symbols are canonical ASCII.
        prefixes = {og: "[" + json.dumps(og) + "," for og in set(ogs)}
        frame = pl.DataFrame({
            "_prefix": [prefixes[ogs[i]] for i in rows],
            "modality": ["amino_acid"] * len(rows),
            "orthogroup": [ogs[i] for i in rows], "alignment_pos_1based": columns[rows],
            "residue": alphabet[states], "aggregation": [fitted.settings.aggregation] * len(rows),
            "training_mean": block["mean"][rows, states],
        }).with_columns(
            pl.concat_str([pl.col("_prefix"), pl.col("alignment_pos_1based").cast(pl.String),
                           pl.lit(',"'), pl.col("residue"), pl.lit('"]')]).alias("feature"),
            *[pl.lit(value).alias(key) for key, value in context.items()],
        ).select([*context, *keys])
        _append_frame(schema_path, frame)


def write_site_scores(destination: Path, members: list[AAMember]) -> None:
    """Write both scores without accumulating all site dictionaries in memory."""
    from phenoradar.amino_acid_reporting import AAReportWriter

    with AAReportWriter(destination) as writer:
        writer.add(members, features=False)
        writer.write_site_scores()


def write_site_stability(destination: Path, members: list[AAMember]) -> None:
    from phenoradar.amino_acid_reporting import AAReportWriter

    with AAReportWriter(destination) as writer:
        writer.add(members, features=False)
        writer.write_site_stability()


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
        requested = {(og, position) for og, position, _ in parsed}
        display = replace(
            member.transform,
            sites=[
                site for site in member.transform.sites
                if (site.orthogroup, site.column + 1) in requested
            ],
        )
        if display.settings.aggregation == "best_covered":
            display.representative_columns = {
                site.orthogroup: member.transform.scoring_columns(site.orthogroup)
                for site in display.sites
            }
        site_index = {
            (site.orthogroup, site.column + 1): j for j, site in enumerate(display.sites)
        }
        batch = PREDICTION_BATCH_SIZE
        for start in range(0, len(targets), batch):
            samples = targets[start : start + batch]
            raw, _, _ = display.raw(store, samples)
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

    table = pl.DataFrame({
        "species": np.repeat(species, len(features)),
        "feature": features * len(species), "frequency": values.ravel(),
    })
    table.write_csv(destination.parent / "tables/top_amino_acid_frequencies.tsv", separator="\t")
