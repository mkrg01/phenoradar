"""Site-level stability and held-out amino-acid composition figures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from phenoradar.amino_acid import (
    ALPHABET,
    PREDICTION_BATCH_SIZE,
    AATransform,
    AlignmentStore,
)
from phenoradar.amino_acid_audits import ColumnarAudits
from phenoradar.amino_acid_blocks import coefficient_blocks, feature_axes, site_blocks

if TYPE_CHECKING:
    from phenoradar.amino_acid_run import AAMember
    from phenoradar.combined import ExpressionTransform


def _append_frame(path: Path, frame: pl.DataFrame, audit: ColumnarAudits | None = None) -> None:
    if frame.is_empty():
        return
    if audit is not None:
        audit.append(path.name, frame)
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
    *, audit: ColumnarAudits | None = None,
) -> None:
    """Write every eligible site and feature without per-residue Python dictionaries."""
    keys = ["feature", "modality", "orthogroup", "alignment_pos_1based", "residue",
            "aggregation", "training_mean"]
    schema_path = destination / "feature_schema.tsv"
    if expression is not None:
        _append_frame(schema_path, pl.DataFrame([
            {**context, **{key: row.get(key) for key in keys}} for row in expression.schema()
        ]), audit)
    for block in site_blocks(fitted.sites):
        ogs, columns, active = block["orthogroup"], block["column"] + 1, block["active"]
        qc = pl.DataFrame({
            **{key: [value] * len(columns) for key, value in context.items()},
            "orthogroup": ogs, "alignment_pos_1based": columns,
            "sample_fraction": block["sample_fraction"],
            "post_aggregation_sample_fraction": block["post_aggregation_sample_fraction"],
            "minor_samples": block["minor_samples"], "minor_groups": block["minor_groups"],
            "score": block["score"], "n_active_states": np.count_nonzero(active, axis=1),
        })
        _append_frame(destination / "amino_acid_site_qc.tsv", qc, audit)
        rows, states = np.nonzero(active)
        if not len(rows):
            continue
        frame = feature_axes(block, active).with_columns(
            pl.lit("amino_acid").alias("modality"),
            pl.lit(fitted.settings.aggregation).alias("aggregation"),
            pl.Series("training_mean", block["mean"][rows, states]),
            *[pl.lit(value).alias(key) for key, value in context.items()],
        ).select([*context, *keys])
        _append_frame(schema_path, frame, audit)


def _member_context(member: AAMember) -> list[pl.Expr]:
    return [pl.lit(member.scope).alias("scope"),
            pl.lit(member.sample_set_id).alias("sample_set_id"),
            pl.lit(member.candidate.candidate_index).alias("candidate_index")]


def write_site_effects(
    destination: Path, member: AAMember, matrix: Any, shift: np.ndarray | None = None,
    *, audit: ColumnarAudits | None = None,
) -> None:
    """Retain all known states, computing each site's effect and contribution once."""
    alphabet = np.array(list(ALPHABET))
    keys = ["scope", "sample_set_id", "candidate_index", "orthogroup",
            "alignment_pos_1based", "residue", "coefficient", "reference_residue",
            "effect_vs_reference", "site_effect_range", "mean_abs_site_contribution"]
    for block, beta, offset in coefficient_blocks(
        member.transform.sites, member.model.coef_, member.expression_width,
    ):
        known, active = block["known"], block["active"]
        rows, states = np.nonzero(known)
        if not len(rows):
            continue
        reference = np.argmax(block["mean"], axis=1)
        effect = (np.max(np.where(known, beta, -np.inf), axis=1)
                  - np.min(np.where(known, beta, np.inf), axis=1))
        effect[known.sum(axis=1) < 2] = 0
        contribution = np.zeros(len(beta))
        columns = np.r_[0, np.cumsum(active.sum(axis=1))]
        # Keep each nonzero site's original dot-product order exactly.
        for local in np.flatnonzero(np.any(beta, axis=1)):
            left, right = offset + columns[local], offset + columns[local + 1]
            values = matrix[:, left:right]
            if shift is not None:
                values = values.toarray() - shift[left:right]
            contribution[local] = float(np.mean(np.abs(
                values @ beta[local, active[local]]
            )))
        frame = feature_axes(block, known, names=False).with_columns(
            *_member_context(member),
            pl.Series("coefficient", beta[rows, states]),
            pl.Series("reference_residue", alphabet[reference[rows]]),
            pl.Series("effect_vs_reference", beta[rows, states] - beta[rows, reference[rows]]),
            pl.Series("site_effect_range", effect[rows]),
            pl.Series("mean_abs_site_contribution", contribution[rows]),
        ).select(keys)
        _append_frame(destination / "amino_acid_site_effects.tsv", frame, audit)


def write_combined_coefficients(destination: Path, member: AAMember) -> None:
    from phenoradar.cv import _glmnet_candidate_params

    expression = member.expression
    assert expression is not None and expression.scaler is not None
    assert member.aa_standard_deviation is not None
    _, ratio = _glmnet_candidate_params(expression.config, member.candidate.params)
    n = member.expression_width
    coefficient = member.model.coef_[:n]
    frame = pl.DataFrame({
        "feature": expression.features, "modality": ["expression"] * n,
        "orthogroup": expression.orthogroups, "coefficient": coefficient,
        "coefficient_original_unit": coefficient / np.asarray(
            expression.scaler.scale_, dtype=float
        ),
        "coefficient_standardized": coefficient,
        "original_unit": [expression.config.preprocess.expression.expression_transform.method] * n,
        "penalty_factor": np.ones(n),
    })
    keys = ["scope", "sample_set_id", "candidate_index", "feature", "modality", "orthogroup",
            "coefficient", "coefficient_original_unit", "coefficient_standardized",
            "original_unit", "penalty_factor"]
    _append_frame(destination / "combined_coefficients.tsv",
                  frame.with_columns(*_member_context(member)).select(keys))
    for block, beta, offset in coefficient_blocks(member.transform.sites, member.model.coef_, n):
        active = block["active"]
        coefficient = beta[active]
        if not len(coefficient):
            continue
        frame = feature_axes(block, active).with_columns(
            *_member_context(member), pl.lit("amino_acid").alias("modality"),
            pl.Series("coefficient", coefficient),
            pl.Series("coefficient_original_unit", coefficient),
            pl.Series("coefficient_standardized", coefficient * member.aa_standard_deviation[
                offset - n:offset - n + len(coefficient)
            ]),
            pl.lit("residue_frequency").alias("original_unit"),
            pl.lit(ratio).alias("penalty_factor"),
        ).select(keys)
        _append_frame(destination / "combined_coefficients.tsv", frame)


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
    annotations: pl.DataFrame | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phenoradar.figures import _feature_axis_labels

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
        _feature_axis_labels(features, annotations),
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
