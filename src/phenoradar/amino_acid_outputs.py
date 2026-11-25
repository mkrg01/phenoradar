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
    *, audit: ColumnarAudits | None = None, compact: bool = False,
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
        if compact:
            # Selected sites' means and active states live beside their effects.
            continue
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


def write_preprocessing_summary(
    destination: Path, fitted: AATransform, context: dict[str, Any],
    *, audit: ColumnarAudits | None = None,
) -> None:
    """Summarize all retained sites before compact reports select their details."""
    counts = {"n_retained_sites": 0, "n_active_sites": 0, "n_active_features": 0,
              "n_known_states": 0}
    columns = ["sample_fraction", "post_aggregation_sample_fraction", "minor_samples",
               "minor_groups", "score"]
    totals = {key: [0.0, float("inf"), float("-inf")] for key in columns}
    for block in site_blocks(fitted.sites):
        counts["n_retained_sites"] += len(block["column"])
        counts["n_active_sites"] += int(np.any(block["active"], axis=1).sum())
        counts["n_active_features"] += int(np.count_nonzero(block["active"]))
        counts["n_known_states"] += int(np.count_nonzero(block["known"]))
        for key in columns:
            values = block[key]
            totals[key][0] += float(np.sum(values, dtype=np.float64))
            totals[key][1] = min(totals[key][1], float(np.min(values)))
            totals[key][2] = max(totals[key][2], float(np.max(values)))
    row: dict[str, Any] = {**context, **counts}
    n = counts["n_retained_sites"]
    for key, (total, minimum, maximum) in totals.items():
        row.update({f"{key}_mean": total / n if n else None,
                    f"{key}_min": minimum if n else None,
                    f"{key}_max": maximum if n else None})
    _append_frame(destination / "amino_acid_site_qc_summary.tsv", pl.DataFrame([row]), audit)


def _member_context(member: AAMember) -> list[pl.Expr]:
    return [pl.lit(member.scope).alias("scope"),
            pl.lit(member.sample_set_id).alias("sample_set_id"),
            pl.lit(member.candidate.candidate_index).alias("candidate_index")]


def write_site_effects(
    destination: Path, member: AAMember, matrix: Any, shift: np.ndarray | None = None,
    *, audit: ColumnarAudits | None = None, compact: bool = False,
) -> None:
    """Retain all known states, computing each site's effect and contribution once."""
    alphabet = np.array(list(ALPHABET))
    keys = ["scope", "sample_set_id", "candidate_index", "orthogroup",
            "alignment_pos_1based", "residue", "coefficient", "reference_residue",
            "effect_vs_reference", "mean_abs_site_contribution"]
    for block, beta, offset in coefficient_blocks(
        member.transform.sites, member.model.coef_, member.expression_width,
    ):
        known, active = block["known"], block["active"]
        retained = known & np.any(beta != 0, axis=1)[:, None] if compact else known
        rows, states = np.nonzero(retained)
        if not len(rows):
            continue
        reference = np.argmax(block["mean"], axis=1)
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
        frame = feature_axes(block, retained, names=False).with_columns(
            *_member_context(member),
            pl.Series("coefficient", beta[rows, states]),
            pl.Series("reference_residue", alphabet[reference[rows]]),
            pl.Series("effect_vs_reference", beta[rows, states] - beta[rows, reference[rows]]),
            pl.Series("mean_abs_site_contribution", contribution[rows]),
        ).select(keys)
        if compact:
            frame = frame.with_columns(
                pl.Series("training_mean", block["mean"][rows, states]),
                pl.Series("active", active[rows, states]),
                pl.lit(member.transform.settings.aggregation).alias("aggregation"),
            )
        _append_frame(destination / "amino_acid_site_effects.tsv", frame, audit)


def write_combined_coefficients(
    destination: Path, member: AAMember, *, compact: bool = False,
) -> None:
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
    if compact:
        frame = frame.filter(pl.col("coefficient") != 0)
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
        if compact:
            frame = frame.filter(pl.col("coefficient") != 0)
        _append_frame(destination / "combined_coefficients.tsv", frame)


def write_site_scores(destination: Path, members: list[AAMember], *, compact: bool = False) -> None:
    """Write both scores without accumulating all site dictionaries in memory."""
    from phenoradar.amino_acid_reporting import AAReportWriter

    with AAReportWriter(destination, compact=compact) as writer:
        writer.add(members, features=False)
        writer.write_site_scores()


def write_site_stability(destination: Path, members: list[AAMember]) -> None:
    from phenoradar.amino_acid_reporting import AAReportWriter

    with AAReportWriter(destination) as writer:
        writer.add(members, features=False)
        writer.write_site_stability()


def write_frequency_table(
    destination: Path,
    store: AlignmentStore,
    manifest: pl.DataFrame,
    members: list[AAMember],
    importance: pl.DataFrame,
    limit: int,
    *, prediction_species: list[str] | None = None,
) -> None:
    """Save OOF residue frequencies used by the confusion-group plot."""
    features = (
        importance.sort(["importance_mean", "feature"], descending=[True, False])
        .head(limit)["feature"]
        .to_list()
    )
    species = (sorted(prediction_species) if prediction_species is not None else sorted(
        manifest.filter(pl.col("pool") == "validation")["species"].unique().to_list()))
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
        targets = (species if prediction_species is not None else
                   manifest.filter((pl.col("pool") == "validation") &
                                   (pl.col("fold_id") == fold))["species"].to_list())
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
    table = pl.DataFrame({
        "species": np.repeat(species, len(features)),
        "feature": features * len(species), "frequency": values.ravel(),
    })
    destination.mkdir(parents=True, exist_ok=True)
    table.write_csv(destination / "top_amino_acid_frequencies.tsv", separator="\t")
