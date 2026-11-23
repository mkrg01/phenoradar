"""Compact model coefficients and exact per-model feature membership."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from phenoradar.amino_acid_reporting import AAReportWriter
    from phenoradar.config import AppConfig


def compact_reports(config: AppConfig) -> bool:
    # Earlier pickled bundles have no table setting; retain their output layout.
    return getattr(config.report, "amino_acid_tables", "full") == "compact"


def write_dictionary(writer: AAReportWriter, raw: pl.LazyFrame) -> None:
    """One feature axis, with 64 model-retention bits in each integer word."""
    destination = writer.destination
    models = pl.DataFrame(writer.model_details).sort("model_id").with_row_index("retention_index")
    if models["model_id"].n_unique() != models.height:
        raise ValueError("Compact report model IDs must be unique")
    models.write_parquet(destination / "report_models.parquet")
    ogs = (raw.filter(pl.col("modality") == "amino_acid").select("orthogroup")
           .unique().sort("orthogroup").collect(engine="streaming")
           .with_row_index("orthogroup_id"))
    ogs.write_csv(destination / "report_orthogroups.tsv", separator="\t")
    keys = ["modality", "orthogroup", "alignment_pos_1based", "residue", "expression_feature"]
    # Worker IDs reserve million-sized ranges. Only dense retention ordinals
    # determine bitmap width; external model IDs remain unchanged in all joins.
    words = max(1, (models.height + 63) // 64)
    masks = [pl.when(pl.col("retention_index") // 64 == word)
             .then(pl.lit(2, dtype=pl.UInt64).pow(pl.col("retention_index") % 64))
             .otherwise(pl.lit(0, dtype=pl.UInt64)).bitwise_or()
             .alias(f"retained_word_{word}") for word in range(words)]
    indexed = raw.join(models.select(pl.col("model_id").alias("_model"),
                                     "retention_index").lazy(), on="_model", how="left")
    axis = indexed.group_by(keys).agg(masks).join(ogs.lazy(), on="orthogroup", how="left")
    axis.drop("orthogroup").select(
        "modality", "orthogroup_id", pl.col("alignment_pos_1based").cast(pl.UInt32),
        "residue", "expression_feature", *[f"retained_word_{w}" for w in range(words)],
    ).sink_parquet(destination / "feature_dictionary.parquet", engine="streaming")
    # Keep every nonzero, including values below the stability/solver tolerance.
    raw.filter(pl.col("coefficient") != 0).join(
        ogs.lazy(), on="orthogroup", how="left",
    ).select(
        pl.col("_model").alias("model_id"), "modality", "orthogroup_id",
        pl.col("alignment_pos_1based").cast(pl.UInt32), "residue", "expression_feature",
        "coefficient",
    ).sink_parquet(destination / "model_nonzero_coefficients.parquet", engine="streaming")
    manifest = {
        "format": "phenoradar-aa-compact-2",
        "model_table": "report_models.parquet", "orthogroup_table": "report_orthogroups.tsv",
        "feature_dictionary": "feature_dictionary.parquet",
        "coefficients": "model_nonzero_coefficients.parquet",
        "retained_words": words,
        "membership": ("For retention_index r in report_models.parquet, feature retained iff "
                       "retained_word_{r//64} & (1 << (r%64)). model_id is preserved separately."),
        "zero_coefficients": "Omitted exactly; a retained zero differs from an excluded feature.",
        "importance": "abs(coefficient)/modality_l1; coefficient*0 when importance_zeroed is true.",
        "details": ("Selected features (plus plot padding); "
                    "selected model sites retain every known residue."),
        "full_audits": "Use report.amino_acid_tables=full for all preprocessing means and site QC.",
    }
    (destination / "compact_report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_site_scores(raw: pl.LazyFrame, destination: Path) -> None:
    context = ["scope", "sample_set_id", "candidate_index"]
    frame = raw
    for metric in ["pss", "effect_range"]:
        total = pl.col(metric).sum().over("_model")
        frame = frame.with_columns(
            pl.when(total > 0).then(pl.col(metric) / total).otherwise(0.0)
            .alias(f"normalized_{metric}"),
            pl.when(pl.col(metric) > 0)
            .then(pl.col(metric).rank(method="min", descending=True).over(context))
            .otherwise(None).alias(f"{metric}_rank"),
        )
    frame.filter((pl.col("pss") != 0) | (pl.col("effect_range") != 0)).select(
        pl.col("_model").alias("model_id"), *context, "orthogroup", "alignment_pos_1based",
        "n_known_residue_states", "n_nonzero_coefficients", "pss", "effect_range",
        "normalized_pss", "normalized_effect_range", "pss_rank", "effect_range_rank",
    ).sink_csv(destination / "amino_acid_site_scores.tsv", separator="\t", engine="streaming")


def read_model_coefficients(directory: Path) -> pl.DataFrame:
    """Read nonzero model values and derive names/importance without a dense grid."""
    directory = Path(directory)
    values = pl.read_parquet(directory / "model_nonzero_coefficients.parquet")
    ogs = pl.read_csv(directory / "report_orthogroups.tsv", separator="\t",
                      schema_overrides={"orthogroup_id": pl.UInt32, "orthogroup": pl.String})
    models = pl.read_parquet(directory / "report_models.parquet")
    frame = values.join(ogs, on="orthogroup_id", how="left").join(models, on="model_id")
    features = [json.dumps([og, pos, residue], separators=(",", ":"))
                if modality == "amino_acid" else expression
                for modality, og, pos, residue, expression in frame.select(
                    "modality", "orthogroup", "alignment_pos_1based", "residue",
                    "expression_feature").iter_rows()]
    frame = frame.with_columns(pl.Series("feature", features, dtype=pl.String))
    total = (pl.when(pl.col("modality") == "amino_acid").then(pl.col("amino_acid_l1"))
             .otherwise(pl.col("expression_l1")))
    zeroed = (pl.when(pl.col("modality") == "amino_acid")
              .then(pl.col("amino_acid_importance_zeroed"))
              .otherwise(pl.col("expression_importance_zeroed")))
    return frame.with_columns(pl.when(zeroed).then(pl.col("coefficient") * 0)
                             .otherwise(pl.col("coefficient").abs() / total).alias("importance"))
