"""Columnar, disk-backed amino-acid interpretation across CV folds."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import polars as pl

from phenoradar.feature_stability import (
    _BY_FEATURE_SCHEMA,
    _BY_FOLD_PAIR_SCHEMA,
    _SUMMARY_SCHEMA,
    DEFAULT_NONZERO_TOLERANCE,
    _fold_sort_key,
)

if TYPE_CHECKING:
    from phenoradar.amino_acid import AASite
    from phenoradar.amino_acid_run import AAMember

REPORT_BLOCK_ROWS = 65_536


class _Spool:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir()
        self.frames: list[pl.DataFrame] = []
        self.rows = 0
        self.parts = 0

    def add(self, frame: pl.DataFrame) -> None:
        if frame.height:
            self.frames.append(frame)
            self.rows += frame.height
            if self.rows >= REPORT_BLOCK_ROWS:
                self.flush()

    def flush(self) -> None:
        if self.frames:
            pl.concat(self.frames).write_parquet(self.directory / f"{self.parts:06d}.parquet")
            self.parts += 1
            self.frames.clear()
            self.rows = 0

    def scan(self) -> pl.LazyFrame:
        self.flush()
        return pl.scan_parquet(self.directory / "*.parquet")


def coefficient_blocks(
    member: AAMember,
) -> Iterator[tuple[list[AASite], np.ndarray, np.ndarray, np.ndarray]]:
    offset = member.expression_width
    for start in range(0, len(member.transform.sites), 512):
        sites = member.transform.sites[start:start + 512]
        active = np.stack([site.active for site in sites])
        known = np.stack([site.known for site in sites])
        count = int(active.sum())
        beta = np.zeros(active.shape)
        beta[active] = member.model.coef_[offset:offset + count]
        offset += count
        yield sites, beta, known, active


def _sink(frame: pl.LazyFrame, path: Path) -> None:
    frame.sink_csv(path, separator="\t", engine="streaming")


def _concat_csv(parts: list[Path], destination: Path) -> None:
    with destination.open("wb") as target:
        for index, path in enumerate(parts):
            with path.open("rb") as source:
                if index:
                    source.readline()
                shutil.copyfileobj(source, target)


class AAReportWriter:
    """Spool each model once; retain no full feature names or fitted fold models."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        destination.mkdir(parents=True, exist_ok=True)
        self._temp = TemporaryDirectory(prefix="phenoradar_aa_reports_")
        self.path = Path(self._temp.name)
        self.features = _Spool(self.path / "features")
        self.sites = _Spool(self.path / "sites")
        self.models: list[tuple[str, str, int]] = []
        self.contexts: set[tuple[str, int, int]] = set()
        self.transforms: dict[tuple[str, int, int], tuple[int, Path | None]] = {}

    def __enter__(self) -> AAReportWriter:
        return self

    def close(self) -> None:
        self._temp.cleanup()

    def __exit__(self, *args: Any) -> None:
        self.close()

    def add(self, members: list[AAMember], *, features: bool = True) -> None:
        transforms: dict[int, tuple[int, Path | None]] = {}
        for member in members:
            model_index = len(self.models)
            fold = member.scope.removeprefix("outer_fold_")
            self.models.append((fold, member.scope, member.sample_set_id))
            context = member.scope, member.sample_set_id, member.candidate.candidate_index
            self.contexts.add(context)
            key = id(member.transform)
            if key not in transforms:
                path = None
                if features and member.transform.settings.aggregation == "best_covered":
                    scoring = member.transform.representative_columns
                    if scoring is None:
                        grouped: dict[str, list[int]] = defaultdict(list)
                        for site in member.transform.sites:
                            grouped[site.orthogroup].append(site.column)
                        scoring = {og: np.array(cols) for og, cols in grouped.items()}
                    path = self.path / f"scoring_{model_index}.joblib"
                    joblib.dump(scoring, path)
                transforms[key] = model_index, path
            self.transforms[context] = transforms[key]
            n = member.expression_width
            aa_total = float(np.abs(member.model.coef_[n:]).sum())
            expression = getattr(member, "expression", None)
            if features and expression is not None:
                coefficient = member.model.coef_[:n]
                total = float(np.abs(coefficient).sum())
                self._features(expression.features, coefficient, total, "expression",
                               model_index, fold, member.sample_set_id)
            for sites, beta, known, active in coefficient_blocks(member):
                if features:
                    names = [site.feature(int(a)) for site in sites
                             for a in np.flatnonzero(site.active)]
                    self._features(names, beta[active], aa_total, "amino_acid",
                                   model_index, fold, member.sample_set_id)
                n_known = known.sum(axis=1)
                effect = (np.max(np.where(known, beta, -np.inf), axis=1)
                          - np.min(np.where(known, beta, np.inf), axis=1))
                effect[n_known < 2] = 0
                self.sites.add(pl.DataFrame({
                    "orthogroup": [s.orthogroup for s in sites],
                    "alignment_pos_1based": [s.column + 1 for s in sites],
                    "n_known_residue_states": n_known.astype(np.int64),
                    "n_nonzero_coefficients": np.count_nonzero(beta * known, axis=1),
                    "pss": np.sum(np.abs(beta) * known, axis=1),
                    "effect_range": effect,
                    "eligible": active.any(axis=1),
                }).with_columns(
                    pl.lit(model_index).alias("_model"), pl.lit(member.scope).alias("scope"),
                    pl.lit(member.sample_set_id).alias("sample_set_id"),
                    pl.lit(member.candidate.candidate_index).alias("candidate_index"),
                ))

    def frequency_members(
        self, members: list[AAMember], importance: pl.DataFrame,
    ) -> list[AAMember]:
        """Restore plot sites from each fold's full eligibility audit, not its betas."""
        from phenoradar.amino_acid import AASite

        requested = sorted({(str(v[0]), int(v[1]))
                            for f in importance["feature"] for v in [json.loads(f)]})
        axes = pl.DataFrame(requested, schema=["orthogroup", "alignment_pos_1based"],
                            orient="row").lazy()
        result = []
        seen = set()
        for member in members:
            key = member.scope, member.sample_set_id, member.candidate.candidate_index
            model_index, path = self.transforms[key]
            if model_index in seen:
                continue
            seen.add(model_index)
            rows = self.sites.scan().filter(pl.col("_model") == model_index).select(
                "orthogroup", "alignment_pos_1based"
            ).join(axes, on=["orthogroup", "alignment_pos_1based"]).sort(
                ["orthogroup", "alignment_pos_1based"]
            ).collect(engine="streaming")
            sites = [AASite(og, position - 1, 0, 0, 0, 0) for og, position in rows.iter_rows()]
            scoring = None
            if path is not None:
                masks = joblib.load(path)
                scoring = {site.orthogroup: masks[site.orthogroup] for site in sites}
            transform = replace(member.transform, sites=sites, representative_columns=scoring)
            result.append(replace(member, transform=transform))
        return result

    def _features(self, names: list[str], coefficient: np.ndarray, total: float,
                  modality: str, model: int, fold: str, sample: int) -> None:
        if not names:
            return
        importance = np.abs(coefficient) / total if not np.isclose(total, 0) else coefficient * 0
        self.features.add(pl.DataFrame({
            "feature": names, "coefficient": coefficient, "importance": importance,
        }).with_columns(
            pl.lit(modality).alias("modality"), pl.lit(model).alias("_model"),
            pl.lit(fold).alias("fold_id"), pl.lit(sample).alias("sample_set_id"),
        ))

    def write_site_scores(self) -> None:
        raw = self.sites.scan()
        context = ["scope", "sample_set_id", "candidate_index"]
        for metric in ["pss", "effect_range"]:
            parts = []
            for index, (scope, sample, candidate) in enumerate(sorted(self.contexts)):
                frame = raw.filter(
                    (pl.col("scope") == scope) & (pl.col("sample_set_id") == sample)
                    & (pl.col("candidate_index") == candidate)
                ).with_columns(pl.col(metric).sum().over("_model").alias("_total"))
                frame = frame.with_columns(
                    pl.when(pl.col("_total") > 0).then(pl.col(metric) / pl.col("_total"))
                    .otherwise(0.0).alias("normalized_score"),
                    pl.when(pl.col(metric) > 0)
                    .then(pl.col(metric).rank(method="min", descending=True).over(context))
                    .otherwise(None).alias("rank"),
                ).select(*context, "orthogroup", "alignment_pos_1based", "n_known_residue_states",
                         "n_nonzero_coefficients", metric, "normalized_score", "rank")
                path = self.path / f"scores_{metric}_{index}.tsv"
                _sink(frame.sort([metric, "orthogroup", "alignment_pos_1based"],
                                 descending=[True, False, False]), path)
                parts.append(path)
            _concat_csv(parts, self.destination / f"amino_acid_site_{metric}.tsv")
            for path in parts:
                path.unlink()

    def write_site_stability(self) -> None:
        raw = self.sites.scan()
        n = len({scope for _, scope, _ in self.models})
        for keys, name in [(["orthogroup", "alignment_pos_1based"], "site"),
                           (["orthogroup"], "og")]:
            frame = raw.group_by(keys).agg(
                pl.col("scope").filter(pl.col("eligible")).n_unique().alias("n_eligible_folds"),
                pl.col("scope").filter(pl.col("effect_range") > 1e-12)
                .n_unique().alias("n_selected_folds"),
            )
            frame = frame.filter(pl.col("n_eligible_folds") > 0)
            frame = frame.with_columns(pl.lit(n).alias("n_outer_folds")).select(
                *keys, "n_outer_folds", "n_eligible_folds", "n_selected_folds",
                (pl.col("n_selected_folds") / n).alias("selection_frequency"),
                pl.when(pl.col("n_eligible_folds") > 0)
                .then(pl.col("n_selected_folds") / pl.col("n_eligible_folds"))
                .otherwise(0.0).alias("selection_frequency_when_eligible"),
            )
            _sink(frame.sort(keys), self.destination / f"amino_acid_{name}_stability.tsv")

    def finalize(self, *, combined: bool, top_features: int) -> pl.DataFrame:
        raw = self.features.scan()
        counts: dict[str, int] = {}
        for fold, _, _ in self.models:
            counts[fold] = counts.get(fold, 0) + 1
        folds = sorted(counts, key=_fold_sort_key)
        count_frame = pl.DataFrame({"fold_id": list(counts), "_count": list(counts.values())})
        means = raw.group_by("fold_id", "feature", "modality").agg(
            pl.col("coefficient").sum(), pl.col("importance").sum(),
        ).join(count_frame.lazy(), on="fold_id").with_columns(
            (pl.col("coefficient") / pl.col("_count")).alias("coefficient"),
            (pl.col("importance") / pl.col("_count")).alias("importance"),
        ).drop("_count")
        means.sink_parquet(self.path / "means.parquet", engine="streaming")
        means = pl.scan_parquet(self.path / "means.parquet")
        union = means.select("feature", "modality").unique()
        union.sink_parquet(self.path / "union.parquet", engine="streaming")
        union = pl.scan_parquet(self.path / "union.parquet")
        _sink(raw.select(pl.lit("outer_fold").alias("scope"), "fold_id", "sample_set_id",
                         "feature").unique(), self.destination / "retained_features.tsv")
        modalities = ["expression", "amino_acid"] if combined else ["amino_acid"]
        for modality in modalities:
            self._interpret(means.filter(pl.col("modality") == modality),
                            union.filter(pl.col("modality") == modality), counts,
                            prefix=f"{modality}_" if combined else "", combined=combined)
        if combined:
            for name in ["feature_importance", "coefficients", "feature_importance_by_fold",
                         "coefficients_by_fold"]:
                _concat_csv([self.destination / f"{m}_{name}.tsv" for m in modalities],
                            self.destination / f"{name}.tsv")
        self._stability(means, union, folds)
        self.write_site_scores()
        self.write_site_stability()
        path = self.destination / ("amino_acid_feature_importance.tsv" if combined
                                   else "feature_importance.tsv")
        return pl.scan_csv(path, separator="\t").sort(
            ["importance_mean", "feature"], descending=[True, False]
        ).head(top_features).collect(engine="streaming")

    def _interpret(self, means: pl.LazyFrame, union: pl.LazyFrame, counts: dict[str, int],
                   *, prefix: str, combined: bool) -> None:
        n = len(counts)
        stats = means.group_by("feature", "modality").agg(
            pl.len().alias("_k"),
            *(expression for col in ["importance", "coefficient"] for expression in [
                pl.col(col).sum().alias(f"{col}_sum"),
                pl.col(col).mean().alias(f"{col}_present_mean"),
                pl.col(col).var(ddof=0).fill_null(0).alias(f"{col}_present_var"),
            ]),
        ).with_columns(*(
            expression for col in ["importance", "coefficient"] for expression in [
                (pl.col(f"{col}_sum") / n).alias(f"{col}_mean"),
                ((pl.col(f"{col}_present_var") * pl.col("_k")
                  + pl.col(f"{col}_present_mean").pow(2) * pl.col("_k")
                  * (n - pl.col("_k")) / n) / n).sqrt().alias(f"{col}_std"),
            ]
        ))
        for kind, col in [("feature_importance", "importance"), ("coefficients", "coefficient")]:
            coefficient = col == "coefficient"
            label = "coef" if coefficient else "importance"
            method = "coef_signed" if coefficient else "coef_abs_l1_norm"
            extra = [pl.lit("NA").alias("reason")] if coefficient else []
            mod = ["modality"] if combined else []
            frame = stats.select(
                "feature", pl.col(f"{col}_mean").alias(f"{label}_mean"),
                pl.col(f"{col}_std").alias(f"{label}_std"),
                pl.lit(len(self.models)).alias("n_models"), pl.lit(n).alias("n_folds"),
                pl.lit(method).alias("method"), *extra, *mod,
            ).sort("feature")
            _sink(frame, self.destination / f"{prefix}{kind}.tsv")
            parts = []
            for index, fold in enumerate(sorted(counts)):
                local = means.filter(pl.col("fold_id") == fold).select("feature", col)
                frame = union.join(local, on="feature", how="left").select(
                    pl.lit(fold).alias("fold_id"), "feature",
                    pl.col(col).fill_null(0.0).alias(f"{label}_mean"),
                    pl.lit(counts[fold]).alias("n_models"), pl.lit(method).alias("method"),
                    *extra, *mod,
                ).sort("feature")
                path = self.path / f"fold_{index}_{prefix}{kind}.tsv"
                _sink(frame, path)
                parts.append(path)
            _concat_csv(parts, self.destination / f"{prefix}{kind}_by_fold.tsv")
            for path in parts:
                path.unlink()

    def _stability(self, means: pl.LazyFrame, union: pl.LazyFrame, folds: list[str]) -> None:
        tolerance = DEFAULT_NONZERO_TOLERANCE
        n = len(folds)
        selected = means.filter(pl.col("coefficient").abs() > tolerance)
        selected.sink_parquet(self.path / "selected.parquet", engine="streaming")
        selected = pl.scan_parquet(self.path / "selected.parquet")
        retained = means.group_by("feature").agg(pl.len().alias("n_retained_folds"))
        signed = selected.group_by("feature").agg(
            pl.len().alias("n_nonzero_folds"),
            (pl.col("coefficient") > tolerance).sum().alias("n_positive_folds"),
            (pl.col("coefficient") < -tolerance).sum().alias("n_negative_folds"),
            pl.col("coefficient").mean().alias("coef_mean_nonzero"),
            pl.col("coefficient").median().alias("coef_median_nonzero"),
        )
        frame = union.select("feature").join(retained, on="feature").join(
            signed, on="feature", how="left"
        ).with_columns(pl.col("n_nonzero_folds", "n_positive_folds", "n_negative_folds")
                       .fill_null(0))
        positive, negative, nonzero = (pl.col(c) for c in [
            "n_positive_folds", "n_negative_folds", "n_nonzero_folds"])
        frame = frame.with_columns(
            pl.lit(n).alias("n_outer_folds"),
            (pl.col("n_retained_folds") / n).alias("retained_frequency"),
            (nonzero / n).alias("selection_frequency"),
            (nonzero / pl.col("n_retained_folds")).alias("selection_frequency_when_retained"),
            pl.when(nonzero == 0).then(None).when(positive > negative).then(pl.lit("positive"))
            .when(negative > positive).then(pl.lit("negative")).otherwise(pl.lit("tie"))
            .alias("dominant_sign"),
            pl.when(nonzero >= 2).then(pl.max_horizontal(positive, negative) / nonzero)
            .otherwise(None).alias("sign_agreement_rate"),
            pl.when(nonzero == 0).then(pl.lit("never_selected"))
            .when(nonzero < 2).then(pl.lit("selected_in_fewer_than_two_folds"))
            .otherwise(pl.lit("ok")).alias("sign_reason"),
            pl.lit("coef_abs_gt_tol").alias("selection_method"),
            pl.lit(tolerance).alias("nonzero_tolerance"),
        ).select(list(_BY_FEATURE_SCHEMA)).cast(pl.Schema(_BY_FEATURE_SCHEMA))
        destination = self.destination / "feature_stability_by_feature.tsv"
        _sink(frame.sort(["selection_frequency", "retained_frequency", "feature"],
                         descending=[True, True, False]), destination)
        pair_rows = []
        for a, b in combinations(folds, 2):
            left = selected.filter(pl.col("fold_id") == a).select("feature")
            right = selected.filter(pl.col("fold_id") == b).select("feature")
            ca = left.select(pl.len()).collect(engine="streaming").item()
            cb = right.select(pl.len()).collect(engine="streaming").item()
            intersection = left.join(right, on="feature").select(pl.len()).collect(
                engine="streaming").item()
            union_count = ca + cb - intersection
            pair_rows.append({
                "fold_id_a": a, "fold_id_b": b, "n_nonzero_a": ca, "n_nonzero_b": cb,
                "n_intersection": intersection, "n_union": union_count,
                "jaccard": intersection / union_count if union_count else None,
                "reason": "ok" if union_count else "both_feature_sets_empty",
                "selection_method": "coef_abs_gt_tol", "nonzero_tolerance": tolerance,
            })
        pairs = pl.DataFrame(pair_rows, schema=_BY_FOLD_PAIR_SCHEMA)
        pairs.write_csv(self.destination / "feature_stability_by_fold_pair.tsv", separator="\t")
        numeric = pl.scan_csv(destination, separator="\t",
                              schema_overrides=_BY_FEATURE_SCHEMA).select(
            pl.len().alias("n_features_evaluated"),
            (pl.col("n_retained_folds") > 0).sum().alias("n_features_ever_retained"),
            (pl.col("n_nonzero_folds") > 0).sum().alias("n_features_ever_selected"),
            (2 * pl.col("n_nonzero_folds") >= n).sum()
            .alias("n_features_selected_in_half_folds"),
            (pl.col("n_nonzero_folds") == n).sum().alias("n_features_selected_in_all_folds"),
            pl.col("sign_agreement_rate").count().alias("n_features_with_sign_agreement"),
            pl.col("sign_agreement_rate").mean().alias("sign_agreement_mean"),
        ).collect(engine="streaming").row(0, named=True)
        jaccard = pairs["jaccard"].drop_nulls()
        summary = {
            "n_outer_folds": n, "n_fold_pairs": pairs.height,
            "n_valid_jaccard_pairs": len(jaccard), **numeric,
            "jaccard_mean": jaccard.mean(), "jaccard_median": jaccard.median(),
            "jaccard_min": jaccard.min(), "jaccard_max": jaccard.max(),
            "selection_method": "coef_abs_gt_tol", "sign_available": True,
            "nonzero_tolerance": tolerance,
        }
        pl.DataFrame([summary], schema=_SUMMARY_SCHEMA).write_csv(
            self.destination / "feature_stability_summary.tsv", separator="\t"
        )
