"""Expression-equivalent figures from bounded slices of amino-acid output tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from phenoradar import figures
from phenoradar.amino_acid_diagnostics import (
    site_score_summary,
    write_site_scores,
    write_site_stability,
)
from phenoradar.amino_acid_diagnostics import (
    write_frequency_by_confusion as _frequency_by_confusion,
)
from phenoradar.amino_acid_diagnostics import (
    write_site_filter_funnel as _site_filter_funnel,
)
from phenoradar.config import AppConfig
from phenoradar.group_summary_outputs import write_group_summary_artifacts
from phenoradar.metrics import (
    FIXED_PROBABILITY_THRESHOLD_NAME,
    FIXED_PROBABILITY_THRESHOLD_VALUE,
)
from phenoradar.tree_prediction import write_run_tree_prediction_artifacts

if TYPE_CHECKING:
    from phenoradar.amino_acid import AlignmentStore
    from phenoradar.amino_acid_reporting import AAReportWriter
    from phenoradar.amino_acid_run import AAMember
    from phenoradar.combined import ExpressionInput


def _read(directory: Path, name: str) -> pl.DataFrame | None:
    path = directory / f"{name}.tsv"
    if not path.exists():
        return None
    return pl.read_csv(path, separator="\t", infer_schema_length=10000,
                       schema_overrides={"fold_id": pl.String, "pss": pl.Float64,
                                         "effect_range": pl.Float64})


def _scan(directory: Path, name: str) -> pl.LazyFrame:
    return pl.scan_csv(directory / f"{name}.tsv", separator="\t", infer_schema_length=10000,
                       schema_overrides={"fold_id": pl.String, "pss": pl.Float64,
                                         "effect_range": pl.Float64})


def _selection(directory: Path, name: str) -> pl.DataFrame | None:
    frame = _read(directory, name)
    if frame is None:
        return None
    if name == "model_selection_selected":
        frame = frame.with_columns(
            pl.col("candidate_index").cum_count().over("scope", "sample_set_id").alias("rank")
        )
    return frame.with_columns(
        pl.col("scope").str.replace("^outer_fold_", "").alias("fold_id"),
        pl.when(pl.col("scope") == "final_refit")
        .then(pl.lit("final_refit"))
        .otherwise(pl.lit("outer_fold"))
        .alias("selection_scope"),
    )


def _final_features(members: list[AAMember]) -> tuple[pl.DataFrame, pl.DataFrame]:
    from phenoradar.amino_acid_prediction import prediction_members

    compact = prediction_members(members, padding_sites=100)
    names = sorted({f for m in compact for f in m.features})
    indices = {name: i for i, name in enumerate(names)}
    beta = np.zeros((len(members), len(names)))
    importance = beta.copy()
    for i, m in enumerate(compact):
        cols = [indices[f] for f in m.features]
        beta[i, cols] = m.model.coef_
        for start, end in [(0, m.expression_width), (m.expression_width, len(cols))]:
            values = np.abs(m.model.coef_[start:end])
            total = float(values.sum())
            if not np.isclose(total, 0):
                importance[i, cols[start:end]] = values / total
    common = {"feature": names, "n_models": [len(members)] * len(names)}
    return (
        pl.DataFrame(
            {
                **common,
                "importance_mean": importance.mean(axis=0),
                "importance_std": importance.std(axis=0),
                "method": "coef_abs_l1_norm",
            }
        ),
        pl.DataFrame(
            {
                **common,
                "coef_mean": beta.mean(axis=0),
                "coef_std": beta.std(axis=0),
                "method": "coef_signed",
                "reason": "NA",
            }
        ),
    )


def write_amino_acid_figures(
    run_dir: Path,
    config: AppConfig,
    members: list[AAMember] | None = None,
    *,
    report: AAReportWriter | None = None,
    store: AlignmentStore | None = None,
    expression: ExpressionInput | None = None,
) -> list[str]:
    """Reuse common plots, without loading millions of zero coefficients into RAM."""
    cv = run_dir / "cv/tables"
    model = run_dir / "model/tables"
    limit = config.figures.top_features
    importance = (
        _scan(cv, "feature_importance")
        .sort(["importance_mean", "feature"], descending=[True, False])
        .head(limit)
        .collect(engine="streaming")
    )
    coefficients = (
        _scan(cv, "coefficients")
        .with_columns(pl.col("coef_mean").abs().alias("_abs"))
        .sort(["_abs", "feature"], descending=[True, False])
        .head(limit)
        .drop("_abs")
        .collect(engine="streaming")
    )
    requested = sorted(set(importance["feature"]) | set(coefficients["feature"]))
    if report is None:
        by_fold = {
            name: _scan(cv, name)
            .filter(pl.col("feature").is_in(requested))
            .collect(engine="streaming")
            for name in ["feature_importance_by_fold", "coefficients_by_fold"]
        }
        means = None
    else:
        means = pl.scan_parquet(report.path / "means" / "*.parquet")
        fold_model_counts: dict[str, int] = {}
        for fold, _, _ in report.models:
            fold_model_counts[fold] = fold_model_counts.get(fold, 0) + 1
        axes = pl.DataFrame(
            {
                "fold_id": list(fold_model_counts),
                "n_models": list(fold_model_counts.values()),
            }
        ).lazy()
        axes = axes.join(pl.DataFrame({"feature": requested}).lazy(), how="cross")
        selected_means = means.filter(pl.col("feature").is_in(requested))
        grid = axes.join(selected_means, on=["fold_id", "feature"], how="left")
        by_fold = {}
        for name, column, label, method in [
            ("feature_importance_by_fold", "importance", "importance_mean", "coef_abs_l1_norm"),
            ("coefficients_by_fold", "coefficient", "coef_mean", "coef_signed"),
        ]:
            frame = grid.select(
                "fold_id",
                "feature",
                "n_models",
                pl.col(column).fill_null(0).alias(label),
                pl.lit(method).alias("method"),
            )
            if column == "coefficient":
                frame = frame.with_columns(pl.lit("NA").alias("reason"))
            by_fold[name] = frame.collect(engine="streaming")
    selected = _selection(cv, "model_selection_selected")
    final_selected = _selection(model, "model_selection_selected")
    if final_selected is not None:
        selected = (
            final_selected
            if selected is None
            else pl.concat([selected, final_selected], how="diagonal_relaxed")
        )
    final_trials = _selection(model, "model_selection_trials")
    final_summary = (
        None
        if final_trials is None
        else figures._summarize_model_selection_trials_for_figure(final_trials)
    )
    final_importance, final_coefficients = _final_features(members) if members else (None, None)
    expression_importance = (
        _scan(cv, "expression_feature_importance")
        .sort(["importance_mean", "feature"], descending=[True, False])
        .head(limit)
        .collect(engine="streaming")
        if expression is not None
        else None
    )
    oof = _read(cv, "prediction_cv")
    assert oof is not None
    coefficient_source = (
        _scan(cv, "coefficients_by_fold")
        if means is None
        else means.rename({"coefficient": "coef_mean"})
    )
    sparsity = (
        coefficient_source.group_by("fold_id")
        .agg((pl.col("coef_mean").abs() > 1e-12).sum().alias("n_nonzero_features"))
        .with_columns(pl.lit("outer_fold").alias("scope"))
        .collect(engine="streaming")
    )
    counts = _selection(cv, "model_selection_selected")
    assert counts is not None
    top_sites = {}
    site_folds = {}
    for metric in ["pss", "effect_range"]:
        site_source = (
            report.sites.scan() if report is not None else _scan(
                cv, "amino_acid_site_scores" if (cv / "amino_acid_site_scores.tsv").exists()
                else f"amino_acid_site_{metric}",
            )
        )
        top_sites[metric], site_folds[metric] = site_score_summary(
            site_source, counts, metric, limit,
        )
    from phenoradar.orthogroup_annotation import load_orthogroup_annotations

    annotation_features = set(requested)
    for annotation_frame in [
        expression_importance,
        final_importance,
        final_coefficients,
        _read(cv, "top_amino_acid_frequencies"),
    ]:
        if annotation_frame is not None:
            annotation_features.update(annotation_frame["feature"])
    ogs = {figures._feature_address(f)[0] for f in annotation_features}
    for top in top_sites.values():
        ogs.update(top["orthogroup"])
    annotations = load_orthogroup_annotations(
        Path(config.data.orthogroup_annotation_path)
        if config.data.orthogroup_annotation_path
        else None,
        feature_names=sorted(ogs),
    )
    metrics = _read(cv, "metrics_cv")
    assert metrics is not None
    external = _read(run_dir / "external_test/tables", "pred_external_test")
    inference = _read(run_dir / "inference/tables", "pred_inference")
    warnings = figures.write_run_figures(
        run_dir=run_dir,
        metrics_cv=metrics,
        oof_predictions=oof,
        feature_importance=importance,
        coefficients=coefficients,
        ensemble_model_probs=None,
        model_selection_trials=_selection(cv, "model_selection_trials"),
        model_selection_selected=selected,
        feature_importance_by_fold=by_fold["feature_importance_by_fold"],
        coefficients_by_fold=by_fold["coefficients_by_fold"],
        feature_stability_by_fold_pair=_read(cv, "feature_stability_by_fold_pair"),
        loss_by_split_cv=_read(cv, "loss_by_split_cv"),
        model_sparsity=sparsity,
        final_refit_feature_importance=final_importance,
        final_refit_coefficients=final_coefficients,
        final_refit_model_selection_trials_summary=final_summary,
        group_bootstrap_metrics=_read(cv, "group_bootstrap_metrics"),
        pred_external_test=external,
        pred_inference=inference,
        trait_name=config.data.trait_col,
        top_features=limit,
        orthogroup_annotations=annotations,
    )
    if expression is not None and expression_importance is not None:
        _write_combined_expression_figures(
            run_dir,
            config,
            expression,
            oof,
            external,
            expression_importance,
            final_importance,
            final_coefficients,
            annotations,
        )
    for stage, predictions, source in [
        ("cv", oof, "prediction_cv.tsv"),
        ("external_test", external, "pred_external_test.tsv"),
        ("inference", inference, "pred_inference.tsv"),
    ]:
        if predictions is not None and predictions.height:
            warnings.extend(
                write_group_summary_artifacts(
                    run_dir=run_dir,
                    stage=stage,
                    predictions=predictions,
                    source_table_name=source,
                    config=config,
                )
            )
    for metric, top in top_sites.items():
        write_site_scores(run_dir, top, site_folds[metric], metric, annotations)
    write_site_stability(run_dir, top_sites["pss"], annotations)
    warnings.extend(
        write_run_tree_prediction_artifacts(
            run_dir=run_dir,
            tree_path=Path(config.data.tree_path) if config.data.tree_path else None,
            metadata_path=Path(config.data.metadata_path),
            tpm_path=Path(config.data.tpm_path),
            species_col=config.data.species_col,
            feature_col=config.data.feature_col,
            value_col=config.data.value_col,
            trait_col=config.data.trait_col,
            group_col=config.data.contrast_group_col or config.split.group_col,
            oof_predictions=oof,
            thresholds=pl.DataFrame(
                {
                    "threshold_name": [FIXED_PROBABILITY_THRESHOLD_NAME],
                    "threshold_value": [FIXED_PROBABILITY_THRESHOLD_VALUE],
                }
            ),
            feature_importance=importance,
            coefficients=coefficients,
            pred_external_test=external,
            include_feature_heatmaps=False,
        )
    )
    if store is not None:
        from phenoradar.amino_acid_tree import write_residue_tree

        warnings.extend(
            write_residue_tree(
                run_dir,
                Path(config.data.tree_path) if config.data.tree_path else None,
                store,
                top_sites["pss"],
                oof,
                aggregation=config.preprocess.amino_acid.aggregation,
                annotations=annotations,
            )
        )
    _frequency_by_confusion(run_dir, oof, limit, annotations)
    _site_filter_funnel(run_dir)
    return warnings


def _write_combined_expression_figures(
    run_dir: Path,
    config: AppConfig,
    expression: ExpressionInput,
    oof: pl.DataFrame,
    external: pl.DataFrame | None,
    importance: pl.DataFrame,
    final_importance: pl.DataFrame | None,
    final_coefficients: pl.DataFrame | None,
    annotations: pl.DataFrame | None,
) -> None:
    """Adapt the existing raw-expression cache to the shared distribution writer."""
    from phenoradar.cv import _build_top_feature_expression_from_matrix
    from phenoradar.missing_expression import mark_expression_observations

    cv_coefficients = (
        _scan(run_dir / "cv/tables", "expression_coefficients")
        .filter(pl.col("feature").is_in(importance["feature"].to_list()))
        .collect(engine="streaming")
    )
    pools = [("cv", oof, importance, cv_coefficients, "label")]
    if external is not None and final_importance is not None and final_coefficients is not None:
        expression_keys = [
            name for name in final_importance["feature"] if json.loads(name)[0] == "expression"
        ]
        final_top = (
            final_importance.filter(pl.col("feature").is_in(expression_keys))
            .sort(["importance_mean", "feature"], descending=[True, False])
            .head(config.figures.top_features)
        )
        pools.append(("external_test", external, final_top, final_coefficients, "true_label"))
    for stage, predictions, top, coefficients, label in pools:
        if top.is_empty() or predictions.is_empty():
            continue
        names = top["feature"].to_list()
        species = predictions["species"].to_list()
        matrix = expression.matrix(species, [json.loads(name)[1] for name in names])
        raw = _build_top_feature_expression_from_matrix(
            species_order=species,
            matrix=matrix,
            feature_names=names,
            feature_importance=top,
            feature_limit=config.figures.top_features,
        )
        missing = config.preprocess.expression.missing_expression
        if missing.method == "neutral":
            raw = mark_expression_observations(raw, zero_as_missing=missing.zero_as_missing)
        raw.write_csv(run_dir / stage / "tables/top_feature_expression.tsv", separator="\t")
        figures._top_feature_expression_by_confusion(
            oof_predictions=predictions,
            top_feature_expression=raw,
            feature_importance=top,
            coefficients=coefficients,
            out_path=run_dir / stage / "figures/top_feature_expression_by_confusion.svg",
            top_features=config.figures.top_features,
            orthogroup_annotations=annotations,
            label_col=label,
        )
