"""Expression-equivalent figures from bounded slices of amino-acid output tables."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from phenoradar import figures
from phenoradar.config import AppConfig

if TYPE_CHECKING:
    from phenoradar.amino_acid_reporting import AAReportWriter
    from phenoradar.amino_acid_run import AAMember


def _read(directory: Path, name: str) -> pl.DataFrame | None:
    path = directory / f"{name}.tsv"
    if not path.exists():
        return None
    return pl.read_csv(path, separator="\t", infer_schema_length=10000,
                       schema_overrides={"fold_id": pl.String})


def _scan(directory: Path, name: str) -> pl.LazyFrame:
    return pl.scan_csv(directory / f"{name}.tsv", separator="\t", infer_schema_length=10000,
                       schema_overrides={"fold_id": pl.String})


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
        pl.when(pl.col("scope") == "final_refit").then(pl.lit("final_refit"))
        .otherwise(pl.lit("outer_fold")).alias("selection_scope"),
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
        pl.DataFrame({**common, "importance_mean": importance.mean(axis=0),
                      "importance_std": importance.std(axis=0), "method": "coef_abs_l1_norm"}),
        pl.DataFrame({**common, "coef_mean": beta.mean(axis=0), "coef_std": beta.std(axis=0),
                      "method": "coef_signed", "reason": "NA"}),
    )


def write_amino_acid_figures(
    run_dir: Path, config: AppConfig, members: list[AAMember] | None = None,
    *, report: AAReportWriter | None = None,
) -> list[str]:
    """Reuse common plots, without loading millions of zero coefficients into RAM."""
    cv = run_dir / "cv/tables"
    model = run_dir / "model/tables"
    limit = config.figures.top_features
    importance = _scan(cv, "feature_importance").sort(
        ["importance_mean", "feature"], descending=[True, False]
    ).head(limit).collect(engine="streaming")
    coefficients = _scan(cv, "coefficients").with_columns(
        pl.col("coef_mean").abs().alias("_abs")
    ).sort(["_abs", "feature"], descending=[True, False]).head(limit).drop("_abs").collect(
        engine="streaming")
    requested = sorted(set(importance["feature"]) | set(coefficients["feature"]))
    if report is None:
        by_fold = {
            name: _scan(cv, name).filter(pl.col("feature").is_in(requested)).collect(
                engine="streaming")
            for name in ["feature_importance_by_fold", "coefficients_by_fold"]
        }
        means = None
    else:
        means = pl.scan_parquet(report.path / "means" / "*.parquet")
        fold_model_counts: dict[str, int] = {}
        for fold, _, _ in report.models:
            fold_model_counts[fold] = fold_model_counts.get(fold, 0) + 1
        axes = pl.DataFrame({
            "fold_id": list(fold_model_counts), "n_models": list(fold_model_counts.values()),
        }).lazy()
        axes = axes.join(pl.DataFrame({"feature": requested}).lazy(), how="cross")
        selected_means = means.filter(pl.col("feature").is_in(requested))
        grid = axes.join(selected_means, on=["fold_id", "feature"], how="left")
        by_fold = {}
        for name, column, label, method in [
            ("feature_importance_by_fold", "importance", "importance_mean", "coef_abs_l1_norm"),
            ("coefficients_by_fold", "coefficient", "coef_mean", "coef_signed"),
        ]:
            frame = grid.select("fold_id", "feature", "n_models",
                                pl.col(column).fill_null(0).alias(label),
                                pl.lit(method).alias("method"))
            if column == "coefficient":
                frame = frame.with_columns(pl.lit("NA").alias("reason"))
            by_fold[name] = frame.collect(engine="streaming")
    # The saved table already has deterministic descending stability order.
    stability = _scan(cv, "feature_stability_by_feature").head(limit).collect(engine="streaming")
    selected = _selection(cv, "model_selection_selected")
    final_selected = _selection(model, "model_selection_selected")
    if final_selected is not None:
        selected = (final_selected if selected is None else
                    pl.concat([selected, final_selected], how="diagonal_relaxed"))
    final_trials = _selection(model, "model_selection_trials")
    final_summary = (None if final_trials is None else
                     figures._summarize_model_selection_trials_for_figure(final_trials))
    final_importance, final_coefficients = _final_features(members) if members else (None, None)
    final_null = final_importance is not None and final_importance.height == 0
    if final_null:
        # A valid intercept-only bundle has no feature rows after compaction.
        final_importance, final_coefficients = None, None
    oof = _read(cv, "prediction_cv")
    assert oof is not None
    coefficient_source = (_scan(cv, "coefficients_by_fold") if means is None else
                          means.rename({"coefficient": "coef_mean"}))
    sparsity = coefficient_source.group_by("fold_id").agg(
        (pl.col("coef_mean").abs() > 1e-12).sum().alias("n_nonzero_features")
    ).with_columns(pl.lit("outer_fold").alias("scope")).collect(engine="streaming")
    metrics = _read(cv, "metrics_cv")
    assert metrics is not None
    warnings = figures.write_run_figures(
        run_dir=run_dir, metrics_cv=metrics, oof_predictions=oof,
        feature_importance=importance, coefficients=coefficients, ensemble_model_probs=None,
        model_selection_trials=_selection(cv, "model_selection_trials"),
        model_selection_selected=selected, feature_importance_by_fold=by_fold[
            "feature_importance_by_fold"], coefficients_by_fold=by_fold["coefficients_by_fold"],
        feature_stability_by_feature=stability,
        feature_stability_by_fold_pair=_read(cv, "feature_stability_by_fold_pair"),
        loss_by_split_cv=_read(cv, "loss_by_split_cv"), model_sparsity=sparsity,
        final_refit_feature_importance=final_importance,
        final_refit_coefficients=final_coefficients,
        final_refit_model_selection_trials_summary=final_summary,
        group_bootstrap_metrics=_read(cv, "group_bootstrap_metrics"),
        pred_external_test=_read(run_dir / "external_test/tables", "pred_external_test"),
        pred_inference=_read(run_dir / "inference/tables", "pred_inference"),
        trait_name=config.data.trait_col, top_features=limit,
    )
    if final_null:
        for name, title in [
            ("final_refit_feature_importance_top", "Final Refit Feature Importance"),
            ("final_refit_coefficients_signed_top", "Final Refit Coefficients"),
        ]:
            figures._write_message_figure(
                title=title, message="Intercept-only model: no nonzero features.",
                out_path=run_dir / "model/figures" / (name + ".svg"),
                width_px=600, height_px=220,
            )
    # Site scores are aggregated after filtering zeros; denominator includes all
    # models, including sites absent from individual fitted feature schemas.
    counts = _selection(cv, "model_selection_selected")
    assert counts is not None
    n_models = counts.group_by("scope").len().lazy()
    for metric in ["pss", "effect_range"]:
        site_source = (_scan(cv, f"amino_acid_site_{metric}") if report is None
                       else report.sites.scan())
        top = site_source.filter(pl.col(metric) > 0).group_by(
            "scope", "orthogroup", "alignment_pos_1based"
        ).agg(pl.col(metric).sum()).join(n_models, on="scope").with_columns(
            (pl.col(metric) / pl.col("len")).alias(metric)
        ).group_by("orthogroup", "alignment_pos_1based").agg(
            (pl.col(metric).sum() / counts["scope"].n_unique()).alias(metric)
        ).sort([metric, "orthogroup", "alignment_pos_1based"],
               descending=[True, False, False]).head(limit).collect(engine="streaming")
        path = run_dir / "cv/figures" / f"amino_acid_site_{metric}_top.svg"
        _site_bar(top, metric, path)
    _frequency_by_confusion(run_dir, oof, limit)
    _site_filter_funnel(run_dir)
    if config.data.feature_mode == "combined":
        from phenoradar.combined_figures import write_combined_figures

        write_combined_figures(run_dir, limit, final_importance=final_importance,
                               final_coefficients=final_coefficients)
    return warnings


def _frequency_by_confusion(run_dir: Path, oof: pl.DataFrame, limit: int) -> None:
    import matplotlib.pyplot as plt

    data = _read(run_dir / "cv/tables", "top_amino_acid_frequencies")
    if data is None:
        return
    data = data.join(oof.select("species", "label", "prob"), on="species")
    names = data["feature"].unique(maintain_order=True).head(limit).to_list()
    if not names:
        return
    figure, axes = plt.subplots(
        len(names), 1, figsize=(7, max(2, 1.4 * len(names))), squeeze=False
    )
    groups = [("TN", 0, False), ("FP", 0, True), ("FN", 1, False), ("TP", 1, True)]
    for axis, name in zip(axes[:, 0], names, strict=True):
        rows = data.filter((pl.col("feature") == name) & pl.col("frequency").is_finite())
        values = [rows.filter((pl.col("label") == label) & ((pl.col("prob") >= .5) == pred))[
            "frequency"].to_numpy() for _, label, pred in groups]
        axis.boxplot(values, tick_labels=[g[0] for g in groups], widths=.6)
        axis.set_ylim(-.05, 1.05)
        axis.set_ylabel("Frequency")
        axis.set_title(figures._feature_axis_labels([name], None)[0], fontsize=9)
    figure.tight_layout()
    figures._save_svg_figure(figure, run_dir / "cv/figures/amino_acid_frequency_by_confusion.svg")


def _site_filter_funnel(run_dir: Path) -> None:
    cv = run_dir / "cv/tables"
    raw = _scan(cv, "amino_acid_og_qc").group_by("scope", "sample_set_id").agg(
        pl.col("n_columns").sum(), pl.col("n_after_call_filter").sum(),
        pl.col("n_after_variant_filter").sum(),
    ).collect(engine="streaming")
    path = cv / "amino_acid_site_qc.tsv"
    if path.exists():
        retained = _scan(cv, "amino_acid_site_qc").group_by("scope", "sample_set_id").agg(
            pl.len().alias("n_retained_sites")
        ).collect(engine="streaming")
        raw = raw.join(retained, on=["scope", "sample_set_id"], how="left").with_columns(
            pl.col("n_retained_sites").fill_null(0))
    else:
        raw = raw.with_columns(pl.lit(0).alias("n_retained_sites"))
    stages = ["alignment_sites", "observed_sites", "variable_sites", "retained_sites"]
    rows = []
    for stage, column in zip(stages, ["n_columns", "n_after_call_filter", "n_after_variant_filter",
                                      "n_retained_sites"],
                             strict=True):
        values = raw[column].to_numpy()
        quantiles = np.quantile(values, [0, .25, .5, .75, 1])
        rows.append({"scope": "outer_fold", "stage": stage, "n_records": len(values),
                     **dict(zip(["n_features_min", "n_features_q1", "n_features_median",
                                 "n_features_q3", "n_features_max"], quantiles, strict=True))})
    summary = pl.DataFrame(rows)
    summary.write_csv(cv / "amino_acid_site_filter_counts_summary.tsv", separator="\t")
    figures._feature_filter_funnel(
        summary, run_dir / "cv/figures/amino_acid_site_filter_funnel.svg", stage_order=stages,
    )


def _site_bar(frame: pl.DataFrame, metric: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{og}: {position}" for og, position in
              frame.select("orthogroup", "alignment_pos_1based").iter_rows()]
    figure, axis = plt.subplots(figsize=(7, max(2, len(labels) * .24)))
    axis.barh(np.arange(len(labels)), frame[metric].to_numpy(), color="#4477aa")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.invert_yaxis()
    label = "PSS" if metric == "pss" else "R (coefficient range)"
    axis.set_xlabel(f"Mean site {label} across outer folds (absent sites = 0)")
    axis.set_ylabel("Orthogroup / alignment site\n(input MSA, 1-based)")
    figure.tight_layout()
    figures._save_svg_figure(figure, path)
