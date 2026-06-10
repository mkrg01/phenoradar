from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import polars as pl
import pytest
from matplotlib.colors import to_hex

import phenoradar.figures as figures_mod
from phenoradar.figures import (
    FigureError,
    write_predict_figures,
    write_report_figures,
    write_run_figures,
)


def _svg_text_y_and_viewbox_height(svg_path: Path, text: str) -> tuple[float, float]:
    root = ET.parse(svg_path).getroot()
    viewbox_height = float(root.attrib["viewBox"].split()[3])
    for element in root.iter():
        if element.tag.endswith("text") and "".join(element.itertext()) == text:
            return float(element.attrib["y"]), viewbox_height
    raise AssertionError(f"{text!r} not found in {svg_path}")


def _minimal_metrics_cv() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "aggregate_scope": "NA",
                "fold_id": "0",
                "metric": "mcc",
                "metric_value": 0.5,
                "n_pos": 1,
                "n_neg": 1,
                "n_valid_folds": None,
            },
            {
                "aggregate_scope": "macro",
                "fold_id": "NA",
                "metric": "mcc",
                "metric_value": 0.5,
                "n_pos": 1,
                "n_neg": 1,
                "n_valid_folds": 1,
            },
            {
                "aggregate_scope": "micro",
                "fold_id": "NA",
                "metric": "mcc",
                "metric_value": 0.5,
                "n_pos": 1,
                "n_neg": 1,
                "n_valid_folds": 1,
            },
        ]
    )


def test_write_predict_figures_requires_uncertainty_when_requested(tmp_path) -> None:
    with pytest.raises(FigureError):
        write_predict_figures(
            run_dir=tmp_path / "predict_run",
            pred_predict=pl.DataFrame(
                {
                    "species": ["sp1", "sp2"],
                    "prob": [0.1, 0.9],
                    "pred_label_fixed_threshold": [0, 1],
                }
            ),
            require_uncertainty=True,
        )


def test_write_run_figures_does_not_emit_ensemble_uncertainty(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=pl.DataFrame(
            {
                "fold_id": ["0"],
                "model_index": [0],
                "species": ["sp1"],
                "prob": [0.2],
            }
        ),
        model_selection_trials=None,
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    assert not (figures_dir / "ensemble_uncertainty.svg").exists()
    assert warnings == []


def _minimal_oof() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fold_id": ["0", "0", "1", "1"],
            "species": ["sp1", "sp2", "sp3", "sp4"],
            "label": [0, 1, 0, 1],
            "prob": [0.2, 0.8, 0.3, 0.7],
        }
    )


def _minimal_loss_by_split() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fold_id": ["0", "0", "1", "1"],
            "split": ["train", "validation", "train", "validation"],
            "metric": ["log_loss", "log_loss", "log_loss", "log_loss"],
            "metric_value": [0.30, 0.45, 0.35, 0.50],
        }
    )


def _minimal_final_refit_loss_by_split() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "split": ["train", "external_test"],
            "metric": ["log_loss", "log_loss"],
            "metric_value": [0.32, 0.51],
        }
    )


def _minimal_classification_summary() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pool": ["validation_oof", "external_test"],
            "fold_id": ["NA", "NA"],
            "threshold_name": [
                "fixed_probability_threshold",
                "fixed_probability_threshold",
            ],
            "threshold_value": [0.5, 0.5],
            "n_total": [4, 2],
            "tp": [2, 1],
            "fp": [0, 0],
            "tn": [2, 1],
            "fn": [0, 0],
            "accuracy": [1.0, 1.0],
            "precision": [1.0, 1.0],
            "recall": [1.0, 1.0],
            "f1": [1.0, 1.0],
            "mcc": [1.0, 1.0],
        }
    )


def _minimal_feature_importance() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["OG1", "OG2"],
            "importance_mean": [0.7, 0.3],
            "importance_std": [0.0, 0.0],
            "n_models": [1, 1],
            "method": ["coef_abs_l1_norm", "coef_abs_l1_norm"],
        }
    )


def _minimal_feature_importance_by_fold() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fold_id": ["0", "1", "0", "1"],
            "feature": ["OG1", "OG1", "OG2", "OG2"],
            "importance_mean": [0.8, 0.6, 0.2, 0.4],
            "n_models": [1, 1, 1, 1],
            "method": ["coef_abs_l1_norm"] * 4,
        }
    )


def _minimal_feature_filter_counts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scope": ["outer_fold", "outer_fold", "outer_fold", "outer_fold"],
            "fold_id": ["0", "0", "1", "1"],
            "sample_set_id": [0, 1, 0, 1],
            "n_features_before": [100, 100, 100, 100],
            "n_features_after_sparse_feature_filter": [80, 82, 78, 79],
            "n_features_after_low_variance": [60, 61, 59, 60],
            "n_features_after_correlation": [52, 53, 50, 52],
            "n_features_after_all": [52, 53, 50, 52],
        }
    )


def _minimal_feature_filter_counts_summary() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scope": ["outer_fold"] * 6,
            "stage": [
                "n_features_before",
                "n_features_after_sparse_feature_filter",
                "n_features_after_low_variance",
                "n_features_after_pair_aware",
                "n_features_after_correlation",
                "n_features_after_all",
            ],
            "n_records": [4, 4, 4, 4, 4, 4],
            "n_features_min": [100, 78, 59, 58, 50, 50],
            "n_features_q1": [100.0, 78.75, 59.75, 58.5, 51.5, 51.5],
            "n_features_median": [100.0, 79.5, 60.0, 59.0, 52.0, 52.0],
            "n_features_mean": [100.0, 79.75, 60.0, 59.0, 51.75, 51.75],
            "n_features_q3": [100.0, 80.5, 60.25, 59.5, 52.25, 52.25],
            "n_features_max": [100, 82, 61, 60, 53, 53],
            "retained_ratio_min": [1.0, 0.78, 0.59, 0.58, 0.50, 0.50],
            "retained_ratio_q1": [1.0, 0.7875, 0.5975, 0.585, 0.515, 0.515],
            "retained_ratio_median": [1.0, 0.795, 0.60, 0.59, 0.52, 0.52],
            "retained_ratio_mean": [1.0, 0.7975, 0.60, 0.59, 0.5175, 0.5175],
            "retained_ratio_q3": [1.0, 0.805, 0.6025, 0.595, 0.5225, 0.5225],
            "retained_ratio_max": [1.0, 0.82, 0.61, 0.60, 0.53, 0.53],
        }
    )


def _minimal_model_sparsity() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scope": ["outer_fold", "outer_fold", "outer_fold", "outer_fold"],
            "fold_id": ["0", "0", "1", "1"],
            "sample_set_id": [0, 1, 0, 1],
            "model_index": [0, 1, 2, 3],
            "model_name": ["logistic_elasticnet"] * 4,
            "n_features_after_all": [52, 53, 50, 52],
            "n_nonzero_features": [24, 25, 21, 22],
            "nonzero_ratio": [24 / 52, 25 / 53, 21 / 50, 22 / 52],
            "count_method": ["coef_abs_gt_tol"] * 4,
            "reason": ["ok"] * 4,
        }
    )


def test_format_float_returns_nan_for_none_and_nan() -> None:
    assert figures_mod._format_float(None) == "NaN"
    assert figures_mod._format_float(float("nan")) == "NaN"


def test_place_x_axis_at_zero_moves_bottom_spine() -> None:
    fig, ax = figures_mod.plt.subplots()
    try:
        figures_mod._place_x_axis_at_zero(ax)

        assert ax.spines["bottom"].get_position() == ("data", 0.0)
    finally:
        figures_mod.plt.close(fig)


def _minimal_coefficients() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["OG1", "OG2"],
            "coef_mean": [0.2, -0.1],
            "coef_std": [0.0, 0.0],
            "n_models": [1, 1],
            "method": ["coef_signed", "coef_signed"],
            "reason": ["NA", "NA"],
        }
    )


def test_write_predict_figures_writes_uncertainty_when_available(tmp_path: Path) -> None:
    write_predict_figures(
        run_dir=tmp_path / "predict_run",
        pred_predict=pl.DataFrame(
            {
                "species": ["sp1", "sp2"],
                "prob": [0.1, 0.9],
                "pred_label_fixed_threshold": [0, 1],
                "uncertainty_std": [0.01, 0.02],
            }
        ),
        require_uncertainty=True,
    )

    figures_dir = tmp_path / "predict_run" / "inference" / "figures"
    assert (figures_dir / "predict_probability_distribution.svg").exists()
    assert (figures_dir / "predict_uncertainty.svg").exists()


def test_write_predict_figures_skips_uncertainty_when_not_required(tmp_path: Path) -> None:
    write_predict_figures(
        run_dir=tmp_path / "predict_run",
        pred_predict=pl.DataFrame(
            {
                "species": ["sp1", "sp2"],
                "prob": [0.1, 0.9],
                "pred_label_fixed_threshold": [0, 1],
            }
        ),
        require_uncertainty=False,
    )

    figures_dir = tmp_path / "predict_run" / "inference" / "figures"
    assert (figures_dir / "predict_probability_distribution.svg").exists()
    assert not (figures_dir / "predict_uncertainty.svg").exists()


def test_write_run_figures_writes_required_artifacts(tmp_path: Path) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        feature_importance_by_fold=_minimal_feature_importance_by_fold(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        loss_by_split_cv=_minimal_loss_by_split(),
    )

    cv_figures_dir = tmp_path / "run" / "cv" / "figures"
    external_figures_dir = tmp_path / "run" / "external_test" / "figures"
    inference_figures_dir = tmp_path / "run" / "inference" / "figures"
    assert cv_figures_dir.is_dir()
    assert external_figures_dir.is_dir()
    assert inference_figures_dir.is_dir()
    assert (cv_figures_dir / "cv_metrics_overview.svg").exists()
    cv_loss_path = cv_figures_dir / "cv_loss_by_split.svg"
    assert cv_loss_path.exists()
    label_y, viewbox_height = _svg_text_y_and_viewbox_height(cv_loss_path, "Log loss")
    assert label_y < viewbox_height
    assert (cv_figures_dir / "feature_importance_top.svg").exists()
    assert (cv_figures_dir / "feature_importance_by_fold_heatmap.svg").exists()
    assert (cv_figures_dir / "coefficients_signed_top.svg").exists()
    assert (cv_figures_dir / "cv_species_probability_by_trait.svg").exists()
    assert (cv_figures_dir / "cv_fold_trait_probability.svg").exists()
    assert (cv_figures_dir / "roc_curve_cv.svg").exists()
    assert (cv_figures_dir / "pr_curve_cv.svg").exists()
    assert not (external_figures_dir / "final_refit_loss_by_split.svg").exists()
    assert not (external_figures_dir / "external_species_probability_by_trait.svg").exists()
    assert not (inference_figures_dir / "inference_probability_distribution.svg").exists()
    assert warnings == []


def test_write_run_figures_writes_feature_filter_and_sparsity_figures(tmp_path: Path) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        feature_filter_counts_summary=_minimal_feature_filter_counts_summary(),
        feature_filter_funnel_stage_order=[
            "n_features_before",
            "n_features_after_sparse_feature_filter",
            "n_features_after_pair_aware",
        ],
        model_sparsity=_minimal_model_sparsity(),
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    assert (figures_dir / "feature_filter_funnel.svg").exists()
    funnel_svg = (figures_dir / "feature_filter_funnel.svg").read_text(encoding="utf-8")
    assert "Feature selection step" in funnel_svg
    assert "Number of selected features" in funnel_svg
    assert "Number of features" not in funnel_svg
    assert "Feature Count" not in funnel_svg
    assert "n=4" not in funnel_svg
    assert "median" in funnel_svg
    assert "IQR (25-75%)" in funnel_svg
    assert "min-max" in funnel_svg
    assert "outer_fold (median" not in funnel_svg
    assert "Input" in funnel_svg
    assert "Sparse feature" in funnel_svg
    assert "Pair aware" in funnel_svg
    assert "sparse_feature" not in funnel_svg
    assert "Low variance" not in funnel_svg
    assert "Correlation" not in funnel_svg
    assert "Final" not in funnel_svg
    assert "79.5" in funnel_svg
    assert not (figures_dir / "selected_features_by_fold_after_preprocessing.svg").exists()
    assert not (figures_dir / "selected_features_after_preprocessing.svg").exists()
    assert not (figures_dir / "selected_features_by_fold.svg").exists()
    assert (figures_dir / "non_zero_feature_count_by_fold.svg").exists()
    assert not (figures_dir / "selected_feature_count_by_fold.svg").exists()
    count_svg = (figures_dir / "non_zero_feature_count_by_fold.svg").read_text(encoding="utf-8")
    assert "Number of non-zero features per model" in count_svg
    assert not (figures_dir / "model_sparsity_scatter.svg").exists()
    assert warnings == []


def test_write_run_figures_writes_external_trait_probability_when_available(tmp_path: Path) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        loss_by_split_final_refit=_minimal_final_refit_loss_by_split(),
        classification_summary=_minimal_classification_summary(),
        pred_external_test=pl.DataFrame(
            {
                "species": ["sp5", "sp6"],
                "true_label": [0, 1],
                "prob": [0.3, 0.7],
                "pred_label_fixed_threshold": [0, 1],
            }
        ),
        pred_inference=pl.DataFrame(
            {
                "species": ["sp7", "sp8", "sp9"],
                "true_label": [None, None, None],
                "prob": [0.1, 0.55, 0.9],
            }
        ),
    )

    external_figures_dir = tmp_path / "run" / "external_test" / "figures"
    inference_figures_dir = tmp_path / "run" / "inference" / "figures"
    final_refit_loss_path = external_figures_dir / "final_refit_loss_by_split.svg"
    assert final_refit_loss_path.exists()
    label_y, viewbox_height = _svg_text_y_and_viewbox_height(final_refit_loss_path, "Log Loss")
    assert label_y < viewbox_height
    assert (external_figures_dir / "external_species_probability_by_trait.svg").exists()
    assert (external_figures_dir / "external_confusion_matrix.svg").exists()
    assert (external_figures_dir / "external_roc_curve.svg").exists()
    assert (external_figures_dir / "external_pr_curve.svg").exists()
    comparison_path = external_figures_dir / "cv_external_metric_comparison.svg"
    assert comparison_path.exists()
    comparison_svg = comparison_path.read_text(encoding="utf-8")
    assert "Validation OOF" in comparison_svg
    assert "External test" in comparison_svg
    assert "MCC" in comparison_svg
    assert (inference_figures_dir / "inference_probability_distribution.svg").exists()
    assert not (tmp_path / "run" / "figures" / "cv_metrics_overview.svg").exists()
    assert warnings == []


def test_write_report_figures_stage_breakdown_is_conditional(tmp_path: Path) -> None:
    report_dir_single = tmp_path / "report_single"
    write_report_figures(
        report_dir=report_dir_single,
        report_runs=pl.DataFrame(
            {
                "run_id": ["r1"],
                "metric_value": [0.9],
                "start_time": ["2026-01-01T00:00:00+00:00"],
                "execution_stage": ["full_run"],
            }
        ),
        report_ranking=pl.DataFrame(
            {
                "rank": [1],
                "run_id": ["r1"],
                "metric_value": [0.9],
            }
        ),
    )
    assert not (report_dir_single / "figures" / "report_stage_breakdown.svg").exists()

    report_dir_multi = tmp_path / "report_multi"
    write_report_figures(
        report_dir=report_dir_multi,
        report_runs=pl.DataFrame(
            {
                "run_id": ["r1", "r2"],
                "metric_value": [0.9, 0.8],
                "start_time": ["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"],
                "execution_stage": ["full_run", "predict"],
            }
        ),
        report_ranking=pl.DataFrame(
            {
                "rank": [1, 2],
                "run_id": ["r1", "r2"],
                "metric_value": [0.9, 0.8],
            }
        ),
    )
    assert (report_dir_multi / "figures" / "report_stage_breakdown.svg").exists()


def test_write_report_figures_rejects_invalid_ranking_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError):
        write_report_figures(
            report_dir=tmp_path / "report",
            report_runs=pl.DataFrame(
                {
                    "run_id": ["r1"],
                    "metric_value": [0.9],
                    "start_time": ["2026-01-01T00:00:00+00:00"],
                    "execution_stage": ["full_run"],
                }
            ),
            report_ranking=pl.DataFrame(
                {
                    "run_id": ["r1"],
                    "metric_value": [0.9],
                }
            ),
        )


def test_write_run_figures_ignores_empty_model_selection_trials_when_provided(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=pl.DataFrame(
            schema={
                "fold_id": pl.String,
                "sample_set_id": pl.Int64,
                "candidate_index": pl.Int64,
                "inner_fold_id": pl.String,
                "metric_name": pl.String,
                "metric_value": pl.Float64,
            }
        ),
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    assert not (figures_dir / "model_selection_trials.svg").exists()
    assert warnings == []


def test_write_run_figures_writes_model_selection_trials_when_summary_provided(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        model_selection_trials_summary=pl.DataFrame(
            {
                "fold_id": ["0", "0", "1", "1"],
                "sample_set_id": [0, 0, 0, 0],
                "candidate_index": [0, 1, 0, 1],
                "metric_name": ["mcc", "mcc", "mcc", "mcc"],
                "params_json": ["{}", "{\"C\":1.0}", "{}", "{\"C\":1.0}"],
                "n_inner_folds": [2, 2, 2, 2],
                "n_valid_inner_folds": [2, 2, 2, 2],
                "metric_value_mean": [0.40, 0.55, 0.38, 0.52],
                "metric_value_std": [0.02, 0.03, 0.01, 0.02],
            }
        ),
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    svg_text = (figures_dir / "model_selection_trials.svg").read_text(encoding="utf-8")
    assert (figures_dir / "model_selection_trials.svg").exists()
    assert '1: {"C":1.0}' in svg_text
    assert warnings == []


def test_write_run_figures_uses_log_loss_axis_label_for_model_selection_trials(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        model_selection_trials_summary=pl.DataFrame(
            {
                "fold_id": ["0", "0"],
                "sample_set_id": [0, 0],
                "candidate_index": [0, 1],
                "metric_name": ["log_loss", "log_loss"],
                "params_json": ["{}", "{\"C\":1.0}"],
                "n_inner_folds": [2, 2],
                "n_valid_inner_folds": [2, 2],
                "metric_value_mean": [0.40, 0.55],
                "metric_value_std": [0.02, 0.03],
            }
        ),
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    svg_text = (figures_dir / "model_selection_trials.svg").read_text(encoding="utf-8")
    assert "Log Loss" in svg_text
    assert warnings == []


def test_write_run_figures_hides_fixed_params_in_model_selection_labels(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        model_selection_trials_summary=pl.DataFrame(
            {
                "fold_id": ["0", "0"],
                "sample_set_id": [0, 0],
                "candidate_index": [0, 1],
                "metric_name": ["mcc", "mcc"],
                "params_json": [
                    "{\"C\":1.0,\"l1_ratio\":0.5}",
                    "{\"C\":2.0,\"l1_ratio\":0.5}",
                ],
                "n_inner_folds": [2, 2],
                "n_valid_inner_folds": [2, 2],
                "metric_value_mean": [0.41, 0.58],
                "metric_value_std": [0.02, 0.03],
            }
        ),
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    svg_text = (figures_dir / "model_selection_trials.svg").read_text(encoding="utf-8")
    assert '0: {"C":1.0}' in svg_text
    assert '1: {"C":2.0}' in svg_text
    assert "l1_ratio" not in svg_text
    assert warnings == []


def test_write_run_figures_writes_one_se_model_selection_figure(
    tmp_path: Path,
) -> None:
    summary = pl.DataFrame(
        {
            "fold_id": ["0", "0", "0", "1", "1", "1"],
            "sample_set_id": [0, 0, 0, 0, 0, 0],
            "candidate_index": [0, 1, 2, 0, 1, 2],
            "metric_name": ["log_loss"] * 6,
            "params_json": [
                "{\"C\":0.01}",
                "{\"C\":0.1}",
                "{\"C\":1.0}",
                "{\"C\":0.01}",
                "{\"C\":0.1}",
                "{\"C\":1.0}",
            ],
            "n_inner_folds": [2] * 6,
            "n_valid_inner_folds": [2] * 6,
            "metric_value_mean": [0.40, 0.23, 0.20, 0.35, 0.24, 0.21],
            "metric_value_std": [0.01, 0.01, 0.07, 0.01, 0.01, 0.08],
            "metric_value_se": [0.007, 0.007, 0.049, 0.007, 0.007, 0.057],
        }
    )
    selected = pl.DataFrame(
        {
            "selection_scope": ["outer_fold", "outer_fold", "final_refit"],
            "fold_id": ["0", "1", "NA"],
            "sample_set_id": [0, 0, 0],
            "selection_source_sample_set_id": [0, 0, 0],
            "rank": [1, 1, 1],
            "candidate_index": [1, 1, 1],
            "metric_name": ["log_loss", "log_loss", "log_loss"],
            "metric_value": [0.23, 0.24, 0.22],
            "metric_value_se": [0.007, 0.007, 0.006],
            "selection_rule": ["one_se", "one_se", "one_se"],
            "n_available_candidates": [3, 3, 3],
            "n_scored_candidates": [3, 3, 3],
            "selected_candidate_count_requested": [1, 1, 1],
            "selected_candidate_count_effective": [1, 1, 1],
            "params_json": ["{\"C\":0.1}", "{\"C\":0.1}", "{\"C\":0.1}"],
        }
    )

    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        model_selection_trials_summary=summary,
        model_selection_selected=selected,
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    one_se_svg = (figures_dir / "model_selection_one_se_curve.svg").read_text(
        encoding="utf-8"
    )
    assert "one-SE threshold" in one_se_svg
    assert "Selected candidate" in one_se_svg
    assert "log10(C)" in one_se_svg
    assert not (figures_dir / "selected_hyperparameter_stability.svg").exists()
    assert warnings == []


def test_write_run_figures_limits_model_selection_sample_sets_per_fold(
    tmp_path: Path,
) -> None:
    summary_rows = []
    for sample_set_id in range(7):
        for candidate_index in (0, 1):
            summary_rows.append(
                {
                    "fold_id": "0",
                    "sample_set_id": sample_set_id,
                    "candidate_index": candidate_index,
                    "metric_name": "mcc",
                    "params_json": (
                        "{\"panel_param\":"
                        f"{sample_set_id * 10 + candidate_index}"
                        "}"
                    ),
                    "n_inner_folds": 2,
                    "n_valid_inner_folds": 2,
                    "metric_value_mean": 0.3 + sample_set_id * 0.01 + candidate_index * 0.02,
                    "metric_value_std": 0.01,
                }
            )
    summary = pl.DataFrame(summary_rows)

    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        model_selection_trials_summary=summary,
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    svg_text = (figures_dir / "model_selection_trials.svg").read_text(encoding="utf-8")
    assert "fold=0" not in svg_text
    assert '0: {"panel_param":0}' in svg_text
    assert '"panel_param":10' not in svg_text
    assert warnings == []


def test_write_run_figures_ignores_empty_ensemble_inputs(tmp_path: Path) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=pl.DataFrame(
            schema={
                "fold_id": pl.String,
                "model_index": pl.Int64,
                "species": pl.String,
                "prob": pl.Float64,
            }
        ),
        model_selection_trials=None,
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    assert not (figures_dir / "ensemble_uncertainty.svg").exists()
    assert warnings == []


def test_write_run_figures_collects_warning_when_roc_curves_cannot_be_drawn(tmp_path: Path) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=pl.DataFrame(
            {
                "fold_id": ["0", "0", "1", "1"],
                "species": ["sp1", "sp2", "sp3", "sp4"],
                "label": [1, 1, 1, 1],
                "prob": [0.2, 0.8, 0.3, 0.7],
            }
        ),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
    )

    assert any("no folds with both labels" in warning for warning in warnings)


def test_write_run_figures_skips_external_curves_for_single_class_external_test(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        pred_external_test=pl.DataFrame(
            {
                "species": ["sp5", "sp6"],
                "true_label": [1, 1],
                "prob": [0.7, 0.9],
                "pred_label_fixed_threshold": [1, 1],
            }
        ),
    )

    external_figures_dir = tmp_path / "run" / "external_test" / "figures"
    assert (external_figures_dir / "external_confusion_matrix.svg").exists()
    assert (external_figures_dir / "external_species_probability_by_trait.svg").exists()
    assert not (external_figures_dir / "external_roc_curve.svg").exists()
    assert not (external_figures_dir / "external_pr_curve.svg").exists()
    assert any("external_test requires both labels" in warning for warning in warnings)


def test_write_run_figures_rejects_invalid_metrics_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="metrics_cv.tsv schema is invalid"):
        write_run_figures(
            run_dir=tmp_path / "run",
            metrics_cv=pl.DataFrame({"aggregate_scope": ["macro"]}),
            oof_predictions=_minimal_oof(),
            feature_importance=_minimal_feature_importance(),
            coefficients=_minimal_coefficients(),
            ensemble_model_probs=None,
            model_selection_trials=None,
        )


def test_write_run_figures_rejects_metrics_without_aggregate_rows(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="does not contain macro/micro aggregate rows"):
        write_run_figures(
            run_dir=tmp_path / "run",
            metrics_cv=pl.DataFrame(
                {
                    "aggregate_scope": ["NA"],
                    "fold_id": ["0"],
                    "metric": ["mcc"],
                    "metric_value": [0.5],
                }
            ),
            oof_predictions=_minimal_oof(),
            feature_importance=_minimal_feature_importance(),
            coefficients=_minimal_coefficients(),
            ensemble_model_probs=None,
            model_selection_trials=None,
        )


def test_write_predict_figures_rejects_empty_prediction_table(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="prediction_inference.tsv is empty"):
        write_predict_figures(
            run_dir=tmp_path / "predict_run",
            pred_predict=pl.DataFrame(
                schema={
                    "species": pl.String,
                    "prob": pl.Float64,
                    "pred_label_fixed_threshold": pl.Int64,
                }
            ),
            require_uncertainty=False,
        )


def test_write_report_figures_rejects_invalid_run_schema_for_metric_comparison(
    tmp_path: Path,
) -> None:
    with pytest.raises(FigureError, match="report_runs.tsv schema is invalid"):
        write_report_figures(
            report_dir=tmp_path / "report",
            report_runs=pl.DataFrame(
                {
                    "run_id": ["r1"],
                    "start_time": ["2026-01-01T00:00:00+00:00"],
                    "execution_stage": ["full_run"],
                }
            ),
            report_ranking=pl.DataFrame(
                {
                    "rank": [1],
                    "run_id": ["r1"],
                    "metric_value": [0.9],
                }
            ),
        )


def test_write_report_figures_rejects_missing_execution_stage_column(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="report_stage_breakdown"):
        write_report_figures(
            report_dir=tmp_path / "report",
            report_runs=pl.DataFrame(
                {
                    "run_id": ["r1"],
                    "metric_value": [0.9],
                    "start_time": ["2026-01-01T00:00:00+00:00"],
                }
            ),
            report_ranking=pl.DataFrame(
                {
                    "rank": [1],
                    "run_id": ["r1"],
                    "metric_value": [0.9],
                }
            ),
        )


def test_write_run_figures_ignores_ensemble_model_probs_for_figure_generation(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=pl.DataFrame(
            {
                "fold_id": ["0", "0", "1", "1"],
                "model_index": [0, 1, 0, 1],
                "species": ["sp1", "sp1", "sp2", "sp2"],
                "prob": [0.3, 0.3, 0.6, 0.6],
            }
        ),
        model_selection_trials=None,
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    assert not (figures_dir / "ensemble_uncertainty.svg").exists()
    assert warnings == []


def test_write_run_figures_ignores_model_selection_trials_with_all_null_metrics(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=pl.DataFrame(
            {
                "fold_id": ["0"],
                "sample_set_id": [0],
                "candidate_index": [0],
                "inner_fold_id": ["0"],
                "metric_name": ["mcc"],
                "metric_value": [None],
            }
        ),
    )

    figures_dir = tmp_path / "run" / "cv" / "figures"
    assert not (figures_dir / "model_selection_trials.svg").exists()
    assert warnings == []


def test_species_probability_by_trait_rejects_invalid_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="schema is invalid"):
        figures_mod._species_probability_by_trait(
            predictions=pl.DataFrame({"species": ["sp1"], "prob": [0.2]}),
            trait_col="label",
            out_path=tmp_path / "cv_species_probability_by_trait.svg",
            title="CV Species Probability by Trait",
            subtitle="test",
            source_table_name="prediction_cv.tsv",
            figure_name="cv_species_probability_by_trait.svg",
        )


def test_binary_trait_color_map_uses_expected_colors() -> None:
    assert figures_mod._binary_trait_color_map(
        [0, 1],
        source_table_name="prediction_cv.tsv",
        figure_name="cv_species_probability_by_trait.svg",
    ) == {0: "#d62728", 1: "#1f77b4"}


def test_binary_trait_color_map_rejects_non_binary_values() -> None:
    with pytest.raises(FigureError, match="non-binary trait values"):
        figures_mod._binary_trait_color_map(
            [0, 1, 2],
            source_table_name="prediction_cv.tsv",
            figure_name="cv_species_probability_by_trait.svg",
        )


def test_species_probability_by_trait_writes_svg(tmp_path: Path) -> None:
    out_path = tmp_path / "cv_species_probability_by_trait.svg"
    figures_mod._species_probability_by_trait(
        predictions=pl.DataFrame(
            {
                "species": ["sp1", "sp2", "sp3", "sp4"],
                "label": [0, 0, 1, 1],
                "prob": [0.2, 0.4, 0.7, 0.9],
            }
        ),
        trait_col="label",
        out_path=out_path,
        title="CV Species Probability by Trait",
        subtitle="test",
        source_table_name="prediction_cv.tsv",
        figure_name="cv_species_probability_by_trait.svg",
    )
    assert out_path.exists()
    svg_text = out_path.read_text(encoding="utf-8")
    assert "n=2" in svg_text
    assert "mean=" not in svg_text


def test_cv_fold_trait_probability_rejects_invalid_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="schema is invalid"):
        figures_mod._cv_fold_trait_probability(
            oof_predictions=pl.DataFrame({"label": [0, 1], "prob": [0.2, 0.8]}),
            out_path=tmp_path / "cv_fold_trait_probability.svg",
        )


def test_cv_fold_trait_probability_rejects_empty_table(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="prediction_cv.tsv is empty"):
        figures_mod._cv_fold_trait_probability(
            oof_predictions=pl.DataFrame(
                schema={
                    "fold_id": pl.String,
                    "label": pl.Int64,
                    "prob": pl.Float64,
                }
            ),
            out_path=tmp_path / "cv_fold_trait_probability.svg",
        )


def test_cv_fold_trait_probability_writes_svg(tmp_path: Path) -> None:
    out_path = tmp_path / "cv_fold_trait_probability.svg"
    figures_mod._cv_fold_trait_probability(
        oof_predictions=pl.DataFrame(
            {
                "fold_id": ["0", "0", "1", "1"],
                "label": [0, 1, 0, 1],
                "prob": [0.2, 0.8, 0.3, 0.7],
            }
        ),
        out_path=out_path,
        trait_name="C4",
    )
    assert out_path.exists()
    svg_text = out_path.read_text(encoding="utf-8")
    assert "CV Fold Trait Probability" not in svg_text
    assert "Fold-wise probability distribution" not in svg_text
    assert "fold=0" not in svg_text
    assert "fold=1" not in svg_text
    assert "C4" in svg_text
    assert "C4=0" not in svg_text
    assert "C4=1" not in svg_text
    assert "#f7f7f7" in svg_text
    assert "#d9d9d9" in svg_text


def test_non_zero_feature_count_by_fold_rejects_invalid_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="schema is invalid"):
        figures_mod._non_zero_feature_count_by_fold(
            model_sparsity=pl.DataFrame({"fold_id": ["0"]}),
            out_path=tmp_path / "non_zero_feature_count_by_fold.svg",
        )


def test_non_zero_feature_count_by_fold_writes_svg(tmp_path: Path) -> None:
    out_path = tmp_path / "non_zero_feature_count_by_fold.svg"
    figures_mod._non_zero_feature_count_by_fold(
        model_sparsity=_minimal_model_sparsity(),
        out_path=out_path,
    )
    assert out_path.exists()
    svg_text = out_path.read_text(encoding="utf-8")
    assert "Number of non-zero features per model" in svg_text
    assert "CV fold" in svg_text


def test_feature_importance_top_rejects_invalid_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="feature_importance.tsv schema is invalid"):
        figures_mod._feature_importance_top(
            feature_importance=pl.DataFrame({"feature": ["OG1"]}),
            out_path=tmp_path / "feature_importance_top.svg",
        )


def test_feature_importance_top_rejects_empty_table(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="feature_importance.tsv is empty"):
        figures_mod._feature_importance_top(
            feature_importance=pl.DataFrame(
                schema={
                    "feature": pl.String,
                    "importance_mean": pl.Float64,
                }
            ),
            out_path=tmp_path / "feature_importance_top.svg",
        )


def test_feature_importance_top_handles_zero_importances(tmp_path: Path) -> None:
    out_path = tmp_path / "feature_importance_top.svg"
    figures_mod._feature_importance_top(
        feature_importance=pl.DataFrame(
            {
                "feature": ["OG1", "OG2"],
                "importance_mean": [0.0, 0.0],
            }
        ),
        out_path=out_path,
    )
    assert out_path.exists()
    svg_text = out_path.read_text()
    assert "Feature Importance Top" not in svg_text
    assert "importance_mean" not in svg_text
    assert "Orthogroup ID" in svg_text
    assert "Mean feature importance per fold" in svg_text
    assert "#009e73" not in svg_text
    assert "#666666" in svg_text


def test_feature_importance_top_uses_requested_feature_limit(tmp_path: Path) -> None:
    out_path = tmp_path / "feature_importance_top.svg"
    figures_mod._feature_importance_top(
        feature_importance=pl.DataFrame(
            {
                "feature": ["OG1", "OG2", "OG3"],
                "importance_mean": [0.2, 0.9, 0.1],
            }
        ),
        out_path=out_path,
        top_features=1,
    )

    svg_text = out_path.read_text()
    assert "OG2" in svg_text
    assert "OG1" not in svg_text
    assert "OG3" not in svg_text


def test_feature_importance_top_fold_points_use_neutral_styling(tmp_path: Path) -> None:
    out_path = tmp_path / "feature_importance_top.svg"
    figures_mod._feature_importance_top(
        feature_importance=pl.DataFrame(
            {
                "feature": ["OG1", "OG2"],
                "importance_mean": [0.4, 0.3],
            }
        ),
        feature_importance_by_fold=pl.DataFrame(
            {
                "fold_id": ["0", "1", "0", "1"],
                "feature": ["OG1", "OG1", "OG2", "OG2"],
                "importance_mean": [0.5, 0.3, 0.2, 0.4],
            }
        ),
        out_path=out_path,
    )

    svg_text = out_path.read_text()
    assert "Feature Importance Top" not in svg_text
    assert "fold-level importance_mean" not in svg_text
    assert "Orthogroup ID" in svg_text
    assert "Mean feature importance per fold" in svg_text
    assert "#009e73" not in svg_text
    assert "#d9f0e6" not in svg_text
    assert "#eeeeee" in svg_text
    assert "#666666" in svg_text


def test_feature_importance_by_fold_heatmap_rejects_invalid_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="feature_importance_by_fold.tsv schema is invalid"):
        figures_mod._feature_importance_by_fold_heatmap(
            feature_importance=_minimal_feature_importance(),
            feature_importance_by_fold=pl.DataFrame({"feature": ["OG1"]}),
            out_path=tmp_path / "feature_importance_by_fold_heatmap.svg",
        )


def test_feature_importance_by_fold_heatmap_writes_svg(tmp_path: Path) -> None:
    out_path = tmp_path / "feature_importance_by_fold_heatmap.svg"
    figures_mod._feature_importance_by_fold_heatmap(
        feature_importance=_minimal_feature_importance(),
        feature_importance_by_fold=_minimal_feature_importance_by_fold(),
        out_path=out_path,
    )

    assert out_path.exists()
    svg_text = out_path.read_text(encoding="utf-8")
    assert "CV fold" in svg_text
    assert "Orthogroup ID" in svg_text
    assert "Mean feature importance per fold" in svg_text
    assert "OG1" in svg_text
    assert "OG2" in svg_text


def test_feature_importance_by_fold_heatmap_colormap_starts_at_white() -> None:
    cmap = figures_mod._FEATURE_IMPORTANCE_HEATMAP_CMAP

    assert to_hex(cmap(0.0)) == "#ffffff"
    assert to_hex(cmap(0.5)) != "#ffffff"


def test_feature_importance_by_fold_heatmap_uses_requested_feature_limit(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "feature_importance_by_fold_heatmap.svg"
    figures_mod._feature_importance_by_fold_heatmap(
        feature_importance=pl.DataFrame(
            {
                "feature": ["OG1", "OG2", "OG3"],
                "importance_mean": [0.2, 0.9, 0.1],
            }
        ),
        feature_importance_by_fold=pl.DataFrame(
            {
                "fold_id": ["0", "0", "0"],
                "feature": ["OG1", "OG2", "OG3"],
                "importance_mean": [0.2, 0.9, 0.1],
            }
        ),
        out_path=out_path,
        top_features=1,
    )

    svg_text = out_path.read_text(encoding="utf-8")
    assert "OG2" in svg_text
    assert "OG1" not in svg_text
    assert "OG3" not in svg_text


def test_coefficients_signed_top_rejects_invalid_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="coefficients.tsv schema is invalid"):
        figures_mod._coefficients_signed_top(
            coefficients=pl.DataFrame({"feature": ["OG1"]}),
            out_path=tmp_path / "coefficients_signed_top.svg",
        )


def test_coefficients_signed_top_skips_when_no_coef_signed_rows(tmp_path: Path) -> None:
    out_path = tmp_path / "coefficients_signed_top.svg"
    figures_mod._coefficients_signed_top(
        coefficients=pl.DataFrame(
            {
                "feature": ["OG1"],
                "coef_mean": [0.1],
                "method": ["permutation"],
            }
        ),
        out_path=out_path,
    )
    assert not out_path.exists()


def test_coefficients_signed_top_handles_zero_coefficients(tmp_path: Path) -> None:
    out_path = tmp_path / "coefficients_signed_top.svg"
    figures_mod._coefficients_signed_top(
        coefficients=pl.DataFrame(
            {
                "feature": ["OG1", "OG2"],
                "coef_mean": [0.0, 0.0],
                "method": ["coef_signed", "coef_signed"],
            }
        ),
        out_path=out_path,
    )
    assert out_path.exists()
    svg_text = out_path.read_text()
    assert "Coefficients Signed Top" not in svg_text
    assert "Top 30 by |coef_mean|" not in svg_text
    assert "Orthogroup ID" in svg_text
    assert "#1f77b4" not in svg_text
    assert "#d62728" not in svg_text


def test_coefficients_signed_top_uses_requested_feature_limit(tmp_path: Path) -> None:
    out_path = tmp_path / "coefficients_signed_top.svg"
    figures_mod._coefficients_signed_top(
        coefficients=pl.DataFrame(
            {
                "feature": ["OG1", "OG2", "OG3"],
                "coef_mean": [0.2, -0.9, 0.1],
                "method": ["coef_signed", "coef_signed", "coef_signed"],
            }
        ),
        out_path=out_path,
        top_features=1,
    )

    svg_text = out_path.read_text()
    assert "OG2" in svg_text
    assert "OG1" not in svg_text
    assert "OG3" not in svg_text


def test_coefficients_signed_top_fold_points_use_neutral_styling(tmp_path: Path) -> None:
    out_path = tmp_path / "coefficients_signed_top.svg"
    figures_mod._coefficients_signed_top(
        coefficients=pl.DataFrame(
            {
                "feature": ["OG1", "OG2"],
                "coef_mean": [0.4, -0.3],
                "method": ["coef_signed", "coef_signed"],
            }
        ),
        coefficients_by_fold=pl.DataFrame(
            {
                "fold_id": ["0", "1", "0", "1"],
                "feature": ["OG1", "OG1", "OG2", "OG2"],
                "coef_mean": [0.5, 0.3, -0.2, -0.4],
                "method": ["coef_signed", "coef_signed", "coef_signed", "coef_signed"],
            }
        ),
        out_path=out_path,
    )

    svg_text = out_path.read_text()
    assert "Coefficients Signed Top" not in svg_text
    assert "Top 30 by |mean fold-level coef|" not in svg_text
    assert "Orthogroup ID" in svg_text
    assert "#1f77b4" not in svg_text
    assert "#d62728" not in svg_text
    assert "#eeeeee" in svg_text
    assert "#666666" in svg_text


def test_predict_probability_distribution_rejects_missing_prob_column(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="prediction_inference.tsv schema is invalid"):
        figures_mod._predict_probability_distribution(
            pred_predict=pl.DataFrame({"species": ["sp1"]}),
            out_path=tmp_path / "predict_probability_distribution.svg",
        )


def test_predict_uncertainty_rejects_empty_table_when_required(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="prediction_inference.tsv is empty"):
        figures_mod._predict_uncertainty(
            pred_predict=pl.DataFrame(
                schema={"species": pl.String, "uncertainty_std": pl.Float64}
            ),
            out_path=tmp_path / "predict_uncertainty.svg",
            required=True,
        )


def test_predict_uncertainty_handles_zero_values(tmp_path: Path) -> None:
    out_path = tmp_path / "predict_uncertainty.svg"
    figures_mod._predict_uncertainty(
        pred_predict=pl.DataFrame(
            {
                "species": ["sp1", "sp2"],
                "uncertainty_std": [0.0, 0.0],
            }
        ),
        out_path=out_path,
        required=True,
    )
    assert out_path.exists()


def test_roc_pr_curves_rejects_invalid_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="prediction_cv.tsv schema is invalid"):
        figures_mod._roc_pr_curves_cv(
            oof_predictions=pl.DataFrame({"label": [0, 1], "prob": [0.2, 0.8]}),
            roc_out_path=tmp_path / "roc_curve_cv.svg",
            pr_out_path=tmp_path / "pr_curve_cv.svg",
        )


def test_roc_pr_curves_rejects_empty_table(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="prediction_cv.tsv is empty"):
        figures_mod._roc_pr_curves_cv(
            oof_predictions=pl.DataFrame(
                schema={
                    "fold_id": pl.String,
                    "label": pl.Int64,
                    "prob": pl.Float64,
                }
            ),
            roc_out_path=tmp_path / "roc_curve_cv.svg",
            pr_out_path=tmp_path / "pr_curve_cv.svg",
        )


def test_roc_pr_curves_preserves_pr_curve_threshold_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, np.ndarray | str] = {}
    original_subplots = figures_mod.plt.subplots

    def subplots_spy(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        original_plot = ax.plot

        def plot_spy(x, y, *plot_args, **plot_kwargs):
            if plot_kwargs.get("color") == figures_mod._COLOR_ORANGE:
                captured["recall"] = np.asarray(x, dtype=float)
                captured["precision"] = np.asarray(y, dtype=float)
                captured["drawstyle"] = plot_kwargs.get("drawstyle")
            return original_plot(x, y, *plot_args, **plot_kwargs)

        ax.plot = plot_spy
        return fig, ax

    monkeypatch.setattr(figures_mod.plt, "subplots", subplots_spy)
    oof_predictions = pl.DataFrame(
        {
            "fold_id": ["0"] * 6,
            "label": [1, 1, 1, 0, 0, 0],
            "prob": [0.02, 0.81, 0.91, 0.61, 0.73, 0.54],
        }
    )

    figures_mod._roc_pr_curves_cv(
        oof_predictions=oof_predictions,
        roc_out_path=tmp_path / "roc_curve_cv.svg",
        pr_out_path=tmp_path / "pr_curve_cv.svg",
    )

    precision, recall, _ = figures_mod.precision_recall_curve(
        np.array([1, 1, 1, 0, 0, 0]),
        np.array([0.02, 0.81, 0.91, 0.61, 0.73, 0.54]),
    )
    recall_order = np.argsort(recall)
    assert not np.array_equal(precision[recall_order], precision)
    np.testing.assert_allclose(captured["recall"], recall)
    np.testing.assert_allclose(captured["precision"], precision)
    assert captured["drawstyle"] == "steps-post"


def test_predict_probability_distribution_handles_nan_only_probabilities(tmp_path: Path) -> None:
    out_path = tmp_path / "predict_probability_distribution.svg"
    figures_mod._predict_probability_distribution(
        pred_predict=pl.DataFrame({"prob": [float("nan")]}),
        out_path=out_path,
    )

    assert out_path.exists()


def test_predict_uncertainty_optional_empty_table_returns_without_writing(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "predict_uncertainty.svg"
    figures_mod._predict_uncertainty(
        pred_predict=pl.DataFrame(
            schema={
                "species": pl.String,
                "uncertainty_std": pl.Float64,
            }
        ),
        out_path=out_path,
        required=False,
    )

    assert not out_path.exists()
