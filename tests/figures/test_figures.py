from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import phenoradar.figures as figures_mod
from phenoradar.figures import (
    FigureError,
    write_predict_figures,
    write_report_figures,
    write_run_figures,
)


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
                    "pred_label_cv_derived_threshold": [0, 1],
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
        oof_predictions=pl.DataFrame(
            {
                "fold_id": ["0", "0"],
                "species": ["sp1", "sp2"],
                "label": [0, 1],
                "prob": [0.2, 0.8],
            }
        ),
        thresholds=pl.DataFrame(
            {
                "threshold_name": [
                    "fixed_probability_threshold",
                    "cv_derived_threshold",
                ],
                "threshold_value": [0.5, 0.4],
                "source": ["config", "oof_predictions"],
                "selection_metric": ["NA", "mcc"],
                "selection_scope": ["NA", "outer_cv"],
            }
        ),
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
        auto_threshold_metric="mcc",
    )

    figures_dir = tmp_path / "run" / "figures"
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


def _minimal_thresholds() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "threshold_name": [
                "fixed_probability_threshold",
                "cv_derived_threshold",
            ],
            "threshold_value": [0.5, 0.4],
            "source": ["config", "oof_predictions"],
            "selection_metric": ["NA", "mcc"],
            "selection_scope": ["NA", "outer_cv"],
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


def test_format_float_returns_nan_for_none_and_nan() -> None:
    assert figures_mod._format_float(None) == "NaN"
    assert figures_mod._format_float(float("nan")) == "NaN"


def test_metric_score_supports_balanced_accuracy() -> None:
    y_true = [0, 0, 1, 1]
    prob = [0.1, 0.7, 0.4, 0.9]
    score = figures_mod._metric_score(
        y_true=np.array(y_true, dtype=int),
        prob=np.array(prob, dtype=float),
        threshold=0.5,
        metric="balanced_accuracy",
    )
    assert score == pytest.approx(0.5)


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
                "pred_label_cv_derived_threshold": [0, 1],
                "uncertainty_std": [0.01, 0.02],
            }
        ),
        require_uncertainty=True,
    )

    figures_dir = tmp_path / "predict_run" / "figures"
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
                "pred_label_cv_derived_threshold": [0, 1],
            }
        ),
        require_uncertainty=False,
    )

    figures_dir = tmp_path / "predict_run" / "figures"
    assert (figures_dir / "predict_probability_distribution.svg").exists()
    assert not (figures_dir / "predict_uncertainty.svg").exists()


def test_write_run_figures_writes_required_artifacts(tmp_path: Path) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        thresholds=_minimal_thresholds(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        auto_threshold_metric="mcc",
    )

    figures_dir = tmp_path / "run" / "figures"
    assert (figures_dir / "cv_metrics_overview.svg").exists()
    assert (figures_dir / "threshold_selection_curve.svg").exists()
    assert (figures_dir / "feature_importance_top.svg").exists()
    assert (figures_dir / "coefficients_signed_top.svg").exists()
    assert (figures_dir / "roc_pr_curves_cv.svg").exists()
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
        thresholds=_minimal_thresholds(),
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
        auto_threshold_metric="mcc",
    )

    figures_dir = tmp_path / "run" / "figures"
    assert not (figures_dir / "model_selection_trials.svg").exists()
    assert warnings == []


def test_write_run_figures_ignores_empty_ensemble_inputs(tmp_path: Path) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        thresholds=_minimal_thresholds(),
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
        auto_threshold_metric="mcc",
    )

    figures_dir = tmp_path / "run" / "figures"
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
        thresholds=_minimal_thresholds(),
        feature_importance=_minimal_feature_importance(),
        coefficients=_minimal_coefficients(),
        ensemble_model_probs=None,
        model_selection_trials=None,
        auto_threshold_metric="mcc",
    )

    assert any("no folds with both labels" in warning for warning in warnings)
    figures_dir = tmp_path / "run" / "figures"
    assert (figures_dir / "threshold_selection_curve.svg").exists()


def test_write_run_figures_rejects_invalid_metrics_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="metrics_cv.tsv schema is invalid"):
        write_run_figures(
            run_dir=tmp_path / "run",
            metrics_cv=pl.DataFrame({"aggregate_scope": ["macro"]}),
            oof_predictions=_minimal_oof(),
            thresholds=_minimal_thresholds(),
            feature_importance=_minimal_feature_importance(),
            coefficients=_minimal_coefficients(),
            ensemble_model_probs=None,
            model_selection_trials=None,
            auto_threshold_metric="mcc",
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
            thresholds=_minimal_thresholds(),
            feature_importance=_minimal_feature_importance(),
            coefficients=_minimal_coefficients(),
            ensemble_model_probs=None,
            model_selection_trials=None,
            auto_threshold_metric="mcc",
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
                    "pred_label_cv_derived_threshold": pl.Int64,
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
        thresholds=_minimal_thresholds(),
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
        auto_threshold_metric="mcc",
    )

    figures_dir = tmp_path / "run" / "figures"
    assert not (figures_dir / "ensemble_uncertainty.svg").exists()
    assert warnings == []


def test_write_run_figures_ignores_model_selection_trials_with_all_null_metrics(
    tmp_path: Path,
) -> None:
    warnings = write_run_figures(
        run_dir=tmp_path / "run",
        metrics_cv=_minimal_metrics_cv(),
        oof_predictions=_minimal_oof(),
        thresholds=_minimal_thresholds(),
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
        auto_threshold_metric="mcc",
    )

    figures_dir = tmp_path / "run" / "figures"
    assert not (figures_dir / "model_selection_trials.svg").exists()
    assert warnings == []


def test_threshold_selection_curve_rejects_invalid_schema(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="prediction_cv.tsv schema is invalid"):
        figures_mod._threshold_selection_curve(
            oof_predictions=pl.DataFrame({"prob": [0.1, 0.9]}),
            thresholds=_minimal_thresholds(),
            selection_metric="mcc",
            out_path=tmp_path / "threshold_selection_curve.svg",
        )


def test_threshold_selection_curve_rejects_empty_predictions(tmp_path: Path) -> None:
    with pytest.raises(FigureError, match="prediction_cv.tsv is empty"):
        figures_mod._threshold_selection_curve(
            oof_predictions=pl.DataFrame(
                schema={
                    "label": pl.Int64,
                    "prob": pl.Float64,
                }
            ),
            thresholds=_minimal_thresholds(),
            selection_metric="mcc",
            out_path=tmp_path / "threshold_selection_curve.svg",
        )


def test_threshold_selection_curve_handles_all_nan_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figures_mod, "_metric_score", lambda *_args, **_kwargs: float("nan"))
    out_path = tmp_path / "threshold_selection_curve.svg"

    figures_mod._threshold_selection_curve(
        oof_predictions=pl.DataFrame({"label": [0, 1], "prob": [0.1, 0.9]}),
        thresholds=_minimal_thresholds(),
        selection_metric="mcc",
        out_path=out_path,
    )

    assert "No valid threshold scores" in out_path.read_text(encoding="utf-8")


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
            out_path=tmp_path / "roc_pr_curves_cv.svg",
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
            out_path=tmp_path / "roc_pr_curves_cv.svg",
        )


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
