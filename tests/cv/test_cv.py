from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn import config_context
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)

import phenoradar.cv as cv_mod
from phenoradar.config import load_and_resolve_config
from phenoradar.cv import (
    CVError,
    ExpressionMatrixBuilder,
    _aggregate_probabilities,
    _apply_correlation_filter,
    _build_estimator,
    _build_prediction_table,
    _compute_fold_metrics,
    _fit_estimator,
    _group_label_inverse_weights,
    _inner_cv_splits,
    _predict_positive_probability,
    _prepare_source_selection,
    _prepare_source_selection_tpe,
    _preprocess_fold,
    _preprocess_train_and_target,
    _select_feature_indices,
    apply_expression_transform,
    run_final_refit,
    run_outer_cv,
)
from phenoradar.interpret import InterpretationError
from phenoradar.model_selection import Candidate, ModelSelectionError
from phenoradar.split import build_split_artifacts


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "sp1\t1\tg1\tno",
                "sp2\t0\tg1\tno",
                "sp3\t1\tg2\tno",
                "sp4\t0\tg2\tno",
                "sp5\t1\t\tyes",
                "sp6\t\t\tno",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp1\tOG2\t0.5",
                "sp2\tOG1\t2.0",
                "sp2\tOG2\t0.3",
                "sp3\tOG1\t3.0",
                "sp3\tOG2\t2.0",
                "sp4\tOG1\t4.0",
                "sp4\tOG2\t0.1",
                "sp5\tOG1\t5.0",
                "sp5\tOG2\t0.9",
                "sp6\tOG1\t6.0",
                "sp6\tOG2\t0.2",
            ]
        )
        + "\n",
    )
    return metadata, tpm


def _config_path(tmp_path: Path, metadata: Path, tpm: Path, extra: str = "") -> Path:
    return _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
{extra}
""".strip()
        + "\n",
    )


def _selection_source_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float),
        np.array([0, 1, 0, 1], dtype=int),
        np.array(["g1", "g1", "g2", "g2"], dtype=str),
    )


def test_run_outer_cv_generates_metrics_and_thresholds(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.oof_predictions.height == 4
    scopes = set(cv_artifacts.metrics_cv.select("aggregate_scope").to_series().to_list())
    assert {"NA", "macro", "micro"}.issubset(scopes)
    assert {
        "fold_id",
        "split",
        "metric",
        "metric_value",
    }.issubset(cv_artifacts.loss_by_split_cv.columns)
    assert cv_artifacts.loss_by_split_cv.height > 0
    assert set(cv_artifacts.loss_by_split_cv.select("metric").to_series().to_list()) == {
        "log_loss"
    }
    assert set(cv_artifacts.loss_by_split_cv.select("split").to_series().to_list()) == {
        "train",
        "validation",
    }
    threshold_names = set(cv_artifacts.thresholds.select("threshold_name").to_series().to_list())
    assert threshold_names == {"fixed_probability_threshold"}
    threshold_row = cv_artifacts.thresholds.row(0, named=True)
    assert threshold_row["threshold_value"] == pytest.approx(0.5)
    assert threshold_row["source"] == "constant"
    assert threshold_row["policy"] == "fixed_constant"
    assert threshold_row["derived_from_cv"] is False
    assert {
        "feature",
        "importance_mean",
        "importance_std",
        "n_models",
        "n_folds",
        "method",
    }.issubset(cv_artifacts.feature_importance.columns)
    assert {
        "fold_id",
        "feature",
        "importance_mean",
        "n_models",
        "method",
    }.issubset(cv_artifacts.feature_importance_by_fold.columns)
    assert {
        "feature",
        "coef_mean",
        "coef_std",
        "n_models",
        "n_folds",
        "method",
        "reason",
    }.issubset(cv_artifacts.coefficients.columns)
    assert {
        "fold_id",
        "feature",
        "coef_mean",
        "n_models",
        "method",
        "reason",
    }.issubset(cv_artifacts.coefficients_by_fold.columns)
    assert cv_artifacts.feature_stability_by_feature.height > 0
    assert {
        "feature",
        "n_outer_folds",
        "n_retained_folds",
        "retained_frequency",
        "n_nonzero_folds",
        "selection_frequency",
        "selection_frequency_when_retained",
        "n_positive_folds",
        "n_negative_folds",
        "dominant_sign",
        "sign_agreement_rate",
        "sign_reason",
    }.issubset(cv_artifacts.feature_stability_by_feature.columns)
    assert cv_artifacts.feature_stability_by_fold_pair.height == 1
    assert {
        "fold_id_a",
        "fold_id_b",
        "n_intersection",
        "n_union",
        "jaccard",
    }.issubset(cv_artifacts.feature_stability_by_fold_pair.columns)
    assert cv_artifacts.feature_stability_summary.height == 1
    assert {
        "n_outer_folds",
        "n_fold_pairs",
        "jaccard_mean",
        "n_features_ever_selected",
        "sign_agreement_mean",
    }.issubset(cv_artifacts.feature_stability_summary.columns)
    aggregate_rows = cv_artifacts.metrics_cv.filter(pl.col("fold_id") == "NA")
    assert aggregate_rows.height > 0
    assert aggregate_rows.filter(pl.col("n_valid_folds").is_null()).height == 0
    assert "uncertainty_std" not in cv_artifacts.oof_predictions.columns
    assert cv_artifacts.ensemble_model_probs is None
    assert cv_artifacts.model_selection_selected is None
    assert cv_artifacts.model_selection_trials is None
    assert cv_artifacts.model_selection_trials_summary is None
    assert cv_artifacts.feature_filter_counts.height > 0
    assert {
        "scope",
        "fold_id",
        "sample_set_id",
        "n_features_before",
        "n_features_after_sparse_feature_filter",
        "n_features_after_low_variance",
        "n_features_after_pair_aware",
        "n_features_after_correlation",
        "n_features_after_all",
    }.issubset(cv_artifacts.feature_filter_counts.columns)
    assert cv_artifacts.feature_filter_counts_summary.height > 0
    assert {
        "scope",
        "stage",
        "n_records",
        "n_features_q1",
        "n_features_median",
        "n_features_q3",
        "retained_ratio_q1",
        "retained_ratio_median",
        "retained_ratio_q3",
    }.issubset(cv_artifacts.feature_filter_counts_summary.columns)
    assert cv_artifacts.retained_features.height > 0
    assert {
        "scope",
        "fold_id",
        "sample_set_id",
        "feature",
    }.issubset(cv_artifacts.retained_features.columns)
    assert cv_artifacts.retained_features_summary.height > 0
    assert {
        "scope",
        "fold_id",
        "feature",
        "retained_count",
        "n_sample_sets",
        "retained_rate",
    }.issubset(cv_artifacts.retained_features_summary.columns)
    assert cv_artifacts.model_sparsity.height > 0
    assert {
        "scope",
        "fold_id",
        "sample_set_id",
        "model_index",
        "model_name",
        "n_features_after_all",
        "n_nonzero_features",
        "nonzero_ratio",
    }.issubset(cv_artifacts.model_sparsity.columns)
    assert cv_artifacts.model_sparsity_summary.height > 0
    assert {
        "scope",
        "model_name",
        "n_models",
        "n_models_with_nonzero_count",
    }.issubset(cv_artifacts.model_sparsity_summary.columns)
    assert set(cv_artifacts.top_feature_expression.columns) == {"species", "feature", "tpm"}
    assert set(cv_artifacts.top_feature_expression.get_column("species")) == {
        "sp1",
        "sp2",
        "sp3",
        "sp4",
    }
    assert (
        cv_artifacts.top_feature_expression.filter(
            (pl.col("species") == "sp1") & (pl.col("feature") == "OG1")
        ).get_column("tpm").item()
        == 1.0
    )
    assert {
        "scope",
        "stage",
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "started_at_sec",
        "ended_at_sec",
        "duration_sec",
    } == set(cv_artifacts.timing.columns)
    timing_pairs = set(cv_artifacts.timing.select("scope", "stage").iter_rows())
    assert {
        ("outer_cv", "matrix_build"),
        ("outer_cv", "fold_execution"),
        ("outer_cv", "interpretation"),
        ("outer_cv", "total"),
        ("outer_fold", "preprocessing"),
        ("outer_fold", "model_fit"),
        ("outer_fold", "prediction"),
        ("outer_fold", "sample_set_total"),
        ("outer_fold", "total"),
    }.issubset(timing_pairs)
    assert cv_artifacts.timing.filter(pl.col("duration_sec") < 0.0).height == 0


def test_metrics_dataframe_handles_valid_fold_counts_after_schema_inference_limit() -> None:
    fold_metrics = {f"metric_{index}": float(index) for index in range(5)}
    metric_rows = [
        row
        for fold_index in range(21)
        for row in cv_mod._metric_rows(
            fold_id=f"fold_{fold_index}",
            aggregate_scope="NA",
            metrics=fold_metrics,
            n_pos=1,
            n_neg=1,
            n_valid_folds=None,
        )
    ]
    metric_rows.extend(
        cv_mod._metric_rows(
            fold_id=None,
            aggregate_scope="macro",
            metrics=fold_metrics,
            n_pos=21,
            n_neg=21,
            n_valid_folds={metric: 21 for metric in fold_metrics},
        )
    )

    metrics_df = cv_mod._metrics_dataframe(metric_rows)

    assert metrics_df.schema["n_valid_folds"] == pl.Int64
    assert metrics_df.filter(pl.col("aggregate_scope") == "macro").get_column(
        "n_valid_folds"
    ).to_list() == [21] * len(fold_metrics)


def test_run_outer_cv_oof_species_and_fold_match_validation_manifest(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)
    expected_oof_index = (
        split_artifacts.split_manifest.filter(
            (pl.col("pool") == "validation") & (pl.col("fold_id") != "NA")
        )
        .select(["fold_id", "species"])
        .sort(["fold_id", "species"])
    )
    actual_oof_index = cv_artifacts.oof_predictions.select(["fold_id", "species"]).sort(
        ["fold_id", "species"]
    )

    assert actual_oof_index.to_dicts() == expected_oof_index.to_dicts()
    assert cv_artifacts.oof_predictions.get_column("species").n_unique() == (
        cv_artifacts.oof_predictions.height
    )


def test_run_outer_cv_allows_single_class_validation_folds(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "sp1\t1\tg1\tno",
                "sp2\t1\tg1\tno",
                "sp3\t0\tg2\tno",
                "sp4\t0\tg2\tno",
                "sp5\t1\tg3\tno",
                "sp6\t0\tg3\tno",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "species\torthogroup\ttpm\n"
        + "\n".join(
            [
                "sp1\tOG1\t6.0",
                "sp2\tOG1\t5.0",
                "sp3\tOG1\t1.0",
                "sp4\tOG1\t2.0",
                "sp5\tOG1\t4.0",
                "sp6\tOG1\t3.0",
            ]
        )
        + "\n",
    )
    config_path = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
sampling:
  strategy: all_samples
  max_samples_per_label_per_group: null
  sampled_set_count: 1
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([config_path])
    split_artifacts = build_split_artifacts(config)

    artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert artifacts.oof_predictions.height == 6
    per_fold = artifacts.metrics_cv.filter(pl.col("aggregate_scope") == "NA")
    single_class_folds = per_fold.filter(pl.col("fold_id").is_in(["1", "2"]))
    assert single_class_folds.filter(pl.col("metric") == "brier").select(
        pl.col("metric_value").is_finite().all()
    ).item()
    assert single_class_folds.filter(
        pl.col("metric").is_in(["roc_auc", "pr_auc", "balanced_accuracy", "mcc"])
    ).select(pl.col("metric_value").is_nan().all()).item()

    macro = artifacts.metrics_cv.filter(pl.col("aggregate_scope") == "macro")
    valid_counts = dict(macro.select("metric", "n_valid_folds").iter_rows())
    assert valid_counts == {
        "balanced_accuracy": 1,
        "brier": 3,
        "mcc": 1,
        "pr_auc": 1,
        "roc_auc": 1,
    }


def test_run_outer_cv_builds_expression_matrix_once_across_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    build_calls = 0
    cache_calls = 0
    original_build_matrix = cv_mod.ExpressionMatrixBuilder.build_matrix
    original_cache_species = cv_mod.ExpressionMatrixBuilder.cache_species

    def _counting_cache_species(self: object, species_order: list[str]) -> None:
        nonlocal cache_calls
        cache_calls += 1
        original_cache_species(self, species_order)

    def _counting_build_matrix(
        self: object, species_order: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        nonlocal build_calls
        build_calls += 1
        return original_build_matrix(self, species_order)

    monkeypatch.setattr(cv_mod.ExpressionMatrixBuilder, "cache_species", _counting_cache_species)
    monkeypatch.setattr(cv_mod.ExpressionMatrixBuilder, "build_matrix", _counting_build_matrix)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.oof_predictions.height == 4
    assert cache_calls == 1
    assert build_calls == 1


def test_expression_matrix_builder_coordinate_fill_preserves_pivot_semantics() -> None:
    long_df = pl.DataFrame(
        {
            "__species": ["sp2", "sp1", "sp1", "sp1"],
            "__feature": ["OG2", "OG1", "OG1", "OG_extra"],
            "__value": [2.0, 1.0, 3.0, 9.0],
        }
    )
    ordering_df = pl.DataFrame(
        {
            "__species": ["sp2", "sp1", "sp2"],
            "__row_idx": [0, 1, 2],
        }
    )

    matrix = ExpressionMatrixBuilder._matrix_from_long_df(
        long_df,
        ordering_df,
        ["OG2", "OG_missing", "OG1"],
    )

    assert matrix.tolist() == [
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 4.0],
        [2.0, 0.0, 0.0],
    ]


def test_run_outer_cv_with_small_max_pivot_cells_uses_feature_chunking(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  max_pivot_cells: 2
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.oof_predictions.height == 4
    assert cv_artifacts.metrics_cv.height > 0
    assert cv_artifacts.thresholds.height > 0


def test_group_balanced_sampling_caps_requested_count(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
sampling:
  strategy: group_balanced
  sampled_set_count: 5
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert any("sampled_set_count exceeded" in warning for warning in cv_artifacts.warnings)


def test_outer_cv_emits_ensemble_artifacts_when_ensemble_size_gt_one(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "g1_pos1\t1\tg1\tno",
                "g1_pos2\t1\tg1\tno",
                "g1_neg1\t0\tg1\tno",
                "g1_neg2\t0\tg1\tno",
                "g2_pos1\t1\tg2\tno",
                "g2_pos2\t1\tg2\tno",
                "g2_neg1\t0\tg2\tno",
                "g2_neg2\t0\tg2\tno",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "g1_pos1\tOG1\t5.0",
                "g1_pos1\tOG2\t2.0",
                "g1_pos2\tOG1\t4.0",
                "g1_pos2\tOG2\t2.2",
                "g1_neg1\tOG1\t1.0",
                "g1_neg1\tOG2\t0.3",
                "g1_neg2\tOG1\t1.2",
                "g1_neg2\tOG2\t0.2",
                "g2_pos1\tOG1\t5.1",
                "g2_pos1\tOG2\t1.9",
                "g2_pos2\tOG1\t4.9",
                "g2_pos2\tOG2\t2.1",
                "g2_neg1\tOG1\t0.9",
                "g2_neg1\tOG2\t0.4",
                "g2_neg2\tOG1\t1.1",
                "g2_neg2\tOG2\t0.1",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
sampling:
  strategy: group_balanced
  max_samples_per_label_per_group: 1
  sampled_set_count: 2
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)
    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.ensemble_model_probs is not None
    assert cv_artifacts.ensemble_model_probs.height > 0
    assert {"fold_id", "model_index", "species", "prob"}.issubset(
        cv_artifacts.ensemble_model_probs.columns
    )
    assert "uncertainty_std" in cv_artifacts.oof_predictions.columns
    assert cv_artifacts.oof_predictions.filter(pl.col("uncertainty_std").is_null()).height == 0
    assert (
        cv_artifacts.oof_predictions.filter(pl.col("uncertainty_std") < 0.0).height == 0
    )


def test_outer_cv_selection_active_emits_selected_and_trials_tables(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "g1_pos\t1\tg1\tno",
                "g1_neg\t0\tg1\tno",
                "g2_pos\t1\tg2\tno",
                "g2_neg\t0\tg2\tno",
                "g3_pos\t1\tg3\tno",
                "g3_neg\t0\tg3\tno",
                "g4_pos\t1\tg4",
                "g4_neg\t0\tg4",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "g1_pos\tOG1\t5.0",
                "g1_pos\tOG2\t1.5",
                "g1_neg\tOG1\t1.0",
                "g1_neg\tOG2\t0.2",
                "g2_pos\tOG1\t4.8",
                "g2_pos\tOG2\t1.7",
                "g2_neg\tOG1\t0.8",
                "g2_neg\tOG2\t0.4",
                "g3_pos\tOG1\t5.2",
                "g3_pos\tOG2\t1.8",
                "g3_neg\tOG1\t1.1",
                "g3_neg\tOG2\t0.3",
                "g4_pos\tOG1\t5.1",
                "g4_pos\tOG2\t1.6",
                "g4_neg\tOG1\t0.9",
                "g4_neg\tOG2\t0.1",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
split:
  outer_cv_strategy: group_kfold
  outer_cv_n_splits: 2
model_selection:
  search_strategy: grid
  search_space:
    C: [0.5, 1.0]
    max_iter: [1]
  selected_candidate_count: 1
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)
    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.model_selection_selected is not None
    assert cv_artifacts.model_selection_trials is not None
    assert cv_artifacts.model_selection_trials_summary is not None
    assert cv_artifacts.model_selection_selected.height > 0
    assert cv_artifacts.model_selection_trials.height > 0
    assert cv_artifacts.model_selection_trials_summary.height > 0
    assert cv_artifacts.convergence_diagnostics.height > 0
    assert set(cv_artifacts.convergence_diagnostics.get_column("fit_scope")) == {
        "candidate_evaluation",
        "selected_model",
    }
    assert set(cv_artifacts.convergence_diagnostics.get_column("training_scope")) == {
        "outer_fold"
    }
    assert cv_artifacts.convergence_diagnostics.filter(~pl.col("converged")).height > 0
    assert any("non-converged fit(s)" in warning for warning in cv_artifacts.warnings)
    assert {
        "selection_scope",
        "fold_id",
        "sample_set_id",
        "selection_source_sample_set_id",
        "rank",
        "candidate_index",
        "metric_name",
        "metric_value",
        "metric_value_se",
        "selection_rule",
        "n_available_candidates",
        "n_scored_candidates",
        "selected_candidate_count_requested",
        "selected_candidate_count_effective",
        "params_json",
    }.issubset(cv_artifacts.model_selection_selected.columns)
    assert {
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "inner_fold_id",
        "metric_name",
        "metric_value",
        "params_json",
    }.issubset(cv_artifacts.model_selection_trials.columns)
    assert {
        "fold_id",
        "sample_set_id",
        "candidate_index",
        "metric_name",
        "params_json",
        "n_inner_folds",
        "n_valid_inner_folds",
        "metric_value_mean",
        "metric_value_std",
        "metric_value_se",
    }.issubset(cv_artifacts.model_selection_trials_summary.columns)


def test_outer_cv_tpe_selection_active_bypasses_generic_candidate_generation(
    tmp_path: Path, monkeypatch
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "g1_pos\t1\tg1\tno",
                "g1_neg\t0\tg1\tno",
                "g2_pos\t1\tg2\tno",
                "g2_neg\t0\tg2\tno",
                "g3_pos\t1\tg3\tno",
                "g3_neg\t0\tg3\tno",
                "g4_pos\t1\tg4",
                "g4_neg\t0\tg4",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "g1_pos\tOG1\t5.0",
                "g1_pos\tOG2\t1.5",
                "g1_neg\tOG1\t1.0",
                "g1_neg\tOG2\t0.2",
                "g2_pos\tOG1\t4.8",
                "g2_pos\tOG2\t1.7",
                "g2_neg\tOG1\t0.8",
                "g2_neg\tOG2\t0.4",
                "g3_pos\tOG1\t5.2",
                "g3_pos\tOG2\t1.8",
                "g3_neg\tOG1\t1.1",
                "g3_neg\tOG2\t0.3",
                "g4_pos\tOG1\t5.1",
                "g4_pos\tOG2\t1.6",
                "g4_neg\tOG1\t0.9",
                "g4_neg\tOG2\t0.1",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
split:
  outer_cv_strategy: group_kfold
  outer_cv_n_splits: 2
model_selection:
  search_strategy: tpe
  trial_count: 3
  search_space:
    C:
      type: continuous_log_range
      base: 10
      start_exp: -1
      end_exp: 1
    l1_ratio:
      type: continuous_range
      start: 0.0
      end: 1.0
  selected_candidate_count: 2
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)

    def _fail_generate_candidates(**_kwargs) -> list[object]:
        raise AssertionError("generate_candidates should not be called for active TPE selection")

    monkeypatch.setattr("phenoradar.cv.generate_candidates", _fail_generate_candidates)
    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.model_selection_selected is not None
    assert cv_artifacts.model_selection_trials is not None
    assert cv_artifacts.model_selection_trials_summary is not None
    assert cv_artifacts.model_selection_selected.height > 0
    assert cv_artifacts.model_selection_trials.height > 0
    assert cv_artifacts.model_selection_trials_summary.height > 0


def test_outer_cv_selection_active_with_percent_emits_selected_and_trials_tables(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "g1_pos\t1\tg1\tno",
                "g1_neg\t0\tg1\tno",
                "g2_pos\t1\tg2\tno",
                "g2_neg\t0\tg2\tno",
                "g3_pos\t1\tg3\tno",
                "g3_neg\t0\tg3\tno",
                "g4_pos\t1\tg4",
                "g4_neg\t0\tg4",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "g1_pos\tOG1\t5.0",
                "g1_pos\tOG2\t1.5",
                "g1_neg\tOG1\t1.0",
                "g1_neg\tOG2\t0.2",
                "g2_pos\tOG1\t4.8",
                "g2_pos\tOG2\t1.7",
                "g2_neg\tOG1\t0.8",
                "g2_neg\tOG2\t0.4",
                "g3_pos\tOG1\t5.2",
                "g3_pos\tOG2\t1.8",
                "g3_neg\tOG1\t1.1",
                "g3_neg\tOG2\t0.3",
                "g4_pos\tOG1\t5.1",
                "g4_pos\tOG2\t1.6",
                "g4_neg\tOG1\t0.9",
                "g4_neg\tOG2\t0.1",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
split:
  outer_cv_strategy: group_kfold
  outer_cv_n_splits: 2
model_selection:
  search_strategy: grid
  search_space:
    C: [0.5, 1.0]
  selected_candidate_percent: 50
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)
    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.model_selection_selected is not None
    assert cv_artifacts.model_selection_trials is not None
    selected = cv_artifacts.model_selection_selected
    assert selected.height > 0
    requested_values = set(
        selected.select("selected_candidate_count_requested").to_series().to_list()
    )
    assert requested_values == {1}
    source_values = set(selected.select("selection_source_sample_set_id").to_series().to_list())
    sample_set_values = set(selected.select("sample_set_id").to_series().to_list())
    assert source_values == sample_set_values


def test_run_final_refit_generates_external_and_inference_predictions(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    refit_artifacts = run_final_refit(config, split_artifacts.split_manifest)

    assert refit_artifacts.pred_external_test.height == 1
    assert refit_artifacts.pred_inference.height == 1
    assert refit_artifacts.ensemble_size >= 1
    assert len(refit_artifacts.models) == refit_artifacts.ensemble_size
    assert len(refit_artifacts.feature_names) > 0
    assert {
        "species",
        "true_label",
        "prob",
        "pred_label_fixed_threshold",
    }.issubset(
        refit_artifacts.pred_external_test.columns
    )
    assert {
        "split",
        "metric",
        "metric_value",
    }.issubset(refit_artifacts.loss_by_split_final_refit.columns)
    assert set(
        refit_artifacts.loss_by_split_final_refit.select("metric").to_series().to_list()
    ) == {"log_loss"}
    assert set(
        refit_artifacts.loss_by_split_final_refit.select("split").to_series().to_list()
    ) == {"train", "external_test"}
    assert {
        "species",
        "prob",
        "pred_label_fixed_threshold",
        "true_label",
    }.issubset(
        refit_artifacts.pred_inference.columns
    )
    assert refit_artifacts.pred_inference.get_column("true_label").null_count() == 1
    assert refit_artifacts.model_selection_selected is None
    assert refit_artifacts.feature_filter_counts.height > 0
    assert {
        "scope",
        "fold_id",
        "sample_set_id",
        "n_features_before",
        "n_features_after_pair_aware",
        "n_features_after_all",
    }.issubset(refit_artifacts.feature_filter_counts.columns)
    assert refit_artifacts.feature_filter_counts_summary.height > 0
    assert refit_artifacts.retained_features.height > 0
    assert {
        "scope",
        "fold_id",
        "sample_set_id",
        "feature",
    }.issubset(refit_artifacts.retained_features.columns)
    assert refit_artifacts.retained_features_summary.height > 0
    assert {
        "scope",
        "fold_id",
        "feature",
        "retained_count",
        "n_sample_sets",
        "retained_rate",
    }.issubset(refit_artifacts.retained_features_summary.columns)
    assert refit_artifacts.model_sparsity.height > 0
    assert {
        "scope",
        "fold_id",
        "sample_set_id",
        "model_index",
        "model_name",
        "n_nonzero_features",
    }.issubset(refit_artifacts.model_sparsity.columns)
    assert refit_artifacts.model_sparsity_summary.height > 0
    timing_stages = set(refit_artifacts.timing.get_column("stage"))
    assert {
        "pool_preparation",
        "matrix_build",
        "sampling",
        "candidate_generation",
        "preprocessing",
        "model_fit",
        "prediction",
        "sample_set_total",
        "postprocess",
        "total",
    }.issubset(timing_stages)
    assert refit_artifacts.timing.get_column("scope").unique().to_list() == [
        "final_refit"
    ]


def test_run_final_refit_prunes_target_matrix_without_changing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "sp1\t1\tg1\tno",
                "sp2\t0\tg1\tno",
                "sp3\t1\tg2\tno",
                "sp4\t0\tg2\tno",
                "sp5\t1\t\tyes",
                "sp6\t\t\tno",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp1\tOG2\t0.5",
                "sp2\tOG1\t2.0",
                "sp2\tOG2\t0.3",
                "sp3\tOG1\t3.0",
                "sp3\tOG2\t2.0",
                "sp4\tOG1\t4.0",
                "sp4\tOG2\t0.1",
                "sp5\tOG1\t5.0",
                "sp5\tOG2\t0.9",
                "sp5\tOG_target_only\t7.0",
                "sp6\tOG1\t6.0",
                "sp6\tOG2\t0.2",
                "sp6\tOG_target_only\t8.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    original_cache_species = cv_mod.ExpressionMatrixBuilder.cache_species
    original_build_matrix = cv_mod.ExpressionMatrixBuilder.build_matrix
    cache_calls: list[list[str]] = []
    build_calls: list[tuple[list[str], list[str] | None]] = []

    def _counting_cache_species(self: object, species_order: list[str]) -> None:
        cache_calls.append(list(species_order))
        original_cache_species(self, species_order)

    def _counting_build_matrix(
        self: object,
        species_order: list[str],
        feature_order: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        build_calls.append(
            (list(species_order), None if feature_order is None else list(feature_order))
        )
        return original_build_matrix(self, species_order, feature_order=feature_order)

    monkeypatch.setattr(cv_mod.ExpressionMatrixBuilder, "cache_species", _counting_cache_species)
    monkeypatch.setattr(cv_mod.ExpressionMatrixBuilder, "build_matrix", _counting_build_matrix)
    optimized = run_final_refit(config, split_artifacts.split_manifest)

    assert len(cache_calls) == 1
    target_calls = [call for call in build_calls if call[1] is not None]
    assert len(target_calls) == 1
    assert "OG_target_only" not in target_calls[0][1]

    monkeypatch.setattr(cv_mod.ExpressionMatrixBuilder, "build_matrix", original_build_matrix)
    monkeypatch.setattr(cv_mod, "_final_refit_can_prune_target_matrix", lambda _config: False)
    full_matrix = run_final_refit(config, split_artifacts.split_manifest)

    assert optimized.pred_external_test.to_dicts() == full_matrix.pred_external_test.to_dicts()
    assert optimized.pred_inference.to_dicts() == full_matrix.pred_inference.to_dicts()
    assert (
        optimized.loss_by_split_final_refit.to_dicts()
        == full_matrix.loss_by_split_final_refit.to_dicts()
    )
    assert (
        optimized.feature_filter_counts.to_dicts()
        == full_matrix.feature_filter_counts.to_dicts()
    )


def test_summarize_retained_features_aggregates_count_and_rate() -> None:
    retained_features = cv_mod._build_retained_features(
        [
            {
                "scope": "outer_fold",
                "fold_id": "0",
                "sample_set_id": 0,
                "feature": "OG1",
            },
            {
                "scope": "outer_fold",
                "fold_id": "0",
                "sample_set_id": 1,
                "feature": "OG1",
            },
            {
                "scope": "outer_fold",
                "fold_id": "0",
                "sample_set_id": 1,
                "feature": "OG2",
            },
            {
                "scope": "outer_fold",
                "fold_id": "1",
                "sample_set_id": 0,
                "feature": "OG2",
            },
        ]
    )

    summary = cv_mod._summarize_retained_features(retained_features)

    assert summary.to_dicts() == [
        {
            "scope": "outer_fold",
            "fold_id": "0",
            "feature": "OG1",
            "retained_count": 2,
            "n_sample_sets": 2,
            "retained_rate": 1.0,
        },
        {
            "scope": "outer_fold",
            "fold_id": "0",
            "feature": "OG2",
            "retained_count": 1,
            "n_sample_sets": 2,
            "retained_rate": 0.5,
        },
        {
            "scope": "outer_fold",
            "fold_id": "1",
            "feature": "OG2",
            "retained_count": 1,
            "n_sample_sets": 1,
            "retained_rate": 1.0,
        },
    ]


def test_group_label_inverse_weights_are_normalized_and_group_label_balanced() -> None:
    y = np.array([1, 1, 0, 0, 1, 0], dtype=int)
    groups = np.array(["g1", "g1", "g1", "g2", "g2", "g2"], dtype=str)

    weights = _group_label_inverse_weights(y, groups)

    assert weights.shape == (6,)
    assert np.mean(weights) == pytest.approx(1.0)
    assert np.all(weights > 0)
    # In group g1 label=0 is rarer than label=1, so it should receive larger weight.
    assert float(weights[2]) > float(weights[0])
    # In group g2 label=1 is rarer than label=0, so it should receive larger weight.
    assert float(weights[4]) > float(weights[3])


def test_outer_cv_selection_runs_per_sample_set(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "g1_pos1\t1\tg1\tno",
                "g1_pos2\t1\tg1\tno",
                "g1_neg1\t0\tg1\tno",
                "g1_neg2\t0\tg1\tno",
                "g2_pos1\t1\tg2\tno",
                "g2_pos2\t1\tg2\tno",
                "g2_neg1\t0\tg2\tno",
                "g2_neg2\t0\tg2\tno",
                "g3_pos1\t1\tg3",
                "g3_pos2\t1\tg3",
                "g3_neg1\t0\tg3",
                "g3_neg2\t0\tg3",
                "g4_pos1\t1\tg4",
                "g4_pos2\t1\tg4",
                "g4_neg1\t0\tg4",
                "g4_neg2\t0\tg4",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "g1_pos1\tOG1\t5.0",
                "g1_pos2\tOG1\t5.1",
                "g1_neg1\tOG1\t1.0",
                "g1_neg2\tOG1\t1.1",
                "g2_pos1\tOG1\t4.8",
                "g2_pos2\tOG1\t4.9",
                "g2_neg1\tOG1\t0.8",
                "g2_neg2\tOG1\t0.9",
                "g3_pos1\tOG1\t5.2",
                "g3_pos2\tOG1\t5.3",
                "g3_neg1\tOG1\t1.2",
                "g3_neg2\tOG1\t1.3",
                "g4_pos1\tOG1\t5.4",
                "g4_pos2\tOG1\t5.5",
                "g4_neg1\tOG1\t1.4",
                "g4_neg2\tOG1\t1.5",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
sampling:
  strategy: group_balanced
  max_samples_per_label_per_group: 1
  sampled_set_count: 2
model_selection:
  search_strategy: grid
  search_space:
    C: [0.5, 1.0]
  selected_candidate_count: 1
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.model_selection_selected is not None
    assert cv_artifacts.model_selection_trials is not None
    assert cv_artifacts.model_selection_trials_summary is not None
    selected = cv_artifacts.model_selection_selected
    trials = cv_artifacts.model_selection_trials
    trials_summary = cv_artifacts.model_selection_trials_summary
    assert set(selected.select("sample_set_id").to_series().to_list()) == {0, 1}
    assert set(selected.select("selection_source_sample_set_id").to_series().to_list()) == {0, 1}
    assert set(trials.select("sample_set_id").to_series().to_list()) == {0, 1}
    assert set(trials_summary.select("sample_set_id").to_series().to_list()) == {0, 1}
    candidate_timings = cv_artifacts.timing.filter(pl.col("stage") == "candidate_score")
    assert candidate_timings.height > 0
    assert set(candidate_timings.get_column("sample_set_id")) == {0, 1}
    assert set(candidate_timings.get_column("candidate_index")) == {0, 1}
    assert candidate_timings.get_column("fold_id").null_count() == 0
    inner_preprocessing_timings = cv_artifacts.timing.filter(
        pl.col("stage") == "inner_cv_preprocessing"
    )
    assert inner_preprocessing_timings.height > 0
    assert set(inner_preprocessing_timings.get_column("sample_set_id")) == {0, 1}


def test_outer_cv_selection_can_reuse_first_sample_set(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "g1_pos1\t1\tg1\tno",
                "g1_pos2\t1\tg1\tno",
                "g1_neg1\t0\tg1\tno",
                "g1_neg2\t0\tg1\tno",
                "g2_pos1\t1\tg2\tno",
                "g2_pos2\t1\tg2\tno",
                "g2_neg1\t0\tg2\tno",
                "g2_neg2\t0\tg2\tno",
                "g3_pos1\t1\tg3",
                "g3_pos2\t1\tg3",
                "g3_neg1\t0\tg3",
                "g3_neg2\t0\tg3",
                "g4_pos1\t1\tg4",
                "g4_pos2\t1\tg4",
                "g4_neg1\t0\tg4",
                "g4_neg2\t0\tg4",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "g1_pos1\tOG1\t5.0",
                "g1_pos2\tOG1\t5.1",
                "g1_neg1\tOG1\t1.0",
                "g1_neg2\tOG1\t1.1",
                "g2_pos1\tOG1\t4.8",
                "g2_pos2\tOG1\t4.9",
                "g2_neg1\tOG1\t0.8",
                "g2_neg2\tOG1\t0.9",
                "g3_pos1\tOG1\t5.2",
                "g3_pos2\tOG1\t5.3",
                "g3_neg1\tOG1\t1.2",
                "g3_neg2\tOG1\t1.3",
                "g4_pos1\tOG1\t5.4",
                "g4_pos2\tOG1\t5.5",
                "g4_neg1\tOG1\t1.4",
                "g4_neg2\tOG1\t1.5",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
sampling:
  strategy: group_balanced
  max_samples_per_label_per_group: 1
  sampled_set_count: 2
model_selection:
  search_strategy: grid
  search_space:
    C: [0.5, 1.0]
  selected_candidate_count: 1
  inner_cv_strategy: logo
  candidate_source_policy: reuse_first_sample_set
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.model_selection_selected is not None
    assert cv_artifacts.model_selection_trials is not None
    assert cv_artifacts.model_selection_trials_summary is not None
    selected = cv_artifacts.model_selection_selected
    trials = cv_artifacts.model_selection_trials
    trials_summary = cv_artifacts.model_selection_trials_summary
    assert set(selected.select("sample_set_id").to_series().to_list()) == {0, 1}
    assert set(selected.select("selection_source_sample_set_id").to_series().to_list()) == {0}
    assert set(trials.select("sample_set_id").to_series().to_list()) == {0}
    assert set(trials_summary.select("sample_set_id").to_series().to_list()) == {0}


def test_outer_cv_rejects_model_specific_invalid_search_space_params(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model:
  name: logistic_elasticnet
model_selection:
  search_strategy: grid
  search_space:
    n_estimators: [10]
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)

    with pytest.raises(CVError, match="Unsupported model_selection.search_space parameter"):
        run_outer_cv(config, split_artifacts.split_manifest)


def test_outer_cv_is_deterministic_for_same_input_config_and_seed(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    first = run_outer_cv(config, split_artifacts.split_manifest)
    second = run_outer_cv(config, split_artifacts.split_manifest)

    assert first.metrics_cv.to_dicts() == second.metrics_cv.to_dicts()
    assert first.loss_by_split_cv.to_dicts() == second.loss_by_split_cv.to_dicts()
    assert first.thresholds.to_dicts() == second.thresholds.to_dicts()
    assert first.oof_predictions.to_dicts() == second.oof_predictions.to_dicts()
    assert first.feature_importance.to_dicts() == second.feature_importance.to_dicts()
    assert first.coefficients.to_dicts() == second.coefficients.to_dicts()


def test_run_outer_cv_parallel_fold_budget_applies_to_selection_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model:
  name: random_forest
model_selection:
  search_strategy: grid
  selected_candidate_count: 1
  inner_cv_strategy: logo
runtime:
  n_jobs: 5
""".strip(),
            )
        ]
    )
    split_artifacts = build_split_artifacts(config)
    expected_fold_count = len(cv_mod._fold_ids(split_artifacts.split_manifest))

    selection_n_jobs: list[int] = []

    def _fake_prepare_source_selection(**kwargs: object) -> cv_mod.SourceSelectionResult:
        raw_config = kwargs["config"]
        if not hasattr(raw_config, "runtime"):
            raise AssertionError("config argument must have runtime")
        selection_n_jobs.append(int(raw_config.runtime.n_jobs))
        return cv_mod.SourceSelectionResult(
            selected_candidates=[
                cv_mod.SelectedCandidate(
                    candidate=Candidate(candidate_index=0, params={"n_estimators": 5}),
                    score=None,
                )
            ],
            n_available_candidates=1,
            n_scored_candidates=1,
            selected_candidate_count_requested=1,
            selected_candidate_count_effective=1,
            trial_rows=[],
        )

    original_build_estimator = cv_mod._build_estimator
    estimator_n_jobs: list[int] = []

    def _wrapped_build_estimator(
        config: object,
        model_seed: int,
        y_train: np.ndarray,
        model_params: dict[str, object] | None = None,
        rf_n_jobs: int | None = None,
    ) -> object:
        if not hasattr(config, "runtime"):
            raise AssertionError("config argument must have runtime")
        estimator_n_jobs.append(int(config.runtime.n_jobs))
        return original_build_estimator(
            config,
            model_seed,
            y_train,
            model_params=model_params,
            rf_n_jobs=rf_n_jobs,
        )

    monkeypatch.setattr(cv_mod, "_prepare_source_selection", _fake_prepare_source_selection)
    monkeypatch.setattr(cv_mod, "_build_estimator", _wrapped_build_estimator)

    _ = run_outer_cv(config, split_artifacts.split_manifest)

    assert expected_fold_count == 2
    assert len(selection_n_jobs) == expected_fold_count
    assert set(selection_n_jobs) == {2}
    assert estimator_n_jobs
    assert set(estimator_n_jobs) == {2}


def test_compute_fold_metrics_returns_nan_when_metrics_are_undefined() -> None:
    y_true = np.array([], dtype=int)
    prob = np.array([], dtype=float)

    metrics = _compute_fold_metrics(y_true, prob, threshold=0.5)

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["mcc"])
    assert np.isnan(metrics["brier"])


@pytest.mark.parametrize("label", [0, 1])
def test_compute_fold_metrics_single_class_keeps_only_brier(label: int) -> None:
    y_true = np.full(3, label, dtype=int)
    prob = np.array([0.1, 0.4, 0.8], dtype=float)

    metrics = _compute_fold_metrics(y_true, prob, threshold=0.5)

    for metric_name in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc"):
        assert np.isnan(metrics[metric_name])
    assert metrics["brier"] == pytest.approx(brier_score_loss(y_true, prob))


def test_compute_fold_metrics_pr_auc_key_is_average_precision_not_trapezoidal_auc() -> None:
    y_true = np.array([1, 0, 1, 0], dtype=int)
    prob = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)

    metrics = _compute_fold_metrics(y_true, prob, threshold=0.5)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    trapezoidal_pr_auc = auc(recall, precision)

    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y_true, prob))
    assert metrics["pr_auc"] != pytest.approx(trapezoidal_pr_auc)


def test_apply_correlation_filter_drops_highly_correlated_feature_pearson(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  correlation_filter:
    enabled: true
    method: pearson
    max_abs_correlation: 0.9
""".strip(),
            )
        ]
    )
    x_train_log = np.array(
        [
            [0.0, 0.0, 0.1],
            [1.0, 1.0, 0.3],
            [2.0, 2.0, 0.2],
            [3.0, 3.0, 0.4],
        ],
        dtype=float,
    )

    kept = _apply_correlation_filter(
        config=config,
        x_train_log=x_train_log,
        selected=np.array([0, 1, 2], dtype=int),
        feature_names=["OG1", "OG2", "OG3"],
    )

    assert kept.tolist() == [0, 2]


def test_apply_correlation_filter_drops_highly_correlated_feature_spearman(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  correlation_filter:
    enabled: true
    method: spearman
    max_abs_correlation: 0.9
""".strip(),
            )
        ]
    )
    x_train_log = np.array(
        [
            [0.0, 0.0, 0.1],
            [1.0, 1.0, 0.3],
            [2.0, 2.0, 0.2],
            [3.0, 3.0, 0.4],
        ],
        dtype=float,
    )

    kept = _apply_correlation_filter(
        config=config,
        x_train_log=x_train_log,
        selected=np.array([0, 1, 2], dtype=int),
        feature_names=["OG1", "OG2", "OG3"],
    )

    assert kept.tolist() == [0, 2]


def test_preprocess_fold_rejects_negative_tpm_values(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    x_train_raw = np.array([[1.0, -0.1], [2.0, 0.3]], dtype=float)
    x_valid_raw = np.array([[1.0, 0.2]], dtype=float)

    with pytest.raises(CVError, match="TPM values must be non-negative"):
        _preprocess_fold(config, x_train_raw, x_valid_raw, ["OG1", "OG2"])


def test_preprocess_fold_scales_validation_with_training_statistics(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    x_train_raw = np.array(
        [
            [1.0, 3.0],
            [2.0, 5.0],
            [4.0, 9.0],
        ],
        dtype=float,
    )
    x_valid_raw = np.array(
        [
            [6.0, 2.0],
            [8.0, 7.0],
        ],
        dtype=float,
    )

    x_train_scaled, x_valid_scaled, selected_features = _preprocess_fold(
        config,
        x_train_raw,
        x_valid_raw,
        ["OG1", "OG2"],
        y_train=np.array([0, 1, 0], dtype=int),
    )

    x_train_log = np.log1p(x_train_raw)
    x_valid_log = np.log1p(x_valid_raw)
    train_mean = np.mean(x_train_log, axis=0)
    train_std = np.std(x_train_log, axis=0, ddof=0)

    expected_train = (x_train_log - train_mean) / train_std
    expected_valid = (x_valid_log - train_mean) / train_std

    assert selected_features == ["OG1", "OG2"]
    assert x_train_scaled == pytest.approx(expected_train)
    assert x_valid_scaled == pytest.approx(expected_valid)


def test_select_feature_indices_raises_when_filters_remove_all_features(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  sparse_feature_filter:
    enabled: true
    min_nonzero_fraction_in_at_least_one_trait: 0.5
""".strip(),
            )
        ]
    )
    x_train_log = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(CVError, match="removed all features"):
        _select_feature_indices(
            config,
            x_train_log,
            ["OG1", "OG2"],
            y_train=np.array([0, 1, 0], dtype=int),
        )


def test_expression_matrix_builder_rejects_missing_expression_file(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    original_scan_csv = cv_mod.pl.scan_csv

    def _raise_file_not_found(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError

    cv_mod.pl.scan_csv = _raise_file_not_found  # type: ignore[assignment]
    try:
        with pytest.raises(CVError, match="Input file not found"):
            ExpressionMatrixBuilder(config)
    finally:
        cv_mod.pl.scan_csv = original_scan_csv  # type: ignore[assignment]


def test_expression_matrix_builder_rejects_missing_expression_file_without_mock(
    tmp_path: Path,
) -> None:
    metadata, _ = _write_fixture(tmp_path)
    missing_tpm = tmp_path / "missing_tpm.tsv"
    config = load_and_resolve_config([_config_path(tmp_path, metadata, missing_tpm)])

    with pytest.raises(CVError, match="Input file not found"):
        ExpressionMatrixBuilder(config)


def test_expression_matrix_builder_rejects_missing_required_columns(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "sp1\t1\tg1\tno",
                "sp2\t0\tg1\tno",
            ]
        )
        + "\n",
    )
    bad_tpm = _write(
        tmp_path / "bad_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup",
                "sp1\tOG1",
                "sp2\tOG1",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, bad_tpm)])

    with pytest.raises(CVError, match="Missing required columns"):
        ExpressionMatrixBuilder(config)


def test_expression_matrix_builder_wraps_ragged_expression_row(tmp_path: Path) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "ragged_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0\textra",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    with pytest.raises(CVError, match="Failed to read expression data"):
        ExpressionMatrixBuilder(config).build_matrix(["sp1", "sp2"])


def test_expression_matrix_builder_build_matrix_rejects_empty_species(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    builder = ExpressionMatrixBuilder(config)

    with pytest.raises(CVError, match="No species were provided"):
        builder.build_matrix([])


def test_expression_matrix_builder_build_matrix_rejects_missing_selected_species(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    builder = ExpressionMatrixBuilder(config)

    with pytest.raises(CVError, match="Expression data is missing selected species"):
        builder.build_matrix(["sp1", "sp_missing"])


def test_expression_matrix_builder_build_matrix_respects_feature_order_and_zero_fills(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    builder = ExpressionMatrixBuilder(config)

    matrix, features = builder.build_matrix(["sp1", "sp2"], feature_order=["OG2", "OG_missing"])

    assert features == ["OG2", "OG_missing"]
    assert matrix.tolist() == [[0.5, 0.0], [0.3, 0.0]]


@pytest.mark.parametrize(
    "feature",
    ["__species", "__row_idx", "__phenoradar_missing_feature__"],
)
def test_expression_matrix_builder_handles_feature_names_that_match_internal_names(
    tmp_path: Path,
    feature: str,
) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "internal_name_feature_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                f"sp1\t{feature}\t1.5",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    matrix, features = ExpressionMatrixBuilder(config).build_matrix(["sp1"])

    assert features == [feature]
    assert matrix.tolist() == [[1.5]]


def test_expression_matrix_builder_validates_features_outside_requested_order(
    tmp_path: Path,
) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "invalid_extra_feature_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp1\tOG_extra\tbad",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    with pytest.raises(CVError, match=r"OG_extra.*\(non-numeric\)"):
        ExpressionMatrixBuilder(config).build_matrix(["sp1"], feature_order=["OG1"])


@pytest.mark.parametrize(
    ("raw_value", "reasons"),
    [
        ("1.0", "missing-feature"),
        ("bad", "non-numeric,missing-feature"),
    ],
)
def test_expression_matrix_builder_rejects_missing_feature_identifier(
    tmp_path: Path,
    raw_value: str,
    reasons: str,
) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "missing_feature_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                f"sp1\t\t{raw_value}",
                "sp1\tOG1\t1.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    with pytest.raises(CVError, match=rf"feature=<missing>.*\({reasons}\)"):
        ExpressionMatrixBuilder(config).build_matrix(["sp1"])


@pytest.mark.parametrize(
    ("raw_value", "reason"),
    [
        ("", "missing"),
        (" ", "missing"),
        ("NA", "non-numeric"),
        ("bad", "non-numeric"),
        ("NaN", "non-finite"),
        ("inf", "non-finite"),
        ("-inf", "non-finite"),
        ("-0.1", "negative"),
    ],
)
def test_expression_matrix_builder_rejects_invalid_tpm_rows_before_pivot(
    tmp_path: Path,
    raw_value: str,
    reason: str,
) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "invalid_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                f"sp1\tOG1\t{raw_value}",
                "sp1\tOG2\t1.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    builder = ExpressionMatrixBuilder(config)

    with pytest.raises(CVError) as exc_info:
        builder.build_matrix(["sp1"])

    message = str(exc_info.value)
    assert "TPM values must be non-negative finite numbers" in message
    assert "first_invalid_line=2" in message
    assert "species='sp1'" in message
    assert "feature='OG1'" in message
    assert f"({reason})" in message


def test_expression_matrix_builder_rejects_negative_duplicate_before_sum(
    tmp_path: Path,
) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "negative_duplicate_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t-1.0",
                "sp1\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    with pytest.raises(CVError, match=r"first_invalid_line=2.*\(negative\)"):
        ExpressionMatrixBuilder(config).build_matrix(["sp1"])


def test_expression_matrix_builder_sums_valid_duplicate_rows(tmp_path: Path) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "duplicate_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp1\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    matrix, features = ExpressionMatrixBuilder(config).build_matrix(["sp1"])

    assert features == ["OG1"]
    assert matrix.tolist() == [[3.0]]


def test_expression_matrix_builder_rejects_non_finite_duplicate_sum(tmp_path: Path) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "overflow_duplicate_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1e308",
                "sp1\tOG1\t1e308",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    with pytest.raises(CVError, match=r"first_invalid_line=2.*\(non-finite-after-sum\)"):
        ExpressionMatrixBuilder(config).build_matrix(["sp1"])


def test_expression_matrix_builder_rejects_invalid_tpm_beyond_inference_window(
    tmp_path: Path,
) -> None:
    metadata, _ = _write_fixture(tmp_path)
    rows = ["species\torthogroup\ttpm"]
    rows.extend(f"sp1\tOG{index:03d}\t1.0" for index in range(101))
    rows.append("sp1\tOG_bad\tbad")
    tpm = _write(tmp_path / "late_invalid_tpm.tsv", "\n".join(rows) + "\n")
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    with pytest.raises(CVError, match=r"first_invalid_line=103.*\(non-numeric\)"):
        ExpressionMatrixBuilder(config).build_matrix(["sp1"])


def test_expression_matrix_builder_chunking_rejects_invalid_tpm(tmp_path: Path) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "chunked_invalid_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp1\tOG2\tNaN",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  max_pivot_cells: 1
""".strip(),
            )
        ]
    )

    with pytest.raises(CVError, match=r"first_invalid_line=3.*\(non-finite\)"):
        ExpressionMatrixBuilder(config).build_matrix(["sp1"])


def test_expression_matrix_builder_cache_species_rejects_invalid_tpm(
    tmp_path: Path,
) -> None:
    metadata, _ = _write_fixture(tmp_path)
    tpm = _write(
        tmp_path / "cached_invalid_tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\tbad",
                "sp2\tOG1\t1.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    builder = ExpressionMatrixBuilder(config)

    with pytest.raises(CVError, match=r"first_invalid_line=2.*\(non-numeric\)"):
        builder.cache_species(["sp1", "sp2"])

    assert builder._cache_tempdir is None
    assert builder._cached_long_path is None
    assert builder._cached_species is None


def test_expression_matrix_builder_cache_species_reuses_cached_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    builder = ExpressionMatrixBuilder(config)
    builder.cache_species(["sp1", "sp2"])

    def _fail_raw_scan(_species: list[str]) -> pl.LazyFrame:
        raise AssertionError("raw scan should not be used for cached species")

    monkeypatch.setattr(builder, "_raw_long_scan_for_species", _fail_raw_scan)
    matrix, features = builder.build_matrix(["sp2"], feature_order=["OG1"])

    assert features == ["OG1"]
    assert matrix.tolist() == [[2.0]]


def test_preprocess_train_and_target_rejects_negative_tpm_values(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    with pytest.raises(CVError, match="TPM values must be non-negative"):
        _preprocess_train_and_target(
            config,
            np.array([[1.0, -0.1], [2.0, 0.3]], dtype=float),
            np.array([[1.0, 0.2]], dtype=float),
            ["OG1", "OG2"],
        )


def test_preprocess_train_and_target_returns_training_fitted_scaler(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    x_train_raw = np.array(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [7.0, 18.0],
        ],
        dtype=float,
    )
    x_target_raw = np.array(
        [
            [2.0, 11.0],
            [9.0, 20.0],
        ],
        dtype=float,
    )

    x_train_scaled, x_target_scaled, selected_features, scaler = _preprocess_train_and_target(
        config,
        x_train_raw,
        x_target_raw,
        ["OG1", "OG2"],
        y_train=np.array([0, 1, 0], dtype=int),
    )

    x_train_log = np.log1p(x_train_raw)
    x_target_log = np.log1p(x_target_raw)
    train_mean = np.mean(x_train_log, axis=0)
    train_std = np.std(x_train_log, axis=0, ddof=0)

    expected_train = (x_train_log - train_mean) / train_std
    expected_target = (x_target_log - train_mean) / train_std

    assert selected_features == ["OG1", "OG2"]
    assert scaler.mean_ == pytest.approx(train_mean)
    assert scaler.scale_ == pytest.approx(train_std)
    assert x_train_scaled == pytest.approx(expected_train)
    assert x_target_scaled == pytest.approx(expected_target)


def test_sample_percentile_rank_transform_preserves_zero_values() -> None:
    raw = np.array(
        [
            [0.0, 10.0, 5.0],
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    transformed = apply_expression_transform(raw, "sample_percentile_rank")

    expected = np.array(
        [
            [0.0, 1.0, 0.5],
            [0.75, 0.0, 0.75],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    assert transformed == pytest.approx(expected)


def test_preprocess_train_and_target_can_disable_feature_scaling(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                """
preprocess:
  expression_transform:
    method: sample_percentile_rank
  sparse_feature_filter:
    enabled: false
  feature_scaling:
    method: none
""",
            )
        ]
    )
    x_train_raw = np.array(
        [
            [0.0, 10.0, 5.0],
            [2.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    x_target_raw = np.array([[1.0, 3.0, 0.0]], dtype=float)

    x_train, x_target, selected_features, scaler = _preprocess_train_and_target(
        config,
        x_train_raw,
        x_target_raw,
        ["OG1", "OG2", "OG3"],
    )

    assert selected_features == ["OG1", "OG2", "OG3"]
    assert scaler is None
    assert x_train == pytest.approx(
        np.array(
            [
                [0.0, 1.0, 0.5],
                [0.75, 0.0, 0.75],
            ],
            dtype=float,
        )
    )
    assert x_target == pytest.approx(np.array([[0.5, 1.0, 0.0]], dtype=float))


def test_build_prediction_table_rejects_probability_length_mismatch() -> None:
    with pytest.raises(CVError, match="species/probability length mismatch"):
        _build_prediction_table(
            species=["sp1", "sp2"],
            prob=np.array([0.5], dtype=float),
            fixed_threshold=0.5,
            uncertainty_std=None,
        )


def test_build_prediction_table_rejects_uncertainty_length_mismatch() -> None:
    with pytest.raises(CVError, match="species/uncertainty length mismatch"):
        _build_prediction_table(
            species=["sp1", "sp2"],
            prob=np.array([0.2, 0.8], dtype=float),
            fixed_threshold=0.5,
            uncertainty_std=np.array([0.1], dtype=float),
        )


def test_build_prediction_table_rejects_true_label_length_mismatch() -> None:
    with pytest.raises(CVError, match="species/true_label length mismatch"):
        _build_prediction_table(
            species=["sp1", "sp2"],
            prob=np.array([0.2, 0.8], dtype=float),
            fixed_threshold=0.5,
            uncertainty_std=None,
            true_label=np.array([1], dtype=int),
        )


def test_build_prediction_table_can_include_empty_true_label_column() -> None:
    table = _build_prediction_table(
        species=["sp1", "sp2"],
        prob=np.array([0.2, 0.8], dtype=float),
        fixed_threshold=0.5,
        uncertainty_std=None,
        include_true_label_column=True,
    )

    assert "true_label" in table.columns
    assert table.get_column("true_label").null_count() == 2


def test_fit_estimator_rejects_unsupported_sample_weight_before_fit() -> None:
    class _NoWeightEstimator:
        def __init__(self) -> None:
            self.call_count = 0

        def fit(self, _x: np.ndarray, _y: np.ndarray) -> None:
            self.call_count += 1

    estimator = _NoWeightEstimator()
    x = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([0, 1], dtype=int)
    sample_weight = np.array([1.0, 1.0], dtype=float)

    with pytest.raises(CVError, match="does not support sample_weight"):
        _fit_estimator(estimator, x, y, sample_weight)  # type: ignore[arg-type]

    assert estimator.call_count == 0


def test_fit_estimator_does_not_hide_internal_type_error_or_retry() -> None:
    class _BrokenWeightEstimator:
        def __init__(self) -> None:
            self.call_count = 0

        def fit(
            self,
            _x: np.ndarray,
            _y: np.ndarray,
            sample_weight: np.ndarray | None = None,
        ) -> None:
            self.call_count += 1
            assert sample_weight is not None
            raise TypeError("internal fit failure")

    estimator = _BrokenWeightEstimator()
    x = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([0, 1], dtype=int)
    sample_weight = np.array([1.0, 1.0], dtype=float)

    with pytest.raises(
        CVError,
        match="fit failed while applying sample_weight: internal fit failure",
    ) as exc_info:
        _fit_estimator(estimator, x, y, sample_weight)  # type: ignore[arg-type]

    assert estimator.call_count == 1
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_fit_estimator_passes_sample_weight_once_without_modification() -> None:
    class _WeightEstimator:
        def __init__(self) -> None:
            self.call_count = 0
            self.received_sample_weight: np.ndarray | None = None

        def fit(
            self,
            _x: np.ndarray,
            _y: np.ndarray,
            sample_weight: np.ndarray | None = None,
        ) -> None:
            self.call_count += 1
            self.received_sample_weight = sample_weight

    estimator = _WeightEstimator()
    x = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([0, 1], dtype=int)
    sample_weight = np.array([0.5, 1.5], dtype=float)

    _fit_estimator(estimator, x, y, sample_weight)  # type: ignore[arg-type]

    assert estimator.call_count == 1
    assert estimator.received_sample_weight is sample_weight


def test_fit_estimator_captures_logistic_convergence_diagnostic() -> None:
    estimator = LogisticRegression(solver="saga", max_iter=1, random_state=42)
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 1.0],
            [0.2, 2.0],
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    with pytest.warns(ConvergenceWarning):
        diagnostic = _fit_estimator(estimator, x, y, sample_weight=None)

    assert diagnostic.estimator_class == "LogisticRegression"
    assert diagnostic.convergence_applicable is True
    assert diagnostic.converged is False
    assert diagnostic.n_iter_values == (1,)
    assert diagnostic.max_iter == 1
    assert diagnostic.convergence_warning_count == 1
    assert diagnostic.convergence_warning_messages


def test_linear_svm_accepts_sample_weight_with_metadata_routing_enabled(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model:
  name: linear_svm
""".strip(),
            )
        ]
    )
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    sample_weight = np.array([1.0, 1.5, 0.5, 1.0, 1.5, 0.5], dtype=float)
    estimator = _build_estimator(config, model_seed=42, y_train=y)

    with config_context(enable_metadata_routing=True):
        _fit_estimator(estimator, x, y, sample_weight)

    assert hasattr(estimator, "calibrated_classifiers_")


def test_predict_positive_probability_rejects_invalid_shape() -> None:
    class _BadEstimator:
        def predict_proba(self, _x: np.ndarray) -> np.ndarray:
            return np.array([0.1, 0.9], dtype=float)

    with pytest.raises(CVError, match="unexpected shape"):
        _predict_positive_probability(_BadEstimator(), np.array([[0.0], [1.0]], dtype=float))


def test_aggregate_probabilities_rejects_unknown_aggregation() -> None:
    with pytest.raises(CVError, match="Unsupported ensemble.probability_aggregation"):
        _aggregate_probabilities([np.array([0.2, 0.8], dtype=float)], "invalid")


def test_build_estimator_linear_svm_rejects_single_class_training_fold(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model:
  name: linear_svm
""".strip(),
            )
        ]
    )

    with pytest.raises(CVError, match="calibration requires at least 2 samples per class"):
        _build_estimator(config, model_seed=123, y_train=np.array([1, 1, 1], dtype=int))


def test_build_estimator_logistic_elasticnet_uses_l1_ratio_semantics() -> None:
    config = load_and_resolve_config([], allow_empty=True)
    x_train = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    y_train = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=int)

    l2_estimator = _build_estimator(
        config,
        model_seed=123,
        y_train=y_train,
        model_params={"C": 0.7, "l1_ratio": 0.0, "max_iter": 5000},
    )
    elasticnet_estimator = _build_estimator(
        config,
        model_seed=123,
        y_train=y_train,
        model_params={"C": 0.7, "l1_ratio": 0.5, "max_iter": 5000},
    )

    assert isinstance(l2_estimator, LogisticRegression)
    assert isinstance(elasticnet_estimator, LogisticRegression)
    assert l2_estimator.solver == "saga"
    assert elasticnet_estimator.solver == "saga"
    assert l2_estimator.l1_ratio == pytest.approx(0.0)
    assert elasticnet_estimator.l1_ratio == pytest.approx(0.5)

    _fit_estimator(l2_estimator, x_train, y_train, sample_weight=None)
    _fit_estimator(elasticnet_estimator, x_train, y_train, sample_weight=None)

    assert not np.allclose(l2_estimator.coef_, elasticnet_estimator.coef_)


def test_build_estimator_logistic_liblinear_supports_l1_and_sample_weight(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "liblinear.yml",
        """
model:
  name: logistic_elasticnet
  logistic_solver: liblinear
model_selection:
  search_space:
    l1_ratio: [1]
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([config_path])
    x_train = np.array(
        [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [1.5, 1.0], [2.0, 0.0], [2.5, 1.0]],
        dtype=float,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    sample_weight = np.array([1.0, 0.5, 1.5, 1.0, 0.5, 1.5], dtype=float)
    estimator = _build_estimator(
        config,
        model_seed=123,
        y_train=y_train,
        model_params={"C": 1.0, "l1_ratio": 1.0, "max_iter": 5000},
    )

    diagnostic = _fit_estimator(estimator, x_train, y_train, sample_weight)

    assert isinstance(estimator, LogisticRegression)
    assert estimator.solver == "liblinear"
    assert estimator.l1_ratio == pytest.approx(1.0)
    assert diagnostic.convergence_applicable is True
    assert diagnostic.converged is True
    assert _predict_positive_probability(estimator, x_train).shape == (6,)


def test_build_estimator_random_forest_respects_explicit_n_jobs(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model:
  name: random_forest
""".strip(),
            )
        ]
    )

    estimator = _build_estimator(
        config,
        model_seed=123,
        y_train=np.array([0, 1], dtype=int),
        rf_n_jobs=2,
    )

    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.n_jobs == 2


def test_inner_cv_splits_requires_strategy_when_mutated_to_none(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_missing_strategy = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"inner_cv_strategy": None}
            ),
        }
    )

    with pytest.raises(CVError, match="inner_cv_strategy is required"):
        _inner_cv_splits(
            config_missing_strategy,
            np.array([0, 1], dtype=int),
            np.array(["g1", "g2"], dtype=str),
        )


def test_inner_cv_splits_requires_n_splits_for_group_kfold(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_group_kfold = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"inner_cv_strategy": "group_kfold", "inner_cv_n_splits": None}
            ),
        }
    )

    with pytest.raises(CVError, match="inner_cv_n_splits is required"):
        _inner_cv_splits(
            config_group_kfold,
            np.array([0, 1, 0, 1], dtype=int),
            np.array(["g1", "g1", "g2", "g2"], dtype=str),
        )


def test_inner_cv_splits_wraps_value_error_from_splitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingLogo:
        def split(self, *_args: object, **_kwargs: object) -> object:
            def _iter() -> object:
                raise ValueError("boom")
                yield  # pragma: no cover

            return _iter()

    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_with_logo = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"inner_cv_strategy": "logo"}
            ),
        }
    )
    monkeypatch.setattr(cv_mod, "LeaveOneGroupOut", lambda: _FailingLogo())

    with pytest.raises(CVError, match="Inner CV split error"):
        _inner_cv_splits(
            config_with_logo,
            np.array([0, 1], dtype=int),
            np.array(["g1", "g2"], dtype=str),
        )


@pytest.mark.parametrize(
    "method", ["none", "log1p", "sample_rank", "sample_percentile_rank"]
)
def test_inner_cv_preprocessing_applies_row_local_expression_transform_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra=f"""
preprocess:
  expression_transform:
    method: {method}
  sparse_feature_filter:
    enabled: false
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    original_transform = cv_mod._apply_expression_transform_for_config
    call_count = 0
    x_source_raw = np.array(
        [[0.0, 3.0], [2.0, 1.0], [1.0, 4.0], [3.0, 2.0]],
        dtype=float,
    )
    y_source = np.array([0, 1, 0, 1], dtype=int)
    groups_source = np.array(["g1", "g1", "g2", "g2"], dtype=str)
    expected = []
    for train_idx, valid_idx, _inner_fold_id in cv_mod._inner_cv_splits(
        config, y_source, groups_source
    ):
        expected.append(
            cv_mod._preprocess_fold(
                config,
                x_source_raw[train_idx, :],
                x_source_raw[valid_idx, :],
                ["OG1", "OG2"],
                y_train=y_source[train_idx],
                groups_train=None,
            )
        )

    def _counted_transform(config_arg: object, matrix: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return original_transform(config_arg, matrix)  # type: ignore[arg-type]

    monkeypatch.setattr(cv_mod, "_apply_expression_transform_for_config", _counted_transform)
    folds = cv_mod._build_inner_cv_preprocessed_folds(
        config=config,
        x_source_raw=x_source_raw,
        y_source=y_source,
        groups_source=groups_source,
        contrast_groups_source=None,
        feature_names=["OG1", "OG2"],
    )

    assert len(folds) == 2
    assert call_count == 1
    for fold, (expected_train, expected_valid, _expected_features) in zip(
        folds, expected, strict=True
    ):
        np.testing.assert_array_equal(fold.x_train, expected_train)
        np.testing.assert_array_equal(fold.x_valid, expected_valid)


def test_prepare_source_selection_wraps_candidate_generation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: (_ for _ in ()).throw(ModelSelectionError("candidate generation failed")),
    )

    with pytest.raises(CVError, match="candidate generation failed"):
        _prepare_source_selection(
            config=config,
            training_scope_id="fold_0",
            source_sample_set_id=0,
            sampled_idx=np.array([0, 1], dtype=int),
            x_train_raw=np.array([[1.0], [2.0]], dtype=float),
            y_train=np.array([0, 1], dtype=int),
            groups_train=np.array(["g1", "g2"], dtype=str),
            feature_names=["OG1"],
            warnings=[],
        )


def test_prepare_source_selection_rejects_empty_candidate_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    monkeypatch.setattr(cv_mod, "generate_candidates", lambda **_kwargs: [])

    with pytest.raises(CVError, match="produced zero candidates"):
        _prepare_source_selection(
            config=config,
            training_scope_id="fold_0",
            source_sample_set_id=0,
            sampled_idx=np.array([0, 1], dtype=int),
            x_train_raw=np.array([[1.0], [2.0]], dtype=float),
            y_train=np.array([0, 1], dtype=int),
            groups_train=np.array(["g1", "g2"], dtype=str),
            feature_names=["OG1"],
            warnings=[],
        )


def test_prepare_source_selection_reuses_inner_fold_preprocessing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: [
            Candidate(candidate_index=0, params={"C": 0.1}),
            Candidate(candidate_index=1, params={"C": 1.0}),
            Candidate(candidate_index=2, params={"C": 10.0}),
        ],
    )

    original_preprocess_fold = cv_mod._preprocess_transformed_fold_with_counts
    preprocess_call_count = 0

    def _counting_preprocess_fold(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray, list[str], cv_mod.FeatureFilterCounts]:
        nonlocal preprocess_call_count
        preprocess_call_count += 1
        return original_preprocess_fold(*args, **kwargs)

    monkeypatch.setattr(
        cv_mod,
        "_preprocess_transformed_fold_with_counts",
        _counting_preprocess_fold,
    )

    seen_cache_ids: list[int] = []

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        preprocessed_folds = kwargs["preprocessed_folds"]
        if not isinstance(preprocessed_folds, list):
            raise AssertionError("preprocessed_folds must be a list")
        seen_cache_ids.append(id(preprocessed_folds))
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        return float(candidate.candidate_index), []

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)

    result = _prepare_source_selection(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=np.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
            ],
            dtype=float,
        ),
        y_train=np.array([1, 0, 1, 0], dtype=int),
        groups_train=np.array(["g1", "g1", "g2", "g2"], dtype=str),
        feature_names=["OG1", "OG2"],
        warnings=[],
    )

    assert preprocess_call_count == 2
    assert seen_cache_ids
    assert len(seen_cache_ids) == 3
    assert len(set(seen_cache_ids)) == 1
    assert result.n_scored_candidates == 3


def test_prepare_source_selection_warns_when_selected_candidate_count_is_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selected_candidate_count: 3
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: [Candidate(candidate_index=0, params={"C": 1.0})],
    )
    monkeypatch.setattr(
        cv_mod,
        "_score_candidate_inner_cv",
        lambda **_kwargs: (0.5, []),
    )
    warnings: list[str] = []

    result = _prepare_source_selection(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=warnings,
    )

    assert result.selected_candidate_count_effective == 1
    assert any(
        "selected_candidate_count exceeded available candidates" in item for item in warnings
    )


def test_prepare_source_selection_deduplicates_selected_candidates_by_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selected_candidate_count: 2
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: [
            Candidate(candidate_index=0, params={"C": 1.0}),
            Candidate(candidate_index=1, params={"C": 1.0}),
        ],
    )

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        if candidate.candidate_index == 0:
            return 0.20, []
        return 0.40, []

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)
    warnings: list[str] = []

    result = _prepare_source_selection(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=warnings,
    )

    assert result.n_scored_candidates == 2
    assert result.n_available_candidates == 1
    assert result.selected_candidate_count_effective == 1
    assert len(result.selected_candidates) == 1
    assert result.selected_candidates[0].candidate.candidate_index == 0
    assert any("deduplicated candidates with identical params" in item for item in warnings)
    assert any(
        "selected_candidate_count exceeded available candidates" in item for item in warnings
    )


def test_prepare_source_selection_supports_selected_candidate_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selected_candidate_percent: 50
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: [
            Candidate(candidate_index=0, params={"C": 0.1}),
            Candidate(candidate_index=1, params={"C": 1.0}),
            Candidate(candidate_index=2, params={"C": 10.0}),
        ],
    )

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        return float(candidate.candidate_index), []

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)

    result = _prepare_source_selection(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=[],
    )

    assert result.n_available_candidates == 3
    assert result.selected_candidate_count_requested == 2
    assert result.selected_candidate_count_effective == 2
    assert [item.candidate.candidate_index for item in result.selected_candidates] == [0, 1]


def test_prepare_source_selection_parallel_scoring_caps_rf_n_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model:
  name: random_forest
model_selection:
  search_strategy: grid
  selected_candidate_count: 1
  inner_cv_strategy: logo
  selection_metric: mcc
runtime:
  n_jobs: 5
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: [
            Candidate(candidate_index=0, params={"n_estimators": 10}),
            Candidate(candidate_index=1, params={"n_estimators": 20}),
        ],
    )
    captured_n_jobs: list[int] = []

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        captured_n_jobs.append(int(kwargs["estimator_n_jobs"]))
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        return float(candidate.candidate_index), []

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)

    result = _prepare_source_selection(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=[],
    )

    assert captured_n_jobs == [2, 2]
    assert result.selected_candidates[0].candidate.candidate_index == 1
    assert result.selected_candidate_count_effective == 1


def test_prepare_source_selection_prefers_lower_log_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
  selection_metric: log_loss
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: [
            Candidate(candidate_index=0, params={"C": 0.5}),
            Candidate(candidate_index=1, params={"C": 1.0}),
        ],
    )

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        if candidate.candidate_index == 0:
            return 0.25, []
        return 0.40, []

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)

    result = _prepare_source_selection(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=[],
    )

    assert result.selected_candidates[0].candidate.candidate_index == 0


def test_prepare_source_selection_one_se_prefers_simpler_candidate_within_best_se(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
  selection_metric: log_loss
  selection_rule: one_se
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: [
            Candidate(candidate_index=0, params={"C": 0.01}),
            Candidate(candidate_index=1, params={"C": 0.1}),
            Candidate(candidate_index=2, params={"C": 1.0}),
        ],
    )

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        fold_values_by_candidate = {
            0: [0.40, 0.40],
            1: [0.23, 0.23],
            2: [0.15, 0.25],
        }
        fold_values = fold_values_by_candidate[candidate.candidate_index]
        rows = [
            {
                "candidate_index": candidate.candidate_index,
                "inner_fold_id": str(inner_fold_id),
                "metric_name": "log_loss",
                "metric_value": value,
                "params_json": "{}",
            }
            for inner_fold_id, value in enumerate(fold_values, start=1)
        ]
        return float(np.mean(fold_values)), rows

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)

    result = _prepare_source_selection(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=[],
    )

    selected = result.selected_candidates[0]
    assert selected.candidate.candidate_index == 1
    assert selected.score == pytest.approx(0.23)
    assert selected.selection_rule == "one_se"


def test_score_candidate_inner_cv_uses_estimator_n_jobs_for_native_thread_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model:
  name: random_forest
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
runtime:
  n_jobs: 5
""".strip(),
            )
        ]
    )
    captured_native_limits: list[int] = []

    def _fake_with_native_thread_limit(
        n_jobs: int, func: object, *args: object, **kwargs: object
    ) -> tuple[float, list[dict[str, object]]]:
        captured_native_limits.append(int(n_jobs))
        if not callable(func):
            raise AssertionError("func must be callable")
        result = func(*args, **kwargs)
        if not isinstance(result, tuple):
            raise AssertionError("expected tuple result")
        return result

    monkeypatch.setattr(cv_mod, "_with_native_thread_limit", _fake_with_native_thread_limit)

    preprocessed_folds = [
        cv_mod.InnerCvPreprocessedFold(
            inner_fold_id="0",
            x_train=np.array([[0.0], [1.0]], dtype=float),
            x_valid=np.array([[0.5], [1.5]], dtype=float),
            y_train=np.array([0, 1], dtype=int),
            y_valid=np.array([0, 1], dtype=int),
            sample_weight=None,
        ),
        cv_mod.InnerCvPreprocessedFold(
            inner_fold_id="1",
            x_train=np.array([[1.0], [2.0]], dtype=float),
            x_valid=np.array([[1.5], [2.5]], dtype=float),
            y_train=np.array([0, 1], dtype=int),
            y_valid=np.array([0, 1], dtype=int),
            sample_weight=None,
        ),
    ]

    score, trial_rows = cv_mod._score_candidate_inner_cv(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        candidate=Candidate(candidate_index=0, params={"n_estimators": 10}),
        preprocessed_folds=preprocessed_folds,
        estimator_n_jobs=2,
    )

    assert not np.isnan(score)
    assert len(trial_rows) == 2
    assert captured_native_limits == [2]


def test_with_native_thread_limit_reuses_threadpool_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_count = 0
    captured_limits: list[int] = []

    class _FakeLimiter:
        def __init__(self, limit: int) -> None:
            self._limit = int(limit)

        def __enter__(self) -> _FakeLimiter:
            return self

        def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
            return None

    class _FakeController:
        def __init__(self) -> None:
            nonlocal init_count
            init_count += 1

        def limit(self, *, limits: int) -> _FakeLimiter:
            captured_limits.append(int(limits))
            return _FakeLimiter(int(limits))

    monkeypatch.setattr(cv_mod, "ThreadpoolController", _FakeController)
    monkeypatch.setattr(cv_mod, "_THREADPOOL_CONTROLLER", None)

    first = cv_mod._with_native_thread_limit(2, lambda value: value + 1, 1)
    second = cv_mod._with_native_thread_limit(3, lambda: "ok")

    assert first == 2
    assert second == "ok"
    assert init_count == 1
    assert captured_limits == [2, 3]


def test_score_candidate_inner_cv_reuses_logistic_estimators_for_warm_start_path(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "warm_start.yml",
        """
model:
  name: logistic_elasticnet
  logistic_solver: saga
  logistic_warm_start_path: true
model_selection:
  search_space:
    C: [0.1, 1.0]
    l1_ratio: [1]
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([config_path])
    folds = [
        cv_mod.InnerCvPreprocessedFold(
            inner_fold_id="0",
            x_train=np.array(
                [[0.0, 0.0], [0.2, 1.0], [1.0, 0.2], [1.2, 1.0]], dtype=float
            ),
            x_valid=np.array([[0.1, 0.1], [1.1, 0.9]], dtype=float),
            y_train=np.array([0, 0, 1, 1], dtype=int),
            y_valid=np.array([0, 1], dtype=int),
            sample_weight=None,
        ),
        cv_mod.InnerCvPreprocessedFold(
            inner_fold_id="1",
            x_train=np.array(
                [[0.0, 1.0], [0.3, 0.0], [1.0, 1.0], [1.3, 0.0]], dtype=float
            ),
            x_valid=np.array([[0.2, 0.8], [1.2, 0.2]], dtype=float),
            y_train=np.array([0, 0, 1, 1], dtype=int),
            y_valid=np.array([0, 1], dtype=int),
            sample_weight=None,
        ),
    ]
    cache: dict[str, LogisticRegression] = {}

    first_score, first_rows = cv_mod._score_candidate_inner_cv(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        candidate=Candidate(candidate_index=0, params={"C": 0.1, "l1_ratio": 1}),
        preprocessed_folds=folds,
        warm_start_estimators=cache,
    )
    cached_ids = {fold_id: id(estimator) for fold_id, estimator in cache.items()}
    second_score, second_rows = cv_mod._score_candidate_inner_cv(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        candidate=Candidate(candidate_index=1, params={"C": 1.0, "l1_ratio": 1}),
        preprocessed_folds=folds,
        warm_start_estimators=cache,
    )

    assert np.isfinite(first_score)
    assert np.isfinite(second_score)
    assert len(first_rows) == len(second_rows) == 2
    assert set(cache) == {"0", "1"}
    assert {fold_id: id(estimator) for fold_id, estimator in cache.items()} == cached_ids
    assert all(estimator.warm_start for estimator in cache.values())
    assert all(pytest.approx(1.0) == estimator.C for estimator in cache.values())


def test_with_native_thread_limit_falls_back_when_controller_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []

    class _FallbackLimiter:
        def __init__(self, *, limits: int) -> None:
            self._limits = int(limits)
            events.append(("enter", self._limits))

        def __enter__(self) -> _FallbackLimiter:
            return self

        def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
            events.append(("exit", self._limits))
            return None

    monkeypatch.setattr(cv_mod, "ThreadpoolController", None)
    monkeypatch.setattr(cv_mod, "_THREADPOOL_CONTROLLER", None)
    monkeypatch.setattr(cv_mod, "threadpool_limits", _FallbackLimiter)

    result = cv_mod._with_native_thread_limit(4, lambda left, right: left + right, 2, 3)

    assert result == 5
    assert events == [("enter", 4), ("exit", 4)]


def test_fit_final_refit_sample_set_parallel_models_use_per_model_native_thread_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model:
  name: logistic_elasticnet
runtime:
  n_jobs: 2
""".strip(),
            )
        ]
    )
    source_result = cv_mod.SourceSelectionResult(
        selected_candidates=[
            cv_mod.SelectedCandidate(
                candidate=Candidate(candidate_index=0, params={"C": 1.0}),
                score=None,
            ),
            cv_mod.SelectedCandidate(
                candidate=Candidate(candidate_index=1, params={"C": 0.5}),
                score=None,
            ),
        ],
        n_available_candidates=2,
        n_scored_candidates=0,
        selected_candidate_count_requested=None,
        selected_candidate_count_effective=2,
        trial_rows=[],
    )
    captured_native_limits: list[int] = []

    def _fake_with_native_thread_limit_for_config(
        local_config: object, func: object, *args: object, **kwargs: object
    ) -> tuple[int, object, np.ndarray]:
        if not hasattr(local_config, "runtime"):
            raise AssertionError("config argument must have runtime")
        captured_native_limits.append(int(local_config.runtime.n_jobs))
        if not callable(func):
            raise AssertionError("func must be callable")
        result = func(*args, **kwargs)
        if not isinstance(result, tuple):
            raise AssertionError("expected tuple result")
        return result

    monkeypatch.setattr(
        cv_mod,
        "_with_native_thread_limit_for_config",
        _fake_with_native_thread_limit_for_config,
    )

    fit_result = cv_mod._fit_final_refit_sample_set(
        config=config,
        sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        source_result=source_result,
        base_model_index=0,
        x_train=np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float),
        y_train=np.array([0, 1, 0, 1], dtype=int),
        groups_train=np.array(["g1", "g1", "g2", "g2"], dtype=str),
        x_target=np.empty((0, 1), dtype=float),
        target_count=0,
    )

    assert fit_result.model_count == 2
    assert len(fit_result.fitted_models) == 2
    assert len(captured_native_limits) == 2
    assert set(captured_native_limits) == {1}


def test_run_final_refit_supports_no_external_or_inference_species(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "sp1\t1\tg1\tno",
                "sp2\t0\tg1\tno",
                "sp3\t1\tg2\tno",
                "sp4\t0\tg2\tno",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    refit = run_final_refit(config, split_artifacts.split_manifest)

    assert refit.pred_external_test.height == 0
    assert refit.pred_inference.height == 0
    assert set(refit.loss_by_split_final_refit.select("split").to_series().to_list()) == {"train"}


def test_run_final_refit_rejects_empty_training_pool(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_manifest = pl.DataFrame(
        {
            "species": ["sp5"],
            "label": [1],
            "group_id": [None],
            "pool": ["external_test"],
            "fold_id": ["NA"],
        }
    )

    with pytest.raises(CVError, match="No species available for final refit training pool"):
        run_final_refit(config, split_manifest)


def test_run_outer_cv_wraps_interpretation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    monkeypatch.setattr(
        cv_mod,
        "build_interpretation_tables",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(InterpretationError("forced interp fail")),
    )

    with pytest.raises(CVError, match="forced interp fail"):
        run_outer_cv(config, split_artifacts.split_manifest)


def test_prepare_source_selection_tpe_requires_selection_setting(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_missing_selected = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "trial_count": 1,
                    "selected_candidate_count": None,
                }
            )
        }
    )

    with pytest.raises(
        CVError,
        match="selected_candidate_count/selected_candidate_percent is required",
    ):
        _prepare_source_selection_tpe(
            config=config_missing_selected,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            sampled_idx=np.array([0, 1], dtype=int),
            x_train_raw=np.array([[1.0], [2.0]], dtype=float),
            y_train=np.array([0, 1], dtype=int),
            groups_train=np.array(["g1", "g2"], dtype=str),
            feature_names=["OG1"],
            warnings=[],
        )


def test_prepare_source_selection_tpe_requires_trial_count(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_missing_trial_count = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "selected_candidate_count": 1,
                    "trial_count": None,
                }
            )
        }
    )

    with pytest.raises(CVError, match="trial_count is required for TPE strategy"):
        _prepare_source_selection_tpe(
            config=config_missing_trial_count,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            sampled_idx=np.array([0, 1], dtype=int),
            x_train_raw=np.array([[1.0], [2.0]], dtype=float),
            y_train=np.array([0, 1], dtype=int),
            groups_train=np.array(["g1", "g2"], dtype=str),
            feature_names=["OG1"],
            warnings=[],
        )


def test_prepare_source_selection_tpe_rejects_single_class_source_set(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"search_strategy": "tpe", "selected_candidate_count": 1, "trial_count": 1}
            )
        }
    )

    with pytest.raises(CVError, match="Selection source sampled set became single-class"):
        _prepare_source_selection_tpe(
            config=config_tpe,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            sampled_idx=np.array([0, 1], dtype=int),
            x_train_raw=np.array([[1.0], [2.0]], dtype=float),
            y_train=np.array([1, 1], dtype=int),
            groups_train=np.array(["g1", "g2"], dtype=str),
            feature_names=["OG1"],
            warnings=[],
        )


def test_prepare_source_selection_tpe_emits_capping_warnings_for_trials_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeTrial:
        def __init__(self, number: int) -> None:
            self.number = number
            self.user_attrs: dict[str, object] = {}
            self.value: float | None = None

        def suggest_categorical(self, _name: str, values: list[object]) -> object:
            return values[0]

        def suggest_float(self, _name: str, low: float, _high: float) -> float:
            return low

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials: list[_FakeTrial] = []

        def optimize(self, objective: object, n_trials: int) -> None:
            for number in range(n_trials):
                trial = _FakeTrial(number)
                score = objective(trial)  # type: ignore[misc]
                trial.value = float(score)
                self.trials.append(trial)

    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "trial_count": 5,
                    "search_space": {"C": [0.1]},
                    "selected_candidate_count": 3,
                    "inner_cv_strategy": "logo",
                }
            )
        }
    )
    monkeypatch.setattr(cv_mod.optuna, "create_study", lambda **_kwargs: _FakeStudy())
    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", lambda **_kwargs: (0.5, []))
    warnings: list[str] = []

    result = _prepare_source_selection_tpe(
        config=config_tpe,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=warnings,
    )

    assert result.n_available_candidates == 1
    assert result.selected_candidate_count_effective == 1
    assert any("trial_count exceeded discrete candidate space" in item for item in warnings)
    assert any(
        "selected_candidate_count exceeded available candidates" in item for item in warnings
    )


def test_prepare_source_selection_tpe_deduplicates_selected_candidates_by_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeTrial:
        def __init__(self, number: int) -> None:
            self.number = number
            self.user_attrs: dict[str, object] = {}
            self.value: float | None = None

        def suggest_categorical(self, _name: str, values: list[object]) -> object:
            return values[0]

        def suggest_float(self, _name: str, low: float, _high: float) -> float:
            return low

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials: list[_FakeTrial] = []

        def optimize(self, objective: object, n_trials: int) -> None:
            for number in range(n_trials):
                trial = _FakeTrial(number)
                trial.value = float(objective(trial))  # type: ignore[misc]
                self.trials.append(trial)

    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "trial_count": 2,
                    "search_space": {"C": [0.1, 1.0]},
                    "selected_candidate_count": 2,
                    "inner_cv_strategy": "logo",
                }
            )
        }
    )
    monkeypatch.setattr(cv_mod.optuna, "create_study", lambda **_kwargs: _FakeStudy())

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        if candidate.candidate_index == 0:
            return 0.50, []
        return 0.30, []

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)
    warnings: list[str] = []

    result = _prepare_source_selection_tpe(
        config=config_tpe,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=warnings,
    )

    assert result.n_scored_candidates == 2
    assert result.n_available_candidates == 1
    assert result.selected_candidate_count_effective == 1
    assert len(result.selected_candidates) == 1
    assert result.selected_candidates[0].candidate.candidate_index == 1
    assert any("deduplicated candidates with identical params" in item for item in warnings)
    assert any(
        "selected_candidate_count exceeded available candidates" in item for item in warnings
    )


def test_prepare_source_selection_tpe_supports_selected_candidate_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeTrial:
        def __init__(self, number: int) -> None:
            self.number = number
            self.user_attrs: dict[str, object] = {}
            self.value: float | None = None

        def suggest_categorical(self, _name: str, values: list[object]) -> object:
            return values[min(self.number, len(values) - 1)]

        def suggest_float(self, _name: str, low: float, _high: float) -> float:
            return low

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials: list[_FakeTrial] = []

        def optimize(self, objective: object, n_trials: int) -> None:
            for number in range(n_trials):
                trial = _FakeTrial(number)
                trial.value = float(objective(trial))  # type: ignore[misc]
                self.trials.append(trial)

    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "trial_count": 3,
                    "search_space": {"C": [0.1, 1.0, 10.0]},
                    "selected_candidate_count": None,
                    "selected_candidate_percent": 50,
                    "inner_cv_strategy": "logo",
                }
            )
        }
    )
    monkeypatch.setattr(cv_mod.optuna, "create_study", lambda **_kwargs: _FakeStudy())

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        return float(candidate.candidate_index), []

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)

    result = _prepare_source_selection_tpe(
        config=config_tpe,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=[],
    )

    assert result.n_available_candidates == 3
    assert result.selected_candidate_count_requested == 2
    assert result.selected_candidate_count_effective == 2
    assert [item.candidate.candidate_index for item in result.selected_candidates] == [0, 1]


def test_prepare_source_selection_tpe_uses_minimize_direction_for_log_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeTrial:
        def __init__(self, number: int) -> None:
            self.number = number
            self.user_attrs: dict[str, object] = {}
            self.value: float | None = None

        def suggest_categorical(self, _name: str, values: list[object]) -> object:
            return values[min(self.number, len(values) - 1)]

        def suggest_float(self, _name: str, low: float, _high: float) -> float:
            return low

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials: list[_FakeTrial] = []

        def optimize(self, objective: object, n_trials: int) -> None:
            for number in range(n_trials):
                trial = _FakeTrial(number)
                trial.value = float(objective(trial))  # type: ignore[misc]
                self.trials.append(trial)

    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "selection_metric": "log_loss",
                    "trial_count": 2,
                    "search_space": {"C": [0.1, 1.0]},
                    "selected_candidate_count": 1,
                    "inner_cv_strategy": "logo",
                }
            )
        }
    )
    directions: list[str] = []

    def _fake_create_study(**kwargs: object) -> _FakeStudy:
        direction = kwargs.get("direction")
        if not isinstance(direction, str):
            raise AssertionError("direction must be provided")
        directions.append(direction)
        return _FakeStudy()

    monkeypatch.setattr(cv_mod.optuna, "create_study", _fake_create_study)

    def _fake_score_candidate_inner_cv(**kwargs: object) -> tuple[float, list[dict[str, object]]]:
        candidate = kwargs["candidate"]
        if not isinstance(candidate, Candidate):
            raise AssertionError("candidate must be a Candidate instance")
        if candidate.candidate_index == 0:
            return 0.2, []
        return 0.8, []

    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", _fake_score_candidate_inner_cv)

    result = _prepare_source_selection_tpe(
        config=config_tpe,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=[],
    )

    assert directions == ["minimize"]
    assert result.selected_candidates[0].candidate.candidate_index == 0


def test_prepare_source_selection_tpe_rejects_trials_missing_candidate_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TrialWithoutParams:
        def __init__(self) -> None:
            self.number = 0
            self.user_attrs: dict[str, object] = {}
            self.value = 0.5

    class _StudyWithoutParams:
        def __init__(self) -> None:
            self.trials = [_TrialWithoutParams()]

        def optimize(self, _objective: object, n_trials: int) -> None:
            _ = n_trials

    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "trial_count": 1,
                    "selected_candidate_count": 1,
                    "inner_cv_strategy": "logo",
                }
            )
        }
    )
    monkeypatch.setattr(cv_mod.optuna, "create_study", lambda **_kwargs: _StudyWithoutParams())

    with pytest.raises(CVError, match="missing candidate_params user attribute"):
        _prepare_source_selection_tpe(
            config=config_tpe,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            sampled_idx=np.array([0, 1, 2, 3], dtype=int),
            x_train_raw=x_train_raw,
            y_train=y_train,
            groups_train=groups_train,
            feature_names=["OG1"],
            warnings=[],
        )


def test_expression_matrix_builder_rejects_empty_matrix_for_selected_species(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            ["species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout", "sp1\t1\tg1\tno"]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(["species\torthogroup\ttpm", "sp1\t\t1.0"]) + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    builder = ExpressionMatrixBuilder(config)

    with pytest.raises(CVError, match="missing-feature"):
        builder.build_matrix(["sp1"])


def test_expression_matrix_builder_chunking_single_feature_returns_single_chunk(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tcontrast_pair_test_holdout",
                "sp1\t1\tg1\tno",
                "sp2\t0\tg1\tno",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  max_pivot_cells: 1
""".strip(),
            )
        ]
    )
    builder = ExpressionMatrixBuilder(config)

    matrix, features = builder.build_matrix(["sp1", "sp2"])

    assert matrix.shape == (2, 1)
    assert features == ["OG1"]


def test_select_feature_indices_rejects_missing_sparse_feature_threshold_when_mutated(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_bad = config.model_copy(
        update={
            "preprocess": config.preprocess.model_copy(
                update={
                    "sparse_feature_filter": config.preprocess.sparse_feature_filter.model_copy(
                        update={
                            "enabled": True,
                            "min_nonzero_fraction_in_at_least_one_trait": None,
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(
        CVError, match="min_nonzero_fraction_in_at_least_one_trait is missing"
    ):
        _select_feature_indices(
            config_bad,
            np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
            ["OG1", "OG2"],
            y_train=np.array([0, 1], dtype=int),
        )


def test_select_feature_indices_sparse_feature_filter_keeps_any_trait_signal(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    selected = _select_feature_indices(
        config,
        np.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.0, 3.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=float,
        ),
        ["OG_C3", "OG_RARE", "OG_C4"],
        y_train=np.array([0, 0, 0, 1, 1, 1], dtype=int),
    )

    assert selected.tolist() == [0, 2]


def test_select_feature_indices_rejects_missing_low_variance_threshold_when_mutated(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_bad = config.model_copy(
        update={
            "preprocess": config.preprocess.model_copy(
                update={
                    "sparse_feature_filter": config.preprocess.sparse_feature_filter.model_copy(
                        update={"enabled": False}
                    ),
                    "low_variance_filter": config.preprocess.low_variance_filter.model_copy(
                        update={"enabled": True, "min_variance": None}
                    )
                }
            )
        }
    )

    with pytest.raises(CVError, match="min_variance is missing"):
        _select_feature_indices(
            config_bad,
            np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
            ["OG1", "OG2"],
        )


def test_select_feature_indices_applies_low_variance_filter(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  sparse_feature_filter:
    enabled: false
  low_variance_filter:
    enabled: true
    min_variance: 0.01
""".strip(),
            )
        ]
    )
    selected = _select_feature_indices(
        config,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=float,
        ),
        ["OG1", "OG2"],
    )

    assert selected.tolist() == [1]


def test_select_feature_indices_pair_aware_filter_prefers_consistent_signal(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  sparse_feature_filter:
    enabled: false
  pair_aware_filter:
    enabled: true
    max_features: 1
""".strip(),
            )
        ]
    )
    warnings: list[str] = []

    selected = _select_feature_indices(
        config,
        np.array(
            [
                [0.0, 0.0],
                [2.0, 10.0],
                [0.0, 0.0],
                [2.2, 0.0],
                [0.0, 0.0],
                [1.8, 0.0],
                [0.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=float,
        ),
        ["OG1", "OG2"],
        y_train=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int),
        groups_train=np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"], dtype=str),
        warnings=warnings,
    )

    assert selected.tolist() == [0]
    assert warnings == []


def test_pair_aware_filter_breaks_score_ties_by_feature_name(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  sparse_feature_filter:
    enabled: false
  pair_aware_filter:
    enabled: true
    max_features: 2
""".strip(),
            )
        ]
    )

    selected = _select_feature_indices(
        config,
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
        ["OG_z", "OG_a", "OG_m"],
        y_train=np.array([0, 1], dtype=int),
        groups_train=np.array(["g1", "g1"], dtype=str),
        warnings=[],
    )

    assert selected.tolist() == [1, 2]


def test_select_feature_indices_pair_aware_filter_uses_available_valid_contrast_pairs(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  sparse_feature_filter:
    enabled: false
  pair_aware_filter:
    enabled: true
    max_features: 1
    min_contrast_pairs: 2
""".strip(),
            )
        ]
    )
    warnings: list[str] = []

    selected = _select_feature_indices(
        config,
        np.array(
            [
                [0.0, 0.0],
                [2.0, 1.0],
                [0.0, 100.0],
                [5.0, 5.0],
                [0.0, 0.0],
                [2.2, -1.0],
            ],
            dtype=float,
        ),
        ["OG1", "OG2"],
        y_train=np.array([0, 1, 0, 1, 0, 1], dtype=int),
        groups_train=np.array(["g1", "g1", None, "g2", "g3", "g3"], dtype=object),
        warnings=warnings,
    )

    assert selected.tolist() == [0]
    assert warnings == []


def test_select_feature_indices_pair_aware_filter_uses_single_valid_pair_by_default(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  sparse_feature_filter:
    enabled: false
  pair_aware_filter:
    enabled: true
    max_features: 1
""".strip(),
            )
        ]
    )
    warnings: list[str] = []

    selected = _select_feature_indices(
        config,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 5.0],
            ],
            dtype=float,
        ),
        ["OG1", "OG2"],
        y_train=np.array([0, 1], dtype=int),
        groups_train=np.array(["g1", "g1"], dtype=object),
        warnings=warnings,
    )

    assert selected.tolist() == [1]
    assert any("one valid contrast pair" in item for item in warnings)


def test_select_feature_indices_pair_aware_filter_skips_when_too_few_groups(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  sparse_feature_filter:
    enabled: false
  pair_aware_filter:
    enabled: true
    max_features: 1
    min_contrast_pairs: 2
""".strip(),
            )
        ]
    )
    warnings: list[str] = []

    selected = _select_feature_indices(
        config,
        np.array(
            [
                [1.0, 0.0],
                [2.0, 3.0],
            ],
            dtype=float,
        ),
        ["OG1", "OG2"],
        y_train=np.array([0, 1], dtype=int),
        groups_train=np.array(["g1", "g1"], dtype=str),
        warnings=warnings,
    )

    assert selected.tolist() == [0, 1]
    assert any("too few valid contrast pairs" in item for item in warnings)


def test_select_feature_indices_calls_correlation_filter_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  sparse_feature_filter:
    enabled: false
  correlation_filter:
    enabled: true
    max_abs_correlation: 0.9
""".strip(),
            )
        ]
    )
    called = {"value": False}

    def _fake_apply(*_args: object, **_kwargs: object) -> np.ndarray:
        called["value"] = True
        return np.array([1], dtype=int)

    monkeypatch.setattr(cv_mod, "_apply_correlation_filter", _fake_apply)
    selected = _select_feature_indices(
        config,
        np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
        ["OG1", "OG2"],
    )

    assert called["value"] is True
    assert selected.tolist() == [1]


def test_apply_correlation_filter_prefers_pair_aware_priority_when_provided(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
preprocess:
  correlation_filter:
    enabled: true
    max_abs_correlation: 0.9
""".strip(),
            )
        ]
    )

    kept = _apply_correlation_filter(
        config,
        x_train_log=np.array(
            [
                [0.0, 0.0],
                [2.0, 1.0],
                [4.0, 2.0],
                [6.0, 3.0],
            ],
            dtype=float,
        ),
        selected=np.array([0, 1], dtype=int),
        feature_names=["OG1", "OG2"],
        priority_scores=np.array([0.1, 0.9], dtype=float),
    )

    assert kept.tolist() == [1]


def test_apply_correlation_filter_rejects_missing_threshold_when_mutated(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_bad = config.model_copy(
        update={
            "preprocess": config.preprocess.model_copy(
                update={
                    "correlation_filter": config.preprocess.correlation_filter.model_copy(
                        update={"enabled": True, "max_abs_correlation": None}
                    )
                }
            )
        }
    )

    with pytest.raises(CVError, match="max_abs_correlation is missing"):
        _apply_correlation_filter(
            config_bad,
            x_train_log=np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
            selected=np.array([0, 1], dtype=int),
            feature_names=["OG1", "OG2"],
        )


def test_sample_training_sets_rejects_group_without_both_labels(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
sampling:
  strategy: group_balanced""".strip(),
            )
        ]
    )

    with pytest.raises(CVError, match="requires both labels per group"):
        cv_mod._sample_training_sets(
            config=config,
            y_train=np.array([1, 1, 0, 0], dtype=int),
            groups_train=np.array(["g1", "g1", "g2", "g2"], dtype=str),
            training_scope_id="fold_0",
            warnings=[],
        )


def test_sample_training_sets_rejects_k_zero_when_mutated(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
sampling:
  strategy: group_balanced
  max_samples_per_label_per_group: 1
""".strip(),
            )
        ]
    )
    config_bad = config.model_copy(
        update={
            "sampling": config.sampling.model_copy(
                update={"strategy": "group_balanced", "max_samples_per_label_per_group": 0}
            )
        }
    )

    with pytest.raises(CVError, match="produced k=0"):
        cv_mod._sample_training_sets(
            config=config_bad,
            y_train=np.array([0, 1, 0, 1], dtype=int),
            groups_train=np.array(["g1", "g1", "g2", "g2"], dtype=str),
            training_scope_id="fold_0",
            warnings=[],
        )


def test_sample_training_sets_rejects_when_unique_generation_does_not_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FixedRng:
        def choice(self, values: np.ndarray, size: int, replace: bool) -> np.ndarray:
            _ = replace
            return np.array([int(values[0])] * size, dtype=int)

    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
sampling:
  strategy: group_balanced
  max_samples_per_label_per_group: 1
  sampled_set_count: 2
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(cv_mod.np.random, "default_rng", lambda *_args, **_kwargs: _FixedRng())

    with pytest.raises(CVError, match="Failed to generate deterministic unique sampled sets"):
        cv_mod._sample_training_sets(
            config=config,
            y_train=np.array([0, 0, 1, 1], dtype=int),
            groups_train=np.array(["g1", "g1", "g1", "g1"], dtype=str),
            training_scope_id="fold_0",
            warnings=[],
        )


def test_group_label_inverse_weights_rejects_non_positive_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv_mod.np, "mean", lambda _values: 0.0)

    with pytest.raises(CVError, match="non-positive mean"):
        _group_label_inverse_weights(
            np.array([0, 1, 0, 1], dtype=int),
            np.array(["g1", "g1", "g2", "g2"], dtype=str),
        )


def test_fit_sample_weights_returns_group_label_inverse_weights(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
sampling:
  weighting: group_label_inverse
""".strip(),
            )
        ]
    )

    weights = cv_mod._fit_sample_weights(
        config,
        np.array([0, 1, 0, 1], dtype=int),
        np.array(["g1", "g1", "g2", "g2"], dtype=str),
    )

    assert weights is not None
    assert float(np.mean(weights)) == pytest.approx(1.0)


def test_aggregate_probabilities_supports_median() -> None:
    result = _aggregate_probabilities(
        [
            np.array([0.1, 0.8, 0.7], dtype=float),
            np.array([0.2, 0.5, 0.9], dtype=float),
            np.array([0.0, 0.6, 0.8], dtype=float),
        ],
        "median",
    )
    assert result.tolist() == pytest.approx([0.1, 0.6, 0.8])


def test_selection_metric_from_probability_supports_balanced_accuracy(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selection_metric: balanced_accuracy
""".strip(),
            )
        ]
    )
    score = cv_mod._selection_metric_from_probability(
        config,
        y_true=np.array([0, 1, 1, 0], dtype=int),
        prob=np.array([0.1, 0.9, 0.3, 0.2], dtype=float),
    )
    assert score == pytest.approx(0.75)


def test_selection_metric_from_probability_supports_log_loss(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selection_metric: log_loss
""".strip(),
            )
        ]
    )
    y_true = np.array([0, 1, 1, 0], dtype=int)
    prob = np.array([0.1, 0.9, 0.3, 0.2], dtype=float)
    score = cv_mod._selection_metric_from_probability(config, y_true=y_true, prob=prob)
    expected = float(log_loss(y_true, prob, labels=[0, 1]))
    assert score == pytest.approx(expected)


def test_inner_cv_splits_supports_group_kfold(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_group_kfold = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"inner_cv_strategy": "group_kfold", "inner_cv_n_splits": 2}
            )
        }
    )

    rows = _inner_cv_splits(
        config_group_kfold,
        np.array([0, 1, 0, 1], dtype=int),
        np.array(["g1", "g1", "g2", "g2"], dtype=str),
    )

    assert len(rows) == 2


def test_inner_cv_splits_supports_stratified_group_kfold(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    stratified_config = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "inner_cv_strategy": "stratified_group_kfold",
                    "inner_cv_n_splits": 2,
                }
            )
        }
    )
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"], dtype=str)

    rows = _inner_cv_splits(stratified_config, y, groups)

    assert len(rows) == 2
    for train_idx, valid_idx, _fold_id in rows:
        assert set(groups[train_idx]).isdisjoint(set(groups[valid_idx]))
        assert set(y[valid_idx]) == {0, 1}


def test_inner_cv_splits_rejects_empty_train_or_validation_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _EmptySplitGroupKFold:
        def __init__(self, n_splits: int) -> None:
            _ = n_splits

        def split(self, *_args: object, **_kwargs: object) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(np.array([], dtype=int), np.array([0], dtype=int))]

    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_group_kfold = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"inner_cv_strategy": "group_kfold", "inner_cv_n_splits": 2}
            )
        }
    )
    monkeypatch.setattr(cv_mod, "GroupKFold", _EmptySplitGroupKFold)

    with pytest.raises(CVError, match="empty train/validation split"):
        _inner_cv_splits(
            config_group_kfold,
            np.array([0, 1], dtype=int),
            np.array(["g1", "g2"], dtype=str),
        )


def test_inner_cv_splits_rejects_zero_fold_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _NoFoldLogo:
        def split(self, *_args: object, **_kwargs: object) -> list[tuple[np.ndarray, np.ndarray]]:
            return []

    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_logo = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"inner_cv_strategy": "logo"}
            )
        }
    )
    monkeypatch.setattr(cv_mod, "LeaveOneGroupOut", lambda: _NoFoldLogo())

    with pytest.raises(CVError, match="Inner CV produced zero folds"):
        _inner_cv_splits(
            config_logo,
            np.array([0, 1], dtype=int),
            np.array(["g1", "g2"], dtype=str),
        )


def test_score_candidate_inner_cv_returns_nan_when_no_inner_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    score, rows = cv_mod._score_candidate_inner_cv(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        candidate=Candidate(candidate_index=0, params={}),
        preprocessed_folds=[],
    )

    assert np.isnan(score)
    assert rows == []


def test_prepare_source_selection_tpe_maps_nan_objective_score_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeTrial:
        def __init__(self, number: int) -> None:
            self.number = number
            self.user_attrs: dict[str, object] = {}
            self.value: float | None = None

        def suggest_categorical(self, _name: str, values: list[object]) -> object:
            return values[0]

        def suggest_float(self, _name: str, low: float, _high: float) -> float:
            return low

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials: list[_FakeTrial] = []

        def optimize(self, objective: object, n_trials: int) -> None:
            for number in range(n_trials):
                trial = _FakeTrial(number)
                trial.value = float(objective(trial))  # type: ignore[misc]
                self.trials.append(trial)

    metadata, tpm = _write_fixture(tmp_path)
    x_train_raw, y_train, groups_train = _selection_source_arrays()
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "trial_count": 1,
                    "search_space": {"C": [0.1]},
                    "selected_candidate_count": 1,
                    "inner_cv_strategy": "logo",
                }
            )
        }
    )
    monkeypatch.setattr(cv_mod.optuna, "create_study", lambda **_kwargs: _FakeStudy())
    monkeypatch.setattr(cv_mod, "_score_candidate_inner_cv", lambda **_kwargs: (np.nan, []))

    result = _prepare_source_selection_tpe(
        config=config_tpe,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        sampled_idx=np.array([0, 1, 2, 3], dtype=int),
        x_train_raw=x_train_raw,
        y_train=y_train,
        groups_train=groups_train,
        feature_names=["OG1"],
        warnings=[],
    )

    assert len(result.selected_candidates) == 1
    assert result.selected_candidates[0].score is None


def test_prepare_source_selection_non_tpe_rejects_single_class_source_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
""".strip(),
            )
        ]
    )
    monkeypatch.setattr(
        cv_mod,
        "generate_candidates",
        lambda **_kwargs: [Candidate(candidate_index=0, params={"C": 1.0})],
    )

    with pytest.raises(CVError, match="Selection source sampled set became single-class"):
        _prepare_source_selection(
            config=config,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            sampled_idx=np.array([0, 1], dtype=int),
            x_train_raw=np.array([[1.0], [2.0]], dtype=float),
            y_train=np.array([1, 1], dtype=int),
            groups_train=np.array(["g1", "g2"], dtype=str),
            feature_names=["OG1"],
            warnings=[],
        )


def test_build_prediction_table_includes_uncertainty_when_provided() -> None:
    table = _build_prediction_table(
        species=["sp1", "sp2"],
        prob=np.array([0.2, 0.8], dtype=float),
        fixed_threshold=0.5,
        uncertainty_std=np.array([0.01, 0.02], dtype=float),
    )

    assert "uncertainty_std" in table.columns


def test_run_final_refit_rejects_null_label_or_group_values(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_manifest = pl.DataFrame(
        {
            "species": ["sp1", "sp2"],
            "label": [1, 0],
            "group_id": ["g1", None],
            "pool": ["train", "validation"],
            "fold_id": ["0", "0"],
        }
    )

    with pytest.raises(CVError, match="contains null label/group values"):
        run_final_refit(config, split_manifest)


def _one_selected_candidate_result() -> cv_mod.SourceSelectionResult:
    return cv_mod.SourceSelectionResult(
        selected_candidates=[
            cv_mod.SelectedCandidate(candidate=Candidate(candidate_index=0, params={}), score=None)
        ],
        n_available_candidates=1,
        n_scored_candidates=1,
        selected_candidate_count_requested=1,
        selected_candidate_count_effective=1,
        trial_rows=[],
    )


def test_run_final_refit_rejects_when_no_candidates_are_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    monkeypatch.setattr(
        cv_mod,
        "_prepare_source_selection",
        lambda **_kwargs: cv_mod.SourceSelectionResult(
            selected_candidates=[],
            n_available_candidates=0,
            n_scored_candidates=0,
            selected_candidate_count_requested=1,
            selected_candidate_count_effective=0,
            trial_rows=[],
        ),
    )

    with pytest.raises(CVError, match="No candidates were selected for final_refit"):
        run_final_refit(config, split_artifacts.split_manifest)


def test_run_final_refit_rejects_single_class_sampled_training_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    monkeypatch.setattr(
        cv_mod,
        "_sample_training_sets",
        lambda **_kwargs: [np.array([0], dtype=int)],
    )
    monkeypatch.setattr(
        cv_mod,
        "_prepare_source_selection",
        lambda **_kwargs: _one_selected_candidate_result(),
    )

    with pytest.raises(CVError, match="Final refit sampled training set became single-class"):
        run_final_refit(config, split_artifacts.split_manifest)


def test_run_outer_cv_rejects_fold_with_empty_train_or_validation_split(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_manifest = pl.DataFrame(
        {
            "species": ["sp1"],
            "label": [1],
            "group_id": ["g1"],
            "pool": ["validation"],
            "fold_id": ["0"],
        }
    )

    with pytest.raises(CVError, match="empty train/validation split"):
        run_outer_cv(config, split_manifest)


def test_run_outer_cv_rejects_when_no_candidates_selected_for_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    monkeypatch.setattr(
        cv_mod,
        "_prepare_source_selection",
        lambda **_kwargs: cv_mod.SourceSelectionResult(
            selected_candidates=[],
            n_available_candidates=0,
            n_scored_candidates=0,
            selected_candidate_count_requested=1,
            selected_candidate_count_effective=0,
            trial_rows=[],
        ),
    )

    with pytest.raises(CVError, match="No candidates were selected for fold"):
        run_outer_cv(config, split_artifacts.split_manifest)


def test_run_outer_cv_rejects_single_class_sampled_training_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    monkeypatch.setattr(
        cv_mod,
        "_sample_training_sets",
        lambda **_kwargs: [np.array([0], dtype=int)],
    )
    monkeypatch.setattr(
        cv_mod,
        "_prepare_source_selection",
        lambda **_kwargs: _one_selected_candidate_result(),
    )

    with pytest.raises(CVError, match="sampled training set became single-class"):
        run_outer_cv(config, split_artifacts.split_manifest)


def test_run_outer_cv_rejects_when_no_oof_predictions_are_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    monkeypatch.setattr(cv_mod, "_fold_ids", lambda _manifest: [])

    with pytest.raises(CVError, match="No out-of-fold predictions were generated"):
        run_outer_cv(config, split_artifacts.split_manifest)


def test_run_outer_cv_sets_macro_metric_to_nan_when_all_fold_values_are_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    def _all_nan_metrics(
        _y_true: np.ndarray, _prob: np.ndarray, _threshold: float
    ) -> dict[str, float]:
        return {
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "balanced_accuracy": np.nan,
            "mcc": np.nan,
            "brier": np.nan,
        }

    monkeypatch.setattr(cv_mod, "_compute_fold_metrics", _all_nan_metrics)
    artifacts = run_outer_cv(config, split_artifacts.split_manifest)
    macro_rows = artifacts.metrics_cv.filter(pl.col("aggregate_scope") == "macro")

    assert macro_rows.height > 0
    assert macro_rows.filter(pl.col("metric_value").is_nan()).height == macro_rows.height
