from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier

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
    _derive_cv_threshold,
    _fit_estimator,
    _group_label_inverse_weights,
    _inner_cv_splits,
    _predict_positive_probability,
    _prepare_source_selection,
    _prepare_source_selection_tpe,
    _preprocess_fold,
    _preprocess_train_and_target,
    _select_feature_indices,
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
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t0\tg1",
                "sp3\t1\tg2",
                "sp4\t0\tg2",
                "sp5\t1\t",
                "sp6\t\t",
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


def test_run_outer_cv_generates_metrics_and_thresholds(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.oof_predictions.height == 4
    scopes = set(cv_artifacts.metrics_cv.select("aggregate_scope").to_series().to_list())
    assert {"NA", "macro", "micro"}.issubset(scopes)
    threshold_names = set(cv_artifacts.thresholds.select("threshold_name").to_series().to_list())
    assert threshold_names == {"fixed_probability_threshold", "cv_derived_threshold"}
    assert {
        "feature",
        "importance_mean",
        "importance_std",
        "n_models",
        "method",
    }.issubset(cv_artifacts.feature_importance.columns)
    assert {
        "feature",
        "coef_mean",
        "coef_std",
        "n_models",
        "method",
        "reason",
    }.issubset(cv_artifacts.coefficients.columns)
    aggregate_rows = cv_artifacts.metrics_cv.filter(pl.col("fold_id") == "NA")
    assert aggregate_rows.height > 0
    assert aggregate_rows.filter(pl.col("n_valid_folds").is_null()).height == 0
    assert "uncertainty_std" not in cv_artifacts.oof_predictions.columns
    assert cv_artifacts.ensemble_model_probs is None
    assert cv_artifacts.model_selection_selected is None
    assert cv_artifacts.model_selection_trials is None
    assert cv_artifacts.model_selection_trials_summary is None


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


def test_run_outer_cv_builds_expression_matrix_once_across_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)

    build_calls = 0
    original_build_matrix = cv_mod.ExpressionMatrixBuilder.build_matrix

    def _counting_build_matrix(
        self: object, species_order: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        nonlocal build_calls
        build_calls += 1
        return original_build_matrix(self, species_order)

    monkeypatch.setattr(cv_mod.ExpressionMatrixBuilder, "build_matrix", _counting_build_matrix)

    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert cv_artifacts.oof_predictions.height == 4
    assert build_calls == 1


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
                "species\tC4\tcontrast_pair_id",
                "g1_pos1\t1\tg1",
                "g1_pos2\t1\tg1",
                "g1_neg1\t0\tg1",
                "g1_neg2\t0\tg1",
                "g2_pos1\t1\tg2",
                "g2_pos2\t1\tg2",
                "g2_neg1\t0\tg2",
                "g2_neg2\t0\tg2",
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
                "species\tC4\tcontrast_pair_id",
                "g1_pos\t1\tg1",
                "g1_neg\t0\tg1",
                "g2_pos\t1\tg2",
                "g2_neg\t0\tg2",
                "g3_pos\t1\tg3",
                "g3_neg\t0\tg3",
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
    assert {
        "selection_scope",
        "fold_id",
        "sample_set_id",
        "selection_source_sample_set_id",
        "rank",
        "candidate_index",
        "metric_name",
        "metric_value",
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
    }.issubset(cv_artifacts.model_selection_trials_summary.columns)


def test_outer_cv_tpe_selection_active_bypasses_generic_candidate_generation(
    tmp_path: Path, monkeypatch
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "g1_pos\t1\tg1",
                "g1_neg\t0\tg1",
                "g2_pos\t1\tg2",
                "g2_neg\t0\tg2",
                "g3_pos\t1\tg3",
                "g3_neg\t0\tg3",
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
      stop_exp: 1
    l1_ratio:
      type: continuous_range
      start: 0.0
      stop: 1.0
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


def test_run_final_refit_generates_external_and_inference_predictions(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)
    cv_threshold = (
        cv_artifacts.thresholds.filter(pl.col("threshold_name") == "cv_derived_threshold")
        .select("threshold_value")
        .to_series()
        .item()
    )

    refit_artifacts = run_final_refit(config, split_artifacts.split_manifest, float(cv_threshold))

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
        "pred_label_cv_derived_threshold",
    }.issubset(
        refit_artifacts.pred_external_test.columns
    )
    assert {
        "species",
        "prob",
        "pred_label_fixed_threshold",
        "pred_label_cv_derived_threshold",
        "true_label",
    }.issubset(
        refit_artifacts.pred_inference.columns
    )
    assert refit_artifacts.pred_inference.get_column("true_label").null_count() == 1
    assert refit_artifacts.model_selection_selected is None


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


def test_outer_cv_reuse_first_sample_set_uses_source_zero_for_all_sample_sets(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "g1_pos1\t1\tg1",
                "g1_pos2\t1\tg1",
                "g1_neg1\t0\tg1",
                "g1_neg2\t0\tg1",
                "g2_pos1\t1\tg2",
                "g2_pos2\t1\tg2",
                "g2_neg1\t0\tg2",
                "g2_neg2\t0\tg2",
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


def test_derive_cv_threshold_prefers_smallest_threshold_on_tie(
    tmp_path: Path, monkeypatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    y_true = np.array([0, 1], dtype=int)
    prob = np.array([0.25, 0.75], dtype=float)

    monkeypatch.setattr("phenoradar.cv.matthews_corrcoef", lambda _y, _p: 0.5)

    threshold, warning = _derive_cv_threshold(config, y_true, prob)

    assert threshold == pytest.approx(0.0)
    assert warning is None


def test_derive_cv_threshold_falls_back_when_all_scores_nan(
    tmp_path: Path, monkeypatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    y_true = np.array([0, 1], dtype=int)
    prob = np.array([0.25, 0.75], dtype=float)

    monkeypatch.setattr("phenoradar.cv.matthews_corrcoef", lambda _y, _p: float("nan"))

    threshold, warning = _derive_cv_threshold(config, y_true, prob)

    assert threshold == pytest.approx(0.5)
    assert warning is not None
    assert "fallback to 0.5" in warning


def test_derive_cv_threshold_rejects_empty_probabilities(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])

    with pytest.raises(CVError, match="empty prediction set"):
        _derive_cv_threshold(config, np.array([], dtype=int), np.array([], dtype=float))


def test_compute_fold_metrics_returns_nan_when_metrics_are_undefined() -> None:
    y_true = np.array([], dtype=int)
    prob = np.array([], dtype=float)

    metrics = _compute_fold_metrics(y_true, prob, threshold=0.5)

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["mcc"])
    assert np.isnan(metrics["brier"])


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
  low_prevalence_filter:
    enabled: true
    min_species_per_feature: 10
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
        _select_feature_indices(config, x_train_log, ["OG1", "OG2"])


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


def test_expression_matrix_builder_rejects_missing_required_columns(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t0\tg1",
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


def test_build_prediction_table_rejects_probability_length_mismatch() -> None:
    with pytest.raises(CVError, match="species/probability length mismatch"):
        _build_prediction_table(
            species=["sp1", "sp2"],
            prob=np.array([0.5], dtype=float),
            fixed_threshold=0.5,
            cv_threshold=0.4,
            uncertainty_std=None,
        )


def test_build_prediction_table_rejects_uncertainty_length_mismatch() -> None:
    with pytest.raises(CVError, match="species/uncertainty length mismatch"):
        _build_prediction_table(
            species=["sp1", "sp2"],
            prob=np.array([0.2, 0.8], dtype=float),
            fixed_threshold=0.5,
            cv_threshold=0.4,
            uncertainty_std=np.array([0.1], dtype=float),
        )


def test_build_prediction_table_rejects_true_label_length_mismatch() -> None:
    with pytest.raises(CVError, match="species/true_label length mismatch"):
        _build_prediction_table(
            species=["sp1", "sp2"],
            prob=np.array([0.2, 0.8], dtype=float),
            fixed_threshold=0.5,
            cv_threshold=0.4,
            uncertainty_std=None,
            true_label=np.array([1], dtype=int),
        )


def test_build_prediction_table_can_include_empty_true_label_column() -> None:
    table = _build_prediction_table(
        species=["sp1", "sp2"],
        prob=np.array([0.2, 0.8], dtype=float),
        fixed_threshold=0.5,
        cv_threshold=0.4,
        uncertainty_std=None,
        include_true_label_column=True,
    )

    assert "true_label" in table.columns
    assert table.get_column("true_label").null_count() == 2


def test_fit_estimator_falls_back_when_sample_weight_not_supported() -> None:
    class _NoWeightEstimator:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fit(self, *args: object, **kwargs: object) -> None:
            self.calls.append((args, kwargs))
            if "sample_weight" in kwargs:
                raise TypeError("sample_weight is unsupported")

    estimator = _NoWeightEstimator()
    x = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([0, 1], dtype=int)
    sample_weight = np.array([1.0, 1.0], dtype=float)

    _fit_estimator(estimator, x, y, sample_weight)

    assert len(estimator.calls) == 2
    assert "sample_weight" in estimator.calls[0][1]
    assert estimator.calls[1][1] == {}


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


def test_prepare_source_selection_warns_when_selected_candidate_count_is_capped(
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
        sampled_idx=np.array([0, 1], dtype=int),
        x_train_raw=np.array([[1.0], [2.0]], dtype=float),
        y_train=np.array([0, 1], dtype=int),
        groups_train=np.array(["g1", "g2"], dtype=str),
        feature_names=["OG1"],
        warnings=warnings,
    )

    assert result.selected_candidate_count_effective == 1
    assert any(
        "selected_candidate_count exceeded available candidates" in item for item in warnings
    )


def test_prepare_source_selection_parallel_scoring_caps_rf_n_jobs(
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
        sampled_idx=np.array([0, 1], dtype=int),
        x_train_raw=np.array([[1.0], [2.0]], dtype=float),
        y_train=np.array([0, 1], dtype=int),
        groups_train=np.array(["g1", "g2"], dtype=str),
        feature_names=["OG1"],
        warnings=[],
    )

    assert captured_n_jobs == [2, 2]
    assert result.selected_candidates[0].candidate.candidate_index == 1
    assert result.selected_candidate_count_effective == 1


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

    score, trial_rows = cv_mod._score_candidate_inner_cv(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        candidate=Candidate(candidate_index=0, params={"n_estimators": 10}),
        x_source_raw=np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float),
        y_source=np.array([0, 1, 0, 1], dtype=int),
        groups_source=np.array(["g1", "g1", "g2", "g2"], dtype=str),
        feature_names=["OG1"],
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
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t0\tg1",
                "sp3\t1\tg2",
                "sp4\t0\tg2",
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
    cv_artifacts = run_outer_cv(config, split_artifacts.split_manifest)
    cv_threshold = (
        cv_artifacts.thresholds.filter(pl.col("threshold_name") == "cv_derived_threshold")
        .select("threshold_value")
        .to_series()
        .item()
    )

    refit = run_final_refit(config, split_artifacts.split_manifest, float(cv_threshold))

    assert refit.pred_external_test.height == 0
    assert refit.pred_inference.height == 0


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
        run_final_refit(config, split_manifest, cv_threshold=0.5)


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


def test_run_outer_cv_appends_threshold_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    split_artifacts = build_split_artifacts(config)
    monkeypatch.setattr(cv_mod, "_derive_cv_threshold", lambda *_args, **_kwargs: (0.5, "warn"))

    artifacts = run_outer_cv(config, split_artifacts.split_manifest)

    assert "warn" in artifacts.warnings


def test_prepare_source_selection_tpe_requires_selected_candidate_count(
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

    with pytest.raises(CVError, match="selected_candidate_count is required"):
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
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "trial_count": 5,
                    "search_space": {"C": [0.1]},
                    "selected_candidate_count": 3,
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
        sampled_idx=np.array([0, 1], dtype=int),
        x_train_raw=np.array([[1.0], [2.0]], dtype=float),
        y_train=np.array([0, 1], dtype=int),
        groups_train=np.array(["g1", "g2"], dtype=str),
        feature_names=["OG1"],
        warnings=warnings,
    )

    assert result.n_available_candidates == 1
    assert result.selected_candidate_count_effective == 1
    assert any("trial_count exceeded discrete candidate space" in item for item in warnings)
    assert any(
        "selected_candidate_count exceeded available candidates" in item for item in warnings
    )


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
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"search_strategy": "tpe", "trial_count": 1, "selected_candidate_count": 1}
            )
        }
    )
    monkeypatch.setattr(cv_mod.optuna, "create_study", lambda **_kwargs: _StudyWithoutParams())

    with pytest.raises(CVError, match="missing candidate_params user attribute"):
        _prepare_source_selection_tpe(
            config=config_tpe,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            sampled_idx=np.array([0, 1], dtype=int),
            x_train_raw=np.array([[1.0], [2.0]], dtype=float),
            y_train=np.array([0, 1], dtype=int),
            groups_train=np.array(["g1", "g2"], dtype=str),
            feature_names=["OG1"],
            warnings=[],
        )


def test_expression_matrix_builder_rejects_empty_matrix_for_selected_species(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(["species\tC4\tcontrast_pair_id", "sp1\t1\tg1"]) + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(["species\torthogroup\ttpm", "sp1\t\t1.0"]) + "\n",
    )
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    builder = ExpressionMatrixBuilder(config)

    with pytest.raises(CVError, match="Expression matrix is empty for the selected species"):
        builder.build_matrix(["sp1"])


def test_expression_matrix_builder_chunking_single_feature_returns_single_chunk(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t0\tg1",
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


def test_select_feature_indices_rejects_missing_low_prevalence_threshold_when_mutated(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_bad = config.model_copy(
        update={
            "preprocess": config.preprocess.model_copy(
                update={
                    "low_prevalence_filter": config.preprocess.low_prevalence_filter.model_copy(
                        update={"enabled": True, "min_species_per_feature": None}
                    )
                }
            )
        }
    )

    with pytest.raises(CVError, match="min_species_per_feature is missing"):
        _select_feature_indices(
            config_bad,
            np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
            ["OG1", "OG2"],
        )


def test_select_feature_indices_rejects_missing_low_variance_threshold_when_mutated(
    tmp_path: Path,
) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_bad = config.model_copy(
        update={
            "preprocess": config.preprocess.model_copy(
                update={
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
  low_prevalence_filter:
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
    monkeypatch.setattr(cv_mod, "_inner_cv_splits", lambda *_args, **_kwargs: [])

    score, rows = cv_mod._score_candidate_inner_cv(
        config=config,
        training_scope_id="fold_0",
        source_sample_set_id=0,
        candidate=Candidate(candidate_index=0, params={}),
        x_source_raw=np.array([[1.0], [2.0]], dtype=float),
        y_source=np.array([0, 1], dtype=int),
        groups_source=np.array(["g1", "g2"], dtype=str),
        feature_names=["OG1"],
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
    config = load_and_resolve_config([_config_path(tmp_path, metadata, tpm)])
    config_tpe = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={
                    "search_strategy": "tpe",
                    "trial_count": 1,
                    "search_space": {"C": [0.1]},
                    "selected_candidate_count": 1,
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
        sampled_idx=np.array([0, 1], dtype=int),
        x_train_raw=np.array([[1.0], [2.0]], dtype=float),
        y_train=np.array([0, 1], dtype=int),
        groups_train=np.array(["g1", "g2"], dtype=str),
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


def test_derive_cv_threshold_supports_balanced_accuracy_metric(tmp_path: Path) -> None:
    metadata, tpm = _write_fixture(tmp_path)
    config = load_and_resolve_config(
        [
            _config_path(
                tmp_path,
                metadata,
                tpm,
                extra="""
report:
  auto_threshold_selection_metric: balanced_accuracy
""".strip(),
            )
        ]
    )

    threshold, warning = _derive_cv_threshold(
        config,
        y_true=np.array([0, 1], dtype=int),
        prob=np.array([0.2, 0.8], dtype=float),
    )

    assert threshold == pytest.approx(0.8)
    assert warning is None


def test_build_prediction_table_includes_uncertainty_when_provided() -> None:
    table = _build_prediction_table(
        species=["sp1", "sp2"],
        prob=np.array([0.2, 0.8], dtype=float),
        fixed_threshold=0.5,
        cv_threshold=0.4,
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
        run_final_refit(config, split_manifest, cv_threshold=0.5)


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
        run_final_refit(config, split_artifacts.split_manifest, cv_threshold=0.5)


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
        run_final_refit(config, split_artifacts.split_manifest, cv_threshold=0.5)


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
