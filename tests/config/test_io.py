from __future__ import annotations

from pathlib import Path

import pytest

from phenoradar.config import ConfigError, load_and_resolve_config


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_deep_merge_and_seed_default(tmp_path: Path) -> None:
    base = _write(
        tmp_path / "base.yml",
        """
runtime:
  seed: 99
sampling:
  weighting: none
model_selection:
  search_space:
    C: [0.1, 1.0]
""".strip()
        + "\n",
    )
    override = _write(
        tmp_path / "override.yml",
        """
sampling:
  weighting: group_label_inverse
model_selection:
  search_space:
    C: [10.0]
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([base, override])

    assert resolved.runtime.seed == 99
    assert resolved.sampling.weighting == "group_label_inverse"
    assert resolved.model_selection.search_space["C"] == [10.0]


def test_missing_config_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_and_resolve_config([tmp_path / "missing.yml"])


def test_empty_config_file_resolves_to_defaults(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "empty.yml", "")
    resolved = load_and_resolve_config([cfg])

    assert resolved.runtime.seed == 42
    assert resolved.data.species_col == "species"
    assert resolved.data.orthogroup_annotation_path is None
    assert resolved.sampling.strategy == "group_balanced"
    assert resolved.sampling.max_samples_per_label_per_group == 1
    assert resolved.sampling.sampled_set_count == 10
    assert resolved.sampling.weighting == "none"
    assert resolved.model.logistic_solver == "saga"
    assert resolved.model.logistic_warm_start_path is False
    assert resolved.model_selection.selection_metric == "log_loss"
    assert resolved.model_selection.selection_rule == "best"
    assert resolved.model_selection.candidate_source_policy == "per_sample_set"
    assert resolved.preprocess.expression_transform.method == "log1p"
    assert resolved.preprocess.sparse_feature_filter.enabled is True
    assert (
        resolved.preprocess.sparse_feature_filter.min_nonzero_fraction_in_at_least_one_trait
        == 0.8
    )
    assert resolved.preprocess.feature_scaling.method == "standard"
    assert resolved.evaluation.group_bootstrap.enabled is False
    assert resolved.evaluation.group_bootstrap.n_resamples == 2000
    assert resolved.evaluation.group_bootstrap.confidence_level == 0.95
    assert resolved.summary.group_col == "family_id"
    assert resolved.summary.group_name_col == "family_name"
    assert resolved.figures.top_features == 30


def test_allow_empty_config_paths_resolves_to_defaults() -> None:
    resolved = load_and_resolve_config([], allow_empty=True)

    assert resolved.runtime.seed == 42
    assert resolved.data.species_col == "species"
    assert resolved.data.orthogroup_annotation_path is None
    assert resolved.sampling.strategy == "group_balanced"
    assert resolved.sampling.max_samples_per_label_per_group == 1
    assert resolved.sampling.sampled_set_count == 10
    assert resolved.sampling.weighting == "none"
    assert resolved.model_selection.selection_metric == "log_loss"
    assert resolved.model_selection.selection_rule == "best"
    assert resolved.model_selection.candidate_source_policy == "per_sample_set"
    assert resolved.preprocess.expression_transform.method == "log1p"
    assert resolved.preprocess.sparse_feature_filter.enabled is True
    assert (
        resolved.preprocess.sparse_feature_filter.min_nonzero_fraction_in_at_least_one_trait
        == 0.8
    )
    assert resolved.preprocess.feature_scaling.method == "standard"
    assert resolved.evaluation.group_bootstrap.enabled is False
    assert resolved.evaluation.group_bootstrap.n_resamples == 2000
    assert resolved.evaluation.group_bootstrap.confidence_level == 0.95
    assert resolved.summary.group_col == "family_id"
    assert resolved.summary.group_name_col == "family_name"
    assert resolved.figures.top_features == 30


def test_summary_group_columns_are_configurable(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "config.yml",
        """
summary:
  group_col: order_id
  group_name_col: order_name
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.summary.group_col == "order_id"
    assert resolved.summary.group_name_col == "order_name"


def test_group_bootstrap_is_configurable(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "config.yml",
        """
evaluation:
  group_bootstrap:
    enabled: true
    n_resamples: 123
    confidence_level: 0.9
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.evaluation.group_bootstrap.enabled is True
    assert resolved.evaluation.group_bootstrap.n_resamples == 123
    assert resolved.evaluation.group_bootstrap.confidence_level == 0.9


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1])
def test_group_bootstrap_rejects_invalid_confidence_level(
    tmp_path: Path, confidence_level: float
) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        f"""
evaluation:
  group_bootstrap:
    confidence_level: {confidence_level}
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
unknown_section:
  foo: 1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_legacy_data_group_col_is_rejected(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
data:
  group_col: contrast_pair_id
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="group_col"):
        load_and_resolve_config([cfg])


def test_group_kfold_requires_n_splits(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
split:
  outer_cv_strategy: group_kfold
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_stratified_group_kfold_requires_n_splits(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
split:
  outer_cv_strategy: stratified_group_kfold
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_stratified_group_kfold_accepts_valid_n_splits(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "config.yml",
        """
split:
  outer_cv_strategy: stratified_group_kfold
  outer_cv_n_splits: 5
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.split.outer_cv_strategy == "stratified_group_kfold"
    assert resolved.split.outer_cv_n_splits == 5


def test_all_samples_rejects_group_balancing_fields(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
sampling:
  strategy: all_samples
  max_samples_per_label_per_group: 2
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_random_strategy_requires_trial_count(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_strategy: random
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_grid_rejects_continuous_search_space(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_strategy: grid
  search_space:
    C:
      type: continuous_range
      start: 0.1
      end: 1.0
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_search_space_legacy_stop_keys_are_rejected(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "legacy_alias.yml",
        """
model_selection:
  search_strategy: random
  trial_count: 2
  search_space:
    C:
      type: log_range
      base: 10
      start_exp: -1
      stop_exp: 1
      step_exp: 1
      inclusive_stop: true
    l1_ratio:
      type: continuous_range
      start: 0.0
      stop: 1.0
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_selected_candidate_count_requires_inner_cv_strategy(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  selected_candidate_count: 2
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_selected_candidate_percent_requires_inner_cv_strategy(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  selected_candidate_percent: 25
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_selected_candidate_count_and_percent_are_mutually_exclusive(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  selected_candidate_count: 2
  selected_candidate_percent: 25
  inner_cv_strategy: logo
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_selected_candidate_percent_must_be_lte_100(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  selected_candidate_percent: 120
  inner_cv_strategy: logo
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_group_kfold_requires_n_splits_of_at_least_two(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
split:
  outer_cv_strategy: group_kfold
  outer_cv_n_splits: 1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_logo_rejects_outer_cv_n_splits(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
split:
  outer_cv_strategy: logo
  outer_cv_n_splits: 3
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_all_samples_requires_sampled_set_count_one(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
sampling:
  strategy: all_samples
  sampled_set_count: 2
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_preprocess_sparse_feature_rejects_null_fraction_when_enabled(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  sparse_feature_filter:
    enabled: true
    min_nonzero_fraction_in_at_least_one_trait: null
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_preprocess_low_variance_requires_min_variance_when_enabled(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  low_variance_filter:
    enabled: true
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_correlation_filter_requires_threshold_when_enabled(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  correlation_filter:
    enabled: true
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_correlation_filter_rejects_threshold_out_of_bounds(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  correlation_filter:
    enabled: true
    max_abs_correlation: 1.5
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_pair_aware_filter_requires_max_features_when_enabled(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  pair_aware_filter:
    enabled: true
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_generic_group_options_allow_null_contrast_pair_col(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "valid.yml",
        """
data:
  contrast_pair_col: null
split:
  group_col: family_id
sampling:
  weighting: group_label_inverse
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.data.contrast_pair_col is None
    assert resolved.split.group_col == "family_id"
    assert resolved.sampling.weighting == "group_label_inverse"


def test_pair_aware_filter_requires_contrast_pair_col(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
data:
  contrast_pair_col: null
preprocess:
  pair_aware_filter:
    enabled: true
    max_features: 10
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="contrast_pair_col"):
        load_and_resolve_config([cfg])


def test_pair_aware_group_balanced_allows_split_group_to_differ_from_contrast_pair(
    tmp_path: Path,
) -> None:
    cfg = _write(
        tmp_path / "valid.yml",
        """
data:
  contrast_pair_col: contrast_pair_id
split:
  group_col: family_id
preprocess:
  pair_aware_filter:
    enabled: true
    max_features: 10
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.data.contrast_pair_col == "contrast_pair_id"
    assert resolved.split.group_col == "family_id"
    assert resolved.preprocess.pair_aware_filter.min_contrast_pairs == 1


def test_pair_aware_filter_rejects_zero_min_contrast_pairs(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  pair_aware_filter:
    enabled: true
    max_features: 10
    min_contrast_pairs: 0
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="min_contrast_pairs"):
        load_and_resolve_config([cfg])


def test_report_fixed_probability_threshold_is_no_longer_configurable(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
report:
  fixed_probability_threshold: -0.1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_figures_top_features_must_be_between_one_and_one_hundred(tmp_path: Path) -> None:
    cfg_zero = _write(
        tmp_path / "zero.yml",
        """
figures:
  top_features: 0
""".strip()
        + "\n",
    )
    cfg_too_many = _write(
        tmp_path / "too-many.yml",
        """
figures:
  top_features: 101
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg_zero])
    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg_too_many])


def test_runtime_n_jobs_must_be_at_least_one(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
runtime:
  n_jobs: 0
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="runtime.n_jobs must be >= 1"):
        load_and_resolve_config([cfg])


def test_inner_group_kfold_requires_inner_n_splits(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: group_kfold
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_inner_stratified_group_kfold_requires_inner_n_splits(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: stratified_group_kfold
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_inner_logo_rejects_inner_n_splits(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
  inner_cv_n_splits: 3
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_inner_group_kfold_requires_inner_n_splits_of_at_least_two(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: group_kfold
  inner_cv_n_splits: 1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_log_range_rejects_end_exp_less_than_start_exp(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_space:
    C:
      type: log_range
      base: 10
      start_exp: 1.0
      end_exp: 0.0
      step_exp: 0.5
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_continuous_log_range_rejects_end_exp_less_than_start_exp(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_strategy: random
  trial_count: 10
  search_space:
    C:
      type: continuous_log_range
      base: 10
      start_exp: 1.0
      end_exp: 0.0
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_model_selection_search_space_rejects_empty_list(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_space:
    C: []
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_unknown_nested_model_key_is_rejected(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model:
  name: logistic_elasticnet
  calibration: sigmoid
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_liblinear_solver_accepts_explicit_l1_or_l2_ratios(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "liblinear.yml",
        """
model:
  name: logistic_elasticnet
  logistic_solver: liblinear
model_selection:
  search_space:
    l1_ratio: [0, 1]
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.model.logistic_solver == "liblinear"


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            """
model:
  name: logistic_elasticnet
  logistic_solver: liblinear
""",
            "requires an explicit",
        ),
        (
            """
model:
  name: logistic_elasticnet
  logistic_solver: liblinear
model_selection:
  search_space:
    l1_ratio: [0.5]
""",
            "supports only l1_ratio values 0 or 1",
        ),
        (
            """
model:
  name: random_forest
  logistic_solver: liblinear
model_selection:
  search_space:
    l1_ratio: [1]
""",
            "only valid when model.name=logistic_elasticnet",
        ),
    ],
)
def test_liblinear_solver_rejects_incompatible_config(
    tmp_path: Path,
    extra: str,
    message: str,
) -> None:
    cfg = _write(tmp_path / "invalid_liblinear.yml", extra.strip() + "\n")

    with pytest.raises(ConfigError, match=message):
        load_and_resolve_config([cfg])


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            """
model:
  name: random_forest
  logistic_warm_start_path: true
""",
            "only valid when model.name=logistic_elasticnet",
        ),
        (
            """
model:
  name: logistic_elasticnet
  logistic_solver: liblinear
  logistic_warm_start_path: true
model_selection:
  search_space:
    l1_ratio: [1]
""",
            "requires model.logistic_solver=saga",
        ),
        (
            """
model:
  name: logistic_elasticnet
  logistic_warm_start_path: true
model_selection:
  search_strategy: random
  trial_count: 2
""",
            "currently requires model_selection.search_strategy=grid",
        ),
    ],
)
def test_logistic_warm_start_path_rejects_incompatible_config(
    tmp_path: Path,
    extra: str,
    message: str,
) -> None:
    cfg = _write(tmp_path / "invalid_warm_start.yml", extra.strip() + "\n")

    with pytest.raises(ConfigError, match=message):
        load_and_resolve_config([cfg])


def test_tpe_strategy_requires_trial_count(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_strategy: tpe
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_range_requires_stop_greater_or_equal_start(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_space:
    C:
      type: range
      start: 1.0
      end: 0.1
      step: 0.1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_int_range_requires_stop_greater_or_equal_start(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_space:
    max_iter:
      type: int_range
      start: 10
      end: 1
      step: 1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_log_range_requires_valid_base(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_space:
    C:
      type: log_range
      base: 1
      start_exp: -1
      end_exp: 1
      step_exp: 1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_continuous_log_range_requires_valid_base(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_strategy: random
  trial_count: 2
  search_space:
    C:
      type: continuous_log_range
      base: 1
      start_exp: -1
      end_exp: 1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_continuous_range_requires_stop_greater_or_equal_start(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
model_selection:
  search_strategy: random
  trial_count: 2
  search_space:
    l1_ratio:
      type: continuous_range
      start: 1.0
      end: 0.0
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_top_level_yaml_must_be_mapping(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
- runtime:
    seed: 1
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="Top-level YAML must be a mapping"):
        load_and_resolve_config([cfg])


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        "runtime: [1, 2\n",
    )

    with pytest.raises(ConfigError, match="Invalid YAML in config file"):
        load_and_resolve_config([cfg])


def test_at_least_one_config_file_is_required() -> None:
    with pytest.raises(ConfigError, match="At least one config file must be provided"):
        load_and_resolve_config([])


def test_execution_stage_override_is_applied(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "base.yml",
        """
runtime:
  execution_stage: cv_only
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg], execution_stage_override="full_run")

    assert resolved.runtime.execution_stage == "full_run"
