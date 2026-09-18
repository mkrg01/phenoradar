from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phenoradar.config import (
    AppConfig,
    ConfigError,
    load_and_resolve_config,
    serialize_resolved_config,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_annotated_config_round_trips_all_fields_and_quoted_values() -> None:
    config = AppConfig.model_validate(
        {
            "data": {
                "metadata_path": "data/種: # metadata.tsv",
                "tree_path": "data/first line\nsecond line.nwk",
                "species_col": "null",
            },
            "sampling": {"training_group_count": 5, "group_subsample_repeat_index": 3},
        }
    )

    rendered = serialize_resolved_config(config)

    assert yaml.safe_load(rendered) == config.model_dump(mode="python")
    assert AppConfig.model_validate(yaml.safe_load(rendered)) == config
    assert serialize_resolved_config(config) == rendered
    assert "require_both_labels_per_group: false  # choices: true, false" in rendered
    assert "logistic_solver" not in rendered
    assert "logistic_warm_start_path" not in rendered
    assert "higher_in_trait: null  # choices: 0, 1, null" in rendered
    assert (
        "inner_cv_strategy: null  # choices: logo, group_kfold, stratified_group_kfold, null"
    ) in rendered
    assert "tree_path:" in rendered and "# type: string or null" in rendered
    assert "group_subsample_repeat_index: 3" in rendered
    assert "report: {}" in rendered


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "range", "start": 0.1, "end": 1.0, "step": 0.1},
        {"type": "int_range", "start": 1, "end": 3, "step": 1},
        {"type": "log_range", "base": 10, "start_exp": -2, "end_exp": 2, "step_exp": 1},
        {"type": "continuous_range", "start": 0.1, "end": 1.0},
        {"type": "continuous_log_range", "base": 10, "start_exp": -2, "end_exp": 2},
    ],
)
def test_annotated_search_space_preserves_values_and_lists_range_types(
    spec: dict[str, object],
) -> None:
    config = AppConfig.model_validate(
        {
            "model_selection": {
                "search_strategy": "tpe",
                "trial_count": 3,
                "search_space": {"lambda": spec, "alpha": [0.0, 1.0]},
            }
        }
    )

    rendered = serialize_resolved_config(config)

    assert yaml.safe_load(rendered) == config.model_dump(mode="python")
    assert (
        f"type: {spec['type']}  # choices: range, int_range, log_range, "
        "continuous_range, continuous_log_range"
    ) in rendered
    if not str(spec["type"]).startswith("continuous"):
        assert "inclusive_end: false  # choices: true, false" in rendered


def test_checked_in_config_explicitly_contains_all_user_fields() -> None:
    path = Path(__file__).resolve().parents[2] / "config.yml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload == load_and_resolve_config([path]).model_dump(
        mode="python", exclude={"sampling": {"group_subsample_repeat_index"}}
    )


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
    lambda: [0.1, 1.0]
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
    lambda: [10.0]
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([base, override])

    assert resolved.runtime.seed == 99
    assert resolved.sampling.weighting == "group_label_inverse"
    assert resolved.model_selection.search_space["lambda"] == [10.0]


def test_missing_config_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_and_resolve_config([tmp_path / "missing.yml"])


def test_empty_config_file_resolves_to_defaults(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "empty.yml", "")
    resolved = load_and_resolve_config([cfg])

    assert resolved.runtime.seed == 42
    assert resolved.data.species_col == "species"
    assert resolved.data.orthogroup_annotation_path is None
    assert resolved.split.require_both_labels_per_group is False
    assert resolved.sampling.strategy == "group_balanced"
    assert resolved.sampling.max_samples_per_label_per_group == 1
    assert resolved.sampling.sampled_set_count == 10
    assert resolved.sampling.training_group_count is None
    assert resolved.sampling.group_subsample_repeats == 1
    assert resolved.sampling.group_subsample_repeat_index == 1
    assert resolved.sampling.weighting == "none"
    assert resolved.model_selection.selection_metric == "log_loss"
    assert resolved.model_selection.selection_rule == "best"
    assert resolved.model_selection.candidate_source_policy == "per_sample_set"
    assert resolved.preprocess.absent_feature_fill == 0
    assert resolved.preprocess.expression_transform.method == "log1p"
    assert resolved.preprocess.sparse_feature_filter.enabled is True
    assert (
        resolved.preprocess.sparse_feature_filter.min_nonzero_fraction
        == 0.8
    )
    assert resolved.preprocess.sparse_feature_filter.scope == "any_trait"
    assert resolved.preprocess.ranked_feature_filter.higher_in_trait is None
    assert resolved.preprocess.feature_scaling.method == "standard"
    assert resolved.evaluation.group_bootstrap.enabled is False
    assert resolved.evaluation.group_bootstrap.n_resamples == 2000
    assert resolved.evaluation.group_bootstrap.confidence_level == 0.95
    assert resolved.summary.group_col == "family"
    assert resolved.figures.top_features == 30


def test_allow_empty_config_paths_resolves_to_defaults() -> None:
    resolved = load_and_resolve_config([], allow_empty=True)

    assert resolved.runtime.seed == 42
    assert resolved.data.species_col == "species"
    assert resolved.data.orthogroup_annotation_path is None
    assert resolved.split.require_both_labels_per_group is False
    assert resolved.sampling.strategy == "group_balanced"
    assert resolved.sampling.max_samples_per_label_per_group == 1
    assert resolved.sampling.sampled_set_count == 10
    assert resolved.sampling.training_group_count is None
    assert resolved.sampling.group_subsample_repeats == 1
    assert resolved.sampling.group_subsample_repeat_index == 1
    assert resolved.sampling.weighting == "none"
    assert resolved.model_selection.selection_metric == "log_loss"
    assert resolved.model_selection.selection_rule == "best"
    assert resolved.model_selection.candidate_source_policy == "per_sample_set"
    assert resolved.preprocess.absent_feature_fill == 0
    assert resolved.preprocess.expression_transform.method == "log1p"
    assert resolved.preprocess.sparse_feature_filter.enabled is True
    assert (
        resolved.preprocess.sparse_feature_filter.min_nonzero_fraction
        == 0.8
    )
    assert resolved.preprocess.sparse_feature_filter.scope == "any_trait"
    assert resolved.preprocess.ranked_feature_filter.higher_in_trait is None
    assert resolved.preprocess.feature_scaling.method == "standard"
    assert resolved.evaluation.group_bootstrap.enabled is False
    assert resolved.evaluation.group_bootstrap.n_resamples == 2000
    assert resolved.evaluation.group_bootstrap.confidence_level == 0.95
    assert resolved.summary.group_col == "family"
    assert resolved.figures.top_features == 30


def test_summary_group_column_is_configurable(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "config.yml",
        """
summary:
  group_col: order
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.summary.group_col == "order"


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
    lambda:
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
    lambda:
      type: log_range
      base: 10
      start_exp: -1
      stop_exp: 1
      step_exp: 1
      inclusive_stop: true
    alpha:
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


def test_training_group_count_must_support_model_selection_inner_cv(
    tmp_path: Path,
) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
sampling:
  training_group_count: 4
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: group_kfold
  inner_cv_n_splits: 5
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="training_group_count"):
        load_and_resolve_config([cfg])


def test_preprocess_sparse_feature_rejects_null_fraction_when_enabled(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  sparse_feature_filter:
    enabled: true
    min_nonzero_fraction: null
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_preprocess_nan_absent_feature_fill_accepts_random_forest(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "valid.yml",
        """
preprocess:
  absent_feature_fill: nan
model:
  name: random_forest
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.preprocess.absent_feature_fill == "nan"


@pytest.mark.parametrize("value", ["zero", "false", "1"])
def test_preprocess_absent_feature_fill_rejects_values_other_than_numeric_zero_or_nan(
    tmp_path: Path,
    value: str,
) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        f"""
preprocess:
  absent_feature_fill: {value}
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="absent_feature_fill"):
        load_and_resolve_config([cfg])


def test_preprocess_nan_absent_feature_fill_rejects_non_random_forest(
    tmp_path: Path,
) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  absent_feature_fill: nan
model:
  name: logistic_elasticnet
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="only supported.*random_forest"):
        load_and_resolve_config([cfg])


def test_preprocess_sparse_feature_accepts_trait_scope(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "valid.yml",
        """
preprocess:
  sparse_feature_filter:
    scope: trait_1
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.preprocess.sparse_feature_filter.scope == "trait_1"


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


def test_ranked_feature_filter_requires_max_features_for_ranked_method(
    tmp_path: Path,
) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  ranked_feature_filter:
    method: pair_aware
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError):
        load_and_resolve_config([cfg])


def test_ranked_feature_filter_accepts_supervised_higher_in_trait(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "valid.yml",
        """
preprocess:
  ranked_feature_filter:
    method: unpaired
    max_features: 10
    higher_in_trait: 1
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.preprocess.ranked_feature_filter.higher_in_trait == 1


@pytest.mark.parametrize("method", ["none", "variance"])
def test_ranked_feature_filter_rejects_higher_in_trait_for_unsupervised_method(
    tmp_path: Path,
    method: str,
) -> None:
    max_features = "" if method == "none" else "    max_features: 10\n"
    cfg = _write(
        tmp_path / "invalid.yml",
        (
            "preprocess:\n"
            "  ranked_feature_filter:\n"
            f"    method: {method}\n"
            f"{max_features}"
            "    higher_in_trait: 1\n"
        ),
    )

    with pytest.raises(ConfigError, match="higher_in_trait"):
        load_and_resolve_config([cfg])


def test_removed_pair_aware_filter_key_is_rejected(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  pair_aware_filter:
    enabled: true
    max_features: 10
""".strip()
        + "\n",
    )

    with pytest.raises(ConfigError, match="pair_aware_filter"):
        load_and_resolve_config([cfg])


def test_unpaired_ranked_filter_does_not_require_contrast_pair_col(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "valid.yml",
        """
data:
  contrast_pair_col: null
preprocess:
  ranked_feature_filter:
    method: unpaired
    max_features: 10
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.preprocess.ranked_feature_filter.method == "unpaired"


def test_generic_group_options_allow_null_contrast_pair_col(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "valid.yml",
        """
data:
  contrast_pair_col: null
split:
  group_col: family
sampling:
  weighting: group_label_inverse
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.data.contrast_pair_col is None
    assert resolved.split.group_col == "family"
    assert resolved.sampling.weighting == "group_label_inverse"


def test_pair_aware_filter_requires_contrast_pair_col(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
data:
  contrast_pair_col: null
preprocess:
  ranked_feature_filter:
    method: pair_aware
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
  group_col: family
preprocess:
  ranked_feature_filter:
    method: pair_aware
    max_features: 10
""".strip()
        + "\n",
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.data.contrast_pair_col == "contrast_pair_id"
    assert resolved.split.group_col == "family"
    assert resolved.preprocess.ranked_feature_filter.min_contrast_pairs == 1


def test_pair_aware_filter_rejects_zero_min_contrast_pairs(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  ranked_feature_filter:
    method: pair_aware
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
    lambda:
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
    lambda:
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
    lambda: []
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


@pytest.mark.parametrize("solver", ["saga", "liblinear"])
def test_legacy_logistic_solver_is_rejected(tmp_path: Path, solver: str) -> None:
    cfg = _write(tmp_path / "legacy.yml", f"model:\n  logistic_solver: {solver}\n")

    with pytest.raises(ConfigError, match="logistic_solver"):
        load_and_resolve_config([cfg])


def test_glmnet_search_space_accepts_full_elastic_net_range(
    tmp_path: Path
) -> None:
    cfg = _write(
        tmp_path / "glmnet.yml",
        """
model:
  name: logistic_elasticnet
model_selection:
  search_space:
    lambda: [0.0001, 0.01, 0.1]
    alpha: [0, 0.5, 1]
    maxit: [100]
    thresh: [1.0e-14]
""".lstrip(),
    )

    resolved = load_and_resolve_config([cfg])

    assert resolved.model_selection.search_space == {
        "lambda": [0.0001, 0.01, 0.1],
        "alpha": [0, 0.5, 1],
        "maxit": [100],
        "thresh": [1.0e-14],
    }


@pytest.mark.parametrize(
    "parameter", ["C", "tol", "penalty", "l1_ratio", "max_iter", "gradient_tol"]
)
def test_logistic_search_space_rejects_removed_parameters(
    tmp_path: Path, parameter: str
) -> None:
    cfg = _write(
        tmp_path / "legacy.yml",
        f"model_selection:\n  search_space:\n    {parameter}: [1.0]\n",
    )

    with pytest.raises(ConfigError, match="Use glmnet parameters lambda"):
        load_and_resolve_config([cfg])


def test_linear_svm_retains_c_search_parameter(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "svm.yml",
        "model:\n  name: linear_svm\n"
        "model_selection:\n  search_space:\n    C: [0.1, 1.0]\n",
    )

    assert load_and_resolve_config([cfg]).model_selection.search_space == {
        "C": [0.1, 1.0]
    }


@pytest.mark.parametrize(
    "model_name, strategy",
    [
        ("random_forest", "grid"),
        ("linear_svm", "grid"),
        ("logistic_elasticnet", "random"),
        ("logistic_elasticnet", "tpe"),
    ],
)
def test_other_models_and_search_strategies_resolve(
    tmp_path: Path,
    model_name: str,
    strategy: str,
) -> None:
    cfg = _write(
        tmp_path / "model_strategy.yml",
        f"model:\n  name: {model_name}\n"
        f"model_selection:\n  search_strategy: {strategy}\n"
        + ("  trial_count: 2\n" if strategy != "grid" else ""),
    )

    assert load_and_resolve_config([cfg]).model.name == model_name


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
    lambda:
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
    maxit:
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
    lambda:
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
    lambda:
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
    alpha:
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


def test_removed_warm_start_switch_is_rejected(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "removed.yml", "model:\n  logistic_warm_start_path: true\n")
    with pytest.raises(ConfigError, match="logistic_warm_start_path"):
        load_and_resolve_config([cfg])
