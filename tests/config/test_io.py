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
    assert resolved.sampling.strategy == "group_balanced"
    assert resolved.sampling.max_samples_per_label_per_group == 1
    assert resolved.sampling.sampled_set_count == 10
    assert resolved.sampling.weighting == "none"
    assert resolved.model_selection.selection_metric == "log_loss"
    assert resolved.preprocess.low_prevalence_filter.enabled is True
    assert resolved.preprocess.low_prevalence_filter.min_species_per_feature == 2


def test_allow_empty_config_paths_resolves_to_defaults() -> None:
    resolved = load_and_resolve_config([], allow_empty=True)

    assert resolved.runtime.seed == 42
    assert resolved.data.species_col == "species"
    assert resolved.sampling.strategy == "group_balanced"
    assert resolved.sampling.max_samples_per_label_per_group == 1
    assert resolved.sampling.sampled_set_count == 10
    assert resolved.sampling.weighting == "none"
    assert resolved.model_selection.selection_metric == "log_loss"


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


def test_preprocess_low_prevalence_rejects_null_min_species_when_enabled(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "invalid.yml",
        """
preprocess:
  low_prevalence_filter:
    enabled: true
    min_species_per_feature: null
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


def test_report_fixed_probability_threshold_must_be_between_zero_and_one(tmp_path: Path) -> None:
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
