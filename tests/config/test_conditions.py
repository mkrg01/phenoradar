from __future__ import annotations

from pathlib import Path

import pytest

from phenoradar.config import ConfigError, has_condition_dimensions, load_config_conditions


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_scalar_lists_expand_in_config_order(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  ranked_feature_filter:
    method: pair_aware
    max_features: [100, 50]
  feature_scaling:
    method: [standard, none]
""".lstrip(),
    )

    condition_set = load_config_conditions([config_path])

    assert [dimension.dotted_path for dimension in condition_set.dimensions] == [
        "preprocess.ranked_feature_filter.max_features",
        "preprocess.feature_scaling.method",
    ]
    assert [condition.values for condition in condition_set.conditions] == [
        (
            ("preprocess.ranked_feature_filter.max_features", 100),
            ("preprocess.feature_scaling.method", "standard"),
        ),
        (
            ("preprocess.ranked_feature_filter.max_features", 100),
            ("preprocess.feature_scaling.method", "none"),
        ),
        (
            ("preprocess.ranked_feature_filter.max_features", 50),
            ("preprocess.feature_scaling.method", "standard"),
        ),
        (
            ("preprocess.ranked_feature_filter.max_features", 50),
            ("preprocess.feature_scaling.method", "none"),
        ),
    ]
    assert [condition.index for condition in condition_set.conditions] == [1, 2, 3, 4]
    assert len({condition.condition_id for condition in condition_set.conditions}) == 4


def test_none_method_is_independent_of_max_features_dimension(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  ranked_feature_filter:
    method: [none, pair_aware]
    max_features: [5000, 2500, 1000, 500, 250, 100, 50]
    min_contrast_pairs: 1
""".lstrip(),
    )

    condition_set = load_config_conditions([config_path])

    assert len(condition_set.conditions) == 8
    assert [
        condition.config.preprocess.ranked_feature_filter.method
        for condition in condition_set.conditions
    ] == [
        "none",
        "pair_aware",
        "pair_aware",
        "pair_aware",
        "pair_aware",
        "pair_aware",
        "pair_aware",
        "pair_aware",
    ]
    assert [
        condition.config.preprocess.ranked_feature_filter.max_features
        for condition in condition_set.conditions
    ] == [None, 5000, 2500, 1000, 500, 250, 100, 50]
    assert condition_set.conditions[0].values == (
        ("preprocess.ranked_feature_filter.method", "none"),
        ("preprocess.ranked_feature_filter.max_features", None),
    )
    assert [condition.index for condition in condition_set.conditions] == list(range(1, 9))


def test_none_method_still_expands_other_condition_dimensions(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  ranked_feature_filter:
    method: [none, pair_aware]
    max_features: [100, 50]
  feature_scaling:
    method: [standard, none]
""".lstrip(),
    )

    condition_set = load_config_conditions([config_path])

    assert len(condition_set.conditions) == 6
    assert [
        (
            condition.config.preprocess.ranked_feature_filter.method,
            condition.config.preprocess.ranked_feature_filter.max_features,
            condition.config.preprocess.feature_scaling.method,
        )
        for condition in condition_set.conditions
    ] == [
        ("none", None, "standard"),
        ("none", None, "none"),
        ("pair_aware", 100, "standard"),
        ("pair_aware", 100, "none"),
        ("pair_aware", 50, "standard"),
        ("pair_aware", 50, "none"),
    ]


def test_existing_search_space_lists_remain_one_config_value(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
model_selection:
  search_space:
    C: [0.1, 1.0, 10.0]
    l1_ratio: [0.0, 1.0]
""".lstrip(),
    )

    assert has_condition_dimensions([config_path]) is False
    condition_set = load_config_conditions([config_path])
    assert condition_set.dimensions == ()
    assert len(condition_set.conditions) == 1
    assert condition_set.conditions[0].config.model_selection.search_space["C"] == [
        0.1,
        1.0,
        10.0,
    ]


def test_invalid_generated_combination_is_rejected(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
sampling:
  strategy: [all_samples, group_balanced]
  max_samples_per_label_per_group: 1
  sampled_set_count: 10
""".lstrip(),
    )

    with pytest.raises(ConfigError, match="Invalid generated condition 1"):
        load_config_conditions([config_path])


def test_split_and_runtime_condition_lists_are_rejected(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
split:
  group_col: [contrast_pair_id, family_id]
runtime:
  seed: [42, 43]
""".lstrip(),
    )

    with pytest.raises(ConfigError, match="split.group_col, runtime.seed"):
        load_config_conditions([config_path])


def test_duplicate_condition_values_are_rejected(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  ranked_feature_filter:
    method: pair_aware
    max_features: [100, 100]
""".lstrip(),
    )

    with pytest.raises(ConfigError, match="duplicates condition 1"):
        load_config_conditions([config_path])


def test_training_group_count_and_repeats_expand_with_full_set_deduplicated(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
sampling:
  training_group_count: [5, null]
  group_subsample_repeats: 2
""".lstrip(),
    )

    assert has_condition_dimensions([config_path]) is True
    condition_set = load_config_conditions([config_path])

    assert [
        (
            condition.config.sampling.training_group_count,
            condition.config.sampling.group_subsample_repeats,
            condition.config.sampling.group_subsample_repeat_index,
        )
        for condition in condition_set.conditions
    ] == [(5, 1, 1), (5, 1, 2), (None, 1, 1)]
    assert [condition.index for condition in condition_set.conditions] == [1, 2, 3]
    assert condition_set.conditions[-1].values == (
        ("sampling.training_group_count", None),
        ("sampling.group_subsample_repeat_index", 1),
    )


def test_scalar_group_subsample_repeat_count_creates_repeat_conditions(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
sampling:
  training_group_count: 5
  group_subsample_repeats: 3
""".lstrip(),
    )

    condition_set = load_config_conditions([config_path])

    assert [
        condition.config.sampling.group_subsample_repeat_index
        for condition in condition_set.conditions
    ] == [1, 2, 3]
    assert all(
        condition.config.sampling.group_subsample_repeats == 1
        for condition in condition_set.conditions
    )
    assert all(
        condition.values[0] == ("sampling.training_group_count", 5)
        for condition in condition_set.conditions
    )


def test_group_subsample_repeats_rejects_condition_list(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
sampling:
  training_group_count: 5
  group_subsample_repeats: [1, 2]
""".lstrip(),
    )

    with pytest.raises(ConfigError, match="must be one integer repeat count"):
        load_config_conditions([config_path])
