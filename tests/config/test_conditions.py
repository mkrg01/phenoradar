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
