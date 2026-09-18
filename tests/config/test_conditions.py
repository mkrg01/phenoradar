from __future__ import annotations

from pathlib import Path

import pytest

from phenoradar.config import ConfigError, has_condition_dimensions, load_config_conditions


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_single_varying_list_preserves_order_with_fixed_singleton_lists(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  ranked_feature_filter:
    method: [pair_aware]
    max_features: [100, 50]
  feature_scaling:
    method: [standard]
""".lstrip(),
    )

    condition_set = load_config_conditions([config_path])

    assert [condition.values for condition in condition_set.conditions] == [
        (
            ("preprocess.ranked_feature_filter.method", "pair_aware"),
            ("preprocess.ranked_feature_filter.max_features", count),
            ("preprocess.feature_scaling.method", "standard"),
        )
        for count in (100, 50)
    ]
    assert [condition.index for condition in condition_set.conditions] == [1, 2]
    assert len({condition.condition_id for condition in condition_set.conditions}) == 2


@pytest.mark.parametrize(
    "settings, expected_paths",
    [
        (
            "preprocess:\n  ranked_feature_filter:\n"
            "    method: pair_aware\n    max_features: [100, 50]\n"
            "  feature_scaling:\n    method: [standard, none]\n",
            ["preprocess.ranked_feature_filter.max_features", "preprocess.feature_scaling.method"],
        ),
        (
            "preprocess:\n  ranked_feature_filter:\n"
            "    method: [none, pair_aware]\n    max_features: [100, 50]\n",
            [
                "preprocess.ranked_feature_filter.method",
                "preprocess.ranked_feature_filter.max_features",
            ],
        ),
        (
            "sampling:\n  training_group_count: [5, 10]\n  group_subsample_repeats: 2\n"
            "preprocess:\n  feature_scaling:\n    method: [standard, none]\n",
            ["sampling.training_group_count", "preprocess.feature_scaling.method"],
        ),
        (
            "sampling:\n  training_group_count: [5, 10]\n"
            "  group_subsample_repeat_index: [1, 2]\n",
            ["sampling.training_group_count", "sampling.group_subsample_repeat_index"],
        ),
    ],
)
def test_multiple_varying_condition_fields_are_rejected(
    tmp_path: Path, settings: str, expected_paths: list[str]
) -> None:
    config_path = _write(tmp_path / "config.yml", settings)

    with pytest.raises(ConfigError, match="Only one condition variable") as error:
        load_config_conditions([config_path])

    assert all(path in str(error.value) for path in expected_paths)
    assert "Use separate studies" in str(error.value)


def test_directional_filter_values_expand_with_fixed_supervised_method(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  sparse_feature_filter:
    scope: any_trait
  ranked_feature_filter:
    method: pair_aware
    max_features: 100
    higher_in_trait: [null, 1]
""".lstrip(),
    )

    condition_set = load_config_conditions([config_path])
    assert [
        condition.config.preprocess.ranked_feature_filter.higher_in_trait
        for condition in condition_set.conditions
    ] == [None, 1]


def test_none_method_normalizes_fixed_max_features(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  ranked_feature_filter:
    method: [none, pair_aware]
    max_features: [100]
""".lstrip(),
    )

    condition_set = load_config_conditions([config_path])
    assert [
        condition.config.preprocess.ranked_feature_filter.max_features
        for condition in condition_set.conditions
    ] == [None, 100]
    assert condition_set.conditions[0].values == (
        ("preprocess.ranked_feature_filter.method", "none"),
        ("preprocess.ranked_feature_filter.max_features", None),
    )


def test_existing_search_space_lists_remain_one_config_value(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
model_selection:
  search_space:
    lambda: [0.1, 1.0, 10.0]
    alpha: [0.0, 1.0]
""".lstrip(),
    )

    assert has_condition_dimensions([config_path]) is False
    condition_set = load_config_conditions([config_path])
    assert condition_set.dimensions == ()
    assert len(condition_set.conditions) == 1
    assert condition_set.conditions[0].config.model_selection.search_space["lambda"] == [
        0.1,
        1.0,
        10.0,
    ]


def test_search_space_lists_do_not_count_as_condition_variables(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  sparse_feature_filter:
    min_nonzero_fraction: [0.9, 1]
model_selection:
  search_space:
    lambda: [0.1, 1.0]
    alpha: [0.0, 1.0]
""".lstrip(),
    )
    conditions = load_config_conditions([config_path])
    assert len(conditions.conditions) == 2
    assert len(conditions.dimensions) == 1
    assert all(
        condition.config.model_selection.search_space["alpha"] == [0.0, 1.0]
        for condition in conditions.conditions
    )


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
  group_col: [contrast_pair_id, family]
runtime:
  seed: [42, 43]
""".lstrip(),
    )

    with pytest.raises(ConfigError, match="split.group_col, runtime.seed"):
        load_config_conditions([config_path])


def test_absent_feature_fill_condition_list_is_rejected(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yml",
        """
preprocess:
  absent_feature_fill: [0, nan]
model:
  name: random_forest
""".lstrip(),
    )

    with pytest.raises(ConfigError, match="preprocess.absent_feature_fill"):
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
