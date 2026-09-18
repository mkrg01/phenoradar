from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import phenoradar.cv as cv
from phenoradar.config import AppConfig


def _assert_same_preprocessing(expected: tuple[Any, ...], actual: tuple[Any, ...]) -> None:
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:4] == expected[2:4]
    if expected[4] is None:
        assert actual[4] is None
    else:
        for name in ("mean_", "var_", "scale_", "n_samples_seen_"):
            np.testing.assert_array_equal(getattr(actual[4], name), getattr(expected[4], name))


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("scaling", ["none", "standard"])
@pytest.mark.parametrize("transform", ["none", "log1p", "sample_rank", "sample_percentile_rank"])
@pytest.mark.parametrize("indexed_train", [False, True])
@pytest.mark.parametrize("rank_method", ["none", "variance", "unpaired", "pair_aware"])
def test_delayed_validation_gather_preserves_preprocessing(
    layout: str, scaling: str, transform: str, indexed_train: bool, rank_method: str
) -> None:
    neutral = transform == "log1p" and scaling == "standard"
    config = AppConfig.model_validate(
        {
            "preprocess": {
                "absent_feature_fill": "nan" if neutral else 0,
                "missing_expression": {"method": "neutral" if neutral else "none"},
                "expression_transform": {"method": transform},
                "feature_scaling": {"method": scaling},
                "sparse_feature_filter": {"min_nonzero_fraction": 0.5},
                "ranked_feature_filter": {"method": rank_method, "max_features": 3},
            }
        }
    )
    raw = np.random.default_rng(582).lognormal(size=(16, 18))
    raw[::3, ::4] = np.nan if neutral else 0.0
    source = cv._apply_expression_transform_for_config(config, raw)
    if layout == "F":
        source = np.asfortranarray(source)
    elif layout == "strided":
        source = source[:, ::2]
    original = source.copy()
    source.flags.writeable = False
    train_rows = np.array([8, 1, 6, 3, 0, 5, 2, 7])
    valid_rows = np.array([15, 10, 10, 9])
    names = [f"f{i}" for i in range(source.shape[1])]
    train = source[train_rows]
    labels = train_rows % 2
    groups = train_rows // 2
    expected_rows: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    expected_warnings: list[str] = []
    actual_warnings: list[str] = []
    expected = cv._preprocess_transformed_fold_with_counts(
        config,
        train,
        source[valid_rows],
        names,
        y_train=labels,
        groups_train=groups,
        warnings=expected_warnings,
        ranked_feature_score_rows=expected_rows,
    )
    actual = cv._preprocess_transformed_fold_with_counts(
        config,
        source if indexed_train else train,
        source,
        names,
        y_train=labels,
        groups_train=groups,
        train_rows=train_rows if indexed_train else None,
        valid_rows=valid_rows,
        warnings=actual_warnings,
        ranked_feature_score_rows=actual_rows,
    )
    _assert_same_preprocessing(expected, actual)
    assert actual_rows == expected_rows
    assert actual_warnings == expected_warnings
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("scope", ["all_samples", "any_trait", "trait_0", "trait_1"])
@pytest.mark.parametrize("correlation", ["none", "pearson", "spearman"])
@pytest.mark.parametrize("neutral", [False, True])
def test_indexed_filters_use_only_training_rows(
    scope: str, correlation: str, neutral: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv, "_SPARSE_FILTER_BLOCK_CELLS", 24)
    monkeypatch.setattr(cv, "_NEUTRAL_FILTER_BLOCK_CELLS", 8)  # Single-column chunks.
    config = AppConfig.model_validate(
        {
            "preprocess": {
                "absent_feature_fill": "nan" if neutral else 0,
                "missing_expression": {"method": "neutral" if neutral else "none"},
                "sparse_feature_filter": {"scope": scope, "min_nonzero_fraction": 0.5},
                "low_variance_filter": {"enabled": True, "min_variance": 0.001},
                "ranked_feature_filter": {"method": "pair_aware", "max_features": 6},
                "correlation_filter": {
                    "enabled": correlation != "none",
                    "method": "pearson" if correlation == "none" else correlation,
                    "max_abs_correlation": 0.95,
                },
            }
        }
    )
    source = np.random.default_rng(672).lognormal(size=(12, 14))
    train_rows = np.array([8, 1, 4, 5, 0, 9, 2, 3])
    valid_rows = np.array([7, 6, 11, 10])
    source[train_rows[:4], 2:5] = np.nan if neutral else 0.0
    source[train_rows, 0] = 0.0  # Observed only outside this training fold.
    source[:, 1] = source[:, 6]  # Correlation filter must choose identically.
    names = [f"f{i}" for i in range(source.shape[1])]
    kwargs = {"y_train": train_rows % 2, "groups_train": train_rows // 2}
    expected = cv._preprocess_transformed_fold_with_counts(
        config,
        source[train_rows],
        source[valid_rows],
        names,
        **kwargs,
    )
    actual = cv._preprocess_transformed_fold_with_counts(
        config,
        source,
        source,
        names,
        train_rows=train_rows,
        valid_rows=valid_rows,
        **kwargs,
    )
    _assert_same_preprocessing(expected, actual)
    assert "f0" not in actual[2]
    # Arbitrary held-out values must not affect selection, training values or scaler.
    source[valid_rows] = np.arange(source.shape[1]) * 1e10
    changed = cv._preprocess_transformed_fold_with_counts(
        config,
        source,
        source,
        names,
        train_rows=train_rows,
        valid_rows=valid_rows,
        **kwargs,
    )
    np.testing.assert_array_equal(changed[0], actual[0])
    assert changed[2:4] == actual[2:4]
    for name in ("mean_", "var_", "scale_"):
        np.testing.assert_array_equal(getattr(changed[4], name), getattr(actual[4], name))


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("width", [1, 3])
def test_indexed_neutral_single_column_keeps_materialized_row_reduction(
    layout: str, width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv, "_NEUTRAL_FILTER_BLOCK_CELLS", 81)
    config = AppConfig.model_validate(
        {
            "preprocess": {
                "absent_feature_fill": "nan",
                "missing_expression": {"method": "neutral"},
                "sparse_feature_filter": {"min_nonzero_fraction": 1.0},
            }
        }
    )
    source = np.zeros((100, width), order=layout)
    source[:, 0] = 0.1
    rows = np.random.default_rng(122).permutation(100)[:81]
    names = [f"f{i}" for i in range(width)]
    # The reference materializes a C-order training fold even from F-order input.
    try:
        expected = cv._select_feature_indices_with_counts(config, source[rows], names)
    except cv.CVError as exc:
        with pytest.raises(cv.CVError, match=str(exc)):
            cv._select_feature_indices_with_counts(config, source, names, train_rows=rows)
    else:
        actual = cv._select_feature_indices_with_counts(config, source, names, train_rows=rows)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]


def test_indexed_preprocessing_preserves_repeated_training_rows_and_empty_validation() -> None:
    config = AppConfig.model_validate(
        {
            "preprocess": {
                "sparse_feature_filter": {"scope": "any_trait", "min_nonzero_fraction": 0.5},
                "ranked_feature_filter": {"method": "unpaired", "max_features": 2},
            }
        }
    )
    source = np.random.default_rng(592).lognormal(size=(8, 5))
    rows = np.array([7, 0, 7, 2, 3, 4])
    valid = np.array([], dtype=int)
    names = [f"f{i}" for i in range(5)]
    expected = cv._preprocess_transformed_fold_with_counts(
        config,
        source[rows],
        source[valid],
        names,
        y_train=rows % 2,
    )
    actual = cv._preprocess_transformed_fold_with_counts(
        config,
        source,
        source,
        names,
        y_train=rows % 2,
        train_rows=rows,
        valid_rows=valid,
    )
    _assert_same_preprocessing(expected, actual)


def test_inner_fold_builder_never_materializes_all_columns_before_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv, "_SPARSE_FILTER_BLOCK_CELLS", 16)
    monkeypatch.setattr(cv, "_NEUTRAL_FILTER_BLOCK_CELLS", 16)
    config = AppConfig.model_validate(
        {
            "preprocess": {
                "absent_feature_fill": "nan",
                "missing_expression": {"method": "neutral"},
                "sparse_feature_filter": {"min_nonzero_fraction": 1.0},
                "ranked_feature_filter": {"method": "pair_aware", "max_features": 2},
            },
            "model_selection": {"selected_candidate_count": 1, "inner_cv_strategy": "logo"},
        }
    )
    raw = np.random.default_rng(935).lognormal(size=(12, 64))
    labels = np.tile([0, 1], 6)
    groups = np.repeat(np.arange(3), 4)
    kwargs = {
        "config": config,
        "x_source_raw": raw,
        "y_source": labels,
        "groups_source": groups,
        "contrast_groups_source": groups,
        "feature_names": [f"f{i}" for i in range(64)],
    }
    expected = cv._build_inner_cv_preprocessed_folds(**kwargs)
    source = cv._apply_expression_transform_for_config(config, raw)

    class GuardedRows(np.ndarray):
        def __getitem__(self, key: Any) -> Any:
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[0], np.ndarray)
                and isinstance(key[1], slice)
                and self.ndim == 2
                and self.shape[1] == 64
            ):
                start, stop, step = key[1].indices(self.shape[1])
                assert len(range(start, stop, step)) <= 2, "Full-width row copy before selection"
            return super().__getitem__(key)

    guarded = source.view(GuardedRows)
    guarded.flags.writeable = False
    monkeypatch.setattr(cv, "_apply_expression_transform_for_config", lambda *_args: guarded)
    actual = cv._build_inner_cv_preprocessed_folds(**kwargs)
    assert len(actual) == len(expected) == 3
    for before, after in zip(expected, actual, strict=True):
        assert after.x_train.shape[1] == after.x_valid.shape[1] == 2
        assert after.inner_fold_id == before.inner_fold_id
        for name in ("x_train", "x_valid", "y_train", "y_valid", "sample_weight"):
            np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    np.testing.assert_array_equal(guarded, source)
