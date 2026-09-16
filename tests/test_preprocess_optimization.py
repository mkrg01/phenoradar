from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import phenoradar.cv as cv_mod
from phenoradar.config import AppConfig


def _config(*, neutral: bool = False, **preprocess: Any) -> AppConfig:
    settings: dict[str, Any] = {
        "sparse_feature_filter": {"enabled": False},
    }
    if neutral:
        settings.update(
            absent_feature_fill="nan",
            missing_expression={"method": "neutral"},
        )
    settings.update(preprocess)
    return AppConfig.model_validate({"preprocess": settings})


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("zero_as_missing", [False, True])
@pytest.mark.parametrize("method", ["none", "unpaired", "pair_aware"])
@pytest.mark.parametrize("min_fraction", [0.0, 2 / 3, 1.0])
@pytest.mark.parametrize("scope", ["all_samples", "any_trait", "trait_0", "trait_1"])
def test_sparse_and_neutral_intersection_preserves_candidates_and_stage_counts(
    order: str,
    zero_as_missing: bool,
    method: str,
    min_fraction: float,
    scope: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force several column blocks, including blocks with no eligible features.
    monkeypatch.setattr(cv_mod, "_SPARSE_FILTER_BLOCK_CELLS", 18)
    monkeypatch.setattr(cv_mod, "_NEUTRAL_FILTER_BLOCK_CELLS", 12)
    tolerance = cv_mod._NONZERO_TOLERANCE
    values = np.array(
        [
            [1, 2, 3, 4, 5, 6],  # Dense, varying feature.
            [1, 0, 2, 0, 3, 0],  # Trait 0 only.
            [0, 4, 0, 5, 0, 6],  # Trait 1 only.
            [1, 2, np.nan, np.nan, 3, 4],  # Two complete contrast pairs.
            [1, np.nan, 2, np.nan, 3, np.nan],  # No observations in trait 1.
            [2, 2, 2, 2, 2, 2],  # Constant despite passing sparsity.
            [np.nan, np.nan, np.nan, 2, np.nan, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan] * 6,
            [tolerance, np.nextafter(tolerance, np.inf), tolerance, 2 * tolerance, 0, np.nan],
            [1, np.nan, np.nan, 2, np.nan, np.nan],  # Both labels, no paired observations.
            [1, 3, 2, np.nan, np.nan, 5],  # Both labels, only one complete pair.
        ],
        dtype=float,
    ).T
    if zero_as_missing:
        values = np.where(values == 0, np.nan, values)
    values = np.array(values, order=order)
    original = values.copy()
    names = [f"feature_{index:02d}" for index in range(values.shape[1])]
    config = _config(
        neutral=True,
        missing_expression={"method": "neutral", "zero_as_missing": zero_as_missing},
        sparse_feature_filter={"scope": scope, "min_nonzero_fraction": min_fraction},
        ranked_feature_filter={"method": method, "max_features": 20, "min_contrast_pairs": 2},
    )
    # Hand-checked eligibility before sparsity, including pair-specific missingness.
    eligible = {
        (False, "none"): [0, 1, 2, 3, 4, 9, 10, 11],
        (False, "unpaired"): [0, 1, 2, 3, 9, 10, 11],
        (False, "pair_aware"): [0, 1, 2, 3, 9],
        (True, "none"): [0, 1, 2, 3, 4, 9, 10, 11],
        (True, "unpaired"): [0, 3, 9, 10, 11],
        (True, "pair_aware"): [0, 3, 9],
    }[zero_as_missing, method]
    trait0 = np.array([1, 1, 0, 2 / 3, 1, 1, 0, 0, 0, 0, 1 / 3, 2 / 3])
    trait1 = np.array([1, 0, 1, 2 / 3, 0, 1, 1 / 3, 0, 0, 2 / 3, 1 / 3, 2 / 3])
    fractions = {
        "all_samples": (trait0 + trait1) / 2,
        "any_trait": np.maximum(trait0, trait1),
        "trait_0": trait0,
        "trait_1": trait1,
    }[scope]
    expected = [index for index in eligible if fractions[index] >= min_fraction]
    score_rows: list[dict[str, Any]] = []

    selected, counts = cv_mod._select_feature_indices_with_counts(
        config,
        values,
        names,
        y_train=np.array([0, 1, 0, 1, 0, 1]),
        groups_train=np.array(["a", "a", "b", "b", "c", "c"]),
        ranked_feature_score_rows=score_rows,
    )

    assert selected.tolist() == expected
    assert counts.n_features_before == len(names)
    assert counts.n_features_after_sparse_feature_filter == len(expected)
    assert counts.n_features_after_low_variance == len(expected)
    assert counts.n_features_after_ranked_feature_filter == len(expected)
    assert counts.n_features_after_correlation == len(expected)
    assert counts.n_features_after_all == len(expected)
    assert [row["feature"] for row in score_rows] == [names[index] for index in expected]
    np.testing.assert_array_equal(values, original)


def test_early_sparse_screen_reduces_neutral_work_and_preserves_each_stage_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        neutral=True,
        sparse_feature_filter={"scope": "all_samples", "min_nonzero_fraction": 1.0},
        low_variance_filter={"enabled": True, "min_variance": 1.0},
        ranked_feature_filter={"method": "variance", "max_features": 2},
    )
    values = np.array(
        [[2] * 6, [1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 5, 6],
         [1, 3, 5, 7, 9, 11], [1, 4, 7, 10, 13, 16], [0, 0, 0, 0, 0, 2]],
        dtype=float,
    ).T
    original_variance = cv_mod._column_nan_variance
    variance_widths: list[int] = []

    def tracked_variance(matrix: np.ndarray, *, ddof: int) -> np.ndarray:
        variance_widths.append(matrix.shape[1])
        return original_variance(matrix, ddof=ddof)

    monkeypatch.setattr(cv_mod, "_column_nan_variance", tracked_variance)
    rows: list[dict[str, Any]] = []
    selected, counts = cv_mod._select_feature_indices_with_counts(
        config, values, ["constant", "small", "z", "y", "x", "sparse"],
        ranked_feature_score_rows=rows,
    )
    assert variance_widths[0] == 5  # Sparse column never reaches neutral variance.
    assert counts.n_features_before == 6
    assert counts.n_features_after_sparse_feature_filter == 4
    assert counts.n_features_after_low_variance == 3
    assert counts.n_features_after_ranked_feature_filter == 2
    assert counts.n_features_after_correlation == counts.n_features_after_all == 2
    assert selected.tolist() == [3, 4]
    assert [row["feature"] for row in rows] == ["z", "y", "x"]
    assert [row["rank"] for row in rows] == [3, 2, 1]


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("sparse_enabled", [False, True])
@pytest.mark.parametrize("min_fraction", [0.0, 1.0])
@pytest.mark.parametrize("one_survivor", [False, True])
@pytest.mark.parametrize("neutral_block_cells", [81, 1_000_000])
def test_neutral_variance_keeps_original_reduction_at_roundoff_boundary(
    order: str,
    sparse_enabled: bool,
    min_fraction: float,
    one_survivor: bool,
    neutral_block_cells: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv_mod, "_NEUTRAL_FILTER_BLOCK_CELLS", neutral_block_cells)
    values = np.zeros((81, 3), order=order)
    values[:, 0] = 0.1
    values[:, 2] = np.arange(81) + 1
    if one_survivor:
        values[1:, 2] = 0
    # Summing the constant column along contiguous versus strided axes can
    # produce a tiny positive variance. Preserve the original full-width result.
    variance_before_narrowing = cv_mod._column_nan_variance(values, ddof=0)
    eligible = variance_before_narrowing > 0
    if sparse_enabled:
        eligible &= np.count_nonzero(values > cv_mod._NONZERO_TOLERANCE, axis=0) / 81 >= (
            min_fraction
        )
    expected = np.flatnonzero(eligible)
    config = _config(
        neutral=True,
        sparse_feature_filter={
            "enabled": sparse_enabled,
            "scope": "all_samples",
            "min_nonzero_fraction": min_fraction,
        },
    )
    if not expected.size:
        with pytest.raises(cv_mod.CVError, match="removed all features"):
            cv_mod._select_feature_indices(config, values, ["constant", "zero", "varying"])
    else:
        selected = cv_mod._select_feature_indices(
            config, values, ["constant", "zero", "varying"]
        )
        np.testing.assert_array_equal(selected, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_neutral_statistics_bound_workspace_when_all_features_survive(
    order: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv_mod, "_NEUTRAL_FILTER_BLOCK_CELLS", 32)
    config = _config(
        neutral=True,
        sparse_feature_filter={"scope": "all_samples", "min_nonzero_fraction": 1.0},
    )
    values = np.array(np.arange(8)[:, None] + np.arange(20)[None, :] + 1, dtype=float, order=order)
    original = values.copy()
    names = [f"feature_{index}" for index in range(20)]
    original_variance = cv_mod._column_nan_variance
    variance_widths: list[int] = []

    def tracked_variance(matrix: np.ndarray, *, ddof: int) -> np.ndarray:
        variance_widths.append(matrix.shape[1])
        return original_variance(matrix, ddof=ddof)

    monkeypatch.setattr(cv_mod, "_column_nan_variance", tracked_variance)
    selected, counts = cv_mod._select_feature_indices_with_counts(config, values, names)

    assert len(variance_widths) > 1
    assert max(variance_widths) <= 4
    np.testing.assert_array_equal(selected, np.arange(20))
    assert counts.n_features_before == 20
    assert counts.n_features_after_sparse_feature_filter == 20
    assert counts.n_features_after_low_variance == 20
    assert counts.n_features_after_ranked_feature_filter == 20
    assert counts.n_features_after_correlation == counts.n_features_after_all == 20
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("method", ["sample_rank", "sample_percentile_rank"])
def test_sparse_screen_uses_full_sample_transform_and_only_training_rows(method: str) -> None:
    config = _config(
        expression_transform={"method": method},
        feature_scaling={"method": "none"},
        sparse_feature_filter={"scope": "all_samples", "min_nonzero_fraction": 1.0},
    )
    train = np.array([[4, 0, 1], [4, 100, 0]], dtype=float)
    target = np.array([[4, 1, 100], [4, 0, 0]], dtype=float)
    train_before = train.copy()
    expected_train = [2, 1] if method == "sample_rank" else [1, 0.5]
    expected_target = [2, 1] if method == "sample_rank" else [2 / 3, 1]

    fitted, predicted, names, _, counts = cv_mod._preprocess_train_and_target_with_counts(
        config, train, target, ["kept", "training_sparse_a", "training_sparse_b"]
    )
    assert names == ["kept"]
    assert counts.n_features_before == 3
    assert counts.n_features_after_sparse_feature_filter == 1
    np.testing.assert_array_equal(fitted[:, 0], expected_train)
    np.testing.assert_array_equal(predicted[:, 0], expected_target)
    np.testing.assert_array_equal(train, train_before)
    other_fit, _, other_names, _, other_counts = cv_mod._preprocess_train_and_target_with_counts(
        config, train, np.full_like(target, 1000), ["kept", "a", "b"]
    )
    np.testing.assert_array_equal(other_fit, fitted)
    assert other_names == names
    assert other_counts == counts


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("rows", [[3, 1, 1, 0], [True, False, True, False], []])
def test_direct_feature_indexing_preserves_values_order_and_layout(
    layout: str, rows: list[int] | list[bool]
) -> None:
    values = np.arange(40, dtype=float).reshape(4, 10)
    values[1, 3] = np.nan
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = values[:, ::2]
    original = values.copy()
    row_indices = np.asarray(rows, dtype=bool if rows and isinstance(rows[0], bool) else int)
    columns = np.array([4, 1, 1, 0])
    expected = values[row_indices][:, columns]
    actual = cv_mod._take_feature_rows(values, row_indices, columns)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    assert not np.shares_memory(actual, values)
    actual[:] = -1
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize(
    "case",
    ["none", "insufficient_pairs", "zero_scores", "variance", "unpaired", "pair_aware", "fallback"],
)
def test_skipping_diagnostic_rows_preserves_ranking_priorities_and_warnings(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = {
        "insufficient_pairs": "pair_aware",
        "zero_scores": "variance",
        "fallback": "pair_aware",
    }.get(case, case)
    config = _config(
        ranked_feature_filter={
            "method": method,
            "max_features": 2,
            "min_contrast_pairs": 4 if case == "insufficient_pairs" else 1,
        }
    )
    values = np.array([[1, 3, 5], [5, 1, 2], [2, 4, 8], [8, 2, 3], [3, 6, 4], [9, 3, 1]])
    labels = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    if case == "zero_scores":
        values = np.ones_like(values)
    if case == "fallback":
        values, labels, groups = values[:2], labels[:2], groups[:2]
    selected = np.array([2, 0, 1])
    warnings_with: list[str] = []
    expected = cv_mod._apply_ranked_feature_filter(
        config, values, selected, ["z", "a", "m"], labels, groups, warnings_with
    )
    assert expected.score_rows

    def unexpected_diagnostics(**_kwargs: Any) -> list[dict[str, Any]]:
        pytest.fail("Discarded diagnostic dictionaries must not be constructed")

    monkeypatch.setattr(cv_mod, "_ranked_score_rows", unexpected_diagnostics)
    warnings_without: list[str] = []
    actual = cv_mod._apply_ranked_feature_filter(
        config,
        values,
        selected,
        ["z", "a", "m"],
        labels,
        groups,
        warnings_without,
        collect_score_rows=False,
    )
    np.testing.assert_array_equal(actual.selected, expected.selected)
    if expected.priority_scores is None:
        assert actual.priority_scores is None
    else:
        np.testing.assert_array_equal(actual.priority_scores, expected.priority_scores)
    assert actual.score_rows == []
    assert warnings_without == warnings_with
    if case in {"insufficient_pairs", "zero_scores", "fallback"}:
        assert warnings_without


def test_feature_selection_skips_diagnostics_only_without_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(ranked_feature_filter={"method": "variance", "max_features": 1})
    values = np.array([[1, 3], [4, 4], [8, 5]], dtype=float)
    rows: list[dict[str, Any]] = []
    expected, expected_counts = cv_mod._select_feature_indices_with_counts(
        config, values, ["a", "b"], ranked_feature_score_rows=rows
    )
    assert len(rows) == 2

    def unexpected_diagnostics(**_kwargs: Any) -> list[dict[str, Any]]:
        pytest.fail("No diagnostic collector was supplied")

    monkeypatch.setattr(cv_mod, "_ranked_score_rows", unexpected_diagnostics)
    actual, counts = cv_mod._select_feature_indices_with_counts(config, values, ["a", "b"])
    np.testing.assert_array_equal(actual, expected)
    assert counts == expected_counts
