from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from phenoradar.abstention import annotate_abstention
from phenoradar.config import AppConfig
from phenoradar.cv import CVError, RunExpressionCache
from phenoradar.expression_matrix_cache import RawExpressionMatrixCache


def test_raw_cache_preserves_row_column_order_and_nan() -> None:
    cache = RawExpressionMatrixCache(1024)
    values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    cache.put(["a", "b"], ["x", "y", "z"], values, complete_schema=True)
    hit = cache.get(["b", "a", "b"], ["z", "y"])
    assert hit is not None
    actual, features = hit
    np.testing.assert_array_equal(actual, np.array([[6.0, 5.0], [3.0, np.nan], [6.0, 5.0]]))
    assert features == ["z", "y"]
    assert not actual.flags.writeable and not values.flags.writeable
    # Without an explicit feature order, a species subset can have a different schema.
    assert cache.get(["a"], None) is None
    assert cache.get(["a", "b"], None) is not None


def test_raw_cache_is_bounded_and_does_not_invent_full_schemas() -> None:
    cache = RawExpressionMatrixCache(48)
    cache.put(["a", "b"], ["x", "y", "z"], np.ones((2, 3)), complete_schema=True)
    cache.put(["c"], ["x", "y"], np.zeros((1, 2)), complete_schema=False)
    assert cache.nbytes <= 48
    assert cache.get(["a", "b"], ["x"]) is None
    assert cache.get(["c"], None) is None
    assert cache.get(["c"], ["x"]) is not None
    cache.put(["large"], [str(i) for i in range(7)], np.zeros((1, 7)), complete_schema=True)
    assert cache.nbytes <= 48
    cache.clear()
    assert cache.nbytes == 0


def test_overlap_returns_only_complete_rows_in_requested_order() -> None:
    cache = RawExpressionMatrixCache(1024)
    cache.put(["a", "b"], ["x", "y"], np.array([[1.0, 2.0], [3.0, 4.0]]), complete_schema=False)
    overlap = cache.overlapping_rows(["new", "b", "a", "b"], ["y"])
    assert overlap is not None
    rows, values = overlap
    assert rows == [1, 2, 3]
    np.testing.assert_array_equal(values, np.array([[4.0], [2.0], [4.0]]))
    assert cache.overlapping_rows(["a"], ["unavailable"]) is None


def test_run_matrix_cache_rejects_input_changes_and_releases_memory(tmp_path: Path) -> None:
    path = tmp_path / "expression.tsv"
    path.write_text("species\torthogroup\ttpm\na\tx\t1\n")
    config = AppConfig.model_validate({"data": {"metadata_path": "unused", "tpm_path": str(path)}})
    with RunExpressionCache() as owner:
        builder = owner.get_builder(config)
        builder.build_matrix(["a"])
        assert builder._matrix_cache is not None and builder._matrix_cache.nbytes > 0
        path.write_text("species\torthogroup\ttpm\na\tx\t22\n")
        with pytest.raises(CVError, match="changed while its matrices were in use"):
            builder.build_matrix(["a"])
        assert builder._matrix_cache.nbytes == 0
    assert builder._matrix_cache.nbytes == 0


@pytest.mark.parametrize("intercept_only", [False, True])
def test_projected_abstention_keeps_original_coefficient_normalization(
    intercept_only: bool,
) -> None:
    names = ["zero", "tiny", "negative", "unused", "positive"]
    coefficients = np.zeros(5) if intercept_only else np.array([0.0, 1e-200, -2.3, 0.0, 3.2])
    matrix = np.array([[1.0, 0.0, 4.0, np.nan, 5.0], [2.0, np.nan, 0.0, 3.0, np.nan]])
    predictions = pl.DataFrame({"species": ["a", "b"], "prob": [0.9, 0.7]})
    options = dict(
        predictions=predictions,
        species=["a", "b"],
        model_coefficients=[(names, coefficients)],
        zero_as_missing=True,
        threshold=0.8,
    )
    full = annotate_abstention(**options, matrix=matrix, feature_names=names)
    projected = annotate_abstention(
        **options,
        matrix=matrix[:, [1, 2, 4]],
        feature_names=[names[i] for i in [1, 2, 4]],
        allow_missing_zero_coefficients=True,
    )
    assert_frame_equal(full, projected, check_exact=True)
    if not intercept_only:
        with pytest.raises(ValueError, match="missing a nonzero-coefficient feature"):
            annotate_abstention(
                **options,
                matrix=matrix[:, [2, 4]],
                feature_names=[names[i] for i in [2, 4]],
                allow_missing_zero_coefficients=True,
            )
