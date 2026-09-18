from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import phenoradar.abstention as abstention


def _legacy_weights(names: list[str], models: list[tuple[list[str], np.ndarray]]) -> np.ndarray:
    weights = np.zeros(len(names))
    for model_names, coefficients in models:
        magnitude = np.abs(np.asarray(coefficients, dtype=float).ravel())
        maximum = magnitude.max() if magnitude.size else 0.0
        if maximum:
            normalized = magnitude / maximum
            normalized /= normalized.sum()
            weights[[names.index(name) for name in model_names]] += normalized / len(models)
    return weights


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("zero_as_missing", [False, True])
@pytest.mark.parametrize("top_features", [0, 3, 100, -1])
@pytest.mark.parametrize("dense_support", [False, True])
def test_blocked_active_coverage_matches_legacy_dense_calculation(
    layout: str,
    zero_as_missing: bool,
    top_features: int,
    dense_support: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(abstention, "_ABSTENTION_BLOCK_CELLS", 128)
    rng = np.random.default_rng(209)
    values = rng.lognormal(size=(23, 74))
    values[rng.random(values.shape) < 0.3] = np.nan
    values[rng.random(values.shape) < 0.4] = 0
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = values[::-1, ::2]
    before = values.copy()
    values.flags.writeable = False
    names = [f"feature_{i:03d}" for i in reversed(range(values.shape[1]))]
    indices = list(range(len(names))) if dense_support else [1, 5, 7, len(names) - 1]
    coef = np.arange(1, len(indices) + 1, dtype=float)
    coef[::3] *= -1
    models = [
        ([names[i] for i in indices], coef),
        ([names[i] for i in reversed(indices)], -coef),
        ([names[0]], np.array([0.0])),  # Still contributes to the model-count denominator.
    ]
    weights = _legacy_weights(names, models)
    observed = np.isfinite(values) & ((values != 0) if zero_as_missing else True)
    expected_coverage = np.clip(observed @ weights, 0.0, 1.0)
    threshold = 0.4
    expected_accepted = (expected_coverage >= threshold) | np.isclose(
        expected_coverage, threshold, rtol=0, atol=1e-12
    )
    order = sorted(np.flatnonzero(weights > 0).tolist(), key=lambda j: (-weights[j], names[j]))
    expected_details = [
        json.dumps(
            [
                {"feature": names[j], "coefficient_fraction": float(weights[j])}
                for j in order
                if not observed[row, j]
            ][:top_features]
            if not expected_accepted[row]
            else [],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in range(len(values))
    ]
    species = [f"species_{i:03d}" for i in range(len(values))]
    result = abstention.annotate_abstention(
        pl.DataFrame({"species": species[::-1], "prob": [0.9] * len(species)}),
        species=species,
        matrix=values,
        feature_names=names,
        model_coefficients=models,
        zero_as_missing=zero_as_missing,
        threshold=threshold,
        top_features=top_features,
    )
    assert result["species"].to_list() == species[::-1]
    result = result.sort("species")
    np.testing.assert_allclose(
        result["information_coverage"], expected_coverage, rtol=0, atol=1e-15
    )
    assert result["decision_status"].to_list() == [
        "accepted" if accepted else "abstained" for accepted in expected_accepted
    ]
    assert result["pred_label_selective"].to_list() == [
        1 if accepted else None for accepted in expected_accepted
    ]
    assert result["missing_features_json"].to_list() == expected_details
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("offset", [-2e-12, -1e-12, 0, 1e-12, 2e-12])
def test_coverage_threshold_keeps_inclusive_tolerance(offset: float) -> None:
    names = [f"f{i}" for i in range(97)]
    values = np.ones((1, 97))
    values[0, 2] = np.nan
    weights = np.zeros(97)
    weights[[2, 89]] = [0.2, 0.8]
    expected_coverage = (np.isfinite(values) @ weights)[0]
    threshold = 0.8 + offset
    expected = expected_coverage >= threshold or np.isclose(
        expected_coverage, threshold, rtol=0, atol=1e-12
    )
    result = abstention.annotate_abstention(
        pl.DataFrame({"species": ["s"], "prob": [0.2]}),
        species=["s"],
        matrix=values,
        feature_names=names,
        model_coefficients=[([names[2], names[89]], np.array([1.0, -4.0]))],
        zero_as_missing=True,
        threshold=threshold,
    )
    assert result["information_coverage"].item() == expected_coverage
    assert result["decision_status"].item() == ("accepted" if expected else "abstained")


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_, str])
def test_input_numeric_conversion_and_large_coefficients(dtype: type) -> None:
    values = np.array([[1, 0, 1], [0, 1, 1]], dtype=dtype)
    result = abstention.annotate_abstention(
        pl.DataFrame({"species": ["a", "b"], "prob": [0.9, 0.1]}),
        species=["a", "b"],
        matrix=values,
        feature_names=["a", "b", "unused"],
        model_coefficients=[(["b", "a"], np.array([-1e308, 1e308]))],
        zero_as_missing=True,
        threshold=0.8,
    )
    assert result["information_coverage"].to_list() == [0.5, 0.5]
    assert result["decision_status"].to_list() == ["abstained", "abstained"]
    assert [json.loads(s)[0]["feature"] for s in result["missing_features_json"]] == ["b", "a"]


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("bad", [-1.0, np.inf, -np.inf])
def test_invalid_unused_columns_are_still_rejected(
    layout: str, bad: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(abstention, "_ABSTENTION_BLOCK_CELLS", 8)
    values = np.ones((3, 12), order=layout)
    values[-1, -1] = bad
    with pytest.raises(ValueError, match="nonnegative raw TPM"):
        abstention.annotate_abstention(
            pl.DataFrame({"species": ["a", "b", "c"], "prob": [0.9] * 3}),
            species=["a", "b", "c"],
            matrix=values,
            feature_names=[f"f{i}" for i in range(12)],
            model_coefficients=[(["f0"], np.array([1.0]))],
            zero_as_missing=True,
            threshold=0.8,
        )


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3, 3)])
def test_empty_targets_features_and_zero_coefficients(shape: tuple[int, int]) -> None:
    species = [f"s{i}" for i in range(shape[0])]
    names = [f"f{i}" for i in range(shape[1])]
    result = abstention.annotate_abstention(
        pl.DataFrame(
            {"species": pl.Series(species, dtype=pl.String), "prob": [0.9] * len(species)}
        ),
        species=species,
        matrix=np.ones(shape),
        feature_names=names,
        model_coefficients=[(names, np.zeros(shape[1]))],
        zero_as_missing=True,
        threshold=0.8,
    )
    assert result["information_coverage"].to_list() == [0.0] * len(species)
    assert result["abstention_reason"].to_list() == ["no_informative_coefficients"] * len(species)
    assert result["missing_features_json"].to_list() == ["[]"] * len(species)


@pytest.mark.parametrize("layout", ["C", "F"])
def test_large_masks_are_bounded_and_observation_mask_uses_only_active_columns(
    layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(abstention, "_ABSTENTION_BLOCK_CELLS", 128)
    values = np.ones((23, 74), order=layout)
    names = [f"f{i}" for i in range(74)]
    finite_shapes: list[tuple[int, ...]] = []
    infinite_shapes: list[tuple[int, ...]] = []
    isfinite, isinf = np.isfinite, np.isinf

    def finite(array: np.ndarray) -> np.ndarray:
        if isinstance(array, np.ndarray) and array.ndim == 2:
            finite_shapes.append(array.shape)
        return isfinite(array)

    def infinite(array: np.ndarray) -> np.ndarray:
        if isinstance(array, np.ndarray) and array.ndim == 2:
            infinite_shapes.append(array.shape)
        return isinf(array)

    monkeypatch.setattr(abstention.np, "isfinite", finite)
    monkeypatch.setattr(abstention.np, "isinf", infinite)
    result = abstention.annotate_abstention(
        pl.DataFrame({"species": [f"s{i}" for i in range(23)], "prob": [0.9] * 23}),
        species=[f"s{i}" for i in range(23)],
        matrix=values,
        feature_names=names,
        model_coefficients=[(names[:12], np.ones(12))],
        zero_as_missing=True,
        threshold=0.8,
    )
    assert result["decision_status"].to_list() == ["accepted"] * 23
    assert len(finite_shapes) > 1 and len(infinite_shapes) > 1
    assert all(shape[1] == 12 and np.prod(shape) <= 128 for shape in finite_shapes)
    assert all(np.prod(shape) <= 128 for shape in infinite_shapes)
