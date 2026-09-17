from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import phenoradar.local_evidence as evidence_mod
from phenoradar.cv import apply_feature_scaling
from phenoradar.local_evidence import iter_top_contributions
from phenoradar.missing_expression import NeutralStandardScaler


@pytest.mark.parametrize("model_count", [1, 2, 9])
@pytest.mark.parametrize("scaling", ["none", "standard", "neutral"])
def test_blocked_contributions_match_cube(
    monkeypatch: pytest.MonkeyPatch, model_count: int, scaling: str
) -> None:
    rng = np.random.default_rng(23)
    names = [f"f{i:04d}" for i in reversed(range(137))]
    matrix = rng.normal(size=(11, len(names)))
    if scaling == "neutral":
        matrix[::3, ::5] = np.nan
    rows = np.array([9, 1, 3, 1, 10, 0])
    schemas, scalers, coefs = [], [], []
    cube = np.zeros((model_count, len(rows), len(names)))
    for model in range(model_count):
        columns = np.array([i for i in range(len(names)) if (i + model) % 3 != 0])
        values = matrix[:, columns]
        scaler = (
            None
            if scaling == "none"
            else (NeutralStandardScaler() if scaling == "neutral" else StandardScaler()).fit(values)
        )
        coef = rng.normal(size=len(columns))
        coef[::7] = 0.0
        schemas.append([names[i] for i in columns])
        scalers.append(scaler)
        coefs.append(coef)
        cube[model][:, columns] = (
            apply_feature_scaling(values[rows], scaler, "none" if scaling == "none" else "standard")
            * coef
        )
    means = cube.mean(axis=0)
    magnitudes = np.abs(cube).mean(axis=0)
    minima, maxima = cube.min(axis=0), cube.max(axis=0)
    monkeypatch.setattr(evidence_mod, "_CONTRIBUTION_BLOCK_CELLS", 200)
    actual = list(
        iter_top_contributions(
            matrix,
            names,
            rows=rows,
            model_features=schemas,
            scalers=scalers,
            coefficients=coefs,
            scaling_method="none" if scaling == "none" else "standard",
            top_features=13,
        )
    )
    assert [row for row, _ in actual] == list(range(len(rows)))
    for row, contributions in actual:
        expected = sorted(range(len(names)), key=lambda i: (-magnitudes[row, i], names[i]))
        expected = [i for i in expected if magnitudes[row, i] > 1e-12][:13]
        assert [item.feature for item in contributions] == [names[i] for i in expected]
        for item, i in zip(contributions, expected, strict=True):
            np.testing.assert_allclose(
                [item.mean, item.mean_abs, item.minimum, item.maximum],
                [means[row, i], magnitudes[row, i], minima[row, i], maxima[row, i]],
                rtol=1e-12,
                atol=1e-14,
            )


def test_contribution_ties_and_missing_model_features() -> None:
    names = ["z", "b", "a", "c", "zero"]
    result = list(
        iter_top_contributions(
            np.array([[2.0, 1.0, -1.0, 1.0, 0.0]]),
            names,
            rows=np.array([0]),
            model_features=[names, ["z"]],
            scalers=[None, None],
            coefficients=[np.ones(5), np.ones(1)],
            scaling_method="none",
            top_features=3,
        )
    )[0][1]
    assert [item.feature for item in result] == ["z", "a", "b"]
    assert result[0].minimum == result[0].maximum == 2.0
    assert result[1].minimum == -1.0 and result[1].maximum == 0.0
    assert result[2].minimum == 0.0 and result[2].maximum == 1.0


def test_contribution_workspaces_do_not_scale_with_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence_mod, "_CONTRIBUTION_BLOCK_CELLS", 128)
    names = [str(i) for i in range(64)]
    allocated: list[tuple[int, ...]] = []
    original = np.zeros

    def zeros(shape: tuple[int, ...], *args: object, **kwargs: object) -> np.ndarray:
        allocated.append(shape)
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(evidence_mod.np, "zeros", zeros)
    result = list(
        iter_top_contributions(
            np.ones((100, 64)),
            names,
            rows=np.arange(100),
            model_features=[names] * 4,
            scalers=[None] * 4,
            coefficients=[np.ones(64)] * 4,
            scaling_method="none",
            top_features=2,
        )
    )
    assert len(result) == 100
    assert allocated and all(len(shape) == 2 and np.prod(shape) <= 128 for shape in allocated)
