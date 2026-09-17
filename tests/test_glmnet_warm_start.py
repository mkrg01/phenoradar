"""Native warm starts preserve exact candidates and their weighted objective."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit

import phenoradar.glmnet as glmnet


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(731)
    x = rng.normal(size=(80, 8))
    y = np.tile([0, 1], 40)
    x[:, 0] += y
    return x, y, rng.uniform(0.1, 4.0, size=len(y))


@pytest.mark.parametrize("sparse_input", [False, True])
def test_coarse_candidates_use_close_native_warm_starts_only(
    sparse_input: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    x, y, weights = _problem()
    matrix = sparse.csc_matrix(x) if sparse_input else x
    native_name = "splognet" if sparse_input else "lognet"
    native = getattr(glmnet, native_name)
    calls = []

    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls.append((np.array(args[-2]), kwargs["nlam"]))
        return native(*args, **kwargs)

    monkeypatch.setattr(glmnet, native_name, tracked)
    targets = [10.0 ** (-1 - 0.5 * i) for i in range(9)]
    requested = targets[::-1] + [targets[3]]
    models = glmnet.fit_glmnet_path(matrix, y, requested, alpha=0.7, sample_weight=weights)

    assert len(calls) == 1
    path, count = calls[0]
    assert count == len(path) == 81
    assert np.all(np.diff(path) < 0)
    assert np.max(np.diff(np.log10(path[::-1]))) <= 0.05 + 1e-12
    assert all(np.count_nonzero(path == target) == 1 for target in targets)
    assert [m.lambda_ for m in models] == requested
    assert len({id(m) for m in models}) == len(requested)
    assert all(m.path_length_ == 9 and m.native_path_length_ == 81 for m in models)
    # Auxiliary points must not alter the weighted objective or returned targets.
    for model in models:
        residual = weights * (expit(x @ model.coef_ + model.intercept_) - y) / weights.sum()
        gradient = x.T @ residual + model.lambda_ * 0.3 * model.coef_
        active = model.coef_ != 0
        np.testing.assert_allclose(
            gradient[active] + model.lambda_ * 0.7 * np.sign(model.coef_[active]),
            0,
            atol=2e-7,
        )
        assert np.all(np.abs(gradient[~active]) <= model.lambda_ * 0.7 + 2e-7)
        assert abs(residual.sum()) < 2e-7
    assert not np.shares_memory(models[5].coef_, models[-1].coef_)


@pytest.mark.parametrize("targets", [[0.0], [0.017], [0.017, 0.0], [0.02, 0.019]])
def test_single_zero_and_already_close_targets_are_preserved(
    targets: list[float],
) -> None:
    np.testing.assert_array_equal(glmnet._native_lambda_path(np.array(targets)), targets)


def test_extreme_positive_targets_remain_finite_unique_and_exact() -> None:
    targets = np.array(
        [np.finfo(float).max, 1.0, np.finfo(float).tiny, np.nextafter(0.0, 1.0), 0.0]
    )
    path = glmnet._native_lambda_path(targets)
    assert np.isfinite(path).all()
    assert np.all(np.diff(path) < 0)
    assert path[-1] == 0
    assert all(np.count_nonzero(path == value) == 1 for value in targets)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_unregularized_endpoint_is_fitted_exactly_after_internal_points(
    sparse_input: bool,
) -> None:
    x, y, weights = _problem()
    matrix = sparse.csc_matrix(x) if sparse_input else x
    models = glmnet.fit_glmnet_path(matrix, y, [0.01, 0, 0.1], sample_weight=weights)
    reference = glmnet.fit_glmnet_path(matrix, y, [0], sample_weight=weights)[0]
    assert [m.lambda_ for m in models] == [0.01, 0, 0.1]
    assert models[1].path_length_ == 3
    assert models[1].native_path_length_ == 22
    np.testing.assert_allclose(models[1].predict_proba(x), reference.predict_proba(x), atol=2e-7)


def test_native_iteration_limit_still_rejects_auxiliary_path_failure() -> None:
    x, y, weights = _problem()
    with pytest.raises(glmnet.GlmnetError, match="including warm-start points.*Increase maxit"):
        glmnet.fit_glmnet_path(x, y, [0.1, 0.001], maxit=1, sample_weight=weights)
