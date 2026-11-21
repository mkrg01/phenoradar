import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit

from phenoradar import glmnet
from phenoradar.amino_acid_packed import PackedAAMatrix, PackedColumns, packed_arithmetic


@pytest.mark.parametrize("alpha", [.6, 1.])
def test_wide_packed_route_avoids_full_csc_and_preserves_full_kkt(monkeypatch, alpha):
    fused = packed_arithmetic()
    if fused is None:
        pytest.skip('Native packing unavailable')
    rng = np.random.default_rng(2026)
    matrix = sparse.csc_matrix(rng.choice([0., 0., .25, .5, 1.], (150, 180)))
    packed = PackedAAMatrix(150, [PackedColumns.from_csc(matrix)], fused=fused)
    y = rng.binomial(1, expit(matrix[:, :3] @ [2., -1., 1.]))
    weights = rng.uniform(.5, 2., len(y))
    factors = rng.uniform(.8, 1.2, matrix.shape[1])
    options = dict(alpha=alpha, sample_weight=weights, penalty_factor=factors,
                   thresh=1e-13, compact_coefficients=True, compute_kkt=False)
    expected = glmnet.fit_glmnet_path(matrix, y, [.04, .02], **options)
    monkeypatch.setattr(glmnet, '_PACKED_WORKING_SET_CSC_BYTES', 0)
    monkeypatch.setattr(PackedAAMatrix, 'to_csc', lambda self: pytest.fail('Expanded full matrix'))
    actual = glmnet.fit_glmnet_path(packed, y, [.04, .02], **options)
    for model, reference in zip(actual, expected, strict=True):
        assert model.used_working_set_
        np.testing.assert_array_equal(model.coef_ != 0, reference.coef_ != 0)
        np.testing.assert_allclose(model.predict_proba(matrix), reference.predict_proba(matrix),
                                   atol=3e-5, rtol=0)
        beta = model.coef_
        residual = weights / weights.sum() * (expit(matrix @ beta + model.intercept_) - y)
        gradient = matrix.T @ residual + model.lambda_ * (1 - alpha) * factors * beta
        error = np.maximum(np.abs(gradient) - model.lambda_ * alpha * factors, 0)
        selected = beta != 0
        error[selected] = np.abs(
            gradient[selected] + model.lambda_ * alpha * factors[selected] * np.sign(beta[selected])
        )
        assert max(abs(residual.sum()), error.max()) <= 1.0001e-7


@pytest.mark.parametrize('alpha,path', [(0., [.04, .02]), (1., [.04, 0.])])
def test_packed_ridge_or_zero_lambda_keeps_native_supported_path(monkeypatch, alpha, path):
    fused = packed_arithmetic()
    if fused is None:
        pytest.skip('Native packing unavailable')
    x = sparse.csc_matrix(np.tile([[0., 1.], [1., 0.]], (20, 1)))
    p = PackedAAMatrix(40, [PackedColumns.from_csc(x)], fused=fused)
    monkeypatch.setattr(glmnet, '_PACKED_WORKING_SET_CSC_BYTES', 0)
    models = glmnet.fit_glmnet_path(p, np.tile([0, 1], 20), path, alpha=alpha,
                                   compact_coefficients=True, compute_kkt=False)
    assert all(not getattr(model, 'used_working_set_', False) for model in models)
