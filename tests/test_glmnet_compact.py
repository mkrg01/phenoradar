import numpy as np
import pytest
from scipy import sparse

import phenoradar.glmnet as glmnet


@pytest.mark.parametrize("alpha", [0, .4, 1])
@pytest.mark.parametrize("sparse_input", [True, False])
def test_compact_paths_preserve_native_solutions_without_dense_expansion(monkeypatch, alpha,
                                                                         sparse_input):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(80, 35))
    x[rng.random(x.shape) < .7] = 0
    y = np.tile([0, 1], 40)
    if sparse_input:
        x = sparse.csc_matrix(x)
    options = dict(alpha=alpha, compute_kkt=False, penalty_factor=np.r_[np.ones(10),
                                                                    np.full(25, .3)])
    reference = glmnet.fit_glmnet_path(x, y, [.1, .03, .01], **options)
    def unexpected(*args):
        pytest.fail("Compact CV path expanded all feature coefficients")
    monkeypatch.setattr(glmnet, "lsolns", unexpected)
    compact = glmnet.fit_glmnet_path(x, y, [.1, .03, .01], compact_coefficients=True, **options)
    expected = glmnet.predict_glmnet_path(reference, x)
    actual = glmnet.predict_glmnet_path(compact, x)
    for a, b, pa, pb in zip(reference, compact, expected, actual, strict=True):
        np.testing.assert_array_equal(a.coef_, b.coef_)
        assert a.intercept_ == b.intercept_ and a.n_iter_ == b.n_iter_
        np.testing.assert_allclose(pa, pb, atol=3e-16, rtol=0)
        assert "coef_" not in b.__dict__


def test_compact_constant_design_has_no_nonzero_coefficients():
    models = glmnet.fit_glmnet_path(sparse.csc_matrix(np.ones((12, 5))), np.tile([0, 1], 6),
                                    [.1, .01], compact_coefficients=True, compute_kkt=False)
    for model in models:
        assert not len(model.coefficient_indices_)
        np.testing.assert_array_equal(model.predict_proba(np.ones((3, 5)))[:, 1], [.5]*3)


def test_wide_compact_prediction_matches_existing_zero_column_projection():
    rng = np.random.default_rng(31)
    n, p = 40, 100_005
    rows = np.repeat(np.arange(n), 3)
    columns = np.tile([9, 123, p - 1], n)
    x = sparse.csc_matrix((rng.normal(size=3*n), (rows, columns)), shape=(n, p))
    y = np.tile([0, 1], n//2)
    factors = np.ones(p)
    options = dict(alpha=1, compute_kkt=False, penalty_factor=factors)
    before = glmnet.fit_glmnet_path(x, y, [.1, .01], **options)
    after = glmnet.fit_glmnet_path(x, y, [.1, .01], compact_coefficients=True, **options)
    assert after[0].penalty_factor is after[1].penalty_factor
    assert not after[0].penalty_factor.flags.writeable
    assert factors.flags.writeable
    for a, b in zip(glmnet.predict_glmnet_path(before, x),
                    glmnet.predict_glmnet_path(after, x), strict=True):
        np.testing.assert_array_equal(a, b)
