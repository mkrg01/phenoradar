from types import SimpleNamespace

import joblib
import numpy as np
import polars as pl
import pytest
from scipy import sparse
from scipy.special import expit
from test_combined import combined_config

from phenoradar import amino_acid_sparse, glmnet, glmnet_working_set
from phenoradar.amino_acid_run import run_amino_acid
from phenoradar.config import load_and_resolve_config


def test_intercept_only_path_scans_once(monkeypatch):
    x = sparse.csc_matrix([[0., 2.], [1., 0.], [0., 1.], [1., 2.]])
    y = np.array([0., 1., 0., 1.])
    weights = np.full(len(y), .25)
    scan = glmnet_working_set._scan_gradients
    residuals = []

    def record_scan(matrix, residual, gradients, ranges):
        residuals.append(residual.copy())
        return scan(matrix, residual, gradients, ranges)

    monkeypatch.setattr(glmnet_working_set, "_scan_gradients", record_scan)
    solutions = list(glmnet_working_set.solve_working_set(
        x, y, weights, np.ones(x.shape[1]), np.array([4., 3., 2.]),
        alpha=1, thresh=1e-7, maxit=10_000,
    ))
    assert len(residuals) == 1
    assert len(solutions) == 3
    assert all(intercept == kkt == 0 and not columns.size
               for intercept, columns, _, _, _, kkt in solutions)


@pytest.mark.parametrize("strength,error,precisions", [
    (.02, 5e-8, [1e-12]),
    (.02, 2e-7, [1e-12, 1e-15]),
    (2e-6, 5e-8, [1e-12, 1e-15]),
    (1e-9, 5e-10, [1e-12]),
])
def test_kkt_acceptance_retains_penalty_scaling_and_strict_retry(
    monkeypatch, strength, error, precisions,
):
    x = sparse.csc_matrix([[-1.], [1.]])
    y = np.array([0., 1.])
    observed = []

    def native_fit(matrix, labels, path, **kwargs):
        observed.append(kwargs["thresh"])
        # For these symmetric observations, the coefficient KKT residual is
        # |lambda - sigmoid(-beta)| and the intercept residual is zero.
        probability = strength + (error if len(observed) == 1 else 0)
        beta = np.log1p(-probability) - np.log(probability)
        return [SimpleNamespace(coef_=np.array([beta]), intercept_=0., n_iter_=1)]

    monkeypatch.setattr(glmnet_working_set, "fit_glmnet_path", native_fit)
    solution, = glmnet_working_set.solve_working_set(
        x, y, np.full(2, .5), np.ones(1), np.array([strength]),
        alpha=1, thresh=1e-7, maxit=10_000,
    )
    assert observed == precisions
    intercept, columns, coefficients, *_ = solution
    np.testing.assert_array_equal(columns, [0])
    residual = .5 * (expit(x @ coefficients + intercept) - y)
    kkt = abs(float((x.T @ residual)[0]) + strength)
    np.testing.assert_allclose(kkt, error if len(precisions) == 1 else 0, atol=1e-15)


def test_default_working_set_precision_avoids_unnecessary_strict_resolves(monkeypatch):
    rng = np.random.default_rng(2026)
    x = sparse.csc_matrix(rng.normal(size=(180, 210)) * (rng.random((180, 210)) < .28))
    y = rng.binomial(1, expit(x[:, :5] @ [2, -1, .5, 1, -.5]))
    weights = rng.uniform(.4, 1.7, len(y))
    weights[0] = 0
    weights /= weights.sum()
    factors = rng.uniform(.7, 1.3, x.shape[1])
    path = np.array([.055, .035, .024])
    native_fit = glmnet_working_set.fit_glmnet_path
    precisions = []

    def record_fit(*args, **kwargs):
        precisions.append(kwargs["thresh"])
        return native_fit(*args, **kwargs)

    monkeypatch.setattr(glmnet_working_set, "fit_glmnet_path", record_fit)
    solutions = glmnet_working_set.solve_working_set(
        x, y, weights, factors, path, alpha=1, thresh=1e-7, maxit=2_000_000,
    )
    errors = []
    for strength, (intercept, columns, coefficients, *_) in zip(path, solutions, strict=True):
        beta = np.zeros(x.shape[1])
        beta[columns] = coefficients
        residual = weights * (expit(x @ beta + intercept) - y)
        gradient = x.T @ residual
        violation = np.maximum(np.abs(gradient) - strength * factors, 0)
        violation[columns] = np.abs(
            gradient[columns] + strength * factors[columns] * np.sign(coefficients)
        )
        errors.append(max(abs(residual.sum()), float(violation.max())))
    assert 1e-8 < max(errors) <= 1e-7
    assert precisions and set(precisions) == {1e-12}


@pytest.mark.parametrize("alpha", [0.6, 1.0])
@pytest.mark.parametrize("compact", [False, True])
def test_oversized_path_matches_weighted_full_objective(monkeypatch, alpha, compact):
    rng = np.random.default_rng(2026)
    n, width = 180, 210
    x = sparse.csc_matrix(rng.normal(size=(n, width)) * (rng.random((n, width)) < .28))
    y = rng.binomial(1, expit(x[:, :5] @ [2, -1, 0.5, 1, -0.5]))
    weights = rng.uniform(.4, 1.7, n)
    weights[0] = 0
    factors = rng.uniform(.7, 1.3, width)
    lambdas = [.024, .055, .035, .024]
    options = dict(alpha=alpha, sample_weight=weights, penalty_factor=factors, thresh=1e-13,
                   compute_kkt=not compact, compact_coefficients=compact)
    reference = glmnet.fit_glmnet_path(x, y, lambdas, **options)
    monkeypatch.setattr(glmnet, "SPARSE_INDEX_LIMIT", 7_000)
    monkeypatch.setattr(glmnet_working_set, "_SCAN_COLUMNS", 19)
    monkeypatch.setattr(glmnet_working_set, "_ADD_COLUMNS", 8)
    result = glmnet.fit_glmnet_path(x, y, lambdas, **options)

    valid = sparse.csc_matrix(rng.normal(size=(50, width)))
    for fit, expected in zip(result, reference, strict=True):
        assert fit.used_working_set_
        assert fit.n_features_in_ == width and fit.lambda_ == expected.lambda_
        np.testing.assert_array_equal(fit.coef_ != 0, expected.coef_ != 0)
        np.testing.assert_allclose(fit.predict_proba(valid), expected.predict_proba(valid),
                                   atol=2e-5, rtol=0)
        r = (expit(x @ fit.coef_ + fit.intercept_) - y) * weights / weights.sum()
        strength = fit.lambda_
        gradient = x.T @ r + strength * (1 - alpha) * factors * fit.coef_
        selected = fit.coef_ != 0
        np.testing.assert_allclose(
            gradient[selected],
            -strength * alpha * factors[selected] * np.sign(fit.coef_[selected]),
            atol=1e-7, rtol=0,
        )
        assert np.max(np.abs(gradient[~selected]) - strength * alpha * factors[~selected]) <= 1e-7
        assert abs(r.sum()) <= 1e-7
    assert result[0] is not result[-1]


@pytest.mark.parametrize("buffer_bytes", [0, 1024])
def test_oversized_encoding_and_joint_expression_preserve_values(monkeypatch, buffer_bytes):
    training = ["s0", "s1", "s2", "s3"]
    x = np.array([[0, 0, 1, 1, 3], [0, 2, 0, 1, 0], [1, 0, 1, 2, 0], [2, 1, 0, 0, 2]],
                 dtype=float)
    monkeypatch.setattr(amino_acid_sparse, "SPARSE_INDEX_LIMIT", 6)
    encoding = amino_acid_sparse.SparseFoldEncoding([training], training, buffer_bytes=buffer_bytes)
    try:
        encoding.append(x[:, :2])
        encoding.append(x[:, 2:])
        aa, = encoding.matrices()
    finally:
        encoding.close()
    assert aa.has_canonical_format
    assert aa.indptr.dtype == aa.indices.dtype == np.int64
    np.testing.assert_array_equal(aa.toarray(), x + encoding.shift)
    expression = np.array([[0, 2], [3, 0], [1, 1], [0, 5]])
    combined = amino_acid_sparse.combine_sparse_features(expression, aa)
    assert combined.has_canonical_format
    assert combined.indptr.dtype == combined.indices.dtype == np.int64
    np.testing.assert_array_equal(combined.toarray(), np.column_stack((expression, aa.toarray())))


def test_combined_fallback_reaches_cv_refit_and_saved_predictions(tmp_path, monkeypatch):
    config = load_and_resolve_config([combined_config(tmp_path)])
    config.preprocess.amino_acid.ranked_site_filter.method = "none"
    reference = run_amino_acid(config, tmp_path / "reference")
    # Make the test trigger the fallback while leaving local glmnet fits native.
    native_fit = glmnet._fit_oversized_path

    def fit(*args, **kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(glmnet, "SPARSE_INDEX_LIMIT", np.iinfo(np.int32).max)
            return native_fit(*args, **kwargs)

    monkeypatch.setattr(glmnet, "_fit_oversized_path", fit)
    monkeypatch.setattr(glmnet, "SPARSE_INDEX_LIMIT", 12)
    monkeypatch.setattr(amino_acid_sparse, "SPARSE_INDEX_LIMIT", 12)
    result = run_amino_acid(config, tmp_path / "fallback")
    models = joblib.load(result / "model_bundle/model_state.joblib")
    assert all(member.model.used_working_set_ for member in models)
    for filename in ("cv/tables/prediction_cv.tsv", "external_test/tables/pred_external_test.tsv",
                     "inference/tables/pred_inference.tsv"):
        original = pl.read_csv(reference / filename, separator="\t").sort("species")
        actual = pl.read_csv(result / filename, separator="\t").sort("species")
        assert actual["species"].to_list() == original["species"].to_list()
        np.testing.assert_allclose(actual["prob"], original["prob"], rtol=0, atol=3e-4)
