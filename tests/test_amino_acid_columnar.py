"""Columnar preprocessing retains train-only statistics and the original site API."""

from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from test_amino_acid import alignment, settings, tpm_table
from test_glmnet_refit import _source, _weighted_problem

import phenoradar.amino_acid as aa
import phenoradar.cv as cv
from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache
from phenoradar.amino_acid_outputs import write_preprocessing_audits
from phenoradar.amino_acid_run import _write
from phenoradar.combined import ExpressionTransform
from phenoradar.config import AppConfig
from phenoradar.glmnet import GlmnetLogisticRegression, fit_glmnet_path


@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
@pytest.mark.parametrize("disk_backed", [False, True])
def test_columnar_sites_match_sitewise_statistics_and_serialized_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregation: str, disk_backed: bool,
) -> None:
    monkeypatch.setattr(aa, "SITE_BLOCK_SIZE", 2)
    if disk_backed:
        monkeypatch.setattr(aa, "MAX_MATRIX_CELLS", 1)
    records = [("s1_a", "AVAA--V"), ("s1_b", "VVAV-AA"),
               ("s2_a", "VVAAVAA"), ("s3_a", "A-AVVVA"), ("held_a", "XW-AVAA")]
    path = alignment(tmp_path / "aa", records)
    (path / "OG2.faa").write_text((path / "OG1.faa").read_text())
    tpm = tpm_table(tmp_path / "tpm.tsv", [
        (name.split("_")[0], name, i + 1) for i, (name, _) in enumerate(records)
    ])
    with aa.AlignmentStore(path, ["s1", "s2", "s3", "held", "absent"],
                           sequence_tpm_path=str(tpm)) as store:
        fitted = aa.fit_amino_acid(store, ["s3", "s1", "s2"], settings(aggregation),
                                  y=np.array([0, 1, 0]), groups=np.array(["a", "b", "c"]))
        assert isinstance(fitted.sites, aa.AASiteTable)
        inner = aa.fit_amino_acid(store, ["s3", "s1", "s2"], settings(aggregation),
                                 y=np.array([0, 1, 0]), groups=np.array(["a", "b", "c"]),
                                 audit=False)
        np.testing.assert_array_equal(inner.matrix(store, ["s3", "s1", "s2"]),
                                      fitted.matrix(store, ["s3", "s1", "s2"]))
        assert inner.features == fitted.features
        raw, _, _ = fitted.raw(store, ["s3", "s1", "s2"])
        for j, site in enumerate(fitted.sites):
            observed = np.isfinite(raw[:, j, 0])
            values = raw[observed, j]
            np.testing.assert_array_equal(site.mean, values.mean(axis=0, dtype=np.float64))
            np.testing.assert_array_equal(site.active, values.var(axis=0) > 1e-12)
            assert site.post_aggregation_sample_fraction == np.mean(observed)
        plain = replace(fitted, sites=list(fitted.sites))
        target = ["held", "absent", "s2", "s1"]
        expected = plain.transform(store, target)
        actual = fitted.transform(store, target)
        assert actual[0].flags.f_contiguous
        for result, reference in zip(actual[:2], expected[:2], strict=True):
            np.testing.assert_array_equal(result, reference)
        assert actual[2] == expected[2]
        assert fitted.features == plain.features
        assert fitted.schema().equals(plain.schema())
        assert fitted.sites[-1].feature(0) == plain.sites[-1].feature(0)
        assert [s.column for s in fitted.sites[::2]] == [s.column for s in plain.sites[::2]]
        joblib.dump(fitted, tmp_path / "transform.joblib")
        restored = joblib.load(tmp_path / "transform.joblib")
        np.testing.assert_array_equal(restored.matrix(store, target), actual[0])


def test_default_threshold_and_explicit_strict_fit_both_optimize_same_objective() -> None:
    x, y, weights = _weighted_problem()
    config = AppConfig()
    default = cv._build_estimator(config, 42, y, model_params={"lambda": 0.017, "alpha": 0.7})
    assert default.thresh == 1e-7
    default.fit(x, y, sample_weight=weights)
    strict = GlmnetLogisticRegression(lambda_=0.017, alpha=0.7, thresh=1e-14).fit(
        x, y, sample_weight=weights
    )

    def objective(model: GlmnetLogisticRegression) -> float:
        eta = model.decision_function(x)
        loss = np.average(np.logaddexp(0, eta) - y * eta, weights=weights)
        penalty = model.lambda_ * (
            model.alpha * np.abs(model.coef_).sum()
            + (1 - model.alpha) * np.dot(model.coef_, model.coef_) / 2
        )
        return float(loss + penalty)

    assert default.n_iter_ < strict.n_iter_
    assert objective(default) <= objective(strict) + 1e-6
    np.testing.assert_allclose(default.predict_proba(x), strict.predict_proba(x), atol=1e-3)


def test_combined_refit_keeps_penalty_factors_and_matching_ratio() -> None:
    x, y, weights = _weighted_problem()
    factors = np.array([1, 1, 3, 3, 3, 3])
    estimator = GlmnetLogisticRegression(lambda_=0.017, alpha=1, penalty_factor=factors)
    source = _source(
        [{"lambda": 0.017, "alpha": 1, "amino_acid_penalty_ratio": 3}],
        [{"lambda": 0.1, "alpha": 1, "amino_acid_penalty_ratio": 3},
         {"lambda": 0.9, "alpha": 1, "amino_acid_penalty_ratio": 10}],
    )
    cv._fit_selected_estimator(estimator, x, y, weights, source, amino_acid_penalty_ratio=3)
    reference = fit_glmnet_path(x, y, [0.1, 0.017], alpha=1,
                               penalty_factor=factors, sample_weight=weights)[-1]
    assert estimator.path_length_ == 2
    np.testing.assert_array_equal(estimator.penalty_factor, factors)
    np.testing.assert_array_equal(estimator.coef_, reference.coef_)
    assert estimator.intercept_ == reference.intercept_


@pytest.mark.parametrize("combined", [False, True])
def test_columnar_audit_matches_complete_legacy_tables(tmp_path: Path, combined: bool) -> None:
    path = alignment(tmp_path / "aa", [("s1_a", "AAV"), ("s2_a", "VAA")])
    (path / "OG1.faa").rename(path / 'OG_é".faa')
    with aa.AlignmentStore(path, ["s1", "s2"]) as store:
        fitted = aa.fit_amino_acid(store, ["s1", "s2"], settings("aa_frequency"),
                                  y=np.array([0, 1]), groups=np.array(["a", "b"]))
    expr = ExpressionTransform(AppConfig(), ["EXPR_A"], None) if combined else None
    for index in range(2):
        context = {"scope": f"outer_fold_{index}", "sample_set_id": index}
        write_preprocessing_audits(tmp_path / "new", fitted, context, expr)
        keys = ["feature", "modality", "orthogroup", "alignment_pos_1based", "residue",
                "aggregation", "training_mean"]
        rows = ([] if expr is None else expr.schema()) + list(fitted.schema_rows())
        _write(tmp_path / "old/feature_schema.tsv",
               ({**context, **{key: row.get(key) for key in keys}} for row in rows))
        _write(tmp_path / "old/amino_acid_site_qc.tsv", (
            {**context, "orthogroup": s.orthogroup, "alignment_pos_1based": s.column + 1,
             "sample_fraction": s.sample_fraction,
             "post_aggregation_sample_fraction": s.post_aggregation_sample_fraction,
             "minor_samples": s.minor_samples, "minor_groups": s.minor_groups, "score": s.score,
             "n_active_states": int(np.sum(s.active))} for s in fitted.sites
        ))
    for name in ["feature_schema.tsv", "amino_acid_site_qc.tsv"]:
        assert_frame_equal(pl.read_csv(tmp_path / "new" / name, separator="\t"),
                           pl.read_csv(tmp_path / "old" / name, separator="\t"), abs_tol=1e-15)


def test_compact_frequency_storage_preserves_all_states_and_observed_zero_coverage(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(739)
    raw = rng.dirichlet(np.ones(20), size=(3, 5)).astype(np.float32)
    raw[1, 1] = np.nan
    raw[:, 2] = 0  # Finite zero vectors and wholly missing columns remain distinct.
    raw[:, 3] = np.nan
    raw[:, 4] = 0
    raw[:, 4, 19] = 1
    coverage = np.ones((3, 5), dtype=np.float32)
    coverage[1, 1] = coverage[:, 3] = 0
    coverage[0, 0] = 0  # Underflow in coverage must not erase finite frequencies.
    samples = ["a", "b", "c"]
    cache = AminoAcidFrequencyCache(
        tmp_path / "frequencies.bin", dict(zip(samples, range(3), strict=True))
    )

    def compute(species: list[str], columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = [samples.index(s) for s in species]
        return raw[rows][:, columns], coverage[rows][:, columns]

    for columns in [np.array([4, 0, 4]), np.array([3, 1, 2, 0])]:
        actual = cache.get(["c", "a", "a"], columns, compute, block_size=2)
        expected = compute(["c", "a", "a"], columns)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
    missing = cache.get(["absent"], np.array([0]), compute, block_size=2)
    assert np.isnan(missing[0]).all() and not missing[1].any()
