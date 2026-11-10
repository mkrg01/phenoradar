"""Relative paths use training-local KKT scales and bounded inner-CV extensions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import yaml
from scipy import sparse
from test_combined import combined_config

import phenoradar.cv as cv
from phenoradar.amino_acid_run import run_amino_acid
from phenoradar.auto_lambda import _FoldCache, lambda_max, score_auto_paths
from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.glmnet import GlmnetError, fit_glmnet_path
from phenoradar.model_selection import auto_lambda_spec, generate_candidates
from phenoradar.split import build_split_artifacts


def config(**spec):
    return AppConfig.model_validate(
        {
            "model_selection": {
                "selected_candidate_count": 1,
                "inner_cv_strategy": "logo",
                "search_space": {"alpha": [1.0], "lambda": {"type": "auto_log_range", **spec}},
            }
        }
    )


def candidates(c):
    return generate_candidates(
        config=c, training_scope_id="test", source_sample_set_id=0, warnings=[]
    )


@pytest.mark.parametrize("csc", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_kkt_scale_matches_native_objective_and_shift(csc, weighted, alpha):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 5))
    y = np.tile([0, 1], 50)
    x[:, 0] += y * 2
    weights = rng.uniform(0.1, 3, len(y)) if weighted else None
    factors = np.array([1.0, 0.1, 2.0, 3.0, 0.5])
    z = sparse.csc_matrix(x) if csc else x
    maximum = lambda_max(z, y, weights, alpha=alpha, penalty_factor=factors)
    np.testing.assert_allclose(
        maximum, lambda_max(x + 100, y, weights, alpha=alpha, penalty_factor=factors), atol=1e-13
    )
    fitted = fit_glmnet_path(
        z,
        y,
        [maximum * (1 + 1e-8), maximum * 0.99],
        alpha=alpha,
        sample_weight=weights,
        penalty_factor=factors,
        thresh=1e-14,
    )
    assert np.count_nonzero(fitted[0].coef_) == 0
    assert np.count_nonzero(fitted[1].coef_) > 0
    assert fitted[0].kkt_residual_ < 1e-12


def test_streamed_native_solutions_equal_legacy_extraction():
    rng = np.random.default_rng(35)
    x = sparse.csc_matrix(rng.normal(size=(80, 8)))
    y = np.tile([0, 1], 40)
    path = [0.02, 0.1, 0.005]
    expected = fit_glmnet_path(x, y, path, thresh=1e-13)
    actual = []
    assert (
        fit_glmnet_path(
            x, y, path, thresh=1e-13, on_model=lambda i, model: actual.append((i, model))
        )
        == []
    )
    for index, model in actual:
        np.testing.assert_array_equal(model.coef_, expected[index].coef_)
        assert model.intercept_ == expected[index].intercept_


def test_default_and_explicit_compatibility_and_invalid_auto():
    c = AppConfig.model_validate(
        {"model_selection": {"selected_candidate_count": 1, "inner_cv_strategy": "logo"}}
    )
    assert auto_lambda_spec(c) is not None
    assert len(candidates(c)) == 9
    assert auto_lambda_spec(AppConfig()) is None  # Does not enable selection.
    explicit = c.model_dump()
    explicit["model_selection"]["search_space"] = {"lambda": [0, 0.1], "alpha": [0]}
    absolute = AppConfig.model_validate(explicit)
    assert auto_lambda_spec(absolute) is None
    assert [v.params["lambda"] for v in candidates(absolute)] == [0, 0.1]
    for change in [
        {"alpha": [0]},
        {"alpha": [True]},
        {"lambda": {"type": "auto_log_range", "min_ratio": 1}},
    ]:
        invalid = config().model_dump()
        invalid["model_selection"]["search_space"].update(change)
        with pytest.raises(ValueError):
            AppConfig.model_validate(invalid)
    invalid = config().model_dump()
    invalid["model_selection"].update(search_strategy="random", trial_count=9)
    with pytest.raises(ValueError, match="grid"):
        AppConfig.model_validate(invalid)


def fold(scale=1, validation_scale=1):
    x = np.array([[-1.0, 0.2], [1, -0.2], [-2, 0.1], [2, -0.1]]) * scale
    return cv.InnerCvPreprocessedFold(
        "0",
        sparse.csc_matrix(x),
        sparse.csc_matrix(x * validation_scale),
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        None,
    )


def test_extension_cache_preserves_display_and_seed_fold_ids(tmp_path):
    prepared = replace(fold(), inner_fold_id="10", seed_fold_id="2")
    cache = _FoldCache(tmp_path)
    cache.add(prepared, (.25, .5))
    restored, maxima = next(iter(cache))
    assert restored.inner_fold_id == "10" and restored.seed_fold_id == "2"
    assert maxima == (.25, .5)
    np.testing.assert_array_equal(restored.x_train.toarray(), prepared.x_train.toarray())
    np.testing.assert_array_equal(restored.x_valid.toarray(), prepared.x_valid.toarray())


def test_extensions_cache_once_and_only_score_new_candidates():
    c = config(n_points=3, max_extensions=2)
    calls = []
    prepared = []

    def prepare():
        prepared.append(1)
        yield fold()

    def fit(x, y, lambdas, **kw):
        calls.append((list(lambdas), list(kw["warmup_lambdas"])))
        for i, lam in enumerate(lambdas):
            kw["on_model"](i, SimpleNamespace(coef_=np.zeros(x.shape[1]), intercept_=np.log(lam)))

    warnings = []
    result = score_auto_paths(
        c,
        candidates(c),
        prepare(),
        fit_path=fit,
        metric=lambda y, p: float(p.mean()),
        diagnostic=lambda model: None,
        warnings=warnings,
    )
    assert prepared == [1]
    assert len(calls) == 3 and [len(v[0]) for v in calls] == [2, 1, 1]
    assert [len(v[1]) for v in calls] == [3, 4, 5]
    assert len(result.candidates) == 5
    assert min(v.params["lambda_fraction"] for v in result.candidates) == pytest.approx(0.0001)
    assert "lower boundary" in warnings[0]
    assert {r["lambda_path_stop_reason"] for _, rows in result.scores for r in rows} == {
        "extension_limit_at_boundary"
    }


def test_tied_scores_choose_null_endpoint_without_extension():
    c = config(n_points=3)
    result = score_auto_paths(
        c,
        candidates(c),
        [fold()],
        fit_path=fit_glmnet_path,
        metric=lambda y, p: 1.0,
        diagnostic=cv._glmnet_fit_diagnostic,
        warnings=[],
    )
    ranked = cv._rank_selected_candidates(
        [
            cv.SelectedCandidate(v, score)
            for v, (score, _) in zip(result.candidates, result.scores, strict=True)
        ],
        c,
    )
    assert ranked[0].candidate.params["lambda_fraction"] == 1
    assert len(result.candidates) == 3


def test_failed_extension_retains_only_completed_round():
    c = config(n_points=3)
    count = 0

    def fit(x, y, lambdas, **kw):
        nonlocal count
        count += 1
        if count == 2:
            raise GlmnetError("extension test")
        for i, lam in enumerate(lambdas):
            kw["on_model"](i, SimpleNamespace(coef_=np.zeros(x.shape[1]), intercept_=np.log(lam)))

    warnings = []
    result = score_auto_paths(
        c,
        candidates(c),
        [fold()],
        fit_path=fit,
        metric=lambda y, p: float(p.mean()),
        diagnostic=lambda model: None,
        warnings=warnings,
    )
    assert len(result.candidates) == 3
    assert len(result.scores) == 3
    assert result.scores[0][1][0]["lambda_path_stop_reason"] == "extension_failed"
    assert "retaining completed" in warnings[0]


def test_validation_values_do_not_set_scale_and_refit_recomputes_scale():
    c = config(n_points=3, extend_lower=False)
    rows = []
    for validation_scale in [1, 1e8]:
        result = score_auto_paths(
            c,
            candidates(c),
            [fold(validation_scale=validation_scale)],
            fit_path=fit_glmnet_path,
            metric=lambda y, p: float(np.mean(p)),
            diagnostic=cv._glmnet_fit_diagnostic,
            warnings=[],
        )
        rows.append([r["lambda_max"] for _, rs in result.scores for r in rs])
    assert rows[0] == rows[1]
    selected = cv.SelectedCandidate(result.candidates[1], 0.5)
    source = cv.SourceSelectionResult(
        [selected], 3, 3, 1, 1, [r for _, rs in result.scores for r in rs]
    )
    estimator = cv._build_estimator(c, 42, fold().y_train, selected.candidate.params)
    current = fold(scale=3)
    diag = cv._fit_selected_estimator(estimator, current.x_train, current.y_train, None, source)
    assert estimator.lambda_max_ == pytest.approx(rows[0][0] * 3)
    assert estimator.lambda_ == pytest.approx(estimator.lambda_max_ * 0.1)
    assert diag.lambda_metadata["lambda"] == estimator.lambda_


@pytest.mark.parametrize("null_selection", [False, True])
@pytest.mark.parametrize("mode", ["expression", "amino_acid", "combined"])
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_auto_nested_cv_and_final_refit_all_modes(
    tmp_path: Path, monkeypatch, mode, null_selection, n_jobs
):
    if null_selection and mode == "expression":
        import phenoradar.amino_acid_run as aa

        monkeypatch.setattr(cv, "_selection_metric_from_probability", lambda *args: 0.0)
        monkeypatch.setattr(aa, "_selection_metric_from_probability", lambda *args: 0.0)
    path = combined_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["data"]["feature_mode"] = mode
    raw["model_selection"]["search_space"]["lambda"] = {
        "type": "auto_log_range",
        "n_points": 3,
        "extend_lower": False,
    }
    path.write_text(yaml.safe_dump(raw))
    c = load_and_resolve_config([path])
    c.runtime.n_jobs = n_jobs
    if null_selection and mode != "expression":
        # Equal observations within each complete 0/1 contrast pair give an
        # actual zero gradient, including in spawned workers without monkeypatches.
        aa = Path(c.data.alignment_dir) / "OG1.faa"
        records = aa.read_text().split(">")[1:]
        rewritten = []
        for record in records:
            name = record.splitlines()[0]
            if name.startswith("s"):
                letter = "ACDE"[int(name.split("_", 1)[0][1:]) // 2]
                rewritten.append(f">{name}\n-X{letter}{letter}*--\n")
            else:
                rewritten.append(">" + record)
        aa.write_text("".join(rewritten))
        expression = pl.read_csv(c.data.tpm_path, separator="\t")
        expression = expression.with_columns(
            pl.when(pl.col("species").str.starts_with("s"))
            .then(pl.col("species").str.slice(1).cast(pl.Int64, strict=False) // 2 + 1)
            .otherwise(pl.col("tpm")).alias("tpm"))
        expression.write_csv(c.data.tpm_path, separator="\t")
    if mode == "expression":
        manifest = build_split_artifacts(c).split_manifest
        outer = cv.run_outer_cv(c, manifest)
        final = cv.run_final_refit(c, manifest)
        trials = outer.model_selection_trials
        selected = final.model_selection_selected
    else:
        destination = run_amino_acid(c, tmp_path / "run")
        trials = pl.read_csv(destination / "cv/tables/model_selection_trials.tsv", separator="\t")
        selected = pl.read_csv(
            destination / "model/tables/model_selection_selected.tsv", separator="\t"
        )
    assert trials is not None and selected is not None
    assert {
        "lambda_fraction",
        "lambda_max",
        "lambda_actual",
        "lambda_path_stop_reason",
        "nonzero_amino_acid",
        "nonzero_expression",
    } <= set(trials.columns)
    for params in selected["params_json"]:
        p = json.loads(params)
        if null_selection:
            assert p["lambda_fraction"] == 1
        assert p["lambda"] == pytest.approx(p["lambda_fraction"] * p["lambda_max"])
    assert np.all(np.isfinite(trials["lambda_max"].to_numpy()))


def test_aa_auto_scoring_uses_compact_coefficients_without_dense_expansion(monkeypatch):
    import phenoradar.glmnet as glmnet

    c = config(n_points=5, max_extensions=1)
    c.data.feature_mode = "amino_acid"

    def dense(*args, **kwargs):
        kwargs["compact_coefficients"] = False
        return fit_glmnet_path(*args, **kwargs)

    options = dict(metric=lambda y, p: float(-np.mean((p - .5) ** 2)),
                   diagnostic=cv._glmnet_fit_diagnostic, warnings=[])
    expected = score_auto_paths(c, candidates(c), [fold()], fit_path=dense, **options)

    def unexpected(*args):
        pytest.fail("Automatic AA CV expanded dense coefficients")

    monkeypatch.setattr(glmnet._CompactGlmnetModel, "coef_", property(unexpected))
    actual = score_auto_paths(c, candidates(c), [fold()], fit_path=fit_glmnet_path, **options)
    assert actual.candidates == expected.candidates
    for (a, ar), (b, br) in zip(actual.scores, expected.scores, strict=True):
        assert a == pytest.approx(b, abs=1e-15)
        for left, right in zip(ar, br, strict=True):
            for key in ["lambda_actual", "lambda_max", "nonzero_expression",
                        "nonzero_amino_acid", "lambda_extension_round",
                        "lambda_path_stop_reason", "lambda_path_min_fraction"]:
                assert left[key] == right[key]


def test_combined_penalty_families_share_one_gradient_scan(monkeypatch):
    import phenoradar.auto_lambda as automatic

    c = config(n_points=3, extend_lower=False)
    c.data.feature_mode = "combined"
    c.model_selection.search_space["amino_acid_penalty_ratio"] = [0.1, 1.0, 10.0]
    f = replace(fold(), n_expression_features=1)
    original = automatic.null_gradients
    expected = original(f.x_train, f.y_train, f.sample_weight)
    calls = []

    def gradient(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(automatic, "null_gradients", gradient)
    result = score_auto_paths(
        c,
        candidates(c),
        [f],
        fit_path=fit_glmnet_path,
        metric=lambda y, p: float(p.mean()),
        diagnostic=cv._glmnet_fit_diagnostic,
        warnings=[],
    )
    assert calls == [1]
    for candidate, (_, rows) in zip(result.candidates, result.scores, strict=True):
        ratio = candidate.params["amino_acid_penalty_ratio"]
        assert rows[0]["lambda_max"] == pytest.approx(max(expected[0], expected[1] / ratio))
