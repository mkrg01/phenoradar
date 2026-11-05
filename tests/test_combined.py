from __future__ import annotations

import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
import yaml
from scipy import sparse
from scipy.special import expit
from test_amino_acid import fixture_config
from typer.testing import CliRunner

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_run import (
    _site_coefficients,
    fit_scope,
    predict_amino_acid,
    predict_members,
    run_amino_acid,
)
from phenoradar.cli import app
from phenoradar.combined import ExpressionInput, fit_expression
from phenoradar.config import load_and_resolve_config
from phenoradar.config.schema import AppConfig
from phenoradar.cv import InnerCvPreprocessedFold, _score_glmnet_paths
from phenoradar.glmnet import GlmnetError, fit_glmnet_path
from phenoradar.model_selection import Candidate, active_search_space, generate_candidates


def combined_config(tmp_path: Path, aggregation: str = "aa_frequency") -> Path:
    path = fixture_config(tmp_path, aggregation)
    raw = yaml.safe_load(path.read_text())
    raw["data"]["feature_mode"] = "combined"
    tpm = tmp_path / "expression.tsv"
    rows = ["species\torthogroup\ttpm"]
    for i, sample in enumerate([*[f"s{j}" for j in range(8)], "external", "unknown"]):
        # These expression-only OGs must survive without matching AA OG IDs.
        rows.extend([f"{sample}\tEXPR\t{2 + 10 * (i % 2)}",
                     f"{sample}\tNOISE\t{2 + i // 2}"])
    rows.append("external\tTARGET_ONLY\t999999")
    tpm.write_text("\n".join(rows) + "\n")
    raw["data"]["tpm_path"] = str(tpm)
    raw.setdefault("preprocess", {})["expression"] = {
        "absent_feature_fill": "nan",
        "missing_expression": {"method": "neutral", "zero_as_missing": True},
        "sparse_feature_filter": {"enabled": False},
        "ranked_feature_filter": {"method": "none"},
    }
    raw["model_selection"]["search_space"] = {
        "lambda": [0.01, 0.1], "alpha": [1], "amino_acid_penalty_ratio": [0.1, 1, 10],
    }
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("sparse_input", [False, True])
def test_weighted_penalty_matches_exact_lasso_rescaling(sparse_input: bool) -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(200, 5))
    y = rng.binomial(1, expit(x @ np.array([1, -2, 0, 0.5, 0])))
    factors = np.array([1, 0.2, 3, 2, 0.7])
    weights = rng.uniform(0.1, 2, len(y))
    native_x = sparse.csc_matrix(x) if sparse_input else x
    models = fit_glmnet_path(native_x, y, [0.07, 0.02], alpha=1,
                             sample_weight=weights, penalty_factor=factors)
    reference = fit_glmnet_path(x / factors, y, [0.07, 0.02], alpha=1, sample_weight=weights)
    for actual, expected in zip(models, reference, strict=True):
        np.testing.assert_allclose(actual.coef_, expected.coef_ / factors, atol=1e-5)
        np.testing.assert_allclose(actual.intercept_, expected.intercept_, atol=1e-5)
        np.testing.assert_allclose(actual.predict_proba(x), expected.predict_proba(x / factors),
                                   atol=1e-6)
        assert actual.kkt_residual_ < 1e-5
    # Uniform factors must change absolute lambda, not cancel in native normalization.
    uniform = fit_glmnet_path(native_x, y, [0.02], alpha=1, penalty_factor=np.full(5, 3.0))[0]
    ordinary = fit_glmnet_path(native_x, y, [0.06], alpha=1)[0]
    np.testing.assert_allclose(uniform.coef_, ordinary.coef_, atol=1e-7)


@pytest.mark.parametrize("factors", [[1], [1, 0], [1, -1], [1, float("nan")]])
def test_invalid_penalty_vectors(factors: list[float]) -> None:
    with pytest.raises(GlmnetError, match="penalty_factor"):
        fit_glmnet_path(np.array([[0, 1], [1, 0], [1, 1], [0, 0]]), np.array([0, 1, 1, 0]),
                        [0.01], penalty_factor=np.array(factors))


def test_ratio_paths_remain_separate_with_fold_specific_feature_counts(monkeypatch) -> None:
    import phenoradar.cv as cv

    config = AppConfig.model_validate(
        {
            "data": {"feature_mode": "combined", "alignment_dir": "aa",
                     "sequence_tpm_path": "unused.tsv"},
            "preprocess": {
                "expression": {
                    "absent_feature_fill": "nan",
                    "missing_expression": {"method": "neutral"},
                }
            },
        }
    )
    rng = np.random.default_rng(9)
    folds = [InnerCvPreprocessedFold(str(i), rng.normal(size=(12, p)), rng.normal(size=(4, p)),
                                    np.tile([0, 1], 6), np.tile([0, 1], 2), None, n)
             for i, (p, n) in enumerate([(5, 2), (4, 1)])]
    calls = []
    original = cv.fit_glmnet_path

    def record(*args, **kwargs):
        calls.append((args[0].shape[1], list(args[2]), kwargs["penalty_factor"].copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(cv, "fit_glmnet_path", record)
    candidates = [Candidate(i, {"lambda": lam, "amino_acid_penalty_ratio": ratio})
                  for i, (ratio, lam) in enumerate([(0.2, 0.1), (0.2, 0.01),
                                                     (5, 0.1), (5, 0.01)])]
    scores = _score_glmnet_paths(config=config, training_scope_id="test", source_sample_set_id=0,
                                 candidates=candidates, preprocessed_folds=folds,
                                 progress_callback=None, timing_recorder=None)
    assert len(scores) == 4 and len(calls) == 4
    for width, path, factors in calls:
        assert path == [0.1, 0.01]
        n = 2 if width == 5 else 1
        np.testing.assert_array_equal(factors[:n], 1)
        assert len(set(factors[n:])) == 1 and factors[-1] in {0.2, 5}


def test_expression_preprocessing_is_training_only(tmp_path: Path) -> None:
    config = load_and_resolve_config([combined_config(tmp_path)])
    species = [*[f"s{i}" for i in range(8)], "external"]
    with ExpressionInput(config, species, species[:8]) as source:
        fitted, train, _, _ = fit_expression(config, source, species[:8], np.tile([0, 1], 4),
                                             np.repeat(["g0", "g1", "g2", "g3"], 2), [])
        assert "TARGET_ONLY" not in fitted.orthogroups
        raw = source.matrix(species[:8], fitted.orthogroups)
        np.testing.assert_allclose(fitted.scaler.mean_, np.log1p(raw).mean(axis=0))
        np.testing.assert_allclose(train.mean(axis=0), 0, atol=1e-12)
        missing, observed = fitted.transform(source, ["not_in_expression"])
        np.testing.assert_array_equal(missing, 0)
        assert not observed.any()


@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
def test_combined_full_run_and_frozen_predictions(tmp_path: Path, aggregation: str) -> None:
    config_path = combined_config(tmp_path, aggregation)
    if aggregation == "tpm_weighted_frequency":
        payload = yaml.safe_load(config_path.read_text())
        plain = Path(payload["data"]["tpm_path"])
        compressed = plain.with_suffix(".tsv.gz")
        with gzip.open(compressed, "wb") as handle:
            handle.write(plain.read_bytes())
        plain.unlink()
        payload["data"]["tpm_path"] = str(compressed)
        config_path.write_text(yaml.safe_dump(payload))
    config = load_and_resolve_config([config_path])
    messages = []
    destination = run_amino_acid(config, tmp_path / "run", progress=messages.append)
    assert any("prepare inner fold" in message for message in messages)
    assert any("AA features" in message for message in messages)
    assert any("score" in message and "glmnet paths" in message for message in messages)
    assert any("fit selected candidate" in message for message in messages)
    bundle = destination / "model_bundle"
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert manifest["feature_mode"] == "combined"
    assert manifest["bundle_format_version"] == "combined-1"
    state = joblib.load(bundle / "model_state.joblib")
    for member in state:
        assert member.expression_width > 0
        assert len(member.features) == len(member.model.coef_)
        assert member.model.alpha == 1
        assert all(json.loads(f)[0] == "expression" for f in member.expression.features)
    schema = pl.read_csv(destination / "model/tables/feature_schema.tsv", separator="\t")
    assert set(schema["modality"]) == {"expression", "amino_acid"}
    assert "TARGET_ONLY" not in schema["orthogroup"].to_list()
    for stage in ["cv", "model"]:
        tables = destination / stage / "tables"
        coefficients = pl.read_csv(tables / "combined_coefficients.tsv", separator="\t")
        aa = coefficients.filter(pl.col("modality") == "amino_acid")
        np.testing.assert_array_equal(aa["coefficient"], aa["coefficient_original_unit"])
        assert (tables / "amino_acid_site_pss.tsv").is_file()
        assert (tables / "amino_acid_site_effect_range.tsv").is_file()
        if stage == "cv":
            for modality in ["expression", "amino_acid"]:
                assert (tables / f"{modality}_feature_importance.tsv").is_file()
    oof = pl.read_csv(destination / "cv/tables/prediction_cv.tsv", separator="\t")
    assert oof.height == 8 and oof["species"].n_unique() == 8
    for name in ["combined_modality_contributions", "combined_penalty_selection",
                 "expression_feature_filter_funnel",
                 "expression_importance_top", "amino_acid_importance_top",
                 "expression_coefficients_signed_top", "amino_acid_coefficients_signed_top"]:
        assert (destination / "cv/figures" / f"{name}.svg").is_file(), name
    summary = pl.read_csv(destination / "cv/tables/combined_modality_contributions_summary.tsv",
                          separator="\t")
    assert set(summary["species"]) == set(oof["species"])
    np.testing.assert_allclose(summary["logit"], summary["expression_contribution"]
                               + summary["amino_acid_contribution"] + summary["intercept"])
    funnel = pl.read_csv(destination / "cv/tables/amino_acid_site_filter_counts_summary.tsv",
                        separator="\t")
    assert funnel["stage"].to_list()[-1] == "retained_sites"
    assert "Retained sites" in (
        destination / "cv/figures/amino_acid_site_filter_funnel.svg"
    ).read_text()
    for stage, sample in [("external_test", "external"), ("inference", "unknown")]:
        subset = tmp_path / f"{sample}.tsv"
        subset.write_text(f"species\n{sample}\n")
        result = predict_amino_acid(bundle, tmp_path / sample, metadata_path=subset)
        predicted = pl.read_csv(result / "predict/tables/pred_predict.tsv", separator="\t")
        original = pl.read_csv(destination / stage / "tables" / f"pred_{stage}.tsv", separator="\t")
        np.testing.assert_allclose(predicted["prob"], original["prob"], rtol=0, atol=1e-12)
        local = pl.read_csv(result / "predict/tables/combined_modality_contributions.tsv",
                            separator="\t")
        np.testing.assert_allclose(local["logit"], local["expression_contribution"]
                                   + local["amino_acid_contribution"] + local["intercept"])
        np.testing.assert_allclose(np.mean(expit(local["logit"].to_numpy())), predicted["prob"][0])
    cli = CliRunner().invoke(app, ["predict", "--model-bundle", str(bundle),
                                  "--tpm-path", config.data.tpm_path, "--quiet"])
    assert cli.exit_code == 0, cli.output


@pytest.mark.parametrize("overrides", [
    {"model_selection": {"search_space": {"alpha": [0.5]}}},
    {"preprocess": {"expression": {"feature_scaling": {"method": "none"}}}},
    {"phylogenetic_imputation": {"enabled": True}},
])
def test_combined_rejects_unsupported_model_choices(tmp_path: Path, overrides: dict) -> None:
    raw = yaml.safe_load(combined_config(tmp_path).read_text())
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    with pytest.raises(ValueError):
        AppConfig.model_validate(raw)


@pytest.mark.parametrize("ratio", [[0], [-1], [True], [float("inf")],
                                  {"type": "continuous_range", "start": 0, "end": 1}])
def test_invalid_ratios_fail_before_input_import(tmp_path: Path, ratio: object) -> None:
    path = combined_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["model_selection"]["search_space"]["amino_acid_penalty_ratio"] = ratio
    if isinstance(ratio, dict):
        raw["model_selection"].update({"search_strategy": "tpe", "trial_count": 1})
    with pytest.raises(ValueError, match="amino_acid_penalty_ratio"):
        AppConfig.model_validate(raw)


def test_missing_modality_and_unused_modality_coverage(tmp_path: Path) -> None:
    path = combined_config(tmp_path)
    config = load_and_resolve_config([path])
    run_amino_acid(config, tmp_path / "run")
    member = joblib.load(tmp_path / "run/model_bundle/model_state.joblib")[0]
    config.abstention.enabled = True
    expression_path = Path(config.data.tpm_path)
    data = pl.read_csv(expression_path, separator="\t").filter(pl.col("species") != "unknown")
    data.write_csv(expression_path, separator="\t")
    all_species = [*[f"s{i}" for i in range(8)], "external", "unknown"]
    with (ExpressionInput(config, all_species, []) as source,
          AlignmentStore(Path(config.data.alignment_dir), all_species) as store):
        member.model.coef_[:] = np.arange(1, len(member.model.coef_) + 1)
        result = predict_members(config, store, ["unknown"], [member], expression=source)
        assert result["expression_information_coverage"][0] == 0
        assert result["amino_acid_information_coverage"][0] == 1
        assert result["decision_status"][0] == "abstained"
        member.model.coef_[:member.expression_width] = 0
        result = predict_members(config, store, ["unknown"], [member], expression=source)
        assert result["expression_information_coverage"][0] is None
        assert result["decision_status"][0] == "accepted"
        member.model.coef_[:] = 0
        result = predict_members(config, store, ["unknown"], [member], expression=source)
        assert result["information_coverage"][0] == 0
        assert result["decision_status"][0] == "abstained"


def test_combined_cli_run_tpe_and_no_selection(tmp_path: Path, monkeypatch) -> None:
    path = combined_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["runtime"]["execution_stage"] = "cv_only"
    raw["model_selection"].update({"search_strategy": "tpe", "trial_count": 2})
    raw["model_selection"]["search_space"]["amino_acid_penalty_ratio"] = {
        "type": "continuous_range", "start": 0.2, "end": 2,
    }
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "-c", str(path), "--quiet"])
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "runs").glob("*/cv/tables/prediction_cv.tsv"))) == 1
    raw["model_selection"] = {
        "selected_candidate_count": None,
        "search_space": {"lambda": [0.01], "amino_acid_penalty_ratio": [2]},
    }
    path.write_text(yaml.safe_dump(raw))
    run_amino_acid(load_and_resolve_config([path]), tmp_path / "without_selection")


def test_joint_model_uses_both_independent_signals_and_preserves_aa_offsets(tmp_path: Path) -> None:
    config = load_and_resolve_config([combined_config(tmp_path)])
    config.model_selection.selected_candidate_count = None
    config.model_selection.search_space = {"lambda": [0.005], "amino_acid_penalty_ratio": [0.7]}
    config.preprocess.amino_acid.ranked_site_filter.method = "none"
    rng = np.random.default_rng(85)
    species = [f"sample{i}" for i in range(160)]
    expression = rng.normal(size=len(species))
    state = rng.integers(0, 2, len(species))
    y = rng.binomial(1, expit(2 * expression + 2 * (2 * state - 1)))
    pl.DataFrame({"species": species, "orthogroup": ["EXPR"] * len(species),
                  "tpm": np.exp(expression)}).write_csv(config.data.tpm_path, separator="\t")
    aa_path = Path(config.data.alignment_dir) / "OG1.faa"
    aa_path.write_text("".join(f">{s}_gene\n{'A' if value else 'V'}\n"
                               for s, value in zip(species, state, strict=True)))
    rows = pl.DataFrame({"species": species, "label": y,
                         "group_id": ["g"] * len(species),
                         "contrast_group_id": ["g"] * len(species)})
    with (ExpressionInput(config, species, species) as source,
          AlignmentStore(aa_path.parent, species) as store):
        member = fit_scope(config, store, rows, "final_refit", tmp_path / "tables", [], source)[0]
        assert member.expression_width == 1 and member.model.coef_[0] != 0
        aa_beta = _site_coefficients(member)[0]
        assert np.ptp(aa_beta[member.transform.sites[0].known]) > 0.1
        np.testing.assert_array_equal(aa_beta[member.transform.sites[0].active],
                                      member.model.coef_[1:])
        xe, _ = member.expression.transform(source, species)
        xa, _, _ = member.transform.transform(store, species)
        x = np.column_stack((xe, xa))
        model = fit_glmnet_path(x, y, [0.005], alpha=1,
                                penalty_factor=np.array([1, 0.7, 0.7]))[0]
        np.testing.assert_allclose(member.model.coef_, model.coef_, atol=1e-7)
        predicted = predict_members(config, store, species, [member], expression=source)
        np.testing.assert_allclose(predicted["prob"], expit(x @ member.model.coef_
                                                          + member.model.intercept_))


@pytest.mark.parametrize("strategy", ["grid", "random", "tpe"])
@pytest.mark.parametrize("mode,model,param,values", [
    ("expression", "logistic_elasticnet", "lambda", [0.01, 0.1]),
    ("expression", "linear_svm", "C", [0.1, 1]),
    ("expression", "random_forest", "n_estimators", [2, 3]),
    ("amino_acid", "logistic_elasticnet", "lambda", [0.01, 0.1]),
])
def test_inactive_ratio_never_changes_candidate_generation(
    strategy: str, mode: str, model: str, param: str, values: list[float]
) -> None:
    ratios = [0.1, 0.3, 1, 3, 10]
    raw = {
        "data": {"feature_mode": mode, "alignment_dir": "unused",
                 "sequence_tpm_path": "unused.tsv"},
        "model": {"name": model},
        "model_selection": {
            "search_strategy": strategy, "trial_count": 3 if strategy != "grid" else None,
            "search_space": {param: values, "amino_acid_penalty_ratio": ratios},
        },
    }
    with_ratio = AppConfig.model_validate(raw)
    assert with_ratio.model_selection.search_space["amino_acid_penalty_ratio"] == ratios
    assert active_search_space(with_ratio) == {param: values}
    raw["model_selection"]["search_space"].pop("amino_acid_penalty_ratio")
    without = AppConfig.model_validate(raw)
    warnings_a, warnings_b = [], []
    options = {"training_scope_id": "outer_fold_1", "source_sample_set_id": 0}
    a = generate_candidates(config=with_ratio, warnings=warnings_a, **options)
    b = generate_candidates(config=without, warnings=warnings_b, **options)
    assert a == b
    assert warnings_a == warnings_b
    assert all("amino_acid_penalty_ratio" not in candidate.params for candidate in a)


@pytest.mark.parametrize("mode", ["expression", "amino_acid"])
def test_inactive_continuous_ratio_does_not_disable_grid(mode: str) -> None:
    raw = {
        "data": {"feature_mode": mode, "alignment_dir": "unused",
                 "sequence_tpm_path": "unused.tsv"},
        "model_selection": {"search_strategy": "grid", "search_space": {
            "lambda": [0.01], "amino_acid_penalty_ratio": {
                "type": "continuous_range", "start": 0.1, "end": 10,
            },
        }},
    }
    config = AppConfig.model_validate(raw)
    assert active_search_space(config) == {"lambda": [0.01]}
    raw["data"]["feature_mode"] = "combined"
    raw.setdefault("preprocess", {})["expression"] = {"absent_feature_fill": "nan",
                         "missing_expression": {"method": "neutral"}}
    with pytest.raises(ValueError, match="grid.*continuous.*amino_acid_penalty_ratio"):
        AppConfig.model_validate(raw)


@pytest.mark.parametrize("mode", ["expression", "amino_acid"])
@pytest.mark.parametrize("strategy", ["grid", "tpe"])
def test_single_mode_cli_ignores_configured_ratio(
    tmp_path: Path, monkeypatch, mode: str, strategy: str
) -> None:
    path = combined_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["data"]["feature_mode"] = mode
    raw["runtime"]["execution_stage"] = "cv_only"
    raw["evaluation"]["group_bootstrap"]["enabled"] = False
    raw["model_selection"]["search_strategy"] = strategy
    if strategy == "tpe":
        raw["model_selection"]["trial_count"] = 2
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "-c", str(path), "--quiet"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    runs = list((tmp_path / "runs").glob("*run*"))
    assert len(runs) == 1
    audited = []
    for table in (runs[0] / "cv/tables").glob("*model_selection*.tsv"):
        frame = pl.read_csv(table, separator="\t")
        if "params_json" in frame.columns:
            audited.extend(frame["params_json"].drop_nulls().to_list())
    assert audited
    assert all("amino_acid_penalty_ratio" not in value for value in audited)
