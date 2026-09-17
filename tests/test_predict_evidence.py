from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from phenoradar.bundle import (
    BundlePredictionContext,
    LoadedBundle,
    ModelPreprocessEntry,
    predict_with_bundle,
)
from phenoradar.config import PredictConfig
from phenoradar.cv import apply_expression_transform
from phenoradar.figures import write_candidate_evidence_figures
from phenoradar.glmnet import GlmnetLogisticRegression
from phenoradar.missing_expression import NeutralStandardScaler
from phenoradar.predict_evidence import build_predict_evidence_artifacts


@pytest.mark.parametrize("transform", ["none", "log1p", "sample_rank", "sample_percentile_rank"])
def test_shared_prediction_context_avoids_repeated_input_and_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transform: str
) -> None:
    from polars.testing import assert_frame_equal

    config, bundle = _fixture(tmp_path, transform)
    config.data.metadata_path = None
    with Path(config.data.tpm_path).open("a") as handle:
        handle.write(
            "negative\tOG1\t1\nnegative\tOG2\t16\ncandidate\tEXTRA\t99999\n"
            "zpositive\tOG1\t32\nzpositive\tOG2\t1\n"
        )
    # Multiple members preserve their own feature order and ensemble probabilities.
    bundle = replace(bundle, models=bundle.models * 2, probability_aggregation="median")
    context = BundlePredictionContext()
    predictions, _ = predict_with_bundle(config, bundle, context=context)
    expected = build_predict_evidence_artifacts(
        config=config, bundle=bundle, predictions=predictions
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Shared evidence must not read input or call model prediction again")

    monkeypatch.setattr("phenoradar.predict_evidence.prepare_bundle_input", fail)
    monkeypatch.setattr("phenoradar.predict_evidence._predict_with_jobs", fail)
    actual = build_predict_evidence_artifacts(
        config=config, bundle=bundle, predictions=predictions, context=context
    )
    for name in ["candidates", "features", "reference_expression", "model_predictions"]:
        assert_frame_equal(
            getattr(actual, name), getattr(expected, name), rel_tol=1e-12, abs_tol=1e-14
        )
    assert actual.warnings == expected.warnings
    assert context.raw is not None and context.raw.shape == (3, 2)
    context.clear()
    assert context.raw is None and context.transformed is None and not context.model_probabilities


def test_bundle_input_prunes_extra_columns_but_validates_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phenoradar.bundle import BundleError
    from phenoradar.cv import ExpressionMatrixBuilder

    config, bundle = _fixture(tmp_path)
    with Path(config.data.tpm_path).open("a") as handle:
        handle.writelines(f"candidate\tEXTRA{i}\t2\n" for i in range(1000))
    original = ExpressionMatrixBuilder.build_matrix
    widths = []

    def build(self: object, species: list[str], feature_order: list[str] | None = None) -> object:
        assert feature_order == bundle.feature_names
        matrix, names = original(self, species, feature_order)
        widths.append(matrix.shape[1])
        return matrix, names

    monkeypatch.setattr(ExpressionMatrixBuilder, "build_matrix", build)
    _, warnings = predict_with_bundle(config, bundle)
    assert widths == [2]
    assert any("ignored 1000 features" in warning for warning in warnings)
    with Path(config.data.tpm_path).open("a") as handle:
        handle.write("candidate\tIGNORED_BAD\t-1\n")
    with pytest.raises(BundleError, match="IGNORED_BAD.*negative"):
        predict_with_bundle(config, bundle)


def _fixture(tmp_path: Path, transform: str = "log1p") -> tuple[PredictConfig, LoadedBundle]:
    tpm = tmp_path / "input.tsv"
    tpm.write_text("species\torthogroup\ttpm\ncandidate\tOG1\t16\ncandidate\tOG2\t1\n")
    config = PredictConfig.model_validate(
        {"data": {"tpm_path": str(tpm)}, "figures": {"top_features": 2}}
    )
    training = np.array([[1.0, 10.0], [2.0, 8.0], [8.0, 2.0], [9.0, 1.0]])
    scaler = StandardScaler().fit(apply_expression_transform(training, transform))
    model = GlmnetLogisticRegression()
    model.coef_ = np.array([1.5, -0.5])
    model.intercept_ = np.array([0.2])
    model.n_features_in_ = 2
    directory = tmp_path / "model_bundle"
    directory.mkdir()
    (directory / "resolved_config.yml").write_text("data:\n  trait_col: OtherTrait\n")
    features = ["OG1", "OG2"]
    bundle = LoadedBundle(
        bundle_dir=directory,
        manifest={},
        manifest_sha256="test",
        feature_names=features,
        transform_feature_names=features,
        scaler=scaler,
        model_preprocess=[ModelPreprocessEntry(feature_names=features, scaler=scaler)],
        models=[model],
        probability_aggregation="mean",
        threshold_fixed=0.5,
        source_run_id=tmp_path.name,
        expression_transform=transform,
        feature_scaling="standard",
    )
    return config, bundle


@pytest.mark.parametrize("transform", ["log1p", "sample_rank", "sample_percentile_rank"])
def test_evidence_reproduces_actual_model_score_and_ignores_extra_rank_features(
    tmp_path: Path,
    transform: str,
) -> None:
    config, bundle = _fixture(tmp_path, transform)
    with Path(config.data.tpm_path).open("a") as handle:
        handle.write("candidate\tEXTRA\t1000000\n")
    predictions, _ = predict_with_bundle(config, bundle)
    evidence = build_predict_evidence_artifacts(
        config=config, bundle=bundle, predictions=predictions
    )
    expected = bundle.scaler.transform(
        apply_expression_transform(np.array([[16.0, 1.0]]), transform)
    )
    observed = evidence.features.sort("feature")["contribution_mean"].to_numpy()
    np.testing.assert_allclose(observed, expected[0] * bundle.models[0].coef_)
    logit = float(observed.sum() + bundle.models[0].intercept_[0])
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), predictions["prob"][0])
    np.testing.assert_allclose(evidence.model_predictions["prob"], predictions["prob"])
    assert evidence.trait_name == "OtherTrait"
    assert evidence.reference_expression.is_empty()
    assert "no known-trait reference" in evidence.warnings[0]


def test_evidence_skips_abstained_positives_and_unsupported_models(tmp_path: Path) -> None:
    config, bundle = _fixture(tmp_path)
    prediction = pl.DataFrame(
        {
            "species": ["candidate"],
            "prob": [0.99],
            "pred_label_fixed_threshold": [1],
            "pred_label_selective": [None],
        }
    )
    evidence = build_predict_evidence_artifacts(
        config=config, bundle=bundle, predictions=prediction
    )
    assert evidence.features.is_empty() and not evidence.warnings
    prediction = prediction.drop("pred_label_selective")
    evidence = build_predict_evidence_artifacts(
        config=config,
        bundle=replace(bundle, models=[RandomForestClassifier()]),
        predictions=prediction,
    )
    assert evidence.features.is_empty()
    assert "unavailable" in evidence.warnings[0]


@pytest.mark.parametrize("missing_row", ["", "candidate\tOG2\t0\n"])
def test_neutral_missing_features_have_no_local_contribution(
    tmp_path: Path,
    missing_row: str,
) -> None:
    config, bundle = _fixture(tmp_path)
    Path(config.data.tpm_path).write_text(
        "species\torthogroup\ttpm\ncandidate\tOG1\t16\n" + missing_row
    )
    scaler = NeutralStandardScaler().fit(np.log1p(np.array([[1.0, 10.0], [9.0, 1.0]])))
    bundle = replace(
        bundle,
        scaler=scaler,
        model_preprocess=[ModelPreprocessEntry(["OG1", "OG2"], scaler)],
        absent_feature_fill="nan",
        zero_as_missing=True,
        missing_expression_method="neutral",
    )
    predictions, _ = predict_with_bundle(config, bundle)
    evidence = build_predict_evidence_artifacts(
        config=config, bundle=bundle, predictions=predictions
    )
    assert evidence.features["feature"].to_list() == ["OG1"]
    np.testing.assert_allclose(evidence.model_predictions["prob"], predictions["prob"])


def test_legacy_reference_fallback_records_source_and_missing_features(tmp_path: Path) -> None:
    config, bundle = _fixture(tmp_path)
    tables = tmp_path / "inference/tables"
    tables.mkdir(parents=True)
    path = tables / "candidate_reference_expression.tsv"
    path.write_text("species\tlabel\tfeature\ttpm\tlog2_tpm_plus1\nknown\t1\tOG1\t7\t3\n")
    predictions, _ = predict_with_bundle(config, bundle)
    evidence = build_predict_evidence_artifacts(
        config=config, bundle=bundle, predictions=predictions
    )
    assert evidence.reference_expression.height == 1
    assert evidence.input_paths == [path]
    assert "1 feature(s)" in evidence.warnings[0]


def test_predict_figure_labels_models_and_unavailable_references_honestly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, bundle = _fixture(tmp_path)
    predictions, _ = predict_with_bundle(config, bundle)
    evidence = build_predict_evidence_artifacts(
        config=config, bundle=bundle, predictions=predictions
    )
    texts: list[str] = []

    def capture(fig: object, _path: Path, *, title: str) -> None:
        for ax in fig.axes:
            texts.append(ax.get_title(loc="left"))
            texts.extend(text.get_text() for text in ax.texts)
        texts.extend(text.get_text() for text in fig.texts)

    monkeypatch.setattr("phenoradar.figures._save_pdf_figure", capture)
    manifest, warnings = write_candidate_evidence_figures(
        run_dir=tmp_path / "predict",
        candidates=evidence.candidates,
        features=evidence.features,
        reference_expression=evidence.reference_expression,
        cross_fold_predictions=evidence.model_predictions,
        trait_name=evidence.trait_name,
        evidence_context="predict",
    )
    assert not warnings
    assert manifest["n_bundle_models"].to_list() == [1]
    assert not any("cross_fold" in column for column in manifest.columns)
    assert "A   Prediction from fitted model" in texts
    assert "Reference unavailable" in texts
    assert not any("outer-CV" in text for text in texts)
