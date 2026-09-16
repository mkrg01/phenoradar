from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from phenoradar.abstention import (
    abstention_summary,
    annotate_abstention,
    write_abstention_artifacts,
)
from phenoradar.bundle import export_model_bundle, load_model_bundle, predict_with_bundle
from phenoradar.cli import app
from phenoradar.config import AppConfig, write_resolved_config
from phenoradar.cv import (
    CVError,
    _preprocess_train_and_target,
    run_final_refit,
    run_outer_cv,
)
from phenoradar.missing_expression import NeutralStandardScaler
from phenoradar.split import build_split_artifacts


def neutral_config(**updates: object) -> AppConfig:
    data: dict[str, object] = {
        "preprocess": {
            "absent_feature_fill": "nan",
            "missing_expression": {"method": "neutral", "zero_as_missing": True},
            "sparse_feature_filter": {"enabled": False},
        },
        "abstention": {"enabled": True},
        "sampling": {
            "strategy": "all_samples",
            "sampled_set_count": 1,
            "max_samples_per_label_per_group": None,
        },
    }
    data.update(updates)
    return AppConfig.model_validate(data)


def test_observed_only_scaling_and_target_do_not_change_training_reference() -> None:
    config = neutral_config()
    train = np.array([[1.0, 8.0, 4.0, np.nan], [3.0, 0.0, 4.0, np.nan], [0.0, 2.0, 4.0, np.nan]])
    target = np.array([[0.0, 4.0, 100.0, 1.0], [3.0, 0.0, 2.0, 5.0]])
    scaled, transformed, names, scaler = _preprocess_train_and_target(
        config, train, target, ["a", "b", "constant", "absent"]
    )
    assert names == ["a", "b"]
    assert isinstance(scaler, NeutralStandardScaler)
    np.testing.assert_allclose(
        scaler.mean_, [np.log1p([1.0, 3.0]).mean(), np.log1p([8.0, 2.0]).mean()]
    )
    assert scaled[2, 0] == scaled[1, 1] == transformed[0, 0] == transformed[1, 1] == 0
    np.testing.assert_allclose(scaled[:, 0], [-1, 1, 0])
    assert train[2, 0] == 0  # no mutation of raw evidence
    with pytest.raises(CVError, match="removed all features"):
        _preprocess_train_and_target(config, np.zeros((3, 2)), np.zeros((1, 2)), ["a", "b"])


def test_mean_observation_is_not_missing_and_boundary_is_inclusive() -> None:
    predictions = pl.DataFrame(
        {"species": ["missing", "observed", "below"], "prob": [0.99, 0.99, 0.99]}
    )
    result = annotate_abstention(
        predictions,
        species=["observed", "missing", "below"],
        matrix=np.array([[5.0, 2.0], [np.nan, 2.0], [5.0, np.nan]]),
        feature_names=["a", "b"],
        model_coefficients=[(["a", "b"], np.array([-2.0, 8.0]))],
        zero_as_missing=True,
        threshold=0.8,
    ).sort("species")
    assert result["information_coverage"].to_list() == pytest.approx([0.2, 0.8, 1.0])
    assert result["pred_label_selective"].to_list() == [None, 1, 1]
    assert result["decision_status"].to_list() == ["abstained", "accepted", "accepted"]


def test_model_weights_do_not_cancel_and_intercept_only_models_count_as_zero() -> None:
    predictions = pl.DataFrame({"species": ["x"], "prob": [0.99]})
    arguments = dict(
        predictions=predictions,
        species=["x"],
        matrix=np.array([[0.0, 2.0]]),
        feature_names=["a", "b"],
        zero_as_missing=True,
        threshold=0.8,
    )
    result = annotate_abstention(
        **arguments,
        model_coefficients=[
            (["a"], np.array([1.0])),
            (["a"], np.array([-1.0])),
            (["b"], np.array([0.0])),
        ],
    )
    assert result["information_coverage"].item() == 0
    assert result["abstention_reason"].item() == "insufficient_information"
    intercept = annotate_abstention(
        **arguments,
        model_coefficients=[
            (["a"], np.array([0.0])),
        ],
    )
    assert intercept["abstention_reason"].item() == "no_informative_coefficients"
    observed_zero = annotate_abstention(
        **{**arguments, "zero_as_missing": False},
        model_coefficients=[(["a"], np.array([1.0]))],
    )
    assert observed_zero["decision_status"].item() == "accepted"


def test_all_abstained_evaluation_is_undefined_and_evidence_is_readable(tmp_path: Path) -> None:
    pred = annotate_abstention(
        pl.DataFrame({"species": ["a", "b"], "prob": [0.99, 0.01], "true_label": [0, 1]}),
        species=["a", "b"],
        matrix=np.zeros((2, 1)),
        feature_names=["OG1"],
        model_coefficients=[(["OG1"], np.array([-2.0]))],
        zero_as_missing=True,
        threshold=0.8,
    )
    summary = abstention_summary(pred).filter(pl.col("scope") == "overall")
    assert summary["n_abstained"].item() == 2
    assert summary["n_evaluated"].item() == 0
    assert summary["error_rate"].item() is None
    assert summary["decision_rate"].item() == 0
    write_abstention_artifacts(pred, tmp_path)
    evidence = pl.read_csv(tmp_path / "abstention_features.tsv", separator="\t")
    assert evidence["coefficient_fraction"].to_list() == [1.0, 1.0]


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_fixed_threshold(threshold: object) -> None:
    with pytest.raises(ValidationError):
        neutral_config(abstention={"enabled": True, "threshold": threshold})


def test_neutral_mode_contract() -> None:
    with pytest.raises(ValidationError, match="require"):
        AppConfig.model_validate({"abstention": {"enabled": True}})
    for key, value in [
        ("model", {"name": "random_forest"}),
        ("preprocess", {"missing_expression": {"method": "neutral"}}),
    ]:
        with pytest.raises(ValidationError, match="requires"):
            neutral_config(**{key: value})


@pytest.mark.parametrize("method", ["pair_aware", "unpaired"])
def test_supervised_filtering_does_not_treat_missing_label_as_low_expression(method: str) -> None:
    config = neutral_config()
    config.preprocess.ranked_feature_filter.method = method
    config.preprocess.ranked_feature_filter.max_features = 3
    config.preprocess.ranked_feature_filter.min_contrast_pairs = 2
    # one_label has observations only in label 1; one_pair has one valid contrast.
    train = np.array([[1, 0, 1], [8, 8, 8], [2, 0, 0], [9, 9, 9]], dtype=float)
    _, _, names, _ = _preprocess_train_and_target(
        config,
        train,
        train,
        ["observed", "one_label", "one_pair"],
        y_train=np.array([0, 1, 0, 1]),
        groups_train=np.array(["a", "a", "b", "b"]),
    )
    assert "observed" in names
    assert "one_label" not in names
    assert ("one_pair" in names) == (method == "unpaired")


@pytest.mark.parametrize("prune", [True, False])
def test_cv_refit_bundle_with_real_missingness(tmp_path: Path, prune: bool) -> None:
    metadata = ["species\tC4\tcontrast_pair_id"]
    expression = ["species\torthogroup\ttpm"]
    for group in range(4):
        for label in (0, 1):
            species = f"s{group}_{label}"
            metadata.append(f"{species}\t{label}\tg{group}")
            down = 1 + group / 10 if label else 10 + group
            up = 10 + group if label else 1 + group / 10
            # Natural missing entries; one explicit zero and one absent coordinate.
            expression.append(f"{species}\tDOWN\t{0 if group == label == 0 else down}")
            if not (group == 1 and label == 0):
                expression.append(f"{species}\tUP\t{up}")
            expression.append(f"{species}\tCONSTANT\t7")
    metadata.extend(["external\t1\t", "missing\t\t"])
    expression.extend(
        [
            "external\tDOWN\t1.1",
            "external\tUP\t11",
            "missing\tDOWN\t0",
            "missing\tUP\t0",
            "missing\tCONSTANT\t7",
        ]
    )
    metadata_path = tmp_path / "metadata.tsv"
    tpm_path = tmp_path / "tpm.tsv"
    metadata_path.write_text("\n".join(metadata) + "\n")
    tpm_path.write_text("\n".join(expression) + "\n")
    config = neutral_config(
        data={"metadata_path": str(metadata_path), "tpm_path": str(tpm_path)},
        runtime={"execution_stage": "full_run", "n_jobs": 2},
        model_selection={
            "selected_candidate_count": 1,
            "inner_cv_strategy": "group_kfold",
            "inner_cv_n_splits": 2,
            "search_space": {"lambda": [0.01, 0.1]},
        },
    )
    config.preprocess.sparse_feature_filter.enabled = prune
    config.preprocess.sparse_feature_filter.scope = "all_samples"
    config.preprocess.sparse_feature_filter.min_nonzero_fraction = 0.5
    split = build_split_artifacts(config)
    cv = run_outer_cv(config, split.split_manifest)
    assert cv.oof_predictions["information_coverage"].is_not_null().all()
    assert cv.inference_predictions_by_fold is not None
    assert cv.inference_predictions_by_fold["decision_status"].unique().to_list() == ["abstained"]
    assert {
        "validation_abstention", "inference_abstention",
        "inference_preprocessing", "inference_prediction",
    }.issubset(set(cv.timing["stage"]))
    refit = run_final_refit(config, split.split_manifest)
    assert refit.pred_inference["information_coverage"].item() == 0
    assert refit.pred_inference["pred_label_selective"].item() is None
    assert {"external_test_abstention", "inference_abstention"}.issubset(
        set(refit.timing["stage"])
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_resolved_config(config, run_dir / "resolved_config.yml")
    exported = export_model_bundle(
        run_dir=run_dir,
        resolved_config_path=run_dir / "resolved_config.yml",
        config=config,
        final_refit_artifacts=refit,
        thresholds=cv.thresholds,
    )
    bundle = load_model_bundle(exported.bundle_dir)
    # Predict config must not override bundled missingness or gate policies.
    predict_config = AppConfig.model_validate({"data": config.data.model_dump()})
    predicted, _ = predict_with_bundle(predict_config, bundle)
    for expected in (refit.pred_external_test, refit.pred_inference):
        actual = predicted.filter(pl.col("species").is_in(expected["species"].to_list()))
        np.testing.assert_allclose(actual["prob"], expected["prob"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual["information_coverage"], expected["information_coverage"])
        assert actual["pred_label_selective"].to_list() == (
            expected["pred_label_selective"].to_list()
        )
    # No model features in the entire prediction input is still a valid abstention.
    tpm_path.write_text("species\torthogroup\ttpm\nmissing\tUNRELATED\t3\n")
    metadata_path.write_text("species\nmissing\n")
    predicted, _ = predict_with_bundle(predict_config, bundle)
    assert predicted["decision_status"].item() == "abstained"
    if not prune:
        # Full refit with no target species must still produce writable empty schemas.
        metadata_path.write_text("\n".join(metadata[:-2]) + "\n")
        tpm_path.write_text("\n".join(expression) + "\n")
        split = build_split_artifacts(config)
        no_targets = run_final_refit(config, split.split_manifest)
        assert no_targets.pred_inference.height == no_targets.pred_external_test.height == 0
        assert "pred_label_selective" in no_targets.pred_inference.columns
        write_abstention_artifacts(no_targets.pred_inference, tmp_path / "empty")


def test_cli_run_and_predict_preserve_abstention_and_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    metadata = ["species\tC4\tcontrast_pair_id"]
    expression = ["species\torthogroup\ttpm"]
    species_names = []
    for group in range(3):
        for label in (0, 1):
            species = f"s{group}_{label}"
            species_names.append(species)
            metadata.append(f"{species}\t{label}\tg{group}")
            down = 0 if group == label == 0 else 1 + group if label else 10 + group
            expression.append(f"{species}\tDOWN\t{down}")
            expression.append(f"{species}\tUP\t{10 + group if label else 1 + group}")
    for species, label in [("external", "1"), ("missing", "")]:
        species_names.append(species)
        metadata.append(f"{species}\t{label}\t")
        expression.append(f"{species}\tDOWN\t0")  # UP is completely absent.
    metadata_path = tmp_path / "metadata.tsv"
    tpm_path = tmp_path / "tpm.tsv"
    tree_path = tmp_path / "tree.nwk"
    metadata_path.write_text("\n".join(metadata) + "\n")
    tpm_path.write_text("\n".join(expression) + "\n")
    tree_path.write_text("(" + ",".join(f"{s}:1" for s in species_names) + ");\n")
    config = neutral_config(
        data={
            "metadata_path": str(metadata_path),
            "tpm_path": str(tpm_path),
            "tree_path": str(tree_path),
        },
        runtime={"execution_stage": "full_run"},
        figures={"top_features": 2},
        summary={"group_col": "contrast_pair_id"},
    )
    config_path = tmp_path / "config.yml"
    write_resolved_config(config, config_path)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(config_path)])
    assert result.exit_code == 0, (result.output, result.exception)
    run_dir = next((tmp_path / "runs").glob("*_run_*"))
    for relative in (
        "cv/figures/cv_metrics_overview_accepted_only.svg",
        "cv/figures/roc_curve_cv_accepted_only.svg",
        "external_test/figures/external_confusion_matrix_accepted_only.svg",
        "external_test/figures/cv_external_metric_comparison_accepted_only.svg",
        "external_test/figures/tree_prediction_external_accepted_only.svg",
        "inference/figures/inference_probability_distribution_accepted_only.svg",
    ):
        assert (run_dir / relative).exists(), relative
    heatmap = pl.read_csv(
        run_dir / "cv" / "tables" / "tree_feature_heatmap_annotation.tsv",
        separator="\t",
        null_values="NA",
    )
    missing_expression = heatmap.filter(
        (pl.col("species") == "s0_0") & (pl.col("feature") == "DOWN")
    )
    assert missing_expression["is_missing"].item()
    assert missing_expression["log2_tpm_plus1"].item() is None
    assert missing_expression["z_score_log2_tpm"].item() is None
    external_tree = pl.read_csv(
        run_dir / "external_test" / "tables" / "tree_prediction_external_annotation.tsv",
        separator="\t",
        null_values="NA",
    )
    assert external_tree["decision_status"].item() == "abstained"
    assert external_tree["pred_label"].item() is None
    for stage, filename in [
        ("external_test", "prediction_external_test.tsv"),
        ("inference", "prediction_inference.tsv"),
    ]:
        tables = run_dir / stage / "tables"
        predictions = pl.read_csv(tables / filename, separator="\t", null_values="NA")
        assert predictions["decision_status"].item() == "abstained"
        assert predictions["pred_label_selective"].item() is None
        summary = pl.read_csv(tables / "abstention_summary.tsv", separator="\t", null_values="NA")
        overall = summary.filter(pl.col("scope") == "overall")
        assert overall["n_accepted"].item() == 0
        assert overall["error_rate"].item() is None
        grouped = pl.read_csv(
            tables / "group_summary_contrast_pair_id.tsv", separator="\t", null_values="NA"
        )
        assert grouped["n_abstained"].sum() == 1
        assert grouped["n_pred_positive"].sum() == 0
    assert not list((run_dir / "inference" / "figures").glob("candidate_evidence_*.pdf"))
    result = runner.invoke(
        app, ["predict", "-c", str(config_path), "--model-bundle", str(run_dir / "model_bundle")]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    predict_dir = next((tmp_path / "runs").glob("*_predict_*"))
    assert (
        predict_dir / "inference/figures/predict_probability_distribution_accepted_only.svg"
    ).exists()
    predictions = pl.read_csv(
        predict_dir / "inference" / "tables" / "prediction_inference.tsv",
        separator="\t",
        null_values="NA",
    )
    missing = predictions.filter(pl.col("species") == "missing")
    assert missing["decision_status"].item() == "abstained"
    assert missing["information_coverage"].item() == 0
