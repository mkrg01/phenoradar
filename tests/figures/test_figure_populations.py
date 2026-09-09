from pathlib import Path
from xml.etree import ElementTree as ET

import polars as pl
import pytest

from phenoradar import figures
from phenoradar.tree_prediction import _write_tree_prediction_svg


def predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "species": ["tp", "fp", "tn", "fn"],
            "true_label": [1, 0, 0, 1],
            "prob": [0.9, 0.8, 0.1, 0.2],
            "pred_label_fixed_threshold": [1, 1, 0, 0],
            "pred_label_selective": [1, None, 0, None],
            "decision_status": ["accepted", "abstained", "accepted", "abstained"],
        }
    )


def text(path: Path) -> set[str]:
    return {
        "".join(e.itertext()) for e in ET.parse(path).getroot().iter() if e.tag.endswith("text")
    }


def test_confusion_and_comparison_evaluate_the_same_population(tmp_path: Path) -> None:
    external = predictions()
    original = external.clone()
    cv = external.rename({"true_label": "label"}).with_columns(pl.lit("0").alias("fold_id"))
    figures._external_confusion_matrix(external, tmp_path / "confusion.svg")
    figures._cv_external_comparison_by_population(cv, external, tmp_path / "comparison.svg")
    assert external.equals(original)
    for suffix, count, precision in (("", 4, "0.500"), ("_accepted_only", 2, "1.000")):
        confusion = text(tmp_path / f"confusion{suffix}.svg")
        comparison = text(tmp_path / f"comparison{suffix}.svg")
        assert f"n = {count}" in confusion
        assert f"Precision = {precision}" in confusion
        assert precision in comparison
        assert not any("abstained" in value for value in confusion | comparison)


@pytest.mark.parametrize("all_abstained", [False, True])
def test_accepted_curves_explain_undefined_populations(tmp_path: Path, all_abstained: bool) -> None:
    frame = predictions().with_columns(
        pl.when((pl.col("true_label") == 1) & pl.lit(not all_abstained))
        .then(pl.lit("accepted"))
        .otherwise(pl.lit("abstained"))
        .alias("decision_status")
    )
    figures._external_roc_pr_curves(
        frame,
        roc_out_path=tmp_path / "roc.svg",
        pr_out_path=tmp_path / "pr.svg",
    )
    for name in ("roc", "pr"):
        assert (tmp_path / f"{name}.svg").exists()
        messages = " ".join(text(tmp_path / f"{name}_accepted_only.svg"))
        assert ("No accepted species" if all_abstained else "fewer than two labels") in messages
    figures._external_confusion_matrix(frame, tmp_path / "confusion.svg")
    if all_abstained:
        assert "Accuracy = 1.000" not in text(tmp_path / "confusion_accepted_only.svg")


def test_histogram_and_group_counts_exclude_abstained_species(tmp_path: Path) -> None:
    frame = predictions().with_columns(
        pl.lit("family").alias("group_id"), pl.lit("Family").alias("group_name")
    )
    figures._predict_probability_distribution(frame, tmp_path / "hist.svg")
    figures.write_group_probability_figure(
        grouped_predictions=frame,
        out_path=tmp_path / "group.svg",
        group_label="Family",
        source_table_name="prediction.tsv",
        figure_name="group.svg",
    )
    assert "Family (n=4)" in text(tmp_path / "group.svg")
    assert "Family (n=2)" in text(tmp_path / "group_accepted_only.svg")


@pytest.mark.parametrize("enabled", [False, True])
def test_pair_generation_depends_on_policy_not_rejection_count(
    tmp_path: Path, enabled: bool
) -> None:
    frame = predictions().with_columns(pl.lit("accepted").alias("decision_status"))
    if not enabled:
        frame = frame.drop("decision_status", "pred_label_selective")
    figures._predict_probability_distribution(frame, tmp_path / "hist.svg")
    accepted = tmp_path / "hist_accepted_only.svg"
    assert accepted.exists() == enabled
    if enabled:
        assert text(accepted) == text(tmp_path / "hist.svg")


def test_population_figures_keep_original_canvas_without_subtitles(tmp_path: Path) -> None:
    frame = predictions()
    figures._predict_probability_distribution(
        frame.drop("decision_status", "pred_label_selective"), tmp_path / "baseline.svg"
    )
    figures._predict_probability_distribution(frame, tmp_path / "hist.svg")
    baseline = ET.parse(tmp_path / "baseline.svg").getroot()
    for name in ("hist.svg", "hist_accepted_only.svg"):
        root = ET.parse(tmp_path / name).getroot()
        for attribute in ("viewBox", "width", "height"):
            assert root.attrib[attribute] == baseline.attrib[attribute]
        assert not any(e.get("id") == "phenoradar-population" for e in root.iter())
        assert not any(
            phrase in value
            for value in text(tmp_path / name)
            for phrase in ("All species", "Accepted only", "abstained=")
        )


def test_expression_confusion_uses_raw_decisions_and_external_axis(tmp_path: Path) -> None:
    frame = predictions()
    expression = pl.DataFrame(
        {"species": frame["species"], "feature": ["OG"] * 4, "tpm": [1.0, 2.0, 3.0, 4.0]}
    )
    figures._top_feature_expression_by_confusion(
        oof_predictions=frame,
        top_feature_expression=expression,
        feature_importance=pl.DataFrame({"feature": ["OG"], "importance_mean": [1.0]}),
        coefficients=pl.DataFrame(
            {"feature": ["OG"], "coef_mean": [1.0], "method": ["coef_signed"]}
        ),
        out_path=tmp_path / "expression.svg",
        label_col="true_label",
    )
    for suffix in ("", "_accepted_only"):
        values = text(tmp_path / f"expression{suffix}.svg")
        assert "External-test confusion group" in values
        assert "OOF confusion group" not in values
        assert "FP" in values
        assert ("n=0" in values) == bool(suffix)


def test_tree_population_filters_tips_and_restores_raw_labels(tmp_path: Path) -> None:
    tree = tmp_path / "tree.nwk"
    tree.write_text("((tp,fp),(tn,fn));")
    frame = predictions().with_columns(pl.col("pred_label_selective").alias("pred_label"))
    warnings = _write_tree_prediction_svg(
        tree_path=tree,
        annotation=frame,
        out_path=tmp_path / "tree.svg",
        title="",
        tracks=["true_label", "prob", "pred_label"],
    )
    assert warnings == []
    assert {"tp", "fp", "tn", "fn"} <= text(tmp_path / "tree.svg")
    accepted = text(tmp_path / "tree_accepted_only.svg")
    assert {"tp", "tn"} <= accepted
    assert "fp" not in accepted and "fn" not in accepted
    assert "abstained" not in text(tmp_path / "tree.svg")
