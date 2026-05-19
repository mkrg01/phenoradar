from __future__ import annotations

from pathlib import Path

import polars as pl

from phenoradar.tree_prediction import (
    build_contrast_pair_tree_annotation,
    build_cv_tree_prediction_annotation,
    build_external_tree_prediction_annotation,
    build_predict_tree_prediction_annotation,
    build_tree_feature_heatmap_annotation,
    write_run_tree_prediction_artifacts,
)


def _metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "species": ["sp1", "sp2", "sp3"],
            "C4": [0, 1, 0],
            "contrast_pair_id": ["g1", "g1", None],
        }
    )


def _thresholds() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "threshold_name": ["fixed_probability_threshold", "cv_derived_threshold"],
            "threshold_value": [0.5, 0.7],
        }
    )


def _feature_importance() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["OG2", "OG1", "OG3"],
            "importance_mean": [0.9, 0.5, 0.1],
            "importance_std": [0.0, 0.0, 0.0],
            "n_models": [1, 1, 1],
            "method": ["coef_abs_l1_norm", "coef_abs_l1_norm", "coef_abs_l1_norm"],
        }
    )


def _coefficients() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["OG1", "OG2"],
            "coef_mean": [0.2, -0.4],
            "coef_std": [0.0, 0.0],
            "n_models": [1, 1],
            "method": ["coef_signed", "coef_signed"],
            "reason": ["NA", "NA"],
        }
    )


def _write_tpm(path: Path) -> None:
    pl.DataFrame(
        {
            "species": ["sp1", "sp1", "sp2", "sp2", "sp3", "sp3"],
            "orthogroup": ["OG1", "OG2", "OG1", "OG2", "OG1", "OG2"],
            "tpm": [0.0, 3.0, 3.0, 15.0, 7.0, 0.0],
        }
    ).write_csv(path, separator="\t")


def test_build_contrast_pair_tree_annotation_filters_to_grouped_species() -> None:
    metadata = _metadata().with_columns(pl.col("C4").alias("true_label"))

    annotation = build_contrast_pair_tree_annotation(
        metadata=metadata,
        group_col="contrast_pair_id",
    )

    assert annotation.to_dicts() == [
        {
            "label": "sp1",
            "species": "sp1",
            "true_label": 0,
            "contrast_pair_id": "g1",
        },
        {
            "label": "sp2",
            "species": "sp2",
            "true_label": 1,
            "contrast_pair_id": "g1",
        },
    ]


def test_build_tree_feature_heatmap_annotation_outputs_log2_and_zscore(tmp_path: Path) -> None:
    tpm_path = tmp_path / "tpm.tsv"
    _write_tpm(tpm_path)
    metadata = _metadata().with_columns(pl.col("C4").alias("true_label"))

    annotation = build_tree_feature_heatmap_annotation(
        metadata=metadata,
        tpm_path=tpm_path,
        species_col="species",
        feature_col="orthogroup",
        value_col="tpm",
        group_col="contrast_pair_id",
        feature_importance=_feature_importance(),
        coefficients=_coefficients(),
        feature_limit=2,
    )

    assert annotation.select("feature").unique().sort("feature").to_series().to_list() == [
        "OG1",
        "OG2",
    ]
    assert annotation.height == 4
    first = annotation.filter((pl.col("species") == "sp1") & (pl.col("feature") == "OG2"))
    assert first.select("feature_rank").item() == 1
    assert first.select("log2_tpm_plus1").item() == 2.0
    assert first.select("coef_mean").item() == -0.4
    zscore_sum = (
        annotation.filter(pl.col("feature") == "OG2").select("z_score_log2_tpm").sum().item()
    )
    assert zscore_sum == 0.0


def test_build_cv_tree_prediction_annotation_filters_to_contrast_pairs() -> None:
    annotation = build_cv_tree_prediction_annotation(
        metadata=_metadata(),
        oof_predictions=pl.DataFrame(
            {
                "fold_id": ["0", "0", "1"],
                "species": ["sp1", "sp2", "sp3"],
                "label": [0, 1, 0],
                "prob": [0.2, 0.8, 0.9],
            }
        ),
        thresholds=_thresholds(),
        group_col="contrast_pair_id",
    )

    assert annotation.select("species").to_series().to_list() == ["sp1", "sp2"]
    assert annotation.select("pred_label").to_series().to_list() == [0, 1]
    assert annotation.columns == [
        "label",
        "species",
        "true_label",
        "prob",
        "pred_label",
        "uncertainty_std",
        "contrast_pair_id",
        "fold_id",
    ]


def test_build_external_tree_prediction_annotation_keeps_external_species() -> None:
    annotation = build_external_tree_prediction_annotation(
        metadata=_metadata(),
        pred_external_test=pl.DataFrame(
            {
                "species": ["sp3"],
                "true_label": [0],
                "prob": [0.9],
                "pred_label_fixed_threshold": [1],
                "pred_label_cv_derived_threshold": [1],
            }
        ),
        group_col="contrast_pair_id",
    )

    assert annotation.to_dicts() == [
        {
            "label": "sp3",
            "species": "sp3",
            "true_label": 0,
            "prob": 0.9,
            "pred_label": 1,
            "uncertainty_std": None,
            "contrast_pair_id": None,
        }
    ]


def test_build_predict_tree_prediction_annotation_preserves_both_prediction_labels() -> None:
    annotation = build_predict_tree_prediction_annotation(
        metadata=_metadata(),
        pred_predict=pl.DataFrame(
            {
                "species": ["sp1", "sp4"],
                "true_label": [0, None],
                "prob": [0.2, 0.6],
                "pred_label_fixed_threshold": [0, 1],
                "pred_label_cv_derived_threshold": [0, 0],
            }
        ),
        group_col="contrast_pair_id",
    )

    assert annotation.select("species").to_series().to_list() == ["sp1", "sp4"]
    assert annotation.select("pred_label_fixed_threshold").to_series().to_list() == [0, 1]
    assert annotation.select("pred_label_cv_derived_threshold").to_series().to_list() == [0, 0]


def test_write_run_tree_prediction_artifacts_writes_annotation_without_tree_extra(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.tsv"
    tpm_path = tmp_path / "tpm.tsv"
    tree_path = tmp_path / "tree.nwk"
    _metadata().write_csv(metadata_path, separator="\t")
    _write_tpm(tpm_path)
    tree_path.write_text("(sp1,sp2,sp3);\n", encoding="utf-8")
    figures_dir = tmp_path / "run" / "figures"
    figures_dir.mkdir(parents=True)

    warnings = write_run_tree_prediction_artifacts(
        run_dir=tmp_path / "run",
        tree_path=tree_path,
        metadata_path=metadata_path,
        tpm_path=tpm_path,
        species_col="species",
        feature_col="orthogroup",
        value_col="tpm",
        trait_col="C4",
        group_col="contrast_pair_id",
        oof_predictions=pl.DataFrame(
            {
                "fold_id": ["0", "0", "1"],
                "species": ["sp1", "sp2", "sp3"],
                "label": [0, 1, 0],
                "prob": [0.2, 0.8, 0.9],
            }
        ),
        thresholds=_thresholds(),
        feature_importance=_feature_importance(),
        coefficients=_coefficients(),
        pred_external_test=None,
    )

    assert (tmp_path / "run" / "tree_prediction_cv_annotation.tsv").exists()
    assert (tmp_path / "run" / "tree_contrast_pairs_annotation.tsv").exists()
    assert (tmp_path / "run" / "tree_feature_heatmap_annotation.tsv").exists()
    if not (figures_dir / "tree_prediction_cv.svg").exists():
        assert any("Toytree is unavailable" in warning for warning in warnings)
