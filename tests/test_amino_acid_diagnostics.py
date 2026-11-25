from pathlib import Path

import numpy as np
import polars as pl

from phenoradar.amino_acid_diagnostics import (
    site_score_summary,
    write_frequency_by_confusion,
    write_site_filter_funnel,
)


def test_site_scores_weight_folds_equally_and_keep_absent_sites_as_zero() -> None:
    selections = pl.DataFrame(
        {
            "scope": [
                "outer_fold_1",
                "outer_fold_1",
                "outer_fold_2",
                "outer_fold_3",
                "final_refit",
            ],
        }
    )
    source = pl.DataFrame(
        {
            "scope": [
                "outer_fold_1",
                "outer_fold_1",
                "outer_fold_3",
                "outer_fold_1",
                "outer_fold_2",
                "final_refit",
            ],
            "orthogroup": ["OG1", "OG1", "OG1", "OG2", "OG2", "OG3"],
            "alignment_pos_1based": [1] * 6,
            "pss": [2.0, 4.0, 9.0, 2.0, 3.0, 1000.0],
        }
    ).lazy()
    top, folds = site_score_summary(source, selections, "pss", 10)
    assert top["orthogroup"].to_list() == ["OG1", "OG2"]
    np.testing.assert_allclose(top["pss"], [4.0, 4 / 3])
    assert folds.height == 6
    assert (
        folds.filter((pl.col("orthogroup") == "OG1") & (pl.col("scope") == "outer_fold_2"))[
            "pss"
        ].item()
        == 0
    )
    assert (
        top.join(
            folds.group_by("orthogroup").agg(pl.col("pss").mean().alias("mean")), on="orthogroup"
        )
        .select((pl.col("pss") == pl.col("mean")).all())
        .item()
    )


def test_zero_score_summary_is_empty_but_has_schema() -> None:
    source = pl.DataFrame(
        {
            "scope": ["outer_fold_1"],
            "orthogroup": ["OG1"],
            "alignment_pos_1based": [1],
            "pss": [0.0],
        }
    ).lazy()
    top, folds = site_score_summary(source, pl.DataFrame({"scope": ["outer_fold_1"]}), "pss", 5)
    assert top.is_empty() and folds.is_empty()
    assert "pss" in top.columns and "n_models" in folds.columns


def test_site_funnel_includes_ranking_and_zero_retained_folds(tmp_path: Path) -> None:
    cv = tmp_path / "cv/tables"
    cv.mkdir(parents=True)
    pl.DataFrame(
        {
            "scope": ["outer_fold_1", "outer_fold_2", "final_refit"],
            "sample_set_id": [0, 0, 0],
            "n_columns": [100, 100, 10000],
            "n_after_call_filter": [80, 80, 9000],
            "n_after_variant_filter": [50, 50, 5000],
        }
    ).write_csv(cv / "amino_acid_og_qc.tsv", separator="\t")
    pl.DataFrame(
        {
            "scope": ["outer_fold_1", "outer_fold_1", "outer_fold_2", "final_refit"],
            "sample_set_id": [0] * 4,
            "n_active_states": [2, 3, 0, 4],
        }
    ).write_csv(cv / "amino_acid_site_qc.tsv", separator="\t")
    write_site_filter_funnel(tmp_path)
    path = tmp_path / "cv/figures/amino_acid_site_filter_funnel.svg"
    assert "Rank-filtered sites" not in path.read_text()
    assert "Model input sites" not in path.read_text()
    write_site_filter_funnel(tmp_path, ranking_enabled=True)
    assert "Rank-filtered sites" in path.read_text()
    assert "Model input sites" not in path.read_text()
    table = pl.read_csv(cv / "amino_acid_site_filter_counts_summary.tsv", separator="\t")
    retained = table.filter(pl.col("stage") == "retained_sites")
    assert retained["n_records"].item() == 2
    assert retained["n_features_median"].item() == 1
    assert retained["n_features_min"].item() == 0
    assert retained["n_features_max"].item() == 2
    assert table.filter(pl.col("stage") == "alignment_sites")["n_features_median"].item() == 100


def test_empty_accepted_diagnostics_produce_explanatory_figures(tmp_path: Path) -> None:
    cv = tmp_path / "cv/tables"
    cv.mkdir(parents=True)
    pl.DataFrame({"species": ["a"], "feature": ['["OG1",1,"A"]'], "frequency": [1.0]}).write_csv(
        cv / "top_amino_acid_frequencies.tsv", separator="\t"
    )
    predictions = pl.DataFrame(
        {"species": ["a"], "label": [1], "prob": [0.8], "decision_status": ["abstained"]}
    )
    write_frequency_by_confusion(tmp_path, predictions, 20)
    accepted = tmp_path / "cv/figures/accepted_only"
    for stem in ["amino_acid_frequency_by_confusion"]:
        assert "No accepted species" in (accepted / f"{stem}.svg").read_text()
    assert (accepted / "amino_acid_frequency_by_confusion.pdf").stat().st_size > 0
