from pathlib import Path

import numpy as np
import polars as pl
from test_amino_acid import settings
from test_amino_acid_streaming import model

from phenoradar.amino_acid import AASite, AATransform
from phenoradar.amino_acid_composition import write_group_trait_composition
from phenoradar.amino_acid_run import AAMember
from phenoradar.amino_acid_stage_figures import final_model_sites
from phenoradar.cv_display import with_prediction_context
from phenoradar.model_selection import Candidate


def test_final_sites_average_pss_before_ranking_without_coefficient_cancellation() -> None:
    sites = [AASite("OG1", j, 1, 0, 0, 0, known=np.arange(20) < 2,
                    active=np.arange(20) < 2) for j in range(2)]
    members = [AAMember(AATransform(settings(), chosen, []), model(beta), "final_refit", i,
                        Candidate(i, {})) for i, (chosen, beta) in enumerate([
        (sites, np.array([2., -2., 3., -3.])), (sites[:1], np.array([-2., 2.])),
    ])]
    actual = final_model_sites(members, 20)
    assert actual["alignment_pos_1based"].to_list() == [1, 2]
    assert actual["pss"].to_list() == [4., 3.]


def test_unknown_prediction_context_uses_no_synthetic_traits_groups_or_folds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "species.tsv"
    path.write_text("species\nx\ny\n")
    predictions = pl.DataFrame({"species": ["x", "y"], "prob": [.1, .9],
                                "fold_id": ["bad", "bad"]})
    context = with_prediction_context(predictions, metadata_path=path, species_col="species",
                                      group_col="family", trait_col="trait")
    assert context["group_id"].null_count() == 2 and context["true_label"].null_count() == 2
    assert "fold_id" not in context.columns
    assert context["prob"].to_list() == [.1, .9]


def test_external_group_trait_composition_has_final_prediction_labels(tmp_path: Path) -> None:
    from test_amino_acid_composition import inputs

    store, sites, context, annotations = inputs(tmp_path)
    with store:
        write_group_trait_composition(
            tmp_path, store, sites, context, aggregation="aa_frequency",
            annotations=annotations, stage="external_test", trait_name="C4",
        )
    table = pl.read_csv(tmp_path / "external_test/tables/amino_acid_group_trait_composition.tsv",
                        separator="\t")
    assert "mean_probability" in table.columns and "mean_oof_probability" not in table.columns
    svg = (tmp_path / "external_test/figures/amino_acid_group_trait_composition.svg").read_text()
    assert "Mean p" in svg and "OOF" not in svg and ">C4<" in svg


def test_external_composition_without_groups_uses_all_observed_species(tmp_path: Path) -> None:
    from test_amino_acid_composition import inputs

    store, sites, context, annotations = inputs(tmp_path)
    context = context.drop("group_id")
    with store:
        write_group_trait_composition(tmp_path, store, sites, context, aggregation="aa_frequency",
                                      annotations=annotations, stage="external_test")
    table = pl.read_csv(tmp_path / "external_test/tables/amino_acid_group_trait_composition.tsv",
                        separator="\t")
    rows = table.select("group_id", "trait", "n_species").unique().sort("trait")
    assert rows["group_id"].to_list() == ["All", "All"]
    assert rows["n_species"].to_list() == [3, 2]
    assert table["grouping"].unique().to_list() == ["all_species"]
    svg = (tmp_path / "external_test/figures/amino_acid_group_trait_composition.svg").read_text()
    assert "Population" in svg and ">Trait<" in svg


def test_inference_composition_labels_predictions_and_preserves_input(tmp_path: Path) -> None:
    from test_amino_acid_composition import inputs

    store, sites, context, annotations = inputs(tmp_path)
    context = context.with_columns(pl.lit(None, dtype=pl.String).alias("group_id"),
                                   pl.lit(None, dtype=pl.Int8).alias("label"),
                                   pl.Series("decision_status", ["accepted", "abstained",
                                                                 "accepted", "accepted",
                                                                 "accepted"]))
    before = context.clone()
    with store:
        write_group_trait_composition(tmp_path, store, sites, context,
            aggregation="tpm_weighted_frequency", annotations=annotations,
            stage="inference", prediction_labels=True)
    assert context.equals(before)
    stem = "amino_acid_group_prediction_composition"
    table = pl.read_csv(tmp_path / "inference/tables" / f"{stem}.tsv", separator="\t")
    assert "trait" not in table.columns and "mean_oof_probability" not in table.columns
    assert table["classification_threshold"].unique().to_list() == [.5]
    rows = table.select("predicted_class", "n_species").unique().sort("predicted_class")
    assert rows["n_species"].to_list() == [3, 2]
    accepted = pl.read_csv(tmp_path / "inference/tables/accepted_only" / f"{stem}.tsv",
                           separator="\t")
    assert accepted.filter(pl.col("predicted_class") == 0)["n_species"].unique().to_list() == [2]
    np.testing.assert_allclose(accepted.filter((pl.col("predicted_class") == 0) &
        (pl.col("alignment_pos_1based") == 1))["A"][0], .9)
    svg = (tmp_path / "inference/figures" / f"{stem}.svg").read_text()
    assert "Predicted" in svg and "Population" in svg and ">Trait<" not in svg
