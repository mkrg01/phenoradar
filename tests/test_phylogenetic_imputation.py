from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from phenoradar.config import AppConfig, load_predict_config
from phenoradar.config.schema import PhylogeneticImputationConfig
from phenoradar.phylogenetic_imputation import (
    PhylogeneticImputationError,
    interpret_unknown_predictions,
    prepare_tree,
    write_phylogenetic_imputation,
)
from phenoradar.trait_reference import (
    TRAIT_SCHEMA,
    merge_observed_traits,
    read_observed_traits,
)


def references() -> pl.DataFrame:
    return pl.DataFrame({"species": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})


def predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "species": ["u", "v", "w", "absent", "a"],
            "prob": [0.95, 0.9, 0.1, 0.8, 0.99],
            "true_label": [None, None, None, None, 0],
            "pred_label_fixed_threshold": [1, 1, 0, 1, 1],
            "pred_label_selective": [1, None, 0, 1, 1],
            "decision_status": ["accepted", "abstained", "accepted", "accepted", "accepted"],
        }
    )


def test_config_shared_with_predict_and_requires_explicit_branch_mode(tmp_path: Path) -> None:
    config = tmp_path / "predict.yml"
    config.write_text("phylogenetic_imputation:\n  enabled: true\n  branch_length_mode: unit\n")
    parsed = load_predict_config([config], require_tpm=False)
    assert parsed.phylogenetic_imputation == PhylogeneticImputationConfig(
        enabled=True,
        branch_length_mode="unit",
    )
    assert not AppConfig().phylogenetic_imputation.enabled
    with pytest.raises(ValueError):
        PhylogeneticImputationConfig(branch_length_mode="auto")
    with pytest.raises(ValueError):
        PhylogeneticImputationConfig(reference_trait_path="traits.tsv")


def test_reference_preserves_literal_names_and_excludes_flagged_species(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text("id\ttrait\texclude\n001\t1\tno\nNA\t0\t\nunknown\t\tno\nskip\t1\tyes\n")
    result = read_observed_traits(
        str(metadata), species_col="id", trait_col="trait", exclude_col="exclude"
    )
    assert result.rows() == [("001", 1), ("NA", 0)]
    assert merge_observed_traits(result, result).rows() == result.rows()
    with pytest.raises(ValueError, match="Conflicting"):
        merge_observed_traits(result, pl.DataFrame({"species": ["001"], "label": [0]}))


@pytest.mark.parametrize("text", ["((a,b),c);", "((a:1,b):2,c:3);"])
def test_missing_lengths_are_not_silently_replaced(tmp_path: Path, text: str) -> None:
    pytest.importorskip("ete4")
    path = tmp_path / "tree.nwk"
    path.write_text(text)
    with pytest.raises(PhylogeneticImputationError, match="branch lengths are missing"):
        prepare_tree(path, "input")
    tree = prepare_tree(path, "unit")
    assert all(node.dist == 1 for node in tree.traverse() if not node.is_root)
    assert path.read_text() == text


@pytest.mark.parametrize(
    "text",
    [
        "(a:1,a:2,b:3);",
        "(a:-1,b:2);",
        "(a:inf,b:2);",
        "(a:bad,b:2);",
        "[&U](a:1,b:2);",
        "not a tree",
        "(:1,b:2);",
    ],
)
def test_invalid_trees_have_actionable_errors(tmp_path: Path, text: str) -> None:
    pytest.importorskip("ete4")
    path = tmp_path / "tree.nwk"
    path.write_text(text)
    with pytest.raises(PhylogeneticImputationError, match="Invalid imputation tree"):
        prepare_tree(path, "input")


def test_unit_lengths_preserve_polytomies_and_species_ids(tmp_path: Path) -> None:
    pytest.importorskip("ete4")
    path = tmp_path / "tree.nwk"
    path.write_text("[&R](('Taxon A','NA','001'),b,c);")
    tree = prepare_tree(path, "unit")
    assert len(tree.children) == 3
    assert len(tree.children[0].children) == 3
    assert {node.name for node in tree.leaves()} == {"Taxon A", "NA", "001", "b", "c"}


def test_disabled_or_no_unknowns_do_not_require_tree_or_nwkit(tmp_path: Path) -> None:
    disabled = interpret_unknown_predictions(
        AppConfig(), predictions=predictions(), run_dir=tmp_path
    )
    assert disabled.comparison.height == 0
    enabled = write_phylogenetic_imputation(
        settings=PhylogeneticImputationConfig(enabled=True),
        predictions=predictions().filter(pl.col("species") == "a"),
        observed_traits=references(),
        tree_path=None,
        run_dir=tmp_path,
        trait_name="trait",
    )
    assert enabled.comparison.height == 0
    assert not (tmp_path / "inference").exists()


def test_no_reference_produces_unavailable_values_not_prior_only_predictions(
    tmp_path: Path,
) -> None:
    result = write_phylogenetic_imputation(
        settings=PhylogeneticImputationConfig(enabled=True),
        predictions=predictions(),
        observed_traits=pl.DataFrame(schema=TRAIT_SCHEMA),
        tree_path=None,
        run_dir=tmp_path,
        trait_name="trait",
    )
    assert result.comparison["phylo_prob"].null_count() == result.comparison.height
    assert result.comparison["phylo_status"].unique().to_list() == ["missing_reference"]
    assert result.warnings


@pytest.fixture
def real_nwkit(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set PHENORADAR_TEST_NWKIT to explicitly enable real external-tool checks."""
    executable = os.environ.get("PHENORADAR_TEST_NWKIT")
    if not executable:
        pytest.skip("Set PHENORADAR_TEST_NWKIT to a compatible nwkit executable")
    pytest.importorskip("ete4")
    assert Path(executable).is_file()
    monkeypatch.setenv(
        "PATH", str(Path(executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    return executable


def test_real_asr_imputation_candidates_and_immutable_predictions(
    tmp_path: Path,
    real_nwkit: str,
) -> None:
    tree = tmp_path / "tree.nwk"
    tree.write_text("((a,b,u),(c,d,v,w));")
    original = predictions()
    result = write_phylogenetic_imputation(
        settings=PhylogeneticImputationConfig(enabled=True, branch_length_mode="unit"),
        predictions=original,
        observed_traits=references(),
        tree_path=str(tree),
        run_dir=tmp_path,
        trait_name="trait",
    )
    assert_frame_equal(original, predictions())
    rows = {row["species"]: row for row in result.comparison.iter_rows(named=True)}
    # For this symmetric tree the ML ER imputation is 5/18 and 13/18.
    assert rows["u"]["phylo_prob"] == pytest.approx(5 / 18, abs=1e-5)
    assert rows["v"]["phylo_prob"] == pytest.approx(13 / 18, abs=1e-5)
    assert rows["w"]["prob_difference"] < 0
    assert "a" not in rows  # An observed species is never a discrepancy candidate.
    assert rows["absent"]["phylo_status"] == "missing_tree_tip"
    assert rows["absent"]["phylo_prob"] is None
    positive = pl.read_csv(
        tmp_path / "inference/tables/phylogenetic_positive_candidates.tsv", separator="\t"
    )
    assert positive["species"].to_list() == ["u"]  # v is abstained despite its raw positive call.
    assert result.nodes.height == 10  # Polytomies are not resolved.
    metadata = json.loads(
        (tmp_path / "inference/phylogenetic_imputation/metadata.json").read_text()
    )
    assert metadata["nwkit_version"].startswith("nwkit ")
    assert metadata["returncode"] == 0


def test_input_and_explicit_unit_tree_give_same_results(tmp_path: Path, real_nwkit: str) -> None:
    tree = tmp_path / "tree.nwk"
    tree.write_text("((a:1,b:1,u:1):1,(c:1,d:1,v:1,w:1):1);")
    results = [
        write_phylogenetic_imputation(
            settings=PhylogeneticImputationConfig(enabled=True, branch_length_mode=mode),
            predictions=predictions(),
            observed_traits=references(),
            tree_path=str(tree),
            run_dir=tmp_path / mode,
            trait_name="trait",
        ).comparison
        for mode in ["input", "unit"]
    ]
    assert_frame_equal(*results)


def test_failed_fit_is_recorded_without_changing_predictions(
    tmp_path: Path,
    real_nwkit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phenoradar import phylogenetic_imputation as module

    tree = tmp_path / "tree.nwk"
    tree.write_text("((a,b,u),(c,d,v,w));")
    original_run = subprocess.run

    def fail_asr(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "asr" in arguments:
            return subprocess.CompletedProcess(arguments, 1, "", "Unable to fit model")
        return original_run(arguments, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", fail_asr)
    result = write_phylogenetic_imputation(
        settings=PhylogeneticImputationConfig(enabled=True, branch_length_mode="unit"),
        predictions=predictions(),
        observed_traits=references(),
        tree_path=str(tree),
        run_dir=tmp_path,
        trait_name="trait",
    )
    assert result.comparison["phylo_prob"].null_count() == result.comparison.height
    assert "fit_failed" in result.comparison["phylo_status"].to_list()
    assert result.warnings


def test_real_figures_retain_references(tmp_path: Path, real_nwkit: str) -> None:
    from phenoradar.phylogenetic_figures import write_phylogenetic_figures

    tree = tmp_path / "tree.nwk"
    tree.write_text("((a,b,u),(c,d,v,w));")
    result = write_phylogenetic_imputation(
        settings=PhylogeneticImputationConfig(enabled=True, branch_length_mode="unit"),
        predictions=predictions(),
        observed_traits=references(),
        tree_path=str(tree),
        run_dir=tmp_path,
        trait_name="trait",
    )
    write_phylogenetic_figures(result, run_dir=tmp_path, branch_length_mode="unit")
    assert result.annotation.filter(~pl.col("is_prediction_target")).height == 3
    svg = (tmp_path / "inference/figures/tree_phylogenetic_imputation.svg").read_text()
    assert "Observed" in svg and "Phylogenetic" in svg and "abstained" in svg
    assert (tmp_path / "inference/figures/phylogenetic_comparison.svg").exists()


def test_real_root_polytomy_zero_edge_and_length_precision(
    tmp_path: Path,
    real_nwkit: str,
) -> None:
    from ete4 import Tree

    tree = tmp_path / "tree.nwk"
    distance = 0.12345678901234567
    tree.write_text(f"(a:0,b:{distance!r},u:0);")
    result = write_phylogenetic_imputation(
        settings=PhylogeneticImputationConfig(enabled=True),
        predictions=predictions().filter(pl.col("species") == "u"),
        observed_traits=pl.DataFrame({"species": ["a", "b"], "label": [0, 1]}),
        tree_path=str(tree),
        run_dir=tmp_path,
        trait_name="trait",
    )
    assert result.comparison["phylo_prob"].to_list() == [0.0]
    saved = Tree(str(tmp_path / "inference/phylogenetic_imputation/tree.nwk"), parser=1)
    assert len(saved.children) == 3
    assert next(node.dist for node in saved.leaves() if node.name == "b") == distance


def test_real_asr_preserves_quoted_and_numeric_species_ids(
    tmp_path: Path,
    real_nwkit: str,
) -> None:
    tree = tmp_path / "tree.nwk"
    tree.write_text("(('001','002','Taxon A'),('NA','0002',u));")
    result = write_phylogenetic_imputation(
        settings=PhylogeneticImputationConfig(enabled=True, branch_length_mode="unit"),
        predictions=pl.DataFrame({"species": ["Taxon A", "u"], "prob": [0.9, 0.2]}),
        observed_traits=pl.DataFrame(
            {"species": ["001", "002", "NA", "0002"], "label": [0, 0, 1, 1]}
        ),
        tree_path=str(tree),
        run_dir=tmp_path,
        trait_name="trait",
    )
    assert result.comparison["phylo_status"].to_list() == ["imputed", "imputed"]
    assert set(result.annotation["species"]) == {"001", "002", "NA", "0002", "Taxon A", "u"}
