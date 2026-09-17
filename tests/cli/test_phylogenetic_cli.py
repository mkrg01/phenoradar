"""End-to-end inference-only phylogenetic interpretation checks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from phenoradar.bundle import BundleError, load_model_bundle
from phenoradar.cli import app


def test_full_run_and_predict_phylogeny_leave_model_predictions_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = os.environ.get("PHENORADAR_TEST_NWKIT")
    if not executable:
        pytest.skip("Set PHENORADAR_TEST_NWKIT to run ASR integration")
    pytest.importorskip("ete4")
    monkeypatch.setenv(
        "PATH", str(Path(executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    monkeypatch.chdir(tmp_path)
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text(
        "species\tphenotype\tcontrast_pair_id\n"
        "a\t0\tg1\nb\t1\tg1\nc\t0\tg2\nd\t1\tg2\ne\t1\t\nu\t\t\n"
    )
    tpm = tmp_path / "tpm.tsv"
    tpm.write_text(
        "species\torthogroup\ttpm\n"
        "a\tOG1\t1\nb\tOG1\t20\nc\tOG1\t2\nd\tOG1\t30\n"
        "e\tOG1\t15\nu\tOG1\t40\n"
    )
    tree = tmp_path / "tree.nwk"
    tree.write_text("((a,c,u),(b,d,e));")
    config = {
        "data": {
            "metadata_path": str(metadata),
            "tpm_path": str(tpm),
            "tree_path": str(tree),
            "trait_col": "phenotype",
        },
        "runtime": {"execution_stage": "full_run", "n_jobs": 1},
        "phylogenetic_imputation": {"enabled": False, "branch_length_mode": "unit"},
    }
    runner = CliRunner()

    def run(kind: str, payload: dict[str, object], *args: str) -> Path:
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.safe_dump(payload))
        before = set((tmp_path / "runs").glob("*"))
        result = runner.invoke(app, [kind, "-c", str(config_path), *args])
        assert result.exit_code == 0, result.output + repr(result.exception)
        return next(iter(set((tmp_path / "runs").glob("*")) - before))

    baseline = run("run", config)
    assert not (baseline / "inference/tables/phylogenetic_imputation.tsv").exists()
    bundle = load_model_bundle(baseline / "model_bundle")
    assert bundle.observed_traits["species"].to_list() == ["a", "b", "c", "d", "e"]
    assert bundle.manifest["trait_name"] == "phenotype"

    config["phylogenetic_imputation"]["enabled"] = True
    enabled = run("run", config)
    for artifact in [
        "cv/tables/prediction_cv.tsv",
        "cv/tables/metrics_cv.tsv",
        "external_test/tables/prediction_external_test.tsv",
        "inference/tables/prediction_inference.tsv",
    ]:
        assert_frame_equal(
            pl.read_csv(baseline / artifact, separator="\t"),
            pl.read_csv(enabled / artifact, separator="\t"),
        )
    comparison = pl.read_csv(
        enabled / "inference/tables/phylogenetic_comparison.tsv", separator="\t"
    )
    assert comparison["species"].to_list() == ["u"]
    assert comparison["phylo_status"].to_list() == ["imputed"]
    assert not list((enabled / "cv").rglob("*phylogenetic*"))
    assert not list((enabled / "external_test").rglob("*phylogenetic*"))
    candidate = pl.read_csv(
        enabled / "inference/tables/candidate_evidence_candidates.tsv", separator="\t"
    )
    assert candidate["phylo_prob"].to_list() == pytest.approx(comparison["phylo_prob"].to_list())
    assert (enabled / "inference/figures/tree_phylogenetic_imputation.svg").exists()

    # Snapshot works after the original metadata has gone; expression input contains only u.
    metadata.unlink()
    target_tpm = tmp_path / "target.tsv"
    target_tpm.write_text("species\torthogroup\ttpm\nu\tOG1\t40\n")
    predicted = run(
        "predict",
        {
            "data": {
                "tpm_path": str(target_tpm),
                "tree_path": str(tree),
                "contrast_pair_col": None,
            },
            "phylogenetic_imputation": {"enabled": True, "branch_length_mode": "unit"},
        },
        "--model-bundle",
        str(baseline / "model_bundle"),
    )
    restored = pl.read_csv(
        predicted / "inference/tables/phylogenetic_comparison.tsv", separator="\t"
    )
    assert restored["phylo_prob"].to_list() == pytest.approx(comparison["phylo_prob"].to_list())
    assert restored["prob"].to_list() == pytest.approx(comparison["prob"].to_list())
    assert (predicted / "inference/figures/tree_phylogenetic_imputation.svg").exists()

    corrupted = tmp_path / "corrupted"
    shutil.copytree(baseline / "model_bundle", corrupted)
    with (corrupted / "observed_traits.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(BundleError, match="integrity check failed"):
        load_model_bundle(corrupted)


def test_cv_only_does_not_invoke_imputation_or_require_a_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text(
        "species\tC4\tcontrast_pair_id\na\t0\tg1\nb\t1\tg1\nc\t0\tg2\nd\t1\tg2\nu\t\t\n"
    )
    tpm = tmp_path / "tpm.tsv"
    tpm.write_text(
        "species\torthogroup\ttpm\na\tOG1\t1\nb\tOG1\t20\nc\tOG1\t2\nd\tOG1\t30\nu\tOG1\t40\n"
    )
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {"metadata_path": str(metadata), "tpm_path": str(tpm)},
                "phylogenetic_imputation": {"enabled": True},
            }
        )
    )

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("CV-only must never invoke imputation")

    monkeypatch.setattr("phenoradar.cli.interpret_unknown_predictions", unexpected)
    result = CliRunner().invoke(app, ["run", "-c", str(config)])
    assert result.exit_code == 0, result.output
    assert not list((tmp_path / "runs").rglob("*phylogenetic*"))
