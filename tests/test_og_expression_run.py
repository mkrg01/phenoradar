from pathlib import Path

import polars as pl
import pytest
import yaml
from test_amino_acid import fixture_config
from typer.testing import CliRunner

from phenoradar.cli import app


@pytest.mark.parametrize("mode", ["amino_acid", "expression", "combined"])
def test_cli_writes_raw_og_expression_for_cv_species_in_every_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    path = fixture_config(tmp_path, "aa_frequency", full=False)
    config = yaml.safe_load(path.read_text())
    config["data"]["feature_mode"] = mode
    if mode == "combined":
        config["preprocess"]["expression"] = {
            "absent_feature_fill": "nan",
            "missing_expression": {"method": "neutral", "zero_as_missing": False},
        }
        config["model_selection"]["search_space"]["alpha"] = [1]
    tpm = tmp_path / "og_tpm.tsv"
    tpm.write_text("species\torthogroup\ttpm\n" + "".join(
        f"s{i}\tOG1\t{100 if i % 2 else 0}\n" for i in range(8)) +
        "external\tOG1\t1000000000\nunknown\tOG1\t1000000000\n")
    config["data"]["tpm_path"] = str(tpm)
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "-c", str(path), "--quiet"])
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").iterdir())
    summary = pl.read_csv(run / "cv/tables/og_expression_group_trait_summary.tsv", separator="\t")
    points = pl.read_csv(run / "cv/tables/og_expression_species.tsv", separator="\t")
    assert points.height == summary["n_observed"].sum() == 8
    assert summary["n_zero"].sum() == 4
    assert points["tpm"].max() == 100
    assert set(points["group_id"]) == {"g0", "g1", "g2", "g3"}
    svg = (run / "cv/figures/og_expression_by_group_trait/01_OG1.svg").read_text()
    assert ">C4: 0<" in svg and ">C4: 1<" in svg
    assert not list((run / "cv/figures").glob("og_expression_by_group_trait_*.svg"))
    heatmap = (run / "cv/figures/og_expression_group_trait_heatmap.svg").read_text()
    assert ">C4<" in heatmap
