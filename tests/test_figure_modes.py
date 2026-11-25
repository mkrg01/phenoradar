from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from phenoradar import figures
from phenoradar.cli import app


@pytest.mark.parametrize("mode", ["expression", "amino_acid", "combined"])
def test_all_feature_modes_generate_shared_diagnostics_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from test_combined import combined_config

    monkeypatch.chdir(tmp_path)
    path = combined_config(tmp_path, "aa_frequency")
    raw = yaml.safe_load(path.read_text())
    raw["data"]["feature_mode"] = mode
    raw["model_selection"]["search_space"] = {"lambda": [0.01], "alpha": [1.0]}
    raw["evaluation"]["group_bootstrap"]["enabled"] = False
    raw["phylogenetic_imputation"] = {"enabled": False}
    raw["abstention"] = {"enabled": True, "threshold": 0.8}
    path.write_text(yaml.safe_dump(raw))
    calls = []
    original = figures._prediction_calibration

    def record(*args, **kwargs):
        calls.append(args[1].name)
        return original(*args, **kwargs)

    monkeypatch.setattr(figures, "_prediction_calibration", record)
    result = CliRunner().invoke(app, ["run", "-c", str(path), "--quiet"])
    assert result.exit_code == 0, result.output
    assert sorted(calls) == ["cv_calibration.svg", "external_calibration.svg"]
    run = next((tmp_path / "runs").glob("*_run_*"))
    for stage, stems in [
        ("cv", ["cv_calibration", "cv_confusion_matrix"]),
        ("external_test", ["external_calibration", "external_confusion_matrix"]),
    ]:
        for stem in stems:
            for population in ["", "accepted_only"]:
                assert (run / stage / "figures" / population / f"{stem}.svg").exists()
    if mode in {"expression", "combined"}:
        assert (run / "cv/figures/top_feature_expression_by_confusion.svg").exists()
        assert (run / "cv/figures/top_feature_expression_by_confusion.pdf").exists()
    if mode == "combined":
        table = pl.read_csv(run / "cv/tables/top_feature_expression.tsv", separator="\t")
        assert table["species"].n_unique() == 8
        assert set(table["tpm"]) >= {2.0, 12.0}
        assert (run / "external_test/figures/top_feature_expression_by_confusion.svg").exists()
