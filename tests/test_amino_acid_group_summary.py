from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from phenoradar.amino_acid_run import predict_amino_acid, run_amino_acid
from phenoradar.cli import app
from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.group_summary_outputs import write_group_summary_artifacts


def group_config(tmp_path: Path, mode: str, column: str, *, full: bool = True) -> Path:
    from test_amino_acid import fixture_config
    from test_combined import combined_config

    path = (combined_config(tmp_path, "tpm_weighted_frequency") if mode == "combined" else
            fixture_config(tmp_path, "tpm_weighted_frequency", full=full))
    raw = yaml.safe_load(path.read_text())
    raw["runtime"]["execution_stage"] = "full_run" if full else "cv_only"
    raw["summary"] = {"group_col": column}
    metadata = Path(raw["data"]["metadata_path"])
    frame = pl.read_csv(metadata, separator="\t").with_columns(
        (pl.lit("Order ") + pl.col("family")).alias("order")
    )
    frame.write_csv(metadata, separator="\t")
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("mode", ["amino_acid", "combined"])
@pytest.mark.parametrize("column", ["family", "order"])
def test_sequence_modes_write_group_probabilities_for_all_run_stages_and_predict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, column: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = group_config(tmp_path, mode, column)
    config = load_and_resolve_config([path])
    out = run_amino_acid(config, tmp_path / "run")
    for stage, count in [("cv", 8), ("external_test", 1), ("inference", 1)]:
        table = pl.read_csv(out / stage / "tables" / f"group_summary_{column}.tsv", separator="\t")
        assert table["n_species"].sum() == count
        assert table["group_col"].unique().to_list() == [column]
        assert (out / stage / "figures" / f"probability_by_{column}.svg").exists()
        if column == "order":
            assert all(group.startswith("Order ") for group in table["group_id"])
            assert not (out / stage / "figures/probability_by_family.svg").exists()

    metadata = pl.read_csv(config.data.metadata_path, separator="\t")
    subset = tmp_path / "targets.tsv"
    metadata.filter(pl.col("species").is_in(["external", "unknown"])).select(
        "species", "family", "order"
    ).write_csv(subset, separator="\t")
    warnings: list[str] = []
    predicted = predict_amino_acid(out / "model_bundle", tmp_path / "predict",
                                   metadata_path=subset, warning_messages=warnings)
    assert warnings == []
    assert (predicted / "predict/figures" / f"probability_by_{column}.svg").exists()
    actual = pl.read_csv(predicted / "predict/tables/pred_predict.tsv",
                         separator="\t").sort("species")
    expected = pl.concat([
        pl.read_csv(out / f"{stage}/tables/pred_{stage}.tsv", separator="\t").select(
            "species", "prob"
        )
        for stage in ["external_test", "inference"]
    ]).sort("species")
    np.testing.assert_allclose(actual["prob"], expected["prob"], rtol=0, atol=1e-12)

    if mode == "amino_acid" and column == "family":
        # A predict config changes only the display grouping of the frozen model.
        raw = yaml.safe_load(path.read_text())
        raw["summary"]["group_col"] = "order"
        override = tmp_path / "predict.yml"
        override.write_text(yaml.safe_dump(raw))
        result = CliRunner().invoke(app, ["predict", "--model-bundle", str(out / "model_bundle"),
            "--metadata-path", str(subset), "-c", str(override), "--quiet"])
        assert result.exit_code == 0, result.output
        assert len(list((tmp_path / "runs").glob(
            "*_predict_*/predict/figures/probability_by_order.svg"))) == 1


def test_sequence_cv_only_writes_only_oof_group_summary(tmp_path: Path) -> None:
    path = group_config(tmp_path, "amino_acid", "family", full=False)
    out = run_amino_acid(load_and_resolve_config([path]), tmp_path / "run")
    assert (out / "cv/figures/probability_by_family.svg").exists()
    assert not (out / "external_test/figures/probability_by_family.svg").exists()
    assert not (out / "inference/figures/probability_by_family.svg").exists()


def test_shared_writer_preserves_abstention_and_normalizes_custom_column(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text("species\tOrder / Clade\na\tA\nb\tA\nc\tB\nd\t\n")
    config = AppConfig()
    config.data.metadata_path = str(metadata)
    config.summary.group_col = "Order / Clade"
    frame = pl.DataFrame({
        "species": list("abcd"), "prob": [.9, .8, .1, .2], "label": [1, 0, 0, 1],
        "pred_label_fixed_threshold": [1, 1, 0, 0],
        "pred_label_selective": [1, None, 0, None],
        "decision_status": ["accepted", "abstained", "accepted", "abstained"],
    })
    original = frame.clone()
    assert write_group_summary_artifacts(run_dir=tmp_path, stage="cv", predictions=frame,
        source_table_name="prediction_cv.tsv", config=config) == []
    assert frame.equals(original)
    summary = pl.read_csv(tmp_path / "cv/tables/group_summary_order_clade.tsv", separator="\t")
    assert set(summary["group_id"]) == {"A", "B", "unassigned"}
    assert summary["n_accepted"].sum() == 2
    assert (tmp_path / "cv/tables/abstention_by_order_clade.tsv").exists()
    assert (tmp_path / "cv/figures/accepted_only/probability_by_order_clade.svg").exists()


def test_missing_group_column_skips_only_the_summary(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text("species\na\n")
    config = AppConfig()
    config.data.metadata_path = str(metadata)
    warnings = write_group_summary_artifacts(run_dir=tmp_path, stage="cv",
        predictions=pl.DataFrame({"species": ["a"], "prob": [.8]}),
        source_table_name="prediction_cv.tsv", config=config)
    assert len(warnings) == 1 and "missing required column" in warnings[0]
    assert not (tmp_path / "cv/figures/probability_by_family.svg").exists()
