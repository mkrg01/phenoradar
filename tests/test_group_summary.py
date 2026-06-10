from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from phenoradar.group_summary import (
    GroupSummaryError,
    build_group_summary_artifacts,
    group_summary_suffix,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_group_summary_suffix_strips_id_suffix() -> None:
    assert group_summary_suffix("family_id") == "family"
    assert group_summary_suffix("source_project") == "source_project"


def test_build_group_summary_artifacts_joins_metadata_and_summarizes(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tfamily_id\tfamily_name",
                "sp1\tf1\tFamily 1",
                "sp2\tf1\tFamily 1",
                "sp3\tf2\tFamily 2",
            ]
        )
        + "\n",
    )
    predictions = pl.DataFrame(
        {
            "species": ["sp1", "sp2", "sp3"],
            "true_label": [1, 0, None],
            "prob": [0.9, 0.2, 0.7],
            "pred_label_fixed_threshold": [1, 0, 1],
        }
    )

    artifacts = build_group_summary_artifacts(
        predictions=predictions,
        metadata_path=metadata,
        species_col="species",
        group_col="family_id",
        group_name_col="family_name",
        source_table_name="prediction_inference.tsv",
    )

    assert artifacts.suffix == "family"
    assert artifacts.group_label == "family"
    rows = artifacts.summary.sort("group_id").to_dicts()
    assert rows[0]["group_id"] == "f1"
    assert rows[0]["group_name"] == "Family 1"
    assert rows[0]["n_species"] == 2
    assert rows[0]["n_true_positive"] == 1
    assert rows[0]["n_true_negative"] == 1
    assert rows[0]["n_pred_positive"] == 1
    assert rows[0]["top_species"] == "sp1"
    assert rows[1]["group_id"] == "f2"
    assert rows[1]["top_prob"] == pytest.approx(0.7)


def test_build_group_summary_artifacts_skips_missing_group_column(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "species\tC4\nsp1\t1\n",
    )
    with pytest.raises(GroupSummaryError, match="missing required column"):
        build_group_summary_artifacts(
            predictions=pl.DataFrame({"species": ["sp1"], "prob": [0.8]}),
            metadata_path=metadata,
            species_col="species",
            group_col="family_id",
            group_name_col="family_name",
            source_table_name="prediction_inference.tsv",
        )
