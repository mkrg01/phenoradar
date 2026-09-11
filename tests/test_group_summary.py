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


def test_group_summary_suffix_normalizes_column_name() -> None:
    assert group_summary_suffix("family") == "family"
    assert group_summary_suffix("source_project") == "source_project"


@pytest.mark.parametrize("group_col", ["family", "order"])
def test_build_group_summary_artifacts_joins_metadata_and_summarizes(
    tmp_path: Path, group_col: str
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                f"species\t{group_col}",
                "sp1\tFamily 1",
                "sp2\tFamily 1",
                "sp3\tFamily 2",
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
        group_col=group_col,
        source_table_name="prediction_inference.tsv",
    )

    assert artifacts.suffix == group_col
    assert artifacts.group_label == group_col
    rows = artifacts.summary.sort("group_id").to_dicts()
    assert rows[0]["group_id"] == "Family 1"
    assert rows[0]["n_species"] == 2
    assert rows[0]["n_true_positive"] == 1
    assert rows[0]["n_true_negative"] == 1
    assert rows[0]["n_pred_positive"] == 1
    assert rows[0]["top_species"] == "sp1"
    assert rows[1]["group_id"] == "Family 2"
    assert rows[1]["top_prob"] == pytest.approx(0.7)


def test_build_group_summary_normalizes_names_and_missing_values(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "species\tfamily\nsp1\t Poaceae \nsp2\tPoaceae\nsp3\t\nsp4\t   \n",
    )
    artifacts = build_group_summary_artifacts(
        predictions=pl.DataFrame(
            {"species": ["sp1", "sp2", "sp3", "sp4", "sp5"], "prob": [0.8] * 5}
        ),
        metadata_path=metadata,
        species_col="species",
        group_col="family",
        source_table_name="prediction_inference.tsv",
    )

    assert artifacts.summary.select("group_id", "n_species").sort(
        "group_id"
    ).to_dicts() == [
        {"group_id": "Poaceae", "n_species": 2},
        {"group_id": "unassigned", "n_species": 3},
    ]


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
            group_col="family",
            source_table_name="prediction_inference.tsv",
        )
