from __future__ import annotations

from pathlib import Path

from phenoradar.orthogroup_annotation import load_orthogroup_annotations


def test_load_orthogroup_annotations_reads_headerless_tsv(tmp_path: Path) -> None:
    path = tmp_path / "orthogroup_annotations.tsv"
    path.write_text(
        "\n".join(
            [
                "OG1\t3193\tbeta carbonic anhydrase",
                "OG2\t3193\thypothetical protein",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    annotations = load_orthogroup_annotations(path)

    assert annotations is not None
    assert annotations.to_dicts() == [
        {
            "feature": "OG1",
            "orthogroup_annotation_taxid": "3193",
            "orthogroup_annotation": "beta carbonic anhydrase",
        },
        {
            "feature": "OG2",
            "orthogroup_annotation_taxid": "3193",
            "orthogroup_annotation": "hypothetical protein",
        },
    ]


def test_load_orthogroup_annotations_allows_absent_path() -> None:
    assert load_orthogroup_annotations(None) is None


def test_load_orthogroup_annotations_filters_before_materialization(tmp_path: Path) -> None:
    path = tmp_path / "orthogroup_annotations.tsv"
    path.write_text(
        "\n".join(
            [
                "OG1\t3193\tbeta carbonic anhydrase",
                "OG2\t3193\thypothetical protein",
                " OG3 \t3193\tmalate dehydrogenase",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    annotations = load_orthogroup_annotations(
        path,
        feature_names=["OG3", "", "OG_missing"],
    )

    assert annotations is not None
    assert annotations.to_dicts() == [
        {
            "feature": "OG3",
            "orthogroup_annotation_taxid": "3193",
            "orthogroup_annotation": "malate dehydrogenase",
        }
    ]
