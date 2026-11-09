from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from phenoradar.orthogroup_annotation import load_orthogroup_annotations


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("requested", [["OG1", "OG2"], []])
def test_streaming_annotations_preserve_duplicate_and_null_selection(
    tmp_path: Path,
    compressed: bool,
    requested: list[str],
) -> None:
    text = " OG1 \t\tfirst\nOG1\t3193\tlater\nOG2\t3193\t\nOG2\t9999\tretained\n"
    path = tmp_path / ("annotations.tsv.gz" if compressed else "annotations.tsv")
    if compressed:
        with gzip.open(path, "wt") as stream:
            stream.write(text)
    else:
        path.write_text(text)
    result = load_orthogroup_annotations(path, feature_names=requested)
    assert result is not None
    assert result.to_dicts() == (
        [
            {
                "feature": "OG1",
                "orthogroup_annotation_taxid": "3193",
                "orthogroup_annotation": "first",
            },
            {
                "feature": "OG2",
                "orthogroup_annotation_taxid": "3193",
                "orthogroup_annotation": "retained",
            },
        ]
        if requested
        else []
    )
