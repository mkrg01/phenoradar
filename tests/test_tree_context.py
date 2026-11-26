import re
from pathlib import Path
from xml.etree import ElementTree as ET

import polars as pl
import pytest

from phenoradar import figures
from phenoradar.cv_display import fold_group_labels, tree_context_tracks, with_cv_groups
from phenoradar.tree_prediction import _write_tree_feature_heatmap_svg


@pytest.mark.parametrize(("groups", "folds", "expected"), [
    (["A", "A", "B", "B"], ["0", "0", "1", "1"], {"0": "A", "1": "B"}),
    (["A", "B", "C", "D"], ["0", "0", "1", "1"], None),
    (["A", "A", "B", "B"], ["0", "1", "0", "1"], None),
    (["A", None, "B", "B"], ["0", "0", "1", "1"], None),
])
def test_group_labels_require_a_complete_bijection(groups, folds, expected) -> None:
    frame = pl.DataFrame({"group_id": groups, "fold_id": folds})
    assert fold_group_labels(frame) == expected
    assert tree_context_tracks(frame) == (["group_id"] if expected else ["group_id", "fold_id"])


def visible(path: Path) -> set[str]:
    return {"".join(e.itertext()) for e in ET.parse(path).getroot().iter()
            if e.tag.endswith("}text")}


@pytest.mark.parametrize("shared", [False, True])
def test_tree_context_color_and_partition_identity(tmp_path: Path, shared: bool) -> None:
    tree = tmp_path / "tree.nwk"
    tree.write_text("((a,b),(c,d));")
    annotation = pl.DataFrame({
        "species": ["a", "b", "c", "d"], "true_label": [0, 1, 0, 1],
        "prob": [.1, .9, .2, .8], "group_id": ["A", "A" if shared else "B", "C", "C"],
        "fold_id": ["0", "0", "1", "1"], "feature": ["OG"] * 4,
        "orthogroup_annotation": ["Kinase"] * 4, "feature_rank": [1] * 4,
        "log2_tpm_plus1": [1., 2., 3., 4.],
        "decision_status": ["accepted", "abstained", "accepted", "accepted"],
    })
    original = annotation.clone()
    output = tmp_path / "heatmap.svg"
    assert _write_tree_feature_heatmap_svg(tree_path=tree, annotation=annotation,
        value_col="log2_tpm_plus1", out_path=output, title="", cmap_name="viridis",
        trait_name="C4") == []
    assert annotation.equals(original)
    for path in [output, tmp_path / "accepted_only/heatmap.svg"]:
        text = visible(path)
        assert "Classification outcome" not in text
        assert "OOF confusion group" not in path.read_text()
        assert {"C4", "prob", "group", "C4: 0", "C4: 1", "Predicted probability"} <= text
        assert ("fold" in text) != shared
        # Non-LOGO must retain Fold even when the accepted subset happens to be bijective.
        assert "stroke-dasharray" in path.read_text()
        rectangles = [e for e in ET.parse(path).getroot().iter() if e.tag.endswith("}rect")]
        context_cells = [e for e in rectangles if any(
            child.tag.endswith("}title") and (" C4=" in (child.text or "")
                                             or " prob=" in (child.text or ""))
            for child in e)]
        assert len(context_cells) == (6 if path.parent.name == "accepted_only" else 8)
        assert len({e.get("style") for e in context_cells}) >= 4
        root = ET.parse(path).getroot()
        tree_ys = {float(e.get("y1")) for e in root.iter()
                   if e.tag.endswith("}line") and "stroke-dasharray" in e.get("style", "")
                   and e.get("y1") == e.get("y2")}
        segments = [re.fullmatch(r"M ([\d.-]+) ([\d.-]+) L ([\d.-]+) ([\d.-]+)", e.get("d", ""))
                    for e in root.iter()
                    if e.tag.endswith("}path") and "stroke-dasharray" in e.get("style", "")]
        lines = [tuple(map(float, match.groups())) for match in segments if match]
        start_x = min(line[0] for line in lines)
        table_ys = {line[1] for line in lines if line[0] == start_x and line[1] == line[3]}
        assert tree_ys and tree_ys == table_ys


def test_probability_axis_uses_actual_group_ids_and_freezes_population_mapping(
    tmp_path: Path,
) -> None:
    frame = pl.DataFrame({"species": ["a", "b", "c", "d"], "label": [0, 1, 0, 1],
        "fold_id": ["0", "0", "1", "1"], "prob": [.1, .9, .2, .8],
        "decision_status": ["accepted", "abstained", "accepted", "accepted"]})
    metadata = tmp_path / "metadata.tsv"
    pl.DataFrame({"species": frame["species"], "family": ["Alpha", "Beta", "Gamma", "Gamma"]}
                 ).write_csv(metadata, separator="\t")
    grouped = with_cv_groups(
        frame, metadata_path=metadata, species_col="species", group_col="family")
    figures._cv_fold_trait_probability(grouped, tmp_path / "nonlogo.svg")
    assert "CV fold" in visible(tmp_path / "nonlogo.svg")
    assert "CV fold" in visible(tmp_path / "accepted_only/nonlogo.svg")
    grouped = grouped.with_columns(pl.Series("group_id", ["Alpha", "Alpha", "Gamma", "Gamma"]))
    figures._cv_fold_trait_probability(grouped, tmp_path / "logo.svg")
    for path in [tmp_path / "logo.svg", tmp_path / "accepted_only/logo.svg"]:
        assert {"Group", "Alpha", "Gamma"} <= visible(path)
        assert "CV fold" not in visible(path)
