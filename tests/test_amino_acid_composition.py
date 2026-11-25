from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from phenoradar.amino_acid import ALPHABET, AlignmentStore
from phenoradar.amino_acid_composition import (
    _composition_figure,
    build_group_trait_composition,
    write_group_trait_composition,
)


def inputs(tmp_path: Path) -> tuple[AlignmentStore, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    aa = tmp_path / "aa"
    aa.mkdir()
    (aa / "OG1.fa").write_text(
        ">s0_a\nAA\n>s0_v\nVV\n>s1_v\nVV\n>s2_gap\n--\n>s3_a\nA-\n>s4_v\nVV\n")
    store = AlignmentStore(aa, [f"s{i}" for i in range(5)])
    store.ogs["OG1"].tpm = np.array([9e9, 1e9, 1, 2, 1, 1], dtype=float)
    sites = pl.DataFrame({"orthogroup": ["OG1"] * 2,
                          "alignment_pos_1based": [1, 2], "pss": [2., 1.]})
    predictions = pl.DataFrame({"species": [f"s{i}" for i in range(5)],
                                "group_id": ["a", "a", "a", "a", "b"],
                                "label": [0, 0, 0, 1, 1], "prob": [.1, .2, .3, .8, .9]})
    annotations = pl.DataFrame({"feature": ["OG1"], "orthogroup_annotation": [
        "A very long complete orthogroup annotation including membrane transporter function"]})
    return store, sites, predictions, annotations


@pytest.mark.parametrize("aggregation, expected_a", [
    ("tpm_weighted_frequency", .45), ("aa_frequency", .25), ("best_covered", .5),
])
def test_species_equal_weights_missing_sites_and_incomplete_groups(
    tmp_path: Path, aggregation: str, expected_a: float,
) -> None:
    store, sites, predictions, annotations = inputs(tmp_path)
    with store:
        table = build_group_trait_composition(store, sites, predictions,
            aggregation=aggregation, annotations=annotations)
    first = table.row(0, named=True)
    assert (first["group_id"], first["trait"], first["n_species"]) == ("a", 0, 3)
    assert first["n_observed_species"] == 2
    assert first["observed_species_fraction"] == pytest.approx(2 / 3)
    assert first["mean_oof_probability"] == pytest.approx(.2)
    assert first["A"] == pytest.approx(expected_a)
    assert first["V"] == pytest.approx(1 - expected_a)
    missing = table.filter((pl.col("group_id") == "a") & (pl.col("trait") == 1) &
                           (pl.col("alignment_pos_1based") == 2)).row(0, named=True)
    assert missing["n_observed_species"] == 0 and np.isnan(missing["A"])
    assert table.height == 6  # Group b has no trait-0 species; do not invent a row.
    assert table["orthogroup_annotation"].unique().to_list() == annotations[
        "orthogroup_annotation"].to_list()
    observed = table.filter(pl.col("n_observed_species") > 0)
    np.testing.assert_allclose(observed.select(list(ALPHABET)).to_numpy().sum(axis=1), 1)


def test_full_labels_grouped_legend_and_accepted_population(tmp_path: Path) -> None:
    store, sites, predictions, annotations = inputs(tmp_path)
    predictions = predictions.with_columns(pl.Series(
        "decision_status", ["accepted", "abstained", "accepted", "accepted", "accepted"]))
    with store:
        write_group_trait_composition(tmp_path, store, sites, predictions,
            aggregation="tpm_weighted_frequency", annotations=annotations)
    root = tmp_path / "cv"
    name = "amino_acid_group_trait_composition"
    full = pl.read_csv(root / "tables" / f"{name}.tsv", separator="\t")
    accepted = pl.read_csv(root / "tables/accepted_only" / f"{name}.tsv", separator="\t")
    assert full["A"][0] == pytest.approx(.45)
    assert accepted["A"][0] == pytest.approx(.9)
    assert accepted["n_species"][0] == 2 and accepted["n_observed_species"][0] == 1
    assert (root / "figures/accepted_only" / f"{name}.svg").exists()
    fig = _composition_figure(full)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    labels = fig.axes[1].get_xticklabels()
    assert annotations["orthogroup_annotation"][0] in " ".join(labels[0].get_text().split())
    assert "(OG1):1" in labels[0].get_text()
    assert "…" not in labels[0].get_text()
    boxes = [label.get_window_extent(canvas.get_renderer()) for label in labels]
    assert all(fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1) for b in boxes)
    assert not boxes[0].overlaps(boxes[1])
    keys = [label.get_text() for label in fig.legends[0].get_texts()]
    assert len(keys) == 7
    assert sorted("".join(keys).replace(" ", "")) == sorted(ALPHABET)
    plt.close(fig)


def test_no_sites_or_accepted_species_writes_explicit_empty_figure(tmp_path: Path) -> None:
    store, sites, predictions, _ = inputs(tmp_path)
    predictions = predictions.with_columns(pl.lit("abstained").alias("decision_status"))
    with store:
        write_group_trait_composition(tmp_path, store, sites.head(0), predictions,
                                      aggregation="aa_frequency")
    for folder in ["figures", "figures/accepted_only"]:
        svg = tmp_path / "cv" / folder / "amino_acid_group_trait_composition.svg"
        assert "No observed groups or positive-PSS sites" in svg.read_text()
