from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from phenoradar import figures
from phenoradar.amino_acid_figures import _frequency_by_confusion


def test_confusion_points_preserve_species_values_counts_and_deterministic_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = tmp_path / "cv/tables"
    tables.mkdir(parents=True)
    feature = '["OG1",2,"A"]'
    other = '["OG1",3,"V"]'
    data = pl.DataFrame({
        "species": ["e", "a", "d", "c", "b", "f", "g", "h", "a"],
        "feature": [feature] * 8 + [other],
        "frequency": [.4, 0., .4, 1., 0., float("nan"), float("inf"), None, .5],
    })
    source = tables / "top_amino_acid_frequencies.tsv"
    data.write_csv(source, separator="\t")
    oof = pl.DataFrame({
        "species": list("abcdefgh"), "label": [0, 0, 0, 1, 1, 1, 1, 1],
        "prob": [.2, .3, .9, .2, .8, .8, .8, .1],
    })
    annotations = pl.DataFrame({"feature": ["OG1"],
                                "orthogroup_annotation": ["Test kinase"]})
    captured = []
    save = figures._save_svg_figure

    def record(figure, path):
        captured.append(figure)
        save(figure, path)

    monkeypatch.setattr(figures, "_save_svg_figure", record)
    _frequency_by_confusion(tmp_path, oof, 2, annotations)
    axes = captured[-1].axes
    offsets = [np.asarray(points.get_offsets()) for points in axes[0].collections]
    assert [len(values) for values in offsets] == [2, 1, 1, 1]
    for position, (actual, expected) in enumerate(
        zip(offsets, [[0., 0.], [1.], [.4], [.4]], strict=True), 1,
    ):
        np.testing.assert_array_equal(actual[:, 1], expected)
        assert np.all(np.abs(actual[:, 0] - position) <= .18 + 1e-12)
    assert offsets[0][0, 0] != offsets[0][1, 0]  # Coincident residues remain visible.
    assert [tick.get_text() for tick in axes[0].get_xticklabels()] == [
        "TN\nn=2", "FP\nn=1", "FN\nn=1", "TP\nn=1",
    ]
    assert len(axes[1].collections) == 1  # The other three groups are empty.
    np.testing.assert_array_equal(axes[1].collections[0].get_offsets(), [[1., .5]])
    svg = (tmp_path / "cv/figures/amino_acid_frequency_by_confusion.svg").read_text()
    assert "Test kinase" in svg and "n=0" in svg

    # Input row order must not change horizontal placement within a species group.
    data.reverse().write_csv(source, separator="\t")
    _frequency_by_confusion(tmp_path, oof.reverse(), 2, annotations)
    first = next(axis for axis in captured[-1].axes if "OG1:2 A" in axis.get_title())
    for actual, expected in zip(first.collections, offsets, strict=True):
        np.testing.assert_array_equal(actual.get_offsets(), expected)
