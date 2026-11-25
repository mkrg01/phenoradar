from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

from phenoradar import figures
from phenoradar.amino_acid_diagnostics import write_frequency_by_confusion


def test_calibration_includes_probability_endpoints_and_excludes_invalid_rows() -> None:
    data = pl.DataFrame(
        {
            "label": [0, 1, 1, 0, 1, 0, 1, 1, None],
            "prob": [0.0, 0.1, 0.99, 1.0, float("nan"), float("inf"), -0.1, 1.1, 0.5],
        }
    )
    bins = figures._calibration_bins(data)
    assert bins.height == 10
    assert bins["n_species"].sum() == 4
    assert bins.filter(pl.col("bin") == 0)["mean_probability"].item() == 0
    assert bins.filter(pl.col("bin") == 1)["observed_fraction"].item() == 1
    last = bins.filter(pl.col("bin") == 9)
    assert last["n_species"].item() == 2
    assert last["mean_probability"].item() == pytest.approx(0.995)
    assert last["observed_fraction"].item() == 0.5
    assert bins.filter(pl.col("bin") == 5)["mean_probability"].item() is None


def test_cv_and_external_diagnostics_share_populations_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv = pl.DataFrame(
        {
            "species": ["tn", "fp", "fn", "tp"],
            "label": [0, 0, 1, 1],
            "prob": [0.0, 0.8, 0.2, 1.0],
            "decision_status": ["accepted", "abstained", "abstained", "accepted"],
        }
    )
    original = cv.clone()
    rendered: dict[Path, Figure] = {}
    save = figures._save_svg_figure

    def record(fig: Figure, path: Path) -> None:
        rendered[path] = fig
        save(fig, path)

    monkeypatch.setattr(figures, "_save_svg_figure", record)
    for name, data, label in [
        ("cv", cv, "label"),
        ("external", cv.rename({"label": "true_label"}), "true_label"),
    ]:
        figures._prediction_confusion_matrix(data, tmp_path / f"{name}_matrix.svg", label_col=label)
        figures._prediction_calibration(data, tmp_path / f"{name}_calibration.svg", label_col=label)
        for subdir, expected, count in [
            ("", [[1, 1], [1, 1]], 4),
            ("accepted_only", [[1, 0], [0, 1]], 2),
        ]:
            matrix = rendered[tmp_path / subdir / f"{name}_matrix.svg"].axes[0].images[0]
            np.testing.assert_array_equal(matrix.get_array(), expected)
            calibration = rendered[tmp_path / subdir / f"{name}_calibration.svg"]
            assert calibration.axes[0].get_title() == ""
            assert sum(bar.get_height() for bar in calibration.axes[1].patches) == count
            assert not list(tmp_path.rglob("*.caption.txt"))
            assert not calibration.texts
            matrix_path = tmp_path / subdir / f"{name}_matrix.svg"
            assert "Threshold p" not in matrix_path.read_text()
    assert cv.equals(original)


def test_expression_and_frequency_share_layout_counts_and_species_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = pl.DataFrame(
        {"species": list("abcd"), "label": [0, 0, 1, 1], "prob": [0.1, 0.9, 0.1, 0.9]}
    )
    tables = tmp_path / "cv/tables"
    tables.mkdir(parents=True)
    feature = '["OG1",2,"A"]'
    pl.DataFrame(
        {"species": list("abcd"), "feature": [feature] * 4, "frequency": [0.0, 0.2, 0.4, None]}
    ).write_csv(tables / "top_amino_acid_frequencies.tsv", separator="\t")
    rendered: dict[Path, Figure] = {}
    save = figures._save_svg_figure

    def record(fig: Figure, path: Path) -> None:
        rendered[path] = fig
        save(fig, path)

    monkeypatch.setattr(figures, "_save_svg_figure", record)
    write_frequency_by_confusion(tmp_path, predictions, 1)
    expression_path = tmp_path / "expression.svg"
    figures._top_feature_expression_by_confusion(
        oof_predictions=predictions,
        top_feature_expression=pl.DataFrame(
            {"species": list("abcd"), "feature": ["OG1"] * 4, "tpm": [0.0, 3.0, 15.0, None]}
        ),
        feature_importance=pl.DataFrame({"feature": ["OG1"], "importance_mean": [1.0]}),
        coefficients=pl.DataFrame(
            {"feature": ["OG1"], "coef_mean": [1.0], "method": ["coef_signed"]}
        ),
        out_path=expression_path,
    )
    frequency = rendered[tmp_path / "cv/figures/amino_acid_frequency_by_confusion.svg"]
    expression = rendered[expression_path]
    assert tuple(frequency.get_size_inches()) == tuple(expression.get_size_inches())
    for fig, expected in [(frequency, [0.4, 0.0, 0.2]), (expression, [4.0, 0.0, 2.0])]:
        axis = fig.axes[0]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "TP\nn=0/1",
            "FN\nn=1/1",
            "TN\nn=1/1",
            "FP\nn=1/1",
        ]
        # TP is missing; the remaining values must move with their actual group.
        np.testing.assert_array_equal(
            [points.get_offsets()[0, 0] for points in axis.collections], [2.0, 3.0, 4.0]
        )
        np.testing.assert_allclose(
            [points.get_offsets()[0, 1] for points in axis.collections], expected
        )
    assert expression_path.with_suffix(".pdf").exists()


def test_distribution_artwork_has_no_prose_or_caption_sidecar(tmp_path: Path) -> None:
    predictions = pl.DataFrame({"species": ["sp1"], "label": [1], "prob": [0.7]})
    path = tmp_path / "distribution.svg"
    figures._write_feature_values_by_confusion(
        values=pl.DataFrame({"species": ["sp1"], "feature": ["OG1"], "value": [0.0]}),
        predictions=predictions,
        features=["OG1"],
        out_path=path,
        value_label="Residue frequency",
    )
    svg = path.read_text()
    assert "Classification outcome" in svg
    assert "n=1/1" in svg
    for text in ["Missing values", "available / total", "1.5 IQR", "p &gt;="]:
        assert text not in svg
    assert not path.with_suffix(".caption.txt").exists()
    assert path.with_suffix(".pdf").exists()
