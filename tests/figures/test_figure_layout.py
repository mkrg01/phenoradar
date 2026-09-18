"""Layout checks use rendered text extents, so regressions cannot hide in valid SVGs."""

from itertools import combinations
from pathlib import Path

import polars as pl
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from phenoradar import figures
from phenoradar.phylogenetic_figures import _comparison_scatter
from phenoradar.phylogenetic_imputation import PhylogeneticImputationArtifacts


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    result: list[Figure] = []
    save = figures._save_svg_figure

    def capture(fig: Figure, path: Path) -> None:
        save(fig, path)
        result.append(fig)

    monkeypatch.setattr(figures, "_save_svg_figure", capture)
    monkeypatch.setattr("phenoradar.phylogenetic_figures._save_svg_figure", capture)
    return result


def assert_text_inside_canvas(fig: Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox.padded(1)
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend(ax.texts)
        if ax.axison:
            artists.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
            for axis in (ax.xaxis, ax.yaxis):
                for tick in axis._update_ticks():
                    artists.extend([tick.label1, tick.label2])
    for artist in artists:
        if not artist.get_visible() or not artist.get_text():
            continue
        box = Text.get_window_extent(artist, renderer)
        assert canvas.contains(box.x0, box.y0), artist.get_text()
        assert canvas.contains(box.x1, box.y1), artist.get_text()


@pytest.mark.parametrize("selection_rule", ["best", "one_se"])
def test_model_selection_panels_have_separate_label_space(
    tmp_path: Path, rendered: list[Figure], selection_rule: str,
) -> None:
    summary = pl.DataFrame([
        {"fold_id": str(fold), "sample_set_id": 0, "candidate_index": candidate,
         "metric_name": "log_loss", "metric_value_mean": 0.3 + 0.1 * candidate,
         "metric_value_std": 0.03, "n_valid_inner_folds": 3,
         "params_json": '{"C":' + str(10 ** (candidate / 2 - 3)) + '}'}
        for fold in range(5) for candidate in range(4)
    ])
    figures._model_selection_curve(
        summary, None, tmp_path / "model_selection.svg", max_sample_sets_per_fold=1,
        selection_rule=selection_rule,
    )
    fig = rendered[-1]
    assert_text_inside_canvas(fig)
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.axison]
    for first, second in combinations(boxes, 2):
        assert not first.overlaps(second)
    assert fig.get_figwidth() <= 7.2
    # Rule-specific artists, not just their legend text, must be absent for best.
    for ax in fig.axes:
        if ax.axison:
            assert any(line.get_label() == "one-SE threshold" for line in ax.lines) == (
                selection_rule == "one_se"
            )
            assert any(mark.get_label() == "Within one-SE" for mark in ax.collections) == (
                selection_rule == "one_se"
            )
            assert any(len(mark.get_segments()) > 0 for mark in ax.collections
                       if hasattr(mark, "get_segments"))



@pytest.mark.parametrize("parameter_rows", [
    ['{"lambda":1}', '{"lambda":10}', '{"C":1}', '{"C":10}'],
    ['{"lambda":1,"alpha":0.1}', '{"lambda":1,"alpha":0.9}'] * 2,
])
def test_selection_curve_uses_candidate_coordinates_for_mixed_parameter_axes(
    tmp_path: Path, rendered: list[Figure], parameter_rows: list[str],
) -> None:
    summary = pl.DataFrame({
        "fold_id": ["0", "0", "1", "1"], "sample_set_id": [0] * 4,
        "candidate_index": [2, 4, 2, 4], "metric_name": ["mcc"] * 4,
        "metric_value_mean": [0.3, 0.5, 0.4, 0.6], "metric_value_se": [0.02] * 4,
        "params_json": parameter_rows,
    })
    figures._model_selection_curve(
        summary, None, tmp_path / "model_selection.svg", max_sample_sets_per_fold=1,
    )
    for ax in rendered[-1].axes:
        if ax.axison:
            assert ax.get_xlabel() == "candidate_index"
            assert ax.lines[0].get_xdata().tolist() == [2.0, 4.0]


def test_long_feature_labels_and_colorbar_fit_on_page(
    tmp_path: Path, rendered: list[Figure],
) -> None:
    features = [f"OG{i}" for i in range(12)]
    importance = pl.DataFrame({"feature": features, "importance_mean": [0.1] * 12})
    annotations = pl.DataFrame({
        "feature": features,
        "orthogroup_annotation": [
            "A very long photosynthetic enzyme annotation with multiple domains and cofactors"
        ] * 12,
    })
    by_fold = pl.DataFrame([
        {"fold_id": str(fold), "feature": feature, "importance_mean": 0.1}
        for fold in range(3) for feature in features
    ])
    figures._feature_importance_by_fold_heatmap(
        importance, by_fold, tmp_path / "heatmap.svg", orthogroup_annotations=annotations,
    )
    fig = rendered[-1]
    assert_text_inside_canvas(fig)
    renderer = fig.canvas.get_renderer()
    labels = [label.get_window_extent(renderer) for label in fig.axes[0].get_yticklabels()]
    for first, second in combinations(labels, 2):
        assert not first.overlaps(second)
    assert not fig.axes[0].get_tightbbox(renderer).overlaps(fig.axes[1].get_tightbbox(renderer))
    assert fig.get_figwidth() <= 7.2


def test_metric_legend_is_outside_data_panel(tmp_path: Path, rendered: list[Figure]) -> None:
    metrics = pl.DataFrame([
        {"aggregate_scope": scope, "fold_id": "NA", "metric": metric, "metric_value": 1.0}
        for scope in ["macro", "micro"] for metric in ["mcc", "balanced_accuracy", "roc_auc"]
    ])
    figures._cv_metrics_overview(metrics, tmp_path / "metrics.svg")
    fig = rendered[-1]
    assert_text_inside_canvas(fig)
    ax = fig.axes[0]
    assert not ax.bbox.overlaps(ax.get_legend().get_window_extent(fig.canvas.get_renderer()))


def test_crowded_phylogenetic_labels_have_separate_callouts(
    tmp_path: Path, rendered: list[Figure],
) -> None:
    artifacts = PhylogeneticImputationArtifacts(comparison=pl.DataFrame([
        {"species": f"Long_species_name_with_subspecies_{i}", "phylo_prob": 0.02,
         "prob": 0.98, "prob_difference": 0.96, "decision_status": "accepted"}
        for i in range(10)
    ]))
    _comparison_scatter(artifacts, tmp_path / "phylogeny.svg")
    fig = rendered[-1]
    assert_text_inside_canvas(fig)
    renderer = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(label, renderer) for label in fig.axes[1].texts]
    for first, second in combinations(boxes, 2):
        assert not first.overlaps(second)
