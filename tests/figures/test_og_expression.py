import gzip
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from phenoradar.config import AppConfig
from phenoradar.cv import CVError
from phenoradar.og_expression_figures import (
    _distribution_page,
    _heatmap_figure,
    expression_orthogroups,
    expression_tables,
    read_og_expression,
    write_group_trait_expression,
)


def context() -> pl.DataFrame:
    return pl.DataFrame({"species": ["s0", "s1", "s2", "s3", "s4"],
                         "group_id": ["a", "a", "a", "a", "b"],
                         "label": [0, 0, 0, 1, 1]})


def source(tmp_path: Path, text: str) -> AppConfig:
    path = tmp_path / "tpm.tsv"
    path.write_text("species\torthogroup\ttpm\n"+text)
    return AppConfig.model_validate({"data": {"tpm_path": str(path)}})


def test_selected_reader_custom_columns_gzip_duplicates_and_explicit_zero(tmp_path: Path) -> None:
    path = tmp_path / "tpm.tsv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("sample\tog\tvalue\n s0 \t OG1 \t0\ns1\tOG1\t2\ns1\tOG1\t3\n"
                     "s0\tOTHER\tbad\nexternal\tOG1\tbad\n")
    config = AppConfig.model_validate({"data": {"tpm_path": str(path),
        "species_col": "sample", "feature_col": "og", "value_col": "value"}})
    values = read_og_expression(config, ["s0", "s1", "s2"], ["OG1"])
    assert values.sort("species").to_dicts() == [
        {"species": "s0", "orthogroup": "OG1", "tpm": 0.},
        {"species": "s1", "orthogroup": "OG1", "tpm": 5.}]


@pytest.mark.parametrize("values", ["-1\ns0\tOG1\t2", "NA", "inf", "1e308\ns0\tOG1\t1e308"])
def test_invalid_selected_rows_cannot_be_hidden_by_duplicate_sums(
    tmp_path: Path, values: str,
) -> None:
    config = source(tmp_path, "s0\tOG1\t"+values+"\n")
    with pytest.raises(CVError, match="Invalid expression data"):
        read_og_expression(config, ["s0"], ["OG1"])


def test_equal_species_statistics_sorted_points_and_missing_counts() -> None:
    values = pl.DataFrame({"species": ["s1", "s0", "s3"], "orthogroup": ["OG1"]*3,
                           "tpm": [1023., 0., 7.]})
    summary, points = expression_tables(values, context(), ["OG1", "MISSING"])
    first = summary.row(0, named=True)
    assert tuple(first[k] for k in ["n_total", "n_observed", "n_missing", "n_zero"]) == (3, 2, 1, 1)
    assert first["median_log2_tpm_plus1"] == pytest.approx(5.)
    assert first["median_tpm"] == pytest.approx(511.5)
    assert first["median_log2_tpm_plus1"] != pytest.approx(np.log2(first["median_tpm"]+1))
    ordered = points.filter(pl.col("label") == 0)
    assert ordered["species"].to_list() == ["s0", "s1"]
    assert np.all(np.diff(ordered["plot_x"]) > 0)
    assert summary["n_observed"].sum() == points.height == 3
    assert summary.filter(pl.col("orthogroup") == "MISSING")["n_observed"].sum() == 0
    assert summary.filter((pl.col("group_id") == "b") & (pl.col("trait") == 0))[
        "n_total"].to_list() == [0, 0]
    singleton = points.filter(pl.col("species") == "s3")
    assert singleton["plot_x"].item() == pytest.approx(.19)


def test_encoded_feature_addresses_select_unique_ogs_in_rank_order() -> None:
    importance = pl.DataFrame({"feature": ['["OG1",1,"A"]', '["OG1",2,"V"]',
        '["expression","EXPR"]', 'PLAIN'], "importance_mean": [.4, .3, .2, .1]})
    assert expression_orthogroups(importance, 3) == ["OG1", "EXPR"]


def test_per_og_figures_full_names_populations_and_stale_outputs(tmp_path: Path) -> None:
    config = source(tmp_path, "".join(f"s{i}\tOG{j}\t{i*j}\n"
                                    for i in range(5) for j in range(1, 6)))
    predictions = context().with_columns(pl.Series("decision_status", [
        "accepted", "abstained", "accepted", "accepted", "accepted"]))
    ogs = [f"OG{j}" for j in range(1, 6)]
    annotation = "Complete orthogroup functional annotation with a long descriptive protein name"
    annotations = pl.DataFrame({"feature": ogs, "orthogroup_annotation": [annotation]*5})
    for folder in ["", "accepted_only"]:
        figures = tmp_path / "cv/figures" / folder
        figures.mkdir(parents=True, exist_ok=True)
        (figures / "og_expression_by_group_trait_01.svg").write_text("retired page")
        (figures / "og_expression_by_group_trait_02.svg").write_text("retired page")
        og_dir = figures / "og_expression_by_group_trait"
        og_dir.mkdir()
        (og_dir / "OLD_OG.svg").write_text("retired OG")
    assert write_group_trait_expression(tmp_path, config, predictions, ogs,
                                        annotations=annotations) == []
    for folder, n in [("", 5), ("accepted_only", 4)]:
        figures = tmp_path / "cv/figures" / folder
        tables = tmp_path / "cv/tables" / folder
        assert not list(figures.glob("og_expression_by_group_trait_*.svg"))
        assert {p.stem for p in (figures / "og_expression_by_group_trait").glob("*.svg")
                } == set(ogs)
        assert (figures / "og_expression_by_group_trait.pdf").exists()
        assert (figures / "og_expression_group_trait_heatmap.pdf").exists()
        summary = pl.read_csv(tables / "og_expression_group_trait_summary.tsv", separator="\t")
        points = pl.read_csv(tables / "og_expression_species.tsv", separator="\t")
        assert summary["n_observed"].sum() == points.height == 5*n
        for fig in [_distribution_page(summary, points, [5]), _heatmap_figure(summary)]:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            artists = [*fig.texts]
            for ax in fig.axes:
                artists.extend([ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label,
                                *ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels()])
            for artist in artists:
                if artist.get_visible() and artist.get_text():
                    b = artist.get_window_extent(canvas.get_renderer())
                    assert b.x0 >= -1 and b.y0 >= -1
                    assert b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1
            assert annotation in " ".join(" ".join(t.get_text() for t in artists).split())
            plt.close(fig)
    write_group_trait_expression(tmp_path, config, predictions, ["OG1"])
    for folder in ["", "accepted_only"]:
        og_dir = tmp_path / "cv/figures" / folder / "og_expression_by_group_trait"
        assert [p.name for p in og_dir.glob("*.svg")] == ["OG1.svg"]
    empty = predictions.with_columns(pl.lit("abstained").alias("decision_status"))
    write_group_trait_expression(tmp_path, config, empty, ["OG1"])
    assert "No accepted species" in (tmp_path / "cv/figures/accepted_only/"
                                     "og_expression_group_trait_heatmap.svg").read_text()
    assert not (tmp_path / "cv/figures/accepted_only/og_expression_by_group_trait.pdf").exists()
    assert not (tmp_path / "cv/tables/accepted_only/og_expression_species.tsv").exists()
    og_dir = tmp_path / "cv/figures/accepted_only/og_expression_by_group_trait"
    assert [p.name for p in og_dir.glob("*.svg")] == ["no_accepted_species.svg"]


def test_population_with_only_missing_expression(tmp_path: Path) -> None:
    config = source(tmp_path, "s0\tOG1\t0\n")
    predictions = context().with_columns(pl.Series("decision_status", [
        "abstained", "accepted", "accepted", "accepted", "accepted"]))
    write_group_trait_expression(tmp_path, config, predictions, ["OG1", "OG2"])
    summary = pl.read_csv(tmp_path / "cv/tables/accepted_only/"
                          "og_expression_group_trait_summary.tsv", separator="\t")
    assert summary["n_observed"].sum() == 0
    assert (tmp_path / "cv/figures/accepted_only/og_expression_by_group_trait/OG1.svg").exists()


def test_optional_missing_or_invalid_input_does_not_block_training(tmp_path: Path) -> None:
    config = source(tmp_path, "s0\tOG1\tbad\n")
    assert "skipped" in write_group_trait_expression(tmp_path, config, context(), ["OG1"])[0]
    Path(config.data.tpm_path).unlink()
    assert write_group_trait_expression(tmp_path, config, context(), ["OG1"]) == []
    assert not (tmp_path / "cv/figures").exists()


def test_numeric_group_order_shared_by_points_distributions_and_heatmap(tmp_path: Path) -> None:
    predictions = context().with_columns(pl.Series("group_id", ["10", "2", "1", "2", "10"]))
    values = pl.DataFrame({"species": predictions["species"], "orthogroup": ["OG1"]*5,
                           "tpm": [1., 2., 3., 4., 5.]})
    summary, points = expression_tables(values, predictions, ["OG1"])
    assert summary["group_id"].unique(maintain_order=True).to_list() == ["1", "2", "10"]
    assert points.sort("group_index")["group_id"].unique(maintain_order=True).to_list() == [
        "1", "2", "10"]
    for fig in [_distribution_page(summary, points, [1]), _heatmap_figure(summary)]:
        group_labels = [t for t in fig.axes[0].texts if t.get_text() in {"1", "2", "10"}]
        # Distribution x positions and heatmap y positions follow the same group sequence.
        if len(fig.axes) == 1:
            labels = [t for t in group_labels if t.get_position()[1] == -.29]
            assert [t.get_text() for t in sorted(labels, key=lambda t: t.get_position()[0])
                    ] == ["1", "2", "10"]
        else:
            assert [t.get_text() for t in group_labels if t.get_position()[0] == .02] == [
                "1", "2", "10"]
        plt.close(fig)
    config = source(tmp_path, "s0\tOG1\t1\n")
    config.data.tree_path = str(tmp_path / "unused_tree.nwk")
    assert write_group_trait_expression(tmp_path, config, predictions, ["OG1"]) == []


def test_og_ids_are_encoded_as_single_filenames(tmp_path: Path) -> None:
    config = source(tmp_path, "s0\tOG/sub\t1\ns0\t../OG\t2\n")
    write_group_trait_expression(tmp_path, config, context(), ["OG/sub", "../OG"])
    og_dir = tmp_path / "cv/figures/og_expression_by_group_trait"
    assert {p.name for p in og_dir.iterdir()} == {"OG%2Fsub.svg", "..%2FOG.svg"}
