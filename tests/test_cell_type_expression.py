import gzip
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import polars as pl
import pytest
import yaml
from scipy import sparse
from test_amino_acid import fixture_config
from test_combined import combined_config
from typer.testing import CliRunner

from phenoradar.cell_type_enrichment import _null_means
from phenoradar.cell_type_expression import (
    _read_cpm,
    rank_site_orthogroups,
    row_zscores,
    write_cell_type_expression_figures,
)
from phenoradar.cli import app
from phenoradar.config import AppConfig
from phenoradar.config.schema import CellTypeExpressionConfig
from phenoradar.figures import FigureError


@pytest.mark.parametrize("representation", ["dense", "sparse", "layer"])
def test_cpm_pools_counts_and_preserves_missing_values(
    tmp_path: Path, representation: str,
) -> None:
    ad = pytest.importorskip("anndata")
    counts = np.array([[1, 3, 6], [2, 1, 7], [0, 1, 4], [0, 0, 0], [50, 0, 0]])
    atlas = ad.AnnData(X=sparse.csr_matrix(counts) if representation == "sparse" else counts)
    atlas.obs["custom_type"] = ["A", "A", "B", "Empty", "Unknown"]
    atlas.var_names = ["gene1", "gene2", "background"]
    layer = None
    if representation == "layer":
        layer = "counts"
        atlas.layers[layer] = sparse.csr_matrix(counts)
        atlas.X = np.log1p(counts)
    path = tmp_path / "atlas.h5ad"
    atlas.write_h5ad(path)
    mapping = tmp_path / "mapping.tsv.gz"
    with gzip.open(mapping, "wt") as handle:
        handle.write(
            "gene_id\torthogroup\n"
            "gene1\tOG_A\ngene1\tOG_A\ngene2\tOG_A\ngene1\tOG_B\n"
            "absent\tOG_absent\n",
        )
    config = CellTypeExpressionConfig(
        h5ad_path=str(path), gene_to_orthogroup_path=str(mapping), cell_type_col="custom_type",
        cell_types=["B", "A", "Empty"], counts_layer=layer,
    )
    labels, cpm, warnings = _read_cpm(config, ["OG_A", "OG_B", "OG_absent"])
    assert labels == ["B", "A", "Empty"]
    np.testing.assert_allclose(
        cpm, [[200_000, 350_000, np.nan], [0, 150_000, np.nan], [np.nan] * 3],
    )
    expected_z = 1 / np.sqrt(2)
    np.testing.assert_allclose(
        row_zscores(cpm), [[-expected_z, expected_z, np.nan]] * 2 + [[np.nan] * 3],
    )
    assert np.isnan(row_zscores(np.array([[1.0, 1.0], [np.nan, 0.0]]))).all()
    assert "2 of 3" in warnings[0]
    config.cell_types = None
    assert _read_cpm(config, ["OG_A"])[0] == ["A", "B", "Empty"]
    if representation == "layer":
        config.counts_layer = None
        with pytest.raises(FigureError, match="nonnegative raw counts"):
            _read_cpm(config, ["OG_A"])


@pytest.mark.parametrize(
    ("filename", "score"),
    [("amino_acid_site_scores.tsv", "normalized_pss"),
     ("amino_acid_site_pss.tsv", "normalized_score")],
)
def test_rank_og_sums_sites_before_applying_limit(
    tmp_path: Path, filename: str, score: str,
) -> None:
    (tmp_path / filename).write_text(
        f"orthogroup\t{score}\nOG1\t0.4\nOG2\t0.3\nOG2\t0.2\nOG3\t0\n",
    )
    assert rank_site_orthogroups(tmp_path, 1) == ["OG2"]


def test_site_weighted_null_draws_distinct_ogs_with_successive_weights() -> None:
    draws = _null_means(np.eye(3), 2, np.array([1., 2., 5.]), 48, 30_000)
    assert np.isin(draws, [0., 0.5]).all()
    np.testing.assert_array_equal(draws.sum(axis=1), 1)
    # The missing OG probabilities are obtained by summing both possible draw orders.
    expected = 1 - np.array([2 / 8 * 5 / 6 + 5 / 8 * 2 / 3,
                             1 / 8 * 5 / 7 + 5 / 8 * 1 / 3,
                             1 / 8 * 2 / 7 + 2 / 8 * 1 / 6])
    np.testing.assert_allclose(2 * draws.mean(axis=0), expected, atol=0.012, rtol=0)


@pytest.mark.parametrize(
    ("mode", "stage", "report_format", "formats"),
    [
        ("expression", "cv_only", "compact", None),
        ("expression", "full_run", "compact", None),
        ("amino_acid", "cv_only", "compact", None),
        ("amino_acid", "full_run", "full", None),
        ("amino_acid", "full_run", "compact", ["svg", "pdf"]),
        ("combined", "cv_only", "full", None),
        ("combined", "full_run", "compact", None),
        ("expression", "cv_only", "compact", ["pdf"]),
        ("amino_acid", "cv_only", "compact", ["svg", "pdf"]),
        ("combined", "cv_only", "compact", ["svg"]),
    ],
)
def test_run_writes_atlas_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, stage: str, report_format: str,
    formats: list[str] | None,
) -> None:
    ad = pytest.importorskip("anndata")
    path = (fixture_config(tmp_path, "aa_frequency") if mode == "amino_acid"
            else combined_config(tmp_path))
    config = yaml.safe_load(path.read_text())
    config["data"]["feature_mode"] = mode
    config["runtime"]["execution_stage"] = stage
    config["figures"] = {"top_features": 1, "formats": formats}
    expect_enrichment = stage == "full_run" and mode in {"expression", "amino_acid"}
    if expect_enrichment and mode == "amino_acid":
        config["preprocess"]["amino_acid"]["ranked_site_filter"] = {"method": "none"}
        alignments = Path(config["data"]["alignment_dir"])
        extension = next(alignments.iterdir()).suffix
        (alignments / f"NOISE{extension}").write_text("".join(
            f">s{i}_gene\n{'AAA' if (i // 2) % 2 else 'VVV'}\n" for i in range(8)
        ))
    if formats is not None:
        tree = tmp_path / "tree.nwk"
        tree.write_text("((s0,s1),(s2,s3),(s4,s5),(s6,s7),external,unknown);\n")
        config["data"]["tree_path"] = str(tree)
        config["runtime"]["n_jobs"] = 2
        if mode == "amino_acid":
            tpm = tmp_path / "expression.tsv"
            tpm.write_text("species\torthogroup\ttpm\n" + "".join(
                f"s{i}\tOG1\t{i + 1}\n" for i in range(8)
            ))
            config["data"]["tpm_path"] = str(tpm)
    config["report"] = {
        "amino_acid_tables": report_format,
        "cell_type_expression": {
            "h5ad_path": "atlas.h5ad",
            "gene_to_orthogroup_path": "mapping.tsv",
            "cell_types": ["type_b", "type_a"],
        },
    }
    # Expression and residues encode the same signal. At this penalty the
    # combined fit attributes the signal to expression alone.
    if mode == "combined":
        config["model_selection"]["search_space"]["amino_acid_penalty_ratio"] = [1]
    elif mode == "expression":
        config["model_selection"]["search_space"].pop("amino_acid_penalty_ratio")
    annotation = tmp_path / "ogs.tsv"
    annotation.write_text("OG1\t3193\tSequence group\nEXPR\t3193\tExpression group\n")
    config["data"]["orthogroup_annotation_path"] = str(annotation)
    path.write_text(yaml.safe_dump(config))
    atlas = ad.AnnData(X=np.array([[2, 1, 3, 4], [1, 3, 1, 5]]))
    atlas.obs["cell_type"] = ["type_a", "type_b"]
    atlas.var_names = ["seq_gene", "expr_gene", "noise_gene", "background"]
    atlas.write_h5ad(tmp_path / "atlas.h5ad")
    (tmp_path / "mapping.tsv").write_text(
        "gene_id\torthogroup\nseq_gene\tOG1\nexpr_gene\tEXPR\nnoise_gene\tNOISE\n",
    )

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "-c", str(path), "--quiet"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    run = next(directory for directory in (tmp_path / "runs").iterdir() if directory.is_dir()
               and not directory.name.startswith("."))
    figure_dir = run / ("model" if stage == "full_run" else "cv") / "figures"
    selected = formats or ["svg"]
    stems = {"cell_type_expression_cpm": "CPM", "cell_type_expression_zscore": "Row z-score"}
    assert set(run.glob("*/figures/cell_type_expression_*")) == {
        figure_dir / f"{stem}.{extension}" for extension in selected for stem in stems
    }
    for stem, scale in stems.items():
        if "pdf" in selected:
            assert (figure_dir / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        if "svg" in selected:
            figure = figure_dir / f"{stem}.svg"
            labels = [node.text for node in ET.parse(figure).iter("{http://www.w3.org/2000/svg}text")]
            assert scale in labels
            assert labels.index("Type b") < labels.index("Type a")
            assert ("Sequence group (OG1)" in labels) == (mode == "amino_acid")
            assert ("Expression group (EXPR)" in labels) == (mode != "amino_acid")
    enrichment = run / "model/tables/cell_type_enrichment.tsv"
    assert enrichment.is_file() == expect_enrichment
    assert set(run.glob("*/figures/cell_type_enrichment*")) == (
        {figure_dir / f"cell_type_enrichment_null.{extension}" for extension in selected}
        if expect_enrichment else set()
    )
    if expect_enrichment:
        table = pl.read_csv(enrichment, separator="\t")
        assert table["cell_type"].to_list() == ["type_b", "type_a"]
        assert table["n_draws"].to_list() == [50_000, 50_000]
        assert table["n_selected_analyzed"].to_list() == [1, 1]
        assert table["n_eligible_analyzed"].to_list() == [2, 2]
        p = table["p_enrichment"].min()
        assert table["p_holm"].min() == pytest.approx(min(1, 2 * p))
        if mode == "expression":
            assert table["observed_pct"][0] == pytest.approx(75)
            assert table["expected_pct"][0] == pytest.approx(50)
        else:
            assert table["observed_pct"][0] == pytest.approx(100 / 3)
            assert table["expected_pct"][0] == pytest.approx(0.4 * 100 / 3 + 0.6 * 25, abs=0.1)
        if "svg" in selected:
            root = ET.parse(figure_dir / "cell_type_enrichment_null.svg").getroot()
            labels = list(root.itertext())
            assert "Null distribution" in labels and "Mean share of OG expression (%)" in labels
            assert "Type b" in labels and "Type a" in labels
            assert not any("selected OGs analyzed" in text or " mode" in text for text in labels)
        if "pdf" in selected:
            assert (figure_dir / "cell_type_enrichment_null.pdf").read_bytes().startswith(b"%PDF-")
    if formats is not None:
        cv_figures = run / "cv/figures"
        expected = ["cv_confusion_matrix", "og_expression_group_trait_heatmap"]
        if mode == "expression":
            expected.append("tree_feature_heatmap_zscore")
        elif mode == "amino_acid":
            expected.append("tree_feature_heatmap_amino_acid")
        for stem in expected:
            assert {path.suffix for path in cv_figures.glob(f"{stem}.*")} == {
                f".{extension}" for extension in selected
            }
        distributions = cv_figures / "og_expression_by_group_trait"
        assert distributions.with_suffix(".pdf").exists() == ("pdf" in selected)
        assert bool(list(distributions.glob("*.svg"))) == ("svg" in selected)


def test_combined_atlas_uses_both_rankings_and_displays_overlap_once(tmp_path: Path) -> None:
    ad = pytest.importorskip("anndata")
    atlas_path = tmp_path / "atlas.h5ad"
    data = ad.AnnData(X=np.array([[1, 2, 3, 4], [3, 2, 1, 4]]))
    data.obs["cell_type"] = ["A", "B"]
    data.var_names = ["g1", "g2", "g3", "background"]
    data.write_h5ad(atlas_path)
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("gene_id\torthogroup\ng1\tOG1\ng2\tOG2\ng3\tOG3\n")
    config = AppConfig.model_validate({
        "data": {"feature_mode": "combined", "alignment_dir": "unused"},
        "preprocess": {
            "amino_acid": {"aggregation": "aa_frequency"},
            "expression": {
                "missing_expression": {"method": "neutral"}, "absent_feature_fill": "nan",
            },
        },
        "model_selection": {"search_space": {"alpha": [1]}},
        "phylogenetic_imputation": {"enabled": False},
        "figures": {"top_features": 2},
        "runtime": {"execution_stage": "full_run"},
        "report": {"cell_type_expression": {
            "h5ad_path": str(atlas_path), "gene_to_orthogroup_path": str(mapping),
        }},
    })
    tables = tmp_path / "model/tables"
    tables.mkdir(parents=True)
    (tables / "amino_acid_site_scores.tsv").write_text(
        "orthogroup\tnormalized_pss\nOG3\t0.6\nOG2\t0.3\nOG4\t0.1\n",
    )
    importance = pl.DataFrame({
        "feature": ['["expression","OG1"]', '["expression","OG2"]', '["expression","OG0"]',
                    '["OG5",1,"A"]'],
        "importance_mean": [0.7, 0.2, 0.1, 0.9],
    })
    warnings = write_cell_type_expression_figures(
        tmp_path, config, expression_importance=importance,
    )
    assert warnings == []
    figure = tmp_path / "model/figures/cell_type_expression_cpm.svg"
    labels = [node.text for node in ET.parse(figure).iter("{http://www.w3.org/2000/svg}text")]
    assert [label for label in labels if label and label.startswith("OG")] == ["OG1", "OG2", "OG3"]
