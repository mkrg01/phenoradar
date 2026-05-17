from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from phenoradar.cli import app
from phenoradar.metadata import (
    MetadataError,
    build_species_metadata_from_skim,
    fetch_ncbi_tree,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_fetch_ncbi_tree_runs_nwkit_constrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    species_trait = _write(
        tmp_path / "species_trait.tsv",
        "\n".join(
            [
                "species\tC4",
                "Zea_mays\t1",
                "Oryza_sativa\t0",
            ]
        )
        + "\n",
    )
    tree_out = tmp_path / "tree.nwk"
    captured: dict[str, Any] = {}

    def fake_run(
        command: list[str] | tuple[str, ...], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        species_list_path = Path(command[command.index("--species_list") + 1])
        captured["species_list"] = species_list_path.read_text(encoding="utf-8")
        outfile = Path(command[command.index("--outfile") + 1])
        outfile.write_text("(Zea_mays,Oryza_sativa);\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = fetch_ncbi_tree(
        species_trait,
        tree_out,
        rank="genus",
        nwkit_bin="nwkit-test",
    )

    assert result.tree_path == tree_out
    assert result.species_count == 2
    assert result.rank == "genus"
    assert captured["command"] == [
        "nwkit-test",
        "constrain",
        "--backbone",
        "ncbi",
        "--species_list",
        captured["command"][5],
        "--rank",
        "genus",
        "--outfile",
        captured["command"][9],
    ]
    assert captured["kwargs"]["check"] is False
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True
    assert captured["species_list"] == "Zea_mays\nOryza_sativa\n"


def test_fetch_ncbi_tree_runs_nwkit_constrain_with_taxid_tsv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    species_taxid = _write(
        tmp_path / "species_taxid.tsv",
        "\n".join(
            [
                "species\ttaxid",
                "Zea_mays\t4577",
                "Oryza_sativa\t4530",
            ]
        )
        + "\n",
    )
    tree_out = tmp_path / "tree.nwk"
    captured: dict[str, Any] = {}

    def fake_run(
        command: list[str] | tuple[str, ...], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        taxid_tsv_path = Path(command[command.index("--taxid_tsv") + 1])
        captured["taxid_tsv"] = taxid_tsv_path.read_text(encoding="utf-8")
        outfile = Path(command[command.index("--outfile") + 1])
        outfile.write_text("(Zea_mays,Oryza_sativa);\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = fetch_ncbi_tree(
        tmp_path / "missing_species_trait.tsv",
        tree_out,
        species_taxid_path=species_taxid,
        nwkit_bin="nwkit-test",
    )

    assert result.tree_path == tree_out
    assert result.species_count == 2
    assert captured["command"] == [
        "nwkit-test",
        "constrain",
        "--backbone",
        "ncbi",
        "--taxid_tsv",
        captured["command"][5],
        "--rank",
        "family",
        "--outfile",
        captured["command"][9],
    ]
    assert captured["kwargs"]["check"] is False
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True
    assert captured["taxid_tsv"] == "leaf_name\ttaxid\nZea_mays\t4577\nOryza_sativa\t4530\n"


def test_fetch_ncbi_tree_rejects_invalid_taxid_values(tmp_path: Path) -> None:
    species_taxid = _write(
        tmp_path / "species_taxid.tsv",
        "species\ttaxid\nZea_mays\tbad\n",
    )

    with pytest.raises(MetadataError, match="taxid column must contain"):
        fetch_ncbi_tree(
            tmp_path / "missing_species_trait.tsv",
            tmp_path / "tree.nwk",
            species_taxid_path=species_taxid,
        )


def test_fetch_ncbi_tree_rejects_duplicate_species(tmp_path: Path) -> None:
    species_trait = _write(
        tmp_path / "species_trait.tsv",
        "\n".join(
            [
                "species\tC4",
                "Zea_mays\t1",
                "Zea_mays\t0",
            ]
        )
        + "\n",
    )

    with pytest.raises(MetadataError, match="duplicates: Zea_mays"):
        fetch_ncbi_tree(species_trait, tmp_path / "tree.nwk")


def test_fetch_ncbi_tree_rejects_existing_output_without_force(tmp_path: Path) -> None:
    species_trait = _write(tmp_path / "species_trait.tsv", "species\tC4\nZea_mays\t1\n")
    tree_out = _write(tmp_path / "tree.nwk", "(Zea_mays);\n")

    with pytest.raises(MetadataError, match="already exists"):
        fetch_ncbi_tree(species_trait, tree_out)


def test_metadata_cli_writes_ncbi_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    species_trait = _write(tmp_path / "species_trait.tsv", "species\tC4\nZea_mays\t1\n")
    tree_out = tmp_path / "tree.nwk"

    def fake_run(
        command: list[str] | tuple[str, ...], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        outfile = Path(command[command.index("--outfile") + 1])
        outfile.write_text("(Zea_mays);\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "metadata",
            "--species-trait",
            str(species_trait),
            "--tree-out",
            str(tree_out),
            "--rank",
            "genus",
            "--tree-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote NCBI taxonomy tree" in result.output
    assert "species=1" in result.output
    assert tree_out.read_text(encoding="utf-8") == "(Zea_mays);\n"


def test_metadata_cli_writes_ncbi_tree_with_species_taxid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    species_taxid = _write(tmp_path / "species_taxid.tsv", "species\ttaxid\nZea_mays\t4577\n")
    tree_out = tmp_path / "tree.nwk"

    def fake_run(
        command: list[str] | tuple[str, ...], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        assert "--taxid_tsv" in command
        outfile = Path(command[command.index("--outfile") + 1])
        outfile.write_text("(Zea_mays);\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "metadata",
            "--species-taxid",
            str(species_taxid),
            "--tree-out",
            str(tree_out),
            "--tree-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote NCBI taxonomy tree" in result.output
    assert "species=1" in result.output
    assert tree_out.read_text(encoding="utf-8") == "(Zea_mays);\n"


def test_build_species_metadata_from_skim_writes_contrast_pair_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    species_trait = _write(
        tmp_path / "species_trait.tsv",
        "\n".join(
            [
                "species\tC4",
                "sp1\t1",
                "sp2\t0",
                "sp3\t1",
                "sp4\t",
            ]
        )
        + "\n",
    )
    tree = _write(tmp_path / "tree.nwk", "((sp1,sp2),sp3,sp4);\n")
    out = tmp_path / "species_metadata.tsv"
    captured: dict[str, Any] = {}

    def fake_run(
        command: list[str] | tuple[str, ...], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if command[1] == "nwk2table":
            outfile = Path(command[command.index("--outfile") + 1])
            outfile.write_text(
                "branch_id\tparent\tname\tdist\tsupport\tsister\n"
                "0\t-1\t\t\t\t-1\n"
                "1\t0\tsp1\t\t\t2\n"
                "2\t0\tsp2\t\t\t1\n"
                "3\t0\tsp3\t\t\t1\n"
                "4\t0\tsp4\t\t\t1\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        trait_path = Path(command[command.index("--trait") + 1])
        captured["trait"] = trait_path.read_text(encoding="utf-8")
        outfile = Path(command[command.index("--outfile") + 1])
        all_groupfile = Path(f"{str(outfile).removesuffix('.nwk')}.all.tsv")
        all_groupfile.write_text(
            "\n".join(
                [
                    "leaf_name\tC4\tgroup\tcontrastive_clade",
                    "sp1\t1\t1\t1",
                    "sp2\t0\t2\t1",
                    "sp3\t1\t3\t",
                    "sp4\t\t4\t1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        outfile.write_text("(sp1,sp2);\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = build_species_metadata_from_skim(
        species_trait,
        tree,
        out,
        nwkit_bin="nwkit-test",
    )

    assert result.metadata_path == out
    assert result.species_count == 4
    assert result.grouped_species_count == 2
    assert result.contrast_pair_count == 1
    assert result.tree_missing_species_count == 0
    assert captured["command"] == [
        "nwkit-test",
        "skim",
        "--infile",
        str(tree),
        "--outfile",
        captured["command"][5],
        "--trait",
        captured["command"][7],
        "--group-by",
        "C4",
        "--only-contrastive-clades",
        "yes",
        "--output-groupfile",
        "yes",
    ]
    assert captured["kwargs"]["check"] is False
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True
    assert captured["trait"] == "leaf_name\tC4\nsp1\t1\nsp2\t0\nsp3\t1\nsp4\t\n"
    assert out.read_text(encoding="utf-8") == (
        "species\tC4\tcontrast_pair_id\n"
        "sp1\t1\t1\n"
        "sp2\t0\t1\n"
        "sp3\t1\t\n"
        "sp4\t\t\n"
    )


def test_metadata_cli_writes_metadata_with_existing_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    species_trait = _write(tmp_path / "species_trait.tsv", "species\tC4\nsp1\t1\nsp2\t0\n")
    tree = _write(tmp_path / "tree.nwk", "(sp1,sp2);\n")
    out = tmp_path / "species_metadata.tsv"

    def fake_run(
        command: list[str] | tuple[str, ...], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        if command[1] == "nwk2table":
            outfile = Path(command[command.index("--outfile") + 1])
            outfile.write_text(
                "branch_id\tparent\tname\tdist\tsupport\tsister\n"
                "0\t-1\t\t\t\t-1\n"
                "1\t0\tsp1\t\t\t2\n"
                "2\t0\tsp2\t\t\t1\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert command[1] == "skim"
        outfile = Path(command[command.index("--outfile") + 1])
        all_groupfile = Path(f"{str(outfile).removesuffix('.nwk')}.all.tsv")
        all_groupfile.write_text(
            "leaf_name\tC4\tgroup\tcontrastive_clade\nsp1\t1\t1\t1\nsp2\t0\t2\t1\n",
            encoding="utf-8",
        )
        outfile.write_text("(sp1,sp2);\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "metadata",
            "--species-trait",
            str(species_trait),
            "--tree-in",
            str(tree),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Using existing tree" in result.output
    assert "Wrote species metadata" in result.output
    assert "contrast_pairs=1" in result.output
    assert "tree_missing_species=0" in result.output
    assert out.read_text(encoding="utf-8") == (
        "species\tC4\tcontrast_pair_id\nsp1\t1\t1\nsp2\t0\t1\n"
    )


def test_build_species_metadata_from_skim_writes_empty_groups_when_no_contrasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    species_trait = _write(tmp_path / "species_trait.tsv", "species\tC4\nsp1\t1\nsp2\t1\n")
    tree = _write(tmp_path / "tree.nwk", "(sp1,sp2);\n")
    out = tmp_path / "species_metadata.tsv"

    def fake_run(
        command: list[str] | tuple[str, ...], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        if command[1] == "nwk2table":
            outfile = Path(command[command.index("--outfile") + 1])
            outfile.write_text(
                "branch_id\tparent\tname\tdist\tsupport\tsister\n"
                "0\t-1\t\t\t\t-1\n"
                "1\t0\tsp1\t\t\t2\n"
                "2\t0\tsp2\t\t\t1\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="ValueError: No leaves were selected for output.",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = build_species_metadata_from_skim(species_trait, tree, out)

    assert result.grouped_species_count == 0
    assert result.contrast_pair_count == 0
    assert result.tree_missing_species_count == 0
    assert out.read_text(encoding="utf-8") == (
        "species\tC4\tcontrast_pair_id\nsp1\t1\t\nsp2\t1\t\n"
    )


def test_build_species_metadata_from_skim_excludes_species_missing_from_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    species_trait = _write(
        tmp_path / "species_trait.tsv",
        "species\tC4\nsp1\t1\nsp2\t0\nmissing_from_tree\t1\n",
    )
    tree = _write(tmp_path / "tree.nwk", "(sp1,sp2);\n")
    out = tmp_path / "species_metadata.tsv"

    def fake_run(
        command: list[str] | tuple[str, ...], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        outfile = Path(command[command.index("--outfile") + 1])
        if command[1] == "nwk2table":
            outfile.write_text(
                "branch_id\tparent\tname\tdist\tsupport\tsister\n"
                "0\t-1\t\t\t\t-1\n"
                "1\t0\tsp1\t\t\t2\n"
                "2\t0\tsp2\t\t\t1\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        trait_path = Path(command[command.index("--trait") + 1])
        assert trait_path.read_text(encoding="utf-8") == "leaf_name\tC4\nsp1\t1\nsp2\t0\n"
        all_groupfile = Path(f"{str(outfile).removesuffix('.nwk')}.all.tsv")
        all_groupfile.write_text(
            "leaf_name\tC4\tgroup\tcontrastive_clade\nsp1\t1\t1\t1\nsp2\t0\t2\t1\n",
            encoding="utf-8",
        )
        outfile.write_text("(sp1,sp2);\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = build_species_metadata_from_skim(species_trait, tree, out)

    assert result.species_count == 2
    assert result.grouped_species_count == 2
    assert result.tree_missing_species_count == 1
    assert out.read_text(encoding="utf-8") == (
        "species\tC4\tcontrast_pair_id\n"
        "sp1\t1\t1\n"
        "sp2\t0\t1\n"
    )
