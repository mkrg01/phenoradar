from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from phenoradar.amino_acid import AlignmentStore
from phenoradar.cli import app
from phenoradar.site_reference import (
    ReferencePolicy,
    annotate_sites,
    core_columns,
    project_references,
    reference_order,
    score_reference,
)
from phenoradar.site_reference_resources import (
    ResourceCache,
    SiteAnnotationError,
    sha256,
    write_json,
)
from phenoradar.site_structure import sifts_mapping, structure_atoms

MAFFT = os.environ.get("PHENORADAR_TEST_MAFFT") or shutil.which("mafft")
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
REFERENCE = "ACPP" + SEQUENCE[2:]
THREE = dict(
    zip(
        SEQUENCE,
        [
            "ALA",
            "CYS",
            "ASP",
            "GLU",
            "PHE",
            "GLY",
            "HIS",
            "ILE",
            "LYS",
            "LEU",
            "MET",
            "ASN",
            "PRO",
            "GLN",
            "ARG",
            "SER",
            "THR",
            "VAL",
            "TRP",
            "TYR",
        ],
        strict=True,
    )
)


def reference(accession: str = "P00001", sequence: str = REFERENCE) -> dict[str, Any]:
    return dict(
        orthogroup="OG1",
        accession=accession,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
        reviewed=True,
        fragment=False,
        membership_verified=True,
        orthodb_gene_ids=["odb_gene"],
    )


def make_pdb(
    path: Path, sequence: str, *, missing: set[int] | None = None, insertion: int | None = None
) -> None:
    lines = []
    for i, aa in enumerate(sequence, 1):
        if i in (missing or set()):
            continue
        code = "A" if i == insertion else " "
        x, y, z = 6 * np.cos(i * 0.8), 6 * np.sin(i * 0.8), i * 1.5
        lines.append(
            f"ATOM  {i:5d}  CA  {THREE[aa]} A{i + 100:4d}{code}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{85.0:6.2f}           C  "
        )
    path.write_text("\n".join(lines) + "\nTER\nEND\n")


def make_sifts(path: Path, *, missing: set[int] | None = None) -> None:
    residues = []
    for i, aa in enumerate(REFERENCE, 1):
        ins = "A" if i == 5 else ""
        note = (
            '<residueDetail property="Annotation">Not_Observed</residueDetail>'
            if i in (missing or set())
            else ""
        )
        residues.append(
            f'<residue><crossRefDb dbSource="PDB" dbChainId="A" '
            f'dbResNum="{i + 100}{ins}" dbResName="{THREE[aa]}"/>'
            f'<crossRefDb dbSource="UniProt" dbAccessionId="P00001" '
            f'dbResNum="{i}"/>{note}</residue>'
        )
    path.write_text(
        '<entry xmlns="http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd">'
        "<entity><segment><listResidue>"
        + "".join(residues)
        + "</listResidue></segment></entity></entry>"
    )


def fixture_run(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    run = root / "run"
    align = root / "align"
    align.mkdir()
    (align / "OG1.fa").write_text("".join(f">s{i}_gene\n{SEQUENCE}\n" for i in range(4)))
    for stage in ["model", "cv", "split"]:
        (run / stage / "tables").mkdir(parents=True)
    write_json(run / "run_metadata.json", dict(status="full_run_completed"))
    (run / "resolved_config.yml").write_text(
        yaml.safe_dump(
            dict(
                data=dict(
                    feature_mode="amino_acid",
                    alignment_dir=str(align),
                    orthogroup_annotation_path="odb12v2_OGs.tab.gz",
                )
            )
        )
    )
    pl.DataFrame(
        dict(species=[f"s{i}" for i in range(4)], pool=["validation"] * 4, fold_id=[1, 1, 2, 2])
    ).write_csv(run / "split/tables/split_manifest.tsv", separator="\t")
    pl.DataFrame(
        dict(orthogroup=["OG1"], sha256=[sha256(align / "OG1.fa")], alignment_length=[20])
    ).write_csv(run / "model/tables/alignment_inputs.tsv", separator="\t")
    base = dict(
        scope=["final_refit"] * 2,
        sample_set_id=[0, 0],
        candidate_index=[1, 1],
        orthogroup=["OG1"] * 2,
        alignment_pos_1based=[3, 18],
    )
    for name, metric in [("effect_range", [2.0, 1.0]), ("pss", [2.0, 1.0])]:
        pl.DataFrame({**base, name: metric}).write_csv(
            run / "model/tables" / f"amino_acid_site_{name}.tsv", separator="\t"
        )
    pl.DataFrame(
        dict(orthogroup=["OG1"] * 2, alignment_pos_1based=[3, 18], selection_frequency=[1.0, 0.5])
    ).write_csv(run / "cv/tables/amino_acid_site_stability.tsv", separator="\t")
    make_pdb(root / "model.pdb", REFERENCE, missing={20}, insertion=5)
    make_sifts(root / "sifts.xml", missing={20})
    catalog = root / "catalog.json"
    write_json(
        catalog,
        dict(
            schema_version="site-reference-catalog-1",
            orthodb_release="v12.2",
            references=[reference()],
            structures=[
                dict(
                    accession="P00001",
                    source="pdb",
                    structure_id="test1",
                    method="X-ray diffraction",
                    resolution=2.0,
                    chain="A",
                    path="model.pdb",
                    format="pdb",
                    sha256=sha256(root / "model.pdb"),
                    sifts_path="sifts.xml",
                    sifts_sha256=sha256(root / "sifts.xml"),
                )
            ],
        ),
    )
    return run, catalog


def test_species_balance_and_copy_count_independence(tmp_path: Path) -> None:
    aa = tmp_path / "align"
    aa.mkdir()
    (aa / "OG1.fa").write_text(
        "".join(f">s1_gene{i}\nAAAA\n" for i in range(40)) + ">s2_gene\nVVVV\n>s3_gene\nVVVV\n"
    )
    policy = ReferencePolicy()
    with AlignmentStore(aa, ["s1", "s2", "s3"]) as store:
        og = store.ogs["OG1"]
        core = core_columns(og, ["s1", "s2", "s3"], policy)
        a = score_reference(reference("A", "AAAA"), "AAAA", og, ["s1", "s2", "s3"], core, policy)
        v = score_reference(reference("V", "VVVV"), "VVVV", og, ["s1", "s2", "s3"], core, policy)
    assert v["score"] == 1 and a["score"] == 0
    assert v["qualified"] and not a["qualified"]
    assert len(v["species_matches"]) == 3
    rows = [dict(v, accession="Z"), dict(v, accession="B"), dict(v, accession="A", reviewed=False)]
    assert [r["accession"] for r in sorted(rows, key=reference_order)] == ["B", "Z", "A"]


@pytest.mark.parametrize("value", [0, -1, 1.01, float("nan"), float("inf")])
def test_policy_rejects_invalid_thresholds(value: float) -> None:
    with pytest.raises(SiteAnnotationError):
        ReferencePolicy(minimum_reference_coverage=value)


def test_sifts_insertion_and_unobserved_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "sifts.xml"
    make_sifts(path, missing={20})
    rows = sifts_mapping(path, "P00001", "A")
    assert rows[4]["auth_seq_id"] == 105 and rows[4]["insertion_code"] == "A"
    assert rows[19]["observed"] is False
    assert sifts_mapping(path, "P00002", "A") == []
    pdb = tmp_path / "model.pdb"
    make_pdb(pdb, REFERENCE, missing={20}, insertion=5)
    atoms, _ = structure_atoms(pdb)
    assert ("A", 105, "A") in atoms and ("A", 120, "") not in atoms


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
def test_mapout_preserves_full_sequence_positions_and_backbone(tmp_path: Path) -> None:
    aa = tmp_path / "align"
    aa.mkdir()
    (aa / "OG1.fa").write_text(">s1_gene\n" + SEQUENCE + "\n")
    with AlignmentStore(aa, ["s1"]) as store:
        aligned, mapping, _ = project_references(
            store.ogs["OG1"], [reference()], tmp_path / "mafft", str(MAFFT)
        )
    assert aligned["P00001"] == SEQUENCE
    assert mapping["P00001"][3] == 5
    assert mapping["P00001"][18] == 20
    assert mapping["P00001"][20] == 22


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
def test_offline_postprocessing_preserves_models_and_records_missing_sites(tmp_path: Path) -> None:
    run, catalog = fixture_run(tmp_path)
    before = {str(p): sha256(p) for p in run.rglob("*") if p.is_file()}
    out = tmp_path / "sites"
    annotate_sites(
        run=run,
        catalog=catalog,
        out=out,
        mafft=str(MAFFT),
        render=False,
        cache=ResourceCache(tmp_path / "cache", offline=True),
    )
    after = {str(p): sha256(p) for p in run.rglob("*") if p.is_file()}
    assert before == after
    rows = pl.read_csv(out / "OG1/site_reference_map.tsv", separator="\t")
    assert rows["reference_pos_1based"].to_list() == [5, 20]
    mapped = pl.read_csv(
        out / "OG1/structure_0000/site_structure_map.tsv", separator="\t", null_values="NA"
    )
    assert mapped["structure_status"].to_list() == ["mapped", "unobserved"]
    assert mapped["auth_seq_id"].to_list() == [105, 120]
    assert mapped["insertion_code"][0] == "A"
    assert mapped["structure_residue"].to_list() == ["D", "V"]
    assert (out / "OG1/structure_0000/sites.pml").is_file()
    assert (
        json.loads((out / "manifest.json").read_text())["status"] == "completed_without_rendering"
    )
    with pytest.raises(SiteAnnotationError, match="already exists"):
        annotate_sites(
            run=run,
            catalog=catalog,
            out=out,
            mafft=str(MAFFT),
            render=False,
            cache=ResourceCache(tmp_path / "cache", offline=True),
        )


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
@pytest.mark.parametrize("change", ["release", "alignment", "reference", "structure"])
def test_annotation_rejects_changed_inputs(tmp_path: Path, change: str) -> None:
    run, catalog = fixture_run(tmp_path)
    data = json.loads(catalog.read_text())
    if change == "release":
        data["orthodb_release"] = "v13.0"
    elif change == "reference":
        data["references"][0]["sequence"] = "A" * 22
    elif change == "alignment":
        path = tmp_path / "align/OG1.fa"
        path.write_text(path.read_text().replace(SEQUENCE, "A" * 20))
    else:
        (tmp_path / "model.pdb").write_text("tampered\n")
    write_json(catalog, data)
    with pytest.raises(SiteAnnotationError):
        annotate_sites(
            run=run,
            catalog=catalog,
            out=tmp_path / "sites",
            mafft=str(MAFFT),
            render=False,
            cache=ResourceCache(tmp_path / "cache", offline=True),
        )


def test_commands_help_and_invalid_run(tmp_path: Path) -> None:
    runner = CliRunner()
    for command in ["prepare-site-references", "annotate-sites"]:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    result = runner.invoke(
        app,
        [
            "annotate-sites",
            "--run-dir",
            str(tmp_path),
            "--catalog",
            str(catalog),
            "--out",
            str(tmp_path / "out"),
            "--no-render",
            "--offline",
        ],
    )
    assert result.exit_code != 0 and "completed run" in result.output


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
def test_portable_replay_and_policy_binding(tmp_path: Path) -> None:
    run, catalog = fixture_run(tmp_path)
    first = tmp_path / "first"
    annotate_sites(
        run=run,
        catalog=catalog,
        out=first,
        mafft=str(MAFFT),
        render=False,
        cache=ResourceCache(tmp_path / "empty", offline=True),
    )
    portable = first / "reference_catalog.json"
    second = tmp_path / "second"
    annotate_sites(
        run=run,
        catalog=portable,
        out=second,
        mafft=str(MAFFT),
        render=False,
        cache=ResourceCache(tmp_path / "another_empty", offline=True),
    )
    for name in ["site_reference_map.tsv", "structure_0000/site_structure_map.tsv"]:
        assert (first / "OG1" / name).read_bytes() == (second / "OG1" / name).read_bytes()
    with pytest.raises(SiteAnnotationError, match="bound to different"):
        annotate_sites(
            run=run,
            catalog=portable,
            out=tmp_path / "changed",
            mafft=str(MAFFT),
            render=False,
            policy=ReferencePolicy(core_species_fraction=0.9),
            cache=ResourceCache(tmp_path / "empty", offline=True),
        )


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
def test_override_keeps_failed_automatic_qc_explicit(tmp_path: Path) -> None:
    run, catalog = fixture_run(tmp_path)
    data = json.loads(catalog.read_text())
    data["references"][0]["fragment"] = True
    write_json(catalog, data)
    auto = tmp_path / "automatic"
    annotate_sites(
        run=run,
        catalog=catalog,
        out=auto,
        mafft=str(MAFFT),
        render=False,
        cache=ResourceCache(tmp_path / "empty", offline=True),
    )
    assert json.loads((auto / "manifest.json").read_text())["orthogroups"][0]["status"] == (
        "reference_unavailable"
    )
    override = tmp_path / "override.tsv"
    override.write_text("orthogroup\taccession\nOG1\tP00001\n")
    manual = tmp_path / "manual"
    annotate_sites(
        run=run,
        catalog=catalog,
        out=manual,
        mafft=str(MAFFT),
        render=False,
        overrides=override,
        cache=ResourceCache(tmp_path / "empty", offline=True),
    )
    ref = json.loads((manual / "OG1/selected_reference.json").read_text())
    assert ref["selection_mode"] == "manual_override" and not ref["reference_qualified"]
    structure = json.loads((manual / "OG1/structure_0000/structure.json").read_text())
    assert structure["reference_selection"] == "manual_override"
    assert structure["reference_qualified"] is False


def test_structure_ranking_and_first_deposited_alternate_location(tmp_path: Path) -> None:
    from phenoradar.site_structure import structure_order

    rows = [
        dict(structure_id="a", chain="A", core_coverage=0.8, sequence_identity=1.0, resolution=1.0),
        dict(structure_id="b", chain="A", core_coverage=0.9, sequence_identity=0.9, resolution=2.0),
        dict(structure_id="c", chain="A", core_coverage=0.9, sequence_identity=0.9, resolution=1.5),
    ]
    assert [r["structure_id"] for r in sorted(rows, key=structure_order)] == ["c", "b", "a"]


@pytest.mark.skipif(
    MAFFT is None or not os.environ.get("PHENORADAR_TEST_PYMOL"),
    reason="optional MAFFT/PyMOL rendering environment unavailable",
)
@pytest.mark.parametrize("source", ["pdb", "alphafold"])
def test_real_pymol_six_views_and_pdf(tmp_path: Path, source: str) -> None:
    run, catalog = fixture_run(tmp_path)
    if source == "alphafold":
        make_pdb(tmp_path / "model.pdb", REFERENCE)
        data = json.loads(catalog.read_text())
        data["structures"][0].update(
            source=source,
            method="AlphaFold",
            sha256=sha256(tmp_path / "model.pdb"),
            reference_start=1,
            reference_end=len(REFERENCE),
        )
        write_json(catalog, data)
    out = tmp_path / "rendered"
    annotate_sites(
        run=run,
        catalog=catalog,
        out=out,
        mafft=str(MAFFT),
        pymol=str(os.environ["PHENORADAR_TEST_PYMOL"]),
        cache=ResourceCache(tmp_path / "empty", offline=True),
    )
    folder = out / "OG1/structure_0000"
    render = json.loads((folder / "rendering.json").read_text())
    assert len(render["views"]) == 6
    translations = []
    for view in render["views"].values():
        rotation = np.array(view[:9]).reshape(3, 3)
        assert np.linalg.det(rotation) == pytest.approx(1, abs=1e-5)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
        translations.append(view[9:])
    assert all(t == translations[0] for t in translations)
    assert len(list(folder.glob("*.png"))) == (30 if source == "alphafold" else 24)
    assert ("confidence" in render["pages"]) == (source == "alphafold")
    assert (folder / "sites.pse").stat().st_size > 100
    assert (folder / "sites.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
def test_alphafold_sequence_positions_and_confidence_are_distinct(tmp_path: Path) -> None:
    run, catalog = fixture_run(tmp_path)
    make_pdb(tmp_path / "af.pdb", REFERENCE)
    data = json.loads(catalog.read_text())
    data["structures"].append(
        dict(
            accession="P00001",
            source="alphafold",
            structure_id="AF-P00001-F1",
            method="AlphaFold",
            resolution=None,
            chain="A",
            path="af.pdb",
            format="pdb",
            sha256=sha256(tmp_path / "af.pdb"),
            reference_start=1,
            reference_end=len(REFERENCE),
        )
    )
    write_json(catalog, data)
    out = tmp_path / "af_annotations"
    annotate_sites(
        run=run,
        catalog=catalog,
        out=out,
        mafft=str(MAFFT),
        render=False,
        cache=ResourceCache(tmp_path / "cache", offline=True),
    )
    mapped = pl.read_csv(out / "OG1/structure_0001/site_structure_map.tsv", separator="\t")
    assert mapped["reference_pos_1based"].to_list() == [5, 20]
    assert mapped["auth_seq_id"].to_list() == [105, 120]
    assert mapped["plddt"].to_list() == [85.0, 85.0]
    assert mapped["pss"].to_list() == [2.0, 1.0]
    assert mapped["analysis_scope"].unique().to_list() == ["final"]
    # A shifted fragment interval cannot silently inherit the same residue numbers.
    data["structures"][1]["reference_start"] = 2
    write_json(catalog, data)
    with pytest.raises(SiteAnnotationError, match="disagree"):
        annotate_sites(
            run=run,
            catalog=catalog,
            out=tmp_path / "bad_af",
            mafft=str(MAFFT),
            render=False,
            cache=ResourceCache(tmp_path / "cache", offline=True),
        )


def test_first_deposited_alternate_ca_is_retained(tmp_path: Path) -> None:
    pdb = tmp_path / "alt.pdb"
    make_pdb(pdb, "A")
    first = pdb.read_text().splitlines()[0]
    second = first[:16] + "A" + first[17:54] + "  0.90" + first[60:]
    first = first[:16] + "B" + first[17:54] + "  0.10" + first[60:]
    first = first[:60] + " 10.00" + first[66:]
    second = second[:60] + " 90.00" + second[66:]
    pdb.write_text(first + "\n" + second + "\nEND\n")
    atoms, _ = structure_atoms(pdb)
    assert len(atoms) == 1
    assert atoms[("A", 101, "")]["b"] == 10.0


def test_first_deposited_model_is_used_even_when_numbers_are_out_of_order(tmp_path: Path) -> None:
    pdb = tmp_path / "models.pdb"
    make_pdb(pdb, "A")
    line = pdb.read_text().splitlines()[0]
    pdb.write_text(
        "MODEL        9\n"
        + line
        + "\nENDMDL\nMODEL        1\n"
        + line[:60]
        + " 10.00"
        + line[66:]
        + "\nENDMDL\n"
    )
    atoms, model = structure_atoms(pdb)
    assert model == 9 and atoms[("A", 101, "")]["b"] == 85.0
    pdb.write_text("MODEL        9\nENDMDL\nMODEL        1\n" + line + "\nENDMDL\n")
    with pytest.raises(SiteAnnotationError, match="no protein CA"):
        structure_atoms(pdb)
    cif = tmp_path / "models.cif"
    fields = [
        "id",
        "pdbx_PDB_model_num",
        "label_atom_id",
        "label_seq_id",
        "auth_asym_id",
        "auth_seq_id",
        "label_comp_id",
        "B_iso_or_equiv",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
    ]
    cif.write_text(
        "data_test\nloop_\n"
        + "".join("_atom_site." + k + "\n" for k in fields)
        + "1 9 CA 1 A 101 ALA 85 0 0 0\n2 1 CA 1 A 101 ALA 10 1 1 1\n#\n"
    )
    atoms, model = structure_atoms(cif)
    assert model == 9 and atoms[("A", 101, "")]["b"] == 85.0


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
def test_stop_and_unknown_symbols_are_preserved_without_changing_columns(tmp_path: Path) -> None:
    aa = tmp_path / "align"
    aa.mkdir()
    (aa / "OG1.fa").write_text(">s1_gene\n" + SEQUENCE + "*X\n>s2_gene\n" + SEQUENCE + "-X\n")
    with AlignmentStore(aa, ["s1", "s2"]) as store:
        projected, positions, _ = project_references(
            store.ogs["OG1"], [reference(sequence=REFERENCE)], tmp_path / "project", str(MAFFT)
        )
    assert len(projected["P00001"]) == 22
    assert positions["P00001"][3] == 5
    assert "*X" in (tmp_path / "project/reference_alignment.fa").read_text()
