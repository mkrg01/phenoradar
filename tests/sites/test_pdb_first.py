from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import polars as pl
import pytest
from test_site_reference import MAFFT, REFERENCE, THREE, fixture_run, reference

from phenoradar.site_pdb_search import prepare_pdb_catalog
from phenoradar.site_reference import annotate_sites
from phenoradar.site_reference_resources import ResourceCache, SiteAnnotationError, write_json


class SearchCache(ResourceCache):
    def fetch(self, url: str, *, missing_ok: bool = False) -> Path | None:
        path = self.directory / "response"
        if "rcsbsearch" in url:
            search = json.loads(parse_qs(urlsplit(url).query)["json"][0])
            assert search["request_options"]["results_content_type"] == ["experimental"]
            assert search["query"]["parameters"]["value"] == REFERENCE
            write_json(path, dict(total_count=2, result_set=[
                dict(identifier="1ABC_1", score=0.75), dict(identifier="2ABC_1", score=1.0),
            ]))
        else:
            assert "/fasta?id=OG1" in url
            path.write_text(f">g1 member\n{REFERENCE}\n")
        return path

    def json(self, url: str, *, missing_ok: bool = False) -> Any:
        if url.endswith("/orthodb_release_id"):
            return "v12.2"
        if "/ogdetails" in url:
            return dict(status="ok", data=dict(
                gene_id=dict(id="g1"), uniprot=[dict(id="P00001")],
            ))
        if "/graphql" in url:
            return dict(data=dict(polymer_entities=[
                dict(
                    rcsb_id=entity,
                    rcsb_polymer_entity_container_identifiers=dict(
                        auth_asym_ids=[chain],
                        reference_sequence_identifiers=[
                            dict(database_name="UniProt", database_accession=accession),
                        ],
                    ),
                    entry=dict(exptl=[dict(method="X-RAY DIFFRACTION")]),
                )
                for entity, chain, accession in [
                    ("1ABC_1", "A", "P00001"), ("2ABC_1", "B", "P00002"),
                ]
            ]))
        accession = url.split("/")[-1].removesuffix(".json")
        return dict(
            primaryAccession=accession,
            sequence=dict(value=REFERENCE),
            entryType="UniProtKB reviewed (Swiss-Prot)",
            entryAudit=dict(sequenceVersion=2),
            organism=dict(scientificName="Example species"),
        )


def test_pdb_catalog_requires_same_og_cross_reference_even_for_identical_proteins(
    tmp_path: Path,
) -> None:
    queries = tmp_path / "queries.fa"
    queries.write_text(f">OG1 full_length_gene\n{REFERENCE}*\n")
    catalog = tmp_path / "catalog.json"
    prepare_pdb_catalog(
        orthogroups=["OG1"], release="v12.2", queries=queries,
        out=catalog, cache=SearchCache(tmp_path / "cache"),
    )
    data = json.loads(catalog.read_text())
    assert [ref["accession"] for ref in data["references"]] == ["P00001"]
    assert data["references"][0]["orthodb_gene_ids"] == ["g1"]
    assert data["pdb_search"]["candidates"]["OG1"][0]["chain"] == "A"
    assert data["pdb_search"]["queries"]["OG1"]["header"] == "OG1 full_length_gene"
    assert data["exclusions"] == [dict(
        accession="P00002", orthogroup="OG1", reason="no_exact_linked_orthodb_member",
    )]


def write_cif(path: Path, accession: str, observed: set[int]) -> None:
    # The author and label chain IDs differ. SIFTS duplicates also occur on domain annotations.
    atom_fields = [
        "id", "label_atom_id", "label_comp_id", "label_asym_id", "auth_asym_id",
        "label_seq_id", "auth_seq_id", "pdbx_PDB_ins_code", "B_iso_or_equiv",
        "Cartn_x", "Cartn_y", "Cartn_z",
    ]
    lines = ["data_example", "loop_", *(f"_atom_site.{name}" for name in atom_fields)]
    for pos in sorted(observed):
        lines.append(f"{pos} CA {THREE[REFERENCE[pos - 1]]} L A {pos} {100 + pos} ? 90 0 0 0")
    lines.extend([
        "#", "loop_", "_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.pdb_strand_id",
        "L A", "#", "loop_",
        *(f"_pdbx_sifts_xref_db.{field}" for field in [
            "unp_acc", "asym_id", "seq_id", "unp_num", "mon_id", "observed", "unp_res",
        ]),
    ])
    for pos in (5, 20, 20):
        # SIFTS can mark a residue as observed even when the deposited model lacks its CA.
        residue = REFERENCE[pos - 1]
        lines.append(f"{accession} L {pos} {pos} {THREE[residue]} y {residue}")
    path.write_text("\n".join(lines) + "\n")


class StructureCache(ResourceCache):
    def __init__(self, root: Path) -> None:
        super().__init__(root / "cache")
        self.files = {
            "1abc": root / "1abc.cif",
            "2abc": root / "2abc.cif",
            "alphafold": root / "alphafold.cif",
        }
        write_cif(self.files["1abc"], "P00001", {5, 20})
        write_cif(self.files["2abc"], "P00002", {20})
        write_cif(self.files["alphafold"], "P00002", {5, 20})

    def fetch(self, url: str, *, missing_ok: bool = False) -> Path | None:
        return self.files[url.split("/")[-1].split("_")[0].removesuffix(".cif")]

    def json(self, url: str, *, missing_ok: bool = False) -> Any:
        assert "api/prediction/" in url
        if not url.endswith("P00002"):
            return None
        return [dict(
            entryId="AF-P00002-F1", uniprotAccession="P00002", sequence=REFERENCE,
            uniprotStart=1, uniprotEnd=len(REFERENCE), chainId="A", latestVersion=6,
            cifUrl="https://alphafold.ebi.ac.uk/files/alphafold.cif",
        )]


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
@pytest.mark.parametrize("has_experimental", [True, False])
def test_pdb_first_uses_actual_coordinates_and_replays_offline(
    tmp_path: Path, has_experimental: bool,
) -> None:
    run, catalog = fixture_run(tmp_path)
    data = json.loads(catalog.read_text())
    data["orthogroups"] = ["OG1"]
    data["structures"] = []
    data["references"].append(reference("P00002"))
    pdbs = [("1abc", "P00001", 0.1), ("2abc", "P00002", 1.0)]
    data["pdb_search"] = dict(candidates={"OG1": [
        dict(source="pdb", structure_id=pdb, accession=acc, chain="A",
             search_score=score, method="X-ray diffraction", resolution=2)
        for pdb, acc, score in (pdbs if has_experimental else pdbs[1:])
    ]})
    write_json(catalog, data)
    site_tsv = tmp_path / "sites.tsv"
    site_tsv.write_text(
        "orthogroup\talignment_pos_1based\trequired\nOG1\t3\ttrue\nOG1\t18\tfalse\n"
    )
    out = tmp_path / "out"
    annotate_sites(
        run=run, catalog=catalog, out=out, cache=StructureCache(tmp_path),
        mafft=str(MAFFT), render=False, sites_path=site_tsv,
    )
    table_path = "OG1/structure_0000/site_structure_map.tsv"
    table = pl.read_csv(out / table_path, separator="\t")
    assert table["structure_id"].unique().to_list() == [
        "1abc" if has_experimental else "AF-P00002-F1",
    ]
    assert table["auth_seq_id"].to_list() == [105, 120]
    audit = pl.read_csv(out / "OG1/pdb_site_candidates.tsv", separator="\t")
    missing = audit.filter(pl.col("structure_id") == "2abc").row(0, named=True)
    assert missing["exclusion_reason"] == "required_site_unobserved"
    assert missing["n_observed_sites"] == 1
    replay = tmp_path / "replay"
    replay_cache = ResourceCache(tmp_path / "empty-cache", offline=True)
    annotate_sites(
        run=run, catalog=out / "reference_catalog.json", out=replay, cache=replay_cache,
        mafft=str(MAFFT), render=False, sites_path=site_tsv,
    )
    assert (replay / table_path).read_bytes() == (out / table_path).read_bytes()
    with pytest.raises(SiteAnnotationError, match="different run inputs or policy"):
        annotate_sites(
            run=run, catalog=out / "reference_catalog.json", out=tmp_path / "other",
            cache=replay_cache, mafft=str(MAFFT), render=False,
        )
