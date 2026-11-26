from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import polars as pl
import pytest
import yaml
from test_site_reference import MAFFT, REFERENCE, THREE, fixture_run, reference

from phenoradar import site_annotation_run
from phenoradar.config.schema import StructureReportConfig
from phenoradar.site_pdb_search import prepare_pdb_catalog
from phenoradar.site_reference import annotate_sites
from phenoradar.site_reference_resources import (
    ResourceCache,
    SiteAnnotationError,
    sha256,
    write_json,
)
from phenoradar.site_structure import rank_pdb_site_candidates


class SearchCache(ResourceCache):
    def fetch(self, url: str, *, missing_ok: bool = False) -> Path | None:
        path = self.directory / "response"
        if "rcsbsearch" in url:
            search = json.loads(parse_qs(urlsplit(url).query)["json"][0])
            assert search["request_options"]["results_content_type"] == ["experimental"]
            assert search["query"]["parameters"]["value"] == REFERENCE
            assert search["request_options"]["results_verbosity"] == "verbose"
            match = dict(sequence_identity=0.9, evalue=0.01, bitscore=42, subject_beg=1,
                         subject_end=len(REFERENCE), subject_length=len(REFERENCE),
                         query_beg=1, query_end=len(REFERENCE))
            services = [dict(service_type="sequence", nodes=[dict(match_context=[match])])]
            write_json(path, dict(total_count=2, result_set=[
                dict(identifier="1ABC_1", score=0.75, services=services),
                dict(identifier="2ABC_1", score=1.0, services=services),
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


class PdbOnlyCache(SearchCache):
    def fetch(self, url: str, *, missing_ok: bool = False) -> Path | None:
        assert "orthodb" not in url
        return super().fetch(url, missing_ok=missing_ok)

    def json(self, url: str, *, missing_ok: bool = False) -> Any:
        assert "orthodb" not in url
        result = super().json(url, missing_ok=missing_ok)
        if url.endswith("/P00002.json"):
            result["sequence"]["value"] = "W" + REFERENCE[1:]
        return result


def test_pdb_catalog_admits_homologs_without_orthodb(tmp_path: Path) -> None:
    queries = tmp_path / "queries.fa"
    queries.write_text(f">OG1 full_length_gene\n{REFERENCE}*\n")
    catalog = tmp_path / "catalog.json"
    prepare_pdb_catalog(
        orthogroups=["OG1"], queries=queries, out=catalog,
        cache=PdbOnlyCache(tmp_path / "cache"),
    )
    data = json.loads(catalog.read_text())
    assert data["orthodb_release"] is None
    assert [r["accession"] for r in data["references"]] == ["P00001", "P00002"]
    assert data["references"][1]["sequence"] != REFERENCE
    assert all(r["orthodb_gene_ids"] == [] and not r["membership_verified"]
               for r in data["references"])
    candidates = data["pdb_search"]["candidates"]["OG1"]
    assert {(r["accession"], r["chain"]) for r in candidates} == {("P00001", "A"), ("P00002", "B")}
    assert candidates[0]["search_subject_end"] == len(REFERENCE)
    assert data["pdb_search"]["queries"]["OG1"]["header"] == "OG1 full_length_gene"
    assert data["exclusions"] == []


def test_short_alignment_query_prepares_without_orthodb(tmp_path: Path) -> None:
    queries = tmp_path / "queries.fa"
    queries.write_text(f">OG1 g1\n{REFERENCE[:15]}\n")
    catalog = tmp_path / "catalog.json"
    prepare_pdb_catalog(
        orthogroups=["OG1"], release="v12.2", queries=queries,
        out=catalog, cache=PdbOnlyCache(tmp_path / "cache"), alignment_sequence_ids={"OG1": "g1"},
    )
    data = json.loads(catalog.read_text())
    assert not data["references"]
    assert data["pdb_search"]["queries"]["OG1"]["pdb_search_skipped"] == (
        "query_shorter_than_25_residues"
    )
    assert data["pdb_search"]["candidates"] == {"OG1": []}


@pytest.mark.parametrize("fails", [False, True])
def test_automatic_report_uses_top_og_and_training_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool,
) -> None:
    run, _ = fixture_run(tmp_path)
    aln = tmp_path / "align/OG1.fa"
    aln.write_text(
        ">s2_gene\nCD..\n>s0_gene\nA-CC\n>s1_gene\nC.A-\n>s3_heldout\nFF-F\n"
    )
    inventory = pl.DataFrame(
        {"orthogroup": ["OG1"], "sha256": [sha256(aln)], "alignment_length": [4]},
    )
    inventory.write_csv(run / "model/tables/alignment_inputs.tsv", separator="\t")
    score_path = run / "model/tables/amino_acid_site_pss.tsv"
    scores = pl.read_csv(score_path, separator="\t")
    lower_scores = scores.with_columns(pl.lit("OG2").alias("orthogroup"), pl.lit(0.3).alias("pss"))
    pl.concat([scores, lower_scores]).write_csv(score_path, separator="\t")
    split_path = run / "split/tables/split_manifest.tsv"
    pl.read_csv(split_path, separator="\t").with_columns(
        pl.when(pl.col("species") == "s3").then(pl.lit("external_test"))
        .otherwise(pl.col("pool")).alias("pool")
    ).write_csv(split_path, separator="\t")
    monkeypatch.setattr(site_annotation_run, "_check_annotation_tools", lambda *args: None)
    calls = []

    def prepare(**kwargs: Any) -> None:
        assert kwargs["orthogroups"] == ["OG1"]
        assert kwargs["queries"].read_text() == ">OG1 s1_gene\nCA\n"
        assert kwargs["alignment_sequence_ids"] == {"OG1": "s1_gene"}
        assert kwargs["release"] == "v12.2"
        if fails:
            raise SiteAnnotationError("External resource unavailable")
        calls.append("prepared")

    def annotate(**kwargs: Any) -> None:
        assert kwargs["orthogroups"] == ["OG1"]
        assert kwargs["scope"] == "final"
        assert kwargs["render"] is True
        calls.append("annotated")

    monkeypatch.setattr(site_annotation_run, "prepare_pdb_catalog", prepare)
    monkeypatch.setattr(site_annotation_run, "annotate_sites", annotate)
    options = StructureReportConfig(top_orthogroups=1, render=True, cache_dir=str(tmp_path))
    site_annotation_run.run_site_annotation_report(run, options, "final")
    status = json.loads((run / "site_annotation_inputs/status.json").read_text())
    assert status["status"] == ("failed" if fails else "completed")
    assert status["orthogroups"] == ["OG1"]
    assert json.loads((run / "run_metadata.json").read_text())["status"] == "full_run_completed"
    assert calls == ([] if fails else ["prepared", "annotated"])


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


def test_inconsistent_sifts_candidate_does_not_discard_other_chains(tmp_path: Path) -> None:
    cache = StructureCache(tmp_path)
    bad = cache.files["1abc"]
    bad.write_text(re.sub(r"^(5 CA) \w+", r"\1 TRP", bad.read_text(), flags=re.MULTILINE))
    candidates = [
        dict(accession=acc, source="pdb", structure_id=pdb, chain="A", search_score=score)
        for acc, pdb, score in [("P00001", "1abc", 1.0), ("P00002", "2abc", 0.5)]
    ]
    sites = [
        dict(alignment_pos_1based=18, reference_pos_1based=20,
             reference_residue=REFERENCE[19], reference_status="mapped"),
    ]
    winner, audit = rank_pdb_site_candidates(
        candidates, {c["accession"]: reference(c["accession"]) for c in candidates},
        {c["accession"]: sites for c in candidates}, {18}, cache,
    )
    assert winner is not None and winner["structure_id"] == "2abc"
    assert audit[0]["exclusion_reason"] == "inconsistent_sifts_mapping"
    assert "differs from structure atoms" in audit[0]["exclusion_detail"]
    assert audit[1]["selected"]


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


@pytest.mark.skipif(MAFFT is None, reason="optional MAFFT executable unavailable")
@pytest.mark.parametrize("scenario", ["eligible", "partial", "fallback", "no_hits"])
def test_homolog_pdb_annotation_and_lazy_fallback(tmp_path: Path, scenario: str) -> None:
    run, _ = fixture_run(tmp_path)
    requires_fallback = scenario in {"fallback", "no_hits"}
    if not requires_fallback:
        config_path = run / "resolved_config.yml"
        config = yaml.safe_load(config_path.read_text())
        config["data"].pop("orthogroup_annotation_path", None)
        config_path.write_text(yaml.safe_dump(config))
    queries = tmp_path / "query.fa"
    queries.write_text(f">OG1\n{REFERENCE[:12] if scenario == 'no_hits' else REFERENCE}\n")
    catalog = tmp_path / "pdb_catalog.json"
    prepare_pdb_catalog(
        orthogroups=["OG1"], queries=queries, out=catalog,
        cache=PdbOnlyCache(tmp_path / "preparation"),
        alignment_sequence_ids={"OG1": "g1"} if scenario == "no_hits" else None,
    )
    data = json.loads(catalog.read_text())
    if scenario != "no_hits":
        only_chain = next(r for r in data["pdb_search"]["candidates"]["OG1"]
                          if r["structure_id"] == "1abc")
        if scenario != "eligible":
            only_chain["search_subject_start"] = 6
        data["pdb_search"]["candidates"]["OG1"] = [only_chain]
    write_json(catalog, data)
    sites = tmp_path / "sites.tsv"
    required = "false" if scenario == "partial" else "true"
    sites.write_text(f"orthogroup\talignment_pos_1based\trequired\n"
                     f"OG1\t3\t{required}\nOG1\t18\tfalse\n")

    class MappingCache(StructureCache):
        def fetch(self, url: str, *, missing_ok: bool = False) -> Path | None:
            if "orthodb" in url:
                assert requires_fallback
                return lookup.fetch(url, missing_ok=missing_ok)
            return super().fetch(url, missing_ok=missing_ok)

        def json(self, url: str, *, missing_ok: bool = False) -> Any:
            if "orthodb" in url or "uniprot" in url:
                assert requires_fallback
                if "/ogdetails" in url:
                    return dict(status="ok", data=dict(
                        gene_id=dict(id="g1"), uniprot=[dict(id="P00002")],
                    ))
                return lookup.json(url, missing_ok=missing_ok)
            return super().json(url, missing_ok=missing_ok)

    lookup = SearchCache(tmp_path / "fallback_lookup")
    out = tmp_path / "annotated"
    annotate_sites(
        run=run, catalog=catalog, out=out, cache=MappingCache(tmp_path),
        mafft=str(MAFFT), render=False, sites_path=sites,
    )
    path = "OG1/structure_0000/site_structure_map.tsv"
    mapped = pl.read_csv(out / path, separator="\t")
    assert mapped["structure_id"].unique().to_list() == [
        "AF-P00002-F1" if requires_fallback else "1abc",
    ]
    assert mapped["orthodb_membership_verified"].unique().to_list() == [requires_fallback]
    assert mapped["structure_status"].to_list() == [
        "outside_search_alignment" if scenario == "partial" else "mapped", "mapped",
    ]
    replay = tmp_path / "replay"
    annotate_sites(
        run=run, catalog=out / "reference_catalog.json", out=replay,
        cache=ResourceCache(tmp_path / "offline", offline=True), mafft=str(MAFFT),
        render=False, sites_path=sites,
    )
    assert (replay / path).read_bytes() == (out / path).read_bytes()
