"""Discover experimental PDB chains using one protein search query per OG."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from phenoradar.site_reference_resources import (
    ResourceCache,
    SiteAnnotationError,
    _odb_data,
    orthodb_api,
    read_fasta,
    sha256,
    uniprot_reference,
    write_json,
)

_RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
_RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
_UNIPROT = "https://rest.uniprot.org/uniprotkb/"
_ENTITY_BATCH = 40
_SEARCH_HITS = 100
_RCSB_MINIMUM_QUERY_LENGTH = 25
PDB_SEARCH_POLICY = "pdb-homolog-site-coverage-1"


def _pdb_hits(sequence: str, cache: ResourceCache) -> tuple[dict[str, dict[str, Any]], int]:
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "sequence_type": "protein",
                "value": sequence,
                "evalue_cutoff": 1,
                "identity_cutoff": 0.25,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "results_content_type": ["experimental"],
            "results_verbosity": "verbose",
            "scoring_strategy": "sequence",
            "paginate": {"start": 0, "rows": _SEARCH_HITS},
        },
    }
    response = cache.fetch(_RCSB_SEARCH + "?" + urlencode({"json": json.dumps(query)}))
    assert response is not None
    raw = response.read_bytes()
    if not raw:  # RCSB returns an empty HTTP 204 when the search has no hits.
        return {}, 0
    result = json.loads(raw)
    hits = {}
    for row in result.get("result_set", []):
        matches = [
            match
            for service in row["services"] if service["service_type"] == "sequence"
            for node in service["nodes"]
            for match in node.get("match_context", [])
        ]
        if not matches:
            raise SiteAnnotationError("RCSB sequence-search hit lacks alignment coordinates.")
        match = max(matches, key=lambda r: r["bitscore"])
        if not (
            1 <= match["query_beg"] <= match["query_end"] <= len(sequence)
            and 1 <= match["subject_beg"] <= match["subject_end"] <= match["subject_length"]
        ):
            raise SiteAnnotationError("Invalid RCSB sequence alignment interval.")
        hits[row["identifier"]] = dict(
            search_score=float(row["score"]),
            search_identity=float(match["sequence_identity"]),
            search_evalue=float(match["evalue"]),
            search_query_start=match["query_beg"],
            search_query_end=match["query_end"],
            search_subject_start=match["subject_beg"],
            search_subject_end=match["subject_end"],
        )
    total: int = result["total_count"]
    if len(hits) != min(total, _SEARCH_HITS):
        raise SiteAnnotationError("Incomplete or duplicated RCSB sequence-search hits.")
    return hits, total


def _pdb_chains(hits: dict[str, dict[str, Any]], cache: ResourceCache) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    identifiers = sorted(hits)
    for start in range(0, len(identifiers), _ENTITY_BATCH):
        batch = identifiers[start : start + _ENTITY_BATCH]
        query = (
            "query { polymer_entities(entity_ids:"
            + json.dumps(batch)
            + ") { rcsb_id rcsb_polymer_entity_container_identifiers { "
            "auth_asym_ids reference_sequence_identifiers { database_name database_accession } } "
            "entry { exptl { method } rcsb_entry_info { resolution_combined } } } }"
        )
        response = cache.json(_RCSB_GRAPHQL + "?" + urlencode({"query": query}))
        if response.get("errors"):
            raise SiteAnnotationError(f"RCSB polymer-entity lookup failed: {response['errors']}")
        entities = response["data"]["polymer_entities"]
        if {entity["rcsb_id"] for entity in entities if entity} != set(batch):
            raise SiteAnnotationError("Incomplete RCSB polymer-entity metadata.")
        for entity in entities:
            if entity is None:
                continue
            entity_id = entity["rcsb_id"]
            ids = entity["rcsb_polymer_entity_container_identifiers"]
            entry = entity["entry"]
            methods = sorted({x["method"] for x in entry.get("exptl") or []})
            resolutions = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            for link in ids.get("reference_sequence_identifiers") or []:
                if link["database_name"] == "UniProt":
                    chains.extend(
                        dict(
                            accession=link["database_accession"],
                            source="pdb",
                            structure_id=entity_id.rsplit("_", 1)[0].lower(),
                            entity_id=entity_id,
                            chain=chain,
                            method=" / ".join(methods),
                            resolution=min(resolutions, default=None),
                            **hits[entity_id],
                        )
                        for chain in ids["auth_asym_ids"]
                    )
    return chains


def _orthodb_gene_links(gene: str, base: str, cache: ResourceCache) -> set[str]:
    url = base + "/ogdetails?" + urlencode({"id": gene})
    info = _odb_data(cache.json(url), gene)
    if info["gene_id"]["id"] != gene:
        raise SiteAnnotationError(f"OrthoDB gene identity mismatch: {gene}")
    return {link["id"] for link in info.get("uniprot", [])}


def _uniprot_entry(accession: str, cache: ResourceCache) -> dict[str, Any] | None:
    entry = cache.json(_UNIPROT + accession + ".json", missing_ok=True)
    if entry is None:
        return None
    if not isinstance(entry, dict):
        raise SiteAnnotationError(f"Invalid UniProt response: {accession}")
    if entry.get("entryType") == "Inactive":
        return None
    if not entry.get("sequence", {}).get("value"):
        raise SiteAnnotationError(f"UniProt response lacks sequence: {accession}")
    return entry


def _closest_member_sequences(query: str, members: dict[str, list[str]]) -> list[str]:
    """Order full-length proteins by BLOSUM62 alignment score for AlphaFold fallback."""
    bio_align: Any = import_module("Bio.Align")
    substitution_matrices: Any = import_module("Bio.Align.substitution_matrices")
    matrix = substitution_matrices.load("BLOSUM62")
    allowed = set(matrix.alphabet)
    aligner = bio_align.PairwiseAligner(
        substitution_matrix=matrix, open_gap_score=-11, extend_gap_score=-1,
    )

    def normalize(sequence: str) -> str:
        return "".join(c if c in allowed else "X" for c in sequence)

    fixed = normalize(query)
    return sorted(members, key=lambda seq: (-aligner.score(fixed, normalize(seq)), members[seq][0]))


def orthodb_fallback_references(
    og: str,
    release: str,
    query_info: dict[str, Any],
    pdb_references: list[dict[str, Any]],
    cache: ResourceCache,
) -> list[dict[str, Any]]:
    """Find exact, linked OrthoDB references for AlphaFold after PDB selection fails."""
    query = query_info["sequence"]
    if hashlib.sha256(query.encode()).hexdigest() != query_info["sha256"]:
        raise SiteAnnotationError(f"Search query checksum mismatch for {og}.")
    _, base = orthodb_api(release, cache)
    members = cache.fetch(base + "/fasta?" + urlencode({"id": og}))
    assert members is not None
    sequences_by_gene = read_fasta(members.read_text())
    member_sequences: dict[str, list[str]] = defaultdict(list)
    for gene, sequence in sequences_by_gene.items():
        member_sequences[sequence].append(gene)
    gene_links: dict[str, set[str]] = {}

    def links(gene: str) -> set[str]:
        if gene not in gene_links:
            gene_links[gene] = _orthodb_gene_links(gene, base, cache)
        return gene_links[gene]

    refs: dict[str, dict[str, Any]] = {}
    checked: set[str] = set()

    def check(accession: str) -> None:
        if accession in checked:
            return
        checked.add(accession)
        entry = _uniprot_entry(accession, cache)
        if entry is None:
            return
        sequence = entry["sequence"]["value"].upper()
        aliases = {accession, entry["primaryAccession"], *entry.get("secondaryAccessions", [])}
        genes = sorted(gene for gene in member_sequences.get(sequence, []) if links(gene) & aliases)
        if genes:
            ref = uniprot_reference(og, accession, entry, genes, sequence)
            if not ref["fragment"]:
                refs.setdefault(ref["accession"], ref)

    query_genes = set(member_sequences.get(query, []))
    if (sequence_id := query_info.get("alignment_sequence_id")) in sequences_by_gene:
        query_genes.add(sequence_id)
    accessions = {link for gene in sorted(query_genes) for link in links(gene)}
    accessions.update(
        r["requested_accession"] for r in pdb_references if r["sequence"] in member_sequences
    )
    for accession in sorted(accessions):
        check(accession)
    if not refs:
        for sequence in _closest_member_sequences(query, member_sequences):
            for gene in member_sequences[sequence]:
                for accession in sorted(links(gene)):
                    check(accession)
            if refs:
                break
    return list(refs.values())


def _prepare_og(
    og: str, query: str, cache: ResourceCache, *, alignment_sequence_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    search_pdb = alignment_sequence_id is None or len(query) >= _RCSB_MINIMUM_QUERY_LENGTH
    hits, total_hits = _pdb_hits(query, cache) if search_pdb else ({}, 0)
    chains = _pdb_chains(hits, cache)
    query_info: dict[str, Any] = dict(
        sequence=query, sha256=hashlib.sha256(query.encode()).hexdigest(),
        pdb_total_hits=total_hits, pdb_screened_entities=len(hits),
    )
    if alignment_sequence_id is not None:
        query_info.update(alignment_sequence_id=alignment_sequence_id, length=len(query))
    if not search_pdb:
        query_info["pdb_search_skipped"] = "query_shorter_than_25_residues"

    exclusions = []
    aliases = {}
    refs: dict[str, dict[str, Any]] = {}
    for accession in sorted({chain["accession"] for chain in chains}):
        entry = _uniprot_entry(accession, cache)
        if entry is None:
            exclusions.append(dict(
                orthogroup=og, accession=accession, reason="uniprot_unavailable",
            ))
            continue
        ref = uniprot_reference(og, accession, entry, [], entry["sequence"]["value"].upper())
        ref["membership_verified"] = False
        existing = refs.setdefault(ref["accession"], ref)
        if existing["sequence_sha256"] != ref["sequence_sha256"]:
            raise SiteAnnotationError("Conflicting UniProt sequences for one accession.")
        aliases[accession] = ref["accession"]

    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for chain in chains:
        accession = aliases.get(chain["accession"])
        if accession is None:
            continue
        key = accession, chain["structure_id"], chain["chain"]
        previous = candidates.get(key)
        if previous is None or chain["search_score"] > previous["search_score"]:
            candidates[key] = {**chain, "accession": accession}
    return list(refs.values()), list(candidates.values()), exclusions, query_info


def prepare_pdb_catalog(
    *,
    orthogroups: list[str],
    release: str | None = None,
    queries: Path,
    out: Path,
    cache: ResourceCache,
    alignment_sequence_ids: dict[str, str] | None = None,
) -> Path:
    """Freeze PDB sequence-search candidates for site mapping."""
    if out.exists():
        raise SiteAnnotationError(f"Output already exists: {out}")
    fasta = queries.read_text()
    sequences = {key: seq.removesuffix("*") for key, seq in read_fasta(fasta).items()}
    headers = {line[1:].split()[0]: line[1:] for line in fasta.splitlines()
               if line.startswith(">")}
    if set(sequences) != set(orthogroups):
        raise SiteAnnotationError("Supply one search FASTA record per requested OG, with OG IDs.")
    for og, sequence in sequences.items():
        if not re.fullmatch(r"[A-Z]+", sequence):
            raise SiteAnnotationError(f"Expected an ungapped protein query for {og}.")
    references: list[dict[str, Any]] = []
    pdb_candidates: dict[str, list[dict[str, Any]]] = {}
    queries_info: dict[str, dict[str, Any]] = {}
    exclusions = []
    for og, query in sorted(sequences.items()):
        refs, chains, rejected, query_info = _prepare_og(
            og, query, cache,
            alignment_sequence_id=(alignment_sequence_ids or {}).get(og),
        )
        references.extend(refs)
        pdb_candidates[og] = chains
        exclusions.extend(rejected)
        queries_info[og] = {**query_info, "header": headers[og]}

    write_json(
        out,
        dict(
            schema_version="site-reference-catalog-1",
            orthodb_release=release,
            orthogroups=sorted(orthogroups),
            references=references,
            exclusions=exclusions,
            sources=[dict(path=str(queries.resolve()), sha256=sha256(queries))],
            resource_receipts=list(cache.receipts.values()),
            structures=[],
            pdb_search=dict(
                reference_policy=PDB_SEARCH_POLICY,
                queries=queries_info,
                candidates=pdb_candidates,
            ),
        ),
    )
    return out
