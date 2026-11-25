"""Discover experimental PDB chains using one protein search query per OG."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from phenoradar.amino_acid import ALPHABET
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
_MAX_SAME_GENE_SUBSTITUTION_FRACTION = 0.02


def _pdb_hits(sequence: str, cache: ResourceCache) -> tuple[dict[str, float], int]:
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
    hits = {row["identifier"]: float(row["score"]) for row in result.get("result_set", [])}
    total: int = result["total_count"]
    if len(hits) != min(total, _SEARCH_HITS):
        raise SiteAnnotationError("Incomplete or duplicated RCSB sequence-search hits.")
    return hits, total


def _pdb_chains(hits: dict[str, float], cache: ResourceCache) -> list[dict[str, Any]]:
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
                            search_score=hits[entity_id],
                        )
                        for chain in ids["auth_asym_ids"]
                    )
    return chains


def _verified_candidate(
    og: str,
    accession: str,
    entry: dict[str, Any],
    member_sequences: dict[str, list[str]],
    gene_details: Callable[[str], dict[str, Any]],
) -> dict[str, Any] | None:
    if "sequence" not in entry:
        raise SiteAnnotationError(f"UniProt response lacks sequence: {accession}")
    sequence = entry["sequence"]["value"].upper()
    aliases = {accession, entry["primaryAccession"], *entry.get("secondaryAccessions", [])}
    genes = [
        gene for gene in member_sequences.get(sequence, [])
        if aliases & _gene_accessions(gene_details(gene))
    ]
    if not genes:
        return None
    return uniprot_reference(og, accession, entry, sorted(genes), sequence)


def _orthodb_gene_details(gene: str, base: str, cache: ResourceCache) -> dict[str, Any]:
    url = base + "/ogdetails?" + urlencode({"id": gene})
    info = _odb_data(cache.json(url), gene)
    if not isinstance(info, dict) or info["gene_id"]["id"] != gene:
        raise SiteAnnotationError(f"OrthoDB gene identity mismatch: {gene}")
    return info


def _gene_accessions(info: dict[str, Any]) -> set[str]:
    return {link["id"] for link in info.get("uniprot", [])}


def _pdb_same_gene_candidate(
    og: str,
    accession: str,
    entry: dict[str, Any],
    members: list[tuple[str, str]],
    gene_details: Callable[[str], dict[str, Any]],
) -> dict[str, Any] | None:
    """Verify a reviewed PDB reference against a same-taxon member's gene and sequence."""
    sequence = entry["sequence"]["value"].upper()
    if entry["entryType"] != "UniProtKB reviewed (Swiss-Prot)" or not sequence:
        return None
    if not set(sequence) <= set(ALPHABET):
        return None
    symbol = {
        g["geneName"]["value"].casefold() for g in entry.get("genes", []) if g.get("geneName")
    }
    gene_ids = {
        link["id"] for link in entry.get("uniProtKBCrossReferences", [])
        if link["database"] == "GeneID"
    }
    aliases = {accession, entry["primaryAccession"], *entry.get("secondaryAccessions", [])}
    mature_chain = any(
        f["type"] == "Chain"
        and f["location"]["start"] == {"value": 1, "modifier": "EXACT"}
        and f["location"]["end"] == {"value": len(sequence), "modifier": "EXACT"}
        for f in entry.get("features", [])
    )
    matches: list[dict[str, Any]] = []
    for gene, protein in members:
        if len(protein) == len(sequence):
            start = 1
            differences = [
                i for i, (a, b) in enumerate(zip(sequence, protein, strict=True), 1) if a != b
            ]
            if (
                len(differences) > len(sequence) * _MAX_SAME_GENE_SUBSTITUTION_FRACTION
                or any(protein[i - 1] not in ALPHABET for i in differences)
            ):
                continue
            method = "same_gene_full_length"
        elif mature_chain and len(protein) > len(sequence) and protein.endswith(sequence):
            start = len(protein) - len(sequence) + 1
            differences = []
            method = "same_gene_exact_c_terminal_mature_chain"
        else:
            continue
        info = gene_details(gene)
        # The primary reviewed symbol must match a same-taxon NCBI Gene alias exactly.
        evidence = [
            {"database": "NCBI Gene", "id": link["id"], "matched_by": "id"}
            for link in info.get("entrezgene", []) if link["id"] in gene_ids
        ]
        if aliases & _gene_accessions(info):
            evidence.append({"database": "OrthoDB", "id": gene, "matched_by": "uniprot_xref"})
        for link in info.get("entrezgene", []):
            synonyms = (link.get("synonyms") or "").casefold().split("|")
            for name in sorted(symbol.intersection(synonyms)):
                evidence.append({
                    "database": "NCBI Gene", "id": link["id"],
                    "matched_by": "primary_gene_symbol", "symbol": name,
                })
        if evidence:
            matches.append(dict(
                method=method, orthodb_gene_id=gene, orthodb_sequence_length=len(protein),
                orthodb_start_1based=start, orthodb_end_1based=len(protein),
                differences_at_uniprot_positions_1based=differences, gene_evidence=evidence,
            ))
    if len(matches) != 1:
        return None
    match = matches[0]
    ref = uniprot_reference(og, accession, entry, [match["orthodb_gene_id"]], sequence)
    return {**ref, "pdb_orthodb_match": match}


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


def _prepare_og(
    og: str, query: str, base: str, cache: ResourceCache,
    *, alignment_sequence_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    exclusions: list[dict[str, Any]] = []
    search_pdb = alignment_sequence_id is None or len(query) >= _RCSB_MINIMUM_QUERY_LENGTH
    hits, total_hits = _pdb_hits(query, cache) if search_pdb else ({}, 0)
    chains = _pdb_chains(hits, cache)
    query_info: dict[str, Any] = dict(
        sha256=hashlib.sha256(query.encode()).hexdigest(),
        pdb_total_hits=total_hits, pdb_screened_entities=len(hits),
    )
    if alignment_sequence_id is not None:
        query_info.update(alignment_sequence_id=alignment_sequence_id, length=len(query))
        if not search_pdb:
            query_info["pdb_search_skipped"] = "query_shorter_than_25_residues"
    members = cache.fetch(base + "/fasta?" + urlencode({"id": og}))
    assert members is not None
    sequences_by_gene = read_fasta(members.read_text())
    member_sequences: dict[str, list[str]] = defaultdict(list)
    members_by_taxon: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for gene, sequence in sequences_by_gene.items():
        member_sequences[sequence].append(gene)
        members_by_taxon[gene.split("_", 1)[0]].append((gene, sequence))
    details: dict[str, dict[str, Any]] = {}

    def gene_details(gene: str) -> dict[str, Any]:
        if gene not in details:
            details[gene] = _orthodb_gene_details(gene, base, cache)
        return details[gene]

    query_genes = set(member_sequences.get(query, []))
    if alignment_sequence_id is not None and alignment_sequence_id in sequences_by_gene:
        query_genes.add(alignment_sequence_id)
    # Query-linked references also give AlphaFold a fallback when PDB has no useful hit.
    pdb_accessions = {chain["accession"] for chain in chains}
    accessions = pdb_accessions.copy()
    accessions.update(link for gene in sorted(query_genes)
                      for link in _gene_accessions(gene_details(gene)))
    verified: dict[str, str] = {}
    refs: dict[str, dict[str, Any]] = {}

    def check(accession: str) -> None:
        entry = cache.json(_UNIPROT + accession + ".json", missing_ok=True)
        if entry is None or entry.get("entryType") == "Inactive":
            reason = "uniprot_unavailable"
        else:
            ref = _verified_candidate(
                og, accession, entry, member_sequences, gene_details,
            )
            if ref is None and accession in pdb_accessions:
                taxon = str(entry["organism"].get("taxonId", ""))
                ref = _pdb_same_gene_candidate(
                    og, accession, entry, members_by_taxon[taxon], gene_details,
                )
            if ref is not None:
                if not ref["fragment"]:
                    existing = refs.setdefault(ref["accession"], ref)
                    if existing["sequence_sha256"] != ref["sequence_sha256"]:
                        raise SiteAnnotationError(
                            "Conflicting UniProt sequences for one accession."
                        )
                    if "pdb_orthodb_match" in existing and "pdb_orthodb_match" not in ref:
                        refs[ref["accession"]] = ref
                    verified[accession] = ref["accession"]
                    return
                reason = "uniprot_fragment"
            else:
                reason = "no_exact_linked_orthodb_member"
        exclusions.append(dict(orthogroup=og, accession=accession, reason=reason))

    for accession in sorted(accessions):
        check(accession)
    if not any("pdb_orthodb_match" not in ref for ref in refs.values()):
        # A transcript query need not match a database sequence exactly.
        for sequence in _closest_member_sequences(query, member_sequences):
            for gene in member_sequences[sequence]:
                for accession in sorted(_gene_accessions(gene_details(gene)) - accessions):
                    accessions.add(accession)
                    check(accession)
            if any("pdb_orthodb_match" not in ref for ref in refs.values()):
                break
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for chain in chains:
        accession = verified.get(chain["accession"])
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
    release: str,
    queries: Path,
    out: Path,
    cache: ResourceCache,
    alignment_sequence_ids: dict[str, str] | None = None,
) -> Path:
    """Freeze site-mapping candidates without querying every OrthoDB gene."""
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
    release, base = orthodb_api(release, cache)
    references: list[dict[str, Any]] = []
    pdb_candidates: dict[str, list[dict[str, Any]]] = {}
    queries_info: dict[str, dict[str, Any]] = {}
    exclusions = []
    for og, query in sorted(sequences.items()):
        refs, chains, rejected, query_info = _prepare_og(
            og, query, base, cache,
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
                queries=queries_info,
                candidates=pdb_candidates,
            ),
        ),
    )
    return out
