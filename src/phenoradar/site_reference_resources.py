"""Frozen public resources for optional, post-fit site annotation."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SiteAnnotationError(ValueError):
    """Invalid annotation inputs, incomplete resources, or failed external tools."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_fasta(text: str) -> dict[str, str]:
    records: dict[str, str] = {}
    name = ""
    for line in text.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            if not name or name in records:
                raise SiteAnnotationError("Empty or duplicated FASTA identifier.")
            records[name] = ""
        elif line.strip():
            if not name:
                raise SiteAnnotationError("FASTA sequence precedes its header.")
            records[name] += line.strip().upper()
    if not records or any(not seq for seq in records.values()):
        raise SiteAnnotationError("Empty FASTA record or file.")
    return records


def _publish_cache(path: Path, payload: bytes) -> None:
    """Publish a complete file without replacing a snapshot another process won."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            with suppress(FileExistsError):
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class ResourceCache:
    """Immutable URL receipts; timeouts are errors, explicit 404s are auditable absences."""

    def __init__(self, directory: Path, *, offline: bool = False, timeout: float = 30) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.offline, self.timeout = offline, timeout
        self.receipts: dict[str, dict[str, Any]] = {}
        self._last_orthodb = 0.0

    def fetch(self, url: str, *, missing_ok: bool = False) -> Path | None:
        key = hashlib.sha256(url.encode()).hexdigest()
        receipt_path = self.directory / f"{key}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("url") != url:
                raise SiteAnnotationError("Cache URL receipt mismatch.")
            if receipt["status"] == 404:
                if not missing_ok:
                    raise SiteAnnotationError(f"Resource is absent: {url}")
                self.receipts[url] = receipt
                return None
            path = self.directory / str(receipt["sha256"])
            if not path.is_file() or sha256(path) != receipt["sha256"]:
                raise SiteAnnotationError(f"Corrupt resource cache: {url}")
            self.receipts[url] = receipt
            return path
        if self.offline:
            raise SiteAnnotationError(f"Offline resource is not cached: {url}")
        if url.startswith("https://data.orthodb.org/") and url.split("?")[0].endswith(
            ("/fasta", "/tab")
        ):
            time.sleep(max(0, 1.05 - (time.monotonic() - self._last_orthodb)))
            self._last_orthodb = time.monotonic()
        for attempt in range(3):
            try:
                request = Request(url, headers={"User-Agent": "PhenoRadar-site-annotation/1"})
                with urlopen(request, timeout=self.timeout) as response:
                    payload = response.read(64 * 1024 * 1024 + 1)
                    headers = dict(response.headers.items())
                if len(payload) > 64 * 1024 * 1024:
                    raise SiteAnnotationError(f"Resource exceeds 64 MiB: {url}")
                digest = hashlib.sha256(payload).hexdigest()
                path = self.directory / digest
                _publish_cache(path, payload)
                receipt = dict(
                    url=url,
                    status=200,
                    sha256=digest,
                    size=len(payload),
                    retrieved_at=datetime.now(UTC).isoformat(),
                    headers=headers,
                )
                _publish_cache(receipt_path, (json.dumps(receipt, sort_keys=True) + "\n").encode())
                return self.fetch(url, missing_ok=missing_ok)
            except HTTPError as exc:
                code = exc.code
                exc.close()
                if code == 404 and missing_ok:
                    receipt = dict(url=url, status=404, retrieved_at=datetime.now(UTC).isoformat())
                    _publish_cache(
                        receipt_path, (json.dumps(receipt, sort_keys=True) + "\n").encode()
                    )
                    return self.fetch(url, missing_ok=missing_ok)
                if code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise SiteAnnotationError(f"HTTP {code}: {url}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise SiteAnnotationError(f"Resource retrieval failed: {url}: {exc}") from exc
            time.sleep(2**attempt)
        raise SiteAnnotationError(f"Resource retrieval failed: {url}")

    def json(self, url: str, *, missing_ok: bool = False) -> Any:
        path = self.fetch(url, missing_ok=missing_ok)
        return None if path is None else json.loads(path.read_text())


def _odb_data(value: Any, label: str) -> Any:
    if not isinstance(value, dict) or value.get("status") != "ok" or value.get("data") is None:
        raise SiteAnnotationError(f"Invalid OrthoDB response: {label}")
    return value["data"]


def _table_members(
    ogs: set[str],
    members: Path,
    xrefs: Path,
) -> tuple[dict[str, dict[str, set[str]]], list[dict[str, Any]]]:
    gene_ogs: dict[str, set[str]] = defaultdict(set)
    sources = []
    for path in [members, xrefs]:
        sources.append(dict(path=str(path.resolve()), sha256=sha256(path)))
    opener = gzip.open if members.suffix == ".gz" else open
    with opener(members, "rt") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2 and row[0] in ogs:
                gene_ogs[row[1]].add(row[0])
    result: dict[str, dict[str, set[str]]] = {og: defaultdict(set) for og in ogs}
    opener = gzip.open if xrefs.suffix == ".gz" else open
    with opener(xrefs, "rt") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3 and row[0] in gene_ogs and row[2].lower() == "uniprot":
                for og in gene_ogs[row[0]]:
                    result[og][row[1]].add(row[0])
    return result, sources


def prepare_reference_catalog(
    *,
    orthogroups: list[str],
    release: str,
    out: Path,
    cache: ResourceCache,
    members: Path | None = None,
    xrefs: Path | None = None,
) -> Path:
    """Enumerate all same-release members, then freeze all available UniProt candidates."""
    if out.exists():
        raise SiteAnnotationError(f"Output already exists: {out}")
    if not re.fullmatch(r"v?\d+\.\d+", release):
        raise SiteAnnotationError("An exact OrthoDB release is required, e.g. v12.2.")
    release = "v" + release.removeprefix("v")
    base = "https://data.orthodb.org/" + release.split(".")[0]
    actual = cache.json(base + "/orthodb_release_id")
    if actual != release:
        raise SiteAnnotationError(f"OrthoDB release mismatch: requested {release}, got {actual}")
    ogs = sorted(set(orthogroups))
    sources: list[dict[str, Any]] = []
    if (members is None) != (xrefs is None):
        raise SiteAnnotationError("Supply both OG2genes and gene_xrefs, or neither.")
    if members is not None and xrefs is not None:
        candidates, sources = _table_members(set(ogs), members, xrefs)
    else:
        candidates = {og: defaultdict(set) for og in ogs}
        for og in ogs:
            group = _odb_data(cache.json(base + "/group?" + urlencode({"id": og})), og)
            entries = _odb_data(cache.json(base + "/orthologs?" + urlencode({"id": og})), og)
            genes = {g["gene_id"]["param"] for entry in entries for g in entry["genes"]}
            if len(genes) != int(group["phyletic_profile"]["gene_count"]):
                raise SiteAnnotationError(f"Incomplete OrthoDB membership response: {og}")
            print(f"Discovering {og}: {len(genes)} same-release genes", flush=True)
            for gene_index, gene in enumerate(sorted(genes)):
                if gene_index % 50 == 0:
                    print(f"  gene metadata {gene_index}/{len(genes)}", flush=True)
                info = _odb_data(cache.json(base + "/ogdetails?" + urlencode({"id": gene})), gene)
                if info["gene_id"]["id"] != gene:
                    raise SiteAnnotationError(f"OrthoDB gene identity mismatch: {gene}")
                for record in info.get("uniprot", []):
                    candidates[og][record["id"]].add(gene)
    references = []
    exclusions = []
    for og in ogs:
        fasta_path = cache.fetch(base + "/fasta?" + urlencode({"id": og}))
        assert fasta_path is not None
        sequences = read_fasta(fasta_path.read_text())
        print(f"Freeze {og}: {len(candidates[og])} linked UniProt accessions", flush=True)
        for candidate_index, (accession, genes) in enumerate(sorted(candidates[og].items())):
            if candidate_index % 25 == 0:
                print(f"  UniProt metadata {candidate_index}/{len(candidates[og])}", flush=True)
            entry = cache.json(
                "https://rest.uniprot.org/uniprotkb/" + accession + ".json", missing_ok=True
            )
            if entry is None:
                exclusions.append(dict(orthogroup=og, accession=accession, reason="uniprot_absent"))
                continue
            if entry.get("entryType") == "Inactive":
                exclusions.append(
                    dict(
                        orthogroup=og,
                        accession=accession,
                        reason="uniprot_inactive",
                        inactive_reason=entry.get("inactiveReason"),
                    )
                )
                continue
            if "sequence" not in entry:
                raise SiteAnnotationError(f"UniProt response lacks sequence: {accession}")
            sequence = entry["sequence"]["value"].upper()
            matching = sorted(g for g in genes if sequences.get(g) == sequence)
            # Version/isoform changes must not become an unproven membership assignment.
            if not matching:
                exclusions.append(
                    dict(
                        orthogroup=og,
                        accession=accession,
                        reason="sequence_differs_from_pinned_orthodb_members",
                    )
                )
                continue
            primary = entry["primaryAccession"]
            references.append(
                dict(
                    orthogroup=og,
                    accession=primary,
                    requested_accession=accession,
                    sequence=sequence,
                    sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                    sequence_version=entry["entryAudit"]["sequenceVersion"],
                    reviewed=entry["entryType"] == "UniProtKB reviewed (Swiss-Prot)",
                    fragment=str(entry.get("proteinDescription", {}).get("flag", "")).lower()
                    in {"fragment", "fragments"}
                    or any(
                        x.get("type") == "Non-terminal residue" for x in entry.get("features", [])
                    ),
                    organism=entry["organism"]["scientificName"],
                    orthodb_gene_ids=matching,
                    membership_verified=True,
                )
            )
    # Aliases may resolve to the same current accession, but never to conflicting sequences.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in references:
        key = record["orthogroup"], record["accession"]
        if key in unique:
            if unique[key]["sequence"] != record["sequence"]:
                raise SiteAnnotationError(f"Conflicting reference sequences: {key}")
            unique[key]["orthodb_gene_ids"] = sorted(
                set(unique[key]["orthodb_gene_ids"] + record["orthodb_gene_ids"])
            )
        else:
            unique[key] = record
    write_json(
        out,
        dict(
            schema_version="site-reference-catalog-1",
            orthodb_release=release,
            orthogroups=ogs,
            references=[unique[k] for k in sorted(unique)],
            exclusions=exclusions,
            sources=sources,
            resource_receipts=list(cache.receipts.values()),
            structures=[],
        ),
    )
    return out
