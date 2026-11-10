from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from phenoradar.site_reference_resources import (
    ResourceCache,
    SiteAnnotationError,
    prepare_reference_catalog,
    read_fasta,
)


class Response(io.BytesIO):
    headers = {"ETag": "frozen-test"}


def test_resource_receipts_offline_reuse_corruption_and_negative_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def get(request: Any, **kwargs: Any) -> Response:
        calls.append(request.full_url)
        if request.full_url.endswith("missing"):
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        return Response(b'{"value": 1}')

    monkeypatch.setattr("phenoradar.site_reference_resources.urlopen", get)
    cache = ResourceCache(tmp_path)
    assert cache.json("https://example.org/data") == {"value": 1}
    assert cache.json("https://example.org/missing", missing_ok=True) is None
    offline = ResourceCache(tmp_path, offline=True)
    assert offline.json("https://example.org/data") == {"value": 1}
    assert offline.json("https://example.org/missing", missing_ok=True) is None
    assert len(calls) == 2
    with pytest.raises(SiteAnnotationError, match="not cached"):
        offline.json("https://example.org/other")
    path = cache.fetch("https://example.org/data")
    assert path is not None
    path.write_bytes(b"changed")
    with pytest.raises(SiteAnnotationError, match="Corrupt"):
        offline.fetch("https://example.org/data")


def test_timeout_is_not_cached_as_absence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def get(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError("test timeout")

    monkeypatch.setattr("phenoradar.site_reference_resources.urlopen", get)
    monkeypatch.setattr("phenoradar.site_reference_resources.time.sleep", lambda _: None)
    with pytest.raises(SiteAnnotationError, match="retrieval failed"):
        ResourceCache(tmp_path).json("https://example.org/data", missing_ok=True)
    assert list(tmp_path.iterdir()) == []


def test_concurrent_first_fetch_keeps_one_complete_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    barrier, lock = Barrier(2), Lock()
    payloads = [b'{"version": 1}', b'{"version": 2}']

    def get(request: Any, **kwargs: Any) -> Response:
        with lock:
            payload = payloads.pop()
        barrier.wait(timeout=5)
        return Response(payload)

    monkeypatch.setattr("phenoradar.site_reference_resources.urlopen", get)

    def fetch(_: int) -> Any:
        return ResourceCache(tmp_path).json("https://example.org/race")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, range(2)))
    assert results[0] == results[1]
    assert ResourceCache(tmp_path, offline=True).json("https://example.org/race") == results[0]


@pytest.mark.parametrize("fasta", ["", "AAA", ">a\nAA\n>a\nVV", ">a\n"])
def test_reject_bad_fasta(fasta: str) -> None:
    with pytest.raises(SiteAnnotationError):
        read_fasta(fasta)


class CatalogCache:
    def __init__(self, tmp: Path) -> None:
        self.receipts: dict[str, Any] = {}
        self.tmp = tmp
        self.release = "v12.2"
        self.bad_sequence = False

    def json(self, url: str, **kwargs: Any) -> Any:
        if url.endswith("/orthodb_release_id"):
            return self.release
        if "/group?" in url:
            return dict(status="ok", data=dict(phyletic_profile=dict(gene_count=1)))
        if "/orthologs?" in url:
            return dict(status="ok", data=[dict(genes=[dict(gene_id=dict(param="g1"))])])
        if "/ogdetails?" in url:
            return dict(status="ok", data=dict(gene_id=dict(id="g1"), uniprot=[dict(id="P00001")]))
        return dict(
            sequence=dict(value="VVV" if self.bad_sequence else "AAA"),
            primaryAccession="P00001",
            entryAudit=dict(sequenceVersion=1),
            entryType="UniProtKB reviewed (Swiss-Prot)",
            organism=dict(scientificName="Species one"),
        )

    def fetch(self, url: str, **kwargs: Any) -> Path:
        path = self.tmp / "member.fa"
        path.write_text(">g1 membership\nAAA\n")
        return path


def test_preparation_proves_pinned_membership_and_records_sequence_change(tmp_path: Path) -> None:
    cache = CatalogCache(tmp_path)
    prepare_reference_catalog(
        orthogroups=["OG1"], release="v12.2", out=tmp_path / "catalog.json", cache=cache
    )  # type: ignore[arg-type]
    data = json.loads((tmp_path / "catalog.json").read_text())
    assert data["references"][0]["orthodb_gene_ids"] == ["g1"]
    assert data["references"][0]["membership_verified"] is True
    cache.bad_sequence = True
    prepare_reference_catalog(
        orthogroups=["OG1"], release="v12.2", out=tmp_path / "changed.json", cache=cache
    )  # type: ignore[arg-type]
    data = json.loads((tmp_path / "changed.json").read_text())
    assert data["references"] == []
    assert data["exclusions"][0]["reason"] == "sequence_differs_from_pinned_orthodb_members"
    cache.release = "v13.0"
    with pytest.raises(SiteAnnotationError, match="release mismatch"):
        prepare_reference_catalog(
            orthogroups=["OG1"], release="v12.2", out=tmp_path / "wrong.json", cache=cache
        )  # type: ignore[arg-type]


def test_inactive_uniprot_is_a_recorded_exclusion(tmp_path: Path) -> None:
    cache = CatalogCache(tmp_path)
    original = cache.json

    def get(url: str, **kwargs: Any) -> Any:
        if "/uniprotkb/" in url:
            return dict(entryType="Inactive", inactiveReason=dict(inactiveReasonType="DELETED"))
        return original(url, **kwargs)

    cache.json = get  # type: ignore[method-assign]
    prepare_reference_catalog(
        orthogroups=["OG1"], release="v12.2", out=tmp_path / "inactive.json", cache=cache
    )  # type: ignore[arg-type]
    data = json.loads((tmp_path / "inactive.json").read_text())
    assert data["references"] == []
    assert data["exclusions"][0]["reason"] == "uniprot_inactive"


def test_annotation_launcher_defaults_to_one_thread_and_preserves_user_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phenoradar.launcher import _maybe_set_polars_max_threads

    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    _maybe_set_polars_max_threads(["annotate-sites"])
    assert os.environ["POLARS_MAX_THREADS"] == "1"
    monkeypatch.setenv("POLARS_MAX_THREADS", "3")
    _maybe_set_polars_max_threads(["prepare-site-references"])
    assert os.environ["POLARS_MAX_THREADS"] == "3"
