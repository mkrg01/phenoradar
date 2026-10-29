"""Reusable, label-independent Parquet representation of sequence TPM input."""

from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import warnings
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

import polars as pl

from phenoradar.provenance import file_identity

_VERSION = 1
_COLUMNS = ["species", "sequence_id", "tpm"]
_CHUNK_BYTES = 32 * 1024 * 1024
_QUERY_PARTS = 8


class SequenceTPMCacheError(ValueError):
    """Invalid sequence TPM input or a source modified during cache creation."""


class _Reader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...
    def readline(self, size: int = -1, /) -> bytes: ...


def _signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _restore(entry: Path, source: dict[str, Any]) -> dict[str, Any] | None:
    try:
        manifest = json.loads((entry / "manifest.json").read_text())
        if manifest["version"] != _VERSION or manifest["source_signature"] != source:
            return None
        if manifest["payload_signatures"] != _payload_signatures(entry):
            return None
        first = next(entry.glob("part-*.parquet"))
        if dict(pl.read_parquet_schema(first)) != dict.fromkeys(_COLUMNS, pl.String):
            return None
        if len(manifest["source_identity"]["sha256"]) != 64:
            return None
        return dict(manifest)
    except (OSError, ValueError, KeyError, TypeError, StopIteration, pl.exceptions.PolarsError):
        return None


def _payload_signatures(entry: Path) -> dict[str, dict[str, Any]]:
    signatures = {}
    for path in sorted(entry.glob("part-*.parquet")):
        stat = _signature(path)
        stat.pop("path")  # Publishing renames the parent directory only.
        signatures[path.name] = stat
    return signatures


def _block(handle: _Reader) -> bytes:
    chunk = handle.read(_CHUNK_BYTES)
    if not chunk:
        return chunk
    if not chunk.endswith(b"\n"):
        chunk += handle.readline(_CHUNK_BYTES)
        if len(chunk) >= 2 * _CHUNK_BYTES and not chunk.endswith(b"\n"):
            raise SequenceTPMCacheError("Sequence TPM record exceeds the block size limit")
    # Finish a quoted CSV record if it spans physical lines. Escaped quotes
    # occur in pairs and do not change parity.
    while chunk.count(b'"') % 2:
        extra = handle.readline(_CHUNK_BYTES)
        if not extra:
            raise SequenceTPMCacheError("Unterminated quoted sequence TPM record")
        chunk += extra
        if len(chunk) > 2 * _CHUNK_BYTES:
            raise SequenceTPMCacheError("Sequence TPM record exceeds the block size limit")
    return chunk


def _build(
    path: Path, entry: Path, source: dict[str, Any], progress: Callable[[str], None] | None
) -> dict[str, Any]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with TemporaryDirectory(prefix=".building-", dir=entry.parent) as directory:
        temporary = Path(directory)
        # Keep TPM as text: validation remains restricted to requested genes,
        # identically on cold/warm reads. No labels or metadata enter this file.
        count = rows = 0
        species_index: dict[str, list[str]] = {}
        with opener(path, "rb") as handle:
            header = next(csv.reader([handle.readline().decode("utf-8")], delimiter="\t"), [])
            if any(header.count(name) != 1 for name in _COLUMNS):
                raise SequenceTPMCacheError(
                    "Sequence TPM requires unique species, sequence_id, tpm columns"
                )
            while block := _block(handle):
                if not block.strip():
                    continue
                frame = pl.read_csv(
                    BytesIO(block),
                    separator="\t",
                    has_header=False,
                    schema=dict.fromkeys(header, pl.String),
                    missing_utf8_is_empty_string=True,
                ).select(_COLUMNS)
                name = f"part-{count:06}.parquet"
                frame.write_parquet(
                    temporary / name,
                    compression="zstd",
                    row_group_size=65536,
                )
                species_index[name] = frame["species"].drop_nulls().unique().to_list()
                count += 1
                rows += frame.height
                if progress and count % 64 == 0:
                    progress(f"Sequence TPM cache: {rows:,} input rows converted.")
        if count == 0:
            pl.DataFrame(schema=dict.fromkeys(_COLUMNS, pl.String)).write_parquet(
                temporary / "part-000000.parquet"
            )
            species_index["part-000000.parquet"] = []
        identity = {**file_identity(path), "n_source_rows": rows}
        if _signature(path) != source:
            raise SequenceTPMCacheError(f"Sequence TPM changed during cache creation: {path}")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "version": _VERSION,
                    "source_signature": source,
                    "source_identity": identity,
                    "payload_signatures": _payload_signatures(temporary),
                    "part_species": species_index,
                },
                sort_keys=True,
            )
            + "\n"
        )
        if entry.exists():
            shutil.rmtree(entry)
        os.replace(temporary, entry)
    return identity


def _species_index(entry: Path, manifest: dict[str, Any]) -> dict[str, list[str]]:
    index = manifest.get("part_species")
    names = sorted(manifest["payload_signatures"])
    if (
        isinstance(index, dict)
        and set(index) == set(names)
        and all(isinstance(v, list) and all(isinstance(s, str) for s in v) for v in index.values())
    ):
        return index
    # Upgrade an existing complete cache without reading the original gzip.
    index = {
        name: pl.read_parquet(entry / name, columns=["species"])["species"]
        .drop_nulls()
        .unique()
        .to_list()
        for name in names
    }
    manifest["part_species"] = index
    with TemporaryDirectory(prefix=".index-", dir=entry) as directory:
        temporary = Path(directory) / "manifest.json"
        temporary.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        os.replace(temporary, entry / "manifest.json")
    return index


def read_sequence_tpm_cache(
    path: Path,
    keys: pl.DataFrame,
    cache_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    """Create once, then read only matching species/genes with predicate pushdown.

    Stat identities avoid rehashing the gzip input on every hit. Source changes
    (including ctime/inode changes with restored mtime) get a new entry. Locks
    and atomic publication prevent readers from seeing a partial cache.
    """
    source = _signature(path)
    key = hashlib.sha256(json.dumps([_VERSION, source], sort_keys=True).encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    entry = cache_root / key
    with (cache_root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _restore(entry, source)
        hit = manifest is not None
        if progress:
            progress(
                "Reuse sequence TPM Parquet cache."
                if hit
                else "Create sequence TPM Parquet cache (first read)."
            )
        if manifest is None:
            _build(path, entry, source, progress)
            manifest = json.loads((entry / "manifest.json").read_text())
        identity = manifest["source_identity"]
        try:
            index = _species_index(entry, manifest)
            by_species = {k[0]: v for k, v in keys.partition_by("species", as_dict=True).items()}
            requested = set(by_species)
            names = [n for n in sorted(index) if requested.intersection(index[n])]
            matches: list[tuple[Any, ...]] = []
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the `streaming` parameter was deprecated",
                    category=DeprecationWarning,
                )
                for start in range(0, len(names), _QUERY_PARTS):
                    batch = names[start : start + _QUERY_PARTS]
                    samples = requested.intersection(s for n in batch for s in index[n])
                    part_keys = pl.concat([by_species[s] for s in sorted(samples)])
                    scan = (
                        pl.scan_parquet([entry / n for n in batch])
                        .filter(pl.col("species").is_in(sorted(samples)))
                        .filter(
                            pl.col("sequence_id").is_in(part_keys["sequence_id"].unique().implode())
                        )
                        .join(part_keys.lazy(), on=["species", "sequence_id"], how="semi")
                    )
                    # Supports Polars >=1.9, including its old keyword spelling.
                    collect = cast(Callable[..., pl.DataFrame], scan.collect)
                    frame = collect(streaming=True)
                    # Copy sparse matches to Python strings. Retaining Polars
                    # StringView results can retain entire decoded part buffers.
                    matches.extend(frame.rows())
                    del frame
        except pl.exceptions.PolarsError as exc:
            raise SequenceTPMCacheError(
                f"Cannot read sequence TPM cache; remove {entry}: {exc}"
            ) from exc
        if _signature(path) != source:
            raise SequenceTPMCacheError(f"Sequence TPM changed during import: {path}")
    return matches, {**identity, "cache_hit": hit, "cache_path": str(entry)}
