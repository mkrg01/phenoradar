"""Private runtime arenas retain exact immutable frequency extents in fewer files."""

from __future__ import annotations

import hashlib
import shutil
from copy import copy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

MIN_ARENA_BYTES = 1024**3
MIN_ARENA_FILES = 128


def prepare_snapshot(store: Any) -> None:
    """Attempt one optional arena for a large, already-validated input snapshot."""
    if not getattr(store, "cache_frequencies", False) or getattr(
        store, "_frequency_arena_attempted", False
    ):
        return
    caches = store._frequency_caches.values()
    files = {chunk.path for cache in caches for chunk in cache._chunks}
    if (
        len(files) < MIN_ARENA_FILES
        or sum(cache.length for cache in store._frequency_caches.values()) < MIN_ARENA_BYTES
    ):
        return
    try:
        summary = consolidate(store)
    except OSError as error:
        summary = dict(status="original_extents_retained", error=str(error))
    store._frequency_arena_summary = summary
    store._frequency_arena_attempted = True


def consolidate(store: Any, *, file_bytes: int = 512 * 1024**2) -> dict[str, Any]:
    """Publish cloned cache descriptors only after all private copies are verified."""
    if file_bytes < 1:
        raise ValueError("Arena size must be positive")
    extents = {
        (chunk.path, chunk.start, chunk.stop)
        for cache in store._frequency_caches.values()
        for chunk in cache._chunks
    }
    total = sum(stop - start for _, start, stop in extents)
    parent = Path(store._temp.name)
    if not extents:
        return dict(status="empty", bytes=0, source_files=0, arena_files=0)
    if shutil.disk_usage(parent).free < total + 64 * 1024**2:
        return dict(
            status="insufficient_scratch",
            bytes=total,
            source_files=len({p for p, _, _ in extents}),
            arena_files=0,
        )
    arena = TemporaryDirectory(prefix="frequency-arena-", dir=parent)
    destination = Path(arena.name)
    translated: dict[tuple[Path, int, int], tuple[Path, int, int]] = {}
    lengths: dict[Path, int] = {}
    checksums: dict[Path, str] = {}
    target: Any = None
    path = destination / "0.bin"
    expected: Any = None

    def finish() -> None:
        if target is not None:
            target.flush()
            target.close()
            with path.open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != expected.hexdigest():
                raise OSError("Frequency arena copy failed its checksum")
            checksums[path] = observed

    try:
        for key in sorted(extents):
            source, start, stop = key
            size = stop - start
            if target is None or (target.tell() and target.tell() + size > file_bytes):
                finish()
                path = destination / f"{len(lengths)}.bin"
                target = path.open("wb")
                expected = hashlib.sha256()
            offset = target.tell()
            with source.open("rb") as stream:
                stream.seek(start)
                remaining = size
                while remaining:
                    block = stream.read(min(1024**2, remaining))
                    if not block:
                        raise OSError("Truncated frequency source extent")
                    expected.update(block)
                    target.write(block)
                    remaining -= len(block)
            translated[key] = path, offset, offset + size
            lengths[path] = target.tell()
        finish()
        caches = {}
        for key, original in store._frequency_caches.items():
            cloned = copy(original)
            cloned._chunks = [
                replace(
                    chunk,
                    path=translated[chunk.path, chunk.start, chunk.stop][0],
                    start=translated[chunk.path, chunk.start, chunk.stop][1],
                    stop=translated[chunk.path, chunk.start, chunk.stop][2],
                )
                for chunk in original._chunks
            ]
            cloned._file_lengths = {
                path: size for path, size in original._file_lengths.items() if path == original.path
            }
            cloned._file_lengths.update(lengths)
            # Classification cache keys use physical extent locations.
            cloned._binary_eligibility = {}
            caches[key] = cloned
        previous = getattr(store, "_frequency_arenas", [])
        owners = [*previous, arena]
        result = dict(
            status="complete",
            bytes=total,
            extents=len(extents),
            source_files=len({p for p, _, _ in extents}),
            arena_files=len(lengths),
            checksum_verified=True,
            arena_sha256={str(p): v for p, v in checksums.items()},
        )
        store._frequency_arenas = owners
        store._frequency_caches = caches
        return result
    except BaseException:
        if target is not None and not target.closed:
            target.close()
        arena.cleanup()
        raise
