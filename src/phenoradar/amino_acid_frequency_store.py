"""Immutable raw-frequency segments; concurrent folds never overwrite shared data."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast
from zipfile import BadZipFile

import numpy as np

from phenoradar.cache_identity import signature

if TYPE_CHECKING:
    from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache


def _payload(entry: Path) -> dict[str, dict[str, int]]:
    return {name: signature(entry / name) for name in ("values.bin", "index.npz")}


def _index_digest(arrays: dict[str, np.ndarray]) -> Any:
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest


def _content_matches(entry: Path, arrays: dict[str, np.ndarray] | None = None) -> bool:
    # Same-size writes can share a filesystem timestamp tick. The segment name
    # already contains a checksum of both the numeric index and exact payload.
    try:
        if arrays is None:
            with np.load(entry / "index.npz", allow_pickle=False) as raw:
                arrays = {name: raw[name] for name in raw.files}
        digest = _index_digest(arrays)
        with (entry / "values.bin").open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return entry.name == f"part-{digest.hexdigest()}"
    except (OSError, ValueError, EOFError, KeyError, TypeError, BadZipFile):
        return False


def restore(cache: AminoAcidFrequencyCache, directory: Path) -> None:
    from phenoradar.amino_acid_frequency_cache import _Chunk

    for entry in sorted(directory.glob("part-*")):
        try:
            manifest = json.loads((entry / "manifest.json").read_text())
            if manifest != {"version": 1, "payload": _payload(entry)}:
                continue
            with np.load(entry / "index.npz", allow_pickle=False) as raw:
                arrays = {name: raw[name] for name in raw.files}
            if not _content_matches(entry, arrays):
                continue
            columns, states, ids, rows = (arrays[name] for name in
                                        ("columns", "states", "chunk_ids", "rows"))
            descriptors = arrays["chunks"]
            if (columns.ndim != 1 or not len(columns)
                    or not (columns.shape == states.shape == ids.shape == rows.shape)
                    or np.any(columns < 0) or np.any(np.diff(columns) <= 0)
                    or descriptors.ndim != 2 or descriptors.shape[1] != 4
                    or np.any(ids < 0) or np.any(ids >= len(descriptors))):
                continue
            chunks = []
            path = entry / "values.bin"
            size = path.stat().st_size
            for i, (start, stop, count, packed) in enumerate(descriptors):
                offsets, positive = arrays[f"offsets_{i}"], arrays[f"positive_{i}"]
                selected_rows = rows[ids == i]
                if (not (0 <= start <= stop <= size) or count < 1 or packed not in (0, 1)
                        or offsets.shape != (count + 1,) or offsets[0] != 0
                        or np.any(np.diff(offsets) < 0)
                        or positive.shape != (int(offsets[-1]), (len(cache.species) + 7) // 8)
                        or np.any(selected_rows < 0) or np.any(selected_rows >= count)):
                    raise ValueError("Invalid frequency segment index")
                chunks.append(_Chunk(path, int(start), int(stop), int(count), offsets,
                                     positive, bool(packed)))
            # Overlapping segments are harmless: all contain the same raw values.
            keep = ~np.isin(columns, cache.columns)
            cache._file_lengths[path] = size
            base = len(cache._chunks)
            cache._chunks.extend(chunks)
            order = np.argsort(np.r_[cache.columns, columns[keep]])
            cache.columns = np.r_[cache.columns, columns[keep]][order]
            cache.states = np.r_[cache.states, states[keep]][order]
            cache._chunk_ids = np.r_[cache._chunk_ids, ids[keep] + base][order].astype(np.int32)
            cache._rows = np.r_[cache._rows, rows[keep]][order]
            cache.length += size
        except (OSError, ValueError, EOFError, KeyError, TypeError, BadZipFile):
            continue


def persist(cache: AminoAcidFrequencyCache, directory: Path) -> bool:
    own = [i for i, chunk in enumerate(cache._chunks)
           if chunk.path == cache.path and chunk.start >= cache._published_length]
    if not own:
        return False
    directory.mkdir(parents=True, exist_ok=True)
    selected = np.isin(cache._chunk_ids, own)
    ids = np.searchsorted(own, cache._chunk_ids[selected]).astype(np.int32)
    arrays: dict[str, np.ndarray] = {
        "columns": cache.columns[selected], "states": cache.states[selected],
        "chunk_ids": ids, "rows": cache._rows[selected],
    }
    chunks = [cache._chunks[i] for i in own]
    arrays["chunks"] = np.array([
        [c.start - cache._published_length, c.stop - cache._published_length,
         c.columns, int(c.packed)] for c in chunks], dtype=np.int64)
    for i, chunk in enumerate(chunks):
        arrays[f"offsets_{i}"], arrays[f"positive_{i}"] = chunk.offsets, chunk.positive
    with TemporaryDirectory(prefix=".building-", dir=directory) as temporary:
        pending = Path(temporary)
        digest = _index_digest(arrays)
        with cache.path.open("rb") as source, (pending / "values.bin").open("wb") as output:
            source.seek(cache._published_length)
            remaining = cache._writer_length - cache._published_length
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("Truncated private frequency cache")
                digest.update(block)
                output.write(block)
                remaining -= len(block)
        np.savez_compressed(pending / "index.npz", **cast(dict[str, Any], arrays))
        (pending / "manifest.json").write_text(json.dumps({"version": 1,
                                                          "payload": _payload(pending)}))
        token = digest.hexdigest()
        entry = directory / f"part-{token}"
        with (directory / f"{token}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                good = json.loads((entry / "manifest.json").read_text()) == {
                    "version": 1, "payload": _payload(entry)}
                good = good and _content_matches(entry)
            except (OSError, ValueError, KeyError, TypeError):
                good = False
            if not good:
                if entry.exists():
                    shutil.rmtree(entry)
                os.replace(pending, entry)
    cache._published_length = cache._writer_length
    return True
