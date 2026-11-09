"""Exact, label-independent species masks; duplicate samples retain multiplicity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import BinaryIO

import numpy as np

_COUNTS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1, dtype=np.uint8,
)


def _write_presence(source: Path, handle: BinaryIO, width: int,
                    shape: tuple[int, int] | None = None, offset: int = 0) -> int:
    packed = np.load(source, mmap_mode="r", allow_pickle=False)
    if shape is not None:
        packed = packed.reshape(-1)[offset:offset + shape[0] * shape[1]].reshape(shape)
    block = max(1, 1_000_000 // max(1, len(packed)))
    for start in range(0, packed.shape[1], block):
        stop = min(packed.shape[1], start + block)
        bits = np.unpackbits(packed[:, start:stop], axis=1)
        end = min(width, stop * 8)
        values = np.packbits(bits[:, :end - start * 8].T, axis=1, bitorder="little")
        handle.write(memoryview(np.ascontiguousarray(values)))
    return (len(packed) + 7) // 8


@dataclass(frozen=True)
class _Extent:
    path: Path
    offset: int
    width: int
    nbytes: int


class SpeciesPresenceCache:
    """Share immutable masks; serialize only extents, never the writer lock."""

    def __init__(self, path: Path, entries: dict[tuple[str, bool], _Extent] | None = None) -> None:
        self.path = path
        self.entries = dict(entries or {})
        self._lock = Lock()

    def clear(self) -> None:
        self.entries.clear()

    def counts(self, key: tuple[str, bool], source: Path,
               width: int, rows: np.ndarray, *, shape: tuple[int, int] | None = None,
               offset: int = 0) -> np.ndarray:
        if not len(rows) or not width:
            return np.zeros(width, dtype=np.int32)
        extent = self.entries.get(key)
        if extent is None:
            with self._lock:
                extent = self.entries.get(key)
                if extent is None:
                    with self.path.open("ab") as handle:
                        start = handle.tell()
                        nbytes = _write_presence(source, handle, width, shape, offset)
                    extent = _Extent(self.path, start, width, nbytes)
                    self.entries[key] = extent
        packed = np.memmap(extent.path, mode="r", dtype=np.uint8, offset=extent.offset,
                           shape=(extent.width, extent.nbytes))
        return count_presence(packed, rows)


def species_mask(rows: np.ndarray, nbytes: int) -> np.ndarray:
    mask = np.zeros(nbytes, dtype=np.uint8)
    np.bitwise_or.at(mask, rows // 8, np.left_shift(np.uint8(1), rows % 8).astype(np.uint8))
    return mask


def any_presence(packed: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.asarray(np.any(packed & species_mask(rows, packed.shape[1]), axis=1), dtype=bool)


def count_presence(packed: np.ndarray, rows: np.ndarray) -> np.ndarray:
    result = np.zeros(len(packed), dtype=np.int32)
    unique, multiplicity = np.unique(rows, return_counts=True)
    block = max(1, 1_000_000 // max(1, packed.shape[1]))
    for weight in np.unique(multiplicity):
        mask = species_mask(unique[multiplicity == weight], packed.shape[1])
        for start in range(0, len(packed), block):
            values = _COUNTS[packed[start:start + block] & mask]
            result[start:start + block] += int(weight) * values.sum(axis=1, dtype=np.int32)
    return result
