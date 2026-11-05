"""Run-local, disk-backed residue compositions shared across training folds."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from phenoradar.amino_acid_presence import any_presence

_MAX_OPEN_MAPS = 64
_MAPS: OrderedDict[tuple[Path, int], np.ndarray] = OrderedDict()
_MAP_LOCK = Lock()


def _mapped_file(path: Path, length: int) -> np.ndarray:
    # Per-process LRU: one mapping per OG must not exhaust RLIMIT_NOFILE.
    key = path, length
    with _MAP_LOCK:
        data = _MAPS.pop(key, None)
        if data is None:
            data = np.memmap(path, mode="r", dtype=np.float32, shape=(length,))
        _MAPS[key] = data
        while len(_MAPS) > _MAX_OPEN_MAPS:
            _MAPS.popitem(last=False)
        return data


def release_frequency_maps(directory: Path) -> None:
    with _MAP_LOCK:
        for key in list(_MAPS):
            if key[0].is_relative_to(directory):
                del _MAPS[key]


@dataclass(frozen=True)
class _Chunk:
    path: Path
    start: int
    stop: int
    columns: int
    offsets: np.ndarray
    positive: np.ndarray


class AminoAcidFrequencyCache:
    """Append immutable state-major extents; forks write only their new columns."""

    def __init__(self, path: Path, species_rows: dict[str, int]) -> None:
        self.path = path
        self.species_rows = species_rows
        self.species = sorted(species_rows, key=species_rows.__getitem__)
        self.columns = np.empty(0, dtype=np.int64)
        self.states = np.empty(0, dtype=np.uint32)
        self._chunks: list[_Chunk] = []
        self._chunk_ids = np.empty(0, dtype=np.int32)
        self._rows = np.empty(0, dtype=np.int64)
        self._file_lengths: dict[Path, int] = {}
        self._writer_length = 0
        self.length = 0

    def fork(self, path: Path) -> AminoAcidFrequencyCache:
        result = copy(self)
        result.path = path
        result._chunks = list(self._chunks)
        result._file_lengths = dict(self._file_lengths)
        result._writer_length = 0
        return result

    def _ensure(
        self, columns: np.ndarray,
        compute: Callable[[list[str], np.ndarray], tuple[np.ndarray, np.ndarray]],
        block_size: int,
    ) -> None:
        indices = np.searchsorted(self.columns, columns)
        if (len(self.columns) and np.all(indices < len(self.columns))
                and np.array_equal(self.columns[indices], columns)):
            return
        missing = np.setdiff1d(columns, self.columns)
        if not len(missing):
            return
        states, chunk_ids, local_rows = [], [], []
        with self.path.open("ab") as handle:
            for start in range(0, len(missing), block_size):
                selected = missing[start:start + block_size]
                values, coverage = compute(self.species, selected)
                present = np.any(values > 0, axis=0)
                site, aa = np.nonzero(present)
                offsets = np.r_[0, np.cumsum(present.sum(axis=1))]
                observed_coverage = np.where(np.isfinite(values[:, :, 0]), coverage, np.nan)
                data = np.ascontiguousarray(values[:, site, aa].T, dtype=np.float32)
                np.ascontiguousarray(observed_coverage.T, dtype=np.float32).tofile(handle)
                data.tofile(handle)
                size = (len(selected) + len(site)) * len(self.species)
                self._chunks.append(_Chunk(
                    self.path, self._writer_length, self._writer_length + size,
                    len(selected), offsets, np.packbits(data > 0, axis=1, bitorder="little"),
                ))
                chunk_ids.append(np.full(len(selected), len(self._chunks) - 1, dtype=np.int32))
                local_rows.append(np.arange(len(selected)))
                bits = np.left_shift(np.uint32(1), np.arange(20, dtype=np.uint32))
                states.append(np.sum(present * bits, axis=1, dtype=np.uint32))
                self._writer_length += size
                self.length += size
        self._file_lengths[self.path] = self._writer_length
        order = np.argsort(np.concatenate((self.columns, missing)))
        self.columns = np.concatenate((self.columns, missing))[order]
        self.states = np.concatenate([self.states, *states])[order]
        self._chunk_ids = np.concatenate([self._chunk_ids, *chunk_ids])[order]
        self._rows = np.concatenate([self._rows, *local_rows])[order]

    def _parts(
        self, columns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Iterator[tuple[_Chunk, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]]]:
        indices = np.searchsorted(self.columns, columns)
        bits = np.left_shift(np.uint32(1), np.arange(20, dtype=np.uint32))
        mask = (self.states[indices, None] & bits) != 0
        sites, residues = np.nonzero(mask)
        rank = np.cumsum(mask, axis=1) - 1
        chunks = self._chunk_ids[indices]

        def parts() -> Iterator[tuple[_Chunk, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            for chunk_id in np.unique(chunks):
                chunk = self._chunks[int(chunk_id)]
                selected = np.flatnonzero(chunks == chunk_id)
                features = np.flatnonzero(chunks[sites] == chunk_id)
                rows = self._rows[indices[selected]]
                source = (chunk.offsets[self._rows[indices[sites[features]]]]
                          + rank[sites[features], residues[features]])
                yield chunk, selected, rows, features, source

        return sites, residues, parts()

    def eligible_columns(
        self, species: list[str], columns: np.ndarray,
        compute: Callable[[list[str], np.ndarray], tuple[np.ndarray, np.ndarray]],
        *, block_size: int, min_states: int,
    ) -> np.ndarray:
        self._ensure(columns, compute, block_size)
        indices = np.searchsorted(self.columns, columns)
        chunks = self._chunk_ids[indices]
        rows = np.array([self.species_rows[s] for s in species if s in self.species_rows],
                        dtype=np.int64)
        eligible = np.zeros(len(columns), dtype=bool)
        for chunk_id in np.unique(chunks):
            chunk = self._chunks[int(chunk_id)]
            selected = np.flatnonzero(chunks == chunk_id)
            source = self._rows[indices[selected]]
            # Prefix counts use the packed support directly; the 20-state schema
            # and its feature indices are expanded only for surviving columns.
            counts = np.r_[0, np.cumsum(any_presence(chunk.positive, rows))]
            eligible[selected] = (counts[chunk.offsets[source + 1]]
                                  - counts[chunk.offsets[source]]) >= min_states
        return eligible

    def get_compact(
        self, species: list[str], columns: np.ndarray,
        compute: Callable[[list[str], np.ndarray], tuple[np.ndarray, np.ndarray]],
        *, block_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._ensure(columns, compute, block_size)
        sites, residues, parts = self._parts(columns)
        values = np.full((len(species), len(sites)), np.nan, dtype=np.float32)
        coverage = np.zeros((len(species), len(columns)), dtype=np.float32)
        observed = np.zeros(coverage.shape, dtype=bool)
        present = [(i, self.species_rows[s]) for i, s in enumerate(species)
                   if s in self.species_rows]
        if present and len(columns):
            targets, rows = np.array(present, dtype=int).T
            for chunk, selected, source_rows, features, source_states in parts:
                data = _mapped_file(chunk.path, self._file_lengths[chunk.path])[
                    chunk.start:chunk.stop
                ].reshape(-1, len(self.species))
                raw_coverage = data[np.ix_(source_rows, rows)].T
                observed[np.ix_(targets, selected)] = np.isfinite(raw_coverage)
                coverage[np.ix_(targets, selected)] = np.where(
                    np.isfinite(raw_coverage), raw_coverage, 0,
                )
                values[np.ix_(targets, features)] = data[
                    np.ix_(chunk.columns + source_states, rows)
                ].T
        return values, coverage, observed, sites, residues

    def get(
        self, species: list[str], columns: np.ndarray,
        compute: Callable[[list[str], np.ndarray], tuple[np.ndarray, np.ndarray]],
        *, block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        compact, coverage, observed, sites, residues = self.get_compact(
            species, columns, compute, block_size=block_size,
        )
        values = np.full((len(species), len(columns), 20), np.nan, dtype=np.float32)
        values[observed] = 0
        values[:, sites, residues] = compact
        return values, coverage
