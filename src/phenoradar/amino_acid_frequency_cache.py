"""Run-local, disk-backed residue compositions shared across training folds."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from shutil import copyfile
from threading import Lock

import numpy as np

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


class AminoAcidFrequencyCache:
    """Store exact positive state frequencies; never cache a learned filter or mean."""

    def __init__(self, path: Path, species_rows: dict[str, int]) -> None:
        self.path = path
        self.species_rows = species_rows
        self.species = sorted(species_rows, key=species_rows.__getitem__)
        self.columns = np.empty(0, dtype=np.int64)
        self.offsets = np.empty(0, dtype=np.int64)
        self.states = np.empty(0, dtype=np.uint32)
        self.length = 0
        self.shared = False

    def get_compact(
        self,
        species: list[str],
        columns: np.ndarray,
        compute: Callable[[list[str], np.ndarray], tuple[np.ndarray, np.ndarray]],
        *,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.searchsorted(self.columns, columns)
        complete = (len(self.columns) > 0 and np.all(indices < len(self.columns))
                    and np.array_equal(self.columns[indices], columns))
        missing = (np.empty(0, dtype=columns.dtype) if complete
                   else np.setdiff1d(columns, self.columns))
        if missing.size:
            if self.shared:
                local = self.path.with_name(self.path.name + ".private")
                copyfile(self.path, local)
                self.path = local
                self.shared = False
            offsets, states = [], []
            sample_rows = np.arange(len(self.species))
            with self.path.open("ab") as handle:
                for start in range(0, len(missing), block_size):
                    selected = missing[start : start + block_size]
                    values, coverage = compute(self.species, selected)
                    present = np.any(values > 0, axis=0)
                    counts = present.sum(axis=1)
                    lengths = (counts + 1) * len(self.species)
                    starts = np.cumsum(np.concatenate(([0], lengths[:-1])))
                    block = np.empty(int(lengths.sum()), dtype=np.float32)
                    # NaN coverage marks unobserved cells; finite zero coverage stays observed.
                    block[starts[:, None] + sample_rows] = np.where(
                        np.isfinite(values[:, :, 0].T), coverage.T, np.nan
                    )
                    columns_with_state, aa = np.nonzero(present)
                    rank = np.cumsum(present, axis=1) - 1
                    locations = (
                        starts[columns_with_state, None] + len(self.species)
                        + sample_rows * counts[columns_with_state, None]
                        + rank[columns_with_state, aa, None]
                    )
                    block[locations] = values[:, columns_with_state, aa].T
                    block.tofile(handle)
                    offsets.append(starts + self.length)
                    bits = np.left_shift(np.uint32(1), np.arange(20, dtype=np.uint32))
                    states.append(np.sum(present * bits, axis=1, dtype=np.uint32))
                    self.length += len(block)
            order = np.argsort(np.concatenate((self.columns, missing)))
            self.columns = np.concatenate((self.columns, missing))[order]
            self.offsets = np.concatenate([self.offsets, *offsets])[order]
            self.states = np.concatenate([self.states, *states])[order]

        indices = np.searchsorted(self.columns, columns)
        bits = np.left_shift(np.uint32(1), np.arange(20, dtype=np.uint32))
        mask = (self.states[indices, None] & bits) != 0
        state_sites, residues = np.nonzero(mask)
        values = np.full((len(species), len(state_sites)), np.nan, dtype=np.float32)
        coverage = np.zeros((len(species), len(columns)), dtype=np.float32)
        observed = np.zeros(coverage.shape, dtype=bool)
        present = [(i, self.species_rows[s]) for i, s in enumerate(species)
                   if s in self.species_rows]
        if present and len(columns):
            targets, rows = np.array(present, dtype=int).T
            starts = self.offsets[indices]
            counts = mask.sum(axis=1)
            data = _mapped_file(self.path, self.length)
            observed_coverage = data[starts[:, None] + rows]
            rank = np.cumsum(mask, axis=1) - 1
            locations = (
                starts[state_sites, None] + len(self.species)
                + rows * counts[state_sites, None]
                + rank[state_sites, residues, None]
            )
            values[targets] = data[locations].T
            observed[targets] = np.isfinite(observed_coverage).T
            coverage[targets] = np.where(np.isfinite(observed_coverage), observed_coverage, 0).T
        return values, coverage, observed, state_sites, residues

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
