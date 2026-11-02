"""Run-local, disk-backed residue compositions shared across training folds."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np


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

    def get(
        self,
        species: list[str],
        columns: np.ndarray,
        compute: Callable[[list[str], np.ndarray], tuple[np.ndarray, np.ndarray]],
        *,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        missing = np.setdiff1d(columns, self.columns)
        if missing.size:
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

        values = np.full((len(species), len(columns), 20), np.nan, dtype=np.float32)
        coverage = np.zeros((len(species), len(columns)), dtype=np.float32)
        present = [(i, self.species_rows[s]) for i, s in enumerate(species)
                   if s in self.species_rows]
        if present and len(columns):
            targets, rows = np.array(present, dtype=int).T
            indices = np.searchsorted(self.columns, columns)
            starts = self.offsets[indices]
            bits = np.left_shift(np.uint32(1), np.arange(20, dtype=np.uint32))
            mask = (self.states[indices, None] & bits) != 0
            counts = mask.sum(axis=1)
            data = np.memmap(self.path, mode="r", dtype=np.float32, shape=(self.length,))
            observed_coverage = data[starts[:, None] + rows]
            block = np.full((len(columns), len(rows), 20), np.nan, dtype=np.float32)
            block[np.isfinite(observed_coverage)] = 0
            columns_with_state, aa = np.nonzero(mask)
            rank = np.cumsum(mask, axis=1) - 1
            locations = (
                starts[columns_with_state, None] + len(self.species)
                + rows * counts[columns_with_state, None]
                + rank[columns_with_state, aa, None]
            )
            block[columns_with_state, :, aa] = data[locations]
            values[targets] = block.transpose(1, 0, 2)
            coverage[targets] = np.nan_to_num(observed_coverage, nan=0).T
        return values, coverage
