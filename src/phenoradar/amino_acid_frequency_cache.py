"""Run-local, disk-backed residue compositions shared across training folds."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np


class AminoAcidFrequencyCache:
    """Append only requested columns; never cache a learned site filter or mean."""

    def __init__(self, path: Path, species_rows: dict[str, int]) -> None:
        self.path = path
        self.species_rows = species_rows
        self.species = sorted(species_rows, key=species_rows.__getitem__)
        self.columns = np.empty(0, dtype=np.int64)
        self.slots = np.empty(0, dtype=np.int64)

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
            old_size = len(self.columns)
            with self.path.open("ab") as handle:
                for start in range(0, len(missing), block_size):
                    selected = missing[start : start + block_size]
                    values, coverage = compute(self.species, selected)
                    block = np.empty((len(selected), len(self.species), 21), dtype=np.float32)
                    block[:, :, :20] = values.transpose(1, 0, 2)
                    block[:, :, 20] = coverage.T
                    block.tofile(handle)
            order = np.argsort(np.concatenate((self.columns, missing)))
            self.columns = np.concatenate((self.columns, missing))[order]
            self.slots = np.concatenate(
                (self.slots, np.arange(old_size, old_size + len(missing)))
            )[order]

        values = np.full((len(species), len(columns), 20), np.nan, dtype=np.float32)
        coverage = np.zeros((len(species), len(columns)), dtype=np.float32)
        present = [(i, self.species_rows[s]) for i, s in enumerate(species)
                   if s in self.species_rows]
        if present and len(columns):
            targets, rows = np.array(present, dtype=int).T
            slots = self.slots[np.searchsorted(self.columns, columns)]
            data = np.memmap(
                self.path, mode="r", dtype=np.float32,
                shape=(len(self.columns), len(self.species), 21),
            )
            block = data[slots[:, None], rows[None, :]]
            values[targets] = block[:, :, :20].transpose(1, 0, 2)
            coverage[targets] = block[:, :, 20].T
        return values, coverage
