"""Bounded, run-local reuse of immutable raw expression matrices."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import numpy as np


@dataclass(frozen=True)
class _RawMatrix:
    species: tuple[str, ...]
    features: tuple[str, ...]
    values: np.ndarray
    complete_schema: bool


class RawExpressionMatrixCache:
    """Reuse raw values only; never share feature selection or fitted state."""

    def __init__(self, byte_limit: int) -> None:
        self.byte_limit = max(0, byte_limit)
        self._entries: OrderedDict[tuple[tuple[str, ...], tuple[str, ...], bool], _RawMatrix] = (
            OrderedDict()
        )
        self._bytes = 0
        self._lock = RLock()

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def put(
        self,
        species: list[str],
        features: list[str],
        matrix: np.ndarray,
        *,
        complete_schema: bool,
    ) -> None:
        if matrix.nbytes > self.byte_limit or not self.byte_limit:
            return
        key = (tuple(species), tuple(features), complete_schema)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.values.nbytes
            while self._entries and (
                self._bytes + matrix.nbytes > self.byte_limit or len(self._entries) >= 8
            ):
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed.values.nbytes
            matrix.setflags(write=False)
            self._entries[key] = _RawMatrix(key[0], key[1], matrix, complete_schema)
            self._bytes += matrix.nbytes

    @staticmethod
    def _take(
        entry: _RawMatrix,
        species: list[str],
        features: list[str],
    ) -> np.ndarray:
        if tuple(species) == entry.species and tuple(features) == entry.features:
            return entry.values
        rows = {name: i for i, name in enumerate(entry.species)}
        columns = {name: i for i, name in enumerate(entry.features)}
        selected: np.ndarray = entry.values[
            np.ix_(
                [rows[name] for name in species],
                [columns[name] for name in features],
            )
        ]
        selected.setflags(write=False)
        return selected

    def get(
        self,
        species: list[str],
        features: list[str] | None,
    ) -> tuple[np.ndarray, list[str]] | None:
        requested = set(species)
        with self._lock:
            for key, entry in reversed(self._entries.items()):
                if not requested.issubset(entry.species):
                    continue
                if features is None:
                    if not entry.complete_schema or requested != set(entry.species):
                        continue
                    names = list(entry.features)
                else:
                    if not set(features).issubset(entry.features):
                        continue
                    names = features
                self._entries.move_to_end(key)
                return self._take(entry, species, names), list(names)
        return None

    def overlapping_rows(
        self,
        species: list[str],
        features: list[str],
    ) -> tuple[list[int], np.ndarray] | None:
        """Return rows available for every requested column in one cached matrix."""
        requested_features = set(features)
        with self._lock:
            best: (
                tuple[tuple[tuple[str, ...], tuple[str, ...], bool], _RawMatrix, list[int]] | None
            ) = None
            for key, entry in reversed(self._entries.items()):
                if not requested_features.issubset(entry.features):
                    continue
                available = set(entry.species)
                rows = [i for i, name in enumerate(species) if name in available]
                if rows and (best is None or len(rows) > len(best[2])):
                    best = (key, entry, rows)
            if best is None:
                return None
            key, entry, rows = best
            self._entries.move_to_end(key)
            return rows, self._take(entry, [species[i] for i in rows], features)
