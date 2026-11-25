"""Keep cached alignment strings out of the grid-worker transport."""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    from phenoradar.amino_acid import AlignmentOG

_MIN_CACHED_RECORDS = 250_000
_Field = Literal["sequence_ids", "sequence_hashes"]


@lru_cache(maxsize=8)
def _read_strings(path: Path, digest: str, rows: int) -> dict[str, tuple[str, ...]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        metadata = json.load(handle)
    alignment = metadata["alignment"]
    fields = ("sequence_ids", "sequence_hashes")
    if (metadata["sha256"] != digest or alignment["sha256"] != digest
            or any(len(alignment[field]) != rows for field in fields)):
        raise ValueError(f"Alignment metadata changed after worker snapshot: {path}")
    return {field: tuple(alignment[field]) for field in fields}


@dataclass(frozen=True)
class CachedStrings(Sequence[str]):
    path: Path
    digest: str
    rows: int
    field: _Field

    def __len__(self) -> int:
        return self.rows

    def _strings(self) -> tuple[str, ...]:
        return _read_strings(self.path, self.digest, self.rows)[self.field]

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: int | slice) -> str | Sequence[str]:
        return self._strings()[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self._strings())


def _cache_metadata(og: AlignmentOG) -> Path | None:
    path = Path(og.cache)
    if path.name == "residues.bin" and path.parent.name == og.sha256:
        # AlignmentInputCache has already validated or published this entry.
        return path.with_name("metadata.json.gz")
    return None


def compact_alignment_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    store = payload["store"]
    ogs: dict[str, AlignmentOG] = store["ogs"]
    if sum(len(og.sequence_ids) for og in ogs.values()) < _MIN_CACHED_RECORDS:
        return payload
    compact = {}
    for key, og in ogs.items():
        path = _cache_metadata(og)
        if path is not None:
            rows = len(og.sequence_ids)
            og = replace(og,
                         sequence_ids=CachedStrings(path, og.sha256, rows, "sequence_ids"),
                         sequence_hashes=CachedStrings(path, og.sha256, rows, "sequence_hashes"))
        compact[key] = og
    return {**payload, "store": {**store, "ogs": compact}}
