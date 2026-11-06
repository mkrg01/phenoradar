"""Reusable alignment bytes and masks, independent of traits and CV folds."""

from __future__ import annotations

import fcntl
import gzip
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from phenoradar.cache_identity import SourceIndex, digest_key, signature

if TYPE_CHECKING:
    from phenoradar.amino_acid import AlignmentOG


def _payload(entry: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ["residues.bin", "presence.npy"]:
        result[name] = signature(entry / name)
    return result


class AlignmentInputCache:
    """Publish complete OG entries atomically; changes invalidate cached validation."""

    def __init__(self, directory: Path, species: set[str], retained: set[str], *,
                 parser_version: str) -> None:
        context = {"version": 2, "parser": parser_version,
                   "species": sorted(species), "retained": sorted(retained)}
        self.sources = SourceIndex(directory)
        self.directory = directory / "v2" / digest_key(context)
        self.hits = 0

    @staticmethod
    def _restore(entry: Path, digest: str) -> AlignmentOG | None:
        from phenoradar.amino_acid import AlignmentOG

        try:
            with gzip.open(entry / "metadata.json.gz", "rt") as handle:
                data = json.load(handle)
            if data["sha256"] != digest or data["payload"] != _payload(entry):
                return None
            row = data["alignment"]
            row["cache"] = str(entry / "residues.bin")
            row["mask_cache"] = str(entry / "presence.npy")
            # JSON object keys are species IDs; row indices retain their exact order.
            return AlignmentOG(**row)
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            return None

    def read(self, path: Path, build: Callable[[], AlignmentOG]) -> AlignmentOG:
        before = signature(path)
        digest = self.sources.lookup(path)
        alignment = None
        if digest is None:
            if self.sources.hint(path):
                digest = self.sources.identify(path)
            else:
                alignment = build()
                digest = self.sources.register(path, alignment.sha256, before)
        key = digest
        parent = self.directory / key[:2]
        entry = parent / key
        restored = self._restore(entry, digest)
        if restored is not None:
            self.hits += 1
            restored.source, restored.orthogroup = str(path.resolve()), path.stem
            if signature(path) != before:
                raise ValueError(f"Alignment changed during cache lookup: {path}")
            return restored
        parent.mkdir(parents=True, exist_ok=True)
        with (parent / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            restored = self._restore(entry, digest)
            if restored is not None:
                self.hits += 1
                restored.source, restored.orthogroup = str(path.resolve()), path.stem
                if signature(path) != before:
                    raise ValueError(f"Alignment changed during cache lookup: {path}")
                return restored
            alignment = build() if alignment is None else alignment
            if signature(path) != before:
                raise ValueError(f"Alignment changed during cache creation: {path}")
            with TemporaryDirectory(prefix=".building-", dir=parent) as temporary:
                pending = Path(temporary)
                shutil.move(alignment.cache, pending / "residues.bin")
                shutil.move(alignment.mask_cache, pending / "presence.npy")
                row = asdict(alignment)
                row["cache"], row["mask_cache"] = "residues.bin", "presence.npy"
                with gzip.open(pending / "metadata.json.gz", "wt", compresslevel=1) as handle:
                    json.dump({"sha256": digest, "payload": _payload(pending),
                               "alignment": row}, handle, separators=(",", ":"))
                if entry.exists():
                    shutil.rmtree(entry)
                os.replace(pending, entry)
            alignment.cache = str(entry / "residues.bin")
            alignment.mask_cache = str(entry / "presence.npy")
            return alignment
