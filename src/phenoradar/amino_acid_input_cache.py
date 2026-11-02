"""Reusable alignment bytes and masks, independent of traits and CV folds."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phenoradar.amino_acid import AlignmentOG


def _signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "name": path.name, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def _payload(entry: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ["residues.bin", "presence.npy"]:
        signature = _signature(entry / name)
        signature.pop("path")
        result[name] = signature
    return result


class AlignmentInputCache:
    """Publish complete OG entries atomically; changes invalidate cached validation."""

    def __init__(self, directory: Path, species: set[str], retained: set[str]) -> None:
        context = json.dumps({"version": 1, "species": sorted(species),
                              "retained": sorted(retained)}, sort_keys=True).encode()
        self.directory = directory / hashlib.sha256(context).hexdigest()
        self.hits = 0

    @staticmethod
    def _restore(entry: Path, source: dict[str, Any]) -> AlignmentOG | None:
        from phenoradar.amino_acid import AlignmentOG

        try:
            with gzip.open(entry / "metadata.json.gz", "rt") as handle:
                data = json.load(handle)
            if data["source"] != source or data["payload"] != _payload(entry):
                return None
            row = data["alignment"]
            row["cache"] = str(entry / "residues.bin")
            row["mask_cache"] = str(entry / "presence.npy")
            # JSON object keys are species IDs; row indices retain their exact order.
            return AlignmentOG(**row)
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            return None

    def read(self, path: Path, build: Callable[[], AlignmentOG]) -> AlignmentOG:
        source = _signature(path)
        key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        parent = self.directory / key[:2]
        entry = parent / key
        restored = self._restore(entry, source)
        if restored is not None:
            self.hits += 1
            return restored
        parent.mkdir(parents=True, exist_ok=True)
        with (parent / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            restored = self._restore(entry, source)
            if restored is not None:
                self.hits += 1
                return restored
            alignment = build()
            if _signature(path) != source:
                raise ValueError(f"Alignment changed during cache creation: {path}")
            with TemporaryDirectory(prefix=".building-", dir=parent) as temporary:
                pending = Path(temporary)
                shutil.move(alignment.cache, pending / "residues.bin")
                shutil.move(alignment.mask_cache, pending / "presence.npy")
                row = asdict(alignment)
                row["cache"], row["mask_cache"] = "residues.bin", "presence.npy"
                with gzip.open(pending / "metadata.json.gz", "wt", compresslevel=1) as handle:
                    json.dump({"source": source, "payload": _payload(pending),
                               "alignment": row}, handle, separators=(",", ":"))
                if entry.exists():
                    shutil.rmtree(entry)
                os.replace(pending, entry)
            alignment.cache = str(entry / "residues.bin")
            alignment.mask_cache = str(entry / "presence.npy")
            return alignment
