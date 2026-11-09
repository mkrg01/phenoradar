"""Validated float64 TPMs and positive-residue masks in alignment record order."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np

from phenoradar.cache_identity import SourceIndex, digest_key, signature

if TYPE_CHECKING:
    from phenoradar.amino_acid import AlignmentOG


class AlignedTPMCache:
    def __init__(self, directory: Path, source: Path,
                 ogs: Mapping[str, AlignmentOG], *, mask_version: str) -> None:
        self.source, self.ogs = source, ogs
        self.before = signature(source)
        digest = SourceIndex(directory).identify(source)
        axis = hashlib.sha256()
        for og in ogs.values():
            axis.update(json.dumps([og.orthogroup, og.sha256, og.sequence_ids,
                                    og.species_rows, og.mask_rows], sort_keys=True).encode())
        self.identity = {"version": 1, "input_sha256": digest, "axis_sha256": axis.hexdigest(),
                         "mask_version": mask_version, "gzip": source.suffix.lower() == ".gz"}
        self.entry = directory / "aligned" / digest_key(self.identity)
        self.records = sum(len(og.sequence_ids) for og in ogs.values())
        self.mask_bytes = sum(len(og.mask_rows) * ((og.width + 7) // 8) for og in ogs.values())

    @staticmethod
    def _payload(entry: Path) -> dict[str, dict[str, int]]:
        return {name: signature(entry / name) for name in ("weights.npy", "presence.npy")}

    def restore(self) -> dict[str, Any] | None:
        try:
            manifest = json.loads((self.entry / "manifest.json").read_text())
            if (manifest["identity"] != self.identity
                    or manifest["payload"] != self._payload(self.entry)):
                return None
            weights = np.load(self.entry / "weights.npy", mmap_mode="r", allow_pickle=False)
            masks = np.load(self.entry / "presence.npy", mmap_mode="r", allow_pickle=False)
            if (weights.dtype != np.float64 or weights.shape != (self.records,)
                    or masks.dtype != np.uint8 or masks.shape != (self.mask_bytes,)
                    or not np.isfinite(weights).all() or np.any(weights < 0)):
                return None
            self.check_source()
            row = offset = 0
            for og in self.ogs.values():
                og.tpm = weights[row:row + len(og.sequence_ids)]
                row += len(og.sequence_ids)
                og.tpm_mask_cache = str(self.entry / "presence.npy")
                og.tpm_mask_offset = offset
                offset += len(og.mask_rows) * ((og.width + 7) // 8)
            self.check_source()
            return {**manifest["input"], "path": str(self.source.resolve()),
                    "cache_hit": True, "aligned_cache_hit": True,
                    "aligned_cache_path": str(self.entry)}
        except (OSError, ValueError, EOFError, KeyError, TypeError):
            return None

    def check_source(self) -> None:
        if signature(self.source) != self.before:
            raise ValueError(f"Sequence TPM changed during import: {self.source}")

    def save(self, identity: dict[str, Any], mask: Callable[[AlignmentOG], np.ndarray]) -> None:
        parent = self.entry.parent
        parent.mkdir(parents=True, exist_ok=True)
        with (parent / f"{self.entry.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.restore() is not None:
                return
            with TemporaryDirectory(prefix=".building-", dir=parent) as temporary:
                pending = Path(temporary)
                weights = np.lib.format.open_memmap(pending / "weights.npy", mode="w+",
                                                   dtype=np.float64, shape=(self.records,))
                masks = np.lib.format.open_memmap(pending / "presence.npy", mode="w+",
                                                 dtype=np.uint8, shape=(self.mask_bytes,))
                row = offset = 0
                for og in self.ogs.values():
                    weights[row:row + len(og.sequence_ids)] = og.tpm
                    row += len(og.sequence_ids)
                    packed = mask(og).reshape(-1)
                    masks[offset:offset + len(packed)] = packed
                    offset += len(packed)
                weights.flush()
                masks.flush()
                del weights, masks
                self.check_source()
                (pending / "manifest.json").write_text(json.dumps({
                    "identity": self.identity, "payload": self._payload(pending), "input": identity,
                }))
                if self.entry.exists():
                    shutil.rmtree(self.entry)
                os.replace(pending, self.entry)
