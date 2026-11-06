"""Fast file identities with content-addressed, location-independent payloads."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "device": stat.st_dev, "inode": stat.st_ino}


def digest_key(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, separators=(",", ":"), sort_keys=True)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SourceIndex:
    """Stat hits avoid reads; changed identities require an actual SHA-256 check.

    A basename/size hint only chooses hashing before parsing. It never establishes
    content equality. Cold FASTA imports can register their existing streaming hash.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory / "sources"

    def _path(self, source: Mapping[str, int]) -> Path:
        key = digest_key(source)
        return self.directory / key[:2] / f"{key}.json"

    def lookup(self, path: Path) -> str | None:
        source = signature(path)
        try:
            row = json.loads(self._path(source).read_text())
            digest = row["sha256"]
            if (row["signature"] == source and isinstance(digest, str)
                    and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)):
                return digest
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def hint(self, path: Path) -> bool:
        return (self.directory / "hints" / digest_key([path.name, path.stat().st_size])).exists()

    def register(self, path: Path, digest: str, before: dict[str, int]) -> str:
        if signature(path) != before:
            raise ValueError(f"Input changed during cache creation: {path}")
        atomic_json(self._path(before), {"signature": before, "sha256": digest})
        hint = self.directory / "hints" / digest_key([path.name, before["size"]])
        hint.parent.mkdir(parents=True, exist_ok=True)
        hint.touch(exist_ok=True)
        return digest

    def identify(self, path: Path) -> str:
        cached = self.lookup(path)
        if cached is not None:
            return cached
        before = signature(path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        return self.register(path, digest, before)
