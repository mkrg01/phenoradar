"""Helpers for installing compact bundled test datasets."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

BUNDLED_C4_TINY_SOURCE = "bundled package data"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_C4_TINY_MANIFEST_NAME = "SHA256SUMS"
_C4_TINY_FILE_NAMES = (
    "species_metadata.tsv",
    "species_trait.tsv",
    "ncbi_tree.nwk",
    "tpm.tsv",
)


@dataclass(frozen=True)
class _DatasetFile:
    name: str
    sha256: str


class TestDataError(ValueError):
    """Raised when test-data installation or validation fails."""


def _c4_tiny_resource_root() -> Traversable:
    return files("phenoradar").joinpath("data").joinpath("c4_tiny")


def _load_c4_tiny_manifest() -> tuple[_DatasetFile, ...]:
    manifest_resource = _c4_tiny_resource_root().joinpath(_C4_TINY_MANIFEST_NAME)
    try:
        manifest_text = manifest_resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise TestDataError(
            f"Bundled test-data manifest is unavailable: {_C4_TINY_MANIFEST_NAME}"
        ) from exc

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(manifest_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise TestDataError(
                f"Invalid bundled test-data manifest line {line_number}: {raw_line!r}"
            )
        sha256, name = parts
        if len(sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in sha256):
            raise TestDataError(
                f"Invalid SHA-256 in bundled test-data manifest line {line_number}"
            )
        if Path(name).name != name or name not in _C4_TINY_FILE_NAMES:
            raise TestDataError(
                f"Unexpected file in bundled test-data manifest line {line_number}: {name}"
            )
        if name in entries:
            raise TestDataError(f"Duplicate file in bundled test-data manifest: {name}")
        entries[name] = sha256.lower()

    missing = [name for name in _C4_TINY_FILE_NAMES if name not in entries]
    if missing:
        raise TestDataError(
            "Bundled test-data manifest is missing required files: " + ", ".join(missing)
        )
    return tuple(_DatasetFile(name=name, sha256=entries[name]) for name in _C4_TINY_FILE_NAMES)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    temp_path = destination.with_name(f".{destination.name}.tmp")
    try:
        with urlopen(url, timeout=60) as response, temp_path.open("wb") as out_handle:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                out_handle.write(chunk)
        temp_path.replace(destination)
    except (URLError, ValueError) as exc:
        temp_path.unlink(missing_ok=True)
        raise TestDataError(f"Failed to download test data from {url}: {exc}") from exc
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise TestDataError(f"Failed to write downloaded test data: {destination}") from exc
    return digest.hexdigest()


def _copy_bundled_file(source: Traversable, destination: Path) -> str:
    digest = hashlib.sha256()
    temp_path = destination.with_name(f".{destination.name}.tmp")
    try:
        with source.open("rb") as source_handle, temp_path.open("wb") as out_handle:
            while True:
                chunk = source_handle.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                out_handle.write(chunk)
        temp_path.replace(destination)
    except (FileNotFoundError, OSError) as exc:
        temp_path.unlink(missing_ok=True)
        raise TestDataError(f"Failed to copy bundled test data: {source.name}") from exc
    return digest.hexdigest()


def resolve_c4_tiny_base_url(base_url: str | None) -> str | None:
    """Resolve an optional external source; ``None`` selects bundled data."""
    raw_source = base_url if base_url is not None else os.environ.get(
        "PHENORADAR_TESTDATA_BASE_URL"
    )
    if raw_source is None:
        return None
    resolved = raw_source.strip()
    if not resolved:
        raise TestDataError("Test data base URL resolved to empty value")
    return resolved.rstrip("/")


def fetch_c4_tiny_test_data(
    out_dir: Path,
    *,
    base_url: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Install compact C4 example data into ``out_dir`` and validate checksums."""
    if out_dir.exists() and not out_dir.is_dir():
        raise TestDataError(f"Output path exists and is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_base_url = resolve_c4_tiny_base_url(base_url)
    dataset_files = _load_c4_tiny_manifest()
    resource_root = _c4_tiny_resource_root()

    written_paths: list[Path] = []
    for dataset_file in dataset_files:
        destination = out_dir / dataset_file.name
        if destination.exists():
            actual_existing_sha = _sha256_file(destination)
            if actual_existing_sha == dataset_file.sha256:
                written_paths.append(destination)
                continue
            if not overwrite:
                raise TestDataError(
                    "Existing file has unexpected checksum; "
                    f"use --force to overwrite: {destination}"
                )
        if resolved_base_url is None:
            actual_sha = _copy_bundled_file(
                resource_root.joinpath(dataset_file.name),
                destination,
            )
        else:
            file_url = f"{resolved_base_url}/{quote(dataset_file.name)}"
            actual_sha = _download_file(file_url, destination)
        if actual_sha != dataset_file.sha256:
            destination.unlink(missing_ok=True)
            raise TestDataError(
                "Test data checksum mismatch for "
                f"{destination.name}: expected {dataset_file.sha256}, got {actual_sha}"
            )
        written_paths.append(destination)
    return written_paths
