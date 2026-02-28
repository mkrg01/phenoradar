"""Helpers for downloading compact test datasets."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

DEFAULT_C4_TINY_BASE_URL = "https://raw.githubusercontent.com/mkrg01/phenoradar/main/testdata/c4_tiny"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class _DatasetFile:
    name: str
    sha256: str


_C4_TINY_FILES = (
    _DatasetFile(
        name="species_metadata.tsv",
        sha256="c0c4732607da374e70f7ec8c7906aceb902c90aff3111fd03559852df1ab9327",
    ),
    _DatasetFile(
        name="tpm.tsv",
        sha256="ff782ac12288e5f7240c7d4ccf3a029ca6d867c8cd1ec9d0d1429d25bb110602",
    ),
)


class TestDataError(ValueError):
    """Raised when test-data download or validation fails."""


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
    except URLError as exc:
        temp_path.unlink(missing_ok=True)
        raise TestDataError(f"Failed to download test data from {url}: {exc}") from exc
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise TestDataError(f"Failed to write downloaded test data: {destination}") from exc
    temp_path.replace(destination)
    return digest.hexdigest()


def _resolve_base_url(base_url: str | None) -> str:
    env_override = os.environ.get("PHENORADAR_TESTDATA_BASE_URL")
    resolved = (base_url or env_override or DEFAULT_C4_TINY_BASE_URL).strip()
    if not resolved:
        raise TestDataError("Test data base URL resolved to empty value")
    return resolved.rstrip("/")


def fetch_c4_tiny_test_data(
    out_dir: Path,
    *,
    base_url: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download compact C4 example data into ``out_dir`` and validate checksums."""
    if out_dir.exists() and not out_dir.is_dir():
        raise TestDataError(f"Output path exists and is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_base_url = _resolve_base_url(base_url)

    written_paths: list[Path] = []
    for dataset_file in _C4_TINY_FILES:
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
        file_url = f"{resolved_base_url}/{quote(dataset_file.name)}"
        downloaded_sha = _download_file(file_url, destination)
        if downloaded_sha != dataset_file.sha256:
            destination.unlink(missing_ok=True)
            raise TestDataError(
                "Downloaded test data checksum mismatch for "
                f"{destination.name}: expected {dataset_file.sha256}, got {downloaded_sha}"
            )
        written_paths.append(destination)
    return written_paths
