"""Run provenance and reproducibility metadata helpers."""

from __future__ import annotations

import platform
import subprocess
from hashlib import sha256
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any


class ProvenanceError(ValueError):
    """Raised when provenance metadata cannot be collected."""


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    """Build deterministic file identity record."""
    if not path.exists():
        raise ProvenanceError(f"Input file not found for provenance: {path}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def runtime_environment_snapshot() -> dict[str, Any]:
    """Capture deterministic runtime environment metadata."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": {
            "polars": package_version("polars"),
            "scikit-learn": package_version("scikit-learn"),
            "pydantic": package_version("pydantic"),
            "typer": package_version("typer"),
        },
    }


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_snapshot(cwd: Path) -> dict[str, Any]:
    """Capture git commit/dirty/patch checksum snapshot for a working tree."""
    commit = _run_git(["git", "rev-parse", "HEAD"], cwd=cwd) or "unknown"
    status = _run_git(["git", "status", "--porcelain"], cwd=cwd)
    dirty = bool(status) if status is not None else False
    patch = _run_git(["git", "diff", "HEAD"], cwd=cwd)
    patch_sha = sha256((patch or "").encode("utf-8")).hexdigest()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_worktree_patch_sha256": patch_sha,
    }


def bundle_payload_sha256(bundle_dir: Path) -> str:
    """Compute deterministic digest over bundle payload files (excluding manifest)."""
    if not bundle_dir.exists():
        raise ProvenanceError(f"Model bundle directory not found: {bundle_dir}")
    file_hashes: list[str] = []
    for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if path.name == "bundle_manifest.json":
            continue
        file_hashes.append(f"{path.name}:{sha256_file(path)}")
    digest = sha256()
    digest.update("\n".join(file_hashes).encode("utf-8"))
    return digest.hexdigest()


def collect_input_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Collect sorted file identity records for metadata."""
    unique_paths = sorted({path.resolve() for path in paths}, key=lambda item: str(item))
    return [file_identity(path) for path in unique_paths]
