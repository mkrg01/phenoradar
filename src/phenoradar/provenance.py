"""Run provenance and reproducibility metadata helpers."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from hashlib import sha256
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import polars as pl

FINGERPRINT_SCHEMA_VERSION = 1
_SPLIT_FINGERPRINT_COLUMNS = (
    "species",
    "pool",
    "fold_id",
    "group_id",
    "contrast_group_id",
    "label",
)


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("Fingerprint payload is not canonically JSON serializable") from exc
    return sha256(serialized.encode("utf-8")).hexdigest()


def input_hashes_by_role(
    input_files: Sequence[Mapping[str, Any]],
    role_paths: Mapping[str, Path],
) -> dict[str, str]:
    """Resolve content hashes from provenance records without hashing files again."""
    hashes_by_path: dict[Path, str] = {}
    for record in input_files:
        raw_path = record.get("path")
        raw_sha256 = record.get("sha256")
        if not isinstance(raw_path, str) or not _is_sha256(raw_sha256):
            raise ProvenanceError("Input file provenance record has invalid path or SHA-256")
        hashes_by_path[Path(raw_path).resolve()] = str(raw_sha256)

    resolved: dict[str, str] = {}
    for role, path in sorted(role_paths.items()):
        digest = hashes_by_path.get(path.resolve())
        if digest is None:
            raise ProvenanceError(f"Missing input file provenance for dataset role: {role}")
        resolved[role] = digest
    return resolved


def dataset_fingerprint(file_sha256_by_role: Mapping[str, str]) -> str:
    """Fingerprint dataset file contents by semantic role, independent of file paths."""
    required_roles = {"metadata", "tpm"}
    missing_roles = sorted(required_roles - set(file_sha256_by_role))
    if missing_roles:
        raise ProvenanceError(
            "Dataset fingerprint is missing required role(s): " + ", ".join(missing_roles)
        )
    invalid_roles = sorted(
        role for role, digest in file_sha256_by_role.items() if not _is_sha256(digest)
    )
    if invalid_roles:
        raise ProvenanceError(
            "Dataset fingerprint has invalid SHA-256 for role(s): " + ", ".join(invalid_roles)
        )
    return _canonical_payload_sha256(
        {
            "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            "files": dict(sorted(file_sha256_by_role.items())),
        }
    )


def split_fingerprint(split_manifest: pl.DataFrame) -> str:
    """Fingerprint the realized species pools, folds, groups, and labels."""
    missing_columns = sorted(set(_SPLIT_FINGERPRINT_COLUMNS) - set(split_manifest.columns))
    if missing_columns:
        raise ProvenanceError(
            "Split fingerprint manifest is missing column(s): " + ", ".join(missing_columns)
        )
    rows = split_manifest.select(_SPLIT_FINGERPRINT_COLUMNS).to_dicts()
    canonical_rows = sorted(
        rows,
        key=lambda row: json.dumps(
            row,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    return _canonical_payload_sha256(
        {
            "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            "columns": list(_SPLIT_FINGERPRINT_COLUMNS),
            "rows": canonical_rows,
        }
    )


def experiment_fingerprint(
    *,
    dataset_sha256: str,
    split_sha256: str,
    evaluation_contract: Mapping[str, Any],
) -> str:
    """Fingerprint the data, realized split, and stable evaluation contract."""
    if not _is_sha256(dataset_sha256):
        raise ProvenanceError("Experiment fingerprint received invalid dataset SHA-256")
    if not _is_sha256(split_sha256):
        raise ProvenanceError("Experiment fingerprint received invalid split SHA-256")
    return _canonical_payload_sha256(
        {
            "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            "dataset_fingerprint": dataset_sha256,
            "split_fingerprint": split_sha256,
            "evaluation_contract": dict(evaluation_contract),
        }
    )
