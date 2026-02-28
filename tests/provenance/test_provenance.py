from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import phenoradar.provenance as provenance_mod
from phenoradar.provenance import (
    ProvenanceError,
    bundle_payload_sha256,
    collect_input_files,
    git_snapshot,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_bundle_payload_sha256_is_deterministic_and_ignores_manifest(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write(bundle_dir / "b.txt", "bbb\n")
    _write(bundle_dir / "a.txt", "aaa\n")
    _write(bundle_dir / "bundle_manifest.json", '{"ignored":"yes"}\n')

    digest_first = bundle_payload_sha256(bundle_dir)
    _write(bundle_dir / "bundle_manifest.json", '{"ignored":"changed"}\n')
    digest_second = bundle_payload_sha256(bundle_dir)
    assert digest_first == digest_second

    _write(bundle_dir / "a.txt", "aaa-modified\n")
    digest_third = bundle_payload_sha256(bundle_dir)
    assert digest_third != digest_first


def test_collect_input_files_deduplicates_and_sorts_paths(tmp_path: Path) -> None:
    file_b = _write(tmp_path / "b.txt", "b\n")
    file_a = _write(tmp_path / "a.txt", "a\n")

    payload = collect_input_files([file_b, file_a, file_b])

    assert len(payload) == 2
    assert payload[0]["path"] == str(file_a.resolve())
    assert payload[1]["path"] == str(file_b.resolve())


def test_bundle_payload_sha256_requires_existing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing_bundle"

    with pytest.raises(ProvenanceError, match="Model bundle directory not found"):
        bundle_payload_sha256(missing)


def test_collect_input_files_raises_when_path_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    with pytest.raises(ProvenanceError, match="Input file not found for provenance"):
        collect_input_files([missing])


def test_run_git_returns_none_on_file_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise_file_not_found(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(provenance_mod.subprocess, "run", _raise_file_not_found)

    assert provenance_mod._run_git(["git", "status"], cwd=tmp_path) is None


def test_run_git_returns_none_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        provenance_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )

    assert provenance_mod._run_git(["git", "status"], cwd=tmp_path) is None


def test_run_git_returns_stripped_stdout_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        provenance_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="abc123\n"),
    )

    assert provenance_mod._run_git(["git", "rev-parse", "HEAD"], cwd=tmp_path) == "abc123"


def test_bundle_payload_sha256_ignores_subdirectories(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write(bundle_dir / "a.txt", "aaa\n")
    subdir = bundle_dir / "nested"
    subdir.mkdir()
    _write(subdir / "inner.txt", "bbb\n")

    digest_first = bundle_payload_sha256(bundle_dir)
    _write(subdir / "inner.txt", "changed\n")
    digest_second = bundle_payload_sha256(bundle_dir)

    assert digest_first == digest_second


def test_git_snapshot_uses_unknown_when_git_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(provenance_mod, "_run_git", lambda *_args, **_kwargs: None)

    snapshot = git_snapshot(tmp_path)

    assert snapshot["git_commit"] == "unknown"
    assert snapshot["git_dirty"] is False
