from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from polars.testing import assert_frame_equal

from phenoradar import asr_cache
from phenoradar import phylogenetic_imputation as imputation
from phenoradar.config.schema import PhylogeneticImputationConfig


@pytest.fixture
def identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    script = tmp_path / "nwkit"
    script.write_text(f"#!{sys.executable}\n# entry point\n")
    package = tmp_path / "package"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("# nwkit implementation\n")
    monkeypatch.setattr(
        asr_cache.importlib.util, "find_spec",
        lambda _: importlib.util.spec_from_file_location("nwkit", source),
    )
    directory = tmp_path / "input"
    directory.mkdir()
    (directory / "tree.nwk").write_text("(a:1,b:1);")
    (directory / "traits.tsv").write_text("leaf_name\tstate\na\t1\n")
    return script, source, directory


def key(script: Path, directory: Path, model: str = "ER") -> str | None:
    return asr_cache.cache_key(
        str(script), "nwkit 0.43.21",
        [str(script), "asr", "--infile", str(directory / "tree.nwk"), "--model", model],
        directory,
    )


@pytest.mark.parametrize(
    "changed", ["tree.nwk", "traits.tsv", "source", "entry", "model", "env", "dependency"],
)
def test_fingerprint_invalidates_changed_analysis(
    identity: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    script, source, directory = identity
    expected = key(script, directory)
    assert expected is not None
    model = "ER"
    if changed in {"tree.nwk", "traits.tsv"}:
        path = directory / changed
        path.write_text(path.read_text() + "\n")
    elif changed == "source":
        source.write_text("# edited without a version bump\n")
    elif changed == "entry":
        script.write_text(script.read_text() + "# changed entry\n")
    elif changed == "model":
        model = "ARD"
    elif changed == "env":
        monkeypatch.setenv("OMP_NUM_THREADS", "139")
    else:
        monkeypatch.setattr(
            asr_cache.importlib.metadata, "distributions",
            lambda: [SimpleNamespace(metadata={"Name": "numpy"}, version="changed")],
        )
    assert key(script, directory, model) != expected


def test_fingerprint_is_independent_of_run_directory(identity: tuple[Path, Path, Path]) -> None:
    script, _, directory = identity
    other = directory.parent / "other-run"
    other.mkdir()
    for name in ("tree.nwk", "traits.tsv"):
        (other / name).write_bytes((directory / name).read_bytes())
    assert key(script, directory) == key(script, other)


def test_unknown_wrapper_disables_cache(identity: tuple[Path, Path, Path]) -> None:
    script, _, directory = identity
    script.write_text("#!/usr/bin/env python\n")
    assert key(script, directory) is None


def test_venv_directory_alias_shares_cache(
    identity: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    script, _, directory = identity
    python = Path(sys.executable)
    alias = tmp_path / "venv_alias"
    alias.symlink_to(python.parent, target_is_directory=True)
    expected = key(script, directory)
    assert expected is not None
    monkeypatch.setattr(sys, "executable", str(alias / python.name))
    assert key(script, directory) == expected

    other_venv = tmp_path / "other_venv"
    other_venv.mkdir()
    (other_venv / python.name).symlink_to(python)
    monkeypatch.setattr(sys, "executable", str(other_venv / python.name))
    assert key(script, directory) is None


def test_archive_roundtrip_and_corruption(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for name in asr_cache.OUTPUT_NAMES:
        (source / name).write_text(f"content of {name}\n")
    archive = tmp_path / "cache" / "key.zip"
    command = ["nwkit", "asr"]
    asr_cache.store(archive, "key", source, command)
    restored = asr_cache.restore(archive, "key", target)
    assert restored is not None and restored["command"] == command
    for name in asr_cache.OUTPUT_NAMES:
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert asr_cache.restore(archive, "different-key", target) is None
    with ZipFile(archive) as zipped:
        members = {name: zipped.read(name) for name in zipped.namelist()}
    # A readable archive with altered outputs must also be rejected.
    members["asr.tsv"] = b"corrupted output"
    with ZipFile(archive, "w") as zipped:
        for name, value in members.items():
            zipped.writestr(name, value)
    assert asr_cache.restore(archive, "key", target) is None
    archive.write_bytes(b"truncated zip")
    assert asr_cache.restore(archive, "key", target) is None


def test_unwritable_cache_is_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in asr_cache.OUTPUT_NAMES:
        (tmp_path / name).write_text("data")
    def fail(*args: object) -> None:
        raise PermissionError("read-only cache")
    monkeypatch.setattr(asr_cache.os, "replace", fail)
    asr_cache.store(tmp_path / "cache" / "key.zip", "key", tmp_path, [])
    assert list((tmp_path / "cache").iterdir()) == []


def test_asr_reuses_validated_outputs_and_preserves_provenance(
    identity: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, _, directory = identity
    monkeypatch.setattr(imputation, "_nwkit_executable", lambda: str(script))
    calls: list[list[str]] = []
    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[-1] == "--version":
            return subprocess.CompletedProcess(arguments, 0, "nwkit 0.43.21\n", "")
        calls.append(arguments)
        output = Path(arguments[arguments.index("--outfile") + 1])
        output.write_text("branch_id\tparent\tnode_class\tname\tp_0\tp_1\n0\t-1\troot\tr\t0.2\t0.8\n")
        (output.parent / "model.tsv").write_text("model\nER\n")
        (output.parent / "annotated_tree.nhx").write_text("(a,b);")
        return subprocess.CompletedProcess(arguments, 0, "output", "warning: example")
    monkeypatch.setattr(imputation.subprocess, "run", run)
    settings = PhylogeneticImputationConfig(enabled=True)
    cache_root = tmp_path / "cache"
    nodes, provenance, warnings = imputation._run_asr(settings, directory, cache_root=cache_root)
    warm, reused, warm_warnings = imputation._run_asr(settings, directory, cache_root=cache_root)
    assert len(calls) == 1
    assert_frame_equal(nodes, warm, check_exact=True)
    assert warnings == warm_warnings == ["warning: example"]
    assert provenance["asr_cache"]["hit"] is False
    assert reused["asr_cache"]["hit"] is True
    assert reused["command"] == calls[0]
    # A damaged cache is replaced by a fresh successful execution.
    archive = Path(reused["asr_cache"]["path"])
    archive.write_bytes(b"broken")
    imputation._run_asr(settings, directory, cache_root=cache_root)
    assert len(calls) == 2
    imputation._run_asr(settings, directory, cache_root=cache_root)
    assert len(calls) == 2
    # Changed fitted model never reuses the first result.
    settings.model = "ARD"
    imputation._run_asr(settings, directory, cache_root=cache_root)
    assert len(calls) == 3


@pytest.mark.parametrize("invalid_output", [False, True])
def test_failed_or_invalid_asr_is_never_cached(
    identity: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    invalid_output: bool,
) -> None:
    script, _, directory = identity
    monkeypatch.setattr(imputation, "_nwkit_executable", lambda: str(script))
    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[-1] == "--version":
            return subprocess.CompletedProcess(arguments, 0, "nwkit version\n", "")
        if invalid_output:
            (directory / "asr.tsv").write_text("invalid\n1\n")
        return subprocess.CompletedProcess(arguments, 0 if invalid_output else 1, "", "failed")
    monkeypatch.setattr(imputation.subprocess, "run", run)
    cache_root = tmp_path / "cache"
    settings = PhylogeneticImputationConfig(enabled=True)
    if invalid_output:
        with pytest.raises(imputation.PhylogeneticImputationError, match="Invalid nwkit"):
            imputation._run_asr(settings, directory, cache_root=cache_root)
    else:
        nodes, _, warnings = imputation._run_asr(settings, directory, cache_root=cache_root)
        assert nodes.is_empty() and warnings
    assert not cache_root.exists()
