from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import phenoradar.testdata as testdata_mod
from phenoradar.testdata import (
    TestDataError as DatasetError,
)
from phenoradar.testdata import (
    fetch_c4_tiny_test_data,
    resolve_c4_tiny_base_url,
)


def test_bundled_manifest_matches_all_packaged_dataset_files() -> None:
    resource_root = testdata_mod._c4_tiny_resource_root()
    manifest = testdata_mod._load_c4_tiny_manifest()

    assert [entry.name for entry in manifest] == [
        "species_metadata.tsv",
        "species_trait.tsv",
        "ncbi_tree.nwk",
        "tpm.tsv",
    ]
    for entry in manifest:
        payload = resource_root.joinpath(entry.name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry.sha256


def test_default_install_uses_bundled_data_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("bundled installation must not access the network")

    monkeypatch.delenv("PHENORADAR_TESTDATA_BASE_URL", raising=False)
    monkeypatch.setattr(testdata_mod, "urlopen", _fail_urlopen)

    written = fetch_c4_tiny_test_data(tmp_path / "dataset")

    assert [path.name for path in written] == [
        "species_metadata.tsv",
        "species_trait.tsv",
        "ncbi_tree.nwk",
        "tpm.tsv",
    ]


def test_base_url_resolution_prefers_explicit_value_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHENORADAR_TESTDATA_BASE_URL", "https://environment.example/data/")

    assert resolve_c4_tiny_base_url(None) == "https://environment.example/data"
    assert resolve_c4_tiny_base_url("https://explicit.example/data/") == (
        "https://explicit.example/data"
    )


def test_base_url_resolution_rejects_empty_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHENORADAR_TESTDATA_BASE_URL", "   ")

    with pytest.raises(DatasetError, match="resolved to empty"):
        resolve_c4_tiny_base_url(None)
