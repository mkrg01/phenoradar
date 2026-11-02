"""Alignment cache reuse must retain validation, identity, and exact byte axes."""

from pathlib import Path

import numpy as np
import pytest
from test_amino_acid import alignment

from phenoradar.amino_acid import AlignmentError, AlignmentStore


def test_persistent_input_cache_reuses_exact_bytes_and_survives_store_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = alignment(tmp_path / "aa", [("s1_a", "A-X"), ("s2_a", "VVA"), ("s1_b", "VAA")])
    cache = tmp_path / "cache"
    original = AlignmentStore._read
    calls = []

    def tracked(self, path, index):
        calls.append(path)
        return original(self, path, index)

    monkeypatch.setattr(AlignmentStore, "_read", tracked)
    for _ in range(2):
        with AlignmentStore(path, ["s1", "s2"], cache_species=["s1"],
                            input_cache_dir=cache) as store:
            og = store.ogs["OG1"]
            assert og.sequence_ids == ["s1_a", "s1_b"]
            assert og.species_rows == {"s1": [0, 1]}
            assert list(store.correspondence) == [
                {"orthogroup": "OG1", "sequence_id": name, "species": "s1"}
                for name in ["s1_a", "s1_b"]
            ]
            np.testing.assert_array_equal(og.matrix(), np.array([list(b"A-X"), list(b"VAA")]))
            np.testing.assert_array_equal(store.call_counts(og, ["s1"]), [1, 1, 1])
            persistent = Path(og.cache)
            identities = store.identities()
        assert persistent.exists()
    assert len(calls) == 1
    with AlignmentStore(path, ["s1", "s2"], cache_species=["s1"]) as plain:
        assert plain.identities() == identities
        assert plain.ogs["OG1"].sequence_hashes == og.sequence_hashes


@pytest.mark.parametrize("changed", ["source", "bytes", "masks", "metadata", "retained"])
def test_changed_inputs_and_incomplete_payloads_are_rebuilt(tmp_path: Path, changed: str) -> None:
    path = alignment(tmp_path / "aa", [("s1_a", "AVA"), ("s2_a", "VVA")])
    cache = tmp_path / "cache"
    with AlignmentStore(path, ["s1", "s2"], input_cache_dir=cache) as store:
        og = store.ogs["OG1"]
        raw, masks = Path(og.cache), Path(og.mask_cache)
    if changed == "source":
        (path / "OG1.faa").write_text(">s1_a\nVVV\n>s2_a\nVVA\n")
    elif changed == "bytes":
        raw.write_bytes(b"AAA" * 2)
    elif changed == "masks":
        masks.unlink()
    elif changed == "metadata":
        (raw.parent / "metadata.json.gz").write_bytes(b"interrupted")
    retained = ["s1"] if changed == "retained" else ["s1", "s2"]
    with (
        AlignmentStore(path, ["s1", "s2"], cache_species=retained,
                       input_cache_dir=cache) as cached,
        AlignmentStore(path, ["s1", "s2"], cache_species=retained) as plain,
    ):
        assert cached.identities() == plain.identities()
        np.testing.assert_array_equal(cached.ogs["OG1"].matrix(), plain.ogs["OG1"].matrix())
        np.testing.assert_array_equal(cached.call_counts(cached.ogs["OG1"], retained),
                                      plain.call_counts(plain.ogs["OG1"], retained))


def test_changed_id_resolution_does_not_bypass_ambiguous_id_validation(tmp_path: Path) -> None:
    path = alignment(tmp_path / "aa", [("s1_a", "AVA"), ("s2_a", "VVA")])
    cache = tmp_path / "cache"
    with AlignmentStore(path, ["s1", "s2"], input_cache_dir=cache):
        pass
    with pytest.raises(AlignmentError, match="Cannot uniquely map"):
        AlignmentStore(path, ["s1", "s1_a", "s2"], input_cache_dir=cache)


def test_renamed_symlink_keeps_its_orthogroup_identity(tmp_path: Path) -> None:
    path = alignment(tmp_path / "aa", [("s1_a", "AVA"), ("s2_a", "VVA")])
    (path / "OG2.faa").symlink_to(path / "OG1.faa")
    for _ in range(2):
        with AlignmentStore(path, ["s1", "s2"], input_cache_dir=tmp_path / "cache") as store:
            assert [og.orthogroup for og in store.ogs.values()] == ["OG1", "OG2"]
            assert [row["orthogroup"] for row in store.identities()] == ["OG1", "OG2"]
