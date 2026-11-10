"""Byte validation remains strict for every sequence, including discarded rows."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from phenoradar.amino_acid import AlignmentError, AlignmentStore


def test_import_preserves_bytes_hashes_and_species_presence(tmp_path: Path) -> None:
    source = tmp_path / "OG1.faa"
    source.write_bytes(b">s1_a description\naCx-.\n>held_a\nVVVVV\n>s1_b\n--d*.\n>s2_a\nvv..-\n")
    with AlignmentStore(tmp_path, ["s1", "s2", "held"], cache_species=["s1", "s2"]) as store:
        og = store.ogs["OG1"]
        assert og.sequence_ids == ["s1_a", "s1_b", "s2_a"]
        assert og.sequence_hashes == [hashlib.sha256(s).hexdigest() for s in [b"ACX", b"D*", b"VV"]]
        assert og.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert og.species_rows == {"s1": [0, 1], "s2": [2]}
        assert Path(og.cache).read_bytes() == b"ACX-.--D*.VV..-"
        np.testing.assert_array_equal(store.call_counts(og, ["s1", "s2"]), [2, 2, 1, 0, 0])


@pytest.mark.parametrize("value", [
    i for i in range(256)
    if i not in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*-.\t\n\r\v\f "
])
def test_discarded_sequences_still_reject_invalid_bytes(tmp_path: Path, value: int) -> None:
    (tmp_path / "OG1.faa").write_bytes(b">s1_g\nAA\n>held_g\nA" + bytes([value]) + b"\n")
    with pytest.raises(AlignmentError, match="Invalid sequence character"):
        AlignmentStore(tmp_path, ["s1", "held"], cache_species=["s1"])


@pytest.mark.parametrize("record,message", [
    (b">held_g\nA\n", "Unequal alignment lengths"),
    (b">held_g\n\n", "Empty sequence"),
    (b">held_g\nAA\n>held_g\nVV\n", "Duplicate FASTA ID"),
    (b">unknown_g\nAA\n", "Cannot uniquely map"),
])
def test_discarded_sequences_keep_schema_checks(
    tmp_path: Path, record: bytes, message: str,
) -> None:
    (tmp_path / "OG1.faa").write_bytes(b">s1_g\nAA\n" + record)
    with pytest.raises(AlignmentError, match=message):
        AlignmentStore(tmp_path, ["s1", "held"], cache_species=["s1"])
