"""Byte validation remains strict for every sequence, including discarded rows."""

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from phenoradar.amino_acid import ALPHABET, AlignmentError, AlignmentStore


@pytest.mark.parametrize("compressed", [False, True])
def test_import_preserves_bytes_hashes_and_species_presence(
    tmp_path: Path, compressed: bool
) -> None:
    source = tmp_path / ("OG1.faa.gz" if compressed else "OG1.faa")
    data = b">s1_a description\naCx-.\n>held_a\nVVVVV\n>s1_b\n--d*.\n>s2_a\nvv..-\n"
    source.write_bytes(gzip.compress(data, mtime=0) if compressed else data)
    with AlignmentStore(tmp_path, ["s1", "s2", "held"], cache_species=["s1", "s2"]) as store:
        og = store.ogs["OG1"]
        assert og.sequence_ids == ["s1_a", "s1_b", "s2_a"]
        assert og.sequence_hashes == [hashlib.sha256(s).hexdigest() for s in [b"ACX", b"D*", b"VV"]]
        assert og.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert og.species_rows == {"s1": [0, 1], "s2": [2]}
        assert Path(og.cache).read_bytes() == b"ACX-.--D*.VV..-"
        np.testing.assert_array_equal(store.call_counts(og, ["s1", "s2"]), [2, 2, 1, 0, 0])


def test_gzip_alignment_cache_preserves_og_names_and_rejects_duplicate_formats(
    tmp_path: Path,
) -> None:
    source = tmp_path / "alignments"
    source.mkdir()
    data = b">s1_a\nAC\n>s2_a\nVC\n"
    (source / "OG1.faa.gz").write_bytes(gzip.compress(data, mtime=0))
    cache = tmp_path / "cache"
    for _ in range(2):
        with AlignmentStore(
            source, ["s1", "s2"], input_cache_dir=cache, only_orthogroups={"OG1"}
        ) as store:
            assert store.ogs["OG1"].orthogroup == "OG1"
            assert Path(store.ogs["OG1"].cache).read_bytes() == b"ACVC"
            assert (
                store.identities()[0]["sha256"]
                == hashlib.sha256((source / "OG1.faa.gz").read_bytes()).hexdigest()
            )
    (source / "OG1.faa").write_bytes(data)
    with pytest.raises(AlignmentError, match="Duplicate orthogroup"):
        AlignmentStore(source, ["s1", "s2"])


@pytest.mark.parametrize("gzip_alignment", [False, True])
@pytest.mark.parametrize("gzip_tpm", [False, True])
def test_weighted_frequencies_accept_both_input_encodings(
    tmp_path: Path, gzip_alignment: bool, gzip_tpm: bool
) -> None:
    alignments = tmp_path / "alignments"
    alignments.mkdir()
    records = b">s1_a\nA\n>s1_v\nV\n>s2_a\nA\n>s2_v\nV\n"
    fasta = alignments / ("OG1.faa.gz" if gzip_alignment else "OG1.faa")
    fasta.write_bytes(gzip.compress(records, mtime=0) if gzip_alignment else records)
    values = b"species\tsequence_id\ttpm\ns1\ts1_a\t3\ns1\ts1_v\t1\ns2\ts2_a\t1\ns2\ts2_v\t3\n"
    tpm = tmp_path / ("sequence_tpm.tsv.gz" if gzip_tpm else "sequence_tpm.tsv")
    tpm.write_bytes(gzip.compress(values, mtime=0) if gzip_tpm else values)
    for _ in range(2):
        with AlignmentStore(
            alignments, ["s1", "s2"], input_cache_dir=tmp_path / "alignment_cache",
            sequence_tpm_path=str(tpm), sequence_tpm_cache_dir=tmp_path / "tpm_cache",
        ) as store:
            frequencies, _, _ = store.frequencies(
                store.ogs["OG1"], ["s1", "s2"], np.array([0]), weighted=True
            )
            np.testing.assert_allclose(
                frequencies[:, 0][:, [ALPHABET.index("A"), ALPHABET.index("V")]],
                [[0.75, 0.25], [0.25, 0.75]],
            )


@pytest.mark.parametrize(
    "value",
    [
        i
        for i in range(256)
        if i not in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*-.\t\n\r\v\f "
    ],
)
def test_discarded_sequences_still_reject_invalid_bytes(tmp_path: Path, value: int) -> None:
    (tmp_path / "OG1.faa").write_bytes(b">s1_g\nAA\n>held_g\nA" + bytes([value]) + b"\n")
    with pytest.raises(AlignmentError, match="Invalid sequence character"):
        AlignmentStore(tmp_path, ["s1", "held"], cache_species=["s1"])


@pytest.mark.parametrize(
    "record,message",
    [
        (b">held_g\nA\n", "Unequal alignment lengths"),
        (b">held_g\n\n", "Empty sequence"),
        (b">held_g\nAA\n>held_g\nVV\n", "Duplicate FASTA ID"),
        (b">unknown_g\nAA\n", "Cannot uniquely map"),
    ],
)
def test_discarded_sequences_keep_schema_checks(
    tmp_path: Path,
    record: bytes,
    message: str,
) -> None:
    (tmp_path / "OG1.faa").write_bytes(b">s1_g\nAA\n" + record)
    with pytest.raises(AlignmentError, match=message):
        AlignmentStore(tmp_path, ["s1", "held"], cache_species=["s1"])
