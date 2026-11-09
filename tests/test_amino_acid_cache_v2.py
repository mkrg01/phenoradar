"""Cross-run reuse must preserve all input validation and exact numerical values."""
import gzip
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from test_amino_acid import alignment
from test_sequence_tpm_cache import inputs, load

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache
from phenoradar.packed_float import encode, take


@pytest.mark.parametrize("copy", [False, True])
def test_relocated_alignment_reuses_bytes(tmp_path: Path, monkeypatch, copy: bool) -> None:
    source = alignment(tmp_path / "aa", [("s1_a", "AX-"), ("s2_a", "VVA")])
    root = tmp_path / "cache"
    with AlignmentStore(source, ["s1", "s2"], input_cache_dir=root) as store:
        original = store.ogs["OG1"].cache
    destination = tmp_path / "relocated"
    if copy:
        shutil.copytree(source, destination)
    else:
        source.rename(destination)
    def forbidden(*args, **kwargs):
        raise AssertionError("Same content must reuse the parsed alignment")
    monkeypatch.setattr(AlignmentStore, "_read", forbidden)
    with AlignmentStore(destination, ["s1", "s2"], input_cache_dir=root) as store:
        assert store.ogs["OG1"].cache == original
        assert store.identities()[0]["path"] == str((destination / "OG1.faa").resolve())


def test_changed_alignment_with_restored_mtime_is_not_a_hit(tmp_path: Path) -> None:
    source = alignment(tmp_path / "aa", [("s1_a", "AAA"), ("s2_a", "VVA")])
    root = tmp_path / "cache"
    with AlignmentStore(source, ["s1", "s2"], input_cache_dir=root) as store:
        old = store.ogs["OG1"].sha256
    p = source / "OG1.faa"
    before = p.stat()
    p.write_bytes(p.read_bytes().replace(b"AAA", b"VVV"))
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
    with AlignmentStore(source, ["s1", "s2"], input_cache_dir=root) as store:
        assert store.ogs["OG1"].sha256 != old
        assert store.ogs["OG1"].matrix()[0].tobytes() == b"VVV"


def test_aligned_tpm_reuses_numeric_weights_and_masks(tmp_path: Path, monkeypatch) -> None:
    import phenoradar.amino_acid as aa_module
    aa, tpm, root = inputs(tmp_path)
    before, coverage, identity = load(aa, tpm, root, ["s1", "s2"])
    assert identity["aligned_cache_hit"] is False
    moved = tpm.with_name("relocated.tsv")
    shutil.copy2(tpm, moved)
    def forbidden(*args, **kwargs):
        raise AssertionError("Numeric cache must skip both Parquet lookup and mask construction")
    monkeypatch.setattr(aa_module, "read_sequence_tpm_cache", forbidden)
    monkeypatch.setattr(AlignmentStore, "_positive_tpm_mask", forbidden)
    after, cov, identity = load(aa, moved, root, ["s1", "s2"])
    assert identity["aligned_cache_hit"] is True
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(coverage, cov)


@pytest.mark.parametrize("dense", [False, True])
def test_float_codec_preserves_every_float32_bit(tmp_path: Path, dense: bool) -> None:
    rng = np.random.default_rng(3391)
    raw = rng.integers(0, 2**32, size=(33, 117), dtype=np.uint32)
    if not dense:
        raw[:, :110] = rng.choice(np.array([0, 0x3F800000, 0x7FC00000], dtype=np.uint32),
                                  size=(33, 110))
    raw[0, -7:] = [0, 0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000, 1, 0x3F000000]
    values = raw.view(np.float32)
    data, packed = encode(values)
    assert packed is not dense
    rows, columns = np.array([32, 0, 17, 0]), np.array([116, 0, 4, 110, 111, 112, 113, 114, 115])
    restored = take(data, len(values), values.shape[1], rows, columns, packed=packed)
    np.testing.assert_array_equal(restored.view(np.uint32), raw[np.ix_(rows, columns)])


def test_frequency_segments_survive_restart_and_worker_extensions(tmp_path: Path) -> None:
    samples = {"a": 0, "b": 1, "held": 2}
    calls = []
    def compute(species, columns):
        calls.extend(columns.tolist())
        raw = np.zeros((len(species), len(columns), 20), dtype=np.float32)
        raw[0, :, 0] = 1
        raw[1, :, 1:3] = [.25, .75]
        raw[2] = np.nan
        return raw, np.array([[1] * len(columns), [.6] * len(columns), [0] * len(columns)],
                             dtype=np.float32)
    parent = AminoAcidFrequencyCache(tmp_path / "parent.bin", samples,
                                    persistent=tmp_path / "persistent")
    expected = parent.get(["b", "a", "held", "a"], np.array([2, 0, 2]), compute, block_size=2)
    assert parent.persist()
    assert not parent.persist()
    child = parent.fork(tmp_path / "worker.bin")
    child.get(["a"], np.array([3]), compute, block_size=2)
    assert child.persist()
    child.get(["a"], np.array([4]), compute, block_size=2)
    assert child.persist()
    # Each commit writes only new bytes, not the worker's preceding generations.
    def forbidden(*args, **kwargs):
        raise AssertionError("Persisted columns must not recompute frequencies")
    restored = AminoAcidFrequencyCache(tmp_path / "fresh.bin", samples,
                                      persistent=tmp_path / "persistent")
    actual = restored.get(["b", "a", "held", "a"], np.array([2, 0, 2]), forbidden, block_size=2)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    restored.get(["a"], np.array([3, 4]), forbidden, block_size=2)
    assert not restored.path.exists()
    assert calls == [0, 2, 3, 4]


@pytest.mark.parametrize("same_stat", [False, True])
def test_corrupt_frequency_segment_is_recomputed(tmp_path: Path, monkeypatch,
                                                same_stat: bool) -> None:
    directory = tmp_path / "persistent"
    def compute(species, columns):
        raw = np.zeros((1, len(columns), 20), dtype=np.float32)
        raw[:, :, 0] = 1
        return raw, np.ones(raw.shape[:2], dtype=np.float32)
    first = AminoAcidFrequencyCache(tmp_path / "first.bin", {"a": 0}, persistent=directory)
    first.get(["a"], np.array([0]), compute, block_size=2)
    first.persist()
    if same_stat:
        import phenoradar.amino_acid_frequency_store as storage

        entry = next(directory.glob("part-*"))
        payload = storage._payload(entry)
        monkeypatch.setattr(storage, "_payload", lambda _: payload)
    next(directory.glob("part-*/values.bin")).write_bytes(b"broken")
    second = AminoAcidFrequencyCache(tmp_path / "second.bin", {"a": 0}, persistent=directory)
    assert not len(second.columns)
    raw, _ = second.get(["a"], np.array([0]), compute, block_size=2)
    assert raw[0, 0, 0] == 1


def test_store_reuses_raw_frequencies_without_reusing_fitted_filters(tmp_path: Path, monkeypatch):
    source = alignment(tmp_path / "aa", [("s1_a", "AAV"), ("s2_a", "VVV")])
    root = tmp_path / "cache"
    with AlignmentStore(source, ["s1", "s2"], input_cache_dir=root,
                        cache_frequencies=True) as store:
        expected, _, _ = store.frequencies(store.ogs["OG1"], ["s1", "s2"], np.arange(3))
    def forbidden(*args, **kwargs):
        raise AssertionError("Raw frequencies must survive closing the input store")
    monkeypatch.setattr(AlignmentStore, "_compute_frequencies", forbidden)
    with AlignmentStore(source, ["s1", "s2"], input_cache_dir=root,
                        cache_frequencies=True) as store:
        actual, _, _ = store.frequencies(store.ogs["OG1"], ["s2", "s1"], np.arange(3))
        np.testing.assert_array_equal(actual, expected[::-1])
        assert not store._frequency_caches[("OG1", False)].path.exists()


def test_concurrent_frequency_segments_restore_the_union(tmp_path: Path) -> None:
    root = tmp_path / "persistent"
    def compute(species, columns):
        raw = np.zeros((2, len(columns), 20), dtype=np.float32)
        raw[0, :, 0], raw[1, :, 1] = 1, 1
        return raw, np.ones(raw.shape[:2], dtype=np.float32)
    parent = AminoAcidFrequencyCache(tmp_path / "parent.bin", {"a": 0, "b": 1}, persistent=root)
    parent.get(["a", "b"], np.array([0]), compute, block_size=2)
    parent.persist()
    def publish(i):
        child = parent.fork(tmp_path / f"worker{i}.bin")
        child.get(["a"], np.array([1, i + 2]), compute, block_size=2)
        child.persist()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(publish, range(2)))
    restored = AminoAcidFrequencyCache(tmp_path / "fresh.bin", {"a": 0, "b": 1}, persistent=root)
    np.testing.assert_array_equal(restored.columns, np.arange(4))
    actual, _ = restored.get(["a", "b"], np.arange(4), compute, block_size=2)
    np.testing.assert_array_equal(actual, compute(["a", "b"], np.arange(4))[0])
    assert not restored.path.exists()


def test_corrupt_numeric_tpm_rebuilds_from_raw_cache(tmp_path: Path) -> None:
    aa, tpm, root = inputs(tmp_path)
    before, _, identity = load(aa, tpm, root, ["s1", "s2"])
    (Path(identity["aligned_cache_path"]) / "weights.npy").write_bytes(b"broken")
    actual, _, repaired = load(aa, tpm, root, ["s1", "s2"])
    assert repaired["cache_hit"] is True
    assert repaired["aligned_cache_hit"] is False
    np.testing.assert_array_equal(actual, before)


def test_changed_compression_mode_does_not_bypass_input_validation(tmp_path: Path) -> None:
    aa, tpm, root = inputs(tmp_path)
    compressed = tpm.with_suffix(".tsv.gz")
    with gzip.open(compressed, "wb") as handle:
        handle.write(tpm.read_bytes())
    load(aa, compressed, root, ["s1", "s2"])
    # Equal bytes require a different decoder when the filename loses .gz.
    wrong = tpm.with_name("wrong.tsv")
    shutil.copy2(compressed, wrong)
    with pytest.raises((ValueError, OSError)):
        load(aa, wrong, root, ["s1", "s2"])


def test_repaired_segment_does_not_reuse_an_old_memory_mapping(tmp_path: Path) -> None:
    root = tmp_path / "persistent"
    def compute(species, columns):
        raw = np.zeros((1, len(columns), 20), dtype=np.float32)
        raw[:, :, 0] = .25
        raw[:, :, 1] = .75
        return raw, np.ones(raw.shape[:2], dtype=np.float32)
    def cache(name):
        return AminoAcidFrequencyCache(tmp_path / name, {"a": 0}, persistent=root)
    first = cache("first.bin")
    expected = first.get(["a"], np.array([0]), compute, block_size=2)[0]
    first.persist()
    mapped = cache("mapped.bin")
    mapped.get(["a"], np.array([0]), compute, block_size=2)
    path = next(root.glob("part-*/values.bin"))
    original_inode = path.stat().st_ino
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x20
    path.write_bytes(data)
    repaired = cache("repair.bin")
    repaired.get(["a"], np.array([0]), compute, block_size=2)
    repaired.persist()
    assert path.stat().st_ino != original_inode
    actual = cache("last.bin").get(["a"], np.array([0]), compute, block_size=2)[0]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
