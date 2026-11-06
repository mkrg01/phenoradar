from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from phenoradar import sequence_tpm_cache as cache
from phenoradar.amino_acid import AlignmentStore


def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    aa = tmp_path / "aa"
    aa.mkdir()
    (aa / "OG.faa").write_text(">s1_a\nA\n>s1_v\nV\n>s2_a\nA\n>s2_v\nV\n")
    tpm = tmp_path / "tpm.tsv"
    tpm.write_text(
        "species\tsequence_id\ttpm\ns1\ts1_a\t3\ns1\ts1_v\t1\ns2\ts2_a\t1\ns2\ts2_v\t3\n"
    )
    return aa, tpm, tmp_path / "cache"


def load(aa: Path, tpm: Path, root: Path, species: list[str]):
    with AlignmentStore(
        aa,
        ["s1", "s2", "new_metadata_sample"],
        cache_species=species,
        sequence_tpm_path=str(tpm),
        sequence_tpm_cache_dir=root,
    ) as store:
        assert store.tpm_identity is not None
        q, coverage, _ = store.frequencies(store.ogs["OG"], species, np.array([0]), weighted=True)
        return q, coverage, store.tpm_identity


def test_reuses_all_species_without_source_parse_or_hash(tmp_path: Path, monkeypatch) -> None:
    aa, tpm, root = inputs(tmp_path)
    q1, _, cold = load(aa, tpm, root, ["s1"])
    assert cold["cache_hit"] is False

    def forbidden(*args, **kwargs):
        raise AssertionError("Warm cache must not scan/hash the TPM source")

    monkeypatch.setattr(cache, "_build", forbidden)
    monkeypatch.setattr(cache, "file_identity", forbidden)
    q2, _, warm = load(aa, tpm, root, ["s2"])
    assert warm["cache_hit"] is True
    assert warm["sha256"] == cold["sha256"]
    np.testing.assert_allclose(q1[0, 0, [0, 17]], [0.75, 0.25])
    np.testing.assert_allclose(q2[0, 0, [0, 17]], [0.25, 0.75])
    q3, _, _ = load(aa, tpm, root, ["s1", "s2"])
    np.testing.assert_array_equal(q3, np.concatenate([q1, q2]))


def test_source_change_invalidates_even_with_restored_mtime(tmp_path: Path) -> None:
    aa, tpm, root = inputs(tmp_path)
    _, _, old = load(aa, tpm, root, ["s1"])
    stat = tpm.stat()
    tpm.write_text(tpm.read_text().replace("s1_a\t3", "s1_a\t1"))
    os.utime(tpm, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    q, _, new = load(aa, tpm, root, ["s1"])
    assert new["cache_hit"] is False
    assert old["sha256"] != new["sha256"]
    np.testing.assert_allclose(q[0, 0, [0, 17]], [0.5, 0.5])


def test_corrupt_payload_is_rebuilt(tmp_path: Path) -> None:
    aa, tpm, root = inputs(tmp_path)
    before, _, identity = load(aa, tpm, root, ["s1"])
    payload = next(Path(identity["cache_path"]).glob("part-*.parquet"))
    payload.write_bytes(b"incomplete")
    shutil.rmtree(root / "aligned")  # Exercise the raw cache, independently of numeric reuse.
    after, _, repaired = load(aa, tpm, root, ["s1"])
    assert repaired["cache_hit"] is False
    np.testing.assert_array_equal(before, after)


def test_failed_build_does_not_publish_partial_cache(tmp_path: Path, monkeypatch) -> None:
    aa, tpm, root = inputs(tmp_path)
    original = cache.pl.DataFrame.write_parquet

    def fail(self, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(cache.pl.DataFrame, "write_parquet", fail)
    with pytest.raises(OSError, match="interrupted"):
        load(aa, tpm, root, ["s1"])
    assert not list(root.rglob("*.parquet"))
    monkeypatch.setattr(cache.pl.DataFrame, "write_parquet", original)
    _, _, identity = load(aa, tpm, root, ["s1"])
    assert identity["cache_hit"] is False


def test_concurrent_readers_publish_once(tmp_path: Path) -> None:
    aa, tpm, root = inputs(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda s: load(aa, tpm, root, [s]), ["s1", "s2"]))
    assert sorted(r[2]["cache_hit"] for r in results) == [False, True]
    assert len(list(root.rglob("part-*.parquet"))) == 1


def test_blocks_keep_complete_rows_and_preserve_values(tmp_path: Path, monkeypatch) -> None:
    aa, tpm, root = inputs(tmp_path)
    monkeypatch.setattr(cache, "_CHUNK_BYTES", 12)
    monkeypatch.setattr(cache, "_QUERY_PARTS", 1)
    original = cache.pl.scan_parquet

    def bounded(paths, **kwargs):
        assert len(paths) <= 1
        return original(paths, **kwargs)

    monkeypatch.setattr(cache.pl, "scan_parquet", bounded)
    q, _, identity = load(aa, tpm, root, ["s1", "s2"])
    assert len(list(Path(identity["cache_path"]).glob("part-*.parquet"))) > 1
    np.testing.assert_allclose(q[:, 0, [0, 17]], [[0.75, 0.25], [0.25, 0.75]])


def test_blocks_allow_quoted_multiline_extra_columns(tmp_path: Path, monkeypatch) -> None:
    aa, tpm, root = inputs(tmp_path)
    lines = tpm.read_text().splitlines()
    tpm.write_text(
        lines[0] + "\tnote\n" + "".join(line + '\t"first\nsecond"\n' for line in lines[1:])
    )
    monkeypatch.setattr(cache, "_CHUNK_BYTES", 32)
    q, _, _ = load(aa, tpm, root, ["s1", "s2"])
    np.testing.assert_allclose(q[:, 0, [0, 17]], [[0.75, 0.25], [0.25, 0.75]])


def test_species_index_upgrade_does_not_reread_source(tmp_path: Path, monkeypatch) -> None:
    aa, tpm, root = inputs(tmp_path)
    before, _, identity = load(aa, tpm, root, ["s1", "s2"])
    path = Path(identity["cache_path"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    shutil.rmtree(root / "aligned")
    manifest.pop("part_species")
    path.write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        raise AssertionError("Index upgrade must reuse Parquet, not the source TSV")

    monkeypatch.setattr(cache, "_build", forbidden)
    after, _, updated = load(aa, tpm, root, ["s1", "s2"])
    assert updated["cache_hit"] is True
    assert "part_species" in json.loads(path.read_text())
    np.testing.assert_array_equal(before, after)
