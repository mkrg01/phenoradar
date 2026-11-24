from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from phenoradar.amino_acid_frequency_arena import consolidate
from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache


def fixture(tmp_path):
    rng = np.random.default_rng(817)
    raw = np.zeros((4, 71, 20), dtype=np.float32)
    raw[:, :, :3] = rng.dirichlet([1, 1, 1], size=(4, 71))
    raw[1, 2] = np.nan
    raw[2, 1, 0] = -0.0
    rows = dict(zip(["a", "b", "c", "held"], range(4), strict=True))

    def compute(samples, columns):
        data = raw[[rows[s] for s in samples]][:, columns]
        return data, np.isfinite(data[:, :, 0]).astype(np.float32)

    caches = {i: AminoAcidFrequencyCache(tmp_path / f"{i}.bin", rows) for i in range(3)}
    for cache in caches.values():
        cache._ensure(np.arange(60), compute, 19)
    store = SimpleNamespace(_frequency_caches=caches, _temp=TemporaryDirectory(dir=tmp_path))
    return store, compute


@pytest.mark.parametrize("file_bytes", [200, 10**8])
def test_arena_preserves_raw_bits_and_future_private_extensions(tmp_path: Path, file_bytes):
    store, compute = fixture(tmp_path)
    old = store._frequency_caches.copy()
    contents = {c.path: c.path.read_bytes() for c in old.values()}
    result = consolidate(store, file_bytes=file_bytes)
    assert result["checksum_verified"]
    if file_bytes > 200:
        assert result["arena_files"] == 1
    samples, columns = ["held", "a", "b", "a", "absent"], np.array([3, 59, 1, 3])
    for key, cache in store._frequency_caches.items():
        for a, b in zip(
            cache.get_compact(samples, columns, compute, block_size=19),
            old[key].get_compact(samples, columns, compute, block_size=19),
            strict=True,
        ):
            assert a.tobytes() == b.tobytes()
        fork = cache.fork(tmp_path / f"child-{key}.bin")
        fork.get_compact(samples, np.array([1, 70]), compute, block_size=19)
        assert fork.path.exists() and cache.path.read_bytes() == contents[cache.path]
    assert all(path.read_bytes() == data for path, data in contents.items())


def test_failed_copy_keeps_original_descriptors(tmp_path, monkeypatch):
    store, _ = fixture(tmp_path)
    original = store._frequency_caches
    before = sorted(Path(store._temp.name).iterdir())
    monkeypatch.setattr(
        "phenoradar.amino_acid_frequency_arena.hashlib.file_digest",
        lambda *a: SimpleNamespace(hexdigest=lambda: "wrong"),
    )
    with pytest.raises(OSError, match="checksum"):
        consolidate(store)
    assert store._frequency_caches is original
    assert sorted(Path(store._temp.name).iterdir()) == before


def test_snapshot_gate_is_optional_and_attempts_a_large_input_only_once(tmp_path, monkeypatch):
    import phenoradar.amino_acid_frequency_arena as arena

    store, _ = fixture(tmp_path)
    store.cache_frequencies = True
    original = store._frequency_caches
    arena.prepare_snapshot(store)
    assert store._frequency_caches is original
    monkeypatch.setattr(arena, "MIN_ARENA_BYTES", 1)
    monkeypatch.setattr(arena, "MIN_ARENA_FILES", 1)
    arena.prepare_snapshot(store)
    assert store._frequency_arena_summary["status"] == "complete"
    current = store._frequency_caches
    arena.prepare_snapshot(store)
    assert store._frequency_caches is current and len(store._frequency_arenas) == 1


def test_optional_failed_copy_keeps_the_original_input_usable(tmp_path, monkeypatch):
    import phenoradar.amino_acid_frequency_arena as arena

    store, _ = fixture(tmp_path)
    store.cache_frequencies = True
    original = store._frequency_caches
    monkeypatch.setattr(arena, "MIN_ARENA_BYTES", 1)
    monkeypatch.setattr(arena, "MIN_ARENA_FILES", 1)
    monkeypatch.setattr(
        arena.hashlib, "file_digest", lambda *a: SimpleNamespace(hexdigest=lambda: "wrong")
    )
    arena.prepare_snapshot(store)
    assert store._frequency_caches is original
    assert store._frequency_arena_summary["status"] == "original_extents_retained"


def test_automatic_snapshot_arena_survives_worker_use_and_closes_with_store(tmp_path, monkeypatch):
    from test_amino_acid import alignment, settings

    import phenoradar.amino_acid_frequency_arena as arena
    import phenoradar.amino_acid_parallel as parallel
    from phenoradar.amino_acid import AlignmentStore
    from phenoradar.amino_acid_parallel import initialize, snapshot
    from phenoradar.amino_acid_sparse import fit_sparse_encoded

    monkeypatch.setattr(arena, "MIN_ARENA_BYTES", 1)
    monkeypatch.setattr(arena, "MIN_ARENA_FILES", 1)
    samples = ["a", "b", "held"]
    path = alignment(tmp_path / "aa", [("a_1", "AA"), ("b_1", "VV"), ("held_1", "AV")])
    with AlignmentStore(path, samples, cache_frequencies=True) as store:
        args = dict(y=np.array([0, 1]), groups=np.array(["a", "b"]))
        _, expected, _ = fit_sparse_encoded(
            store, samples[:2], settings(), [samples[:2], samples[2:]], packed=True, **args
        )
        payload = snapshot(store, None)
        location = Path(store._frequency_arenas[0].name)
        assert location.exists()
        assert "_frequency_arenas" not in payload["store"]
        previous_worker = parallel._WORKER
        initialize(payload, 1)
        child, _ = parallel._WORKER
        try:
            _, actual, _ = fit_sparse_encoded(
                child, samples[:2], settings(), [samples[:2], samples[2:]], packed=True, **args
            )
            for a, b in zip(actual, expected, strict=True):
                a, b = a.to_csc(), b.to_csc()
                for name in ["data", "indices", "indptr"]:
                    assert getattr(a, name).tobytes() == getattr(b, name).tobytes()
        finally:
            child.close()
            parallel._WORKER = previous_worker
        assert location.exists()
    assert not location.exists()
