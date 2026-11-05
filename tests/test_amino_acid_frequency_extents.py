from pathlib import Path

import numpy as np

from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache


def test_worker_extensions_share_existing_data_and_write_only_new_columns(tmp_path: Path) -> None:
    calls = []

    def compute(species, columns):
        calls.extend(columns.tolist())
        values = np.zeros((len(species), len(columns), 20), dtype=np.float32)
        values[0, :, 0], values[1, :, 1] = 1, 1
        values[2, :, 2] = 1
        return values, np.ones(values.shape[:2], dtype=np.float32)

    parent = AminoAcidFrequencyCache(tmp_path / "parent.bin", {"a": 0, "b": 1, "held": 2})
    parent.get(["a", "b"], np.arange(20), compute, block_size=7)
    original = parent.path.read_bytes()
    fork = parent.fork(tmp_path / "worker.bin")
    fork.get(["b", "a", "b", "absent"], np.array([19, 1, 19]), compute, block_size=7)
    assert not fork.path.exists()
    fork.get(["a", "b"], np.array([20, 1]), compute, block_size=7)
    assert parent.path.read_bytes() == original
    assert fork.path.stat().st_size < len(original)
    assert calls == list(range(20)) + [20]
    np.testing.assert_array_equal(parent.columns, np.arange(20))
    assert set(fork._file_lengths) == {parent.path, fork.path}
    known = fork.eligible_columns(["a", "a"], np.array([0, 20]), compute,
                                 block_size=7, min_states=2)
    assert not known.any()
    assert fork.eligible_columns(["a", "b"], np.array([0, 20]), compute,
                                 block_size=7, min_states=2).all()


def test_presence_uses_stored_float32_frequencies_and_training_species_only(tmp_path: Path) -> None:
    def compute(species, columns):
        raw = np.zeros((3, len(columns), 20), dtype=np.float64)
        raw[:2, :, 0] = 1
        raw[0, :, 1] = 1e-100  # Positive input rounds to zero in the float32 cache.
        raw[2, :, 1] = 1
        raw[2, :, 0] = 0
        return raw, np.ones(raw.shape[:2])

    cache = AminoAcidFrequencyCache(tmp_path / "data.bin", {"a": 0, "b": 1, "held": 2})
    columns = np.array([2, 0, 2])
    assert not cache.eligible_columns(["a", "b", "a"], columns, compute,
                                      block_size=2, min_states=2).any()
    assert cache.eligible_columns(["a", "held"], columns, compute,
                                  block_size=2, min_states=2).all()
    raw, coverage = cache.get(["b", "absent", "a"], columns, compute, block_size=2)
    assert raw[2, :, 1].tolist() == [0, 0, 0]
    assert np.isnan(raw[1]).all()
    assert coverage[1].tolist() == [0, 0, 0]
