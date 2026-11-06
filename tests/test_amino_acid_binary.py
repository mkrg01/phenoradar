from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from test_amino_acid import settings

from phenoradar.amino_acid_audits import ColumnarAudits, merge_audits
from phenoradar.amino_acid_binary import encode_cached_block
from phenoradar.amino_acid_encoding import compact_statistics
from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache
from phenoradar.amino_acid_outputs import _append_frame
from phenoradar.amino_acid_sparse import SparseFoldEncoding


@pytest.mark.parametrize("n", [7, 64, 113])
@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
def test_binary_and_mixed_columns_preserve_all_statistics_and_CSC_bits(
    tmp_path: Path, n: int, audit: bool, duplicate: bool,
) -> None:
    rng = np.random.default_rng(853)
    samples = [f"s{i}" for i in range(n)]
    raw = np.zeros((n, 43, 20), dtype=np.float32)
    for j in range(raw.shape[1]):
        if j % 3:
            state = rng.integers(0, 4, n)
            raw[np.arange(n), j, state] = 1
        else:
            raw[:, j, :3] = rng.dirichlet(np.ones(3), size=n)
    raw[rng.random(raw.shape[:2]) < .15] = np.nan
    raw[:, 1] = 0  # Finite zeros are observed but have zero known-residue mass.
    raw[0, 1, 2] = 1
    raw[1, 1, 3] = 1
    raw[:, 2] = 0
    raw[:, 2, 4] = 1  # Ineligible monomorphic sites still have correct minor counts.
    raw[0, 4, 10] = -0.0  # Noncanonical bits use the floating-point path.
    raw[-1, 5] = 0
    raw[-1, 5, 19] = 1  # A validation-only state must not enter the training schema.
    coverage = np.isfinite(raw[:, :, 0]).astype(np.float32)
    columns = np.arange(raw.shape[1])
    rows = dict(zip(samples, range(n), strict=True))
    cache = AminoAcidFrequencyCache(tmp_path / "cache.bin", rows)

    def compute(species, selected):
        indices = [rows[s] for s in species]
        return raw[indices][:, selected], coverage[indices][:, selected]

    cache._ensure(columns, compute, 11)
    store = SimpleNamespace(
        _frequency_cache=lambda *_: (cache, compute, 11),
        compact_frequencies=lambda og, species, columns, **_: cache.get_compact(
            species, columns, compute, block_size=11),
    )
    training = samples[:-1] + ([samples[0], samples[1]] if duplicate else [])
    targets = [training, [samples[-1], "absent", samples[0], samples[0]], []]
    groups = np.array([str(i % 3) for i in range(len(training))])
    quality = settings().site_filter
    new = SparseFoldEncoding(targets, training, buffer_bytes=1024**2)
    old = SparseFoldEncoding(targets, training, buffer_bytes=1024**2)
    try:
        values, _, observed, sites, aa = cache.get_compact(
            old.species, columns, compute, block_size=11,
        )
        expected = compact_statistics(old._selected(values, old.training),
                                      old._selected(observed, old.training),
                                      sites, aa, groups, quality, audit=audit)
        old.add_compact(values, sites, aa, expected[0], *expected[4:])
        actual = encode_cached_block(store, None, columns, training, groups, quality, new,
                                     weighted=False, audit=audit)
        assert actual is not None
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
            if a.dtype == np.float64:
                np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        np.testing.assert_array_equal(new.shift.view(np.uint64), old.shift.view(np.uint64))
        for a, b in zip(new.matrices(), old.matrices(), strict=True):
            np.testing.assert_array_equal(a.data.view(np.uint64), b.data.view(np.uint64))
            np.testing.assert_array_equal(a.indices, b.indices)
            np.testing.assert_array_equal(a.indptr, b.indptr)
    finally:
        new.close()
        old.close()


def test_worker_parquet_audits_reproduce_TSV_bytes_and_order(tmp_path: Path, monkeypatch) -> None:
    import phenoradar.amino_acid_audits as audits

    monkeypatch.setattr(audits, "AUDIT_BLOCK_ROWS", 3)
    original, staged, merged = [tmp_path / name for name in ["old", "worker", "merged"]]
    merged.mkdir()
    with ColumnarAudits(staged) as writer:
        for i in range(4):
            frame = pl.DataFrame({
                "feature": ['quoted"\t\n', 'é', str(i)],
                "coefficient": [-0.0, np.nextafter(0., -1), .12345678901234567],
                "optional": [None, None, None] if not i else ["A", None, "V"],
            })
            _append_frame(original / "table.tsv", frame)
            writer.append("table.tsv", frame)
    merge_audits(staged, merged)
    assert (original / "table.tsv").read_bytes() == (merged / "table.tsv").read_bytes()
