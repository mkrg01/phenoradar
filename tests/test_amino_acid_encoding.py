from pathlib import Path

import numpy as np
import pytest
from test_amino_acid import alignment, settings, tpm_table

from phenoradar.amino_acid import AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_encoding import fit_encoded


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
@pytest.mark.parametrize("method", ["none", "unpaired", "pair_aware"])
def test_fused_encoding_is_exact_and_validation_does_not_fit_sites(
    tmp_path: Path, monkeypatch, disk: bool, aggregation: str, method: str,
) -> None:
    import phenoradar.amino_acid_encoding as encoding

    if disk:
        monkeypatch.setattr(encoding, "ENCODING_MEMORY_LIMIT_BYTES", 0)
    records = [("a_1", "AC-DVAX"), ("a_2", "ADDDVAX"), ("b_1", "ADCEVAX"),
               ("c_1", "ADCDAAX"), ("d_1", "ACCEAAX"), ("held_1", "W-XWWVX")]
    path = alignment(tmp_path / "aa", records)
    tpm = tpm_table(tmp_path / "tpm.tsv", [(s.split('_')[0], s, i) for i, (s, _) in
                                          enumerate(records)])
    cfg = settings(aggregation)
    cfg.site_filter.min_sample_fraction = .5
    cfg.ranked_site_filter.method = method
    cfg.ranked_site_filter.min_contrast_pairs = 1
    training = ["a", "b", "c", "d", "a"]  # Duplicate biological rows are preserved.
    kwargs = dict(y=np.array([0, 1, 0, 1, 0]), groups=np.array(["1", "1", "2", "2", "1"]),
                  contrast_groups=np.array(["1", "1", "2", "2", "1"]))
    with AlignmentStore(path, ["a", "b", "c", "d", "held", "absent"],
                        cache_frequencies=True, sequence_tpm_path=str(tpm)) as store:
        expected = fit_amino_acid(store, training, cfg, **kwargs)
        fitted, matrices = fit_encoded(store, training, cfg,
                                       [training, ["held", "absent"], []], **kwargs)
        assert fitted.features == expected.features
        np.testing.assert_array_equal([s.mean for s in fitted.sites],
                                      [s.mean for s in expected.sites])
        for species, matrix in zip([training, ["held", "absent"], []], matrices, strict=True):
            np.testing.assert_array_equal(matrix, expected.matrix(store, species))
            assert matrix.flags.f_contiguous


def test_frequency_mapping_cache_is_bounded_and_released(tmp_path: Path, monkeypatch) -> None:
    import phenoradar.amino_acid_frequency_cache as frequency

    monkeypatch.setattr(frequency, "_MAX_OPEN_MAPS", 3)
    samples = ["a", "b"]
    columns = np.arange(2)
    raw = np.zeros((2, 2, 20), dtype=np.float32)
    raw[0, :, 0] = 1
    raw[1, :, 1] = 1
    for i in range(10):
        cache = frequency.AminoAcidFrequencyCache(tmp_path / f"{i}.bin", {"a": 0, "b": 1})
        actual, _ = cache.get(samples, columns, lambda *_: (raw, np.ones((2, 2))), block_size=2)
        np.testing.assert_array_equal(actual, raw)
        assert len(frequency._MAPS) <= 3
    frequency.release_frequency_maps(tmp_path)
    assert not any(path.is_relative_to(tmp_path) for path, _ in frequency._MAPS)


@pytest.mark.parametrize("n", [7, 64, 113])
def test_compact_moments_keep_float32_variance_and_missing_semantics(n: int) -> None:
    from phenoradar.amino_acid import _quality_summary, _site_statistics
    from phenoradar.amino_acid_encoding import compact_statistics

    rng = np.random.default_rng(834)
    raw = np.zeros((n, 83, 20), dtype=np.float32)
    for j in range(raw.shape[1]):
        states = rng.choice(20, 1 + j % 20, replace=False)
        raw[:, j, states] = rng.dirichlet(np.full(len(states), .2), size=n)
    raw[rng.random(raw.shape[:2]) < .15] = np.nan
    raw[:, 0] = 0
    raw[:, 0, :2] = [.1, .9]  # A constant composition close to float32 variance tolerance.
    sites, aa = np.nonzero(np.any(raw > 0, axis=0))
    observed = np.isfinite(raw[:, :, 0])
    groups = np.array([str(i % 3) for i in range(n)])
    quality = settings().site_filter
    expected = _site_statistics(raw)
    eligible, minor, group_minor = _quality_summary(raw, groups, quality)
    actual = compact_statistics(raw[:, sites, aa], observed, sites, aa, groups, quality, audit=True)
    np.testing.assert_array_equal(actual[0], eligible)
    np.testing.assert_array_equal(actual[1], minor)
    np.testing.assert_array_equal(actual[2], group_minor)
    np.testing.assert_array_equal(actual[3], expected[0])
    for new, old in zip(actual[4:], expected[1:], strict=True):
        np.testing.assert_array_equal(new[eligible], old[eligible])


def test_encoding_can_spill_after_buffering_without_changing_columns() -> None:
    from phenoradar.amino_acid_encoding import FoldEncoding

    encoder = FoldEncoding([["a", "b"], ["b"], []])
    encoder.memory_limit = 100
    chunks = []
    try:
        for i in range(5):
            block = np.arange(4).reshape(2, 2) + i * 5
            chunks.append(block)
            encoder.append(block)
        assert encoder.spilled
        matrices = encoder.matrices()
    finally:
        encoder.close()
    expected = np.column_stack(chunks)
    np.testing.assert_array_equal(matrices[0], expected)
    np.testing.assert_array_equal(matrices[1], expected[[1]])
    assert matrices[2].shape == (0, expected.shape[1])
