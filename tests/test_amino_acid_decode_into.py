import hashlib

import numpy as np
import pytest

from phenoradar import amino_acid_frequency_cache as frequency
from phenoradar import amino_acid_kernels as kernels
from phenoradar.packed_float import take


def payload(values, packed):
    if not packed:
        return np.ascontiguousarray(values).reshape(-1).view(np.uint8)
    bits = values.view(np.uint32)
    one, missing = bits == 0x3F800000, bits == 0x7FC00000
    other = ~((bits == 0) | one | missing)
    planes = np.packbits(np.stack((one, missing, other), axis=1), axis=2, bitorder="little")
    return np.r_[planes.reshape(-1), values[other].view(np.uint8)]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("include_coverage", [False, True])
@pytest.mark.parametrize("width", [1, 7, 8, 9, 113])
def test_scatter_matches_existing_decoder_bits_and_axes(packed, include_coverage, width):
    if not hasattr(kernels.native, "decode_compact"):
        pytest.skip("Optional scatter decoder unavailable")
    rng = np.random.default_rng(width)
    raw = rng.choice(np.array([0., -0., 1., .125, np.nan, np.inf, -np.inf],
                              dtype=np.float32), (9, width))
    raw.view(np.uint32)[1, 0] = 0x7FC00123
    raw.view(np.uint32)[4, 0] = 0xFFC00157
    data = payload(raw, packed)
    digest = hashlib.sha256(data).digest()
    data.flags.writeable = False
    coverage_rows, state_rows = np.array([1, 7, 1]), np.array([4, 0, 5, 8])
    samples, targets = np.array([width - 1, 0, width - 1]), np.array([4, 0, 2])
    selected, features = np.array([3, 1, 3]), np.array([4, 1, 4, 0])
    decoded = take(data, 9, width, np.r_[coverage_rows, state_rows], samples, packed=packed)
    expected = np.full((6, 6), np.nan, dtype=np.float32)
    expected[np.ix_(targets, features)] = decoded[3:].T
    raw_coverage = decoded[:3].T
    expected_observed = np.zeros((6, 5), dtype=bool)
    expected_observed[np.ix_(targets, selected)] = np.isfinite(raw_coverage)
    expected_coverage = np.zeros((6, 5), dtype=np.float32)
    expected_coverage[np.ix_(targets, selected)] = np.where(
        np.isfinite(raw_coverage), raw_coverage, 0,
    )
    actual = np.full_like(expected, np.nan)
    observed = np.zeros_like(expected_observed)
    coverage = np.zeros_like(expected_coverage) if include_coverage else None
    assert kernels.decode_compact_into(
        data, 9, width, coverage_rows, state_rows, samples, targets, selected, features,
        actual, observed, coverage, packed=packed,
    )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(observed, expected_observed)
    if coverage is not None:
        np.testing.assert_array_equal(coverage.view(np.uint32), expected_coverage.view(np.uint32))
    assert hashlib.sha256(data).digest() == digest


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("empty_samples", [False, True])
def test_empty_axes_leave_output_unchanged(packed, empty_samples):
    if not hasattr(kernels.native, "decode_compact"):
        pytest.skip("Optional scatter decoder unavailable")
    raw = np.zeros((3, 0 if empty_samples else 4), dtype=np.float32)
    data = payload(raw, packed)
    empty = np.empty(0, dtype=np.int64)
    values = np.full((2, 0), np.nan, dtype=np.float32)
    observed = np.zeros((2, 0), dtype=bool)
    assert kernels.decode_compact_into(data, 3, raw.shape[1], empty, empty, empty, empty,
                                      empty, empty, values, observed, None, packed=packed)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("axis", range(6))
def test_invalid_axis_is_rejected_before_output_mutation(packed, axis):
    if not hasattr(kernels.native, "decode_compact"):
        pytest.skip("Optional scatter decoder unavailable")
    data = payload(np.zeros((4, 5), dtype=np.float32), packed)
    axes = [np.array([1]), np.array([2]), np.array([3]), np.array([0]),
            np.array([0]), np.array([1])]
    axes[axis][0] = -1
    values = np.full((2, 2), .25, dtype=np.float32)
    observed = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="shape/index"):
        kernels.native.decode_compact(data, 4, 5, axes, (values, observed, None), packed)
    np.testing.assert_array_equal(values, .25)
    assert not observed.any()


def test_truncated_exceptions_and_overlapping_buffers_are_rejected():
    if not hasattr(kernels.native, "decode_compact"):
        pytest.skip("Optional scatter decoder unavailable")
    raw = np.zeros((4, 5), dtype=np.float32)
    raw[0, 0] = .125
    data = payload(raw, True)
    axes = [np.array([0])] * 6
    values, observed = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="Truncated"):
        kernels.native.decode_compact(data[:-4], 4, 5, axes, (values, observed, None), True)
    assert values[0, 0] == 0 and not observed.any()
    data = payload(raw, False)
    aliased = data.view(np.float32).reshape(4, 5)
    with pytest.raises(ValueError, match="shape/index"):
        kernels.native.decode_compact(data, 4, 5, axes,
                                      (aliased, np.zeros((4, 1), dtype=bool), None), False)


@pytest.mark.parametrize("raw_payload", [False, True])
def test_multichunk_cache_preserves_bits_with_missing_and_duplicate_species(
    tmp_path, monkeypatch, raw_payload,
):
    rng = np.random.default_rng(817)
    species = [f"s{i}" for i in range(8)]
    source = np.zeros((8, 13, 20), dtype=np.float32)
    source[:, :, [0, 3, 8]] = rng.dirichlet([.3, 1., 2.], (8, 13)).astype(np.float32)
    source[0, 1, 0] = -0.
    source[2, 2, :] = np.nan
    source.view(np.uint32)[3, 7, 3] = 0x7FC00057
    coverage = rng.choice([0., .5, 1.], (8, 13)).astype(np.float32)

    def compute(names, columns):
        rows = [species.index(s) for s in names]
        return source[rows][:, columns], coverage[rows][:, columns]

    if raw_payload:
        monkeypatch.setattr(frequency, "encode",
                            lambda a: (np.ascontiguousarray(a).reshape(-1).view(np.uint8), False))
    cache = frequency.AminoAcidFrequencyCache(
        tmp_path / "raw.bin", {s: i for i, s in enumerate(species)},
    )
    cache._ensure(np.arange(6), compute, 3)
    cache._ensure(np.arange(6, 13), compute, 4)
    digest = hashlib.sha256(cache.path.read_bytes()).digest()
    request = ["s7", "absent", "s0", "s7", "s2"]
    columns = np.array([12, 1, 12, 4, 7])
    with monkeypatch.context() as scoped:
        scoped.setattr(kernels, "decode_compact_into", lambda *a, **k: False)
        expected = cache.get_compact(request, columns, compute, block_size=4)
    for include_coverage in [False, True]:
        actual = cache.get_compact(request, columns, compute, block_size=4,
                                   include_coverage=include_coverage)
        for j in [0, 2, 3, 4]:
            a, b = actual[j], expected[j]
            np.testing.assert_array_equal(a.view(np.uint32) if j == 0 else a,
                                          b.view(np.uint32) if j == 0 else b)
        if include_coverage:
            np.testing.assert_array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))
        else:
            assert actual[1] is None
    assert hashlib.sha256(cache.path.read_bytes()).digest() == digest
