import numpy as np
import pytest

from phenoradar import amino_acid_kernels as kernels
from phenoradar.amino_acid_encoding import compact_statistics
from phenoradar.amino_acid_sparse import SparseFoldEncoding
from phenoradar.config.schema import AminoAcidSiteFilter
from phenoradar.packed_float import encode, take


@pytest.mark.parametrize("width", [1, 7, 8, 9, 113, 116])
def test_native_decoder_preserves_exception_bits_and_duplicate_axes(monkeypatch, width):
    if kernels.native is None:
        pytest.skip("Optional C compiler unavailable")
    rng = np.random.default_rng(width)
    a = rng.choice(np.array([0, 0, 0, 1, np.nan, .125, -0.0], dtype=np.float32), (17, width))
    a.view(np.uint32)[0, 0] = 0x7FC00123  # Noncanonical NaN payload.
    data, packed = encode(a)
    assert packed or width == 1
    rows = np.array([12, 0, 12, 4, 16])
    columns = np.array([width - 1, 0, width - 1])
    actual = take(data, 17, width, rows, columns, packed=packed)
    with monkeypatch.context() as m:
        m.setattr(kernels, "native", None)
        expected = take(data, 17, width, rows, columns, packed=packed)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("seed", range(6))
def test_native_fractional_centering_and_csc_are_exact(monkeypatch, seed):
    if kernels.native is None:
        pytest.skip("Optional C compiler unavailable")
    rng = np.random.default_rng(seed)
    species = [f"s{i}" for i in range(13)]
    sites = np.repeat(np.arange(9), 3)
    residues = np.tile(np.array([0, 2, 6]), 9)
    values = rng.dirichlet([.3, 1, 2], (13, 9)).reshape(13, -1).astype(np.float32)
    observed = rng.random((13, 9)) > .15
    values[~observed[:, sites]] = np.nan
    train = np.array([0, 3, 3, 5, 7, 9, 10])
    stats = compact_statistics(values[train], observed[train], sites, residues,
                                np.array([str(i) for i in train]), AminoAcidSiteFilter(),
                                audit=False)
    groups = [[species[i] for i in train], [species[i] for i in [12, 1, 12, 4]]]
    blocks = []
    for native in [True, False]:
        encoding = SparseFoldEncoding(groups, groups[0], buffer_bytes=10**8)
        try:
            with monkeypatch.context() as m:
                if not native:
                    m.setattr(kernels, "native", None)
                encoding.add_compact(values.copy(), sites, residues, stats[0], *stats[4:])
            blocks.append((encoding.matrices(), encoding.shift))
        finally:
            encoding.close()
    for a, b in zip(blocks[0][0], blocks[1][0], strict=True):
        for name in ["data", "indices", "indptr"]:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(blocks[0][1], blocks[1][1])


def test_native_decoder_rejects_truncated_exceptions():
    if kernels.native is None:
        pytest.skip("Optional C compiler unavailable")
    a = np.zeros((8, 20), dtype=np.float32)
    a[0, 0] = .125
    data, packed = encode(a)
    assert packed
    with pytest.raises(ValueError, match="Truncated"):
        take(data[:-4], 8, 20, np.array([0]), np.array([0]), packed=True)
