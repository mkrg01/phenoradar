import numpy as np
import pytest

from phenoradar import amino_acid_kernels as kernels
from phenoradar.amino_acid_encoding import compact_statistics
from phenoradar.config.schema import AminoAcidSiteFilter


@pytest.mark.parametrize("samples", [2, 7, 16, 31, 1038])
@pytest.mark.parametrize("states", [2, 3, 7, 20])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fractional_statistics_are_bitwise_exact(monkeypatch, samples, states, seed):
    if not hasattr(kernels.native, "observed_state_statistics"):
        pytest.skip("Native statistics kernel unavailable")
    rng = np.random.default_rng(seed)
    sites = np.repeat(np.arange(5), states)
    residues = np.tile(np.arange(states), 5)
    raw = rng.dirichlet(np.ones(states), size=(samples, 5)).reshape(samples, -1).astype(
        np.float32)
    raw[:, states:2*states] = 0
    raw[:, states] = 1  # Constant/ineligible site mixed with fractional sites.
    observed = rng.random((samples, 5)) > .15
    raw[~observed[:, sites]] = np.nan
    before = raw.view(np.uint32).copy()
    quality = AminoAcidSiteFilter()
    groups = np.arange(samples).astype(str)
    actual = compact_statistics(raw, observed, sites, residues, groups, quality, audit=False)
    with monkeypatch.context() as m:
        m.setattr(kernels, "native", None)
        expected = compact_statistics(raw, observed, sites, residues, groups, quality, audit=False)
    for a, b in zip(actual, expected, strict=True):
        assert a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(raw.view(np.uint32), before)


@pytest.mark.parametrize("value", [np.inf, -1., 2.])
def test_nonfrequency_inputs_keep_numpy_fallback(value):
    raw = np.array([[value, .2], [.4, .6]], dtype=np.float32)
    result = kernels.compact_statistics_native(raw, np.ones((2, 1), dtype=bool),
                                               np.array([0, 0]), np.array([0, 1]), 2)
    assert result is None


def test_single_state_retains_numpy_pairwise_reduction():
    assert kernels.compact_statistics_native(np.ones((100, 1), dtype=np.float32),
        np.ones((100, 1), dtype=bool), np.array([0]), np.array([0]), 1) is None


@pytest.mark.parametrize("delta", [0., 1e-8, 1e-6, 2e-6, 1e-5])
def test_observed_states_keep_constant_and_tiny_variation_columns(monkeypatch, delta):
    if not hasattr(kernels.native, "observed_state_statistics"):
        pytest.skip("Native statistics kernel unavailable")
    raw = np.array([[.25, .75], [.25+delta, .75-delta], [np.nan, np.nan]], dtype=np.float32)
    raw.view(np.uint32)[2] = 0x7FC00123
    raw = raw[np.array([0, 0, 1, 1, 1, 2, 2])]
    observed = ~np.isnan(raw[:, :1])
    args = (raw, observed, np.array([0, 0]), np.array([0, 1]),
            np.zeros(len(raw)), AminoAcidSiteFilter())
    actual = compact_statistics(*args, audit=False)
    assert actual[-1].sum() == 2
    with monkeypatch.context() as m:
        m.setattr(kernels, "native", None)
        expected = compact_statistics(*args, audit=False)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(actual, expected, strict=True))


def test_unordered_columns_use_numpy_fallback():
    result = kernels.compact_statistics_native(np.ones((5, 2), dtype=np.float32),
        np.ones((5, 1), dtype=bool), np.array([0, 0]), np.array([1, 0]), 2)
    assert result is None


def test_old_compiled_statistics_use_observed_state_numpy_fallback(monkeypatch):
    from types import SimpleNamespace

    def old_kernel(*args):
        raise AssertionError("The old variance-screening kernel must not run")

    monkeypatch.setattr(kernels, "native", SimpleNamespace(compact_statistics=old_kernel))
    raw = np.tile(np.array([.25, .75], dtype=np.float32), (4, 1))
    observed = np.ones((4, 1), dtype=bool)
    result = compact_statistics(
        raw, observed, np.array([0, 0]), np.array([0, 1]), np.zeros(4),
        AminoAcidSiteFilter(), audit=False,
    )
    assert result[-1].sum() == 2
