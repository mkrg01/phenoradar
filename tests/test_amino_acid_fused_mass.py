import numpy as np
import pytest

from phenoradar import amino_acid_kernels as kernels
from phenoradar.amino_acid_encoding import _compact_mass, compact_statistics
from phenoradar.amino_acid_packed_encoding import PackedFoldEncoding
from phenoradar.amino_acid_sparse import SparseFoldEncoding
from phenoradar.config.schema import AminoAcidSiteFilter

pytestmark = pytest.mark.skipif(
    not hasattr(kernels.native, "centering_mass"), reason="Optional mass kernel unavailable",
)


def legacy_mass(values, sites, residues, eligible, known):
    keep = eligible[sites]
    local = np.cumsum(eligible) - 1
    raw = np.nan_to_num(values[:, keep], copy=False)
    raw *= known[sites[keep], residues[keep]][None, :]
    return _compact_mass(raw, local[sites[keep]], residues[keep], int(eligible.sum()))


@pytest.mark.parametrize("states", [3, 20])
@pytest.mark.parametrize("seed", range(12))
def test_guarded_mass_matches_legacy_bits_for_partial_and_extreme_sites(seed, states):
    rng = np.random.default_rng(seed)
    n, width = 43, 17
    sites = np.repeat(np.arange(width), states)
    residues = np.tile(np.arange(states), width)
    values = rng.dirichlet(np.full(states, .4), (n, width)).astype(np.float32)
    values[rng.random((n, width)) < .1] = np.nan
    values[0, 0] = 0
    values[0, 0, 0] = 1
    values[0, 0, 1] = np.finfo(np.float32).smallest_subnormal
    values[1, 1, 0] = -0.
    values[2, 2, 0] = np.inf
    values[3, 3, 0] = -np.inf
    values[4, 4, 0] = -1e-6
    eligible = rng.random(width) > .1
    eligible[:5] = True
    known = rng.random((width, 20)) > .2
    known[:5] = True
    known[5:7] = False
    values = values.reshape(n, -1)
    values.flags.writeable = False
    expected = legacy_mass(values, sites, residues, eligible, known)
    actual = kernels.centering_mass(values, sites, residues, eligible, known)
    assert actual is not None
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_guarded_mass_clipping_boundaries_and_unknown_states_match_bits():
    sites = np.repeat(np.arange(4), 3)
    residues = np.tile([0, 3, 8], 4)
    values = np.zeros((9, 12), dtype=np.float32)
    values[:, 0] = [1, np.nextafter(np.float32(1), np.float32(0)),
                    1 - 1e-7, 1 - 2e-7, 1 + 1e-7, 1 + 2e-7, 0, -0., np.nan]
    values[:, 3:6] = [1, .5, np.finfo(np.float32).smallest_subnormal]
    values[:, 6:9] = [-0., np.inf, -np.inf]
    known = np.ones((4, 20), dtype=bool)
    known[1, 8] = False
    known[2] = False
    eligible = np.ones(4, dtype=bool)
    expected = legacy_mass(values, sites, residues, eligible, known)
    actual = kernels.centering_mass(values, sites, residues, eligible, known)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_mass_kernel_routes_inexact_sum_and_negative_zero_to_legacy():
    values = np.array([[.25, .75, 0., .5, 1e-30, 1e-30, -0., .5, .5]], dtype=np.float32)
    sites = np.repeat(np.arange(3), 3)
    mass = np.empty((1, 3))
    fallback = np.empty(3, dtype=bool)
    count = kernels.native.centering_mass(values, np.arange(9), sites, mass, fallback, 1, 9, 3)
    assert count == 2
    np.testing.assert_array_equal(fallback, [False, True, True])
    assert mass[0, 0] == 1

    values[0, 3] = 1
    count = kernels.native.centering_mass(values, np.arange(9), sites, mass, fallback, 1, 9, 3)
    assert count == 1
    np.testing.assert_array_equal(fallback, [False, False, True])
    assert mass[0, 1] == 1


def test_mass_preserves_legacy_order_when_three_terms_can_round_differently():
    values = np.zeros((1, 20), dtype=np.float32)
    values[0, [0, 1, 9]] = [.5, 2**-54, 2**-54]
    sites = np.zeros(20, dtype=np.int64)
    mass, fallback = np.empty((1, 1)), np.empty(1, dtype=bool)
    assert kernels.native.centering_mass(values, np.arange(20), sites, mass, fallback,
                                         1, 20, 1) == 1
    actual = kernels.centering_mass(values, sites, np.arange(20), np.ones(1, dtype=bool),
                                    np.ones((1, 20), dtype=bool))
    expected = legacy_mass(values, sites, np.arange(20), np.ones(1, dtype=bool),
                           np.ones((1, 20), dtype=bool))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_mass_kernel_rejects_more_than_twenty_terms():
    with pytest.raises(ValueError, match="shape/index"):
        kernels.native.centering_mass(np.ones((2, 21), dtype=np.float32), np.arange(21),
                                       np.zeros(21, dtype=np.int64), np.empty((2, 1)),
                                       np.empty(1, dtype=bool), 2, 21, 1)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("heldout_state", [False, True])
def test_mass_preserves_encoded_matrices_and_shifts(monkeypatch, seed, packed, heldout_state):
    rng = np.random.default_rng(seed)
    species = [f"s{i}" for i in range(80)]
    training = species[:60] + [species[3], species[3]]
    groups = [training, species[60:] + [species[65]]]
    sites = np.repeat(np.arange(16), 3)
    residues = np.tile([0, 3, 8], 16)
    values = rng.dirichlet([.3, 1, 2], (80, 16)).reshape(80, -1).astype(np.float32)
    observed = rng.random((80, 16)) > .1
    values[~observed[:, sites]] = np.nan
    if heldout_state:
        values[:60, residues == 8] = 0
    train = np.array([species.index(s) for s in training])
    stats = compact_statistics(values[train], observed[train], sites, residues,
                               np.array(training), AminoAcidSiteFilter(), audit=False)
    results = []
    for fused in [False, True]:
        encoding = (PackedFoldEncoding(groups, training, fused=False, budget=10**8) if packed
                    else SparseFoldEncoding(groups, training, buffer_bytes=0))
        try:
            with monkeypatch.context() as scoped:
                if not fused:
                    scoped.setattr(kernels, "centering_mass", lambda *args: None)
                encoding.add_compact(values.copy(), sites, residues, stats[0], *stats[4:])
            results.append((encoding.matrices(), encoding.shift.copy()))
        finally:
            encoding.close()
    np.testing.assert_array_equal(results[0][1].view(np.uint64), results[1][1].view(np.uint64))
    for actual, expected in zip(results[1][0], results[0][0], strict=True):
        assert actual.nnz == expected.nnz
        columns = np.arange(actual.shape[1])
        np.testing.assert_array_equal(actual[:, columns].toarray().view(np.uint64),
                                      expected[:, columns].toarray().view(np.uint64))
        if not packed:
            for name in ["data", "indices", "indptr"]:
                a, b = getattr(actual, name), getattr(expected, name)
                np.testing.assert_array_equal(a.view(np.uint64) if name == "data" else a,
                                              b.view(np.uint64) if name == "data" else b)
