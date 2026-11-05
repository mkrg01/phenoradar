from pathlib import Path

import numpy as np
import pytest

from phenoradar.amino_acid_presence import SpeciesPresenceCache, any_presence, count_presence


@pytest.mark.parametrize("samples,width", [
    (0, 17), (1, 1), (7, 9), (8, 8), (9, 513), (63, 127), (64, 128),
    (65, 257), (116, 1025), (129, 25), (7, 0), (17, 480003),
])
def test_species_bits_match_counts_with_padding_duplicates_and_empty_queries(
    tmp_path: Path, samples: int, width: int,
) -> None:
    rng = np.random.default_rng(161)
    present = rng.random((samples, width)) > .3
    source, target = tmp_path / "source.npy", tmp_path / "target.npy"
    np.save(source, np.packbits(present, axis=1))
    cache = SpeciesPresenceCache(target)
    packed = np.packbits(present.T, axis=1, bitorder="little")
    queries = [[], list(range(samples))]
    if samples:
        queries.extend([[samples - 1, 0, samples - 1], list(range(samples))[::-2]])
    for query in queries:
        rows = np.array(query, dtype=int)
        np.testing.assert_array_equal(cache.counts(("OG", False), source, width, rows),
                                      present[rows].sum(axis=0, dtype=np.int32))
        np.testing.assert_array_equal(count_presence(packed, rows),
                                      present[rows].sum(axis=0, dtype=np.int32))
        np.testing.assert_array_equal(any_presence(packed, rows), np.any(present[rows], axis=0))


def test_presence_extents_share_files_and_parallel_appends_are_complete(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(234)
    matrices = [rng.random((9, width)) > .2 for width in [17, 33, 65, 1001]]
    paths = [tmp_path / f"source_{i}.npy" for i in range(4)]
    for path, matrix in zip(paths, matrices, strict=True):
        np.save(path, np.packbits(matrix, axis=1))
    parent = SpeciesPresenceCache(tmp_path / "parent.bin")
    rows = np.array([8, 0, 8])
    parent.counts(("0", True), paths[0], 17, rows)
    original = parent.path.read_bytes()
    child = SpeciesPresenceCache(tmp_path / "child.bin", parent.entries)
    np.testing.assert_array_equal(child.counts(("0", True), paths[0], 17, rows),
                                  matrices[0][rows].sum(axis=0))
    assert not child.path.exists()

    def count(i):
        return child.counts((str(i), True), paths[i], matrices[i].shape[1], rows)

    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(count, [1, 2, 3]))
    for i, actual in enumerate(result, 1):
        np.testing.assert_array_equal(actual, matrices[i][rows].sum(axis=0))
    assert parent.path.read_bytes() == original
    assert len(parent.entries) == 1 and len(child.entries) == 4
