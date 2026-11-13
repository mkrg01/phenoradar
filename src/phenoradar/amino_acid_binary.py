"""Exact statistics and shifted CSC columns for cached 0/1/missing compositions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from phenoradar.amino_acid_encoding import EncodedBlock, compact_statistics
from phenoradar.amino_acid_frequency_cache import _mapped_file
from phenoradar.amino_acid_kernels import packed_csc
from phenoradar.amino_acid_presence import _COUNTS, any_presence, count_presence

if TYPE_CHECKING:
    from phenoradar.amino_acid import AlignmentOG, AlignmentStore
    from phenoradar.amino_acid_encoding import FoldEncoding
    from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache, _Chunk
    from phenoradar.config.schema import AminoAcidSiteFilter


@dataclass
class _Binary:
    sites: np.ndarray
    residues: np.ndarray
    ones: np.ndarray
    observed: np.ndarray
    species_rows: dict[str, int]

    def rows(self, species: list[str]) -> np.ndarray:
        return np.array([self.species_rows[s] for s in species if s in self.species_rows],
                        dtype=np.int64)


def _site_or(bits: np.ndarray, sites: np.ndarray, n: int) -> np.ndarray:
    """State rows are sorted by site; reduce whole byte rows in one native pass."""
    result = np.zeros((n, bits.shape[1]), dtype=np.uint8)
    if len(sites):
        starts = np.r_[0, np.flatnonzero(np.diff(sites)) + 1]
        result[sites[starts]] = np.bitwise_or.reduceat(bits, starts, axis=0)
    return result


def _planes(cache: AminoAcidFrequencyCache, chunk: _Chunk) -> np.ndarray:
    nbytes = (len(cache.species) + 7) // 8
    data = _mapped_file(chunk.path, cache._file_lengths[chunk.path])[chunk.start:chunk.stop]
    nrows = chunk.columns + int(chunk.offsets[-1])
    return data[:nrows * 3 * nbytes].reshape(nrows, 3, nbytes)


def _binary_columns(cache: AminoAcidFrequencyCache, columns: np.ndarray) -> np.ndarray:
    """Reuse the encoding classification of immutable chunks across training folds."""
    indices = np.searchsorted(cache.columns, columns)
    chunk_ids = cache._chunk_ids[indices]
    eligible = np.zeros(len(columns), dtype=bool)
    for chunk_id in np.unique(chunk_ids):
        chunk = cache._chunks[int(chunk_id)]
        if not chunk.packed:
            continue
        key = chunk.path, chunk.start
        valid = cache._binary_eligibility.get(key)
        if valid is None:
            counts = np.diff(chunk.offsets)
            populated = counts > 0
            valid = np.zeros(chunk.columns, dtype=bool)
            if populated.any():
                planes = _planes(cache, chunk)
                states = planes[chunk.columns:]
                state_sites = np.repeat(np.arange(chunk.columns), counts)
                invalid = np.any((states[:, 2] != 0) |
                                 (states[:, 1] != planes[state_sites, 1]), axis=1)
                starts = chunk.offsets[:-1][populated]
                ones = states[:, 0]
                mass = np.bitwise_or.reduceat(ones, starts, axis=0)
                positives = np.add.reduceat(_COUNTS[ones].sum(axis=1), starts)
                valid[populated] = (~np.logical_or.reduceat(invalid, starts)
                                    & (positives == _COUNTS[mass].sum(axis=1)))
            cache._binary_eligibility[key] = valid
        selected = chunk_ids == chunk_id
        eligible[selected] = valid[cache._rows[indices[selected]]]
    return eligible


def _read(cache: AminoAcidFrequencyCache, columns: np.ndarray) -> _Binary:
    sites, residues, parts = cache._parts(columns)
    nbytes = (len(cache.species) + 7) // 8
    ones = np.empty((len(sites), nbytes), dtype=np.uint8)
    observed = np.empty((len(columns), nbytes), dtype=np.uint8)
    for chunk, selected, source_rows, features, source_states in parts:
        planes = _planes(cache, chunk)
        ones[features] = planes[chunk.columns + source_states, 0]
        observed[selected] = ~planes[source_rows, 1]
    padding = len(cache.species) % 8
    if padding:
        observed[:, -1] &= np.uint8((1 << padding) - 1)
    return _Binary(sites, residues, ones, observed, cache.species_rows)


def _statistics(binary: _Binary, species: list[str], groups: np.ndarray,
                quality: AminoAcidSiteFilter, *, audit: bool) -> tuple[Any, ...]:
    rows = binary.rows(species)
    n = len(binary.observed)
    counts = count_presence(binary.observed, rows)
    positives = count_presence(binary.ones, rows)
    known = np.zeros((n, 20), dtype=bool)
    known[binary.sites, binary.residues] = positives > 0
    eligible = known.sum(axis=1) >= quality.min_residue_states
    mean = np.zeros((n, 20), dtype=float)
    active = np.zeros((n, 20), dtype=bool)
    keep = eligible[binary.sites]
    sites, aa = binary.sites[keep], binary.residues[keep]
    mean[sites, aa] = positives[keep] / counts[sites]
    # For at most 2**24 samples, a varying binary column's float32 variance is
    # far above 1e-12; constant columns have exactly zero variance.
    active[sites, aa] = (positives[keep] > 0) & (positives[keep] < counts[sites])
    minor = np.zeros(n, dtype=int)
    minor_groups = np.zeros(n, dtype=int)
    if audit:
        state_counts = np.zeros((n, 20), dtype=np.int32)
        state_counts[binary.sites, binary.residues] = positives
        major = state_counts.argmax(axis=1)
        nonmajor = binary.residues != major[binary.sites]
        bits = _site_or(binary.ones[nonmajor], binary.sites[nonmajor], n)
        minor = count_presence(bits, rows).astype(int)
        for group in np.unique(groups):
            local = [s for s, g in zip(species, groups, strict=True) if g == group]
            minor_groups += any_presence(bits, binary.rows(local))
    return eligible, minor, minor_groups, counts / len(species), mean, known, active


def _encode(binary: _Binary, training: list[str], targets: list[list[str]],
            mean: np.ndarray, known: np.ndarray, active: np.ndarray) -> EncodedBlock:
    selected = active[binary.sites, binary.residues]
    sites, aa = binary.sites[selected], binary.residues[selected]
    ones = binary.ones[selected]
    keep = known[binary.sites, binary.residues]
    mass = _site_or(binary.ones[keep], binary.sites[keep], len(binary.observed))
    rows = binary.rows(training)
    positive = count_presence(ones, rows)
    mass_count = count_presence(mass, rows)[sites]
    mu = mean[sites, aa]
    category_values = np.stack((-mu, np.zeros(len(mu)), 1.0 - mu))
    category_counts = np.stack((mass_count - positive, len(training) - mass_count, positive))
    shift = -category_values[category_counts.argmax(axis=0), np.arange(len(mu))]
    category_values += shift
    matrices = []
    for species in targets:
        sources = np.array([binary.species_rows.get(s, -1) for s in species], dtype=np.int64)
        shape = (len(species), len(mu))
        encoded = packed_csc(ones, mass, sites, sources, category_values, len(binary.species_rows))
        if encoded is not None:
            matrices.append(sparse.csc_matrix(encoded, shape=shape))
            continue
        present = sources >= 0
        source = sources[present]
        bit = np.left_shift(np.uint8(1), source % 8).astype(np.uint8)
        q = np.zeros((len(ones), len(species)), dtype=bool)
        q[:, present] = (ones[:, source // 8] & bit) != 0
        m = np.zeros((len(mass), len(species)), dtype=bool)
        m[:, present] = (mass[:, source // 8] & bit) != 0
        m = m[sites]
        nonzero = np.where(q, category_values[2, :, None] != 0,
                           np.where(m, category_values[0, :, None] != 0,
                                    category_values[1, :, None] != 0))
        columns, indices = np.nonzero(nonzero)
        values = np.where(q[columns, indices], category_values[2, columns],
                          np.where(m[columns, indices], category_values[0, columns],
                                   category_values[1, columns]))
        indptr = np.r_[0, np.cumsum(np.bincount(columns, minlength=len(mu)))]
        matrices.append(sparse.csc_matrix((values, indices, indptr), shape=shape))
    return EncodedBlock(matrices, shift)


def encode_cached_block(store: AlignmentStore, og: AlignmentOG, columns: np.ndarray,
                        species: list[str], groups: np.ndarray, quality: AminoAcidSiteFilter,
                        encoding: FoldEncoding, *, weighted: bool,
                        audit: bool) -> tuple[Any, ...] | None:
    from phenoradar.amino_acid_sparse import SparseFoldEncoding

    if not isinstance(encoding, SparseFoldEncoding) or len(species) > 2**24:
        return None
    cache, _, _ = store._frequency_cache(og, weighted)
    mask = _binary_columns(cache, columns)
    if not mask.any():
        return None
    binary = _read(cache, columns[mask])
    binary_stats = _statistics(binary, species, groups, quality, audit=audit)
    binary_block = _encode(binary, species, encoding.groups, *binary_stats[4:])
    if mask.all():
        encoding.extend(binary_block)
        return binary_stats

    values, _, observed, sites, residues = store.compact_frequencies(
        og, encoding.species, columns[~mask], weighted=weighted,
    )
    fractional_stats = compact_statistics(encoding._selected(values, encoding.training),
                                          encoding._selected(observed, encoding.training),
                                          sites, residues, groups, quality, audit=audit)
    child = encoding.fork()
    try:
        child.add_compact(values, sites, residues, fractional_stats[0], *fractional_stats[4:])
        fractional_block = child.block()
    finally:
        child.close()

    n = len(columns)
    arrays = [np.zeros(n, dtype=bool), np.zeros(n, dtype=int), np.zeros(n, dtype=int),
              np.zeros(n), np.zeros((n, 20)), np.zeros((n, 20), dtype=bool),
              np.zeros((n, 20), dtype=bool)]
    for dest, bit_stats, float_stats in zip(arrays, binary_stats, fractional_stats, strict=True):
        dest[mask] = bit_stats
        dest[~mask] = float_stats
    positions = np.arange(n * 20).reshape(n, 20)
    order = np.argsort(np.r_[positions[mask][binary_stats[-1]],
                            positions[~mask][fractional_stats[-1]]])
    matrices = [sparse.hstack((bit_matrix, float_matrix), format="csc")[:, order]
                for bit_matrix, float_matrix in zip(binary_block.matrices,
                                                    fractional_block.matrices, strict=True)]
    shift = np.r_[binary_block.shift, fractional_block.shift][order]
    encoding.extend(EncodedBlock(matrices, shift))
    return tuple(arrays)
