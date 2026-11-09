"""Exact statistics and shifted CSC columns for cached 0/1/missing compositions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from phenoradar.amino_acid_encoding import EncodedBlock, compact_statistics
from phenoradar.amino_acid_frequency_cache import _mapped_file
from phenoradar.amino_acid_presence import _COUNTS, any_presence, count_presence

if TYPE_CHECKING:
    from phenoradar.amino_acid import AlignmentOG, AlignmentStore
    from phenoradar.amino_acid_encoding import FoldEncoding
    from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache
    from phenoradar.config.schema import AminoAcidSiteFilter


@dataclass
class _Binary:
    sites: np.ndarray
    residues: np.ndarray
    ones: np.ndarray
    observed: np.ndarray
    eligible: np.ndarray
    species_rows: dict[str, int]

    def rows(self, species: list[str]) -> np.ndarray:
        return np.array([self.species_rows[s] for s in species if s in self.species_rows],
                        dtype=np.int64)

    def gather(self, packed: np.ndarray, species: list[str]) -> np.ndarray:
        result = np.zeros((len(packed), len(species)), dtype=bool)
        present = [(i, self.species_rows[s]) for i, s in enumerate(species)
                   if s in self.species_rows]
        if present:
            targets, source = np.array(present).T
            bit = np.left_shift(np.uint8(1), source % 8).astype(np.uint8)
            result[:, targets] = (packed[:, source // 8]
                                  & bit) != 0
        return result


def _site_or(bits: np.ndarray, sites: np.ndarray, n: int) -> np.ndarray:
    """State rows are sorted by site; reduce whole byte rows in one native pass."""
    result = np.zeros((n, bits.shape[1]), dtype=np.uint8)
    if len(sites):
        starts = np.r_[0, np.flatnonzero(np.diff(sites)) + 1]
        result[sites[starts]] = np.bitwise_or.reduceat(bits, starts, axis=0)
    return result


def _read(cache: AminoAcidFrequencyCache, columns: np.ndarray) -> _Binary:
    sites, residues, parts = cache._parts(columns)
    nbytes = (len(cache.species) + 7) // 8
    ones = np.zeros((len(sites), nbytes), dtype=np.uint8)
    observed = np.zeros((len(columns), nbytes), dtype=np.uint8)
    bad = np.zeros(len(columns), dtype=bool)
    for chunk, selected, source_rows, features, source_states in parts:
        if not chunk.packed or not len(features):
            bad[selected] = True
            continue
        data = _mapped_file(chunk.path, cache._file_lengths[chunk.path])[chunk.start:chunk.stop]
        nrows = chunk.columns + int(chunk.offsets[-1])
        planes = data[:nrows * 3 * nbytes].reshape(nrows, 3, nbytes)
        state_rows = chunk.columns + source_states
        ones[features] = planes[state_rows, 0]
        observed[selected] = ~planes[source_rows, 1]
        exception = np.any(planes[state_rows, 2], axis=1)
        # Malformed/noncanonical missing values take the ordinary float path too.
        mismatch = np.any(planes[state_rows, 1] != planes[source_rows, 1][
            np.searchsorted(selected, sites[features])], axis=1)
        local_sites = sites[features]
        starts = np.r_[0, np.flatnonzero(np.diff(local_sites)) + 1]
        bad[local_sites[starts]] |= np.logical_or.reduceat(exception | mismatch, starts)
    padding = len(cache.species) % 8
    if padding:
        observed[:, -1] &= np.uint8((1 << padding) - 1)
    mass = _site_or(ones, sites, len(columns))
    # Integer masses above one cannot use one-hot categories; preserve the general path.
    positive = np.bincount(sites, weights=_COUNTS[ones].sum(axis=1), minlength=len(columns))
    bad |= positive != _COUNTS[mass].sum(axis=1)
    return _Binary(sites, residues, ones, observed, ~bad, cache.species_rows)


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
        q = binary.gather(ones, species)
        m = binary.gather(mass, species)[sites]
        nonzero = np.where(q, category_values[2, :, None] != 0,
                           np.where(m, category_values[0, :, None] != 0,
                                    category_values[1, :, None] != 0))
        columns, indices = np.nonzero(nonzero)
        values = np.where(q[columns, indices], category_values[2, columns],
                          np.where(m[columns, indices], category_values[0, columns],
                                   category_values[1, columns]))
        indptr = np.r_[0, np.cumsum(np.bincount(columns, minlength=len(mu)))]
        matrices.append(sparse.csc_matrix((values, indices, indptr),
                                          shape=(len(species), len(mu))))
    return EncodedBlock(matrices, shift)


def encode_cached_block(store: AlignmentStore, og: AlignmentOG, columns: np.ndarray,
                        species: list[str], groups: np.ndarray, quality: AminoAcidSiteFilter,
                        encoding: FoldEncoding, *, weighted: bool,
                        audit: bool) -> tuple[Any, ...] | None:
    from phenoradar.amino_acid_sparse import SparseFoldEncoding

    if not isinstance(encoding, SparseFoldEncoding) or len(species) > 2**24:
        return None
    cache, _, _ = store._frequency_cache(og, weighted)
    raw = _read(cache, columns)
    if not raw.eligible.any():
        return None
    n = len(columns)
    arrays = [np.zeros(n, dtype=bool), np.zeros(n, dtype=int), np.zeros(n, dtype=int),
              np.zeros(n), np.zeros((n, 20)), np.zeros((n, 20), dtype=bool),
              np.zeros((n, 20), dtype=bool)]
    blocks = []
    positions = []
    for is_binary in [True, False]:
        selected = np.flatnonzero(raw.eligible == is_binary)
        if not len(selected):
            continue
        keep = np.isin(raw.sites, selected)
        site = np.searchsorted(selected, raw.sites[keep])
        if is_binary:
            binary = _Binary(site, raw.residues[keep], raw.ones[keep], raw.observed[selected],
                             np.ones(len(selected), dtype=bool), raw.species_rows)
            stats = _statistics(binary, species, groups, quality, audit=audit)
            block = _encode(binary, species, encoding.groups, *stats[4:])
        else:
            values, _, observed, state_sites, residues = store.compact_frequencies(
                og, encoding.species, columns[selected], weighted=weighted,
            )
            stats = compact_statistics(encoding._selected(values, encoding.training),
                                       encoding._selected(observed, encoding.training),
                                       state_sites, residues, groups, quality, audit=audit)
            child = encoding.fork()
            try:
                child.add_compact(values, state_sites, residues, stats[0], *stats[4:])
                block = child.block()
            finally:
                child.close()
        for dest, source in zip(arrays, stats, strict=True):
            dest[selected] = source
        blocks.append(block)
        positions.append((selected[:, None] * 20 + np.arange(20))[stats[-1]])
    order = np.argsort(np.concatenate(positions))
    matrices = [sparse.hstack([b.matrices[i] for b in blocks], format="csc")[:, order]
                for i in range(len(encoding.groups))]
    shift = np.concatenate([b.shift for b in blocks])[order]
    encoding.extend(EncodedBlock(matrices, shift))
    return tuple(arrays)
