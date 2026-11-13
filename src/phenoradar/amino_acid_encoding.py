"""Encode fold matrices while fitting training-only site statistics."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any

import numpy as np

from phenoradar.resource_budget import memory_budget

ENCODING_MEMORY_LIMIT_BYTES = 8 * 1024**3

if TYPE_CHECKING:
    from phenoradar.amino_acid import AATransform, AlignmentStore
    from phenoradar.config.schema import AminoAcidConfig, AminoAcidSiteFilter


@dataclass
class EncodedBlock:
    matrices: list[Any]
    shift: np.ndarray | None = None


def _compact_mass(values: np.ndarray, sites: np.ndarray, residues: np.ndarray,
                  n_sites: int) -> np.ndarray:
    """Keep the original 20-slot sum except for provably clipped 0/1 masses."""
    result = np.zeros((len(values), n_sites), dtype=float)
    if not n_sites:
        return result
    counts = np.bincount(sites, minlength=n_sites)
    site_steps = np.diff(sites)
    ordered = np.all((site_steps > 0) | ((site_steps == 0) & (np.diff(residues) > 0)))
    fallback = np.ones(n_sites, dtype=bool)
    if (np.all(counts > 0) and ordered
            and values.min(initial=0) >= 0 and values.max(initial=1) <= 1):
        offsets = np.r_[0, np.cumsum(counts)]
        approximate = np.add.reduceat(values, offsets[:-1], axis=1, dtype=np.float64)
        # At most 20 nonnegative float32 terms: this guard exceeds the float64
        # summation error. Boundary and partial masses use the original sum.
        clipped = np.abs(approximate - 1) < 1e-7 - 64 * np.finfo(float).eps
        fallback = np.any(~clipped & (approximate != 0), axis=0)
        result[clipped] = 1
    if fallback.any():
        local = np.cumsum(fallback) - 1
        keep = fallback[sites]
        dense = np.zeros((len(values), int(fallback.sum()), 20), dtype=np.float32)
        dense[:, local[sites[keep]], residues[keep]] = values[:, keep]
        mass = np.sum(dense, axis=2, dtype=np.float64)
        result[:, fallback] = np.where(np.isclose(mass, 1, rtol=0, atol=1e-7), 1, mass)
    return result


class FoldEncoding:
    """Append whole F-order column blocks; validation never determines the schema."""

    def __init__(self, groups: list[list[str]]) -> None:
        self.species = list(dict.fromkeys(s for group in groups for s in group))
        lookup = {s: i for i, s in enumerate(self.species)}
        self.rows = [np.array([lookup[s] for s in group], dtype=int) for group in groups]
        self.stack = ExitStack()
        self.files = [self.stack.enter_context(TemporaryFile(prefix="phenoradar_aa_encoded_"))  # noqa: SIM115
                      for _ in groups]
        self.width = 0
        self.capacities = [0] * len(groups)
        self.memory_limit = min(ENCODING_MEMORY_LIMIT_BYTES, memory_budget() // 4)
        self.chunks: list[list[np.ndarray]] = [[] for _ in groups]
        self.spilled = False

    def fork(self) -> FoldEncoding:
        return FoldEncoding([self.species])

    def block(self) -> EncodedBlock:
        return EncodedBlock(self.matrices())

    def extend(self, block: EncodedBlock) -> None:
        matrix = block.matrices[0]
        for start in range(0, matrix.shape[1], 4096):
            self.append(matrix[:, start:start + 4096])

    def add(self, raw: np.ndarray, mean: np.ndarray,
            known: np.ndarray, active: np.ndarray) -> None:
        q = np.nan_to_num(raw, nan=0.0)
        mass = np.sum(q * known[None, :, :], axis=2, dtype=np.float64)
        mass = np.where(np.isclose(mass, 1, rtol=0, atol=1e-7), 1, mass)
        columns = np.flatnonzero(active.ravel())
        values = q.reshape(len(self.species), -1)[:, columns].astype(float)
        values -= mass[:, columns // 20] * mean[active][None, :]
        self.append(values)

    def add_compact(self, values: np.ndarray, sites: np.ndarray, residues: np.ndarray,
                    eligible: np.ndarray, mean: np.ndarray, known: np.ndarray,
                    active: np.ndarray) -> None:
        keep = eligible[sites]
        local = np.cumsum(eligible) - 1
        q = np.nan_to_num(values[:, keep], copy=False)
        q *= known[sites[keep], residues[keep]][None, :]
        q = q.astype(np.float32, copy=False)
        mass = _compact_mass(q, local[sites[keep]], residues[keep], int(eligible.sum()))
        selected = keep & active[sites, residues]
        from phenoradar.amino_acid_kernels import center, native

        if native is not None and hasattr(self, "_append_csc"):
            encoded = center(values, np.flatnonzero(selected), local[sites[selected]],
                             mean[sites[selected], residues[selected]], mass)
            from phenoradar.amino_acid_kernels import append

            append(self, encoded)
            return
        encoded = np.nan_to_num(values[:, selected], copy=False).astype(float)
        encoded -= mass[:, local[sites[selected]]] * mean[sites[selected], residues[selected]]
        self.append(encoded)

    def _selected(self, values: np.ndarray, rows: np.ndarray) -> np.ndarray:
        if len(rows) and np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
            return np.asarray(values[rows[0]:rows[0] + len(rows)])
        return np.asarray(values[rows])

    def _write(self, index: int, values: np.ndarray) -> None:
        if not values.size:
            return
        handle = self.files[index]
        n, width = values.shape
        offset = self.width * n * 8
        required = offset + values.size * 8
        if required > self.capacities[index]:
            self.capacities[index] = max(required, self.capacities[index] * 2, 1024**2)
            handle.truncate(self.capacities[index])
        target = np.memmap(handle, mode="r+", dtype=float, shape=(n, width),
                           offset=offset, order="F")
        target[:] = values
        del target

    def append(self, values: np.ndarray) -> None:
        width = values.shape[1]
        if not width:
            return
        required = (self.width + width) * sum(map(len, self.rows)) * 8
        if not self.spilled and required <= self.memory_limit:
            for rows, chunks in zip(self.rows, self.chunks, strict=True):
                chunks.append(np.array(self._selected(values, rows), order="F", copy=True))
            self.width += width
            return
        if not self.spilled:
            previous, self.chunks = self.chunks, [[] for _ in self.rows]
            self.spilled = True
            self.width = 0
            for blocks in zip(*previous, strict=True):
                for index, block in enumerate(blocks):
                    self._write(index, block)
                self.width += blocks[0].shape[1]
            del previous
        for i, rows in enumerate(self.rows):
            self._write(i, self._selected(values, rows))
        self.width += width

    def matrices(self) -> list[np.ndarray]:
        result = []
        for i, (rows, handle) in enumerate(zip(self.rows, self.files, strict=True)):
            shape = (len(rows), self.width)
            if not self.spilled:
                matrix = np.empty(shape, order="F")
                offset = 0
                for block in self.chunks[i]:
                    matrix[:, offset:offset + block.shape[1]] = block
                    offset += block.shape[1]
                self.chunks[i].clear()
                result.append(matrix)
            else:
                handle.truncate(len(rows) * self.width * 8)
                handle.flush()
                result.append(np.memmap(handle, mode="r+", dtype=float, shape=shape, order="F")
                              if len(rows) and self.width else np.empty(shape, order="F"))
        return result

    def close(self) -> None:
        self.stack.close()


def fit_encoded(
    store: AlignmentStore, training: list[str], settings: AminoAcidConfig,
    targets: list[list[str]], **kwargs: Any,
) -> tuple[AATransform, list[np.ndarray]]:
    from phenoradar.amino_acid import fit_amino_acid

    if settings.ranked_site_filter.method != "none":
        fitted = fit_amino_acid(store, training, settings, **kwargs)
        return fitted, [fitted.matrix(store, species) for species in targets]
    encoding = FoldEncoding(targets)
    try:
        fitted = fit_amino_acid(store, training, settings, _encoding=encoding, **kwargs)
        return fitted, encoding.matrices()
    finally:
        encoding.close()


def compact_statistics(
    raw: np.ndarray, observed: np.ndarray, sites: np.ndarray, residues: np.ndarray,
    groups: np.ndarray, quality: AminoAcidSiteFilter, *, audit: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from phenoradar.amino_acid import _quality_summary

    n = observed.shape[1]
    known = np.zeros((n, 20), dtype=bool)
    known[sites, residues] = np.any(raw > 0, axis=0)
    eligible = known.sum(axis=1) >= quality.min_residue_states
    minor_samples = np.zeros(n, dtype=int)
    minor_groups = np.zeros(n, dtype=int)
    if audit:
        dense = np.full((len(raw), n, 20), np.nan, dtype=np.float32)
        dense[observed] = 0
        dense[:, sites, residues] = raw
        _, minor_samples, minor_groups = _quality_summary(dense, groups, quality)
    counts = observed.sum(axis=0)
    means = np.zeros((n, 20), dtype=float)
    active = np.zeros((n, 20), dtype=bool)
    selected = eligible[sites]
    if selected.any():
        state_sites = sites[selected]
        aa = residues[selected]
        # Two or more states per eligible site keep axis 0 as the same strided
        # float32 reduction used by the original (sample, site, 20) design.
        data = np.ascontiguousarray(np.nan_to_num(raw[:, selected], copy=False))
        denominator = counts[state_sites]
        mean = np.zeros(data.shape[1], dtype=float)
        np.divide(data.sum(axis=0, dtype=np.float64), denominator, out=mean,
                  where=denominator > 0)
        mean32 = np.zeros(data.shape[1], dtype=np.float32)
        np.divide(data.sum(axis=0, dtype=np.float32), denominator, out=mean32,
                  where=denominator > 0)
        data -= mean32
        data[~observed[:, state_sites]] = 0
        np.square(data, out=data)
        variance = np.zeros_like(mean32)
        np.divide(np.sum(data, axis=0, dtype=np.float32), denominator,
                  out=variance, where=denominator > 0)
        means[state_sites, aa] = mean
        active[state_sites, aa] = variance > 1e-12
    return eligible, minor_samples, minor_groups, counts / len(raw), means, known, active
