"""Training-only affine shifts and disk-backed CSC encoding for AA models."""
from __future__ import annotations

from contextlib import ExitStack
from tempfile import TemporaryFile
from typing import Any

import numpy as np
from scipy import sparse

from phenoradar.amino_acid import AATransform, AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_encoding import EncodedBlock, FoldEncoding
from phenoradar.config.schema import AminoAcidConfig


def modal_shift(values: np.ndarray) -> np.ndarray:
    """Choose an exact column mode; no rounding, thresholding or label access."""
    ordered = np.sort(values, axis=0)
    lengths = np.ones(ordered.shape, dtype=np.int64)
    for row in range(1, len(ordered)):
        lengths[row] = np.where(ordered[row] == ordered[row - 1], lengths[row - 1] + 1, 1)
    ends = np.argmax(lengths, axis=0)
    return np.asarray(-ordered[ends, np.arange(values.shape[1])])


class SparseFoldEncoding(FoldEncoding):
    """Build shifted CSC blocks without materializing a dense fold matrix.

    The returned design is Z=X+c. Every target uses c learned from training
    rows, including duplicated sampling rows. Fitted intercepts return to the
    original centered basis via b_X=b_Z+c@beta before saving or interpretation.
    """

    def __init__(self, groups: list[list[str]], training: list[str], *,
                 buffer_bytes: int = 0) -> None:
        self.groups, self.training_species = groups, training
        self.species = list(dict.fromkeys(s for group in [training, *groups] for s in group))
        lookup = {s: i for i, s in enumerate(self.species)}
        self.rows = [np.array([lookup[s] for s in group], dtype=int) for group in groups]
        self.training = np.array([lookup[s] for s in training], dtype=int)
        self.stack = ExitStack()
        self.handles: list[list[Any] | None] = [None for _ in groups]
        self.memory_limit = buffer_bytes
        self._buffered_chunks: list[list[list[np.ndarray]]] | None = (
            [[] for _ in groups] if buffer_bytes else None
        )
        self._buffered_bytes = 0
        self.nonzero = [0] * len(groups)
        self.width = 0
        self.shifts: list[np.ndarray] = []

    def fork(self) -> SparseFoldEncoding:
        return SparseFoldEncoding(self.groups, self.training_species, buffer_bytes=64 * 1024**2)

    def block(self) -> EncodedBlock:
        return EncodedBlock(self.matrices(), self.shift)

    def _write_arrays(self, index: int, arrays: list[np.ndarray]) -> None:
        handles = self.handles[index]
        if handles is None:
            handles = [self.stack.enter_context(TemporaryFile(  # noqa: SIM115
                prefix="phenoradar_aa_csc_")) for _ in range(3)]
            self.handles[index] = handles
        for handle, array in zip(handles, arrays, strict=True):
            handle.write(memoryview(np.ascontiguousarray(array)))

    def _append_csc(self, index: int, data: np.ndarray,
                    rows: np.ndarray, ends: np.ndarray) -> None:
        count = self.nonzero[index] + len(data)
        if count >= np.iinfo(np.int32).max:
            raise ValueError("AA sparse matrix exceeds glmnet's 32-bit nonzero limit")
        arrays = [data, np.asarray(rows, dtype=np.int32),
                  np.asarray(ends + self.nonzero[index], dtype=np.int32)]
        size = sum(array.nbytes for array in arrays)
        if self._buffered_chunks is not None and self._buffered_bytes + size > self.memory_limit:
            for i, blocks in enumerate(self._buffered_chunks):
                for block in blocks:
                    self._write_arrays(i, block)
            self._buffered_chunks = None
        if self._buffered_chunks is None:
            self._write_arrays(index, arrays)
        else:
            self._buffered_chunks[index].append(arrays)
            self._buffered_bytes += size
        self.nonzero[index] = count

    def extend(self, block: EncodedBlock) -> None:
        assert block.shift is not None
        for i, matrix in enumerate(block.matrices):
            self._append_csc(i, matrix.data, matrix.indices, matrix.indptr[1:])
        self.shifts.append(block.shift)
        self.width += len(block.shift)

    def append(self, values: np.ndarray) -> None:
        if not values.shape[1]:
            return
        shift = modal_shift(values[self.training])
        self.shifts.append(shift)
        for i, rows in enumerate(self.rows):
            # Transposed blocks enumerate nonzeros in CSC order directly, without
            # a COO conversion, column sort, or whole-OG dense assembly.
            shifted = (values[rows] + shift).T
            columns, indices = np.nonzero(shifted)
            ends = np.cumsum(np.bincount(columns, minlength=values.shape[1]), dtype=np.int64)
            self._append_csc(i, shifted[columns, indices], indices, ends)
        self.width += values.shape[1]

    @property
    def shift(self) -> np.ndarray:
        return np.concatenate(self.shifts) if self.shifts else np.empty(0)

    def matrices(self) -> list[Any]:
        result = []
        for i, (rows, handles, count) in enumerate(
            zip(self.rows, self.handles, self.nonzero, strict=True)
        ):
            if not count:
                result.append(sparse.csc_matrix((len(rows), self.width)))
                continue
            if self._buffered_chunks is not None:
                chunks = self._buffered_chunks[i]
                data, indices, ends = [np.concatenate([chunk[j] for chunk in chunks])
                                       for j in range(3)]
                self._buffered_chunks[i] = []
            else:
                assert handles is not None
                for handle in handles:
                    handle.flush()
                data = np.memmap(handles[0], mode="r", dtype=np.float64, shape=(count,))
                indices = np.memmap(handles[1], mode="r", dtype=np.int32, shape=(count,))
                ends = np.memmap(handles[2], mode="r", dtype=np.int32, shape=(self.width,))
            indptr = np.r_[np.int32(0), ends]
            result.append(sparse.csc_matrix((data, indices, indptr),
                                            shape=(len(rows), self.width), copy=False))
        return result


def fit_sparse_encoded(
    store: AlignmentStore, training: list[str], settings: AminoAcidConfig,
    targets: list[list[str]], **kwargs: Any,
) -> tuple[AATransform, list[Any], np.ndarray]:
    encoding = SparseFoldEncoding(targets, training)
    try:
        if settings.ranked_site_filter.method == "none":
            fitted = fit_amino_acid(store, training, settings, _encoding=encoding, **kwargs)
        else:
            fitted = fit_amino_acid(store, training, settings, **kwargs)
            encoding.append(fitted.matrix(store, encoding.species))
        return fitted, encoding.matrices(), encoding.shift
    finally:
        encoding.close()


def training_information(
    fitted: AATransform, store: AlignmentStore, species: list[str],
    matrix: Any, shift: np.ndarray, *, standard_deviation: bool,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    """Compute optional interpretation metadata in bounded site blocks."""
    representatives: list[dict[str, Any]] = []
    scales = []
    offset = 0
    for indices, raw, coverage in fitted.raw_blocks(store, species, representatives):
        sites = [fitted.sites[j] for j in indices]
        counts = [int(site.active.sum()) for site in sites]
        width = sum(counts)
        if standard_deviation:
            known = np.stack([site.known for site in sites])
            mass = np.asarray(np.sum(np.nan_to_num(raw) * known, axis=2, dtype=np.float64))
            centered = matrix[:, offset:offset + width].toarray() - shift[offset:offset + width]
            left = 0
            for j, count in enumerate(counts):
                scales.extend(np.std(centered[(coverage[:, j] * mass[:, j]) > 0,
                                               left:left + count], axis=0))
                left += count
        offset += width
    return (np.asarray(scales) if standard_deviation else None), representatives
