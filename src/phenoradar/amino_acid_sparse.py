"""Training-only affine shifts and memory-bounded CSC encoding for AA models."""
from __future__ import annotations

from contextlib import ExitStack
from tempfile import TemporaryFile
from typing import Any

import numpy as np
from scipy import sparse

from phenoradar.amino_acid import AATransform, AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_encoding import EncodedBlock, FoldEncoding
from phenoradar.amino_acid_resources import matrix_buffer_limit, record_sparse_usage
from phenoradar.config.schema import AminoAcidConfig


def modal_shift(values: np.ndarray) -> np.ndarray:
    """Choose an exact column mode; no rounding, thresholding or label access."""
    result = np.empty(values.shape[1], dtype=values.dtype)
    majority = np.zeros(values.shape[1], dtype=bool)
    if len(values):
        candidate = values[0]
        # A strict majority is the unique mode. Keep signed-zero columns on the
        # original sort path so even their zero sign and tie rules are preserved.
        majority = ((candidate != 0)
                    & (np.count_nonzero(values == candidate, axis=0) > len(values) // 2))
        result[majority] = -candidate[majority]
    if majority.all():
        return result
    ordered = np.sort(values[:, ~majority], axis=0)
    rows = np.arange(len(ordered))[:, None]
    starts = np.zeros(ordered.shape, dtype=np.int64)
    starts[1:] = np.where(ordered[1:] != ordered[:-1], rows[1:], 0)
    np.maximum.accumulate(starts, axis=0, out=starts)
    lengths = rows - starts + 1
    ends = np.argmax(lengths, axis=0)
    result[~majority] = -ordered[ends, np.arange(ordered.shape[1])]
    return result


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
        shift = modal_shift(self._selected(values, self.training))
        self.shifts.append(shift)
        for i, rows in enumerate(self.rows):
            # Transposed blocks enumerate nonzeros in CSC order directly, without
            # a COO conversion, column sort, or whole-OG dense assembly.
            shifted = (self._selected(values, rows) + shift).T
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
                del chunks
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
    targets: list[list[str]], *, prefix: list[np.ndarray] | None = None, **kwargs: Any,
) -> tuple[AATransform, list[Any], np.ndarray]:
    """Encode both modes with one CSC assembly, optionally starting with expression.

    Prefix columns are already scaled and retain their original basis. Appending
    AA blocks directly avoids copying the complete AA matrix in a later hstack.
    """
    encoding = SparseFoldEncoding(targets, training, buffer_bytes=matrix_buffer_limit(store))
    try:
        if prefix is not None:
            if len(prefix) != len(targets):
                raise ValueError("One prefix matrix is required per target group")
            width = prefix[0].shape[1]
            for i, (values, species) in enumerate(zip(prefix, targets, strict=True)):
                if values.shape != (len(species), width):
                    raise ValueError("Prefix shapes must match target rows and feature width")
                matrix = sparse.csc_matrix(values, dtype=np.float64)
                encoding._append_csc(i, matrix.data, matrix.indices, matrix.indptr[1:])
            encoding.width = width
            encoding.shifts.append(np.zeros(width))
        if settings.ranked_site_filter.method == "none":
            fitted = fit_amino_acid(store, training, settings, _encoding=encoding, **kwargs)
        else:
            fitted = fit_amino_acid(store, training, settings, **kwargs)
            encoding.append(fitted.matrix(store, encoding.species))
        matrices = encoding.matrices()
        record_sparse_usage(store, fitted, matrices)
        return fitted, matrices, encoding.shift
    finally:
        encoding.close()


def training_information(
    fitted: AATransform, store: AlignmentStore, species: list[str],
    matrix: Any, shift: np.ndarray, *, standard_deviation: bool, offset: int = 0,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    """Compute optional interpretation metadata in bounded site blocks."""
    representatives: list[dict[str, Any]] = []
    scales = []
    for indices, raw, coverage in fitted.raw_blocks(store, species, representatives):
        sites = [fitted.sites[j] for j in indices]
        counts = [int(site.active.sum()) for site in sites]
        width = sum(counts)
        if standard_deviation:
            known = np.stack([site.known for site in sites])
            mass = np.asarray(np.sum(np.nan_to_num(raw) * known, axis=2, dtype=np.float64))
            centered = matrix[:, offset:offset + width].toarray() - shift[offset:offset + width]
            columns = np.repeat(np.arange(len(sites)), counts)
            observed = (coverage * mass) > 0
            # The same observed-only population SD, reduced once per site block.
            # This affects interpretation only; model inputs remain unchanged.
            scales.extend(np.std(centered, axis=0, where=observed[:, columns]))
        offset += width
    return (np.asarray(scales) if standard_deviation else None), representatives
