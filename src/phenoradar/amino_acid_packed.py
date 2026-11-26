"""Exact column blocks for amino-acid matrices that exceed glmnet's CSC limit."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import sparse

from phenoradar.amino_acid_kernels import native, packed_csc

INDEX_LIMIT = np.iinfo(np.int32).max


def _csc_bytes(matrix: sparse.csc_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


@dataclass
class PackedColumns:
    """Store up to three nonzero values per compressed column, in CSC row order."""

    indptr: np.ndarray
    mapping: np.ndarray
    values: np.ndarray
    planes: np.ndarray
    fallback: sparse.csc_matrix
    _scan_plan: tuple[Any, tuple[np.ndarray, ...]] | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    @property
    def shape(self) -> tuple[int, int]:
        return self.fallback.shape[0], len(self.mapping)

    @property
    def nbytes(self) -> int:
        return (_csc_bytes(self.fallback) + self.indptr.nbytes + self.mapping.nbytes
                + self.values.nbytes + self.planes.nbytes)

    @classmethod
    def from_csc(cls, matrix: sparse.csc_matrix) -> PackedColumns:
        height, width = matrix.shape
        if max(height, width, matrix.nnz) >= INDEX_LIMIT or not matrix.has_canonical_format:
            raise ValueError("AA packing requires a small canonical CSC block")
        mapping = np.empty(width, dtype=np.int32)
        palette = np.empty((width, 3))
        planes = np.empty((width, 2, (height + 63) // 64), dtype=np.uint64)
        count = native.pack_csc_columns(
            height, matrix.data, matrix.indices, matrix.indptr, mapping, palette, planes,
        )
        fallback = matrix[:, mapping < 0] if count else matrix
        return cls(matrix.indptr, mapping, palette[:count].copy(), planes[:count].copy(), fallback)

    @classmethod
    def from_masks(
        cls, ones: np.ndarray, mass: np.ndarray, sites: np.ndarray,
        sources: np.ndarray, categories: np.ndarray, species_count: int,
    ) -> PackedColumns:
        height, width = len(sources), len(sites)
        palette = np.ascontiguousarray(categories.T)
        words = (height + 63) // 64
        counts = np.empty(width, dtype=np.int32)
        planes = np.empty((width, 2, words), dtype=np.uint64)
        inputs = [np.ascontiguousarray(a) for a in (ones, mass, sites, sources)]
        native.pack_mask_columns(*inputs, palette, planes, counts)
        keep = 12 * counts.astype(np.int64) > 16 * words + 24
        skip = np.flatnonzero(~keep)
        ptr = _pointers(counts)
        if len(skip):
            arrays = packed_csc(
                ones[skip], mass, sites[skip], sources, categories[:, skip], species_count,
            )
            if arrays is None:
                raise ValueError("AA binary block exceeds the native CSC builder's capacity")
            # The native builder allocates worst-case capacity. Retain only populated entries.
            data, indices, indptr = arrays
            fallback = sparse.csc_matrix((data.copy(), indices.copy(), indptr),
                                         shape=(height, len(skip)))
        else:
            fallback = sparse.csc_matrix((height, 0), dtype=float)
        mapping = np.empty(width, dtype=np.int32)
        mapping[keep] = np.arange(keep.sum(), dtype=np.int32)
        mapping[~keep] = -1 - np.arange(len(skip), dtype=np.int32)
        return cls(ptr, mapping, palette[keep], planes[keep], fallback)

    def interleave(self, other: PackedColumns, order: np.ndarray) -> PackedColumns:
        mapping = other.mapping.copy()
        mapping[mapping >= 0] += len(self.values)
        mapping[mapping < 0] -= self.fallback.shape[1]
        return PackedColumns(
            _pointers(np.r_[np.diff(self.indptr), np.diff(other.indptr)][order]),
            np.r_[self.mapping, mapping][order],
            np.concatenate([self.values, other.values]),
            np.concatenate([self.planes, other.planes]),
            sparse.hstack([self.fallback, other.fallback], format="csc"),
        )

    def _buffers(self) -> tuple[np.ndarray, ...]:
        return (self.mapping, self.values, self.planes, self.fallback.data,
                self.fallback.indices, self.fallback.indptr)

    def _prepare_scan(self) -> None:
        structural = (self.mapping, self.planes, self.fallback.indices, self.fallback.indptr)
        owners = tuple(array.tobytes() for array in structural)
        plan = native.prepare_packed_scan(self.shape[0], len(self.values), *owners)
        mapping, planes, rows, offsets = (
            np.frombuffer(owner, dtype=array.dtype).reshape(array.shape)
            for array, owner in zip(structural, owners, strict=True)
        )
        # Replace only this block's views. The original CSC matrix may have a caller.
        fallback = copy(self.fallback)
        fallback.indices, fallback.indptr = rows, offsets
        self.mapping, self.planes, self.fallback = mapping, planes, fallback
        self._scan_plan = (plan, (mapping, planes, rows, offsets))

    def gradient(self, residual: np.ndarray, out: np.ndarray, *, fused: bool) -> None:
        scan = getattr(native, "scan_validated_packed_columns", None)
        if scan is not None and len(residual) == self.shape[0]:
            if self._scan_plan is None:
                self._prepare_scan()
            assert self._scan_plan is not None
            plan, arrays = self._scan_plan
            mapping, planes, rows, offsets = arrays
            if (self.mapping is mapping and self.planes is planes
                    and self.fallback.indices is rows and self.fallback.indptr is offsets):
                scan(plan, fused, self.values, self.fallback.data, residual, out)
                return
        native.scan_packed_columns(fused, *self._buffers(), residual, out)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_scan_plan", None)
        return state

    def gather(
        self, columns: np.ndarray, indptr: np.ndarray, data: np.ndarray, indices: np.ndarray,
    ) -> None:
        native.gather_packed_columns(
            self.shape[0], *self._buffers(), columns, indptr, data, indices
        )


def _pointers(counts: np.ndarray) -> np.ndarray:
    if counts.sum(dtype=np.int64) >= INDEX_LIMIT:
        raise ValueError("Selected AA columns exceed the local CSC index limit")
    ptr = np.empty(len(counts) + 1, dtype=np.int32)
    ptr[0] = 0
    np.cumsum(counts, out=ptr[1:])
    return ptr


@lru_cache(maxsize=1)
def packed_arithmetic() -> bool | None:
    """Match SciPy's sequential sparse accumulation, with or without FMA."""
    if not all(hasattr(native, name) for name in (
        "pack_csc_columns", "pack_mask_columns", "scan_packed_columns", "gather_packed_columns",
    )):
        return None
    rng = np.random.default_rng(12)
    values = rng.choice([0., .13, -.21, .5], size=(65, 12))
    values[:, :3] *= rng.random((65, 3))
    matrix = sparse.csc_matrix(values)
    block = PackedColumns.from_csc(matrix)
    residual = rng.standard_normal(matrix.shape[0])
    result = np.empty(matrix.shape[1])
    expected = (matrix.T @ residual).view(np.uint64)
    for fused in (False, True):
        block.gradient(residual, result, fused=fused)
        if np.array_equal(expected, result.view(np.uint64)):
            return fused
    return None


class PackedAAMatrix:
    """A fold matrix built from bounded blocks; the logical offsets are 64-bit."""

    def __init__(self, rows: int, blocks: list[PackedColumns], *, fused: bool) -> None:
        self.blocks = tuple(blocks)
        self.fused = fused
        self.boundaries = np.r_[0, np.cumsum([block.shape[1] for block in blocks], dtype=np.int64)]
        self.shape = (rows, int(self.boundaries[-1]))
        self.indptr = np.empty(self.shape[1] + 1, dtype=np.int64)
        self.indptr[0] = 0
        for left, right, block in zip(
            self.boundaries[:-1], self.boundaries[1:], blocks, strict=True
        ):
            np.add(block.indptr[1:], self.indptr[left], out=self.indptr[left + 1:right + 1],
                   dtype=np.int64)
        self.nnz = int(self.indptr[-1])
        self.nbytes = self.indptr.nbytes + self.boundaries.nbytes + sum(b.nbytes for b in blocks)

    def gradient(self, residual: np.ndarray, out: np.ndarray) -> None:
        from phenoradar.amino_acid_gradient_parallel import scan_blocks

        residual = np.ascontiguousarray(residual, dtype=float)
        if not scan_blocks(self, residual, out):
            for left, right, block in zip(self.boundaries[:-1], self.boundaries[1:],
                                          self.blocks, strict=True):
                block.gradient(residual, out[left:right], fused=self.fused)
        np.abs(out, out=out)

    def to_csc(self) -> sparse.csc_matrix:
        if self.nnz >= INDEX_LIMIT:
            raise ValueError("AA fold exceeds the native CSC index limit")
        data = np.empty(self.nnz)
        indices = np.empty(self.nnz, dtype=np.int32)
        for left, right, block in zip(self.boundaries[:-1], self.boundaries[1:],
                                      self.blocks, strict=True):
            start, stop = self.indptr[left], self.indptr[right]
            block.gather(np.arange(right - left, dtype=np.int64), block.indptr,
                         data[start:stop], indices[start:stop])
        return sparse.csc_matrix((data, indices, self.indptr.astype(np.int32)), shape=self.shape)

    def select_rows(self, rows: np.ndarray) -> PackedAAMatrix:
        """Retain canonical row order when zero sample weights remove rows."""
        blocks = []
        for block in self.blocks:
            width = block.shape[1]
            size = int(block.indptr[-1])
            data = np.empty(size)
            indices = np.empty(size, dtype=np.int32)
            block.gather(np.arange(width, dtype=np.int64), block.indptr, data, indices)
            local = sparse.csc_matrix((data, indices, block.indptr), shape=block.shape)
            blocks.append(PackedColumns.from_csc(local.tocsr()[rows].tocsc()))
        return PackedAAMatrix(int(rows.sum()), blocks, fused=self.fused)

    def __getitem__(self, key: tuple[slice, slice | np.ndarray]) -> sparse.csc_matrix:
        rows, columns = key
        if rows != slice(None):
            raise ValueError("Packed AA column selection requires all rows")
        if isinstance(columns, slice):
            columns = np.arange(*columns.indices(self.shape[1]), dtype=np.int64)
        else:
            columns = np.ascontiguousarray(columns, dtype=np.int64)
        if (columns.ndim != 1 or np.any(columns < 0) or np.any(columns >= self.shape[1])
                or np.any(columns[1:] <= columns[:-1])):
            raise ValueError("Expected sorted, unique AA column indices")
        ptr = _pointers(self.indptr[columns + 1] - self.indptr[columns])
        data = np.empty(int(ptr[-1]))
        indices = np.empty(len(data), dtype=np.int32)
        which = np.searchsorted(self.boundaries[1:], columns, side="right")
        cuts = np.r_[0, 1 + np.flatnonzero(which[1:] != which[:-1]), len(columns)]
        for left, right in zip(cuts[:-1], cuts[1:], strict=True):
            if left == right:
                continue
            index = which[left]
            start, stop = ptr[left], ptr[right]
            self.blocks[index].gather(columns[left:right] - self.boundaries[index],
                                      ptr[left:right + 1] - start,
                                      data[start:stop], indices[start:stop])
        return sparse.csc_matrix((data, indices, ptr), shape=(self.shape[0], len(columns)))
