"""Assemble packed fold blocks directly from the frequency cache where possible."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from phenoradar.amino_acid_encoding import EncodedBlock
from phenoradar.amino_acid_packed import PackedAAMatrix, PackedColumns
from phenoradar.amino_acid_sparse import SparseFoldEncoding
from phenoradar.resource_budget import memory_budget


class PackedFoldEncoding(SparseFoldEncoding):
    def __init__(self, groups: list[list[str]], training: list[str], *,
                 fused: bool, budget: int | None = None) -> None:
        super().__init__(groups, training)
        self.fused = fused
        self.budget = int(memory_budget() * .65) if budget is None else budget
        self.blocks: list[list[PackedColumns]] = [[] for _ in groups]
        self.retained_bytes = 0

    def fork(self) -> PackedFoldEncoding:
        return PackedFoldEncoding(
            self.groups, self.training_species, fused=self.fused, budget=self.budget,
        )

    def _retain(self, index: int, block: PackedColumns) -> None:
        self.retained_bytes += block.nbytes + 8 * block.shape[1]
        if self.retained_bytes > self.budget:
            raise MemoryError("Packed AA fold exceeds its available memory budget")
        if block.shape[1]:
            self.blocks[index].append(block)

    def _append_csc(self, index: int, data: np.ndarray,
                    rows: np.ndarray, ends: np.ndarray) -> None:
        if not len(ends):
            return
        ptr = np.empty(len(ends) + 1, dtype=np.int32)
        ptr[0] = 0
        ptr[1:] = ends
        shape = (len(self.rows[index]), len(ends))
        # Trim the producer's worst-case arrays before retaining a CSC fallback.
        matrix = sparse.csc_matrix((data.copy(), rows.astype(np.int32), ptr), shape=shape)
        self._retain(index, PackedColumns.from_csc(matrix))

    def extend(self, block: EncodedBlock) -> None:
        assert block.shift is not None
        for index, matrix in enumerate(block.matrices):
            if isinstance(matrix, PackedAAMatrix):
                for local in matrix.blocks:
                    self._retain(index, local)
            elif isinstance(matrix, PackedColumns):
                self._retain(index, matrix)
            else:
                self._append_csc(index, matrix.data, matrix.indices, matrix.indptr[1:])
        self.width += len(block.shift)
        self.shifts.append(block.shift)

    def matrices(self) -> list[Any]:
        return [PackedAAMatrix(len(rows), blocks, fused=self.fused)
                for rows, blocks in zip(self.rows, self.blocks, strict=True)]

    def block(self) -> EncodedBlock:
        return EncodedBlock(self.matrices(), self.shift)
