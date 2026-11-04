"""Training-only affine shifts and disk-backed CSC encoding for AA models."""
from __future__ import annotations

from contextlib import ExitStack
from tempfile import TemporaryFile
from typing import Any

import numpy as np
from scipy import sparse

from phenoradar.amino_acid import AATransform, AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_encoding import FoldEncoding
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

    def __init__(self, groups: list[list[str]], training: list[str]) -> None:
        self.species = list(dict.fromkeys(s for group in [training, *groups] for s in group))
        lookup = {s: i for i, s in enumerate(self.species)}
        self.rows = [np.array([lookup[s] for s in group], dtype=int) for group in groups]
        self.training = np.array([lookup[s] for s in training], dtype=int)
        self.stack = ExitStack()
        self.handles = [[self.stack.enter_context(TemporaryFile(  # noqa: SIM115
            prefix="phenoradar_aa_csc_")) for _ in range(3)] for _ in groups]
        self.nonzero = [0] * len(groups)
        self.width = 0
        self.shifts: list[np.ndarray] = []

    def append(self, values: np.ndarray) -> None:
        if not values.shape[1]:
            return
        shift = modal_shift(values[self.training])
        self.shifts.append(shift)
        for i, rows in enumerate(self.rows):
            block = sparse.csc_matrix(values[rows] + shift)
            count = self.nonzero[i] + block.nnz
            if count >= np.iinfo(np.int32).max:
                raise ValueError("AA sparse matrix exceeds glmnet's 32-bit nonzero limit")
            for handle, array in zip(self.handles[i], [block.data, block.indices.astype(np.int32),
                                                      np.asarray(block.indptr[1:] + self.nonzero[i],
                                                                 dtype=np.int32)],
                                     strict=True):
                handle.write(array.tobytes())
            self.nonzero[i] = count
        self.width += values.shape[1]

    @property
    def shift(self) -> np.ndarray:
        return np.concatenate(self.shifts) if self.shifts else np.empty(0)

    def matrices(self) -> list[Any]:
        result = []
        for rows, handles, count in zip(self.rows, self.handles, self.nonzero, strict=True):
            for handle in handles:
                handle.flush()
            if not count:
                result.append(sparse.csc_matrix((len(rows), self.width)))
                continue
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
