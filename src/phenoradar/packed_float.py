"""Lossless float32 blocks: bit planes for zero/one/NaN, exact exceptional bits."""
from __future__ import annotations

import hashlib
import inspect

import numpy as np

from phenoradar.amino_acid_presence import _COUNTS


def encode(values: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    ones, missing = bits == 0x3F800000, bits == 0x7FC00000
    other = ~((bits == 0) | ones | missing)
    planes = np.packbits(np.stack((ones, missing, other), axis=1), axis=2, bitorder="little")
    exceptional = values[other].view(np.uint8)
    if planes.nbytes + exceptional.nbytes >= values.nbytes:
        return values.reshape(-1).view(np.uint8), False
    return np.concatenate((planes.reshape(-1), exceptional)), True


def take(data: np.ndarray, nrows: int, width: int, rows: np.ndarray,
         columns: np.ndarray, *, packed: bool) -> np.ndarray:
    if not packed:
        return np.asarray(data.view(np.float32).reshape(nrows, width)[np.ix_(rows, columns)])
    from phenoradar.amino_acid_kernels import decode

    decoded = decode(data, nrows, width, rows, columns)
    if decoded is not None:
        return decoded
    nbytes = (width + 7) // 8
    end = nrows * 3 * nbytes
    planes = data[:end].reshape(nrows, 3, nbytes)
    byte = columns // 8
    bit = np.left_shift(np.uint8(1), columns % 8).astype(np.uint8)
    selected = planes[rows[:, None, None], np.arange(3)[None, :, None], byte[None, None, :]]
    flags = (selected & bit[None, None, :]) != 0
    result = np.zeros((len(rows), len(columns)), dtype=np.float32)
    result[flags[:, 0]] = 1
    result[flags[:, 1]] = np.float32(np.nan)
    rr, cc = np.nonzero(flags[:, 2])
    if len(rr):
        counts = _COUNTS[planes[:, 2]].astype(np.int64)
        starts = np.r_[0, np.cumsum(counts.sum(axis=1))]
        prefix = np.c_[np.zeros(nrows, dtype=np.int64), counts.cumsum(axis=1)]
        before = _COUNTS[selected[rr, 2, cc] & (bit[cc] - np.uint8(1))]
        rank = starts[rows[rr]] + prefix[rows[rr], byte[cc]] + before
        result[rr, cc] = data[end:].view(np.float32)[rank]
    return result


# Decoder optimizations do not change immutable payloads produced by this encoder.
FORMAT_SHA = hashlib.sha256(inspect.getsource(encode).encode()).hexdigest()
