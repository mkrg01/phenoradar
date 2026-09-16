"""Observed-only scaling with an explicit neutral value for unknown expression."""

from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler


def mark_expression_observations(
    frame: pl.DataFrame,
    *,
    zero_as_missing: bool,
    value_col: str = "tpm",
    log_col: str = "log2_tpm_plus1",
) -> pl.DataFrame:
    """Keep source TPM, expose its observation status, and mask plotting values."""
    value = pl.col(value_col)
    missing = value.is_null() | ~value.is_finite()
    if zero_as_missing:
        missing = missing | (value == 0)
    result = frame.with_columns(missing.alias("is_missing"))
    if log_col in frame.columns:
        result = result.with_columns(
            pl.when(pl.col("is_missing")).then(None).otherwise(pl.col(log_col)).alias(log_col)
        )
    return result


def mask_expression(matrix: np.ndarray, *, zero_as_missing: bool) -> np.ndarray:
    """Keep raw input intact; mark uncertain zeros before any feature statistics."""
    values = np.asarray(matrix, dtype=float)
    if zero_as_missing:
        return np.where(values == 0.0, np.nan, values)
    return values


class NeutralStandardScaler(StandardScaler):  # type: ignore[misc]
    """Fit StandardScaler on observations, neutralize only missing transform entries.

    StandardScaler ignores NaNs in fit and preserves them in transform. Keeping
    this operation in the persisted scaler makes every prediction/evidence path
    use the exact same model input, including predictions after bundle reload.
    """

    def transform(self, X: Any, copy: bool | None = None) -> np.ndarray:
        missing = np.isnan(np.asarray(X, dtype=float))
        scaled = np.asarray(super().transform(X, copy=copy), dtype=float)
        result = np.where(missing, 0.0, scaled)
        if not np.isfinite(result).all():
            raise ValueError("Neutral standardization produced non-finite observed values")
        return result
