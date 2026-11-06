from __future__ import annotations

import numpy as np
import pytest

from phenoradar.config import AppConfig
from phenoradar.cv import (
    _apply_expression_transform_for_config,
    _compact_prediction_entry,
    _transform_target_for_selected_features,
)
from phenoradar.glmnet import GlmnetLogisticRegression
from phenoradar.missing_expression import NeutralStandardScaler


@pytest.mark.parametrize("zero_as_missing", [False, True])
@pytest.mark.parametrize("intercept_only", [False, True])
def test_compact_predictions_preserve_missing_values_and_original_model(
    zero_as_missing: bool,
    intercept_only: bool,
) -> None:
    config = AppConfig.model_validate(
        {
            "data": {"metadata_path": "unused.tsv", "tpm_path": "unused.tsv"},
            "preprocess": {
                "expression": {
                    "expression_transform": {"method": "log1p"},
                    "absent_feature_fill": "nan",
                    "missing_expression": {"method": "neutral", "zero_as_missing": zero_as_missing},
                }
            },
        }
    )
    names = ["inactive", "tiny", "negative", "unused", "positive"]
    train = np.array(
        [[1.0, 3.0, 5.0, 0.0, 2.0], [3.0, 5.0, 7.0, 4.0, 0.0], [5.0, 0.0, 9.0, 6.0, 8.0]]
    )
    target = np.array(
        [[0.0, 3.0, 0.0, 9.0, 10.0], [np.nan, 5.0, 7.0, np.nan, 0.0], [3.0, 1e10, 9.0, 6.0, 2.0]]
    )
    scaler = NeutralStandardScaler().fit(_apply_expression_transform_for_config(config, train))
    model = GlmnetLogisticRegression()
    model.coef_ = np.zeros(5) if intercept_only else np.array([0.0, 1e-15, -2.3, 0.0, 3.2])
    model.intercept_ = -0.3
    model.n_features_in_ = 5
    expected = model.predict_proba(
        _transform_target_for_selected_features(
            config,
            target,
            names,
            names,
            scaler,
        )
    )
    original_coef = model.coef_.copy()
    original_mean = scaler.mean_.copy()
    entry = _compact_prediction_entry(config, names, scaler, model)
    actual = entry.model.predict_proba(
        _transform_target_for_selected_features(
            config,
            target,
            names,
            entry.feature_names,
            entry.scaler,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(model.coef_, original_coef)
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    assert model.n_features_in_ == scaler.n_features_in_ == 5
    assert len(entry.feature_names) == (1 if intercept_only else 3)
    if not intercept_only:
        assert "tiny" in entry.feature_names


@pytest.mark.parametrize("transform", ["sample_rank", "sample_percentile_rank"])
def test_rank_predictions_keep_the_complete_feature_row(transform: str) -> None:
    config = AppConfig.model_validate(
        {
            "data": {"metadata_path": "unused.tsv", "tpm_path": "unused.tsv"},
            "preprocess": {"expression": {"expression_transform": {"method": transform}}},
        }
    )
    model = GlmnetLogisticRegression()
    model.coef_ = np.array([0.0, 1.0])
    names = ["zero", "active"]
    entry = _compact_prediction_entry(config, names, None, model)
    assert entry.feature_names == names
    assert entry.model is model
