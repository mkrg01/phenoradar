from __future__ import annotations

import polars as pl
import pytest

from phenoradar.feature_stability import (
    FeatureStabilityError,
    build_feature_stability_tables,
)


def _retained(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scope": ["outer_fold"] * len(rows),
            "fold_id": [fold_id for fold_id, _feature in rows],
            "sample_set_id": [0] * len(rows),
            "feature": [feature for _fold_id, feature in rows],
        }
    )


def test_build_feature_stability_tables_for_signed_linear_coefficients() -> None:
    fold_ids = ["1", "2", "3"]
    features = ["a", "b", "c"]
    importance = pl.DataFrame(
        {
            "fold_id": [fold for fold in fold_ids for _feature in features],
            "feature": features * len(fold_ids),
            "importance_mean": [0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
        }
    )
    coefficients = pl.DataFrame(
        {
            "fold_id": [fold for fold in fold_ids for _feature in features],
            "feature": features * len(fold_ids),
            "coef_mean": [1.0, -1.0, 0.0, 2.0, 1.0, 0.0, 0.0, -2.0, 3.0],
            "method": ["coef_signed"] * 9,
            "reason": ["NA"] * 9,
        }
    )
    retained = _retained([("1", "a"), ("1", "b"), ("2", "a"), ("2", "b"), ("3", "b"), ("3", "c")])

    artifacts = build_feature_stability_tables(
        feature_importance_by_fold=importance,
        coefficients_by_fold=coefficients,
        retained_features=retained,
    )

    by_feature = {row["feature"]: row for row in artifacts.by_feature.iter_rows(named=True)}
    assert by_feature["a"]["n_retained_folds"] == 2
    assert by_feature["a"]["n_nonzero_folds"] == 2
    assert by_feature["a"]["selection_frequency"] == pytest.approx(2 / 3)
    assert by_feature["a"]["selection_frequency_when_retained"] == pytest.approx(1.0)
    assert by_feature["a"]["dominant_sign"] == "positive"
    assert by_feature["a"]["sign_agreement_rate"] == pytest.approx(1.0)

    assert by_feature["b"]["n_positive_folds"] == 1
    assert by_feature["b"]["n_negative_folds"] == 2
    assert by_feature["b"]["dominant_sign"] == "negative"
    assert by_feature["b"]["sign_agreement_rate"] == pytest.approx(2 / 3)

    assert by_feature["c"]["n_nonzero_folds"] == 1
    assert by_feature["c"]["sign_agreement_rate"] is None
    assert by_feature["c"]["sign_reason"] == "selected_in_fewer_than_two_folds"

    pair_values = {
        (row["fold_id_a"], row["fold_id_b"]): row["jaccard"]
        for row in artifacts.by_fold_pair.iter_rows(named=True)
    }
    assert pair_values == pytest.approx({("1", "2"): 1.0, ("1", "3"): 1 / 3, ("2", "3"): 1 / 3})
    summary = artifacts.summary.row(0, named=True)
    assert summary["n_outer_folds"] == 3
    assert summary["n_fold_pairs"] == 3
    assert summary["n_features_ever_selected"] == 3
    assert summary["n_features_selected_in_half_folds"] == 2
    assert summary["n_features_selected_in_all_folds"] == 1
    assert summary["jaccard_mean"] == pytest.approx(5 / 9)
    assert summary["sign_available"] is True


def test_build_feature_stability_tables_uses_importance_for_random_forest() -> None:
    importance = pl.DataFrame(
        {
            "fold_id": ["1", "1", "2", "2"],
            "feature": ["a", "b", "a", "b"],
            "importance_mean": [0.7, 0.3, 0.0, 1.0],
        }
    )
    coefficients = pl.DataFrame(
        {
            "fold_id": ["1", "1", "2", "2"],
            "feature": ["a", "b", "a", "b"],
            "coef_mean": [None, None, None, None],
            "method": ["NA"] * 4,
            "reason": ["unsupported_model_non_linear"] * 4,
        }
    )
    retained = _retained([("1", "a"), ("1", "b"), ("2", "a"), ("2", "b")])

    artifacts = build_feature_stability_tables(
        feature_importance_by_fold=importance,
        coefficients_by_fold=coefficients,
        retained_features=retained,
    )

    by_feature = {row["feature"]: row for row in artifacts.by_feature.iter_rows(named=True)}
    assert by_feature["a"]["selection_frequency"] == pytest.approx(0.5)
    assert by_feature["b"]["selection_frequency"] == pytest.approx(1.0)
    assert by_feature["b"]["dominant_sign"] is None
    assert by_feature["b"]["sign_agreement_rate"] is None
    assert by_feature["b"]["sign_reason"] == "signed_coefficients_unavailable"
    assert artifacts.by_fold_pair.row(0, named=True)["jaccard"] == pytest.approx(0.5)
    assert artifacts.summary.row(0, named=True)["selection_method"] == ("feature_importance_gt_tol")
    assert artifacts.summary.row(0, named=True)["sign_available"] is False


def test_build_feature_stability_tables_rejects_duplicate_fold_feature_rows() -> None:
    importance = pl.DataFrame(
        {
            "fold_id": ["1", "1"],
            "feature": ["a", "a"],
            "importance_mean": [0.5, 0.5],
        }
    )
    coefficients = pl.DataFrame(
        {
            "fold_id": ["1"],
            "feature": ["a"],
            "coef_mean": [1.0],
            "method": ["coef_signed"],
            "reason": ["NA"],
        }
    )

    with pytest.raises(FeatureStabilityError, match="duplicate fold/feature"):
        build_feature_stability_tables(
            feature_importance_by_fold=importance,
            coefficients_by_fold=coefficients,
            retained_features=_retained([("1", "a")]),
        )
