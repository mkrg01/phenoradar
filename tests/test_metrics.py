from __future__ import annotations

from typing import get_args

import pytest

from phenoradar.config.schema import SelectionMetricName
from phenoradar.metrics import (
    EVALUATION_METRIC_CONTRACT_VERSION,
    FIXED_PROBABILITY_THRESHOLD_NAME,
    FIXED_PROBABILITY_THRESHOLD_VALUE,
    METRIC_DIRECTIONS,
    evaluation_metric_contract,
    evaluation_metric_contract_rows,
    metric_contract,
    metric_direction,
    metric_sort_value,
)
from phenoradar.reporting import PrimaryMetric


def test_metric_direction_registry_covers_report_and_selection_metrics() -> None:
    configured_metrics = set(get_args(PrimaryMetric)) | set(get_args(SelectionMetricName))

    assert set(METRIC_DIRECTIONS) == configured_metrics


@pytest.mark.parametrize(
    ("metric_name", "direction"),
    [
        ("mcc", "maximize"),
        ("balanced_accuracy", "maximize"),
        ("roc_auc", "maximize"),
        ("pr_auc", "maximize"),
        ("brier", "minimize"),
        ("log_loss", "minimize"),
    ],
)
def test_metric_direction_is_explicit(metric_name: str, direction: str) -> None:
    assert metric_direction(metric_name) == direction


def test_metric_direction_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported metric direction"):
        metric_direction("unknown")


def test_metric_sort_value_changes_only_the_sort_key() -> None:
    assert metric_sort_value("mcc", -0.2) == 0.2
    assert metric_sort_value("brier", 0.2) == 0.2


def test_pr_auc_contract_explicitly_defines_average_precision_compatibility_key() -> None:
    contract = metric_contract("pr_auc")

    assert contract["display_name"] == "Average Precision"
    assert contract["implementation"] == "sklearn.metrics.average_precision_score"
    assert contract["input_type"] == "probability"
    assert contract["threshold_name"] is None
    assert contract["compatibility_note"] is not None
    assert "not trapezoidal" in contract["compatibility_note"]


def test_threshold_dependent_metric_contracts_share_fixed_threshold() -> None:
    for metric_name in ("mcc", "balanced_accuracy"):
        contract = metric_contract(metric_name)
        assert contract["threshold_name"] == FIXED_PROBABILITY_THRESHOLD_NAME
        assert contract["threshold_value"] == FIXED_PROBABILITY_THRESHOLD_VALUE

    complete_contract = evaluation_metric_contract()
    assert complete_contract["metric_contract_version"] == EVALUATION_METRIC_CONTRACT_VERSION
    assert complete_contract["classification_threshold"] == {
        "threshold_name": FIXED_PROBABILITY_THRESHOLD_NAME,
        "threshold_value": FIXED_PROBABILITY_THRESHOLD_VALUE,
        "policy": "fixed_constant",
        "derived_from_cv": False,
    }


def test_evaluation_metric_contract_rows_cover_direction_registry() -> None:
    rows = evaluation_metric_contract_rows()

    assert {str(row["metric_name"]) for row in rows} == set(METRIC_DIRECTIONS)
    assert {int(row["metric_contract_version"]) for row in rows} == {
        EVALUATION_METRIC_CONTRACT_VERSION
    }
