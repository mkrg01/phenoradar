from __future__ import annotations

from typing import get_args

import pytest

from phenoradar.config.schema import SelectionMetricName
from phenoradar.metrics import METRIC_DIRECTIONS, metric_direction, metric_sort_value
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
