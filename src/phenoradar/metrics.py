"""Shared metric-direction semantics."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

MetricDirection = Literal["maximize", "minimize"]

METRIC_DIRECTIONS: Final[Mapping[str, MetricDirection]] = MappingProxyType(
    {
        "mcc": "maximize",
        "balanced_accuracy": "maximize",
        "roc_auc": "maximize",
        "pr_auc": "maximize",
        "brier": "minimize",
        "log_loss": "minimize",
    }
)


def metric_direction(metric_name: str) -> MetricDirection:
    """Return the optimization direction for a supported metric."""
    try:
        return METRIC_DIRECTIONS[metric_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric direction: {metric_name}") from exc


def metric_higher_is_better(metric_name: str) -> bool:
    """Return whether larger values indicate better performance."""
    return metric_direction(metric_name) == "maximize"


def metric_sort_value(metric_name: str, metric_value: float) -> float:
    """Return an ascending sort value that places better metrics first."""
    return -metric_value if metric_higher_is_better(metric_name) else metric_value
