"""Shared metric-direction semantics."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, Literal, TypedDict, cast

MetricDirection = Literal["maximize", "minimize"]
MetricInput = Literal["probability", "predicted_label"]

EVALUATION_METRIC_CONTRACT_VERSION: Final = 1
FIXED_PROBABILITY_THRESHOLD_NAME: Final = "fixed_probability_threshold"
FIXED_PROBABILITY_THRESHOLD_POLICY: Final = "fixed_constant"
FIXED_PROBABILITY_THRESHOLD_VALUE: Final = 0.5
FIXED_PROBABILITY_THRESHOLD_DERIVED_FROM_CV: Final = False


class MetricContract(TypedDict):
    """Machine-readable definition of one metric key."""

    display_name: str
    implementation: str
    better_direction: MetricDirection
    input_type: MetricInput
    threshold_name: str | None
    threshold_value: float | None
    compatibility_note: str | None


_METRIC_CONTRACTS: Final[Mapping[str, MetricContract]] = MappingProxyType(
    {
        "mcc": {
            "display_name": "Matthews correlation coefficient",
            "implementation": "sklearn.metrics.matthews_corrcoef",
            "better_direction": "maximize",
            "input_type": "predicted_label",
            "threshold_name": FIXED_PROBABILITY_THRESHOLD_NAME,
            "threshold_value": FIXED_PROBABILITY_THRESHOLD_VALUE,
            "compatibility_note": None,
        },
        "balanced_accuracy": {
            "display_name": "Balanced accuracy",
            "implementation": "sklearn.metrics.balanced_accuracy_score",
            "better_direction": "maximize",
            "input_type": "predicted_label",
            "threshold_name": FIXED_PROBABILITY_THRESHOLD_NAME,
            "threshold_value": FIXED_PROBABILITY_THRESHOLD_VALUE,
            "compatibility_note": None,
        },
        "roc_auc": {
            "display_name": "ROC AUC",
            "implementation": "sklearn.metrics.roc_auc_score",
            "better_direction": "maximize",
            "input_type": "probability",
            "threshold_name": None,
            "threshold_value": None,
            "compatibility_note": None,
        },
        "pr_auc": {
            "display_name": "Average Precision",
            "implementation": "sklearn.metrics.average_precision_score",
            "better_direction": "maximize",
            "input_type": "probability",
            "threshold_name": None,
            "threshold_value": None,
            "compatibility_note": (
                "The pr_auc key is retained for compatibility; its value is Average "
                "Precision, not trapezoidal precision-recall curve area."
            ),
        },
        "brier": {
            "display_name": "Brier score loss",
            "implementation": "sklearn.metrics.brier_score_loss",
            "better_direction": "minimize",
            "input_type": "probability",
            "threshold_name": None,
            "threshold_value": None,
            "compatibility_note": None,
        },
        "log_loss": {
            "display_name": "Log loss",
            "implementation": "sklearn.metrics.log_loss",
            "better_direction": "minimize",
            "input_type": "probability",
            "threshold_name": None,
            "threshold_value": None,
            "compatibility_note": None,
        },
    }
)

METRIC_DIRECTIONS: Final[Mapping[str, MetricDirection]] = MappingProxyType(
    {name: contract["better_direction"] for name, contract in _METRIC_CONTRACTS.items()}
)


def metric_contract(metric_name: str) -> MetricContract:
    """Return a detached machine-readable contract for one metric key."""
    try:
        return cast(MetricContract, dict(_METRIC_CONTRACTS[metric_name]))
    except KeyError as exc:
        raise ValueError(f"Unsupported metric contract: {metric_name}") from exc


def evaluation_metric_contract() -> dict[str, Any]:
    """Return the complete evaluation metric and classification-threshold contract."""
    return {
        "metric_contract_version": EVALUATION_METRIC_CONTRACT_VERSION,
        "classification_threshold": {
            "threshold_name": FIXED_PROBABILITY_THRESHOLD_NAME,
            "threshold_value": FIXED_PROBABILITY_THRESHOLD_VALUE,
            "policy": FIXED_PROBABILITY_THRESHOLD_POLICY,
            "derived_from_cv": FIXED_PROBABILITY_THRESHOLD_DERIVED_FROM_CV,
        },
        "metrics": {name: dict(contract) for name, contract in _METRIC_CONTRACTS.items()},
    }


def evaluation_metric_contract_rows() -> list[dict[str, Any]]:
    """Return tabular rows for the evaluation contract artifact."""
    rows: list[dict[str, Any]] = []
    for metric_name, contract in _METRIC_CONTRACTS.items():
        rows.append(
            {
                "metric_contract_version": EVALUATION_METRIC_CONTRACT_VERSION,
                "metric_name": metric_name,
                **dict(contract),
            }
        )
    return rows


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
