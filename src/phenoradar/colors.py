"""Shared colors for PhenoRadar figures."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

CONFUSION_GROUP_ORDER: Final[tuple[str, ...]] = ("TP", "FN", "TN", "FP")
CONFUSION_GROUP_COLORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "TP": "#0072B2",
        "FN": "#CC79A7",
        "TN": "#009E73",
        "FP": "#E69F00",
    }
)
CONFUSION_GROUP_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "TP": "true positive",
        "FN": "false negative",
        "TN": "true negative",
        "FP": "false positive",
    }
)
