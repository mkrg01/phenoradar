"""Prediction populations shared by SVG figures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

import polars as pl

from phenoradar.metrics import FIXED_PROBABILITY_THRESHOLD_VALUE

_SVG_NS = "http://www.w3.org/2000/svg"


def prediction_figure_populations(
    predictions: dict[str, pl.DataFrame],
) -> Iterator[tuple[str, dict[str, pl.DataFrame]]]:
    """Yield all-species and accepted-only plotting copies when abstention is present.

    All-species classification shows the raw fixed-threshold decision, including
    decisions that were withheld. Saved prediction tables retain selective labels.
    """
    if not any("decision_status" in frame.columns for frame in predictions.values()):
        yield "", predictions
        return
    for accepted_only in (False, True):
        frames = {}
        for name, frame in predictions.items():
            selected = (
                frame.filter(pl.col("decision_status") == "accepted")
                if accepted_only and "decision_status" in frame.columns
                else frame
            )
            selected = selected.drop("pred_label_selective", "decision_status", strict=False)
            # Tree annotations already contain nullable selective prediction labels.
            if "prob" in selected.columns:
                selected = selected.with_columns(
                    (pl.col("prob") >= FIXED_PROBABILITY_THRESHOLD_VALUE).cast(pl.Int64).alias(col)
                    for col in ("pred_label", "pred_label_fixed_threshold")
                    if col in selected.columns
                )
            frames[name] = selected
        yield "_accepted_only" if accepted_only else "", frames


def population_figure_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def write_population_message_svg(path: Path, message: str) -> None:
    """Make undefined populations explicit, including trees with too few tips."""
    root = ET.Element(
        f"{{{_SVG_NS}}}svg",
        {
            "width": "720px",
            "height": "180px",
            "viewBox": "0 0 720 180",
        },
    )
    ET.SubElement(
        root,
        f"{{{_SVG_NS}}}rect",
        {
            "width": "100%",
            "height": "100%",
            "fill": "white",
        },
    )
    text = ET.SubElement(
        root,
        f"{{{_SVG_NS}}}text",
        {
            "x": "15",
            "y": "65",
            "style": "font-family: sans-serif; font-size: 12px",
        },
    )
    text.text = message
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
