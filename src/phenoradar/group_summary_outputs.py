"""Write stage-level group summaries identically for every feature mode."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from phenoradar.abstention import abstention_summary
from phenoradar.config import AppConfig, PredictConfig
from phenoradar.figures import FigureError, write_group_probability_figure
from phenoradar.group_summary import GroupSummaryError, build_group_summary_artifacts
from phenoradar.timing import TimingRecorder


def write_group_summary_artifacts(
    *,
    run_dir: Path,
    stage: str,
    predictions: pl.DataFrame,
    source_table_name: str,
    config: AppConfig | PredictConfig,
    timing_recorder: TimingRecorder | None = None,
) -> list[str]:
    summary_started = None if timing_recorder is None else timing_recorder.start()
    if config.data.metadata_path is None:
        return []
    try:
        artifacts = build_group_summary_artifacts(
            predictions=predictions,
            metadata_path=Path(config.data.metadata_path),
            species_col=config.data.species_col,
            group_col=config.summary.group_col,
            source_table_name=source_table_name,
        )
    except GroupSummaryError as exc:
        return [str(exc)]

    tables_dir = run_dir / stage / "tables"
    figures_dir = run_dir / stage / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts.summary.write_csv(
        tables_dir / f"group_summary_{artifacts.suffix}.tsv",
        separator="\t",
        float_precision=8,
        null_value="NA",
    )
    if "decision_status" in artifacts.predictions.columns:
        abstention_summary(artifacts.predictions).write_csv(
            tables_dir / f"abstention_by_{artifacts.suffix}.tsv",
            separator="\t",
            null_value="NA",
        )
    if timing_recorder is not None and summary_started is not None:
        timing_recorder.record_since(
            summary_started, scope="artifacts", stage=f"{stage}_group_summary_tables"
        )
    figure_started = None if timing_recorder is None else timing_recorder.start()
    try:
        write_group_probability_figure(
            grouped_predictions=artifacts.predictions,
            out_path=figures_dir / f"probability_by_{artifacts.suffix}.svg",
            group_label=artifacts.group_label,
            source_table_name=source_table_name,
            figure_name=f"probability_by_{artifacts.suffix}.svg",
        )
    except FigureError as exc:
        return [str(exc)]
    finally:
        if timing_recorder is not None and figure_started is not None:
            timing_recorder.record_since(
                figure_started, scope="artifacts", stage=f"{stage}_group_summary_figure"
            )
    return []
