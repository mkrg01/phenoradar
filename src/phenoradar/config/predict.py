"""Prediction-only settings and compatibility with existing run configs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field, PositiveInt, ValidationError

from .io import ConfigError, _deep_merge_dicts, _load_yaml_mapping
from .schema import AppConfig, StrictModel, SummaryConfig


class PredictDataConfig(StrictModel):
    tpm_path: str = ""
    metadata_path: str | None = None
    tree_path: str | None = None
    species_col: str = Field(default="species", min_length=1)
    feature_col: str = Field(default="orthogroup", min_length=1)
    value_col: str = Field(default="tpm", min_length=1)
    trait_col: str = Field(default="C4", min_length=1)
    contrast_pair_col: str | None = "contrast_pair_id"


class PredictPreprocessConfig(StrictModel):
    # This is a memory guard, not a learned preprocessing choice.
    max_pivot_cells: PositiveInt = 50_000_000


class PredictRuntimeConfig(StrictModel):
    n_jobs: PositiveInt = 1


class PredictConfig(StrictModel):
    """Inputs and execution settings; learned state comes only from the bundle."""

    data: PredictDataConfig = Field(default_factory=PredictDataConfig)
    preprocess: PredictPreprocessConfig = Field(default_factory=PredictPreprocessConfig)
    runtime: PredictRuntimeConfig = Field(default_factory=PredictRuntimeConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)


def load_predict_config(
    config_paths: Sequence[Path],
    *,
    overrides: Mapping[str, Any] | None = None,
    require_tpm: bool = True,
) -> PredictConfig:
    """Resolve CLI > config > defaults, ignoring recognized training-only keys."""
    if len(config_paths) > 1:
        raise ConfigError("`--config` / `-c` can be specified at most once.")
    raw = _load_yaml_mapping(config_paths[0]) if config_paths else {}
    unknown = set(raw) - AppConfig.model_fields.keys()
    if unknown:
        raise ConfigError(f"Unknown config field(s): {', '.join(sorted(unknown))}")

    selected: dict[str, Any] = {}
    for section, field in PredictConfig.model_fields.items():
        if section not in raw:
            continue
        value = raw[section]
        if not isinstance(value, Mapping):
            raise ConfigError(f"{section} must be a mapping")
        legacy_type = AppConfig.model_fields[section].annotation
        predict_type = field.annotation
        assert isinstance(legacy_type, type) and issubclass(legacy_type, StrictModel)
        assert isinstance(predict_type, type) and issubclass(predict_type, StrictModel)
        unknown = set(value) - legacy_type.model_fields.keys()
        if unknown:
            raise ConfigError(f"Unknown {section} field(s): {', '.join(sorted(unknown))}")
        selected[section] = {
            key: item for key, item in value.items() if key in predict_type.model_fields
        }
    if overrides:
        selected = _deep_merge_dicts(selected, overrides)
    try:
        config = PredictConfig.model_validate(selected)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    if require_tpm and not config.data.tpm_path.strip():
        raise ConfigError("Provide --tpm-path or data.tpm_path in --config.")
    return config
