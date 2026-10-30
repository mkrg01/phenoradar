"""Prediction inputs; learned preprocessing comes from the model bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field, PositiveInt, ValidationError

from .io import ConfigError, _deep_merge_dicts, _load_yaml_mapping
from .schema import (
    AppConfig,
    FiguresConfig,
    PhylogeneticImputationConfig,
    StrictModel,
    SummaryConfig,
)


class PredictDataConfig(StrictModel):
    feature_mode: str = "expression"
    alignment_dir: str | None = None
    sequence_tpm_path: str | None = None
    sequence_tpm_cache_dir: str | None = "runs/.phenoradar_cache/sequence_tpm"
    tpm_path: str = ""
    metadata_path: str | None = None
    tree_path: str | None = None
    orthogroup_annotation_path: str | None = None
    species_col: str = Field(default="species", min_length=1)
    feature_col: str = Field(default="orthogroup", min_length=1)
    value_col: str = Field(default="tpm", min_length=1)
    trait_col: str = Field(default="C4", min_length=1)
    contrast_pair_col: str | None = "contrast_pair_id"


class PredictRuntimeConfig(StrictModel):
    n_jobs: PositiveInt = 1


class PredictConfig(StrictModel):
    """Inputs and execution settings; learned state comes only from the bundle."""

    data: PredictDataConfig = Field(default_factory=PredictDataConfig)
    runtime: PredictRuntimeConfig = Field(default_factory=PredictRuntimeConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    figures: FiguresConfig = Field(default_factory=FiguresConfig)
    phylogenetic_imputation: PhylogeneticImputationConfig = Field(
        default_factory=PhylogeneticImputationConfig
    )


def _check_config_keys(
    value: Mapping[str, Any], model: type[StrictModel], prefix: str = ""
) -> None:
    unknown = set(value) - model.model_fields.keys()
    if unknown:
        context = f"{prefix} " if prefix else "config "
        raise ConfigError(f"Unknown {context}field(s): {', '.join(sorted(unknown))}")
    for key, item in value.items():
        nested = model.model_fields[key].annotation
        if (
            isinstance(item, Mapping)
            and isinstance(nested, type)
            and issubclass(nested, StrictModel)
        ):
            _check_config_keys(item, nested, f"{prefix}.{key}" if prefix else key)


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
    _check_config_keys(raw, AppConfig)

    selected: dict[str, Any] = {}
    for section, field in PredictConfig.model_fields.items():
        if section not in raw:
            continue
        value = raw[section]
        if not isinstance(value, Mapping):
            raise ConfigError(f"{section} must be a mapping")
        predict_type = field.annotation
        assert isinstance(predict_type, type) and issubclass(predict_type, StrictModel)
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
