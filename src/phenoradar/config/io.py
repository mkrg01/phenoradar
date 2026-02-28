"""Config file loading, deep merge, and serialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import AppConfig, ExecutionStage


class ConfigError(ValueError):
    """Raised when configuration loading or validation fails."""


def _deep_merge_dicts(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _deep_merge_dicts(dict(base_value), override_value)
            continue
        merged[key] = override_value
    return merged


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: {path}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Top-level YAML must be a mapping: {path}")
    return dict(raw)


def load_and_resolve_config(
    config_paths: Sequence[Path],
    execution_stage_override: ExecutionStage | None = None,
    *,
    allow_empty: bool = False,
) -> AppConfig:
    """Load one or more YAML files, deep merge them, and validate with Pydantic."""
    if not config_paths and not allow_empty:
        raise ConfigError("At least one config file must be provided")

    merged: dict[str, Any] = {}
    for config_path in config_paths:
        merged = _deep_merge_dicts(merged, _load_yaml_mapping(config_path))

    if execution_stage_override is not None:
        runtime: dict[str, Any] = {}
        runtime_raw = merged.get("runtime")
        if isinstance(runtime_raw, Mapping):
            runtime = dict(runtime_raw)
        runtime["execution_stage"] = execution_stage_override
        merged["runtime"] = runtime

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def serialize_resolved_config(config: AppConfig) -> str:
    """Serialize a validated config into deterministic YAML."""
    config_dict = config.model_dump(mode="python")
    return yaml.safe_dump(
        config_dict,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )


def write_resolved_config(config: AppConfig, output_path: Path) -> None:
    """Write resolved YAML config to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_resolved_config(config), encoding="utf-8")
