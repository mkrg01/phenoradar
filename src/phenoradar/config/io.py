"""Config file loading, deep merge, and serialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml.nodes import MappingNode, ScalarNode

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


def _schema_variants(
    schema: dict[str, Any], definitions: dict[str, Any]
) -> list[dict[str, Any]]:
    if "$ref" in schema:
        return _schema_variants(definitions[schema["$ref"].rsplit("/", 1)[-1]], definitions)
    if "anyOf" in schema:
        return [
            variant
            for member in schema["anyOf"]
            for variant in _schema_variants(member, definitions)
        ]
    return [schema]


def _schema_value_comment(variants: list[dict[str, Any]]) -> str | None:
    choices: list[str] = []
    for variant in variants:
        if "enum" in variant:
            values = variant["enum"]
        elif "const" in variant:
            values = [variant["const"]]
        elif variant.get("type") == "boolean":
            values = [True, False]
        elif variant.get("type") == "null":
            values = [None]
        else:
            types = list(dict.fromkeys(member.get("type", "any") for member in variants))
            return "type: " + " or ".join(types) if "null" in types else None
        for value in values:
            if value is None:
                choices.append("null")
            elif isinstance(value, bool):
                choices.append(str(value).lower())
            else:
                choices.append(str(value))
    return "choices: " + ", ".join(dict.fromkeys(choices)) if choices else None


def serialize_resolved_config(config: AppConfig, *, include_internal: bool = True) -> str:
    """Serialize config with choices in comments, optionally omitting internal state."""
    payload = config.model_dump(mode="python")
    if not include_internal:
        payload["sampling"].pop("group_subsample_repeat_index", None)
    serialized = yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    schema = type(config).model_json_schema()
    definitions = schema.get("$defs", {})
    inline_comments: dict[int, str] = {}
    preceding_comments: dict[int, str] = {}

    def annotate(node: MappingNode, variants: list[dict[str, Any]]) -> None:
        for key, value in node.value:
            field_variants = [
                expanded
                for variant in variants
                if isinstance(
                    field_schema := variant.get("properties", {}).get(
                        key.value, variant.get("additionalProperties")
                    ),
                    dict,
                )
                for expanded in _schema_variants(field_schema, definitions)
            ]
            if isinstance(value, MappingNode):
                annotate(value, field_variants)
            elif isinstance(value, ScalarNode):
                comment = _schema_value_comment(field_variants)
                if comment is not None:
                    if key.start_mark.line == value.end_mark.line:
                        inline_comments[key.start_mark.line] = comment
                    else:
                        preceding_comments[key.start_mark.line] = comment

    root = yaml.compose(serialized, Loader=yaml.SafeLoader)
    if isinstance(root, MappingNode):
        annotate(root, [schema])
    lines: list[str] = []
    for index, line in enumerate(serialized.splitlines()):
        if index in preceding_comments:
            indent = len(line) - len(line.lstrip())
            lines.append(" " * indent + "# " + preceding_comments[index])
        if index in inline_comments:
            line += "  # " + inline_comments[index]
        lines.append(line)
    return "\n".join(lines) + "\n"


def write_resolved_config(
    config: AppConfig, output_path: Path, *, include_internal: bool = True
) -> None:
    """Write resolved YAML config to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        serialize_resolved_config(config, include_internal=include_internal), encoding="utf-8"
    )
