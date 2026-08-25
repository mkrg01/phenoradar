"""Ordered multi-condition expansion for scalar configuration fields."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, TypeAdapter, ValidationError

from .io import ConfigError, _deep_merge_dicts, _load_yaml_mapping
from .schema import AppConfig, ExecutionStage


@dataclass(frozen=True)
class ConditionDimension:
    """One ordered config path whose scalar value varies across conditions."""

    path: tuple[str, ...]
    values: tuple[Any, ...]

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True)
class ConfigCondition:
    """One fully validated condition generated from a source config."""

    index: int
    condition_id: str
    label: str
    values: tuple[tuple[str, Any], ...]
    config: AppConfig


@dataclass(frozen=True)
class ConfigConditionSet:
    """Ordered conditions and the dimensions used to generate them."""

    dimensions: tuple[ConditionDimension, ...]
    conditions: tuple[ConfigCondition, ...]


_FORBIDDEN_DIMENSION_PREFIXES = (
    ("data",),
    ("split",),
    ("runtime",),
    ("evaluation",),
    ("figures",),
    ("summary",),
    ("report",),
    ("model_selection", "inner_cv_strategy"),
    ("model_selection", "inner_cv_n_splits"),
    ("preprocess", "max_pivot_cells"),
)

_RANKED_MAX_FEATURES_PATH = (
    "preprocess",
    "ranked_feature_filter",
    "max_features",
)
_RANKED_MAX_FEATURES_DOTTED_PATH = ".".join(_RANKED_MAX_FEATURES_PATH)


def _merged_raw_config(
    config_paths: list[Path],
    execution_stage_override: ExecutionStage | None,
) -> dict[str, Any]:
    if not config_paths:
        raise ConfigError("At least one config file must be provided")
    merged: dict[str, Any] = {}
    for config_path in config_paths:
        merged = _deep_merge_dicts(merged, _load_yaml_mapping(config_path))
    if execution_stage_override is not None:
        runtime_raw = merged.get("runtime")
        runtime = dict(runtime_raw) if isinstance(runtime_raw, dict) else {}
        runtime["execution_stage"] = execution_stage_override
        merged["runtime"] = runtime
    return merged


def _nested_model_type(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for member in get_args(annotation):
        if isinstance(member, type) and issubclass(member, BaseModel):
            return member
    return None


def _valid_as_whole(annotation: Any, value: Any) -> bool:
    try:
        TypeAdapter(annotation).validate_python(value)
    except (TypeError, ValueError, ValidationError):
        return False
    return True


def _collect_dimensions(
    raw: dict[str, Any],
    model_type: type[BaseModel],
    *,
    prefix: tuple[str, ...] = (),
) -> list[ConditionDimension]:
    dimensions: list[ConditionDimension] = []
    for key, value in raw.items():
        field = model_type.model_fields.get(key)
        if field is None:
            continue
        path = (*prefix, key)
        nested_model = _nested_model_type(field.annotation)
        if nested_model is not None and isinstance(value, dict):
            dimensions.extend(
                _collect_dimensions(value, nested_model, prefix=path)
            )
            continue
        if not isinstance(value, list):
            continue
        if _valid_as_whole(field.annotation, value):
            continue
        if nested_model is not None:
            raise ConfigError(
                f"Condition values must target scalar fields, not {'.'.join(path)}"
            )
        if not value:
            raise ConfigError(f"Condition value list cannot be empty: {'.'.join(path)}")
        adapter: TypeAdapter[Any] = TypeAdapter(field.annotation)
        validated_values: list[Any] = []
        for position, item in enumerate(value, start=1):
            try:
                validated_values.append(adapter.validate_python(item))
            except (TypeError, ValueError, ValidationError) as exc:
                raise ConfigError(
                    f"Invalid condition value at {'.'.join(path)}[{position}]: {exc}"
                ) from exc
        dimensions.append(ConditionDimension(path=path, values=tuple(validated_values)))
    return dimensions


def _is_forbidden_dimension(path: tuple[str, ...]) -> bool:
    return any(path[: len(prefix)] == prefix for prefix in _FORBIDDEN_DIMENSION_PREFIXES)


def _set_path(raw: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = raw
    for key in path[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise ConfigError(f"Condition path is not a mapping: {'.'.join(path)}")
        target = child
    target[path[-1]] = value


def _normalize_inactive_condition_fields(
    raw: dict[str, Any],
    values: list[tuple[str, Any]],
) -> None:
    """Canonicalize fields that are inactive for the selected condition."""
    preprocess = raw.get("preprocess")
    if not isinstance(preprocess, dict):
        return
    ranked_filter = preprocess.get("ranked_feature_filter")
    if (
        not isinstance(ranked_filter, dict)
        or ranked_filter.get("method", "none") != "none"
    ):
        return

    ranked_filter["max_features"] = None
    for position, (path, _value) in enumerate(values):
        if path == _RANKED_MAX_FEATURES_DOTTED_PATH:
            values[position] = (path, None)


def _differs_only_by_inactive_ranked_max_features(
    previous_values: tuple[tuple[str, Any], ...],
    current_values: tuple[tuple[str, Any], ...],
    resolved: AppConfig,
) -> bool:
    if resolved.preprocess.ranked_feature_filter.method != "none":
        return False
    differing_paths = {
        previous_path
        for (previous_path, previous_value), (current_path, current_value) in zip(
            previous_values,
            current_values,
            strict=True,
        )
        if previous_path != current_path or previous_value != current_value
    }
    return differing_paths == {_RANKED_MAX_FEATURES_DOTTED_PATH}


def _config_sha256(config: AppConfig) -> str:
    canonical = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _display_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def has_condition_dimensions(
    config_paths: list[Path],
    execution_stage_override: ExecutionStage | None = None,
) -> bool:
    """Return whether a config contains any scalar field expressed as a list."""
    raw = _merged_raw_config(config_paths, execution_stage_override)
    return bool(_collect_dimensions(raw, AppConfig))


def load_config_conditions(
    config_paths: list[Path],
    execution_stage_override: ExecutionStage | None = None,
) -> ConfigConditionSet:
    """Expand and validate all ordered scalar-list conditions in a config."""
    raw = _merged_raw_config(config_paths, execution_stage_override)
    dimensions = tuple(_collect_dimensions(raw, AppConfig))
    forbidden = [
        dimension.dotted_path
        for dimension in dimensions
        if _is_forbidden_dimension(dimension.path)
    ]
    if forbidden:
        raise ConfigError(
            "Condition lists are not supported for split/runtime/report-control fields: "
            + ", ".join(forbidden)
        )

    combinations = product(*(dimension.values for dimension in dimensions))
    conditions: list[ConfigCondition] = []
    seen_hashes: dict[str, tuple[int, tuple[tuple[str, Any], ...]]] = {}
    for combination in combinations:
        condition_raw = deepcopy(raw)
        values: list[tuple[str, Any]] = []
        for dimension, value in zip(dimensions, combination, strict=True):
            _set_path(condition_raw, dimension.path, value)
            values.append((dimension.dotted_path, value))
        source_values = tuple(values)
        _normalize_inactive_condition_fields(condition_raw, values)
        index = len(conditions) + 1
        try:
            resolved = AppConfig.model_validate(condition_raw)
        except ValidationError as exc:
            details = ", ".join(f"{path}={_display_value(value)}" for path, value in values)
            raise ConfigError(f"Invalid generated condition {index} ({details}): {exc}") from exc
        digest = _config_sha256(resolved)
        duplicate = seen_hashes.get(digest)
        if duplicate is not None:
            duplicate_index, duplicate_source_values = duplicate
            if _differs_only_by_inactive_ranked_max_features(
                duplicate_source_values,
                source_values,
                resolved,
            ):
                continue
            raise ConfigError(
                f"Generated condition {index} duplicates condition {duplicate_index}; "
                "remove repeated or inactive condition values"
            )
        seen_hashes[digest] = (index, source_values)
        label = "; ".join(f"{path}={_display_value(value)}" for path, value in values)
        conditions.append(
            ConfigCondition(
                index=index,
                condition_id=f"cond_{digest[:12]}",
                label=label,
                values=tuple(values),
                config=resolved,
            )
        )
    return ConfigConditionSet(dimensions=dimensions, conditions=tuple(conditions))
