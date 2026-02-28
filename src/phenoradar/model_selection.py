"""Model-selection search-space expansion and candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from phenoradar.config import AppConfig
from phenoradar.config.schema import (
    ContinuousLogRangeSpec,
    ContinuousRangeSpec,
    DiscreteRangeSpec,
    IntRangeSpec,
    LogRangeSpec,
    SearchSpaceValue,
)

_MAX_GENERATED_VALUES = 1_000_000
ContinuousSpec = ContinuousRangeSpec | ContinuousLogRangeSpec


class ModelSelectionError(ValueError):
    """Raised when candidate generation fails."""


@dataclass(frozen=True)
class Candidate:
    """One generated hyperparameter candidate."""

    candidate_index: int
    params: dict[str, Any]


def _deterministic_int_seed(seed_text: str) -> int:
    from hashlib import sha256

    seed_hash = sha256(seed_text.encode("utf-8")).hexdigest()[:16]
    return int(seed_hash, 16) % (2**31 - 1)


def _dedupe_preserve_order(values: list[Any]) -> list[Any]:
    seen: list[Any] = []
    deduped: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.append(value)
        deduped.append(value)
    return deduped


def _expand_float_range(spec: DiscreteRangeSpec) -> list[float]:
    values: list[float] = []
    index = 0
    tolerance = abs(spec.step) * 1e-9 + 1e-12
    while index < _MAX_GENERATED_VALUES:
        value = float(spec.start + spec.step * index)
        if spec.inclusive_stop:
            if value > float(spec.stop) + tolerance:
                break
        elif value >= float(spec.stop) - tolerance:
            break
        values.append(value)
        index += 1
    if not values:
        raise ModelSelectionError("range expansion produced zero values")
    return _dedupe_preserve_order(values)


def _expand_int_range(spec: IntRangeSpec) -> list[int]:
    stop_value = spec.stop + (1 if spec.inclusive_stop else 0)
    values = list(range(spec.start, stop_value, spec.step))
    if not values:
        raise ModelSelectionError("int_range expansion produced zero values")
    return _dedupe_preserve_order(values)


def _expand_log_range(spec: LogRangeSpec) -> list[float]:
    exponents: list[float] = []
    index = 0
    tolerance = abs(spec.step_exp) * 1e-9 + 1e-12
    while index < _MAX_GENERATED_VALUES:
        exponent = float(spec.start_exp + spec.step_exp * index)
        if spec.inclusive_stop:
            if exponent > float(spec.stop_exp) + tolerance:
                break
        elif exponent >= float(spec.stop_exp) - tolerance:
            break
        exponents.append(exponent)
        index += 1
    if not exponents:
        raise ModelSelectionError("log_range expansion produced zero values")
    values = [float(spec.base**exponent) for exponent in exponents]
    return _dedupe_preserve_order(values)


def _expand_discrete_value(value: SearchSpaceValue) -> list[Any]:
    if isinstance(value, list):
        values = _dedupe_preserve_order(list(value))
        if not values:
            raise ModelSelectionError("search_space list cannot be empty")
        return values
    if isinstance(value, DiscreteRangeSpec):
        return _expand_float_range(value)
    if isinstance(value, IntRangeSpec):
        return _expand_int_range(value)
    if isinstance(value, LogRangeSpec):
        return _expand_log_range(value)
    raise ModelSelectionError("continuous search-space values are not discrete")


def _full_discrete_candidate_space(discrete_values: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not discrete_values:
        return [{}]
    param_names = sorted(discrete_values.keys())
    params_product = product(*(discrete_values[name] for name in param_names))
    return [
        {name: value for name, value in zip(param_names, combo, strict=True)}
        for combo in params_product
    ]


def _search_space_parts(
    search_space: dict[str, SearchSpaceValue],
) -> tuple[dict[str, list[Any]], dict[str, ContinuousSpec]]:
    discrete_values: dict[str, list[Any]] = {}
    continuous_values: dict[str, ContinuousSpec] = {}

    for param_name in sorted(search_space.keys()):
        value = search_space[param_name]
        if isinstance(value, (ContinuousRangeSpec, ContinuousLogRangeSpec)):
            continuous_values[param_name] = value
            continue
        discrete_values[param_name] = _expand_discrete_value(value)
    return discrete_values, continuous_values


def expanded_search_space(
    search_space: dict[str, SearchSpaceValue],
) -> tuple[dict[str, list[Any]], dict[str, ContinuousSpec]]:
    """Expand validated search space into deterministic discrete and continuous parts."""
    return _search_space_parts(search_space)


def discrete_candidate_space_size(discrete_values: dict[str, list[Any]]) -> int:
    """Return deterministic size of discrete full candidate space."""
    if not discrete_values:
        return 1
    return int(prod(len(values) for values in discrete_values.values()))


def _draw_continuous_random(
    *,
    discrete_values: dict[str, list[Any]],
    continuous_values: dict[str, ContinuousRangeSpec | ContinuousLogRangeSpec],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    for _ in range(count):
        params: dict[str, Any] = {}
        for param_name in sorted(discrete_values.keys()):
            values = discrete_values[param_name]
            params[param_name] = values[int(rng.integers(0, len(values)))]
        for param_name in sorted(continuous_values.keys()):
            spec = continuous_values[param_name]
            if isinstance(spec, ContinuousRangeSpec):
                params[param_name] = float(rng.uniform(spec.start, spec.stop))
            else:
                exponent = float(rng.uniform(spec.start_exp, spec.stop_exp))
                params[param_name] = float(spec.base**exponent)
        candidates.append(params)
    return candidates


def _draw_tpe_candidates(
    *,
    discrete_values: dict[str, list[Any]],
    continuous_values: dict[str, ContinuousRangeSpec | ContinuousLogRangeSpec],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
    )

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for param_name in sorted(discrete_values.keys()):
            params[param_name] = trial.suggest_categorical(param_name, discrete_values[param_name])
        for param_name in sorted(continuous_values.keys()):
            spec = continuous_values[param_name]
            if isinstance(spec, ContinuousRangeSpec):
                params[param_name] = trial.suggest_float(param_name, spec.start, spec.stop)
            else:
                exponent_name = f"{param_name}__exp"
                exponent = trial.suggest_float(exponent_name, spec.start_exp, spec.stop_exp)
                params[param_name] = float(spec.base**exponent)
        trial.set_user_attr("params", params)
        return 0.0

    study.optimize(_objective, n_trials=count)
    ordered_trials = sorted(study.trials, key=lambda item: item.number)
    candidates: list[dict[str, Any]] = []
    for trial in ordered_trials:
        attrs = trial.user_attrs.get("params")
        if not isinstance(attrs, dict):
            raise ModelSelectionError("TPE trial did not record params")
        candidates.append(dict(attrs))
    return candidates


def generate_candidates(
    *,
    config: AppConfig,
    training_scope_id: str,
    source_sample_set_id: int,
    warnings: list[str],
) -> list[Candidate]:
    """Generate deterministic candidate params for one source sample set."""
    search_space = config.model_selection.search_space
    strategy = config.model_selection.search_strategy
    runtime_seed = int(config.runtime.seed)

    discrete_values, continuous_values = _search_space_parts(search_space)
    full_discrete = _full_discrete_candidate_space(discrete_values)
    has_continuous = bool(continuous_values)

    if strategy == "grid":
        params_list = full_discrete
    elif strategy == "random":
        if config.model_selection.trial_count is None:
            raise ModelSelectionError("trial_count is required for random strategy")
        requested = int(config.model_selection.trial_count)
        local_seed = _deterministic_int_seed(
            f"{runtime_seed}|{training_scope_id}|sample_set_{source_sample_set_id}|random"
        )
        if not has_continuous:
            effective = min(requested, len(full_discrete))
            if effective < requested:
                warnings.append(
                    "model_selection.trial_count exceeded discrete candidate space; "
                    f"capped from {requested} to {effective} "
                    f"for source sample_set_id={source_sample_set_id}"
                )
            rng = np.random.default_rng(local_seed)
            order = rng.permutation(len(full_discrete)).tolist()
            params_list = [full_discrete[idx] for idx in order[:effective]]
        else:
            params_list = _draw_continuous_random(
                discrete_values=discrete_values,
                continuous_values=continuous_values,
                count=requested,
                seed=local_seed,
            )
    else:
        if config.model_selection.trial_count is None:
            raise ModelSelectionError("trial_count is required for tpe strategy")
        requested = int(config.model_selection.trial_count)
        local_seed = _deterministic_int_seed(
            f"{runtime_seed}|{training_scope_id}|sample_set_{source_sample_set_id}|tpe"
        )
        effective = requested
        if not has_continuous:
            effective = min(requested, len(full_discrete))
            if effective < requested:
                warnings.append(
                    "model_selection.trial_count exceeded discrete candidate space; "
                    f"capped from {requested} to {effective} "
                    f"for source sample_set_id={source_sample_set_id}"
                )
        params_list = _draw_tpe_candidates(
            discrete_values=discrete_values,
            continuous_values=continuous_values,
            count=effective,
            seed=local_seed,
        )

    return [
        Candidate(candidate_index=index, params=dict(params))
        for index, params in enumerate(params_list)
    ]
