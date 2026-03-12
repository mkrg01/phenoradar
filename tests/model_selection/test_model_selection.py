from __future__ import annotations

from pathlib import Path
from textwrap import dedent, indent

import pytest

import phenoradar.model_selection as model_selection_mod
from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.config.schema import (
    ContinuousRangeSpec,
    DiscreteRangeSpec,
    IntRangeSpec,
    LogRangeSpec,
)
from phenoradar.model_selection import (
    Candidate,
    ModelSelectionError,
    discrete_candidate_space_size,
    expanded_search_space,
    generate_candidates,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _config(tmp_path: Path, extra: str) -> AppConfig:
    extra_block = indent(dedent(extra).strip(), "  ")
    config_path = _write(
        tmp_path / "config.yml",
        f"""
runtime:
  seed: 123
model_selection:
{extra_block}
""".rstrip()
        + "\n",
    )
    return load_and_resolve_config([config_path])


def _params(candidates: list[Candidate]) -> list[dict[str, object]]:
    return [candidate.params for candidate in candidates]


def test_generate_candidates_grid_expands_discrete_space_deterministically(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: grid
  search_space:
    l1_ratio: [0.0, 0.5, 0.5]
    C:
      type: log_range
      base: 10
      start_exp: -1
      end_exp: 1
      step_exp: 1
      inclusive_end: true
""",
    )
    warnings: list[str] = []
    candidates = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=warnings,
    )

    params = _params(candidates)
    assert len(params) == 6
    assert warnings == []
    assert params[0]["C"] == pytest.approx(0.1)
    assert params[0]["l1_ratio"] == pytest.approx(0.0)
    assert params[-1]["C"] == pytest.approx(10.0)
    assert params[-1]["l1_ratio"] == pytest.approx(0.5)


def test_generate_candidates_random_caps_discrete_space_and_is_deterministic(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: random
  trial_count: 10
  search_space:
    C: [0.1, 1.0]
    solver: ["lbfgs", "liblinear"]
""",
    )
    warnings_first: list[str] = []
    warnings_second: list[str] = []
    first = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=warnings_first,
    )
    second = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=warnings_second,
    )

    first_params = _params(first)
    second_params = _params(second)
    assert len(first_params) == 4
    assert first_params == second_params
    assert any("capped from 10 to 4" in warning for warning in warnings_first)
    assert warnings_first == warnings_second


def test_generate_candidates_random_with_continuous_space_is_deterministic(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: random
  trial_count: 5
  search_space:
    C: [0.1, 1.0]
    l1_ratio:
      type: continuous_range
      start: 0.0
      end: 1.0
""",
    )
    warnings_first: list[str] = []
    warnings_second: list[str] = []
    first = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=warnings_first,
    )
    second = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=warnings_second,
    )

    first_params = _params(first)
    second_params = _params(second)
    assert len(first_params) == 5
    assert first_params == second_params
    assert warnings_first == []
    assert warnings_second == []
    for params in first_params:
        assert params["C"] in {0.1, 1.0}
        assert 0.0 <= float(params["l1_ratio"]) <= 1.0


def test_generate_candidates_tpe_with_continuous_space_is_deterministic(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: tpe
  trial_count: 4
  search_space:
    C:
      type: continuous_log_range
      base: 10
      start_exp: -1
      end_exp: 1
    l1_ratio:
      type: continuous_range
      start: 0.0
      end: 1.0
""",
    )
    warnings_first: list[str] = []
    warnings_second: list[str] = []
    first = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=warnings_first,
    )
    second = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=warnings_second,
    )

    first_params = _params(first)
    second_params = _params(second)
    assert len(first_params) == 4
    assert warnings_first == []
    assert warnings_second == []
    for first_item, second_item in zip(first_params, second_params, strict=True):
        assert first_item["C"] == pytest.approx(float(second_item["C"]))
        assert first_item["l1_ratio"] == pytest.approx(float(second_item["l1_ratio"]))
        assert 0.1 <= float(first_item["C"]) <= 10.0
        assert 0.0 <= float(first_item["l1_ratio"]) <= 1.0


def test_generate_candidates_tpe_caps_discrete_space_and_warns(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: tpe
  trial_count: 10
  search_space:
    C: [0.1, 1.0]
    l1_ratio: [0.0, 0.5]
""",
    )
    warnings: list[str] = []

    candidates = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=warnings,
    )

    assert len(candidates) == 4
    assert any("capped from 10 to 4" in warning for warning in warnings)


def test_generate_candidates_uses_runtime_seed(tmp_path: Path) -> None:
    config_seed_123 = _config(
        tmp_path,
        extra="""
  search_strategy: random
  trial_count: 3
  search_space:
    C:
      type: continuous_range
      start: 0.0
      end: 1.0
""",
    )
    config_seed_124 = config_seed_123.model_copy(
        update={"runtime": config_seed_123.runtime.model_copy(update={"seed": 124})}
    )

    params_seed_123 = _params(
        generate_candidates(
            config=config_seed_123,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            warnings=[],
        )
    )
    params_seed_124 = _params(
        generate_candidates(
            config=config_seed_124,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            warnings=[],
        )
    )

    assert params_seed_123 != params_seed_124


def test_generate_candidates_fails_when_discrete_range_expands_to_zero_values(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: random
  trial_count: 1
  search_space:
    C:
      type: range
      start: 1.0
      end: 1.0
      step: 0.1
      inclusive_end: false
""",
    )

    with pytest.raises(ModelSelectionError, match="range expansion produced zero values"):
        generate_candidates(
            config=config,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            warnings=[],
        )


def test_expanded_search_space_and_discrete_size_helpers(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: random
  trial_count: 2
  search_space:
    C:
      type: log_range
      base: 10
      start_exp: -1
      end_exp: 1
      step_exp: 1
      inclusive_end: true
    l1_ratio: [0.0, 0.5]
    alpha:
      type: continuous_range
      start: 0.0
      end: 1.0
""",
    )

    discrete, continuous = expanded_search_space(config.model_selection.search_space)

    assert sorted(discrete.keys()) == ["C", "l1_ratio"]
    assert discrete_candidate_space_size(discrete) == 6
    assert list(continuous.keys()) == ["alpha"]


def test_discrete_candidate_space_size_returns_one_for_empty_space() -> None:
    assert discrete_candidate_space_size({}) == 1


def test_generate_candidates_random_requires_trial_count_when_mutated_to_none(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: random
  trial_count: 3
  search_space:
    C: [0.1, 1.0]
""",
    )
    config_missing_trial_count = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(update={"trial_count": None}),
        }
    )

    with pytest.raises(ModelSelectionError, match="trial_count is required for random strategy"):
        generate_candidates(
            config=config_missing_trial_count,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            warnings=[],
        )


def test_generate_candidates_tpe_requires_trial_count_when_mutated_to_none(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: tpe
  trial_count: 3
  search_space:
    C: [0.1, 1.0]
""",
    )
    config_missing_trial_count = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(update={"trial_count": None}),
        }
    )

    with pytest.raises(ModelSelectionError, match="trial_count is required for tpe strategy"):
        generate_candidates(
            config=config_missing_trial_count,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            warnings=[],
        )


def test_generate_candidates_rejects_empty_search_space_list_when_mutated(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: grid
  search_space:
    C: [1.0]
""",
    )
    config_with_empty_list = config.model_copy(
        update={
            "model_selection": config.model_selection.model_copy(
                update={"search_space": {"C": []}}
            ),
        }
    )

    with pytest.raises(ModelSelectionError, match="search_space list cannot be empty"):
        generate_candidates(
            config=config_with_empty_list,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            warnings=[],
        )


def test_generate_candidates_random_with_continuous_log_space_is_deterministic(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        extra="""
  search_strategy: random
  trial_count: 5
  search_space:
    C:
      type: continuous_log_range
      base: 10
      start_exp: -2
      end_exp: 0
""",
    )
    first = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=[],
    )
    second = generate_candidates(
        config=config,
        training_scope_id="outer_fold_0",
        source_sample_set_id=0,
        warnings=[],
    )

    first_params = _params(first)
    second_params = _params(second)
    assert first_params == second_params
    assert len(first_params) == 5
    for params in first_params:
        assert 0.01 <= float(params["C"]) <= 1.0


def test_generate_candidates_tpe_raises_when_trial_did_not_store_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeTrial:
        def __init__(self) -> None:
            self.number = 0
            self.user_attrs: dict[str, object] = {}

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials = [_FakeTrial()]

        def optimize(self, _objective: object, n_trials: int) -> None:
            _ = n_trials

    monkeypatch.setattr(
        model_selection_mod.optuna,
        "create_study",
        lambda **_kwargs: _FakeStudy(),
    )
    config = _config(
        tmp_path,
        extra="""
  search_strategy: tpe
  trial_count: 1
  search_space:
    C: [0.1]
""",
    )

    with pytest.raises(ModelSelectionError, match="TPE trial did not record params"):
        generate_candidates(
            config=config,
            training_scope_id="outer_fold_0",
            source_sample_set_id=0,
            warnings=[],
        )


def test_expand_float_range_inclusive_end_includes_endpoint() -> None:
    spec = DiscreteRangeSpec(
        type="range",
        start=0.0,
        end=0.2,
        step=0.1,
        inclusive_end=True,
    )

    values = model_selection_mod._expand_float_range(spec)

    assert values == pytest.approx([0.0, 0.1, 0.2])


def test_expand_int_range_inclusive_end_includes_endpoint() -> None:
    spec = IntRangeSpec(
        type="int_range",
        start=1,
        end=3,
        step=1,
        inclusive_end=True,
    )

    values = model_selection_mod._expand_int_range(spec)

    assert values == [1, 2, 3]


def test_expand_int_range_rejects_zero_values() -> None:
    spec = IntRangeSpec(
        type="int_range",
        start=1,
        end=1,
        step=1,
        inclusive_end=False,
    )

    with pytest.raises(ModelSelectionError, match="int_range expansion produced zero values"):
        model_selection_mod._expand_int_range(spec)


def test_expand_log_range_exclusive_end_excludes_endpoint() -> None:
    spec = LogRangeSpec(
        type="log_range",
        base=10.0,
        start_exp=0.0,
        end_exp=2.0,
        step_exp=1.0,
        inclusive_end=False,
    )

    values = model_selection_mod._expand_log_range(spec)

    assert values == pytest.approx([1.0, 10.0])


def test_expand_log_range_rejects_zero_values() -> None:
    spec = LogRangeSpec(
        type="log_range",
        base=10.0,
        start_exp=0.0,
        end_exp=0.0,
        step_exp=1.0,
        inclusive_end=False,
    )

    with pytest.raises(ModelSelectionError, match="log_range expansion produced zero values"):
        model_selection_mod._expand_log_range(spec)


def test_expand_discrete_value_accepts_int_range_spec() -> None:
    spec = IntRangeSpec(
        type="int_range",
        start=1,
        end=3,
        step=1,
        inclusive_end=False,
    )

    values = model_selection_mod._expand_discrete_value(spec)

    assert values == [1, 2]


def test_expand_discrete_value_rejects_continuous_spec() -> None:
    spec = ContinuousRangeSpec(type="continuous_range", start=0.0, end=1.0)

    with pytest.raises(
        ModelSelectionError, match="continuous search-space values are not discrete"
    ):
        model_selection_mod._expand_discrete_value(spec)
