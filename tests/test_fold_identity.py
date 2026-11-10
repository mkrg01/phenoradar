from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from sklearn.model_selection import LeaveOneGroupOut
from test_combined import combined_config

from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.cv import (
    _build_inner_cv_preprocessed_folds,
    _fold_ids,
    _inner_cv_splits,
    _sample_training_sets,
    _score_candidate_inner_cv,
)
from phenoradar.fold_identity import configure_fold_seed_ids, seed_scope, validation_fold_ids
from phenoradar.model_selection import Candidate, generate_candidates
from phenoradar.split import build_split_artifacts


def numeric_config(tmp_path: Path, mode: str = "expression") -> AppConfig:
    path = combined_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["data"]["feature_mode"] = mode
    metadata = Path(raw["data"]["metadata_path"])
    frame = pl.read_csv(metadata, separator="\t").with_columns(
        pl.col("contrast_pair_id").replace_strict(
            {"g0": "0", "g1": "2", "g2": "10", "g3": "20"}, default=None,
        )
    )
    frame.write_csv(metadata, separator="\t")
    path.write_text(yaml.safe_dump(raw))
    return load_and_resolve_config([path])


@pytest.mark.parametrize("mode", ["expression", "amino_acid", "combined"])
def test_logo_fold_ids_match_noncontiguous_group_numbers_without_repartitioning(
    tmp_path: Path, mode: str,
) -> None:
    config = numeric_config(tmp_path, mode)
    artifacts = build_split_artifacts(config)
    manifest = artifacts.split_manifest
    valid = manifest.filter(pl.col("pool") == "validation")
    assert valid.filter(pl.col("fold_id") != pl.col("group_id")).is_empty()
    assert set(valid["fold_id"]) == {"0", "2", "10", "20"}
    assert _fold_ids(manifest) == ["0", "10", "2", "20"]  # Original execution order.
    assert set(artifacts.fold_diagnostics["fold_id"]) == set(valid["fold_id"])
    assert artifacts.fold_validation_groups.filter(
        pl.col("fold_id") != pl.col("group_id")
    ).is_empty()
    for row in artifacts.fold_validation_groups.iter_rows(named=True):
        held_out = row["group_id"]
        train = manifest.filter((pl.col("fold_id") == held_out) & (pl.col("pool") == "train"))
        assert held_out not in train["group_id"]
        assert train.height == 6
    configure_fold_seed_ids(config, manifest)
    assert seed_scope(config, "outer_fold_10") == "outer_fold_2"
    assert seed_scope(config, "fold_2|sample_set_3") == "fold_3|sample_set_3"
    assert seed_scope(config, "final_refit") == "final_refit"
    assert pickle.loads(pickle.dumps(config))._fold_seed_ids == config._fold_seed_ids
    assert "_fold_seed_ids" not in config.model_dump()
    assert config.model_copy(deep=True)._fold_seed_ids == config._fold_seed_ids


@pytest.mark.parametrize("groups,valid,expected", [
    (["1", "10", "2"], [[0], [1], [2]], ["1", "10", "2"]),
    (["g1", "g10", "g2"], [[0], [1], [2]], ["1", "2", "3"]),
    (["1", "10", "2"], [[0, 1], [2]], ["1", "2"]),
    (["01", "1"], [[0], [1]], ["1", "2"]),
    (["1", "2"], [[0], [0]], ["1", "2"]),
])
def test_fold_number_reuse_requires_an_unambiguous_integer_group_partition(
    groups: list[str], valid: list[list[int]], expected: list[str],
) -> None:
    assert validation_fold_ids(groups, valid) == expected


def test_inner_logo_keeps_membership_and_uses_original_group_numbers() -> None:
    config = AppConfig()
    config.model_selection.inner_cv_strategy = "logo"
    groups = np.repeat(["0", "2", "10", "20"], 2)
    y = np.tile([0, 1], 4)
    actual = _inner_cv_splits(config, y, groups)
    expected = list(LeaveOneGroupOut().split(np.arange(8), y, groups))
    for (train, valid, fold), (old_train, old_valid) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(train, old_train)
        np.testing.assert_array_equal(valid, old_valid)
        assert set(groups[valid]) == {fold}
    config.model_selection.inner_cv_strategy = "group_kfold"
    config.model_selection.inner_cv_n_splits = 2
    assert [fold for _, _, fold in _inner_cv_splits(config, y, groups)] == ["1", "2"]
    config.model_selection.inner_cv_n_splits = 4
    assert all(set(groups[valid]) == {fold}
               for _, valid, fold in _inner_cv_splits(config, y, groups))


def test_group_balanced_sampling_does_not_change_when_display_ids_change() -> None:
    old, new = AppConfig(), AppConfig()
    new._fold_seed_ids = {"10": "2"}
    groups = np.repeat(["0", "2", "20"], 6)
    y = np.tile([0, 0, 0, 1, 1, 1], 3)
    before = _sample_training_sets(old, y, groups, "fold_2", [])
    after = _sample_training_sets(new, y, groups, "fold_10", [])
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)


def test_random_forest_inner_scores_keep_original_outer_and_inner_seeds(monkeypatch) -> None:
    import phenoradar.cv as cv

    config = AppConfig()
    config.model_selection.inner_cv_strategy = "logo"
    config.model.name = "random_forest"
    config._fold_seed_ids = {"10": "2"}
    config.preprocess.expression.sparse_feature_filter.enabled = False
    config.preprocess.expression.ranked_feature_filter.method = "none"
    rng = np.random.default_rng(9)
    x = rng.uniform(1, 5, size=(24, 3))
    y = np.tile([0, 0, 0, 1, 1, 1], 4)
    groups = np.repeat(["0", "2", "10", "20"], 6)
    folds = _build_inner_cv_preprocessed_folds(
        config=config, x_source_raw=x, y_source=y, groups_source=groups,
        contrast_groups_source=None, feature_names=["a", "b", "c"],
    )
    legacy = [replace(fold, inner_fold_id=str(i), seed_fold_id=None)
              for i, fold in enumerate(folds, 1)]
    seeds = []
    build = cv._build_estimator

    def capture(*args, **kwargs):
        seeds.append(args[1])
        return build(*args, **kwargs)

    monkeypatch.setattr(cv, "_build_estimator", capture)
    kwargs = dict(config=config, source_sample_set_id=0,
                  candidate=Candidate(0, {"n_estimators": 8, "max_depth": 2}))
    before = _score_candidate_inner_cv(**kwargs, training_scope_id="outer_fold_2",
                                       preprocessed_folds=legacy)
    old_seeds = seeds.copy()
    seeds.clear()
    after = _score_candidate_inner_cv(**kwargs, training_scope_id="outer_fold_10",
                                      preprocessed_folds=folds)
    assert seeds == old_seeds
    assert before[0] == after[0]
    assert [row["metric_value"] for row in before[1]] == [row["metric_value"] for row in after[1]]


@pytest.mark.parametrize("strategy", ["random", "tpe"])
def test_candidate_generation_preserves_seed_when_fold_id_changes(strategy: str) -> None:
    before, after = AppConfig(), AppConfig()
    for config in [before, after]:
        config.model_selection.search_strategy = strategy
        config.model_selection.trial_count = 4
        config.model_selection.search_space = {"lambda": [.001, .01, .1], "alpha": [.5, 1.]}
    after._fold_seed_ids = {"10": "2"}
    old = generate_candidates(config=before, training_scope_id="outer_fold_2",
                              source_sample_set_id=1, warnings=[])
    new = generate_candidates(config=after, training_scope_id="outer_fold_10",
                              source_sample_set_id=1, warnings=[])
    assert old == new


def test_inner_label_failure_identifies_the_held_out_group() -> None:
    config = AppConfig()
    config.model_selection.inner_cv_strategy = "logo"
    with pytest.raises(ValueError, match="Inner CV fold 10 training split"):
        _inner_cv_splits(config, np.array([0, 0, 1, 1]), np.array(["10", "10", "20", "20"]))
