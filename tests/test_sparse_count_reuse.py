from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import phenoradar.cv as cv
from phenoradar.config import AppConfig
from phenoradar.split import build_split_artifacts


def _config(scope: str = "any_trait", strategy: str = "logo", **preprocess: Any) -> AppConfig:
    settings: dict[str, Any] = {
        "sparse_feature_filter": {"enabled": True, "scope": scope, "min_nonzero_fraction": 0.5},
        "ranked_feature_filter": {"method": "pair_aware", "max_features": 5},
    }
    settings.update(preprocess)
    return AppConfig.model_validate({
        "preprocess": settings,
        "sampling": {"weighting": "group_label_inverse"},
        "model_selection": {
            "selected_candidate_count": 1, "inner_cv_strategy": strategy,
            "inner_cv_n_splits": None if strategy == "logo" else 4,
        },
        "runtime": {"n_jobs": 1},
    })


@pytest.mark.parametrize("scope", ["all_samples", "any_trait", "trait_0", "trait_1"])
@pytest.mark.parametrize("strategy", ["logo", "group_kfold", "stratified_group_kfold"])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("threshold", [0.0, np.nextafter(0.5, 0), 0.5, np.nextafter(0.5, 1), 1.0])
def test_cached_counts_match_direct_scan_at_boundaries(
    scope: str, strategy: str, layout: str, threshold: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv, "_SPARSE_FILTER_BLOCK_CELLS", 35)
    config = _config(scope, strategy)
    config.preprocess.sparse_feature_filter.min_nonzero_fraction = threshold
    matrix = np.random.default_rng(681).choice(
        [0, cv._NONZERO_TOLERANCE, np.nextafter(cv._NONZERO_TOLERANCE, np.inf),
         2.0, np.nan, np.inf, -np.inf], size=(40, 26),
    )
    matrix[:, 0] = 1.0
    matrix[:, 2] = np.nan
    if layout == "F":
        matrix = np.asfortranarray(matrix)
    elif layout == "strided":
        matrix = matrix[:, ::2]
    matrix.flags.writeable = False
    labels = np.arange(40) % 2
    groups = np.repeat(np.arange(8), [1, 3, 4, 6, 5, 7, 8, 6])
    splits = cv._inner_cv_splits(config, labels, groups)
    cache = cv._build_inner_sparse_count_cache(config, matrix, labels, splits)
    assert cache is not None
    for train, valid, _ in splits:
        expected = cv._sparse_feature_indices(config, matrix, labels[train], train_rows=train)
        np.testing.assert_array_equal(cache.select(valid), expected)


@pytest.mark.parametrize("scope", ["all_samples", "any_trait", "trait_0", "trait_1"])
@pytest.mark.parametrize("strategy", ["logo", "group_kfold", "stratified_group_kfold"])
@pytest.mark.parametrize("transform,neutral", [
    ("none", False), ("log1p", False), ("log1p", True),
    ("sample_rank", False), ("sample_percentile_rank", False),
])
@pytest.mark.parametrize("rank_method", ["none", "pair_aware", "unpaired", "variance"])
def test_inner_preprocessing_matches_direct_path_exactly(
    scope: str, strategy: str, transform: str, neutral: bool, rank_method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        scope, strategy,
        expression_transform={"method": transform},
        absent_feature_fill="nan" if neutral else 0,
        missing_expression={"method": "neutral" if neutral else "none", "zero_as_missing": neutral},
        ranked_feature_filter={"method": rank_method, "max_features": 5},
    )
    raw = np.random.default_rng(523).lognormal(size=(36, 24))
    raw[::3, ::3] = 0.0
    if neutral:
        raw[::4, 1::3] = np.nan
    labels = np.arange(36) % 2
    groups = np.repeat(np.arange(6), 6)
    arguments = dict(
        config=config, x_source_raw=raw, y_source=labels, groups_source=groups,
        contrast_groups_source=groups, feature_names=[f"f{i}" for i in range(raw.shape[1])],
    )
    with monkeypatch.context() as direct:
        direct.setattr(cv, "_build_inner_sparse_count_cache", lambda *_: None)
        expected = cv._build_inner_cv_preprocessed_folds(**arguments)
    actual = cv._build_inner_cv_preprocessed_folds(**arguments)
    for before, after in zip(expected, actual, strict=True):
        assert after.inner_fold_id == before.inner_fold_id
        for name in ("x_train", "x_valid", "y_train", "y_valid", "sample_weight"):
            np.testing.assert_array_equal(getattr(after, name), getattr(before, name))


@pytest.mark.parametrize("scope", ["all_samples", "any_trait", "trait_0", "trait_1"])
@pytest.mark.parametrize("strategy", ["logo", "group_kfold", "stratified_group_kfold"])
def test_validation_values_and_labels_cannot_change_training_preprocessing(
    scope: str, strategy: str,
) -> None:
    config = _config(scope, strategy)
    raw = np.random.default_rng(118).lognormal(size=(36, 20))
    raw[::3, ::3] = 0
    labels = np.arange(36) % 2
    groups = np.repeat(np.arange(6), 6)
    splits = cv._inner_cv_splits(config, labels, groups)
    train, valid, _ = splits[0]
    names = [f"f{i}" for i in range(raw.shape[1])]

    def prepare(matrix: np.ndarray, source_labels: np.ndarray) -> tuple[Any, ...]:
        transformed = cv._apply_expression_transform_for_config(config, matrix)
        cache = cv._build_inner_sparse_count_cache(config, transformed, source_labels, splits)
        assert cache is not None
        scores: list[dict[str, Any]] = []
        warnings: list[str] = []
        prepared = cv._preprocess_transformed_fold_with_counts(
            config, transformed, transformed, names, y_train=source_labels[train],
            groups_train=groups[train], train_rows=train, valid_rows=valid,
            sparse_selected=cache.select(valid), ranked_feature_score_rows=scores,
            warnings=warnings,
        )
        return (*prepared, scores, warnings)

    expected = prepare(raw, labels)
    changed = raw.copy()
    changed[valid] = 1e10
    changed[np.ix_(valid, np.arange(0, raw.shape[1], 2))] = 0.0
    changed_labels = labels.copy()
    changed_labels[valid] = 1 - changed_labels[valid]
    actual = prepare(changed, changed_labels)  # Keep splits fixed, including for stratification.
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[2:4] == expected[2:4]  # Feature order and every filter-stage count.
    assert actual[5:] == expected[5:]  # Ranked scores and warnings.
    for name in ("mean_", "var_", "scale_", "n_samples_seen_"):
        np.testing.assert_array_equal(getattr(actual[4], name), getattr(expected[4], name))


@pytest.mark.parametrize("scope", ["any_trait", "trait_0", "trait_1"])
def test_single_class_training_retains_direct_behavior(scope: str) -> None:
    config = _config(scope)
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    matrix = np.arange(32, dtype=float).reshape(8, 4)
    splits = [(np.flatnonzero(np.arange(8) // 2 != g), np.array([2*g, 2*g+1]), str(g))
              for g in range(4)]
    cache = cv._build_inner_sparse_count_cache(config, matrix, labels, splits)
    assert cache is not None
    for train, valid, _ in splits:
        try:
            expected = cv._sparse_feature_indices(config, matrix, labels[train], train_rows=train)
        except cv.CVError as error:
            with pytest.raises(cv.CVError, match=str(error)):
                cache.select(valid)
        else:
            np.testing.assert_array_equal(cache.select(valid), expected)


@pytest.mark.parametrize(
    "case", ["few_folds", "disabled", "duplicate", "omit", "overlap", "memory"],
)
def test_unsupported_or_unhelpful_splits_fall_back(
    case: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    matrix = np.ones((12, 3))
    labels = np.arange(12) % 2
    groups = np.repeat(np.arange(6), 2)
    splits = cv._inner_cv_splits(config, labels, groups)
    if case == "few_folds":
        splits = splits[:3]
    elif case == "disabled":
        config.preprocess.sparse_feature_filter.enabled = False
    elif case == "overlap":
        splits[-1] = splits[0]
    elif case == "memory":
        monkeypatch.setattr(cv, "_INNER_SPARSE_CACHE_MAX_BYTES", 1)
    else:
        train, valid, key = splits[0]
        train = train.copy()
        if case == "duplicate":
            train[-1] = train[0]
        else:
            train = train[:-1]
        splits[0] = (train, valid, key)
    assert cv._build_inner_sparse_count_cache(config, matrix, labels, splits) is None


def test_sampled_source_can_include_repeated_species_rows() -> None:
    config = _config()
    original = np.random.default_rng(203).choice([0., 1., 2.], size=(24, 12))
    original_groups = np.repeat(np.arange(6), 4)
    # A source may have selected only part of a group or repeated a species.
    sampled = np.array([i for g in range(6) for i in (4*g, 4*g, 4*g+1)])
    source, labels, groups = original[sampled], sampled % 2, original_groups[sampled]
    splits = cv._inner_cv_splits(config, labels, groups)
    cache = cv._build_inner_sparse_count_cache(config, source, labels, splits)
    assert cache is not None
    for train, valid, _ in splits:
        expected = cv._sparse_feature_indices(config, source, labels[train], train_rows=train)
        np.testing.assert_array_equal(cache.select(valid), expected)


@pytest.mark.parametrize("n_jobs", [1, 3])
@pytest.mark.parametrize("sampling", ["all_samples", "group_balanced"])
def test_complete_nested_cv_predictions_and_artifacts_match(
    n_jobs: int, sampling: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.arange(36) % 2
    groups = np.repeat(np.arange(6), 6)
    names = [f"sp{i:02d}" for i in range(36)]
    raw = np.random.default_rng(388).lognormal(size=(36, 12))
    raw[:, 0] += labels * 3
    raw[::3, 1::3] = 0
    metadata_path, tpm_path = tmp_path / "metadata.tsv", tmp_path / "tpm.tsv"
    pl.DataFrame({"species": names, "C4": labels, "contrast_pair_id": groups}).write_csv(
        metadata_path, separator="\t"
    )
    pl.DataFrame({
        "species": np.repeat(names, raw.shape[1]),
        "orthogroup": np.tile([f"OG{i}" for i in range(raw.shape[1])], len(names)),
        "tpm": raw.ravel(),
    }).write_csv(tpm_path, separator="\t")
    config = _config()
    config.data.metadata_path = str(metadata_path)
    config.data.tpm_path = str(tpm_path)
    config.runtime.n_jobs = n_jobs
    config.runtime.execution_stage = "cv_only"
    config.model_selection.search_space = {"lambda": [0.1, 0.01], "alpha": [1.0]}
    config.sampling.strategy = sampling
    if sampling == "group_balanced":
        config.sampling.max_samples_per_label_per_group = 2
    manifest = build_split_artifacts(config).split_manifest
    with monkeypatch.context() as direct:
        direct.setattr(cv, "_build_inner_sparse_count_cache", lambda *_: None)
        expected = cv.run_outer_cv(config, manifest)
    actual = cv.run_outer_cv(config, manifest)
    for field in fields(expected):
        if field.name == "timing":
            continue
        before, after = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(before, pl.DataFrame):
            if n_jobs > 1 and field.name == "metrics_cv":
                # Existing macro means follow thread completion order. Even two
                # direct-scan runs differ at roundoff; per-fold/micro metrics,
                # coefficients, selected models and every prediction stay exact.
                macro = pl.col("aggregate_scope") == "macro"
                assert_frame_equal(before.filter(~macro), after.filter(~macro), check_exact=True)
                assert_frame_equal(
                    before.filter(macro), after.filter(macro),
                    check_exact=False, rel_tol=0, abs_tol=5e-16,
                )
            else:
                assert_frame_equal(before, after, check_exact=True)
        else:
            assert after == before, field.name
