from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import phenoradar.cv as cv_mod
from phenoradar.config import AppConfig
from phenoradar.cv import (
    CVArtifacts,
    CVError,
    FinalRefitArtifacts,
    RunExpressionCache,
    run_final_refit,
    run_outer_cv,
)
from phenoradar.split import build_split_artifacts


def _config(tmp_path: Path) -> AppConfig:
    metadata = ["species\tC4\tcontrast_pair_id"]
    expression = ["species\torthogroup\ttpm"]
    for group in range(4):
        for label in (0, 1):
            species = f"s{group}_{label}"
            metadata.append(f"{species}\t{label}\tg{group}")
            expression.append(f"{species}\tUP\t{10 + group if label else 1 + group / 10}")
            if not (group == 0 and label == 0):
                expression.append(f"{species}\tDOWN\t{1 + group / 10 if label else 10 + group}")
            expression.append(f"{species}\tBASE\t3")
    metadata.extend(["external\t1\t", "inference\t\t"])
    expression.extend(
        [
            "external\tUP\t12",
            "external\tDOWN\t0",
            "external\tBASE\t3",
            "external\tEXTERNAL_ONLY\t1000000",
            "inference\tUP\t1",
            "inference\tUP\t1",  # Duplicate coordinates must sum.
            "inference\tBASE\t3",
            "inference\tINFERENCE_ONLY\t2000000",
            "unselected\tIGNORED\tbad",  # Never consumed by either stage.
        ]
    )
    metadata_path = tmp_path / "metadata.tsv"
    tpm_path = tmp_path / "tpm.tsv"
    metadata_path.write_text("\n".join(metadata) + "\n")
    tpm_path.write_text("\n".join(expression) + "\n")
    return AppConfig.model_validate(
        {
            "data": {"metadata_path": str(metadata_path), "tpm_path": str(tpm_path)},
            "runtime": {"execution_stage": "full_run", "n_jobs": 2},
            "sampling": {
                "strategy": "all_samples",
                "sampled_set_count": 1,
                "max_samples_per_label_per_group": None,
            },
            "preprocess": {
                "sparse_feature_filter": {
                    "enabled": True,
                    "scope": "all_samples",
                    "min_nonzero_fraction": 0.25,
                },
            },
            "model_selection": {
                "selected_candidate_count": 1,
                "inner_cv_strategy": "group_kfold",
                "inner_cv_n_splits": 2,
                "search_space": {"lambda": [0.01, 0.1]},
            },
        }
    )


def _assert_tables_equal(
    expected: CVArtifacts | FinalRefitArtifacts, actual: CVArtifacts | FinalRefitArtifacts
) -> None:
    for field in fields(expected):
        if field.name == "timing":
            continue
        a, b = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(a, pl.DataFrame):
            assert_frame_equal(a, b, check_exact=False, rel_tol=1e-12, abs_tol=1e-14)
        elif a is None:
            assert b is None, field.name
    assert actual.warnings == expected.warnings


@pytest.mark.parametrize("max_pivot_cells", [8, 50_000_000])
@pytest.mark.parametrize(
    "transform,neutral",
    [
        ("none", False),
        ("log1p", False),
        ("sample_rank", False),
        ("sample_percentile_rank", False),
        ("log1p", True),
    ],
)
def test_shared_cache_matches_independent_cv_and_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_pivot_cells: int,
    transform: str,
    neutral: bool,
) -> None:
    config = _config(tmp_path)
    config.preprocess.max_pivot_cells = max_pivot_cells
    config.preprocess.expression_transform.method = transform
    if neutral:
        config.preprocess.absent_feature_fill = "nan"
        config.preprocess.missing_expression.method = "neutral"
        config.preprocess.missing_expression.zero_as_missing = True
        config.abstention.enabled = True
    split = build_split_artifacts(config).split_manifest
    expected_cv = run_outer_cv(config, split)
    expected_refit = run_final_refit(config, split)

    raw_calls: list[list[str]] = []
    original = cv_mod.ExpressionMatrixBuilder._raw_long_scan_for_species

    def tracked(self: cv_mod.ExpressionMatrixBuilder, species: list[str]) -> pl.LazyFrame:
        raw_calls.append(list(species))
        return original(self, species)

    monkeypatch.setattr(cv_mod.ExpressionMatrixBuilder, "_raw_long_scan_for_species", tracked)
    with RunExpressionCache() as cache:
        actual_cv = run_outer_cv(config, split, expression_cache=cache)
        builder = cache.get_builder(config)
        cache_path = builder._cached_long_path
        assert cache_path is not None and cache_path.exists()
        actual_refit = run_final_refit(config, split, expression_cache=cache)
        assert builder._cached_long_path == cache_path

    assert not cache_path.exists()
    assert builder._cached_long_path is None
    assert len(raw_calls) == 1
    assert set(raw_calls[0]) == set(split["species"])
    _assert_tables_equal(expected_cv, actual_cv)
    _assert_tables_equal(expected_refit, actual_refit)
    assert not set(actual_cv.retained_features["feature"]) & {
        "EXTERNAL_ONLY",
        "INFERENCE_ONLY",
    }
    assert actual_refit.transform_feature_names == expected_refit.transform_feature_names
    assert len(actual_refit.model_entries) == len(expected_refit.model_entries)
    for a, b in zip(expected_refit.model_entries, actual_refit.model_entries, strict=True):
        assert a.feature_names == b.feature_names
        np.testing.assert_allclose(a.model.coef_, b.model.coef_, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(a.model.intercept_, b.model.intercept_, rtol=1e-12, atol=1e-14)


def test_cv_only_cache_does_not_consume_target_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.runtime.execution_stage = "cv_only"
    tpm_path = Path(config.data.tpm_path)
    with tpm_path.open("a") as out:
        out.write("external\tBAD\t-1\ninference\tBAD\tnot_numeric\n")
    split = build_split_artifacts(config).split_manifest
    expected = run_outer_cv(config, split)
    with RunExpressionCache() as cache:
        actual = run_outer_cv(config, split, expression_cache=cache)
        builder = cache.get_builder(config)
        assert builder._cached_species == {f"s{g}_{label}" for g in range(4) for label in (0, 1)}
    _assert_tables_equal(expected, actual)


@pytest.mark.parametrize("target_pools", [[], ["external_test"], ["discovery_inference"]])
def test_shared_cache_supports_empty_target_pools(tmp_path: Path, target_pools: list[str]) -> None:
    config = _config(tmp_path)
    split = build_split_artifacts(config).split_manifest.filter(
        pl.col("pool").is_in(["train", "validation", *target_pools])
    )
    expected_cv = run_outer_cv(config, split)
    expected_refit = run_final_refit(config, split)
    with RunExpressionCache() as cache:
        actual_cv = run_outer_cv(config, split, expression_cache=cache)
        actual_refit = run_final_refit(config, split, expression_cache=cache)
    _assert_tables_equal(expected_cv, actual_cv)
    _assert_tables_equal(expected_refit, actual_refit)


def test_full_run_cache_validates_ignored_target_features_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    with Path(config.data.tpm_path).open("a") as out:
        out.write("external\tUNUSED_BAD\t-1\n")
    split = build_split_artifacts(config).split_manifest
    paths: list[Path] = []
    original = cv_mod.TemporaryDirectory

    def tempdir(*args: object, **kwargs: object) -> object:
        directory = original(*args, **kwargs)
        paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(cv_mod, "TemporaryDirectory", tempdir)
    with (
        pytest.raises(CVError, match=r"external.*UNUSED_BAD.*negative"),
        RunExpressionCache() as cache,
    ):
        run_outer_cv(config, split, expression_cache=cache)
    assert paths and all(not path.exists() for path in paths)


def test_run_cache_cleans_up_if_a_later_stage_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="later stage failed"), RunExpressionCache() as cache:
        builder = cache.get_builder(config)
        builder.cache_species(["s0_0"])
        path = builder._cached_long_path
        assert path is not None and path.exists()
        raise RuntimeError("later stage failed")
    assert not path.exists()
    assert builder._cached_species is None
    cache.close()  # Idempotent cleanup.


@pytest.mark.parametrize(
    "setting", ["tpm_path", "value_col", "absent_feature_fill", "max_pivot_cells"]
)
def test_run_cache_rejects_changed_input_settings(tmp_path: Path, setting: str) -> None:
    config = _config(tmp_path)
    with RunExpressionCache() as cache:
        cache.get_builder(config)
        owner = config.data if setting in {"tpm_path", "value_col"} else config.preprocess
        value = {
            "tpm_path": "different.tsv",
            "value_col": "other",
            "absent_feature_fill": "nan",
            "max_pivot_cells": 1,
        }[setting]
        setattr(owner, setting, value)
        with pytest.raises(CVError, match="different input settings"):
            cache.get_builder(config)
