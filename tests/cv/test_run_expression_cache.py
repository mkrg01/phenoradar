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
from phenoradar.timing import TimingRecorder


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
    # Fold completion order is intentionally unconstrained under parallelism.
    assert sorted(actual.warnings) == sorted(expected.warnings)


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


@pytest.mark.parametrize("max_pivot_cells", [64, 50_000_000])
@pytest.mark.parametrize("fill", [0, "nan"])
def test_streamed_cache_preserves_wide_matrix_and_duplicate_coordinates(
    tmp_path: Path, max_pivot_cells: int, fill: int | str
) -> None:
    config = _config(tmp_path)
    config.preprocess.max_pivot_cells = max_pivot_cells
    config.preprocess.absent_feature_fill = fill
    features = [f"f{i:04d}" for i in range(1024)]
    # Split duplicate coordinates across distant input rows and reverse the
    # requested feature order so hash-group output order cannot affect alignment.
    rows = ["species\torthogroup\ttpm"]
    for species in ["a", "b"]:
        rows.extend(f" {species} \t {feature} \t0.25" for feature in features)
    rows.append("a\tONLY_A\t0")
    rows.append("ignored\t\tbad")
    for species in ["b", "a"]:
        rows.extend(f"{species}\t{feature}\t0.5" for feature in reversed(features))
    Path(config.data.tpm_path).write_text("\n".join(rows) + "\n")
    order = ["ONLY_A", *reversed(features), "ABSENT"]
    species_order = ["b", "a", "b"]
    direct = cv_mod.ExpressionMatrixBuilder(config)
    expected, expected_names = direct.build_matrix(species_order, feature_order=order)
    with RunExpressionCache() as cache:
        builder = cache.get_builder(config)
        builder.cache_species(["a", "b"])
        actual, names = builder.build_matrix(species_order, feature_order=order)
    assert names == expected_names == order
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 1:-1], np.full((3, len(features)), 0.75))
    assert actual[1, 0] == 0.0  # Observed zero must remain distinct from absence.
    missing = np.nan if fill == "nan" else 0.0
    np.testing.assert_array_equal(actual[:, -1], np.full(3, missing))
    np.testing.assert_array_equal(actual[[0, 2], 0], np.full(2, missing))


@pytest.mark.parametrize(
    "values,feature,total,line,reason",
    [
        (["1", "bad", "-2"], "F", 2, 3, "negative"),
        (["1", "", "2"], "F", 1, 3, "missing"),
        (["1", "NaN", "2"], "F", 1, 3, "non-finite"),
        (["1", "1e308", "1e308"], "F", 1, 2, "non-finite-after-sum"),
        (["1", "2", "3"], " ", 3, 2, "missing-feature"),
        (["bad", "1", "2"], "", 3, 2, "non-numeric,missing-feature"),
    ],
)
def test_streamed_cache_preserves_invalid_duplicate_diagnostics(
    tmp_path: Path,
    values: list[str],
    feature: str,
    total: int,
    line: int,
    reason: str,
) -> None:
    config = _config(tmp_path)
    rows = ["species\torthogroup\ttpm", *(f"a\t{feature}\t{v}" for v in values)]
    rows.extend(["a\tVALID\t1", "ignored\t\tbad"])
    Path(config.data.tpm_path).write_text("\n".join(rows) + "\n")
    with pytest.raises(CVError) as direct_error:
        cv_mod.ExpressionMatrixBuilder(config).build_matrix(["a"], feature_order=["VALID"])
    with RunExpressionCache() as cache, pytest.raises(CVError) as cached_error:
        cache.get_builder(config).cache_species(["a"])
    message = str(cached_error.value)
    assert message == str(direct_error.value)
    assert f"invalid_rows={total};" in message
    assert f"first_invalid_line={line}:" in message
    assert f"example_reasons_in_coordinate=({reason})" in message


def test_streamed_cache_sums_many_floating_point_duplicates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rng = np.random.default_rng(42)
    values = rng.uniform(0.0, 100.0, size=(4000, 8))
    rows = ["species\torthogroup\ttpm"]
    rows.extend(
        f"a\tF{feature}\t{value}"
        for row in values
        for feature, value in enumerate(row)
    )
    Path(config.data.tpm_path).write_text("\n".join(rows) + "\n")
    direct, names = cv_mod.ExpressionMatrixBuilder(config).build_matrix(["a"])
    with RunExpressionCache() as cache:
        builder = cache.get_builder(config)
        builder.cache_species(["a"])
        actual, actual_names = builder.build_matrix(["a"])
    assert actual_names == names == [f"F{i}" for i in range(8)]
    # Parallel reductions can differ in their final floating-point bits.
    np.testing.assert_allclose(actual, direct, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual[0], values.sum(axis=0), rtol=1e-12, atol=1e-14)


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


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("aggregation", ["mean", "median"])
@pytest.mark.parametrize("mode", ["none", "log1p", "neutral", "forest", "rank", "sampled", "dense"])
def test_deferred_inference_matches_eager_full_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_jobs: int,
    aggregation: str,
    mode: str,
) -> None:
    config = _config(tmp_path)
    config.runtime.n_jobs = n_jobs
    config.ensemble.probability_aggregation = aggregation
    config.preprocess.ranked_feature_filter.method = "variance"
    config.preprocess.ranked_feature_filter.max_features = 1
    config.preprocess.max_pivot_cells = 8
    config.model_selection.selected_candidate_count = 2
    if mode in {"none", "log1p"}:
        config.preprocess.expression_transform.method = mode
    elif mode == "rank":
        config.preprocess.expression_transform.method = "sample_percentile_rank"
    elif mode == "neutral":
        config.preprocess.absent_feature_fill = "nan"
        config.preprocess.missing_expression.method = "neutral"
        config.preprocess.missing_expression.zero_as_missing = True
        config.abstention.enabled = True
    elif mode == "forest":
        config.model.name = "random_forest"
        config.preprocess.absent_feature_fill = "nan"
        config.model_selection.search_space = {"n_estimators": [3, 5]}
    elif mode == "dense":
        config.preprocess.sparse_feature_filter.enabled = False
        config.preprocess.low_variance_filter.enabled = False
        config.preprocess.ranked_feature_filter.method = "none"
        config.preprocess.correlation_filter.enabled = False
    elif mode == "sampled":
        metadata_path, tpm_path = Path(config.data.metadata_path), Path(config.data.tpm_path)
        with metadata_path.open("a") as out:
            for line in metadata_path.read_text().splitlines()[1:]:
                species, label, group = line.split("\t")
                if group:
                    out.write(f"{species}_copy\t{label}\t{group}\n")
        with tpm_path.open("a") as out:
            for line in tpm_path.read_text().splitlines()[1:]:
                species, feature, value = line.split("\t")
                if species.startswith("s"):
                    out.write(f"{species}_copy\t{feature}\t{float(value) * 1.1}\n")
        config.sampling.strategy = "group_balanced"
        config.sampling.max_samples_per_label_per_group = 1
        config.sampling.sampled_set_count = 2
    split = build_split_artifacts(config).split_manifest
    with monkeypatch.context() as reference:
        reference.setattr(cv_mod, "_can_defer_outer_inference", lambda _config: False)
        expected = run_outer_cv(config, split)
    actual = run_outer_cv(config, split)
    _assert_tables_equal(expected, actual)
    stages = set(actual.timing["stage"])
    assert ("inference_matrix_build" in stages) == (mode != "rank")
    assert ("deferred_inference" in stages) == (mode != "rank")
    assert "inference" in set(actual.top_feature_expression["species"])
    if mode == "sampled":
        assert set(actual.retained_features["sample_set_id"]) == {0, 1}
    if mode == "dense":
        assert set(actual.retained_features["feature"]) == {"BASE", "DOWN", "UP"}


def test_deferred_inference_builds_only_retained_union_after_all_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.preprocess.ranked_feature_filter.method = "variance"
    config.preprocess.ranked_feature_filter.max_features = 1
    split = build_split_artifacts(config)
    completed_fits: list[str] = []
    inference_builds: list[tuple[tuple[int, ...], list[str]]] = []
    build = cv_mod.ExpressionMatrixBuilder.build_matrix
    fit = cv_mod._fit_outer_sample_set

    def tracked_fit(**kwargs: object) -> object:
        matrix = kwargs["x_inference_matrix"]
        assert isinstance(matrix, np.ndarray) and matrix.shape[0] == 0
        result = fit(**kwargs)
        completed_fits.append(str(kwargs["fold_id"]))
        return result

    def tracked_build(
        self: cv_mod.ExpressionMatrixBuilder,
        species_order: list[str],
        feature_order: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        if "inference" in species_order:
            assert len(completed_fits) == split.fold_count
            assert species_order == ["inference"] and feature_order is not None
        matrix, names = build(self, species_order, feature_order)
        if "inference" in species_order:
            inference_builds.append((matrix.shape, names))
        return matrix, names

    monkeypatch.setattr(cv_mod, "_fit_outer_sample_set", tracked_fit)
    monkeypatch.setattr(cv_mod.ExpressionMatrixBuilder, "build_matrix", tracked_build)
    result = run_outer_cv(config, split.split_manifest)
    retained = set(result.retained_features["feature"])
    assert len(inference_builds) == 1
    shape, names = inference_builds[0]
    assert shape == (1, len(retained)) and set(names) == retained
    assert len(names) < 3


def test_deferred_inference_preserves_zero_coefficient_abstention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.preprocess.absent_feature_fill = "nan"
    config.preprocess.missing_expression.method = "neutral"
    config.preprocess.missing_expression.zero_as_missing = True
    config.abstention.enabled = True
    config.model_selection.search_space = {"lambda": [1e6]}
    split = build_split_artifacts(config).split_manifest
    with monkeypatch.context() as reference:
        reference.setattr(cv_mod, "_can_defer_outer_inference", lambda _config: False)
        expected = run_outer_cv(config, split)
    actual = run_outer_cv(config, split)
    _assert_tables_equal(expected, actual)
    predictions = actual.inference_predictions_by_fold
    assert predictions is not None
    assert set(predictions["abstention_reason"]) == {"no_informative_coefficients"}


def test_input_timing_separates_preparation_from_matrix_work(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.preprocess.max_pivot_cells = 8
    split = build_split_artifacts(config).split_manifest
    recorder = TimingRecorder()
    with RunExpressionCache() as cache:
        cache.prepare(config, split, recorder)
        preparation = recorder.to_frame()
        assert set(preparation["scope"]) == {"expression_input"}
        assert {"input_normalize_cache", "input_validation_read"} == set(preparation["stage"])
        cv = run_outer_cv(config, split, expression_cache=cache, timing_recorder=recorder)
        refit = run_final_refit(config, split, expression_cache=cache, timing_recorder=recorder)
    for artifact in (cv, refit):
        assert {"input_schema_read", "input_matrix_read", "input_dense_assembly"}.issubset(
            set(artifact.timing["stage"])
        )
        assert "input_normalize_cache" not in artifact.timing["stage"]
    assert cv.timing["started_at_sec"].min() >= preparation["ended_at_sec"].max()
