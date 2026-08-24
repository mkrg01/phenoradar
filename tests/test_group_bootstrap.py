from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from phenoradar.group_bootstrap import GroupBootstrapError, run_oof_group_bootstrap
from phenoradar.metrics import FIXED_PROBABILITY_THRESHOLD_VALUE, binary_probability_metrics


def _oof_predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fold_id": ["1", "1", "2", "3", "3", "3"],
            "species": ["a1", "a2", "b1", "c1", "c2", "c3"],
            "label": [1, 0, 1, 0, 1, 0],
            "prob": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        }
    )


def _split_manifest() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "species": ["a1", "a2", "b1", "c1", "c2", "c3"],
            "pool": ["validation"] * 6,
            "fold_id": ["1", "1", "2", "3", "3", "3"],
            "group_id": ["g1", "g1", "g2", "g3", "g3", "g3"],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )


def test_group_bootstrap_resamples_intact_groups_and_reports_expected_schema() -> None:
    artifacts = run_oof_group_bootstrap(
        oof_predictions=_oof_predictions(),
        split_manifest=_split_manifest(),
        group_col="contrast_pair_id",
        n_resamples=20,
        confidence_level=0.95,
        runtime_seed=42,
    )

    assert artifacts.n_groups == 3
    assert artifacts.summary.height == 6
    assert artifacts.replicates.height == 120
    assert set(artifacts.summary.select("metric").to_series().to_list()) == {
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "brier",
        "log_loss",
    }
    assert artifacts.summary.select("group_col").unique().item() == "contrast_pair_id"
    assert artifacts.summary.select("n_groups").unique().item() == 3
    assert artifacts.replicates.select("n_sampled_groups").unique().item() == 3

    rng = np.random.default_rng(artifacts.seed)
    sampled_group_indices = rng.integers(0, 3, size=3)
    rows_by_group = [np.array([0, 1]), np.array([2]), np.array([3, 4, 5])]
    sampled_rows = np.concatenate([rows_by_group[index] for index in sampled_group_indices])
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=int)[sampled_rows]
    probability = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], dtype=float)[sampled_rows]
    expected_brier = binary_probability_metrics(
        y_true,
        probability,
        threshold=FIXED_PROBABILITY_THRESHOLD_VALUE,
    )["brier"]
    first_brier = artifacts.replicates.filter(
        (pl.col("resample_id") == 1) & (pl.col("metric") == "brier")
    ).row(0, named=True)
    assert first_brier["metric_value"] == pytest.approx(expected_brier)
    assert first_brier["n_unique_sampled_groups"] == int(
        np.unique(sampled_group_indices).size
    )
    assert first_brier["n_species_with_multiplicity"] == int(sampled_rows.size)


def test_group_bootstrap_is_deterministic_for_runtime_seed() -> None:
    kwargs = {
        "oof_predictions": _oof_predictions(),
        "split_manifest": _split_manifest(),
        "group_col": "family_id",
        "n_resamples": 25,
        "confidence_level": 0.9,
    }

    first = run_oof_group_bootstrap(**kwargs, runtime_seed=123)
    second = run_oof_group_bootstrap(**kwargs, runtime_seed=123)
    changed = run_oof_group_bootstrap(**kwargs, runtime_seed=124)

    assert first.summary.equals(second.summary)
    assert first.replicates.equals(second.replicates)
    assert not first.replicates.equals(changed.replicates)


def test_group_bootstrap_marks_single_label_two_class_metrics_invalid() -> None:
    oof = pl.DataFrame(
        {
            "species": ["p1", "p2", "n1", "n2"],
            "label": [1, 1, 0, 0],
            "prob": [0.9, 0.8, 0.2, 0.1],
        }
    )
    manifest = pl.DataFrame(
        {
            "species": ["p1", "p2", "n1", "n2"],
            "pool": ["validation"] * 4,
            "group_id": ["positive", "positive", "negative", "negative"],
        }
    )

    artifacts = run_oof_group_bootstrap(
        oof_predictions=oof,
        split_manifest=manifest,
        group_col="family_id",
        n_resamples=200,
        confidence_level=0.95,
        runtime_seed=42,
    )

    summary = {row["metric"]: row for row in artifacts.summary.iter_rows(named=True)}
    for metric_name in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc"):
        assert 0 < summary[metric_name]["n_valid_resamples"] < 200
    for metric_name in ("brier", "log_loss"):
        assert summary[metric_name]["n_valid_resamples"] == 200
    assert any("invalid counts" in warning for warning in artifacts.warnings)


def test_group_bootstrap_rejects_oof_split_species_mismatch() -> None:
    manifest = _split_manifest().filter(pl.col("species") != "c3")

    with pytest.raises(GroupBootstrapError, match="OOF/split species mismatch"):
        run_oof_group_bootstrap(
            oof_predictions=_oof_predictions(),
            split_manifest=manifest,
            group_col="family_id",
            n_resamples=10,
            confidence_level=0.95,
            runtime_seed=42,
        )


def test_group_bootstrap_requires_at_least_two_groups() -> None:
    manifest = _split_manifest().with_columns(pl.lit("one").alias("group_id"))

    with pytest.raises(GroupBootstrapError, match="at least two groups"):
        run_oof_group_bootstrap(
            oof_predictions=_oof_predictions(),
            split_manifest=manifest,
            group_col="family_id",
            n_resamples=10,
            confidence_level=0.95,
            runtime_seed=42,
        )
