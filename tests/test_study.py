from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from phenoradar.study import (
    _ranked_feature_sensitivity_figures,
    generate_study_report,
)

_METRICS = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "mcc",
    "brier",
    "log_loss",
)


def _write_condition(
    run_dir: Path,
    *,
    points: dict[str, float],
    replicates: dict[str, list[float]],
) -> None:
    cv_tables = run_dir / "cv" / "tables"
    split_tables = run_dir / "split" / "tables"
    cv_tables.mkdir(parents=True)
    split_tables.mkdir(parents=True)
    (run_dir / "resolved_config.yml").write_text("runtime:\n  seed: 42\n", encoding="utf-8")
    pl.DataFrame(
        {
            "aggregate_scope": [None] * len(_METRICS) + ["micro"] * len(_METRICS),
            "fold_id": [1] * len(_METRICS) + [None] * len(_METRICS),
            "metric": [*list(_METRICS), *list(_METRICS)],
            "metric_value": [
                *[points[metric] for metric in _METRICS],
                *[points[metric] for metric in _METRICS],
            ],
        }
    ).write_csv(cv_tables / "metrics_cv.tsv", separator="\t", null_value="NA")
    pl.DataFrame(
        {
            "metric": list(_METRICS),
            "point_estimate": [points[metric] for metric in _METRICS],
            "ci_lower": [points[metric] - 0.1 for metric in _METRICS],
            "ci_upper": [points[metric] + 0.1 for metric in _METRICS],
            "confidence_level": [0.95] * len(_METRICS),
        }
    ).write_csv(cv_tables / "group_bootstrap_metrics.tsv", separator="\t")
    replicate_rows = [
        {
            "resample_id": resample_id,
            "metric": metric,
            "metric_value": value,
        }
        for metric in _METRICS
        for resample_id, value in enumerate(replicates[metric], start=1)
    ]
    pl.DataFrame(replicate_rows).write_csv(
        cv_tables / "group_bootstrap_replicates.tsv",
        separator="\t",
    )
    pl.DataFrame(
        {
            "species": ["sp1"],
            "pool": ["validation"],
            "fold_id": ["0"],
            "group_id": ["g1"],
            "contrast_group_id": ["g1"],
            "label": [1],
        }
    ).write_csv(split_tables / "split_manifest.tsv", separator="\t")
    pl.DataFrame(
        {"fold_id": ["0"], "species": ["sp1"], "label": [1], "prob": [0.8]}
    ).write_csv(cv_tables / "prediction_cv.tsv", separator="\t")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "cv_completed",
                "dataset_fingerprint": "a" * 64,
                "split_fingerprint": "b" * 64,
            }
        ),
        encoding="utf-8",
    )


def test_generate_study_report_preserves_order_and_uses_paired_replicates(
    tmp_path: Path,
) -> None:
    run_a = tmp_path / "conditions" / "a"
    run_b = tmp_path / "conditions" / "b"
    points_a = {
        "roc_auc": 0.9,
        "pr_auc": 0.8,
        "balanced_accuracy": 0.75,
        "mcc": 0.7,
        "brier": 0.1,
        "log_loss": 0.2,
    }
    points_b = {
        "roc_auc": 0.7,
        "pr_auc": 0.6,
        "balanced_accuracy": 0.55,
        "mcc": 0.4,
        "brier": 0.3,
        "log_loss": 0.5,
    }
    replicate_a = {
        metric: [value - 0.02, value, value + 0.02] for metric, value in points_a.items()
    }
    replicate_b = {
        metric: [value - 0.02, value, value + 0.02] for metric, value in points_b.items()
    }
    _write_condition(run_a, points=points_a, replicates=replicate_a)
    _write_condition(run_b, points=points_b, replicates=replicate_b)
    manifest_rows = [
        {
            "condition_index": 1,
            "condition_id": "cond_a",
            "condition_label": "first",
            "status": "completed",
            "run_dir": str(run_a),
        },
        {
            "condition_index": 2,
            "condition_id": "cond_b",
            "condition_label": "second",
            "status": "completed",
            "run_dir": str(run_b),
        },
    ]

    artifacts = generate_study_report(tmp_path, manifest_rows)

    assert (
        artifacts.condition_metrics.select("condition_index")
        .unique(maintain_order=True)
        .get_column("condition_index")
        .to_list()
        == [1, 2]
    )
    comparisons = artifacts.pairwise_comparisons
    assert comparisons.get_column("condition_a_id").unique().to_list() == ["cond_a"]
    assert comparisons.get_column("condition_b_id").unique().to_list() == ["cond_b"]
    assert comparisons.filter(pl.col("improvement_a_over_b") <= 0).height == 0
    assert comparisons.filter(pl.col("ci_lower") <= 0).height == 0
    assert comparisons.get_column("probability_a_better").unique().to_list() == [1.0]
    assert comparisons.get_column("n_valid_resamples").unique().to_list() == [3]
    assert len(artifacts.figure_paths) == 3
    assert all(path.exists() for path in artifacts.figure_paths)
    assert not (tmp_path / "figures" / "pairwise_improvement.svg").exists()
    condition_svg = (tmp_path / "figures" / "condition_metrics.svg").read_text(
        encoding="utf-8"
    )
    assert "first" in condition_svg
    assert "second" in condition_svg


def test_ranked_feature_sensitivity_figures_use_matched_feature_counts(
    tmp_path: Path,
) -> None:
    condition_specs = [
        (1, "pair_50", "pair_aware", 50, 0.70),
        (2, "pair_100", "pair_aware", 100, 0.72),
        (3, "unpaired_50", "unpaired", 50, 0.80),
        (4, "unpaired_100", "unpaired", 100, 0.84),
    ]
    metric_rows = [
        {
            "condition_index": index,
            "condition_id": condition_id,
            "condition_label": f"method={method}; max_features={max_features}",
            "metric": metric,
            "point_estimate": point,
            "ci_lower": point - 0.05,
            "ci_upper": point + 0.05,
            "confidence_level": 0.95,
        }
        for index, condition_id, method, max_features, point in condition_specs
        for metric in _METRICS
    ]
    pairwise_rows = [
        {
            "condition_a_id": f"pair_{max_features}",
            "condition_b_id": f"unpaired_{max_features}",
            "metric": metric,
            "improvement_a_over_b": -0.1,
            "ci_lower": -0.15,
            "ci_upper": -0.05,
        }
        for max_features in (50, 100)
        for metric in _METRICS
    ]
    manifest_rows = [
        {
            "condition_id": condition_id,
            "varying_parameters_json": json.dumps(
                {
                    "preprocess.ranked_feature_filter.method": method,
                    "preprocess.ranked_feature_filter.max_features": max_features,
                }
            ),
        }
        for _index, condition_id, method, max_features, _point in condition_specs
    ]
    output_dir = tmp_path / "figures"
    output_dir.mkdir()

    paths = _ranked_feature_sensitivity_figures(
        pl.DataFrame(metric_rows),
        pl.DataFrame(pairwise_rows),
        manifest_rows,
        output_dir=output_dir,
    )

    assert len(paths) == 6
    assert all(path.exists() for path in paths)
    sensitivity_svg = (output_dir / "ranked_feature_sensitivity.svg").read_text(
        encoding="utf-8"
    )
    difference_svg = (
        output_dir / "ranked_feature_method_difference.svg"
    ).read_text(encoding="utf-8")
    assert "pair_aware" in sensitivity_svg
    assert "unpaired" in sensitivity_svg
    assert "unpaired improvement over pair_aware" in difference_svg


def test_generate_study_report_aggregates_training_group_subsample_repeats(
    tmp_path: Path,
) -> None:
    specs = [
        (1, "limited_1", 2, 1, 0.60),
        (2, "limited_2", 2, 2, 0.80),
        (3, "full", None, 1, 0.90),
    ]
    manifest_rows: list[dict[str, object]] = []
    for index, condition_id, max_groups, repeat, point in specs:
        run_dir = tmp_path / "conditions" / condition_id
        points = {metric: point for metric in _METRICS}
        replicates = {metric: [point - 0.01, point, point + 0.01] for metric in _METRICS}
        _write_condition(run_dir, points=points, replicates=replicates)
        model_tables = run_dir / "model" / "tables"
        model_tables.mkdir(parents=True)
        effective_count = 4 if max_groups is None else max_groups
        pl.DataFrame(
            {
                "scope": ["outer_fold", "outer_fold"],
                "fold_id": ["1", "2"],
                "group_subsample_repeat": [repeat, repeat],
                "n_training_groups_selected": [effective_count, effective_count],
            }
        ).write_csv(
            model_tables / "training_group_subsets.tsv",
            separator="\t",
        )
        manifest_rows.append(
            {
                "condition_index": index,
                "condition_id": condition_id,
                "condition_label": condition_id,
                "varying_parameters_json": json.dumps(
                    {
                        "sampling.max_training_groups": max_groups,
                        "sampling.group_subsample_repeat": repeat,
                    }
                ),
                "status": "completed",
                "run_dir": str(run_dir),
            }
        )

    artifacts = generate_study_report(tmp_path, manifest_rows)

    assert artifacts.training_group_sensitivity is not None
    sensitivity = artifacts.training_group_sensitivity
    limited = sensitivity.filter(
        (pl.col("max_training_groups") == 2) & (pl.col("metric") == "roc_auc")
    ).row(0, named=True)
    assert limited["n_subset_repeats"] == 2
    assert limited["point_estimate_mean"] == pytest.approx(0.7)
    assert limited["point_estimate_q1"] == pytest.approx(0.65)
    assert limited["point_estimate_q3"] == pytest.approx(0.75)
    full = sensitivity.filter(
        pl.col("full_training_set") & (pl.col("metric") == "roc_auc")
    ).row(0, named=True)
    assert full["max_training_groups_label"] == "full"
    assert full["effective_training_groups_mean"] == 4.0
    assert (tmp_path / "tables" / "training_group_sensitivity.tsv").exists()
    for extension in ("svg", "pdf", "png"):
        assert (tmp_path / "figures" / f"training_group_sensitivity.{extension}").exists()
