from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import phenoradar.split as split_mod
from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.split import SplitError, build_split_artifacts


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_config(tmp_path: Path, metadata: Path, tpm: Path) -> Path:
    return _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )


def _fixture_data(tmp_path: Path) -> tuple[Path, Path]:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t0\tg1",
                "sp3\t1\tg2",
                "sp4\t0\tg2",
                "sp5\t1\t",
                "sp6\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
                "sp6\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    return metadata, tpm


def test_build_split_artifacts_success(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    artifacts = build_split_artifacts(config)

    assert artifacts.fold_count == 2
    assert artifacts.pool_counts["training_validation"] == 4
    assert artifacts.pool_counts["external_test"] == 1
    assert artifacts.pool_counts["discovery_inference"] == 1
    assert artifacts.expression_rows_excluded == 0
    assert artifacts.split_manifest.height > 0
    assert artifacts.fold_validation_groups.height == 2
    assert artifacts.fold_diagnostics.height == 2


def test_split_group_col_can_differ_from_contrast_pair_col(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\tfamily",
                "sp1\t1\tcp1\tfamily_a",
                "sp2\t0\tcp1\tfamily_a",
                "sp3\t1\tcp2\tfamily_b",
                "sp4\t0\tcp2\tfamily_b",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
            ]
        )
        + "\n",
    )
    cfg = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
  contrast_pair_col: contrast_pair_id
split:
  group_col: family
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([cfg])

    manifest = build_split_artifacts(config).split_manifest
    rows = (
        manifest.filter(pl.col("pool").is_in(["train", "validation"]))
        .group_by("species")
        .agg(
            pl.col("group_id").drop_nulls().first().alias("group_id"),
            pl.col("contrast_group_id").drop_nulls().first().alias("contrast_group_id"),
        )
        .sort("species")
        .to_dicts()
    )

    assert rows == [
        {"species": "sp1", "group_id": "family_a", "contrast_group_id": "cp1"},
        {"species": "sp2", "group_id": "family_a", "contrast_group_id": "cp1"},
        {"species": "sp3", "group_id": "family_b", "contrast_group_id": "cp2"},
        {"species": "sp4", "group_id": "family_b", "contrast_group_id": "cp2"},
    ]


def test_pair_aware_filter_allows_missing_contrast_pairs_for_rank_split(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\torder",
                "sp1\t1\tcp1\torder_a",
                "sp2\t0\tcp1\torder_a",
                "sp3\t1\t\torder_b",
                "sp4\t0\t\torder_b",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
            ]
        )
        + "\n",
    )
    cfg = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
  contrast_pair_col: contrast_pair_id
split:
  group_col: order
preprocess:
  ranked_feature_filter:
    method: pair_aware
    max_features: 1
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([cfg])

    manifest = build_split_artifacts(config).split_manifest

    assert manifest.filter(pl.col("pool").is_in(["train", "validation"])).height > 0
    assert manifest.filter(pl.col("contrast_group_id").is_null()).height > 0


def _both_label_groups_config(
    tmp_path: Path,
    *,
    require_both_labels: bool | None = True,
    outer_cv_strategy: str = "logo",
    outer_cv_n_splits: int | None = None,
    sampling_strategy: str = "all_samples",
    rows: list[str] | None = None,
) -> AppConfig:
    if rows is None:
        rows = [
            "a_pos\t1\tcp_a\tfamily_a\tno",
            "a_neg\t0\tcp_a\tfamily_a\tno",
            "b_pos\t1\t\tfamily_b\tno",
            "b_neg\t0\t\tfamily_b\tno",
            "c_pos\t1\tcp_c\tfamily_c\tno",
            "c_neg\t0\tcp_c\tfamily_c\tno",
            "positive_1\t1\t\tpositive_only\tno",
            "positive_2\t1\t\tpositive_only\tno",
            "negative_1\t0\t\tnegative_only\tno",
            "negative_2\t0\t\tnegative_only\tno",
            "remaining_neg\t0\t\texcluded_positive\tno",
            "excluded_pos\t1\t\texcluded_positive\tyes",
            "unknown_positive_family\t\t\tpositive_only\tno",
            "unknown_only\t\t\t\tno",
        ]
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "species\tC4\tcontrast_pair_id\tfamily\texclude\n"
        + "\n".join(rows)
        + "\n",
    )
    species = [row.split("\t")[0] for row in rows if row.split("\t")[-1] != "yes"]
    tpm = _write(
        tmp_path / "tpm.tsv",
        "species\torthogroup\ttpm\n"
        + "\n".join(f"{name}\tOG1\t1.0" for name in species)
        + "\n",
    )
    enabled_line = (
        ""
        if require_both_labels is None
        else f"  require_both_labels_per_group: {str(require_both_labels).lower()}\n"
    )
    config_path = _write(
        tmp_path / "config.yml",
        f"""data:
  metadata_path: {metadata}
  tpm_path: {tpm}
  contrast_pair_col: contrast_pair_id
split:
  group_col: family
  exclude_col: exclude
  outer_cv_strategy: {outer_cv_strategy}
  outer_cv_n_splits: {outer_cv_n_splits or "null"}
{enabled_line}sampling:
  strategy: {sampling_strategy}
  max_samples_per_label_per_group: null
  sampled_set_count: 1
""",
    )
    return load_and_resolve_config([config_path])


@pytest.mark.parametrize("sampling_strategy", ["all_samples", "group_balanced"])
@pytest.mark.parametrize(
    ("outer_cv_strategy", "outer_cv_n_splits", "expected_folds"),
    [("logo", None, 3), ("group_kfold", 2, 2), ("stratified_group_kfold", 2, 2)],
)
def test_require_both_labels_routes_single_label_groups_before_cv(
    tmp_path: Path,
    sampling_strategy: str,
    outer_cv_strategy: str,
    outer_cv_n_splits: int | None,
    expected_folds: int,
) -> None:
    config = _both_label_groups_config(
        tmp_path,
        sampling_strategy=sampling_strategy,
        outer_cv_strategy=outer_cv_strategy,
        outer_cv_n_splits=outer_cv_n_splits,
    )

    artifacts = build_split_artifacts(config)
    manifest = artifacts.split_manifest
    cv_rows = manifest.filter(pl.col("pool").is_in(["train", "validation"]))
    expected_cv_species = {"a_pos", "a_neg", "b_pos", "b_neg", "c_pos", "c_neg"}

    assert artifacts.fold_count == expected_folds
    assert artifacts.pool_counts == {
        "training_validation": 6,
        "external_test": 5,
        "discovery_inference": 2,
        "excluded": 1,
    }
    assert set(cv_rows.get_column("species")) == expected_cv_species
    assert cv_rows.height == len(expected_cv_species) * expected_folds
    validation = cv_rows.filter(pl.col("pool") == "validation")
    assert validation.height == len(expected_cv_species)
    assert validation.get_column("species").n_unique() == len(expected_cv_species)
    assert cv_rows.filter(pl.col("group_id") == "family_b").get_column(
        "contrast_group_id"
    ).null_count() == 2 * expected_folds
    assert set(manifest.filter(pl.col("pool") == "external_test").get_column("species")) == {
        "positive_1", "positive_2", "negative_1", "negative_2", "remaining_neg",
    }
    assert set(
        manifest.filter(pl.col("pool") == "discovery_inference").get_column("species")
    ) == {"unknown_positive_family", "unknown_only"}
    assert "excluded_pos" not in set(manifest.get_column("species"))
    assert artifacts.fold_diagnostics.get_column("two_class_validation_metrics_defined").all()
    for fold_id in cv_rows.get_column("fold_id").unique():
        fold = cv_rows.filter(pl.col("fold_id") == fold_id)
        train = fold.filter(pl.col("pool") == "train")
        valid = fold.filter(pl.col("pool") == "validation")
        assert set(train.get_column("group_id")).isdisjoint(valid.get_column("group_id"))
        assert set(train.get_column("label")) == {0, 1}
        assert set(valid.get_column("label")) == {0, 1}


def test_family_split_allows_single_label_validation_groups_by_default(
    tmp_path: Path,
) -> None:
    default_config = _both_label_groups_config(tmp_path, require_both_labels=None)
    default_artifacts = build_split_artifacts(default_config)
    disabled_config = _both_label_groups_config(tmp_path, require_both_labels=False)
    disabled_artifacts = build_split_artifacts(disabled_config)

    assert default_config.split.require_both_labels_per_group is False
    assert default_artifacts.split_manifest.equals(disabled_artifacts.split_manifest)
    assert default_artifacts.fold_count == 6
    assert default_artifacts.pool_counts["training_validation"] == 11
    assert default_artifacts.pool_counts["external_test"] == 0
    assert set(default_artifacts.fold_diagnostics.get_column("validation_label_profile")) == {
        "both", "positive_only", "negative_only",
    }
    assert not default_artifacts.fold_diagnostics.get_column(
        "two_class_validation_metrics_defined"
    ).all()


@pytest.mark.parametrize(
    ("remaining_mixed_groups", "message"),
    [(0, "require_both_labels_per_group"), (1, "at least two split groups"), (2, "n_splits")],
)
def test_require_both_labels_rejects_insufficient_remaining_cv_groups(
    tmp_path: Path, remaining_mixed_groups: int, message: str
) -> None:
    rows = [
        "positive\t1\t\tpositive_only\tno",
        "negative\t0\t\tnegative_only\tno",
    ]
    for index in range(remaining_mixed_groups):
        rows.extend([
            f"mixed_{index}_pos\t1\t\tmixed_{index}\tno",
            f"mixed_{index}_neg\t0\t\tmixed_{index}\tno",
        ])
    config = _both_label_groups_config(
        tmp_path,
        rows=rows,
        outer_cv_strategy="group_kfold",
        outer_cv_n_splits=3,
    )

    with pytest.raises(SplitError, match=message):
        build_split_artifacts(config)


@pytest.mark.parametrize("contrast_pair_col", ["contrast_pair_id", None])
@pytest.mark.parametrize("require_both_labels", [True, False])
def test_non_contrast_splits_reject_missing_group_values(
    tmp_path: Path, contrast_pair_col: str | None, require_both_labels: bool
) -> None:
    config = _both_label_groups_config(
        tmp_path,
        rows=["missing\t1\t\t\tno"],
        require_both_labels=require_both_labels,
    )
    config.data.contrast_pair_col = contrast_pair_col

    with pytest.raises(SplitError, match="non-empty split group"):
        build_split_artifacts(config)


def test_null_contrast_pair_col_uses_split_group_without_contrast_column(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tfamily",
                "sp1\t1\tfamily_a",
                "sp2\t0\tfamily_a",
                "sp3\t1\tfamily_b",
                "sp4\t0\tfamily_b",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
            ]
        )
        + "\n",
    )
    cfg = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
  contrast_pair_col: null
split:
  group_col: family
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([cfg])

    manifest = build_split_artifacts(config).split_manifest

    assert set(manifest.get_column("group_id").drop_nulls().to_list()) == {
        "family_a",
        "family_b",
    }
    assert manifest.get_column("contrast_group_id").null_count() == manifest.height


def test_paired_labeled_species_and_unknowns_create_no_external_test(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t0\tg1",
                "sp3\t1\tg2",
                "sp4\t0\tg2",
                "sp5\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
            ]
        )
        + "\n",
    )
    cfg = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([cfg])

    artifacts = build_split_artifacts(config)

    assert artifacts.pool_counts["external_test"] == 0
    assert artifacts.pool_counts["discovery_inference"] == 1
    assert "external_test" not in set(artifacts.split_manifest.get_column("pool").to_list())


@pytest.mark.parametrize("contrast_pair_col", ["contrast_pair_id", "custom_pair"])
@pytest.mark.parametrize("require_both_labels", [True, False])
def test_unpaired_labeled_species_are_automatically_external_test(
    tmp_path: Path, contrast_pair_col: str, require_both_labels: bool
) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    table = pl.read_csv(metadata, separator="\t").rename(
        {"contrast_pair_id": contrast_pair_col}
    )
    table = table.with_columns(
        pl.when(pl.col("species") == "sp5")
        .then(pl.lit("   "))
        .otherwise(pl.col(contrast_pair_col))
        .alias(contrast_pair_col)
    )
    table.write_csv(metadata, separator="\t")
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    config.data.contrast_pair_col = contrast_pair_col
    config.split.group_col = contrast_pair_col
    config.split.require_both_labels_per_group = require_both_labels

    artifacts = build_split_artifacts(config)

    assert artifacts.pool_counts == {
        "training_validation": 4,
        "external_test": 1,
        "discovery_inference": 1,
        "excluded": 0,
    }
    manifest = artifacts.split_manifest
    assert manifest.filter(pl.col("species") == "sp5").get_column("pool").to_list() == [
        "external_test"
    ]
    assert manifest.filter(pl.col("species") == "sp6").get_column("pool").to_list() == [
        "discovery_inference"
    ]


def test_exclude_col_removes_species_from_all_pools_and_expression_requirements(
    tmp_path: Path,
) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\ttaxon_exclude",
                "sp1\t1\tg1\tno",
                "sp2\t0\tg1\tno",
                "sp3\t1\tg2\tno",
                "sp4\t0\tg2\tno",
                "excluded_sp\t1\t\tyes",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
            ]
        )
        + "\n",
    )
    cfg = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
split:
  exclude_col: taxon_exclude
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([cfg])

    artifacts = build_split_artifacts(config)

    assert artifacts.pool_counts["excluded"] == 1
    assert "excluded_sp" not in set(artifacts.split_manifest.get_column("species").to_list())


def test_invalid_exclude_col_values_are_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id\ttaxon_exclude",
                "sp1\t1\tg1\tmaybe",
                "sp2\t0\tg1\tno",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    cfg = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
split:
  exclude_col: taxon_exclude
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([cfg])

    with pytest.raises(SplitError, match="Exclude column"):
        build_split_artifacts(config)


def test_fold_validation_groups_map_logo_folds_to_held_out_groups(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    fold_validation_groups = build_split_artifacts(config).fold_validation_groups

    assert fold_validation_groups.columns == [
        "fold_id",
        "group_id",
        "n_validation_species",
        "n_validation_pos",
        "n_validation_neg",
        "validation_label_profile",
    ]
    assert fold_validation_groups.to_dicts() == [
        {
            "fold_id": "1",
            "group_id": "g1",
            "n_validation_species": 2,
            "n_validation_pos": 1,
            "n_validation_neg": 1,
            "validation_label_profile": "both",
        },
        {
            "fold_id": "2",
            "group_id": "g2",
            "n_validation_species": 2,
            "n_validation_pos": 1,
            "n_validation_neg": 1,
            "validation_label_profile": "both",
        },
    ]


def test_split_manifest_fold_train_validation_are_disjoint_by_group_and_species(
    tmp_path: Path,
) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    manifest = build_split_artifacts(config).split_manifest
    fold_ids = (
        manifest.filter(pl.col("pool") == "validation")
        .select("fold_id")
        .unique()
        .sort("fold_id")
        .to_series()
        .to_list()
    )
    assert fold_ids

    for fold_id in fold_ids:
        train_rows = manifest.filter((pl.col("pool") == "train") & (pl.col("fold_id") == fold_id))
        valid_rows = manifest.filter(
            (pl.col("pool") == "validation") & (pl.col("fold_id") == fold_id)
        )
        train_groups = set(train_rows.select("group_id").to_series().to_list())
        valid_groups = set(valid_rows.select("group_id").to_series().to_list())
        train_species = set(train_rows.select("species").to_series().to_list())
        valid_species = set(valid_rows.select("species").to_series().to_list())

        assert train_groups.isdisjoint(valid_groups)
        assert train_species.isdisjoint(valid_species)


def test_invalid_trait_values_are_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\tmaybe\tg1",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError):
        build_split_artifacts(config)


def test_missing_species_in_expression_are_rejected(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    # Remove sp6 from expression to trigger coverage error.
    tpm.write_text(
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError):
        build_split_artifacts(config)


def test_single_class_validation_groups_are_allowed_and_diagnosed(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t1\tg1",
                "sp3\t0\tg2",
                "sp4\t0\tg2",
                "sp5\t0\tg3",
                "sp6\t1\tg3",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
                "sp6\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    config_path = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
sampling:
  strategy: all_samples
  max_samples_per_label_per_group: null
  sampled_set_count: 1
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([config_path])

    artifacts = build_split_artifacts(config)

    assert artifacts.fold_count == 3
    diagnostics = artifacts.fold_diagnostics.sort("fold_id")
    assert diagnostics.select("validation_label_profile").to_series().to_list() == [
        "positive_only",
        "negative_only",
        "both",
    ]
    assert diagnostics.select("two_class_validation_metrics_defined").to_series().to_list() == [
        False,
        False,
        True,
    ]


def test_single_class_training_fold_is_still_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t1\tg1",
                "sp3\t0\tg2",
                "sp4\t0\tg2",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "species\torthogroup\ttpm\n"
        + "\n".join(f"sp{index}\tOG1\t{float(index)}" for index in range(1, 5))
        + "\n",
    )
    config_path = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
sampling:
  strategy: all_samples
  max_samples_per_label_per_group: null
  sampled_set_count: 1
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([config_path])

    with pytest.raises(SplitError, match="training split contains fewer than two labels"):
        build_split_artifacts(config)


def test_stratified_group_kfold_keeps_groups_disjoint_and_balances_labels(
    tmp_path: Path,
) -> None:
    species_rows = [
        f"sp{index}\t{1 if index <= 5 else 0}\tg{index}"
        for index in range(1, 11)
    ]
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "species\tC4\tcontrast_pair_id\n"
        + "\n".join(species_rows)
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "species\torthogroup\ttpm\n"
        + "\n".join(f"sp{index}\tOG1\t{float(index)}" for index in range(1, 11))
        + "\n",
    )
    config_path = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
split:
  outer_cv_strategy: stratified_group_kfold
  outer_cv_n_splits: 5
sampling:
  strategy: all_samples
  max_samples_per_label_per_group: null
  sampled_set_count: 1
""".strip()
        + "\n",
    )
    config = load_and_resolve_config([config_path])

    artifacts = build_split_artifacts(config)
    repeated_artifacts = build_split_artifacts(config)

    assert artifacts.fold_count == 5
    assert artifacts.split_manifest.equals(repeated_artifacts.split_manifest)
    assert artifacts.fold_diagnostics.select(
        pl.col("two_class_validation_metrics_defined").all()
    ).item()
    for fold_id in artifacts.fold_diagnostics.select("fold_id").to_series().to_list():
        train_groups = set(
            artifacts.split_manifest.filter(
                (pl.col("fold_id") == fold_id) & (pl.col("pool") == "train")
            )
            .select("group_id")
            .to_series()
            .to_list()
        )
        valid_groups = set(
            artifacts.split_manifest.filter(
                (pl.col("fold_id") == fold_id) & (pl.col("pool") == "validation")
            )
            .select("group_id")
            .to_series()
            .to_list()
        )
        assert train_groups.isdisjoint(valid_groups)


def test_invalid_trait_values_error_lists_offending_values(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\tmaybe\tg1",
                "sp2\t2\tg1",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="offending values: 2, maybe"):
        build_split_artifacts(config)


def test_duplicate_species_in_metadata_is_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp1\t0\tg1",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="must be unique"):
        build_split_artifacts(config)


def test_empty_species_identifier_in_metadata_is_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "\t1\tg1",
                "sp2\t0\tg1",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="empty species identifiers"):
        build_split_artifacts(config)


def test_expression_rows_excluded_counts_rows_not_in_metadata(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    tpm.write_text(
        tpm.read_text(encoding="utf-8")
        + "\n".join(
            [
                "sp_extra\tOG1\t1.0",
                "sp_extra\tOG2\t2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    artifacts = build_split_artifacts(config)

    assert artifacts.expression_rows_excluded == 2


def test_split_manifest_is_sorted_by_contract_keys(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "z_sp\t1\tg1",
                "a_sp\t0\tg1",
                "y_sp\t1\tg2",
                "b_sp\t0\tg2",
                "ext\t1\t",
                "inf\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "z_sp\tOG1\t1.0",
                "a_sp\tOG1\t2.0",
                "y_sp\tOG1\t3.0",
                "b_sp\tOG1\t4.0",
                "ext\tOG1\t5.0",
                "inf\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    artifacts = build_split_artifacts(config)
    manifest = artifacts.split_manifest
    sorted_manifest = manifest.sort(["pool", "fold_id", "group_id", "species"], nulls_last=False)

    assert manifest.to_dicts() == sorted_manifest.to_dicts()


def test_missing_required_metadata_columns_are_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4",
                "sp1\t1",
                "sp2\t0",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="Missing required columns in metadata"):
        build_split_artifacts(config)


def test_missing_required_expression_species_column_is_rejected(tmp_path: Path) -> None:
    metadata, _tpm = _fixture_data(tmp_path)
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "not_species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
                "sp6\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="Missing required column in expression data"):
        build_split_artifacts(config)


def test_missing_metadata_file_is_rejected(tmp_path: Path) -> None:
    metadata = tmp_path / "missing_metadata.tsv"
    _metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="Input file not found"):
        build_split_artifacts(config)


def test_expression_scan_file_not_found_is_wrapped_as_split_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    def _raise_file_not_found(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(split_mod.pl, "scan_csv", _raise_file_not_found)

    with pytest.raises(SplitError, match="Input file not found"):
        build_split_artifacts(config)


def test_no_training_validation_pool_is_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\t",
                "sp2\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="No species available in training and validation pool"):
        build_split_artifacts(config)


def test_group_kfold_without_n_splits_is_rejected_when_mutated(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    config_missing_n_splits = config.model_copy(
        update={
            "split": config.split.model_copy(
                update={"outer_cv_strategy": "group_kfold", "outer_cv_n_splits": None}
            )
        }
    )

    with pytest.raises(SplitError, match="outer_cv_n_splits must be set"):
        build_split_artifacts(config_missing_n_splits)


def test_outer_cv_splitter_value_error_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingLogo:
        def split(self, *_args: object, **_kwargs: object) -> object:
            def _iter() -> object:
                raise ValueError("boom")
                yield  # pragma: no cover

            return _iter()

    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    monkeypatch.setattr(split_mod, "LeaveOneGroupOut", lambda: _FailingLogo())

    with pytest.raises(SplitError, match="boom"):
        build_split_artifacts(config)


def test_outer_cv_splitter_rejects_zero_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _EmptyLogo:
        def split(self, *_args: object, **_kwargs: object) -> object:
            return iter([])

    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    monkeypatch.setattr(split_mod, "LeaveOneGroupOut", lambda: _EmptyLogo())

    with pytest.raises(SplitError, match="Outer CV produced zero folds"):
        build_split_artifacts(config)


def test_split_manifest_builder_rejects_empty_rows() -> None:
    empty_training = split_mod.pl.DataFrame(
        schema={
            "__species": split_mod.pl.String,
            "__group": split_mod.pl.String,
            "__contrast_group": split_mod.pl.String,
            "__label": split_mod.pl.Int8,
        }
    )
    empty_external = split_mod.pl.DataFrame(
        schema={
            "__species": split_mod.pl.String,
            "__contrast_group": split_mod.pl.String,
            "__label": split_mod.pl.Int8,
        }
    )
    empty_inference = split_mod.pl.DataFrame(
        schema={
            "__species": split_mod.pl.String,
            "__contrast_group": split_mod.pl.String,
            "__label": split_mod.pl.Int8,
        }
    )

    with pytest.raises(SplitError, match="Split manifest is empty"):
        split_mod._build_split_manifest(
            training_df=empty_training,
            external_df=empty_external,
            inference_df=empty_inference,
            folds=[],
        )


def test_expression_rows_excluded_rejects_non_integer_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeSchema:
        def names(self) -> list[str]:
            return ["species"]

    class _FakeSpeciesSummary:
        def item(self, _row: int, column: str) -> pl.Series | str:
            if column == "__expression_species":
                return pl.Series(["sp1"])
            return "not-an-int"

    class _FakeScan:
        def __init__(self) -> None:
            self.collect_calls = 0

        def collect_schema(self) -> _FakeSchema:
            return _FakeSchema()

        def select(self, *_args: object, **_kwargs: object) -> _FakeScan:
            return self

        def collect(self) -> _FakeSpeciesSummary:
            self.collect_calls += 1
            return _FakeSpeciesSummary()

    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    fake_scan = _FakeScan()
    monkeypatch.setattr(split_mod.pl, "scan_csv", lambda *_args, **_kwargs: fake_scan)

    with pytest.raises(
        SplitError, match="Failed to compute expression rows excluded from metadata"
    ):
        build_split_artifacts(config)
    assert fake_scan.collect_calls == 1
