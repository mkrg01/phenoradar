"""Legacy names resolve to identical contrast groups, splits, and settings."""

import pickle
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from phenoradar.config import (
    AppConfig,
    ConfigError,
    load_and_resolve_config,
    load_config_conditions,
)
from phenoradar.config.predict import PredictConfig
from phenoradar.contrast_metadata import canonical_metadata
from phenoradar.split import SplitError, build_split_artifacts


def test_legacy_settings_resolve_to_canonical_configuration() -> None:
    legacy = AppConfig.model_validate(
        {
            "data": {"contrast_pair_col": "expanded_pair_clade_id"},
            "split": {"group_col": "expanded_pair_clade_id"},
            "preprocess": {
                "expression": {"ranked_feature_filter": {"min_contrast_pairs": 3}},
                "amino_acid": {"ranked_site_filter": {"min_contrast_pairs": 4}},
            },
        }
    )
    canonical = AppConfig.model_validate(
        {
            "data": {"contrast_group_col": "phylogenetic_block_id"},
            "split": {"group_col": "phylogenetic_block_id"},
            "preprocess": {
                "expression": {"ranked_feature_filter": {"min_contrast_groups": 3}},
                "amino_acid": {"ranked_site_filter": {"min_contrast_groups": 4}},
            },
        }
    )
    assert legacy.model_dump() == canonical.model_dump()
    assert "contrast_pair_col" not in legacy.model_dump()["data"]
    assert "min_contrast_pairs" not in legacy.model_dump_json()
    assert (
        PredictConfig.model_validate(
            {"data": {"contrast_pair_col": "expanded_pair_clade_id"}}
        ).data.contrast_group_col
        == "phylogenetic_block_id"
    )


def test_legacy_configuration_objects_in_model_bundles_remain_readable() -> None:
    config = AppConfig.model_validate(
        {
            "data": {"contrast_group_col": "phylogenetic_block_id"},
            "split": {"group_col": "phylogenetic_block_id"},
            "preprocess": {"expression": {"ranked_feature_filter": {"min_contrast_groups": 3}}},
        }
    )
    expected = config.model_dump()
    config.data.__dict__["contrast_pair_col"] = "expanded_pair_clade_id"
    del config.data.__dict__["contrast_group_col"]
    config.data.__pydantic_fields_set__ = {"contrast_pair_col"}
    config.split.__dict__["group_col"] = "expanded_pair_clade_id"
    ranked = config.preprocess.expression.ranked_feature_filter
    ranked.__dict__["min_contrast_pairs"] = ranked.__dict__.pop("min_contrast_groups")
    ranked.__pydantic_fields_set__ = {"min_contrast_pairs"}
    restored = pickle.loads(pickle.dumps(config))
    assert restored.model_dump() == expected
    assert restored.data.model_fields_set == {"contrast_group_col"}


@pytest.mark.parametrize("section", ["data", "expression", "amino_acid"])
def test_conflicting_configuration_aliases_fail(section: str) -> None:
    if section == "data":
        payload = {
            "data": {
                "contrast_pair_col": "contrast_pair_id",
                "contrast_group_col": "phylogenetic_block_id",
            }
        }
    else:
        key = "ranked_feature_filter" if section == "expression" else "ranked_site_filter"
        payload = {
            "preprocess": {section: {key: {"min_contrast_pairs": 2, "min_contrast_groups": 3}}}
        }
    with pytest.raises(ValidationError, match="conflicting configuration aliases"):
        AppConfig.model_validate(payload)


def test_equal_aliases_are_accepted_and_layers_override_by_canonical_name(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {
            "data": {
                "contrast_pair_col": "expanded_pair_clade_id",
                "contrast_group_col": "phylogenetic_block_id",
            }
        }
    )
    assert config.data.contrast_group_col == "phylogenetic_block_id"
    base, override = tmp_path / "old.yml", tmp_path / "new.yml"
    base.write_text("data:\n  contrast_pair_col: expanded_pair_clade_id\n")
    override.write_text("data:\n  contrast_group_col: contrast_pair_id\n")
    assert load_and_resolve_config([base, override]).data.contrast_group_col == "contrast_pair_id"
    assert (
        load_and_resolve_config([override, base]).data.contrast_group_col == "phylogenetic_block_id"
    )
    base.write_text("data:\n  contrast_pair_col: old\n  contrast_group_col: new\n")
    with pytest.raises(ConfigError, match="conflicting configuration aliases"):
        load_and_resolve_config([base, override])


@pytest.mark.parametrize("suffix", ["", "_all", "_phenotyped", "_representatives"])
def test_metadata_aliases_preserve_ids_and_reject_conflicts(suffix: str) -> None:
    old, new = "expanded_pair_clade_id" + suffix, "phylogenetic_block_id" + suffix
    frame = pl.DataFrame({old: ["1", "2", None], new: ["1", "2", ""]})
    assert canonical_metadata(frame).columns == [new]
    assert canonical_metadata(frame.drop(new))[new].to_list() == ["1", "2", None]
    with pytest.raises(ValueError, match="conflicting metadata aliases"):
        canonical_metadata(frame.with_columns(pl.lit("other").alias(new)))


def test_legacy_condition_lists_use_canonical_dimension_names(tmp_path: Path) -> None:
    path = tmp_path / "conditions.yml"
    path.write_text(
        "preprocess:\n  expression:\n    ranked_feature_filter:\n      min_contrast_pairs: [1, 2]\n"
    )
    conditions = load_config_conditions([path]).conditions
    assert [
        item.config.preprocess.expression.ranked_feature_filter.min_contrast_groups
        for item in conditions
    ] == [1, 2]
    assert all(
        item.values[0][0] == "preprocess.expression.ranked_feature_filter.min_contrast_groups"
        for item in conditions
    )


def test_condition_ids_agree_for_legacy_and_canonical_group_columns(tmp_path: Path) -> None:
    old, new = tmp_path / "old.yml", tmp_path / "new.yml"
    old.write_text(
        "data:\n  contrast_pair_col: expanded_pair_clade_id\n"
        "split:\n  group_col: expanded_pair_clade_id\n"
        "preprocess:\n  expression:\n    ranked_feature_filter:\n      min_contrast_pairs: [1, 2]\n"
    )
    new.write_text(
        "data:\n  contrast_group_col: phylogenetic_block_id\n"
        "split:\n  group_col: phylogenetic_block_id\n"
        "preprocess:\n  expression:\n    ranked_feature_filter:\n"
        "      min_contrast_groups: [1, 2]\n"
    )
    legacy = load_config_conditions([old]).conditions
    canonical = load_config_conditions([new]).conditions
    assert [item.condition_id for item in legacy] == [item.condition_id for item in canonical]
    assert [item.config.model_dump() for item in legacy] == [
        item.config.model_dump() for item in canonical
    ]


def test_old_and_new_metadata_produce_identical_folds_and_pools(tmp_path: Path) -> None:
    species = ["A", "B", "C", "D", "external", "unknown"]
    metadata = pl.DataFrame(
        {
            "species": species,
            "C4": [0, 1, 0, 1, 1, None],
            "expanded_pair_clade_id": ["1", "1", "2", "2", None, None],
        }
    )
    old, new, expression = (tmp_path / name for name in ["old.tsv", "new.tsv", "tpm.tsv"])
    metadata.write_csv(old, separator="\t")
    canonical_metadata(metadata).write_csv(new, separator="\t")
    pl.DataFrame(
        {"species": species, "orthogroup": ["OG1"] * len(species), "tpm": [1.0] * len(species)}
    ).write_csv(expression, separator="\t")
    results = []
    for path, column in [(old, "expanded_pair_clade_id"), (new, "phylogenetic_block_id")]:
        config = AppConfig.model_validate(
            {
                "data": {"metadata_path": str(path), "tpm_path": str(expression)},
                "split": {"group_col": column},
            }
        )
        results.append(build_split_artifacts(config))
    assert results[0].split_manifest.equals(results[1].split_manifest)
    assert results[0].fold_validation_groups.equals(results[1].fold_validation_groups)
    assert results[0].pool_counts == results[1].pool_counts
    assert results[0].pool_counts["external_test"] == 1
    assert results[0].pool_counts["discovery_inference"] == 1
    metadata.with_columns(pl.lit("wrong").alias("phylogenetic_block_id")).write_csv(
        old, separator="\t"
    )
    with pytest.raises(SplitError, match="conflicting metadata aliases"):
        build_split_artifacts(
            AppConfig.model_validate(
                {
                    "data": {"metadata_path": str(old), "tpm_path": str(expression)},
                    "split": {"group_col": "phylogenetic_block_id"},
                }
            )
        )
