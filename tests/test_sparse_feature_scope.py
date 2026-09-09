from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from phenoradar.config import (
    AppConfig,
    ConfigError,
    has_condition_dimensions,
    load_and_resolve_config,
    load_config_conditions,
    serialize_resolved_config,
)
from phenoradar.cv import CVError, _preprocess_train_and_target, _select_feature_indices


def _config(**settings: object) -> AppConfig:
    return AppConfig.model_validate({"preprocess": {"sparse_feature_filter": settings}})


@pytest.mark.parametrize("legacy,scope", [(None, "any_trait"), (0, "trait_0"), (1, "trait_1")])
def test_legacy_config_resolves_and_serializes_only_new_scope(
    tmp_path: Path,
    legacy: int | None,
    scope: str,
) -> None:
    path = tmp_path / "config.yml"
    source = {"preprocess": {"sparse_feature_filter": {"within_trait": legacy}}}
    path.write_text(yaml.safe_dump(source))
    config = load_and_resolve_config([path])
    assert config == _config(scope=scope)
    assert config.preprocess.sparse_feature_filter.scope == scope
    resolved = yaml.safe_load(serialize_resolved_config(config))
    assert "within_trait" not in resolved["preprocess"]["sparse_feature_filter"]
    assert resolved["preprocess"]["sparse_feature_filter"]["scope"] == scope
    assert AppConfig.model_validate(source) == config
    assert source["preprocess"]["sparse_feature_filter"] == {"within_trait": legacy}


@pytest.mark.parametrize("invalid", [True, False, 2, "1", [], {}])
def test_legacy_scope_rejects_invalid_values(invalid: object) -> None:
    with pytest.raises(ValidationError, match="within_trait"):
        _config(within_trait=invalid)


@pytest.mark.parametrize("invalid", [None, True, 0, "all", "both_traits"])
def test_scope_rejects_ambiguous_or_unsupported_values(invalid: object) -> None:
    with pytest.raises(ValidationError, match="scope"):
        _config(scope=invalid)


@pytest.mark.parametrize("legacy,scope", [(None, "any_trait"), (1, "trait_1"), (0, "all_samples")])
def test_new_and_legacy_keys_cannot_be_combined(
    tmp_path: Path,
    legacy: int | None,
    scope: str,
) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "preprocess": {
                    "sparse_feature_filter": {
                        "within_trait": legacy,
                        "scope": scope,
                    }
                }
            }
        )
    )
    for loader in (load_and_resolve_config, load_config_conditions, has_condition_dimensions):
        with pytest.raises(ConfigError, match="both scope and within_trait"):
            loader([path])


def test_scope_comparison_and_legacy_comparison_have_canonical_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("preprocess:\n  sparse_feature_filter:\n    within_trait: [null, 0, 1]\n")
    assert has_condition_dimensions([path])
    legacy = load_config_conditions([path])
    path.write_text(
        "preprocess:\n  sparse_feature_filter:\n    scope: [any_trait, trait_0, trait_1]\n"
    )
    canonical = load_config_conditions([path])
    assert legacy == canonical
    assert canonical.dimensions[0].dotted_path == "preprocess.sparse_feature_filter.scope"
    path.write_text(
        "preprocess:\n  sparse_feature_filter:\n"
        "    scope: [all_samples, any_trait, trait_0, trait_1]\n"
    )
    assert [
        c.config.preprocess.sparse_feature_filter.scope
        for c in load_config_conditions([path]).conditions
    ] == [
        "all_samples",
        "any_trait",
        "trait_0",
        "trait_1",
    ]


@pytest.mark.parametrize(
    "scope,expected",
    [
        ("all_samples", [0]),
        ("any_trait", [0, 1, 2]),
        ("trait_0", [0]),
        ("trait_1", [1, 2]),
    ],
)
def test_scope_uses_species_denominators_with_imbalanced_classes_and_missing_values(
    scope: str,
    expected: list[int],
) -> None:
    # Eight trait-0 and two trait-1 species. Fractions in the third column are
    # 5/8 and 2/2: their unweighted mean exceeds .8, but the pooled 7/10 does not.
    values = np.full((10, 5), np.nan)
    values[:8, 0] = 1
    values[8, 0] = 0
    values[:8, 1] = 0
    values[8:, 1] = 1
    values[:, 2] = 0
    values[:5, 2] = values[8:, 2] = 1
    values[0, 3] = 1
    names = ["majority", "minority", "unweighted_mean_only", "mostly_missing", "all_missing"]
    config = _config(scope=scope, min_nonzero_fraction=0.8)
    labels = np.array([0] * 8 + [1] * 2)
    assert _select_feature_indices(config, values, names, y_train=labels).tolist() == expected
    if scope == "all_samples":
        assert _select_feature_indices(config, values, names).tolist() == expected
        assert (
            _select_feature_indices(config, values, names, y_train=1 - labels).tolist() == expected
        )
        # Zero and NA both remain in the denominator: 8/10 fails a .9 threshold.
        with pytest.raises(CVError, match="removed all features"):
            _select_feature_indices(_config(scope=scope, min_nonzero_fraction=0.9), values, names)


def test_all_samples_filters_before_neutralization_without_using_target_values() -> None:
    config = AppConfig.model_validate(
        {
            "preprocess": {
                "absent_feature_fill": "nan",
                "missing_expression": {"method": "neutral", "zero_as_missing": True},
                "sparse_feature_filter": {"scope": "all_samples", "min_nonzero_fraction": 0.75},
            }
        }
    )
    train = np.array([[1, 1], [2, 2], [3, 0], [0, np.nan]], dtype=float)
    target = np.array([[0, 100], [100, 0]], dtype=float)
    scaled, predicted, names, scaler = _preprocess_train_and_target(
        config, train, target, ["kept", "removed"]
    )
    assert names == ["kept"]
    assert scaled[-1, 0] == predicted[0, 0] == 0
    np.testing.assert_allclose(scaler.mean_, [np.log1p([1, 2, 3]).mean()])
    assert train[3, 0] == 0 and np.isnan(train[3, 1])


def test_all_samples_rejects_an_empty_training_population() -> None:
    with pytest.raises(CVError, match="at least one training row"):
        _select_feature_indices(_config(scope="all_samples"), np.empty((0, 1)), ["OG1"])
