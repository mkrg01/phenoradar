from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import sparse
from test_combined import combined_config

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_sparse import fit_sparse_encoded, training_information
from phenoradar.combined import ExpressionInput
from phenoradar.config import load_and_resolve_config
from phenoradar.cv import (
    CVError,
    RunExpressionCache,
    _fit_sample_weights,
    _inner_cv_splits,
    _preprocess_fold_with_counts,
    expression_input_species,
    iter_inner_cv_preprocessed_folds,
)
from phenoradar.split import build_split_artifacts


def test_shared_input_pool_skips_unused_targets_and_releases_replaced_cache(tmp_path: Path) -> None:
    config = load_and_resolve_config([combined_config(tmp_path)])
    data = pl.read_csv(config.data.tpm_path, separator="\t").with_columns(
        pl.when(pl.col("species") == "external").then(-1.).otherwise(pl.col("tpm")).alias("tpm")
    )
    data.write_csv(config.data.tpm_path, separator="\t")
    config.runtime.execution_stage = "cv_only"
    manifest = build_split_artifacts(config).split_manifest
    training = [f"s{i}" for i in range(8)]
    all_species = [*training, "external", "unknown"]
    assert expression_input_species(config, manifest) == training
    with ExpressionInput(config, all_species, training) as joint:
        assert joint.builder._cached_species == set(training)
    expression = config.model_copy(deep=True)
    expression.data.feature_mode = "expression"
    with RunExpressionCache() as cache:
        cache.prepare(expression, manifest)
        builder = cache.get_builder(expression)
        assert builder._cached_species == set(training)
        previous = builder._cached_long_path
        cache.prepare_species(expression, ["s0"])
        assert builder._cached_long_path == previous  # A subset reuses the validated cache.
        with pytest.raises(CVError, match="negative"):
            cache.prepare_species(expression, all_species)
        assert previous is not None and previous.exists()  # A failed replacement is transactional.
        data.filter(pl.col("species") != "external").write_csv(config.data.tpm_path, separator="\t")
        cache.prepare_species(expression, [*training, "unknown"])
        assert not previous.exists()


@pytest.mark.parametrize("ranked", ["none", "pair_aware", "unpaired", "variance"])
def test_streamed_expression_folds_preserve_repeated_rows_and_all_preprocessing(
    tmp_path: Path, ranked: str,
) -> None:
    config = load_and_resolve_config([combined_config(tmp_path)])
    config.preprocess.expression.ranked_feature_filter.method = ranked
    config.preprocess.expression.ranked_feature_filter.max_features = 2
    config.preprocess.expression.sparse_feature_filter.enabled = True
    config.preprocess.expression.sparse_feature_filter.min_nonzero_fraction = .5
    config.model_selection.inner_cv_strategy = "logo"
    species = [f"s{i}" for i in range(8)]
    sampled = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7])
    labels = (np.arange(8) % 2)[sampled]
    groups = np.repeat(["g0", "g1", "g2", "g3"], 2)[sampled]
    with ExpressionInput(config, species, species) as source:
        raw = source.matrix([species[i] for i in sampled])
        features = source.features
        options = dict(x_source_raw=raw,
                       y_source=labels, groups_source=groups, contrast_groups_source=groups,
                       feature_names=source.features, warnings=[])
        streamed = list(iter_inner_cv_preprocessed_folds(config=config, **options))
        config.data.feature_mode = "expression"
    for actual, (train, valid, fold) in zip(
        streamed, _inner_cv_splits(config, labels, groups), strict=True,
    ):
        # Independent per-fold reference: no shared transformed matrix or count cache.
        expected_train, expected_valid, *_ = _preprocess_fold_with_counts(
            config, raw[train], raw[valid], features,
            y_train=labels[train], groups_train=groups[train], warnings=[],
        )
        assert actual.inner_fold_id == fold
        np.testing.assert_array_equal(actual.x_train, expected_train)
        np.testing.assert_array_equal(actual.x_valid, expected_valid)
        np.testing.assert_array_equal(actual.sample_weight,
                                      _fit_sample_weights(config, labels[train], groups[train]))


@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
@pytest.mark.parametrize("ranked", ["none", "pair_aware"])
@pytest.mark.parametrize("buffer_bytes", [0, 1024**2])
def test_joint_csc_assembly_matches_separate_blocks_and_observed_training_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregation: str, ranked: str,
    buffer_bytes: int,
) -> None:
    import phenoradar.amino_acid_sparse as encoding

    monkeypatch.setattr(encoding, "matrix_buffer_limit", lambda store: buffer_bytes)
    config = load_and_resolve_config([combined_config(tmp_path, aggregation)])
    config.preprocess.amino_acid.ranked_site_filter.method = ranked
    species = [f"s{i}" for i in range(8)]
    training = [species[0], *species]
    targets = [training, ["s0", "external", "unknown"]]
    rng = np.random.default_rng(18)
    prefix = [rng.normal(size=(len(s), 3)) for s in targets]
    prefix[0] = prefix[0].astype(np.float32)
    prefix[0][:, 0] = 0
    settings = dict(y=np.array([0, *np.arange(8) % 2]),
                    groups=np.array(["g0", *np.repeat(["g0", "g1", "g2", "g3"], 2)]))
    settings["contrast_groups"] = settings["groups"]
    with AlignmentStore(Path(config.data.alignment_dir), [*species, "external", "unknown"],
                        sequence_tpm_path=config.data.sequence_tpm_path
                        if aggregation == "tpm_weighted_frequency" else None) as store:
        fitted, separate, shift = fit_sparse_encoded(
            store, training, config.preprocess.amino_acid, targets, **settings)
        joint, matrices, joint_shift = fit_sparse_encoded(
            store, training, config.preprocess.amino_acid, targets, prefix=prefix, **settings)
        for actual, left, right in zip(matrices, prefix, separate, strict=True):
            expected = sparse.hstack((left, right), format="csc")
            for field in ["data", "indices", "indptr"]:
                np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
        np.testing.assert_array_equal(joint_shift, np.r_[np.zeros(3), shift])
        actual, _ = training_information(joint, store, training, matrices[0], joint_shift,
                                          standard_deviation=True, offset=3)
        expected = []
        offset = 0
        for indices, raw, coverage in fitted.raw_blocks(store, training, []):
            sites = [fitted.sites[i] for i in indices]
            mass = np.sum(np.nan_to_num(raw) * np.stack([s.known for s in sites]),
                          axis=2, dtype=np.float64)
            for j, site in enumerate(sites):
                width = int(site.active.sum())
                centered = (separate[0][:, offset:offset + width].toarray()
                            - shift[offset:offset + width])
                expected.extend(np.std(centered[(coverage[:, j] * mass[:, j]) > 0], axis=0))
                offset += width
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
