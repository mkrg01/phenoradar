from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from test_amino_acid import alignment, settings, tpm_table

from phenoradar.amino_acid import AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_sparse import (
    SparseFoldEncoding,
    fit_sparse_encoded,
    modal_shift,
    training_information,
)
from phenoradar.glmnet import fit_glmnet_path


@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
@pytest.mark.parametrize("method", ["none", "unpaired", "pair_aware"])
@pytest.mark.parametrize("jobs", [1, 3])
@pytest.mark.parametrize("cache_frequencies", [False, True])
def test_sparse_encoding_preserves_filters_missing_novel_and_duplicate_rows(
    tmp_path: Path, aggregation: str, method: str, jobs: int, cache_frequencies: bool,
) -> None:
    records = [("a_1", "AC-DVAX"), ("a_2", "ADDDVAX"), ("b_1", "ADCEVAX"),
               ("c_1", "ADCDAAX"), ("d_1", "ACCEAAX"), ("held_1", "W-XWWVX")]
    path = alignment(tmp_path / "aa", records)
    (path / "OG2.faa").write_text((path / "OG1.faa").read_text())
    tpm = tpm_table(tmp_path / "tpm.tsv", [(s.split('_')[0], s, i+1)
                                          for i, (s, _) in enumerate(records)])
    cfg = settings(aggregation)
    cfg.site_filter.min_sample_fraction = .5
    cfg.ranked_site_filter.method = method
    cfg.ranked_site_filter.min_contrast_pairs = 1
    training = ["d", "b", "a", "c", "a"]
    targets = [training, ["held", "absent", "c"], []]
    kwargs = dict(y=np.array([0, 1, 0, 1, 0]), groups=np.array(["1", "1", "2", "2", "1"]),
                  contrast_groups=np.array(["1", "1", "2", "2", "1"]))
    with AlignmentStore(path, ["a", "b", "c", "d", "held", "absent"],
                        cache_frequencies=cache_frequencies, sequence_tpm_path=str(tpm)) as store:
        expected = fit_amino_acid(store, training, cfg, **kwargs)
        fitted, matrices, shift = fit_sparse_encoded(store, training, cfg, targets,
                                                    n_jobs=jobs, **kwargs)
        assert fitted.features == expected.features
        assert fitted.qc == expected.qc
        for samples, matrix in zip(targets, matrices, strict=True):
            assert sparse.isspmatrix_csc(matrix) and matrix.has_canonical_format
            original = expected.matrix(store, samples)
            np.testing.assert_array_equal(matrix.toarray(), original + shift)
        np.testing.assert_array_equal(shift, modal_shift(expected.matrix(store, training)))
        scales, representatives = training_information(fitted, store, training, matrices[0],
                                                       shift, standard_deviation=True)
        dense, observed, expected_representatives = fitted.transform(store, training)
        expected_scales = []
        offset = 0
        for j, site in enumerate(fitted.sites):
            width = int(site.active.sum())
            expected_scales.extend(np.std(dense[observed[:, j] > 0, offset:offset+width], axis=0))
            offset += width
        np.testing.assert_allclose(scales, expected_scales, atol=2e-16)
        assert representatives == expected_representatives


def test_modes_are_training_only_and_sparse_files_survive_encoder_close() -> None:
    with np.errstate(all="raise"):
        encoder = SparseFoldEncoding([["a", "b", "a"], ["held"], []], ["a", "b", "a"])
        values = np.array([[.2, .7], [.8, .3], [999., -999.]])
        encoder.append(values[:, :1])
        encoder.append(values[:, 1:])
        result = encoder.matrices()
        shift = encoder.shift
        encoder.close()
    np.testing.assert_array_equal(shift, [-.2, -.7])
    np.testing.assert_array_equal(result[0].toarray(), values[[0, 1, 0]]+shift)
    np.testing.assert_array_equal(result[1].toarray(), values[[2]]+shift)
    assert result[2].shape == (0, 2)


def test_affine_lasso_preserves_centered_saved_model_and_combined_penalties() -> None:
    rng = np.random.default_rng(204)
    x = rng.integers(0, 2, size=(160, 8)).astype(float)
    x -= x.mean(axis=0)
    shift = modal_shift(x)
    expression = rng.normal(size=(160, 3))
    design = sparse.hstack((expression, sparse.csc_matrix(x+shift)), format="csc")
    full_shift = np.r_[np.zeros(3), shift]
    centered = np.column_stack((expression, x))
    y = rng.binomial(1, 1/(1+np.exp(-centered[:, 0]-centered[:, 4])))
    weights = rng.uniform(.3, 2, size=len(y))
    for ratio in [.1, 1, 10]:
        factors = np.r_[np.ones(3), np.full(8, ratio)]
        models = fit_glmnet_path(design, y, [.1, .01], alpha=1, thresh=1e-12,
                                sample_weight=weights, penalty_factor=factors)
        expected = fit_glmnet_path(centered, y, [.1, .01], alpha=1, thresh=1e-12,
                                  sample_weight=weights, penalty_factor=factors)
        for model, reference in zip(models, expected, strict=True):
            native = model.predict_proba(design)
            model.intercept_ += float(full_shift @ model.coef_)
            np.testing.assert_allclose(model.predict_proba(centered), native, atol=1e-15)
            np.testing.assert_allclose(model.predict_proba(centered),
                                       reference.predict_proba(centered), atol=2e-7)


@pytest.mark.parametrize("canonical", [True, False])
def test_sparse_path_capacity_retries_and_keeps_readonly_input(
    monkeypatch, canonical: bool,
) -> None:
    import phenoradar.glmnet as glmnet

    rng = np.random.default_rng(1)
    x = sparse.csc_matrix(rng.normal(size=(120, 14)))
    y = rng.integers(0, 2, 120)
    weights = rng.uniform(.1, 2, 120)
    if not canonical:
        # Split entries into duplicates; canonicalization must copy the caller's arrays.
        x = sparse.csc_matrix((np.repeat(x.data/2, 2), np.repeat(x.indices, 2), x.indptr*2),
                              shape=x.shape)
    originals = [a.copy() for a in [x.data, x.indices, x.indptr]]
    for a in [x.data, x.indices, x.indptr]:
        a.flags.writeable = False
    expected = fit_glmnet_path(x, y, [.1, .01], sample_weight=weights)
    monkeypatch.setattr(glmnet, "_SPARSE_PATH_CAPACITY", 1)
    actual = fit_glmnet_path(x, y, [.1, .01], sample_weight=weights)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.coef_, b.coef_)
        assert a.intercept_ == b.intercept_ and a.n_iter_ == b.n_iter_
    for a, b in zip([x.data, x.indices, x.indptr], originals, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("buffer_bytes", [1, 1024**2])
def test_sparse_og_blocks_merge_without_dense_reconstruction_and_survive_close(
    buffer_bytes: int,
) -> None:
    training = ["b", "a", "b"]
    targets = [["b", "a", "b"], ["held", "a"], []]
    encoder = SparseFoldEncoding(targets, training)
    values = np.array([[0, 1, 0, .3], [1, 0, 0, .9], [999, -999, 0, .1]])
    try:
        for selected in [slice(0, 2), slice(2, 4)]:
            child = encoder.fork()
            child.memory_limit = buffer_bytes
            try:
                child.append(values[:, selected])
                block = child.block()
            finally:
                child.close()
            encoder.extend(block)
        actual, shift = encoder.matrices(), encoder.shift
    finally:
        encoder.close()
    expected_shift = modal_shift(values[[0, 1, 0]])
    np.testing.assert_array_equal(shift, expected_shift)
    for rows, matrix in zip([[0, 1, 0], [2, 1], []], actual, strict=True):
        expected = sparse.csc_matrix(values[rows] + shift)
        assert matrix.has_canonical_format
        for name in ["data", "indices", "indptr"]:
            np.testing.assert_array_equal(getattr(matrix, name), getattr(expected, name))


@pytest.mark.parametrize("spill", [False, True])
def test_parallel_sparse_encoding_does_not_build_dense_og_matrices(
    tmp_path: Path, monkeypatch, spill: bool,
):
    from phenoradar.amino_acid_encoding import FoldEncoding

    if spill:
        monkeypatch.setattr("phenoradar.amino_acid_og_parallel.OG_BUFFER_BYTES", 1)
    path = alignment(tmp_path / "aa", [("a_1", "AC-D"), ("b_1", "ADCE"), ("held_1", "W-XW")])
    (path / "OG2.faa").write_text((path / "OG1.faa").read_text())
    cfg = settings("aa_frequency")
    kwargs = dict(y=np.array([0, 1]), groups=np.array(["a", "b"]))
    with AlignmentStore(path, ["a", "b", "held"], cache_frequencies=True) as store:
        expected = fit_amino_acid(store, ["a", "b"], cfg, **kwargs)
        dense = expected.matrix(store, ["a", "b", "held"])

        def forbidden(*args):
            pytest.fail("Sparse OG preprocessing must not assemble a dense OG matrix")

        monkeypatch.setattr(FoldEncoding, "matrices", forbidden)
        fitted, matrices, shift = fit_sparse_encoded(
            store, ["a", "b"], cfg, [["a", "b", "held"]], n_jobs=3, **kwargs,
        )
        assert fitted.features == expected.features
        np.testing.assert_array_equal(matrices[0].toarray(), dense + shift)
