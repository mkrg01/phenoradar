import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from test_amino_acid import alignment, fixture_config, settings, tpm_table

from phenoradar import glmnet
from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_packed import PackedColumns, native, packed_arithmetic
from phenoradar.amino_acid_prediction import DeferredAAValidation
from phenoradar.amino_acid_run import run_amino_acid
from phenoradar.amino_acid_sparse import fit_sparse_encoded
from phenoradar.config import load_and_resolve_config


def test_packed_scan_reuses_validated_immutable_indices():
    if not hasattr(native, "prepare_packed_scan"):
        pytest.skip("Validated scan kernel is unavailable")
    matrix = sparse.csc_matrix(np.column_stack([np.ones(65), np.arange(65.0)]))
    block = PackedColumns.from_csc(matrix)
    assert len(block.values) == 1 and block.fallback.nnz
    residual = np.linspace(-0.3, 0.7, 65)
    out = np.empty(2)
    checked = np.empty_like(out)
    for fused in (False, True):
        native.scan_packed_columns(fused, *block._buffers(), residual, checked)
        block.gradient(residual, out, fused=fused)
        assert out.tobytes() == checked.tobytes()
    state = block._scan_plan
    for array in state[1]:
        with pytest.raises(ValueError):
            array.setflags(write=True)

    restored = pickle.loads(pickle.dumps(block))
    assert restored._scan_plan is None
    restored.gradient(residual, out, fused=True)
    assert out.tobytes() == checked.tobytes()

    block.fallback.indices = block.fallback.indices.copy()
    block.fallback.indices[0] = len(residual)
    with pytest.raises(ValueError, match="shape/index"):
        block.gradient(residual, out, fused=True)
    with pytest.raises(ValueError, match="shape/index"):
        native.scan_validated_packed_columns(
            state[0], True, block.values, block.fallback.data, residual[:-1], out,
        )


@pytest.mark.parametrize("aggregation", ["aa_frequency", "tpm_weighted_frequency", "best_covered"])
def test_deferred_validation_preserves_shifted_columns(tmp_path, aggregation):
    if packed_arithmetic() is None:
        pytest.skip("Compatible native packing kernels are unavailable")
    genes = [
        ("s0", "s0_a", "AAA", 1), ("s0", "s0_b", "VAA", 4),
        ("s1", "s1_a", "AVA", 1), ("s2", "s2_a", "VAV", 1),
        ("s3", "s3_a", "AAV", 1), ("s3", "s3_b", "R-V", 2),
        ("s4", "s4_a", "V-R", 1),
    ]
    path = alignment(tmp_path / "aa", [(gene, seq) for _, gene, seq, _ in genes])
    tpm = tpm_table(tmp_path / "tpm.tsv",
                    [(species, gene, weight) for species, gene, _, weight in genes])
    train = ["s0", "s1", "s2", "s0"]
    valid = ["s3", "missing", "s4", "s3"]
    prep = settings(aggregation)
    with AlignmentStore(
        path, list(dict.fromkeys(train + valid)), cache_frequencies=True,
        sequence_tpm_path=str(tpm) if aggregation == "tpm_weighted_frequency" else None,
    ) as store:
        kwargs = dict(y=np.array([0, 1, 1, 0]), groups=np.array(["a", "b", "c", "a"]),
                      audit=False, packed=True)
        _, (expected_train, expected_valid), expected_shift = fit_sparse_encoded(
            store, train, prep, [train, valid], **kwargs,
        )
        fitted, (train_only,), shift = fit_sparse_encoded(store, train, prep, [train], **kwargs)
        deferred = DeferredAAValidation(fitted, store, valid, shift, lambda: pytest.fail(
            "A narrow validation projection must not generate all validation columns",
        ))
        np.testing.assert_array_equal(expected_shift, shift)
        np.testing.assert_array_equal(
            expected_train.to_csc().toarray(), train_only.to_csc().toarray(),
        )
        columns = np.arange(0, len(shift), 2)
        np.testing.assert_array_equal(expected_valid[:, columns].toarray(),
                                      deferred._columns(columns).toarray())
        models = glmnet.fit_glmnet_path(
            train_only, kwargs["y"], [1, 0.1, 0.01], compact_coefficients=True, compute_kkt=False,
        )
        assert any(len(model.coefficient_indices_) for model in models)
        assert any(not len(model.coefficient_indices_) for model in models)
        np.testing.assert_array_equal(glmnet.predict_glmnet_path(models, expected_valid),
                                      glmnet.predict_glmnet_path(models, deferred))


@pytest.mark.parametrize("oversized", [False, True])
def test_packed_amino_acid_run_preserves_cv_refit_and_reports(tmp_path, monkeypatch, oversized):
    if packed_arithmetic() is None:
        pytest.skip("Compatible native packing kernels are unavailable")
    config = load_and_resolve_config([fixture_config(tmp_path, "tpm_weighted_frequency")])
    config.preprocess.amino_acid.ranked_site_filter.method = "none"
    if oversized:
        # Force the full-matrix limit while leaving each local solve native.
        fit = glmnet._fit_oversized_path

        def fit_with_local_limit(*args, **kwargs):
            with monkeypatch.context() as local:
                local.setattr(glmnet, "SPARSE_INDEX_LIMIT", np.iinfo(np.int32).max)
                return fit(*args, **kwargs)

        monkeypatch.setattr(glmnet, "_fit_oversized_path", fit_with_local_limit)
        monkeypatch.setattr(glmnet, "SPARSE_INDEX_LIMIT", 9)
    csc = run_amino_acid(config, tmp_path / "csc")
    config.runtime.amino_acid_matrix = "packed"
    packed = run_amino_acid(config, tmp_path / "packed")
    for path in [
        "cv/tables/model_selection_trials.tsv",
        "cv/tables/model_selection_selected.tsv",
        "cv/tables/prediction_cv.tsv",
        "cv/tables/amino_acid_site_effects.tsv",
        "external_test/tables/pred_external_test.tsv",
        "inference/tables/pred_inference.tsv",
    ]:
        assert (Path(packed) / path).read_bytes() == (Path(csc) / path).read_bytes(), path
