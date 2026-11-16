from pathlib import Path

import numpy as np
import pytest
from test_amino_acid import fixture_config

from phenoradar import glmnet
from phenoradar.amino_acid_packed import packed_arithmetic
from phenoradar.amino_acid_run import run_amino_acid
from phenoradar.config import load_and_resolve_config


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
