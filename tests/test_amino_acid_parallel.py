from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from test_amino_acid import fixture_config

from phenoradar.amino_acid_run import run_amino_acid
from phenoradar.config import load_and_resolve_config


@pytest.mark.parametrize("aggregation, combined", [
    ("best_covered", False), ("aa_frequency", False), ("tpm_weighted_frequency", False),
    ("tpm_weighted_frequency", True),
])
def test_process_outer_folds_match_serial_and_render_common_figures(
    tmp_path: Path, monkeypatch, aggregation: str, combined: bool,
) -> None:
    from test_combined import combined_config

    import phenoradar.amino_acid_parallel as parallel

    path = (combined_config(tmp_path, aggregation) if combined
            else fixture_config(tmp_path, aggregation))
    config = load_and_resolve_config([path])
    alignments = Path(config.data.alignment_dir)
    (alignments / "OG2.faa").write_text((alignments / "OG1.faa").read_text())
    # Keep this integration test small; the complete output tables are compared.
    config.figures.top_features = 3
    first = run_amino_acid(config, tmp_path / "serial")
    config.runtime.n_jobs = 4
    monkeypatch.setattr(parallel, "fold_workers", lambda *args: 2)
    second = run_amino_acid(config, tmp_path / "parallel")
    for stage, names in {
        "cv": ["prediction_cv", "model_selection_trials", "model_selection_selected",
               "feature_importance", "coefficients", "coefficients_by_fold", "amino_acid_site_pss",
               "amino_acid_site_effect_range", "amino_acid_site_stability",
               "selected_site_sequence_map", "amino_acid_site_effects"],
        "external_test": ["pred_external_test"], "inference": ["pred_inference"],
    }.items():
        for name in names:
            frames = [pl.read_csv(root / stage / "tables" / f"{name}.tsv", separator="\t")
                      for root in [first, second]]
            columns = frames[0].columns
            assert_frame_equal(frames[0].sort(columns), frames[1].sort(columns), abs_tol=1e-14)
    for name in ["roc_curve_cv", "pr_curve_cv",
                 "feature_importance_top", "coefficients_signed_top", "feature_stability_top",
                 "feature_importance_by_fold_heatmap", "model_selection",
                 "amino_acid_site_pss_top", "amino_acid_site_effect_range_top",
                 "amino_acid_frequency_by_confusion", "amino_acid_site_filter_funnel"]:
        assert (second / "cv/figures" / f"{name}.svg").exists(), name
    np.testing.assert_array_equal(
        pl.read_csv(first / "cv/tables/prediction_cv.tsv", separator="\t")["prob"],
        pl.read_csv(second / "cv/tables/prediction_cv.tsv", separator="\t")["prob"],
    )
