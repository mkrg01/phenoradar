import shutil
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

    monkeypatch.chdir(tmp_path)

    import phenoradar.amino_acid_parallel as parallel

    path = (combined_config(tmp_path, aggregation) if combined
            else fixture_config(tmp_path, aggregation))
    config = load_and_resolve_config([path])
    # Noncontiguous numeric IDs exercise seed-ID transport into spawned workers.
    metadata_path = Path(config.data.metadata_path)
    pl.read_csv(metadata_path, separator="\t").with_columns(
        pl.col(config.split.group_col).replace_strict(
            {"g0": "0", "g1": "2", "g2": "10", "g3": "20"}, default=None,
        )
    ).write_csv(metadata_path, separator="\t")
    config.preprocess.amino_acid.ranked_site_filter.method = "none"
    alignments = Path(config.data.alignment_dir)
    (alignments / "OG2.faa").write_text((alignments / "OG1.faa").read_text())
    # This OG fails coverage in the pilot (g0 held out), but passes in later folds.
    # OG threads allocate its cache entry before its data file has been created.
    records = (alignments / "OG1.faa").read_text().split(">")[1:]
    (alignments / "OG_later.faa").write_text("".join(
        ">" + record for record in records if record.split("_", 1)[0] in {"s0", "s1", "s2"}
    ))
    # Keep this integration test small; the complete output tables are compared.
    config.figures.top_features = 3
    first = run_amino_acid(config, tmp_path / "serial")
    validation = pl.read_csv(first / "split/tables/split_manifest.tsv", separator="\t").filter(
        pl.col("pool") == "validation"
    )
    assert validation.filter(pl.col("fold_id").cast(pl.String) != pl.col("group_id").cast(
        pl.String)).is_empty()
    # Keep exercising genuinely empty worker caches despite cross-run reuse.
    shutil.rmtree(Path("runs/.phenoradar_cache/alignments/frequencies"), ignore_errors=True)
    config.runtime.n_jobs = 4
    monkeypatch.setattr(parallel, "fold_workers", lambda *args: 2)
    original_snapshot = parallel.snapshot

    def snapshot_with_empty_cache(store, expression):
        payload = original_snapshot(store, expression)
        key = ("OG_later", aggregation == "tpm_weighted_frequency")
        if aggregation != "best_covered":
            assert payload["store"]["_frequency_caches"][key].length == 0
        return payload

    monkeypatch.setattr(parallel, "snapshot", snapshot_with_empty_cache)
    second = run_amino_acid(config, tmp_path / "parallel")
    later_sites = pl.read_csv(
        second / "cv/tables/amino_acid_site_pss.tsv", separator="\t",
    ).filter(pl.col("orthogroup") == "OG_later")
    assert later_sites["scope"].n_unique() == 2
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
                 "feature_importance_top", "coefficients_signed_top",
                 "feature_importance_by_fold_heatmap", "model_selection",
                 "amino_acid_site_pss_top", "amino_acid_site_effect_range_top",
                 "amino_acid_frequency_by_confusion", "amino_acid_site_filter_funnel"]:
        assert (second / "cv/figures" / f"{name}.svg").exists(), name
    for root in [first, second]:
        assert not (root / "cv/figures/feature_stability_top.svg").exists()
        assert (root / "cv/tables/feature_stability_by_feature.tsv").exists()
    np.testing.assert_array_equal(
        pl.read_csv(first / "cv/tables/prediction_cv.tsv", separator="\t")["prob"],
        pl.read_csv(second / "cv/tables/prediction_cv.tsv", separator="\t")["prob"],
    )


@pytest.mark.parametrize("empty_file_exists", [False, True])
def test_worker_initializes_empty_caches_privately_and_shares_immutable_extents(
    tmp_path: Path, monkeypatch, empty_file_exists: bool,
) -> None:
    import phenoradar.amino_acid_parallel as parallel
    from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache

    monkeypatch.setattr(parallel, "_WORKER", None)
    monkeypatch.setenv("POLARS_MAX_THREADS", "1")
    species = ["a", "b"]

    def compute(samples, columns):
        values = np.zeros((len(samples), len(columns), 20), dtype=np.float32)
        for row, sample in enumerate(samples):
            values[row, np.arange(len(columns)), (columns + species.index(sample)) % 20] = 1
        return values, np.ones(values.shape[:2], dtype=np.float32)

    empty = AminoAcidFrequencyCache(tmp_path / "empty.bin", {s: i for i, s in enumerate(species)})
    populated = AminoAcidFrequencyCache(
        tmp_path / "populated.bin", {s: i for i, s in enumerate(species)},
    )
    if empty_file_exists:
        empty.path.touch()
    populated.get(species, np.array([0]), compute, block_size=1)
    original_bytes = populated.path.read_bytes()
    original_columns = populated.columns.copy()
    parent_caches = {("empty", True): empty, ("populated", True): populated}
    payload = {"store": {"ogs": {}, "species": set(species), "retained_species": set(species),
                         "cache_frequencies": True, "_frequency_caches": parent_caches,
                         "tpm_identity": None, "_site_presence": {},
                         "_frequency_directory": None}, "expression": None}
    parallel.initialize(payload, 1)
    worker = parallel._WORKER[0]
    try:
        fresh = worker._frequency_caches[("empty", True)]
        shared = worker._frequency_caches[("populated", True)]
        assert not fresh.path.is_symlink()
        assert not fresh.path.exists()
        assert not shared.path.exists()
        assert shared._chunks[0].path == populated.path
        # Existing columns reuse the shared data without recomputing or copying it.
        def unexpected_compute(*args):
            pytest.fail("A cached site must not be recomputed")

        values, coverage = shared.get(species, np.array([0]), unexpected_compute, block_size=1)
        expected, expected_coverage = compute(species, np.array([0]))
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(coverage, expected_coverage)
        assert not shared.path.exists()
        # Only new columns are written privately; shared extents are never copied.
        for cache in [fresh, shared]:
            values, coverage = cache.get(species, np.array([0, 1]), compute, block_size=1)
            expected, expected_coverage = compute(species, np.array([0, 1]))
            np.testing.assert_array_equal(values, expected)
            np.testing.assert_array_equal(coverage, expected_coverage)
            assert cache.path.is_file() and not cache.path.is_symlink()
        assert empty.path.exists() == empty_file_exists
        if empty_file_exists:
            assert empty.path.stat().st_size == 0
        assert empty.length == 0 and empty.columns.size == 0
        assert populated.path.read_bytes() == original_bytes
        np.testing.assert_array_equal(populated.columns, original_columns)
    finally:
        worker.close()
    assert parent_caches == {("empty", True): empty, ("populated", True): populated}


def test_later_fold_starts_while_earlier_result_is_waiting(tmp_path: Path, monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import joblib

    import phenoradar.amino_acid_parallel as parallel
    from phenoradar.config import AppConfig

    started, released = Event(), Event()

    class Executor(ThreadPoolExecutor):
        def __init__(self, max_workers, **kwargs):
            super().__init__(max_workers=max_workers)

    def pilot(config, store, expression, manifest, fold, directory, model_base, progress):
        directory.mkdir()
        joblib.dump((None, None, None, 2), directory / "result.joblib")
        return directory

    def worker(config, manifest, fold, directory, model_base):
        assert model_base == (int(fold) - 1) * 1_000_000
        if fold == "2":
            started.set()
            assert released.wait(5), "A free worker must start fold 4 before fold 2 finishes"
        elif fold == "3":
            assert started.wait(5)
        elif fold == "4":
            released.set()
        return directory

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(parallel, "run_fold", pilot)
    monkeypatch.setattr(parallel, "worker", worker)
    monkeypatch.setattr(parallel, "fold_workers", lambda *args: 2)
    monkeypatch.setattr(parallel, "snapshot", lambda *args: {})
    manifest = pl.DataFrame({"species": ["a", "b"], "pool": ["train", "validation"]})
    config = AppConfig()
    paths = list(parallel.outer_folds(config, None, None, manifest, ["1", "2", "3", "4"], None))
    assert [p.name for p in paths] == ["1", "2", "3", "4"]
