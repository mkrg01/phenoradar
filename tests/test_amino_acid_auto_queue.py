from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from test_combined import combined_config

from phenoradar.amino_acid_run import run_amino_acid
from phenoradar.auto_lambda import AutoPathState
from phenoradar.config import AppConfig, load_and_resolve_config
from phenoradar.model_selection import generate_candidates


def _config(tmp_path, mode, aggregation, policy="per_sample_set"):
    config = load_and_resolve_config([combined_config(tmp_path, aggregation)])
    config.data.feature_mode = mode
    config.preprocess.amino_acid.ranked_site_filter.method = "none"
    config.figures.top_features = 3
    payload = config.model_dump()
    payload["model_selection"]["search_space"]["lambda"] = {
        "type": "auto_log_range", "n_points": 3, "max_extensions": 2,
    }
    payload["model_selection"]["candidate_source_policy"] = policy
    return AppConfig.model_validate(payload)


def _compare_public_tables(a, b):
    assert sorted(p.relative_to(a) for p in a.rglob("*.tsv")) == sorted(
        p.relative_to(b) for p in b.rglob("*.tsv"))
    for path in a.rglob("*.tsv"):
        if path.name in {"alignment_inputs.tsv", "sequence_correspondence.tsv",
                         "sequence_tpm_input.tsv"}:
            continue
        other = b / path.relative_to(a)
        if not path.stat().st_size:
            assert not other.stat().st_size
            continue
        left = pl.read_csv(path, separator="\t", infer_schema_length=None)
        right = pl.read_csv(other, separator="\t", infer_schema_length=None,
                            schema_overrides=left.schema)
        assert_frame_equal(left.sort(left.columns), right.sort(right.columns), check_exact=True)
    assert sorted(p.relative_to(a) for p in a.rglob("*.svg")) == sorted(
        p.relative_to(b) for p in b.rglob("*.svg"))
    original, queued = [joblib.load(p / "model_bundle/model_state.joblib") for p in [a, b]]
    for left, right in zip(original, queued, strict=True):
        np.testing.assert_array_equal(left.model.coef_, right.model.coef_)
        assert left.model.intercept_ == right.model.intercept_
        assert_frame_equal(left.transform.schema(), right.transform.schema(), check_exact=True)


@pytest.mark.parametrize("mode, aggregation", [
    ("amino_acid", "aa_frequency"), ("amino_acid", "best_covered"),
    ("amino_acid", "tpm_weighted_frequency"), ("combined", "tpm_weighted_frequency"),
])
def test_automatic_queue_matches_serial_models_trials_coordinates_and_bootstrap(
    tmp_path: Path, monkeypatch, mode, aggregation,
):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, mode, aggregation)
    serial = run_amino_acid(config, tmp_path / "serial")
    config.runtime.n_jobs = 4
    events = []
    queued = run_amino_acid(config, tmp_path / "queued", progress=events.append)
    assert any("AA automatic queue:" in event for event in events)
    _compare_public_tables(serial, queued)
    trials = pl.read_csv(queued / "cv/tables/model_selection_trials.tsv", separator="\t")
    assert (trials["lambda_path_attempted_rounds"] == 2).all()
    assert set(trials["lambda_path_stop_reason"]) == {"extension_limit_at_boundary"}


@pytest.mark.parametrize("policy", ["per_sample_set", "reuse_first_sample_set"])
def test_automatic_queue_preserves_sample_sources_and_multiple_selected_candidates(
    tmp_path: Path, monkeypatch, policy,
):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, "amino_acid", "aa_frequency", policy)
    config.sampling.strategy = "group_balanced"
    config.sampling.max_samples_per_label_per_group = 1
    config.sampling.sampled_set_count = 2
    config.model_selection.selected_candidate_count = 2
    serial = run_amino_acid(config, tmp_path / "serial")
    config.runtime.n_jobs = 4
    queued = run_amino_acid(config, tmp_path / "queued")
    _compare_public_tables(serial, queued)


@pytest.mark.parametrize("mode", ["amino_acid", "combined"])
def test_automatic_queue_preserves_numeric_fold_ids_and_execution_order(
    tmp_path: Path, monkeypatch, mode,
):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, mode, "aa_frequency")
    metadata = Path(config.data.metadata_path)
    pl.read_csv(metadata, separator="\t").with_columns(
        pl.col("contrast_pair_id").replace_strict(
            {"g0": "0", "g1": "2", "g2": "10", "g3": "20"}, default=None,
        )
    ).write_csv(metadata, separator="\t")
    serial = run_amino_acid(config, tmp_path / "serial")
    config.runtime.n_jobs = 4
    events = []
    queued = run_amino_acid(config, tmp_path / "queued", progress=events.append)
    assert [event for event in events if event.startswith("Outer CV fold")] == [
        f"Outer CV fold {fold} completed." for fold in ["0", "10", "2", "20"]
    ]
    _compare_public_tables(serial, queued)


def test_complete_round_uses_aggregate_optimum_not_individual_fold_boundary():
    config = AppConfig.model_validate({"model_selection": {
        "selected_candidate_count": 1, "inner_cv_strategy": "logo",
        "search_space": {"lambda": {"type": "auto_log_range", "n_points": 3}},
    }})
    candidates = generate_candidates(config=config, training_scope_id="test",
                                     source_sample_set_id=0, warnings=[])
    state = AutoPathState(config, candidates)
    # Fold 0 prefers the lower endpoint, but the full-CV mean prefers the interior.
    results = [{i: [{"metric_value": value}] for i, value in enumerate(values)}
               for values in [[1., .2, 0.], [1., 0., 2.]]]
    assert not state.commit_round(results)
    assert state.reason == "interior_or_null_optimum" and state.round_id == 0
    assert len(state.result().candidates) == 3


@pytest.mark.parametrize("extension_failure", [False, True])
def test_automatic_queue_round_barrier_cached_extensions_and_order(
    tmp_path: Path, monkeypatch, extension_failure,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock
    from types import SimpleNamespace

    import phenoradar.amino_acid_auto_queue as queue
    import phenoradar.amino_acid_run as run
    from phenoradar.amino_acid_resources import SparseUsage
    from phenoradar.glmnet import GlmnetError
    from phenoradar.split import build_split_artifacts

    config = _config(tmp_path, "amino_acid", "aa_frequency")
    config.runtime.n_jobs = 4
    manifest = build_split_artifacts(config).split_manifest
    release, lock = Event(), Lock()
    completed = {}
    selections = []

    class Executor(ThreadPoolExecutor):
        def __init__(self, max_workers, **kwargs):
            super().__init__(max_workers=max_workers)

    def scored(pending):
        return {c.candidate_index: [{"metric_value": float(c.params["lambda_fraction"])}]
                for c in pending}

    def initial(config, pending, *args):
        return queue._Result(scored(pending), [])

    def task(config, source, index, directory, pending, complete, round_id):
        key = tuple(source.species)
        if round_id == 0:
            if "s0" not in source.species and index == 1:
                assert release.wait(5), "Other scopes must progress while one fold waits"
            elif "s0" in source.species:
                release.set()
            with lock:
                completed.setdefault(key, set()).add(index)
            return queue._Result(scored(pending), [])
        with lock:
            done = completed[key] | ({0} if "s0" not in source.species else set())
            assert len(done) == len(source.splits), "An extension started before its round barrier"
        assert len(complete) >= 3 and len(pending) == 1
        if extension_failure and "s0" not in source.species and index == 1:
            return queue._Result({}, [], GlmnetError("cached extension failure"))
        return queue._Result(scored(pending), [])

    def selection(config, state, source, messages):
        selections.append((state.reason, len(state.complete), state.round_id))
        return [], [], state.warnings

    def refit(config, plan, *args):
        plan.directory.mkdir()
        return plan.directory

    store = SimpleNamespace(_sparse_usage=SparseUsage(), flush_frequency_caches=lambda: None)
    monkeypatch.setattr(run, "_prepare_inner_fold", lambda *args: SimpleNamespace(
        y_train=np.array([0, 1]), x_train=np.zeros((2, 2))))
    monkeypatch.setattr(queue, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(queue, "queue_workers", lambda *args: (2, 1, 1024))
    monkeypatch.setattr(queue, "snapshot", lambda *args: {})
    monkeypatch.setattr(queue, "_initial", initial)
    monkeypatch.setattr(queue, "_task", task)
    monkeypatch.setattr(queue, "_selection", selection)
    monkeypatch.setattr(queue, "_refit", refit)
    result = list(queue.automatic_scopes(config, store, None, manifest, ["1", "2", "3", "4"], None))
    assert [name for name, _ in result] == ["1", "2", "3", "4", "final_refit"]
    if extension_failure:
        assert ("extension_failed", 3, 1) in selections
    else:
        assert all(item == ("extension_limit_at_boundary", 5, 2) for item in selections)
