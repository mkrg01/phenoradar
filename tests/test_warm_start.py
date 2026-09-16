from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any

import numpy as np
import pytest

import phenoradar.cv as cv
from phenoradar.config import AppConfig


def _folds() -> list[cv.InnerCvPreprocessedFold]:
    rng = np.random.default_rng(817)
    folds = []
    # Different feature schemas ensure warm-start state cannot cross inner folds.
    for fold_index, feature_count in enumerate((1, 4, 3)):
        y = np.tile([0, 1], 50)
        x = rng.normal(size=(100, feature_count))
        x[:, 0] += 1.5 * y
        folds.append(
            cv.InnerCvPreprocessedFold(
                inner_fold_id=str(fold_index),
                x_train=x[:80],
                x_valid=x[80:],
                y_train=y[:80],
                y_valid=y[80:],
                sample_weight=rng.uniform(0.5, 2.0, size=80),
            )
        )
    return folds


def _config(*, warm: bool, n_jobs: int) -> AppConfig:
    return AppConfig.model_validate(
        {
            "model": {"logistic_warm_start_path": warm},
            "runtime": {"n_jobs": n_jobs},
            "model_selection": {
                "selected_candidate_count": 6,
                "inner_cv_strategy": "logo",
                "selection_rule": "one_se",
                "search_space": {
                    "alpha": [0.01, 0.1, 0.03],
                    "l1_ratio": [0.4, 1.0],
                    "gradient_tol": [1e-8],
                },
            },
        }
    )


def _select(config: AppConfig, progress: list[tuple[str, str | None]]) -> cv.SourceSelectionResult:
    return cv._prepare_source_selection(
        config=config,
        training_scope_id="outer_fold_1",
        source_sample_set_id=0,
        sampled_idx=np.arange(4),
        x_train_raw=np.array([[0.0], [1.0], [2.0], [3.0]]),
        y_train=np.array([0, 1, 0, 1]),
        groups_train=np.array(["a", "a", "b", "b"]),
        feature_names=["OG1"],
        warnings=[],
        progress_callback=lambda event, detail: progress.append((event, detail)),
    )


def test_parallel_warm_paths_match_serial_warm_and_cold_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds = _folds()
    monkeypatch.setattr(cv, "_build_inner_cv_preprocessed_folds", lambda **_kwargs: folds)
    results = []
    for warm, n_jobs in ((False, 1), (True, 1), (True, 4), (False, 4)):
        progress: list[tuple[str, str | None]] = []
        result = _select(_config(warm=warm, n_jobs=n_jobs), progress)
        assert result.n_scored_candidates == 6
        assert len(result.trial_rows) == 18
        assert all(row["fit_diagnostic"].converged for row in result.trial_rows)
        assert len(progress) == 6
        assert all(event == "selection_candidate_done" for event, _ in progress)
        assert "candidate_progress=6/6" in str(progress[-1][1])
        results.append(result)

    reference = results[0]
    reference_scores = {
        (row["candidate_index"], row["inner_fold_id"]): row["metric_value"]
        for row in reference.trial_rows
    }
    reference_selected = {
        item.candidate.candidate_index: item for item in reference.selected_candidates
    }
    for result in results[1:]:
        scores = {
            (row["candidate_index"], row["inner_fold_id"]): row["metric_value"]
            for row in result.trial_rows
        }
        assert scores == pytest.approx(reference_scores, abs=1e-6)
        assert result.selected_candidates[0].candidate == reference.selected_candidates[0].candidate
        for selected in result.selected_candidates:
            expected = reference_selected[selected.candidate.candidate_index]
            assert selected.score == pytest.approx(expected.score, abs=1e-6)
            assert selected.score_std_error == pytest.approx(expected.score_std_error, abs=1e-6)


def test_parallel_warm_paths_keep_descending_alphas_and_isolate_fitted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds = _folds()
    monkeypatch.setattr(cv, "_build_inner_cv_preprocessed_folds", lambda **_kwargs: folds)
    original_score = cv._score_candidate_inner_cv
    histories: dict[int, list[tuple[str, float, float]]] = defaultdict(list)
    retained_caches: list[dict[str, Any]] = []
    lock = Lock()

    def tracked_score(**kwargs: Any) -> tuple[float, list[dict[str, Any]]]:
        cache = kwargs["warm_start_estimators"]
        candidate = kwargs["candidate"]
        candidate_folds = kwargs["preprocessed_folds"]
        assert len(candidate_folds) == 1
        assert kwargs["estimator_n_jobs"] == 1
        with lock:
            retained_caches.append(cache)
            histories[id(cache)].append(
                (
                    candidate_folds[0].inner_fold_id,
                    candidate.params["l1_ratio"],
                    candidate.params["alpha"],
                )
            )
        return original_score(**kwargs)

    monkeypatch.setattr(cv, "_score_candidate_inner_cv", tracked_score)
    _select(_config(warm=True, n_jobs=4), [])
    _select(_config(warm=True, n_jobs=4), [])

    # Separate caches for each fold/ratio path, including across independent searches.
    assert len(histories) == 12
    for history in histories.values():
        assert len({(fold_id, ratio) for fold_id, ratio, _ in history}) == 1
        assert [alpha for _, _, alpha in history] == [0.1, 0.03, 0.01]
