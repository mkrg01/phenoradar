from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np
import pytest

import phenoradar.cv as cv
from phenoradar.config import AppConfig
from phenoradar.glmnet import GlmnetLogisticRegression


def _folds() -> list[cv.InnerCvPreprocessedFold]:
    rng = np.random.default_rng(817)
    folds = []
    # Different feature schemas ensure fitted state cannot cross inner folds.
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


def _config(*, n_jobs: int) -> AppConfig:
    return AppConfig.model_validate(
        {
            "runtime": {"n_jobs": n_jobs},
            "model_selection": {
                "selected_candidate_count": 6,
                "inner_cv_strategy": "logo",
                "selection_rule": "one_se",
                "search_space": {
                    "lambda": [0.01, 0.1, 0.03],
                    "alpha": [0.4, 1.0],
                    "thresh": [1e-14],
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


def test_parallel_paths_match_serial_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds = _folds()
    monkeypatch.setattr(cv, "_build_inner_cv_preprocessed_folds", lambda **_kwargs: folds)
    results = []
    for n_jobs in (1, 4):
        progress: list[tuple[str, str | None]] = []
        result = _select(_config(n_jobs=n_jobs), progress)
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


def test_native_paths_are_batched_by_fold_and_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    folds = _folds()
    monkeypatch.setattr(cv, "_build_inner_cv_preprocessed_folds", lambda **_kwargs: folds)
    original_fit = cv.fit_glmnet_path
    calls: list[tuple[int, float, list[float]]] = []
    lock = Lock()

    def tracked_fit(
        x: np.ndarray, y: np.ndarray, lambdas: list[float], **kwargs: Any
    ) -> list[GlmnetLogisticRegression]:
        with lock:
            calls.append((id(x), kwargs["alpha"], lambdas))
        return original_fit(x, y, lambdas, **kwargs)

    monkeypatch.setattr(cv, "fit_glmnet_path", tracked_fit)
    config = _config(n_jobs=4)
    result = _select(config, [])
    assert len(calls) == 6  # Three folds times two alpha values, not 18 separate fits.
    assert len({(fold, alpha) for fold, alpha, _ in calls}) == 6
    assert all(lambdas == [0.1, 0.03, 0.01] for _, _, lambdas in calls)
    # An independently fitted candidate must reproduce each path score.
    for selected in result.selected_candidates:
        score, rows = cv._score_candidate_inner_cv(
            config=config,
            training_scope_id="outer_fold_1",
            source_sample_set_id=0,
            candidate=selected.candidate,
            preprocessed_folds=folds,
            estimator_n_jobs=1,
        )
        assert score == pytest.approx(selected.score, abs=1e-6)
        assert all(row["fit_diagnostic"].converged for row in rows)
