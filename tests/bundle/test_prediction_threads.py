from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_info

from phenoradar.bundle import _predict_with_jobs


@pytest.mark.parametrize("fail", [False, True])
def test_predict_overrides_fitted_workers_and_native_threads_then_restores(
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    model = RandomForestClassifier(n_estimators=2, n_jobs=1, random_state=42).fit(x, [0, 0, 1, 1])
    original_predict = model.predict_proba
    expected = original_predict(x)[:, 1]
    model.n_jobs = 7

    def observe(matrix: np.ndarray) -> np.ndarray:
        assert model.n_jobs == 2
        assert all(pool["num_threads"] <= 2 for pool in threadpool_info())
        if fail:
            raise ValueError("prediction failed")
        return original_predict(matrix)

    monkeypatch.setattr(model, "predict_proba", observe)
    if fail:
        with pytest.raises(ValueError, match="prediction failed"):
            _predict_with_jobs(model, x, 2)
    else:
        np.testing.assert_allclose(_predict_with_jobs(model, x, 2), expected)
    assert model.n_jobs == 7
