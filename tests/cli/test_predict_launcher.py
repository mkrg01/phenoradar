from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from phenoradar.launcher import _maybe_set_polars_max_threads


@pytest.mark.parametrize(
    "args, configured, expected",
    [
        ([], None, "1"),
        (["--n-jobs", "3"], None, "3"),
        (["--n-jobs=4"], 2, "4"),
        ([], 2, "2"),
        (["--n-jobs", "3"], 0, "3"),
    ],
)
def test_predict_sets_pool_from_cli_then_config_then_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    configured: int | None,
    expected: str,
) -> None:
    monkeypatch.setenv("POLARS_MAX_THREADS", "20")
    argv = ["predict", *args]
    if configured is not None:
        config = tmp_path / "config.yml"
        config.write_text(f"runtime:\n  n_jobs: {configured}\n")
        argv += ["-c", str(config)]
    _maybe_set_polars_max_threads(argv)
    assert os.environ["POLARS_MAX_THREADS"] == expected


def test_predict_configures_actual_polars_pool_before_import() -> None:
    code = """
import json
from phenoradar.launcher import _maybe_set_polars_max_threads
_maybe_set_polars_max_threads(['predict', '--n-jobs', '3'])
import polars as pl
print(json.dumps({'threads': pl.thread_pool_size()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "POLARS_MAX_THREADS": "1"},
    )
    assert json.loads(result.stdout)["threads"] == 3


def test_predict_applies_path_overrides_before_resolving_thread_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLARS_MAX_THREADS", "20")
    config = tmp_path / "config.yml"
    config.write_text("data:\n  tpm_path: null\n  metadata_path: 42\n")
    _maybe_set_polars_max_threads(
        [
            "predict",
            "-c",
            str(config),
            "--n-jobs",
            "3",
            "--tpm-path=tpm.tsv",
            "--metadata-path",
            "metadata.tsv",
        ]
    )
    assert os.environ["POLARS_MAX_THREADS"] == "3"
