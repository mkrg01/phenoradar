from __future__ import annotations

import os
import sys
import types

import pytest

import phenoradar.launcher as launcher_mod


def test_main_sets_polars_max_threads_from_runtime_n_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"value": False}
    call_args: dict[str, object] = {}
    fake_cli = types.ModuleType("phenoradar.cli")

    def _fake_app() -> None:
        called["value"] = True

    fake_cli.app = _fake_app

    class _Runtime:
        n_jobs = 3

    class _Resolved:
        runtime = _Runtime()

    def _fake_load_and_resolve_config(_paths: object, *, allow_empty: bool = False) -> object:
        call_args["allow_empty"] = allow_empty
        return _Resolved()

    monkeypatch.setattr(launcher_mod, "load_and_resolve_config", _fake_load_and_resolve_config)
    monkeypatch.setattr(sys, "argv", ["phenoradar", "run", "-c", "config.yml"])
    monkeypatch.setitem(sys.modules, "phenoradar.cli", fake_cli)
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)

    launcher_mod.main()

    assert os.environ.get("POLARS_MAX_THREADS") == "3"
    assert call_args["allow_empty"] is False
    assert called["value"] is True


def test_main_keeps_existing_polars_max_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"value": False}
    fake_cli = types.ModuleType("phenoradar.cli")

    def _fake_app() -> None:
        called["value"] = True

    fake_cli.app = _fake_app

    class _Runtime:
        n_jobs = 3

    class _Resolved:
        runtime = _Runtime()

    monkeypatch.setattr(
        launcher_mod,
        "load_and_resolve_config",
        lambda _paths, *, allow_empty=False: _Resolved(),
    )
    monkeypatch.setattr(sys, "argv", ["phenoradar", "run", "-c", "config.yml"])
    monkeypatch.setitem(sys.modules, "phenoradar.cli", fake_cli)
    monkeypatch.setenv("POLARS_MAX_THREADS", "9")

    launcher_mod.main()

    assert os.environ.get("POLARS_MAX_THREADS") == "9"
    assert called["value"] is True


def test_main_skips_config_resolution_for_non_config_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"value": False}
    fake_cli = types.ModuleType("phenoradar.cli")

    def _fake_app() -> None:
        called["value"] = True

    fake_cli.app = _fake_app

    def _raise_if_called(_paths: object) -> object:
        raise AssertionError("load_and_resolve_config should not be called")

    monkeypatch.setattr(launcher_mod, "load_and_resolve_config", _raise_if_called)
    monkeypatch.setattr(sys, "argv", ["phenoradar", "report"])
    monkeypatch.setitem(sys.modules, "phenoradar.cli", fake_cli)
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)

    launcher_mod.main()

    assert os.environ.get("POLARS_MAX_THREADS") is None
    assert called["value"] is True


def test_main_allows_empty_config_for_config_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"value": False}
    call_args: dict[str, object] = {}
    fake_cli = types.ModuleType("phenoradar.cli")

    def _fake_app() -> None:
        called["value"] = True

    fake_cli.app = _fake_app

    class _Runtime:
        n_jobs = 2

    class _Resolved:
        runtime = _Runtime()

    def _fake_load_and_resolve_config(_paths: object, *, allow_empty: bool = False) -> object:
        call_args["allow_empty"] = allow_empty
        return _Resolved()

    monkeypatch.setattr(launcher_mod, "load_and_resolve_config", _fake_load_and_resolve_config)
    monkeypatch.setattr(sys, "argv", ["phenoradar", "config"])
    monkeypatch.setitem(sys.modules, "phenoradar.cli", fake_cli)
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)

    launcher_mod.main()

    assert os.environ.get("POLARS_MAX_THREADS") == "2"
    assert call_args["allow_empty"] is True
    assert called["value"] is True
