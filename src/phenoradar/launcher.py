"""CLI launcher that configures process-level thread caps before importing Polars."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from phenoradar.config import ConfigError, load_and_resolve_config

_COMMANDS_WITH_CONFIG = {"run", "predict", "config"}


def _infer_command(argv: list[str]) -> str | None:
    for token in argv:
        if token.startswith("-"):
            continue
        return token
    return None


def _extract_config_paths(argv: list[str]) -> list[Path]:
    config_paths: list[Path] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"--config", "-c"}:
            if index + 1 < len(argv):
                config_paths.append(Path(argv[index + 1]))
                index += 2
                continue
            index += 1
            continue
        if token.startswith("--config="):
            config_paths.append(Path(token.split("=", 1)[1]))
            index += 1
            continue
        if token.startswith("-c") and len(token) > 2:
            config_paths.append(Path(token[2:]))
            index += 1
            continue
        index += 1
    return config_paths


def _maybe_set_polars_max_threads(argv: list[str]) -> None:
    if os.environ.get("POLARS_MAX_THREADS") is not None:
        return

    command = _infer_command(argv)
    if command not in _COMMANDS_WITH_CONFIG:
        return

    config_paths = _extract_config_paths(argv)
    if len(config_paths) > 1:
        return
    allow_empty = command in {"config"}
    try:
        resolved = load_and_resolve_config(config_paths, allow_empty=allow_empty)
    except (ConfigError, OSError, ValueError):
        return

    runtime_n_jobs = int(resolved.runtime.n_jobs)
    if runtime_n_jobs >= 1:
        os.environ["POLARS_MAX_THREADS"] = str(runtime_n_jobs)


def main() -> None:
    """CLI script entrypoint."""
    argv = sys.argv[1:]
    _maybe_set_polars_max_threads(argv)
    from phenoradar.cli import app

    app()
