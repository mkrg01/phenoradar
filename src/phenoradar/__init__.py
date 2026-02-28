"""PhenoRadar package."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

__all__ = ["__version__"]


def _version_from_pyproject() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return "0+unknown"
    project = pyproject.get("project")
    if not isinstance(project, dict):
        return "0+unknown"
    version = project.get("version")
    if not isinstance(version, str):
        return "0+unknown"
    return version


try:
    __version__ = package_version("phenoradar")
except PackageNotFoundError:
    __version__ = _version_from_pyproject()
