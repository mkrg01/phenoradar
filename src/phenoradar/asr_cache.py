"""Content-addressed reuse of successful, deterministic nwkit ASR outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from phenoradar.provenance import sha256_file

OUTPUT_NAMES = (
    "asr.tsv", "model.tsv", "annotated_tree.nhx", "nwkit.stdout.txt", "nwkit.stderr.txt",
)


def cache_key(executable: str, version: str, arguments: list[str], directory: Path) -> str | None:
    """Reuse only a Python entry point whose runtime we can identify locally.

    Unknown wrappers and other environments run normally without a cache. Source
    hashes also invalidate editable installs whose version number did not change.
    """
    try:
        script = Path(executable)
        interpreter = Path(script.read_bytes().splitlines()[0].decode().removeprefix("#!"))
        current = Path(sys.executable)
        # A venv can be mounted at multiple paths. Comparing only the executable
        # would also match different venvs that symlink to the same base Python.
        if not interpreter.is_absolute() or not interpreter.parent.samefile(current.parent):
            return None
        if not interpreter.samefile(current):
            return None
        spec = importlib.util.find_spec("nwkit")
        if spec is None or spec.origin is None:
            return None
        package = Path(spec.origin).parent
        source_hashes = {
            str(path.relative_to(package)): sha256_file(path)
            for path in sorted(package.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".so", ".pyd"}
        }
        if not source_hashes:
            return None
        # Record the complete installed dependency versions as well as the
        # actual nwkit source. This includes scipy/numpy/ete4 and editable builds.
        distributions = sorted(
            (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
        )
        normalized_arguments = [
            "<run>/" + Path(arg).name if Path(arg).parent == directory else arg
            for arg in arguments[1:]
        ]
        identity = {
            "schema": 1,
            "version": version,
            "entry_point": sha256_file(script),
            "python": sys.version,
            "python_binary": sha256_file(Path(sys.executable)),
            "nwkit_source": source_hashes,
            "distributions": distributions,
            "arguments": normalized_arguments,
            "tree": sha256_file(directory / "tree.nwk"),
            "traits": sha256_file(directory / "traits.tsv"),
            "environment": {
                name: os.environ.get(name) for name in (
                    "PYTHONPATH", "PYTHONHOME", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                )
            },
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    except (OSError, ValueError, IndexError, UnicodeError, TypeError):
        return None


def restore(cache_path: Path, key: str, directory: Path) -> dict[str, Any] | None:
    """Validate every member before restoring; a missing/broken cache is a miss."""
    try:
        with ZipFile(cache_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if not isinstance(manifest, dict) or manifest.get("key") != key:
                return None
            if manifest.get("returncode") != 0 or not isinstance(manifest.get("command"), list):
                return None
            hashes = manifest.get("sha256")
            if not isinstance(hashes, dict) or set(hashes) != set(OUTPUT_NAMES):
                return None
            contents = {name: archive.read(name) for name in OUTPUT_NAMES}
            for name, content in contents.items():
                if hashlib.sha256(content).hexdigest() != hashes[name]:
                    return None
            for name, content in contents.items():
                (directory / name).write_bytes(content)
            return manifest
    except (OSError, ValueError, KeyError, BadZipFile, EOFError, RuntimeError):
        return None


def store(cache_path: Path, key: str, directory: Path, command: list[str]) -> None:
    """Publish one complete archive atomically; cache failures cannot fail a run."""
    temporary: Path | None = None
    try:
        contents = {name: (directory / name).read_bytes() for name in OUTPUT_NAMES}
        manifest = {
            "key": key,
            "command": command,
            "returncode": 0,
            "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache_path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
            for name, data in contents.items():
                archive.writestr(name, data)
        os.replace(temporary, cache_path)
    except OSError:
        pass
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
