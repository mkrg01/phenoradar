"""Content-addressed normalized expression, independent of labels and fitted state."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

NORMALIZATION_VERSION = 2
_PARTITION_SPECIES = 64
_MAX_PARTITIONS = 64

type SourceSignature = tuple[int, int, int, int, int]


class ExpressionCacheError(ValueError):
    """An input changed or a normalized expression entry could not be built."""


def source_signature(path: Path) -> SourceSignature:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExpressionSource:
    digest: str
    signature: SourceSignature


def identify_source(
    path: Path,
    *,
    digest: str | None = None,
    expected_signature: SourceSignature | None = None,
) -> ExpressionSource:
    before = source_signature(path)
    if expected_signature is not None and before != expected_signature:
        raise ExpressionCacheError(f"Expression input changed during provenance collection: {path}")
    digest = _sha256(path) if digest is None else digest
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ExpressionCacheError("Expression cache requires a valid input SHA-256")
    if source_signature(path) != before:
        raise ExpressionCacheError(f"Expression input changed while hashing: {path}")
    return ExpressionSource(digest=digest, signature=before)


@contextlib.contextmanager
def _entry_lock(path: Path) -> Iterator[None]:
    with path.open("a+b") as handle:
        if os.name == "nt":
            windows_lock = importlib.import_module("msvcrt")

            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            windows_lock.locking(handle.fileno(), windows_lock.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                windows_lock.locking(handle.fileno(), windows_lock.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _restore(entry: Path, key_payload: Mapping[str, Any]) -> bool:
    try:
        manifest = json.loads((entry / "manifest.json").read_text())
        if manifest["identity"] != key_payload or not manifest["parts"]:
            return False
        expected = {row["name"] for row in manifest["parts"]}
        if expected != {path.name for path in entry.glob("part-*.parquet")}:
            return False
        catalogs = manifest["catalogs"]
        if {row["name"] for row in catalogs} != {"catalog.json", "features.parquet"}:
            return False
        for row in [*manifest["parts"], *catalogs]:
            name = row["name"]
            if Path(name).name != name:
                return False
            path = entry / name
            if path.stat().st_size != row["size"] or _sha256(path) != row["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _normalizer_digest() -> str:
    from phenoradar.cv import ExpressionMatrixBuilder

    functions = (
        ExpressionMatrixBuilder.__init__,
        ExpressionMatrixBuilder._normalized_rows_for_species,
        ExpressionMatrixBuilder._aggregate_normalized_rows,
        ExpressionMatrixBuilder._validate_tpm_scan,
        ExpressionMatrixBuilder._raise_invalid_tpm_values,
        _build,
    )
    code = "\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(code.encode()).hexdigest()


def normalized_cache(
    directory: Path,
    path: Path,
    species: Sequence[str],
    columns: tuple[str, str, str],
    *,
    source: ExpressionSource | None = None,
    workers: int = 1,
) -> tuple[Path, bool, ExpressionSource]:
    source = identify_source(path) if source is None else source
    if source_signature(path) != source.signature:
        raise ExpressionCacheError(f"Expression input changed before cache lookup: {path}")
    payload = {
        "version": NORMALIZATION_VERSION,
        "polars": pl.__version__,
        "normalizer_sha256": _normalizer_digest(),
        "input_sha256": source.digest,
        "species": sorted(set(species)),
        "columns": list(columns),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    entry = directory / key
    with _entry_lock(directory / f"{key}.lock"):
        hit = _restore(entry, payload)
        if not hit:
            with tempfile.TemporaryDirectory(prefix=f".{key}.building-", dir=directory) as raw:
                pending = Path(raw)
                request = {
                    "path": str(path.resolve()),
                    "species": payload["species"],
                    "columns": list(columns),
                    "output": str(pending.resolve()),
                }
                spec = pending / "request.json"
                spec.write_text(json.dumps(request))
                env = dict(os.environ)
                env["POLARS_MAX_THREADS"] = str(max(1, workers))
                # The child imports this same checkout, including editable/source installs.
                package_root = str(Path(__file__).resolve().parent.parent)
                env["PYTHONPATH"] = os.pathsep.join(
                    [package_root, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
                )
                result = subprocess.run(
                    [sys.executable, "-m", "phenoradar.expression_input_cache", str(spec)],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                response_path = pending / "response.json"
                if result.returncode:
                    if response_path.exists():
                        message = json.loads(response_path.read_text()).get("error")
                    else:
                        message = result.stderr[-4000:] or f"worker exit code {result.returncode}"
                    raise ExpressionCacheError(str(message))
                if source_signature(path) != source.signature:
                    raise ExpressionCacheError(
                        f"Expression input changed during cache creation: {path}"
                    )
                if _normalizer_digest() != payload["normalizer_sha256"]:
                    raise ExpressionCacheError(
                        "Expression normalization code changed; retry the run"
                    )
                parts = [
                    {"name": part.name, "size": part.stat().st_size, "sha256": _sha256(part)}
                    for part in sorted(pending.glob("part-*.parquet"))
                ]
                if not parts:
                    raise ExpressionCacheError("Expression cache worker produced no parts")
                catalogs = [
                    {
                        "name": name,
                        "size": (pending / name).stat().st_size,
                        "sha256": _sha256(pending / name),
                    }
                    for name in ("catalog.json", "features.parquet")
                ]
                (pending / "manifest.json").write_text(
                    json.dumps(
                        {"identity": payload, "parts": parts, "catalogs": catalogs}, sort_keys=True
                    )
                )
                spec.unlink()
                response_path.unlink()
                if entry.exists():
                    shutil.rmtree(entry)
                os.replace(pending, entry)
        if source_signature(path) != source.signature:
            raise ExpressionCacheError(f"Expression input changed while reading its cache: {path}")
    return entry, hit, source


def _build(request: Mapping[str, Any]) -> None:
    # Import the numerical pipeline only inside the short-lived normalization worker.
    from phenoradar.config import AppConfig
    from phenoradar.cv import ExpressionMatrixBuilder

    species_col, feature_col, value_col = request["columns"]
    config = AppConfig.model_validate(
        {
            "data": {
                "tpm_path": request["path"],
                "species_col": species_col,
                "feature_col": feature_col,
                "value_col": value_col,
            }
        }
    )
    builder = ExpressionMatrixBuilder(config)
    output = Path(request["output"])
    species = request["species"]
    count = min(_MAX_PARTITIONS, max(1, math.ceil(len(species) / _PARTITION_SPECIES)))
    stage = output / ".partitioned"
    stage.mkdir()
    # Explicit species-to-bucket assignment keeps every duplicate coordinate together.
    buckets = {name: index % count for index, name in enumerate(sorted(species))}
    bucket_table = pl.DataFrame({"__species": list(buckets), "__bucket": list(buckets.values())})
    normalized = (
        builder._normalized_rows_for_species(species)
        .select(
            "__species",
            "__feature",
            "__value",
            "__invalid_count",
            "__source_line",
            "__invalid_line",
            "__invalid_reason_mask",
        )
        .join(bucket_table.lazy(), on="__species", how="inner")
    )
    partition_by = getattr(pl, "PartitionBy", None)
    if partition_by is not None:
        normalized.sink_parquet(
            partition_by(stage, key="__bucket", include_key=False, approximate_bytes_per_file=None),
            compression="lz4",
            row_group_size=64_000,
        )
        groups = [sorted(folder.glob("*.parquet")) for folder in sorted(stage.iterdir())]
    else:
        # Older supported Polars can still use bounded per-species aggregation.
        normalized.sink_parquet(stage / "raw.parquet", compression="lz4", row_group_size=64_000)
        groups = [[stage / "raw.parquet"] for _ in range(count)]
    for index, paths in enumerate(groups):
        if not paths:
            continue
        scan = pl.scan_parquet(paths)
        if partition_by is None:
            scan = scan.filter(pl.col("__bucket") == index).drop("__bucket")
        builder._aggregate_normalized_rows(scan).sort("__species").sink_parquet(
            output / f"part-{index:04d}.parquet",
            compression="zstd",
            row_group_size=64_000,
        )
    parts = sorted(output.glob("part-*.parquet"))
    if not parts:
        # Empty selected input must retain the same schema and downstream coverage error.
        builder._raw_long_scan_for_species(species).sink_parquet(output / "part-0000.parquet")
        parts = [output / "part-0000.parquet"]
    builder._validate_tpm_scan(pl.scan_parquet(parts))
    catalogs = output / ".catalogs"
    catalogs.mkdir()
    part_species: dict[str, list[str]] = {}
    for index, part in enumerate(parts):
        scan = pl.scan_parquet(part)
        part_species[part.name] = sorted(
            str(value)
            for value in scan.select("__species").unique().collect().to_series().to_list()
        )
        scan.select("__feature").unique().sink_parquet(catalogs / f"{index:04d}.parquet")
    pl.scan_parquet(sorted(catalogs.glob("*.parquet"))).unique().sort("__feature").sink_parquet(
        output / "features.parquet"
    )
    (output / "catalog.json").write_text(
        json.dumps(
            {
                "species": sorted({name for names in part_species.values() for name in names}),
                "part_species": part_species,
            }
        )
    )
    shutil.rmtree(catalogs)
    shutil.rmtree(stage)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    response = Path(request["output"]) / "response.json"
    try:
        _build(request)
    except Exception as exc:
        response.write_text(json.dumps({"error": str(exc)}))
        raise SystemExit(1) from None
    response.write_text(json.dumps({"complete": True}))


if __name__ == "__main__":
    _main()
