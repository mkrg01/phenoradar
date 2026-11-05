from __future__ import annotations

import gzip
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import phenoradar.expression_input_cache as cache_mod
from phenoradar.config import AppConfig
from phenoradar.cv import CVError, ExpressionMatrixBuilder, RunExpressionCache
from phenoradar.timing import TimingRecorder


def config_for(path: Path) -> AppConfig:
    return AppConfig.model_validate({"data": {"tpm_path": str(path)}, "runtime": {"n_jobs": 1}})


def write_input(path: Path) -> AppConfig:
    path.write_text(
        "species\torthogroup\ttpm\talternate\na\tf\t1\t11\nb\tf\t2\t12\na\tf\t0.25\t13\nb\tg\t0\t14\n"
    )
    return config_for(path)


def cached_matrix(
    config: AppConfig, directory: Path, species: list[str]
) -> tuple[np.ndarray, list[str], Path, set[str]]:
    recorder = TimingRecorder()
    with RunExpressionCache(cache_directory=directory) as cache:
        builder = cache.get_builder(config)
        builder.configure_timing(recorder, scope="test")
        builder.cache_species(species)
        matrix, features = builder.build_matrix(species)
        assert builder._cached_long_path is not None
        entry = builder._cached_long_path
    assert entry.exists()
    return matrix, features, entry, set(recorder.to_frame()["stage"])


def test_reuse_across_runs_and_identical_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_input(tmp_path / "input.tsv")
    expected, names = ExpressionMatrixBuilder(config).build_matrix(["b", "a", "b"])
    actual, features, entry, stages = cached_matrix(config, tmp_path / "cache", ["b", "a", "b"])
    np.testing.assert_array_equal(actual, expected)
    assert features == names and "input_normalize_cache" in stages
    copied = tmp_path / "copied.tsv"
    copied.write_bytes(Path(config.data.tpm_path).read_bytes())
    config.data.tpm_path = str(copied)
    config.preprocess.expression.absent_feature_fill = "nan"

    def no_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("Cache hit must not start normalization")

    monkeypatch.setattr(cache_mod.subprocess, "run", no_build)
    actual, features, reused, stages = cached_matrix(config, tmp_path / "cache", ["a", "b", "a"])
    assert reused == entry and "input_cache_hit" in stages
    assert actual[0, features.index("f")] == 1.25
    assert np.isnan(actual[0, features.index("g")])
    assert actual[1, features.index("g")] == 0


def test_content_change_with_restored_mtime_invalidates_cache(tmp_path: Path) -> None:
    path = tmp_path / "input.tsv"
    config = write_input(path)
    _, _, before, _ = cached_matrix(config, tmp_path / "cache", ["a", "b"])
    stat = path.stat()
    path.write_text(path.read_text().replace("a\tf\t1\t11", "a\tf\t3\t11"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    matrix, features, after, _ = cached_matrix(config, tmp_path / "cache", ["a", "b"])
    assert before != after
    assert matrix[0, features.index("f")] == 3.25


def test_columns_species_and_format_are_part_of_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_input(tmp_path / "input.tsv")
    _, _, first, _ = cached_matrix(config, tmp_path / "cache", ["a", "b"])
    _, _, subset, _ = cached_matrix(config, tmp_path / "cache", ["a"])
    config.data.value_col = "alternate"
    values, names, alternate, _ = cached_matrix(config, tmp_path / "cache", ["a", "b"])
    assert values[0, names.index("f")] == 24
    monkeypatch.setattr(cache_mod, "NORMALIZATION_VERSION", cache_mod.NORMALIZATION_VERSION + 1)
    _, _, versioned, _ = cached_matrix(config, tmp_path / "cache", ["a", "b"])
    assert len({first, subset, alternate, versioned}) == 4


@pytest.mark.parametrize("corruption", ["payload", "missing_part", "manifest"])
def test_incomplete_or_corrupted_entries_are_rebuilt(tmp_path: Path, corruption: str) -> None:
    config = write_input(tmp_path / "input.tsv")
    expected, features, entry, _ = cached_matrix(config, tmp_path / "cache", ["a", "b"])
    part = next(entry.glob("part-*.parquet"))
    if corruption == "payload":
        data = bytearray(part.read_bytes())
        data[len(data) // 2] ^= 1
        part.write_bytes(data)
    elif corruption == "missing_part":
        part.unlink()
    else:
        (entry / "manifest.json").write_text("{broken")
    actual, names, restored, stages = cached_matrix(config, tmp_path / "cache", ["a", "b"])
    assert restored == entry and names == features and "input_normalize_cache" in stages
    np.testing.assert_array_equal(actual, expected)


def test_concurrent_requests_publish_one_complete_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_input(tmp_path / "input.tsv")
    original = cache_mod.subprocess.run
    calls: list[object] = []

    def tracked(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return original(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(cache_mod.subprocess, "run", tracked)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(cached_matrix, config, tmp_path / "cache", ["a", "b"]) for _ in range(2)
        ]
        first, second = [future.result() for future in futures]
    assert len(calls) == 1 and first[2] == second[2]
    np.testing.assert_array_equal(first[0], second[0])


@pytest.mark.parametrize("compressed", [False, True])
def test_partitioned_cache_preserves_duplicates_and_missing_values(
    tmp_path: Path, compressed: bool
) -> None:
    species = [f"s{i:03d}" for i in range(130)]
    lines = ["species\torthogroup\ttpm"]
    for fraction in ["0.25", "0.5"]:
        lines.extend(
            f" {name} \t f{index:03d} \t{fraction}"
            for name in reversed(species)
            for index in range(100)
        )
    lines.extend([f"{species[0]}\tZERO\t0", "ignored\t\tbad"])
    path = tmp_path / ("input.tsv.gz" if compressed else "input.tsv")
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(gzip.compress(data) if compressed else data)
    config = config_for(path)
    config.preprocess.expression.absent_feature_fill = "nan"
    expected, names = ExpressionMatrixBuilder(config).build_matrix(species)
    actual, features, entry, _ = cached_matrix(config, tmp_path / "cache", species)
    assert len(list(entry.glob("part-*.parquet"))) > 1 and names == features
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "values", [["1", "bad", "-2"], ["1", "", "2"], ["1", "NaN", "2"], ["1", "1e308", "1e308"]]
)
def test_invalid_consumed_rows_keep_diagnostics_and_are_never_published(
    tmp_path: Path, values: list[str]
) -> None:
    path = tmp_path / "input.tsv"
    path.write_text(
        "species\torthogroup\ttpm\n"
        + "".join(f"a\tBAD\t{v}\n" for v in values)
        + "a\tGOOD\t1\nignored\tBAD\t-1\n"
    )
    config = config_for(path)
    with pytest.raises(CVError) as direct:
        ExpressionMatrixBuilder(config).build_matrix(["a"], feature_order=["GOOD"])
    with pytest.raises(CVError) as persistent:
        cached_matrix(config, tmp_path / "cache", ["a"])
    assert str(persistent.value) == str(direct.value)
    assert not list((tmp_path / "cache").glob("*/manifest.json"))


def test_source_change_is_detected_while_cache_is_in_use(tmp_path: Path) -> None:
    config = write_input(tmp_path / "input.tsv")
    with RunExpressionCache(cache_directory=tmp_path / "cache") as cache:
        builder = cache.get_builder(config)
        builder.cache_species(["a", "b"])
        Path(config.data.tpm_path).write_text("species\torthogroup\ttpm\na\tf\t99\n")
        with pytest.raises(CVError, match="changed while its cache was in use"):
            builder.build_matrix(["a"])


def test_failed_worker_never_publishes_a_partial_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_input(tmp_path / "input.tsv")
    original = cache_mod.subprocess.run
    monkeypatch.setattr(
        cache_mod.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "worker interrupted"),
    )
    with pytest.raises(CVError, match="worker interrupted"):
        cached_matrix(config, tmp_path / "cache", ["a", "b"])
    assert not list((tmp_path / "cache").glob("*/manifest.json"))
    assert not list((tmp_path / "cache").glob(".*.building-*"))
    monkeypatch.setattr(cache_mod.subprocess, "run", original)
    cached_matrix(config, tmp_path / "cache", ["a", "b"])


def test_catalog_does_not_expand_training_schema_and_builder_can_be_borrowed(
    tmp_path: Path,
) -> None:
    config = write_input(tmp_path / "input.tsv")
    with RunExpressionCache(cache_directory=tmp_path / "cache") as cache:
        builder = cache.get_builder(config)
        builder.cache_species(["a", "b"])
        assert builder.feature_names_for_species(["a"]) == ["f"]
        assert builder.feature_names_for_species(["a", "b"]) == ["f", "g"]
        assert cache.prepared_builder is builder
    assert cache.prepared_builder is None
    matrix, features = builder.build_matrix(["a"])
    assert features == ["f"]
    np.testing.assert_array_equal(matrix, [[1.25]])


def test_partitioned_assembly_preserves_custom_order_repeated_species_and_chunks(
    tmp_path: Path,
) -> None:
    species = [f"s{i:03d}" for i in range(130)]
    path = tmp_path / "wide.tsv"
    path.write_text(
        "species\torthogroup\ttpm\n"
        + "".join(f"{name}\tf{index:03d}\t{index + 1}\n" for name in species for index in range(20))
        + f"{species[0]}\tZERO\t0\n"
    )
    config = config_for(path)
    config.preprocess.expression.absent_feature_fill = "nan"
    order = [species[-1], *reversed(species), species[-1]]
    features = ["ABSENT", "ZERO", *[f"f{i:03d}" for i in reversed(range(20))]]
    expected, _ = ExpressionMatrixBuilder(config).build_matrix(order, feature_order=features)
    with RunExpressionCache(cache_directory=tmp_path / "cache") as cache:
        builder = cache.get_builder(config)
        builder.cache_species(species)
        builder._max_pivot_cells = 64
        actual, names = builder.build_matrix(order, feature_order=features)
    assert names == features
    np.testing.assert_array_equal(actual, expected)


def test_normalized_snapshot_must_match_recorded_provenance(tmp_path: Path) -> None:
    config = write_input(tmp_path / "input.tsv")
    with RunExpressionCache(cache_directory=tmp_path / "cache") as cache:
        builder = cache.get_builder(config)
        builder.cache_species(["a", "b"])
        assert builder._cache_source is not None
        cache.validate_input_identity([
            {"path": config.data.tpm_path, "sha256": builder._cache_source.digest}
        ])
        with pytest.raises(CVError, match="recorded input SHA-256"):
            cache.validate_input_identity([{"path": config.data.tpm_path, "sha256": "0" * 64}])
