from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import phenoradar.split as split_mod
from phenoradar.config import load_and_resolve_config
from phenoradar.split import SplitError, build_split_artifacts


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_config(tmp_path: Path, metadata: Path, tpm: Path) -> Path:
    return _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
""".strip()
        + "\n",
    )


def _fixture_data(tmp_path: Path) -> tuple[Path, Path]:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t0\tg1",
                "sp3\t1\tg2",
                "sp4\t0\tg2",
                "sp5\t1\t",
                "sp6\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
                "sp6\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    return metadata, tpm


def test_build_split_artifacts_success(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    artifacts = build_split_artifacts(config)

    assert artifacts.fold_count == 2
    assert artifacts.pool_counts["training_validation"] == 4
    assert artifacts.pool_counts["external_test"] == 1
    assert artifacts.pool_counts["discovery_inference"] == 1
    assert artifacts.expression_rows_excluded == 0
    assert artifacts.split_manifest.height > 0


def test_split_manifest_fold_train_validation_are_disjoint_by_group_and_species(
    tmp_path: Path,
) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    manifest = build_split_artifacts(config).split_manifest
    fold_ids = (
        manifest.filter(pl.col("pool") == "validation")
        .select("fold_id")
        .unique()
        .sort("fold_id")
        .to_series()
        .to_list()
    )
    assert fold_ids

    for fold_id in fold_ids:
        train_rows = manifest.filter((pl.col("pool") == "train") & (pl.col("fold_id") == fold_id))
        valid_rows = manifest.filter(
            (pl.col("pool") == "validation") & (pl.col("fold_id") == fold_id)
        )
        train_groups = set(train_rows.select("group_id").to_series().to_list())
        valid_groups = set(valid_rows.select("group_id").to_series().to_list())
        train_species = set(train_rows.select("species").to_series().to_list())
        valid_species = set(valid_rows.select("species").to_series().to_list())

        assert train_groups.isdisjoint(valid_groups)
        assert train_species.isdisjoint(valid_species)


def test_invalid_trait_values_are_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\tmaybe\tg1",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError):
        build_split_artifacts(config)


def test_missing_species_in_expression_are_rejected(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    # Remove sp6 from expression to trigger coverage error.
    tpm.write_text(
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError):
        build_split_artifacts(config)


def test_single_class_group_is_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp2\t1\tg1",
                "sp3\t0\tg2",
                "sp4\t1\tg2",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError):
        build_split_artifacts(config)


def test_invalid_trait_values_error_lists_offending_values(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\tmaybe\tg1",
                "sp2\t2\tg1",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="offending values: 2, maybe"):
        build_split_artifacts(config)


def test_duplicate_species_in_metadata_is_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\tg1",
                "sp1\t0\tg1",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="must be unique"):
        build_split_artifacts(config)


def test_empty_species_identifier_in_metadata_is_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "\t1\tg1",
                "sp2\t0\tg1",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="empty species identifiers"):
        build_split_artifacts(config)


def test_expression_rows_excluded_counts_rows_not_in_metadata(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    tpm.write_text(
        tpm.read_text(encoding="utf-8")
        + "\n".join(
            [
                "sp_extra\tOG1\t1.0",
                "sp_extra\tOG2\t2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    artifacts = build_split_artifacts(config)

    assert artifacts.expression_rows_excluded == 2


def test_split_manifest_is_sorted_by_contract_keys(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "z_sp\t1\tg1",
                "a_sp\t0\tg1",
                "y_sp\t1\tg2",
                "b_sp\t0\tg2",
                "ext\t1\t",
                "inf\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "z_sp\tOG1\t1.0",
                "a_sp\tOG1\t2.0",
                "y_sp\tOG1\t3.0",
                "b_sp\tOG1\t4.0",
                "ext\tOG1\t5.0",
                "inf\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    artifacts = build_split_artifacts(config)
    manifest = artifacts.split_manifest
    sorted_manifest = manifest.sort(["pool", "fold_id", "group_id", "species"], nulls_last=False)

    assert manifest.to_dicts() == sorted_manifest.to_dicts()


def test_missing_required_metadata_columns_are_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4",
                "sp1\t1",
                "sp2\t0",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="Missing required columns in metadata"):
        build_split_artifacts(config)


def test_missing_required_expression_species_column_is_rejected(tmp_path: Path) -> None:
    metadata, _tpm = _fixture_data(tmp_path)
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "not_species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
                "sp3\tOG1\t3.0",
                "sp4\tOG1\t4.0",
                "sp5\tOG1\t5.0",
                "sp6\tOG1\t6.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="Missing required column in expression data"):
        build_split_artifacts(config)


def test_missing_metadata_file_is_rejected(tmp_path: Path) -> None:
    metadata = tmp_path / "missing_metadata.tsv"
    _metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="Input file not found"):
        build_split_artifacts(config)


def test_expression_scan_file_not_found_is_wrapped_as_split_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    def _raise_file_not_found(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(split_mod.pl, "scan_csv", _raise_file_not_found)

    with pytest.raises(SplitError, match="Input file not found"):
        build_split_artifacts(config)


def test_no_training_validation_pool_is_rejected(tmp_path: Path) -> None:
    metadata = _write(
        tmp_path / "species_metadata.tsv",
        "\n".join(
            [
                "species\tC4\tcontrast_pair_id",
                "sp1\t1\t",
                "sp2\t\t",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "sp1\tOG1\t1.0",
                "sp2\tOG1\t2.0",
            ]
        )
        + "\n",
    )
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])

    with pytest.raises(SplitError, match="No species available in training and validation pool"):
        build_split_artifacts(config)


def test_group_kfold_without_n_splits_is_rejected_when_mutated(tmp_path: Path) -> None:
    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    config_missing_n_splits = config.model_copy(
        update={
            "split": config.split.model_copy(
                update={"outer_cv_strategy": "group_kfold", "outer_cv_n_splits": None}
            )
        }
    )

    with pytest.raises(SplitError, match="outer_cv_n_splits must be set"):
        build_split_artifacts(config_missing_n_splits)


def test_outer_cv_splitter_value_error_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingLogo:
        def split(self, *_args: object, **_kwargs: object) -> object:
            def _iter() -> object:
                raise ValueError("boom")
                yield  # pragma: no cover

            return _iter()

    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    monkeypatch.setattr(split_mod, "LeaveOneGroupOut", lambda: _FailingLogo())

    with pytest.raises(SplitError, match="boom"):
        build_split_artifacts(config)


def test_outer_cv_splitter_rejects_zero_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _EmptyLogo:
        def split(self, *_args: object, **_kwargs: object) -> object:
            return iter([])

    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    monkeypatch.setattr(split_mod, "LeaveOneGroupOut", lambda: _EmptyLogo())

    with pytest.raises(SplitError, match="Outer CV produced zero folds"):
        build_split_artifacts(config)


def test_split_manifest_builder_rejects_empty_rows() -> None:
    empty_training = split_mod.pl.DataFrame(
        schema={
            "__species": split_mod.pl.String,
            "__group": split_mod.pl.String,
            "__label": split_mod.pl.Int8,
        }
    )
    empty_external = split_mod.pl.DataFrame(
        schema={"__species": split_mod.pl.String, "__label": split_mod.pl.Int8}
    )
    empty_inference = split_mod.pl.DataFrame(
        schema={"__species": split_mod.pl.String, "__label": split_mod.pl.Int8}
    )

    with pytest.raises(SplitError, match="Split manifest is empty"):
        split_mod._build_split_manifest(
            training_df=empty_training,
            external_df=empty_external,
            inference_df=empty_inference,
            folds=[],
        )


def test_expression_rows_excluded_rejects_non_integer_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeSchema:
        def names(self) -> list[str]:
            return ["species"]

    class _FakeSeries:
        def to_list(self) -> list[str]:
            return ["sp1"]

    class _FakeCollectedSpecies:
        def to_series(self) -> _FakeSeries:
            return _FakeSeries()

    class _FakeCollectedCount:
        def item(self) -> str:
            return "not-an-int"

    class _FakeScan:
        def __init__(self) -> None:
            self._collect_calls = 0

        def collect_schema(self) -> _FakeSchema:
            return _FakeSchema()

        def select(self, *_args: object, **_kwargs: object) -> _FakeScan:
            return self

        def filter(self, *_args: object, **_kwargs: object) -> _FakeScan:
            return self

        def unique(self, *_args: object, **_kwargs: object) -> _FakeScan:
            return self

        def collect(self) -> object:
            self._collect_calls += 1
            if self._collect_calls == 1:
                return _FakeCollectedSpecies()
            return _FakeCollectedCount()

    metadata, tpm = _fixture_data(tmp_path)
    config = load_and_resolve_config([_write_config(tmp_path, metadata, tpm)])
    monkeypatch.setattr(split_mod.pl, "scan_csv", lambda *_args, **_kwargs: _FakeScan())

    with pytest.raises(
        SplitError, match="Failed to compute expression rows excluded from metadata"
    ):
        build_split_artifacts(config)
