"""Metadata preparation helpers."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl


class MetadataError(ValueError):
    """Raised when metadata preparation fails."""


@dataclass(frozen=True)
class NCBITreeResult:
    """Artifacts produced by NCBI taxonomy tree retrieval."""

    tree_path: Path
    species_count: int
    rank: str


@dataclass(frozen=True)
class SpeciesMetadataResult:
    """Artifacts produced by species metadata generation."""

    metadata_path: Path
    species_count: int
    grouped_species_count: int
    contrast_pair_count: int
    tree_missing_species_count: int


def _load_species_trait(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path, separator="\t")
    except FileNotFoundError as exc:
        raise MetadataError(f"Input file not found: {path}") from exc
    except Exception as exc:
        raise MetadataError(f"Failed to read species trait TSV: {path}") from exc


def _require_columns(frame: pl.DataFrame, required: list[str], context: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        missing_str = ", ".join(missing)
        raise MetadataError(f"Missing required columns in {context}: {missing_str}")


def _with_normalized_species(
    frame: pl.DataFrame, *, species_col: str, context: str = "species trait TSV"
) -> pl.DataFrame:
    _require_columns(frame, [species_col], context)
    normalized = frame.with_columns(
        pl.col(species_col).cast(pl.String, strict=False).str.strip_chars().alias("__species")
    )
    invalid_species = normalized.filter(
        pl.col("__species").is_null() | (pl.col("__species") == "")
    ).height
    if invalid_species > 0:
        raise MetadataError(
            f"{context} contains {invalid_species} rows with empty species identifiers"
        )

    duplicated_species = (
        normalized.group_by("__species")
        .len()
        .filter(pl.col("len") > 1)
        .select("__species")
        .to_series()
        .to_list()
    )
    if duplicated_species:
        duplicates = ", ".join(str(v) for v in sorted(duplicated_species)[:10])
        raise MetadataError(
            f"{context} species identifiers must be unique; duplicates: {duplicates}"
        )
    return normalized


def _load_species(species_trait_path: Path, *, species_col: str) -> list[str]:
    species_trait = _with_normalized_species(
        _load_species_trait(species_trait_path), species_col=species_col
    )
    species = species_trait.select("__species").to_series().to_list()
    return [str(value) for value in species]


def _normalize_species_taxid(
    species_taxid_path: Path,
    *,
    species_col: str,
    taxid_col: str,
) -> pl.DataFrame:
    try:
        species_taxid = pl.read_csv(species_taxid_path, separator="\t")
    except FileNotFoundError as exc:
        raise MetadataError(f"Input file not found: {species_taxid_path}") from exc
    except Exception as exc:
        raise MetadataError(f"Failed to read species taxid TSV: {species_taxid_path}") from exc

    species_taxid = _with_normalized_species(
        species_taxid,
        species_col=species_col,
        context="species taxid TSV",
    )
    _require_columns(species_taxid, [taxid_col], "species taxid TSV")
    normalized = species_taxid.with_columns(
        pl.col(taxid_col).cast(pl.String, strict=False).str.strip_chars().alias("__taxid_raw")
    )
    invalid_taxid = normalized.filter(
        pl.col("__taxid_raw").is_null()
        | (pl.col("__taxid_raw") == "")
        | pl.col("__taxid_raw").cast(pl.Int64, strict=False).is_null()
    )
    if invalid_taxid.height > 0:
        offenders = (
            invalid_taxid.select("__taxid_raw")
            .unique()
            .sort("__taxid_raw")
            .to_series()
            .to_list()
        )
        preview = ", ".join("<empty>" if value is None else str(value) for value in offenders[:10])
        raise MetadataError(
            "Species taxid TSV taxid column must contain non-empty integer values; "
            f"offending values: {preview}"
        )
    return normalized.select(
        pl.col("__species").alias("leaf_name"),
        pl.col("__taxid_raw").cast(pl.Int64).alias("taxid"),
    )


def _format_process_output(stdout: str | None, stderr: str | None) -> str:
    parts: list[str] = []
    if stderr:
        parts.append(stderr.strip())
    if stdout:
        parts.append(stdout.strip())
    return "\n".join(part for part in parts if part)


def _run_nwkit(command: Sequence[str], *, action: str) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MetadataError(
            "Failed to run nwkit. Install nwkit or pass --nwkit-bin with the executable path."
        ) from exc
    except OSError as exc:
        raise MetadataError(f"Failed to run nwkit: {exc}") from exc

    if completed.returncode != 0:
        details = _format_process_output(completed.stdout, completed.stderr)
        message = f"nwkit {action} failed with exit code {completed.returncode}"
        if details:
            message += f":\n{details}"
        raise MetadataError(message)
    return completed


def fetch_ncbi_tree(
    species_trait_path: Path,
    tree_out: Path,
    *,
    species_taxid_path: Path | None = None,
    species_col: str = "species",
    taxid_col: str = "taxid",
    rank: str = "family",
    nwkit_bin: str = "nwkit",
    overwrite: bool = False,
) -> NCBITreeResult:
    """Generate an NCBI-taxonomy constrained Newick tree with ``nwkit constrain``."""
    resolved_rank = rank.strip()
    if not resolved_rank:
        raise MetadataError("NCBI taxonomy rank must be non-empty")
    if not nwkit_bin.strip():
        raise MetadataError("nwkit executable path must be non-empty")
    if tree_out.exists() and not overwrite:
        raise MetadataError(f"Output tree already exists; use --force to overwrite: {tree_out}")

    tree_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phenoradar_metadata_",
        dir=tree_out.parent,
    ) as temp_dir:
        temp_dir_path = Path(temp_dir)
        species_list_path = temp_dir_path / "species_list.txt"
        taxid_tsv_path = temp_dir_path / "species_taxid.tsv"
        temp_tree_out = Path(temp_dir) / "ncbi_tree.nwk"
        command_parts: list[str] = [
            nwkit_bin,
            "constrain",
            "--backbone",
            "ncbi",
        ]
        if species_taxid_path is None:
            species = _load_species(species_trait_path, species_col=species_col)
            if not species:
                raise MetadataError("Species trait TSV contains no species")
            species_count = len(species)
            species_list_path.write_text("\n".join(species) + "\n", encoding="utf-8")
            command_parts.extend(["--species_list", str(species_list_path)])
        else:
            species_taxid = _normalize_species_taxid(
                species_taxid_path,
                species_col=species_col,
                taxid_col=taxid_col,
            )
            if species_taxid.height == 0:
                raise MetadataError("Species taxid TSV contains no species")
            species_count = species_taxid.height
            species_taxid.write_csv(taxid_tsv_path, separator="\t")
            command_parts.extend(["--taxid_tsv", str(taxid_tsv_path)])
        command_parts.extend(
            [
                "--rank",
                resolved_rank,
                "--outfile",
                str(temp_tree_out),
            ]
        )
        command: Sequence[str] = tuple(command_parts)
        completed = _run_nwkit(command, action="constrain")
        if not temp_tree_out.exists():
            details = _format_process_output(completed.stdout, completed.stderr)
            message = f"nwkit constrain completed but did not create output tree: {tree_out}"
            if details:
                message += f":\n{details}"
            raise MetadataError(message)
        temp_tree_out.replace(tree_out)

    return NCBITreeResult(tree_path=tree_out, species_count=species_count, rank=resolved_rank)


def _normalize_species_trait_for_metadata(
    species_trait_path: Path,
    *,
    species_col: str,
    trait_col: str,
) -> pl.DataFrame:
    species_trait = _with_normalized_species(
        _load_species_trait(species_trait_path), species_col=species_col
    )
    _require_columns(species_trait, [trait_col], "species trait TSV")
    normalized = species_trait.with_columns(
        pl.col(trait_col).cast(pl.String, strict=False).str.strip_chars().alias("__trait_raw")
    )
    invalid_trait_values = (
        normalized.filter(
            pl.col("__trait_raw").is_not_null()
            & (pl.col("__trait_raw") != "")
            & ~pl.col("__trait_raw").is_in(["0", "1"])
        )
        .select("__trait_raw")
        .unique()
        .sort("__trait_raw")
        .to_series()
        .to_list()
    )
    if invalid_trait_values:
        invalid = ", ".join(str(v) for v in invalid_trait_values)
        raise MetadataError(
            "Trait column must contain only 0/1 or null/empty values; "
            f"offending values: {invalid}"
        )
    return normalized.with_columns(
        pl.when(pl.col("__trait_raw").is_null() | (pl.col("__trait_raw") == ""))
        .then(None)
        .otherwise(pl.col("__trait_raw").cast(pl.Int8))
        .alias("__trait")
    )


def _skim_groupfile_path(skim_tree_path: Path, suffix: str) -> Path:
    stem = str(skim_tree_path)
    if stem.endswith(".nwk"):
        stem = stem.removesuffix(".nwk")
    return Path(f"{stem}.{suffix}.tsv")


def _read_tree_names_with_nwkit(
    tree_path: Path, *, out_dir: Path, nwkit_bin: str
) -> set[str]:
    table_path = out_dir / "tree_table.tsv"
    command: Sequence[str] = (
        nwkit_bin,
        "nwk2table",
        "--infile",
        str(tree_path),
        "--outfile",
        str(table_path),
    )
    completed = _run_nwkit(command, action="nwk2table")
    if not table_path.exists():
        details = _format_process_output(completed.stdout, completed.stderr)
        message = f"nwkit nwk2table completed but did not create table: {table_path}"
        if details:
            message += f":\n{details}"
        raise MetadataError(message)
    try:
        table = pl.read_csv(table_path, separator="\t")
    except Exception as exc:
        raise MetadataError(f"Failed to read nwkit tree table: {table_path}") from exc
    _require_columns(table, ["name"], "nwkit tree table")
    return set(
        str(value)
        for value in table.select(
            pl.col("name").cast(pl.String, strict=False).str.strip_chars()
        )
        .filter(pl.col("name").is_not_null() & (pl.col("name") != ""))
        .to_series()
        .to_list()
    )


def build_species_metadata_from_skim(
    species_trait_path: Path,
    tree_path: Path,
    out: Path,
    *,
    species_col: str = "species",
    trait_col: str = "C4",
    group_col: str = "contrast_pair_id",
    nwkit_bin: str = "nwkit",
    overwrite: bool = False,
) -> SpeciesMetadataResult:
    """Build PhenoRadar species metadata using ``nwkit skim`` contrastive clades."""
    if not nwkit_bin.strip():
        raise MetadataError("nwkit executable path must be non-empty")
    if out.exists() and not overwrite:
        raise MetadataError(f"Output metadata already exists; use --force to overwrite: {out}")
    if not tree_path.exists():
        raise MetadataError(f"Input tree not found: {tree_path}")

    species_trait = _normalize_species_trait_for_metadata(
        species_trait_path,
        species_col=species_col,
        trait_col=trait_col,
    )
    if species_trait.height == 0:
        raise MetadataError("Species trait TSV contains no species")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phenoradar_metadata_",
        dir=out.parent,
    ) as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_trait_path = temp_dir_path / "trait_for_skim.tsv"
        temp_skim_tree = temp_dir_path / "skim.nwk"
        tree_names = _read_tree_names_with_nwkit(
            tree_path, out_dir=temp_dir_path, nwkit_bin=nwkit_bin
        )
        species_for_skim = species_trait.filter(pl.col("__species").is_in(tree_names))
        tree_missing_species_count = species_trait.height - species_for_skim.height
        if species_for_skim.height == 0:
            raise MetadataError(
                "No species in species trait TSV were found in the input tree; "
                "cannot build species metadata"
            )

        species_for_skim.select(
            pl.col("__species").alias("leaf_name"),
            pl.col("__trait").alias(trait_col),
        ).write_csv(temp_trait_path, separator="\t")

        command = (
            nwkit_bin,
            "skim",
            "--infile",
            str(tree_path),
            "--outfile",
            str(temp_skim_tree),
            "--trait",
            str(temp_trait_path),
            "--group-by",
            trait_col,
            "--only-contrastive-clades",
            "yes",
            "--output-groupfile",
            "yes",
        )
        try:
            completed = _run_nwkit(command, action="skim")
        except MetadataError as exc:
            if "No leaves were selected for output" not in str(exc):
                raise
            skim_groups = species_for_skim.select(
                pl.col("__species").alias("leaf_name"),
                pl.lit(None, dtype=pl.String).alias("contrastive_clade"),
            )
        else:
            all_groupfile = _skim_groupfile_path(temp_skim_tree, "all")
            if not all_groupfile.exists():
                details = _format_process_output(completed.stdout, completed.stderr)
                message = (
                    "nwkit skim completed but did not create group table: "
                    f"{all_groupfile}"
                )
                if details:
                    message += f":\n{details}"
                raise MetadataError(message)
            try:
                skim_groups = pl.read_csv(all_groupfile, separator="\t")
            except Exception as exc:
                raise MetadataError(
                    f"Failed to read nwkit skim group table: {all_groupfile}"
                ) from exc

    _require_columns(skim_groups, ["leaf_name", "contrastive_clade"], "nwkit skim group table")
    group_assignments = skim_groups.select(
        pl.col("leaf_name").cast(pl.String, strict=False).str.strip_chars().alias("__species"),
        pl.col("contrastive_clade")
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .alias("__contrast_pair"),
    )
    metadata = (
        species_for_skim.select(
            pl.col("__species"),
            pl.col("__species").alias(species_col),
            pl.col("__trait").alias(trait_col),
        )
        .join(group_assignments, on="__species", how="left")
        .with_columns(
            pl.when(
                pl.col(trait_col).is_null()
                | pl.col("__contrast_pair").is_null()
                | (pl.col("__contrast_pair") == "")
            )
            .then(None)
            .otherwise(pl.col("__contrast_pair"))
            .alias(group_col)
        )
        .select([species_col, trait_col, group_col])
    )

    grouped = metadata.filter(pl.col(group_col).is_not_null() & (pl.col(group_col) != ""))
    invalid_groups = (
        grouped.group_by(group_col)
        .agg(pl.col(trait_col).n_unique().alias("n_labels"))
        .filter(pl.col("n_labels") < 2)
        .select(group_col)
        .to_series()
        .to_list()
    )
    if invalid_groups:
        groups = ", ".join(str(v) for v in sorted(invalid_groups)[:10])
        raise MetadataError(f"Generated contrast groups must contain both 0/1 labels: {groups}")

    metadata.write_csv(out, separator="\t")
    contrast_pair_count = grouped.select(group_col).n_unique() if grouped.height > 0 else 0
    return SpeciesMetadataResult(
        metadata_path=out,
        species_count=metadata.height,
        grouped_species_count=grouped.height,
        contrast_pair_count=contrast_pair_count,
        tree_missing_species_count=tree_missing_species_count,
    )
