"""Metadata preparation helpers."""

from __future__ import annotations

import importlib
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Protocol, cast

import polars as pl


class MetadataError(ValueError):
    """Raised when metadata preparation fails."""


@dataclass(frozen=True)
class NCBITreeResult:
    """Artifacts produced by NCBI taxonomy tree retrieval."""

    tree_path: Path
    species_count: int


@dataclass(frozen=True)
class SpeciesTaxidResult:
    """Artifacts produced by species-to-NCBI-taxid resolution."""

    taxid_path: Path
    species_count: int
    resolved_species_count: int
    unresolved_species_count: int


@dataclass(frozen=True)
class SpeciesMetadataResult:
    """Artifacts produced by species metadata generation."""

    metadata_path: Path
    species_count: int
    grouped_species_count: int
    contrast_pair_count: int
    contrast_pair_test_holdout_count: int
    taxon_block_counts: dict[str, int]
    taxon_block_test_holdout_counts: dict[str, int]
    taxon_block_exclude_counts: dict[str, int]
    tree_missing_species_count: int
    species_taxid_path: Path | None = None
    species_taxid_generated: bool = False
    species_taxid_resolved_count: int = 0
    species_taxid_unresolved_count: int = 0


_NWKIT_DEFAULT_SPECIES_PATTERN = re.compile(r"^([^_]+_[^_]+)(?:_|$)")


class _Closable(Protocol):
    def close(self) -> None: ...


class _NCBITaxa(Protocol):
    db: _Closable | None

    def get_name_translator(
        self, names: Sequence[str]
    ) -> Mapping[str, Sequence[int | str]]: ...

    def get_lineage(self, taxid: int) -> Sequence[int | str]: ...

    def get_rank(self, taxids: Sequence[int]) -> Mapping[int | str, str]: ...

    def get_taxid_translator(self, taxids: Sequence[int]) -> Mapping[int | str, str]: ...


class _NCBITaxaFactory(Protocol):
    def __call__(self, *, dbfile: str | None = None) -> _NCBITaxa: ...


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


def _taxonomy_query_from_label(label: str) -> str | None:
    normalized = re.sub(r"\s+", "_", str(label).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return None
    match = _NWKIT_DEFAULT_SPECIES_PATTERN.search(normalized)
    query = match.group(1).replace("_", " ") if match is not None else normalized.replace("_", " ")
    query = re.sub(r"\s+", " ", query).strip()
    return query or None


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
            invalid_taxid.select("__taxid_raw").unique().sort("__taxid_raw").to_series().to_list()
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
    nwkit_bin: str = "nwkit",
    overwrite: bool = False,
) -> NCBITreeResult:
    """Generate an NCBI-taxonomy constrained Newick tree with ``nwkit constrain``."""
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

    return NCBITreeResult(tree_path=tree_out, species_count=species_count)


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
            f"Trait column must contain only 0/1 or null/empty values; offending values: {invalid}"
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


def _validate_output_column_names(column_names: Sequence[str]) -> None:
    duplicates = sorted({name for name in column_names if column_names.count(name) > 1})
    if duplicates:
        duplicate_str = ", ".join(duplicates)
        raise MetadataError(f"Output metadata column names must be unique: {duplicate_str}")


def _normalize_taxon_block_ranks(ranks: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    for rank in ranks or []:
        value = rank.strip().lower()
        if not value:
            raise MetadataError("Taxon block ranks must be non-empty")
        if not value.replace("_", "").isalnum():
            raise MetadataError(
                "Taxon block ranks must contain only letters, numbers, or underscores; "
                f"offending value: {rank}"
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


def _taxon_block_columns(rank: str) -> tuple[str, str, str, str]:
    prefix = f"taxon_{rank}"
    return (
        f"{prefix}_id",
        f"{prefix}_name",
        f"{prefix}_test_holdout",
        f"{prefix}_exclude",
    )


def _load_ncbi_taxa(ncbi_taxonomy_db: Path | None = None) -> _NCBITaxa:
    try:
        ete4_module = importlib.import_module("ete4")
    except ImportError as exc:
        raise MetadataError(
            "Taxonomic rank blocking requires ete4, which is installed with phenoradar. "
            "Reinstall phenoradar or install ete4 manually."
        ) from exc
    try:
        ncbi_taxa_factory = cast(_NCBITaxaFactory, ete4_module.NCBITaxa)
        dbfile = None if ncbi_taxonomy_db is None else str(ncbi_taxonomy_db)
        return ncbi_taxa_factory(dbfile=dbfile)
    except Exception as exc:
        raise MetadataError("Failed to initialize ete4 NCBITaxa") from exc


def _close_ncbi_taxa(ncbi_taxa: _NCBITaxa) -> None:
    db = cast(_Closable | None, getattr(ncbi_taxa, "db", None))
    if db is None:
        return
    try:
        db.close()
    except Exception:
        return


def _lookup_ncbi_taxid(taxonomy_query: str, ncbi_taxa: _NCBITaxa) -> int | None:
    def _name_translator_value(query: str) -> list[int | str]:
        try:
            name_map = ncbi_taxa.get_name_translator([query])
        except Exception:
            return []
        values = name_map.get(query)
        if values is None:
            return []
        return list(values)

    candidates = _name_translator_value(taxonomy_query)
    if not candidates:
        genus = taxonomy_query.split(" ", 1)[0].strip()
        if genus and genus != taxonomy_query:
            candidates = _name_translator_value(genus)
    if not candidates:
        return None
    try:
        return int(candidates[0])
    except (TypeError, ValueError):
        return None


def _build_species_taxid_frame(
    species_for_taxid: pl.DataFrame,
    *,
    ncbi_taxonomy_db: Path | None,
) -> tuple[pl.DataFrame, int]:
    ncbi_taxa = _load_ncbi_taxa(ncbi_taxonomy_db)
    rows: list[dict[str, object]] = []
    unresolved_count = 0
    try:
        for species_value in species_for_taxid.select("__species").to_series().to_list():
            species = str(species_value)
            taxonomy_query = _taxonomy_query_from_label(species)
            taxid = (
                None if taxonomy_query is None else _lookup_ncbi_taxid(taxonomy_query, ncbi_taxa)
            )
            if taxid is None:
                unresolved_count += 1
                continue
            rows.append({"leaf_name": species, "taxid": taxid})
    finally:
        _close_ncbi_taxa(ncbi_taxa)

    schema = {"leaf_name": pl.String, "taxid": pl.Int64}
    if not rows:
        return pl.DataFrame(schema=schema), unresolved_count
    return pl.DataFrame(rows, schema=schema).sort("leaf_name"), unresolved_count


def _write_species_taxid_from_frame(
    species_for_taxid: pl.DataFrame,
    out: Path,
    *,
    species_col: str,
    taxid_col: str,
    ncbi_taxonomy_db: Path | None,
    overwrite: bool,
) -> SpeciesTaxidResult:
    if out.exists() and not overwrite:
        raise MetadataError(
            f"Output species taxid TSV already exists; use --force to overwrite: {out}"
        )
    species_count = species_for_taxid.height
    species_taxid, unresolved_count = _build_species_taxid_frame(
        species_for_taxid,
        ncbi_taxonomy_db=ncbi_taxonomy_db,
    )
    if species_taxid.height == 0:
        raise MetadataError("No species could be resolved to NCBI taxids")
    out.parent.mkdir(parents=True, exist_ok=True)
    rename_columns = {}
    if species_col != "leaf_name":
        rename_columns["leaf_name"] = species_col
    if taxid_col != "taxid":
        rename_columns["taxid"] = taxid_col
    species_taxid_out = species_taxid.rename(rename_columns) if rename_columns else species_taxid
    species_taxid_out.write_csv(out, separator="\t")
    return SpeciesTaxidResult(
        taxid_path=out,
        species_count=species_count,
        resolved_species_count=species_taxid.height,
        unresolved_species_count=unresolved_count,
    )


def build_species_taxid_tsv(
    species_trait_path: Path,
    out: Path,
    *,
    species_col: str = "species",
    taxid_col: str = "taxid",
    ncbi_taxonomy_db: Path | None = None,
    overwrite: bool = False,
) -> SpeciesTaxidResult:
    """Resolve species labels to NCBI taxids and write a species_taxid TSV."""
    species_trait = _with_normalized_species(
        _load_species_trait(species_trait_path), species_col=species_col
    )
    if species_trait.height == 0:
        raise MetadataError("Species trait TSV contains no species")
    return _write_species_taxid_from_frame(
        species_trait,
        out,
        species_col=species_col,
        taxid_col=taxid_col,
        ncbi_taxonomy_db=ncbi_taxonomy_db,
        overwrite=overwrite,
    )


def _dict_get_any_key(mapping: Mapping[int | str, str], key: int) -> str | None:
    return mapping.get(key, mapping.get(str(key)))


def _rank_taxon_lookup(
    taxids: Sequence[int],
    ranks: Sequence[str],
    ncbi_taxa: _NCBITaxa,
) -> dict[int, dict[str, tuple[str, str]]]:
    lookup: dict[int, dict[str, tuple[str, str]]] = {}
    rank_taxids: set[int] = set()
    lineages_by_taxid: dict[int, list[int]] = {}
    for taxid in sorted(set(int(value) for value in taxids)):
        try:
            lineage = [int(value) for value in ncbi_taxa.get_lineage(taxid)]
        except Exception:
            lookup[taxid] = {}
            continue
        lineages_by_taxid[taxid] = lineage
        try:
            rank_map = dict(ncbi_taxa.get_rank(lineage))
        except Exception:
            lookup[taxid] = {}
            continue
        by_rank: dict[str, tuple[str, str]] = {}
        for lineage_taxid in lineage:
            lineage_rank = _dict_get_any_key(rank_map, lineage_taxid)
            if lineage_rank is None:
                continue
            lineage_rank_str = str(lineage_rank).strip().lower()
            if lineage_rank_str in ranks and lineage_rank_str not in by_rank:
                lineage_rank_taxid = int(lineage_taxid)
                by_rank[lineage_rank_str] = (str(lineage_rank_taxid), "")
                rank_taxids.add(lineage_rank_taxid)
        lookup[taxid] = by_rank

    try:
        names = dict(ncbi_taxa.get_taxid_translator(sorted(rank_taxids)))
    except Exception:
        names = {}
    for by_rank in lookup.values():
        for rank, (rank_taxid_text, _name) in list(by_rank.items()):
            translated = _dict_get_any_key(names, int(rank_taxid_text))
            by_rank[rank] = (
                rank_taxid_text,
                "" if translated is None else translated,
            )
    return lookup


def _taxon_block_holdout_groups(
    group_ids: list[str],
    *,
    fraction: float,
    seed: int,
) -> set[str]:
    if fraction <= 0.0 or len(group_ids) <= 1:
        return set()
    holdout_count = max(1, int(round(len(group_ids) * fraction)))
    holdout_count = min(holdout_count, len(group_ids) - 1)
    shuffled = sorted(group_ids)
    Random(seed).shuffle(shuffled)
    return set(shuffled[:holdout_count])


def _build_taxon_block_metadata(
    species_for_metadata: pl.DataFrame,
    species_taxid_path: Path,
    *,
    species_col: str,
    taxid_col: str,
    ranks: Sequence[str],
    min_species_per_label: int,
    mixed_test_fraction: float,
    mixed_test_seed: int,
    ncbi_taxonomy_db: Path | None,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, int], dict[str, int]]:
    if min_species_per_label < 1:
        raise MetadataError("taxon_block_min_species_per_label must be >= 1")
    if mixed_test_fraction < 0.0 or mixed_test_fraction >= 1.0:
        raise MetadataError("taxon_block_mixed_test_fraction must be >= 0 and < 1")

    species_taxid = _normalize_species_taxid(
        species_taxid_path,
        species_col=species_col,
        taxid_col=taxid_col,
    ).rename({"leaf_name": "__species"})
    metadata = species_for_metadata.select("__species", "__trait").join(
        species_taxid, on="__species", how="left"
    )
    taxids = [
        int(value) for value in metadata.select("taxid").drop_nulls().unique().to_series().to_list()
    ]
    ncbi_taxa = _load_ncbi_taxa(ncbi_taxonomy_db)
    rank_lookup = _rank_taxon_lookup(taxids, ranks, ncbi_taxa)

    rows: list[dict[str, object]] = []
    for row in metadata.to_dicts():
        species = str(row["__species"])
        trait = row["__trait"]
        taxid = row["taxid"]
        payload: dict[str, object] = {"__species": species, "__trait": trait}
        rank_values = {} if taxid is None else rank_lookup.get(int(taxid), {})
        for rank in ranks:
            raw_id_col = f"__taxon_{rank}_raw_id"
            raw_name_col = f"__taxon_{rank}_raw_name"
            rank_taxon = rank_values.get(rank)
            if rank_taxon is None:
                payload[raw_id_col] = None
                payload[raw_name_col] = None
            else:
                payload[raw_id_col] = rank_taxon[0]
                payload[raw_name_col] = rank_taxon[1]
        rows.append(payload)

    rank_frame = pl.DataFrame(rows)
    output = rank_frame.select("__species")
    block_counts: dict[str, int] = {}
    holdout_counts: dict[str, int] = {}
    exclude_counts: dict[str, int] = {}

    for rank in ranks:
        raw_id_col = f"__taxon_{rank}_raw_id"
        raw_name_col = f"__taxon_{rank}_raw_name"
        out_id_col, out_name_col, holdout_col, exclude_col = _taxon_block_columns(rank)
        label_counts = (
            rank_frame.filter(pl.col("__trait").is_not_null() & pl.col(raw_id_col).is_not_null())
            .group_by(raw_id_col)
            .agg(
                (pl.col("__trait") == 0).sum().alias("n_label_0"),
                (pl.col("__trait") == 1).sum().alias("n_label_1"),
            )
        )
        valid_group_ids = [
            str(row[raw_id_col])
            for row in label_counts.filter(
                (pl.col("n_label_0") >= min_species_per_label)
                & (pl.col("n_label_1") >= min_species_per_label)
            ).to_dicts()
        ]
        valid_groups = set(valid_group_ids)
        mixed_holdout_groups = _taxon_block_holdout_groups(
            valid_group_ids,
            fraction=mixed_test_fraction,
            seed=mixed_test_seed,
        )
        kept_groups = valid_groups - mixed_holdout_groups

        rank_output = rank_frame.with_columns(
            (
                pl.col("__trait").is_not_null()
                & pl.col(raw_id_col).is_not_null()
                & pl.col(raw_id_col).is_in(sorted(kept_groups))
            ).alias("__taxon_keep"),
            (
                pl.col("__trait").is_not_null()
                & pl.col(raw_id_col).is_not_null()
                & pl.col(raw_id_col).is_in(sorted(valid_groups))
                & pl.col(raw_id_col).is_in(sorted(mixed_holdout_groups))
            ).alias("__taxon_mixed_holdout"),
            (pl.col("__trait").is_not_null() & pl.col(raw_id_col).is_null()).alias(
                "__taxon_exclude"
            ),
        ).with_columns(
            pl.when(pl.col("__taxon_keep"))
            .then(pl.col(raw_id_col))
            .otherwise(None)
            .alias(out_id_col),
            pl.when(pl.col("__taxon_keep"))
            .then(pl.col(raw_name_col))
            .otherwise(None)
            .alias(out_name_col),
            pl.when(
                pl.col("__trait").is_not_null()
                & ~pl.col("__taxon_keep")
                & ~pl.col("__taxon_exclude")
            )
            .then(pl.lit("yes"))
            .otherwise(pl.lit("no"))
            .alias(holdout_col),
            pl.when(pl.col("__taxon_exclude"))
            .then(pl.lit("yes"))
            .otherwise(pl.lit("no"))
            .alias(exclude_col),
        )
        output = output.join(
            rank_output.select(["__species", out_id_col, out_name_col, holdout_col, exclude_col]),
            on="__species",
            how="left",
        )
        block_counts[rank] = len(kept_groups)
        holdout_counts[rank] = rank_output.filter(pl.col(holdout_col) == "yes").height
        exclude_counts[rank] = rank_output.filter(pl.col(exclude_col) == "yes").height

    return output, block_counts, holdout_counts, exclude_counts


def _read_tree_names_with_nwkit(tree_path: Path, *, out_dir: Path, nwkit_bin: str) -> set[str]:
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
        for value in table.select(pl.col("name").cast(pl.String, strict=False).str.strip_chars())
        .filter(pl.col("name").is_not_null() & (pl.col("name") != ""))
        .to_series()
        .to_list()
    )


def build_species_metadata_from_skim(
    species_trait_path: Path,
    tree_path: Path,
    out: Path,
    *,
    species_taxid_path: Path | None = None,
    species_taxid_out_path: Path | None = None,
    species_col: str = "species",
    taxid_col: str = "taxid",
    trait_col: str = "C4",
    contrast_pair_col: str = "contrast_pair_id",
    contrast_pair_test_holdout_col: str = "contrast_pair_test_holdout",
    taxon_block_ranks: Sequence[str] | None = None,
    taxon_block_min_species_per_label: int = 1,
    taxon_block_mixed_test_fraction: float = 0.0,
    taxon_block_mixed_test_seed: int = 42,
    ncbi_taxonomy_db: Path | None = None,
    nwkit_bin: str = "nwkit",
    overwrite: bool = False,
) -> SpeciesMetadataResult:
    """Build PhenoRadar species metadata using ``nwkit skim`` contrastive clades."""
    if not nwkit_bin.strip():
        raise MetadataError("nwkit executable path must be non-empty")
    if out.exists() and not overwrite:
        raise MetadataError(f"Output metadata already exists; use --force to overwrite: {out}")
    if species_taxid_path is not None and species_taxid_out_path is not None:
        raise MetadataError("species_taxid_out_path cannot be used with species_taxid_path")
    if not tree_path.exists():
        raise MetadataError(f"Input tree not found: {tree_path}")
    taxon_ranks = _normalize_taxon_block_ranks(taxon_block_ranks)
    output_columns = [species_col, trait_col, contrast_pair_col, contrast_pair_test_holdout_col]
    for rank in taxon_ranks:
        output_columns.extend(_taxon_block_columns(rank))
    _validate_output_column_names(output_columns)

    species_trait = _normalize_species_trait_for_metadata(
        species_trait_path,
        species_col=species_col,
        trait_col=trait_col,
    )
    if species_trait.height == 0:
        raise MetadataError("Species trait TSV contains no species")

    resolved_species_taxid_path = species_taxid_path
    generated_species_taxid_result: SpeciesTaxidResult | None = None
    if resolved_species_taxid_path is None and (taxon_ranks or species_taxid_out_path is not None):
        generated_species_taxid_result = _write_species_taxid_from_frame(
            species_trait,
            species_taxid_out_path or out.with_name("species_taxid.tsv"),
            species_col=species_col,
            taxid_col=taxid_col,
            ncbi_taxonomy_db=ncbi_taxonomy_db,
            overwrite=overwrite,
        )
        resolved_species_taxid_path = generated_species_taxid_result.taxid_path

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
                message = f"nwkit skim completed but did not create group table: {all_groupfile}"
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
            .alias(contrast_pair_col),
            pl.when(
                pl.col(trait_col).is_not_null()
                & (pl.col("__contrast_pair").is_null() | (pl.col("__contrast_pair") == ""))
            )
            .then(pl.lit("yes"))
            .otherwise(pl.lit("no"))
            .alias(contrast_pair_test_holdout_col),
        )
    )
    taxon_block_counts: dict[str, int] = {}
    taxon_block_test_holdout_counts: dict[str, int] = {}
    taxon_block_exclude_counts: dict[str, int] = {}
    if taxon_ranks:
        if resolved_species_taxid_path is None:
            raise MetadataError("Taxonomic rank blocking requires species_taxid_path")
        (
            taxon_blocks,
            taxon_block_counts,
            taxon_block_test_holdout_counts,
            taxon_block_exclude_counts,
        ) = _build_taxon_block_metadata(
            species_for_skim,
            resolved_species_taxid_path,
            species_col=species_col,
            taxid_col=taxid_col,
            ranks=taxon_ranks,
            min_species_per_label=taxon_block_min_species_per_label,
            mixed_test_fraction=taxon_block_mixed_test_fraction,
            mixed_test_seed=taxon_block_mixed_test_seed,
            ncbi_taxonomy_db=ncbi_taxonomy_db,
        )
        metadata = metadata.join(taxon_blocks, on="__species", how="left")
    metadata = metadata.select(output_columns)

    grouped = metadata.filter(
        pl.col(contrast_pair_col).is_not_null() & (pl.col(contrast_pair_col) != "")
    )
    invalid_groups = (
        grouped.group_by(contrast_pair_col)
        .agg(pl.col(trait_col).n_unique().alias("n_labels"))
        .filter(pl.col("n_labels") < 2)
        .select(contrast_pair_col)
        .to_series()
        .to_list()
    )
    if invalid_groups:
        groups = ", ".join(str(v) for v in sorted(invalid_groups)[:10])
        raise MetadataError(f"Generated contrast groups must contain both 0/1 labels: {groups}")

    metadata.write_csv(out, separator="\t")
    contrast_pair_count = grouped.select(contrast_pair_col).n_unique() if grouped.height > 0 else 0
    contrast_pair_test_holdout_count = metadata.filter(
        pl.col(contrast_pair_test_holdout_col) == "yes"
    ).height
    return SpeciesMetadataResult(
        metadata_path=out,
        species_count=metadata.height,
        grouped_species_count=grouped.height,
        contrast_pair_count=contrast_pair_count,
        contrast_pair_test_holdout_count=contrast_pair_test_holdout_count,
        taxon_block_counts=taxon_block_counts,
        taxon_block_test_holdout_counts=taxon_block_test_holdout_counts,
        taxon_block_exclude_counts=taxon_block_exclude_counts,
        tree_missing_species_count=tree_missing_species_count,
        species_taxid_path=resolved_species_taxid_path,
        species_taxid_generated=generated_species_taxid_result is not None,
        species_taxid_resolved_count=0
        if generated_species_taxid_result is None
        else generated_species_taxid_result.resolved_species_count,
        species_taxid_unresolved_count=0
        if generated_species_taxid_result is None
        else generated_species_taxid_result.unresolved_species_count,
    )
