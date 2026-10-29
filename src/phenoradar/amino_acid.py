"""Alignment input and train-fitted, site-addressable amino-acid features."""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import polars as pl

from phenoradar.config.schema import AminoAcidConfig, AminoAcidSiteFilter
from phenoradar.provenance import file_identity
from phenoradar.sequence_tpm_cache import SequenceTPMCacheError, read_sequence_tpm_cache

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SITE_BLOCK_SIZE = 512
PREDICTION_BATCH_SIZE = 128
MAX_MATRIX_CELLS = 50_000_000
_CODE = np.full(256, 255, dtype=np.uint8)
_CODE[list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ*")] = 20
_CODE[list(b"-.")] = 21
_CODE[list(ALPHABET.encode())] = np.arange(20)


class AlignmentError(ValueError):
    """Invalid alignment, correspondence, or sequence preprocessing state."""


@dataclass
class AlignmentOG:
    orthogroup: str
    source: str
    sha256: str
    width: int
    cache: str
    mask_cache: str
    sequence_ids: list[str]
    sequence_hashes: list[str]
    species_rows: dict[str, list[int]]
    mask_rows: dict[str, int]
    tpm: np.ndarray | None = None
    tpm_mask_cache: str | None = None

    def matrix(self) -> np.ndarray:
        if not self.sequence_ids:
            return np.empty((0, self.width), dtype=np.uint8)
        return np.memmap(
            self.cache, mode="r", dtype=np.uint8, shape=(len(self.sequence_ids), self.width)
        )


class AlignmentStore:
    """Read each FASTA once; keep byte matrices on disk and packed presence masks.

    Only requested cache species are retained. Resolving IDs uses the complete
    metadata species list, independently of training labels and CV membership.
    """

    def __init__(
        self,
        directory: Path,
        species: list[str],
        *,
        cache_species: list[str] | None = None,
        only_orthogroups: set[str] | None = None,
        sequence_tpm_path: str | None = None,
        sequence_tpm_cache_dir: Path | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._temp = TemporaryDirectory(prefix="phenoradar_aa_")
        self.ogs: dict[str, AlignmentOG] = {}
        self.species = set(species)
        self.retained_species = self.species if cache_species is None else set(cache_species)
        self.tpm_identity: dict[str, Any] | None = None
        try:
            paths = sorted(
                p
                for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in {".faa", ".fa", ".fasta"}
            )
            for path in paths:
                if only_orthogroups is not None and path.stem not in only_orthogroups:
                    continue
                if path.stem in self.ogs:
                    raise AlignmentError(f"Duplicate orthogroup filename stem: {path.stem}")
                self.ogs[path.stem] = self._read(path, len(self.ogs))
            if not self.ogs:
                raise AlignmentError(f"No alignment FASTA files found in {directory}")
            if only_orthogroups is not None and only_orthogroups - self.ogs.keys():
                raise AlignmentError("Prediction input is missing required orthogroup alignments")
            if sequence_tpm_path is not None:
                self._load_tpm(Path(sequence_tpm_path), sequence_tpm_cache_dir, progress)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._temp.cleanup()

    def __enter__(self) -> AlignmentStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _resolve(self, og: str, sequence_id: str) -> str:
        candidates = []
        candidate = sequence_id
        while candidate:
            if candidate in self.species:
                candidates.append(candidate)
            candidate = candidate.rsplit("_", 1)[0] if "_" in candidate else ""
        if len(candidates) != 1:
            raise AlignmentError(
                f"Cannot uniquely map {og}/{sequence_id} to metadata species; "
                "FASTA IDs must match a metadata species ID or that ID followed by '_' "
                "and a gene suffix; resolve unmatched or ambiguous IDs"
            )
        return candidates[0]

    def _read(self, path: Path, index: int) -> AlignmentOG:
        cache = Path(self._temp.name) / f"{index}.bin"
        mask_cache = Path(self._temp.name) / f"{index}_presence.npy"
        ids: list[str] = []
        hashes: list[str] = []
        rows: dict[str, list[int]] = defaultdict(list)
        masks: dict[str, np.ndarray] = {}
        seen: set[str] = set()
        digest = hashlib.sha256()
        before = path.stat()
        width: int | None = None
        header: str | None = None
        parts: list[bytes] = []

        def consume(handle: Any) -> None:
            nonlocal width
            if header is None:
                return
            sequence_id = header.split()[0]
            if sequence_id in seen:
                raise AlignmentError(f"Duplicate FASTA ID in {path}: {sequence_id}")
            seen.add(sequence_id)
            raw = b"".join(parts).upper()
            if not raw:
                raise AlignmentError(f"Empty sequence in {path}: {sequence_id}")
            if width is None:
                width = len(raw)
            if len(raw) != width:
                raise AlignmentError(f"Unequal alignment lengths in {path}: {sequence_id}")
            values = np.frombuffer(raw, dtype=np.uint8)
            code = _CODE[values]
            if np.any(code == 255):
                raise AlignmentError(f"Invalid sequence character in {path}: {sequence_id}")
            species = self._resolve(path.stem, sequence_id)
            if species not in self.retained_species:
                return
            row = len(ids)
            rows[species].append(row)
            ids.append(sequence_id)
            hashes.append(hashlib.sha256(values[code != 21].tobytes()).hexdigest())
            handle.write(raw)
            present = np.packbits(code < 20)
            if species in masks:
                masks[species] |= present
            else:
                masks[species] = present

        with path.open("rb") as source, cache.open("wb") as target:
            for line in source:
                digest.update(line)
                if line.startswith(b">"):
                    consume(target)
                    try:
                        header = line[1:].decode("utf-8").strip()
                    except UnicodeDecodeError as exc:
                        raise AlignmentError(f"Invalid FASTA header in {path}") from exc
                    if not header:
                        raise AlignmentError(f"Empty FASTA header in {path}")
                    parts = []
                elif line.strip():
                    if header is None:
                        raise AlignmentError(f"Sequence before FASTA header in {path}")
                    parts.append(line.strip())
            consume(target)
        if width is None:
            raise AlignmentError(f"Empty alignment: {path}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise AlignmentError(f"Alignment changed during import: {path}")
        order = sorted(masks)
        packed = (
            np.stack([masks[s] for s in order])
            if order
            else np.empty((0, (width + 7) // 8), dtype=np.uint8)
        )
        np.save(mask_cache, packed, allow_pickle=False)
        return AlignmentOG(
            path.stem,
            str(path.resolve()),
            digest.hexdigest(),
            width,
            str(cache),
            str(mask_cache),
            ids,
            hashes,
            dict(rows),
            {s: i for i, s in enumerate(order)},
        )

    def identities(self) -> list[dict[str, Any]]:
        return [
            {
                "orthogroup": og.orthogroup,
                "path": og.source,
                "sha256": og.sha256,
                "alignment_length": og.width,
            }
            for og in self.ogs.values()
        ]

    def _load_tpm(
        self, path: Path, cache_dir: Path | None, progress: Callable[[str], None] | None
    ) -> None:
        """Stream a possibly large table, retaining only cached alignment keys.

        The same species/ID may occur in several OGs. It has one input TPM,
        assigned to all those records; duplicate matching input keys are errors.
        """
        requests: dict[str, dict[str, list[tuple[AlignmentOG, int]]]] = {}
        for og in self.ogs.values():
            og.tpm = np.full(len(og.sequence_ids), np.nan, dtype=np.float64)
            for sample, rows in og.species_rows.items():
                requested = requests.setdefault(sample, {})
                for row in rows:
                    requested.setdefault(og.sequence_ids[row], []).append((og, row))
        before = path.stat()
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        matched = 0
        identity: dict[str, Any] | None = None
        with ExitStack() as stack:
            reader: Any
            if cache_dir is not None:
                keys = pl.DataFrame(
                    [(sample, gene) for sample, genes in requests.items() for gene in genes],
                    schema={"species": pl.String, "sequence_id": pl.String},
                    orient="row",
                )
                try:
                    tpm_rows, identity = read_sequence_tpm_cache(
                        path, keys, cache_dir, progress=progress
                    )
                except (SequenceTPMCacheError, pl.exceptions.PolarsError) as exc:
                    raise AlignmentError(str(exc)) from exc
                reader = iter(tpm_rows)
                header = ["species", "sequence_id", "tpm"]
            else:
                handle = stack.enter_context(opener(path, "rt", encoding="utf-8", newline=""))
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, [])
            required = ["species", "sequence_id", "tpm"]
            if any(header.count(name) != 1 for name in required):
                raise AlignmentError(
                    "Sequence TPM requires unique species, sequence_id, tpm columns"
                )
            si, gi, ti = (header.index(name) for name in required)
            for line, values in enumerate(reader, start=2):
                if not values:
                    continue
                if len(values) != len(header):
                    raise AlignmentError(f"Malformed sequence TPM row {line}")
                targets = requests.get(values[si], {}).get(values[gi])
                if targets is None:
                    continue
                first, row = targets[0]
                assert first.tpm is not None
                if np.isfinite(first.tpm[row]):
                    raise AlignmentError(f"Duplicate sequence TPM key: {values[si]}/{values[gi]}")
                try:
                    value = float(values[ti])
                except (TypeError, ValueError) as exc:
                    raise AlignmentError(
                        f"Invalid sequence TPM for {values[si]}/{values[gi]}"
                    ) from exc
                if not np.isfinite(value) or value < 0:
                    raise AlignmentError(
                        f"Sequence TPM must be finite and nonnegative for {values[si]}/{values[gi]}"
                    )
                for og, row in targets:
                    assert og.tpm is not None
                    og.tpm[row] = value
                matched += 1
        if cache_dir is not None:
            del keys, tpm_rows
        del requests
        missing: list[str] = []
        missing_count = 0
        zero_count = 0
        for og in self.ogs.values():
            assert og.tpm is not None
            absent = np.flatnonzero(np.isnan(og.tpm))
            missing_count += len(absent)
            missing.extend(f"{og.orthogroup}/{og.sequence_ids[r]}" for r in absent[:3])
            missing = missing[:5]
            zero_count += int(np.count_nonzero(og.tpm == 0))
        if missing_count:
            raise AlignmentError(
                f"Missing sequence TPM for {missing_count} cached alignment records "
                f"(not assumed zero): {', '.join(missing)}"
            )
        self.tpm_identity = {
            **(
                identity
                if identity is not None
                else {**file_identity(path), "cache_hit": False, "cache_path": None}
            ),
            "n_matched_keys": matched,
            "n_alignment_records": sum(len(og.sequence_ids) for og in self.ogs.values()),
            "n_zero_tpm_records": zero_count,
        }
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise AlignmentError(f"Sequence TPM changed during import: {path}")
        # Build once; each training fold can cheaply count species with at least
        # one positive-TPM canonical residue without expanding 20-state features.
        for index, og in enumerate(self.ogs.values()):
            assert og.tpm is not None
            matrix = og.matrix()
            packed = np.zeros((len(og.mask_rows), (og.width + 7) // 8), dtype=np.uint8)
            for sample, rows in og.species_rows.items():
                for row in rows:
                    if og.tpm[row] > 0:
                        packed[og.mask_rows[sample]] |= np.packbits(_CODE[matrix[row]] < 20)
            og.tpm_mask_cache = str(Path(self._temp.name) / f"{index}_tpm_presence.npy")
            np.save(og.tpm_mask_cache, packed, allow_pickle=False)

    def call_counts(
        self, og: AlignmentOG, species: list[str], *, weighted: bool = False
    ) -> np.ndarray:
        counts = np.zeros(og.width, dtype=np.int32)
        cache = og.tpm_mask_cache if weighted else og.mask_cache
        if cache is None:
            raise AlignmentError("Weighted AA features require sequence TPM input")
        packed = np.load(cache, mmap_mode="r", allow_pickle=False)
        for sample in species:
            if sample in og.mask_rows:
                counts += np.unpackbits(packed[og.mask_rows[sample]], count=og.width)
        return counts

    @property
    def correspondence(self) -> Any:
        return (
            {"orthogroup": og.orthogroup, "sequence_id": og.sequence_ids[row], "species": sample}
            for og in self.ogs.values()
            for sample, rows in og.species_rows.items()
            for row in rows
        )

    def representatives(
        self, og: AlignmentOG, species: list[str], columns: np.ndarray
    ) -> list[int | None]:
        matrix = og.matrix()
        chosen: list[int | None] = []
        for sample in species:
            best: tuple[int, str, int] | None = None
            for row in og.species_rows.get(sample, []):
                count = int(np.count_nonzero(_CODE[matrix[row, columns]] < 20))
                key = (-count, og.sequence_ids[row], row)
                if best is None or key < best:
                    best = key
            chosen.append(best[2] if best is not None and best[0] < 0 else None)
        return chosen

    def frequencies(
        self,
        og: AlignmentOG,
        species: list[str],
        columns: np.ndarray,
        *,
        representative: bool = False,
        scoring_columns: np.ndarray | None = None,
        chosen_rows: list[int | None] | None = None,
        weighted: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
        if weighted and (og.tpm is None or representative):
            raise AlignmentError(
                "Weighted frequencies require TPM and cannot select a representative"
            )
        if chosen_rows is not None and (not representative or len(chosen_rows) != len(species)):
            raise AlignmentError("Invalid frozen representative rows")
        if representative and chosen_rows is None:
            chosen_rows = self.representatives(
                og, species, scoring_columns if scoring_columns is not None else columns
            )
        values = np.full((len(species), len(columns), 20), np.nan, dtype=np.float32)
        coverage = np.zeros((len(species), len(columns)), dtype=np.float32)
        chosen: list[str | None] = []
        matrix = og.matrix()
        for i, sample in enumerate(species):
            gene_rows = og.species_rows.get(sample, [])
            selected: str | None = None
            if representative:
                assert chosen_rows is not None
                row = chosen_rows[i]
                gene_rows = [] if row is None else [row]
                selected = None if row is None else og.sequence_ids[row]
            chosen.append(selected)
            if not gene_rows:
                continue
            weights = np.ones(len(gene_rows), dtype=np.float64)
            if weighted:
                assert og.tpm is not None
                weights = og.tpm[gene_rows].copy()
                maximum = float(weights.max())
                if maximum == 0:
                    continue
                weights /= maximum  # Stable even for very large, finite TPMs.
            counts = np.zeros((len(columns), 20), dtype=np.float64 if weighted else np.int32)
            for start in range(0, len(gene_rows), 256):
                codes = _CODE[matrix[np.ix_(np.asarray(gene_rows[start : start + 256]), columns)]]
                for a in range(20):
                    if weighted:
                        counts[:, a] += np.sum(
                            (codes == a) * weights[start : start + 256, None], axis=0
                        )
                    else:
                        counts[:, a] += np.sum(codes == a, axis=0)
            total = counts.sum(axis=1)
            observed = total > 0
            values[i, observed] = counts[observed] / total[observed, None]
            coverage[i] = total / weights.sum()
        return values, coverage, chosen


@dataclass
class AASite:
    orthogroup: str
    column: int
    sample_fraction: float
    minor_samples: int
    minor_groups: int
    score: float
    mean: np.ndarray = field(default_factory=lambda: np.zeros(20))
    known: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=bool))
    active: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=bool))
    post_aggregation_sample_fraction: float = 0.0

    def feature(self, residue: int) -> str:
        return json.dumps(
            [self.orthogroup, self.column + 1, ALPHABET[residue]], separators=(",", ":")
        )


@dataclass
class AATransform:
    settings: AminoAcidConfig
    sites: list[AASite]
    qc: list[dict[str, Any]]
    # None preserves the retained-site rule when loading pre-aa-3 bundles.
    representative_columns: dict[str, np.ndarray] | None = None

    def scoring_columns(self, orthogroup: str) -> np.ndarray:
        if self.representative_columns is not None:
            return self.representative_columns[orthogroup]
        return np.array([s.column for s in self.sites if s.orthogroup == orthogroup], dtype=int)

    @property
    def features(self) -> list[str]:
        return [site.feature(int(a)) for site in self.sites for a in np.flatnonzero(site.active)]

    def schema(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "feature": site.feature(int(a)),
                    "modality": "amino_acid",
                    "orthogroup": site.orthogroup,
                    "alignment_pos_1based": site.column + 1,
                    "residue": ALPHABET[a],
                    "aggregation": self.settings.aggregation,
                    "training_mean": float(site.mean[a]),
                }
                for site in self.sites
                for a in np.flatnonzero(site.active)
            ]
        )

    def raw(
        self, store: AlignmentStore, species: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if len(species) * max(1, len(self.sites) * 20) > MAX_MATRIX_CELLS:
            raise AlignmentError(
                "AA matrix limit exceeded; reduce amino_acid.ranked_site_filter.max_sites"
            )
        values = np.full((len(species), len(self.sites), 20), np.nan, dtype=np.float32)
        coverage = np.zeros((len(species), len(self.sites)), dtype=np.float32)
        representatives: list[dict[str, Any]] = []
        grouped: dict[str, list[int]] = defaultdict(list)
        for j, site in enumerate(self.sites):
            grouped[site.orthogroup].append(j)
        for name, indices in grouped.items():
            columns = np.array([self.sites[j].column for j in indices], dtype=int)
            representative = self.settings.aggregation == "best_covered"
            scoring = self.scoring_columns(name) if representative else columns
            og = store.ogs[name]
            chosen_rows = store.representatives(og, species, scoring) if representative else None
            q, c, chosen = store.frequencies(
                og,
                species,
                columns,
                representative=representative,
                chosen_rows=chosen_rows,
                weighted=self.settings.aggregation == "tpm_weighted_frequency",
            )
            values[:, indices] = q
            coverage[:, indices] = c
            if representative:
                assert chosen_rows is not None
                matrix = og.matrix()
                for sample_index, (sample, sequence_id) in enumerate(
                    zip(species, chosen, strict=True)
                ):
                    representatives.append(
                        {
                            "species": sample,
                            "orthogroup": name,
                            "sequence_id": sequence_id,
                            "n_scoring_sites": len(scoring),
                            "n_valid_scoring_sites": (
                                int(np.count_nonzero(_CODE[matrix[row, scoring]] < 20))
                                if (row := chosen_rows[sample_index]) is not None
                                else 0
                            ),
                            "n_valid_model_sites": int(np.isfinite(q[sample_index, :, 0]).sum()),
                        }
                    )
        return values, coverage, representatives

    def transform(
        self, store: AlignmentStore, species: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        raw, coverage, representatives = self.raw(store, species)
        columns = []
        information = []
        for j, site in enumerate(self.sites):
            assert site.mean is not None and site.known is not None and site.active is not None
            q = np.nan_to_num(raw[:, j], nan=0.0).astype(float)
            known_mass = q[:, site.known].sum(axis=1)
            known_mass = np.where(np.isclose(known_mass, 1, rtol=0, atol=1e-7), 1, known_mass)
            # Missing and novel-residue mass receive the training mean. A
            # partially novel composition retains its known fraction's effect.
            centered = q - known_mass[:, None] * site.mean
            columns.append(centered[:, site.active])
            information.append(coverage[:, j] * known_mass)
        return (np.concatenate(columns, axis=1), np.column_stack(information), representatives)


def _quality_summary(
    q: np.ndarray, groups: np.ndarray, quality: AminoAcidSiteFilter
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.nan_to_num(q)
    major = np.nanmean(q, axis=0).argmax(axis=1)
    states = np.any(values > 0, axis=0).sum(axis=1)
    minor = values.sum(axis=2) - np.take_along_axis(values, major[None, :, None], axis=2)[:, :, 0]
    minor_present = minor > 1e-7
    support_groups = np.zeros(q.shape[1], dtype=int)
    for group in np.unique(groups):
        support_groups += np.any(minor_present[groups == group], axis=0)
    # Major-relative counts are descriptive QC, never site-selection criteria.
    eligible = states >= quality.min_residue_states
    return eligible, minor_present.sum(axis=0), support_groups


def fit_amino_acid(
    store: AlignmentStore,
    species: list[str],
    settings: AminoAcidConfig,
    *,
    y: np.ndarray,
    groups: np.ndarray,
    contrast_groups: np.ndarray | None = None,
) -> AATransform:
    """Every reduction and decision below uses exactly these training species."""
    if len(species) != len(y) or len(groups) != len(y) or not species:
        raise AlignmentError("Invalid AA training sample schema")
    quality = settings.site_filter
    ranked = settings.ranked_site_filter
    required = int(np.ceil(quality.min_sample_fraction * len(species)))
    heap: list[tuple[float, int, AASite]] = []
    qc = []
    ordinal = 0
    representative = settings.aggregation == "best_covered"
    representative_columns: dict[str, np.ndarray] | None = {} if representative else None
    groups = groups.astype(str)
    block = min(SITE_BLOCK_SIZE, max(1, MAX_MATRIX_CELLS // (len(species) * 20)))
    for og in store.ogs.values():
        counts: dict[str, Any] = {
            "orthogroup": og.orthogroup,
            "n_columns": og.width,
            "n_after_call_filter": 0,
            "n_after_variant_filter": 0,
            "n_representative_scoring_sites": None,
        }
        qc.append(counts)
        if sum(s in og.species_rows for s in species) < required:
            continue
        weighted = settings.aggregation == "tpm_weighted_frequency"
        calls = store.call_counts(og, species, weighted=weighted)
        columns = np.flatnonzero(calls >= required)
        chosen_rows = None
        if representative:
            # Learn an unsupervised scoring mask before phenotype ranking.
            # Freeze gene choices on this mask, even if later filters remove sites.
            parts = []
            for start in range(0, len(columns), block):
                selected = columns[start : start + block]
                q, _, _ = store.frequencies(og, species, selected)
                eligible, _, _ = _quality_summary(q, groups, quality)
                parts.append(selected[eligible])
            scoring = np.concatenate(parts) if parts else np.empty(0, dtype=int)
            counts["n_representative_scoring_sites"] = len(scoring)
            if not len(scoring):
                continue
            assert representative_columns is not None
            chosen_rows = store.representatives(og, species, scoring)
            matrix = og.matrix()
            calls = np.zeros(og.width, dtype=np.int32)
            for row in chosen_rows:
                if row is not None:
                    calls[scoring] += _CODE[matrix[row, scoring]] < 20
            columns = scoring[calls[scoring] >= required]
        counts["n_after_call_filter"] = len(columns)
        variant_count = 0
        for start in range(0, len(columns), block):
            selected = columns[start : start + block]
            q, _, _ = store.frequencies(
                og,
                species,
                selected,
                representative=representative,
                chosen_rows=chosen_rows,
                weighted=weighted,
            )
            eligible, minor_samples, minor_groups = _quality_summary(q, groups, quality)
            scores = np.zeros(len(selected))
            if ranked.method == "unpaired":
                both = np.any(np.isfinite(q[y == 0, :, 0]), axis=0) & np.any(
                    np.isfinite(q[y == 1, :, 0]), axis=0
                )
                n0 = np.isfinite(q[y == 0, :, 0]).sum(axis=0)
                n1 = np.isfinite(q[y == 1, :, 0]).sum(axis=0)
                difference = np.nansum(q[y == 1], axis=0) / np.maximum(n1[:, None], 1)
                difference -= np.nansum(q[y == 0], axis=0) / np.maximum(n0[:, None], 1)
                scores = np.linalg.norm(difference, axis=1)
                eligible &= both
            elif ranked.method == "pair_aware":
                if contrast_groups is None:
                    raise AlignmentError("pair_aware AA ranking requires contrast groups")
                sums = np.zeros((len(selected), 20))
                n_pairs = np.zeros(len(selected), dtype=int)
                for group in sorted(set(contrast_groups.astype(str)) - {"", "None"}):
                    rows0 = (contrast_groups == group) & (y == 0)
                    rows1 = (contrast_groups == group) & (y == 1)
                    if not rows0.any() or not rows1.any():
                        continue
                    n0 = np.isfinite(q[rows0, :, 0]).sum(axis=0)
                    n1 = np.isfinite(q[rows1, :, 0]).sum(axis=0)
                    valid = (n0 > 0) & (n1 > 0)
                    difference = np.nansum(q[rows1], axis=0) / np.maximum(n1[:, None], 1)
                    difference -= np.nansum(q[rows0], axis=0) / np.maximum(n0[:, None], 1)
                    sums[valid] += difference[valid]
                    n_pairs += valid
                scores = np.linalg.norm(sums / np.maximum(n_pairs[:, None], 1), axis=1)
                eligible &= n_pairs >= ranked.min_contrast_pairs
            for raw_j in np.flatnonzero(eligible):
                j = int(raw_j)
                variant_count += 1
                site = AASite(
                    og.orthogroup,
                    int(selected[j]),
                    float(calls[selected[j]] / len(species)),
                    int(minor_samples[j]),
                    int(minor_groups[j]),
                    float(scores[j]),
                )
                entry = (site.score, -ordinal, site)
                ordinal += 1
                if len(heap) < ranked.max_sites:
                    heapq.heappush(heap, entry)
                elif ranked.method == "none":
                    raise AlignmentError(
                        "AA eligible sites exceed max_sites; enable site ranking, "
                        "tighten filters, or increase the explicit limit"
                    )
                else:
                    heapq.heappushpop(heap, entry)
        counts["n_after_variant_filter"] = variant_count
        if representative_columns is not None:
            needed = {item[2].orthogroup for item in heap}
            if og.orthogroup in needed:
                representative_columns[og.orthogroup] = scoring
            for name in list(representative_columns):
                if name not in needed:
                    del representative_columns[name]
    sites = sorted((item[2] for item in heap), key=lambda s: (s.orthogroup, s.column))
    if not sites:
        raise AlignmentError(
            "AA preprocessing removed all sites; inspect training coverage/filters"
        )
    if representative_columns is not None:
        needed = {site.orthogroup for site in sites}
        representative_columns = {
            name: columns for name, columns in representative_columns.items() if name in needed
        }
    fitted = AATransform(settings.model_copy(deep=True), sites, qc, representative_columns)
    raw, _, _ = fitted.raw(store, species)
    for j, site in enumerate(sites):
        observed = np.isfinite(raw[:, j, 0])
        site.post_aggregation_sample_fraction = float(np.mean(observed))
        site.mean = (
            np.asarray(np.mean(raw[observed, j], axis=0, dtype=np.float64))
            if observed.any()
            else np.zeros(20)
        )
        site.known = np.any(np.nan_to_num(raw[:, j]) > 0, axis=0)
        site.active = (
            (np.var(raw[observed, j], axis=0) > 1e-12)
            if observed.any()
            else np.zeros(20, dtype=bool)
        )
    if not fitted.features:
        raise AlignmentError("AA aggregation produced no variable features in this training fold")
    return fitted
