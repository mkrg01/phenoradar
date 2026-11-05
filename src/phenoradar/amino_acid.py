"""Alignment input and train-fitted, site-addressable amino-acid features."""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import polars as pl

from phenoradar.amino_acid_frequency_cache import (
    AminoAcidFrequencyCache,
    release_frequency_maps,
)
from phenoradar.amino_acid_input_cache import AlignmentInputCache
from phenoradar.amino_acid_presence import SpeciesPresenceCache
from phenoradar.config.schema import AminoAcidConfig, AminoAcidSiteFilter
from phenoradar.provenance import file_identity
from phenoradar.sequence_tpm_cache import SequenceTPMCacheError, read_sequence_tpm_cache

if TYPE_CHECKING:
    from phenoradar.amino_acid_encoding import FoldEncoding

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


def _matrix(
    shape: tuple[int, ...], dtype: Any = np.float64, *, order: Literal["C", "F"] = "C",
) -> np.ndarray:
    """Keep large output matrices on temporary disk; never select fewer sites."""
    if int(np.prod(shape, dtype=np.int64)) <= MAX_MATRIX_CELLS:
        return np.empty(shape, dtype=dtype, order=order)
    with TemporaryFile(prefix="phenoradar_aa_matrix_") as handle:
        return np.memmap(handle, mode="w+", dtype=dtype, shape=shape, order=order)


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
        cache_frequencies: bool = False,
        input_cache_dir: Path | None = None,
        only_orthogroups: set[str] | None = None,
        sequence_tpm_path: str | None = None,
        sequence_tpm_cache_dir: Path | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._temp = TemporaryDirectory(prefix="phenoradar_aa_")
        self.ogs: dict[str, AlignmentOG] = {}
        self.species = set(species)
        self.retained_species = self.species if cache_species is None else set(cache_species)
        self.cache_frequencies = cache_frequencies
        self._frequency_caches: dict[tuple[str, bool], AminoAcidFrequencyCache] = {}
        self._site_presence = SpeciesPresenceCache(Path(self._temp.name) / "species_presence.bin")
        self.tpm_identity: dict[str, Any] | None = None
        input_cache = (
            AlignmentInputCache(input_cache_dir, self.species, self.retained_species)
            if input_cache_dir is not None else None
        )
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
                index = len(self.ogs)
                self.ogs[path.stem] = (
                    self._read(path, index) if input_cache is None
                    else input_cache.read(path, partial(self._read, path, index))
                )
            if not self.ogs:
                raise AlignmentError(f"No alignment FASTA files found in {directory}")
            if only_orthogroups is not None and only_orthogroups - self.ogs.keys():
                raise AlignmentError("Prediction input is missing required orthogroup alignments")
            if progress and input_cache is not None:
                progress(f"Reuse {input_cache.hits:,}/{len(self.ogs):,} alignment byte caches.")
            if sequence_tpm_path is not None:
                self._load_tpm(Path(sequence_tpm_path), sequence_tpm_cache_dir, progress)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        release_frequency_maps(Path(self._temp.name))
        self._frequency_caches.clear()
        self._site_presence.clear()
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
        rows = np.array([og.mask_rows[s] for s in species if s in og.mask_rows], dtype=int)
        if not len(rows):
            return counts
        return self._site_presence.counts((og.orthogroup, weighted), Path(cache), og.width, rows)

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
        if (self.cache_frequencies and not representative and len(columns)
                and any(s in og.species_rows for s in species)):
            cache, compute, block = self._frequency_cache(og, weighted)
            q, c = cache.get(species, columns, compute, block_size=block)
            return q, c, [None] * len(species)
        return self._compute_frequencies(
            og, species, columns, representative=representative,
            chosen_rows=chosen_rows, weighted=weighted,
        )

    def _frequency_cache(
        self, og: AlignmentOG, weighted: bool,
    ) -> tuple[AminoAcidFrequencyCache,
               Callable[[list[str], np.ndarray], tuple[np.ndarray, np.ndarray]], int]:
        key = og.orthogroup, weighted
        if key not in self._frequency_caches:
            self._frequency_caches[key] = AminoAcidFrequencyCache(
                Path(self._temp.name) / f"{len(self._frequency_caches)}_frequency.bin",
                og.mask_rows,
            )
        cache = self._frequency_caches[key]
        block = min(SITE_BLOCK_SIZE, max(1, MAX_MATRIX_CELLS // max(1, len(cache.species) * 21)))

        def compute(samples: list[str], selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q, c, _ = self._compute_frequencies(og, samples, selected, weighted=weighted)
            return q, c

        return cache, compute, block

    def compact_frequencies(
        self, og: AlignmentOG, species: list[str], columns: np.ndarray, *, weighted: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache, compute, block = self._frequency_cache(og, weighted)
        return cache.get_compact(species, columns, compute, block_size=block)

    def frequency_candidates(
        self, og: AlignmentOG, species: list[str], columns: np.ndarray, *,
        weighted: bool, min_states: int,
    ) -> np.ndarray:
        cache, compute, block = self._frequency_cache(og, weighted)
        return cache.eligible_columns(species, columns, compute, block_size=block,
                                      min_states=min_states)

    def _compute_frequencies(
        self,
        og: AlignmentOG,
        species: list[str],
        columns: np.ndarray,
        *,
        representative: bool = False,
        chosen_rows: list[int | None] | None = None,
        weighted: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
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
            if len(gene_rows) == 1:
                row = gene_rows[0]
                if weighted:
                    assert og.tpm is not None
                    if og.tpm[row] == 0:
                        continue
                codes = _CODE[matrix[row, columns]]
                observed = codes < 20
                values[i, observed] = 0
                values[i, np.flatnonzero(observed), codes[observed]] = 1
                coverage[i, observed] = 1
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


class AASiteTable(Sequence[AASite]):
    """Columnar fitted sites; materialize small site views only for interpretation."""

    column: np.ndarray
    sample_fraction: np.ndarray
    minor_samples: np.ndarray
    minor_groups: np.ndarray
    score: np.ndarray
    mean: np.ndarray
    known: np.ndarray
    active: np.ndarray
    post_aggregation_sample_fraction: np.ndarray

    def __init__(self, blocks: list[dict[str, Any]]) -> None:
        blocks = sorted(blocks, key=lambda b: (b["orthogroup"], int(b["column"][0])))
        self.orthogroups = np.concatenate([
            np.full(len(b["column"]), b["orthogroup"], dtype=object) for b in blocks
        ])
        for name in (
            "column", "sample_fraction", "minor_samples", "minor_groups", "score",
            "mean", "known", "active", "post_aggregation_sample_fraction",
        ):
            setattr(self, name, np.concatenate([b[name] for b in blocks]))
        boundaries = np.flatnonzero(self.orthogroups[1:] != self.orthogroups[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(self.column)]))
        self.groups = [(str(self.orthogroups[a]), int(a), int(b))
                       for a, b in zip(starts, ends, strict=True)]

    def __len__(self) -> int:
        return len(self.column)

    @overload
    def __getitem__(self, index: int) -> AASite: ...

    @overload
    def __getitem__(self, index: slice) -> list[AASite]: ...

    def __getitem__(self, index: int | slice) -> AASite | list[AASite]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return AASite(
            str(self.orthogroups[index]), int(self.column[index]),
            float(self.sample_fraction[index]), int(self.minor_samples[index]),
            int(self.minor_groups[index]), float(self.score[index]),
            self.mean[index], self.known[index], self.active[index],
            float(self.post_aggregation_sample_fraction[index]),
        )

    def __iter__(self) -> Iterator[AASite]:
        for i in range(len(self)):
            yield self[i]


@dataclass
class AATransform:
    settings: AminoAcidConfig
    sites: Sequence[AASite]
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

    def schema_rows(self) -> Iterator[dict[str, Any]]:
        for site in self.sites:
            for a in np.flatnonzero(site.active):
                yield {
                    "feature": site.feature(int(a)),
                    "modality": "amino_acid",
                    "orthogroup": site.orthogroup,
                    "alignment_pos_1based": site.column + 1,
                    "residue": ALPHABET[a],
                    "aggregation": self.settings.aggregation,
                    "training_mean": float(site.mean[a]),
                }

    def schema(self) -> pl.DataFrame:
        return pl.DataFrame(self.schema_rows())

    def raw_blocks(
        self,
        store: AlignmentStore,
        species: list[str],
        representatives: list[dict[str, Any]] | None = None,
    ) -> Iterator[tuple[list[int], np.ndarray, np.ndarray]]:
        """Read bounded site blocks, choosing each representative once per OG."""
        grouped: dict[str, Sequence[int]] = {}
        if isinstance(self.sites, AASiteTable):
            grouped = {name: range(start, end) for name, start, end in self.sites.groups}
        else:
            lists: dict[str, list[int]] = defaultdict(list)
            for j, site in enumerate(self.sites):
                lists[site.orthogroup].append(j)
            grouped.update(lists)
        block = min(SITE_BLOCK_SIZE, max(1, MAX_MATRIX_CELLS // max(1, len(species) * 20)))
        for name, indices in grouped.items():
            representative = self.settings.aggregation == "best_covered"
            scoring = self.scoring_columns(name) if representative else np.empty(0, dtype=int)
            og = store.ogs[name]
            chosen_rows = store.representatives(og, species, scoring) if representative else None
            valid_model = np.zeros(len(species), dtype=int)
            for start in range(0, len(indices), block):
                selected = indices[start : start + block]
                columns = (
                    self.sites.column[selected[0]:selected[-1] + 1]
                    if isinstance(self.sites, AASiteTable)
                    else np.array([self.sites[j].column for j in selected], dtype=int)
                )
                q, c, _ = store.frequencies(
                    og,
                    species,
                    columns,
                    representative=representative,
                    chosen_rows=chosen_rows,
                    weighted=self.settings.aggregation == "tpm_weighted_frequency",
                )
                if representative and representatives is not None:
                    valid_model += np.isfinite(q[:, :, 0]).sum(axis=1)
                yield list(selected), q, c
            if representative and representatives is not None:
                assert chosen_rows is not None
                matrix = og.matrix()
                for sample_index, (sample, row) in enumerate(
                    zip(species, chosen_rows, strict=True)
                ):
                    representatives.append(
                        {
                            "species": sample,
                            "orthogroup": name,
                            "sequence_id": None if row is None else og.sequence_ids[row],
                            "n_scoring_sites": len(scoring),
                            "n_valid_scoring_sites": (
                                int(np.count_nonzero(_CODE[matrix[row, scoring]] < 20))
                                if row is not None
                                else 0
                            ),
                            "n_valid_model_sites": int(valid_model[sample_index]),
                        }
                    )

    def raw(
        self, store: AlignmentStore, species: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        values = _matrix((len(species), len(self.sites), 20), np.float32)
        coverage = _matrix((len(species), len(self.sites)), np.float32)
        representatives: list[dict[str, Any]] = []
        for indices, q, c in self.raw_blocks(store, species, representatives):
            values[:, indices] = q
            coverage[:, indices] = c
        return values, coverage, representatives

    def matrix(self, store: AlignmentStore, species: list[str]) -> np.ndarray:
        """Encode inner-CV data without allocating unused observation outputs."""
        return self._transform(store, species, with_information=False)[0]

    def transform(
        self, store: AlignmentStore, species: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        matrix, information, representatives = self._transform(
            store, species, with_information=True
        )
        assert information is not None
        return matrix, information, representatives

    def _transform(
        self, store: AlignmentStore, species: list[str], *, with_information: bool
    ) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, Any]]]:
        counts = (self.sites.active.sum(axis=1) if isinstance(self.sites, AASiteTable)
                  else np.array([np.count_nonzero(site.active) for site in self.sites]))
        offsets = np.cumsum(np.concatenate(([0], counts)))
        # Site blocks fill whole columns; native glmnet also consumes column-major input.
        matrix = _matrix((len(species), int(offsets[-1])), order="F")
        information = _matrix((len(species), len(self.sites))) if with_information else None
        representatives: list[dict[str, Any]] = []
        for indices, raw, coverage in self.raw_blocks(
            store, species, representatives if with_information else None
        ):
            if isinstance(self.sites, AASiteTable):
                selected = slice(indices[0], indices[-1] + 1)
                means = self.sites.mean[selected]
                known = self.sites.known[selected]
                active = self.sites.active[selected]
            else:
                sites = [self.sites[j] for j in indices]
                means = np.stack([site.mean for site in sites])
                known = np.stack([site.known for site in sites])
                active = np.stack([site.active for site in sites])
            q = np.nan_to_num(raw, nan=0.0)
            known_mass = np.sum(q * known[None, :, :], axis=2, dtype=np.float64)
            known_mass = np.where(np.isclose(known_mass, 1, rtol=0, atol=1e-7), 1, known_mass)
            # Missing and novel-residue mass receive the training mean.
            state_columns = np.flatnonzero(active.ravel())
            values = q.reshape(len(species), len(indices) * 20)[:, state_columns].astype(float)
            values -= known_mass[:, state_columns // 20] * means[active][None, :]
            destination: slice | np.ndarray = slice(offsets[indices[0]], offsets[indices[-1] + 1])
            if indices[-1] - indices[0] + 1 != len(indices):
                destination = np.concatenate(
                    [np.arange(offsets[j], offsets[j + 1]) for j in indices]
                )
            matrix[:, destination] = values
            if information is not None:
                information[:, indices] = coverage * known_mass
        return matrix, information, representatives


def _site_statistics(
    raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit all site moments in a block, preserving float32 variance semantics."""
    observed = np.isfinite(raw[:, :, 0])
    counts = observed.sum(axis=0)
    values = np.nan_to_num(raw)
    means = np.zeros(raw.shape[1:], dtype=np.float64)
    np.divide(
        values.sum(axis=0, dtype=np.float64), counts[:, None],
        out=means, where=counts[:, None] > 0,
    )
    mean32 = np.zeros(raw.shape[1:], dtype=np.float32)
    np.divide(
        values.sum(axis=0, dtype=np.float32), counts[:, None],
        out=mean32, where=counts[:, None] > 0,
    )
    differences = values - mean32[None, :, :]
    differences[~observed] = 0
    variance = np.zeros_like(mean32)
    np.divide(
        np.sum(differences * differences, axis=0, dtype=np.float32), counts[:, None],
        out=variance, where=counts[:, None] > 0,
    )
    return counts / raw.shape[0], means, np.any(values > 0, axis=0), variance > 1e-12


def _quality_summary(
    q: np.ndarray, groups: np.ndarray, quality: AminoAcidSiteFilter, *, audit: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.any(q > 0, axis=0).sum(axis=1)
    eligible = states >= quality.min_residue_states
    if not audit:
        zeros = np.zeros(q.shape[1], dtype=int)
        return eligible, zeros, zeros
    values = np.nan_to_num(q)
    major = np.nanmean(q, axis=0).argmax(axis=1)
    minor = values.sum(axis=2) - np.take_along_axis(values, major[None, :, None], axis=2)[:, :, 0]
    minor_present = minor > 1e-7
    support_groups = np.zeros(q.shape[1], dtype=int)
    for group in np.unique(groups):
        support_groups += np.any(minor_present[groups == group], axis=0)
    # Major-relative counts are descriptive QC, never site-selection criteria.
    return eligible, minor_present.sum(axis=0), support_groups


def fit_amino_acid(
    store: AlignmentStore,
    species: list[str],
    settings: AminoAcidConfig,
    *,
    y: np.ndarray,
    groups: np.ndarray,
    contrast_groups: np.ndarray | None = None,
    audit: bool = True,
    n_jobs: int = 1,
    _encoding: FoldEncoding | None = None,
    _allow_empty: bool = False,
) -> AATransform:
    """Every reduction and decision below uses exactly these training species."""
    if len(species) != len(y) or len(groups) != len(y) or not species:
        raise AlignmentError("Invalid AA training sample schema")
    if n_jobs > 1 and len(store.ogs) > 1 and settings.ranked_site_filter.method == "none":
        weighted = settings.aggregation == "tpm_weighted_frequency"
        warm = any(c.length for (_, is_weighted), c in store._frequency_caches.items()
                   if is_weighted == weighted)
        # Cold cache creation avoids the per-OG assembly overhead.
        if not store.cache_frequencies or warm:
            from phenoradar.amino_acid_og_parallel import fit_orthogroups

            return fit_orthogroups(store, species, settings, n_jobs=n_jobs, encoding=_encoding,
                                   y=y, groups=groups, contrast_groups=contrast_groups, audit=audit)
    quality = settings.site_filter
    ranked = settings.ranked_site_filter
    required = int(np.ceil(quality.min_sample_fraction * len(species)))
    heap: list[tuple[float, int, AASite]] = []
    blocks: list[dict[str, Any]] = []
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
                eligible, _, _ = _quality_summary(q, groups, quality, audit=False)
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
        samples = species if _encoding is None else _encoding.species
        block_choices = chosen_rows
        if _encoding is not None and representative:
            block_choices = store.representatives(og, samples, scoring)
        sample_index = {s: i for i, s in enumerate(samples)}
        training_rows = [sample_index[s] for s in species]
        for start in range(0, len(columns), block):
            selected = columns[start : start + block]
            if _encoding is not None and not representative and store.cache_frequencies:
                from phenoradar.amino_acid_encoding import compact_statistics

                candidates = store.frequency_candidates(
                    og, species, selected, weighted=weighted,
                    min_states=quality.min_residue_states,
                )
                selected = selected[candidates]
                if not len(selected):
                    continue
                values, _, observed, state_sites, residues = store.compact_frequencies(
                    og, samples, selected, weighted=weighted,
                )
                eligible, minor_samples, minor_groups, fractions, means, known, active = (
                    compact_statistics(values[training_rows], observed[training_rows],
                                       state_sites, residues, groups, quality, audit=audit)
                )
                variant_count += int(eligible.sum())
                if eligible.any():
                    _encoding.add_compact(values, state_sites, residues, eligible,
                                          means, known, active)
                    blocks.append({
                        "orthogroup": og.orthogroup, "column": selected[eligible],
                        "sample_fraction": calls[selected[eligible]] / len(species),
                        "minor_samples": minor_samples[eligible],
                        "minor_groups": minor_groups[eligible],
                        "score": np.zeros(int(eligible.sum())), "mean": means[eligible],
                        "known": known[eligible], "active": active[eligible],
                        "post_aggregation_sample_fraction": fractions[eligible],
                    })
                continue
            raw, _, _ = store.frequencies(
                og, samples, selected, representative=representative,
                chosen_rows=block_choices, weighted=weighted,
            )
            q = raw if _encoding is None else raw[training_rows]
            eligible, minor_samples, minor_groups = _quality_summary(
                q, groups, quality, audit=audit
            )
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
            if ranked.method == "none":
                variant_count += int(np.count_nonzero(eligible))
                if np.any(eligible):
                    fractions, means, known, active = _site_statistics(q)
                    if _encoding is not None:
                        _encoding.add(raw[:, eligible], means[eligible], known[eligible],
                                      active[eligible])
                    blocks.append({
                        "orthogroup": og.orthogroup,
                        "column": selected[eligible],
                        "sample_fraction": calls[selected[eligible]] / len(species),
                        "minor_samples": minor_samples[eligible],
                        "minor_groups": minor_groups[eligible],
                        "score": scores[eligible],
                        "mean": means[eligible], "known": known[eligible],
                        "active": active[eligible],
                        "post_aggregation_sample_fraction": fractions[eligible],
                    })
                continue
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
                else:
                    heapq.heappushpop(heap, entry)
        counts["n_after_variant_filter"] = variant_count
        if representative_columns is not None and ranked.method == "none":
            if variant_count:
                representative_columns[og.orthogroup] = scoring
        elif representative_columns is not None:
            needed = {item[2].orthogroup for item in heap}
            if og.orthogroup in needed:
                representative_columns[og.orthogroup] = scoring
            for name in list(representative_columns):
                if name not in needed:
                    del representative_columns[name]
    sites: Sequence[AASite] = (
        AASiteTable(blocks) if blocks
        else sorted((item[2] for item in heap), key=lambda s: (s.orthogroup, s.column))
    )
    if not sites and not _allow_empty:
        raise AlignmentError(
            "AA preprocessing removed all sites; inspect training coverage/filters"
        )
    if representative_columns is not None:
        needed = {site.orthogroup for site in sites}
        representative_columns = {
            name: columns for name, columns in representative_columns.items() if name in needed
        }
    fitted = AATransform(settings.model_copy(deep=True), sites, qc, representative_columns)
    if isinstance(sites, AASiteTable):
        if not np.any(sites.active) and not _allow_empty:
            raise AlignmentError(
                "AA aggregation produced no variable features in this training fold"
            )
        return fitted
    for indices, raw, _ in fitted.raw_blocks(store, species):
        fractions, means, known, active = _site_statistics(raw)
        for k, j in enumerate(indices):
            site = sites[j]
            site.post_aggregation_sample_fraction = float(fractions[k])
            site.mean = means[k]
            site.known = known[k]
            site.active = active[k]
    if not _allow_empty and not any(np.any(site.active) for site in sites):
        raise AlignmentError("AA aggregation produced no variable features in this training fold")
    return fitted
