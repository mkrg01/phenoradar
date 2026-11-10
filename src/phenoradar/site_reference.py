"""Species-balanced external reference selection, independent of model fitting."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from phenoradar.amino_acid import ALPHABET, AlignmentOG, AlignmentStore
from phenoradar.site_reference_resources import (
    ResourceCache,
    SiteAnnotationError,
    read_fasta,
    sha256,
    write_json,
)


@dataclass(frozen=True)
class ReferencePolicy:
    core_species_fraction: float = 0.8
    minimum_reference_coverage: float = 0.8
    minimum_median_identity: float = 0.3

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0 < value <= 1:
                raise SiteAnnotationError(f"{name} must be finite and in (0, 1].")


_DEFAULT_POLICY = ReferencePolicy()


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pl.DataFrame(rows, infer_schema_length=None).write_csv(
            path, separator="\t", null_value="NA"
        )
    else:
        path.write_text("status\n")


def selected_sites(run: Path, scope: str) -> pl.DataFrame:
    """Bounded aggregation of nonzero site effects; preserve all selected positions."""
    if scope not in {"final", "cv"}:
        raise SiteAnnotationError("Scope must be final or cv.")
    tables = run / ("model/tables" if scope == "final" else "cv/tables")
    effects = tables / "amino_acid_site_effect_range.tsv"
    if not effects.is_file():
        raise SiteAnnotationError(f"Site effects are not available for {scope}: {effects}")
    context = ["scope", "sample_set_id", "candidate_index"]
    scan = pl.scan_csv(effects, separator="\t", infer_schema_length=10000)
    count = scan.select(context).unique().select(pl.len()).collect().item()
    if not count:
        raise SiteAnnotationError("Site effect table has no models.")
    frame = (
        scan.filter(pl.col("effect_range") > 1e-12)
        .group_by("orthogroup", "alignment_pos_1based")
        .agg((pl.col("effect_range").sum() / count).alias("effect_range"))
        .collect(engine="streaming")
    )
    pss = (
        pl.scan_csv(tables / "amino_acid_site_pss.tsv", separator="\t")
        .filter(pl.col("pss") > 0)
        .group_by("orthogroup", "alignment_pos_1based")
        .agg((pl.col("pss").sum() / count).alias("pss"))
        .collect(engine="streaming")
    )
    frame = frame.join(pss, on=["orthogroup", "alignment_pos_1based"], how="left")
    stability = run / "cv/tables/amino_acid_site_stability.tsv"
    if stability.is_file():
        freq = (
            pl.scan_csv(stability, separator="\t")
            .select("orthogroup", "alignment_pos_1based", "selection_frequency")
            .collect(engine="streaming")
        )
        frame = frame.join(freq, on=["orthogroup", "alignment_pos_1based"], how="left")
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("selection_frequency"))
    return frame.sort(["orthogroup", "alignment_pos_1based"])


def _run_inputs(run: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    metadata_path = run / "run_metadata.json"
    if not metadata_path.is_file():
        raise SiteAnnotationError("A completed run with run_metadata.json is required.")
    metadata = json.loads(metadata_path.read_text())
    status = str(metadata.get("status", ""))
    if not status.endswith("completed"):
        raise SiteAnnotationError(f"Run is not complete: {status}")
    config = yaml.safe_load((run / "resolved_config.yml").read_text())
    if config["data"].get("feature_mode") not in {"amino_acid", "combined"}:
        raise SiteAnnotationError("Site annotation requires amino_acid or combined mode.")
    manifest = pl.read_csv(
        run / "split/tables/split_manifest.tsv",
        separator="\t",
        schema_overrides={"fold_id": pl.String},
    )
    training = sorted(
        manifest.filter(pl.col("pool").is_in(["train", "validation"]))["species"].unique().to_list()
    )
    species = sorted(manifest["species"].unique().to_list())
    if not training:
        raise SiteAnnotationError("Split manifest has no training species.")
    return config, species, training


def core_columns(og: AlignmentOG, training: list[str], policy: ReferencePolicy) -> np.ndarray:
    matrix = og.matrix()
    canonical = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)
    counts = np.zeros(og.width, dtype=np.int64)
    for species in training:
        indices = og.species_rows.get(species, [])
        if indices:
            counts += np.isin(matrix[indices], canonical).any(axis=0)
    return counts >= policy.core_species_fraction * len(training)


def project_references(
    og: AlignmentOG,
    references: list[dict[str, Any]],
    work: Path,
    mafft: str,
) -> tuple[dict[str, str], dict[str, dict[int, int]], str]:
    """Keep columns fixed and obtain full-length positions from MAFFT's map file."""
    work.mkdir(parents=True, exist_ok=True)
    matrix = og.matrix()
    backbone = work / "training_alignment.fa"
    with backbone.open("w") as handle:
        for index, name in enumerate(og.sequence_ids):
            handle.write(f">{name}\n{matrix[index].tobytes().decode()}\n")
    if not og.sequence_ids:
        raise SiteAnnotationError(f"No training sequences for {og.orthogroup}")
    candidates = work / "references.fa"
    # Identical sequences are aligned once, while accession-level ranking remains explicit.
    representatives: dict[str, str] = {}
    aliases = {}
    for record in references:
        sequence = record["sequence"]
        name = "ref_" + hashlib.sha256(sequence.encode()).hexdigest()
        representatives[name] = sequence
        aliases[record["accession"]] = name
    candidates.write_text(
        "".join(f">{name}\n{seq}\n" for name, seq in sorted(representatives.items()))
    )
    version = subprocess.run([mafft, "--version"], capture_output=True, text=True, check=False)
    version_text = (version.stdout + version.stderr).strip()
    command = [
        mafft,
        "--anysymbol",
        "--add",
        candidates.name,
        "--keeplength",
        "--mapout",
        "--thread",
        "1",
        "--inputorder",
        backbone.name,
    ]
    try:
        result = subprocess.run(command, cwd=work, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise SiteAnnotationError(f"Cannot run MAFFT: {mafft}") from exc
    (work / "mafft.stderr.txt").write_text(result.stderr)
    write_json(work / "mafft_command.json", dict(command=command, version=version_text))
    if result.returncode:
        raise SiteAnnotationError(f"MAFFT failed for {og.orthogroup}: {result.stderr[-1000:]}")
    (work / "reference_alignment.fa").write_text(result.stdout)
    aligned = read_fasta(result.stdout)
    for index, name in enumerate(og.sequence_ids):
        if aligned.get(name) != matrix[index].tobytes().decode():
            raise SiteAnnotationError("MAFFT modified the training alignment.")
    maps: dict[str, dict[int, int]] = {name: {} for name in representatives}
    name = ""
    map_path = candidates.with_suffix(".fa.map")
    if not map_path.is_file():
        raise SiteAnnotationError("MAFFT did not write the coordinate map.")
    for line in map_path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
        elif line.strip() and not line.startswith("#"):
            letter, original, column = (part.strip() for part in line.split(","))
            if column != "-":
                position, col = int(original), int(column)
                if name not in maps or not 1 <= col <= og.width:
                    raise SiteAnnotationError("Invalid MAFFT coordinate map.")
                if (
                    representatives[name][position - 1] != letter
                    or aligned[name][col - 1] != letter
                ):
                    raise SiteAnnotationError("MAFFT residue/map mismatch.")
                maps[name][col] = position
    projected, positions = {}, {}
    for accession, alias in aliases.items():
        if len(aligned.get(alias, "")) != og.width:
            raise SiteAnnotationError("Reference alignment width mismatch.")
        projected[accession], positions[accession] = aligned[alias], maps[alias]
    return projected, positions, version_text


def score_reference(
    record: dict[str, Any],
    aligned: str,
    og: AlignmentOG,
    training: list[str],
    core: np.ndarray,
    policy: ReferencePolicy,
) -> dict[str, Any]:
    ref = np.frombuffer(aligned.encode(), dtype=np.uint8)[core]
    canonical = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)
    ref_known = np.isin(ref, canonical)
    n_ref = int(ref_known.sum())
    coverage = n_ref / len(ref) if len(ref) else 0.0
    scores, identities, gene_choices = [], [], []
    matrix = og.matrix()
    for species in training:
        choices = []
        for row in og.species_rows.get(species, []):
            seq = matrix[row][core]
            known = np.isin(seq, canonical)
            paired = known & ref_known
            n_pair = int(paired.sum())
            n_gene = int(known.sum())
            identical = int(((seq == ref) & paired).sum())
            identity = identical / n_pair if n_pair else 0.0
            # identity * min(two coverages) = identical / max(nonmissing lengths).
            score = identical / max(n_gene, n_ref) if max(n_gene, n_ref) else 0.0
            choices.append((-score, og.sequence_ids[row], identity))
        if choices:
            negative, gene, identity = min(choices)
            scores.append(-negative)
            identities.append(identity)
            gene_choices.append(
                dict(species=species, sequence_id=gene, score=-negative, identity=identity)
            )
        else:
            scores.append(0.0)
            identities.append(0.0)
            gene_choices.append(dict(species=species, sequence_id=None, score=0.0, identity=0.0))
    identity = float(np.median(identities))
    reasons = []
    if not core.any():
        reasons.append("no_core_columns")
    if record.get("fragment", False):
        reasons.append("fragment")
    if not record.get("membership_verified", False):
        reasons.append("membership_unverified")
    if coverage < policy.minimum_reference_coverage:
        reasons.append("low_core_coverage")
    if identity < policy.minimum_median_identity:
        reasons.append("low_median_identity")
    return dict(
        accession=record["accession"],
        score=float(np.median(scores)),
        core_coverage=coverage,
        median_identity=identity,
        reviewed=bool(record.get("reviewed", False)),
        qualified=not reasons,
        exclusion_reason=";".join(reasons),
        species_matches=gene_choices,
    )


def reference_order(row: dict[str, Any]) -> tuple[Any, ...]:
    return (-row["score"], -row["core_coverage"], -int(row["reviewed"]), row["accession"])


def annotate_sites(
    *,
    run: Path,
    catalog: Path,
    out: Path,
    cache: ResourceCache,
    scope: str = "final",
    orthogroups: list[str] | None = None,
    policy: ReferencePolicy = _DEFAULT_POLICY,
    mafft: str = "mafft",
    pymol: str = "pymol",
    render: bool = True,
    overrides: Path | None = None,
    orthodb_release: str | None = None,
) -> Path:
    """Annotate a completed run in a new output directory; never modify its models."""
    from phenoradar import __version__
    from phenoradar.site_structure import acquire_structures, structure_annotations
    from phenoradar.site_structure_render import render_structure

    run, catalog, out = run.resolve(), catalog.resolve(), out.resolve()
    if out.exists():
        raise SiteAnnotationError(f"Output already exists: {out}")
    config, species, training = _run_inputs(run)
    for module in ["Bio", "reportlab"] if render else ["Bio"]:
        if importlib.util.find_spec(module) is None:
            raise SiteAnnotationError("Install phenoradar[structure] for structure annotations.")
    if shutil.which(mafft) is None:
        raise SiteAnnotationError(f"MAFFT executable not found: {mafft}")
    if render and shutil.which(pymol) is None:
        raise SiteAnnotationError(
            f"PyMOL executable not found: {pymol}; install PyMOL or use --no-render."
        )
    sites = selected_sites(run, scope)
    if orthogroups:
        unknown = set(orthogroups) - set(sites["orthogroup"])
        if unknown:
            raise SiteAnnotationError(f"OGs have no selected sites in {scope}: {sorted(unknown)}")
        sites = sites.filter(pl.col("orthogroup").is_in(orthogroups))
    for name in sites["orthogroup"].unique():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise SiteAnnotationError("Unsafe orthogroup identifier.")
    data = json.loads(catalog.read_text())
    if data.get("schema_version") != "site-reference-catalog-1":
        raise SiteAnnotationError("Unsupported reference catalog schema.")
    release = data.get("orthodb_release", "")
    if not release:
        raise SiteAnnotationError("Reference catalog has no frozen OrthoDB release.")
    match = re.search(
        r"odb(\d+)v(\d+)_OGs", str(config["data"].get("orthogroup_annotation_path", ""))
    )
    inferred = f"v{match[1]}.{match[2]}" if match else None
    expected_release = orthodb_release or inferred
    if expected_release is None:
        raise SiteAnnotationError(
            "Supply --orthodb-release when the run does not identify its release."
        )
    expected_release = "v" + expected_release.removeprefix("v")
    if release != expected_release or (inferred is not None and inferred != expected_release):
        raise SiteAnnotationError("Reference catalog and run OrthoDB releases differ.")
    automatic = {}
    if overrides:
        for row in pl.read_csv(overrides, separator="\t").iter_rows(named=True):
            override_og, accession = str(row["orthogroup"]), str(row["accession"])
            if override_og in automatic:
                raise SiteAnnotationError(f"Duplicate reference override: {override_og}")
            automatic[override_og] = accession
        if set(automatic) - set(sites["orthogroup"]):
            raise SiteAnnotationError("Override refers to an unrequested OG.")
    binding = dict(
        split_manifest_sha256=sha256(run / "split/tables/split_manifest.tsv"),
        alignment_inputs_sha256=sha256(run / "model/tables/alignment_inputs.tsv"),
        policy=asdict(policy),
    )
    if data.get("frozen_annotation_binding") not in (None, binding):
        raise SiteAnnotationError("Portable catalog is bound to different run inputs or policy.")
    out.mkdir(parents=True)
    artifact_paths = [
        run / p
        for p in [
            "run_metadata.json",
            "resolved_config.yml",
            "split/tables/split_manifest.tsv",
            "model/tables/alignment_inputs.tsv",
            ("model" if scope == "final" else "cv") + "/tables/amino_acid_site_effect_range.tsv",
            ("model" if scope == "final" else "cv") + "/tables/amino_acid_site_pss.tsv",
            "cv/tables/amino_acid_site_stability.tsv",
        ]
    ]
    from phenoradar.provenance import phenoradar_build_snapshot

    manifest: dict[str, Any] = dict(
        schema_version="site-annotation-1",
        reference_policy="species-balanced-reference-1",
        status="running",
        run=str(run),
        catalog=str(catalog),
        catalog_sha256=sha256(catalog),
        orthodb_release=release,
        policy=asdict(policy),
        scope=scope,
        training_species=training,
        reference_selection_uses="training_sequences_only",
        overrides_sha256=sha256(overrides) if overrides else None,
        input_hashes={},
        python_version=platform.python_version(),
        phenoradar_version=__version__,
        annotation_build=phenoradar_build_snapshot(Path(__file__)),
        annotation_sources={p.name: sha256(p) for p in Path(__file__).parent.glob("site_*.py")},
        run_artifacts={str(p.relative_to(run)): sha256(p) for p in artifact_paths if p.is_file()},
        library_versions={
            name: package_version(name)
            for name in (
                ["numpy", "polars", "biopython", "reportlab"]
                if render
                else ["numpy", "polars", "biopython"]
            )
        },
        orthogroups=[],
    )
    portable = {**data, "structures": [], "structures_complete_for": []}
    portable["references"] = [
        r for r in data["references"] if r["orthogroup"] in set(sites["orthogroup"])
    ]
    portable["orthogroups"] = sorted(set(sites["orthogroup"]))
    portable["frozen_annotation_binding"] = binding
    write_json(out / "reference_catalog.json", portable)
    write_json(out / "manifest.json", manifest)
    try:
        inputs = pl.read_csv(run / "model/tables/alignment_inputs.tsv", separator="\t")
        expected = {str(r["orthogroup"]): r for r in inputs.iter_rows(named=True)}
        for name in sites["orthogroup"].unique().sort().to_list():
            print(f"Annotating {name}", flush=True)
            target = out / name
            target.mkdir()
            records = [r for r in data["references"] if r["orthogroup"] == name]
            accessions = [r["accession"] for r in records]
            if len(set(accessions)) != len(accessions):
                raise SiteAnnotationError(f"Duplicate catalog accessions for {name}")
            for record in records:
                if not re_full_accession(record["accession"]):
                    raise SiteAnnotationError("Unsafe or empty reference accession.")
                seq = record["sequence"]
                if not seq or not seq.isascii() or not seq.isalpha() or seq != seq.upper():
                    raise SiteAnnotationError(
                        "Reference must be an ungapped uppercase AA sequence."
                    )
                if hashlib.sha256(seq.encode()).hexdigest() != record["sequence_sha256"]:
                    raise SiteAnnotationError("Reference catalog sequence checksum mismatch.")
            if not records:
                manifest["orthogroups"].append(
                    dict(orthogroup=name, status="reference_unavailable")
                )
                continue
            with AlignmentStore(
                Path(config["data"]["alignment_dir"]),
                species,
                cache_species=training,
                only_orthogroups={name},
            ) as store:
                og = store.ogs[name]
                if (
                    name not in expected
                    or og.sha256 != expected[name]["sha256"]
                    or (og.width != int(expected[name]["alignment_length"]))
                ):
                    raise SiteAnnotationError(
                        f"Input alignment changed since model fitting: {name}"
                    )
                manifest["input_hashes"][name] = og.sha256
                core = core_columns(og, training, policy)
                projected, positions, mafft_version = project_references(
                    og, records, target / "alignment", mafft
                )
                manifest["mafft_version"] = mafft_version
                ranked = sorted(
                    [
                        score_reference(r, projected[r["accession"]], og, training, core, policy)
                        for r in records
                    ],
                    key=reference_order,
                )
            _write_table(
                target / "reference_candidates.tsv",
                [
                    dict(rank=i, **{k: v for k, v in row.items() if k != "species_matches"})
                    for i, row in enumerate(ranked, 1)
                ],
            )
            forced = automatic.get(name)
            eligible = [r for r in ranked if r["qualified"]]
            choice = (
                next((r for r in ranked if r["accession"] == forced), None)
                if forced
                else (eligible[0] if eligible else None)
            )
            if forced and choice is None:
                raise SiteAnnotationError(
                    f"Override accession is not in the frozen catalog: {forced}"
                )
            if choice is None:
                manifest["orthogroups"].append(
                    dict(orthogroup=name, status="reference_unavailable")
                )
                continue
            accession = choice["accession"]
            reference = next(r for r in records if r["accession"] == accession)
            reference["selection_mode"] = "manual_override" if forced else "automatic"
            reference["reference_qualified"] = choice["qualified"]
            write_json(
                target / "selected_reference.json",
                dict(
                    **reference,
                    metrics=choice,
                    n_core_columns=int(core.sum()),
                ),
            )
            _write_table(target / "reference_species_matches.tsv", choice["species_matches"])
            selected = sites.filter(pl.col("orthogroup") == name).to_dicts()
            rows = []
            for site in selected:
                col = int(site["alignment_pos_1based"])
                if not 1 <= col <= len(projected[accession]):
                    raise SiteAnnotationError("Selected site is outside its original alignment.")
                pos = positions[accession].get(col)
                letter = None if pos is None else reference["sequence"][pos - 1]
                status = (
                    "reference_gap"
                    if pos is None
                    else "reference_unknown"
                    if letter is None or letter not in ALPHABET
                    else "mapped"
                )
                rows.append(
                    dict(
                        **site,
                        analysis_scope=scope,
                        accession=accession,
                        reference_pos_1based=pos,
                        reference_residue=letter,
                        reference_status=status,
                    )
                )
            _write_table(target / "site_reference_map.tsv", rows)
            (target / "reference.fa").write_text(f">{accession}\n{reference['sequence']}\n")
            core_positions = {
                positions[accession].get(int(i) + 1, -int(i) - 1) for i in np.flatnonzero(core)
            }
            structures = [r for r in data.get("structures", []) if r["accession"] == accession]
            if not structures and accession not in data.get("structures_complete_for", []):
                structures = acquire_structures(
                    reference, cache, core_positions, policy.minimum_reference_coverage
                )
            results = structure_annotations(
                reference,
                rows,
                core_positions,
                structures,
                catalog.parent,
                target,
                policy.minimum_reference_coverage,
            )
            portable["structures_complete_for"].append(accession)
            for file in sorted(target.glob("structure_*/structure.json")):
                frozen = json.loads(file.read_text())
                frozen["path"] = str(Path(frozen["path"]).relative_to(out))
                if frozen["source"] == "pdb":
                    frozen["sifts_path"] = str(file.parent.relative_to(out) / "sifts.xml.gz")
                for key in ["folder", "sites", "structure_key"]:
                    frozen.pop(key, None)
                portable["structures"].append(frozen)
            write_json(out / "reference_catalog.json", portable)
            rendered = []
            for result in results:
                render_structure(result, target, pymol=pymol, render=render)
                rendered.append(result["structure_key"])
            manifest["orthogroups"].append(
                dict(
                    orthogroup=name,
                    accession=accession,
                    status="annotated" if results else "reference_only",
                    structures=rendered,
                )
            )
            write_json(out / "manifest.json", manifest)
        manifest["resource_receipts"] = list(cache.receipts.values())
        manifest["status"] = "completed" if render else "completed_without_rendering"
        write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            status="failed", error=str(exc), resource_receipts=list(cache.receipts.values())
        )
        write_json(out / "manifest.json", manifest)
        raise
    return out


def re_full_accession(value: str) -> bool:
    return bool(value) and all(c.isascii() and (c.isalnum() or c in "_-") for c in value)
