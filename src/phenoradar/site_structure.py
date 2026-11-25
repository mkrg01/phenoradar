"""Residue-level structure mapping and deterministic selection of experimental chains."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import io
import json
import math
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from phenoradar.amino_acid import ALPHABET
from phenoradar.site_functional_context import deposited_ligands
from phenoradar.site_reference_resources import ResourceCache, SiteAnnotationError, sha256


def _letter(name: str) -> str:
    from Bio.Data.PDBData import protein_letters_3to1_extended

    return str(protein_letters_3to1_extended.get(name.upper(), "X"))


def _residue_id(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", value)
    if not match:
        raise SiteAnnotationError(f"Unsupported PDB residue identifier: {value}")
    return int(match[1]), match[2]


def sifts_mapping(path: Path, accession: str, chain: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(128 * 1024 * 1024 + 1)
    if len(raw) > 128 * 1024 * 1024 or b"<!ENTITY" in raw:
        raise SiteAnnotationError("Oversized or unsafe SIFTS XML.")
    root = ET.fromstring(raw)
    rows = []
    for residue in root.iter():
        if residue.tag.rsplit("}", 1)[-1] != "residue":
            continue
        refs = [x.attrib for x in residue if x.tag.rsplit("}", 1)[-1] == "crossRefDb"]
        pdb = [x for x in refs if x.get("dbSource") == "PDB" and x.get("dbChainId") == chain]
        uni = [
            x
            for x in refs
            if x.get("dbSource") == "UniProt" and x.get("dbAccessionId") == accession
        ]
        if not pdb or not uni:
            continue
        if len(pdb) != 1 or len(uni) != 1:
            raise SiteAnnotationError("Ambiguous SIFTS residue correspondence.")
        if pdb[0]["dbResNum"] in {"null", "None", "?", ""}:
            continue
        number, insertion = _residue_id(pdb[0]["dbResNum"])
        unobserved = any(
            x.tag.rsplit("}", 1)[-1] == "residueDetail" and (x.text or "").strip() == "Not_Observed"
            for x in residue
        )
        rows.append(
            dict(
                reference_pos_1based=int(uni[0]["dbResNum"]),
                chain=chain,
                auth_seq_id=number,
                insertion_code=insertion,
                residue=_letter(pdb[0]["dbResName"]),
                observed=not unobserved,
            )
        )
    keys = [(r["reference_pos_1based"], r["chain"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SiteAnnotationError("Duplicated UniProt positions within a SIFTS chain.")
    return rows


def structure_atoms(
    path: Path, *, coordinate_format: str | None = None,
    cif_data: dict[str, Any] | None = None,
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], int]:
    """First model's CA atoms; author chains, residue IDs, and insertion codes stay distinct."""
    cif_module: Any = importlib.import_module("Bio.PDB.MMCIF2Dict")

    atoms = {}
    if (coordinate_format or path.suffix.lower().removeprefix(".")) in {"cif", "mmcif"}:
        data = cif_data if cif_data is not None else cif_module.MMCIF2Dict(str(path))
        count = len(data["_atom_site.id"])
        model_numbers = data.get("_atom_site.pdbx_PDB_model_num", ["1"] * count)
        first = int(model_numbers[0])
        label_chains = data.get("_atom_site.label_asym_id", data["_atom_site.auth_asym_id"])
        for i in range(count):
            if data["_atom_site.label_atom_id"][i] != "CA" or int(model_numbers[i]) != first:
                continue
            label = data["_atom_site.label_seq_id"][i]
            if label in {".", "?"}:
                continue
            chain = data["_atom_site.auth_asym_id"][i]
            number = int(data["_atom_site.auth_seq_id"][i])
            insertion = data.get("_atom_site.pdbx_PDB_ins_code", ["?"] * count)[i]
            insertion = "" if insertion in {".", "?"} else insertion
            key = chain, number, insertion
            record = dict(
                residue=_letter(data["_atom_site.label_comp_id"][i]),
                label_seq_id=int(label),
                label_chain=label_chains[i],
                b=float(data["_atom_site.B_iso_or_equiv"][i]),
                xyz=[float(data[f"_atom_site.Cartn_{a}"][i]) for a in "xyz"],
            )
            if key not in atoms:  # First alternate location, in deposited order.
                atoms[key] = record
    else:
        first = 1
        active_model = 1
        model_seen = False
        label_by_chain: dict[str, int] = {}
        for line in path.read_text().splitlines():
            if line.startswith("MODEL "):
                active_model = int(line[10:14].strip())
                if not model_seen:
                    first = active_model
                    model_seen = True
            elif line.startswith("ENDMDL"):
                break
            elif (
                line.startswith(("ATOM  ", "HETATM"))
                and active_model == first
                and line[12:16].strip() == "CA"
            ):
                chain, number, insertion = (
                    line[21:22].strip(),
                    int(line[22:26]),
                    line[26:27].strip(),
                )
                residue_name = line[17:20].strip()
                if _letter(residue_name) == "X" and residue_name != "UNK":
                    continue
                key = chain, number, insertion
                if key not in atoms:
                    label_by_chain[chain] = label_by_chain.get(chain, 0) + 1
                    atoms[key] = dict(
                        residue=_letter(residue_name),
                        label_seq_id=label_by_chain[chain],
                        b=float(line[60:66]),
                        xyz=[float(line[start : start + 8]) for start in [30, 38, 46]],
                    )
    if not atoms:
        raise SiteAnnotationError(f"Structure has no protein CA atoms: {path}")
    return atoms, first


def pdbe_cif_sites(
    path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int, str], dict[str, Any]], int]:
    """Load the PDBe mmCIF file containing coordinates together with residue-level SIFTS."""
    cif_module: Any = importlib.import_module("Bio.PDB.MMCIF2Dict")
    data = cif_module.MMCIF2Dict(str(path))
    atoms, first = structure_atoms(path, coordinate_format="cif", cif_data=data)
    return data, atoms, first


def pdbe_sifts_mapping(
    data: dict[str, Any],
    atoms: dict[tuple[str, int, str], dict[str, Any]],
    accession: str,
    chain: str,
) -> list[dict[str, Any]]:
    prefix = "_pdbx_sifts_xref_db."
    if prefix + "unp_acc" not in data:
        return []
    scheme = "_pdbx_poly_seq_scheme."
    label_chains = {label for label, author in zip(
        data[scheme + "asym_id"], data[scheme + "pdb_strand_id"], strict=True,
    ) if author == chain}
    atom_ids = {(atom["label_chain"], str(atom["label_seq_id"])): key
                for key, atom in atoms.items() if key[0] == chain}
    mappings: dict[int, tuple[tuple[str, str], dict[str, Any]]] = {}
    columns = ["unp_acc", "asym_id", "seq_id", "unp_num", "mon_id", "observed"]
    uniprot_residues = data.get(prefix + "unp_res", ["?"] * len(data[prefix + "unp_acc"]))
    for acc, label, seq, pos, residue, observed, uniprot_residue in zip(
        *(data[prefix + name] for name in columns), uniprot_residues, strict=True,
    ):
        if acc != accession or label not in label_chains or pos in {".", "?"}:
            continue
        position = int(pos)
        label_key = label, seq
        if position in mappings and mappings[position][0] != label_key:
            raise SiteAnnotationError("Ambiguous SIFTS residue correspondence in PDBe mmCIF.")
        key = atom_ids.get(label_key)
        mappings[position] = (label_key, dict(
            reference_pos_1based=position, chain=chain,
            auth_seq_id=key[1] if key is not None else None,
            insertion_code=key[2] if key is not None else "",
            residue=_letter(residue), observed=observed == "y" and key is not None,
            uniprot_residue=uniprot_residue if uniprot_residue not in {".", "?", "X"} else None,
        ))
    return [row for _, row in mappings.values()]


def _coverage(rows: list[dict[str, Any]], core: set[int]) -> float:
    observed = {r["reference_pos_1based"] for r in rows if r["observed"]}
    return len(observed & core) / len(core) if core else 0.0


def _identity(rows: list[dict[str, Any]], sequence: str) -> float:
    observed = [r for r in rows if r["observed"]]
    if any(not 1 <= r["reference_pos_1based"] <= len(sequence) for r in rows):
        raise SiteAnnotationError("Structure maps outside the full reference sequence.")
    return (
        sum(r["residue"] == sequence[r["reference_pos_1based"] - 1] for r in observed)
        / len(observed)
        if observed
        else 0.0
    )


def structure_order(row: dict[str, Any]) -> tuple[Any, ...]:
    resolution = row.get("resolution")
    return (
        -row["core_coverage"],
        -row["sequence_identity"],
        float(resolution) if resolution is not None else math.inf,
        row["structure_id"],
        row["chain"],
    )


def acquire_structures(
    reference: dict[str, Any],
    cache: ResourceCache,
    core: set[int],
    minimum_coverage: float,
    *,
    include_pdb: bool = True,
    required_positions: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Screen all PDB/SIFTS candidates; fetch coordinates only for chosen method representatives."""
    accession = reference["accession"]
    best = None
    if include_pdb:
        best = cache.json(
            "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/" + accession, missing_ok=True
        )
    candidates = []
    if best is not None:
        if not isinstance(best, dict) or accession not in best:
            raise SiteAnnotationError("Unexpected PDBe structure-list response.")
        for row in best[accession]:
            pdb = row["pdb_id"].lower()
            chain = row["chain_id"]
            path = cache.fetch(
                "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/" + pdb + ".xml.gz"
            )
            assert path is not None
            mapping = sifts_mapping(path, accession, chain)
            candidates.append(
                dict(
                    accession=accession,
                    source="pdb",
                    structure_id=pdb,
                    method=row["experimental_method"],
                    resolution=row.get("resolution"),
                    chain=chain,
                    sifts_path=str(path),
                    sifts_sha256=sha256(path),
                    core_coverage=_coverage(mapping, core),
                    sequence_identity=_identity(mapping, reference["sequence"]),
                )
            )
    # The complete screening table is retained, including rejected partial structures.
    results = []
    for method in sorted({r["method"] for r in candidates}):
        eligible = [
            r
            for r in candidates
            if r["method"] == method and r["core_coverage"] >= minimum_coverage
        ]
        if eligible:
            row = min(eligible, key=structure_order).copy()
            path = cache.fetch("https://files.rcsb.org/download/" + row["structure_id"] + ".cif")
            assert path is not None
            row.update(path=str(path), sha256=sha256(path), format="cif")
            results.append(row)
    af = cache.json("https://alphafold.ebi.ac.uk/api/prediction/" + accession, missing_ok=True)
    if af is not None:
        if not isinstance(af, list):
            raise SiteAnnotationError("Unexpected AlphaFold API response.")
        target_positions = core if required_positions is None else required_positions
        for row in sorted(af, key=lambda r: r["entryId"]):
            if row.get("uniprotAccession") != accession:
                raise SiteAnnotationError("AlphaFold accession differs from selected reference.")
            start = int(row.get("uniprotStart", row.get("sequenceStart", 1)))
            end = int(row.get("uniprotEnd", row.get("sequenceEnd", 0)))
            if row.get("sequence") != reference["sequence"][start - 1 : end]:
                raise SiteAnnotationError(
                    "AlphaFold sequence differs from frozen UniProt reference."
                )
            af_coverage = len(core.intersection(range(start, end + 1))) / len(core) if core else 0.0
            candidates.append(
                dict(
                    accession=accession,
                    source="alphafold",
                    structure_id=row["entryId"],
                    method="AlphaFold",
                    resolution=None,
                    chain=row.get("chainId", "A"),
                    core_coverage=af_coverage,
                    sequence_identity=1.0,
                )
            )
            if not target_positions.intersection(range(start, end + 1)):
                continue
            path = cache.fetch(row["cifUrl"])
            assert path is not None
            results.append(
                dict(
                    accession=accession,
                    source="alphafold",
                    structure_id=row["entryId"],
                    method="AlphaFold",
                    resolution=None,
                    chain=row.get("chainId", "A"),
                    reference_start=start,
                    reference_end=end,
                    path=str(path),
                    sha256=sha256(path),
                    format="cif",
                    model_version=row["latestVersion"],
                )
            )
    reference["_structure_screening"] = candidates
    return results


def map_structure_sites(
    reference: dict[str, Any],
    sites: list[dict[str, Any]],
    record: dict[str, Any],
    atoms: dict[tuple[str, int, str], dict[str, Any]],
    pdb_mapping: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Map one chain and reconcile SIFTS observation flags with deposited CA atoms."""
    chain = record["chain"]
    if record["source"] == "pdb":
        if pdb_mapping is None:
            raise SiteAnnotationError("PDB structure has no SIFTS mapping.")
        mapping = pdb_mapping
    elif record["source"] == "alphafold":
        start, end = int(record["reference_start"]), int(record["reference_end"])
        if not 1 <= start <= end <= len(reference["sequence"]):
            raise SiteAnnotationError("Invalid AlphaFold reference interval.")
        mapping = []
        for (atom_chain, number, insertion), atom in atoms.items():
            if atom_chain != chain:
                continue
            position = start + atom["label_seq_id"] - 1
            if position > end or atom["residue"] != reference["sequence"][position - 1]:
                raise SiteAnnotationError("AlphaFold coordinates/reference sequence disagree.")
            mapping.append(
                dict(
                    reference_pos_1based=position,
                    chain=chain,
                    auth_seq_id=number,
                    insertion_code=insertion,
                    observed=True,
                    residue=atom["residue"],
                )
            )
    else:
        raise SiteAnnotationError("Structure source must be pdb or alphafold.")
    coordinate_map = {}
    for residue in mapping:
        key = residue["chain"], residue["auth_seq_id"], residue["insertion_code"]
        atom_at_residue = atoms.get(key)
        observed = residue["observed"] and atom_at_residue is not None
        if (
            observed
            and atom_at_residue is not None
            and atom_at_residue["residue"] != residue["residue"]
        ):
            raise SiteAnnotationError("Observed SIFTS residue differs from structure atoms.")
        coordinate_map[residue["reference_pos_1based"]] = {
            **residue,
            "observed": observed,
            "plddt": atom_at_residue["b"]
            if observed and atom_at_residue is not None and record["source"] == "alphafold"
            else None,
        }
    rows = []
    for site in sites:
        site_residue = coordinate_map.get(site["reference_pos_1based"])
        observed = site_residue is not None and site_residue["observed"]
        status = (
            site["reference_status"]
            if site["reference_status"] != "mapped"
            else "outside_structure"
            if site_residue is None
            else "unobserved"
            if not observed
            else "sifts_reference_differs"
            if site_residue.get("uniprot_residue") not in {None, site["reference_residue"]}
            else "structure_residue_differs"
            if site_residue["residue"] != site["reference_residue"]
            else "mapped"
        )
        rows.append(
            dict(
                **site,
                structure_id=record["structure_id"],
                source=record["source"],
                chain=chain,
                auth_seq_id=site_residue["auth_seq_id"] if site_residue is not None else None,
                insertion_code=site_residue["insertion_code"]
                if site_residue is not None
                else None,
                structure_residue=site_residue["residue"] if site_residue is not None else None,
                plddt=site_residue["plddt"] if observed and site_residue is not None else None,
                structure_status=status,
            )
        )
    return rows, coordinate_map


def rank_pdb_site_candidates(
    candidates: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    sites_by_accession: dict[str, list[dict[str, Any]]],
    required_sites: set[int],
    cache: ResourceCache,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Rank chains with SIFTS and verified coordinates at requested sites."""

    @lru_cache(maxsize=2)
    def coordinates(pdb: str) -> Path | None:
        return cache.fetch(
            f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}_updated.cif", missing_ok=True
        )

    @lru_cache(maxsize=2)
    def read_structure(
        path: Path,
    ) -> tuple[dict[str, Any], dict[tuple[str, int, str], dict[str, Any]], int]:
        return pdbe_cif_sites(path)

    audit = []
    eligible = []
    for candidate in sorted(candidates, key=lambda r: (r["structure_id"], r["chain"])):
        accession = candidate["accession"]
        pdb = candidate["structure_id"]
        path = coordinates(pdb)
        observed: set[int] = set()
        reason = "coordinates_unavailable"
        if path is not None:
            cif_data, atoms, _ = read_structure(path)
            mapping = pdbe_sifts_mapping(cif_data, atoms, accession, candidate["chain"])
            rows, _ = map_structure_sites(
                references[accession], sites_by_accession[accession], candidate, atoms, mapping,
            )
            observed = {
                r["alignment_pos_1based"] for r in rows
                if r["structure_status"] in {"mapped", "structure_residue_differs"}
            }
            reason = "no_observed_sites" if not observed else "required_site_unobserved"
            if observed and required_sites <= observed:
                reason = ""
                eligible.append((len(observed), {
                    **candidate, "path": str(path), "sha256": sha256(path),
                    "format": "cif", "sifts_format": "mmcif",
                }))
        audit.append({
            **candidate,
            "n_observed_sites": len(observed),
            "observed_columns": ",".join(map(str, sorted(observed))),
            "exclusion_reason": reason,
        })
    winner = None
    if eligible:
        _, winner = min(eligible, key=lambda item: (
            -item[0], -item[1]["search_score"], item[1]["structure_id"], item[1]["chain"],
            item[1]["accession"],
        ))
    for row in audit:
        row["selected"] = winner is not None and all(
            row[key] == winner[key] for key in ("accession", "structure_id", "chain")
        )
        if not row["selected"] and not row["exclusion_reason"]:
            row["exclusion_reason"] = "lower_site_coverage_or_search_rank"
    return winner, audit


def structure_annotations(
    reference: dict[str, Any],
    sites: list[dict[str, Any]],
    core: set[int],
    structures: list[dict[str, Any]],
    catalog_dir: Path,
    out: Path,
    minimum_coverage: float,
) -> list[dict[str, Any]]:
    from phenoradar.site_reference import _write_table
    from phenoradar.site_reference_resources import write_json

    candidates = []
    for index, record in enumerate(structures):
        record = dict(record)
        source_path = (catalog_dir / record["path"]).resolve()
        if sha256(source_path) != record["sha256"]:
            raise SiteAnnotationError("Structure checksum differs from its catalog.")
        folder = out / f"structure_{index:04d}"
        folder.mkdir()
        suffix = record.get("format", source_path.suffix.removeprefix("."))
        if suffix not in {"cif", "mmcif", "pdb"}:
            raise SiteAnnotationError("Structure format must be cif, mmcif, or pdb.")
        path = folder / ("structure." + suffix)
        shutil.copyfile(source_path, path)
        pdb_mapping = None
        cif_data = None
        if record["source"] == "pdb" and record.get("sifts_format") == "mmcif":
            cif_data, atoms, first_model = pdbe_cif_sites(path)
            pdb_mapping = pdbe_sifts_mapping(
                cif_data, atoms, reference["accession"], record["chain"],
            )
        else:
            atoms, first_model = structure_atoms(path)
        if record["source"] == "pdb" and pdb_mapping is None:
            sifts_path = (catalog_dir / record["sifts_path"]).resolve()
            if sha256(sifts_path) != record["sifts_sha256"]:
                raise SiteAnnotationError("SIFTS checksum differs from its catalog.")
            pdb_mapping = sifts_mapping(sifts_path, reference["accession"], record["chain"])
            shutil.copyfile(sifts_path, folder / "sifts.xml.gz")
        functional = []
        for feature in reference.get("uniprot_site_features", []):
            for position in range(feature["feature_start"], feature["feature_end"] + 1):
                letter = reference["sequence"][position - 1]
                functional.append(dict(
                    **feature, accession=reference["accession"], reference_pos_1based=position,
                    reference_residue=letter,
                    reference_status="mapped" if letter in ALPHABET else "reference_unknown",
                ))
        mapped, coordinate_map = map_structure_sites(
            reference, sites + functional, record, atoms, pdb_mapping,
        )
        ligands = (deposited_ligands(path, record["structure_id"], first_model, cif_data)
                   if record["source"] == "pdb" else [])
        record.update(
            folder=str(folder),
            path=str(path),
            structure_key=folder.name,
            first_model=first_model,
            sites=mapped[:len(sites)],
            functional_sites=mapped[len(sites):],
            pdb_ligands=ligands,
            orthogroup_annotation=reference.get("orthogroup_annotation"),
            site_scope=sites[0].get("analysis_scope", "final"),
            reference_selection=reference.get("selection_mode", "automatic"),
            reference_qualified=reference.get("reference_qualified", True),
            core_coverage=_coverage(list(coordinate_map.values()), core),
            sequence_identity=_identity(list(coordinate_map.values()), reference["sequence"]),
        )
        if (
            "core_coverage" in structures[index]
            and abs(record["core_coverage"] - structures[index]["core_coverage"]) > 1e-12
        ):
            raise SiteAnnotationError("SIFTS observed coverage disagrees with structure atoms.")
        candidates.append(record)
    chosen = []
    for method in sorted({r["method"] for r in candidates if r["source"] == "pdb"}):
        eligible = [
            r
            for r in candidates
            if r["method"] == method and r["core_coverage"] >= minimum_coverage
        ]
        if eligible:
            chosen.append(min(eligible, key=structure_order))
    chosen.extend(r for r in candidates if r["source"] == "alphafold")
    selected_paths = {r["folder"] for r in chosen}
    selected_keys = {(r["source"], r["structure_id"], r["chain"]) for r in chosen}
    screening = reference.get("_structure_screening", []) or candidates
    _write_table(
        out / "structure_candidates.tsv",
        [
            dict(
                {
                    k: r.get(k)
                    for k in [
                        "structure_id",
                        "source",
                        "method",
                        "chain",
                        "resolution",
                        "core_coverage",
                        "sequence_identity",
                    ]
                },
                selected=(r["source"], r["structure_id"], r["chain"]) in selected_keys,
                exclusion_reason=(
                    ""
                    if (r["source"], r["structure_id"], r["chain"]) in selected_keys
                    else "no_core_overlap"
                    if r["source"] == "alphafold"
                    else "low_core_coverage"
                    if r["core_coverage"] < minimum_coverage
                    else "lower_method_rank"
                ),
            )
            for r in screening
        ],
    )
    for record in candidates:
        folder = Path(record["folder"])
        record["selected"] = record["folder"] in selected_paths
        write_json(folder / "structure.json", record)
        _write_table(folder / "site_structure_map.tsv", record["sites"])
        if record["functional_sites"]:
            _write_table(folder / "functional_site_map.tsv", record["functional_sites"])
        if record["pdb_ligands"]:
            _write_table(folder / "pdb_ligands.tsv", record["pdb_ligands"])
    write_json(
        out / "structure_selection.json",
        dict(
            policy="one_per_experimental_method_plus_same_reference_alphafold",
            selected=[r["structure_key"] for r in chosen],
            screening_sha256=hashlib.sha256(
                json.dumps(screening, sort_keys=True, default=str).encode()
            ).hexdigest(),
        ),
    )
    return chosen
