"""UniProt functional features and non-polymer components in deposited coordinates."""

from __future__ import annotations

import json
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

_WATER = {"HOH", "DOD", "WAT", "H2O"}
type _LigandInstance = tuple[str, str, str, str, str]


def uniprot_site_features(entry: dict[str, Any], length: int) -> list[dict[str, Any]]:
    """Keep exact active/binding positions on the canonical sequence and their evidence."""
    rows = []
    for feature in entry.get("features", []):
        kind = feature.get("type")
        if kind not in {"Active site", "Binding site"}:
            continue
        location = feature.get("location", {})
        if location.get("sequence") not in (None, entry["primaryAccession"]):
            continue
        start, end = (location.get(key, {}) for key in ("start", "end"))
        if any(p.get("modifier", "EXACT") != "EXACT" for p in (start, end)):
            continue
        first, last = start.get("value"), end.get("value")
        if (
            not isinstance(first, int) or not isinstance(last, int)
            or not 1 <= first <= last <= length
        ):
            continue
        ligand = feature.get("ligand") or {}
        rows.append(dict(
            feature_type=kind, feature_start=first, feature_end=last,
            description=feature.get("description"), ligand_name=ligand.get("name"),
            ligand_id=ligand.get("id"),
            evidence=json.dumps(feature.get("evidences") or [], sort_keys=True),
        ))
    return rows


def _cif_ligands(data: dict[str, list[str]], model: int) -> dict[_LigandInstance, str]:
    prefix = "_atom_site."
    entity = "_pdbx_entity_nonpoly."
    nonpolymer_ids = data.get(entity + "entity_id", [])
    declared = dict(zip(
        nonpolymer_ids, data.get(entity + "name", [""] * len(nonpolymer_ids)), strict=True,
    ))
    count = len(data[prefix + "id"])

    def column(key: str, default: str = "") -> list[str]:
        return data.get(prefix + key, [default] * count)

    columns = zip(
        column("pdbx_PDB_model_num", "1"), column("group_PDB"), column("label_entity_id"),
        column("label_seq_id"), column("label_comp_id"), column("auth_asym_id"),
        column("label_asym_id"), column("auth_seq_id"), column("pdbx_PDB_ins_code"), strict=True,
    )
    component_ids = data.get("_chem_comp.id", [])
    names = dict(zip(
        component_ids, data.get("_chem_comp.name", [""] * len(component_ids)), strict=True,
    ))
    ligands = {}
    for state, group, entity_id, seq, code, chain, segment, number, ins in columns:
        nonpolymer = entity_id in declared if declared else group == "HETATM" and seq in {".", "?"}
        if int(state) != model or not nonpolymer or code in _WATER or number in {"", ".", "?"}:
            continue
        key = chain, segment, number, "" if ins in {".", "?"} else ins, code
        ligands[key] = declared.get(entity_id) or names.get(code, "")
    return ligands


def _pdb_ligands(path: Path, model: int) -> dict[_LigandInstance, str]:
    modified = set()
    names: dict[str, list[str]] = defaultdict(list)
    ligands = {}
    state = 1
    for line in path.read_text().splitlines():
        if line.startswith("MODRES"):
            modified.add((line[16:17].strip(), line[18:22].strip(), line[22:23].strip()))
        elif line.startswith("HETNAM"):
            names[line[11:14].strip()].append(line[15:].strip())
        elif line.startswith("MODEL "):
            state = int(line[10:14])
        elif line.startswith("ENDMDL"):
            break
        elif line.startswith("HETATM") and state == model:
            chain, number, ins = line[21:22].strip(), line[22:26].strip(), line[26:27].strip()
            code = line[17:20].strip()
            if code not in _WATER and (chain, number, ins) not in modified:
                ligands[(chain, "", number, ins, code)] = " ".join(names[code])
    return ligands


def deposited_ligands(
    path: Path, structure_id: str, model: int, cif_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """List unique observed non-water components in the displayed experimental model."""
    if path.suffix in {".cif", ".mmcif"}:
        if cif_data is None:
            cif: Any = import_module("Bio.PDB.MMCIF2Dict")
            cif_data = cif.MMCIF2Dict(str(path))
        assert cif_data is not None
        ligands = _cif_ligands(cif_data, model)
    else:
        ligands = _pdb_ligands(path, model)
    return [
        dict(structure_id=structure_id, model=model, chain=chain, label_asym_id=segment,
             auth_seq_id=number, insertion_code=ins, component_id=code, name=" ".join(name.split()))
        for (chain, segment, number, ins, code), name in sorted(ligands.items())
    ]
