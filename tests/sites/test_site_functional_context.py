from __future__ import annotations

import json
from pathlib import Path

from phenoradar.site_functional_context import deposited_ligands, uniprot_site_features


def test_uniprot_features_require_exact_canonical_coordinates() -> None:
    evidence = [{"evidenceCode": "ECO:0000269", "source": "PubMed", "id": "123"}]
    location = {"start": {"value": 2}, "end": {"value": 3}}
    rows = uniprot_site_features({
        "primaryAccession": "P00001",
        "features": [
            {"type": "Binding site", "location": location, "evidences": evidence,
             "ligand": {"name": "zinc", "id": "ChEBI:CHEBI:29105"}},
            {"type": "Active site", "location": {**location, "sequence": "P00001-2"}},
            {"type": "Active site", "location": {**location, "end": {"value": 10}}},
            {"type": "Active site", "location": {**location,
                                                "start": {"value": 1, "modifier": "UNCERTAIN"}}},
            {"type": "Region", "location": location},
        ],
    }, 9)
    assert len(rows) == 1
    assert rows[0]["feature_type"] == "Binding site"
    assert (rows[0]["feature_start"], rows[0]["feature_end"]) == (2, 3)
    assert rows[0]["ligand_id"] == "ChEBI:CHEBI:29105"
    assert json.loads(rows[0]["evidence"]) == evidence


def test_deposited_mmcif_ligands_keep_instances_and_exclude_water_and_other_models() -> None:
    columns = {
        "id": list("123456"), "pdbx_PDB_model_num": list("111112"),
        "group_PDB": ["HETATM"] * 6,
        "label_entity_id": ["2", "2", "2", "3", "1", "2"],
        "label_seq_id": [".", ".", ".", ".", "10", "."],
        "label_comp_id": ["PLP", "PLP", "PLP", "HOH", "MSE", "PLP"],
        "auth_asym_id": ["A"] * 6, "label_asym_id": ["B", "B", "C", "D", "A", "E"],
        "auth_seq_id": ["4", "4", "4", "8", "10", "4"],
    }
    data = {"_atom_site." + key: value for key, value in columns.items()}
    data.update({"_pdbx_entity_nonpoly.entity_id": ["2", "3"],
                 "_pdbx_entity_nonpoly.name": ["pyridoxal phosphate", "water"]})
    rows = deposited_ligands(Path("already_parsed.cif"), "6pk1", 1, data)
    assert [(r["label_asym_id"], r["component_id"]) for r in rows] == [("B", "PLP"), ("C", "PLP")]
