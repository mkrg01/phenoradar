"""Fixed Zappo residue colors (Jalview), independent of alignment conservation."""

ZAPPO_GROUPS = (
    ("ILVAM", "Aliphatic / hydrophobic", "#ffafaf"),
    ("FWY", "Aromatic", "#ffc800"),
    ("KRH", "Positive", "#6464ff"),
    ("DE", "Negative", "#ff0000"),
    ("STNQ", "Hydrophilic", "#00ff00"),
    ("PG", "Conformationally special", "#ff00ff"),
    ("C", "Cysteine", "#ffff00"),
)
RESIDUE_COLORS = {aa: color for residues, _, color in ZAPPO_GROUPS for aa in residues}


def residue_color(residue: object) -> str:
    if residue in {"-", "."}:
        return "#ffffff"
    return RESIDUE_COLORS.get(str(residue), "#dddddd")
