"""Reproducible PyMOL scenes and six-view, publication-readable PDF reports."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from phenoradar.site_reference_resources import SiteAnnotationError, write_json

_SITE_COLORS = (
    "#d62728", "#6941a5", "#007f72", "#9d6500",
    "#1964ad", "#b02a78", "#617b19", "#49545c",
)
_POSITION_STATUSES = {"mapped", "structure_residue_differs"}

# Standalone script: PyMOL may run in a different environment from PhenoRadar.
_PYMOL_SCRIPT = r"""from pathlib import Path
import json
import math
import numpy as np
from pymol import cmd

# PyMOL does not bind __file__ to the executed script. CLI sets this working directory.
root = Path.cwd()
scene = json.loads((root / 'scene.json').read_text())
cmd.reinitialize()
cmd.set('max_threads', 1)
cmd.load(str(root / scene['file']), 'structure')
if scene.get('orthogroup_group'):
    cmd.group(scene['orthogroup_group'], 'structure')
    cmd.set_title('structure', 1, scene['orthogroup_label'])
cmd.frame(1)
# Keep the first deposited alternate location for each protein atom.
seen, duplicates = set(), []
for atom in cmd.get_model('structure and polymer.protein', state=1).atom:
    key = (atom.chain, atom.segi, atom.resi, atom.name)
    if key in seen:
        duplicates.append(str(atom.index))
    seen.add(key)
if duplicates:
    cmd.remove('structure and index ' + '+'.join(duplicates))
cmd.hide('everything', 'all')
def quote(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'
target = 'model structure and chain ' + quote(scene['chain'])
cmd.show('cartoon', 'structure and polymer.protein')
cmd.color('gray80', 'structure')
cmd.set('cartoon_transparency', .65, 'structure and not (' + target + ')')
cmd.bg_color('white')
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)
cmd.set('label_color', 'black')
cmd.set('label_size', 26)
cmd.set('label_outline_color', 'white')
cmd.set('label_shadow_mode', 0)
coords = cmd.get_coords(target + ' and name CA', state=1)
if coords is None or len(coords) == 0:
    raise RuntimeError('Selected chain has no CA atoms.')
cmd.orient(target + ' and name CA', state=1)
cmd.zoom(target, buffer=5, state=1, complete=1)
base = list(cmd.get_view())
# Fixed camera scale for all rotations, with room for the full target chain.
base[11] *= 1.3
radius = max(float(np.linalg.norm(x - np.mean(coords, axis=0))) for x in coords) + 5
base[15], base[16] = -base[11] - 1.5 * radius, -base[11] + 1.5 * radius
cmd.set_view(base)
def rx(degrees):
    a = math.radians(degrees)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
                     [0, math.sin(a), math.cos(a)]])
def ry(degrees):
    a = math.radians(degrees)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                     [-math.sin(a), 0, math.cos(a)]])
rotations = [('front', np.eye(3)), ('back', ry(180)), ('right', ry(90)),
             ('left', ry(-90)), ('top', rx(90)), ('bottom', rx(-90))]
views = {}
for name, rotation in rotations:
    assert abs(np.linalg.det(rotation) - 1) < 1e-12
    view = base.copy()
    view[:9] = (rotation @ np.array(base[:9]).reshape(3, 3)).ravel().tolist()
    views[name] = view

def selection(site):
    return target + ' and resi ' + quote(str(site['auth_seq_id']) +
                                      (site['insertion_code'] or ''))
selected = [s for s in scene['sites'] if s['structure_status'] in
            ('mapped', 'structure_residue_differs')]
site_selections = []
for site in selected:
    sel = selection(site)
    if cmd.count_atoms(sel + ' and name CA', state=1) != 1:
        raise RuntimeError('Mapped residue does not uniquely identify a CA atom: ' + sel)
    cmd.show('sticks', sel)
    cmd.show('spheres', sel + ' and name CA')
    cmd.set('sphere_scale', .5, sel)
    label = '{}{} / MSA {}'.format(site['reference_residue'],
        site['reference_pos_1based'], site['alignment_pos_1based'])
    cmd.label(sel + ' and name CA', repr(label))
    name = 'MSA_{}'.format(site['alignment_pos_1based'])
    cmd.select(name, sel, enable=0)
    site_selections.append(name)
if site_selections:
    cmd.select('selected_sites', ' or '.join('%' + name for name in site_selections), enable=0)
    cmd.hide('labels', 'selected_sites')

context_layers = []
def context_layer(name, expression, color, representation):
    if not expression or not cmd.count_atoms(expression, state=1):
        return
    cmd.create(name, expression, source_state=1, target_state=1)
    cmd.hide('everything', name)
    cmd.show(representation, name)
    cmd.color(color, name)
    cmd.disable(name)
    context_layers.append(name)

for kind, name, color in [('Active site', 'UniProt_active_sites', '0x2274c7'),
                          ('Binding site', 'UniProt_binding_sites', '0x17844b')]:
    expressions = sorted({
        '(' + selection(site) + ' and polymer.protein and name CA)'
        for site in scene.get('functional_sites', [])
        if site['feature_type'] == kind and site['structure_status'] == 'mapped'
    })
    context_layer(name, ' or '.join(expressions), color, 'spheres')
    if name in context_layers:
        cmd.set('sphere_scale', .45, name)

ligands = {
    (row['chain'], row['label_asym_id'], str(row['auth_seq_id']) + row['insertion_code'],
     row['component_id']) for row in scene.get('pdb_ligands', [])
}
if ligands:
    indices, matched = [], set()
    for atom in cmd.get_model('structure', state=1).atom:
        exact = (atom.chain, atom.segi, atom.resi, atom.resn)
        unsegmented = (atom.chain, '', atom.resi, atom.resn)
        matches = ligands.intersection((exact, unsegmented))
        if matches:
            matched.update(matches)
            indices.append(str(atom.index))
    if matched != ligands:
        print('Recorded PDB ligand instances unavailable in PyMOL:', sorted(ligands - matched))
    if indices:
        context_layer('PDB_ligands', 'structure and index ' + '+'.join(indices), '0xf28e2b',
                      'sticks')
        cmd.show('spheres', 'PDB_ligands')
        cmd.set('sphere_scale', .19, 'PDB_ligands')

pages = ['selected', 'pss', 'selection_frequency']
if scene['source'] == 'alphafold':
    pages.append('confidence')
if context_layers:
    pages.append('functional_context')
scales = {}
for page in pages:
    cmd.color('gray80', target)
    if page == 'functional_context':
        for layer in context_layers:
            cmd.enable(layer)
        for site in selected:
            cmd.color('0xd62728', selection(site))
    elif page == 'confidence':
        atoms = cmd.get_model(target, state=1).atom
        for lo, hi, color in [(0, 50, '0xff7d45'), (50, 70, '0xffdb13'),
                              (70, 90, '0x65cbf3'), (90, 101, '0x0053d6')]:
            indices = [str(atom.index) for atom in atoms if lo <= atom.b < hi]
            if indices:
                cmd.color(color, target + ' and index ' + '+'.join(indices))
    else:
        values = [float(s[page]) for s in scene['sites'] if page != 'selected' and
                  s.get(page) is not None and math.isfinite(float(s[page]))]
        upper = 1.0 if page == 'selection_frequency' else max(values, default=0)
        scales[page] = upper
        for site in selected:
            if page == 'selected':
                color = '0x' + scene['site_colors'].get(
                    str(site['alignment_pos_1based']), '#d62728')[1:]
            elif site.get(page) is None:
                color = 'gray70'
            else:
                fraction = max(0, min(1, float(site[page]) / upper)) if upper > 0 else 0
                rgb = [1 - .25 * fraction, .9 * (1 - fraction), .9 * (1 - fraction)]
                color = 'metric_{}'.format(site['alignment_pos_1based'])
                cmd.set_color(color, rgb)
            cmd.color(color, selection(site))
    for name, view in views.items():
        cmd.set_view(view)
        if page == 'selected':
            cmd.scene(name, 'store')
        cmd.png(str(root / (page + '_' + name + '.png')), width=1000, height=1000,
                dpi=300, ray=1, quiet=1)
    if page == 'functional_context':
        cmd.set_view(views['front'])
        cmd.scene('functional_context', 'store')
cmd.scene('front', 'recall')
cmd.save(str(root / 'sites.pse'))
(root / 'rendering.json').write_text(json.dumps(dict(
    pymol_version=cmd.get_version()[0], views=views, scales=scales,
    functional_layers=context_layers,
    projection='orthographic', pixel_size=[1000, 1000], pages=pages,
    orientation='CA principal axes, saved initial camera; proper rotations only',
    first_model=scene['first_model']), indent=2) + '\n')
cmd.quit()
"""


def render_structure(
    record: dict[str, Any],
    out: Path,
    *,
    pymol: str,
    render: bool,
) -> None:
    folder = Path(record["folder"])
    orthogroup = record["sites"][0]["orthogroup"]
    annotation = record.get("orthogroup_annotation")
    mapped = [site for site in record["sites"] if site["structure_status"] in _POSITION_STATUSES]
    site_colors = {}
    if len(mapped) <= len(_SITE_COLORS):
        site_colors = dict(zip(
            (site["alignment_pos_1based"] for site in mapped), _SITE_COLORS, strict=False,
        ))
    group = None
    if annotation:
        name_part = re.sub(r"[^A-Za-z0-9]+", "_", annotation).strip("_")[:55]
        id_part = re.sub(r"[^A-Za-z0-9_]+", "_", orthogroup)
        group = f"OG_{name_part}_{id_part}" if name_part else f"OG_{id_part}"
    write_json(
        folder / "scene.json",
        dict(
            file=Path(record["path"]).name,
            source=record["source"],
            chain=record["chain"],
            first_model=record["first_model"],
            sites=record["sites"],
            site_colors=site_colors,
            functional_sites=record.get("functional_sites", []),
            pdb_ligands=record.get("pdb_ligands", []),
            orthogroup_label=f"{annotation} ({orthogroup})" if annotation else orthogroup,
            orthogroup_group=group,
        ),
    )
    (folder / "render.py").write_text(_PYMOL_SCRIPT)
    (folder / "sites.pml").write_text("run render.py\n")
    if not render:
        return
    try:
        result = subprocess.run(
            [pymol, "-cq", "render.py"], cwd=folder, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise SiteAnnotationError(f"Cannot run PyMOL: {pymol}") from exc
    (folder / "pymol.log").write_text(result.stdout + result.stderr)
    if (
        result.returncode
        or not (folder / "rendering.json").is_file()
        or not (folder / "sites.pse").is_file()
    ):
        raise SiteAnnotationError(f"PyMOL rendering failed; see {folder / 'pymol.log'}")
    _write_pdf(record, folder, site_colors)


def _write_pdf(record: dict[str, Any], folder: Path, site_colors: dict[int, str]) -> None:
    from html import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    rendering = json.loads((folder / "rendering.json").read_text())
    width, height = A4
    canvas = Canvas(str(folder / "sites.pdf"), pagesize=A4)
    canvas.setTitle("PhenoRadar site mapping: " + record["structure_id"])
    canvas.setAuthor("PhenoRadar")
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 7
    styles["BodyText"].leading = 9
    page_number = 0
    title = record["sites"][0]["orthogroup"] + " | " + record["accession"]
    annotation = record.get("orthogroup_annotation")
    subtitle = "{} | chain {} | {} | model {}".format(
        record["structure_id"], record["chain"], record["method"], record["first_model"]
    )

    def header(label: str) -> None:
        nonlocal page_number
        page_number += 1
        canvas.setFillColor(colors.HexColor("#17324d"))
        if annotation:
            label_width = canvas.stringWidth(annotation, "Helvetica-Bold", 13)
            font_size = min(13, 13 * (width - 56) / max(label_width, 1))
            canvas.setFont("Helvetica-Bold", font_size)
            canvas.drawString(28, height - 27, annotation)
            canvas.setFont("Helvetica", 9)
            canvas.drawString(28, height - 41, title)
        else:
            canvas.setFont("Helvetica-Bold", 13)
            canvas.drawString(28, height - 30, title)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(28, height - (54 if annotation else 46), subtitle)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(28, height - (68 if annotation else 64), label)
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(width - 28, 18, f"PhenoRadar | {page_number}")

    names = ["front", "back", "right", "left", "top", "bottom"]
    labels = dict(
        selected="Selected sites",
        pss="PSS (mean across models)",
        selection_frequency="CV selection frequency",
        confidence="AlphaFold local confidence (pLDDT)",
        functional_context="Functional sites and deposited ligands",
    )
    mapped = [site for site in record["sites"] if site["structure_status"] in _POSITION_STATUSES]
    for page in rendering["pages"]:
        header(record.get("site_scope", "final") + " | " + labels[page])
        for i, direction in enumerate(names):
            image = folder / f"{page}_{direction}.png"
            if not image.is_file():
                raise SiteAnnotationError(f"Missing six-view panel: {image}")
            column, row = i % 2, i // 2
            x = 28 + column * (width - 56) / 2
            y = height - 267 - row * 203
            canvas.setFont("Helvetica", 8)
            canvas.drawString(x + 10, y + 190, direction.capitalize())
            canvas.drawImage(
                ImageReader(str(image)),
                x + 32,
                y,
                width=184,
                height=184,
                preserveAspectRatio=True,
                mask="auto",
            )
        if page in {"pss", "selection_frequency"}:
            for index in range(100):
                fraction = index / 99
                canvas.setFillColor(
                    colors.Color(1 - 0.25 * fraction, 0.9 * (1 - fraction), 0.9 * (1 - fraction))
                )
                canvas.rect(width - 160 + index * 1.25, 102, 1.3, 7, stroke=0, fill=1)
            canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 8)
        if page == "selected":
            if site_colors:
                for index, site in enumerate(mapped):
                    x = 33 + (index % 2) * (width - 56) / 2
                    y = 157 - (index // 2) * 15
                    column = site["alignment_pos_1based"]
                    canvas.setFillColor(colors.HexColor(site_colors[column]))
                    canvas.circle(x, y + 2.5, 3, stroke=0, fill=1)
                    canvas.setFillColor(colors.black)
                    reference = f"{site['reference_residue']}{site['reference_pos_1based']}"
                    structure = (
                        f"{site['chain']}:{site['structure_residue']}{site['auth_seq_id']}"
                        f"{site['insertion_code'] or ''}"
                    )
                    canvas.drawString(x + 9, y, f"MSA {column}   |   {reference}   |   {structure}")
                note = "Site key shows the MSA column, UniProt residue and structure residue."
            else:
                note = "Red marks mapped sites. See the site correspondence table for residue IDs."
        elif page == "confidence":
            note = "pLDDT: orange <50; yellow 50-70; cyan 70-90; blue >=90. Not mapping confidence."
        elif page == "functional_context":
            entries = {
                "UniProt_active_sites": "blue: UniProt active sites",
                "UniProt_binding_sites": "green: UniProt binding sites",
                "PDB_ligands": "orange: PDB ligands",
            }
            parts = ["Red: selected sites", *(entries[n] for n in rendering["functional_layers"])]
            note = ". ".join(parts) + "."
        else:
            upper = rendering["scales"][page]
            note = (
                f"Pale to dark red: 0 to {upper:.4g}; gray: unavailable. Shared scale across views."
            )
        canvas.drawString(28, 90, note)
        canvas.drawString(
            28,
            76,
            f"{len(mapped)}/{len(record['sites'])} selected sites have coordinates; "
            f"core coverage {record['core_coverage']:.1%}.",
        )
        canvas.drawString(
            28,
            62,
            "Six views share one camera scale. Full mapping, omissions and "
            "residue differences follow.",
        )
        canvas.drawString(
            28,
            48,
            "Reference: "
            + record.get("reference_selection", "automatic")
            + ("; QC passed" if record.get("reference_qualified", True) else "; QC FAILED"),
        )
        canvas.showPage()
    # Complete, paginated site listing; no top-N truncation of failed mappings.
    for start in range(0, len(record["sites"]), 23):
        header("Site correspondence and missing-coordinate audit")
        rows: list[list[Any]] = [
            ["MSA", "Reference", "Structure", "PSS", "CV freq.", "pLDDT", "Status"]
        ]
        for site in record["sites"][start : start + 23]:

            def number(value: Any) -> str:
                return "-" if value is None else f"{float(value):.3g}"

            ref = (
                "-"
                if site["reference_pos_1based"] is None
                else str(site["reference_residue"]) + str(site["reference_pos_1based"])
            )
            residue = (
                "-"
                if site["auth_seq_id"] is None
                else str(site["chain"])
                + ":"
                + str(site["auth_seq_id"])
                + (site["insertion_code"] or "")
            )
            status = Paragraph(
                escape(site["structure_status"]).replace("_", " "), styles["BodyText"]
            )
            rows.append(
                [
                    str(site["alignment_pos_1based"]),
                    ref,
                    residue,
                    number(site.get("pss")),
                    number(site.get("selection_frequency")),
                    number(site.get("plddt")),
                    status,
                ]
            )
        table = Table(rows, colWidths=[43, 58, 65, 46, 48, 44, width - 360])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eff5")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#9bafc2")),
                ]
            )
        )
        _, table_height = table.wrap(width - 56, height - 120)
        table.drawOn(canvas, 28, height - 86 - table_height)
        canvas.showPage()
    canvas.save()
