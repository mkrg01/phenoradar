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
_CONTEXT_COLORS = {
    "UniProt_active_sites": "#2274c7",
    "UniProt_binding_sites": "#17844b",
    "PDB_ligands": "#f28e2b",
}
_CONFIDENCE_LEVELS = (
    (0, 50, "#ff7d45", "<50"),
    (50, 70, "#ffdb13", "50 to <70"),
    (70, 90, "#65cbf3", "70 to <90"),
    (90, 101, "#0053d6", "90 to 100"),
)
_POSITION_STATUSES = {"mapped", "structure_residue_differs"}
_LegendKey = list[tuple[str, str, tuple[str, ...]]]

# Standalone script: PyMOL may run in a different environment from PhenoRadar.
_PYMOL_SCRIPT = r"""from pathlib import Path
import json
import math
import numpy as np
from pymol import cgo, cmd

# PyMOL does not bind __file__ to the executed script. CLI sets this working directory.
root = Path.cwd()
scene = json.loads((root / 'scene.json').read_text())
cmd.reinitialize()
cmd.set('max_threads', 1)
cmd.load(str(root / scene['file']), 'structure')
if scene.get('orthogroup_group'):
    cmd.group(scene['orthogroup_group'], 'structure')
title = scene['orthogroup_label']
if scene.get('pdb_search_organism'):
    title += ' | PDB sequence match | ' + scene['pdb_search_organism']
cmd.set_title('structure', 1, title)
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
other_chains = 'structure and not (' + target + ')'
cmd.color('gray90', other_chains)
cmd.set('cartoon_transparency', .65, other_chains)
cmd.bg_color('white')
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)
cmd.set('label_color', 'black')
cmd.set('label_size', 26)
cmd.set('label_outline_color', 'white')
cmd.set('label_shadow_mode', 0)
target_ca = target + ' and polymer.protein and name CA'
proteins = 'structure and polymer.protein'
if not cmd.count_atoms(target_ca, state=1):
    raise RuntimeError('Selected chain has no CA atoms.')
cmd.orient(target_ca, state=1)
cmd.zoom(proteins, buffer=5, state=1, complete=1)
base = list(cmd.get_view())
# Frame the displayed protein chains at the same scale in every rotation.
base[11] *= 1.15
coords = cmd.get_coords(proteins, state=1)
radius = float(np.linalg.norm(coords - base[12:15], axis=1).max()) + 5
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
selected_columns = {selection(site): site['alignment_pos_1based'] for site in selected}
site_selections = []
for site in selected:
    sel = selection(site)
    if cmd.count_atoms(sel + ' and name CA', state=1) != 1:
        raise RuntimeError('Mapped residue does not uniquely identify a CA atom: ' + sel)
    cmd.show('sticks', sel)
    cmd.show('spheres', sel + ' and name CA')
    cmd.set('sphere_scale', .6, sel)
    color = scene['site_colors'].get(str(site['alignment_pos_1based']), '#d62728')
    cmd.color('0x' + color[1:], sel)
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
site_context_overlaps = {}
for kind, name, ring_radius in [('Active site', 'UniProt_active_sites', 1.45),
                                ('Binding site', 'UniProt_binding_sites', 1.85)]:
    residues = sorted({
        selection(site)
        for site in scene.get('functional_sites', [])
        if site['feature_type'] == kind and site['structure_status'] == 'mapped'
    })
    hex_color = scene['context_colors'][name]
    color = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    markers = []
    for residue in residues:
        coordinates = cmd.get_coords(residue + ' and polymer.protein and name CA', state=1)
        if coordinates is None or len(coordinates) != 1:
            print('Functional residue lacks a unique CA atom:', residue)
            continue
        center = coordinates[0].tolist()
        if residue in selected_columns:
            # Orthogonal hoops keep the annotation visible as the scene rotates.
            for normal in [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]:
                markers.extend(cgo.torus(center, normal, ring_radius, color,
                                         cradius=.14, samples=20, csamples=6))
            site_context_overlaps.setdefault(selected_columns[residue], []).append(name)
        else:
            markers.extend([cgo.COLOR, *color, cgo.SPHERE, *center, .75])
    if markers:
        cmd.load_cgo(markers, name, state=1)
        context_layers.append(name)

ligands = {
    (row['chain'], row['label_asym_id'], str(row['auth_seq_id']) + row['insertion_code'],
     row['component_id']) for row in scene.get('pdb_ligands', [])
}
matched = set()
if ligands:
    indices = []
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
        cmd.create('PDB_ligands', 'structure and index ' + '+'.join(indices),
                   source_state=1, target_state=1)
        cmd.hide('everything', 'PDB_ligands')
        cmd.show('sticks', 'PDB_ligands')
        cmd.show('spheres', 'PDB_ligands')
        cmd.color('0x' + scene['context_colors']['PDB_ligands'][1:], 'PDB_ligands')
        cmd.set('sphere_scale', .19, 'PDB_ligands')
        context_layers.append('PDB_ligands')

overview = 'functional_context' if context_layers else 'selected'
pages = [overview]
if scene['source'] == 'alphafold':
    pages.append('confidence')
for page in pages:
    if page == 'confidence':
        for layer in context_layers:
            cmd.disable(layer)
        cmd.color('gray80', target)
        atoms = cmd.get_model(target, state=1).atom
        for lo, hi, color, _ in scene['confidence_levels']:
            indices = [str(atom.index) for atom in atoms if lo <= atom.b < hi]
            if indices:
                cmd.color('0x' + color[1:], target + ' and index ' + '+'.join(indices))
    for name, view in views.items():
        cmd.set_view(view)
        if page == overview:
            cmd.scene(name, 'store')
        cmd.png(str(root / (page + '_' + name + '.png')), width=1000, height=1000,
                dpi=300, ray=1, quiet=1)
    if page == 'functional_context':
        cmd.set_view(views['front'])
        cmd.scene('functional_context', 'store')
cmd.scene('front', 'recall')
cmd.save(str(root / 'sites.pse'))
(root / 'rendering.json').write_text(json.dumps(dict(
    pymol_version=cmd.get_version()[0], views=views, functional_layers=context_layers,
    site_context_overlaps=site_context_overlaps,
    ligand_ids=sorted({row[3] for row in matched}),
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
            context_colors=_CONTEXT_COLORS,
            confidence_levels=_CONFIDENCE_LEVELS,
            functional_sites=record.get("functional_sites", []),
            pdb_ligands=record.get("pdb_ligands", []),
            orthogroup_label=f"{annotation} ({orthogroup})" if annotation else orthogroup,
            pdb_search_organism=(
                record["reference_organism"] if record.get("search_subject_start") is not None
                else None
            ),
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
    from reportlab.lib.colors import HexColor, black
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    rendering = json.loads((folder / "rendering.json").read_text())
    width = 180 * mm
    margin = 12
    gutter = 8
    panel_size = (width - 2 * margin - 2 * gutter) / 3
    panel_height = panel_size + 14
    key_width = (width - 2 * margin) / 2
    canvas = Canvas(str(folder / "sites.pdf"))
    structure_id = record["structure_id"]
    if record["source"] == "pdb":
        structure_id = structure_id.upper()
    orthogroup = record["sites"][0]["orthogroup"]
    annotation = record.get("orthogroup_annotation")
    title = f"OrthoDB {orthogroup} | {structure_id} | Chain {record['chain']}"
    canvas.setTitle(f"{annotation} | {title}" if annotation else title)
    canvas.setAuthor("PhenoRadar")

    overlaps = rendering["site_context_overlaps"]
    mapped = [site for site in record["sites"] if site["structure_status"] in _POSITION_STATUSES]
    site_key: _LegendKey = []
    for site in mapped:
        column = site["alignment_pos_1based"]
        if column not in site_colors:
            continue
        pdb_residue = (
            f"{site['structure_residue']}{site['auth_seq_id']}{site['insertion_code'] or ''}"
        )
        label = f"MSA {column}  |  {site['chain']}:{pdb_residue}"
        reference = f"{site['reference_residue']}{site['reference_pos_1based']}"
        if reference != pdb_residue:
            label += f"  |  UniProt {reference}"
        if site.get("pss") is not None:
            label += f"  |  PSS {float(site['pss']):.3g}"
        rings = tuple(_CONTEXT_COLORS[layer] for layer in overlaps.get(str(column), []))
        site_key.append((site_colors[column], label, rings))
    if mapped and not site_key:
        site_key.append((_SITE_COLORS[0], "Selected sites", ()))

    context_labels = {
        "UniProt_active_sites": "UniProt active sites",
        "UniProt_binding_sites": "UniProt binding sites",
        "PDB_ligands": "Deposited ligands",
    }
    if rendering["ligand_ids"]:
        with_ids = "Deposited ligands (" + ", ".join(rendering["ligand_ids"]) + ")"
        if canvas.stringWidth(with_ids, "Helvetica", 8) <= key_width - 14:
            context_labels["PDB_ligands"] = with_ids
    for layer in {layer for layers in overlaps.values() for layer in layers}:
        context_labels[layer] += " (rings at selected sites)"
    context_key: _LegendKey = [
        (_CONTEXT_COLORS[layer], context_labels[layer], ())
        for layer in rendering["functional_layers"]
    ]
    confidence_key: _LegendKey = [
        (color, f"pLDDT {label}", ()) for _, _, color, label in _CONFIDENCE_LEVELS
    ]
    directions = ["front", "right", "top", "back", "left", "bottom"]
    header_height = 29 if annotation else 16
    for page in rendering["pages"]:
        keys = [confidence_key] if page == "confidence" else [site_key, context_key]
        legend_items = sum((len(key) + 1) // 2 for key in keys)
        legend_height = 13 * legend_items
        height = 2 * panel_height + 2 * margin + header_height + legend_height
        canvas.setPageSize((width, height))
        canvas.setFillColor(black)
        title_y = height - margin - 7
        if annotation:
            name_width = canvas.stringWidth(annotation, "Helvetica-Bold", 10)
            canvas.setFont("Helvetica-Bold", min(10, 10 * (width - 2 * margin) / name_width))
            canvas.drawString(margin, title_y, annotation)
            canvas.setFont("Helvetica", 8)
            title_y -= 13
        else:
            canvas.setFont("Helvetica-Bold", 10)
        header = title + (" | Local confidence" if page == "confidence" else "")
        canvas.drawString(margin, title_y, header)

        for i, direction in enumerate(directions):
            image = folder / f"{page}_{direction}.png"
            if not image.is_file():
                raise SiteAnnotationError(f"Missing six-view panel: {image}")
            row, column = divmod(i, 3)
            x = margin + column * (panel_size + gutter)
            y = height - margin - header_height - panel_height * (row + 1)
            canvas.setFont("Helvetica", 8)
            canvas.drawCentredString(x + panel_size / 2, y + panel_size + 3, direction.title())
            canvas.drawImage(ImageReader(str(image)), x, y, width=panel_size, height=panel_size)

        y = margin + legend_height - 8
        for key in keys:
            for i, (color, label, rings) in enumerate(key):
                row, column = divmod(i, 2)
                x = margin + 6 + column * key_width
                baseline = y - 13 * row
                canvas.setFillColor(HexColor(color))
                if key is site_key:
                    canvas.circle(x, baseline + 2.5, 3, stroke=0, fill=1)
                    for index, ring in enumerate(rings):
                        canvas.setStrokeColor(HexColor(ring))
                        canvas.setLineWidth(.7)
                        canvas.circle(x, baseline + 2.5, 4 + 1.3 * index, stroke=1, fill=0)
                else:
                    canvas.rect(x - 3, baseline - .5, 6, 6, stroke=0, fill=1)
                canvas.setFillColor(black)
                canvas.setFont("Helvetica", 8)
                canvas.drawString(x + 9, baseline, label)
            y -= 13 * ((len(key) + 1) // 2)
        canvas.showPage()
    canvas.save()
