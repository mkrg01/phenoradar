"""Reproducible PyMOL scenes and six-view, publication-readable PDF reports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from phenoradar.site_reference_resources import SiteAnnotationError, write_json

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
cmd.set('label_position', [0, 0, radius + 2])
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
for site in selected:
    sel = selection(site)
    if cmd.count_atoms(sel + ' and name CA', state=1) != 1:
        raise RuntimeError('Mapped residue does not uniquely identify a CA atom: ' + sel)
    cmd.show('sticks', sel)
    cmd.show('spheres', sel + ' and name CA')
    cmd.set('sphere_scale', .35, sel)
    label = '{}{}\nMSA{}'.format(site['reference_residue'],
        site['reference_pos_1based'], site['alignment_pos_1based'])
    cmd.label(sel + ' and name CA', repr(label))

pages = ['selected', 'pss', 'effect_range', 'selection_frequency']
if scene['source'] == 'alphafold':
    pages.append('confidence')
scales = {}
for page in pages:
    cmd.color('gray80', target)
    if page == 'confidence':
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
                color = '0xd62728'
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
    if page == 'selected':
        cmd.scene('front', 'recall')
        cmd.save(str(root / 'sites.pse'))
(root / 'rendering.json').write_text(json.dumps(dict(
    pymol_version=cmd.get_version()[0], views=views, scales=scales,
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
    write_json(
        folder / "scene.json",
        dict(
            file=Path(record["path"]).name,
            source=record["source"],
            chain=record["chain"],
            first_model=record["first_model"],
            sites=record["sites"],
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
    _write_pdf(record, folder)


def _write_pdf(record: dict[str, Any], folder: Path) -> None:
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
    subtitle = "{} | chain {} | {} | model {}".format(
        record["structure_id"], record["chain"], record["method"], record["first_model"]
    )

    def header(label: str) -> None:
        nonlocal page_number
        page_number += 1
        canvas.setFillColor(colors.HexColor("#17324d"))
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(28, height - 30, title)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(28, height - 46, subtitle)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(28, height - 64, label)
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(width - 28, 18, f"PhenoRadar | {page_number}")

    names = ["front", "back", "right", "left", "top", "bottom"]
    labels = dict(
        selected="Selected sites",
        pss="PSS (mean across models)",
        effect_range="R (mean coefficient range)",
        selection_frequency="CV selection frequency",
        confidence="AlphaFold local confidence (pLDDT)",
    )
    for page in rendering["pages"]:
        header(record.get("site_scope", "final") + " | " + labels[page])
        for i, direction in enumerate(names):
            image = folder / f"{page}_{direction}.png"
            if not image.is_file():
                raise SiteAnnotationError(f"Missing six-view panel: {image}")
            column, row = i % 2, i // 2
            x = 28 + column * (width - 56) / 2
            y = height - 286 - row * 218
            canvas.setFont("Helvetica", 8)
            canvas.drawString(x + 10, y + 204, direction.capitalize())
            canvas.drawImage(
                ImageReader(str(image)),
                x + 23,
                y,
                width=198,
                height=198,
                preserveAspectRatio=True,
                mask="auto",
            )
        if page not in {"selected", "confidence"}:
            for index in range(100):
                fraction = index / 99
                canvas.setFillColor(
                    colors.Color(1 - 0.25 * fraction, 0.9 * (1 - fraction), 0.9 * (1 - fraction))
                )
                canvas.rect(width - 160 + index * 1.25, 102, 1.3, 7, stroke=0, fill=1)
            canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 8)
        if page == "selected":
            note = "Red: selected sites. Labels: reference residue/position / MSA column."
        elif page == "confidence":
            note = "pLDDT: orange <50; yellow 50-70; cyan 70-90; blue >=90. Not mapping confidence."
        else:
            upper = rendering["scales"][page]
            note = (
                f"Pale to dark red: 0 to {upper:.4g}; gray: unavailable. Shared scale across views."
            )
        canvas.drawString(28, 90, note)
        observed = sum(
            s["structure_status"] in {"mapped", "structure_residue_differs"}
            for s in record["sites"]
        )
        canvas.drawString(
            28,
            76,
            f"{observed}/{len(record['sites'])} selected sites have coordinates; "
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
            ["MSA", "Reference", "Structure", "PSS", "R", "CV freq.", "pLDDT", "Status"]
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
                    number(site.get("effect_range")),
                    number(site.get("selection_frequency")),
                    number(site.get("plddt")),
                    status,
                ]
            )
        table = Table(rows, colWidths=[43, 58, 65, 46, 46, 48, 44, width - 406])
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
