"""Report cell-type expression from an AnnData count atlas."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.colors import FuncNorm, Normalize
from numpy.typing import NDArray
from scipy import sparse

from phenoradar.config import AppConfig
from phenoradar.config.schema import CellTypeExpressionConfig
from phenoradar.figure_runtime import use_figure_formats
from phenoradar.figures import FigureError, _feature_axis_labels, _save_figure
from phenoradar.og_expression_figures import expression_orthogroups
from phenoradar.orthogroup_annotation import load_orthogroup_annotations


def validate_atlas_inputs(config: CellTypeExpressionConfig) -> None:
    """Catch missing report inputs before fitting the models."""
    if find_spec("anndata") is None:
        raise FigureError("Cell-type expression requires AnnData. Install phenoradar[atlas].")
    for path in (config.h5ad_path, config.gene_to_orthogroup_path):
        if not Path(path).is_file():
            raise FigureError(f"Cell-type expression input file not found: {path}")


def rank_site_orthogroups(tables_dir: Path, limit: int) -> list[str]:
    """Rank by summed within-model PSS shares over all sites and fitted models."""
    path = tables_dir / "amino_acid_site_scores.tsv"
    score = "normalized_pss"
    if not path.is_file():
        path = tables_dir / "amino_acid_site_pss.tsv"
        score = "normalized_score"
    return (
        pl.scan_csv(
            path, separator="\t", schema_overrides={"orthogroup": pl.String, score: pl.Float64},
        )
        .filter(pl.col(score).is_finite() & (pl.col(score) > 0))
        .group_by("orthogroup")
        .agg(pl.col(score).sum().alias("score"))
        .sort(["score", "orthogroup"], descending=[True, False])
        .head(limit)
        .collect(engine="streaming")["orthogroup"].to_list()
    )


def _read_cpm(
    config: CellTypeExpressionConfig, orthogroups: list[str],
) -> tuple[list[str], NDArray[np.float64], list[str]]:
    try:
        ad = import_module("anndata")
    except ImportError as exc:
        raise FigureError(
            "Cell-type expression requires AnnData. Install phenoradar[atlas]."
        ) from exc
    mapping = (
        pl.scan_csv(
            config.gene_to_orthogroup_path, separator="\t",
            schema_overrides={"gene_id": pl.String, "orthogroup": pl.String},
        )
        .select("gene_id", "orthogroup")
        .filter(pl.col("gene_id").is_not_null() & pl.col("orthogroup").is_in(orthogroups))
        .unique()
        .collect(engine="streaming")
    )
    atlas = ad.read_h5ad(config.h5ad_path, backed="r")
    try:
        if not atlas.var_names.is_unique:
            raise FigureError("Cell-type atlas var_names must contain unique gene IDs.")
        if config.cell_type_col not in atlas.obs:
            raise FigureError(f"Cell-type atlas obs has no column {config.cell_type_col!r}.")
        observed = atlas.obs[config.cell_type_col].astype("string").fillna("").str.strip()
        if config.cell_types is None:
            labels: list[str] = [
                label for label in observed.unique() if label and label.casefold() != "unknown"
            ]
        else:
            labels = config.cell_types
            available = set(observed)
            missing = [label for label in labels if label not in available]
            if missing:
                raise FigureError(f"Cell types absent from the atlas: {', '.join(missing)}.")
        if not labels:
            raise FigureError("The atlas contains no selected cell types.")
        if config.counts_layer is not None and config.counts_layer not in atlas.layers:
            raise FigureError(f"Cell-type atlas has no layer {config.counts_layer!r}.")
        matrix = atlas.X if config.counts_layer is None else atlas.layers[config.counts_layer]
        if matrix is None or matrix.dtype.kind not in "iuf":
            raise FigureError("The atlas count matrix must contain numeric raw counts.")
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
        codes = observed.map({label: i for i, label in enumerate(labels)}).fillna(-1).to_numpy(
            dtype=np.intp,
        )
        gene_counts = np.zeros((len(labels), atlas.n_vars), dtype=np.float64)
        batch_size = max(1, min(1024, 2_000_000 // max(atlas.n_vars, 1)))
        for start in range(0, atlas.n_obs, batch_size):
            batch_codes = codes[start:start + batch_size]
            if not np.any(batch_codes >= 0):
                continue
            batch = matrix[start:start + batch_size]
            values = batch.data if sparse.issparse(batch) else batch
            if (not np.all(np.isfinite(values)) or np.any(values < 0)
                    or (values.dtype.kind == "f" and np.any(values != np.floor(values)))):
                raise FigureError("The atlas count matrix must contain nonnegative raw counts.")
            for i in np.unique(batch_codes[batch_codes >= 0]):
                gene_counts[i] += np.asarray(
                    batch[batch_codes == i].sum(axis=0, dtype=np.float64),
                ).ravel()
        gene_indices = {name: i for i, name in enumerate(atlas.var_names)}
    finally:
        atlas.file.close()

    indices_by_og: dict[str, list[int]] = {}
    for gene, og in mapping.iter_rows():
        if gene in gene_indices:
            indices_by_og.setdefault(og, []).append(gene_indices[gene])
    library_sizes = gene_counts.sum(axis=1)
    observed_types = library_sizes > 0
    cpm = np.full((len(orthogroups), len(labels)), np.nan)
    for i, og in enumerate(orthogroups):
        if og in indices_by_og:
            totals = gene_counts[:, indices_by_og[og]].sum(axis=1)
            cpm[i, observed_types] = totals[observed_types] / library_sizes[observed_types] * 1e6
    warnings: list[str] = []
    if len(indices_by_og) < len(orthogroups):
        warnings.append(
            f"Cell-type expression: {len(indices_by_og)} of {len(orthogroups)} ranked OGs "
            "matched atlas genes. Unmatched rows are displayed as NA."
        )
    if not np.all(observed_types):
        warnings.append("Cell-type expression: cell types with zero total counts are shown as NA.")
    return labels, cpm, warnings


def row_zscores(cpm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standardize each OG across its observed cell types using the sample SD."""
    result = np.full_like(cpm, np.nan)
    for i, row in enumerate(cpm):
        observed = np.isfinite(row)
        values = row[observed]
        if len(values) >= 2 and np.ptp(values) > 0:
            result[i, observed] = (values - values.mean()) / values.std(ddof=1)
    return result


def _plot(
    path: Path, labels: list[str], cell_types: list[str], values: NDArray[np.float64],
    *, zscore: bool,
) -> None:
    finite = values[np.isfinite(values)]
    if zscore:
        limit = max(2, int(np.ceil(np.abs(finite).max(initial=0))))
        norm = Normalize(vmin=-limit, vmax=limit)
        tick_step = (limit + 3) // 4
        tick_limit = (limit // tick_step) * tick_step
        ticks = list(range(-tick_limit, tick_limit + 1, tick_step))
        scale_label = "Row z-score"
    else:
        peak = max(1, finite.max(initial=0))
        step = 10.0 ** np.floor(np.log10(peak))
        upper = float(np.ceil(peak / step) * step)
        norm = FuncNorm((np.log1p, np.expm1), vmin=0, vmax=upper)
        ticks = [0] + [10 ** i for i in range(int(np.log10(upper)) + 1)]
        if upper >= 1.7 * ticks[-1]:
            ticks.append(int(upper))
        scale_label = "CPM"

    height = max(2.5, 1.3 + sum(max(1, label.count("\n") + 1) * 0.16 + 0.06 for label in labels))
    with matplotlib.rc_context({"font.size": 8.5, "axes.linewidth": 0.55}):
        width = max(7, 3.7 + len(cell_types) * 0.47)
        fig, (ax, scale) = plt.subplots(
            ncols=2, figsize=(width, height), width_ratios=[width - 3.4, 0.14],
        )
        mesh = ax.pcolormesh(
            values, cmap=matplotlib.colormaps["viridis"].with_extremes(bad="#e8e8e8"),
            norm=norm, edgecolors="white", linewidth=0.4,
        )
        ax.invert_yaxis()
        display_types = [label.replace("_", " ") for label in cell_types]
        ax.set_xticks(
            np.arange(len(cell_types)) + 0.5,
            [label[:1].upper() + label[1:] for label in display_types],
            rotation=45, ha="right", va="top", rotation_mode="anchor",
        )
        ax.set_yticks(np.arange(len(labels)) + 0.5, labels, multialignment="right")
        ax.set_xlabel("Cell type")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row, col in np.argwhere(~np.isfinite(values)):
            ax.text(col + 0.5, row + 0.5, "NA", va="center", ha="center", color="#424242")
        bar = fig.colorbar(mesh, cax=scale)
        bar.set_ticks(ticks, labels=[f"{value:,}" for value in ticks])
        bar.set_label(scale_label)
        bar.ax.tick_params(length=2.5, width=0.6)
        _save_figure(fig, path)


def write_cell_type_expression_figures(
    run_dir: Path, config: AppConfig, *, expression_importance: pl.DataFrame | None = None,
) -> list[str]:
    """Write one CPM and one row-standardized heatmap for the active mode."""
    atlas = getattr(config.report, "cell_type_expression", None)
    if atlas is None:
        return []
    stage = "model" if config.runtime.execution_stage == "full_run" else "cv"
    limit = config.figures.top_features
    orthogroups = []
    if expression_importance is not None and {"feature", "importance_mean"}.issubset(
        expression_importance.columns,
    ):
        active = expression_importance.filter(
            pl.col("importance_mean").is_finite() & (pl.col("importance_mean") > 0),
        )
        if config.data.feature_mode == "combined":
            active = active.filter(pl.col("feature").str.starts_with('["expression",'))
        orthogroups = expression_orthogroups(active, limit)
    try:
        if config.data.feature_mode in {"amino_acid", "combined"}:
            ranked_sites = rank_site_orthogroups(run_dir / stage / "tables", limit)
            orthogroups = list(dict.fromkeys([*orthogroups, *ranked_sites]))
        if not orthogroups:
            return ["Cell-type expression: no orthogroups with positive model importance."]
        cell_types, cpm, warnings = _read_cpm(atlas, orthogroups)
    except (OSError, ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
        raise FigureError(f"Could not read cell-type expression inputs: {exc}") from exc

    annotation_path = config.data.orthogroup_annotation_path
    annotations = load_orthogroup_annotations(
        Path(annotation_path) if annotation_path is not None else None, feature_names=orthogroups,
    )
    labels = _feature_axis_labels(orthogroups, annotations)
    out = run_dir / stage / "figures"
    standardized = row_zscores(cpm)
    with use_figure_formats(config.figures.formats):
        _plot(out / "cell_type_expression_cpm.svg", labels, cell_types, cpm, zscore=False)
        _plot(
            out / "cell_type_expression_zscore.svg", labels, cell_types, standardized, zscore=True,
        )
    return warnings
