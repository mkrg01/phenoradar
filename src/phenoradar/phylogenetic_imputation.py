"""Optional nwkit ASR interpretation of unknown-species predictions only."""

from __future__ import annotations

import importlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any

import polars as pl

from phenoradar import asr_cache
from phenoradar.abstention import prediction_label_expr
from phenoradar.config import AppConfig, PredictConfig
from phenoradar.config.schema import PhylogeneticImputationConfig
from phenoradar.provenance import sha256_file
from phenoradar.timing import TimingRecorder
from phenoradar.trait_reference import (
    merge_observed_traits,
    read_observed_traits,
    validate_observed_traits,
)


class PhylogeneticImputationError(ValueError):
    """Invalid phylogenetic interpretation input or incompatible nwkit installation."""


class PhylogeneticImputationUnavailable(PhylogeneticImputationError):
    """A prerequisite for phylogenetic imputation is unavailable."""


@dataclass
class PhylogeneticImputationArtifacts:
    comparison: pl.DataFrame = field(default_factory=pl.DataFrame)
    annotation: pl.DataFrame = field(default_factory=pl.DataFrame)
    nodes: pl.DataFrame = field(default_factory=pl.DataFrame)
    tree: Any = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class _PreparedPhylogeneticFit:
    reference: pl.DataFrame
    unknown: pl.Series
    artifacts: PhylogeneticImputationArtifacts
    status: str
    metadata: dict[str, Any]
    directory: Path
    executable: str | None


def prepare_tree(path: Path, branch_length_mode: str) -> Any:
    """Preserve missing lengths during parsing and retain the supplied rooting."""
    try:
        ete = importlib.import_module("ete4")
    except ImportError as exc:
        raise PhylogeneticImputationUnavailable(
            "Phylogenetic imputation requires ete4; install phenoradar[phylogeny]."
        ) from exc
    try:
        newick = path.read_text(encoding="utf-8-sig").strip()
        if newick.startswith("[&U]"):
            raise ValueError("Supply a rooted tree; the input is marked unrooted")
        newick = newick.removeprefix("[&R]").strip()
        if not newick.endswith(";"):
            raise ValueError("Expected a semicolon-terminated Newick tree")
        tree = ete.Tree(newick, parser=1)
        tips = list(tree.leaves())
        names = [tip.name for tip in tips]
        if any(name is None or not str(name).strip() for name in names):
            raise ValueError("Tree contains unnamed tips")
        if len(set(names)) != len(names):
            raise ValueError("Tree contains duplicate species IDs")
        for index, node in enumerate(tree.traverse("levelorder")):
            node.add_prop("phylo_id", index)
            if node.is_root:
                node.dist = 0.0
            elif branch_length_mode == "unit":
                node.dist = 1.0
            elif node.dist is None:
                raise PhylogeneticImputationUnavailable(
                    "Input branch lengths are missing; select branch_length_mode: unit "
                    "to explicitly assign all branches length 1"
                )
            elif not math.isfinite(node.dist) or node.dist < 0:
                raise ValueError("Input branch lengths must be finite and non-negative")
    except PhylogeneticImputationUnavailable:
        raise
    except Exception as exc:
        raise PhylogeneticImputationError(f"Invalid imputation tree {path}: {exc}") from exc
    return tree


def _nwkit_executable() -> str:
    executable = shutil.which("nwkit")
    if executable is None:
        local = Path(sys.executable).parent / ("nwkit.exe" if sys.platform == "win32" else "nwkit")
        if local.is_file():
            executable = str(local)
    if executable is None:
        raise PhylogeneticImputationUnavailable(
            "Phylogenetic imputation requires the nwkit asr executable on PATH. "
            "See docs/phylogenetic-imputation.md for installation."
        )
    return executable


def _run_asr(
    settings: PhylogeneticImputationConfig,
    directory: Path,
    *,
    cache_root: Path | None = None,
    executable: str | None = None,
    cancel: Event | None = None,
) -> tuple[pl.DataFrame, dict[str, Any], list[str]]:
    executable = executable or _nwkit_executable()
    version = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode:
        raise PhylogeneticImputationError(
            "The installed nwkit is incompatible with ASR integration; "
            "see docs/phylogenetic-imputation.md (tested with nwkit 0.43.21)."
        )
    arguments = [
        executable,
        "asr",
        "--infile",
        str(directory / "tree.nwk"),
        "--input-rooted",
        "yes",
        "--format",
        "1",
        "--trait",
        str(directory / "traits.tsv"),
        "--state-column",
        "state",
        "--trait-type",
        "discrete",
        "--states",
        "0,1",
        "--model",
        settings.model,
        "--root-prior",
        settings.root_prior,
        "--target",
        "all",
        "--output",
        "probabilities",
        "--outfile",
        str(directory / "asr.tsv"),
        "--model-out",
        str(directory / "model.tsv"),
        "--tree-out",
        str(directory / "annotated_tree.nhx"),
        "--tree-annotation",
        "all",
    ]
    key = (
        None if cache_root is None
        else asr_cache.cache_key(executable, version.stdout.strip(), arguments, directory)
    )
    cache_path = None if key is None or cache_root is None else cache_root / f"{key}.zip"
    cached = (
        None if cache_path is None or key is None
        else asr_cache.restore(cache_path, key, directory)
    )
    if cached is None:
        result = (
            subprocess.run(arguments, capture_output=True, text=True, check=False)
            if cancel is None else _run_cancellable_asr(arguments, cancel)
        )
        (directory / "nwkit.stderr.txt").write_text(result.stderr, encoding="utf-8")
        (directory / "nwkit.stdout.txt").write_text(result.stdout, encoding="utf-8")
    else:
        result = subprocess.CompletedProcess(
            cached["command"], 0,
            (directory / "nwkit.stdout.txt").read_text(encoding="utf-8"),
            (directory / "nwkit.stderr.txt").read_text(encoding="utf-8"),
        )
    provenance: dict[str, Any] = {
        "nwkit_version": version.stdout.strip(),
        "command": arguments if cached is None else cached["command"],
        "returncode": result.returncode,
        "asr_cache": {
            "hit": cached is not None,
            "key": key,
            "path": None if cache_path is None else str(cache_path),
            "requested_command": arguments,
        },
    }
    if result.returncode:
        if "unrecognized arguments" in result.stderr or "invalid choice" in result.stderr:
            raise PhylogeneticImputationError(
                "Installed nwkit does not support the required ASR options. "
                "See docs/phylogenetic-imputation.md."
            )
        return (
            pl.DataFrame(),
            provenance,
            [
                "Phylogenetic imputation failed; probabilities remain NA. "
                "See inference/phylogenetic_imputation/nwkit.stderr.txt."
            ],
        )
    try:
        nodes = pl.read_csv(
            directory / "asr.tsv",
            separator="\t",
            schema_overrides={
                "name": pl.String,
                "observed_state": pl.String,
            },
        )
        required = {"branch_id", "parent", "node_class", "name", "p_0", "p_1"}
        if not required.issubset(nodes.columns):
            raise ValueError("missing required columns")
        if nodes["branch_id"].n_unique() != nodes.height:
            raise ValueError("duplicate node IDs")
        for row in nodes.iter_rows(named=True):
            p0, p1 = float(row["p_0"]), float(row["p_1"])
            if not (0 <= p0 <= 1 and 0 <= p1 <= 1 and abs(p0 + p1 - 1) < 1e-7):
                raise ValueError("invalid state probabilities")
    except (OSError, TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        raise PhylogeneticImputationError(f"Invalid nwkit ASR output: {exc}") from exc
    warnings = [line for line in result.stderr.splitlines() if "warning" in line.lower()]
    if cached is None and cache_path is not None and key is not None:
        asr_cache.store(cache_path, key, directory, arguments)
    return nodes, provenance, warnings


def _run_cancellable_asr(arguments: list[str], cancel: Event) -> subprocess.CompletedProcess[str]:
    if cancel.is_set():
        raise CancelledError()
    with subprocess.Popen(
        arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        while not cancel.is_set():
            try:
                stdout, stderr = process.communicate(timeout=0.25)
                return subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
        process.terminate()
        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise CancelledError()


def _validate_node_mapping(tree: Any, nodes: pl.DataFrame) -> None:
    """Check the external command's IDs before interpreting ancestral probabilities."""
    by_id = {row["branch_id"]: row for row in nodes.iter_rows(named=True)}
    tree_nodes = list(tree.traverse("levelorder"))
    if len(tree_nodes) != nodes.height:
        raise PhylogeneticImputationError("nwkit output does not cover the input tree")
    for node in tree_nodes:
        row = by_id.get(node.props["phylo_id"])
        parent = -1 if node.is_root else node.up.props["phylo_id"]
        if (
            row is None
            or row["parent"] != parent
            or (node.is_leaf and (row["node_class"] != "leaf" or row["name"] != node.name))
        ):
            raise PhylogeneticImputationError("nwkit node IDs do not match the input tree")


def _annotation(
    targets: pl.DataFrame,
    reference: pl.DataFrame,
    tree: Any,
    nodes: pl.DataFrame,
    status: str,
) -> pl.DataFrame:
    known = dict(reference.iter_rows())
    probabilities = (
        {
            row["name"]: row["p_1"]
            for row in nodes.iter_rows(named=True)
            if row["node_class"] == "leaf"
        }
        if nodes.height
        else {}
    )
    in_tree = {node.name for node in tree.leaves()} if tree is not None else None
    target_names = set(targets["species"].to_list())
    rows = []
    for name in sorted(target_names | set(known)):
        observed = known.get(name)
        if in_tree is not None and name not in in_tree:
            node_status = "missing_tree_tip"
        elif observed is not None:
            node_status = "observed"
        else:
            node_status = status
        rows.append(
            {
                "species": name,
                "observed_trait": observed,
                "phylo_prob": probabilities.get(name),
                "phylo_status": node_status,
                "phylo_is_imputed": node_status == "imputed",
                "is_prediction_target": name in target_names,
            }
        )
    frame = pl.DataFrame(
        rows,
        schema={
            "species": pl.String,
            "observed_trait": pl.Int8,
            "phylo_prob": pl.Float64,
            "phylo_status": pl.String,
            "phylo_is_imputed": pl.Boolean,
            "is_prediction_target": pl.Boolean,
        },
    )
    prediction_columns = [
        name
        for name in (
            "species",
            "prob",
            "pred_label_fixed_threshold",
            "decision_status",
            "information_coverage",
            "pred_label_selective",
        )
        if name in targets.columns
    ]
    return (
        frame.join(targets.select(prediction_columns), on="species", how="left")
        .with_columns(
            pl.when(pl.col("phylo_is_imputed") & pl.col("is_prediction_target"))
            .then(pl.col("prob") - pl.col("phylo_prob"))
            .otherwise(None)
            .alias("prob_difference")
        )
        .with_columns(pl.col("prob_difference").abs().alias("abs_prob_difference"))
    )


def _prepare_phylogenetic_fit(
    *,
    settings: PhylogeneticImputationConfig,
    target_species: pl.Series,
    reference: pl.DataFrame,
    tree_path: str | None,
    directory: Path,
    trait_name: str,
) -> _PreparedPhylogeneticFit | None:
    artifacts = PhylogeneticImputationArtifacts()
    unknown = target_species.filter(~target_species.is_in(reference["species"].to_list()))
    if unknown.is_empty():
        return None
    mapped = reference.clear()
    n_targets_in_tree = 0
    executable = None
    if reference.height:
        if tree_path is None:
            raise PhylogeneticImputationError(
                "Phylogenetic imputation requires data.tree_path when unknown targets exist."
            )
        artifacts.tree = prepare_tree(Path(tree_path), settings.branch_length_mode)
        names = sorted(node.name for node in artifacts.tree.leaves())
        mapped = reference.filter(pl.col("species").is_in(names))
        n_targets_in_tree = len(unknown.filter(unknown.is_in(names)))
        if mapped.height and n_targets_in_tree:
            executable = _nwkit_executable()
    directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "settings": settings.model_dump(),
        "trait_name": trait_name,
        "reference_scope": "observed_metadata_and_bundle_traits",
        "n_unknown_targets": len(unknown),
        "n_reference_traits": reference.height,
        "rooting": "as_supplied",
        "distance_unit": (
            "branch_count" if settings.branch_length_mode == "unit" else "input_branch_length"
        ),
    }
    reference.write_csv(directory / "reference_traits.tsv", separator="\t")
    metadata["reference_sha256"] = sha256_file(directory / "reference_traits.tsv")
    status = "missing_reference"
    if reference.height == 0:
        artifacts.warnings.append("Skipped phylogenetic imputation: no observed reference traits.")
    else:
        parser = importlib.import_module("ete4.parser.newick").make_parser(1, dist="%.17g")
        artifacts.tree.write(outfile=str(directory / "tree.nwk"), parser=parser)
        metadata["input_tree_sha256"] = sha256_file(Path(tree_path))
        metadata["analysis_tree_sha256"] = sha256_file(directory / "tree.nwk")
        metadata["n_reference_traits_in_tree"] = mapped.height
        metadata["n_unknown_targets_in_tree"] = n_targets_in_tree
        mapped.rename({"species": "leaf_name", "label": "state"}).write_csv(
            directory / "traits.tsv", separator="\t"
        )
        if executable is None:
            artifacts.warnings.append(
                "Skipped phylogenetic imputation: the tree must contain both "
                "observed reference species and unknown prediction targets."
            )
    return _PreparedPhylogeneticFit(
        reference, unknown, artifacts, status, metadata, directory, executable
    )


def _execute_phylogenetic_fit(
    fit: _PreparedPhylogeneticFit,
    settings: PhylogeneticImputationConfig,
    cache_root: Path,
    cancel: Event | None = None,
) -> _PreparedPhylogeneticFit:
    if fit.executable is not None:
        artifacts = fit.artifacts
        artifacts.nodes, provenance, warnings = _run_asr(
            settings, fit.directory, cache_root=cache_root, executable=fit.executable, cancel=cancel
        )
        fit.metadata.update(provenance)
        artifacts.warnings.extend(warnings)
        fit.status = "imputed" if artifacts.nodes.height else "fit_failed"
        if artifacts.nodes.height:
            _validate_node_mapping(artifacts.tree, artifacts.nodes)
    return fit


def write_phylogenetic_imputation(
    *,
    settings: PhylogeneticImputationConfig,
    predictions: pl.DataFrame,
    observed_traits: pl.DataFrame,
    tree_path: str | None,
    run_dir: Path,
    trait_name: str,
    prepared_fit: _PreparedPhylogeneticFit | None = None,
) -> PhylogeneticImputationArtifacts:
    """Write post-prediction artifacts; never train or modify prediction labels."""
    if not settings.enabled or predictions.height == 0:
        return PhylogeneticImputationArtifacts()
    reference = validate_observed_traits(observed_traits)
    target_species = predictions["species"]
    directory = run_dir / "inference" / "phylogenetic_imputation"
    if prepared_fit is not None:
        unknown = target_species.filter(~target_species.is_in(reference["species"].to_list()))
        if (
            prepared_fit.metadata["settings"] == settings.model_dump()
            and prepared_fit.metadata["trait_name"] == trait_name
            and prepared_fit.reference.equals(reference)
            and prepared_fit.unknown.sort().equals(unknown.sort())
            and (
                reference.is_empty()
                or (tree_path is not None
                    and prepared_fit.metadata["input_tree_sha256"] == sha256_file(Path(tree_path)))
            )
        ):
            shutil.copytree(prepared_fit.directory, directory)
            cache = prepared_fit.metadata.get("asr_cache")
            if cache is not None:
                # Keep the actual command, with a usable replay command for the saved inputs.
                cache["requested_command"] = [
                    str(directory / Path(arg).name) if Path(arg).parent == prepared_fit.directory
                    else arg for arg in cache["requested_command"]
                ]
        else:
            prepared_fit = None
    if prepared_fit is None:
        prepared_fit = _prepare_phylogenetic_fit(
            settings=settings, target_species=target_species, reference=reference,
            tree_path=tree_path, directory=directory, trait_name=trait_name,
        )
        if prepared_fit is not None:
            _execute_phylogenetic_fit(
                prepared_fit, settings, run_dir.parent / ".phenoradar_cache" / "asr"
            )
    if prepared_fit is None:
        return PhylogeneticImputationArtifacts()
    artifacts = prepared_fit.artifacts
    metadata = prepared_fit.metadata
    artifacts.annotation = _annotation(
        predictions,
        reference,
        artifacts.tree,
        artifacts.nodes,
        prepared_fit.status,
    )
    artifacts.comparison = artifacts.annotation.filter(
        pl.col("is_prediction_target") & pl.col("observed_trait").is_null()
    ).sort(["abs_prob_difference", "species"], descending=[True, False], nulls_last=True)
    tables = run_dir / "inference" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    positive = artifacts.comparison.filter(
        (prediction_label_expr(artifacts.comparison) == 1)
        & (pl.col("phylo_prob") < 0.5)
        & (pl.col("prob_difference") > 0)
    ).sort(["prob_difference", "species"], descending=[True, False])
    for frame, filename in [
        (artifacts.annotation, "phylogenetic_imputation.tsv"),
        (artifacts.comparison, "phylogenetic_comparison.tsv"),
        (positive, "phylogenetic_positive_candidates.tsv"),
    ]:
        frame.write_csv(tables / filename, separator="\t", null_value="NA", float_precision=10)
    metadata["status"] = prepared_fit.status
    metadata["warnings"] = artifacts.warnings
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return artifacts


def add_phylogenetic_evidence(
    candidates: pl.DataFrame,
    artifacts: PhylogeneticImputationArtifacts,
) -> pl.DataFrame:
    if artifacts.comparison.height == 0 or candidates.height == 0:
        return candidates
    return candidates.join(
        artifacts.comparison.select(
            "species",
            "phylo_prob",
            "prob_difference",
            "phylo_status",
        ),
        on="species",
        how="left",
    )


@contextmanager
def background_phylogenetic_fit(
    config: AppConfig,
    split_manifest: pl.DataFrame,
    *,
    run_root: Path,
    timing_recorder: TimingRecorder,
) -> Iterator[Future[_PreparedPhylogeneticFit | None] | None]:
    """Stage the ASR inputs on the main thread, then fit alongside expression CV."""
    if (
        config.runtime.execution_stage != "full_run"
        or config.runtime.n_jobs < 2
        or not config.phylogenetic_imputation.enabled
        or config.data.tree_path is None
    ):
        yield None
        return
    targets = (
        split_manifest.filter(pl.col("pool") == "discovery_inference")["species"]
        .unique()
    )
    if targets.is_empty():
        yield None
        return

    with TemporaryDirectory(prefix="phenoradar-asr-") as scratch:
        try:
            reference = read_observed_traits(
                config.data.metadata_path, species_col=config.data.species_col,
                trait_col=config.data.trait_col, exclude_col=config.split.exclude_col,
            )
            prepared = _prepare_phylogenetic_fit(
                settings=config.phylogenetic_imputation, target_species=targets,
                reference=reference, tree_path=config.data.tree_path,
                directory=Path(scratch) / "phylogenetic_imputation",
                trait_name=config.data.trait_col,
            )
        except (OSError, ValueError, pl.exceptions.PolarsError):
            # Report input errors in the normal inference stage using actual predictions.
            yield None
            return
        if prepared is None or prepared.executable is None:
            yield None
            return

        cancel = Event()

        def fit() -> _PreparedPhylogeneticFit:
            with timing_recorder.measure(scope="inference", stage="phylogenetic_background_fit"):
                return _execute_phylogenetic_fit(
                    prepared, config.phylogenetic_imputation,
                    run_root / ".phenoradar_cache" / "asr", cancel,
                )

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="phenoradar-asr") as executor:
            future = executor.submit(fit)
            try:
                yield future
            finally:
                cancel.set()
                future.cancel()


def interpret_unknown_predictions(
    config: AppConfig | PredictConfig,
    *,
    predictions: pl.DataFrame,
    run_dir: Path,
    bundled_traits: pl.DataFrame | None = None,
    trait_name: str | None = None,
    background_fit: Future[_PreparedPhylogeneticFit | None] | None = None,
) -> PhylogeneticImputationArtifacts:
    """Shared inference-stage entry point, deliberately absent from CV and testing."""
    if not config.phylogenetic_imputation.enabled or predictions.height == 0:
        return PhylogeneticImputationArtifacts()
    if config.data.tree_path is None:
        return PhylogeneticImputationArtifacts(
            warnings=["Skipped phylogenetic imputation: data.tree_path is unset."]
        )
    from phenoradar.phylogenetic_figures import write_phylogenetic_figures

    metadata_trait = config.data.trait_col
    if isinstance(config, PredictConfig) and "trait_col" not in config.data.model_fields_set:
        metadata_trait = trait_name or metadata_trait
    reference = merge_observed_traits(
        bundled_traits,
        read_observed_traits(
            config.data.metadata_path,
            species_col=config.data.species_col,
            trait_col=metadata_trait,
            exclude_col=config.split.exclude_col if isinstance(config, AppConfig) else None,
        ),
    )
    try:
        artifacts = write_phylogenetic_imputation(
            settings=config.phylogenetic_imputation,
            predictions=predictions,
            observed_traits=reference,
            tree_path=config.data.tree_path,
            run_dir=run_dir,
            trait_name=trait_name or config.data.trait_col,
            prepared_fit=background_fit.result() if background_fit is not None else None,
        )
    except PhylogeneticImputationUnavailable as exc:
        return PhylogeneticImputationArtifacts(
            warnings=[f"Skipped phylogenetic imputation: {exc}"]
        )
    write_phylogenetic_figures(
        artifacts,
        run_dir=run_dir,
        branch_length_mode=config.phylogenetic_imputation.branch_length_mode,
    )
    return artifacts
