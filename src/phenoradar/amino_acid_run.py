"""Sequence-mode orchestration using the shared split, sampling and glmnet code."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import polars as pl
from optuna.samplers import TPESampler
from scipy import sparse

from phenoradar.amino_acid import (
    ALPHABET,
    PREDICTION_BATCH_SIZE,
    AATransform,
    AlignmentError,
    AlignmentStore,
)
from phenoradar.amino_acid_audits import ColumnarAudits
from phenoradar.amino_acid_outputs import (
    write_combined_coefficients,
    write_frequency_table,
    write_preprocessing_audits,
    write_site_effects,
    write_site_scores,
)
from phenoradar.amino_acid_prediction import prediction_members
from phenoradar.amino_acid_reporting import AAReportWriter
from phenoradar.amino_acid_resources import record_sparse_usage
from phenoradar.amino_acid_sparse import fit_sparse_encoded, training_information
from phenoradar.combined import ExpressionInput, ExpressionTransform, fit_expression
from phenoradar.config import AppConfig
from phenoradar.config.io import write_resolved_config
from phenoradar.cv import (
    InnerCvPreprocessedFold,
    SelectedCandidate,
    SourceSelectionResult,
    _aggregate_probabilities,
    _build_estimator,
    _dedupe_ranked_candidates_by_params,
    _deterministic_int_seed,
    _fit_sample_weights,
    _fit_selected_estimator,
    _fold_ids,
    _glmnet_candidate_params,
    _inner_cv_splits,
    _metric_rows,
    _metrics_dataframe,
    _rank_selected_candidates,
    _requested_selected_candidate_count,
    _sample_training_sets,
    _score_glmnet_paths,
    _score_std_error_from_trial_rows,
    _select_training_groups,
    _selection_is_active,
    _selection_metric_nan_sentinel,
    _selection_metric_optuna_direction,
    _selection_source_sample_set_id,
    _tpe_trial_params,
    _with_native_thread_limit,
)
from phenoradar.fold_identity import configure_fold_seed_ids, seed_scope
from phenoradar.glmnet import GlmnetLogisticRegression
from phenoradar.group_bootstrap import run_oof_group_bootstrap
from phenoradar.metrics import (
    binary_log_loss,
    binary_probability_metrics,
    evaluation_metric_contract,
    evaluation_metric_contract_rows,
)
from phenoradar.model_selection import (
    Candidate,
    active_search_space,
    expanded_search_space,
    generate_candidates,
)
from phenoradar.provenance import (
    dataset_fingerprint,
    experiment_fingerprint,
    file_identity,
    phenoradar_build_snapshot,
    sha256_file,
    split_fingerprint,
)
from phenoradar.split import SplitArtifacts, build_split_artifacts


@dataclass
class AAMember:
    transform: AATransform
    model: GlmnetLogisticRegression
    scope: str
    sample_set_id: int
    candidate: Candidate
    expression: ExpressionTransform | None = None
    aa_standard_deviation: np.ndarray | None = None

    @property
    def expression_width(self) -> int:
        expression = getattr(self, "expression", None)
        return len(expression.features) if expression is not None else 0

    @property
    def features(self) -> list[str]:
        expression = getattr(self, "expression", None)
        return ([] if expression is None else expression.features) + self.transform.features


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        return
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def _arrays(rows: pl.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    return (
        rows["species"].to_list(),
        rows["label"].to_numpy().astype(int),
        rows["group_id"].cast(pl.String).to_numpy(),
        rows["contrast_group_id"].cast(pl.String).fill_null("").to_numpy(),
    )


def _prepare_inner_fold(
    config: AppConfig, store: AlignmentStore, expression: ExpressionInput | None,
    species: list[str], y: np.ndarray, groups: np.ndarray, pairs: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, str], warnings: list[str],
) -> InnerCvPreprocessedFold:
    train, valid, fold = split
    train_species = [species[i] for i in train]
    valid_species = [species[i] for i in valid]
    fitted, matrices, _ = fit_sparse_encoded(
        store, train_species, config.preprocess.amino_acid,
        [train_species, valid_species], y=y[train], groups=groups[train],
        contrast_groups=pairs[train], audit=False, n_jobs=config.runtime.n_jobs,
    )
    x_train, x_valid = matrices
    n_expression = None
    if expression is not None:
        expr, xe, _, _ = fit_expression(config, expression, train_species, y[train],
                                        pairs[train], warnings)
        xv, _ = expr.transform(expression, valid_species)
        n_expression = xe.shape[1]
        # Two CSC inputs let SciPy concatenate without converting AA to COO.
        x_train = sparse.hstack((sparse.csc_matrix(xe), x_train), format="csc")
        x_valid = sparse.hstack((sparse.csc_matrix(xv), x_valid), format="csc")
        record_sparse_usage(store, fitted, [x_train, x_valid])
    return InnerCvPreprocessedFold(fold, x_train, x_valid, y[train], y[valid],
                                   _fit_sample_weights(config, y[train], groups[train]),
                                   n_expression)


def _select(
    config: AppConfig,
    store: AlignmentStore,
    rows: pl.DataFrame,
    scope: str,
    sample_set_id: int,
    warnings: list[str],
    expression: ExpressionInput | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[SelectedCandidate], list[dict[str, Any]]]:
    species, y, groups, pairs = _arrays(rows)
    def prepare() -> Generator[InnerCvPreprocessedFold, None, None]:
        for train, valid, fold in _inner_cv_splits(config, y, groups):
            if progress:
                progress(f"{scope}: prepare inner fold {fold}.")
            prepared_fold = _prepare_inner_fold(config, store, expression, species, y,
                                                  groups, pairs, (train, valid, fold), warnings)
            if progress:
                progress(
                    f"{scope}: inner fold {fold} ready "
                    f"({len(train)} train, {len(valid)} validation, "
                    f"{prepared_fold.x_train.shape[1]} AA features)."
                )
            yield prepared_fold
            del prepared_fold


    # TPE revisits the same folds for successive candidate batches. Grid paths
    # can instead consume each fold once and immediately release its matrices.
    prepared = (list(prepare()) if _selection_is_active(config)
                and config.model_selection.search_strategy == "tpe" else None)

    def selection_progress(event: str, detail: str | None) -> None:
        if progress:
            progress(f"{scope}: {detail or event}")

    def score(candidates: list[Candidate]) -> list[tuple[float, list[dict[str, Any]]]]:
        if progress:
            progress(f"{scope}: score {len(candidates)} candidates along glmnet paths.")
        if prepared is not None:
            return _score_glmnet_paths(
                config=config, training_scope_id=scope, source_sample_set_id=sample_set_id,
                candidates=candidates, preprocessed_folds=prepared,
                progress_callback=selection_progress, timing_recorder=None, max_path_workers=1,
            )
        collected: list[list[dict[str, Any]]] = [[] for _ in candidates]
        from phenoradar.amino_acid_inner import inner_workers, score_parallel

        iterator = prepare()
        first = next(iterator)
        splits = list(_inner_cv_splits(config, y, groups))[1:]
        workers = inner_workers(config, store, first, len(splits))

        pending_first = [first]

        def serial_scores() -> Iterable[list[tuple[float, list[dict[str, Any]]]]]:
            while True:
                fold = pending_first.pop() if pending_first else next(iterator, None)
                if fold is None:
                    return
                yield _score_glmnet_paths(
                    config=config, training_scope_id=scope, source_sample_set_id=sample_set_id,
                    candidates=candidates, preprocessed_folds=[fold],
                    progress_callback=None, timing_recorder=None,
                    consume_training_matrix=True, max_path_workers=1,
                )
                del fold

        scores = (score_parallel(config, store, expression, scope, sample_set_id, candidates,
                                 species, y, groups, pairs, first, splits, workers,
                                 warnings, progress)
                  if workers > 1 else serial_scores())
        if workers > 1:
            pending_first.clear()
            iterator.close()
        del first
        for scored in scores:
            for rows, (_, trials) in zip(collected, scored, strict=True):
                rows.extend(trials)
            del scored
        result = []
        for rows in collected:
            values = [row["metric_value"] for row in rows if not np.isnan(row["metric_value"])]
            result.append((float(np.mean(values)) if values else float("nan"), rows))
        return result

    trials: list[dict[str, Any]] = []
    scored: list[SelectedCandidate] = []
    if _selection_is_active(config) and config.model_selection.search_strategy == "tpe":
        discrete, continuous = expanded_search_space(active_search_space(config))
        study = optuna.create_study(
            direction=_selection_metric_optuna_direction(config.model_selection.selection_metric),
            sampler=TPESampler(
                seed=_deterministic_int_seed(
                    f"{config.runtime.seed}|{seed_scope(config, scope)}|{sample_set_id}|aa_tpe"
                )
            ),
        )

        def objective(trial: optuna.Trial) -> float:
            candidate = Candidate(
                trial.number,
                _tpe_trial_params(trial, discrete_values=discrete, continuous_values=continuous),
            )
            mean, detail = score([candidate])[0]
            trials.extend(detail)
            scored.append(
                SelectedCandidate(
                    candidate,
                    mean,
                    _score_std_error_from_trial_rows(detail),
                    config.model_selection.selection_rule,
                )
            )
            return (
                mean
                if np.isfinite(mean)
                else _selection_metric_nan_sentinel(config.model_selection.selection_metric)
            )

        study.optimize(objective, n_trials=int(config.model_selection.trial_count or 1))
    else:
        candidates = generate_candidates(
            config=config,
            training_scope_id=scope,
            source_sample_set_id=sample_set_id,
            warnings=warnings,
        )
        if not _selection_is_active(config):
            return _dedupe_ranked_candidates_by_params(
                [SelectedCandidate(c, None) for c in candidates]
            ), []
        for candidate, (mean, detail) in zip(candidates, score(candidates), strict=True):
            trials.extend(detail)
            scored.append(
                SelectedCandidate(
                    candidate,
                    mean,
                    _score_std_error_from_trial_rows(detail),
                    config.model_selection.selection_rule,
                )
            )
    return _selected_from_scores(config, scored), trials


def _selected_from_scores(config: AppConfig, scored: list[SelectedCandidate]
                          ) -> list[SelectedCandidate]:
    ranked = _dedupe_ranked_candidates_by_params(_rank_selected_candidates(scored, config))
    if not any(item.score is not None and np.isfinite(item.score) for item in ranked):
        raise AlignmentError("No finite inner-CV scores for amino-acid candidate selection")
    count = _requested_selected_candidate_count(
        config=config, available_unique_candidates=len(ranked)
    )
    return ranked[:count]


def fit_scope(
    config: AppConfig,
    store: AlignmentStore,
    rows: pl.DataFrame,
    scope: str,
    tables: Path,
    warnings: list[str],
    expression: ExpressionInput | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    audit: ColumnarAudits | None = None,
    selections: dict[int, tuple[list[SelectedCandidate], list[dict[str, Any]], list[str]]]
    | None = None,
) -> list[AAMember]:
    species, y, groups, pairs = _arrays(rows)
    selection = _select_training_groups(config, y, groups, scope=scope, fold_id=scope)
    _write(tables / "training_group_subsets.tsv", selection.audit_rows)
    sampled = _sample_training_sets(config, y, groups, scope, warnings, selection.group_ids)
    source_results: dict[int, SourceSelectionResult] = {}
    members = []
    for sample_set_id, indices in enumerate(sampled):
        source_id = _selection_source_sample_set_id(config, sample_set_id)
        if source_id not in source_results:
            if selections is None:
                selected, trials = _select(
                    config, store, rows[sampled[source_id].tolist()], scope, source_id, warnings,
                    expression, progress=progress,
                )
            else:
                selected, trials, messages = selections[source_id]
                warnings.extend(messages)
            scored_count = len({row["candidate_index"] for row in trials})
            source_results[source_id] = SourceSelectionResult(
                selected, max(scored_count, len(selected)), scored_count,
                len(selected), len(selected), trials,
            )
            _write(
                tables / "model_selection_trials.tsv",
                (
                    {
                        "scope": scope,
                        "sample_set_id": source_id,
                        **{k: v for k, v in row.items() if k != "fit_diagnostic"},
                    }
                    for row in trials
                ),
            )
        selected = source_results[source_id].selected_candidates
        train_species = [species[i] for i in indices]
        if progress:
            progress(f"{scope}: prepare refit features for sample set {sample_set_id}.")
        fitted, matrices, shift = fit_sparse_encoded(
            store, train_species, config.preprocess.amino_acid, [train_species],
            y=y[indices], groups=groups[indices], contrast_groups=pairs[indices],
            n_jobs=config.runtime.n_jobs,
        )
        matrix = matrices[0]
        del matrices
        context = {"scope": scope, "sample_set_id": sample_set_id}
        expr = None
        aa_std = None
        representatives: list[dict[str, Any]] = []
        if expression is not None or config.preprocess.amino_acid.aggregation == "best_covered":
            aa_std, representatives = training_information(
                fitted, store, train_species, matrix, shift,
                standard_deviation=expression is not None,
            )
        if expression is not None:
            expr, xe, counts, scores = fit_expression(
                config, expression, train_species, y[indices], pairs[indices], warnings
            )
            matrix = sparse.hstack((sparse.csc_matrix(xe), matrix), format="csc")
            record_sparse_usage(store, fitted, [matrix])
            shift = np.r_[np.zeros(xe.shape[1]), shift]
            _write(tables / "expression_feature_filter_counts.tsv", [{**context, **counts}])
            _write(tables / "expression_ranked_feature_scores.tsv",
                   ({**context, **row} for row in scores))
        _write(tables / "amino_acid_og_qc.tsv", ({**context, **q} for q in fitted.qc))
        if progress:
            progress(f"{scope}: refit matrix ready; write preprocessing audits.")
        write_preprocessing_audits(tables, fitted, context, expr, audit=audit)
        _write(tables / "representative_sequences.tsv", ({**context, **r} for r in representatives))
        if fitted.representative_columns is not None:
            _write(
                tables / "representative_scoring_sites.tsv",
                (
                    {**context, "orthogroup": name, "alignment_pos_1based": int(column) + 1}
                    for name, columns in sorted(fitted.representative_columns.items())
                    for column in columns
                ),
            )
        for result in selected:
            model = _build_estimator(
                config, config.runtime.seed, y[indices], model_params=result.candidate.params
            )
            assert isinstance(model, GlmnetLogisticRegression)
            if expr is not None:
                _, ratio = _glmnet_candidate_params(config, result.candidate.params)
                model.penalty_factor = np.concatenate((
                    np.ones(len(expr.features)),
                    np.full(matrix.shape[1] - len(expr.features), ratio),
                ))
            if progress:
                progress(
                    f"{scope}: fit selected candidate {result.candidate.candidate_index} "
                    f"({matrix.shape[0]} species, {matrix.shape[1]} features)."
                )
            _with_native_thread_limit(
                config.runtime.n_jobs,
                _fit_selected_estimator,
                model,
                matrix,
                y[indices],
                sample_weight=_fit_sample_weights(config, y[indices], groups[indices]),
                source_result=source_results[source_id],
                amino_acid_penalty_ratio=(
                    _glmnet_candidate_params(config, result.candidate.params)[1]
                    if expr is not None else None
                ),
            )
            # Native fitting uses Z=X+c; all saved models and reports use X.
            model.intercept_ += float(shift @ model.coef_)
            members.append(AAMember(fitted, model, scope, sample_set_id, result.candidate,
                                    expr, aa_std))
            _write(
                tables / "model_selection_selected.tsv",
                [
                    {
                        **context,
                        "candidate_index": result.candidate.candidate_index,
                        "params_json": json.dumps(result.candidate.params, sort_keys=True),
                        "score": result.score,
                        "score_std_error": result.score_std_error,
                        "selection_rule": result.selection_rule,
                    }
                ],
            )
            write_site_effects(tables, members[-1], matrix, shift, audit=audit)
            if expr is not None:
                write_combined_coefficients(tables, members[-1])
    return members


def _site_coefficients(member: AAMember) -> list[np.ndarray]:
    offset = member.expression_width
    coefficients = []
    for site in member.transform.sites:
        beta = np.zeros(20)
        count = int(np.sum(site.active))
        beta[site.active] = member.model.coef_[offset : offset + count]
        coefficients.append(beta)
        offset += count
    return coefficients


def _effect_range(known: np.ndarray, beta: np.ndarray) -> float:
    return float(np.ptp(beta[known])) if np.count_nonzero(known) > 1 else 0.0


def predict_members(
    config: AppConfig,
    store: AlignmentStore,
    species: list[str],
    members: list[AAMember],
    *,
    tables: Path | None = None,
    expression: ExpressionInput | None = None,
) -> pl.DataFrame:
    members = prediction_members(
        members, padding_sites=config.figures.top_features if tables is not None else 0
    )
    output = []
    size = PREDICTION_BATCH_SIZE
    for start in range(0, len(species), size):
        samples = species[start : start + size]
        probabilities = []
        coverages = []
        expression_coverages = []
        aa_coverages = []
        matrices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        gene_matrices: dict[str, np.ndarray] = {}
        local_rows = []
        for model_index, member in enumerate(members):
            key = id(member.transform)
            if key not in matrices:
                x, observed, representatives = member.transform.transform(store, samples)
                matrices[key] = x, observed
                if tables is not None:
                    _write(
                        tables / "prediction_representative_sequences.tsv",
                        (
                            {"scope": member.scope, "sample_set_id": member.sample_set_id, **r}
                            for r in representatives
                        ),
                    )
            x, observed = matrices[key]
            expr_observed = None
            if member.expression_width:
                assert expression is not None and member.expression is not None
                xe, expr_observed = member.expression.transform(expression, samples)
                x = np.column_stack((xe, x))
            probabilities.append(member.model.predict_proba(x)[:, 1])
            betas = _site_coefficients(member)
            weights = np.array(
                [
                    _effect_range(site.known, beta)
                    for site, beta in zip(member.transform.sites, betas, strict=True)
                ]
            )
            total = float(weights.sum())
            aa_coverage = observed @ (weights / total) if total > 0 else np.zeros(len(samples))
            if expr_observed is None:
                coverages.append(aa_coverage)
            else:
                ew = np.abs(member.model.coef_[:member.expression_width])
                et = float(ew.sum())
                ec = expr_observed @ (ew / et) if et > 0 else np.ones(len(samples))
                ac = aa_coverage if total > 0 else np.ones(len(samples))
                coverages.append(np.minimum(ec, ac) if et > 0 or total > 0
                                 else np.zeros(len(samples)))
                expression_coverages.append(ec if et > 0 else np.full(len(samples), np.nan))
                aa_coverages.append(ac if total > 0 else np.full(len(samples), np.nan))
            if tables is not None:
                offset = member.expression_width
                if expr_observed is not None:
                    ec = x[:, :offset] @ member.model.coef_[:offset]
                    ac = x[:, offset:] @ member.model.coef_[offset:]
                    _write(tables / "combined_modality_contributions.tsv", (
                        {"species": sample, "model_index": model_index, "scope": member.scope,
                         "sample_set_id": member.sample_set_id,
                         "candidate_index": member.candidate.candidate_index,
                         "expression_contribution": float(ec[i]),
                         "amino_acid_contribution": float(ac[i]),
                         "intercept": float(member.model.intercept_),
                         "logit": float(ec[i] + ac[i] + member.model.intercept_)}
                        for i, sample in enumerate(samples)
                    ))
                    assert member.expression is not None
                    _write(tables / "expression_local_contributions.tsv", (
                        {"species": sample, "model_index": model_index, "scope": member.scope,
                         "sample_set_id": member.sample_set_id,
                         "candidate_index": member.candidate.candidate_index,
                         "orthogroup": member.expression.orthogroups[j],
                         "contribution": float(x[i, j] * member.model.coef_[j]),
                         "observed": bool(expr_observed[i, j])}
                        for i, sample in enumerate(samples)
                        for j in np.argsort(-np.abs(x[i, :offset] * member.model.coef_[:offset]),
                                            kind="stable")[:config.figures.top_features]
                    ))
                contributions = []
                for site, beta in zip(member.transform.sites, betas, strict=True):
                    count = int(np.sum(site.active))
                    contributions.append(x[:, offset : offset + count] @ beta[site.active])
                    offset += count
                contribution_matrix = np.column_stack(contributions)
                for i, sample in enumerate(samples):
                    top = np.argsort(-np.abs(contribution_matrix[i]), kind="stable")[
                        : config.figures.top_features
                    ]
                    for j in top:
                        site = member.transform.sites[j]
                        og = store.ogs[site.orthogroup]
                        if og.orthogroup not in gene_matrices:
                            if len(gene_matrices) >= 32:
                                gene_matrices.pop(next(iter(gene_matrices)))
                            gene_matrices[og.orthogroup] = og.matrix()
                        gene_rows = og.species_rows.get(sample, [])
                        n_callable = sum(
                            chr(int(gene_matrices[og.orthogroup][r, site.column])) in ALPHABET
                            for r in gene_rows
                        )
                        tpm_evidence = {}
                        if member.transform.settings.aggregation == "tpm_weighted_frequency":
                            assert og.tpm is not None
                            weights = og.tpm[gene_rows]
                            callable_rows = np.array(
                                [
                                    chr(int(gene_matrices[og.orthogroup][r, site.column]))
                                    in ALPHABET
                                    for r in gene_rows
                                ],
                                dtype=bool,
                            )
                            tpm_evidence = {
                                "n_og_positive_tpm_genes": int(np.count_nonzero(weights > 0)),
                                "n_og_callable_positive_tpm_genes": int(
                                    np.count_nonzero(weights[callable_rows] > 0)
                                ),
                            }
                        local_rows.append(
                            {
                                "species": sample,
                                "model_index": model_index,
                                "scope": member.scope,
                                "orthogroup": site.orthogroup,
                                "alignment_pos_1based": site.column + 1,
                                "contribution": float(contribution_matrix[i, j]),
                                "observed_fraction": float(observed[i, j]),
                                "n_og_genes": len(gene_rows),
                                "n_og_callable_genes": n_callable,
                                **tpm_evidence,
                            }
                        )
        probability = _aggregate_probabilities(
            probabilities, config.ensemble.probability_aggregation
        )
        coverage = np.mean(coverages, axis=0)
        uncertainty = np.std(probabilities, axis=0)
        for i, sample in enumerate(samples):
            accepted = not config.abstention.enabled or bool(
                coverage[i] >= config.abstention.threshold - 1e-12
            )
            output.append(
                {
                    "species": sample,
                    "prob": float(probability[i]),
                    "pred_label_fixed_threshold": int(probability[i] >= 0.5),
                    "uncertainty_std": float(uncertainty[i]),
                    "information_coverage": float(coverage[i]),
                    "abstention_threshold": config.abstention.threshold,
                    "pred_label_selective": int(probability[i] >= 0.5) if accepted else None,
                    "decision_status": "accepted" if accepted else "abstained",
                    "abstention_reason": None if accepted else "insufficient_information",
                }
            )
            if expression_coverages:
                for name, values in [("expression", expression_coverages),
                                     ("amino_acid", aa_coverages)]:
                    finite = np.asarray(values)[:, i]
                    finite = finite[np.isfinite(finite)]
                    output[-1][f"{name}_information_coverage"] = (
                        float(np.mean(finite)) if finite.size else None
                    )
        if tables is not None:
            _write(tables / "amino_acid_local_contributions.tsv", local_rows)
    schema = {"pred_label_selective": pl.Int64, "abstention_reason": pl.String}
    if expression is not None:
        schema.update({"expression_information_coverage": pl.Float64,
                       "amino_acid_information_coverage": pl.Float64})
    return pl.DataFrame(output, schema_overrides=schema)


def _coordinate_rows(
    store: AlignmentStore, members: list[AAMember], species: list[str]
) -> Iterable[dict[str, Any]]:
    members = prediction_members(members)
    seen = set()
    for member in members:
        key = id(member.transform)
        if key in seen:
            continue
        seen.add(key)
        grouped: dict[str, list[tuple[Any, np.ndarray]]] = defaultdict(list)
        # Coordinates include all active-model sites, even if a different member
        # sharing this transform selected the coefficient at that site.
        selected = set()
        for other in members:
            if other.transform is member.transform:
                for site, beta in zip(
                    other.transform.sites, _site_coefficients(other), strict=True
                ):
                    if _effect_range(site.known, beta) > 1e-12:
                        selected.add((site.orthogroup, site.column))
        for site in member.transform.sites:
            if (site.orthogroup, site.column) in selected:
                grouped[site.orthogroup].append((site, site.known))
        for name, sites in grouped.items():
            og = store.ogs[name]
            matrix = og.matrix()
            scoring = member.transform.scoring_columns(name)
            for sample in species:
                rows = og.species_rows.get(sample, [])
                weighted = member.transform.settings.aggregation == "tpm_weighted_frequency"
                normalized: dict[int, float] = {}
                denominators: dict[int, float] = {}
                if weighted and rows:
                    assert og.tpm is not None
                    maximum = float(og.tpm[rows].max())
                    normalized = {r: float(og.tpm[r] / maximum) if maximum > 0 else 0 for r in rows}
                    denominators = {
                        site.column: sum(
                            normalized[r]
                            for r in rows
                            if chr(int(matrix[r, site.column])) in ALPHABET
                        )
                        for site, _ in sites
                    }
                if member.transform.settings.aggregation == "best_covered":
                    chosen = store.representatives(og, [sample], scoring)[0]
                    rows = [] if chosen is None else [chosen]
                for row in rows:
                    raw = matrix[row]
                    positions = np.cumsum((raw != ord("-")) & (raw != ord(".")))
                    for site, _ in sites:
                        residue = chr(int(raw[site.column]))
                        tpm_evidence = {}
                        if weighted:
                            assert og.tpm is not None
                            denominator = denominators[site.column]
                            tpm_evidence = {
                                "sequence_tpm": float(og.tpm[row]),
                                "site_tpm_fraction": (
                                    normalized[row] / denominator
                                    if residue in ALPHABET and denominator > 0
                                    else 0.0
                                ),
                            }
                        yield {
                            "scope": member.scope,
                            "sample_set_id": member.sample_set_id,
                            "species": sample,
                            "orthogroup": name,
                            "alignment_pos_1based": site.column + 1,
                            "sequence_id": og.sequence_ids[row],
                            "residue": residue,
                            "sequence_pos_1based": None
                            if residue in "-."
                            else int(positions[site.column]),
                            "is_canonical": residue in ALPHABET,
                            "sequence_sha256": og.sequence_hashes[row],
                            **tpm_evidence,
                        }


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        **binary_probability_metrics(y, probabilities, threshold=0.5),
        "log_loss": binary_log_loss(y, probabilities),
    }


def run_amino_acid(
    config: AppConfig,
    run_dir: Path,
    *,
    split_override: SplitArtifacts | None = None,
    study_context: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    started = datetime.now(UTC)
    splits = split_override or build_split_artifacts(config)
    configure_fold_seed_ids(config, splits.split_manifest)
    manifest = splits.split_manifest
    metadata = pl.read_csv(config.data.metadata_path, separator="\t")
    all_species = metadata[config.data.species_col].cast(pl.String).str.strip_chars().to_list()
    training_species = sorted(
        manifest.filter(pl.col("pool").is_in(["train", "validation"]))["species"].unique().to_list()
    )
    # Keep training bytes on disk; held-out target bytes are imported only for
    # selected OGs after fitting. No full-width inference frequency cube exists.
    run_dir.mkdir(parents=True, exist_ok=True)
    split_dir = run_dir / "split/tables"
    tables = run_dir / "cv/tables"
    model_dir = run_dir / "model/tables"
    split_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir / "resolved_config.yml")
    manifest.write_csv(split_dir / "split_manifest.tsv", separator="\t")
    splits.fold_diagnostics.write_csv(split_dir / "fold_diagnostics.tsv", separator="\t")
    splits.fold_validation_groups.write_csv(
        split_dir / "fold_validation_groups.tsv", separator="\t"
    )
    warnings: list[str] = []
    oof = []
    metrics = []
    losses = []
    fold_values = []
    coordinate_members: list[AAMember] = []
    if progress:
        progress("Import alignment bytes and packed observation masks.")
    weighted = config.preprocess.amino_acid.aggregation == "tpm_weighted_frequency"
    tpm_path = config.data.sequence_tpm_path if weighted else None
    tpm_cache = (
        Path(config.data.sequence_tpm_cache_dir) if config.data.sequence_tpm_cache_dir else None
    )
    with ExitStack() as stack:
        reports = stack.enter_context(AAReportWriter(tables))
        expression = (
            stack.enter_context(ExpressionInput(config, all_species, training_species))
            if config.data.feature_mode == "combined" else None
        )
        expression_identity = file_identity(Path(config.data.tpm_path)) if expression else None
        if expression_identity:
            _write(model_dir / "expression_input.tsv", [expression_identity])
        store = stack.enter_context(
            AlignmentStore(
                Path(config.data.alignment_dir or ""),
                all_species,
                cache_species=training_species,
                cache_frequencies=True,
                input_cache_dir=Path("runs/.phenoradar_cache/alignments"),
                sequence_tpm_path=tpm_path,
                sequence_tpm_cache_dir=tpm_cache,
                progress=progress,
            )
        )
        if expression is not None:
            present = {s for og in store.ogs.values() for s in og.species_rows}
            missing = sorted(set(training_species) - present)
            if missing:
                raise AlignmentError(
                    "Combined training requires sequences for every training species; missing: "
                    + ", ".join(missing[:10])
                )
        identities = store.identities()
        _write(model_dir / "alignment_inputs.tsv", identities)
        _write(model_dir / "sequence_correspondence.tsv", store.correspondence)
        if store.tpm_identity:
            _write(
                model_dir / "sequence_tpm_input.tsv",
                [{"scope": "training_pool", **store.tpm_identity}],
            )
        from phenoradar.amino_acid_parallel import merge_tables, outer_folds
        from phenoradar.amino_acid_queue import enabled, grid_scopes

        folds = _fold_ids(manifest)
        scopes = (grid_scopes(config, store, expression, manifest, folds, progress)
                  if enabled(config) else zip(
                      folds, outer_folds(config, store, expression, manifest, folds, progress),
                      strict=True))
        final = None
        final_messages: list[str] = []
        for fold, directory in scopes:
            if fold == "final_refit":
                final, final_messages = joblib.load(directory / "members.joblib")
                merge_tables(directory / "tables", model_dir)
                continue
            compact, pred, fold_warnings, _ = joblib.load(directory / "result.joblib")
            reports.merge(directory / "reports")
            merge_tables(directory / "tables", tables)
            warnings.extend(fold_warnings)
            coordinate_members.extend(compact)
            oof.append(pred)
            values = _metrics(pred["label"].to_numpy(), pred["prob"].to_numpy())
            fold_values.append(values)
            positive = int(pred["label"].sum())
            metrics.extend(_metric_rows(fold, "NA", values, positive, len(pred) - positive, 1))
            losses.append({"fold_id": fold, "split": "validation", "metric": "log_loss",
                           "metric_value": values["log_loss"]})
            if progress:
                progress(f"Outer CV fold {fold} completed.")
        predictions = pl.concat(oof).sort("species")
        predictions.write_csv(tables / "prediction_cv.tsv", separator="\t")
        micro = _metrics(predictions["label"].to_numpy(), predictions["prob"].to_numpy())
        macro = {
            key: float(np.mean([v[key] for v in fold_values if np.isfinite(v[key])]))
            if any(np.isfinite(v[key]) for v in fold_values)
            else float("nan")
            for key in micro
        }
        positive = int(predictions["label"].sum())
        for label, values in [("micro", micro), ("macro", macro)]:
            metrics.extend(
                _metric_rows(
                    None,
                    label,
                    values,
                    positive,
                    len(predictions) - positive,
                    {k: sum(np.isfinite(v[k]) for v in fold_values) for k in micro},
                )
            )
        _metrics_dataframe(metrics).write_csv(tables / "metrics_cv.tsv", separator="\t")
        _write(tables / "loss_by_split_cv.tsv", losses)
        if progress:
            progress("Finalize disk-backed feature and site interpretation tables.")
        importance = (reports.finalize_isolated(
            combined=expression is not None, top_features=config.figures.top_features,
            n_jobs=config.runtime.n_jobs,
        ) if config.runtime.n_jobs >= 3 else reports.finalize(
            combined=expression is not None, top_features=config.figures.top_features,
        ))
        write_frequency_table(
            tables, store, manifest,
            reports.frequency_members(coordinate_members, importance),
            importance, config.figures.top_features,
        )
        _write(
            tables / "selected_site_sequence_map.tsv",
            _coordinate_rows(store, coordinate_members, training_species),
        )
        if config.evaluation.group_bootstrap.enabled:
            boot = run_oof_group_bootstrap(
                oof_predictions=predictions,
                split_manifest=manifest,
                group_col=config.split.group_col,
                n_resamples=config.evaluation.group_bootstrap.n_resamples,
                confidence_level=config.evaluation.group_bootstrap.confidence_level,
                runtime_seed=config.runtime.seed,
            )
            boot.summary.write_csv(tables / "group_bootstrap_metrics.tsv", separator="\t")
            boot.replicates.write_csv(tables / "group_bootstrap_replicates.tsv", separator="\t")
            warnings.extend(boot.warnings)
        if config.runtime.execution_stage == "full_run":
            if progress:
                progress("Fit final sequence ensemble and predict external/inference species.")
            train = (
                manifest.filter(pl.col("pool").is_in(["train", "validation"]))
                .unique(subset="species")
                .sort("species")
            )
            if final is None:
                final = fit_scope(
                    config, store, train, "final_refit", model_dir, warnings,
                    expression, progress=progress,
                )
                write_site_scores(model_dir, final)
            else:
                warnings.extend(final_messages)
            needed = _prediction_orthogroups(config, final)
            target_species = (
                sorted(set(all_species) - set(training_species))
                if weighted
                else (
                    manifest.filter(pl.col("pool").is_in(["external_test", "discovery_inference"]))[
                        "species"
                    ]
                    .unique()
                    .to_list()
                )
            )
            if expression is not None and target_species:
                needed_expression = sorted({
                    og for member in final if member.expression is not None
                    for og in member.expression.orthogroups
                })
                expression.matrix(target_species, needed_expression)
            if target_species:
                with AlignmentStore(
                    Path(config.data.alignment_dir or ""),
                    all_species,
                    cache_species=target_species,
                    only_orthogroups=needed,
                    sequence_tpm_path=tpm_path,
                    sequence_tpm_cache_dir=tpm_cache,
                    progress=progress,
                ) as targets:
                    _verify_axes(
                        targets, {item["orthogroup"]: item["sha256"] for item in identities}
                    )
                    if weighted:
                        assert targets.tpm_identity is not None and store.tpm_identity is not None
                        if targets.tpm_identity["sha256"] != store.tpm_identity["sha256"]:
                            raise AlignmentError(
                                "Sequence TPM changed between training and prediction"
                            )
                        _write(
                            model_dir / "sequence_tpm_input.tsv",
                            [{"scope": "prediction_pool", **targets.tpm_identity}],
                        )
                    _save_bundle(
                        config,
                        run_dir / "model_bundle",
                        final,
                        store,
                        all_species,
                        target_store=targets,
                    )
                    for pool, stage in [
                        ("external_test", "external_test"),
                        ("discovery_inference", "inference"),
                    ]:
                        target = manifest.filter(pl.col("pool") == pool).unique(subset="species")
                        if target.height:
                            dest = run_dir / stage / "tables"
                            dest.mkdir(parents=True, exist_ok=True)
                            prediction = predict_members(
                                config, targets, target["species"].to_list(), final, tables=dest,
                                expression=expression,
                            )
                            prediction = prediction.join(
                                target.select("species", pl.col("label").alias("true_label")),
                                on="species",
                            )
                            prediction.write_csv(dest / f"pred_{stage}.tsv", separator="\t")
                            _write(
                                dest / "selected_site_sequence_map.tsv",
                                _coordinate_rows(targets, final, target["species"].to_list()),
                            )
            else:
                _save_bundle(config, run_dir / "model_bundle", final, store, all_species)
            _write(
                model_dir / "selected_site_sequence_map.tsv",
                _coordinate_rows(store, final, training_species),
            )
        from phenoradar.amino_acid_figures import write_amino_acid_figures

        if progress:
            progress("Generate evaluation, selection and interpretation figures.")
        warnings.extend(write_amino_acid_figures(
            run_dir, config, final if config.runtime.execution_stage == "full_run" else None,
            report=reports, store=store,
        ))
        _write(
            model_dir / "thresholds.tsv",
            [
                {
                    "threshold_name": "fixed_probability_threshold",
                    "threshold_value": 0.5,
                    "source": "constant",
                    "policy": "fixed_constant",
                    "derived_from_cv": False,
                }
            ],
        )
        _write(model_dir / "evaluation_contract.tsv", evaluation_metric_contract_rows())
        contract = {
            "evaluation_contract_version": 1,
            "label_unit": "species",
            "trait_col": config.data.trait_col,
            "group_col": config.split.group_col,
            "metric_contract": evaluation_metric_contract(),
        }
        input_hashes = {
            "metadata": sha256_file(Path(config.data.metadata_path)),
            **({"tpm": expression_identity["sha256"]} if expression_identity else {}),
            **{f"alignment:{i['orthogroup']}": i["sha256"] for i in identities},
            **(
                {"sequence_tpm": store.tpm_identity["sha256"]}
                if weighted and store.tpm_identity
                else {}
            ),
        }
        dataset = dataset_fingerprint(
            input_hashes,
            feature_mode=config.data.feature_mode,
        )
        split_hash = split_fingerprint(manifest)
        metadata_out = {
            "run_id": run_dir.name,
            "feature_mode": config.data.feature_mode,
            "aggregation": config.preprocess.amino_acid.aggregation,
            "status": "full_run_completed"
            if config.runtime.execution_stage == "full_run"
            else "cv_completed",
            "execution_stage": config.runtime.execution_stage,
            "start_time": started.isoformat(),
            "end_time": datetime.now(UTC).isoformat(),
            "duration_sec": (datetime.now(UTC) - started).total_seconds(),
            "dataset_fingerprint": dataset,
            "input_sha256": input_hashes,
            "split_fingerprint": split_hash,
            "experiment_fingerprint": experiment_fingerprint(
                dataset_sha256=dataset, split_sha256=split_hash, evaluation_contract=contract
            ),
            "evaluation_contract": contract,
            "build": phenoradar_build_snapshot(),
            "warnings": warnings,
            "study_context": study_context,
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata_out, indent=2) + "\n")
    return run_dir


def _prediction_orthogroups(config: AppConfig, members: list[AAMember]) -> set[str]:
    compact = prediction_members(members, padding_sites=config.figures.top_features)
    return {site.orthogroup for member in compact for site in member.transform.sites}


def _save_bundle(
    config: AppConfig,
    destination: Path,
    members: list[AAMember],
    store: AlignmentStore,
    species: list[str],
    *,
    target_store: AlignmentStore | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(members, destination / "model_state.joblib")
    write_resolved_config(config, destination / "resolved_config.yml")
    needed = _prediction_orthogroups(config, members)
    payload_files = ["model_state.joblib", "resolved_config.yml"]
    weighted = config.preprocess.amino_acid.aggregation == "tpm_weighted_frequency"
    if weighted:
        with gzip.open(
            destination / "sequence_tpm.tsv.gz", "wt", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["species", "sequence_id", "tpm"])
            seen: set[tuple[str, str]] = set()
            for source in [store, *([target_store] if target_store is not None else [])]:
                for name in sorted(needed):
                    og = source.ogs[name]
                    assert og.tpm is not None
                    for sample, rows in og.species_rows.items():
                        for row in rows:
                            key = sample, og.sequence_ids[row]
                            if key not in seen:
                                writer.writerow([*key, float(og.tpm[row])])
                                seen.add(key)
        payload_files.append("sequence_tpm.tsv.gz")
    manifest = {
        "feature_mode": config.data.feature_mode,
        "bundle_format_version": (
            "combined-1" if config.data.feature_mode == "combined" else
            "aa-3"
            if any(member.transform.representative_columns is not None for member in members)
            else "aa-2"
            if weighted
            else "aa-1"
        ),
        "sequence_tpm_source": store.tpm_identity,
        "expression_source": (
            file_identity(Path(config.data.tpm_path))
            if config.data.feature_mode == "combined" else None
        ),
        "representative_selection": (
            "quality_filtered_sites"
            if any(member.transform.representative_columns is not None for member in members)
            else "retained_sites"
            if config.preprocess.amino_acid.aggregation == "best_covered"
            else None
        ),
        "species": species,
        "alignment_hashes": {
            i["orthogroup"]: i["sha256"] for i in store.identities() if i["orthogroup"] in needed
        },
        "files": {name: file_identity(destination / name)["sha256"] for name in payload_files},
    }
    (destination / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _verify_axes(store: AlignmentStore, expected: dict[str, str]) -> None:
    for name, og in store.ogs.items():
        if expected.get(name) != og.sha256:
            raise AlignmentError(
                f"Alignment coordinate identity mismatch for {name}; "
                "use the original frozen alignment"
            )


def predict_amino_acid(
    bundle: Path,
    destination: Path,
    *,
    metadata_path: Path | None = None,
    alignment_dir: Path | None = None,
    sequence_tpm_path: Path | None = None,
    sequence_tpm_cache_dir: Path | None = None,
    n_jobs: int | None = None,
    tpm_path: Path | None = None,
    summary_group_col: str | None = None,
    warning_messages: list[str] | None = None,
) -> Path:
    from phenoradar.config import load_and_resolve_config

    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    if manifest.get("bundle_format_version") not in {"aa-1", "aa-2", "aa-3", "combined-1"}:
        raise AlignmentError("Unsupported amino-acid bundle version")
    for name in [
        "model_state.joblib",
        "resolved_config.yml",
        *(["sequence_tpm.tsv.gz"] if "sequence_tpm.tsv.gz" in manifest["files"] else []),
    ]:
        if sha256_file(bundle / name) != manifest["files"][name]:
            raise AlignmentError(f"Bundle checksum mismatch: {name}")
    config = load_and_resolve_config([bundle / "resolved_config.yml"])
    combined = manifest.get("feature_mode") == "combined"
    if combined != (config.data.feature_mode == "combined"):
        raise AlignmentError("Bundle feature mode does not match its frozen configuration")
    if tpm_path is not None:
        if not combined:
            raise AlignmentError("OG expression input requires a combined model bundle")
        config.data.tpm_path = str(tpm_path)
    weighted = config.preprocess.amino_acid.aggregation == "tpm_weighted_frequency"
    if weighted and "sequence_tpm.tsv.gz" not in manifest["files"]:
        raise AlignmentError("Weighted bundle is missing its sequence TPM payload")
    if sequence_tpm_path is not None and not weighted:
        raise AlignmentError("Sequence TPM input requires a tpm_weighted_frequency model bundle")
    weights_path = (sequence_tpm_path or bundle / "sequence_tpm.tsv.gz") if weighted else None
    if n_jobs is not None:
        config.runtime.n_jobs = n_jobs
    members = joblib.load(bundle / "model_state.joblib")
    required_ogs = _prediction_orthogroups(config, members)
    if required_ogs - set(manifest["alignment_hashes"]):
        raise AlignmentError("Frozen bundle is missing required alignment identities")
    species = manifest["species"]
    targets = (
        species
        if metadata_path is None
        else pl.read_csv(metadata_path, separator="\t")[config.data.species_col]
        .cast(pl.String)
        .str.strip_chars()
        .to_list()
    )
    if not targets or len(set(targets)) != len(targets) or any(not s for s in targets):
        raise AlignmentError("Prediction species must be nonempty and unique")
    if set(targets) - set(species):
        raise AlignmentError(
            "Initial sequence prediction requires species in the frozen alignments"
        )
    with ExitStack() as stack:
        expression = (
            stack.enter_context(ExpressionInput(config, targets, [])) if combined else None
        )
        if expression is not None:
            names = sorted({og for m in members for og in m.expression.orthogroups})
            expression.matrix(targets, names)
        store = stack.enter_context(
            AlignmentStore(
                alignment_dir or Path(config.data.alignment_dir or ""),
                species,
                cache_species=targets,
                only_orthogroups=required_ogs,
                sequence_tpm_path=str(weights_path) if weights_path is not None else None,
                sequence_tpm_cache_dir=sequence_tpm_cache_dir,
            )
        )
        _verify_axes(store, manifest["alignment_hashes"])
        destination.mkdir(parents=True, exist_ok=True)
        tables = destination / "predict/tables"
        tables.mkdir(parents=True, exist_ok=True)
        write_site_scores(tables, members)
        if expression is not None:
            for member in members:
                write_combined_coefficients(tables, member)
            _write(tables / "expression_input.tsv", [file_identity(Path(config.data.tpm_path))])
        if store.tpm_identity:
            _write(
                tables / "sequence_tpm_input.tsv",
                [{"source": "override" if sequence_tpm_path else "bundle", **store.tpm_identity}],
            )
        predictions = predict_members(config, store, targets, members, tables=tables,
                                      expression=expression)
        predictions.write_csv(tables / "pred_predict.tsv", separator="\t")
        from phenoradar.group_summary_outputs import write_group_summary_artifacts

        summary_config = config.model_copy(deep=True)
        if metadata_path is not None:
            summary_config.data.metadata_path = str(metadata_path)
        if summary_group_col is not None:
            summary_config.summary.group_col = summary_group_col
        summary_warnings = write_group_summary_artifacts(
            run_dir=destination, stage="predict", predictions=predictions,
            source_table_name="pred_predict.tsv", config=summary_config,
        )
        if warning_messages is not None:
            warning_messages.extend(summary_warnings)
        _write(tables / "selected_site_sequence_map.tsv", _coordinate_rows(store, members, targets))
    return destination
