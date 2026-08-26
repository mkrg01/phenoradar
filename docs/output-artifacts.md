# Output Artifacts and Interpretation

This page describes run-time outputs, when they are written, and how to interpret
them for `run` / `predict` / `report`.

## Quick reading order

For one `run` result directory, a practical order is:

1. `run_metadata.json` (status, warnings, pool counts, timing)
2. `runtime/tables/timing.tsv` (stage, fold, sample-set, and candidate bottlenecks)
3. `cv/tables/metrics_cv.tsv`, `cv/tables/loss_by_split_cv.tsv`,
   `model/tables/evaluation_contract.tsv`, `model/tables/thresholds.tsv`, and
   `summary/tables/classification_summary.tsv`
   (overall quality, train/validation loss gap, thresholds, and threshold-wise classification tradeoffs)
4. `cv/tables/prediction_cv.tsv`, `cv/figures/roc_curve_cv.svg`, and
   `cv/figures/pr_curve_cv.svg`
   (overall CV ranking behavior)
5. `external_test/tables/prediction_external_test.tsv` /
   `inference/tables/prediction_inference.tsv` (`full_run` only)
6. `cv/tables/feature_importance.tsv`, `cv/tables/coefficients.tsv`, and
   `cv/tables/feature_stability_summary.tsv`
   (model interpretation and outer-fold stability)

## Run directory layout

`phenoradar run` writes:

- `runs/<timestamp>_run_<id>/...`

When scalar-list condition expansion is used, it instead writes:

- `runs/<timestamp>_study_<id>/...`

The study directory contains the archived source config, one shared split,
ordered condition manifest, complete run artifacts under `conditions/`, and
cross-condition outputs:

- `condition_manifest.tsv`
- `config_differences.tsv`
- `study_metadata.json`
- `split/tables/split_manifest.tsv`
- `tables/condition_metrics.tsv`
- `tables/pairwise_comparisons.tsv`
- `tables/training_group_sensitivity.tsv` for a pure
  `sampling.training_group_count`/`group_subsample_repeats` study
- `figures/condition_metrics.{svg,pdf,png}`
- `figures/training_group_sensitivity.{svg,pdf,png}` for that training-group
  sweep
- `figures/ranked_feature_sensitivity.{svg,pdf,png}` when both ranked-filter
  method and `max_features` vary as a complete two-method grid
- `figures/ranked_feature_method_difference.{svg,pdf,png}` for the matched
  second-method improvement over the first method at each `max_features`

`condition_metrics.tsv` reports absolute OOF metrics and available group-bootstrap
intervals for every condition. `pairwise_comparisons.tsv` contains every unordered
condition pair. `improvement_a_over_b` is oriented so that positive values always
favor condition A, including for loss metrics. Pairwise intervals use matched
bootstrap replicate IDs generated from the shared groups and seed.
`training_group_sensitivity.tsv` aggregates condition point estimates across
group-subset repeats and reports their mean, standard deviation, quartiles, and
range. The corresponding figure shows the mean, interquartile band, and observed
minimum-to-maximum range across repeats against the effective mean number of
training groups per fold.

Run outputs use a stage-first layout. Stage-specific TSVs are placed in
`<stage>/tables/`, and stage-specific SVGs are placed in `<stage>/figures/`.
The main stage directories are `split/`, `cv/`, `model/`, `summary/`, `runtime/`,
`external_test/`, and `inference/`.

`phenoradar predict` writes:

- `runs/<timestamp>_predict_<id>/...`

`phenoradar report` writes:

- `reports/<timestamp>_report_<id>/...`

## `run` artifacts (schemas and write conditions)

Always written:

- `resolved_config.yml`
  - composed + validated config used in execution
- `split/tables/split_manifest.tsv`
  - columns: `species`, `pool`, `fold_id`, `group_id`, `contrast_group_id`, `label`
  - pools: `train`, `validation`, `external_test`, `discovery_inference`
  - `train` / `validation` rows are the per-fold expansion of the internal
    `training_validation` pool.
- `split/tables/fold_validation_groups.tsv`
  - columns: `fold_id`, `group_id`, `n_validation_species`, `n_validation_pos`,
    `n_validation_neg`, `validation_label_profile`
  - one row per validation-side group in each outer fold
  - `validation_label_profile`: `both`, `positive_only`, or `negative_only`
  - for `logo`, each `fold_id` has exactly one row
  - for `group_kfold` and `stratified_group_kfold`, a `fold_id` can have
    multiple rows
- `split/tables/fold_diagnostics.tsv`
  - one row per outer fold, written before CV execution results
  - columns: `fold_id`, train/validation group and species counts,
    train/validation positive and negative counts, `train_label_profile`,
    `validation_label_profile`, and `two_class_validation_metrics_defined`
  - use this table with `fold_validation_groups.tsv` to audit taxonomic-block
    assignments and identify single-label validation folds
- `model/tables/training_group_subsets.tsv`
  - one audit row per available group in every outer fold, plus final-refit rows
    during `full_run`
  - records the requested and effective group counts, deterministic group rank,
    generated repeat index, selected flag, and per-group label/species counts
  - `group_id` is a value of `split.group_col`; selection changes training rows
    only and does not alter `split_manifest.tsv`
- `cv/tables/metrics_cv.tsv`
  - columns: `aggregate_scope`, `fold_id`, `metric`, `metric_value`, `n_pos`, `n_neg`, `n_valid_folds`
  - `aggregate_scope`: per-fold rows use `NA`, aggregate rows use `macro`/`micro`
- `cv/tables/loss_by_split_cv.tsv`
  - columns: `fold_id`, `split`, `metric`, `metric_value`
  - current `split` values: `train`, `validation`
  - current `metric` value: `log_loss`
- `model/tables/thresholds.tsv`
  - columns: `threshold_name`, `threshold_value`, `source`, `policy`,
    `derived_from_cv`, `selection_metric`, `selection_scope`
  - threshold names: `fixed_probability_threshold`
- `model/tables/evaluation_contract.tsv`
  - one row per metric key with its display name, exact implementation, optimization
    direction, input type, threshold dependency, and compatibility note
  - `pr_auc` is explicitly mapped to `sklearn.metrics.average_precision_score`
- `cv/tables/feature_importance.tsv`
  - columns: `feature`, `importance_mean`, `importance_std`, `n_models`, `n_folds`, `method`
- `cv/tables/feature_importance_by_fold.tsv`
  - columns: `fold_id`, `feature`, `importance_mean`, `n_models`, `method`
- `cv/tables/coefficients.tsv`
  - columns: `feature`, `coef_mean`, `coef_std`, `n_models`, `n_folds`, `method`, `reason`
- `cv/tables/coefficients_by_fold.tsv`
  - columns: `fold_id`, `feature`, `coef_mean`, `n_models`, `method`, `reason`
  - for non-linear models, coefficient values can be `NA` with `reason=unsupported_model_non_linear`
- `cv/tables/feature_stability_by_feature.tsv`
  - one row per feature, computed after all outer folds have completed
  - reports retention frequency, non-zero selection frequency, and selection
    frequency conditional on having survived preprocessing
  - for signed linear models, also reports positive/negative fold counts and
    coefficient-sign agreement; these fields are `NA` for non-linear models
- `cv/tables/feature_stability_by_fold_pair.tsv`
  - one row per outer-fold pair
  - `jaccard` compares the two final non-zero feature sets as
    `intersection / union`; it is `NA` when both sets are empty
- `cv/tables/feature_stability_summary.tsv`
  - one-row summary of selection counts, pairwise Jaccard statistics, and
    coefficient-sign agreement
  - stability artifacts reuse already fitted outer-fold models and are purely
    diagnostic; they do not affect feature selection, predictions, or scores
- `cv/tables/prediction_cv.tsv`
  - columns: `fold_id`, `species`, `label`, `prob`
  - optional `uncertainty_std` (ensemble size > 1)
- `<stage>/tables/group_summary_<group>.tsv` (when `summary.group_col` is present in metadata)
  - written for `cv`; also for `external_test` and `inference` in `full_run` when those prediction pools are non-empty
  - `<group>` is derived from `summary.group_col`; for example `family_id` writes `group_summary_family.tsv`
  - columns:
    - `group_col`, `group_id`, `group_name`
    - `n_species`, `n_true_positive`, `n_true_negative`, `n_pred_positive`,
      `pred_positive_rate`
    - `prob_min`, `prob_q1`, `prob_median`, `prob_mean`, `prob_q3`, `prob_max`
    - `uncertainty_mean`, `top_species`, `top_prob`
- `model/tables/feature_filter_counts.tsv`
  - columns:
    - `scope`, `fold_id`, `sample_set_id`
    - `n_features_before`
    - `n_features_after_sparse_feature_filter`
    - `n_features_after_low_variance`
    - `n_features_after_ranked_feature_filter`
    - `n_features_after_correlation`
    - `n_features_after_all`
- `model/tables/feature_filter_counts_summary.tsv`
  - columns:
    - `scope`, `stage`, `n_records`
    - `n_features_min`, `n_features_q1`, `n_features_median`, `n_features_mean`,
      `n_features_q3`, `n_features_max`
    - `retained_ratio_min`, `retained_ratio_q1`, `retained_ratio_median`,
      `retained_ratio_mean`, `retained_ratio_q3`, `retained_ratio_max`
- `model/tables/ranked_feature_scores.tsv`
  - one row per ranked-filter candidate in each outer-fold or final-refit sample set
  - includes `method`, `feature`, `effect`, `standard_error`, `score`, `rank`,
    `retained`, label/pair counts, requested/effective feature counts, and any
    skip reason
- `model/tables/retained_features.tsv`
  - columns:
    - `scope`, `fold_id`, `sample_set_id`, `feature`
- `model/tables/retained_features_summary.tsv`
  - columns:
    - `scope`, `fold_id`, `feature`
    - `retained_count`, `n_sample_sets`, `retained_rate`
- `model/tables/model_sparsity.tsv`
  - columns:
    - `scope`, `fold_id`, `sample_set_id`, `model_index`, `model_name`
    - `n_features_after_all`, `n_nonzero_features`, `nonzero_ratio`
    - `count_method`, `reason`
- `model/tables/model_sparsity_summary.tsv`
  - columns:
    - `scope`, `model_name`
    - `n_models`, `n_models_with_nonzero_count`
    - `n_nonzero_min`, `n_nonzero_median`, `n_nonzero_mean`, `n_nonzero_max`
    - `nonzero_ratio_min`, `nonzero_ratio_median`, `nonzero_ratio_mean`, `nonzero_ratio_max`
- `model/tables/convergence_diagnostics.tsv`
  - one row per estimator fit, including inner-CV candidate evaluation, selected outer-fold
    models, and final-refit candidate/model fits
  - columns:
    - `training_scope`, `fit_scope`, `fold_id`, `sample_set_id`
    - `selection_source_sample_set_id`, `candidate_index`, `inner_fold_id`, `model_index`
    - `model_name`, `estimator_class`, `convergence_applicable`, `converged`
    - `n_iter_max`, `n_iter_values_json`, `max_iter`
    - `convergence_warning_count`, `convergence_warning_message`, `params_json`
- `summary/tables/classification_summary.tsv`
  - columns:
    - `pool`, `fold_id`
    - `threshold_name`, `threshold_value`
    - `n_total`, `tp`, `fp`, `tn`, `fn`
    - `accuracy`, `precision`, `recall`, `f1`, `mcc`
    - `precision`: `NA` when `tp + fp = 0`
    - `recall`: `NA` when `tp + fn = 0`
    - `f1`: `NA` when precision/recall is undefined or both are zero
  - includes:
    - `validation_oof` pooled row (`fold_id=NA`)
    - `validation_oof` per-fold rows (`fold_id=<outer fold id>`)
    - `external_test` pooled row (`full_run` only, `fold_id=NA`)
- `run_metadata.json`
  - provenance and execution metadata (`status`, timings, seed policy, git/runtime snapshot, warnings)
  - software provenance: `provenance_schema_version`, `phenoradar_version`,
    `phenoradar_install_type`, `git_source`, `git_commit`, `git_dirty`, and
    `git_worktree_patch_sha256`
  - Git fields describe the imported PhenoRadar source checkout only. For an installed
    distribution they are left explicitly unavailable instead of inspecting the invocation
    directory or an enclosing unrelated repository.
  - comparison identity: `fingerprint_schema_version`, `dataset_fingerprint`,
    `split_fingerprint`, `experiment_fingerprint`, and `evaluation_contract`
  - `evaluation_contract.metric_contract` records exact metric implementations and the
    fixed, non-CV-derived classification threshold policy
  - `dataset_fingerprint` hashes metadata/TPM contents by semantic role (not their paths)
  - `split_fingerprint` hashes realized species pool/fold/group/label assignments
- `runtime/tables/timing.tsv`
  - columns: `scope`, `stage`, `fold_id`, `sample_set_id`, `candidate_index`,
    `started_at_sec`, `ended_at_sec`, `duration_sec`
  - always written for successful `run` commands
  - uses a monotonic wall clock; offsets are relative to one recorder created at
    command start
  - records top-level run stages, outer-CV/final-refit stages, each outer fold,
    sampled-set preprocessing/fitting, and model-selection candidate scoring
  - `run_metadata.json.timing.stage_duration_sec` repeats the top-level
    `scope=run` durations for machine-readable comparison
- stage-specific `figures/` directories
  - always creates `cv/figures/`, `external_test/figures/`, and `inference/figures/`.
  - always attempts:
    - `cv/figures/cv_metrics_overview.svg`
    - `cv/figures/cv_loss_by_split.svg`
    - `cv/figures/feature_importance_top.svg`
    - `cv/figures/top_feature_expression_by_confusion.svg`
    - `cv/figures/feature_importance_by_fold_heatmap.svg`
    - `cv/figures/coefficients_signed_top.svg`
    - `cv/figures/feature_stability_top.svg`
    - `cv/figures/feature_set_jaccard_heatmap.svg`
    - `cv/figures/cv_species_probability_by_trait.svg`
    - `cv/figures/cv_fold_trait_probability.svg`
    - `cv/figures/feature_filter_funnel.svg`
    - `cv/figures/non_zero_feature_count_by_fold.svg`
    - `cv/figures/probability_by_<group>.svg` (attempted when `summary.group_col` is present in metadata)
    - `cv/figures/model_selection_trials.svg` (candidate selection active)
    - `cv/figures/model_selection_one_se_curve.svg` (candidate selection active)
    - `cv/figures/roc_curve_cv.svg` (may be skipped with warning for degenerate folds)
    - `cv/figures/pr_curve_cv.svg` (may be skipped with warning for degenerate folds)
    - `external_test/figures/final_refit_loss_by_split.svg` (attempted in `full_run`)
    - `external_test/figures/feature_filter_funnel.svg` (attempted in `full_run`)
    - `external_test/figures/external_species_probability_by_trait.svg` (attempted in `full_run`; may be skipped with warning when external test set is empty)
    - `external_test/figures/external_confusion_matrix.svg` (attempted in `full_run`; may be skipped with warning when external test set is empty)
    - `external_test/figures/cv_external_metric_comparison.svg` (attempted in `full_run`; may be skipped with warning when the pooled CV or external-test summary is missing)
    - `external_test/figures/external_roc_curve.svg` (attempted in `full_run`; may be skipped with warning when external test labels are single-class)
    - `external_test/figures/external_pr_curve.svg` (attempted in `full_run`; may be skipped with warning when external test labels are single-class)
    - `inference/figures/inference_probability_distribution.svg` (attempted in `full_run`; may be skipped with warning when inference set is empty)
    - `inference/figures/species_probability_cv_and_inference.svg` (attempted in `full_run`; may be skipped with warning when inference set is empty)
    - `<stage>/figures/probability_by_<group>.svg` (attempted for non-empty prediction stages when `summary.group_col` is present in metadata)

Conditionally written:

- `cv/tables/group_bootstrap_metrics.tsv`
  (`evaluation.group_bootstrap.enabled=true`)
  - one row per metric (`roc_auc`, `pr_auc`, `balanced_accuracy`, `mcc`,
    `brier`, `log_loss`)
  - columns: `metric`, `point_estimate`, `ci_lower`, `ci_upper`,
    `confidence_level`, `n_resamples`, `n_valid_resamples`,
    `valid_resample_fraction`, `n_groups`, `group_col`, `bootstrap_method`,
    `seed`
- `cv/tables/group_bootstrap_replicates.tsv`
  (`evaluation.group_bootstrap.enabled=true`)
  - one row per bootstrap replicate and metric
  - columns: `resample_id`, `metric`, `metric_value`, `n_sampled_groups`,
    `n_unique_sampled_groups`, `n_species_with_multiplicity`, `n_pos`, `n_neg`
- `cv/figures/group_bootstrap_metrics.svg`
  (`evaluation.group_bootstrap.enabled=true`)
  - pooled OOF point estimates with configured percentile confidence intervals
- `external_test/tables/prediction_external_test.tsv` (`full_run` only)
  - columns:
    - `species`, `true_label`, `prob`
    - `pred_label_fixed_threshold`
    - optional `uncertainty_std`
- `inference/tables/prediction_inference.tsv` (`full_run` only)
  - columns:
    - `species`, `true_label`, `prob`
    - `pred_label_fixed_threshold`
    - optional `uncertainty_std`
  - `true_label` values are `NA` because inference labels are unknown
- `external_test/tables/loss_by_split_final_refit.tsv` (`full_run` only)
  - columns: `split`, `metric`, `metric_value`
  - current `split` values: `train`, `external_test` (external row is omitted when external pool is empty)
  - current `metric` value: `log_loss`
- `model_bundle/` (`full_run` only)
  - reusable inference bundle (see bundle section below)
- `cv/tables/ensemble_model_probs.tsv` (ensemble size > 1)
  - columns: `fold_id`, `model_index`, `species`, `prob`
- `cv/tables/model_selection_trials.tsv` (candidate selection active)
  - columns: `fold_id`, `sample_set_id`, `candidate_index`, `inner_fold_id`, `metric_name`, `metric_value`, `params_json`
- `cv/tables/model_selection_trials_summary.tsv` (candidate selection active)
  - columns: `fold_id`, `sample_set_id`, `candidate_index`, `metric_name`, `params_json`
  - columns: `n_inner_folds`, `n_valid_inner_folds`, `metric_value_mean`, `metric_value_std`, `metric_value_se`
- `model/tables/model_selection_selected.tsv` (candidate selection active)
  - columns:
    - `selection_scope`, `fold_id`, `sample_set_id`, `selection_source_sample_set_id`
    - `rank`, `candidate_index`, `metric_name`, `metric_value`, `metric_value_se`, `selection_rule`
    - `n_available_candidates`, `n_scored_candidates`
    - `selected_candidate_count_requested`, `selected_candidate_count_effective`
    - `params_json`

## Run interpretation guide

The section headings below use short artifact names. The write locations follow
the run layout above, for example CV tables live under `cv/tables/`, model
diagnostic tables under `model/tables/`, and CV figures under `cv/figures/`.

### Metric semantics (`metrics_cv.tsv`)

`metric` column meanings:

| metric | better direction | threshold dependent | interpretation note |
| --- | --- | --- | --- |
| `roc_auc` | higher | no | Ranking quality across all thresholds. |
| `pr_auc` | higher | no | Compatibility key for Average Precision (`sklearn.metrics.average_precision_score`), not trapezoidal PR-curve area. |
| `balanced_accuracy` | higher | yes | Mean of sensitivity and specificity at fixed threshold. |
| `mcc` | higher | yes | Correlation-like binary metric; robust under imbalance. |
| `brier` | lower | no | Probability calibration error (squared). |

For a single-label validation fold, `roc_auc`, `pr_auc`,
`balanced_accuracy`, and `mcc` are explicitly written as `NA`. Brier score and
validation log loss remain defined. Aggregate `micro` metrics are recomputed
from all pooled out-of-fold predictions and can therefore remain defined even
when individual folds are single-label.

`aggregate_scope` meaning:

- `NA`: per-fold row (`fold_id` is concrete 1-based fold index).
- `macro`: mean of fold metrics (each fold weighted equally; NaN-aware).
- `micro`: metric recomputed from all out-of-fold predictions pooled together.

`n_valid_folds` is metric-specific for aggregate rows:

- If some folds are not computable for a metric (for example single-class fold for AUC), this value shows how many folds were valid.

### Timing semantics (`runtime/tables/timing.tsv`)

- `scope=run` contains the major command stages such as split construction,
  outer CV, optional group bootstrap, final refit, artifact writing, and figure
  generation.
- `scope=outer_cv` and `scope=final_refit` contain stage-level measurements.
- `scope=outer_fold` adds `fold_id`; sampled-set and candidate rows also add
  `sample_set_id` and, where applicable, `candidate_index`.
- `started_at_sec` and `ended_at_sec` expose overlap between parallel work.
  Concurrent row durations must not be summed to estimate elapsed wall time.
- Use `stage=total` rows for outer-CV, final-refit, and per-fold wall-clock
  comparisons. Use the finer rows to locate the expensive preprocessing,
  candidate scoring, fitting, or prediction path.
- `stage=inner_cv_preprocessing` measures creation of the train-only
  preprocessed inner-fold cache before candidate scoring. It appears once per
  outer fold and source sample set when model selection is active.

### Core run artifact interpretation

#### `split_manifest.tsv`

- `pool`:
  - `train` / `validation`: species used in outer CV (same species appears across folds).
  - `external_test`: labeled species marked by `split.test_holdout_col`;
    evaluated only in `full_run`.
  - `discovery_inference`: unlabeled species; inference target in `full_run`.
  - `train` / `validation` are the per-fold representation of the internal
    `training_validation` pool.
  - Species marked by `split.exclude_col` are omitted from `split_manifest.tsv`.
- `fold_id`:
  - 1-based fold index for `train`/`validation`.
  - `NA` for `external_test` and `discovery_inference`.
- `group_id`: the `split.group_col` value used for CV groups.
- `contrast_group_id`: the `data.contrast_pair_col` value when configured;
  used by contrast-pair-specific features.
- `label`: known only where metadata has trait label.

#### `fold_validation_groups.tsv`

- Lookup table from outer `fold_id` to validation-side `group_id`.
- Use this when numeric `fold_id` values need to be interpreted later.
- In `logo`, this is the held-out group for each fold.
- In `group_kfold` and `stratified_group_kfold`, multiple validation groups can
  map to the same fold.
- `validation_label_profile` identifies two-class, positive-only, and
  negative-only validation groups.

#### `fold_diagnostics.tsv`

- Pre-CV audit table with one row per realized outer fold.
- Reports train/validation group counts, species counts, positive/negative
  counts, and label profiles.
- `two_class_validation_metrics_defined=false` means fold-level ROC AUC,
  Average Precision, balanced accuracy, and MCC will be written as `NA`.
- A single-label validation fold is allowed, but every training fold must still
  contain both labels.

#### `evaluation_contract.tsv`

- Machine-readable registry for every metric emitted or used for model selection.
- `implementation` names the exact scikit-learn function used.
- `input_type=predicted_label` metrics use the named fixed threshold; probability metrics
  are threshold-independent.
- The legacy-compatible key `pr_auc` has `display_name=Average Precision` and
  `implementation=sklearn.metrics.average_precision_score`.

#### `group_bootstrap_metrics.tsv` and `group_bootstrap_replicates.tsv` (optional)

- The resampling unit is the actual `split.group_col`, not necessarily family.
  For example, selecting `contrast_pair_id` or `order_id` bootstraps contrast
  pairs or orders, respectively.
- With `G` unique OOF groups, each replicate draws `G` groups with replacement
  and includes all member species. Repeated groups contribute their species
  repeatedly.
- The models are not refit. The tables quantify sampling uncertainty of pooled
  OOF performance under group-level resampling.
- `point_estimate` is computed once from all OOF rows; `ci_lower` and `ci_upper`
  are percentile bounds from finite bootstrap values.
- A replicate containing only one label has `NA` for metrics that require two
  labels. Use `n_valid_resamples` and `valid_resample_fraction` to assess how
  much information supported each interval.

#### `thresholds.tsv`

- `fixed_probability_threshold`:
  - Constant probability threshold `0.5`.
  - Used for `pred_label_fixed_threshold` and threshold-dependent metrics in CV.
- `policy` is `fixed_constant` and `derived_from_cv` is `false`.
- `selection_scope` is `NA` because the threshold is fixed rather than selected from CV.

#### `loss_by_split_cv.tsv`

- Fold-level final loss diagnostics using `log_loss`.
- `train`:
  - loss on sampled training subsets used for model fitting.
  - when multiple sampled sets exist, reported value is the mean across sampled-set ensembles.
- `validation`:
  - loss on fold validation data.
  - when multiple sampled sets exist, reported value is the mean across sampled-set ensembles.
- Use this table to compare train/validation gap by fold as an overfitting check.

#### `loss_by_split_final_refit.tsv` (`full_run`)

- Final-refit loss diagnostics using `log_loss`.
- `train`:
  - loss on sampled training subsets used in final refit.
  - when multiple sampled sets exist, reported value is the mean across sampled-set ensembles.
- `external_test`:
  - loss on external labeled data.
  - omitted when external-test pool is empty.
- Use this table to compare final-refit train vs external generalization gap.

#### `prediction_cv.tsv`

- One row per validation sample prediction in outer CV.
- `prob` is class-1 probability.
- `label` is true binary label.
- `uncertainty_std` (optional):
  - Standard deviation of per-model probabilities in ensemble.
  - Larger value means lower ensemble agreement.
- Use this file to inspect separation, calibration, and threshold effects without touching final-refit outputs.

#### `tree_prediction_cv_annotation.tsv` (optional)

- Written when `data.tree_path` is set.
- ggtree/Toytree-friendly tip annotation for CV species with non-empty
  `split.group_col`.
- Columns: `label`, `species`, `true_label`, `prob`, `pred_label`, `uncertainty_std`,
  `group_id`, `group_name`, `fold_id`.
- `group_id` is the `split.group_col` value. `group_name` is populated when a matching
  name column is available, such as `family_name` for `family_id`.
- `pred_label` uses the fixed threshold recorded in `thresholds.tsv` (currently `0.5`).

#### `tree_contrast_pairs_annotation.tsv` (optional)

- Written when `data.tree_path` is set.
- ggtree/Toytree-friendly metadata QC annotation for all species with non-empty
  `split.group_col`.
- Columns: `label`, `species`, `true_label`, `group_id`, `group_name`.
- Use this file to inspect which tree tips participate in split groups before
  interpreting prediction probabilities.

#### `tree_feature_heatmap_annotation.tsv` (optional)

- Written when `data.tree_path` is set.
- Long-form ggtree/Toytree-friendly feature heatmap values for species with non-empty
  `split.group_col` and the top `figures.top_features` features by `importance_mean`.
- Columns: `label`, `species`, `true_label`, `prob`, `group_id`, `group_name`, `feature_rank`,
  `feature`, `orthogroup_annotation_taxid`, `orthogroup_annotation`, `importance_mean`,
  `coef_mean`, `tpm`, `log2_tpm_plus1`, `z_score_log2_tpm`.
- `prob` is the out-of-fold predicted probability of label `1` when available.
- `log2_tpm_plus1` is `log2(TPM + 1)` after duplicate `(species, feature)` rows are
  summed; `z_score_log2_tpm` is computed within each feature across included species.
- `orthogroup_annotation_taxid` and `orthogroup_annotation` are populated when
  `data.orthogroup_annotation_path` is set.

#### `prediction_external_test.tsv` / `prediction_inference.tsv`

- `prob`: predicted probability of label `1`.
- `pred_label_fixed_threshold`: hard label from fixed threshold.
- `true_label`:
  - present in `prediction_external_test.tsv` (known trait labels).
  - present in `prediction_inference.tsv` as `NA` (labels are unknown).
- `uncertainty_std` (optional):
  - Standard deviation of per-model probabilities in ensemble.
  - Larger value means lower ensemble agreement.

#### `tree_prediction_external_annotation.tsv` / `tree_prediction_predict_annotation.tsv` (optional)

- Written when `data.tree_path` is set and the corresponding prediction table exists.
- External-test columns: `label`, `species`, `true_label`, `prob`, `pred_label`,
  `uncertainty_std`, `group_id`, `group_name`.
- Predict columns: `label`, `species`, `true_label`, `prob`,
  `pred_label_fixed_threshold`, `uncertainty_std`, `group_id`, `group_name`.
- The annotation TSV retains predicted species even when a species is absent from the tree;
  Toytree SVG output is pruned to species present in the tree.

#### `feature_filter_counts.tsv`

- One row per preprocessing result (`scope`, `fold_id`, `sample_set_id`).
- Values are stage-wise feature counts through:
  - raw (`n_features_before`)
  - sparse feature
  - low variance
  - pair aware
  - correlation
  - final (`n_features_after_all`)
- Use this table to inspect fold/sample-set-specific filtering behavior.

#### `feature_filter_counts_summary.tsv`

- Summary of `feature_filter_counts.tsv` grouped by (`scope`, `stage`).
- `*_q1`, `*_median`, and `*_q3` report the 25th, 50th, and 75th percentiles.
- `retained_ratio_*` is the ratio relative to `n_features_before`.
- Use this table for quick stage-wise trend checks without scanning all folds/sample sets.

#### `retained_features.tsv`

- One row per retained feature after preprocessing for a given
  (`scope`, `fold_id`, `sample_set_id`).
- `feature` is the feature name that survived all preprocessing filters.
- Use this table when you need the exact retained-feature list for each fold/sample set.

#### `retained_features_summary.tsv`

- Summary of `retained_features.tsv` grouped by (`scope`, `fold_id`, `feature`).
- `retained_count` is how many sampled sets retained that feature in the fold.
- `retained_rate = retained_count / n_sample_sets`.
- Use this table to compare feature retention across folds without scanning every sample set.

#### `model_sparsity.tsv`

- One row per fitted model in CV/final-refit scopes.
- `n_nonzero_features`:
  - count of non-zero features derived from model-specific signals
    (`coef_` for linear models, `feature_importances_` for random forest).
  - can be `NA` when unavailable, with the reason in `reason`.
- `nonzero_ratio`:
  - `n_nonzero_features / n_features_after_all` when count is available.

#### `model_sparsity_summary.tsv`

- Summary grouped by (`scope`, `model_name`).
- `n_models_with_nonzero_count` helps identify how many models exposed usable sparsity counts.

#### `convergence_diagnostics.tsv`

- `fit_scope=candidate_evaluation` identifies fits used to score hyperparameter candidates;
  `fit_scope=selected_model` identifies models used for predictions.
- `convergence_applicable=false` means the estimator has no iterative convergence contract
  (for example, Random Forest); it does not mean that fitting failed.
- For iterative estimators, `converged=false` means scikit-learn emitted a
  `ConvergenceWarning` during that fit. `n_iter_max` is the largest observed `n_iter_`, while
  `n_iter_values_json` preserves all observed values, including calibrated SVM sub-estimators.
- When any iterative fit is not converged, a compact summary is also stored in
  `run_metadata.json` `warnings`. Use this table to locate the affected fold/candidate before
  increasing `max_iter` or changing regularization.

#### `feature_importance.tsv`

- `importance_mean`:
  - Mean of fold-level normalized importance values.
  - Within each fold, normalized importance is averaged across fitted models first.
- `importance_std`:
  - Variation across fold-level mean importance values.
  - Large value suggests unstable feature reliance.
- `n_models` / `n_folds`:
  - Total fitted model count and outer-fold count used for the summary.
- `method`:
  - `coef_abs_l1_norm`: linear model coefficients (absolute, L1-normalized per model).
  - `feature_importances_l1_norm`: random forest importances (L1-normalized per model).

#### `feature_importance_by_fold.tsv`

- One row per (`fold_id`, `feature`).
- `importance_mean` is the mean normalized importance across fitted models in that fold.
- These fold-level values are the points and boxplot distribution in
  `cv/figures/feature_importance_top.svg` and the cells in
  `cv/figures/feature_importance_by_fold_heatmap.svg`.

#### `coefficients.tsv`

- `coef_mean` / `coef_std` summarize fold-level mean signed linear coefficients.
- Positive `coef_mean`: higher standardized feature value pushes probability toward class `1`.
- Negative `coef_mean`: pushes toward class `0`.
- For non-linear models, coefficient columns can be `NA` with
  `reason=unsupported_model_non_linear`.

#### `coefficients_by_fold.tsv`

- One row per (`fold_id`, `feature`).
- `coef_mean` is the mean signed coefficient across fitted linear models in that fold.
- These fold-level values are the points and boxplot distribution in
  `cv/figures/coefficients_signed_top.svg`.

#### Feature-stability tables

- `selection_frequency` is the fraction of all outer folds in which a feature's
  final model signal is non-zero (`abs(coef_mean) > 1e-12` for signed linear
  models; `importance_mean > 1e-12` otherwise).
- `selection_frequency_when_retained` separates model-selection instability from
  preprocessing instability by using only folds where the feature was retained
  in the denominator.
- `sign_agreement_rate` is the larger of the positive and negative non-zero fold
  counts divided by the number of non-zero folds. It is reported only when a
  signed coefficient is available and the feature is selected in at least two folds.
- Pairwise `jaccard` is `1` for identical non-empty sets, `0` for disjoint sets,
  and closer to `1` as outer folds choose more similar features.
- These are post-hoc outer-CV diagnostics. They must not be used to alter the
  evaluated folds retrospectively; doing so would leak validation information.

#### `classification_summary.tsv`

- Purpose:
  - threshold-wise confusion-matrix and standard classification metrics summary.
- `pool`:
  - `validation_oof`: out-of-fold validation predictions.
  - `external_test`: final-refit external test predictions (`full_run` only).
- `fold_id`:
  - per-fold rows for `validation_oof` (`1`, `2`, ...).
  - pooled row uses `NA`.
- `tp`, `fp`, `tn`, `fn`:
  - confusion matrix counts at each threshold.
- `accuracy`:
  - overall fraction of correct predictions (`(tp + tn) / n_total`).
- `precision`:
  - predicted-positive purity (`tp / (tp + fp)`).
  - `NA` when no positive predictions are made.
- `recall`:
  - true positive recall (`tp / (tp + fn)`).
  - `NA` when a pool has no positive labels.
- `f1`:
  - harmonic mean of precision and recall.
  - `NA` when precision/recall is undefined or both are zero.
- `mcc`:
  - Matthews correlation coefficient in `[-1, 1]`.
  - `1` is perfect agreement, `0` is no better than random-like agreement, `-1` is total disagreement.
- `threshold_name`:
  - currently `fixed_probability_threshold`.

#### `model_selection_trials.tsv` (when candidate selection is enabled)

- One row per inner-CV trial result.
- `metric_value`: score for one candidate on one inner fold.
- `params_json`: exact hyperparameter set for that candidate.
- Compare candidates by averaging `metric_value` within
  (`fold_id`, `sample_set_id`, `candidate_index`).

#### `model_selection_trials_summary.tsv` (when candidate selection is enabled)

- One row per candidate after aggregating all inner folds.
- Grouping keys are
  (`fold_id`, `sample_set_id`, `candidate_index`, `metric_name`, `params_json`).
- `metric_value_mean`:
  - inner-fold mean score (NaN values are ignored).
- `metric_value_std`:
  - population std (`ddof=0`) across valid inner-fold scores.
- `metric_value_se`:
  - `metric_value_std / sqrt(n_valid_inner_folds)`.
- `n_inner_folds` / `n_valid_inner_folds`:
  - total inner folds vs folds with valid numeric score.

#### `model_selection_selected.tsv` (when candidate selection is enabled)

- `selection_scope`:
  - `outer_fold`: selections used in outer CV training.
  - `final_refit`: selections used for full-run refit.
- `rank`: rank among selected candidates after applying `selection_rule`.
  - With `best`, direction depends on `metric_name`:
    - `mcc` / `balanced_accuracy`: higher is better.
    - `log_loss`: lower is better.
  - With `one_se`, candidates within one standard error of the best mean score
    are ranked by model simplicity first.
- `metric_value_se`:
  - standard error used by `selection_rule=one_se`; NaN when not available.
- `selection_source_sample_set_id`:
  - sampled set used for candidate selection (`reuse_first_sample_set` uses `0` for all rows).
- `selected_candidate_count_requested` vs `selected_candidate_count_effective`:
  - shows requested vs effective top-K after candidate availability and deduplication.
  - when `selected_candidate_percent` is used, `selected_candidate_count_requested`
    is the per-sampled-set count derived from that percentage.

### Run figures

- `cv/figures/cv_metrics_overview.svg`
  - Blue: macro, orange: micro.
  - Axis is drawn from observed metric range; the x-axis is placed at the zero score baseline.
- `cv/figures/cv_loss_by_split.svg`
  - Fold-wise final `log_loss` comparison of `train` vs `validation`.
  - Useful for quick overfitting diagnostics without per-iteration learning curves.
- `cv/figures/group_bootstrap_metrics.svg`
  (`evaluation.group_bootstrap.enabled=true`)
  - Pooled OOF metric estimates with group-bootstrap percentile intervals.
  - The annotation records the grouping column, group count, resample count,
    and smallest valid-replicate count across plotted metrics.
- `cv/figures/roc_curve_cv.svg`
  - Pooled OOF ROC curve.
  - Curve summarizes all folds together (not per-fold overlays).
- `cv/figures/pr_curve_cv.svg`
  - Pooled OOF precision-recall curve.
  - Curve summarizes all folds together (not per-fold overlays).
  - The title reports Average Precision, matching the implementation behind the
    compatibility metric key `pr_auc`.
- `cv/figures/feature_importance_top.svg`
  - Top `figures.top_features` features by mean fold-level `importance_mean`.
  - Horizontal boxplot plus fold-level points.
  - When `data.orthogroup_annotation_path` is set, labels are rendered as
    `full annotation (orthogroup ID)` instead of ID-only labels.
- `cv/figures/top_feature_expression_by_confusion.svg`
  - Small multiples for the same top-importance features, ordered by
    `importance_mean`.
  - Each panel shows `log2(TPM + 1)` boxplots and species-level points in the
    OOF `TP`, `FN`, `TN`, and `FP` groups at the fixed probability threshold.
  - Panel headings include the orthogroup annotation and ID, mean feature
    importance, and signed linear coefficient when available.
  - Correct predictions use circles and errors use crosses.
- `cv/figures/feature_importance_by_fold_heatmap.svg`
  - Top `figures.top_features` features by mean fold-level `importance_mean`.
  - Rows are features, columns are CV folds, and color is fold-level
    `importance_mean` from `feature_importance_by_fold.tsv`.
  - The continuous white-to-blue scale starts at zero, so unimportant fold-feature
    cells remain white and larger importances become darker blue.
  - When `data.orthogroup_annotation_path` is set, row labels are rendered as
    `full annotation (orthogroup ID)` instead of ID-only labels.
- `cv/figures/coefficients_signed_top.svg`
  - Top `figures.top_features` by absolute mean fold-level coefficient magnitude.
  - Horizontal boxplot plus fold-level points; right is positive and left is negative.
  - When `data.orthogroup_annotation_path` is set, labels are rendered as
    `full annotation (orthogroup ID)` instead of ID-only labels.
- `cv/figures/feature_stability_top.svg`
  - Top `figures.top_features` features ranked by outer-fold selection frequency.
  - Gray bars show preprocessing retention; colored bars show final non-zero
    selection. Color indicates the dominant coefficient sign when available.
- `cv/figures/feature_set_jaccard_heatmap.svg`
  - Symmetric outer-fold heatmap of final non-zero feature-set overlap.
  - The diagonal is `1`; off-diagonal values come from
    `feature_stability_by_fold_pair.tsv`.
- `cv/figures/cv_species_probability_by_trait.svg`
  - Out-of-fold species probabilities grouped by trait (`label`).
  - Boxplot with per-species points and trait-wise mean markers.
  - The dashed horizontal line marks the fixed probability threshold at `0.5`.
- `cv/figures/cv_fold_trait_probability.svg`
  - Fold-level probability distribution grouped by trait.
  - Useful for checking fold-to-fold drift or fold-specific overlap.
- `cv/figures/feature_filter_funnel.svg`
  - Outer-CV feature-count trend through the enabled `preprocess.*_filter` steps.
  - Line is median count; shaded band is IQR; dashed lines are min-max.
  - Legend identifies median/IQR/min-max.
- `cv/figures/non_zero_feature_count_by_fold.svg`
  - Fold-wise distribution of `n_nonzero_features` from `model_sparsity.tsv`.
  - Boxplots are shown when a fold has multiple models; points show individual models.
- `cv/figures/model_selection_trials.svg` (candidate selection active)
  - Panels are laid out automatically in a compact grid.
  - Candidate scores are shown as `metric_value_mean ± metric_value_se`.
  - All folds are shown; per fold, only the first `sample_set_id` is plotted.
  - Y-axis labels include `candidate_index` and parameter JSON
    (keys fixed across candidates in the panel are omitted).
- `cv/figures/model_selection_one_se_curve.svg` (candidate selection active)
  - Shows candidate mean score with SE, one-SE threshold, best mean candidate,
    one-SE-eligible candidates, and the selected candidate.
  - Uses `log10(C)` on the x-axis when all candidates expose positive `C`;
    otherwise falls back to `candidate_index`.
  - All folds are shown; per fold, only the first `sample_set_id` is plotted.
- `external_test/figures/external_species_probability_by_trait.svg` (`full_run` with external samples)
  - External-test species probabilities grouped by `true_label`.
  - Boxplot with per-species points and trait-wise mean markers.
- `external_test/figures/external_confusion_matrix.svg` (`full_run` with external samples)
  - Fixed-threshold external-test confusion matrix.
  - Cells show counts and the row-wise percentage within each true-label class.
  - Side annotations report accuracy, precision, recall, specificity, F1, and MCC.
- `external_test/figures/cv_external_metric_comparison.svg` (`full_run` with external samples)
  - Grouped-bar comparison of pooled out-of-fold CV and external-test classification metrics.
  - Uses the fixed-threshold rows from `summary/tables/classification_summary.tsv`.
  - Shows accuracy, precision, recall, F1, and MCC.
- `external_test/figures/external_roc_curve.svg` / `external_test/figures/external_pr_curve.svg` (`full_run` with both external-test labels)
  - External-test ROC and precision-recall curves from `prediction_external_test.tsv`.
  - The ROC panel annotates ROC AUC. The PR panel annotates average precision and the external-test positive rate.
- `external_test/figures/feature_filter_funnel.svg` (`full_run`)
  - Final-refit feature-count trend through the enabled `preprocess.*_filter` steps.
  - Uses the model fit on the full training/validation pool rather than outer-CV folds.
- `<stage>/figures/probability_by_<group>.svg` (when `summary.group_col` is present in metadata)
  - Group-wise predicted probability distributions for the configured summary group.
  - Uses `summary.group_name_col` for y-axis labels when available.
  - For example, `summary.group_col: family_id` writes `probability_by_family.svg`.
- `inference/figures/inference_probability_distribution.svg` (`full_run` with inference samples)
  - Histogram of `prediction_inference.tsv` probabilities in bins
    `[0.0, 0.1), ... , [0.9, 1.0]`.
- `inference/figures/species_probability_cv_and_inference.svg` (`full_run` with inference samples)
  - Three-column comparison of CV out-of-fold probabilities for trait `0` and `1`
    plus unannotated inference probabilities.
  - The x-axis label uses the configured trait name, and the dashed horizontal line
    marks the fixed probability threshold at `0.5`.
- `external_test/figures/final_refit_loss_by_split.svg` (`full_run`)
  - Final-refit `log_loss` comparison of `train` and `external_test`.
  - Useful for quick train-vs-external generalization diagnostics.
- `cv/figures/tree_prediction_cv.svg` / `external_test/figures/tree_prediction_external.svg` (optional)
  - Written when `data.tree_path` is set and Toytree is available.
  - Rectangular Toytree view with aligned tracks for trait label, probability,
    predicted label, uncertainty, group, and fold where applicable.
- `cv/figures/tree_group.svg` (optional)
  - Written when `data.tree_path` is set and Toytree is available.
  - Rectangular Toytree view with trait-label and split-group tracks for metadata QC.
- `cv/figures/tree_feature_heatmap_zscore.svg` / `cv/figures/tree_feature_heatmap_log2_tpm.svg` (optional)
  - Written when `data.tree_path` is set and Toytree is available.
  - Rectangular Toytree views with top-feature heatmap tiles ordered by
    `importance_mean`; the feature count is controlled by `figures.top_features`.
  - Show the numeric trait label and predicted probability immediately before the
    feature heatmap.
  - Include an inline continuous color-bar legend showing the plotted value scale
    and missing-value color.
  - When `data.orthogroup_annotation_path` is set, feature labels are rendered as
    `orthogroup ID: full annotation` instead of ID-only labels.
  - The z-score figure emphasizes relative per-feature expression patterns; the
    log2-TPM figure preserves absolute expression scale after `log2(TPM + 1)`.

## `predict` artifacts (schemas and interpretation)

- `resolved_config.yml`
- `inference/tables/prediction_inference.tsv`
  - columns:
    - `species`, `true_label`, `prob`
    - `pred_label_fixed_threshold`
    - optional `uncertainty_std`
  - `true_label` values are `NA` because inference labels are unknown
  - `prob` is predicted probability of label `1`
- `inference/tables/group_summary_<group>.tsv` (when `summary.group_col` is present in metadata)
  - Same schema as run-stage grouped summaries.
- `run_metadata.json`
  - includes bundle manifest/payload hash values and bundle source metadata
  - includes the same PhenoRadar version/build provenance fields as `run`
  - copies `bundle_source_provenance_schema_version`,
    `bundle_source_phenoradar_version`, and `bundle_source_git_commit` from the bundle
    manifest without inferring values for legacy bundles
- `inference/figures/`
  - `predict_probability_distribution.svg`
  - `probability_by_<group>.svg` (when `summary.group_col` is present in metadata)
  - optional `predict_uncertainty.svg` (bundle ensemble size > 1)

### Predict figures

- `inference/figures/predict_probability_distribution.svg`
  - Histogram of predicted probabilities in bins `[0.0, 0.1), ... , [0.9, 1.0]`.
- `inference/figures/predict_uncertainty.svg` (ensemble only)
  - Top species by `uncertainty_std`; high bars indicate less stable predictions.
- `inference/figures/probability_by_<group>.svg`
  - Group-wise predicted probability distributions for `summary.group_col`.
- `inference/figures/tree_prediction_predict.svg` (optional)
  - Written when `data.tree_path` is set and Toytree is available.
  - Tree view with aligned tracks for true label when known, probability,
    fixed-threshold prediction, uncertainty, and group when available.

## `report` artifacts (schemas and interpretation)

- `report_manifest.json`
  - selected runs, options, skipped runs, ranked count
  - `generated_by` records the PhenoRadar version/build that generated the report
  - `report_options.metric_direction` is `maximize` or `minimize`
  - `report_options` also records the metric display name, implementation, and any
    fixed-threshold dependency
  - `experiment_compatibility` records verification status, fingerprints, unknown legacy
    runs, and whether mixed comparison was explicitly enabled
  - `software_compatibility` records observed PhenoRadar versions, legacy runs with unknown
    versions, dirty source-checkout runs, and whether versions are mixed
- `report_runs.tsv`
  - one row per included run after selection/filtering
  - columns: `run_id`, `run_dir`, `command`, `execution_stage`, `status`, `start_time`,
    `end_time`, `duration_sec`, software-provenance columns, `primary_metric`,
    `metric_contract_version`,
    `metric_display_name`,
    `metric_implementation`, `metric_threshold_name`, `metric_threshold_value`,
    `aggregate_scope`, `metric_value`,
    `fingerprint_schema_version`, `dataset_fingerprint`, `split_fingerprint`,
    `experiment_fingerprint`
  - `metric_value` can be `NA` (for example missing/invalid metrics in non-strict mode)
  - metric-definition columns are `NA` for legacy runs whose calculation contract cannot
    be verified; they are never inferred from the currently installed version
- `report_ranking.tsv`
  - ranked runs with non-null metric
  - columns: `run_id`, `run_dir`, `execution_stage`, `start_time`, `metric_name`,
    software-provenance columns, metric-definition columns, `aggregate_scope`, `metric_value`,
    fingerprint columns, `rank`
  - `rank` follows the selected metric's documented better direction: descending for
    `mcc`, `balanced_accuracy`, `roc_auc`, and `pr_auc`; ascending for `brier`
  - ties are ordered by `start_time`, then `run_id`, both ascending
- `report_warnings.tsv`
  - columns: `run_id`, `run_dir`, `warning_type`, `message`
  - aggregated ingestion warnings across runs; prioritize recurring `warning_type`
- optional narrative output by `--output-format`:
  - `report.md`, `report.html`, or `report.json`
- `figures/`
  - `report_metric_ranking.svg`
  - `report_metric_comparison.svg`
  - `report_stage_breakdown.svg` (only when more than one stage appears)

### Report figures

- `report_metric_ranking.svg`
  - Top ranked runs from `report_ranking.tsv`; the axis states whether higher or lower is better.
- `report_metric_comparison.svg`
  - Comparable runs with non-null metric from `report_runs.tsv`, ordered in the same better
    direction as `report_ranking.tsv`.
- `report_stage_breakdown.svg`
  - Count of runs by `execution_stage` (shown only when more than one stage exists).

## Model bundle layout (`model_bundle/`)

Files:

- `bundle_manifest.json`
  - records `threshold_name`, `threshold_fixed`, `threshold_policy`, and
    `threshold_derived_from_cv=false`
  - records the exporting build in `source_provenance_schema_version`,
    `source_phenoradar_version`, `source_phenoradar_install_type`, `source_git_source`,
    `source_git_commit`, `source_git_dirty`, and `source_git_worktree_patch_sha256`
- `feature_schema.tsv`
  - ordered union of features consumed by at least one fitted model
- `transform_feature_schema.tsv`
  - complete ordered raw-feature schema supplied to the expression transform during final refit
  - prediction raw values are aligned to this schema before sample-rank transforms
- `preprocess_state.joblib`
  - contains both feature schemas plus preprocessing method metadata
    (`expression_transform`, `feature_scaling`)
  - may include model-local preprocessing entries (`model_preprocess`),
    including selected features and optional scaler state
- `model_state.joblib`
- `thresholds.tsv`
- `resolved_config.yml`

Bundle loading enforces:

- supported `bundle_format_version`
- required file presence
- file inventory size/SHA-256 checks
- feature schema continuity and consistency with preprocess/model states

Bundle format compatibility:

- current exports use format version `2`.
- version `1` bundles using feature-wise `none` or `log1p` transforms remain loadable.
- version `1` rank-transform bundles cannot reconstruct the complete pre-transform schema and
  must be regenerated with the current PhenoRadar version.

## Status and warning interpretation

- Primary status values:
  - run: `cv_completed`, `full_run_completed`
  - predict: `predict_completed`
- CLI commands print warning summaries at completion when warnings are present.
- `run_metadata.json` `warnings` aggregates runtime and figure-generation warnings.
- Estimator non-convergence is captured in those warnings and detailed in
  `model/tables/convergence_diagnostics.tsv`; raw `ConvergenceWarning` messages are not left
  only on stderr.
- `report_warnings.tsv` includes per-run ingestion warnings during report aggregation.
