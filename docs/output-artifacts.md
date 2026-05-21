# Output Artifacts and Interpretation

This page describes run-time outputs, when they are written, and how to interpret
them for `run` / `predict` / `report`.

## Quick reading order

For one `run` result directory, a practical order is:

1. `run_metadata.json` (status, warnings, pool counts, timing)
2. `metrics_cv.tsv`, `loss_by_split_cv.tsv`, `thresholds.tsv`, and `classification_summary.tsv`
   (overall quality, train/validation loss gap, thresholds, and threshold-wise classification tradeoffs)
3. `prediction_cv.tsv` and `figures/roc_pr_curves_cv.svg` (overall CV ranking behavior)
4. `prediction_external_test.tsv` / `prediction_inference.tsv` (`full_run` only)
5. `feature_importance.tsv` and `coefficients.tsv` (model interpretation)

## Run directory layout

`phenoradar run` writes:

- `runs/<timestamp>_run_<id>/...`

`phenoradar predict` writes:

- `runs/<timestamp>_predict_<id>/...`

`phenoradar report` writes:

- `reports/<timestamp>_report_<id>/...`

## `run` artifacts (schemas and write conditions)

Always written:

- `resolved_config.yml`
  - composed + validated config used in execution
- `split_manifest.tsv`
  - columns: `species`, `pool`, `fold_id`, `group_id`, `contrast_group_id`, `label`
  - pools: `train`, `validation`, `external_test`, `discovery_inference`
  - `train` / `validation` rows are the per-fold expansion of the internal
    `training_validation` pool.
- `fold_validation_groups.tsv`
  - columns: `fold_id`, `group_id`, `n_validation_species`, `n_validation_pos`, `n_validation_neg`
  - one row per validation-side group in each outer fold
  - for `logo`, each `fold_id` has exactly one row
  - for `group_kfold`, a `fold_id` can have multiple rows
- `metrics_cv.tsv`
  - columns: `aggregate_scope`, `fold_id`, `metric`, `metric_value`, `n_pos`, `n_neg`, `n_valid_folds`
  - `aggregate_scope`: per-fold rows use `NA`, aggregate rows use `macro`/`micro`
- `loss_by_split_cv.tsv`
  - columns: `fold_id`, `split`, `metric`, `metric_value`
  - current `split` values: `train`, `validation`
  - current `metric` value: `log_loss`
- `thresholds.tsv`
  - columns: `threshold_name`, `threshold_value`, `source`, `selection_metric`, `selection_scope`
  - threshold names: `fixed_probability_threshold`, `cv_derived_threshold`
- `feature_importance.tsv`
  - columns: `feature`, `importance_mean`, `importance_std`, `n_models`, `n_folds`, `method`
- `feature_importance_by_fold.tsv`
  - columns: `fold_id`, `feature`, `importance_mean`, `n_models`, `method`
- `coefficients.tsv`
  - columns: `feature`, `coef_mean`, `coef_std`, `n_models`, `n_folds`, `method`, `reason`
- `coefficients_by_fold.tsv`
  - columns: `fold_id`, `feature`, `coef_mean`, `n_models`, `method`, `reason`
  - for non-linear models, coefficient values can be `NA` with `reason=unsupported_model_non_linear`
- `prediction_cv.tsv`
  - columns: `fold_id`, `species`, `label`, `prob`
  - optional `uncertainty_std` (ensemble size > 1)
- `feature_filter_counts.tsv`
  - columns:
    - `scope`, `fold_id`, `sample_set_id`
    - `n_features_before`
    - `n_features_after_low_prevalence`
    - `n_features_after_low_variance`
    - `n_features_after_pair_aware`
    - `n_features_after_correlation`
    - `n_features_after_all`
- `feature_filter_counts_summary.tsv`
  - columns:
    - `scope`, `stage`, `n_records`
    - `n_features_min`, `n_features_q1`, `n_features_median`, `n_features_mean`,
      `n_features_q3`, `n_features_max`
    - `retained_ratio_min`, `retained_ratio_q1`, `retained_ratio_median`,
      `retained_ratio_mean`, `retained_ratio_q3`, `retained_ratio_max`
- `retained_features.tsv`
  - columns:
    - `scope`, `fold_id`, `sample_set_id`, `feature`
- `retained_features_summary.tsv`
  - columns:
    - `scope`, `fold_id`, `feature`
    - `retained_count`, `n_sample_sets`, `retained_rate`
- `model_sparsity.tsv`
  - columns:
    - `scope`, `fold_id`, `sample_set_id`, `model_index`, `model_name`
    - `n_features_after_all`, `n_nonzero_features`, `nonzero_ratio`
    - `count_method`, `reason`
- `model_sparsity_summary.tsv`
  - columns:
    - `scope`, `model_name`
    - `n_models`, `n_models_with_nonzero_count`
    - `n_nonzero_min`, `n_nonzero_median`, `n_nonzero_mean`, `n_nonzero_max`
    - `nonzero_ratio_min`, `nonzero_ratio_median`, `nonzero_ratio_mean`, `nonzero_ratio_max`
- `classification_summary.tsv`
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
- `figures/`
  - always attempts:
    - `cv_metrics_overview.svg`
    - `cv_loss_by_split.svg`
    - `threshold_selection_curve.svg`
    - `feature_importance_top.svg`
    - `feature_importance_by_fold_heatmap.svg`
    - `coefficients_signed_top.svg`
    - `cv_species_probability_by_trait.svg`
    - `cv_fold_trait_probability.svg`
    - `feature_filter_funnel.svg`
    - `selected_features_by_fold_after_preprocessing.svg`
    - `non_zero_feature_count_by_fold.svg`
    - `model_selection_trials.svg` (candidate selection active)
    - `roc_pr_curves_cv.svg` (may be skipped with warning for degenerate folds)
    - `final_refit_loss_by_split.svg` (attempted in `full_run`)
    - `external_species_probability_by_trait.svg` (attempted in `full_run`; may be skipped with warning when external test set is empty)

Conditionally written:

- `prediction_external_test.tsv` (`full_run` only)
  - columns:
    - `species`, `true_label`, `prob`
    - `pred_label_fixed_threshold`, `pred_label_cv_derived_threshold`
    - optional `uncertainty_std`
- `prediction_inference.tsv` (`full_run` only)
  - columns:
    - `species`, `true_label`, `prob`
    - `pred_label_fixed_threshold`, `pred_label_cv_derived_threshold`
    - optional `uncertainty_std`
  - `true_label` values are `NA` because inference labels are unknown
- `loss_by_split_final_refit.tsv` (`full_run` only)
  - columns: `split`, `metric`, `metric_value`
  - current `split` values: `train`, `external_test` (external row is omitted when external pool is empty)
  - current `metric` value: `log_loss`
- `model_bundle/` (`full_run` only)
  - reusable inference bundle (see bundle section below)
- `ensemble_model_probs.tsv` (ensemble size > 1)
  - columns: `fold_id`, `model_index`, `species`, `prob`
- `model_selection_trials.tsv` (candidate selection active)
  - columns: `fold_id`, `sample_set_id`, `candidate_index`, `inner_fold_id`, `metric_name`, `metric_value`, `params_json`
- `model_selection_trials_summary.tsv` (candidate selection active)
  - columns: `fold_id`, `sample_set_id`, `candidate_index`, `metric_name`, `params_json`
  - columns: `n_inner_folds`, `n_valid_inner_folds`, `metric_value_mean`, `metric_value_std`
- `model_selection_selected.tsv` (candidate selection active)
  - columns:
    - `selection_scope`, `fold_id`, `sample_set_id`, `selection_source_sample_set_id`
    - `rank`, `candidate_index`, `metric_name`, `metric_value`
    - `n_available_candidates`, `n_scored_candidates`
    - `selected_candidate_count_requested`, `selected_candidate_count_effective`
    - `params_json`

## Run interpretation guide

### Metric semantics (`metrics_cv.tsv`)

`metric` column meanings:

| metric | better direction | threshold dependent | interpretation note |
| --- | --- | --- | --- |
| `roc_auc` | higher | no | Ranking quality across all thresholds. |
| `pr_auc` | higher | no | Useful under class imbalance; focuses on positive class retrieval. |
| `balanced_accuracy` | higher | yes | Mean of sensitivity and specificity at fixed threshold. |
| `mcc` | higher | yes | Correlation-like binary metric; robust under imbalance. |
| `brier` | lower | no | Probability calibration error (squared). |

`aggregate_scope` meaning:

- `NA`: per-fold row (`fold_id` is concrete 1-based fold index).
- `macro`: mean of fold metrics (each fold weighted equally; NaN-aware).
- `micro`: metric recomputed from all out-of-fold predictions pooled together.

`n_valid_folds` is metric-specific for aggregate rows:

- If some folds are not computable for a metric (for example single-class fold for AUC), this value shows how many folds were valid.

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
- In `group_kfold`, multiple validation groups can map to the same fold.

#### `thresholds.tsv`

- `fixed_probability_threshold`:
  - Comes from config (`report.fixed_probability_threshold`).
  - Used for `pred_label_fixed_threshold` and threshold-dependent metrics in CV.
- `cv_derived_threshold`:
  - Chosen from out-of-fold probabilities to maximize
    `report.auto_threshold_selection_metric` (`mcc` or `balanced_accuracy`).
  - Used for `pred_label_cv_derived_threshold`.
- `selection_scope`:
  - `outer_cv` indicates threshold was derived from CV OOF predictions.

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
  name column is available, such as `taxon_family_name` for `taxon_family_id`.
- `pred_label` uses the CV-derived threshold when available.

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
  `feature`, `importance_mean`, `coef_mean`, `tpm`, `log2_tpm_plus1`,
  `z_score_log2_tpm`.
- `prob` is the out-of-fold predicted probability of label `1` when available.
- `log2_tpm_plus1` is `log2(TPM + 1)` after duplicate `(species, feature)` rows are
  summed; `z_score_log2_tpm` is computed within each feature across included species.

#### `prediction_external_test.tsv` / `prediction_inference.tsv`

- `prob`: predicted probability of label `1`.
- `pred_label_fixed_threshold`: hard label from fixed threshold.
- `pred_label_cv_derived_threshold`: hard label from CV-derived threshold.
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
  `pred_label_fixed_threshold`, `pred_label_cv_derived_threshold`, `uncertainty_std`,
  `group_id`, `group_name`.
- The annotation TSV retains predicted species even when a species is absent from the tree;
  Toytree SVG output is pruned to species present in the tree.

#### `feature_filter_counts.tsv`

- One row per preprocessing result (`scope`, `fold_id`, `sample_set_id`).
- Values are stage-wise feature counts through:
  - raw (`n_features_before`)
  - low prevalence
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
  `feature_importance_top.svg` and the cells in
  `feature_importance_by_fold_heatmap.svg`.

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
  `coefficients_signed_top.svg`.

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
  - compare classification tradeoffs under `fixed_probability_threshold` and
    `cv_derived_threshold`.

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
- `n_inner_folds` / `n_valid_inner_folds`:
  - total inner folds vs folds with valid numeric score.

#### `model_selection_selected.tsv` (when candidate selection is enabled)

- `selection_scope`:
  - `outer_fold`: selections used in outer CV training.
  - `final_refit`: selections used for full-run refit.
- `rank`: rank among selected candidates by inner-CV score.
  - Direction depends on `metric_name`:
    - `mcc` / `balanced_accuracy`: higher is better.
    - `log_loss`: lower is better.
- `selection_source_sample_set_id`:
  - sampled set used for candidate selection (`reuse_first_sample_set` uses `0` for all rows).
- `selected_candidate_count_requested` vs `selected_candidate_count_effective`:
  - shows requested vs effective top-K after candidate availability and deduplication.
  - when `selected_candidate_percent` is used, `selected_candidate_count_requested`
    is the per-sampled-set count derived from that percentage.

### Run figures

- `cv_metrics_overview.svg`
  - Blue: macro, orange: micro.
  - Axis is drawn from observed metric range; the x-axis is placed at the zero score baseline.
- `threshold_selection_curve.svg`
  - x-axis: threshold, y-axis: selected score metric.
  - The x-axis is placed at the zero score baseline.
  - Red marker is the chosen `cv_derived_threshold`.
- `cv_loss_by_split.svg`
  - Fold-wise final `log_loss` comparison of `train` vs `validation`.
  - Useful for quick overfitting diagnostics without per-iteration learning curves.
- `roc_pr_curves_cv.svg`
  - Left: pooled OOF ROC, right: pooled OOF PR.
  - Curves summarize all folds together (not per-fold overlays).
- `feature_importance_top.svg`
  - Top `figures.top_features` features by mean fold-level `importance_mean`.
  - Horizontal boxplot plus fold-level points.
- `feature_importance_by_fold_heatmap.svg`
  - Top `figures.top_features` features by mean fold-level `importance_mean`.
  - Rows are features, columns are CV folds, and color is fold-level
    `importance_mean` from `feature_importance_by_fold.tsv`.
  - The continuous white-to-blue scale starts at zero, so unimportant fold-feature
    cells remain white and larger importances become darker blue.
- `coefficients_signed_top.svg`
  - Top `figures.top_features` by absolute mean fold-level coefficient magnitude.
  - Horizontal boxplot plus fold-level points; right is positive and left is negative.
- `cv_species_probability_by_trait.svg`
  - Out-of-fold species probabilities grouped by trait (`label`).
  - Boxplot with per-species points and trait-wise mean markers.
- `cv_fold_trait_probability.svg`
  - Fold-level probability distribution grouped by trait.
  - Useful for checking fold-to-fold drift or fold-specific overlap.
- `feature_filter_funnel.svg`
  - Feature-count trend by scope through the enabled `preprocess.*_filter` steps.
  - Line is median count; shaded band is IQR; dashed lines are min-max.
  - Legend identifies median/IQR/min-max; the figure annotates the `n_records` count.
- `selected_features_by_fold_after_preprocessing.svg`
  - Outer-fold retained-feature heatmap after preprocessing.
  - Rows are features, columns are folds, color is the feature retention rate across sampled sets.
  - Useful for spotting fold-specific preprocessing retention differences.
- `non_zero_feature_count_by_fold.svg`
  - Fold-wise distribution of `n_nonzero_features` from `model_sparsity.tsv`.
  - Boxplots are shown when a fold has multiple models; points show individual models.
- `model_selection_trials.svg` (candidate selection active)
  - Panels are laid out automatically in a compact grid.
  - Candidate scores are shown as `metric_value_mean ± metric_value_std`.
  - All folds are shown; per fold, only the first `sample_set_id` is plotted.
  - Y-axis labels include `candidate_index` and parameter JSON
    (keys fixed across candidates in the panel are omitted).
- `external_species_probability_by_trait.svg` (`full_run` with external samples)
  - External-test species probabilities grouped by `true_label`.
  - Boxplot with per-species points and trait-wise mean markers.
- `final_refit_loss_by_split.svg` (`full_run`)
  - Final-refit `log_loss` comparison of `train` and `external_test`.
  - Useful for quick train-vs-external generalization diagnostics.
- `tree_prediction_cv.svg` / `tree_prediction_external.svg` (optional)
  - Written when `data.tree_path` is set and Toytree is available.
  - Rectangular Toytree view with aligned tracks for trait label, probability,
    predicted label, uncertainty, group, and fold where applicable.
- `tree_group.svg` (optional)
  - Written when `data.tree_path` is set and Toytree is available.
  - Rectangular Toytree view with trait-label and split-group tracks for metadata QC.
- `tree_feature_heatmap_zscore.svg` / `tree_feature_heatmap_log2_tpm.svg` (optional)
  - Written when `data.tree_path` is set and Toytree is available.
  - Rectangular Toytree views with top-feature heatmap tiles ordered by
    `importance_mean`; the feature count is controlled by `figures.top_features`.
  - Show the numeric trait label and predicted probability immediately before the
    feature heatmap.
  - Include an inline continuous color-bar legend showing the plotted value scale
    and missing-value color.
  - The z-score figure emphasizes relative per-feature expression patterns; the
    log2-TPM figure preserves absolute expression scale after `log2(TPM + 1)`.

## `predict` artifacts (schemas and interpretation)

- `resolved_config.yml`
- `prediction_inference.tsv`
  - columns:
    - `species`, `true_label`, `prob`
    - `pred_label_fixed_threshold`, `pred_label_cv_derived_threshold`
    - optional `uncertainty_std`
  - `true_label` values are `NA` because inference labels are unknown
  - `prob` is predicted probability of label `1`
- `run_metadata.json`
  - includes bundle manifest/payload hash values and bundle source metadata
- `figures/`
  - `predict_probability_distribution.svg`
  - optional `predict_uncertainty.svg` (bundle ensemble size > 1)

### Predict figures

- `predict_probability_distribution.svg`
  - Histogram of predicted probabilities in bins `[0.0, 0.1), ... , [0.9, 1.0]`.
- `predict_uncertainty.svg` (ensemble only)
  - Top species by `uncertainty_std`; high bars indicate less stable predictions.
- `tree_prediction_predict.svg` (optional)
  - Written when `data.tree_path` is set and Toytree is available.
  - Tree view with aligned tracks for true label when known, probability,
    CV-threshold prediction, uncertainty, and group when available.

## `report` artifacts (schemas and interpretation)

- `report_manifest.json`
  - selected runs, options, skipped runs, ranked count
- `report_runs.tsv`
  - one row per included run after selection/filtering
  - columns: `run_id`, `run_dir`, `command`, `execution_stage`, `status`, `start_time`, `end_time`, `duration_sec`, `primary_metric`, `aggregate_scope`, `metric_value`
  - `metric_value` can be `NA` (for example missing/invalid metrics in non-strict mode)
- `report_ranking.tsv`
  - ranked runs with non-null metric
  - columns: `run_id`, `run_dir`, `execution_stage`, `start_time`, `metric_name`, `aggregate_scope`, `metric_value`, `rank`
  - `rank` order follows selected metric/scope, then tie-break by `start_time`, then `run_id`
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
  - Top ranked runs from `report_ranking.tsv`.
- `report_metric_comparison.svg`
  - Comparable runs with non-null metric from `report_runs.tsv`.
- `report_stage_breakdown.svg`
  - Count of runs by `execution_stage` (shown only when more than one stage exists).

## Model bundle layout (`model_bundle/`)

Files:

- `bundle_manifest.json`
- `feature_schema.tsv`
- `preprocess_state.joblib`
  - contains bundle feature schema plus preprocessing method metadata
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

## Status and warning interpretation

- Primary status values:
  - run: `cv_completed`, `full_run_completed`
  - predict: `predict_completed`
- CLI commands print warning summaries at completion when warnings are present.
- `run_metadata.json` `warnings` aggregates runtime and figure-generation warnings.
- `report_warnings.tsv` includes per-run ingestion warnings during report aggregation.
