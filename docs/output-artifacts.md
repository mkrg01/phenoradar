# Outputs and interpretation

Start with the files below in a run directory. Paths on this page describe
expression mode; [sequence and combined modes](amino-acid.md#outputs) add their
own tables and use different prediction filenames.

| Question | Read first |
| --- | --- |
| Did the run finish? | `run_metadata.json`: status and warnings |
| How well did it predict held-out species? | `cv/tables/metrics_cv.tsv` and `prediction_cv.tsv` |
| Which groups were held out? | `split/tables/fold_validation_groups.tsv` and `fold_diagnostics.tsv` |
| Are decisions withheld for missing information? | Each stage's `abstention_summary.tsv` |
| Which features mattered consistently? | `cv/tables/feature_importance.tsv` and `feature_stability_by_feature.tsv` |
| What did the final model predict? | `external_test/tables/prediction_external_test.tsv` and `inference/tables/prediction_inference.tsv` |
| What took the time? | `runtime/tables/timing.tsv` |

## Directory layout

```text
runs/<timestamp>_run_<id>/
  resolved_config.yml
  run_metadata.json
  split/tables/          # species pools and fold assignments
  cv/{tables,figures}/   # held-out evaluation and interpretation
  model/{tables,figures}/
  summary/tables/
  runtime/tables/
  external_test/{tables,figures}/  # full_run evaluation
  inference/{tables,figures}/      # full_run unknown-trait predictions
  model_bundle/                   # full_run reusable model
```

Stage directories may exist without data. Conditional tables and figures require
the corresponding pool, model support, or enabled setting. Missing or degenerate
plot inputs are recorded as warnings.

`predict` writes `runs/<timestamp>_predict_<id>/`;
`report` writes `reports/<timestamp>_report_<id>/`.
For multi-condition runs, see [studies](#studies).

## Predictions

| Table | Core columns | Meaning |
| --- | --- | --- |
| `cv/tables/prediction_cv.tsv` | `fold_id`, `species`, `label`, `prob` | One held-out prediction per CV species |
| `external_test/tables/prediction_external_test.tsv` | `species`, `true_label`, `prob`, `pred_label_fixed_threshold` | Final-model evaluation, `full_run` |
| `inference/tables/prediction_inference.tsv` | Same as external test; `true_label=NA` | Unknown-species predictions from `full_run` or expression `predict` |
| `inference/tables/prediction_inference_by_fold.tsv` | `fold_id`, `species`, `prob` | Unknown-species sensitivity to outer-fold training composition |
| `cv/tables/ensemble_model_probs.tsv` | `fold_id`, `model_index`, `species`, `prob` | Individual member probabilities when the ensemble has multiple models |

`prob` is class-1 probability. Hard labels use the fixed threshold 0.5;
`model/tables/thresholds.tsv` records `policy=fixed_constant` and
`derived_from_cv=false`.

Optional `uncertainty_std` is the standard deviation of member probabilities.
It measures ensemble disagreement, not a confidence interval. Unknown-species
predictions from outer folds are sensitivity diagnostics, not OOF performance
estimates for those species.

## Metrics

`cv/tables/metrics_cv.tsv` contains `aggregate_scope`, `fold_id`, `metric`,
`metric_value`, `n_pos`, `n_neg`, and `n_valid_folds`.

| Metric | Better | Interpretation |
| --- | --- | --- |
| `roc_auc` | Higher | Ranking across thresholds |
| `pr_auc` | Higher | Average Precision (`sklearn.metrics.average_precision_score`), not trapezoidal PR area |
| `balanced_accuracy` | Higher | Mean sensitivity and specificity at 0.5 |
| `mcc` | Higher | Matthews correlation coefficient at 0.5 |
| `brier` | Lower | Mean squared probability error |

`model/tables/evaluation_contract.tsv` records exact implementations, directions,
input types, and threshold dependencies, including model-selection log loss.

| `aggregate_scope` | Meaning |
| --- | --- |
| `NA` | One fold, with a concrete `fold_id` |
| `macro` | Mean of defined fold scores; each fold has equal weight |
| `micro` | Recomputed from all pooled OOF predictions |

Single-label validation folds have `NA` for ROC AUC, Average Precision, balanced
accuracy, and MCC. Brier score and log loss remain defined. `n_valid_folds` shows
the number contributing to each aggregate; pooled micro metrics can remain
defined when individual folds are single-label.

`summary/tables/classification_summary.tsv` gives pooled and per-fold OOF
confusion counts (`tp`, `fp`, `tn`, `fn`), accuracy, precision, recall, F1, and MCC.
`full_run` adds pooled external-test results. Precision is `NA` without predicted
positives; recall is `NA` without positive labels in the population; F1 is `NA`
when precision/recall is undefined or both are zero.

### Loss and group-bootstrap intervals

| Table | Content / condition |
| --- | --- |
| `cv/tables/loss_by_split_cv.tsv` | `fold_id`, `split`, `metric`, `metric_value`; train vs validation log loss |
| `external_test/tables/loss_by_split_final_refit.tsv` | Train vs external-test log loss in `full_run`; external row omitted if empty |
| `cv/tables/group_bootstrap_metrics.tsv` | Pooled OOF estimates, percentile bounds, confidence level, valid replicate counts, group and seed metadata |
| `cv/tables/group_bootstrap_replicates.tsv` | Per-replicate metric values, group counts, multiplicities, and label counts |

Loss diagnostics average sampled-set ensemble losses when multiple sampled sets
exist. Use the training/validation gap to inspect overfitting.

Bootstrap tables require `evaluation.group_bootstrap.enabled: true`. Each
replicate draws G groups with replacement from the G unique `split.group_col`
values, retaining every species and repeated-group multiplicity. Models are not
refitted. Check `n_valid_resamples` and `valid_resample_fraction`: single-label
replicates cannot support all metrics.

## Abstention

When enabled, prediction tables add:

| Column | Meaning |
| --- | --- |
| `information_coverage` | Model-weighted observation coverage, averaged across members |
| `abstention_threshold` | Saved fixed coverage threshold |
| `pred_label_selective` | Accepted fixed-0.5 label, or `NA` for abstention |
| `decision_status` | `accepted` or `abstained` |
| `abstention_reason` | `insufficient_information`, `no_informative_coefficients`, or `NA` |
| `missing_features_json` | Missing important features, ordered by coefficient weight |

Use `pred_label_selective` for downstream classification and candidate
selection. Coverage measures available information, not calibrated correctness;
see [coverage policies](configuration.md#missing-expression-and-abstention).

Each prediction stage also writes:

| Table under `<stage>/tables/` | Content |
| --- | --- |
| `abstention_summary.tsv` | Overall, label, fold, and available group summaries of accepted/abstained counts and selective performance |
| `abstention_features.tsv` | Missing evidence: `species`, `rank`, `feature`, `coefficient_fraction`; up to `figures.top_features` per species |
| `abstention_by_<group>.tsv` | Additional grouped decision summaries when metadata allows |

`decision_rate = n_accepted / n_species`. Selective metrics use accepted labeled
species only; all-abstained metrics are `NA`. `positive_error_rate` is
`FP / (TP + FP)` and `negative_error_rate` is `FN / (TN + FN)`; undefined ratios
are `NA`.

Raw metrics, loss, and classification summaries retain all species. Always
compare selective performance with decision rate. Grouped positive rates use
accepted species as the denominator, while probability summaries retain all
species. Candidate evidence includes accepted positives only.

Prediction/evaluation figures have all-species versions directly under
`<stage>/figures/` and accepted-only versions with the same names under
`<stage>/figures/accepted_only/`. These pairs are written even if nobody abstains;
empty accepted sets receive explanatory figures. Model interpretation, selection,
training loss, and bootstrap figures keep their original populations.

Neutral-expression evidence preserves raw TPM and adds `is_missing`. Missing
plot values appear unavailable rather than as zero expression. Missing inputs
and observations at the training mean both have zero standardized contribution,
but only the latter count as observed.

## Splits and model diagnostics

| Table | Content / use |
| --- | --- |
| `split/tables/split_manifest.tsv` | `species`, `pool`, `fold_id`, `fold_index`, `group_id`, `contrast_group_id`, `label` |
| `split/tables/fold_validation_groups.tsv` | Held-out groups per fold, species/label counts, and `validation_label_profile` |
| `split/tables/fold_diagnostics.tsv` | Train/validation group and label counts; `two_class_validation_metrics_defined` |
| `model/tables/training_group_subsets.tsv` | Requested/effective group counts, deterministic ranks, repeat index, selection flags |
| `model/tables/feature_filter_counts.tsv` | Feature counts before/after each filter by scope, fold, and sampled set |
| `model/tables/feature_filter_counts_summary.tsv` | Stage-wise count and retained-ratio summaries |
| `model/tables/ranked_feature_scores.tsv` | Signed effects, direction eligibility, errors/scores/ranks, retention flags, and skip reasons |
| `model/tables/retained_features.tsv` | Exact retained feature list per scope/fold/sampled set |
| `model/tables/retained_features_summary.tsv` | Retained counts and rates across sampled sets |
| `model/tables/model_sparsity.tsv` | Per-model nonzero counts, ratios, counting method, and unavailable reasons |
| `model/tables/model_sparsity_summary.tsv` | Sparsity summaries by scope and model |
| `model/tables/convergence_diagnostics.tsv` | Fit scope, candidate/model IDs, convergence status, iteration counts, warnings, and parameters |

Manifest pools are per-fold `train`/`validation`, held-out `external_test`, and
unknown `discovery_inference`. Excluded species are absent. `group_id` is the
CV group; `contrast_group_id` is the separately configured contrast pair.
External/inference rows have no fold ID.

For convergence diagnostics, `fit_scope=candidate_evaluation` identifies search
fits; `selected_model` identifies prediction models. `convergence_applicable=false`
means no iterative convergence contract, as for random forest. Logistic native
failures stop training; successful path counts include internal warm-start
points and are shared by candidates on that path. For SVM, `converged=false`
means at least one calibrated sub-estimator reached its iteration limit.

## Feature interpretation

| Table under `cv/tables/` | Content |
| --- | --- |
| `feature_importance.tsv` | `feature`, `importance_mean`, `importance_std`, model/fold counts, method |
| `feature_importance_by_fold.tsv` | Per-fold mean normalized importance |
| `coefficients.tsv` | Mean/std signed coefficients, counts, method, unavailable reason |
| `coefficients_by_fold.tsv` | Per-fold mean signed coefficients |
| `feature_stability_by_feature.tsv` | Preprocessing retention, nonzero selection, conditional selection, sign agreement |
| `feature_stability_by_fold_pair.tsv` | Nonzero-feature-set intersections, unions, and Jaccard similarity |
| `feature_stability_summary.tsv` | Overall selection, Jaccard, and sign-agreement summaries |

Importances are normalized per model, averaged within each fold, then across
folds. Standard deviations describe fold-level variation. Linear importance uses
absolute coefficients; random forest uses native feature importance. Signed
coefficients describe the configured transformed/scaled inputs; positive values
push toward class 1. Nonlinear coefficients are `NA` with an explanatory reason.

`selection_frequency` counts nonzero fold signals across all outer folds;
`selection_frequency_when_retained` conditions on surviving preprocessing.
`sign_agreement_rate` is the dominant nonzero sign fraction, reported for signed
features selected in at least two folds. Jaccard is intersection/union and is
`NA` for two empty sets. These diagnostics reuse fitted folds; changing the
evaluated feature set retrospectively would leak validation information.

`full_run` adds `model/tables/final_refit_feature_importance.tsv` and
`final_refit_coefficients.tsv`, plus corresponding `*_by_model.tsv` tables.
Their variation is across final ensemble members, not CV folds. A feature absent
from a member has zero importance/coefficient in that member's summary.

### Candidate selection diagnostics

| Table | Content / condition |
| --- | --- |
| `cv/tables/model_selection_trials.tsv` | Inner-fold scores and exact `params_json` per candidate/sample set |
| `cv/tables/model_selection_trials_summary.tsv` | Candidate mean, population std, SE, and valid fold counts |
| `model/tables/model_selection_selected.tsv` | Selected ranks, rule, source sample set, requested/effective counts |
| `model/tables/final_refit_model_selection_trials.tsv` and `*_summary.tsv` | Equivalent search evidence for final refit |

These require active candidate selection. Standard error is population standard
deviation divided by the square root of valid inner folds. The selected table
records `outer_fold` or `final_refit` scope and the rule used; percentage-based
selection records the derived requested count. See
[model selection](configuration.md#model_selection) for ranking rules.

## Figures and species evidence

Most figures are SVGs under the matching stage. `figures.top_features` limits
global or species-local features; optional OG annotations replace ID-only labels.

| Purpose | Figures under `cv/figures/` |
| --- | --- |
| Performance | `cv_metrics_overview.svg`, `roc_curve_cv.svg`, `pr_curve_cv.svg`, optional `group_bootstrap_metrics.svg` |
| Training gap | `cv_loss_by_split.svg` |
| Probability distributions | `cv_species_probability_by_trait.svg`, `cv_fold_trait_probability.svg`, `probability_by_<group>.svg` |
| Feature reliance | `feature_importance_top.svg`, `feature_importance_by_fold_heatmap.svg`, `coefficients_signed_top.svg` |
| Expression by TP/FN/TN/FP | `top_feature_expression_by_confusion.svg` using `log2(TPM + 1)` |
| Stability | `feature_set_jaccard_heatmap.svg` |
| Filtering and sparsity | `feature_filter_funnel.svg`, `non_zero_feature_count_by_fold.svg` |
| Candidate selection | `model_selection.svg` for `best`, or `model_selection_one_se_curve.svg` for `one_se` |

Selection curves show inner-CV means and SEs; only one-SE curves show the
one-SE eligibility threshold. Candidate details remain in tables. Final-refit
figures under `model/figures/` use the `final_refit_` prefix and summarize final
ensemble members.

External-test figures include probability distributions, confusion matrix,
ROC/PR curves, CV/external metric comparison, top-feature expression, and
`final_refit_loss_by_split.svg`. Inference figures include
`inference_probability_distribution.svg` and
`species_probability_cv_and_inference.svg`. Group plots require the configured
metadata column; ROC/PR plots require a usable label population.

### Local evidence PDFs

Linear models can produce:

| Directory | Species / fitted models used |
| --- | --- |
| `cv/figures/species_evidence/{false_positive,false_negative}/` | Raw fixed-0.5 OOF errors; only their actual held-out-fold models |
| `cv/figures/species_evidence/abstained/` | The same OOF error diagnostics for withheld decisions |
| `inference/figures/candidate_evidence/<probability_bin>/` | Accepted positive candidates; final models or loaded bundle members |

PDFs show member probabilities, signed local contributions, and target expression
against known-trait references when available. Local features are ranked by mean
absolute contribution; absent member features contribute zero. Contributions
add on the linear-score scale, not as probability components.

CV references include only species sampled for that target's training fold.
Full-run candidates use known-trait training/CV references and can show
outer-fold prediction sensitivity alongside the final probability. Expression
`predict` shows bundle-member probabilities instead. Nonlinear models skip
local linear PDFs with a warning.

Underlying tables include `cv_species_evidence.tsv`,
`cv_species_feature_evidence.tsv`, and `cv_species_reference_expression.tsv`
under `cv/tables/`; candidates use `candidate_evidence_candidates.tsv`,
`candidate_feature_evidence.tsv`, and `candidate_reference_expression.tsv` under
`inference/tables/`. Each PDF directory has a species/candidate manifest with
relative paths and probability summaries.

### Groups and trees

`<stage>/tables/group_summary_<group>.tsv` reports species/label counts, positive
decision rates, probability quantiles, uncertainty, and top species. Group names
come directly from `summary.group_col`.

With `data.tree_path`, tree tables include `tree_prediction_cv_annotation.tsv`,
`tree_contrast_pairs_annotation.tsv`, `tree_feature_heatmap_annotation.tsv`, and
stage-specific external/predict annotations. They support ggtree reuse; predicted
species absent from the tree stay in annotation tables.

Toytree figures include `tree_prediction_cv.svg`, `tree_group.svg`,
`tree_feature_heatmap_log2_tpm.svg`, `tree_feature_heatmap_zscore.svg`, and
stage-specific prediction trees. Heatmap values and annotations are retained in
the long-form table. Top features follow importance ranking; z-scores are within
feature across included species. Optional
[phylogenetic outputs](phylogenetic-imputation.md#outputs) live under `inference/`.

## Timing and provenance

`runtime/tables/timing.tsv` has `scope`, `stage`, `fold_id`, `sample_set_id`,
`candidate_index`, `started_at_sec`, `ended_at_sec`, and `duration_sec`.

Start with `scope=run, stage=total` (or `scope=predict` for expression prediction).
Narrow to outer folds, sampled sets, candidate scoring, or figure jobs to locate
cost. All offsets use one monotonic clock. Nested and parallel durations overlap;
do not sum them as elapsed time. Logistic path-scoring rows may have no candidate
index because one interval covers a whole path.

`run_metadata.json` records status, warnings, timing, input hashes, seed policy,
PhenoRadar version/build, and dataset/split/experiment fingerprints. Git fields
describe the imported PhenoRadar checkout; installed distributions do not borrow
Git identity from the invocation directory. Prediction additionally records
source-bundle hashes and available exporting-build provenance.

Successful statuses are `cv_completed`, `full_run_completed`, and
`predict_completed`. Warnings include skipped figures and convergence diagnostics;
review them before interpreting missing results.

## `predict` artifacts

Expression prediction writes effective `resolved_config.yml`, `run_metadata.json`,
`runtime/tables/timing.tsv`, and the inference predictions described above.
Optional inference outputs include grouped summaries, abstention tables, tree
annotations, and local evidence PDFs.

`inference/tables/candidate_model_probabilities.tsv` contains `species`,
zero-based `model_index`, and `prob` for fitted bundle members. Its manifest
reports `n_bundle_models` and probability summaries. Missing reference expression
is marked unavailable; it is never inferred from predictions.

Main figures are `predict_probability_distribution.svg`, optional
`predict_uncertainty.svg`, `probability_by_<group>.svg`, and
`tree_prediction_predict.svg` under `inference/figures/`.

## Studies

Multi-condition runs write `runs/<timestamp>_study_<id>/` with an archived source
config, shared split, and full run artifacts under `conditions/`.

| File | Meaning |
| --- | --- |
| `condition_manifest.tsv`, `config_differences.tsv` | Ordered conditions and changed settings |
| `study_metadata.json` | Study status and provenance |
| `tables/condition_metrics.tsv` | Absolute OOF metrics and available bootstrap intervals |
| `tables/pairwise_comparisons.tsv` | Every unordered condition pair; positive `improvement_a_over_b` favors A, including losses |
| `tables/training_group_sensitivity.tsv` | Group-count study summaries across subset repeats |
| `figures/condition_metrics.{svg,pdf,png}` | Metrics in configured condition order |
| `figures/training_group_sensitivity.{svg,pdf,png}` | Mean, IQR, and min–max across subset repeats |

Pairwise intervals match bootstrap replicate IDs on the shared groups and seed.
Sensitivity plots use effective mean training groups per fold. Legacy study
reports may also contain ranked-filter sensitivity and method-difference plots.

## `report` artifacts

| File | Meaning |
| --- | --- |
| `report_manifest.json` | Selection, options, skipped runs, comparison compatibility, and report-generating build |
| `report_runs.tsv` | Included runs, status/timing, software provenance, metric definitions/values, and fingerprints |
| `report_ranking.tsv` | Runs with valid metrics, sorted by the selected better direction and assigned ranks |
| `report_warnings.tsv` | `run_id`, `run_dir`, `warning_type`, `message` |
| `report.md`, `report.html`, or `report.json` | Optional narrative selected by `--output-format` |
| `figures/` | Metric ranking/comparison and multi-stage breakdown SVGs |

Legacy unknown metric definitions remain `NA`; they are not guessed from the
installed version. The manifest separates experiment compatibility from software
version/build warnings. See [report controls](cli-reference.md#report).

## Model bundle

Expression `full_run` exports:

| File | Purpose |
| --- | --- |
| `bundle_manifest.json` | Format, file inventory/checksums, threshold policy, exporting-build provenance |
| `feature_schema.tsv` | Ordered union of model-consumed features |
| `transform_feature_schema.tsv` | Complete raw schema required before rank transforms |
| `preprocess_state.joblib` | Model-local selected features, transformations, scalers, missing/abstention policies |
| `model_state.joblib` | Fitted models |
| `thresholds.tsv`, `resolved_config.yml` | Saved decision contract and training settings |
| `reference_expression.parquet`, `orthogroup_annotations.parquet` | Optional portable interpretation snapshots |
| `observed_traits.parquet` | Optional known-trait references for phylogenetic context |

Loading validates format, required files, sizes/checksums, and schema/state
consistency. Current expression exports use format `3`; readers support `1` and
`2` with their original policies. Version-1 rank-transform bundles lack the full
transform schema and must be regenerated.

Expression references cover nonzero final-model features and known-trait
training/CV species, excluding external-test and unknown species. Older bundles
can reuse the source run's candidate reference TSV when accessible; unavailable
references are marked and prediction continues. No training data are refitted.
Sequence/combined bundles have [separate formats and alignment requirements](amino-acid.md#prediction).

When validation groups and folds have a one-to-one correspondence (including
LOGO), non-negative integer group IDs are reused as `fold_id`. For example,
held-out group `10` is fold `10`, including noncontiguous or zero-based IDs.
This applies to outer and inner CV in every feature mode. Other group labels or
multi-group validation folds retain sequential IDs and their group mapping.
`split_manifest.tsv` also records the original 1-based `fold_index` (null outside
CV). Execution and random seeds retain this index, so renaming IDs does not
change splits, sample selection or stochastic model initialization.
