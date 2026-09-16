# CLI Reference

Both commands are equivalent:

- `phenoradar`
- `phrad`

## Global command tree

```text
phenoradar run
phenoradar config
phenoradar dataset
phenoradar predict
phenoradar report
```

Input preparation belongs in the separate `phenoradar_prep` repository.
See [data-format.md](data-format.md) for the input files accepted by PhenoRadar.

Global options:

- `--help`, `-h`: show help
- `--version`, `-V`: show installed `phenoradar` version and exit

## Logging behavior

- default: concise progress logs (timestamps + key milestones)
- `--verbose`, `-v`: detailed stage-level progress logs
- `--quiet`, `-q`: suppress progress logs and show only final summaries/warnings
- `--verbose` and `--quiet` cannot be used together
- warnings are printed at command end and also persisted:
  - `run` / `predict`: `run_metadata.json` `warnings`
  - `report`: `report_warnings.tsv`
- `report` warning summaries are shown as `warning_type` aggregates (rows/runs)

Related detail docs:

- [output-artifacts.md](output-artifacts.md) for file-level schemas, generation conditions, and interpretation guidance.
- [pipeline-details.md](pipeline-details.md) for step-by-step runtime behavior.

## `run`

Run training/evaluation pipeline.

```bash
phenoradar run -c config.yml [--execution-stage cv_only|full_run]
phenoradar run -c config.yml --resume runs/<study_id>
```

Options:

- `--config`, `-c` (required): YAML config path (exactly once).
- `--execution-stage`: temporary override of `runtime.execution_stage`
- `--verbose`, `-v`: detailed stage-level logs
- `--quiet`, `-q`: suppress progress logs
- `--resume`: resume a multi-condition study directory generated from the same
  ordered conditions

When a schema-scalar config field contains a list, `run` expands the values into
ordered conditions, reuses one outer split for every condition, and writes a
study directory. All conditions are reported symmetrically; there is no default
or reference condition.

Always written:

- `resolved_config.yml`
- `split/tables/split_manifest.tsv`
- `split/tables/fold_validation_groups.tsv`
- `split/tables/fold_diagnostics.tsv`
- `cv/tables/metrics_cv.tsv`
- `cv/tables/loss_by_split_cv.tsv`
- `cv/tables/feature_importance.tsv`
- `cv/tables/feature_importance_by_fold.tsv`
- `cv/tables/coefficients.tsv`
- `cv/tables/coefficients_by_fold.tsv`
- `cv/tables/feature_stability_by_feature.tsv`
- `cv/tables/feature_stability_by_fold_pair.tsv`
- `cv/tables/feature_stability_summary.tsv`
- `cv/tables/prediction_cv.tsv`
- `model/tables/thresholds.tsv`
- `model/tables/evaluation_contract.tsv`
- `model/tables/feature_filter_counts.tsv`
- `model/tables/feature_filter_counts_summary.tsv`
- `model/tables/retained_features.tsv`
- `model/tables/retained_features_summary.tsv`
- `model/tables/model_sparsity.tsv`
- `model/tables/model_sparsity_summary.tsv`
- `model/tables/convergence_diagnostics.tsv`
- `summary/tables/classification_summary.tsv`
- `runtime/tables/timing.tsv`
- `run_metadata.json`
- stage-specific figure directories (`cv/figures/`, `model/figures/`,
  `external_test/figures/`, `inference/figures/`)

Notes:

- `prediction_cv.tsv` may include optional `uncertainty_std` when ensemble size > 1.
- Timing rows use one monotonic clock. Parallel fold/sample/candidate intervals
  can overlap and should not be summed as elapsed wall time.
- Stage figure directories include:
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
  - `cv/figures/probability_by_<group>.svg` (when `summary.group_col` is present in metadata)
  - `cv/figures/model_selection_trials.svg` (model selection enabled)
  - `cv/figures/roc_curve_cv.svg` (may be skipped when degenerate)
  - `cv/figures/pr_curve_cv.svg` (may be skipped when degenerate)
  - `model/figures/final_refit_feature_importance_top.svg` (`full_run`)
  - `model/figures/final_refit_coefficients_signed_top.svg` (`full_run`, linear model)
  - `model/figures/final_refit_feature_filter_funnel.svg` (`full_run`)
  - `model/figures/final_refit_model_selection_trials.svg` (model selection enabled)
  - `model/figures/final_refit_model_selection_one_se_curve.svg` (model selection enabled)
  - `external_test/figures/final_refit_loss_by_split.svg` (`full_run`)
  - `external_test/figures/top_feature_expression_by_confusion.svg` (`full_run` when external test rows exist)
  - `external_test/figures/external_species_probability_by_trait.svg` (`full_run` when external test rows exist)
  - `external_test/figures/external_confusion_matrix.svg` (`full_run` when external test rows exist)
  - `external_test/figures/cv_external_metric_comparison.svg` (`full_run` when external test rows exist)
  - `external_test/figures/external_roc_curve.svg` / `external_test/figures/external_pr_curve.svg` (`full_run` when external test rows contain both labels)
  - `inference/figures/inference_probability_distribution.svg` (`full_run` when inference rows exist)
  - `inference/figures/species_probability_cv_and_inference.svg` (`full_run` when inference rows exist)
  - `<stage>/figures/probability_by_<group>.svg` (`full_run` prediction stages when `summary.group_col` is present in metadata)

Conditionally written:

- `cv/tables/group_bootstrap_metrics.tsv`,
  `cv/tables/group_bootstrap_replicates.tsv`, and
  `cv/figures/group_bootstrap_metrics.svg`
  (`evaluation.group_bootstrap.enabled=true`)
- `external_test/tables/prediction_external_test.tsv` (`full_run` only)
- `inference/tables/prediction_inference.tsv` (`full_run` only)
- `external_test/tables/loss_by_split_final_refit.tsv` (`full_run` only)
- `model_bundle/` (`full_run` only)
- `cv/tables/ensemble_model_probs.tsv` (ensemble size > 1)
- `cv/tables/model_selection_trials.tsv` (model selection enabled)
- `cv/tables/model_selection_trials_summary.tsv` (model selection enabled)
- `model/tables/model_selection_selected.tsv` (candidate selection enabled)
- `model/tables/final_refit_feature_importance.tsv` /
  `model/tables/final_refit_feature_importance_by_model.tsv` (`full_run`)
- `model/tables/final_refit_coefficients.tsv` /
  `model/tables/final_refit_coefficients_by_model.tsv` (`full_run`)
- `model/tables/final_refit_model_selection_trials.tsv` /
  `model/tables/final_refit_model_selection_trials_summary.tsv`
  (candidate selection enabled in `full_run`)
- `cv/figures/model_selection_one_se_curve.svg` (candidate selection enabled)

## `config`

Resolve and validate config without running the pipeline. The output includes
every user-facing setting with defaults filled in, plus comments listing
available choices and nullable value types. Internal repeat indices are omitted;
set `sampling.group_subsample_repeats` to control repeated group subsampling.

```bash
phenoradar config [-c config.yml] [--out config.yml]
```

Options:

- `--config`, `-c` (optional): YAML config path (at most once; omitted means built-in defaults only)
- `--out` (optional): output YAML path (default: `config.yml`)
- `--verbose`, `-v`: detailed stage-level logs
- `--quiet`, `-q`: suppress progress logs

## `predict`

Predict from an exported bundle (no retraining).

```bash
phenoradar predict --model-bundle runs/<run_id>/model_bundle -c predict_config.yml
```

Options:

- `--model-bundle` (required): bundle directory
- `--config`, `-c` (required): prediction config (exactly once)
- `--verbose`, `-v`: detailed stage-level logs
- `--quiet`, `-q`: suppress progress logs

Outputs:

- `resolved_config.yml`
- `inference/tables/prediction_inference.tsv`
- `run_metadata.json`
- `inference/figures/`
  - `predict_probability_distribution.svg`
  - optional `predict_uncertainty.svg` (bundle ensemble size > 1)

Prediction-time feature alignment policy:

- rank transforms align raw input to the complete bundled transform schema before ranking
- for rank transforms, transform-schema features missing in input -> filled
  with the bundle's training-time `preprocess.absent_feature_fill` policy before ranking
- for rank transforms, extra input features outside the transform schema -> ignored before ranking
- feature-wise transforms may align directly to the model-feature union; missing
  model features use the same bundled fill policy (`0` or `NA`)
- missing bundled features produce a warning, but prediction continues when at
  least one model feature overlaps
- zero overlap with the model-feature union -> error

## `dataset`

Install compact test data bundled with PhenoRadar.

```bash
phenoradar dataset [--out testdata/c4_tiny] [--base-url URL] [--force]
```

Options:

- `--out`: output directory (default: `testdata/c4_tiny`)
- `--base-url`: optional external source URL containing the c4_tiny dataset files;
  `PHENORADAR_TESTDATA_BASE_URL` provides the same override
- `--force`: overwrite existing files if checksum does not match expected values
- `--verbose`, `-v`: detailed stage-level logs
- `--quiet`, `-q`: suppress progress logs

Without an external source override, this command copies package resources and
does not access the network. All files are checked against the bundled
`SHA256SUMS` manifest.

## `report`

Aggregate multiple run directories into comparison artifacts.

```bash
phenoradar report --runs-root runs [options]
```

Selection options:

- `--run-dir` (repeatable): explicit run directories
- `--runs-root`: scan root directory for runs
- `--glob` (default: `*`)
- `--latest N`: keep only latest N directories after glob

Ranking options:

- `--primary-metric`: `mcc|balanced_accuracy|roc_auc|pr_auc|brier`
  - `pr_auc` is the compatibility key for Average Precision computed by
    `sklearn.metrics.average_precision_score`
  - ranking is descending for `mcc`, `balanced_accuracy`, `roc_auc`, and `pr_auc`, and
    ascending for `brier`
- `--aggregate-scope`: `macro|micro`
- `--include-stage`: `cv_only|full_run|predict|all`
- `--strict`: fail instead of non-strict warn-and-continue behavior
- `--allow-mixed-experiments`: explicitly allow ranking runs whose experiment
  fingerprints differ or cannot be verified
- `--output-format`: `tsv|md|html|json`
- `--out`: output directory (default auto-generated under `reports/`)
- `--verbose`, `-v`: detailed stage-level logs
- `--quiet`, `-q`: suppress progress logs

Outputs:

- `report_manifest.json`
- `report_runs.tsv`
- `report_ranking.tsv`
- `report_warnings.tsv`
- optional narrative file (`report.md`, `report.html`, or `report.json`)
- `figures/` (`report_metric_ranking.svg`, `report_metric_comparison.svg`,
  and `report_stage_breakdown.svg` when more than one stage appears)

By default, ranked runs must share one experiment fingerprint. The fingerprint is based on
the metadata/TPM file contents, the realized split manifest, and the evaluation contract;
file paths and model hyperparameters are excluded. Legacy runs without fingerprints remain
reportable in non-strict mode with warnings, while `--strict` rejects them. Report metric
definitions are read from each run's persisted contract, so missing legacy definitions are
left unknown rather than inferred from the installed version.
The report also carries each run's persisted PhenoRadar version/build fields and emits
`mixed_phenoradar_versions`, `missing_phenoradar_version`, or `dirty_phenoradar_build`
warnings when applicable. The report-generating build is recorded separately under
`report_manifest.json.generated_by`.
