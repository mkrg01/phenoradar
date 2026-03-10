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
```

Options:

- `--config`, `-c` (required): YAML config path (exactly once).
- `--execution-stage`: temporary override of `runtime.execution_stage`
- `--verbose`, `-v`: detailed stage-level logs
- `--quiet`, `-q`: suppress progress logs

Always written:

- `resolved_config.yml`
- `split_manifest.tsv`
- `metrics_cv.tsv`
- `loss_by_split_cv.tsv`
- `thresholds.tsv`
- `feature_importance.tsv`
- `coefficients.tsv`
- `prediction_cv.tsv`
- `classification_summary.tsv`
- `run_metadata.json`
- `figures/` (SVG files)

Notes:

- `prediction_cv.tsv` may include optional `uncertainty_std` when ensemble size > 1.
- `figures/` includes:
  - `cv_metrics_overview.svg`
  - `cv_loss_by_split.svg`
  - `threshold_selection_curve.svg`
  - `feature_importance_top.svg`
  - `coefficients_signed_top.svg`
  - `cv_species_probability_by_trait.svg`
  - `cv_fold_trait_probability.svg`
  - `roc_pr_curves_cv.svg` (may be skipped when degenerate)
  - `final_refit_loss_by_split.svg` (`full_run`)
  - `external_species_probability_by_trait.svg` (`full_run` when external test rows exist)

Conditionally written:

- `prediction_external_test.tsv` (`full_run` only)
- `prediction_inference.tsv` (`full_run` only)
- `loss_by_split_final_refit.tsv` (`full_run` only)
- `model_bundle/` (`full_run` only)
- `ensemble_model_probs.tsv` (ensemble size > 1)
- `model_selection_trials.tsv` (model selection enabled)
- `model_selection_trials_summary.tsv` (model selection enabled)
- `model_selection_selected.tsv` (candidate selection enabled)

## `config`

Resolve and validate config without running the pipeline.

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
- `prediction_inference.tsv`
- `run_metadata.json`
- `figures/`
  - `predict_probability_distribution.svg`
  - optional `predict_uncertainty.svg` (bundle ensemble size > 1)

Prediction-time feature alignment policy:

- bundle features missing in input -> filled with `0`
- extra input features not in bundle -> ignored
- zero overlap with bundle feature schema -> error

## `dataset`

Download compact test data from GitHub raw content.

```bash
phenoradar dataset [--out testdata/c4_tiny] [--base-url URL] [--force]
```

Options:

- `--out`: output directory (default: `testdata/c4_tiny`)
- `--base-url`: alternate source URL containing `species_metadata.tsv` and `tpm.tsv`
- `--force`: overwrite existing files if checksum does not match expected values
- `--verbose`, `-v`: detailed stage-level logs
- `--quiet`, `-q`: suppress progress logs

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
- `--aggregate-scope`: `macro|micro`
- `--include-stage`: `cv_only|full_run|predict|all`
- `--strict`: fail instead of non-strict warn-and-continue behavior
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
