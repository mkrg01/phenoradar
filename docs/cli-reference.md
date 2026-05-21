# CLI Reference

Both commands are equivalent:

- `phenoradar`
- `phrad`

## Global command tree

```text
phenoradar run
phenoradar config
phenoradar metadata
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
- `fold_validation_groups.tsv`
- `metrics_cv.tsv`
- `loss_by_split_cv.tsv`
- `thresholds.tsv`
- `feature_importance.tsv`
- `feature_importance_by_fold.tsv`
- `coefficients.tsv`
- `coefficients_by_fold.tsv`
- `prediction_cv.tsv`
- `feature_filter_counts.tsv`
- `feature_filter_counts_summary.tsv`
- `retained_features.tsv`
- `retained_features_summary.tsv`
- `model_sparsity.tsv`
- `model_sparsity_summary.tsv`
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
  - `feature_importance_by_fold_heatmap.svg`
  - `coefficients_signed_top.svg`
  - `cv_species_probability_by_trait.svg`
  - `cv_fold_trait_probability.svg`
  - `feature_filter_funnel.svg`
  - `selected_features_by_fold_after_preprocessing.svg`
  - `non_zero_feature_count_by_fold.svg`
  - `model_selection_trials.svg` (model selection enabled)
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

## `metadata`

Generate metadata-adjacent artifacts from a raw species trait table.

```bash
phenoradar metadata \
  --species-trait species_trait.tsv \
  --tree-out ncbi_tree.nwk \
  --out species_metadata.tsv
```

This command uses the external `nwkit` executable. Taxonomic-rank blocking also uses
`ete4.NCBITaxa`, which is included in standard PhenoRadar installs. For a local uv
environment, install the recorded `nwkit` dependency group with:

```bash
uv sync --group taxonomy
```

For conda-based environments, install `nwkit` from Bioconda. For pip-only environments,
install `nwkit` directly from the upstream repository. Make sure the executable is on
`PATH` or pass `--nwkit-bin`.
Tree retrieval uses the default `nwkit constrain` taxonomy depth; taxonomic-rank split
blocks are controlled separately with `--taxon-block-rank`.

Options:

- `--species-trait`: input TSV containing species and binary trait columns (default: `species_trait.tsv`)
- `--species-taxid`: optional TSV containing species and NCBI taxid columns for tree retrieval and taxonomic-rank blocking
- `--species-taxid-out`: output generated species/taxid TSV when `--species-taxid` is omitted; defaults to `species_taxid.tsv` next to `--out` when taxonomic-rank blocking needs it
- `--out`: output PhenoRadar metadata TSV (default: `species_metadata.tsv`)
- `--tree-in`: existing Newick tree to use for group assignment; skips NCBI tree retrieval
- `--tree-out`: output Newick tree path when retrieving from NCBI Taxonomy (default: `ncbi_tree.nwk`)
- `--species-col`: species column name in `species_trait.tsv` and `species_taxid.tsv` (default: `species`)
- `--taxid-col`: taxid column name in `species_taxid.tsv` (default: `taxid`)
- `--trait-col`: binary trait column name in `species_trait.tsv` and output metadata (default: `C4`)
- `--contrast-pair-col`: output contrast-pair column name (default: `contrast_pair_id`)
- `--contrast-pair-test-holdout-col`: output column marking known-trait species without a contrast pair as test holdouts (default: `contrast_pair_test_holdout`)
- `--taxon-block-rank`: NCBI taxonomy rank to emit as a split block; repeat for multiple ranks such as `family` and `order`
- `--taxon-block-min-species-per-label`: minimum labeled species per trait value required for a taxon block to enter CV (default: `1`)
- `--taxon-block-mixed-test-fraction`: fraction of mixed-label taxon blocks to reserve as external test blocks (default: `0.0`)
- `--taxon-block-mixed-test-seed`: random seed for selecting mixed-label test blocks (default: `42`)
- `--ncbi-taxonomy-db`: optional ete4 NCBI taxonomy SQLite database path
- `--nwkit-bin`: `nwkit` executable path (default: `nwkit`)
- `--force`: overwrite existing tree or metadata outputs
- `--tree-only`: fetch/write only the tree and skip metadata generation
- `--verbose`, `-v`: detailed stage-level logs
- `--quiet`, `-q`: suppress progress logs

Group assignment uses `nwkit skim --only-contrastive-clades yes --output-groupfile yes`.
Species that are not present in the tree are excluded from the generated metadata. The
generated `contrast_pair_id` is based on `contrastive_clade`, so each assigned training
contrast pair contains both non-missing trait labels (`0` and `1`). Known-trait species
without an assigned contrast pair are marked in `contrast_pair_test_holdout`.

When `--taxon-block-rank` is supplied, the command also writes
`taxon_<rank>_id`, `taxon_<rank>_name`, `taxon_<rank>_test_holdout`, and
`taxon_<rank>_exclude`. If `--species-taxid` is omitted, `phenoradar metadata`
first resolves species names with `ete4.NCBITaxa`, writes a generated
`species_taxid.tsv` (or `--species-taxid-out`), and reuses that file for rank
blocking. Rank blocks with both labels are usable as CV groups. Single-label
rank blocks are marked as test holdout, while labeled species with missing
taxid/rank are marked as excluded.

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
- `--base-url`: alternate source URL containing the c4_tiny dataset files
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
