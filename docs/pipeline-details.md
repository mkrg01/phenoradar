# Pipeline Details

This page explains execution flow for `run`, `predict`, and `report`.

## High-level flow

`run`:

1. Resolve and validate config.
2. Build split manifest from metadata + expression coverage.
3. Run outer CV (training + validation predictions + metrics).
4. Derive CV threshold from OOF predictions.
5. If `full_run`, refit on train+validation and predict external/inference.
6. Write artifacts, figures, metadata, and optional model bundle.

`predict`:

1. Resolve config.
2. Load and verify bundle integrity.
3. Build expression matrix and align features to bundle schema.
4. Run deterministic inference using bundled preprocessing/model state.
5. Write artifacts, figures, metadata.

`report`:

1. Select run directories (`--run-dir` or scan `--runs-root`).
2. Validate required artifacts.
3. Load metrics and compute ranking with selected metric/scope.
4. Write report tables, manifest, optional narrative, and figures.

## `run`: step-by-step

### 1) Config composition and validation

- Optional single `-c` config overrides built-in defaults.
- Unknown keys are rejected.
- Cross-field rules (sampling/model-selection/filter constraints) are enforced.

### 2) Split construction

Metadata normalization:

- species IDs are trimmed and must be non-empty and unique.
- trait must be `0/1` or null/empty.
- group can be null/empty.

Pool assignment:

- trait present + group present -> `training_validation`
- trait present + group missing -> `external_test`
- trait missing -> `discovery_inference`

`training_validation` is an internal pool label before fold expansion.
In `split_manifest.tsv`, these species appear as `train` and `validation`.

Expression coverage checks:

- all metadata species must exist in expression table.
- rows in expression with species not present in metadata are counted and reported in metadata.

Preflight before CV:

- each training group must contain both labels (applies to all sampling modes, including `all_samples`).

Outer CV splits:

- `logo` -> `LeaveOneGroupOut`
- `group_kfold` -> `GroupKFold(n_splits=outer_cv_n_splits)`

### 3) Outer CV training/evaluation

Outer folds can execute in parallel (up to `runtime.n_jobs`) with per-fold CPU budgeting.

Before fold execution:

- build one shared species x feature matrix from long TPM table for all outer-CV train/validation species.

For each outer fold:

1. Slice shared matrix rows into fold-local train/valid arrays by species.
2. Build one or more sampled training sets:
  - `all_samples`: single full set
  - `group_balanced`: deterministic per-group balanced subsets
  - sampled sets can execute in parallel within each fold budget
3. Candidate handling:
  - when `selected_candidate_count` is null:
    - generate candidates and fit them directly (no inner CV ranking)
  - when `selected_candidate_count` is set:
    - score candidates on inner CV
    - keep top-K by `selection_metric`
    - for `search_strategy=grid|random`, candidate scoring can run in parallel within each fold budget
    - during that parallel scoring, per-model `random_forest` threads are auto-limited so combined fold/candidate/model concurrency stays within `runtime.n_jobs`
    - NumPy/SciPy/scikit-learn native thread pools are also limited to the active runtime budget in these execution paths
    - `search_strategy=tpe` remains sequential
4. For each sampled training set, preprocess sampled-train/valid and fit selected models:
  - `log1p`
  - optional low-prevalence filter
  - optional low-variance filter
  - optional correlation filter (pearson/spearman)
  - standard scaling (fit on sampled train, transform valid)
  - fit selected model(s), predict fold-valid probabilities
5. Aggregate model probabilities (`mean` or `median`).
6. Compute fold metrics.

After all folds:

- write macro/micro aggregate metrics.
- derive `cv_derived_threshold` from OOF predictions by scanning candidate thresholds.
- build interpretation tables (`feature_importance`, `coefficients`).

### 4) Final refit (`execution_stage=full_run`)

- Training pool is `train + validation` species.
- Candidate generation/selection is repeated in `final_refit` scope.
- sampled sets can run in parallel up to `runtime.n_jobs` budget.
- each sampled set is preprocessed independently before fit:
  - `log1p`
  - optional low-prevalence filter
  - optional low-variance filter
  - optional correlation filter (pearson/spearman)
  - standard scaling (fit on sampled train, transform target pools)
- within each sampled set, selected model fits can also run in parallel.
- `random_forest` thread count is auto-limited per task so combined concurrency stays within `runtime.n_jobs`.
- NumPy/SciPy/scikit-learn native thread pools are also limited to the active runtime budget.
- Fit selected model(s) and predict:
  - `external_test`
  - `discovery_inference`
- Produce `prediction_external_test.tsv` and `prediction_inference.tsv`.
- Export `model_bundle/` with model-local preprocessing state.

## Model selection behavior details

Candidate generation strategy:

- `grid`: deterministic full discrete Cartesian product.
- `random`: deterministic sampling by hashed seed and scope.
- `tpe`: deterministic Optuna `TPESampler(seed=...)`.

Search-space notes:

- empty `search_space` is valid and yields one empty candidate `{}`.
- for discrete-only search, `trial_count` is capped to candidate-space size with warning.

Inner-CV selection:

- enabled only when `selected_candidate_count` is set.
- `inner_cv_strategy`: `logo` or `group_kfold`.
- threshold-dependent candidate metrics (`mcc`/`balanced_accuracy`) use
  `report.fixed_probability_threshold`; `log_loss` is threshold-independent.

## Reproducibility and seeding

- `runtime.seed` controls global deterministic behavior.
- model-selection candidate generation and TPE selection also use `runtime.seed`.
- model/sample-set/fold scoped seeds are derived via deterministic hash formulas.
- selection/trial ordering is deterministic for fixed input data and config.

## `predict`: step-by-step

Bundle verification:

- verify format version.
- verify required files.
- verify file inventory checksums/sizes.
- verify feature schema and preprocessing/model state consistency.

Feature alignment:

- align input features to bundle feature schema.
- missing bundle features are filled with `0`.
- extra input features are ignored.
- zero overlap is an error.

Inference:

- apply `log1p`, then per-model bundled preprocessing state
  (feature subset alignment + model-local scaler transform).
- run all bundled models.
- aggregate probs by bundled aggregation mode.
- derive `pred_label_fixed_threshold` by bundled fixed threshold.
- derive `pred_label_cv_derived_threshold` by bundled CV-derived threshold.

## `report`: step-by-step

Run selection:

- explicit `--run-dir` list, or scan `--runs-root` with `--glob` and optional `--latest`.

Per-run ingestion:

- requires `run_metadata.json` and `resolved_config.yml`.
- for non-predict runs, missing `metrics_cv.tsv` is an error in `--strict`.
- in non-strict mode, missing/invalid `metrics_cv.tsv` records warnings and can leave runs included but unranked.
- applies stage filter (`--include-stage`).

Ranking:

- rank by selected `primary_metric` and `aggregate_scope`.
- tie-break by `start_time`, then `run_id`.

Error policy:

- `--strict`: fail fast on invalid/missing artifacts.
- default non-strict: warn and continue (skip some invalid runs; keep some runs as unranked when metrics are missing/invalid).

## Where to go next

- For config keys and constraints: [configuration.md](configuration.md)
- For output file schemas and interpretation guidance: [output-artifacts.md](output-artifacts.md)
