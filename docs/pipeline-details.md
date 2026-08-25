# Pipeline Details

This page explains execution flow for `run`, `predict`, and `report`.

## High-level flow

`run`:

1. Resolve and validate config.
2. Build split manifest from metadata + expression coverage.
3. Run outer CV (training + validation predictions + metrics).
4. Record the fixed probability-threshold contract (`0.5`); it is not derived from OOF data.
5. If `full_run`, refit on train+validation and predict external/inference.
6. Write artifacts, figures, metadata, optional model bundle, and a monotonic
   timing trace.

`predict`:

1. Resolve config.
2. Load and verify bundle integrity.
3. Build the expression matrix, align raw features to the bundled transform schema, and apply
   bundled preprocessing.
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
- `split.group_col` is required for labeled species unless
  `split.test_holdout_col` marks the species as a test holdout.
- `split.test_holdout_col` is optional; when it is `null`, no external-test
  holdout column is read.
- `split.exclude_col` is optional; rows marked true are removed before pool
  assignment and expression coverage checks.

Pool assignment:

- exclude true -> removed from all pools
- trait present + test holdout true -> `external_test`
- trait present + test holdout false + split group present -> `training_validation`
- trait missing -> `discovery_inference`

`training_validation` is an internal pool label before fold expansion.
In `split/tables/split_manifest.tsv`, these species appear as `train` and `validation`.

Expression coverage checks:

- all non-excluded metadata species must exist in expression table.
- rows in expression with species not present in metadata are counted and reported in metadata.

Preflight before CV:

- the full `training_validation` pool must contain both labels.
- each outer fold's training side must contain both labels.
- a validation side may contain one label. In that case, two-class
  discrimination metrics are written as `NA`, while Brier score and log loss
  remain computable.
- `sampling.strategy: group_balanced` requires each `split.group_col` group to
  contain both labels.
- `preprocess.ranked_feature_filter.method: pair_aware` requires
  `data.contrast_pair_col`.
  Pair-aware scoring uses only contrast pairs in the training fold that contain
  both labels. The stage is skipped with a warning when fewer than
  `preprocess.ranked_feature_filter.min_contrast_pairs` valid pairs are available.

Outer CV splits:

- `logo` -> `LeaveOneGroupOut`
- `group_kfold` -> `GroupKFold(n_splits=outer_cv_n_splits)`
- `stratified_group_kfold` -> `StratifiedGroupKFold` with intact groups,
  approximate label stratification, reproducible shuffling, and
  `n_splits=outer_cv_n_splits`

### 3) Outer CV training/evaluation

Outer folds can execute in parallel (up to `runtime.n_jobs`) with per-fold CPU budgeting.

Before fold execution:

- scan and normalize the long TPM rows for all outer-CV train/validation
  species once into a temporary Parquet cache;
- build one shared species x feature matrix by mapping the cached long rows to
  integer coordinates and accumulating values into a zero-filled NumPy array;
- retain a small raw-expression table for only the top interpreted features so
  tree heatmaps do not rescan the full TPM input when species coverage matches;
- retain `preprocess.max_pivot_cells` as the dense-cell limit used to split
  oversized matrices into feature chunks.

For each outer fold:

1. Slice shared matrix rows into fold-local train/valid arrays by species.
2. Build one or more sampled training sets:
  - `all_samples`: single full set
  - `group_balanced`: deterministic per-group balanced subsets
  - sampled sets can execute in parallel within each fold budget
3. Candidate handling:
  - when both `selected_candidate_count` and `selected_candidate_percent` are null:
    - generate candidates and fit them directly (no inner CV ranking)
  - when selection is active (`selected_candidate_count` or `selected_candidate_percent` is set):
    - score candidates on inner CV
    - keep top-K by `selection_metric`
    - selection source depends on `model_selection.candidate_source_policy`:
      - `per_sample_set`: select independently per sampled set
      - `reuse_first_sample_set`: select once from sampled set `0` and reuse
    - selected candidates are deduplicated by hyperparameter set (no duplicate params in one sampled-set selection)
    - for `search_strategy=grid|random`, candidate scoring can run in parallel within each fold budget
    - during that parallel scoring, per-model `random_forest` threads are auto-limited so combined fold/candidate/model concurrency stays within `runtime.n_jobs`
    - NumPy/SciPy/scikit-learn native thread pools are also limited to the active runtime budget in these execution paths
    - `search_strategy=tpe` remains sequential
4. For each sampled training set, preprocess sampled-train/valid and fit selected models:
  - `preprocess.expression_transform` (`none`, `log1p`, `sample_rank`, or `sample_percentile_rank`)
  - optional sparse feature filter
  - optional low-variance filter
  - optional pair-aware filter (train-only group-contrast ranking)
  - optional correlation filter (pearson/spearman)
  - `preprocess.feature_scaling` (`none` or train-fitted standard scaling)
  - fit selected model(s), predict fold-valid probabilities
5. Aggregate model probabilities (`mean` or `median`).
6. Compute fold metrics.

After all folds:

- write macro/micro aggregate metrics.
- write the configured fixed prediction threshold.
- build interpretation tables (`feature_importance`, `coefficients`).
- when `evaluation.group_bootstrap.enabled=true`, pool the OOF predictions and
  resample the intact `split.group_col` groups to estimate percentile confidence
  intervals for ROC AUC, Average Precision, balanced accuracy, MCC, Brier score,
  and log loss. This is prediction-level resampling and does not refit models.

### 4) Final refit (`execution_stage=full_run`)

- Training pool is `train + validation` species.
- Candidate generation/selection is repeated in `final_refit` scope.
- sampled sets can run in parallel up to `runtime.n_jobs` budget.
- each sampled set is preprocessed independently before fit:
  - `preprocess.expression_transform` (`none`, `log1p`, `sample_rank`, or `sample_percentile_rank`)
  - optional sparse feature filter
  - optional low-variance filter
  - optional pair-aware filter (train-only group-contrast ranking)
  - optional correlation filter (pearson/spearman)
  - `preprocess.feature_scaling` (`none` or train-fitted standard scaling)
- within each sampled set, selected model fits can also run in parallel.
- `random_forest` thread count is auto-limited per task so combined concurrency stays within `runtime.n_jobs`.
- NumPy/SciPy/scikit-learn native thread pools are also limited to the active runtime budget.
- Fit selected model(s) and predict:
  - `external_test`
  - `discovery_inference`
- Produce `external_test/tables/prediction_external_test.tsv` and
  `inference/tables/prediction_inference.tsv`.
- Export `model_bundle/` with model-local preprocessing state.

### 5) Timing instrumentation

- `runtime/tables/timing.tsv` uses one `time.perf_counter` origin for the whole
  successful `run` command.
- Top-level rows cover config, provenance, split construction, outer CV,
  optional group bootstrap/final refit, artifact writing, and figures.
- Nested rows cover outer-CV matrix construction, individual folds,
  sample-set preprocessing, inner-CV preprocessing, selected-model
  fitting/prediction, and inner-CV candidate scoring. Final refit uses the same
  sample/candidate identifiers.
- Supported expression transforms are row-local. During model selection, each
  sampled source matrix is therefore transformed once before its inner-CV rows
  are sliced; train-fitted feature filtering and scaling still run separately
  inside every inner fold to prevent leakage.
- Sparse-feature filtering computes the nonzero mask once per inner fold and
  reuses it for trait-specific retention fractions.
- Orthogroup annotation loading is restricted to the union of top features
  that can appear in importance, coefficient, or tree-heatmap figures.
- Start/end offsets make concurrent intervals explicit. Nested or parallel
  durations are diagnostic measurements and are not expected to sum to the
  top-level wall-clock duration.
- `run_metadata.json.timing.stage_duration_sec` stores the `scope=run` summary;
  the TSV remains the detailed source of truth.

## Model selection behavior details

Candidate generation strategy:

- `grid`: deterministic full discrete Cartesian product.
- `random`: deterministic sampling by hashed seed and scope.
- `tpe`: deterministic Optuna `TPESampler(seed=...)`.

Search-space notes:

- empty `search_space` is valid and yields one empty candidate `{}`.
- for discrete-only search, `trial_count` is capped to candidate-space size with warning.

Inner-CV selection:

- enabled only when selection is active (`selected_candidate_count` or `selected_candidate_percent` is set).
- `inner_cv_strategy`: `logo`, `group_kfold`, or `stratified_group_kfold`.
- `selection_rule=best` ranks by mean inner-CV score.
- `selection_rule=one_se` first identifies candidates within one standard error
  of the best mean score, then prefers the simpler candidate.
- threshold-dependent candidate metrics (`mcc`/`balanced_accuracy`) use the fixed
  probability threshold `0.5`; `log_loss` is threshold-independent.

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

- for `sample_rank` and `sample_percentile_rank`, align raw input to the complete
  `transform_feature_schema.tsv` before applying the expression transform.
- raw transform features missing from the input are filled with `0`; extra input features are
  removed before rank calculation and therefore cannot change retained-feature ranks.
- for feature-wise `none` and `log1p`, alignment may be restricted to the model-feature union
  because feature selection and transformation commute.
- zero overlap with the model-feature union is an error.

Inference:

- apply per-model bundled preprocessing state
  (transformed feature subset selection + optional model-local scaler transform).
- run all bundled models.
- aggregate probs by bundled aggregation mode.
- derive `pred_label_fixed_threshold` by bundled fixed threshold.

## `report`: step-by-step

Run selection:

- explicit `--run-dir` list, or scan `--runs-root` with `--glob` and optional `--latest`.

Per-run ingestion:

- requires `run_metadata.json` and `resolved_config.yml`.
- for non-predict runs, missing `cv/tables/metrics_cv.tsv` is an error in `--strict`.
- in non-strict mode, missing/invalid `cv/tables/metrics_cv.tsv` records warnings and can leave runs included but unranked.
- applies stage filter (`--include-stage`).
- verifies that metric-bearing runs share one experiment fingerprint.
- reads each run's persisted metric contract rather than inferring its implementation from
  the currently installed version.
- reads each run's persisted PhenoRadar version/build provenance and warns about missing
  legacy versions, dirty source checkouts, or reports that mix PhenoRadar versions.
- legacy runs without fingerprints or metric contracts are warned in non-strict mode and
  rejected in `--strict`.
- software-version warnings are diagnostic: experiment fingerprints and metric contracts
  remain the comparison guard, so a version mismatch is surfaced without silently changing
  ranking eligibility.

Ranking:

- rank by selected `primary_metric` and `aggregate_scope`.
- reject mixed known/unknown or different experiment fingerprints unless
  `--allow-mixed-experiments` is specified.
- tie-break by `start_time`, then `run_id`.

Error policy:

- `--strict`: fail fast on invalid/missing artifacts.
- default non-strict: warn and continue (skip some invalid runs; keep some runs as unranked when metrics are missing/invalid).

## Where to go next

- For config keys and constraints: [configuration.md](configuration.md)
- For output file schemas and interpretation guidance: [output-artifacts.md](output-artifacts.md)
