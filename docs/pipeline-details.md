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
- the `split.group_col` column is required. For contrast-pair splits (the
  column matches non-null `data.contrast_pair_col`), labeled species with
  missing group values automatically enter `external_test`.
- for other split groups, labeled, non-excluded species with missing group
  values cause an error; missing contrast pairs do not affect pool assignment.
- `split.exclude_col` is optional; rows marked true are removed before pool
  assignment and expression coverage checks.

Pool assignment:

- exclude true -> removed from all pools
- trait missing -> `discovery_inference`
- trait present + contrast-pair split + pair missing -> `external_test`
- trait present + split group present -> `training_validation`

When `split.require_both_labels_per_group=true`, inspect the provisional
`training_validation` pool by `split.group_col` before constructing folds.
Only groups with both `0` and `1` remain eligible; labeled species in
single-label groups move to `external_test`. Excluded species, unpaired
external-test species, and trait-missing species do not contribute to that
check. The default `false` preserves the pool assignment above.

This filters CV eligibility before training: moved species are absent from
all CV training and validation sets and from final-refit training.

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

- scan and normalize the long TPM rows for the required species once into a
  temporary Parquet cache;
- build one shared species x feature matrix by mapping the cached long rows to
  integer coordinates; absent coordinates are initialized according to
  `preprocess.absent_feature_fill` (`0` by default, optionally `nan` for
  random forest or neutral logistic regression);
- retain a small raw-expression table for only the top interpreted features so
  tree heatmaps do not rescan the full TPM input when species coverage matches;
- retain `preprocess.max_pivot_cells` as the dense-cell limit used to split
  oversized matrices into feature chunks.

The CLI keeps one normalized-expression cache through outer CV and final refit.
For `full_run`, its species set includes train/validation, discovery inference,
and external test, so refit reuses the validated raw rows without parsing and
aggregating the TSV again. CV feature names still come only from train/validation
species; refit preserves its existing feature schema and target-matrix pruning
rules. Feature selection, transforms, scaling, and model fitting retain their
existing training scopes. This sharing does not retain CV's dense matrices for
refit. For `cv_only`, the cache contains only train/validation species.

Cache construction expresses missing-feature normalization and invalid-row
diagnostics as row-wise operations followed by scalar sum/min/max aggregates.
This lets the Parquet sink stream these operations without first materializing
all normalized input rows. Duplicate species/feature coordinates are still
summed, and validation retains invalid-row counts, source-line examples, and
checks for overflow after summation. The aggregation still needs state for the
distinct coordinates, so its memory usage can grow with the input size. This
optimization applies even when every feature is retained for modeling.

The cache is local to one run (including one condition in a study), and is
removed after these stages or on an exception. It is not a persistent cache
between runs. Because `full_run` prepares the complete raw species set before
CV, invalid external/inference input can now fail at that preparation step.
No such extra validation is performed for `cv_only`. Python callers can opt
into the same lifecycle with a `RunExpressionCache` context passed via
`expression_cache` to `run_outer_cv` and `run_final_refit`.

For `none` and `log1p` transforms, outer CV initially builds only the raw
train/validation matrix. After all folds finish fitting and validation, it
builds one inference matrix from the union of their retained features, in the
original feature order. Each sampled set applies its own selected columns,
transform, and fitted scaler, then predicts with its fitted models in the
original ensemble order. Abstention and inference-species figure data reuse
this narrow raw matrix. Model selection and training never use inference rows.
Only fitted models, scalers, and feature schemas are retained for this second
phase; sample-set training arrays are not retained for inference.

`sample_rank` and `sample_percentile_rank` keep the existing full-row inference
path because their results depend on features outside the retained union.
When most features survive filtering, the union may still be wide. The raw
input cache also continues to validate all consumed expression values.

For each outer fold:

1. Slice shared matrix rows into fold-local train/valid arrays by species.
2. Rank the fold-local `split.group_col` training groups reproducibly using
   `runtime.seed` and the internal index generated from
   `sampling.group_subsample_repeats`, then retain exactly
   `sampling.training_group_count` groups. Validation rows remain unchanged.
   With a fixed repeat index, increasing counts produce nested training-group
   subsets. A numeric count that is unavailable in any fold is an error; `null`
   retains every available training group.
3. Build one or more sampled training sets from the retained groups:
  - `all_samples`: single full set
  - `group_balanced`: deterministic per-group balanced subsets
  - sampled sets can execute in parallel within each fold budget
4. Candidate handling:
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
5. For each sampled training set, preprocess sampled-train/valid and fit selected models:
  - `preprocess.expression_transform` (`none`, `log1p`, `sample_rank`, or `sample_percentile_rank`)
  - optional sparse feature filter; `scope: all_samples` uses the nonzero
    fraction across all sampled training species, `any_trait` uses the best
    fraction across trait classes, and `trait_0` / `trait_1` uses only that class;
    zero and missing values stay in the denominator, with no validation or
    prediction species contributing to these fractions
  - optional low-variance filter
  - optional ranked feature filter; supervised methods can require the
    train-fold effect `trait 1 - trait 0` to have a configured direction
  - optional correlation filter (pearson/spearman)
  - `preprocess.feature_scaling` (`none` or train-fitted standard scaling)
  - fit selected model(s), predict fold-valid probabilities
6. Aggregate model probabilities (`mean` or `median`).
7. Compute fold metrics.

Feature filtering screens sparse columns immediately after the complete expression
transform, before computing neutral missing-expression eligibility statistics.
The sparsity fractions use bounded column blocks; only surviving columns need
neutral variance, label-observation, and contrast-pair calculations. Neutral
eligibility statistics also use bounded column blocks to limit temporary memory
when most features survive sparsity. Rank and percentile-rank transforms still
see all features before screening. The recorded
`n_features_after_sparse_feature_filter` includes both neutral eligibility and
sparsity, preserving its existing meaning and the ranking candidate population.
Pair-aware and unpaired statistics gather only the required rows and columns,
preserving the prior reduction layout. Inner CV computes the same rankings and
warnings but omits unused per-feature diagnostic rows; outer and final-refit
diagnostic tables remain available.

Inner CV shares the complete transformed source matrix across its splits and
passes integer row indices to feature filtering. Sparsity and neutral statistics
read bounded column blocks from training rows; supervised and correlation filters
also use only those rows. Full-width training and validation copies are avoided.
Once feature selection is complete, only the selected rows and columns are
gathered for scaling. Scaling is fitted on training rows, and the resulting
per-fold matrices remain cached for candidate evaluation. Row order and reduction
layout are preserved, including when the source matrix uses Fortran layout.

After all folds:

- write macro/micro aggregate metrics.
- write the configured fixed prediction threshold.
- build interpretation tables (`feature_importance`, `coefficients`).
- for linear models, build one local-evidence record for every fixed-threshold OOF
  misclassification from only the models used in its held-out fold; expression
  references use the unique species sampled for that fold's training.
- when `evaluation.group_bootstrap.enabled=true`, pool the OOF predictions and
  resample the intact `split.group_col` groups to estimate percentile confidence
  intervals for ROC AUC, Average Precision, balanced accuracy, MCC, Brier score,
  and log loss. This is prediction-level resampling and does not refit models.

### 4) Final refit (`execution_stage=full_run`)

- Training pool is `train + validation` species, after any
  `split.require_both_labels_per_group` filtering. Automatically held-out
  single-label groups remain in `external_test`.
- `sampling.training_group_count` is applied again to this full refit pool using
  the same reproducible group ranking. Thus a numeric count also determines the
  exact number of groups used by the deployed refit model; use `cv_only` when
  the setting is only being explored.
- Candidate generation/selection is repeated in `final_refit` scope.
- sampled sets can run in parallel up to `runtime.n_jobs` budget.
- each sampled set is preprocessed independently before fit:
  - `preprocess.expression_transform` (`none`, `log1p`, `sample_rank`, or `sample_percentile_rank`)
  - optional sparse feature filter, optionally restricted to one trait class
  - optional low-variance filter
  - optional ranked feature filter, including pair-aware or unpaired
    directional-effect eligibility
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
- Full runs record shared input preparation as the independent run stage
  `expression_preparation`, before the outer-CV timer. Its `expression_input`
  rows include `input_normalize_cache` (TSV scan, normalization, aggregation,
  and Parquet write) and `input_validation_read`.
- Builders also record `input_schema_read`, `input_matrix_read`, and
  `input_dense_assembly` under the current stage's scope. The latter includes
  frame validation, coordinate mapping, allocation, filling, and chunk copying.
  Chunked matrices emit multiple rows; sum the same scope/stage for their cost.
- Deferred inference records outer-CV `inference_matrix_build` and
  `inference_execution`, and per-fold `deferred_inference` with preprocessing,
  prediction, and abstention details. Fold `total` and `fold_execution` cover
  fitting/validation in this path; the outer-CV total also includes deferred
  inference. Nested timings must not be added to their enclosing durations.
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

Logistic elastic net calls glmnet's native binomial coordinate-descent solver
through `python-glmnet`, with an unpenalized intercept and internal
standardization disabled. The `lambda` parameter penalizes the sample-weighted
mean log loss; larger `lambda` means stronger regularization. `alpha` is the L1
fraction. SVM and random forest retain their scikit-learn implementations.

Grid and random searches fit one descending lambda path for each inner fold and
combination of `alpha`, `thresh`, and `maxit`. The native solver reuses coefficients
along that path; coefficients are never shared between folds. Fold preprocessing
is reused across candidates. Independent folds and parameter paths are scheduled
within `runtime.n_jobs`. TPE uses independent fits for sequentially proposed
candidates. Candidate scores use the exact requested lambda, without interpolation
or additional internal cross-validation. The one-SE rule prefers larger `lambda`
and then larger `alpha` among eligible logistic candidates.

Selected logistic models in outer CV and final refit use the descending prefix
of that source sample set's candidate lambdas, ending at the exact selected
lambda. Only candidates with matching `alpha`, `thresh`, and `maxit` contribute
to the prefix. Evaluated candidates are available for grid, random, and TPE
selection; without inner CV, the configured ensemble candidates supply the path.
If no matching stronger lambda is available, the model uses a single-lambda fit.
Each refit starts a new native path on its own preprocessed training samples and
weights. Inner-fold coefficients and scalers are not reused. Intermediate models
are discarded, and candidate selection, ensemble ordering, and the convergence
threshold are preserved. Reported coordinate passes cover the entire refit path.
Changing the optimization path can produce small numerical differences in fitted
coefficients and predictions, particularly with correlated Lasso features.

The backend accepts dense and CSC sparse matrices, sample weights, and one-feature
folds. An all-constant training matrix is fitted with the exact weighted intercept
and zero coefficients. Native errors and incomplete paths stop training. Stored
models contain ordinary coefficients and intercepts, and prediction uses the
sigmoid directly. This does not change the current expression preprocessing
pipeline into a sparse sequence-feature pipeline.

Neutral logistic regression retains missing values throughout feature filtering
and fits standardization only on training observations. Its persisted scaler
replaces missing standardized values with zero immediately before the model.
The original raw matrix supplies the observation mask for information coverage.
Final refit, pruned-target prediction, outer-fold inference, and bundle prediction
all apply the same policy. Information coverage is computed from each prediction's
own fitted models; no coefficients from other CV folds are used.

When enabled, abstention applies the same configured fixed threshold in every
fold and in final prediction. It adds no learning or threshold-selection step.
All-species probability metrics remain available; separate abstention tables
report selective performance and the number of accepted/rejected species.

Abstention validates all supplied expression values in bounded blocks, including
columns that do not contribute to the model. It preserves per-model absolute
coefficient normalization and the averaging denominator for intercept-only
models. Coverage and missing-feature evidence then use only columns with positive
combined coefficient weight. Observation masks are computed directly from finite
values and the configured zero-as-missing policy in row blocks; no full-width
NaN-filled expression copy is needed. The existing threshold and inclusive
`1e-12` comparison tolerance are unchanged. Floating-point summation can differ
at the last few bits when zero-weight columns are omitted.

Timing records distinguish outer-fold `inference_preprocessing`,
`inference_prediction`, `validation_abstention`, and `inference_abstention`.
Final refit records `external_test_abstention` and `inference_abstention`.
These are nested measurements included in the existing enclosing stages, so
their durations must not be added to those enclosing stages.

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
- raw transform features missing from the input use the fill policy stored in
  the bundle (`0` by default or `NA` for a random-forest bundle trained with
  `preprocess.absent_feature_fill=nan`); extra input features are removed before
  rank calculation and therefore cannot change retained-feature ranks.
- for feature-wise `none` and `log1p`, alignment may be restricted to the model-feature union
  because feature selection and transformation commute.
- zero overlap with the model-feature union is an error.

Missing-value semantics:

- a bundle feature absent from the prediction input is filled according to the
  bundled training policy (`0` or `NA`) and reported as a warning; this applies
  to feature-wise transforms as well as rank transforms
- the long expression format maps an absent `(species, feature)` coordinate to
  the same bundled policy; default `0` mode cannot by itself distinguish an
  unmeasured value from a measured biological zero
- directional feature filtering reduces reliance on absence of trait-0-high
  features, but it does not change these input semantics or constrain final
  multivariable model coefficient signs

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
