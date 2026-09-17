# Configuration

PhenoRadar config files are YAML mappings validated by Pydantic.

This page is the canonical reference for config input behavior, defaults, and
key-by-key definitions.
For runtime execution flow tied to config keys, see
[pipeline-details.md](pipeline-details.md).

## Config input behavior

CLI config input is one YAML file:

- `run`: required (`-c config.yml`)
- `predict`: optional (`-c predict_config.yml`); see [prediction settings](#prediction-settings)
- `config`: optional (`-c config.yml`), omitted means built-in defaults only

Resolution and override rules:

- CLI accepts one `-c` file (`run` required, `predict`/`config` optional).
- Unspecified keys are filled by built-in defaults.
- Generated YAML explicitly includes every user-facing setting, including inactive fields,
  `null` values, and empty sections. Comments list enum and boolean choices;
  nullable fields also show `null` or their accepted types.
- Internal group-subsampling repeat indices are generated automatically and
  recorded in each run's `resolved_config.yml` for reproducibility.
- Unknown keys are rejected.
- `runtime.execution_stage` can be overridden from CLI
  (`phenoradar run -c config.yml --execution-stage ...`).

Use `config` to inspect resolved output:

```bash
phenoradar config [--out resolved.yml]
```

### Prediction settings

`predict` requires a model bundle and an explicit TPM path, supplied through
`--tpm-path` or `data.tpm_path`. It never falls back to the bundled example data.
`--tpm-path`, `--metadata-path`, and `--n-jobs` override corresponding config
values; otherwise prediction-specific defaults apply.

Only these settings are used and saved in prediction `resolved_config.yml`:

| Setting | Default / meaning |
| --- | --- |
| `data.tpm_path` | Required input expression TSV |
| `data.metadata_path` | `null`; all TPM species, or the supplied metadata's species subset |
| `data.species_col`, `data.feature_col`, `data.value_col` | `species`, `orthogroup`, `tpm` |
| `data.tree_path` | `null`; optional prediction tree |
| `data.orthogroup_annotation_path` | `null`; optional labels overriding bundled annotations |
| `figures.top_features` | Bundle's training-time value when `figures` is omitted; otherwise `30` unless specified |
| `data.trait_col`, `data.contrast_pair_col` | `C4`, `contrast_pair_id`; optional tree annotations |
| `preprocess.max_pivot_cells` | `50000000`; memory guard |
| `runtime.n_jobs` | `1`; positive prediction worker/thread count |
| `summary.group_col` | `family`; grouped summaries when metadata supplies this column |

A metadata TSV needs only the species column. Missing trait and contrast-pair
columns do not prevent prediction or tree output. Without metadata, no group
summary is attempted. Use metadata with a grouping column for grouped summaries.

Legacy run configs remain accepted. Recognized training sections and training-only
keys in the sections above are ignored, without running training validation;
for example, `model`, `split`, `sampling`, `model_selection`, and learned
preprocessing choices cannot alter predictions. Unknown prediction-setting keys
are rejected. Bundle state supplies all learned preprocessing and decision
policies. Its source and hashes are saved in `run_metadata.json`, together with
the effective worker count and hashes of the input files actually supplied.
Prediction is deterministic from the fitted bundle and does not use `runtime.seed`.

For `predict`, the launcher sets `POLARS_MAX_THREADS` from the effective worker
count before importing Polars, replacing any inherited value. The inference
calls also cap native BLAS/OpenMP threads and temporarily set the loaded
estimator's worker count when supported. This does not change saved bundle files.

The remaining sections describe the training/config-generation schema.

### Ordered multi-condition runs

For `phenoradar run`, a list assigned to a field that is scalar in the schema
creates one condition per value. No separate experiment section or reference
condition is used. For example:

```yaml
preprocess:
  ranked_feature_filter:
    method: [none, pair_aware, unpaired, variance]
    max_features: 100
```

Multiple scalar-list fields are expanded as a Cartesian product. Field order in
the YAML and value order within each list determine `condition_index`; study
tables and figures retain this order and are not sorted by value or performance.

Inactive scalar fields are canonicalized instead of producing duplicate
conditions. One such case is when
`preprocess.ranked_feature_filter.method` is `none`, `max_features` is inactive.
That condition is generated once with `max_features: null`, independently of a
`max_features` list. Other varying fields are still expanded normally. Thus,
`method: [none, pair_aware]` with seven `max_features` values generates eight
conditions rather than fourteen.

Training-group count sensitivity uses the same mechanism. The repeat count
creates independently ranked group subsets, and subsets with the same internal
repeat index are nested as the training-group count increases:

```yaml
sampling:
  training_group_count: [5, 10, 15, null]
  group_subsample_repeats: 3
```

This creates three repeat conditions for each numeric training-group count.
Here `null` means all fold-local training groups. Because repeating that
condition would select the same groups, the all-available run is generated once
rather than three times.

Lists that are already part of a field's schema remain ordinary single-run
values. In particular, this remains one inner model-selection search space:

```yaml
model_selection:
  search_space:
    alpha: [0.0001, 0.001, 0.01, 0.1]
```

Multi-condition runs share one realized outer split. Condition lists are
therefore rejected for `data`, `split`, `runtime`, evaluation/report controls,
inner-CV split controls, and operational preprocessing controls. Use a separate
config and study when comparing generalization regimes or datasets.

Resume an interrupted study with the same source config:

```bash
phenoradar run -c config.yml --resume runs/<study_id>
```

## Default config

Generate resolved defaults with:

```bash
phenoradar config
```

```yaml
data:
  metadata_path: testdata/c4_tiny/species_metadata.tsv
  tpm_path: testdata/c4_tiny/tpm.tsv
  tree_path: null  # type: string or null
  orthogroup_annotation_path: null  # type: string or null
  species_col: species
  feature_col: orthogroup
  value_col: tpm
  trait_col: C4
  contrast_pair_col: contrast_pair_id  # type: string or null
split:
  group_col: contrast_pair_id
  exclude_col: null  # type: string or null
  require_both_labels_per_group: false  # choices: true, false
  outer_cv_strategy: logo  # choices: logo, group_kfold, stratified_group_kfold
  outer_cv_n_splits: null  # type: integer or null
sampling:
  strategy: group_balanced  # choices: all_samples, group_balanced
  max_samples_per_label_per_group: 1  # type: integer or null
  sampled_set_count: 10
  training_group_count: null  # type: integer or null
  group_subsample_repeats: 1
  weighting: none  # choices: none, group_label_inverse
preprocess:
  max_pivot_cells: 50000000
  absent_feature_fill: 0  # choices: 0, nan
  missing_expression:
    method: none  # choices: none, neutral
    zero_as_missing: false  # choices: true, false
  expression_transform:
    method: log1p  # choices: none, log1p, sample_rank, sample_percentile_rank
  sparse_feature_filter:
    enabled: true  # choices: true, false
    min_nonzero_fraction: 0.8  # type: number or null
    scope: any_trait  # choices: all_samples, any_trait, trait_0, trait_1
  low_variance_filter:
    enabled: false  # choices: true, false
    min_variance: null  # type: number or null
  ranked_feature_filter:
    method: none  # choices: none, pair_aware, unpaired, variance
    max_features: null  # type: integer or null
    min_contrast_pairs: 1
    higher_in_trait: null  # choices: 0, 1, null
  correlation_filter:
    enabled: false  # choices: true, false
    method: pearson  # choices: pearson, spearman
    max_abs_correlation: null  # type: number or null
  feature_scaling:
    method: standard  # choices: none, standard
model:
  name: logistic_elasticnet  # choices: logistic_elasticnet, linear_svm, random_forest
abstention:
  enabled: false  # choices: true, false
  threshold: 0.8
model_selection:
  selected_candidate_count: null  # type: integer or null
  selected_candidate_percent: null  # type: number or null
  candidate_source_policy: per_sample_set  # choices: per_sample_set, reuse_first_sample_set
  search_strategy: grid  # choices: grid, random, tpe
  trial_count: null  # type: integer or null
  search_space: {}
  inner_cv_strategy: null  # choices: logo, group_kfold, stratified_group_kfold, null
  inner_cv_n_splits: null  # type: integer or null
  selection_metric: log_loss  # choices: mcc, balanced_accuracy, log_loss
  selection_rule: best  # choices: best, one_se
ensemble:
  probability_aggregation: mean  # choices: mean, median
evaluation:
  group_bootstrap:
    enabled: false  # choices: true, false
    n_resamples: 2000
    confidence_level: 0.95
summary:
  group_col: family
figures:
  top_features: 30
report: {}
runtime:
  seed: 42
  n_jobs: 1
  execution_stage: cv_only  # choices: cv_only, full_run
```

## Top-level structure

- `data`
- `split`
- `sampling`
- `preprocess`
- `model`
- `abstention`
- `model_selection`
- `ensemble`
- `evaluation`
- `summary`
- `figures`
- `report`
- `runtime`

## Key enums (quick lookup)

- `runtime.execution_stage`: `cv_only` | `full_run`
- `split.outer_cv_strategy`: `logo` | `group_kfold` | `stratified_group_kfold`
- `model.name`: `logistic_elasticnet` | `linear_svm` | `random_forest`
- `sampling.strategy`: `all_samples` | `group_balanced`
- `sampling.weighting`: `none` | `group_label_inverse`
- `preprocess.expression_transform.method`: `none` | `log1p` | `sample_rank` | `sample_percentile_rank`
- `preprocess.absent_feature_fill`: `0` | `nan`
- `preprocess.feature_scaling.method`: `none` | `standard`
- `ensemble.probability_aggregation`: `mean` | `median`
- `model_selection.search_strategy`: `grid` | `random` | `tpe`
- `model_selection.selection_metric`: `mcc` | `balanced_accuracy` | `log_loss`
- `model_selection.selection_rule`: `best` | `one_se`

## `data`

- `data.metadata_path`
  - type: `str`
  - default: `testdata/c4_tiny/species_metadata.tsv`
- `data.tpm_path`
  - type: `str`
  - default: `testdata/c4_tiny/tpm.tsv`
- `data.tree_path`
  - type: `str | null`
  - default: `null`
  - optional Newick tree used to write tree prediction annotation TSVs and, when
    Toytree is available, Toytree SVG figures.
- `data.orthogroup_annotation_path`
  - type: `str | null`
  - default: `null`
  - optional headerless TSV/TSV.GZ with orthogroup ID, annotation taxid, and
    annotation text columns. When set, run-level feature interpretation figures
    and tree feature heatmaps use annotation labels instead of ID-only labels.
- `data.species_col`
  - type: `str`
  - default: `species`
- `data.feature_col`
  - type: `str`
  - default: `orthogroup`
- `data.value_col`
  - type: `str`
  - default: `tpm`
- `data.trait_col`
  - type: `str`
  - default: `C4`
- `data.contrast_pair_col`
  - type: `str | null`
  - default: `contrast_pair_id`
  - optional contrast-pair column used by contrast-specific features such as
    `preprocess.ranked_feature_filter.method=pair_aware` and tree contrast-pair
    QC. Set to `null`
    when the workflow does not use contrast pairs.

## `split`

- `split.group_col`
  - type: `str`
  - default: `contrast_pair_id`
  - grouping column used for outer CV, group-balanced sampling, and
    group-label inverse weighting.
  - when this column matches the non-null `data.contrast_pair_col`, labeled
    species without a contrast pair automatically enter `external_test`.
  - with other split groups, labeled species must have a non-empty group
    unless explicitly excluded. Missing contrast pairs do not affect their
    pool assignment.
- `split.exclude_col`
  - type: `str | null`
  - default: `null`
  - optional metadata column marking species to remove from CV, external test,
    and inference pools. Accepted true values are `yes`, `true`, and `1`;
    false values are `no`, `false`, `0`, empty, or null.
- `split.require_both_labels_per_group`
  - type: `bool`
  - default: `false`
  - when true, only groups containing both labels (`0` and `1`) are eligible
    for CV. Eligibility is checked using labeled, non-excluded species with
    a split group, before any folds are constructed.
    Labeled species in single-label groups are assigned to `external_test`
    and are absent from both CV training/validation and final-refit training.
    Trait-missing species remain in `discovery_inference`.
  - with `false` and `sampling.strategy: all_samples`, single-label groups
    remain in CV. Validation folds may contain one label; every training fold
    must contain both. `group_balanced` sampling requires both labels in each
    group.
- `split.outer_cv_strategy`
  - type: `logo | group_kfold | stratified_group_kfold`
  - default: `logo`
  - `stratified_group_kfold` keeps groups intact while approximating the
    overall trait-label ratio in each fold. It shuffles groups reproducibly
    using `runtime.seed`.
- `split.outer_cv_n_splits`
  - type: `int >= 1 | null`
  - default: `null`
  - rule:
    - required when `outer_cv_strategy=group_kfold|stratified_group_kfold`
    - must be `>= 2`
    - must be `null` when `outer_cv_strategy=logo`

For family-level CV using existing `family` annotations:

```yaml
split:
  group_col: family
  require_both_labels_per_group: true
```

This uses all eligible labeled species in families containing both labels,
including species without contrast pairs. Single-label families become
`external_test`. This filters the training population before CV; it does not
merely omit single-label validation folds from the reported metrics.

## `sampling`

- `sampling.strategy`
  - type: `all_samples | group_balanced`
  - default: `group_balanced`
- `sampling.max_samples_per_label_per_group`
  - type: `int >= 1 | null`
  - default: `1`
- `sampling.sampled_set_count`
  - type: `int >= 1`
  - default: `10`
- `sampling.training_group_count`
  - type: `int >= 1 | null`
  - default: `null`
  - exact number of fold-local training groups retained before species-level
    sample-set construction
  - groups are values of `split.group_col`; validation groups are never removed
    or changed by this setting
  - `null` keeps every available training group
  - a numeric count is an error when a fold or final-refit scope has fewer
    available training groups
- `sampling.group_subsample_repeats`
  - type: `int >= 1`
  - default: `1`
  - number of independently ranked training-group subsets evaluated for each
    numeric `training_group_count`
  - values greater than `1` automatically create separate study conditions with
    internal repeat indices `1..group_subsample_repeats`
  - each internal repeat index is combined with `runtime.seed`, so increasing
    the repeat count preserves all previously generated subsets
  - for a fixed internal repeat index, group rankings do not depend on
    `training_group_count`, so smaller counts are strict prefixes of larger
    counts
  - effectively inactive when `training_group_count=null`; the all-available
    condition is run once
- `sampling.weighting`
  - type: `none | group_label_inverse`
  - default: `none`

Compatibility rules:

- when `sampling.strategy=all_samples`
  - `sampling.max_samples_per_label_per_group` must be `null`
  - `sampling.sampled_set_count` must be `1`
- `group_balanced` requires each `split.group_col` group in the
  `training_validation` pool to contain both labels (`0` and `1`) before CV
  split execution.
- `group_label_inverse` weights labels within `split.group_col` groups and does
  not require contrast-pair metadata.
- training-group subsampling happens after the outer split and only on its
  training side. Increase `group_subsample_repeats` to measure sensitivity to
  group composition; repeats are separate study conditions, not members of the
  `sampled_set_count` ensemble.
- when model selection is active, `training_group_count` must be at least `2`
  for inner `logo`, or at least `model_selection.inner_cv_n_splits` for inner
  `group_kfold`/`stratified_group_kfold`.

## `preprocess`

- `preprocess.max_pivot_cells`
  - type: `int >= 1`
  - default: `50000000`
  - meaning: upper bound for direct species x feature pivot size before chunked
    pivot mode
- `preprocess.absent_feature_fill`
  - type: `0 | nan`
  - default: `0`
  - behavior:
    - `0`: represent an absent `(species, feature)` coordinate as numeric zero
    - `nan`: preserve it as a floating-point missing value through expression
      transforms, optional scaling, model fitting, and prediction
  - rule: `nan` is supported with `model.name=random_forest` or the neutral
    logistic-regression mode described below
  - sparse filtering still counts an absent coordinate as not nonzero; the
    remaining feature statistics use available finite observations
  - the selected policy is stored in full-run model bundles and reused during
    `predict`, including for bundle features entirely absent from prediction input

### `preprocess.missing_expression` and `abstention`

These features are opt-in in built-in defaults. The checked-in `config.yml`
enables neutral expression handling and fixed-threshold abstention:

```yaml
model:
  name: logistic_elasticnet
preprocess:
  absent_feature_fill: nan
  missing_expression:
    method: neutral
    zero_as_missing: true
  expression_transform:
    method: log1p
  feature_scaling:
    method: standard
abstention:
  enabled: true
  threshold: 0.8
```

- `preprocess.missing_expression.method`: `none` (default) or `neutral`.
  Neutral mode requires logistic regression, `log1p`, standard scaling, and
  `absent_feature_fill: nan`. Retrain the model when enabling this mode.
- `zero_as_missing`: default `false`. With `true`, explicit numeric zeros are
  treated as unknown after duplicate coordinates have been summed. Missing
  coordinates are unknown with either setting. Positive TPM remains observed;
  no small-positive-expression cutoff is inferred. This policy cannot distinguish
  biological silence from technical failure and discards zero-expression evidence.
- Unknown values stay missing during feature selection. All-missing, singleton,
  and constant-observation features are removed. Supervised ranking requires
  observations in both labels; pair-aware filtering also requires the configured
  number of valid contrasts for each retained feature.
- Means and population standard deviations are fitted on observed values in each
  model's sampled training set. Standardized missing inputs become exactly zero,
  so their additive linear contribution is zero. Raw TPM and observation status
  are kept separately for evidence. Validation/inference never refits these statistics.
- `abstention.enabled`: default `false`; requires neutral mode.
- `abstention.threshold`: fixed value in `(0, 1]`, default `0.8`. No CV threshold
  search, artificial missingness training, or minimum validation sample count is
  introduced. Existing model selection and evaluation CV remain available.

For each model, information coverage is
`sum(abs(coef[j]) * observed[j]) / sum(abs(coef[j]))`. Intercept-only models
have zero coverage. For ensembles, coverages are averaged arithmetically,
independently of mean/median probability aggregation. A genuinely observed value
equal to the training mean counts as observed, even though its standardized value
is zero. Opposite-signed model coefficients never cancel in this calculation.

Predictions with coverage at least the threshold are accepted; lower coverage
produces a null selective label. Probabilities and the raw fixed-0.5 labels remain
available. A model with no informative coefficients always abstains. The gate
does not reject a probability merely because it is close to 0.5. Coverage is a
measurement-availability heuristic, not a calibrated confidence or error bound.
Removing negative evidence can still increase the predicted probability.

The bundle persists both policies, the scalers, and the fixed gate threshold;
`predict` uses the saved policies rather than the prediction config's overrides.

### `preprocess.expression_transform`

- `method`
  - type: `none | log1p | sample_rank | sample_percentile_rank`
  - default: `log1p`
  - behavior:
    - applied after the expression matrix is built and before feature filters
    - `none`: use input values as-is
    - `log1p`: use `log(1 + value)`; values must be non-negative
    - `sample_rank`: within each sample, rank positive feature values and keep
      zero values at `0`
    - `sample_percentile_rank`: same zero-preserving sample-wise ranking, scaled
      by the number of positive features so the largest positive feature is `1`

### `preprocess.sparse_feature_filter`

- `enabled`
  - type: `bool`
  - default: `true`
- `min_nonzero_fraction`
  - type: `float in [0, 1] | null`
  - default: `0.8`
  - rule: required when `enabled=true`
  - behavior: keeps a feature when its nonzero fraction is at least this value
    in the training population selected by `scope`
- `scope`
  - type: `all_samples | any_trait | trait_0 | trait_1`
  - default: `any_trait`
  - behavior:
    - `all_samples`: divides the nonzero count by the total number of sampled
      training species, without using their trait labels
    - `any_trait`: calculates the fraction separately in each trait class and
      keeps a feature if at least one class meets the threshold
    - `trait_0` / `trait_1`: calculates the fraction only in the selected class;
      that class must be present in the training rows

For example, with five species of each trait, a feature detected in one trait-0
species and all five trait-1 species has overall fraction `6/10 = 0.6`. At a
threshold of `0.8`, `all_samples` removes it, while `any_trait` and `trait_1`
retain it. With unequal class sizes, `all_samples` still divides by total species
count; it does not take an unweighted mean of the two class fractions and does
not use model sample weights.

The filter runs on each model's sampled training set, including inner-CV training
partitions when model selection is enabled. Validation and prediction species
never affect its fractions. Values are counted after the expression transform
and before standardization or neutral filling, using the existing nonzero
tolerance. Zeros and missing values at this stage remain in the denominator but
not the numerator. Under neutral `log1p` handling, uncertain raw zeros are
already missing when the filter runs.

To select features by their overall training prevalence:

```yaml
preprocess:
  sparse_feature_filter:
    enabled: true
    min_nonzero_fraction: 0.8
    scope: all_samples
```

Condition lists such as `scope: [all_samples, any_trait, trait_1]` are supported.

### `preprocess.low_variance_filter`

- `enabled`
  - type: `bool`
  - default: `false`
- `min_variance`
  - type: `float >= 0 | null`
  - default: `null`
  - rule: required when `enabled=true`

### `preprocess.ranked_feature_filter`

- `method`
  - type: `none | pair_aware | unpaired | variance`
  - default: `none`
- `max_features`
  - type: `int >= 1 | null`
  - default: `null`
  - rule: required when `method` is not `none`
- `min_contrast_pairs`
  - type: `int >= 1`
  - default: `1`
- `higher_in_trait`
  - type: `0 | 1 | null`
  - default: `null`
  - rule: a non-`null` value is valid only with `pair_aware` or `unpaired`
- behavior:
  - all methods are fitted using training rows only, after sparse and
    low-variance filtering and before correlation filtering
  - `none` keeps all candidates and does not require `max_features`
  - `pair_aware` calculates `mean(trait 1) - mean(trait 0)` within each valid
    contrast group and ranks by a stabilized paired t-like score computed from
    per-group label contrasts; with no usable standard errors it ranks by
    unstandardized effect magnitude while still enforcing directional
    eligibility
  - `unpaired` calculates `mean(trait 1) - mean(trait 0)`, ignores contrast
    groups, and ranks by a stabilized label-mean
    difference divided by its Welch standard error; with no usable standard
    errors it ranks by unstandardized effect magnitude while still enforcing
    directional eligibility
  - `variance` ranks by train-set sample variance without using labels
  - keeps the top `max_features`
  - `null` ranks both effect directions by magnitude, preserving the
    original behavior
  - `1` makes only positive effects eligible;
    `0` makes only negative effects eligible
  - directional modes keep at most `max_features` eligible features and never
    fill unused slots with zero-effect or opposite-direction features
  - directional modes fail closed when no candidate has an eligible effect
  - when fewer than `min_contrast_pairs` valid contrast pairs are available in a
    split, `null` mode is skipped with a warning; directional modes fail
    closed because their direction constraint cannot be verified
  - `pair_aware` requires `data.contrast_pair_col`; this is independent from
    `split.group_col`, so taxonomic-rank splits can still use contrast-pair
    feature scoring where contrast pairs are available.

To retain features that provide positive expression evidence for trait `1`, use
both filters together:

```yaml
preprocess:
  sparse_feature_filter:
    enabled: true
    min_nonzero_fraction: 0.9
    scope: trait_1
  ranked_feature_filter:
    method: pair_aware
    max_features: 200
    min_contrast_pairs: 1
    higher_in_trait: 1
```

`scope: trait_1` only requires frequent nonzero expression within trait `1`;
it does not by itself require expression to be higher than in trait `0`.
Conversely, `higher_in_trait` constrains the univariate train-fold filtering
effect but does not constrain the sign or monotonicity learned by the final
multivariable model.

### `preprocess.correlation_filter`

- `enabled`
  - type: `bool`
  - default: `false`
- `method`
  - type: `pearson | spearman`
  - default: `pearson`
- `max_abs_correlation`
  - type: `float | null`
  - default: `null`
  - rules:
    - required when `enabled=true`
    - must be in `(0, 1]`

### `preprocess.feature_scaling`

- `method`
  - type: `none | standard`
  - default: `standard`
  - behavior:
    - applied after all feature filters and before model fitting/prediction
    - `none`: do not scale selected features
    - `standard`: fit a scikit-learn `StandardScaler` on the sampled training
      matrix and transform validation/target matrices with that fitted state

## `model`

- `model.name`
  - type: `logistic_elasticnet | linear_svm | random_forest`
  - default: `logistic_elasticnet`
  - behavior:
    - `logistic_elasticnet` uses the native glmnet binomial solver through
      `python-glmnet`; no R installation or subprocess is needed.
    - `alpha=0` gives L2, `alpha=1` gives L1, and `0 < alpha < 1` gives elastic net.
    - `lambda` controls regularization strength; larger values shrink more.
    - The intercept is unpenalized. Internal standardization is disabled:
      feature scaling is controlled by `preprocess.feature_scaling`, including
      observed-only scaling for neutral missing-expression inputs.
    - `linear_svm` and `random_forest` use scikit-learn.

For grid and random searches, candidates with identical `alpha`, `thresh`, and
`maxit` are fitted in a single descending lambda path per inner fold. Each fold
has independent fitted state. Independent folds and paths are scheduled within
`runtime.n_jobs`. TPE proposes candidates sequentially and uses individual fits.
Path fitting is automatic; there is no solver or warm-start switch.

Positive lambda gaps are automatically filled with internal warm-start points,
with at most 0.05 decades between successive values. These points aid convergence;
they are not additional search candidates and receive no CV scores. Requested
lambdas and convergence settings remain exact and unchanged. A single candidate
stays a single fit, and lambda zero has no logarithmic bridge.

Outer-CV and final-refit training also use the available stronger candidate lambdas
with matching `alpha`, `thresh`, and `maxit` as a descending path to the selected
lambda. Each refit uses its own training data and retains only the selected model.
Only the original candidate lambdas are eligible for selection; if no matching
stronger candidate exists, the refit uses the selected lambda alone. This behavior
requires no config change.

The logistic search parameters are now `lambda`, `alpha`, `maxit`, and `thresh`.
Configs using removed keys must be updated, and previously saved logistic model
bundles must be regenerated. In particular, `alpha` now means the L1 fraction.

## `model_selection`

- `model_selection.selected_candidate_count`
  - type: `int >= 1 | null`
  - default: `null`
- `model_selection.selected_candidate_percent`
  - type: `float > 0 and <= 100 | null`
  - default: `null`
  - rule: mutually exclusive with `selected_candidate_count`
- `model_selection.candidate_source_policy`
  - type: `per_sample_set | reuse_first_sample_set`
  - default: `per_sample_set`
- `model_selection.search_strategy`
  - type: `grid | random | tpe`
  - default: `grid`
- `model_selection.trial_count`
  - type: `int >= 1 | null`
  - default: `null`
  - rule: required when `search_strategy=random|tpe`
- `model_selection.search_space`
  - type: mapping (`dict[str, SearchSpaceValue]`)
  - default: `{}`
- `model_selection.inner_cv_strategy`
  - type: `logo | group_kfold | stratified_group_kfold | null`
  - default: `null`
- `model_selection.inner_cv_n_splits`
  - type: `int >= 1 | null`
  - default: `null`
  - rules:
    - required when `inner_cv_strategy=group_kfold|stratified_group_kfold`
    - must be `>= 2`
    - must be `null` when `inner_cv_strategy=logo|null`
- `model_selection.selection_metric`
  - type: `mcc | balanced_accuracy | log_loss`
  - default: `log_loss`
- `model_selection.selection_rule`
  - type: `best | one_se`
  - default: `best`
  - behavior:
    - `best`: rank candidates by the inner-CV mean `selection_metric`.
    - `one_se`: find the best inner-CV mean score, keep candidates within one
      standard error of that best score, then prefer the simpler model among
      that eligible set.

Compatibility rules:

- `selected_candidate_count` and `selected_candidate_percent` are mutually exclusive.
- `selected_candidate_count` or `selected_candidate_percent` requires `inner_cv_strategy`.
- when selection is active, top-N selection is applied per sampled set and selected models are always distinct by hyperparameter set.
- with `selection_rule=one_se`, "simpler" means larger `lambda` for logistic
  elastic net (then larger `alpha`), or smaller `C` for linear SVM.
  For random forest, shallower trees, larger split/leaf minima, and fewer trees
  are preferred in that order.
- `candidate_source_policy=per_sample_set`: select candidates independently for each sampled set.
- `candidate_source_policy=reuse_first_sample_set`: select candidates once from sampled set `0` and reuse them for all sampled sets.
- `search_strategy=grid` cannot use
  `continuous_range`/`continuous_log_range`.
- search-space list values cannot be empty.

### `model_selection.search_space` value formats

Each parameter value can be one of:

- explicit list, e.g. `lambda: [0.0001, 0.001, 0.01]`
- `range`
- `int_range`
- `log_range`
- `continuous_range`
- `continuous_log_range`

`continuous_*` types are valid only for `random`/`tpe`.

For `grid`, explicit lists are expanded as-is. If a parameter list has exactly
one value (for example, `max_iter: [200]`), that parameter is effectively fixed
while other parameters are searched.

#### `range` (discrete float values)

Use this when you want evenly spaced floating-point candidates.

Fields:

- `type`: must be `range`
- `start`: first value
- `end`: upper boundary
- `step`: increment (`> 0`)
- `inclusive_end`: whether `end` can be included (default `false`)

Behavior:

- `inclusive_end=false`:
  - generate `start, start + step, ...` while value `< end`
- `inclusive_end=true`:
  - generate `start, start + step, ...` while value `<= end`
- if `end < start`, config validation fails
- if expansion produces zero values, run fails (for example:
  `start=1.0, end=1.0, step=0.1, inclusive_end=false`)

Example:

```yaml
search_space:
  lambda:
    type: range
    start: 0.001
    end: 0.011
    step: 0.002
    inclusive_end: true
```

Expanded values: `0.001, 0.003, 0.005, 0.007, 0.009, 0.011`

#### `int_range` (discrete integer values)

Same idea as `range`, but integer-only.

Fields:

- `type`: must be `int_range`
- `start`: first integer value
- `end`: upper boundary
- `step`: increment (`> 0`)
- `inclusive_end`: whether `end` can be included (default `false`)

Example:

```yaml
search_space:
  max_iter:
    type: int_range
    start: 100
    end: 301
    step: 100
    inclusive_end: false
```

Expanded values: `100, 200, 300`

#### `log_range` (discrete logarithmic values)

Use this for multiplicative spacing such as `0.001, 0.01, 0.1, ...`.

Fields:

- `type`: must be `log_range`
- `base`: logarithm base (`> 0`, and not `1`)
- `start_exp`: first exponent
- `end_exp`: exponent upper boundary
- `step_exp`: exponent increment (`> 0`)
- `inclusive_end`: whether `end_exp` can be included (default `false`)

Values are generated as `base ** exponent`.

Example:

```yaml
search_space:
  lambda:
    type: log_range
    base: 10
    start_exp: -5
    end_exp: -1
    step_exp: 1
    inclusive_end: true
```

Expanded values: `0.00001, 0.0001, 0.001, 0.01, 0.1`

#### `continuous_range` and `continuous_log_range`

These are sampled continuously (not expanded into a full discrete list), and
can be used only with `search_strategy=random|tpe`.

- `continuous_range`: uniform sample in `[start, end]`
- `continuous_log_range`: sample exponent in `[start_exp, end_exp]`, then
  transform with `base ** exponent`

Example (`random` + spec-based ranges):

```yaml
model_selection:
  search_strategy: random
  trial_count: 30
  search_space:
    lambda:
      type: log_range
      base: 10
      start_exp: -5
      end_exp: -1
      step_exp: 1
    alpha:
      type: continuous_range
      start: 0.0
      end: 1.0
```

Example (`grid` + explicit `[]` lists):

```yaml
model_selection:
  search_strategy: grid
  trial_count: null
  search_space:
    lambda: [0.0001, 0.001, 0.01]
    alpha: [0.2, 0.5, 0.8]
    maxit: [2000000]
```

### Allowed search-space parameter names by model

- `logistic_elasticnet`: `lambda`, `alpha`, `maxit`, `thresh`
- `linear_svm`: `C`, `max_iter`
- `random_forest`: `n_estimators`, `max_depth`, `min_samples_split`,
  `min_samples_leaf`

Unknown logistic parameter names are rejected during config validation;
unsupported parameters for other models are rejected at training time.

For logistic elastic net, defaults are `lambda=0.01`, `alpha=0.5`,
`maxit=2000000`, and `thresh=1e-14`. `lambda >= 0` sets regularization
strength directly: larger values shrink coefficients more. `alpha` is in
`[0, 1]`. `maxit` is a positive integer limiting coordinate-descent passes across
the entire lambda path; it is not an IRLS iteration count. `thresh > 0` controls
glmnet's relative objective-improvement stopping criterion. The strict default
was chosen to preserve prediction accuracy at weak regularization. If the native
solver fails or returns an incomplete path, training fails with an error rather
than scoring partially converged candidates. Increase `maxit` when its limit is
reached. Native convergence does not imply a fixed gradient-residual tolerance.

The optimized objective is the sample-weighted mean binary log loss plus
`lambda * alpha * sum(abs(beta))` and
`0.5 * lambda * (1 - alpha) * sum(beta ** 2)`. The intercept is unpenalized.
Multiplying all sample weights by the same positive constant leaves this
objective unchanged. There is no conversion from the previous `C` scale.

A starting grid for the working expression model is:

```yaml
model_selection:
  search_space:
    lambda:
      type: log_range
      base: 10
      start_exp: -5
      end_exp: -1
      step_exp: 0.5
      inclusive_end: true
    alpha: [1]
    maxit: [2000000]
    thresh: [1.0e-14]
```

This evaluates nine regularization strengths. Re-tune this grid for the data;
it is not a numerical translation of a previous scikit-learn run.

## `ensemble`

- `ensemble.probability_aggregation`
  - type: `mean | median`
  - default: `mean`

## `evaluation`

- `evaluation.group_bootstrap.enabled`
  - type: `bool`
  - default: `false`
  - behavior: resample intact outer-CV groups from pooled OOF predictions and
    estimate percentile confidence intervals without refitting models.
- `evaluation.group_bootstrap.n_resamples`
  - type: `int`
  - default: `2000`
  - rule: must be `>= 1`
- `evaluation.group_bootstrap.confidence_level`
  - type: `float`
  - default: `0.95`
  - rule: must be strictly between `0` and `1`

The bootstrap group is always `split.group_col`. For example, it is
`contrast_pair_id`, `family`, or `order` when that column is selected for
the split. If there are `G` unique OOF groups, every replicate draws `G` groups
with replacement and includes all species in every selected group. The seed is
derived deterministically from `runtime.seed`.

## `summary`

- `summary.group_col`
  - type: `str`
  - default: `family`
  - behavior:
    - metadata column used for stage-level grouped prediction summaries and display labels.
    - values are readable names, such as `Poaceae` in the `family` column;
      a separate ID or name column is not required.
    - if the column is absent from `data.metadata_path`, grouped summary tables
      and figures are skipped with a warning; model training/prediction still
      proceeds.

## `figures`

- `figures.top_features`
  - type: `int`
  - default: `30`
  - rule: must be in `[1, 100]`
  - behavior:
    - controls how many top-ranked features are shown in
      `cv/figures/feature_importance_top.svg`,
      `cv/figures/feature_importance_by_fold_heatmap.svg`,
      `cv/figures/coefficients_signed_top.svg`,
      `cv/figures/feature_stability_top.svg`, and tree feature heatmaps when
      `data.tree_path` is set.
    - also limits the candidate-local features in each
      `inference/figures/candidate_evidence/**/*.pdf`; candidate features are ranked
      separately for each species by mean absolute local contribution.
    - also limits the species-local features in each misclassified-OOF diagnostic under
      `cv/figures/species_evidence/**/*.pdf`; features are ranked separately for each
      species using only the models that produced its held-out-fold prediction.

## `report`

- No user-configurable report settings are currently defined.
- Prediction labels use the fixed probability threshold `0.5`.

## `runtime`

- `runtime.seed`
  - type: `int`
  - default: `42`
- `runtime.n_jobs`
  - type: `int`
  - default: `1`
  - rule: must be `>= 1`
  - behavior:
    - global CPU upper bound for training-time parallel work.
    - outer-CV folds can run in parallel up to this limit.
    - within each running fold, model-selection candidate scoring
      (`search_strategy=grid|random` with selection active) can
      also run in parallel.
    - per-model `random_forest` threads are auto-adjusted against the remaining
      fold budget so combined fold/candidate/model parallel work stays within
      this limit.
    - NumPy/SciPy/scikit-learn native thread pools (BLAS/OpenMP) are
      runtime-limited to this budget at execution points.
    - Polars thread-pool size is process-initialized.
    - when using `phenoradar` / `phrad` CLI entrypoints, `POLARS_MAX_THREADS`
      is auto-set from resolved `runtime.n_jobs` unless already set.
    - when running without the launcher entrypoint (for example
      `python -m phenoradar.cli`), set `POLARS_MAX_THREADS` before process
      start when you need a strict cap.
- `runtime.execution_stage`
  - type: `cv_only | full_run`
  - default: `cv_only`

## Practical notes

- Empty `search_space` is valid and means "no hyperparameter variation".
- With default config, model selection is effectively disabled
  (`selected_candidate_count: null` and `selected_candidate_percent: null`).
- Use `phenoradar config` to inspect resolved and validated config before a long
  run.
