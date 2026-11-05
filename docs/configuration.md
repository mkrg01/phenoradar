# Configuration

Use one YAML file for all feature modes. Generate a commented config or validate
an existing one before a long run:

```bash
phenoradar config --out defaults.yml
phenoradar config -c config.yml --out resolved.yml
```

Omitted keys use built-in defaults; unknown keys are rejected. Generated configs
list all user-facing settings, including inactive and nullable fields. Each run
saves its effective settings in `resolved_config.yml`.

The repository's working `config.yml` differs from built-in defaults: it uses
custom data paths, neutral missing-expression handling, abstention, and model
selection. The defaults below describe `phenoradar config` without `-c`.

Use these sections to find settings:

- [Inputs](#data), [splits](#split), and [sampling](#sampling)
- [Expression preprocessing](#preprocess) and [missing expression](#missing-expression-and-abstention)
- [Models](#model) and [model selection](#model_selection)
- [Output and runtime settings](#output-and-runtime-settings)
- [Prediction settings](#prediction-settings) and [condition comparisons](#ordered-multi-condition-runs)

## `data`

Keys in each table are relative to the section heading.

| Key | Default | Purpose / accepted values |
| --- | --- | --- |
| `metadata_path` | `testdata/c4_tiny/species_metadata.tsv` | Species metadata TSV |
| `tpm_path` | `testdata/c4_tiny/tpm.tsv` | Long-format OG expression TSV or gzip-compressed TSV |
| `species_col` | `species` | Species ID column |
| `feature_col` | `orthogroup` | Expression feature column |
| `value_col` | `tpm` | Expression value column |
| `trait_col` | `C4` | Binary label column |
| `contrast_pair_col` | `contrast_pair_id` | Pair-aware scoring column; `null` disables contrast-pair features |
| `tree_path` | `null` | Optional Newick tree for annotations and figures |
| `orthogroup_annotation_path` | `null` | Optional headerless annotation TSV/TSV.GZ |
| `feature_mode` | `expression` | `expression`, `amino_acid`, or `combined` |
| `alignment_dir` | `null` | Required for sequence and combined modes |
| `sequence_tpm_path` | `null` | Gene-level TPM TSV/TSV.GZ for weighted sequence frequencies |

See [data format](data-format.md) for input validation and
[amino-acid and combined modes](amino-acid.md) for sequence-specific behavior.
Cache paths and matrix-construction chunk sizes are managed internally and
omitted from generated user configs. Expression runs reuse validated normalized
TPM under `runs/.phenoradar_cache/expression/`. Input contents, selected species,
column names, normalization code, and Polars version determine cache identity;
changed or damaged entries are rebuilt. Filters and scaling are fitted within
each training scope. Delete the cache directory between runs to reclaim disk space.

## `split`

| Key | Default | Purpose / accepted values |
| --- | --- | --- |
| `group_col` | `contrast_pair_id` | Metadata group kept intact during CV; also used for sampling and weighting |
| `exclude_col` | `null` | Optional column removing flagged species from every pool |
| `require_both_labels_per_group` | `false` | Reserve labeled single-label groups for external testing |
| `outer_cv_strategy` | `logo` | `logo`, `group_kfold`, or `stratified_group_kfold` |
| `outer_cv_n_splits` | `null` | Required integer ≥ 2 for either K-fold strategy; must be `null` for `logo` |

`logo` holds out one group at a time. `stratified_group_kfold` approximates the
label ratio while preserving groups and shuffling reproducibly with
`runtime.seed`.

With a contrast-pair split, labeled species missing a pair become external-test
species. For other split groups, missing groups are errors unless the species
is excluded or unlabeled. `require_both_labels_per_group: true` removes
single-label groups from both CV and final-refit training. See
[pool assignment](data-format.md#metadata-tsv) for all routing rules.

Every training fold must contain both labels. Validation folds may contain only
one label; metrics requiring both labels are then `NA`.

## `sampling`

| Key | Default | Purpose / accepted values |
| --- | --- | --- |
| `strategy` | `group_balanced` | Balanced subsets per group, or `all_samples` |
| `max_samples_per_label_per_group` | `1` | Positive per-label cap for balanced sampling; `null` for no cap |
| `sampled_set_count` | `10` | Positive number of sampled training sets |
| `training_group_count` | `null` | Exact positive number of training groups; `null` keeps all |
| `group_subsample_repeats` | `1` | Positive number of independent group-subset conditions |
| `weighting` | `none` | `none` or `group_label_inverse` |

For `all_samples`, set both related fields explicitly:

```yaml
sampling:
  strategy: all_samples
  max_samples_per_label_per_group: null
  sampled_set_count: 1
```

`group_balanced` requires both labels in every training group.
`group_label_inverse` assigns each species weight proportional to `1 / N(g,y)`,
then normalizes weights to mean 1 within each fitted subset. Every observed
group-label combination has equal total weight; a single-label group therefore
has half the total weight of a two-label group.

`training_group_count` changes only the training side of each outer fold and is
also applied during final refit. It fails if any scope has too few groups.
With model selection, it must support at least two groups for inner `logo`, or
at least `inner_cv_n_splits` for inner K-fold CV.

Group-subset repeats are separate study conditions, not ensemble members. For a
fixed repeat index, smaller group counts are prefixes of the same deterministic
group ranking. Increasing the repeat count preserves existing subsets. The
all-groups (`null`) condition runs once. See [condition comparisons](#ordered-multi-condition-runs).

## `preprocess`

Group settings under `expression` and `amino_acid`, in that order. Expression-only
runs use `preprocess.expression`; sequence-only runs use `preprocess.amino_acid`;
combined runs use both. All modes share one config file. Flat preprocessing keys,
top-level `amino_acid`, and removed settings are rejected.

The settings below are relative to `preprocess.expression`; see the
[sequence guide](amino-acid.md#preprocessing) for `preprocess.amino_acid`.
Expression processing runs in this order:

1. Expression transform
2. Sparse and missing-observation eligibility filters
3. Low-variance filter
4. Ranked feature filter
5. Correlation filter
6. Feature scaling

Each model learns filters and scaling from its sampled training rows, including
within inner CV. Validation and prediction rows never fit preprocessing.
Expression preprocessing is inactive in sequence-only mode.

| Key | Default | Purpose / accepted values |
| --- | --- | --- |
| `absent_feature_fill` | `0` | `0` or `nan`; `nan` requires random forest or neutral logistic regression |
| `expression_transform.method` | `log1p` | `none`, `log1p`, `sample_rank`, `sample_percentile_rank` |
| `feature_scaling.method` | `standard` | `standard` or `none` |

`log1p` is the natural logarithm of `1 + TPM`. Rank transforms rank positive
features within each species and keep zeros at zero. Percentile ranks divide by
the number of positive features, so the largest positive feature is 1. Ranking
uses the full transform schema before feature filtering.

`standard` fits training means and standard deviations, then reuses them for
validation and prediction. The absent-feature policy is also saved in bundles.

### Sparse feature filter

| Key under `sparse_feature_filter` | Default | Purpose / accepted values |
| --- | --- | --- |
| `enabled` | `true` | Enable prevalence filtering |
| `min_nonzero_fraction` | `0.8` | Required fraction in `[0, 1]` when enabled |
| `scope` | `any_trait` | `all_samples`, `any_trait`, `trait_0`, or `trait_1` |

`all_samples` uses all sampled training species. `any_trait` keeps a feature if
at least one label class meets the threshold; `trait_0` and `trait_1` use only
that class. Zeros and missing values count in the denominator, never the
numerator. Model sample weights do not change these fractions.

For example, a feature observed in all five trait-1 species and one of five
trait-0 species has prevalence 0.6 overall and 1.0 in trait 1. At a 0.8 threshold,
`all_samples` removes it; `any_trait` and `trait_1` retain it.

### Low-variance and correlation filters

| Key | Default | Rule |
| --- | --- | --- |
| `low_variance_filter.enabled` | `false` | Enable variance filtering |
| `low_variance_filter.min_variance` | `null` | Required value ≥ 0 when enabled |
| `correlation_filter.enabled` | `false` | Enable correlated-feature removal |
| `correlation_filter.method` | `pearson` | `pearson` or `spearman` |
| `correlation_filter.max_abs_correlation` | `null` | Required value in `(0, 1]` when enabled |

### Ranked feature filter

| Key under `ranked_feature_filter` | Default | Purpose / accepted values |
| --- | --- | --- |
| `method` | `none` | `none`, `pair_aware`, `unpaired`, or `variance` |
| `max_features` | `null` | Positive feature limit; `null` disables the entire ranked filter |
| `min_contrast_pairs` | `1` | Positive minimum valid-pair count for pair-aware scoring |
| `higher_in_trait` | `null` | Both directions (`null`), higher in trait `1`, or higher in trait `0` |

`pair_aware` ranks stabilized t-like scores from within-pair label contrasts.
It requires `data.contrast_pair_col`, independently of `split.group_col`.
`unpaired` ranks label-mean differences divided by stabilized Welch standard
errors. Both fall back to effect magnitude when usable standard errors are
unavailable. `variance` ranks training sample variance without labels.

Direction applies only to `pair_aware` or `unpaired`: the signed effect is
`mean(trait 1) - mean(trait 0)`. Directional filters keep at most `max_features`
eligible features, never filling with opposite or zero effects. They fail if
no eligible features remain or direction cannot be verified. With too few
valid pairs, two-sided pair-aware filtering instead skips with a warning.

For positive trait-1 expression evidence:

```yaml
preprocess:
  expression:
    sparse_feature_filter:
      min_nonzero_fraction: 0.9
      scope: trait_1
    ranked_feature_filter:
      method: pair_aware
      max_features: 200
      higher_in_trait: 1
```

Prevalence in trait 1 alone does not establish higher expression than trait 0.
The ranking direction constrains univariate filtering, not the final model's
coefficient signs.

## Missing expression and abstention

Both features are disabled in built-in defaults. To make missing expression
neutral in a logistic model and withhold decisions with low information coverage:

```yaml
preprocess:
  expression:
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

| Key | Default | Rule |
| --- | --- | --- |
| `preprocess.expression.missing_expression.method` | `none` | `neutral` requires logistic regression, `log1p`, standard scaling, and `absent_feature_fill: nan` |
| `preprocess.expression.missing_expression.zero_as_missing` | `false` | With `true`, treat summed numeric zeros as unknown; requires neutral mode for expression |
| `abstention.enabled` | `false` | Requires neutral mode for expression; sequence modes use site coverage |
| `abstention.threshold` | `0.8` | Fixed information-coverage threshold in `(0, 1]` |

Neutral mode fits means and population standard deviations on observed training
values. Unknown standardized inputs become zero, giving zero linear contribution.
All-missing, singleton, and constant-observation features are removed; supervised
filters require observations in both labels and, where applicable, valid pairs.
Retraining is required when enabling this policy.

Missing coordinates are unknown regardless of `zero_as_missing`. Enabling it
also discards measured zero-expression evidence; positive TPM remains observed.
Raw TPM and observation status remain available in evidence tables.

For each expression model, information coverage is:

```text
sum(abs(coefficient) × observed) / sum(abs(coefficient))
```

Ensemble coverage is the arithmetic mean of member coverages; intercept-only
members contribute zero. An observed value at its training mean still counts
as observed. Sequence and combined coverage are defined in the
[mode guide](amino-acid.md#information-coverage).

Coverage at or above the threshold accepts the fixed-0.5 decision; otherwise
`pred_label_selective` is null. Raw probabilities and fixed-threshold labels
remain available. This gate measures information availability; it is not a
calibrated probability of correctness and does not reject probabilities near
0.5. Removing negative evidence can still raise a probability.

The bundle freezes preprocessing, coverage policy, and threshold. See
[selective outputs](output-artifacts.md#abstention) for downstream use.

## `model`

`model.name` defaults to `logistic_elasticnet`. It uses native glmnet through
`python-glmnet`, with an unpenalized intercept and no internal standardization.
Scaling is controlled by `preprocess`. `linear_svm` and `random_forest` use
scikit-learn. Sequence and combined modes require logistic regression.

Hyperparameters belong in `model_selection.search_space`:

| Model | Parameters and defaults |
| --- | --- |
| `logistic_elasticnet` | `lambda=0.01`, `alpha=0.5`, `maxit=2000000`, `thresh=1e-7` |
| `linear_svm` | `C`, `max_iter` |
| `random_forest` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Combined logistic model | Fixed `alpha=1`; optional positive `amino_acid_penalty_ratio=1` |

For logistic regression, `lambda ≥ 0` increases regularization strength.
`alpha` is the L1 fraction in `[0, 1]`: 0 gives L2, 1 gives L1. The objective is
sample-weighted mean binary log loss plus:

```text
lambda × alpha × sum(abs(beta)) + 0.5 × lambda × (1 − alpha) × sum(beta²)
```

`maxit` limits coordinate-descent passes for the entire native lambda path;
`thresh > 0` controls relative objective improvement. Native errors or incomplete
paths stop training. Increase `maxit` when its limit is reached. Path fitting is
automatic; see [solver behavior](pipeline-details.md#solver-behavior).

Old logistic search keys must be replaced with glmnet parameters and old logistic
bundles regenerated. There is no direct conversion from the previous `C` scale.

## `model_selection`

| Key | Default | Purpose / accepted values |
| --- | --- | --- |
| `selected_candidate_count` | `null` | Positive number to retain per sampled set |
| `selected_candidate_percent` | `null` | Percentage in `(0, 100]`; mutually exclusive with count |
| `candidate_source_policy` | `per_sample_set` | Select independently, or `reuse_first_sample_set` (set 0) |
| `search_strategy` | `grid` | `grid`, `random`, or `tpe` |
| `trial_count` | `null` | Required positive count for `random` and `tpe` |
| `search_space` | `{}` | Parameter lists or range specifications |
| `inner_cv_strategy` | `null` | `logo`, `group_kfold`, or `stratified_group_kfold` |
| `inner_cv_n_splits` | `null` | Integer ≥ 2 for K-fold strategies; `null` otherwise |
| `selection_metric` | `log_loss` | `log_loss` (lower), `mcc` or `balanced_accuracy` (higher) |
| `selection_rule` | `best` | Best mean inner-CV score, or `one_se` |

Selection is enabled only when count or percent is set, and then requires an
inner-CV strategy. Otherwise candidates are fitted directly as ensemble members.
An empty search space yields one default candidate; lists cannot be empty.
Selected hyperparameter sets are deduplicated.

`one_se` keeps candidates within one standard error of the best score and prefers
simpler models: larger `lambda`, then larger `alpha` for logistic regression;
smaller `C` for SVM; shallower trees, larger split/leaf minima, then fewer trees
for random forest.

A complete selection example:

```yaml
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
  search_space:
    lambda: [0.0001, 0.001, 0.01, 0.1]
    alpha: [1]
```

### Search-space formats

| Value / `type` | Fields | Behavior |
| --- | --- | --- |
| List, e.g. `[0.001, 0.01]` | Explicit values | One value fixes a parameter |
| `range` | `start`, `end`, `step` | Discrete float increments |
| `int_range` | `start`, `end`, `step` | Discrete integer increments |
| `log_range` | `base`, `start_exp`, `end_exp`, `step_exp` | Discrete `base ** exponent` values |
| `continuous_range` | `start`, `end` | Uniform sampling in the interval |
| `continuous_log_range` | `base`, `start_exp`, `end_exp` | Uniform exponent sampling, then exponentiation |

Discrete ranges exclude the endpoint unless `inclusive_end: true`. Steps must be
positive and bounds ordered; expansions must be non-empty. Log bases must be
positive and differ from 1. Continuous ranges require `random` or `tpe`.

For example, this produces nine lambda values from `1e-5` through `0.1`:

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
```

## Output and runtime settings

| Key | Default | Purpose / accepted values |
| --- | --- | --- |
| `ensemble.probability_aggregation` | `mean` | `mean` or `median` of member probabilities |
| `evaluation.group_bootstrap.enabled` | `false` | Group-resampled percentile intervals for pooled OOF metrics |
| `evaluation.group_bootstrap.n_resamples` | `2000` | Positive replicate count |
| `evaluation.group_bootstrap.confidence_level` | `0.95` | Value strictly between 0 and 1 |
| `summary.group_col` | `family` | Metadata column for grouped summaries and display labels |
| `figures.top_features` | `30` | Number of displayed global or species-local features, from 1 to 100 |
| `runtime.seed` | `42` | Reproducible splitting, sampling, and model fitting |
| `runtime.n_jobs` | `1` | Positive CPU upper bound for parallel work |
| `runtime.execution_stage` | `cv_only` | `cv_only` or `full_run`; CLI `--execution-stage` overrides it |

The group bootstrap resamples `split.group_col` from pooled out-of-fold (OOF)
predictions without refitting. Its seed derives from `runtime.seed`.

Training parallelism shares the `n_jobs` budget across folds, candidates, and
supported model threads; sequence outer folds run sequentially to bound memory.
The CLI initializes Polars threads from this value unless `POLARS_MAX_THREADS`
is already set. Native BLAS/OpenMP threads are limited during execution. Direct
Python invocations must set Polars limits before import when a strict cap matters.

`phylogenetic_imputation` defaults to `enabled: false`, `branch_length_mode: input`,
`model: ER`, `root_prior: equal`. See the
[phylogenetic guide](phylogenetic-imputation.md) for options and requirements.
`report: {}` has no configurable fields. Classification uses a fixed probability
threshold of 0.5 throughout.

## Prediction settings

Expression `predict` requires a bundle and an explicit TPM path through
`--tpm-path` or `data.tpm_path`; it never falls back to example data. Config and
metadata are optional. CLI input paths and `--n-jobs` override config values.

| Setting | Prediction default / behavior |
| --- | --- |
| `data.tpm_path` | Required expression TSV or gzip-compressed TSV |
| `data.metadata_path` | `null`; predict all TPM species, or select metadata species |
| `data.species_col`, `data.feature_col`, `data.value_col` | `species`, `orthogroup`, `tpm` |
| `data.tree_path`, `data.orthogroup_annotation_path` | `null`; optional tree and labels |
| `data.trait_col`, `data.contrast_pair_col` | `C4`, `contrast_pair_id`; optional annotations |
| `runtime.n_jobs` | `1`; also overrides inherited `POLARS_MAX_THREADS` for expression prediction |
| `summary.group_col` | `family`; summaries require metadata with this column |
| `figures.top_features` | Saved training value if `figures` is omitted; otherwise 30 unless specified |
| `phylogenetic_imputation` | Disabled; same options as training |

For example:

```yaml
data:
  tpm_path: data/tpm_predict.tsv
runtime:
  n_jobs: 4
```

Training configs using the current schema are accepted: training-only fields are ignored and
cannot alter the learned model, transforms, scaling, aggregation, or decision
policies. Unknown prediction keys are rejected. Prediction is deterministic from
the bundle and does not use `runtime.seed`. Effective prediction settings and
input/bundle hashes are saved with the results.

For sequence and combined prediction inputs and supported overrides, see
[the mode guide](amino-acid.md#prediction).

## Ordered multi-condition runs

For `run`, an ordered list in a normally scalar field creates one condition per
value on a shared outer split:

```yaml
preprocess:
  expression:
    ranked_feature_filter:
      method: [none, pair_aware, unpaired, variance]
      max_features: 100
```

Only one field may contain multiple distinct values per study. Conditions keep
config order; comparisons are symmetric, with no reference condition. Lists in
`search_space` retain their normal meaning as hyperparameter candidates.

Inactive settings are canonicalized: `method: none` ignores `max_features`.
To compare no ranked filtering with several feature limits, keep the method
fixed and include `null`:

```yaml
preprocess:
  expression:
    ranked_feature_filter:
      method: pair_aware
      max_features: [null, 1000, 100]
```

The `null` condition disables the entire ranked filter, including direction and
pair-dependent eligibility, while other preprocessing remains active.

For training-group sensitivity:

```yaml
sampling:
  training_group_count: [5, 10, 15, null]
  group_subsample_repeats: 3
```

Each numeric count gets three independently ranked subsets; the all-groups
condition runs once. Generated repeat indices do not count as another varying
field and are recorded in each run's resolved config.

Condition lists are rejected for input data, outer and inner split controls,
runtime, evaluation, figures, summary/report controls, and operational preprocessing fields.
Use separate studies to compare datasets or generalization regimes.

Resume an interrupted study with the same source config:

```bash
phenoradar run -c config.yml --resume runs/STUDY_ID
```

See [study outputs](output-artifacts.md#studies) for tables and figures.
