# Configuration

PhenoRadar config files are YAML mappings validated by Pydantic.

This page is the canonical reference for config input behavior, defaults, and
key-by-key definitions.
For runtime execution flow tied to config keys, see
[pipeline-details.md](pipeline-details.md).

## Config input behavior

CLI config input is one YAML file:

- `run`: required (`-c config.yml`)
- `predict`: required (`-c config.yml`)
- `config`: optional (`-c config.yml`), omitted means built-in defaults only

Resolution and override rules:

- CLI accepts one `-c` file (`run`/`predict` required, `config` optional).
- Unspecified keys are filled by built-in defaults.
- Unknown keys are rejected.
- `runtime.execution_stage` can be overridden from CLI
  (`phenoradar run -c config.yml --execution-stage ...`).

Use `config` to inspect resolved output:

```bash
phenoradar config [--out resolved.yml]
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
  species_col: species
  feature_col: orthogroup
  value_col: tpm
  trait_col: C4
  group_col: contrast_pair_id
split:
  outer_cv_strategy: logo
  outer_cv_n_splits: null
sampling:
  strategy: group_balanced
  max_samples_per_label_per_group: 1
  sampled_set_count: 10
  weighting: none
preprocess:
  max_pivot_cells: 50000000
  low_prevalence_filter:
    enabled: true
    min_species_per_feature: 2
  low_variance_filter:
    enabled: false
    min_variance: null
  correlation_filter:
    enabled: false
    method: pearson
    max_abs_correlation: null
model:
  name: logistic_elasticnet
model_selection:
  selected_candidate_count: null
  candidate_source_policy: reuse_first_sample_set
  search_strategy: grid
  trial_count: null
  search_space: {}
  inner_cv_strategy: null
  inner_cv_n_splits: null
  selection_metric: mcc
ensemble:
  probability_aggregation: mean
report:
  fixed_probability_threshold: 0.5
  auto_threshold_selection_metric: mcc
runtime:
  seed: 42
  n_jobs: 1
  execution_stage: cv_only
```

## Top-level structure

- `data`
- `split`
- `sampling`
- `preprocess`
- `model`
- `model_selection`
- `ensemble`
- `report`
- `runtime`

## Key enums (quick lookup)

- `runtime.execution_stage`: `cv_only` | `full_run`
- `split.outer_cv_strategy`: `logo` | `group_kfold`
- `model.name`: `logistic_elasticnet` | `linear_svm` | `random_forest`
- `sampling.strategy`: `all_samples` | `group_balanced`
- `sampling.weighting`: `none` | `group_label_inverse`
- `ensemble.probability_aggregation`: `mean` | `median`
- `model_selection.search_strategy`: `grid` | `random` | `tpe`
- `model_selection.selection_metric`: `mcc` | `balanced_accuracy`
- `report.auto_threshold_selection_metric`: `mcc` | `balanced_accuracy`

## `data`

- `data.metadata_path`
  - type: `str`
  - default: `testdata/c4_tiny/species_metadata.tsv`
- `data.tpm_path`
  - type: `str`
  - default: `testdata/c4_tiny/tpm.tsv`
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
- `data.group_col`
  - type: `str`
  - default: `contrast_pair_id`

## `split`

- `split.outer_cv_strategy`
  - type: `logo | group_kfold`
  - default: `logo`
- `split.outer_cv_n_splits`
  - type: `int >= 1 | null`
  - default: `null`
  - rule:
    - required when `outer_cv_strategy=group_kfold`
    - must be `>= 2`
    - must be `null` when `outer_cv_strategy=logo`

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
- `sampling.weighting`
  - type: `none | group_label_inverse`
  - default: `none`

Compatibility rules:

- when `sampling.strategy=all_samples`
  - `sampling.max_samples_per_label_per_group` must be `null`
  - `sampling.sampled_set_count` must be `1`
- each training group must contain both labels (`0` and `1`) before CV split
  execution.
  - this refers to the internal `training_validation` pool (metadata rows with
    both trait and group present), which is expanded to `train`/`validation`
    rows in `split_manifest.tsv`.

## `preprocess`

- `preprocess.max_pivot_cells`
  - type: `int >= 1`
  - default: `50000000`
  - meaning: upper bound for direct species x feature pivot size before chunked
    pivot mode

### `preprocess.low_prevalence_filter`

- `enabled`
  - type: `bool`
  - default: `true`
- `min_species_per_feature`
  - type: `int >= 1 | null`
  - default: `2`
  - rule: required when `enabled=true`

### `preprocess.low_variance_filter`

- `enabled`
  - type: `bool`
  - default: `false`
- `min_variance`
  - type: `float >= 0 | null`
  - default: `null`
  - rule: required when `enabled=true`

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

## `model`

- `model.name`
  - type: `logistic_elasticnet | linear_svm | random_forest`
  - default: `logistic_elasticnet`

## `model_selection`

- `model_selection.selected_candidate_count`
  - type: `int >= 1 | null`
  - default: `null`
- `model_selection.candidate_source_policy`
  - type: `per_sample_set | reuse_first_sample_set`
  - default: `reuse_first_sample_set`
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
  - type: `logo | group_kfold | null`
  - default: `null`
- `model_selection.inner_cv_n_splits`
  - type: `int >= 1 | null`
  - default: `null`
  - rules:
    - required when `inner_cv_strategy=group_kfold`
    - must be `>= 2`
    - must be `null` when `inner_cv_strategy=logo|null`
- `model_selection.selection_metric`
  - type: `mcc | balanced_accuracy`
  - default: `mcc`

Compatibility rules:

- `selected_candidate_count` requires `inner_cv_strategy`.
- `search_strategy=grid` cannot use
  `continuous_range`/`continuous_log_range`.
- search-space list values cannot be empty.

### `model_selection.search_space` value formats

Each parameter value can be one of:

- explicit list, e.g. `C: [0.1, 1.0, 10.0]`
- `range`
- `int_range`
- `log_range`
- `continuous_range`
- `continuous_log_range`

`continuous_*` types are valid only for `random`/`tpe`.

For `grid`, explicit lists are expanded as-is. If a parameter list has exactly
one value (for example, `max_iter: [200]`), that parameter is effectively fixed
while other parameters are searched.

Example (`random` + spec-based ranges):

```yaml
model_selection:
  search_strategy: random
  trial_count: 30
  search_space:
    C:
      type: log_range
      base: 10
      start_exp: -3
      stop_exp: 3
      step_exp: 1
    l1_ratio:
      type: continuous_range
      start: 0.0
      stop: 1.0
```

Example (`grid` + explicit `[]` lists):

```yaml
model_selection:
  search_strategy: grid
  trial_count: null
  search_space:
    C: [0.01, 0.1, 1.0]
    l1_ratio: [0.2, 0.5, 0.8]
    max_iter: [200]
```

### Allowed search-space parameter names by model

- `logistic_elasticnet`: `C`, `l1_ratio`, `max_iter`
- `linear_svm`: `C`, `max_iter`
- `random_forest`: `n_estimators`, `max_depth`, `min_samples_split`,
  `min_samples_leaf`

Unknown parameter names are rejected at training time.

## `ensemble`

- `ensemble.probability_aggregation`
  - type: `mean | median`
  - default: `mean`

## `report`

- `report.fixed_probability_threshold`
  - type: `float`
  - default: `0.5`
  - rule: must be in `[0, 1]`
- `report.auto_threshold_selection_metric`
  - type: `mcc | balanced_accuracy`
  - default: `mcc`

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
      (`search_strategy=grid|random` with `selected_candidate_count` set) can
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
  (`selected_candidate_count: null`).
- Use `phenoradar config` to inspect resolved and validated config before a long
  run.
