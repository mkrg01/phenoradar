# Quickstart

This guide walks through the shortest path to a first PhenoRadar run.

## TL;DR (first successful run)

This is the fastest path:

```bash
pip install phenoradar
phenoradar dataset
phenoradar config
phenoradar run -c config.yml
```

You should get a new run directory under:

```text
runs/<timestamp>_run_<id>/
```

## 1) Install and verify CLI

PhenoRadar requires Python 3.12+.

From source checkout:

```bash
git clone <your-repo-url>
cd phenoradar
uv sync --extra dev
```

From PyPI:

```bash
pip install phenoradar
```

You can use either CLI entrypoint:

- `phenoradar`
- `phrad`

Check installed version:

```bash
phenoradar --version
```

## 2) Install compact test data

Use the built-in dataset command:

```bash
phenoradar dataset
```

This copies the compact dataset bundled with the installed PhenoRadar package into
`testdata/c4_tiny/`; the default operation does not require network access:

- `testdata/c4_tiny/species_metadata.tsv`
- `testdata/c4_tiny/species_trait.tsv`
- `testdata/c4_tiny/ncbi_tree.nwk`
- `testdata/c4_tiny/tpm.tsv`

For development or mirrors, `--base-url URL` (or
`PHENORADAR_TESTDATA_BASE_URL`) selects an external source. Every copied or
downloaded file is checked against the bundled `SHA256SUMS` manifest.

You can also supply your own files; see [data-format.md](data-format.md) for required columns.

## 3) Generate `config.yml`

If you want to use custom files/settings, generate a config first:

```bash
phenoradar config
```

This writes `config.yml` by default.

Then edit `config.yml` as needed. For example, change `runtime.n_jobs` from `1` to `4`:

```yaml
runtime:
  seed: 42
  n_jobs: 4
  execution_stage: cv_only
```

For detailed `config.yml` guidance, see:

- [configuration.md](configuration.md) for config behavior, validation rules, and key-by-key settings.

## 4) Run your first CV-only pipeline

Run with your generated/edited config:

```bash
phenoradar run -c config.yml
```

Config notes:

- Any unspecified keys use defaults.
- Unknown keys are rejected.
- `-c` is required and accepts one YAML file.

Log options:

- default: concise progress logs
- detailed stage logs: `phenoradar run --verbose`
- minimal output: `phenoradar run --quiet`

This writes artifacts under a new run directory:

```text
runs/<timestamp>_run_<id>/
```

To compare multiple values with the same split, replace a schema-scalar value
with an ordered list, for example:

```yaml
preprocess:
  ranked_feature_filter:
    method: [none, pair_aware, unpaired, variance]
    max_features: 100
```

The same `phenoradar run -c config.yml` command then writes
`runs/<timestamp>_study_<id>/`, with one run per condition plus symmetric
pairwise tables and publication-oriented SVG/PDF/PNG figures. Conditions retain
their config order and no reference condition is selected.

Core outputs include:

- `resolved_config.yml`
- `split/tables/split_manifest.tsv`
- `split/tables/fold_validation_groups.tsv`
- `split/tables/fold_diagnostics.tsv`
- `cv/tables/metrics_cv.tsv`
- `cv/tables/feature_importance.tsv`
- `cv/tables/feature_importance_by_fold.tsv`
- `cv/figures/top_feature_expression_by_confusion.svg`
- `cv/tables/coefficients.tsv`
- `cv/tables/coefficients_by_fold.tsv`
- `cv/tables/feature_stability_by_feature.tsv`
- `cv/tables/feature_stability_by_fold_pair.tsv`
- `cv/tables/feature_stability_summary.tsv`
- `cv/tables/prediction_cv.tsv`
- `model/tables/thresholds.tsv`
- `model/tables/evaluation_contract.tsv`
- `summary/tables/classification_summary.tsv`
- `runtime/tables/timing.tsv`
- `run_metadata.json`
- `cv/figures/`

If warnings are recorded, they are printed at command end and stored in
`run_metadata.json` (`warnings` field).

To add group-level 95% confidence intervals for the pooled OOF metrics, enable:

```yaml
evaluation:
  group_bootstrap:
    enabled: true
    n_resamples: 2000
    confidence_level: 0.95
```

This writes `cv/tables/group_bootstrap_metrics.tsv`, the per-replicate audit
table, and `cv/figures/group_bootstrap_metrics.svg`. The resampling unit is the
configured `split.group_col`.

`runtime/tables/timing.tsv` is always written. Start with `scope=run` and
`stage=total`, then inspect `outer_fold`, `sample_set_id`, and
`candidate_index` rows to locate bottlenecks. Parallel intervals can overlap,
so their durations are not additive.

## 5) Run full refit and export a reusable bundle

```bash
phenoradar run -c config.yml --execution-stage full_run
```

`full_run` adds:

- `external_test/tables/prediction_external_test.tsv`
- `inference/tables/prediction_inference.tsv`
- `external_test/figures/`
- `inference/figures/`
- `model_bundle/`

## 6) Predict with a model bundle

Create `predict_config.yml`:

```yaml
data:
  metadata_path: data/species_metadata_predict.tsv
  tpm_path: data/tpm_predict.tsv
```

Run prediction:

```bash
phenoradar predict \
  --model-bundle runs/<run_id>/model_bundle \
  -c predict_config.yml
```

This writes:

```text
runs/<timestamp>_predict_<id>/
```

With:

- `inference/tables/prediction_inference.tsv`
- `run_metadata.json`
- `inference/figures/`

## 7) Aggregate multiple runs

```bash
phenoradar report --runs-root runs
```

The default comparison guard rejects rankings that mix datasets or realized splits. Use
`--allow-mixed-experiments` only when that cross-experiment comparison is intentional.

This writes:

```text
reports/<timestamp>_report_<id>/
```

With:

- `report_manifest.json`
- `report_runs.tsv`
- `report_ranking.tsv`
- `report_warnings.tsv`
- `figures/`

## 8) Next docs (recommended)

1. [data-format.md](data-format.md) for strict TSV requirements.
2. [configuration.md](configuration.md) for config behavior, common validation rules, and key-by-key settings.
3. [cli-reference.md](cli-reference.md) for all command options.
4. [output-artifacts.md](output-artifacts.md) for file-level output schemas and interpretation guidance.
5. [pipeline-details.md](pipeline-details.md) for internal execution behavior.
