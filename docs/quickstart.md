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

## 2) Fetch compact test data

Use the built-in download command:

```bash
phenoradar dataset
```

This downloads a compact dataset from GitHub into `testdata/c4_tiny/`:

- `testdata/c4_tiny/species_metadata.tsv`
- `testdata/c4_tiny/tpm.tsv`

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

Core outputs include:

- `resolved_config.yml`
- `split_manifest.tsv`
- `metrics_cv.tsv`
- `thresholds.tsv`
- `feature_importance.tsv`
- `coefficients.tsv`
- `prediction_cv.tsv`
- `classification_summary.tsv`
- `run_metadata.json`
- `figures/`

If warnings are recorded, they are printed at command end and stored in
`run_metadata.json` (`warnings` field).

## 5) Run full refit and export a reusable bundle

```bash
phenoradar run -c config.yml --execution-stage full_run
```

`full_run` adds:

- `prediction_external_test.tsv`
- `prediction_inference.tsv`
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

- `prediction_inference.tsv`
- `run_metadata.json`
- `figures/`

## 7) Aggregate multiple runs

```bash
phenoradar report --runs-root runs
```

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
