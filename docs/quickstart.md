# Quickstart

PhenoRadar requires Python 3.12+. The commands below use expression data;
for sequence inputs, see [amino-acid and combined modes](amino-acid.md).

## Run the example dataset

```bash
pip install phenoradar
phenoradar dataset
phenoradar config --out demo.yml
phenoradar run -c demo.yml
```

`dataset` copies the bundled data into `testdata/c4_tiny/` without network
access. `config` writes built-in defaults to `demo.yml`; these differ from the
repository's working `config.yml`.

The default run performs cross-validation (CV) and writes
`runs/<timestamp>_run_<id>/`. Start with:

| File | What to check |
| --- | --- |
| `run_metadata.json` | Completion status and warnings |
| `cv/tables/metrics_cv.tsv` | Per-fold and aggregate performance |
| `cv/tables/prediction_cv.tsv` | Held-out predictions for each species |
| `cv/figures/` | Performance and feature plots |

See [outputs and interpretation](output-artifacts.md) for the complete guide.

## Use your own data

Prepare inputs with the separate `phenoradar_prep` repository, then edit the
paths in your config. See [data format](data-format.md) for required columns.

```yaml
data:
  metadata_path: data/species_metadata.tsv
  tpm_path: data/tpm.tsv
split:
  group_col: expanded_pair_clade_id
runtime:
  n_jobs: 4
```

Choose a grouping column present in the metadata. Other omitted settings use
built-in defaults, and unknown keys are rejected. To validate
and inspect settings before training:

```bash
phenoradar config -c demo.yml --out resolved.yml
phenoradar run -c demo.yml
```

Add `--verbose` for detailed logs or `--quiet` for final summaries and warnings.

## Export a model and predict

A full run refits on the eligible labeled training pool, evaluates external-test
species, predicts unknown species, and exports a reusable model bundle:

```bash
phenoradar run -c demo.yml --execution-stage full_run
```

Replace `RUN_ID` below with that run's directory name:

```bash
phenoradar predict \
  --model-bundle runs/RUN_ID/model_bundle \
  --tpm-path data/tpm_predict.tsv \
  --n-jobs 4
```

All species in the TPM file are predicted. To select a subset or supply family
annotations, add `--metadata-path data/prediction_species.tsv`; only its
`species` column is required.

Results appear in `runs/<timestamp>_predict_<id>/`, with predictions in
`inference/tables/prediction_inference.tsv`. Learned preprocessing and decision
policies come from the bundle.

## Compare settings or runs

To compare one setting on the same outer split, give a scalar setting an ordered
list in your config:

```yaml
preprocess:
  expression:
    ranked_feature_filter:
      method: [none, pair_aware, unpaired, variance]
      max_features: 100
```

Run as usual. Results appear under `runs/<timestamp>_study_<id>/`, with one run
per condition and comparison tables and figures. Only one setting may vary per
study; see [multi-condition runs](configuration.md#ordered-multi-condition-runs).

To summarize existing runs:

```bash
phenoradar report --runs-root runs
```

Reports appear under `reports/<timestamp>_report_<id>/`. Rankings require
compatible datasets and splits; see the [CLI reference](cli-reference.md#report)
for selection and comparison options.

For other tasks, use the [documentation index](README.md).
