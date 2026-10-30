# CLI reference

`phenoradar` and `phrad` are equivalent. Use `phenoradar --help` or
`phenoradar COMMAND --help` for terminal help; `--version` / `-V` shows the
installed version.

| Command | Purpose |
| --- | --- |
| `dataset` | Copy the bundled example data |
| `config` | Generate or validate YAML settings |
| `run` | Train and evaluate; optionally export a model |
| `predict` | Apply an exported model bundle |
| `report` | Compare existing runs |

All commands accept `--verbose` / `-v` for detailed logs or `--quiet` / `-q`
for final summaries and warnings. These flags are mutually exclusive. Warnings
are saved in `run_metadata.json` for runs and expression predictions, or
`report_warnings.tsv` for reports.

## `dataset`

```bash
phenoradar dataset [--out testdata/c4_tiny] [--base-url URL] [--force]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--out` | `testdata/c4_tiny` | Destination directory |
| `--base-url` | Bundled resources | External dataset source; also settable with `PHENORADAR_TESTDATA_BASE_URL` |
| `--force` | Off | Overwrite files whose checksums differ |

The default copy needs no network access. Copied or downloaded files are checked
against the bundled `SHA256SUMS` manifest.

## `config`

```bash
phenoradar config [-c config.yml] [--out config.yml]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config`, `-c` | Built-in defaults | One optional input YAML |
| `--out` | `config.yml` | Destination YAML; overwrites the file if it exists |

Resolves defaults and validates settings without training. Output comments list
accepted choices and nullable types. Internal repeat indices and automatic cache
paths are omitted. See [configuration](configuration.md) for all settings.

## `run`

```bash
phenoradar run -c config.yml
phenoradar run -c config.yml --execution-stage full_run
phenoradar run -c config.yml --resume runs/STUDY_ID
```

| Option | Required / default | Meaning |
| --- | --- | --- |
| `--config`, `-c` | Required, exactly once | Input YAML |
| `--execution-stage` | Config value | Override with `cv_only` or `full_run` |
| `--resume` | None | Resume a study using the same ordered conditions |

`cv_only` writes held-out predictions, metrics, diagnostics, and figures.
`full_run` also refits, predicts external-test and unknown species, and exports
`model_bundle/`.

A list in a normally scalar setting creates an ordered study on one shared outer
split. See [multi-condition runs](configuration.md#ordered-multi-condition-runs).
Run and study results are created under `runs/`; file details belong in
[outputs and interpretation](output-artifacts.md).

## `predict`

```bash
phenoradar predict --model-bundle runs/RUN_ID/model_bundle --tpm-path data/tpm.tsv
phenoradar predict --model-bundle runs/RUN_ID/model_bundle -c predict_config.yml
```

| Option | Required / default | Meaning |
| --- | --- | --- |
| `--model-bundle` | Required | Bundle directory |
| `--config`, `-c` | None | One optional prediction YAML |
| `--tpm-path` | Required for expression/combined unless in config | OG expression TSV |
| `--metadata-path` | Config value, otherwise none | Species subset and annotations |
| `--n-jobs` | Config value, otherwise `1` | Positive worker/native-thread limit |
| `--alignment-dir` | Saved/configured path | Frozen alignments for sequence or combined bundles |
| `--sequence-tpm-path` | Bundled weights | Override gene-level TPM for weighted sequence bundles |

Explicit CLI values override config values. Expression prediction uses all TPM
species when metadata is omitted; metadata needs only the species column.
Model state, preprocessing, aggregation, and decision policies come from the
bundle. Training settings in an older config do not retrain or change the model.

Expression results are written to `runs/<timestamp>_predict_<id>/`, with
predictions at `inference/tables/prediction_inference.tsv`. See
[prediction settings](configuration.md#prediction-settings) for accepted config
fields and [prediction outputs](output-artifacts.md#predict-artifacts) for
evidence PDFs and tables.

Sequence and combined bundles require original species and frozen alignments;
their predictions use `predict/tables/pred_predict.tsv`. A sequence-only bundle
does not accept `--tpm-path`. See [sequence prediction](amino-acid.md#prediction).

## `report`

```bash
phenoradar report --runs-root runs
phenoradar report --run-dir runs/RUN_A --run-dir runs/RUN_B --out reports/comparison
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-dir` | None | Explicit run directory; repeatable |
| `--runs-root` | None | Root to scan for run directories |
| `--glob` | `*` | Pattern when scanning the root |
| `--latest` | All matches | Keep latest N directories after glob expansion |
| `--primary-metric` | `mcc` | `mcc`, `balanced_accuracy`, `roc_auc`, `pr_auc`, or `brier` |
| `--aggregate-scope` | `macro` | `macro` or `micro` |
| `--include-stage` | `all` | `cv_only`, `full_run`, `predict`, or `all` |
| `--strict` | Off | Fail on invalid/missing artifacts |
| `--allow-mixed-experiments` | Off | Allow rankings across different or unknown experiment fingerprints |
| `--output-format` | `tsv` | `tsv`, `md`, `html`, or `json` |
| `--out` | New directory under `reports/` | Report destination |

Ranks higher scores first except for Brier score, where lower is better.
`pr_auc` means Average Precision, not trapezoidal PR area. Ties use ascending
start time, then run ID.

Ranked runs must share an experiment fingerprint covering input contents,
realized split, and evaluation contract. File paths and model hyperparameters do
not define that identity. Use `--allow-mixed-experiments` when a cross-experiment
comparison is intentional.

Non-strict mode records ingestion warnings and may retain runs without metrics
as unranked. Legacy runs with missing fingerprints or metric contracts produce
warnings; `--strict` rejects them. Metric definitions are read from saved run
contracts. Reports also flag mixed software versions and dirty source builds.

See [report outputs](output-artifacts.md#report-artifacts) for the resulting
tables, manifest, narrative, and figures.
