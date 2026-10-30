# phenoradar

[![CI](https://github.com/mkrg01/phenoradar/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mkrg01/phenoradar/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/phenoradar.svg)](https://pypi.org/project/phenoradar/) [![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/github/license/mkrg01/phenoradar.svg)](LICENSE)

> [!WARNING]
> This repository is under active development and may change without notice.

PhenoRadar predicts binary phenotypes from orthogroup-level TPM using grouped
cross-validation and reusable model bundles.
Prepare input data in the separate `phenoradar_prep` repository, then supply the
metadata and expression TSVs to PhenoRadar.

For prediction from orthogroup amino-acid alignments, alone or jointly with
expression, see [amino-acid and combined modes](docs/amino-acid.md).

## Quickstart

Requires Python 3.12+.

```bash
pip install phenoradar
phenoradar dataset
phenoradar config --out demo.yml
phenoradar run -c demo.yml
```

This runs CV on the bundled example data and saves results under `runs/`.

## Documentation

- [Quickstart](docs/quickstart.md): own data, model export, and prediction.
- [Configuration](docs/configuration.md): settings, defaults, and condition comparisons.
- [Outputs and interpretation](docs/output-artifacts.md): results and how to read them.
- [All guides](docs/README.md): input formats, CLI, methods, and optional features.
