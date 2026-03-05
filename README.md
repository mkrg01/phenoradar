# phenoradar

[![CI](https://github.com/mkrg01/phenoradar/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mkrg01/phenoradar/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/phenoradar.svg)](https://pypi.org/project/phenoradar/) [![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/github/license/mkrg01/phenoradar.svg)](LICENSE)

> [!WARNING]
> This repository is under active development and may change without notice.

## Overview

PhenoRadar is a CLI tool for binary phenotype prediction from orthogroup-level TPM.

## Install

```bash
pip install phenoradar
```

## Fastest Run

Fetch compact test data, materialize default config, then run:

```bash
phenoradar dataset
phenoradar config
phenoradar run -c config.yml
```

## Documentation (Recommended Order)

1. [docs/quickstart.md](docs/quickstart.md): first successful run, full run, and prediction.
2. [docs/data-format.md](docs/data-format.md): required metadata/expression TSV schema.
3. [docs/configuration.md](docs/configuration.md): config behavior, defaults, and key-by-key definitions.
4. [docs/cli-reference.md](docs/cli-reference.md): all commands and options.
5. [docs/output-artifacts.md](docs/output-artifacts.md): output files, column-level details, and interpretation guidance.
6. [docs/pipeline-details.md](docs/pipeline-details.md): internal execution behavior.
7. [docs/release-automation.md](docs/release-automation.md): release workflow and PyPI publishing.
