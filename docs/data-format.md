# Data Format

PhenoRadar expects tab-separated files (`.tsv`) for metadata and expression.

## Metadata TSV

Default path/key:

- `data.metadata_path` (default: `testdata/c4_tiny/species_metadata.tsv`)

Required columns (default names):

- `species` (`data.species_col`)
- `C4` (`data.trait_col`)
- `contrast_pair_id` (`data.group_col`)

Rules:

- `species` must be non-empty and unique.
- `trait` values must be `0`, `1`, empty, or null.
- `group` can be empty/null.

Pool assignment is derived from `(trait, group)`:

- `trait` present and `group` present -> `training_validation`
- `trait` present and `group` missing -> `external_test`
- `trait` missing -> `discovery_inference`

Training preflight requirement:

- each training group must include both labels (`0` and `1`)

Example:

```tsv
species	C4	contrast_pair_id
sp1	1	g1
sp2	0	g1
sp3	1	g2
sp4	0	g2
sp5	1
sp6
```

## Expression TSV (long format)

Default path/key:

- `data.tpm_path` (default: `testdata/c4_tiny/tpm.tsv`)

Required columns (default names):

- `species` (`data.species_col`)
- `orthogroup` (`data.feature_col`)
- `tpm` (`data.value_col`)

Rules:

- all metadata species must exist in expression data
- `tpm` values must be non-negative before log transform
- duplicate `(species, feature)` rows are summed

Example:

```tsv
species	orthogroup	tpm
sp1	OG1	1.0
sp1	OG2	0.5
sp2	OG1	2.0
sp2	OG2	0.3
```

## Custom column names

If your files use different headers, map them in config:

```yaml
data:
  metadata_path: data/metadata.tsv
  tpm_path: data/expression.tsv
  species_col: taxon_id
  feature_col: og_id
  value_col: abundance
  trait_col: phenotype
  group_col: contrast_id
```
