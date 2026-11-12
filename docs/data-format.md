# Data format

Prepare inputs in the separate `phenoradar_prep` repository. PhenoRadar reads
prepared metadata and expression TSVs for training, evaluation, and prediction.
Sequence inputs are described in [amino-acid and combined modes](amino-acid.md#inputs).

## Metadata TSV

Set `data.metadata_path` to a table with one row per species:

```tsv
species	C4	contrast_pair_id	family
sp1	1	pair1	Poaceae
sp2	0	pair1	Poaceae
sp3	1		Cyperaceae
sp4			Poaceae
```

| Column (default) | Required for `run` | Meaning |
| --- | --- | --- |
| `species` | Yes | Non-empty, unique species ID |
| `C4` | Yes | Binary trait: `0`, `1`, empty, or null |
| `contrast_pair_id` | When selected for grouping or pair-aware scoring | Contrast pair identifier |
| `expanded_pair_clade_id` | When selected for grouping or pair-aware scoring | Expanded contrast-pair clade identifier |
| `family` | No | Readable annotation for grouped summaries and plots |

`split.group_col` must be specified and name an existing column.
`data.contrast_pair_col` controls pair-aware features and may be `null` when
contrast pairs are unused.

Pool assignment follows these rules, in order:

| Species | Pool |
| --- | --- |
| Flagged by `split.exclude_col` | Removed from all pools |
| Missing trait | `discovery_inference`; predicted in `full_run` |
| Labeled, missing group in a contrast-pair split | `external_test` |
| Labeled, in a single-label group with `require_both_labels_per_group: true` | `external_test` |
| Other eligible labeled species | Training and CV |

For splits on other columns, such as `family`, labeled species must have a group
unless excluded. Missing contrast pairs do not prevent training in these splits.
The optional exclusion column accepts `yes/no`, `true/false`, `1/0`, empty, or
null; empty/null means false.

Every training fold must contain both labels. `group_balanced` sampling also
requires both labels in every training group. With `all_samples`, validation
groups may contain a single label. To reserve single-label families for external
testing:

```yaml
split:
  group_col: family
  require_both_labels_per_group: true
```

These reserved species enter neither CV nor final-refit training. Eligibility
uses labeled, non-excluded species only. Unknown-trait species remain inference
targets, including those with missing groups.

`summary.group_col` defaults to `family` and uses its values as display labels.
If the column is absent, grouped summaries are skipped with a warning.

### Metadata for prediction

For expression `predict`, metadata is optional. Without it, all distinct species
in the TPM file become targets; empty species IDs are errors. With metadata,
only `species` is required: it selects targets and can supply annotations.
Trait and contrast-pair columns are unnecessary, including for prediction trees.
Sequence bundles have [additional target restrictions](amino-acid.md#prediction).

## Expression TSV

Set `data.tpm_path` to a long-format TSV, either uncompressed (`.tsv`) or
gzip-compressed (`.tsv.gz`). Compression is detected automatically; no separate
option or manual extraction is needed. Training, combined mode, and prediction
accept both formats, including prediction via `--tpm-path`.

```yaml
data:
  tpm_path: data/tpm.tsv.gz
```

The decompressed table has this format:

```tsv
species	orthogroup	tpm
sp1	OG1	1.0
sp1	OG2	0.5
sp2	OG1	2.0
sp2	OG2	0.3
```

| Column (default) | Meaning |
| --- | --- |
| `species` | Species ID matching metadata |
| `orthogroup` | Non-empty feature ID |
| `tpm` | Present, numeric, finite, non-negative expression value |

All non-excluded metadata species must occur in expression data. Consumed rows
are validated after command-specific species/feature selection; invalid values
are reported with source-line examples and never treated as zero. Duplicate
`(species, orthogroup)` rows are validated individually, then summed.

An absent coordinate is zero by default. To keep absent coordinates missing,
use `preprocess.expression.absent_feature_fill: nan` with random forest or
[neutral logistic regression](configuration.md#missing-expression-and-abstention).
Explicit blank/`NA` TPM values remain invalid. Neutral mode can also treat
numeric zeros as unknown with `zero_as_missing: true`, after duplicate summation.

Default zero filling cannot distinguish an unmeasured coordinate from a measured
zero. Validate measurement coverage upstream when this distinction matters.

## Orthogroup annotations

`data.orthogroup_annotation_path` optionally accepts a **headerless** TSV or
TSV.GZ with three columns: orthogroup ID, annotation taxid, annotation text.
Feature figures and tree heatmaps use these annotations as labels.

## Tree Newick

Set `data.tree_path` to enable tree prediction annotation TSVs and Toytree SVGs.
Tip labels must match species IDs. Annotation TSVs retain predicted species
missing from the tree; figures include only matching tips.

For optional unknown-trait imputation, include both targets and known-trait
references and follow the [rooting and branch-length rules](phylogenetic-imputation.md#trees-and-reference-traits).

## Custom column names

Map different headers in the config:

```yaml
data:
  metadata_path: data/metadata.tsv
  tpm_path: data/expression.tsv
  species_col: taxon_id
  feature_col: og_id
  value_col: abundance
  trait_col: phenotype
  contrast_pair_col: contrast_id
split:
  group_col: contrast_id
  exclude_col: final_exclude
```
