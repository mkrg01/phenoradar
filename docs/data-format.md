# Data Format

PhenoRadar expects tab-separated files (`.tsv`) for metadata and expression.

Input preparation, including metadata and tree generation, belongs in the separate
`phenoradar_prep` repository. PhenoRadar reads prepared inputs for training,
evaluation, and prediction.

## Metadata TSV

Default path/key:

- `data.metadata_path` (default: `testdata/c4_tiny/species_metadata.tsv`)

Required columns (default names):

- `species` (`data.species_col`)
- `C4` (`data.trait_col`)
- `contrast_pair_id` (`split.group_col`; also `data.contrast_pair_col` by default)

Optional annotation columns:

- `family` (`summary.group_col` by default): taxonomic names such as `Poaceae`.
- `order` or other grouping columns, when needed for summaries or splitting.

The selected `summary.group_col` supplies both the grouping value and display
label. No separate ID or name column is required. Group summaries and figures
are skipped with a warning if the selected column is absent.

Example:

```tsv
species	C4	contrast_pair_id	family
sp1	1	pair1	Poaceae
sp2	0	pair1	Poaceae
```

```yaml
summary:
  group_col: family
```

Rules:

- `species` must be non-empty and unique.
- `trait` values must be `0`, `1`, empty, or null.
- The `split.group_col` column is required. When it matches the configured,
  non-null `data.contrast_pair_col`, labeled species with empty/null group
  values automatically enter `external_test`.
- For other split groups, such as `family`, labeled species must have a
  non-empty group value unless explicitly excluded. Missing contrast pairs
  do not prevent these species from entering CV.
- `split.exclude_col`, when configured, accepts `yes/no`, `true/false`, `1/0`,
  empty, or null values and removes matching species from all pools.
  Empty/null values are treated as false.
- `data.contrast_pair_col` can be set to `null` for non-contrast-pair
  workflows. `preprocess.ranked_feature_filter.method=pair_aware` then cannot
  be used.

For family-level CV, the same `family` names can also be selected as the
split groups. To route families with only one observed trait label to
`external_test` automatically:

```yaml
split:
  group_col: family
  require_both_labels_per_group: true
```

With the configuration above, families with both labels enter CV and
single-label families become external-test groups. The check uses only labeled,
non-excluded species. Reserved families stay out of CV and final-refit training.

With the default `require_both_labels_per_group: false` and
`sampling.strategy: all_samples`, single-label families also enter CV.
`logo` holds out each family in turn, so its validation fold may contain
only `0` or only `1`. Every fold's training side must still contain both
labels. `group_balanced` sampling requires both labels in every training
family; use `all_samples` to allow single-label families in CV.

Species with empty/null traits are always inference targets, regardless of
family membership or missing family values. They are excluded from model
training, CV, external-test evaluation, and group-label eligibility checks.
They are predicted in `full_run`; `cv_only` records their pool assignment
without predicting them. `split.exclude_col` removes flagged species from
all pools, including inference.

## Orthogroup annotations

Optional path/key:

- `data.orthogroup_annotation_path` (default: `null`)

`data.orthogroup_annotation_path` points to a headerless TSV or TSV.GZ with
three columns: orthogroup ID, annotation taxid, and annotation text. When set,
PhenoRadar uses annotation labels in feature interpretation figures and tree
feature heatmaps instead of ID-only labels.

## Tree Newick

Optional path/key:

- `data.tree_path` (default: `null`)

When `data.tree_path` is set, `phenoradar run` and `phenoradar predict` write
ggtree-friendly tree prediction annotation TSV files under the relevant stage
`tables/` directory. Toytree SVG figures are also written under the matching
stage `figures/` directory when Toytree is available.

Tree tip labels must match metadata and prediction `species` values. In CV runs, tree
contrast-pair QC and prediction artifacts focus on species with non-empty
`contrast_pair_id`. In external-test and predict outputs, all predicted species are included
in the annotation TSV. Tree feature heatmaps use the top `figures.top_features`
features by fold-level `importance_mean` and write both `log2(TPM + 1)` and
within-feature z-score values.

Training preflight requirements:

- the full `training_validation` pool must include both labels (`0` and `1`)
- with `sampling.strategy: group_balanced`, each `split.group_col` group must
  include both labels before CV
- with `preprocess.ranked_feature_filter.method: pair_aware`, valid contrast pairs used
  for feature scoring must include both labels; species without a valid
  `data.contrast_pair_col` value remain eligible for model training when the
  split metadata assigns them to the training pool

Example:

```tsv
species	C4	contrast_pair_id	family
sp1	1	g1	Family A
sp2	0	g1	Family A
sp3	1	g2	Family B
sp4	0	g2	Family B
sp5	1		Family C
sp6			Family D
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
- every expression row actually consumed after a command's species/feature selection must have a
  non-empty `orthogroup`; its `tpm` must be present, numeric, finite, and non-negative
- invalid consumed rows are reported with source-line examples; invalid values are never
  interpreted as zero
- duplicate `(species, feature)` rows are summed after every contributing value is validated
- an absent `(species, feature)` coordinate is represented as zero by default
- set `preprocess.absent_feature_fill: nan` with `model.name: random_forest`
  or neutral logistic regression (`preprocess.missing_expression.method: neutral`) to
  represent absent coordinates as missing values instead; explicit TPM values
  must still be finite and cannot be written as `NA`
- in the default `0` mode, the long format cannot distinguish a truly
  measured zero from an unmeasured coordinate; encode or validate measurement
  coverage upstream when that distinction matters
- neutral mode can additionally treat explicit zero as unknown with
  `preprocess.missing_expression.zero_as_missing: true`. The original TPM is not
  overwritten. Positive values remain observed; explicit `NA`/blank TPM is still
  invalid. Duplicate coordinates are summed before zero masking.

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
  contrast_pair_col: contrast_id
split:
  group_col: contrast_id
  exclude_col: final_exclude
```
