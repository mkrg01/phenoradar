# Data Format

PhenoRadar expects tab-separated files (`.tsv`) for metadata and expression.

## Species Trait TSV

`phenoradar metadata` can fetch an NCBI taxonomy constrained Newick tree from a raw species
trait table and generate `species_metadata.tsv` with `contrast_pair_id` assignments and
optional taxonomic-rank split blocks.

Required columns (default names):

- `species` (`--species-col`)
- `C4` (`--trait-col`)

Example:

```tsv
species	C4
sp1	1
sp2	0
```

Generate a tree and metadata:

```bash
phenoradar metadata \
  --species-trait species_trait.tsv \
  --species-taxid species_taxid.tsv \
  --tree-out ncbi_tree.nwk \
  --out species_metadata.tsv
```

The tree-generation step requires `nwkit`. Taxonomic-rank blocks also use `ete4.NCBITaxa`.
For a local uv environment, install the recorded dependency group with:

```bash
uv sync --group taxonomy
```

For conda-based environments, install `nwkit` from Bioconda and `ete4` from PyPI or conda.
For pip-only environments, install `nwkit` directly from the upstream repository.

Group assignment uses `nwkit skim` contrastive clades. Species that are not present in the
tree are excluded from the generated `species_metadata.tsv`. Species with known traits inside
minimal clades containing both `0` and `1` receive the same `contrast_pair_id`. Known-trait
species present in the tree but outside contrastive clades keep an empty contrast-pair value
and are marked `contrast_pair_test_holdout=yes` by default.

Taxonomic rank blocking can be requested with repeated `--taxon-block-rank` options, for
example `--taxon-block-rank family --taxon-block-rank order`. This requires
`--species-taxid`. For each requested rank, metadata includes:

- `taxon_<rank>_id`
- `taxon_<rank>_name`
- `taxon_<rank>_test_holdout`
- `taxon_<rank>_exclude`

Rank blocks with both trait labels enter CV. Single-label rank blocks become
`*_test_holdout=yes`. Labeled species with missing taxid or missing requested rank become
`*_exclude=yes`, so they are not used for CV or external test.

Use a generated rank block by selecting its columns in config:

```yaml
split:
  group_col: taxon_family_id
  test_holdout_col: taxon_family_test_holdout
  exclude_col: taxon_family_exclude
```

## Species Taxid TSV

When known NCBI Taxonomy IDs are available, `phenoradar metadata` can use them for tree
retrieval instead of inferring taxids from species names.

Required columns (default names):

- `species` (`--species-col`)
- `taxid` (`--taxid-col`)

Example:

```tsv
species	taxid
Zea_mays	4577
Oryza_sativa	4530
```

Use it with:

```bash
phenoradar metadata \
  --species-trait species_trait.tsv \
  --species-taxid species_taxid.tsv \
  --tree-out ncbi_tree.nwk \
  --out species_metadata.tsv
```

## Metadata TSV

Default path/key:

- `data.metadata_path` (default: `testdata/c4_tiny/species_metadata.tsv`)

Required columns (default names):

- `species` (`data.species_col`)
- `C4` (`data.trait_col`)
- `contrast_pair_id` (`split.group_col`; also `data.contrast_pair_col` by default)
- `contrast_pair_test_holdout` (`split.test_holdout_col`)

Rules:

- `species` must be non-empty and unique.
- `trait` values must be `0`, `1`, empty, or null.
- `split.group_col` is required for labeled species unless
  `split.test_holdout_col` marks the species as a test holdout.
- `split.test_holdout_col` accepts `yes/no`, `true/false`, `1/0`, empty, or
  null values. Empty/null values are treated as false.
- `split.exclude_col`, when configured, accepts the same boolean values and
  removes matching species from CV, external test, and inference.
- `data.contrast_pair_col` can be set to `null` for non-contrast-pair
  workflows. Contrast-pair-specific features such as `pair_aware_filter` then
  cannot be used.

Pool assignment is derived from `(trait, split.group_col, split.test_holdout_col,
split.exclude_col)`:

- `exclude` true -> removed from all pools
- `trait` present and test holdout true -> `external_test`
- `trait` present, test holdout false, and split group present -> `training_validation`
- `trait` missing -> `discovery_inference`
- `trait` present, test holdout false, and split group missing -> error

## Tree Newick

Optional path/key:

- `data.tree_path` (default: `null`)

When `data.tree_path` is set, `phenoradar run` and `phenoradar predict` write
ggtree-friendly tree prediction annotation TSV files. If the optional tree visualization
dependencies are installed with `pip install "phenoradar[tree]"`, Toytree SVG figures are
also written under `figures/`.

Tree tip labels must match metadata and prediction `species` values. In CV runs, tree
contrast-pair QC and prediction artifacts focus on species with non-empty
`contrast_pair_id`. In external-test and predict outputs, all predicted species are included
in the annotation TSV. Tree feature heatmaps use the top 30 features by fold-level
`importance_mean`
and write both `log2(TPM + 1)` and within-feature z-score values.

Training preflight requirements:

- the full `training_validation` pool must include both labels (`0` and `1`)
- with `sampling.strategy: group_balanced`, each `split.group_col` group must
  include both labels before CV
- with `preprocess.pair_aware_filter.enabled: true`, each contrast pair must
  include both labels before CV

Example:

```tsv
species	C4	contrast_pair_id	contrast_pair_test_holdout
sp1	1	g1	no
sp2	0	g1	no
sp3	1	g2	no
sp4	0	g2	no
sp5	1		yes
sp6			no
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
  contrast_pair_col: contrast_id
split:
  group_col: contrast_id
  test_holdout_col: final_test_holdout
  exclude_col: final_exclude
```
