# Data Format

PhenoRadar expects tab-separated files (`.tsv`) for metadata and expression.

## Species Trait TSV

`phenoradar metadata` can fetch an NCBI taxonomy constrained Newick tree from a raw species
trait table and generate `species_metadata.tsv` with `contrast_pair_id` assignments.

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

The tree-generation step requires `nwkit`. For a local uv environment, install the recorded
dependency group with:

```bash
uv sync --group taxonomy
```

For conda-based environments, install `nwkit` from Bioconda. For pip-only environments,
install it directly from the upstream repository.

Group assignment uses `nwkit skim` contrastive clades. Species that are not present in the
tree are excluded from the generated `species_metadata.tsv`. Species with known traits inside
minimal clades containing both `0` and `1` receive the same `contrast_pair_id`. Known-trait
species present in the tree but outside contrastive clades keep an empty group and therefore
become `external_test` under the normal pool assignment rules.

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
- `contrast_pair_id` (`data.group_col`)

Rules:

- `species` must be non-empty and unique.
- `trait` values must be `0`, `1`, empty, or null.
- `group` can be empty/null.

Pool assignment is derived from `(trait, group)`:

- `trait` present and `group` present -> `training_validation`
- `trait` present and `group` missing -> `external_test`
- `trait` missing -> `discovery_inference`

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
