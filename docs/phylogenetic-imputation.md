# Phylogenetic imputation of unknown traits

PhenoRadar can compare expression-based predictions for unknown species with
binary-trait imputation from `nwkit asr`. This is optional interpretation after
prediction: observed labels, expression predictions, model training, CV, and
external-test evaluation are unchanged. It does not classify species into known
or novel evolutionary lineages, and it does not train on imputed labels.

## Installation and configuration

Install the optional tree reader and a compatible `nwkit` executable:

```bash
pip install 'phenoradar[phylogeny]'
pip install 'nwkit @ git+https://github.com/kfuku52/nwkit.git@76b7a1725072cb51a58c8ad43182fd5529d1b39f'
nwkit --version
nwkit asr --help
```

The pinned source above is nwkit 0.43.21, used by the integration tests. A
compatible newer installation can also be used. `nwkit` can live in a separate
environment if its executable is on `PATH`; ETE4 must be available in the
PhenoRadar environment. A nwkit installation alongside PhenoRadar is also found
when that environment's scripts directory is not on `PATH`.

For a checkout, install the reader with `uv sync --extra dev --extra phylogeny`.
An isolated nwkit environment avoids changing PhenoRadar's locked dependencies.

```yaml
data:
  tree_path: species_tree.nwk

phylogenetic_imputation:
  enabled: true
  branch_length_mode: unit
  model: ER
  root_prior: equal
```

The defaults are `enabled: false`, `branch_length_mode: input`, `model: ER`,
and `root_prior: equal`. `model` also accepts `ARD`; `root_prior` also accepts
`empirical`. The same section is accepted by `run` and `predict`.

Only `full_run` and `predict` execute imputation, and only when they predict at
least one species without an observed trait. `cv_only` never executes it.

## Trees and reference traits

Supply a rooted Newick tree containing both prediction targets and known-trait
reference species. The supplied root is retained, including multifurcating
roots. An explicit `[&U]` unrooted marker is rejected. Polytomies are preserved.

- `input`: use every supplied non-root branch length. Missing, negative, and
  non-finite lengths are errors; zero lengths are retained. Root stem length
  does not participate in inference.
- `unit`: replace every non-root branch length with 1 in an analysis copy,
  regardless of whether the input has lengths. The original file is unchanged.

`unit` is useful for taxonomy trees with absent or arbitrary lengths. Its
probabilities are conditional on equal branch lengths; its distances describe
branch counts, not time. Fine-grained taxonomic subdivisions can affect results.
Neither mode integrates uncertainty in topology or branch lengths.

Known binary traits come from the existing metadata, using `data.species_col`
and `data.trait_col`. No additional reference table is required. All observed,
non-excluded species can provide reference traits, including species assigned to
external testing. These labels are used only for interpreting unknown species;
there is no feedback into model fitting or performance evaluation. Rows selected
by `split.exclude_col` are excluded from the saved reference as well.

Every new `full_run` model bundle saves these known traits as an optional,
integrity-verified `observed_traits.parquet`, even if imputation is disabled.
`predict` merges this snapshot with observed traits in its optional metadata;
conflicting labels for the same species are errors. The bundled trait name is
used for metadata lookup unless `data.trait_col` is explicitly specified.
Species with a known trait in either source are not discrepancy candidates.

Older bundles still load. If neither the bundle nor prediction metadata provides
known traits, only imputation is skipped, with unavailable results and a recorded
reason. No prior-only probabilities are substituted. Missing targets in the tree
are retained in comparison tables with `NA` probabilities.

## Outputs

All outputs are confined to `inference/`:

| File | Content |
| --- | --- |
| `tables/phylogenetic_imputation.tsv` | Prediction targets and reference species, with observed traits and ASR probabilities |
| `tables/phylogenetic_comparison.tsv` | Unknown prediction targets, sorted by descending absolute probability difference |
| `tables/phylogenetic_positive_candidates.tsv` | Accepted expression-positive unknown targets with a positive difference, sorted by descending signed difference |
| `figures/tree_phylogenetic_imputation.svg` | Tree, ancestral probabilities, observed traits, expression probabilities, phylogenetic probabilities, differences, and statuses |
| `figures/phylogenetic_comparison.svg` | Unknown-species probability comparison, with up to ten largest-disagreement species labeled |
| `phylogenetic_imputation/reference_traits.tsv` | Snapshot of all supplied known traits |
| `phylogenetic_imputation/traits.tsv` | Known traits that matched the analysis tree, passed to nwkit |
| `phylogenetic_imputation/tree.nwk` | Analysis tree after the selected branch-length treatment |
| `phylogenetic_imputation/asr.tsv` | All-node state probabilities from nwkit |
| `phylogenetic_imputation/model.tsv` | Fitted nwkit model parameters and diagnostics |
| `phylogenetic_imputation/annotated_tree.nhx` | nwkit tree with ancestral and tip-state probability annotations |
| `phylogenetic_imputation/metadata.json` | Settings, reference scope/counts, tree/reference hashes, nwkit version/command, status and warnings |
| `phylogenetic_imputation/nwkit.stderr.txt`, `nwkit.stdout.txt` | External-command diagnostics |

Files requiring a tree or successful fit are absent when that step is skipped
or fails. Nothing is written when the feature is disabled or no unknown targets
exist. An incompatible/missing nwkit executable or invalid input is an error.
A failed numerical fit leaves comparison probabilities unavailable, records the
failure, and retains the expression predictions.

The annotation and comparison tables include:

- `observed_trait`: the actual known 0/1 state, or `NA`; never overwritten by ASR.
- `phylo_prob`: ASR probability of state 1; observed tips are conditioned on
  their supplied states, so their values are not held-out predictions.
- `phylo_status`: `observed`, `imputed`, `missing_tree_tip`, `missing_reference`,
  or `fit_failed`.
- `phylo_is_imputed`: true only for successfully imputed unknown tips.
- `is_prediction_target`: distinguishes targets from reference-only species.
- `prob`: the original expression probability, when the species was predicted.
- `prob_difference`: expression probability minus phylogenetic probability,
  computed only for successfully imputed unknown prediction targets.
- `abs_prob_difference`: absolute value of that difference.
- Available expression decision and abstention columns.

The discrepancy tables retain abstained species for inspection, but the positive
candidate subset uses the existing selective decision. Candidate evidence TSVs
and PDFs also include phylogenetic probability and signed difference when
available. Existing prediction TSVs and their `true_label` columns are unchanged.

The tree includes observed references around unknown targets. Any display
pruning happens after fitting, preserves branch lengths, and uses the original
node IDs for ancestral probabilities. Gray cells mean missing or unavailable
values; the observed-trait column remains gray for imputed species. Expression
targets are shown in bold. Internal-node color represents the ASR probability of
state 1. Neither differences nor their rankings are p-values or probabilities of
a novel evolutionary origin.

## Integration checks

To include checks against an actual nwkit installation:

```bash
PHENORADAR_TEST_NWKIT=/path/to/nwkit uv run --extra phylogeny pytest -q
```

The tests cover polytomies, explicit/missing lengths, known-label references,
disagreement ranking, abstention, bundle reuse, and unchanged expression-model
predictions and evaluation. They do not add phylogenetic cross-validation to the
application pipeline.
