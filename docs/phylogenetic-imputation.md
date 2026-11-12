# Phylogenetic imputation

Compare expression predictions for unknown-trait species with binary-trait
imputation from `nwkit asr`. This optional interpretation runs after prediction;
observed labels, model training, CV, and external-test evaluation remain unchanged.
Imputed labels are never used for training.

## Installation and configuration

Install the optional reader and the nwkit version used by integration tests:

```bash
pip install 'phenoradar[phylogeny]'
pip install 'nwkit @ git+https://github.com/kfuku52/nwkit.git@76b7a1725072cb51a58c8ad43182fd5529d1b39f'
nwkit --version
nwkit asr --help
```

The pin is nwkit 0.43.21. A compatible nwkit may run in another environment if
its executable is on `PATH`; ETE4 must be in the PhenoRadar environment. A nwkit
executable alongside PhenoRadar is also detected. For a checkout, use
`uv sync --extra dev --extra phylogeny` to install the reader.

```yaml
data:
  tree_path: species_tree.nwk
phylogenetic_imputation:
  enabled: true
  branch_length_mode: unit
  model: ER
  root_prior: equal
```

| Setting | Default | Options |
| --- | --- | --- |
| `enabled` | `true` | Set to `false` to skip interpretation in expression `full_run` and `predict` |
| `branch_length_mode` | `input` | `input` or `unit` |
| `model` | `ER` | `ER` or `ARD`, with rates fitted by nwkit |
| `root_prior` | `equal` | `equal` or `empirical` observed-state frequencies |

Imputation runs only for unknown-trait targets in expression `full_run` and
`predict`.

## Trees and reference traits

Supply a rooted Newick tree containing targets and known-trait references.
The supplied root and polytomies are preserved; an explicit `[&U]` marker is
rejected.

- `input` uses supplied non-root lengths. Missing lengths skip imputation.
  Negative or non-finite lengths are errors. Zero is valid, and the root stem
  length is ignored.
- `unit` assigns every non-root length 1 in an analysis copy. This suits taxonomy
  trees with absent/arbitrary lengths; distances are branch counts, not time.

Results are conditional on the tree and branch treatment; neither mode integrates
topology/length uncertainty. Taxonomic subdivisions can affect unit-length results.

Known traits come from metadata using `data.species_col` and `data.trait_col`.
All observed, non-excluded species can serve as interpretation references,
including external-test species. No extra reference table is needed.

New expression `full_run` bundles save an integrity-verified
`observed_traits.parquet` even when imputation is disabled. `predict` merges it
with optional metadata traits; conflicting labels are errors. The saved trait
name is used unless `data.trait_col` is explicitly supplied. Species with known
traits in either source are excluded from discrepancy candidates.

Older bundles remain usable. With no known references, imputation is skipped
with a recorded reason and unavailable probabilities; no prior-only values are
substituted. Targets missing from the tree remain in comparison tables with `NA`.

## Outputs

All paths are under `inference/`:

| File | Content |
| --- | --- |
| `tables/phylogenetic_imputation.tsv` | Targets and references, observed traits, ASR probabilities, and status |
| `tables/phylogenetic_comparison.tsv` | Unknown targets ordered by absolute probability difference |
| `tables/phylogenetic_positive_candidates.tsv` | Accepted positive unknowns with `phylo_prob < 0.5` and positive signed difference |
| `figures/tree_phylogenetic_imputation.svg` | Tree, ancestral probabilities, traits, predictions, differences, and statuses |
| `figures/phylogenetic_comparison.svg` | Probability comparison with the largest disagreements labeled |
| `phylogenetic_imputation/{reference_traits,traits}.tsv` | All supplied references and the tree-matched nwkit input |
| `phylogenetic_imputation/{tree.nwk,annotated_tree.nhx}` | Analysis tree and nwkit annotations |
| `phylogenetic_imputation/{asr,model}.tsv` | Node probabilities and fitted-model diagnostics |
| `phylogenetic_imputation/metadata.json` | Settings, reference scope, hashes, command/version, status, and warnings |
| `phylogenetic_imputation/nwkit.{stderr,stdout}.txt` | External-command diagnostics |

Disabled imputation or no unknown targets produces no files. When required tree
data, ETE4, or nwkit are unavailable, PhenoRadar records a warning and completes
the prediction run. Incompatible nwkit or invalid input is an error. A numerical
fit failure retains expression predictions, records the failure, and leaves
phylogenetic probabilities unavailable. Fit-dependent files may be absent.

| Column | Meaning |
| --- | --- |
| `observed_trait` | Actual known 0/1 state or `NA`; never overwritten |
| `phylo_prob` | ASR probability of state 1; known tips are conditioned on observed states |
| `phylo_status` | `observed`, `imputed`, `missing_tree_tip`, `missing_reference`, or `fit_failed` |
| `phylo_is_imputed` | True only for successfully imputed unknown tips |
| `is_prediction_target` | Distinguishes targets from reference-only species |
| `prob` | Original expression probability |
| `prob_difference` | Expression minus phylogenetic probability, for imputed unknown targets |
| `abs_prob_difference` | Absolute difference |

Comparison tables retain abstained species. Positive candidates require an
accepted positive decision and phylogenetic probability strictly below 0.5;
equal-to-0.5 or unavailable values are excluded. Candidate evidence TSVs include
available phylogenetic values; evidence PDFs leave them to the dedicated plots.

Tree figures retain observed references around targets. Gray means unavailable;
unknown observed-trait cells stay gray, targets are bold, and internal-node color
shows ASR state-1 probability. Differences and rankings are neither p-values nor
probabilities of a novel evolutionary origin. Existing prediction tables and
`true_label` values are unchanged.

For integration checks with a real executable:

```bash
PHENORADAR_TEST_NWKIT=/path/to/nwkit uv run --extra phylogeny pytest -q
```
