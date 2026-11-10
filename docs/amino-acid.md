# Amino-acid and combined modes

Use orthogroup (OG) amino-acid alignments alone or jointly with expression.
All modes share `config.yml`, grouped CV, sampling, weights, model selection,
and mean/median ensembles. Sequence and combined modes require
`model.name: logistic_elasticnet`.

Training encodes AA blocks as RAM-buffered CSC sparse matrices, spilling to
temporary disk when needed. An exact column mode learned from training rows is
shifted to zero; the unpenalized intercept
is converted back to the original centered basis before saving and reporting.
Missing/novel-residue handling, feature filters, penalties and lambda paths are
unchanged. Different floating-point calculations can change coefficients or
selected sites, especially with weak regularization. No configuration is needed.
Combined models keep the expression preprocessing and its separate penalty.

Grid-search inner folds share a memory-bounded process queue across outer scopes.
Refits use separate short-lived processes; final selection can overlap outer CV.
Scores and reports retain their original fold/candidate order. TPE retains its
sequential candidate dependencies. Source installs build optional
C kernels when a C compiler is available; otherwise the NumPy implementation
is used. Neither optimization needs configuration or changes the lambda path.

## Start a sequence run

Generate a config, then set the sequence inputs:

```bash
phenoradar config --out sequence.yml
```

```yaml
data:
  feature_mode: amino_acid
  metadata_path: data/species_metadata.tsv
  alignment_dir: data/alignments
  sequence_tpm_path: data/sequence_tpm.tsv.gz
preprocess:
  amino_acid:
    aggregation: tpm_weighted_frequency
```

```bash
phenoradar run -c sequence.yml
```

Expression `data.tpm_path` and `preprocess.expression` settings are inactive in
sequence-only mode. Metadata follows the usual [training format](data-format.md#metadata-tsv).

## Inputs

Each `.faa`, `.fa`, or `.fasta` file directly in `data.alignment_dir` is one OG;
its filename stem is the OG ID. No manifest is needed. Records within each OG
must have unique IDs and equal alignment lengths.

The FASTA ID is the first header token. It must match an exact metadata species
ID or that ID followed by `_` and a gene suffix. Ambiguous matches fail.
Explicit sequence maps are unsupported.

Canonical residues are `ACDEFGHIKLMNPQRSTVWY`; `-` and `.` are gaps. Other
letters and `*` are unknown: preserved for coordinates, excluded from frequency
denominators. Invalid non-sequence characters fail validation. Missing
species/OG combinations remain unknown, not evidence of gene absence.

### Aggregation

| `preprocess.amino_acid.aggregation` | Species-level representation |
| --- | --- |
| `aa_frequency` | Equal-gene canonical-residue frequencies at each site |
| `best_covered` | One gene per species/OG, chosen on training-frozen quality-filtered columns |
| `tpm_weighted_frequency` (default) | Canonical-residue frequencies weighted by gene TPM |

Weighted mode additionally requires:

```yaml
data:
  sequence_tpm_path: data/sequence_tpm.tsv.gz
preprocess:
  amino_acid:
    aggregation: tpm_weighted_frequency
```

The TSV (optionally gzip-compressed) has `species`, `sequence_id`, `tpm`.
Every imported alignment record needs exactly one finite, non-negative TPM;
missing or duplicate matching keys and invalid matching values are errors.
Keys use complete FASTA IDs; extra rows are ignored. Other aggregation modes
ignore this input.

At each site, residue weights are divided by total TPM of genes with a canonical
residue. Gaps, unknowns, and zero-weight genes contribute nothing. A zero
denominator is missing, with no unweighted fallback. Scaling all TPMs within a
species/OG leaves the frequencies unchanged; total OG expression requires
[combined mode](#joint-expression-and-amino-acid-mode).

An aggregation list compares modes on the same alignments and split. Weighted
conditions retain the additional TPM hash, which must agree across those conditions.

## Preprocessing

Every sampled outer/inner training set and final refit learns its own site set,
representatives, major residues, scores, and means. Validation and inference rows
never determine them.

| Setting | Default | Meaning |
| --- | --- | --- |
| `preprocess.amino_acid.site_filter.min_sample_fraction` | `0.8` | Fraction in `(0, 1]` of all training species callable at a site |
| `preprocess.amino_acid.site_filter.min_residue_states` | `2` | Required distinct canonical states, from 2 to 20 |
| `preprocess.amino_acid.ranked_site_filter.method` | `none` | `none`, `pair_aware`, or `unpaired` |
| `preprocess.amino_acid.ranked_site_filter.max_sites` | `2000` | Retained-site limit for ranking; ignored with `none` |
| `preprocess.amino_acid.ranked_site_filter.min_contrast_pairs` | `2` | Positive valid-pair minimum for pair-aware ranking |

Processing order:

1. Screen callability using all training species, including those without the OG.
   Weighted mode requires a positive-TPM canonical residue.
2. Compute frequencies and screen residue variation.
3. For `best_covered`, select the gene with most canonical residues on the
   quality-filtered columns; ties use sequence ID, and zero coverage selects
   none. Freeze these scoring columns, then recheck coverage/variation on the
   representatives without selecting genes again.
4. Optionally rank sites. `pair_aware` takes the norm of equally averaged
   within-pair phenotype-frequency contrasts and requires enough valid pairs
   per site. `unpaired` uses phenotype mean-composition differences. Neither
   imposes a direction. Ties use OG ID and input column order. With `none`,
   neither ranking nor the `max_sites` limit applies: all quality-passing sites
   are retained.
5. Freeze model sites, center states on training means, and drop constant state
   columns. Missing or previously unseen state mass receives the mean and has
   zero contribution; known mass in a partly novel composition retains its effect.

Sequence features are centered without variance scaling. `best_covered` uses the
same representatives for ranking and fitting, even after sites are removed.
Validation/prediction selects its genes on the training-frozen scoring columns.
Minor-sample/group counts remain descriptive QC fields, without filtering sites.

## Joint expression and amino-acid mode

Set `data.feature_mode: combined`, supply both `data.tpm_path` for OG expression
and `data.alignment_dir`, and use neutral expression preprocessing. The default
TPM-weighted aggregation also requires `data.sequence_tpm_path`:

```yaml
data:
  feature_mode: combined
  metadata_path: data/species_metadata.tsv
  tpm_path: data/tpm.tsv
  alignment_dir: data/alignments
  sequence_tpm_path: data/sequence_tpm.tsv.gz
preprocess:
  expression:
    absent_feature_fill: nan
    missing_expression:
      method: neutral
    expression_transform:
      method: log1p
    feature_scaling:
      method: standard
model_selection:
  selected_candidate_count: 1
  inner_cv_strategy: logo
  search_space:
    lambda: [0.0001, 0.001, 0.01, 0.1]
    alpha: [1]
    amino_acid_penalty_ratio: [0.1, 0.3, 1, 3, 10]
```

`preprocess.expression` controls expression; `preprocess.amino_acid` controls sequence. All three
sequence aggregations are supported. Both blocks share species, sampling,
weights, and splits, with filters/scalers/representatives fitted separately in
each training fold. OGs need not occur in both modalities. Every training species
must occur in both input types; missing individual OGs/sites remain neutral.

Combined mode fixes `alpha=1` (LASSO). The objective is weighted logistic loss plus:

```text
lambda × (sum(abs(expression coefficients)) + ratio × sum(abs(AA coefficients)))
```

The positive AA/expression penalty ratio defaults to 1; larger values penalize
AA more. Inner CV can select both lambda and ratio. Native penalty normalization
is compensated so retained column counts do not change this definition.
The additive model may select only one modality; it has no expression-by-residue
interactions. Combined mode currently rejects phylogenetic imputation.

## Information coverage

When abstention is enabled, sequence coverage weights site observation fractions
by each site's fitted coefficient range and normalizes within each model.
Frequency-mode observation combines callable-gene fraction and known residue
mass. Weighted mode uses callable OG-TPM fraction times known residue mass,
not a fraction of all transcriptome TPM. Intercept-only members contribute zero.

Combined models use absolute-coefficient expression coverage and site-range AA
coverage. Per-model joint coverage is the minimum over modalities with selected
coefficients; unused modalities impose no requirement. An intercept-only model
has zero coverage. Missing prediction evidence is neutralized and assessed
through this gate.

Ensemble coverage is the arithmetic mean of member coverages. It is an
information-availability measure, not calibrated correctness. Raw CV metrics
still include every OOF species, independently of selective decisions.

## Outputs

Standard split, CV metric, coefficient, feature-stability, and optional bootstrap
tables use the normal stage directories. Sequence-specific tables include:

| Table | Content |
| --- | --- |
| `alignment_inputs.tsv`, `sequence_correspondence.tsv` | Input hashes and sequence/species mapping |
| `sequence_tpm_input.tsv` | Weighted input identity, matched/zero counts, cache path/hit status |
| `amino_acid_og_qc.tsv`, `amino_acid_site_qc.tsv` | Train-only coverage, variation, and scoring diagnostics |
| `feature_schema.tsv`, `representative_sequences.tsv` | State columns and gene choices |
| `representative_scoring_sites.tsv` | Pre-ranking columns frozen for gene selection |
| `amino_acid_site_effects.tsv` | Effects versus training major residue, ranges, and local contribution summaries |
| `amino_acid_site_pss.tsv` | Per-model/site sum of absolute residue coefficients |
| `amino_acid_site_effect_range.tsv` | Per-model/site max–min coefficient among training-observed residues, including zero coefficients |
| `amino_acid_site_stability.tsv`, `amino_acid_og_stability.tsv` | Selected-fold fractions, including fractions conditional on eligibility |
| `amino_acid_local_contributions.tsv` | Leading per-model site effects, observation fractions, and gene counts |
| `selected_site_sequence_map.tsv` | Original residues, sequence checksums, and coordinates; weighted mode adds gene TPM and callable-site weight fractions |

PSS and effect-range scores are written in `cv/tables/`, `model/tables/` for
refits, and `predict/tables/` for bundle predictions. Keys include model scope,
sampled set, candidate, OG, and original alignment position. Scores are computed
per model, normalized by that model's total, and ranked with shared ranks for
ties. Zero scores have blank rank; all-zero models have zero normalized scores.
These describe coefficients, not predictive performance. Site selection,
stability, and coverage use effect range.

Coordinates are one-based: alignment positions refer to the original MSA;
sequence positions count non-gap characters, including `X` and `*`. Gap coordinates
are empty. The OOF frequency heatmap uses held-out-fold preprocessing and displays
states ranked by normalized absolute coefficient importance, not PSS; gray is
unavailable. No separate site-score plots are generated.

Combined mode adds:

| Table | Content |
| --- | --- |
| `combined_coefficients.tsv` | Per-model coefficients in model/original units, standardized comparisons, and penalty factors |
| `expression_*`, `amino_acid_*` CV interpretation tables | Importance normalized within each modality; generic tables combine rows without a shared cross-modality ranking |
| `combined_modality_contributions.tsv` | Expression and AA contributions, intercept, total logit per species/model |
| `expression_local_contributions.tsv` | Leading local OG expression contributions |

Original expression units are `log1p(TPM)`, not raw TPM. AA standard deviations
use observed training species and affect reporting only. Expression filter/ranking
and AA score/representative/coordinate tables are retained. Probabilities are
aggregated after the sigmoid, not by summing probability contributions.

## Prediction

Export with `--execution-stage full_run`, then predict using frozen alignments:

```bash
phenoradar predict --model-bundle runs/RUN_ID/model_bundle \
  --metadata-path data/prediction_species.tsv --alignment-dir data/alignments
```

For a combined bundle, also pass `--tpm-path data/tpm_predict.tsv` or provide
`data.tpm_path` through `-c`. Metadata is optional; without it, sequence/combined
prediction targets the bundle's original species. With it, targets must be a
subset of those species. Required OG alignments must match saved content hashes;
new sequences or alignments require retraining.

Predictions are written to `predict/tables/pred_predict.tsv`. Full-run predictions
use `external_test/tables/pred_external_test.tsv` and
`inference/tables/pred_inference.tsv`.

Weighted bundles include gene TPMs for required OGs and original species. The
source TPM file is unnecessary for reproducing predictions. Override weights with
`--sequence-tpm-path new_tpm.tsv.gz` or `data.sequence_tpm_path` in `-c`; only
target species/required OGs need matching TPMs, and the actual hash is recorded.
Input overrides do not refit preprocessing.

Sequence bundles use `aa-3`; readers also accept `aa-1`/`aa-2` with their original
representative rules. Combined bundles use `combined-1`. New representative
bundles retain pre-ranking scoring masks.

## Caching and memory

Prediction byte caches are temporary per run. Frequencies and centering are
computed in site blocks; large output matrices use temporary disk backing.
Unweighted and TPM-weighted raw residue compositions are cached per requested
OG/column across the species retained for training and reused across runs.
The persistent entries live under `runs/.phenoradar_cache/alignments/frequencies`.
Their identities include alignment content, species/gene correspondence, exact
TPM weights, NumPy version and the aggregation implementation. Only
columns requested after the callability screen are added. Only states present
in those raw compositions occupy disk; zero and missing frequencies are restored
exactly. This changes storage, not site or state selection. These per-species
compositions contain no fitted filter, phenotype score, or centering mean;
each training fold still learns those separately. Representative choices remain
fold-specific. Prediction stores compute only each requested batch and do not
build this training cache. Inner CV and frequency-mode AA-only refits omit
unused observation matrices; audit tables are written in columnar batches.
Matrices use column-major storage for block writes and glmnet input.

Callability and positive stored-frequency support use species bitsets; duplicate
sampling rows retain their multiplicity. Raw frequencies occupy immutable,
state-major file extents. Zero, one and canonical NaN values use bit planes;
other float32 bit patterns are stored exactly. Blocks with little compression
use dense storage. Signed zero and noncanonical NaNs also round-trip exactly.
For 0/1/missing compositions, training statistics and shifted CSC columns are
computed directly from bitsets. Mixed frequencies use the general numerical
path; validation-only states still contribute no known-residue mass.
Completed folds publish only newly computed segments atomically; concurrent
workers can extend the same cache without overwriting existing segments. Fold workers share existing extents and append only
new columns to private files, without copying an OG's earlier data.
Set `TMPDIR` to a location with enough space for temporary caches and matrices.
Validated training alignment bytes and presence masks are reused across runs
under `runs/.phenoradar_cache/alignments`. Source changes, changed species ID
resolution, changed retained species, or changed cache files invalidate entries.
A file-stat index avoids hashing unchanged inputs. Payload identity is based
on content, so moving an unchanged file can reuse it. A changed file identity
requires SHA-256 verification; arbitrary copies are not trusted from timestamps
alone. Cold imports record the hash while parsing. Format changes rebuild the
affected entries automatically. These caches contain no learned filters or means. Source alignments must remain
available; deleting the cache safely forces rebuilding.
Unranked preprocessing keeps fitted site statistics in columnar arrays and
computes quality and moments in the same pass. Filters and means remain
training-only. The default glmnet convergence threshold is `1e-7`.
Inner selection skips unused major-relative QC counts; full outer/final audit
tables retain them. Centering expands only model columns to float64.
Selected models refit along matching stronger lambdas on their own training
data, retaining the selected lambda and modality penalty factors.
CV interpretation is spooled to temporary Parquet parts as each model finishes.
Full feature eligibility, coefficients, PSS, effect ranges, and stability tables
are retained; completed folds do not accumulate Python feature dictionaries or
full fitted transforms in RAM. Fold means and feature eligibility are aggregated
one fold at a time; shared feature statistics are computed once for coefficient
and importance outputs. Summary aggregates can differ at float64 roundoff.
Fold coefficient and importance tables restore zero rows on a shared sorted
feature axis, without sorting or copying full temporary TSVs per fold. Worker
schema, site-QC and site-effect audits travel as ordered compressed Parquet
parts; the parent writes each public TSV once. All original audit rows remain.
Score ranking is performed one model context at a time. Heatmaps restore each
fold's original eligible sites from this audit,
including sites whose coefficient is zero in that fold.

Prediction constructs only exact nonzero coefficient columns. Zero-score sites
needed to preserve local-explanation ordering are also observed, and original
known states, training means, and representative scoring masks are preserved.
Prediction alignment identities and bundled TPMs cover these required OGs;
the model's complete training audit remains available. Prediction representative
tables describe the OGs actually evaluated. All-zero models use their intercept.
Heatmaps load only the displayed sites. Model fitting can still allocate
in-memory copies; disk backing does not cap total process memory. No sites are removed
to meet an internal matrix-size threshold.

Weighted mode converts sequence TPM once into compressed Parquet parts under
`runs/.phenoradar_cache/sequence_tpm`, relative to the working directory. This
persistent cache stores raw input only, with no labels or learned preprocessing,
and can be reused across metadata/CV changes. Changed sources or incomplete parts
are rebuilt; source files must remain available. Delete cache directories to
reclaim space or force rebuilding. Validated TPMs are additionally stored as float64 arrays in alignment record
order, with a shared packed positive-residue mask. This avoids repeated string
joins, number conversion and mask construction for the same alignment/species
axis. `aligned_cache_hit` and `aligned_cache_path` identify this layer in
`sequence_tpm_input.tsv`; no cache setting is needed. Missing TPMs are still
errors and zero TPM is retained. Content-identical relocated TPM inputs can
reuse both layers after identity verification.


AA grid selection consumes each inner fold once. With no site ranking, training
statistics and train/validation encoding share one residue-frequency read. Filters,
means and representative choices still use training species only. The first
inner fold sizes the shared scoring queue and refit budget from actual CSC sizes.
Completed scopes can refit while other scopes score; when scoring finishes, its
processes exit and available memory can support more refits. Random/TPE selection
retains the outer-fold scheduling path. Tables and warnings merge in fold order.
Interpretation aggregation runs in a separate process so its allocation arenas
are released before prediction and plotting.
No sites, candidates or CV folds are removed to reduce resource use.

AA runs also write the shared evaluation, model-selection, coefficient,
importance and stability SVGs under each stage's `figures/` directory.
`amino_acid_site_pss_top.svg` and `amino_acid_site_effect_range_top.svg` show
separate site scores. Frequency plots use raw residue frequencies (0–1), with
unavailable values excluded from the confusion-group distributions.
The site-filter funnel counts alignment sites, rather than residue features.

Temporary arithmetic skips canonical states absent from the cached compositions.
Training means, known states and the float32 variance threshold remain unchanged;
known-residue mass keeps the original 20-state reduction for partial sums and
rounding boundaries; provably clipped zero/one sums use compact arithmetic. Encoded columns retain
the original site/residue order. Wide-path scoring checks the full validation
matrix once, then uses the exact nonzero coefficient columns (floating-point
roundoff can differ from dense prediction).
Strict-majority modes skip sorting; unresolved modes retain the original sorting
and tie rules, including signed-zero handling.

Encoded fold matrices use RAM-buffered CSC arrays within an automatic process
budget and spill to temporary disk when that budget is exceeded. Sparse native fitting avoids input copies; its coefficient-path buffer
grows only if the active set needs more space, restarting the same complete path.
CSC sizes and site metadata bound process concurrency. Each worker has its own
matrix-buffer budget. Buffered CSC arrays grow incrementally without a full
end-of-fold data/index concatenation. Inner ratio paths
run sequentially within each AA worker, avoiding simultaneous full-matrix copies.

Unranked preprocessing also distributes independent OGs across the CPUs assigned
to each fold, using up to four threads. Initial composition-cache creation stays
serial; cached folds use parallel OG processing. Sparse OG workers build CSC blocks
directly, without assembling a dense matrix for the entire OG. Small blocks stay
in memory and larger blocks spill to temporary disk; the parent joins CSC arrays
without recomputing their training-only shifts. Results retain the original
OG/site/residue order; training reductions and representative choices are unchanged.
Audit and interpretation tables use columnar
blocks, including zero coefficients and all eligible sites.

## Tree residue heatmap and annotation labels

With `data.tree_path`, AA and combined runs also write `cv/figures/tree_group.svg`
and `tree_prediction_cv.svg`, with the same tracks and annotation TSVs as expression
runs. Groups use `data.contrast_pair_col`, falling back to `split.group_col` when
null. External-test predictions also receive a tree when at least two predicted
species occur in the tree. These plots use saved predictions; they do not refit models.

With `data.tree_path`, `cv/figures/tree_feature_heatmap_amino_acid.svg` shows
OOF species and the top sites by mean outer-fold PSS (absent sites score zero),
up to `figures.top_features`. Each column is one original, 1-based MSA site.
Letters use the [Zappo physicochemical colors](https://www.jalview.org/help/html/colourSchemes/zappo.html).

This is a display of one sequence, rather than a frequency or local contribution:
TPM-weighted mode chooses the highest-TPM gene per species/OG; other modes choose
maximum canonical-residue coverage over the input MSA. Ties use the sequence ID.
This display choice does not replace training-frozen `best_covered` selection.
Gaps remain gaps, unknown letters remain unknown, and no different gene is chosen
at an individual site. Zero total TPM or no available sequence is shown as `?`.
The accompanying `cv/tables/tree_feature_heatmap_amino_acid_annotation.tsv` records
sequence IDs, TPM, selection rule, residue status, PSS and MSA/ungapped coordinates.
Species absent from the tree are omitted with a warning; accepted-only variants
follow the existing abstention figure policy.

When `data.orthogroup_annotation_path` is supplied, AA/combined importance,
coefficient, frequency and PSS/R figures include the annotation name
alongside the OG/site/residue ID. Tree headers may shorten the annotation text; OG IDs and input-MSA site numbers
are always kept in full. SVG tooltips and the annotation TSV retain the complete name. Missing annotations retain IDs.

`amino_acid_frequency_by_confusion.svg` overlays one point per species with an
available OOF residue frequency on each boxplot. Horizontal offsets are fixed by
species order; each group reports its observed species count (`n`).

The standalone `amino_acid_frequency_heatmap.svg` is retired. Use the tree residue
heatmap instead. `top_amino_acid_frequencies.tsv` remains the source of OOF residue
frequencies for `amino_acid_frequency_by_confusion.svg`.

## Grouped probabilities

All feature modes use `summary.group_col` (default `family`) for grouped
probability figures and summary tables. For example, `summary.group_col: order`
uses metadata's `order` column and writes `probability_by_order.svg` and
`group_summary_order.tsv`. Filename suffixes are lowercase, with punctuation
replaced by underscores. This setting is independent of `split.group_col`.

AA/combined runs write these under `cv`, plus `external_test` and `inference`
when those predictions are available. AA/combined `predict` writes them under
`predict`. A predict config can override the bundle's summary column. A missing
metadata column skips that summary with a warning; missing individual group
values are grouped as `unassigned`.
