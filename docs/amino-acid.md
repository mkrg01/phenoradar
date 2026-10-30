# Amino-acid and combined modes

Use orthogroup (OG) amino-acid alignments alone or jointly with expression.
All modes share `config.yml`, grouped CV, sampling, weights, model selection,
and mean/median ensembles. Sequence and combined modes require
`model.name: logistic_elasticnet`.

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
preprocess:
  amino_acid:
    aggregation: aa_frequency
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
| `aa_frequency` (default) | Equal-gene canonical-residue frequencies at each site |
| `best_covered` | One gene per species/OG, chosen on training-frozen quality-filtered columns |
| `tpm_weighted_frequency` | Canonical-residue frequencies weighted by gene TPM |

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
| `preprocess.amino_acid.ranked_site_filter.max_sites` | `2000` | Positive retained-site limit |
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
   imposes a direction. Ties use OG ID and input column order. With ranking
   `none`, exceeding `max_sites` is an error, not implicit truncation.
5. Freeze model sites, center states on training means, and drop constant state
   columns. Missing or previously unseen state mass receives the mean and has
   zero contribution; known mass in a partly novel composition retains its effect.

Sequence features are centered without variance scaling. `best_covered` uses the
same representatives for ranking and fitting, even after sites are removed.
Validation/prediction selects its genes on the training-frozen scoring columns.
Minor-sample/group counts remain descriptive QC fields, without filtering sites.

## Joint expression and amino-acid mode

Set `data.feature_mode: combined`, supply both `data.tpm_path` for OG expression
and `data.alignment_dir`, and use neutral expression preprocessing:

```yaml
data:
  feature_mode: combined
  metadata_path: data/species_metadata.tsv
  tpm_path: data/tpm.tsv
  alignment_dir: data/alignments
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

Alignment-byte caches are temporary per run. Only selected sites are expanded
into model matrices; prediction and frequency work are batched. Outer folds run
sequentially to bound memory, while inner glmnet paths use `runtime.n_jobs`.
Workspace limits are internal and do not cap total process memory.

Weighted mode converts sequence TPM once into compressed Parquet parts under
`runs/.phenoradar_cache/sequence_tpm`, relative to the working directory. This
persistent cache stores raw input only, with no labels or learned preprocessing,
and can be reused across metadata/CV changes. Changed sources or incomplete parts
are rebuilt; source files must remain available. Delete cache directories to
reclaim space or force rebuilding. Cache hits and paths appear in
`sequence_tpm_input.tsv`; no cache setting is needed.
