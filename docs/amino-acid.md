# Amino-acid phenotype prediction

Expression and sequence modes use the same `config.yml`. To enable prediction
from orthogroup alignments, set `data.feature_mode: amino_acid` and the input
paths below. Optional `amino_acid` preprocessing is at the end of the config:

```bash
phenoradar run -c config.yml
```

## Inputs

Supply the normal species metadata and `data.alignment_dir`. Each `.faa`, `.fa`,
or `.fasta` file directly in that directory is one OG; its filename stem is the
OG ID. No input manifest or provenance file is needed. All discovered OGs are
candidates, independent of how they were selected upstream.

Each FASTA record counts as one gene. Sequence IDs are the first header token.
They must be unique within an OG, and sequences within an OG must have equal
alignment lengths. A record matches an exact metadata species ID, or that ID
followed by `_` and a gene suffix. Matching must be unambiguous; otherwise input
loading fails. Explicit sequence maps are no longer supported. Older configs
with `sequence_map_path: null` remain readable.

The canonical alphabet is `ACDEFGHIKLMNPQRSTVWY`. `-` and `.` are gaps. Other
letters and `*` are unknown states: preserved for coordinates, excluded from
frequency denominators. Invalid non-sequence characters are errors. Missing
sample/OG combinations remain missing; they are not treated as gene absence.

For expression-weighted frequencies, select:

```yaml
data:
  feature_mode: amino_acid
  alignment_dir: c4_dataset/alignments
  sequence_tpm_path: c4_dataset/sequence_tpm.tsv.gz
amino_acid:
  aggregation: tpm_weighted_frequency
```

The sequence TPM input is a TSV (optionally gzip-compressed) with `species`,
`sequence_id`, `tpm`. Keys refer to metadata species and complete FASTA IDs.
Each cached alignment record must have exactly one finite, nonnegative TPM;
missing keys, duplicate matching keys and invalid matching values are errors.
Extra rows are ignored. TPM 0 is
valid and gives zero weight. The other two aggregation modes do not read this input.

At each site, weighted frequency is the sum of TPMs for genes carrying a given
residue divided by the sum of TPMs for genes with any canonical residue. Gaps and
unknowns do not enter this denominator. A zero denominator is missing, with no
fallback to unweighted frequencies. Multiplying all TPMs in a species/OG by a
common factor leaves its features unchanged: total OG expression is not included.
An aggregation list compares these modes on the same metadata, alignments and
split. Reports allow the additional TPM input in weighted mode; its hash is
retained and must agree across weighted conditions.

## Preprocessing

Each outer training set, inner training set, sampled set, and final refit learns
its own preprocessing. Validation and inference rows never determine sites,
major residues, ranking scores, or neutral values.

1. Packed masks count species with at least one canonical residue at a site
   (on a positive-TPM gene in weighted mode).
   `min_sample_fraction` (default 0.8) uses all training species as denominator,
   including species without that OG. This screen runs before 20-state frequency
   counting.
2. On surviving columns, compute each species' residue frequencies: equal-gene
   frequencies for `aa_frequency`/`best_covered`, TPM-weighted frequencies for
   `tpm_weighted_frequency`.
   Filter on `min_residue_states` (distinct canonical residues; default 2).
   There is no minimum number of species or groups carrying nonmajor residues.
   The retired `min_minor_sample_count` and `min_minor_group_count` keys are
   ignored with warnings when reading older configurations or bundles; resolved
   configs omit them.
3. For `best_covered`, choose each species' gene with the most canonical residues
   on the quality-filtered columns in its OG; ties use sequence ID and zero
   coverage selects no gene. Freeze these scoring columns before phenotype
   ranking. Recheck coverage and variant filters on the chosen representatives,
   without choosing genes again. Ranking and model features use the same genes;
   validation/prediction chooses its genes on the training-frozen scoring columns.
4. Optionally rank sites using the selected aggregation, including representatives
   or TPM-weighted frequencies. `pair_aware` averages phenotype-frequency differences
   equally over valid contrast pairs, then takes their vector norm; it requires
   `min_contrast_pairs` per site. `unpaired` uses the difference between phenotype
   mean compositions. Neither imposes a direction. Ties use OG ID and original
   column order. `max_sites` limits the retained set. With ranking `none`, exceeding
   that limit is an error rather than an implicit truncation.
5. Freeze the model site set. `aa_frequency` uses all genes, excluding gaps/unknowns
   separately at each site; `tpm_weighted_frequency` uses the weighted frequencies
   above. `best_covered` keeps the representatives chosen before ranking, even
   when ranking or constant-state removal discards some scoring sites.
6. Center each state on its training mean. Missing or previously unseen state
   mass receives that mean and therefore contributes zero. Known mass in a
   partially novel composition retains its proportional effect. Drop constant
   state columns without changing the representative-scoring site set.

Sequence features are not variance-scaled. Existing expression transforms and
filters are inactive in this mode. `lambda` and `alpha` retain their existing
glmnet meanings; grid, random, TPE, best/one-SE selection, grouped sampling,
sample weighting, and mean/median ensembles use the shared implementations.

Alignment bytes for training-pool species are cached on disk once per run.
Presence masks are bit-packed; frequency statistics use site/gene blocks. Only
selected sites are expanded into model matrices. Target import is restricted to
selected OGs and predictions are batched. Frequency workspaces use column blocks.
Outer folds run sequentially to bound memory; inner glmnet paths use
`runtime.n_jobs`.
Sequence TPM is converted once, in bounded blocks, to compressed Parquet parts
automatically under `runs/.phenoradar_cache/sequence_tpm`, relative to the working
directory. No cache location is needed in user configs.
Later imports select required species/genes using a per-part species index,
including across runs and the training/inference stages. Parts are queried in
small batches; only copied matching rows are retained between batches.
The cache includes all original TPM rows and no metadata, labels, sites, ranking
scores or training means. Replacing metadata or changing CV settings reuses it;
all learned preprocessing is still fitted separately within training folds.

The source path, size, nanosecond modification/change times, device/inode and
cache format identify an entry. The source SHA-256 is computed at creation and
reused on a hit; the large source is not reread just to hash it. Changed sources
get new entries. Changed/incomplete cached parts are rebuilt; a lock and atomic
publication protect concurrent runs. The source must remain available. Cache
directories can be deleted to reclaim space or force rebuilding. Older configs
with an explicit cache location or `null` (direct reading) remain accepted.
Alignment-byte caches remain temporary per run. CV folds share imported weights
and positive-TPM masks.

Training-observed-mean centering is fixed. Frequency blocks (512 columns),
prediction batches (128 species) and the workspace
limit (50,000,000 entries in species × sites × 20) are internal constants, not
config options. The limit guards this workspace, not total process memory.
Older configs containing these retired settings are accepted with a warning and
the settings are omitted from resolved output. Cache hits and locations are
recorded in `sequence_tpm_input.tsv`.

## Outputs and prediction

Standard split, OOF prediction, metric, coefficient, feature-stability and optional
group-bootstrap tables are written under the normal stage directories. Sequence
tables include:

- `alignment_inputs.tsv`, `sequence_correspondence.tsv`: input hashes and ID mapping.
- `sequence_tpm_input.tsv` (weighted mode): input hash, matched/zero-weight counts,
  cache location and whether the cached input was reused.
- `amino_acid_og_qc.tsv`, `amino_acid_site_qc.tsv`: train-only filtering counts,
  pre/post-aggregation coverage, variant support and scores. `minor_samples` and
  `minor_groups` are descriptive counts relative to the training major residue
  (defined by mean species frequencies); neither is a filtering criterion.
- `feature_schema.tsv`, `representative_sequences.tsv`: state columns and gene choices.
- `representative_scoring_sites.tsv`: training-frozen, pre-ranking columns used
  to choose representatives; gene tables distinguish scoring-site and model-site coverage.
- `amino_acid_site_effects.tsv`: state effects versus the training major residue,
  effect ranges and mean absolute site contributions.
- `amino_acid_site_pss.tsv`: one row per fitted model/site with PSS, the sum of
  absolute residue coefficients, plus `normalized_score` and within-model
  descending `rank`.
- `amino_acid_site_effect_range.tsv`: the same model/site keys, with the maximum
  minus minimum coefficient over training-observed residues. Observed residues
  with zero coefficients are included; unobserved residues are excluded.
- `amino_acid_site_stability.tsv`, `amino_acid_og_stability.tsv`: selected-fold
  fractions, including fractions conditional on being eligible.
- `amino_acid_local_contributions.tsv`: leading per-model site contributions,
  effective observation fractions, OG gene counts and callable-gene counts.
- `selected_site_sequence_map.tsv`: all source genes for selected sites (the chosen
  gene in representative mode), with original residues and sequence checksums;
  weighted mode also reports `sequence_tpm` and `site_tpm_fraction`, each gene's
  share of the callable TPM at that site (zero for gaps, unknowns or zero weight).

Both score files are written in `cv/tables`, `model/tables` for final refits, and
`predict/tables` when loading a saved bundle. Keys include `scope`, `sample_set_id`,
`candidate_index`, OG and original alignment position. Values are calculated per
model, without averaging coefficients across folds. Each score is normalized by
its total across sites in that model; all-zero models have normalized scores of
zero. Zero-score sites remain in the tables with a blank rank; ties share a rank.
These are coefficient-based interpretation scores, not measures of predictive
performance. Site selection, stability and information coverage still use the
effect range.

Coordinates are one-based. Alignment positions always refer to the input MSA.
Sequence positions count non-gap input characters; `X` and `*` are not silently
deleted. Gap coordinates are empty. The OOF frequency heatmap uses each held-out
species' fold-fitted preprocessing; grey cells are unavailable. The heatmap's
displayed state columns are chosen by normalized absolute state coefficients
averaged across CV models/folds (`feature_importance.tsv`). It does not rank sites
by PSS or effect range; separate site-score plots are not currently generated.

If `abstention.enabled`, information coverage weights site observation fractions
by the fitted within-site coefficient range, normalized within each model and
then averaged across models. Frequency-mode observation includes the fraction
of OG genes callable at that site and the fraction of residue mass known to the
model. Weighted mode uses the fraction of OG TPM callable at the site, multiplied
by known residue mass; it is not a fraction of all transcriptome TPM. Local
contribution tables also report positive-TPM and callable positive-TPM gene counts.
Intercept-only members contribute zero coverage. Metrics still include
all OOF species, independently of selective decisions.

Set `runtime.execution_stage: full_run` to refit and export `model_bundle/`:

```bash
phenoradar predict --model-bundle runs/RUN_ID/model_bundle \
  --metadata-path prediction_species.tsv --alignment-dir path/to/alignments
```

The bundle stores preprocessing, configuration and (for weighted
mode) the TPMs for required OGs and original species. The source TPM file is not
needed to reproduce predictions. To use a different expression input with the
same fitted model and alignments, pass `--sequence-tpm-path new_tpm.tsv.gz`
(or `data.sequence_tpm_path` in `--config`); the actual input hash is recorded.
Only requested species/required OGs need matching TPMs. `--config`
can supply input paths; it cannot retrain preprocessing. Prediction currently
requires species from the original metadata and the original frozen alignments
for required OGs, verified by content hashes. New alignments or added sequences
need a future coordinate-mapping interface. Combined total-expression/sequence
models and structure mapping are deferred.

New representative bundles use `aa-3` and store their pre-ranking scoring masks.
Older `aa-1`/`aa-2` bundles remain readable; legacy representatives retain their
original post-ranking selection rule when predicting.
