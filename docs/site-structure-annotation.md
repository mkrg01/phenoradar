# External reference numbering and protein structures

Structure annotation is optional postprocessing for completed amino-acid and
combined runs. It uses the fitted model's site scores and input alignment.

## Run automatically from YAML

Install `phenoradar[structure]` and the MAFFT executable. To also save annotated
PyMOL sessions and PDFs, install PyMOL and set `render: true`.

```yaml
report:
  structure:
    enabled: true
    top_orthogroups: 20
    orthodb_release: v12.2  # optional if inferred from the OG annotation filename
    render: true
```

Run `phenoradar run -c config.yml` as usual. PhenoRadar uses final-model sites for
`full_run` and outer-CV sites for `cv_only`. It ranks OGs by their maximum mean
site PSS, with ties broken by OG ID, and maps their selected sites. For each OG,
it searches RCSB PDB with the training MSA row closest to the median ungapped
length, breaking ties by sequence ID. Gaps are removed before search. For a query
shorter than 25 residues, it skips the PDB search and still looks for a verified
reference and AlphaFold model. The search sequence is sent to RCSB for eligible
queries and is separate from the full-length coordinate reference.

Results are written to `RUN/site_annotations/`. `RUN/site_annotation_inputs/`
contains the generated query FASTA, candidate catalog and `status.json`, which
records duration and success or failure independently of model fitting. Use
`mafft_exe`, `pymol_exe` or `cache_dir` under `report.structure` to override the
executable paths or resource cache. With `render: false` (the default), the
coordinates, mapping tables and PyMOL scripts are still saved. Open `sites.pse`
directly in PyMOL when `render: true`, or run `pymol sites.pml` from the structure
directory to rebuild the session with the saved coordinates.

## Run separately on completed results

Install `phenoradar[structure]` plus the MAFFT and PyMOL executables separately.
PyMOL can run in a different environment; pass its executable explicitly.

```bash
phenoradar prepare-site-references \
  --run-dir RUN --orthodb-release v12.2 --out references.json \
  --cache-dir site_resource_cache

phenoradar annotate-sites \
  --run-dir RUN --catalog references.json --out site_annotations \
  --cache-dir site_resource_cache --mafft-exe mafft --pymol-exe pymol
```

Both commands accept repeated `--orthogroup OG` to process a subset. `--scope final`
uses sites with PSS > 1e-12 in the final ensemble; `--scope cv` uses the outer-CV
union. Scope is explicit and defaults to final, including for cv-only runs.
All requested sites are retained in the correspondence tables.

## Prioritize experimental structures at selected sites

Supply a FASTA with one ungapped protein sequence per OG. Use the
OG ID as the first token of the header and include the source sequence ID in the
description. The query sequence is sent to the RCSB PDB sequence-search service.
An optional sites TSV has the columns `orthogroup`, `alignment_pos_1based`, and
`required` (optional, `true` or `false`). Use the column numbers in the exact MSA
used by the run. With no sites TSV, selection uses all sites in the requested scope.

```bash
phenoradar prepare-site-references \
  --run-dir RUN --orthodb-release v12.2 --pdb-query-fasta queries.fa \
  --cache-dir site_resource_cache --out pdb_candidates.json

phenoradar annotate-sites \
  --run-dir RUN --catalog pdb_candidates.json --sites sites.tsv \
  --cache-dir site_resource_cache --out site_annotations --no-render
```

The search screens up to 100 experimentally determined PDB polymer entities per
query (E-value cutoff 1, sequence identity cutoff 0.25). It first checks for an
exact full-length match to a UniProt-linked OG member in the run's OrthoDB
release. A reviewed experimental reference can also match one member of the
same NCBI taxon using an accession link, a shared NCBI Gene ID, or its primary
gene symbol in the member's NCBI Gene aliases. This route requires one of the
following sequence matches:

- A position-by-position comparison of equal-length proteins with at least 98%
  identity.
- An exact C-terminal match to a longer OrthoDB protein, with a reviewed
  UniProtKB `Chain` feature spanning the entire shorter sequence.

The catalog records the gene evidence, matched OrthoDB interval and sequence
differences. Site projection uses the actual UniProt sequence. Candidate chains
with verified accessions are fetched as SIFTS-enriched PDBe mmCIF files.

An eligible chain contains a deposited Cα coordinate at every required site and
at least one requested site. Among eligible chains, the largest number of
observed requested sites wins, followed by RCSB sequence-search score and PDB,
chain and UniProt identifier order. The `pdb_site_candidates.tsv` table records
observed columns and reasons for exclusion, including SIFTS inconsistencies
specific to individual chains. Check the local projection for sites interpreted
individually.

When eligible experimental coordinates are unavailable, AlphaFold models of
exact full-length linked references are ranked by observed site coverage and
mean pLDDT over mapped sites. The catalog includes an exact query-linked UniProt
reference where available. If no exact linked reference is verified,
preparation tries closest same-OG members by full-length protein alignment.
A portable catalog freezes the selected structure and the site list. To replay,
use the same sites TSV and `--offline`.

## OrthoDB snapshot and complete candidate collection

An exact OrthoDB release is required. Annotation checks it against the run's
`odb12v2_OGs`-style annotation filename; when that information is absent, supply
`--orthodb-release` explicitly. OG identifiers are not portable across releases.

Preparation enumerates every member of each requested OG from the versioned
OrthoDB API, checks the member count, obtains each gene's UniProt cross-references,
and retrieves its UniProt sequence. This can require many first-use requests.
For larger batches, supply the same-release `--og2genes FILE --gene-xrefs FILE`
dumps instead; they are streamed once and their hashes are recorded. The caller
must supply dumps from the declared release. No candidate limit is applied.
Only UniProt sequences exactly matching a linked full-length OrthoDB FASTA member
enter the verified pool. Missing accessions and sequence/version differences stay
in the catalog's exclusions; they are not silently replaced by a similar hit.

The cache freezes URL responses, checksums, retrieval timestamps and response
headers, including explicit 404 absences. Timeouts, incomplete responses and
server errors fail the command. `--offline` requires all resources to be cached.
Use a new cache directory to intentionally refresh a resource snapshot.

## Reference policy: species-balanced-reference-1

Reference choice uses the training species recorded in the saved split manifest,
not external/inference targets, labels, TPM, PSS, selected-site coverage, or
structure availability. A missing species contributes zero rather than being
removed from the comparison. All folds use the same display-only reference.

1. Define core columns as input-MSA columns containing at least one canonical
   residue in at least 80% of training species. Conserved columns are included.
2. Add full-length candidates with MAFFT `--amino --anysymbol --add --keeplength --mapout --thread 1`.
   Original training rows must remain byte-identical. Identical candidate
   sequences are aligned once; their distinct accessions remain in the ranking.
3. In core columns, let I be identity over jointly canonical positions and let
   C1/C2 be the fractions of each sequence's canonical positions that are jointly
   canonical. Gene similarity is `I * min(C1, C2)`, equivalently the identical
   count divided by the larger canonical-residue count. No overlap scores zero.
4. For each species, use its gene with maximum similarity, breaking ties by
   sequence ID. Rank candidates by the median species similarity, descending.
   Copy number never increases a species' weight. This is a coordinate reference,
   not a new test proving one-to-one orthology or shared biological function.
5. Require verified membership, a non-fragment UniProt record, at least 80% core
   coverage, and at least 30% median identity for those species/gene matches.
   No qualifying candidate gives `reference_unavailable`; thresholds are not
   relaxed separately for an OG. Tie-breaks are core coverage, reviewed status,
   then accession in ascending lexical order.

These are explicit initial QC thresholds, not guarantees of residue homology.
`--policy JSON` can change the three thresholds globally; the full policy is saved.
Its default contents are:

```json
{"core_species_fraction": 0.8, "minimum_reference_coverage": 0.8, "minimum_median_identity": 0.3}
```

Validate changed policies independently, not by maximizing classification metrics.
`--reference-overrides TSV` has `orthogroup` and `accession` columns, must refer to
catalog candidates, and is recorded as manual even if it fails automatic QC.

Reference positions come from MAFFT's map of the *full* input reference sequence.
Do not count non-gap residues in a trimmed MSA to infer whole-protein positions.
Original MSA columns and full-length reference positions are retained separately;
reference gaps and unknown residues remain explicit unmapped states.

## Structures and selection

References are fixed before searching structures. PDB chains must map to that
same UniProt accession through residue-level SIFTS, preserving author chain,
residue number, insertion code, unobserved residues and sequence differences.
Screen all listed candidates using observed core coverage. Within each experimental
method, choose a chain with at least the reference policy's minimum core coverage,
ranked by core coverage, observed sequence identity, resolution where defined,
then PDB/chain ID. Experimental methods are not combined into a quality score.
Rejected partial structures remain in the screening table. Coordinate/SIFTS
inconsistencies fail instead of silently selecting a different structure.

Same-reference AlphaFold models are supplementary, or the main view if PDB is
unavailable. Sequence versions and fragment intervals must match the frozen
reference. Relevant fragments have separate views; do not infer distances across
fragments. Only the first deposited model is displayed for experimental ensembles.
Alternate-location CA coordinates use deposited order. pLDDT remains a distinct
local-confidence measure, not PhenoRadar importance or alignment confidence.

New reference catalogs retain exact UniProtKB `Active site` and `Binding site`
features from the verified reference, including ligand IDs and evidence. The
`functional_site_map.tsv` records both mapped and unresolved feature positions.
Prepare the reference catalog again to add UniProt features to older catalogs.
For experimental structures, `pdb_ligands.tsv` lists deposited non-water,
non-polymer components in the displayed model, including ions and additives.

When these coordinates are available, the PDF has a functional-context page and
PyMOL has a `functional_context` scene. Toggle `UniProt_active_sites` (blue),
`UniProt_binding_sites` (green), and `PDB_ligands` (orange) independently. The
original score pages and front scene keep their score colors.

## Outputs and PDF display

The new output directory contains `manifest.json` and a portable
`reference_catalog.json`. Reuse the latter with `--offline` to replay references
and copied structure/SIFTS inputs in another new directory with the same split,
alignment-input inventory and policy. For changed inputs/policy, use the original
prepared catalog and resource cache so all structure candidates are screened again.
Do not overwrite
annotation directories. Run/alignment hashes, candidate and structure rankings,
QC failures, chosen species genes, commands and tool versions are retained.

Each OG has `reference_candidates.tsv`, `reference_species_matches.tsv`,
`selected_reference.json`, `reference.fa`, `site_reference_map.tsv`, and structure
screening/selection tables. When `data.orthogroup_annotation_path` has a name for
the OG, the mapping tables, PDF header and PyMOL group include that name alongside
the OG ID. The portable catalog retains the names for offline replay.
Each structure folder contains the copied coordinates,
`site_structure_map.tsv`, `scene.json`, `render.py`, `sites.pml`, `sites.pse`,
`rendering.json`, and `sites.pdf`. Run `pymol -cq render.py` **in that folder**
to regenerate its saved scenes. `--no-render` writes mapping and scene scripts
without requiring PyMOL; the manifest explicitly records that rendering was skipped.

PDF pages show front/back/right/left/top/bottom in a 2-column, 3-row layout.
The saved initial CA orientation, orthographic projection, camera scale and proper
rotation matrices are shared across views. Target chains are emphasized and other
protein chains are muted context. PNG panels are ray traced at 1000 x 1000 pixels;
page text, legends and complete mapping tables are PDF text.

Separate pages show selected sites, mean PSS, and CV selection frequency;
each quantitative scale is fixed across six views and labeled per structure/OG.
AlphaFold adds a pLDDT page. The Selected sites page uses colored markers and a
key below the views when up to eight sites have coordinates. The key lists the
MSA column, UniProt residue and structure residue. Larger sets use red markers
and the paginated audit table. PyMOL hides residue labels by default. Use
`show labels, selected_sites` to reveal them or an `MSA_<column>` selection to
locate an individual site. The paginated audit lists every requested site,
including gaps, unobserved coordinates, substitutions in the structure and
unavailable confidence.
PDFs show the reference's automatic/manual status and QC outcome.

SIFTS: https://www.ebi.ac.uk/pdbe/docs/sifts/methodology.html
MAFFT: https://mafft.cbrc.jp/alignment/software/addsequences.html
