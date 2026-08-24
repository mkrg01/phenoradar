# c4_tiny

Compact test dataset for PhenoRadar.

## Files

- `species_metadata.tsv`: 20 species rows (train/validation, external_test, discovery_inference)
- `species_trait.tsv`: raw species trait table used to generate metadata
- `ncbi_tree.nwk`: NCBI taxonomy constrained tree used to assign contrast groups
- `tpm.tsv`: 1,515 long-format expression rows

## Origin

This dataset is a compact subset derived from `c4_dataset/` for smoke testing and documentation examples.

## Integrity

`SHA256SUMS` is the canonical checksum manifest used by `phenoradar dataset`.
The command validates every bundled or externally fetched file before reporting success.
