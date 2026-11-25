"""Exercise alignment metadata references across the worker pickle boundary."""
from __future__ import annotations

import pickle

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_parallel import snapshot
from phenoradar.amino_acid_worker_metadata import CachedStrings, compact_alignment_metadata


def test_worker_metadata_preserves_alignment_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("phenoradar.amino_acid_worker_metadata._MIN_CACHED_RECORDS", 0)
    directory = tmp_path / "alignments"
    directory.mkdir()
    (directory / "OG.faa").write_text(">a_1\nAV-\n>a_2\nV-A\n>b\nAVA\n")
    original = AlignmentStore(directory, ["a", "b"], input_cache_dir=tmp_path / "inputs")
    try:
        payload = compact_alignment_metadata(snapshot(original, None))
        worker = pickle.loads(pickle.dumps(payload))["store"]["ogs"]["OG"]
        reference = original.ogs["OG"]
        assert isinstance(worker.sequence_ids, CachedStrings)
        assert isinstance(reference.sequence_ids, list)
        assert worker.matrix().tobytes() == reference.matrix().tobytes()
        assert list(worker.sequence_ids) == reference.sequence_ids
        assert list(worker.sequence_hashes) == reference.sequence_hashes
        assert original.representatives(worker, ["a", "b"], [0, 1]) == [0, 2]
    finally:
        original.close()


def test_uncached_alignments_keep_strings_inline(tmp_path, monkeypatch):
    monkeypatch.setattr("phenoradar.amino_acid_worker_metadata._MIN_CACHED_RECORDS", 0)
    (tmp_path / "OG.faa").write_text(">a\nA\n")
    store = AlignmentStore(tmp_path, ["a"])
    try:
        payload = compact_alignment_metadata(snapshot(store, None))
        assert payload["store"]["ogs"]["OG"] is store.ogs["OG"]
    finally:
        store.close()
