from pathlib import Path

import numpy as np
import pytest
from test_amino_acid import alignment, settings, tpm_table

from phenoradar.amino_acid import AASiteTable, AlignmentError, AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_encoding import fit_encoded


@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_og_parallel_preserves_all_training_statistics_and_matrix_columns(
    tmp_path: Path, monkeypatch, aggregation: str, disk: bool, cache: bool,
) -> None:
    import phenoradar.amino_acid_og_parallel as parallel

    if disk:
        monkeypatch.setattr(parallel, "OG_BUFFER_BYTES", 1)
    records = [("a_1", "AC-DVAX"), ("a_2", "ADDDVAX"), ("b_1", "ADCEVAX"),
               ("c_1", "ADCDAAX"), ("d_1", "ACCEAAX"), ("held_1", "W-XWWVX")]
    path = alignment(tmp_path / "aa", records)
    (path / 'OG_é".faa').write_text((path / "OG1.faa").read_text())
    (path / "OG_gap.faa").write_text("".join(f">{s}\n-------\n" for s, _ in records))
    (path / "OG_constant.faa").write_text("".join(
        f">{s}_a\nAV\n>{s}_b\nVA\n" for s in ["a", "b", "c", "d", "held"]
    ))
    (path / "OG_few.faa").write_text(">a_1\nAV\n")
    tpm = tpm_table(tmp_path / "tpm.tsv", [
        (s.split("_")[0], s, i + 1) for i, (s, _) in enumerate(records)
    ] + [(s, f"{s}_{g}", 1) for s in ["a", "b", "c", "d", "held"] for g in ["a", "b"]])
    cfg = settings(aggregation)
    training = ["d", "b", "a", "c", "a"]
    kwargs = dict(y=np.array([0, 1, 0, 1, 0]), groups=np.array(["1", "1", "2", "2", "2"]))
    with AlignmentStore(path, ["a", "b", "c", "d", "held", "absent"],
                        cache_frequencies=cache, sequence_tpm_path=str(tpm)) as store:
        targets = [training, ["held", "absent", "c"], []]
        expected, matrices = fit_encoded(store, training, cfg, targets, **kwargs)
        actual, parallel_matrices = fit_encoded(store, training, cfg, targets,
                                               n_jobs=3, **kwargs)
        assert isinstance(actual.sites, AASiteTable)
        plain = fit_amino_acid(store, training, cfg, n_jobs=3, **kwargs)
        assert plain.features == expected.features
        np.testing.assert_array_equal(plain.matrix(store, targets[1]), matrices[1])
        assert actual.features == expected.features
        assert actual.qc == expected.qc
        for field in ["orthogroups", "column", "mean", "known", "active", "sample_fraction",
                      "minor_samples", "minor_groups", "post_aggregation_sample_fraction"]:
            np.testing.assert_array_equal(getattr(actual.sites, field),
                                          getattr(expected.sites, field))
        for actual_matrix, expected_matrix in zip(parallel_matrices, matrices, strict=True):
            np.testing.assert_array_equal(actual_matrix, expected_matrix)
            assert actual_matrix.flags.f_contiguous
        assert (actual.representative_columns is None) == (expected.representative_columns is None)
        for name, columns in (actual.representative_columns or {}).items():
            np.testing.assert_array_equal(columns, expected.representative_columns[name])
        paths = [c.path for c in store._frequency_caches.values()]
        assert len(paths) == len(set(paths))


def test_og_parallel_keeps_global_empty_fold_errors(tmp_path: Path) -> None:
    path = alignment(tmp_path / "aa", [("a_1", "--"), ("b_1", "--")])
    (path / "OG2.faa").write_text((path / "OG1.faa").read_text())
    with AlignmentStore(path, ["a", "b"]) as store, pytest.raises(
        AlignmentError, match="removed all sites",
    ):
        fit_encoded(store, ["a", "b"], settings(), [["a", "b"]], n_jobs=2,
                    y=np.array([0, 1]), groups=np.array(["a", "b"]))


def test_cold_cache_uses_serial_creation_then_parallel_cached_folds(
    tmp_path: Path, monkeypatch,
) -> None:
    import phenoradar.amino_acid_og_parallel as parallel

    original = parallel.fit_orthogroups
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(parallel, "fit_orthogroups", tracked)
    path = alignment(tmp_path / "aa", [("a_1", "AA"), ("b_1", "VV")])
    (path / "OG2.faa").write_text((path / "OG1.faa").read_text())
    with AlignmentStore(path, ["a", "b"], cache_frequencies=True) as store:
        kwargs = dict(n_jobs=2, y=np.array([0, 1]), groups=np.array(["a", "b"]))
        fit_encoded(store, ["a", "b"], settings(), [["a", "b"]], **kwargs)
        assert not calls
        fit_encoded(store, ["a", "b"], settings(), [["a", "b"]], **kwargs)
        assert calls == [True]
