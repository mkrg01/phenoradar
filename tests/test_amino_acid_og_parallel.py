from pathlib import Path
from threading import Event

import numpy as np
import pytest
from test_amino_acid import alignment, settings, tpm_table

from phenoradar.amino_acid import AASiteTable, AlignmentError, AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_encoding import fit_encoded


def test_completed_og_refills_workers_before_slow_first_og_finishes(tmp_path, monkeypatch):
    import phenoradar.amino_acid_og_parallel as parallel

    path = alignment(tmp_path / "aa", [("a_1", "AA"), ("b_1", "VV")])
    for i in range(2, 7):
        (path / f"OG{i}.faa").write_text((path / "OG1.faa").read_text())
    later_started = Event()
    original = parallel.fit_amino_acid

    def fit(local, *args, **kwargs):
        name, = local.ogs
        if name == "OG1":
            assert later_started.wait(5), "Idle worker did not receive the third OG"
        elif name == "OG3":
            later_started.set()
        return original(local, *args, **kwargs)

    monkeypatch.setattr(parallel, "fit_amino_acid", fit)
    with AlignmentStore(path, ["a", "b"]) as store:
        result = parallel.fit_orthogroups(
            store, ["a", "b"], settings(), n_jobs=2, encoding=None,
            y=np.array([0, 1]), groups=np.array(["a", "b"]),
        )
    assert result.sites.orthogroups.tolist() == [f"OG{i}" for i in range(1, 7) for _ in range(2)]


def test_og_completed_byte_accounting_includes_sparse_and_packed_buffers():
    from types import SimpleNamespace

    from scipy import sparse

    from phenoradar.amino_acid_encoding import EncodedBlock
    from phenoradar.amino_acid_og_parallel import _result_bytes

    matrix = sparse.csc_matrix(np.eye(3))
    shift = np.zeros(3)
    fitted = SimpleNamespace(sites=[])
    expected = sum(getattr(matrix, n).nbytes for n in ["data", "indices", "indptr"])
    assert _result_bytes(fitted, EncodedBlock([matrix], shift)) == expected + shift.nbytes
    assert _result_bytes(fitted, EncodedBlock([np.eye(3)], shift)) == 72 + shift.nbytes


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


@pytest.mark.parametrize("packed", [False, True])
def test_parallel_does_not_restore_frequency_cache_for_uncovered_og(
    tmp_path: Path, monkeypatch, packed: bool,
) -> None:
    from phenoradar.amino_acid_packed import packed_arithmetic
    from phenoradar.amino_acid_sparse import fit_sparse_encoded

    if packed and packed_arithmetic() is None:
        pytest.skip("Compatible native packing kernels are unavailable")
    path = alignment(tmp_path / "aa", [("a_1", "AA"), ("b_1", "VV"),
                                        ("c_1", "AV"), ("d_1", "VA")])
    (path / "OG_few.faa").write_text(">a_1\nAV\n")
    samples = ["a", "b", "c", "d"]
    args = dict(y=np.array([0, 1, 0, 1]), groups=np.array(["g0", "g0", "g1", "g1"]))
    with AlignmentStore(path, samples, cache_frequencies=True) as store:
        reference, expected, shift = fit_sparse_encoded(
            store, samples, settings(), [samples], packed=packed, **args,
        )
        original = AlignmentStore._frequency_cache
        calls = []

        def restore(current, og, weighted):
            assert og.orthogroup != "OG_few", "Uncovered OG must not read its frequency cache"
            calls.append(og.orthogroup)
            return original(current, og, weighted)

        monkeypatch.setattr(AlignmentStore, "_frequency_cache", restore)
        actual, matrices, actual_shift = fit_sparse_encoded(
            store, samples, settings(), [samples], packed=packed, n_jobs=3, **args,
        )
        assert calls and actual.features == reference.features and actual.qc == reference.qc
        np.testing.assert_array_equal(actual_shift, shift)
        a = matrices[0].to_csc() if packed else matrices[0]
        b = expected[0].to_csc() if packed else expected[0]
        for name in ["data", "indices", "indptr"]:
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes()


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
