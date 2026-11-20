import gzip
from pathlib import Path
from threading import Lock
from time import sleep

import numpy as np
import pytest
from test_amino_acid import alignment, settings

from phenoradar.amino_acid import AlignmentError, AlignmentStore
from phenoradar.amino_acid_frequency_cache import AminoAcidFrequencyCache
from phenoradar.amino_acid_sparse import fit_sparse_encoded


@pytest.mark.parametrize('threads', [2, 4, 12])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('compressed', [False, True])
def test_input_threads_preserve_axes_bytes_and_cache_hits(tmp_path, threads, cached, compressed):
    path = alignment(tmp_path / 'aa', [('a_1', 'AC-DVA'), ('b_1', 'ADCDVA')])
    for i in range(16):
        (path / f'OG{i + 2:02}.faa').write_bytes((path / 'OG1.faa').read_bytes())
    if compressed:
        for source in path.glob('*.faa'):
            source.with_suffix('.faa.gz').write_bytes(gzip.compress(source.read_bytes()))
            source.unlink()
    cache = tmp_path / 'cache' if cached else None
    reference = None
    for jobs in [1, threads, threads]:
        messages = []
        with AlignmentStore(path, ['a', 'b'], input_cache_dir=cache,
                            n_jobs=jobs, progress=messages.append) as store:
            current = (store.identities(), list(store.ogs),
                       [(og.matrix().tobytes(), Path(og.mask_cache).read_bytes(),
                         og.sequence_ids, og.species_rows) for og in store.ogs.values()])
            if reference is None:
                reference = current
            assert current == reference
        if cached and jobs != 1:
            assert messages == ['Reuse 17/17 alignment byte caches.']


@pytest.mark.parametrize('suffix', ['.fa', '.fa.gz'])
def test_parallel_duplicate_stems_are_rejected(tmp_path, suffix):
    path = alignment(tmp_path / 'aa', [('a_1', 'AC')])
    payload = (path / 'OG1.faa').read_bytes()
    (path / ('OG1' + suffix)).write_bytes(
        gzip.compress(payload) if suffix.endswith('.gz') else payload
    )
    with pytest.raises(AlignmentError, match='Duplicate orthogroup'):
        AlignmentStore(path, ['a'], n_jobs=4)


def test_frequency_restore_overlaps_without_private_path_collisions(tmp_path, monkeypatch):
    path = alignment(tmp_path / 'aa', [('a_1', 'ACDV'), ('b_1', 'ADAV'),
                                     ('c_1', 'VCCA'), ('d_1', 'VDCA')])
    for i in range(7):
        (path / f'OG{i + 2}.faa').write_bytes((path / 'OG1.faa').read_bytes())
    cache = tmp_path / 'cache'
    with AlignmentStore(path, ['a', 'b', 'c', 'd'], cache_frequencies=True,
                        input_cache_dir=cache) as warm:
        fit_sparse_encoded(
            warm, ['a', 'b', 'c', 'd'], settings('aa_frequency'),
            [['a', 'b', 'c', 'd']], y=np.array([0, 1, 0, 1]),
            groups=np.array(['x', 'x', 'y', 'y']), audit=False, packed=True, n_jobs=1,
        )
    original = AminoAcidFrequencyCache.__init__
    lock = Lock()
    active = peak = 0
    destinations = []
    def tracked(self, destination, *args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            destinations.append(destination)
        try:
            sleep(.01)
            original(self, destination, *args, **kwargs)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(AminoAcidFrequencyCache, '__init__', tracked)
    with AlignmentStore(path, ['a', 'b', 'c', 'd'], cache_frequencies=True,
                        input_cache_dir=cache) as store:
        fitted, matrices, shift = fit_sparse_encoded(
            store, ['a', 'b', 'c', 'd'], settings('aa_frequency'),
            [['a', 'b', 'c', 'd']], y=np.array([0, 1, 0, 1]),
            groups=np.array(['x', 'x', 'y', 'y']), audit=False, packed=True, n_jobs=4,
        )
        assert len(fitted.features) and matrices and shift.size
        assert 2 <= peak <= 4
        assert len(destinations) == len(set(destinations)) == 8
