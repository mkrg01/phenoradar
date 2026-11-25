from pathlib import Path

import joblib
import numpy as np
import pytest
from test_amino_acid import alignment, settings, tpm_table

from phenoradar.amino_acid import AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_encoding import fit_encoded
from phenoradar.amino_acid_sparse import fit_sparse_encoded
from phenoradar.glmnet import fit_glmnet_path


@pytest.mark.parametrize("sparse_input", [False, True])
def test_constant_and_tiny_variation_columns_keep_regularized_predictions(
    tmp_path: Path, sparse_input: bool,
) -> None:
    training = [f"s{i}" for i in range(32)]
    targets = ["held0", "held1", "absent"]
    y = np.arange(len(training)) % 2
    records, constant_records, weights = [], [], []
    for i, species in enumerate([*training, *targets[:-1]]):
        if i < len(training):
            trait = "V" if y[i] else "A"
            delta = 2e-7 if y[i] else -2e-7
            sequences = ("A" + trait, "V" + trait)
        else:
            delta = .4
            sequences = ("AA", "VA") if species == "held0" else ("WW", "WW")
        records.extend([(species + "_a", sequences[0]), (species + "_v", sequences[1])])
        constant_records.extend([(species + "_c", "A"), (species + "_d", "V")])
        weights.extend([
            (species, species + "_a", .5 + delta),
            (species, species + "_v", .5 - delta),
            (species, species + "_c", 1. if i < len(training) else 3.),
            (species, species + "_d", 1.),
        ])
    path = alignment(tmp_path / "aa", records)
    (path / "OG2.faa").write_text(
        "".join(f">{name}\n{sequence}\n" for name, sequence in constant_records)
    )
    tpm = tpm_table(tmp_path / "tpm.tsv", weights)
    with AlignmentStore(
        path, [*training, *targets], cache_frequencies=True, sequence_tpm_path=str(tpm),
    ) as store:
        cfg = settings("tpm_weighted_frequency")
        kwargs = dict(y=y, groups=np.array([str(i % 4) for i in range(len(training))]))
        if sparse_input:
            fitted, matrices, _ = fit_sparse_encoded(
                store, training, cfg, [training, targets], **kwargs,
            )
        else:
            fitted, matrices = fit_encoded(store, training, cfg, [training, targets], **kwargs)
        train, held = matrices
        dense = train.toarray() if sparse_input else train
        old_columns = dense.var(axis=0) > 1e-12
        assert train.shape[1] == 6
        assert old_columns.sum() == 2
        assert len(fitted.sites) == 3
        assert not any('"W"' in name for name in fitted.features)
        for penalty_ratio in [.1, 1., 10.]:
            factors = np.full(train.shape[1], penalty_ratio)
            models = fit_glmnet_path(
                train, y, [.1, .01, 1e-5], alpha=1, penalty_factor=factors,
            )
            reference = fit_glmnet_path(
                train[:, old_columns], y, [.1, .01, 1e-5], alpha=1,
                penalty_factor=factors[old_columns],
            )
            for model, old in zip(models, reference, strict=True):
                np.testing.assert_array_equal(model.coef_[~old_columns], 0)
                for matrix in [train, held]:
                    np.testing.assert_allclose(
                        model.predict_proba(matrix), old.predict_proba(matrix[:, old_columns]),
                        rtol=0, atol=1e-7,
                    )


def test_saved_transform_preserves_its_original_feature_mask(tmp_path: Path) -> None:
    path = alignment(tmp_path / "aa", [
        ("s1_a", "AA"), ("s1_v", "VA"),
        ("s2_a", "AV"), ("s2_v", "VV"), ("held_a", "AA"),
    ])
    with AlignmentStore(path, ["s1", "s2", "held"]) as store:
        fitted = fit_amino_acid(
            store, ["s1", "s2"], settings(), y=np.array([0, 1]),
            groups=np.array(["g1", "g2"]),
        )
        # Older bundles omit a site's constant residue columns. Loading must
        # honor that saved mask rather than infer a new axis from known states.
        fitted.sites.active[0] = False
        before = fitted.matrix(store, ["held"])
        features = fitted.features
        joblib.dump(fitted, tmp_path / "legacy-transform.joblib")
        restored = joblib.load(tmp_path / "legacy-transform.joblib")
        assert restored.features == features
        np.testing.assert_array_equal(restored.matrix(store, ["held"]), before)
