from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from glum import GeneralizedLinearRegressor
from sklearn.preprocessing import StandardScaler

from phenoradar.candidate_evidence import build_candidate_evidence_artifacts
from phenoradar.config import load_and_resolve_config
from phenoradar.cv import FinalModelEntry, _build_cv_species_evidence
from phenoradar.figures import (
    write_candidate_evidence_figures,
    write_cv_species_evidence_figures,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _candidate_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    metadata = _write(
        tmp_path / "metadata.tsv",
        "\n".join(
            [
                "species\tC4\tfamily",
                "known_0a\t0\tFamily one",
                "known_0b\t0\tFamily one",
                "known_1a\t1\tFamily two",
                "known_1b\t1\tFamily two",
                "candidate_a\t\tFamily three",
            ]
        )
        + "\n",
    )
    tpm = _write(
        tmp_path / "tpm.tsv",
        "\n".join(
            [
                "species\torthogroup\ttpm",
                "known_0a\tOG1\t1",
                "known_0a\tOG2\t10",
                "known_0b\tOG1\t2",
                "known_0b\tOG2\t8",
                "known_1a\tOG1\t8",
                "known_1a\tOG2\t2",
                "known_1b\tOG1\t9",
                "known_1b\tOG2\t1",
                "candidate_a\tOG1\t16",
                "candidate_a\tOG2\t1",
            ]
        )
        + "\n",
    )
    config = _write(
        tmp_path / "config.yml",
        f"""
data:
  metadata_path: {metadata}
  tpm_path: {tpm}
preprocess:
  ranked_feature_filter:
    method: none
""".strip()
        + "\n",
    )
    return metadata, tpm, config


@pytest.mark.parametrize("group_col", ["family", "order"])
def test_build_candidate_evidence_uses_candidate_local_contribution(
    tmp_path: Path, group_col: str
) -> None:
    _metadata, _tpm, config_path = _candidate_fixture(tmp_path)
    config = load_and_resolve_config([config_path])
    config.summary.group_col = group_col
    split_manifest = pl.DataFrame(
        {
            "species": ["known_0a", "known_0b", "known_1a", "known_1b"],
            "pool": ["train", "validation", "train", "validation"],
            "label": [0, 0, 1, 1],
        }
    )
    train_transformed = np.log1p(np.array([[1.0, 10.0], [2.0, 8.0], [8.0, 2.0], [9.0, 1.0]]))
    scaler = StandardScaler().fit(train_transformed)
    model = GeneralizedLinearRegressor(family="binomial")
    model.coef_ = np.array([1.5, -0.5], dtype=float)
    final_refit = SimpleNamespace(
        pred_inference=pl.DataFrame(
            {
                "species": ["candidate_a"],
                "prob": [0.97],
                "pred_label_fixed_threshold": [1],
            }
        ),
        model_entries=[
            FinalModelEntry(
                feature_names=["OG1", "OG2"],
                scaler=scaler,
                model=model,
            )
        ],
        transform_feature_names=["OG1", "OG2"],
    )
    cross_fold = pl.DataFrame(
        {
            "fold_id": ["1", "2"],
            "species": ["candidate_a", "candidate_a"],
            "prob": [0.90, 0.95],
        }
    )

    artifacts = build_candidate_evidence_artifacts(
        config=config,
        split_manifest=split_manifest,
        final_refit=final_refit,  # type: ignore[arg-type]
        cross_fold_predictions=cross_fold,
        top_features=1,
    )

    assert artifacts.candidates.row(0, named=True)["family"] == "Family three"
    assert artifacts.warnings == []
    assert artifacts.features.height == 1
    assert artifacts.features.row(0, named=True)["feature"] == "OG1"
    assert artifacts.features.row(0, named=True)["local_rank"] == 1
    assert artifacts.reference_expression.height == 4
    assert artifacts.cross_fold_predictions.height == 2


def test_write_candidate_evidence_figures_writes_probability_bin_pdf_and_manifest(
    tmp_path: Path,
) -> None:
    candidates = pl.DataFrame(
        {
            "species": ["Candidate species"],
            "prob": [0.97],
            "family": ["Family three"],
        }
    )
    features = pl.DataFrame(
        {
            "species": ["Candidate species", "Candidate species"],
            "feature": ["OG1", "OG2"],
            "local_rank": [1, 2],
            "contribution_mean": [1.2, -0.4],
            "candidate_log2_tpm_plus1": [4.1, 1.0],
        }
    )
    reference_expression = pl.DataFrame(
        {
            "species": ["s0a", "s0b", "s1a", "s1b"] * 2,
            "label": [0, 0, 1, 1] * 2,
            "feature": ["OG1"] * 4 + ["OG2"] * 4,
            "log2_tpm_plus1": [0.8, 1.2, 3.6, 4.0, 2.8, 3.0, 0.7, 1.1],
        }
    )
    cross_fold = pl.DataFrame(
        {
            "fold_id": ["1", "2", "3"],
            "species": ["Candidate species"] * 3,
            "prob": [0.88, 0.92, 0.95],
        }
    )
    annotations = pl.DataFrame(
        {
            "feature": ["OG1", "OG2"],
            "orthogroup_annotation": [
                "beta carbonic anhydrase",
                "phosphoenolpyruvate carboxylase",
            ],
        }
    )

    manifest, warnings = write_candidate_evidence_figures(
        run_dir=tmp_path,
        candidates=candidates,
        features=features,
        reference_expression=reference_expression,
        cross_fold_predictions=cross_fold,
        trait_name="C4",
        orthogroup_annotations=annotations,
        parallel_workers=1,
    )

    assert warnings == []
    assert manifest.height == 1
    row = manifest.row(0, named=True)
    assert row["probability_bin"] == "p_095_100"
    assert row["n_cross_fold_predictions"] == 3
    root = tmp_path / "inference" / "figures" / "candidate_evidence"
    pdf_path = root / str(row["figure_path"])
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert (root / "candidate_manifest.tsv").exists()


def test_build_cv_species_evidence_uses_held_out_fold_models_and_training_reference(
    tmp_path: Path,
) -> None:
    _metadata, _tpm, config_path = _candidate_fixture(tmp_path)
    config = load_and_resolve_config([config_path])
    x_train_raw = np.array(
        [[1.0, 10.0], [2.0, 8.0], [8.0, 2.0], [9.0, 1.0]], dtype=float
    )
    scaler = StandardScaler().fit(np.log1p(x_train_raw))
    model = GeneralizedLinearRegressor(family="binomial")
    model.coef_ = np.array([1.5, -0.5], dtype=float)

    species, features, reference, warnings = _build_cv_species_evidence(
        config=config,
        fold_id="2",
        model_entries=[
            FinalModelEntry(
                feature_names=["OG1", "OG2"],
                scaler=scaler,
                model=model,
            )
        ],
        feature_names=["OG1", "OG2"],
        train_species=["known_0a", "known_0b", "known_1a", "known_1b"],
        valid_species=["held_out_species"],
        valid_group_ids=["held_out_group"],
        y_train=np.array([0, 0, 1, 1], dtype=int),
        y_valid=np.array([1], dtype=int),
        x_train_raw=x_train_raw,
        x_valid_raw=np.array([[1.0, 10.0]], dtype=float),
        reference_indices=np.array([0, 2], dtype=int),
        mean_prob=np.array([0.1], dtype=float),
        uncertainty_std=None,
        fixed_threshold=0.5,
        top_features=1,
    )

    assert warnings == []
    assert species.row(0, named=True) == {
        "fold_id": "2",
        "species": "held_out_species",
        "group_id": "held_out_group",
        "label": 1,
        "pred_label": 0,
        "confusion_group": "FN",
        "prob": 0.1,
        "log_loss": pytest.approx(-np.log(0.1)),
        "uncertainty_std": None,
        "n_models": 1,
    }
    assert features.height == 1
    assert features.row(0, named=True)["feature"] == "OG1"
    assert reference.height == 2
    assert set(reference.get_column("species")) == {"known_0a", "known_1a"}
    assert set(reference.get_column("target_species")) == {"held_out_species"}


def test_write_cv_species_evidence_figures_writes_error_pdf_and_manifest(
    tmp_path: Path,
) -> None:
    species_evidence = pl.DataFrame(
        {
            "fold_id": ["2"],
            "species": ["Held out species"],
            "group_id": ["g2"],
            "label": [1],
            "pred_label": [0],
            "confusion_group": ["FN"],
            "prob": [0.12],
            "log_loss": [-np.log(0.12)],
            "uncertainty_std": [0.03],
            "n_models": [2],
        }
    )
    features = pl.DataFrame(
        {
            "fold_id": ["2", "2"],
            "species": ["Held out species", "Held out species"],
            "feature": ["OG1", "OG2"],
            "local_rank": [1, 2],
            "contribution_mean": [-1.2, 0.4],
            "target_log2_tpm_plus1": [1.0, 3.2],
        }
    )
    reference_expression = pl.DataFrame(
        {
            "fold_id": ["2"] * 8,
            "target_species": ["Held out species"] * 8,
            "species": ["s0a", "s0b", "s1a", "s1b"] * 2,
            "label": [0, 0, 1, 1] * 2,
            "feature": ["OG1"] * 4 + ["OG2"] * 4,
            "log2_tpm_plus1": [0.8, 1.2, 3.6, 4.0, 2.8, 3.0, 0.7, 1.1],
        }
    )
    ensemble_model_probs = pl.DataFrame(
        {
            "fold_id": ["2", "2"],
            "model_index": [0, 1],
            "species": ["Held out species", "Held out species"],
            "prob": [0.10, 0.14],
        }
    )

    manifest, warnings = write_cv_species_evidence_figures(
        run_dir=tmp_path,
        species_evidence=species_evidence,
        features=features,
        reference_expression=reference_expression,
        ensemble_model_probs=ensemble_model_probs,
        trait_name="C4",
        parallel_workers=1,
    )

    assert warnings == []
    assert manifest.height == 1
    row = manifest.row(0, named=True)
    assert row["confusion_group"] == "FN"
    assert row["n_model_predictions"] == 2
    root = tmp_path / "cv" / "figures" / "species_evidence"
    pdf_path = root / str(row["figure_path"])
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert (root / "species_manifest.tsv").exists()
