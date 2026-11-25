import json
from contextlib import ExitStack
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import log_loss
from test_amino_acid import fixture_config
from test_combined import combined_config

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_parallel import run_fold
from phenoradar.amino_acid_run import predict_members
from phenoradar.combined import ExpressionInput
from phenoradar.config import load_and_resolve_config
from phenoradar.cv import SelectedCandidate, _fold_ids
from phenoradar.figures import _cv_loss_by_split
from phenoradar.model_selection import Candidate
from phenoradar.split import build_split_artifacts


@pytest.mark.parametrize("aggregation, combined", [
    ("aa_frequency", False), ("best_covered", False),
    ("tpm_weighted_frequency", False), ("tpm_weighted_frequency", True),
])
@pytest.mark.parametrize("probability_aggregation", ["mean", "median"])
def test_cv_losses_use_actual_sampled_species_and_frozen_model_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    aggregation: str, combined: bool, probability_aggregation: str,
) -> None:
    import phenoradar.amino_acid_run as run

    monkeypatch.chdir(tmp_path)
    config = load_and_resolve_config([combined_config(tmp_path, aggregation) if combined
                                     else fixture_config(tmp_path, aggregation)])
    config.preprocess.amino_acid.ranked_site_filter.method = "none"
    config.ensemble.probability_aggregation = probability_aggregation
    manifest = build_split_artifacts(config).split_manifest
    fold = _fold_ids(manifest)[0]
    train = manifest.filter((pl.col("fold_id") == fold) & (pl.col("pool") == "train")).sort(
        "species")
    valid = manifest.filter((pl.col("fold_id") == fold) & (pl.col("pool") == "validation"))
    # Different sizes and label proportions expose accidental pooling or double centering.
    indices = [np.array([0, 1, 2]), np.array([1, 2, 3, 4, 5])]
    monkeypatch.setattr(run, "_sample_training_sets", lambda *args: indices)
    candidates = [SelectedCandidate(Candidate(i, {
        "lambda": penalty, "alpha": 1 if combined else .95,
        **({"amino_acid_penalty_ratio": 1} if combined else {}),
    }), .5) for i, penalty in enumerate([.001, .02, .2])]
    trials = [{"candidate_index": item.candidate.candidate_index,
               "params_json": json.dumps(item.candidate.params)} for item in candidates]
    selections = {i: (candidates, trials, []) for i in range(2)}
    species = manifest["species"].unique().to_list()
    with ExitStack() as stack:
        store = stack.enter_context(AlignmentStore(
            Path(config.data.alignment_dir), species,
            sequence_tpm_path=config.data.sequence_tpm_path
            if aggregation == "tpm_weighted_frequency" else None,
        ))
        expression = stack.enter_context(ExpressionInput(
            config, species, train["species"].to_list())) if combined else None
        directory = run_fold(config, store, expression, manifest, fold, tmp_path / "fold", 0,
                             selections=selections)
        members, prediction, _, _ = joblib.load(directory / "result.joblib")
        actual_sets = pl.read_csv(directory / "tables/training_loss_by_sample_set.tsv",
                                  separator="\t").sort("sample_set_id")
        expected_train, expected_valid = [], []
        for sample_id, selected_rows in enumerate(indices):
            selected = train[selected_rows.tolist()]
            subset = [member for member in members if member.sample_set_id == sample_id]
            for rows, expected in [(selected, expected_train), (valid, expected_valid)]:
                probabilities = predict_members(config, store, rows["species"].to_list(), subset,
                                                expression=expression).join(
                    rows.select("species", "label"), on="species")
                expected.append(log_loss(probabilities["label"], probabilities["prob"],
                                         labels=[0, 1]))
        np.testing.assert_allclose(actual_sets["metric_value"], expected_train, atol=1e-12, rtol=0)
        assert actual_sets["n_species"].to_list() == [3, 5]
        assert set(train["species"]).isdisjoint(valid["species"])
        losses = pl.read_csv(directory / "tables/loss_by_split_cv.tsv", separator="\t")
        assert losses["split"].to_list() == ["train", "validation"]
        np.testing.assert_allclose(losses["metric_value"],
                                   [np.mean(expected_train), np.mean(expected_valid)],
                                   atol=1e-12, rtol=0)
        # Loss diagnostics must leave the all-model OOF ensemble unchanged.
        expected = predict_members(config, store, valid["species"].to_list(), members,
                                   expression=expression).sort("species")
        np.testing.assert_array_equal(prediction.sort("species")["prob"], expected["prob"])
    path = tmp_path / "cv_loss_by_split.svg"
    _cv_loss_by_split(losses, path)
    assert ">train<" in path.read_text() and ">validation<" in path.read_text()
