from __future__ import annotations

import json
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from test_amino_acid import alignment, settings, tpm_table

from phenoradar.amino_acid import AASite, AATransform, AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_prediction import prediction_members
from phenoradar.amino_acid_reporting import AAReportWriter
from phenoradar.amino_acid_run import AAMember, _save_bundle, predict_amino_acid, predict_members
from phenoradar.combined import ExpressionTransform
from phenoradar.config.schema import AppConfig
from phenoradar.feature_stability import build_feature_stability_tables
from phenoradar.glmnet import GlmnetLogisticRegression
from phenoradar.interpret import ModelFeatureEntry, build_interpretation_tables
from phenoradar.model_selection import Candidate


def model(beta: np.ndarray) -> GlmnetLogisticRegression:
    fitted = GlmnetLogisticRegression()
    fitted.coef_ = beta
    fitted.intercept_ = 0.37
    fitted.n_features_in_ = len(beta)
    fitted.classes_ = np.array([0, 1])
    return fitted


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_streamed_interpretation_matches_full_tables(
    tmp_path: Path, monkeypatch, combined: bool, zero: bool,
) -> None:
    import phenoradar.amino_acid_reporting as reporting

    monkeypatch.setattr(reporting, "REPORT_BLOCK_ROWS", 3)
    members = []
    for index, (fold, positions) in enumerate([
        ("1", [1, 2, 10]), ("1", [1, 4]), ("2", [2, 4]), ("10", [1, 2, 4]),
    ]):
        sites = []
        for position in positions:
            known = np.arange(20) < 3
            sites.append(AASite("OG_quoted_\"", position, 1, 0, 0, 0,
                                known=known, active=known.copy()))
        # This site passes composition filters but has constant residue frequencies.
        sites.append(AASite("OG_constant", 0, 1, 0, 0, 0,
                            known=np.arange(20) < 2, active=np.zeros(20, dtype=bool)))
        transform = AATransform(settings(), sites, [])
        expr = ExpressionTransform(AppConfig(), ["EXPR_A", "EXPR_B"], None) if combined else None
        n = 2 if combined else 0
        beta = np.zeros(n + sum(int(s.active.sum()) for s in sites))
        if not zero:
            beta[::3] = (index + 1) * (-1.0 if index % 2 else 1.0)
            beta[-1] = 1e-13  # Stability tolerance and exact prediction sparsity differ.
        members.append(AAMember(transform, model(beta), f"outer_fold_{fold}", index,
                                Candidate(index, {}), expr))

    outputs = {}
    for modality in (["expression", "amino_acid"] if combined else ["amino_acid"]):
        entries = []
        for member in members:
            n = member.expression_width
            columns = slice(0, n) if modality == "expression" else slice(n, None)
            fitted = copy(member.model)
            fitted.coef_ = member.model.coef_[columns]
            entries.append(ModelFeatureEntry(member.features[columns], fitted,
                                             member.scope.removeprefix("outer_fold_")))
        result = build_interpretation_tables(entries)
        for name in ["feature_importance", "coefficients", "feature_importance_by_fold",
                     "coefficients_by_fold"]:
            frame = getattr(result, name)
            if combined:
                frame = frame.with_columns(pl.lit(modality).alias("modality"))
                outputs[f"{modality}_{name}"] = frame
            outputs[name] = pl.concat([outputs[name], frame]) if name in outputs else frame
    retained = pl.DataFrame([
        {"scope": "outer_fold", "fold_id": m.scope.removeprefix("outer_fold_"),
         "sample_set_id": m.sample_set_id, "feature": f}
        for m in members for f in m.features
    ]).unique()
    stability = build_feature_stability_tables(
        feature_importance_by_fold=outputs["feature_importance_by_fold"],
        coefficients_by_fold=outputs["coefficients_by_fold"], retained_features=retained,
    )
    for name in ["by_feature", "by_fold_pair", "summary"]:
        outputs[f"feature_stability_{name}"] = getattr(stability, name)
    with AAReportWriter(tmp_path) as writer:
        for member in members:
            writer.add([member])
        top = writer.finalize(combined=combined, top_features=3)
        assert not writer.features.frames and not writer.sites.frames
        temp = writer.path
    assert not temp.exists()
    for name, expected in outputs.items():
        actual = pl.read_csv(tmp_path / f"{name}.tsv", separator="\t",
                             schema_overrides={c: t for c, t in expected.schema.items()
                                               if t == pl.String})
        assert_frame_equal(actual, expected, check_dtypes=False, abs_tol=1e-13)
    actual = pl.read_csv(tmp_path / "retained_features.tsv", separator="\t",
                         schema_overrides={"fold_id": pl.String})
    assert_frame_equal(actual.sort(actual.columns), retained.sort(retained.columns),
                       check_dtypes=False)
    expected = outputs["amino_acid_feature_importance" if combined else "feature_importance"]
    assert_frame_equal(top, expected.sort(["importance_mean", "feature"],
                                         descending=[True, False]).head(3), check_dtypes=False)
    site_stability = pl.read_csv(tmp_path / "amino_acid_site_stability.tsv", separator="\t")
    assert "OG_constant" not in site_stability["orthogroup"].to_list()


@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
@pytest.mark.parametrize("zero", [False, True])
def test_sparse_predictions_match_full_inputs_and_local_explanations(
    tmp_path: Path, monkeypatch, aggregation: str, zero: bool,
) -> None:
    import phenoradar.amino_acid_run as run

    train = [f"s{i}" for i in range(8)]
    records = [(f"{s}_main", ("A" if i % 2 else "V") * 80)
               for i, s in enumerate(train)]
    records += [("held_z", "X-V" * 26 + "AA"), ("held_a", "A" * 15 + "-" * 65)]
    path = alignment(tmp_path / "alignments", records)
    (path / "OG2.faa").write_text((path / "OG1.faa").read_text())
    tpm = tpm_table(tmp_path / "weights.tsv", [
        (name.rsplit("_", 1)[0], name, i + 1) for i, (name, _) in enumerate(records)
    ])
    config = AppConfig.model_validate({
        "data": {"feature_mode": "amino_acid", "alignment_dir": str(path),
                 "sequence_tpm_path": str(tpm)},
        "preprocess": {"amino_acid": {"aggregation": aggregation}},
        "figures": {"top_features": 3},
    })
    with AlignmentStore(path, [*train, "held", "absent"],
                        sequence_tpm_path=str(tpm)) as store:
        transform = fit_amino_acid(store, train, settings(aggregation),
                                  y=np.arange(8) % 2, groups=np.array(train))
        if aggregation == "best_covered":
            # Old bundles also need their full scoring mask frozen before pruning.
            transform.representative_columns = None
        width = len(transform.features)
        members = []
        for index in range(2):
            beta = np.zeros(width)
            if not zero:
                beta[-5 - index * 100] = 0.8 - index * 1.1
                beta[-1] = 1e-16
            members.append(AAMember(transform, model(beta), "final_refit", index,
                                    Candidate(index, {})))
        original_coef = [m.model.coef_.copy() for m in members]
        old = tmp_path / "full"
        new = tmp_path / "sparse"
        monkeypatch.setattr(run, "PREDICTION_BATCH_SIZE", 2)
        with monkeypatch.context() as context:
            context.setattr(run, "prediction_members", lambda members, **kwargs: members)
            expected = predict_members(config, store, ["held", "absent", "s0"], members,
                                       tables=old)
        actual = predict_members(config, store, ["held", "absent", "s0"], members, tables=new)
        assert_frame_equal(actual, expected, abs_tol=5e-15)
        for name in ["amino_acid_local_contributions.tsv"]:
            assert_frame_equal(pl.read_csv(new / name, separator="\t"),
                               pl.read_csv(old / name, separator="\t"), abs_tol=5e-15)
        compact = prediction_members(members, padding_sites=3)
        assert len(compact[0].transform.sites) <= 6
        assert compact[0].transform is compact[1].transform
        for member, original in zip(members, original_coef, strict=True):
            np.testing.assert_array_equal(member.model.coef_, original)
            assert len(member.transform.sites) == 160
        # A prediction-only store needs no coefficient-zero OG outside padding.
        needed = {s.orthogroup for m in compact for s in m.transform.sites}
        with AlignmentStore(path, [*train, "held", "absent"], only_orthogroups=needed,
                            sequence_tpm_path=str(tpm)) as narrow:
            again = predict_members(config, narrow, ["held", "absent", "s0"], members,
                                    tables=tmp_path / "narrow")
        assert_frame_equal(again, expected, abs_tol=5e-15)
        if zero:
            intercept = prediction_members(members)
            assert not intercept[0].transform.sites
            assert intercept[0].model.predict_proba(np.empty((2, 0))).shape == (2, 2)
            bundle = tmp_path / "bundle"
            _save_bundle(config, bundle, members, store, [*train, "held", "absent"])
            manifest = json.loads((bundle / "bundle_manifest.json").read_text())
            assert set(manifest["alignment_hashes"]) == {"OG1"}
            (path / "OG2.faa").unlink()
            metadata = tmp_path / "targets.tsv"
            metadata.write_text("species\nheld\n")
            dest = predict_amino_acid(bundle, tmp_path / "frozen", metadata_path=metadata)
            frozen = pl.read_csv(dest / "predict/tables/pred_predict.tsv", separator="\t")
            assert_frame_equal(frozen, expected.filter(pl.col("species") == "held"))
            audit = pl.read_csv(dest / "predict/tables/amino_acid_site_pss.tsv", separator="\t")
            assert set(audit["orthogroup"]) == {"OG1", "OG2"}


@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
def test_frequency_plot_restores_eligible_sites_even_when_fold_coefficients_are_zero(
    tmp_path: Path, aggregation: str,
) -> None:
    path = alignment(tmp_path / "aa", [("held_z", "AA-"), ("held_a", "--V")])
    tpm = tpm_table(tmp_path / "weights.tsv", [("held", "held_z", 1), ("held", "held_a", 3)])
    known = np.array([a in {0, 17} for a in range(20)])
    sites = [AASite("OG1", i, 1, 0, 0, 0, known=known, active=known) for i in range(3)]
    original = AATransform(settings(aggregation), sites, [])
    members = [AAMember(replace(original), model(np.zeros(6)), "outer_fold_1", 0, Candidate(0, {})),
               AAMember(replace(original), model(np.array([0, 0, 0, 0, 1, 0.])),
                        "outer_fold_2", 0, Candidate(0, {}))]
    with (AlignmentStore(path, ["held"], sequence_tpm_path=str(tpm)) as store,
          AAReportWriter(tmp_path / "reports") as writer):
        for member in members:
            writer.add([member])
        compact = prediction_members(members)
        assert not compact[0].transform.sites
        importance = pl.DataFrame({"feature": [sites[-1].feature(0)]})
        display = writer.frequency_members(compact, importance)
        assert len(display) == 2
        for member in display:
            assert [site.column for site in member.transform.sites] == [2]
            actual = member.transform.raw(store, ["held"])[0]
            expected = original.raw(store, ["held"])[0][:, [2]]
            np.testing.assert_array_equal(actual, expected)
