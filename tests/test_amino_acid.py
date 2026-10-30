from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from phenoradar.amino_acid import AlignmentError, AlignmentStore, fit_amino_acid
from phenoradar.amino_acid_run import predict_amino_acid, run_amino_acid
from phenoradar.cli import app
from phenoradar.config import load_and_resolve_config
from phenoradar.config.schema import AminoAcidConfig, AppConfig


def alignment(path: Path, records: list[tuple[str, str]]) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    (path / "OG1.faa").write_text("".join(f">{name}\n{seq}\n" for name, seq in records))
    return path


def settings(aggregation: str = "aa_frequency", **extra: object) -> AminoAcidConfig:
    return AminoAcidConfig.model_validate(
        {
            "aggregation": aggregation,
            "site_filter": {"min_sample_fraction": 0.5},
            **extra,
        }
    )


def test_site_score_exports_preserve_model_scores_ranks_and_zero_states(tmp_path: Path) -> None:
    from phenoradar.amino_acid import AASite, AATransform
    from phenoradar.amino_acid_outputs import write_site_scores
    from phenoradar.amino_acid_run import AAMember
    from phenoradar.glmnet import GlmnetLogisticRegression
    from phenoradar.model_selection import Candidate

    sites = []
    for column, states in enumerate([4, 2, 2, 2]):
        known = np.arange(20) < states
        sites.append(
            AASite("OG1", column, 1.0, 0, 0, 0.0, known=known, active=known.copy())
        )
    transform = AATransform(settings(), sites, [])
    coefficients = np.array([-1, -1, 1, 1, 3, 0, 0, 0, 2, 1], dtype=float)
    members = []
    for index, beta in enumerate([coefficients, -coefficients, np.zeros_like(coefficients)]):
        model = GlmnetLogisticRegression()
        model.coef_ = beta
        members.append(AAMember(transform, model, f"outer_fold_{index + 1}", 0, Candidate(0, {})))
    write_site_scores(tmp_path, members)
    for metric, expected, ranks in [
        ("pss", [4, 3, 0, 3], [1, 2, None, 2]),
        ("effect_range", [2, 3, 0, 1], [2, 1, None, 3]),
    ]:
        table = pl.read_csv(tmp_path / f"amino_acid_site_{metric}.tsv", separator="\t")
        assert table.height == 12
        first = table.filter(pl.col("scope") == "outer_fold_1").sort("alignment_pos_1based")
        assert first[metric].to_list() == expected
        assert first["rank"].to_list() == ranks
        assert first["n_known_residue_states"].to_list() == [4, 2, 2, 2]
        np.testing.assert_allclose(first["normalized_score"], np.array(expected) / sum(expected))
        # Opposite-signed models must not cancel before absolute values/ranges.
        second = table.filter(pl.col("scope") == "outer_fold_2").sort("alignment_pos_1based")
        assert first.drop("scope").equals(second.drop("scope"))
        zero = table.filter(pl.col("scope") == "outer_fold_3")
        assert zero["normalized_score"].to_list() == [0, 0, 0, 0]
        assert zero["rank"].null_count() == 4


def test_default_coverage_uses_ceiling_and_allows_singleton_residues(tmp_path: Path) -> None:
    path = alignment(
        tmp_path / "aa",
        [
            ("s0_g", "AAA"),
            ("s1_g", "AAA"),
            ("s2_g", "AAA"),
            ("s3_g", "VAA"),
            ("s4_g", "-VA"),
            ("s5_g", "--V"),
        ],
    )
    species = [f"s{i}" for i in range(6)]
    with AlignmentStore(path, species) as store:
        fitted = fit_amino_acid(
            store,
            species,
            AminoAcidConfig(),
            y=np.array([0, 1, 0, 1, 0, 1]),
            groups=np.array(["g0", "g0", "g1", "g1", "g2", "g2"]),
        )
    assert [site.column for site in fitted.sites] == [1, 2]
    assert [site.minor_samples for site in fitted.sites] == [1, 1]
    assert [site.sample_fraction for site in fitted.sites] == [5 / 6, 1.0]


@pytest.mark.parametrize("retired", ["min_minor_sample_count", "min_minor_group_count"])
def test_retired_minor_filters_are_rejected(retired: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AminoAcidConfig.model_validate({"site_filter": {retired: 100}})


@pytest.mark.parametrize("aggregation", ["best_covered", "aa_frequency", "tpm_weighted_frequency"])
def test_major_phenotype_residue_is_not_filtered_by_minor_group_support(
    tmp_path: Path, aggregation: str
) -> None:
    # A occurs in every trait-positive species across three groups and is major.
    # V occurs only in trait-negative species; its group count must not select sites.
    species = [f"s{i}" for i in range(6)]
    records = [(f"{sample}_g", residue) for sample, residue in zip(species, "VAVAAA", strict=True)]
    path = alignment(tmp_path / "aa", records)
    tpm = tpm_table(tmp_path / "tpm.tsv", [(sample, f"{sample}_g", 1) for sample in species])
    config = settings(aggregation)
    schemas, matrices, minor_groups = [], [], []
    with AlignmentStore(
        path,
        species,
        sequence_tpm_path=str(tpm) if aggregation == "tpm_weighted_frequency" else None,
    ) as store:
        for groups in [
            ["g0", "g0", "g1", "g1", "g2", "g2"],
            ["g0", "g0", "g0", "g1", "g2", "g2"],
        ]:
            fitted = fit_amino_acid(
                store,
                species,
                config,
                y=np.array([0, 1, 0, 1, 0, 1]),
                groups=np.array(groups),
            )
            assert [site.column for site in fitted.sites] == [0]
            assert fitted.sites[0].mean[0] == pytest.approx(4 / 6)
            schemas.append(fitted.schema().to_dicts())
            matrices.append(fitted.transform(store, species)[0])
            minor_groups.append(fitted.sites[0].minor_groups)
    assert minor_groups == [2, 1]
    assert schemas[0] == schemas[1]
    np.testing.assert_array_equal(matrices[0], matrices[1])


def test_gap_screen_precedes_frequency_and_counts_species(tmp_path: Path, monkeypatch) -> None:
    path = alignment(
        tmp_path / "aa",
        [
            ("s1_a", "-A-"),
            ("s1_b", "-V-"),
            ("s1_c", "-A-"),
            ("s2_a", "-V-"),
            ("s3_a", "---"),
            ("s4_a", "---"),
        ],
    )
    with AlignmentStore(path, ["s1", "s2", "s3", "s4"]) as store:
        assert store.call_counts(store.ogs["OG1"], ["s1", "s2", "s3", "s4"]).tolist() == [0, 2, 0]
        original = store.frequencies
        checked = []

        def narrow(og, species, columns, **kwargs):
            checked.extend(columns.tolist())
            assert set(columns) <= {1}
            return original(og, species, columns, **kwargs)

        monkeypatch.setattr(store, "frequencies", narrow)
        fitted = fit_amino_acid(
            store,
            ["s1", "s2", "s3", "s4"],
            settings(),
            y=np.array([0, 1, 0, 1]),
            groups=np.array(["a", "a", "b", "b"]),
        )
        assert checked
        assert fitted.sites[0].sample_fraction == 0.5
        raw, coverage, _ = fitted.raw(store, ["s1", "s2", "s3"])
        assert raw[0, 0, 0] == pytest.approx(2 / 3)
        assert raw[0, 0, 17] == pytest.approx(1 / 3)
        assert coverage[:, 0].tolist() == [1.0, 1.0, 0.0]
        assert np.isnan(raw[2]).all()


def test_training_state_and_values_ignore_held_out_changes(tmp_path: Path) -> None:
    path = alignment(tmp_path / "aa", [("s1_g", "A--"), ("s2_g", "V--"), ("held_g", "WWW")])
    signatures = []
    matrices = []
    for held in ["WWW", "---"]:
        alignment(path, [("s1_g", "A--"), ("s2_g", "V--"), ("held_g", held)])
        with AlignmentStore(path, ["s1", "s2", "held"]) as store:
            state = fit_amino_acid(
                store, ["s1", "s2"], settings(), y=np.array([0, 1]), groups=np.array(["a", "b"])
            )
            signatures.append(state.schema().to_dicts())
            matrices.append(state.transform(store, ["s1", "s2"])[0])
            unknown = state.transform(store, ["held"])[0]
            assert np.all(unknown == 0)
    assert signatures[0] == signatures[1]
    np.testing.assert_array_equal(*matrices)


def test_novel_fraction_is_neutral_without_losing_known_fraction(tmp_path: Path) -> None:
    path = alignment(
        tmp_path / "aa", [("s1_g", "A"), ("s2_g", "V"), ("held_a", "A"), ("held_w", "W")]
    )
    with AlignmentStore(path, ["s1", "s2", "held"]) as store:
        state = fit_amino_acid(
            store, ["s1", "s2"], settings(), y=np.array([0, 1]), groups=np.array(["a", "b"])
        )
        x, coverage, _ = state.transform(store, ["s1", "held"])
        np.testing.assert_allclose(x[1], 0.5 * x[0])
        assert coverage[1, 0] == 0.5


def test_representative_uses_filtered_sites_and_stable_id_tie(tmp_path: Path) -> None:
    path = alignment(
        tmp_path / "aa",
        [
            ("s1_z", "--AV"),
            ("s1_a", "AA--"),
            ("s2_g", "VV--"),
            ("held_z", "VVWW"),
            ("held_a", "AA--"),
        ],
    )
    with AlignmentStore(path, ["s1", "s2", "held"]) as store:
        state = fit_amino_acid(
            store,
            ["s1", "s2"],
            settings("best_covered"),
            y=np.array([0, 1]),
            groups=np.array(["a", "b"]),
        )
        assert [site.column for site in state.sites] == [0, 1]
        raw, _, representatives = state.raw(store, ["s1", "held"])
        assert [row["sequence_id"] for row in representatives] == ["s1_a", "held_a"]
        assert np.all(raw[:, :, 0] == 1)


def test_representatives_are_frozen_before_ranking_and_ignore_held_out(tmp_path: Path) -> None:
    species = [f"s{i}" for i in range(4)]
    records = [
        record
        for i, sample in enumerate(species)
        for record in [
            (f"{sample}_a", "V--" if i % 2 == 0 else "A--"),
            (f"{sample}_z", "AAA" if i % 2 == 0 else "VVV"),
        ]
    ]
    signatures = []
    for held in ["AAA", "VVV"]:
        path = alignment(tmp_path / "aa", records + [("held_a", "V--"), ("held_z", held)])
        with AlignmentStore(path, [*species, "held"]) as store:
            state = fit_amino_acid(
                store,
                species,
                settings(
                    "best_covered",
                    ranked_site_filter={
                        "method": "pair_aware",
                        "max_sites": 1,
                        "min_contrast_pairs": 2,
                    },
                ),
                y=np.array([0, 1, 0, 1]),
                groups=np.array(["g0", "g0", "g1", "g1"]),
                contrast_groups=np.array(["g0", "g0", "g1", "g1"]),
            )
            # All-gene frequencies cancel the signal at column 0. Frozen representatives
            # make all three columns equally predictive, so column 0 wins the stable tie.
            assert [site.column for site in state.sites] == [0]
            np.testing.assert_array_equal(state.scoring_columns("OG1"), [0, 1, 2])
            signatures.append(state.schema().to_dicts())
            raw, _, chosen = state.raw(store, ["s0", "held"])
            assert [row["sequence_id"] for row in chosen] == ["s0_z", "held_z"]
            assert [row["n_valid_scoring_sites"] for row in chosen] == [3, 3]
            assert [row["n_valid_model_sites"] for row in chosen] == [1, 1]
            assert raw[1, 0, 0 if held == "AAA" else 17] == 1
            # A pre-aa-3 fitted object has no scoring-mask attribute in its pickle.
            del state.__dict__["representative_columns"]
            assert state.raw(store, ["held"])[2][0]["sequence_id"] == "held_a"
    assert signatures[0] == signatures[1]


def test_constant_composition_is_not_a_variable_feature(tmp_path: Path) -> None:
    path = alignment(tmp_path / "aa", [("s1_a", "A"), ("s1_v", "V"), ("s2_a", "A"), ("s2_v", "V")])
    with (
        AlignmentStore(path, ["s1", "s2"]) as store,
        pytest.raises(AlignmentError, match="no variable features"),
    ):
        fit_amino_acid(
            store, ["s1", "s2"], settings(), y=np.array([0, 1]), groups=np.array(["a", "b"])
        )


@pytest.mark.parametrize(
    "records,message",
    [
        ([("s1_g", "AA"), ("s2_g", "A")], "Unequal"),
        ([("s1_g", "AA"), ("s1_g", "VV")], "Duplicate"),
        ([("s1_g", "A1")], "Invalid sequence character"),
        ([("unknown_g", "A")], "Cannot uniquely map"),
    ],
)
def test_input_validation(tmp_path: Path, records, message: str) -> None:
    path = alignment(tmp_path / "aa", records)
    with pytest.raises(AlignmentError, match=message):
        AlignmentStore(path, ["s1", "s2"])


def fixture_config(tmp_path: Path, aggregation: str, *, full: bool = True) -> Path:
    metadata = tmp_path / "metadata.tsv"
    lines = ["species\tC4\tcontrast_pair_id\tfamily"]
    records = []
    for i in range(8):
        lines.append(f"s{i}\t{i % 2}\tg{i // 2}\tF{i // 2}")
        records.append((f"s{i}_gene", "-X" + ("AA" if i % 2 == 0 else "VV") + "*--"))
    lines.extend(["external\t1\t\tF", "unknown\t\t\tF"])
    records.extend([("external_gene", "-XVV*--"), ("unknown_gene", "-XAA*--")])
    if aggregation == "tpm_weighted_frequency":
        records += [
            (name.replace("_gene", "_other"), seq.translate(str.maketrans("AV", "VA")))
            for name, seq in list(records)
        ]
    metadata.write_text("\n".join(lines) + "\n")
    aa = alignment(tmp_path / "alignments", records)
    config = {
        "data": {
            "feature_mode": "amino_acid",
            "metadata_path": str(metadata),
            "alignment_dir": str(aa),
            "sequence_tpm_cache_dir": str(tmp_path / "tpm_cache"),
        },
        "preprocess": {"amino_acid": {
            "aggregation": aggregation,
            "site_filter": {"min_sample_fraction": 0.5},
            "ranked_site_filter": {
                "method": "pair_aware",
                "max_sites": 10,
                "min_contrast_pairs": 1,
            },
        }},
        "sampling": {
            "strategy": "all_samples",
            "sampled_set_count": 1,
            "max_samples_per_label_per_group": None,
        },
        "model_selection": {
            "selected_candidate_count": 1,
            "inner_cv_strategy": "logo",
            "search_space": {"lambda": [0.01, 0.1], "alpha": [0.95]},
        },
        "runtime": {"execution_stage": "full_run" if full else "cv_only", "n_jobs": 1},
        "evaluation": {"group_bootstrap": {"enabled": True, "n_resamples": 10}},
    }
    if aggregation == "tpm_weighted_frequency":
        tpm = tmp_path / "sequence_tpm.tsv.gz"
        with gzip.open(tpm, "wt") as handle:
            handle.write("species\tsequence_id\ttpm\n")
            for name, _ in records:
                handle.write(
                    f"{name.rsplit('_', 1)[0]}\t{name}\t{9 if name.endswith('_gene') else 1}\n"
                )
        config["data"]["sequence_tpm_path"] = str(tpm)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("aggregation", ["aa_frequency", "best_covered", "tpm_weighted_frequency"])
def test_nested_cv_refit_bundle_and_coordinates(tmp_path: Path, aggregation: str) -> None:
    config_path = fixture_config(tmp_path, aggregation)
    config = load_and_resolve_config([config_path])
    destination = run_amino_acid(config, tmp_path / "run")
    oof = pl.read_csv(destination / "cv/tables/prediction_cv.tsv", separator="\t")
    assert oof.height == 8
    assert oof["prob"].is_finite().all()
    assert (destination / "cv/tables/model_selection_trials.tsv").exists()
    assert (destination / "cv/tables/group_bootstrap_metrics.tsv").exists()
    assert (destination / "cv/tables/amino_acid_site_stability.tsv").exists()
    assert (destination / "cv/figures/amino_acid_frequency_heatmap.svg").exists()
    coordinates = pl.read_csv(
        destination / "cv/tables/selected_site_sequence_map.tsv", separator="\t"
    )
    assert set(coordinates["alignment_pos_1based"]) == {3, 4}
    assert set(coordinates["sequence_pos_1based"]) == {2, 3}
    assert set(coordinates["residue"]) == {"A", "V"}
    if aggregation == "best_covered":
        assert (destination / "cv/tables/representative_sequences.tsv").exists()
    if aggregation == "tpm_weighted_frequency":
        assert set(coordinates["sequence_tpm"]) == {1, 9}
        np.testing.assert_allclose(sorted(set(coordinates["site_tpm_fraction"])), [0.1, 0.9])
        assert (destination / "model/tables/sequence_tpm_input.tsv").exists()
        # A portable bundle needs only selected OGs' weights, not the original input.
        Path(config.data.sequence_tpm_path).unlink()
    subset = tmp_path / "subset.tsv"
    subset.write_text("species\nunknown\n")
    predicted = predict_amino_acid(
        destination / "model_bundle", tmp_path / "predict", metadata_path=subset
    )
    after = pl.read_csv(predicted / "predict/tables/pred_predict.tsv", separator="\t")
    before = pl.read_csv(destination / "inference/tables/pred_inference.tsv", separator="\t")
    np.testing.assert_array_equal(after["prob"].to_numpy(), before["prob"].to_numpy())
    for metric in ["pss", "effect_range"]:
        name = f"amino_acid_site_{metric}.tsv"
        for stage in ["cv", "model"]:
            scores = pl.read_csv(destination / stage / "tables" / name, separator="\t")
            assert set(scores["alignment_pos_1based"]) == {3, 4}
            assert scores[metric].is_finite().all()
            assert (scores[metric] >= 0).all()
        fitted_scores = pl.read_csv(destination / "model/tables" / name, separator="\t")
        predicted_scores = pl.read_csv(predicted / "predict/tables" / name, separator="\t")
        # Model interpretation must not depend on the requested prediction species.
        assert fitted_scores.equals(predicted_scores)
    state_path = destination / "model_bundle/model_state.joblib"
    state_path.write_bytes(state_path.read_bytes() + b"tampered")
    with pytest.raises(AlignmentError, match="checksum"):
        predict_amino_acid(destination / "model_bundle", tmp_path / "bad")


def test_representative_bundle_and_coordinates_keep_pre_rank_gene_choices(tmp_path: Path) -> None:
    config_path = fixture_config(tmp_path, "best_covered")
    payload = yaml.safe_load(config_path.read_text())
    fasta = Path(payload["data"]["alignment_dir"]) / "OG1.faa"
    lines = fasta.read_text().splitlines()
    for name, sequence in zip(lines[::2], lines[1::2], strict=True):
        other = name.removesuffix("_gene") + "_a"
        residue = "V" if sequence[2] == "A" else "A"
        with fasta.open("a") as handle:
            handle.write(f"{other}\n-X{residue}-*--\n")
    payload["preprocess"]["amino_acid"]["ranked_site_filter"]["max_sites"] = 1
    config_path.write_text(yaml.safe_dump(payload))
    destination = run_amino_acid(load_and_resolve_config([config_path]), tmp_path / "run")
    manifest = json.loads((destination / "model_bundle/bundle_manifest.json").read_text())
    assert manifest["bundle_format_version"] == "aa-3"
    assert manifest["representative_selection"] == "quality_filtered_sites"
    scoring = pl.read_csv(
        destination / "model/tables/representative_scoring_sites.tsv", separator="\t"
    )
    assert set(scoring["alignment_pos_1based"]) == {3, 4}
    subset = tmp_path / "subset.tsv"
    subset.write_text("species\nunknown\n")
    predicted = predict_amino_acid(
        destination / "model_bundle", tmp_path / "prediction", metadata_path=subset
    )
    selected = pl.read_csv(
        predicted / "predict/tables/prediction_representative_sequences.tsv", separator="\t"
    )
    assert selected["sequence_id"].to_list() == ["unknown_gene"]
    assert selected["n_valid_scoring_sites"].to_list() == [2]
    assert selected["n_valid_model_sites"].to_list() == [1]
    coordinates = pl.read_csv(
        predicted / "predict/tables/selected_site_sequence_map.tsv", separator="\t"
    )
    assert coordinates["sequence_id"].to_list() == ["unknown_gene"]
    assert coordinates["alignment_pos_1based"].to_list() == [3]
    before = pl.read_csv(destination / "inference/tables/pred_inference.tsv", separator="\t")
    after = pl.read_csv(predicted / "predict/tables/pred_predict.tsv", separator="\t")
    np.testing.assert_array_equal(after["prob"].to_numpy(), before["prob"].to_numpy())


def test_cli_run_and_predict_need_no_tpm(tmp_path: Path, monkeypatch) -> None:
    config = fixture_config(tmp_path, "aa_frequency")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(config), "--quiet"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    runs = list((tmp_path / "runs").glob("*run*"))
    assert len(runs) == 1
    result = runner.invoke(
        app, ["predict", "--model-bundle", str(runs[0] / "model_bundle"), "--quiet"]
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    result = runner.invoke(
        app,
        ["predict", "--model-bundle", str(runs[0] / "model_bundle"), "-c", str(config), "--quiet"],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    metadata = json.loads((runs[0] / "run_metadata.json").read_text())
    assert metadata["feature_mode"] == "amino_acid"


def tpm_table(path: Path, rows: list[tuple[str, str, object]]) -> Path:
    path.write_text("species\tsequence_id\ttpm\n" + "".join(f"{s}\t{g}\t{v}\n" for s, g, v in rows))
    return path


@pytest.mark.parametrize("cached", [False, True])
def test_weighted_frequencies_gaps_zero_and_scale(tmp_path: Path, cached: bool) -> None:
    path = alignment(
        tmp_path / "aa",
        [
            ("s1_a", "AAA-"),
            ("s1_v", "V-XX"),
            ("s1_z", "WWWW"),
            ("s2_a", "AAAA"),
            ("s2_v", "VVVV"),
        ],
    )
    for scale in [1, 1e300]:
        tpm = tpm_table(
            tmp_path / "tpm.tsv",
            [
                ("s1", "s1_a", scale),
                ("s1", "s1_v", 3 * scale),
                ("s1", "s1_z", 0),
                ("s2", "s2_a", 0),
                ("s2", "s2_v", 0),
                ("unrelated", "other", "not-a-number"),
            ],
        )
        with AlignmentStore(
            path,
            ["s1", "s2"],
            sequence_tpm_path=str(tpm),
            sequence_tpm_cache_dir=tmp_path / "cache" if cached else None,
        ) as store:
            og = store.ogs["OG1"]
            q, c, _ = store.frequencies(og, ["s1", "s2"], np.arange(4), weighted=True)
            np.testing.assert_allclose(q[0, 0, [0, 17, 18]], [0.25, 0.75, 0])
            np.testing.assert_allclose(q[0, 1:3, 0], [1, 1])
            np.testing.assert_allclose(c[0], [1, 0.25, 0.25, 0])
            assert np.isnan(q[0, 3]).all()
            assert np.isnan(q[1]).all()
            assert store.call_counts(og, ["s1", "s2"], weighted=True).tolist() == [1, 1, 1, 0]
            # Unweighted mode continues to count zero-TPM genes.
            unweighted, _, _ = store.frequencies(og, ["s1"], np.arange(4))
            assert unweighted[0, 3, 18] == 1


@pytest.mark.parametrize(
    "values,message",
    [
        ([], "Missing sequence TPM"),
        ([1, 2], "Duplicate sequence TPM"),
        ([-1], "finite and nonnegative"),
        (["nan"], "finite and nonnegative"),
        (["inf"], "finite and nonnegative"),
        (["NA"], "Invalid sequence TPM"),
    ],
)
@pytest.mark.parametrize("cached", [False, True])
def test_weighted_tpm_validation(tmp_path: Path, values, message: str, cached: bool) -> None:
    path = alignment(tmp_path / "aa", [("s1_a", "A")])
    tpm = tpm_table(tmp_path / "tpm.tsv", [("s1", "s1_a", v) for v in values])
    for _ in range(2):
        with pytest.raises(AlignmentError, match=message):
            AlignmentStore(
                path,
                ["s1"],
                sequence_tpm_path=str(tpm),
                sequence_tpm_cache_dir=tmp_path / "cache" if cached else None,
            )


def test_weighted_selection_uses_weights_and_training_only(tmp_path: Path) -> None:
    path = alignment(
        tmp_path / "aa",
        [
            ("s1_a", "AA"),
            ("s1_v", "V-"),
            ("s2_a", "AA"),
            ("s2_v", "V-"),
            ("held_a", "WW"),
            ("held_v", "V-"),
        ],
    )
    schemas = []
    xs = []
    for held_weights in [(100, 0), (0, 100)]:
        tpm = tpm_table(
            tmp_path / "tpm.tsv",
            [
                ("s1", "s1_a", 3),
                ("s1", "s1_v", 1),
                ("s2", "s2_a", 0),
                ("s2", "s2_v", 10),
                ("held", "held_a", held_weights[0]),
                ("held", "held_v", held_weights[1]),
            ],
        )
        with AlignmentStore(path, ["s1", "s2", "held"], sequence_tpm_path=str(tpm)) as store:
            state = fit_amino_acid(
                store,
                ["s1", "s2"],
                settings(
                    "tpm_weighted_frequency",
                    site_filter={
                        "min_sample_fraction": 1.0,
                    },
                ),
                y=np.array([0, 1]),
                groups=np.array(["a", "b"]),
            )
            assert [s.column for s in state.sites] == [0]
            schemas.append(state.schema().to_dicts())
            xs.append(state.transform(store, ["s1", "s2"])[0])
            raw, coverage, _ = state.raw(store, ["held"])
            x, information, _ = state.transform(store, ["held"])
            assert coverage[0, 0] == 1
            if held_weights[1] == 0:
                assert information[0, 0] == 0
                assert np.all(x == 0)  # Previously unseen W has neutral effect.
            else:
                assert raw[0, 0, 17] == 1
    assert schemas[0] == schemas[1]
    np.testing.assert_array_equal(*xs)


def test_weighted_bundle_override_and_checksums(tmp_path: Path, monkeypatch) -> None:
    config_path = fixture_config(tmp_path, "tpm_weighted_frequency")
    config = load_and_resolve_config([config_path])
    destination = run_amino_acid(config, tmp_path / "run")
    subset = tmp_path / "subset.tsv"
    subset.write_text("species\nunknown\n")
    override = tpm_table(
        tmp_path / "new_tpm.tsv",
        [
            ("unknown", "unknown_gene", 1),
            ("unknown", "unknown_other", 9),
        ],
    )
    predicted = predict_amino_acid(
        destination / "model_bundle",
        tmp_path / "override",
        metadata_path=subset,
        sequence_tpm_path=override,
    )
    probability = pl.read_csv(predicted / "predict/tables/pred_predict.tsv", separator="\t")[
        "prob"
    ][0]
    original = pl.read_csv(destination / "inference/tables/pred_inference.tsv", separator="\t")[
        "prob"
    ][0]
    external = pl.read_csv(
        destination / "external_test/tables/pred_external_test.tsv", separator="\t"
    )["prob"][0]
    assert probability > original
    assert probability == pytest.approx(external)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(destination / "model_bundle"),
            "--metadata-path",
            str(subset),
            "--sequence-tpm-path",
            str(override),
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    payload = destination / "model_bundle/sequence_tpm.tsv.gz"
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(AlignmentError, match="checksum"):
        predict_amino_acid(destination / "model_bundle", tmp_path / "bad")


def test_tpm_cache_is_automatic_without_a_config_key(tmp_path: Path, monkeypatch) -> None:
    config_path = fixture_config(tmp_path, "tpm_weighted_frequency")
    payload = yaml.safe_load(config_path.read_text())
    payload["data"].pop("sequence_tpm_cache_dir")
    config_path.write_text(yaml.safe_dump(payload))
    monkeypatch.chdir(tmp_path)
    config = load_and_resolve_config([config_path])
    first = run_amino_acid(config, tmp_path / "first")
    second = run_amino_acid(config, tmp_path / "second")
    cache = tmp_path / "runs/.phenoradar_cache/sequence_tpm"
    assert list(cache.glob("*/part-*.parquet"))
    first_input = pl.read_csv(first / "model/tables/sequence_tpm_input.tsv", separator="\t")
    second_input = pl.read_csv(second / "model/tables/sequence_tpm_input.tsv", separator="\t")
    assert first_input.filter(pl.col("scope") == "training_pool")["cache_hit"].to_list() == [False]
    assert second_input["cache_hit"].all()
    first_oof = pl.read_csv(first / "cv/tables/prediction_cv.tsv", separator="\t").sort("species")
    second_oof = pl.read_csv(second / "cv/tables/prediction_cv.tsv", separator="\t").sort("species")
    np.testing.assert_array_equal(first_oof["prob"].to_numpy(), second_oof["prob"].to_numpy())


def test_weighted_config_requires_tpm() -> None:
    with pytest.raises(ValueError, match="sequence_tpm_path"):
        AppConfig.model_validate(
            {
                "data": {"feature_mode": "amino_acid", "alignment_dir": "aa"},
                "preprocess": {"amino_acid": {"aggregation": "tpm_weighted_frequency"}},
            }
        )


def test_sequence_config_preserves_expression_defaults() -> None:
    assert AppConfig().data.feature_mode == "expression"
    with pytest.raises(ValueError, match="alignment_dir"):
        AppConfig.model_validate({"data": {"feature_mode": "amino_acid"}})


@pytest.mark.parametrize("strategy", ["random", "tpe"])
def test_shared_search_strategies(tmp_path: Path, strategy: str) -> None:
    config_path = fixture_config(tmp_path, "aa_frequency", full=False)
    payload = yaml.safe_load(config_path.read_text())
    payload["model_selection"].update(
        search_strategy=strategy, trial_count=2, selection_rule="one_se"
    )
    config_path.write_text(yaml.safe_dump(payload))
    run_amino_acid(load_and_resolve_config([config_path]), tmp_path / "run")
    selected = pl.read_csv(tmp_path / "run/cv/tables/model_selection_selected.tsv", separator="\t")
    assert selected.height == 4
    assert selected["score"].is_finite().all()
    assert set(selected["selection_rule"]) == {"one_se"}


def test_condition_comparison_uses_shared_split(tmp_path: Path, monkeypatch) -> None:
    config_path = fixture_config(tmp_path, "aa_frequency", full=False)
    payload = yaml.safe_load(config_path.read_text())
    payload["preprocess"]["amino_acid"]["aggregation"] = [
        "best_covered",
        "aa_frequency",
        "tpm_weighted_frequency",
    ]
    records = [
        line[1:]
        for line in (Path(payload["data"]["alignment_dir"]) / "OG1.faa").read_text().splitlines()
        if line.startswith(">")
    ]
    tpm = tpm_table(tmp_path / "tpm.tsv", [(r.rsplit("_", 1)[0], r, 1) for r in records])
    payload["data"]["sequence_tpm_path"] = str(tpm)
    config_path.write_text(yaml.safe_dump(payload))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "-c", str(config_path), "--quiet"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    studies = list((tmp_path / "runs").glob("*study*"))
    assert len(studies) == 1
    fingerprints = [
        json.loads(p.read_text())["split_fingerprint"]
        for p in studies[0].glob("conditions/*/run_metadata.json")
    ]
    assert len(fingerprints) == 3
    assert len(set(fingerprints)) == 1
    metadata = [
        json.loads(p.read_text()) for p in studies[0].glob("conditions/*/run_metadata.json")
    ]
    assert len({m["dataset_fingerprint"] for m in metadata}) == 2
    assert (studies[0] / "tables/condition_metrics.tsv").exists()


def test_frozen_coordinate_hash_is_required(tmp_path: Path) -> None:
    config_path = fixture_config(tmp_path, "aa_frequency")
    config = load_and_resolve_config([config_path])
    run_amino_acid(config, tmp_path / "run")
    aa = Path(config.data.alignment_dir) / "OG1.faa"
    aa.write_text(aa.read_text().replace("-XAA*--", "-XAW*--", 1))
    with pytest.raises(AlignmentError, match="coordinate identity"):
        predict_amino_acid(tmp_path / "run/model_bundle", tmp_path / "bad")


def test_empty_site_after_representative_choice_is_filtered(tmp_path: Path) -> None:
    # First two sites win gene selection; the third site remains in the fixed
    # scoring mask but has no observation in any chosen representative.
    path = alignment(
        tmp_path / "aa", [("s1_a", "AA-"), ("s1_b", "--A"), ("s2_a", "VV-"), ("s2_b", "--V")]
    )
    with AlignmentStore(path, ["s1", "s2"]) as store:
        state = fit_amino_acid(
            store,
            ["s1", "s2"],
            settings("best_covered"),
            y=np.array([0, 1]),
            groups=np.array(["a", "b"]),
        )
        assert [site.column for site in state.sites] == [0, 1]
        np.testing.assert_array_equal(state.scoring_columns("OG1"), [0, 1, 2])
        from phenoradar.amino_acid_run import AAMember, predict_members
        from phenoradar.glmnet import GlmnetLogisticRegression
        from phenoradar.model_selection import Candidate

        x, _, _ = state.transform(store, ["s1", "s2"])
        model = GlmnetLogisticRegression(lambda_=0.1).fit(x, np.array([0, 1]))
        config = AppConfig.model_validate(
            {
                "data": {"feature_mode": "amino_acid", "alignment_dir": str(path)},
                "abstention": {"enabled": True},
            }
        )
        result = predict_members(
            config, store, ["s1", "s2"], [AAMember(state, model, "test", 0, Candidate(0, {}))]
        )
        assert result["prob"].is_finite().all()
        assert result["information_coverage"].min() == pytest.approx(1)


def test_automatic_species_assignment_supports_exact_ids_and_gene_suffixes(tmp_path: Path) -> None:
    path = alignment(
        tmp_path / "aa",
        [("Genus_species", "A"), ("Genus_species_gene_isoform_1", "V"), ("Other_taxon_g", "A")],
    )
    with AlignmentStore(path, ["Genus_species", "Other_taxon"]) as store:
        assert store.ogs["OG1"].species_rows == {"Genus_species": [0, 1], "Other_taxon": [2]}


def test_automatic_species_assignment_rejects_ambiguous_metadata_prefixes(tmp_path: Path) -> None:
    path = alignment(tmp_path / "aa", [("s1_sub_gene", "A")])
    with pytest.raises(AlignmentError, match="Cannot uniquely map.*ambiguous IDs"):
        AlignmentStore(path, ["s1", "s1_sub"])


def test_site_limit_is_a_guard_without_ranking(tmp_path: Path) -> None:
    path = alignment(tmp_path / "aa", [("s1_g", "AA"), ("s2_g", "VV")])
    with (
        AlignmentStore(path, ["s1", "s2"]) as store,
        pytest.raises(AlignmentError, match="exceed max_sites"),
    ):
        fit_amino_acid(
            store,
            ["s1", "s2"],
            settings(ranked_site_filter={"method": "none", "max_sites": 1}),
            y=np.array([0, 1]),
            groups=np.array(["a", "b"]),
        )
