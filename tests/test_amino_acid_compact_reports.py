from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from test_amino_acid import settings
from test_amino_acid_streaming import model

from phenoradar.amino_acid import AASite, AATransform
from phenoradar.amino_acid_compact_reports import read_model_coefficients, write_dictionary
from phenoradar.amino_acid_outputs import write_site_effects
from phenoradar.amino_acid_reporting import AAReportWriter
from phenoradar.amino_acid_run import AAMember
from phenoradar.combined import ExpressionTransform
from phenoradar.config.schema import AppConfig
from phenoradar.model_selection import Candidate


def members_for_test(combined: bool = False, zero: bool = False) -> list[AAMember]:
    result = []
    for i, fold in enumerate(["1", "1", "2", "10"]):
        sites = [AASite('OG_é"', j, 1, 0, 0, 0,
                        known=np.arange(20) < 3, active=np.arange(20) < 2)
                 for j in range(300 + i)]
        expression = ExpressionTransform(AppConfig(), ["EXPR_A", "EXPR_B"], None) \
            if combined else None
        n = 2 if combined else 0
        beta = np.zeros(n + len(sites) * 2)
        if not zero:
            beta[:4] = [1.0, -2.0, 1e-13, -0.0]
            if i == 1:
                beta *= -1  # Exact ensemble cancellation must not erase model coefficients.
        result.append(AAMember(AATransform(settings(), sites, []), model(beta),
                               f"outer_fold_{fold}", i, Candidate(i, {}), expression))
    return result


def test_default_and_older_pickled_report_settings() -> None:
    from phenoradar.amino_acid_compact_reports import compact_reports

    config = AppConfig()
    assert compact_reports(config)
    del config.report.__dict__['amino_acid_tables']
    assert not compact_reports(config)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_compact_values_membership_and_global_summaries_match_full(
    tmp_path: Path, combined: bool, zero: bool,
) -> None:
    members = members_for_test(combined, zero)
    tops = []
    for compact in [False, True]:
        destination = tmp_path / ("compact" if compact else "full")
        with AAReportWriter(destination, compact=compact) as writer:
            writer.add(members)
            tops.append(writer.finalize(combined=combined, top_features=3))
    assert_frame_equal(*tops, check_exact=True)
    compact, full = tmp_path / "compact", tmp_path / "full"
    for directory in [compact, full]:
        sparsity = pl.read_csv(directory / "model_sparsity.tsv", separator="\t")
        assert sparsity["n_nonzero_features"].to_list() == ([0] * 4 if zero else [2] * 4)
        assert sparsity.filter(pl.col("fold_id") == 1).height == 2
        assert sparsity["scope"].unique().to_list() == ["outer_fold"]
    values = read_model_coefficients(compact)
    actual = {(r['model_id'], r['feature']): r['coefficient'] for r in values.to_dicts()}
    expected = {(i, feature): float(value) for i, member in enumerate(members)
                for feature, value in zip(member.features, member.model.coef_, strict=True)
                if value != 0}
    assert actual.keys() == expected.keys()
    assert all(actual[key].hex() == expected[key].hex() for key in expected)
    for row in values.to_dicts():
        member = members[row['model_id']]
        offset = member.expression_width
        segment = member.model.coef_[:offset] if row['modality'] == 'expression' \
            else member.model.coef_[offset:]
        total = float(np.abs(segment).sum())
        importance = row['coefficient'] * 0 if np.isclose(total, 0) \
            else abs(row['coefficient']) / total
        assert row['importance'].hex() == importance.hex()
    axes = pl.read_parquet(compact / "feature_dictionary.parquet").join(
        pl.read_csv(compact / "report_orthogroups.tsv", separator='\t'), on='orthogroup_id',
        how='left')
    retained = set()
    for row in axes.to_dicts():
        feature = (row['expression_feature'] if row['modality'] == 'expression' else
                   json.dumps([row['orthogroup'], row['alignment_pos_1based'], row['residue']],
                              separators=(',', ':')))
        for i in range(len(members)):
            if row['retained_word_0'] & (1 << i):
                retained.add((i, feature))
    assert retained == {(i, feature) for i, member in enumerate(members)
                        for feature in member.features}
    for name in ['feature_stability_summary', 'feature_stability_by_fold_pair',
                 'amino_acid_og_stability']:
        assert_frame_equal(*[pl.read_csv(p / f'{name}.tsv', separator='\t')
                             for p in [compact, full]], check_exact=True)
    for name in ['feature_stability_by_feature', 'coefficients', 'coefficients_by_fold',
                 'feature_importance', 'feature_importance_by_fold']:
        current, previous = [pl.read_csv(p / f'{name}.tsv', separator='\t',
                                         schema_overrides={'fold_id': pl.String})
                             for p in [compact, full]]
        previous = previous.filter(pl.col('feature').is_in(current['feature'].to_list()))
        columns = current.columns
        assert_frame_equal(current.sort(columns), previous.sort(columns), check_exact=True)
    assert not (compact / 'retained_features.tsv').exists()
    assert not (compact / 'amino_acid_site_pss.tsv').exists()
    assert not (compact / 'amino_acid_site_effect_range.tsv').exists()
    keys = ['scope', 'sample_set_id', 'candidate_index', 'orthogroup', 'alignment_pos_1based']
    score_schema = {**dict.fromkeys(keys[1:3] + [keys[-1], 'n_known_residue_states',
                                                'n_nonzero_coefficients'], pl.Int64),
                    **dict.fromkeys(['pss', 'effect_range', 'normalized_pss',
                                     'normalized_effect_range', 'pss_rank', 'effect_range_rank',
                                     'normalized_score', 'rank'], pl.Float64)}
    scores = pl.read_csv(compact / 'amino_acid_site_scores.tsv', separator='\t',
                         schema_overrides=score_schema)
    for metric in ['pss', 'effect_range']:
        expected_scores = pl.read_csv(full / f'amino_acid_site_{metric}.tsv', separator='\t',
                                      schema_overrides=score_schema)
        expected_scores = expected_scores.join(scores.select(keys), on=keys, how='semi')
        actual_scores = scores.rename({f'normalized_{metric}': 'normalized_score',
                                       f'{metric}_rank': 'rank'}).select(expected_scores.columns)
        assert_frame_equal(actual_scores.sort(keys), expected_scores.sort(keys), check_exact=True)
    if not combined:
        assert sum(p.stat().st_size for p in compact.iterdir()) < \
            sum(p.stat().st_size for p in full.iterdir()) * .15


@pytest.mark.parametrize("count", [4, 19, 129])
@pytest.mark.parametrize("zero", [False, True])
def test_worker_bit_words_and_empty_coefficients_survive_merge(
    tmp_path: Path, count: int, zero: bool,
) -> None:
    member = members_for_test(zero=True)[0]
    member.transform.sites = member.transform.sites[:1]
    member.model = model(np.array([0. if zero else 1e-15, 0.]))
    with AAReportWriter(tmp_path / 'merged', compact=True) as merged:
        for index in range(count):
            model_id = index * 1_000_000
            member.scope = f'outer_fold_{index + 1}'
            with AAReportWriter(tmp_path / f'worker{model_id}', model_base=model_id,
                                compact=True) as writer:
                writer.add([member])
                writer.export(tmp_path / f'spool{model_id}')
            merged.merge(tmp_path / f'spool{model_id}')
        write_dictionary(merged, merged.features.scan())
    dictionary = pl.read_parquet(tmp_path / 'merged/feature_dictionary.parquet')
    words = (count + 63) // 64
    assert [c for c in dictionary.columns if c.startswith('retained_word_')] == \
        [f'retained_word_{i}' for i in range(words)]
    for i in range(words):
        assert dictionary[f'retained_word_{i}'].unique().to_list() == \
            [(1 << min(64, count - i * 64)) - 1]
    models = pl.read_parquet(tmp_path / 'merged/report_models.parquet')
    assert models['model_id'].to_list() == [i * 1_000_000 for i in range(count)]
    assert models['retention_index'].to_list() == list(range(count))
    coefficients = read_model_coefficients(tmp_path / 'merged')
    assert coefficients.is_empty() == zero
    if not zero:
        assert sorted(coefficients['model_id']) == models['model_id'].to_list()
        assert all(v.hex() == (1e-15).hex() for v in coefficients['coefficient'])
    manifest = json.loads((tmp_path / 'merged/compact_report_manifest.json').read_text())
    assert manifest['format'] == 'phenoradar-aa-compact-2' and manifest['retained_words'] == words


def test_selected_site_keeps_zero_alternative_and_training_mean(tmp_path: Path) -> None:
    known, active = np.arange(20) < 3, np.arange(20) < 2
    mean = np.r_[.8, .1, .1, np.zeros(17)]
    sites = [AASite('OG1', i, 1, 0, 0, 0, mean=mean, known=known, active=active)
             for i in range(3)]
    member = AAMember(AATransform(settings(), sites, []),
                      model(np.array([2., 0., 0., 0., 1e-15, 0.])),
                      'outer_fold_1', 0, Candidate(0, {}))
    matrix = np.arange(24, dtype=float).reshape(4, 6)
    for compact in [False, True]:
        write_site_effects(tmp_path / str(compact), member, matrix, compact=compact)
    current, previous = [pl.read_csv(tmp_path / str(value) / 'amino_acid_site_effects.tsv',
                                     separator='\t') for value in [True, False]]
    assert_frame_equal(current.select(previous.columns),
                       previous.filter(pl.col('alignment_pos_1based').is_in([1, 3])),
                       check_exact=True)
    assert current.height == 6
    alternative = current.filter((pl.col('alignment_pos_1based') == 1) &
                                 (pl.col('residue') == 'C')).row(0, named=True)
    assert alternative['coefficient'] == 0 and alternative['effect_vs_reference'] == -2
    assert alternative['training_mean'] == .1
    assert set(current['alignment_pos_1based']) == {1, 3}


@pytest.mark.parametrize('aggregation,combined', [
    ('aa_frequency', False), ('best_covered', False), ('tpm_weighted_frequency', False),
    ('tpm_weighted_frequency', True),
])
@pytest.mark.parametrize('n_jobs', [1, 4])
def test_report_modes_leave_training_predictions_and_coordinates_unchanged(
    tmp_path: Path, monkeypatch, aggregation: str, combined: bool, n_jobs: int,
) -> None:
    import joblib
    from test_amino_acid import fixture_config
    from test_combined import combined_config

    from phenoradar.amino_acid_run import run_amino_acid
    from phenoradar.config import load_and_resolve_config

    monkeypatch.chdir(tmp_path)
    config_path = (combined_config(tmp_path, aggregation) if combined else
                   fixture_config(tmp_path, aggregation))
    config = load_and_resolve_config([config_path])
    assert config.report.amino_acid_tables == 'compact'
    config.runtime.n_jobs = n_jobs
    config.preprocess.amino_acid.ranked_site_filter.method = 'none'
    if n_jobs == 4 and not combined and aggregation == 'tpm_weighted_frequency':
        config.runtime.amino_acid_matrix = 'packed'
    config.figures.top_features = 3
    for mode in ['full', 'compact']:
        config.report.amino_acid_tables = mode
        run_amino_acid(config, tmp_path / mode)
    for relative in ['split/tables/split_manifest.tsv', 'cv/tables/prediction_cv.tsv',
                     'cv/tables/metrics_cv.tsv', 'cv/tables/model_selection_selected.tsv',
                     'cv/tables/model_selection_trials.tsv',
                     'cv/tables/selected_site_sequence_map.tsv',
                     'cv/tables/feature_stability_summary.tsv',
                     'cv/tables/amino_acid_site_filter_counts_summary.tsv',
                     'model/tables/model_selection_selected.tsv',
                     'external_test/tables/pred_external_test.tsv',
                     'inference/tables/pred_inference.tsv']:
        old, new = [pl.read_csv(tmp_path / mode / relative, separator='\t',
                               schema_overrides={'fold_id': pl.String})
                    for mode in ['full', 'compact']]
        assert_frame_equal(old, new, check_exact=True)
    old, new = [joblib.load(tmp_path / mode / 'model_bundle/model_state.joblib')
                for mode in ['full', 'compact']]
    for a, b in zip(old, new, strict=True):
        np.testing.assert_array_equal(a.model.coef_, b.model.coef_)
        assert a.model.intercept_.hex() == b.model.intercept_.hex()
    for stage in ['cv', 'model']:
        directory = tmp_path / 'compact' / stage / 'tables'
        assert (directory / 'compact_report_manifest.json').exists()
        assert (directory / 'feature_dictionary.parquet').exists()
        assert not (directory / 'retained_features.tsv').exists()
        if not combined:
            assert not (directory / 'feature_schema.tsv').exists()
        assert (directory / 'amino_acid_site_scores.tsv').exists()
        assert (directory / 'amino_acid_site_qc_summary.tsv').exists()


def test_compact_qc_summary_covers_unselected_sites(tmp_path: Path) -> None:
    from phenoradar.amino_acid_outputs import (
        write_preprocessing_audits,
        write_preprocessing_summary,
    )

    fitted = members_for_test()[0].transform
    context = dict(scope='outer_fold_1', sample_set_id=0)
    write_preprocessing_audits(tmp_path / 'full', fitted, context)
    write_preprocessing_summary(tmp_path / 'compact', fitted, context)
    full = pl.read_csv(tmp_path / 'full/amino_acid_site_qc.tsv', separator='\t')
    row = pl.read_csv(tmp_path / 'compact/amino_acid_site_qc_summary.tsv',
                       separator='\t').row(0, named=True)
    assert row['n_retained_sites'] == full.height
    assert row['n_active_features'] == full['n_active_states'].sum()
    for column in ['sample_fraction', 'post_aggregation_sample_fraction', 'minor_samples',
                   'minor_groups', 'score']:
        assert row[column + '_mean'] == full[column].mean()
        assert row[column + '_min'] == full[column].min()
        assert row[column + '_max'] == full[column].max()


def test_intercept_only_cv_can_replot_empty_compact_site_scores(
    tmp_path: Path, monkeypatch,
) -> None:
    from test_amino_acid import fixture_config

    from phenoradar.amino_acid_figures import write_amino_acid_figures
    from phenoradar.amino_acid_run import run_amino_acid
    from phenoradar.config import load_and_resolve_config

    monkeypatch.chdir(tmp_path)
    config = load_and_resolve_config([fixture_config(tmp_path, 'aa_frequency')])
    config.runtime.execution_stage = 'cv_only'
    config.model_selection.search_space = {'lambda': [1000.], 'alpha': [1.]}
    run = run_amino_acid(config, tmp_path / 'run')
    for stage in ['cv']:
        directory = run / stage / 'tables'
        assert read_model_coefficients(directory).is_empty()
        assert pl.read_csv(directory / 'amino_acid_site_scores.tsv', separator='\t').is_empty()
        assert pl.read_csv(directory / 'amino_acid_site_qc_summary.tsv',
                           separator='\t')['n_retained_sites'].sum() > 0
    write_amino_acid_figures(run, config)
