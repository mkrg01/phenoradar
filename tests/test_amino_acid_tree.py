from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from phenoradar.amino_acid import AlignmentStore
from phenoradar.amino_acid_tree import build_residue_annotation, write_residue_tree
from phenoradar.figures import _feature_axis_labels, _feature_label_annotations_subset
from phenoradar.residue_colors import RESIDUE_COLORS, residue_color


def inputs(tmp_path: Path) -> tuple[AlignmentStore, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    aa = tmp_path / 'aa'
    aa.mkdir()
    (aa / 'OG1.fa').write_text('>s1_z\nAAAA\n>s1_a\nV-X*\n>s2_a\nACDE\n')
    store = AlignmentStore(aa, ['s1', 's2', 's3'])
    store.ogs['OG1'].tpm = np.array([1., 9., 0.])
    sites = pl.DataFrame({'orthogroup': ['OG1'] * 3,
                          'alignment_pos_1based': [2, 1, 3], 'pss': [3., 2., 1.]})
    predictions = pl.DataFrame({'species': ['s1', 's2', 's3'], 'label': [1, 0, 1],
                                'prob': [.9, .2, .4]})
    annotations = pl.DataFrame({'feature': ['OG1'], 'orthogroup_annotation': ['Test kinase']})
    return store, sites, predictions, annotations


def test_max_tpm_is_per_og_and_keeps_gap_unknown_and_coordinates(tmp_path: Path) -> None:
    store, sites, predictions, annotations = inputs(tmp_path)
    with store:
        before = store.ogs['OG1'].tpm.copy()
        table = build_residue_annotation(store, sites, predictions,
                                        aggregation='tpm_weighted_frequency',
                                        annotations=annotations)
        rows = table.filter(pl.col('species') == 's1').sort('feature_rank')
        assert rows['residue'].to_list() == ['-', 'V', 'X']
        assert rows['representative_sequence_id'].to_list() == ['s1_a'] * 3
        assert rows['sequence_pos_1based'].to_list() == [None, 1, 2]
        assert rows['residue_status'].to_list() == ['gap', 'observed', 'unknown_residue']
        assert rows['orthogroup_annotation'].to_list() == ['Test kinase'] * 3
        assert table.filter(pl.col('species') == 's2')['residue_status'].unique().to_list() == [
            'zero_tpm']
        assert table.filter(pl.col('species') == 's3')['residue_status'].unique().to_list() == [
            'missing_sequence']
        np.testing.assert_array_equal(store.ogs['OG1'].tpm, before)
        store.ogs['OG1'].tpm[:2] = 9
        tie = build_residue_annotation(store, sites.head(1), predictions,
                                       aggregation='tpm_weighted_frequency')
        assert tie.filter(pl.col('species') == 's1')['representative_sequence_id'][0] == 's1_a'


@pytest.mark.parametrize('aggregation', ['aa_frequency', 'best_covered'])
def test_unweighted_display_coverage_is_site_independent(tmp_path: Path, aggregation: str) -> None:
    store, sites, predictions, _ = inputs(tmp_path)
    with store:
        table = build_residue_annotation(store, sites, predictions, aggregation=aggregation)
        assert table.filter(pl.col('species') == 's1')['representative_sequence_id'].to_list() == [
            's1_z'] * 3
        with pytest.raises(ValueError, match='outside input alignment'):
            build_residue_annotation(store, sites.with_columns(
                pl.lit(0).alias('alignment_pos_1based')), predictions, aggregation=aggregation)


def test_annotation_resolves_aa_site_and_combined_keys_without_losing_ids() -> None:
    annotations = pl.DataFrame({'feature': ['OG1'], 'orthogroup_annotation': ['Test kinase']})
    names = ['OG1', json.dumps(['OG1', 8, 'V']), json.dumps(['OG1', 8]),
             json.dumps(['expression', 'OG1'])]
    labels = _feature_axis_labels(names, annotations)
    assert labels == ['Test kinase (OG1)', 'Test kinase (OG1:8 V)',
                      'Test kinase (OG1:8)', 'Test kinase (OG1 (expression))']
    importance = pl.DataFrame({'feature': [names[1]], 'importance_mean': [1.]})
    coefficients = pl.DataFrame({
        'feature': [names[1]], 'coef_mean': [1.], 'method': ['coef_signed']})
    subset = _feature_label_annotations_subset(annotations, feature_importance=importance,
                                              coefficients=coefficients, top_features=1)
    assert subset is not None and subset.equals(annotations)
    assert _feature_axis_labels([names[1]], None) == ['OG1:8 V']


def test_residue_tree_svg_table_palette_and_accepted_population(tmp_path: Path) -> None:
    store, sites, predictions, annotations = inputs(tmp_path)
    annotations = annotations.with_columns(
        pl.lit('Test kinase & membrane <family>').alias('orthogroup_annotation'))
    tree = tmp_path / 'tree.nwk'
    tree.write_text('((s1:1,s2:1):1,s3:2,extra:1);')
    predictions = predictions.with_columns(pl.Series('decision_status', ['accepted', 'accepted',
                                                                         'abstained']))
    with store:
        assert write_residue_tree(tmp_path, tree, store, sites, predictions,
                                  aggregation='tpm_weighted_frequency',
                                  annotations=annotations) == []
    svg = (tmp_path / 'cv/figures/tree_feature_heatmap_amino_acid.svg').read_text()
    assert 'Test kinase' in svg and 'OG1:2' in svg and 'Zappo' in svg
    assert 's1_a' in svg and 'max_tpm' in svg
    assert (tmp_path / 'cv/figures/accepted_only/tree_feature_heatmap_amino_acid.svg').exists()
    table = pl.read_csv(tmp_path / 'cv/tables/tree_feature_heatmap_amino_acid_annotation.tsv',
                        separator='\t')
    assert table.height == 9
    assert len(RESIDUE_COLORS) == 20
    assert residue_color('K') == '#6464ff'
    assert residue_color('-') != residue_color('X')


def test_aa_run_generates_annotated_figures_and_tree(tmp_path: Path) -> None:
    from test_amino_acid import fixture_config

    from phenoradar.amino_acid_run import run_amino_acid
    from phenoradar.config import load_and_resolve_config

    path = fixture_config(tmp_path, 'tpm_weighted_frequency')
    config = yaml.safe_load(path.read_text())
    metadata = pl.read_csv(config['data']['metadata_path'], separator='\t')
    tree = tmp_path / 'tree.nwk'
    tree.write_text('(' + ','.join(f'{s}:1' for s in metadata['species']) + ');')
    annotation = tmp_path / 'og.tab'
    annotation.write_text('OG1\t1\tTest kinase\n')
    config['data'].update(tree_path=str(tree), orthogroup_annotation_path=str(annotation))
    path.write_text(yaml.safe_dump(config))
    out = run_amino_acid(load_and_resolve_config([path]), tmp_path / 'run')
    for figure in ['tree_feature_heatmap_amino_acid', 'feature_importance_top',
                   'coefficients_signed_top', 'amino_acid_site_pss_top',
                   'amino_acid_site_effect_range_top',
                   'amino_acid_frequency_by_confusion']:
        text = (out / 'cv/figures' / f'{figure}.svg').read_text()
        assert 'Test kinase' in text, figure
    assert not (out / 'cv/figures/amino_acid_frequency_heatmap.svg').exists()
    table = pl.read_csv(out / 'cv/tables/tree_feature_heatmap_amino_acid_annotation.tsv',
                        separator='\t')
    assert table['species'].n_unique() == 8
    assert set(table['representative_sequence_id']) == {f's{i}_gene' for i in range(8)}


@pytest.mark.parametrize('mode', ['amino_acid', 'combined'])
@pytest.mark.parametrize('full', [False, True])
def test_sequence_modes_write_group_and_cv_prediction_trees(
    tmp_path: Path, mode: str, full: bool,
) -> None:
    from test_amino_acid import fixture_config
    from test_combined import combined_config

    from phenoradar.amino_acid_run import run_amino_acid
    from phenoradar.config import load_and_resolve_config

    path = (combined_config(tmp_path, 'tpm_weighted_frequency') if mode == 'combined' else
            fixture_config(tmp_path, 'tpm_weighted_frequency'))
    raw = yaml.safe_load(path.read_text())
    raw['runtime']['execution_stage'] = 'full_run' if full else 'cv_only'
    if mode == 'amino_acid':
        # Context trees must never read the inactive expression input.
        raw['data']['tpm_path'] = str(tmp_path / 'does-not-exist.tsv')
    metadata = Path(raw['data']['metadata_path'])
    frame = pl.read_csv(metadata, separator='\t').with_columns(
        pl.col('contrast_pair_id').replace(
            {'g0': '1', 'g1': '10', 'g2': '2', 'g3': '20'})
    )
    frame.write_csv(metadata, separator='\t')
    tree = tmp_path / 'tree.nwk'
    tree.write_text('(' + ','.join(f'{s}:1' for s in frame['species']) + ');')
    raw['data']['tree_path'] = str(tree)
    path.write_text(yaml.safe_dump(raw))
    out = run_amino_acid(load_and_resolve_config([path]), tmp_path / 'run')

    for name in ['tree_group', 'tree_prediction_cv', 'tree_feature_heatmap_amino_acid']:
        assert (out / 'cv/figures' / f'{name}.svg').exists()
    assert (out / 'cv/figures/accepted_only/tree_prediction_cv.svg').exists()
    assert not (out / 'cv/tables/tree_feature_heatmap_annotation.tsv').exists()
    for name in ['tree_feature_heatmap_log2_tpm', 'tree_feature_heatmap_zscore']:
        assert not (out / 'cv/figures' / f'{name}.svg').exists()

    tree_table = pl.read_csv(out / 'cv/tables/tree_prediction_cv_annotation.tsv',
        separator='\t', schema_overrides={'fold_id': pl.String, 'group_id': pl.String}
    ).sort('species')
    oof = pl.read_csv(out / 'cv/tables/prediction_cv.tsv', separator='\t',
                     schema_overrides={'fold_id': pl.String}).sort('species')
    assert tree_table.height == 8
    assert tree_table['species'].to_list() == oof['species'].to_list()
    assert tree_table['fold_id'].to_list() == oof['fold_id'].to_list()
    assert tree_table['group_id'].to_list() == tree_table['fold_id'].to_list()
    assert tree_table['true_label'].to_list() == oof['label'].to_list()
    assert tree_table['pred_label'].to_list() == oof['pred_label_selective'].to_list()
    np.testing.assert_allclose(tree_table['prob'], oof['prob'], rtol=0, atol=5e-9)
    groups = pl.read_csv(out / 'cv/tables/tree_contrast_pairs_annotation.tsv', separator='\t')
    assert set(groups['species']) == set(oof['species'])
    external_table = out / 'external_test/tables/tree_prediction_external_annotation.tsv'
    assert external_table.exists() == full
    if full:
        assert pl.read_csv(external_table, separator='\t')['species'].to_list() == ['external']


@pytest.mark.parametrize('mode', ['amino_acid', 'combined'])
def test_sequence_context_trees_fall_back_to_split_group_without_contrast_pairs(
    tmp_path: Path, mode: str,
) -> None:
    from test_amino_acid import fixture_config
    from test_combined import combined_config

    from phenoradar.amino_acid_run import run_amino_acid
    from phenoradar.config import load_and_resolve_config

    path = combined_config(tmp_path) if mode == 'combined' else fixture_config(
        tmp_path, 'aa_frequency', full=False)
    raw = yaml.safe_load(path.read_text())
    raw['runtime']['execution_stage'] = 'cv_only'
    raw['data']['contrast_pair_col'] = None
    raw['split'] = {'group_col': 'family'}
    raw['preprocess']['amino_acid']['ranked_site_filter']['method'] = 'none'
    metadata = pl.read_csv(raw['data']['metadata_path'], separator='\t')
    tree = tmp_path / 'tree.nwk'
    tree.write_text('(' + ','.join(f'{s}:1' for s in metadata['species']) + ');')
    raw['data']['tree_path'] = str(tree)
    path.write_text(yaml.safe_dump(raw))
    out = run_amino_acid(load_and_resolve_config([path]), tmp_path / 'run')
    assert (out / 'cv/figures/tree_group.svg').exists()
    assert (out / 'cv/figures/tree_prediction_cv.svg').exists()
    annotation = pl.read_csv(out / 'cv/tables/tree_prediction_cv_annotation.tsv', separator='\t')
    oof = pl.read_csv(out / 'cv/tables/prediction_cv.tsv', separator='\t')
    expected = oof.select('species').join(metadata.select('species', 'family'), on='species')
    actual = annotation.select('species', pl.col('group_id').alias('family'))
    assert actual.sort('species').equals(expected.sort('species'))


def test_tree_site_identifiers_survive_long_annotations_and_ids(tmp_path: Path) -> None:
    from xml.etree import ElementTree as ET

    from phenoradar.tree_prediction import _write_tree_feature_heatmap_svg

    store, sites, predictions, annotations = inputs(tmp_path)
    with store:
        table = build_residue_annotation(store, sites.head(1), predictions,
            aggregation='tpm_weighted_frequency', annotations=annotations)
    og = 'OG_' + 'x' * 120
    annotation = 'A very long annotated protein description ' * 12
    table = table.with_columns(
        pl.lit(json.dumps([og, 123456])).alias('feature'),
        pl.lit(annotation).alias('orthogroup_annotation'),
    )
    tree = tmp_path / 'tree.nwk'
    tree.write_text('(s1:1,s2:1,s3:1);')
    path = tmp_path / 'tree.svg'
    assert _write_tree_feature_heatmap_svg(
        tree_path=tree, annotation=table, value_col='residue', cmap_name='', out_path=path,
        title='', annotate_features=True,
    ) == []
    xml = ET.parse(path)
    visible = [''.join(element.itertext()) for element in xml.getroot().iter()
               if element.tag.endswith('text')]
    assert any(f'{og}:123456:' in label and '…' in label for label in visible)
    assert any('input MSA site (1-based)' in label for label in visible)
    assert annotation.strip() in path.read_text()  # Tooltips keep the complete annotation.
