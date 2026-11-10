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
