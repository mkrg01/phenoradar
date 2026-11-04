import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler
from test_amino_acid import settings
from test_amino_acid_streaming import model

from phenoradar.amino_acid import ALPHABET, AASite, AATransform
from phenoradar.amino_acid_outputs import write_combined_coefficients, write_site_effects
from phenoradar.amino_acid_run import AAMember, _write
from phenoradar.combined import ExpressionTransform
from phenoradar.config.schema import AppConfig
from phenoradar.model_selection import Candidate


@pytest.mark.parametrize("combined", [False, True])
def test_columnar_effects_and_units_match_all_states_and_append_contexts(
    tmp_path: Path, combined: bool,
) -> None:
    sites = []
    for column, (n_known, n_active) in enumerate([(3, 2), (2, 0), (1, 1), (4, 4)]):
        mean = np.zeros(20)
        mean[:n_known] = np.arange(1, n_known + 1) / sum(range(1, n_known + 1))
        sites.append(AASite('OG_é"', column, 1, 0, 0, 0, mean,
                            np.arange(20) < n_known, np.arange(20) < n_active))
    config = AppConfig()
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = np.zeros(2), np.array([2.0, 4.0])
    expression = ExpressionTransform(config, ["EXPR_A", "EXPR_B"], scaler) if combined else None
    beta = np.r_[([4.0, -2.0] if combined else []), [-1.0, 2.0, .5, 0, 0, 0, 0]]
    matrix = np.arange(3 * len(beta), dtype=float).reshape(3, -1) / 7
    scales = np.arange(1, 8, dtype=float) / 10
    for candidate in range(2):
        member = AAMember(AATransform(settings(), sites, []), model(beta),
                          "outer_fold_2", 0, Candidate(candidate, {}), expression, scales)
        write_site_effects(tmp_path / "new", member, matrix)
        context = dict(scope=member.scope, sample_set_id=0, candidate_index=candidate)
        offset = member.expression_width
        effects = []
        for site in sites:
            count = int(site.active.sum())
            coefficients = np.zeros(20)
            coefficients[site.active] = beta[offset:offset + count]
            contribution = float(np.mean(np.abs(matrix[:, offset:offset + count]
                                                @ coefficients[site.active])))
            reference = int(np.argmax(site.mean))
            for state in np.flatnonzero(site.known):
                effects.append({**context, "orthogroup": site.orthogroup,
                                "alignment_pos_1based": site.column + 1,
                                "residue": ALPHABET[state], "coefficient": coefficients[state],
                                "reference_residue": ALPHABET[reference],
                                "effect_vs_reference": float(
                                    coefficients[state] - coefficients[reference]
                                ),
                                "site_effect_range": float(np.ptp(coefficients[site.known])),
                                "mean_abs_site_contribution": contribution})
            offset += count
        _write(tmp_path / "old/amino_acid_site_effects.tsv", effects)
        if combined:
            write_combined_coefficients(tmp_path / "new", member)
            table = pl.read_csv(tmp_path / "new/combined_coefficients.tsv", separator="\t")
            local = table.filter(pl.col("candidate_index") == candidate)
            assert local["feature"].to_list() == member.features
            assert local["coefficient"].to_list() == beta.tolist()
            np.testing.assert_array_equal(local["coefficient_original_unit"],
                                          np.r_[beta[:2] / scaler.scale_, beta[2:]])
            np.testing.assert_array_equal(local["coefficient_standardized"],
                                          np.r_[beta[:2], beta[2:] * scales])
            assert [json.loads(f)[0] for f in local["feature"][2:]] == ['OG_é"'] * 7
    frames = [pl.read_csv(tmp_path / d / "amino_acid_site_effects.tsv", separator="\t")
              for d in ["old", "new"]]
    assert_frame_equal(*frames, check_exact=True)
