"""Expression blocks for joint expression/amino-acid LASSO models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from phenoradar.config import AppConfig
from phenoradar.cv import (
    CVError,
    FeatureScaler,
    RunExpressionCache,
    _apply_expression_transform_for_config,
    _preprocess_fold_with_counts,
    apply_feature_scaling,
)


class ExpressionInput:
    """Reuse the existing validated disk cache and bounded selected target matrices.

    Only the training pool determines the candidate feature axis. Target rows
    are materialized later, with the union of fitted expression features.
    """

    def __init__(self, config: AppConfig, all_species: list[str], training: list[str], *,
                 cache_species: list[str] | None = None) -> None:
        self.cache = RunExpressionCache()
        self.builder = self.cache.get_builder(config)
        self.caches: list[tuple[dict[str, int], dict[str, int], np.ndarray]] = []
        self.features: list[str] = []
        try:
            consumed = (cache_species if cache_species is not None else
                        all_species if not training or config.runtime.execution_stage == "full_run"
                        else training)
            self.cache.prepare_species(config, consumed)
            self.present = set(
                self.builder._collect_expression(
                    self.builder._long_scan_for_species(consumed).select("__species").unique()
                )["__species"].to_list()
            )
            missing = sorted(set(training) - self.present)
            if missing:
                raise CVError(
                    "Combined training requires expression for every training species; missing: "
                    + ", ".join(missing[:10])
                )
            if training:
                matrix, self.features = self.builder.build_matrix(training)
                self._cache(training, self.features, matrix)
        except BaseException:
            self.close()
            raise

    def _cache(self, species: list[str], features: list[str], matrix: np.ndarray) -> None:
        self.caches.append((
            {s: i for i, s in enumerate(species)},
            {f: j for j, f in enumerate(features)},
            matrix,
        ))

    def matrix(self, species: list[str], features: list[str] | None = None) -> np.ndarray:
        names = self.features if features is None else features
        for rows, columns, matrix in self.caches:
            if all(s in rows for s in species) and all(f in columns for f in names):
                return np.asarray(
                    matrix[np.ix_([rows[s] for s in species], [columns[f] for f in names])]
                )
        result = np.full((len(species), len(names)), np.nan)
        available = [s for s in species if s in self.present]
        if available:
            values, _ = self.builder.build_matrix(available, names)
            rows = {s: i for i, s in enumerate(species)}
            result[[rows[s] for s in available]] = values
        self._cache(species, names, result)
        return result

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> ExpressionInput:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass
class ExpressionTransform:
    config: AppConfig
    orthogroups: list[str]
    scaler: FeatureScaler

    @property
    def features(self) -> list[str]:
        return [json.dumps(["expression", og], separators=(",", ":")) for og in self.orthogroups]

    def transform(
        self, source: ExpressionInput, species: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = source.matrix(species, self.orthogroups)
        values = _apply_expression_transform_for_config(self.config, raw)
        observed = np.isfinite(values)
        return (
            apply_feature_scaling(
                values, self.scaler, self.config.preprocess.expression.feature_scaling.method
            ),
            observed,
        )

    def schema(self) -> list[dict[str, Any]]:
        return [
            {"feature": f, "modality": "expression", "orthogroup": og,
             "alignment_pos_1based": None, "residue": None}
            for f, og in zip(self.features, self.orthogroups, strict=True)
        ]


def fit_expression(
    config: AppConfig,
    source: ExpressionInput,
    species: list[str],
    y: np.ndarray,
    pairs: np.ndarray,
    warnings: list[str],
) -> tuple[ExpressionTransform, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    raw = source.matrix(species)
    scores: list[dict[str, Any]] = []
    train, _, names, counts, scaler = _preprocess_fold_with_counts(
        config, raw, np.empty((0, len(source.features))), source.features,
        y_train=y, groups_train=pairs, warnings=warnings, ranked_feature_score_rows=scores,
    )
    return (
        ExpressionTransform(config.model_copy(deep=True), names, scaler), train,
        dict(vars(counts)), scores,
    )
