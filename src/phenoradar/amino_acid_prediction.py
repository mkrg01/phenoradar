"""Prediction views that retain exact nonzero effects and frozen preprocessing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.special import expit

from phenoradar.amino_acid import AASite, AASiteTable, AATransform, AlignmentStore
from phenoradar.amino_acid_packed import PackedAAMatrix

if TYPE_CHECKING:
    from phenoradar.amino_acid_run import AAMember
    from phenoradar.glmnet import _CompactGlmnetModel


def _site_offsets(sites: Sequence[AASite]) -> np.ndarray:
    counts = (sites.active.sum(axis=1) if isinstance(sites, AASiteTable)
              else np.fromiter((np.count_nonzero(site.active) for site in sites),
                               dtype=np.int64, count=len(sites)))
    return np.cumsum(np.concatenate(([0], counts)))


def site_offsets(member: AAMember) -> np.ndarray:
    return _site_offsets(member.transform.sites)


@dataclass
class DeferredAAValidation:
    """Decode supported validation columns in the fitted, shifted basis."""

    transform: AATransform
    store: AlignmentStore
    species: list[str]
    shift: np.ndarray
    full_matrix: Callable[[], PackedAAMatrix]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.species), len(self.shift)

    def _columns(self, columns: np.ndarray) -> sparse.csc_matrix:
        if not len(columns):
            return sparse.csc_matrix((len(self.species), 0))
        offsets = _site_offsets(self.transform.sites)
        positions = np.searchsorted(offsets, columns, side="right") - 1
        indices, starts = np.unique(positions, return_index=True)
        sites = []
        for i, start, stop in zip(indices, starts, np.r_[starts[1:], len(columns)], strict=True):
            site = self.transform.sites[int(i)]
            residues = np.flatnonzero(site.active)[columns[start:stop] - offsets[i]]
            active = np.zeros(20, dtype=bool)
            active[residues] = True
            sites.append(replace(site, active=active))
        # Keep every known training residue for centering, including states
        # whose coefficients are zero, and keep the original representative mask.
        projected = replace(self.transform, sites=sites, qc=[])
        return sparse.csc_matrix(projected.matrix(self.store, self.species) + self.shift[columns])

    def predict(self, models: Sequence[_CompactGlmnetModel]) -> list[np.ndarray]:
        supports = [model.coefficient_indices_ for model in models]
        columns = np.unique(np.concatenate(supports))
        # Bound the dense decoding workspace for unusually dense paths.
        if len(columns) * len(self.species) > 1_000_000:
            full = self.full_matrix()
            return [model.predict_proba(full)[:, 1] for model in models]
        selected = self._columns(columns)
        return [
            expit(np.asarray(selected[:, np.searchsorted(columns, support)]
                             @ model.coefficient_values_ + model.intercept_))
            for model, support in zip(models, supports, strict=True)
        ]


def prediction_members(
    members: list[AAMember],
    *,
    padding_sites: int = 0,
) -> list[AAMember]:
    """Keep nonzero sites, plus zero-score padding for stable local explanations.

    Known states and means remain complete at every retained site. Representative
    scoring masks come from the original transform, including legacy bundles.
    The training members and their full audit schemas are never modified.
    """
    grouped: dict[int, list[AAMember]] = defaultdict(list)
    for member in members:
        grouped[id(member.transform)].append(member)
    result = {}
    for sharing in grouped.values():
        first = sharing[0]
        original = first.transform
        offsets = site_offsets(first)
        selected = set(range(min(padding_sites, len(original.sites))))
        columns = np.unique(np.concatenate([
            np.flatnonzero(m.model.coef_[m.expression_width:] != 0) for m in sharing
        ]))
        selected.update((np.searchsorted(offsets, columns, side="right") - 1).tolist())
        indices = sorted(selected)
        sites = []
        for i in indices:
            site = original.sites[i]
            states = np.flatnonzero(site.active)
            selected_columns = columns[(columns >= offsets[i]) & (columns < offsets[i + 1])]
            active = np.zeros(20, dtype=bool)
            active[states[selected_columns - offsets[i]]] = True
            sites.append(replace(site, active=active))
        scoring = original.representative_columns
        if original.settings.aggregation == "best_covered":
            needed = {s.orthogroup for s in sites}
            if scoring is None:
                grouped_columns: dict[str, list[int]] = defaultdict(list)
                for site in original.sites:
                    if site.orthogroup in needed:
                        grouped_columns[site.orthogroup].append(site.column)
                scoring = {og: np.array(cols) for og, cols in grouped_columns.items()}
            else:
                scoring = {og: scoring[og] for og in needed}
        transform = replace(original, sites=sites, qc=[], representative_columns=scoring)
        for member in sharing:
            n = member.expression_width
            projection = np.concatenate((np.arange(n), n + columns))
            model = copy(member.model)
            model.coef_ = member.model.coef_[projection]
            model.n_features_in_ = len(projection)
            if model.penalty_factor is not None:
                model.penalty_factor = model.penalty_factor[projection]
            scales = getattr(member, "aa_standard_deviation", None)
            result[id(member)] = replace(
                member, transform=transform, model=model,
                aa_standard_deviation=None if scales is None else scales[columns],
            )
    return [result[id(member)] for member in members]
