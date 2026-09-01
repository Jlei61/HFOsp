"""TRAIN-only normalization and route templates for the early ictal field.

H2b spec §4 forbids a held-out seizure from touching any clustering, template,
threshold or normalization. Here that is structural rather than documentary:
both fitters take an explicit ``train_index`` and validate it, and the fitted
objects are frozen -- ``apply`` and ``assign_route`` never update state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FieldNormalization:
    """Per-contact centering/scaling estimated on TRAIN seizures only."""

    mean: np.ndarray
    scale: np.ndarray
    n_train: int
    train_index: tuple[int, ...]

    def apply(self, field: np.ndarray) -> np.ndarray:
        """Normalise one field. Pure: never re-estimates from ``field``."""
        return (np.asarray(field, float) - self.mean) / self.scale


@dataclass(frozen=True)
class RouteTemplates:
    """Frozen early-field routes, with their TRAIN support kept visible."""

    templates: np.ndarray          # (n_routes, n_contacts)
    support: tuple[int, ...]       # TRAIN seizures behind each route
    under_supported: tuple[bool, ...]
    n_train: int
    min_support: int

    @property
    def n_routes(self) -> int:
        return int(self.templates.shape[0])


def _validate(n_rows: int, train_index: Sequence[int]) -> np.ndarray:
    idx = np.asarray(list(train_index), dtype=int)
    if idx.size == 0:
        raise ValueError("need at least one TRAIN row to fit")
    if idx.min() < 0 or idx.max() >= n_rows:
        raise ValueError(f"train_index out of range for {n_rows} rows: {list(train_index)}")
    return idx


def fit_field_normalization(
    fields: np.ndarray,
    train_index: Sequence[int],
) -> FieldNormalization:
    """Per-contact mean/scale over the TRAIN rows only."""

    f = np.asarray(fields, float)
    idx = _validate(f.shape[0], train_index)
    train = f[idx]
    mean = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    # A contact that never varies across TRAIN carries no scale information;
    # unit scale keeps it finite instead of exploding to inf.
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return FieldNormalization(mean=mean, scale=scale, n_train=int(idx.size),
                              train_index=tuple(int(i) for i in idx))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    na, nb = np.linalg.norm(a[ok]), np.linalg.norm(b[ok])
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a[ok], b[ok]) / (na * nb))


def fit_route_templates(
    fields: np.ndarray,
    train_index: Sequence[int],
    max_routes: int = 3,
    min_support: int = 2,
) -> RouteTemplates:
    """Agglomerate TRAIN early fields into at most ``max_routes`` routes.

    Under-supported routes are reported, never merged away: "支持不足的 route
    不强行合并" (H2b spec §4). Correlation-style cosine similarity is used so
    routes describe spatial *pattern*, not overall seizure intensity.
    """

    f = np.asarray(fields, float)
    idx = _validate(f.shape[0], train_index)
    if max_routes > idx.size:
        raise ValueError(f"max_routes={max_routes} exceeds {idx.size} TRAIN seizures")

    train = f[idx]
    clusters = [[i] for i in range(train.shape[0])]
    while len(clusters) > max_routes:
        best, pair = -np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                ca = np.nanmean(train[clusters[a]], axis=0)
                cb = np.nanmean(train[clusters[b]], axis=0)
                s = _cosine(ca, cb)
                if np.isfinite(s) and s > best:
                    best, pair = s, (a, b)
        if pair is None:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)

    clusters.sort(key=len, reverse=True)
    templates = np.vstack([np.nanmean(train[c], axis=0) for c in clusters])
    support = tuple(len(c) for c in clusters)
    return RouteTemplates(
        templates=templates,
        support=support,
        under_supported=tuple(s < min_support for s in support),
        n_train=int(idx.size),
        min_support=int(min_support),
    )


def assign_route(field: np.ndarray, routes: RouteTemplates) -> tuple[int, float]:
    """Label a held-out field against frozen templates. Never moves them."""

    sims = [_cosine(field, t) for t in routes.templates]
    if not np.any(np.isfinite(sims)):
        return -1, float("nan")
    best = int(np.nanargmax(sims))
    return best, float(sims[best])
