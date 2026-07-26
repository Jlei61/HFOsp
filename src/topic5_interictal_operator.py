"""Within-event interictal propagation operator for Topic 5 / Figure 6.

The recurrent step in this module is a recruitment *pseudo-time* step inside
one interictal group event.  This module intentionally has no event-history,
inter-event-interval, seizure-seed, or ictal-time rollout interface.

Non-participating contacts are always missing.  Legacy ``lagPatRank`` values
are re-ranked only among ``eventsBool`` participants before they enter any
model or metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.propagation_skeleton_geometry import parse_shaft


EPS = 1e-8
CONTACT_FEATURE_NAMES = (
    "prefix_participation_support",
    "prefix_participation_support_centered",
    "within_shaft_position",
    "shaft_size_fraction",
    "coord_x_centered_scaled",
    "coord_y_centered_scaled",
    "coord_z_centered_scaled",
    "geometry_present",
)


def masked_local_ranks(ranks: np.ndarray, participation: np.ndarray) -> np.ndarray:
    """Return local [0, 1] ranks with non-participants left as NaN."""
    from src.lagpat_rank_audit import mask_phantom_ranks

    ranks = np.asarray(ranks, float)
    participation = np.asarray(participation, bool)
    if ranks.shape != participation.shape:
        raise ValueError("ranks and participation must have identical shapes")
    return mask_phantom_ranks(ranks, participation, normalize=True)


def recruitment_groups(
    rank: np.ndarray,
    participation: np.ndarray,
    *,
    lag_raw: Optional[np.ndarray] = None,
    tie_tolerance_seconds: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Encode one event as ordered, possibly set-valued recruitment groups.

    ``group_ids[c]`` is ``-1`` for a non-participant and otherwise runs from
    zero (earliest set) to ``n_groups - 1``.  With the v0.3 primary tolerance
    of zero, only exactly equal finite raw centroids form a tied set.  A
    positive tolerance merges adjacent centroids whose separation is no more
    than the frozen tolerance.
    """
    rank = np.asarray(rank, float)
    participation = np.asarray(participation, bool)
    if rank.ndim != 1 or participation.shape != rank.shape:
        raise ValueError("rank and participation must be aligned 1D arrays")
    tol = float(tie_tolerance_seconds)
    if tol < 0:
        raise ValueError("tie_tolerance_seconds must be non-negative")

    valid = participation & np.isfinite(rank)
    out = np.full(rank.shape, -1, dtype=np.int16)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out, 0

    if lag_raw is not None:
        lag = np.asarray(lag_raw, float)
        if lag.shape != rank.shape:
            raise ValueError("lag_raw must match rank shape")
    else:
        lag = np.full(rank.shape, np.nan)

    # Rank is the canonical ordering. Raw lag is used only to decide whether
    # adjacent contacts are tied; missing raw lag falls back to exact rank.
    order = idx[np.argsort(rank[idx], kind="stable")]
    group = 0
    out[order[0]] = group
    for left, right in zip(order[:-1], order[1:]):
        if np.isfinite(lag[left]) and np.isfinite(lag[right]):
            gap = float(lag[right] - lag[left])
            tied = (gap == 0.0) if tol == 0.0 else (gap <= tol + EPS)
        else:
            gap = float(rank[right] - rank[left])
            tied = gap == 0.0
        if not tied:
            group += 1
        out[right] = group
    return out, int(group + 1)


def encode_recruitment_matrix(
    ranks: np.ndarray,
    participation: np.ndarray,
    lag_raw: Optional[np.ndarray] = None,
    *,
    tie_tolerance_seconds: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode channel x event arrays into event x contact model arrays."""
    ranks = np.asarray(ranks, float)
    participation = np.asarray(participation, bool)
    if ranks.ndim != 2 or participation.shape != ranks.shape:
        raise ValueError("ranks and participation must be aligned channel x event arrays")
    lag = None if lag_raw is None else np.asarray(lag_raw, float)
    if lag is not None and lag.shape != ranks.shape:
        raise ValueError("lag_raw shape mismatch")

    local = masked_local_ranks(ranks, participation)
    n_contacts, n_events = ranks.shape
    groups = np.full((n_events, n_contacts), -1, dtype=np.int16)
    counts = np.zeros(n_events, dtype=np.int16)
    for event_index in range(n_events):
        group, count = recruitment_groups(
            local[:, event_index],
            participation[:, event_index],
            lag_raw=None if lag is None else lag[:, event_index],
            tie_tolerance_seconds=tie_tolerance_seconds,
        )
        groups[event_index] = group
        counts[event_index] = count
    return local.T.astype(np.float32), groups, counts


def _normalized_geometry(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    coords = np.asarray(coords, float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [contact, 3]")
    mapped = np.isfinite(coords).all(axis=1)
    out = np.zeros_like(coords, dtype=float)
    if not np.any(mapped):
        return out, mapped, 1.0
    centered = coords[mapped] - np.mean(coords[mapped], axis=0, keepdims=True)
    if centered.shape[0] >= 2:
        delta = centered[:, None, :] - centered[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        upper = dist[np.triu_indices(centered.shape[0], 1)]
        upper = upper[np.isfinite(upper) & (upper > EPS)]
        scale = float(np.median(upper)) if upper.size else 1.0
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    out[mapped] = centered / scale
    return out, mapped, scale


def build_contact_features(
    channel_names: Sequence[str],
    participation_support: np.ndarray,
    coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build patient-local, ID-free contact side information.

    String channel/shaft identifiers are used only to derive a normalized
    within-shaft ordinal.  They are never exposed as model features.
    """
    names = [str(x) for x in channel_names]
    support = np.asarray(participation_support, float)
    if support.shape != (len(names),):
        raise ValueError("participation_support must match channel_names")
    support = np.clip(np.nan_to_num(support, nan=0.0), 0.0, 1.0)
    centered_support = support - float(np.mean(support)) if support.size else support

    parsed = [parse_shaft(name) for name in names]
    by_shaft: Dict[str, list[Tuple[int, int]]] = {}
    for index, (shaft, ordinal) in enumerate(parsed):
        if shaft is not None and ordinal is not None:
            by_shaft.setdefault(str(shaft), []).append((index, int(ordinal)))
    shaft_position = np.zeros(len(names), dtype=float)
    shaft_size = np.zeros(len(names), dtype=float)
    for members in by_shaft.values():
        ordinals = np.asarray([ordinal for _, ordinal in members], float)
        lo, hi = float(np.min(ordinals)), float(np.max(ordinals))
        for index, ordinal in members:
            shaft_position[index] = (
                0.0 if hi <= lo else 2.0 * (float(ordinal) - lo) / (hi - lo) - 1.0
            )
            shaft_size[index] = len(members) / max(len(names), 1)

    if coords is None:
        coords = np.full((len(names), 3), np.nan, dtype=float)
    norm_coords, geometry_mask, geometry_scale = _normalized_geometry(coords)
    features = np.column_stack(
        [
            support,
            centered_support,
            shaft_position,
            shaft_size,
            norm_coords,
            geometry_mask.astype(float),
        ]
    ).astype(np.float32)
    if features.shape[1] != len(CONTACT_FEATURE_NAMES):
        raise AssertionError("contact feature contract drift")
    metadata = {
        "feature_names": list(CONTACT_FEATURE_NAMES),
        "n_contacts": len(names),
        "n_geometry_mapped": int(np.sum(geometry_mask)),
        "geometry_scale": float(geometry_scale),
        "n_parseable_shafts": len(by_shaft),
        "string_identifiers_exposed_to_model": False,
    }
    return features, metadata


def prefix_targets(group_ids: np.ndarray, tau: int) -> Dict[str, np.ndarray | bool]:
    """Construct targets after observing groups ``0 .. tau-1``."""
    group_ids = np.asarray(group_ids, int)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be 1D")
    valid_groups = group_ids[group_ids >= 0]
    if valid_groups.size == 0:
        raise ValueError("event has no participating contacts")
    n_groups = int(np.max(valid_groups) + 1)
    if tau < 1 or tau > n_groups:
        raise ValueError(f"tau must be in [1, {n_groups}]")
    recruited = (group_ids >= 0) & (group_ids < tau)
    terminal = tau == n_groups
    next_set = np.zeros(group_ids.shape, dtype=bool)
    if not terminal:
        next_set = group_ids == tau
    remaining = group_ids >= tau
    suffix_group = np.where(remaining, group_ids - tau, -1).astype(np.int16)
    return {
        "recruited": recruited,
        "next_set": next_set,
        "terminal": terminal,
        "remaining": remaining,
        "suffix_group": suffix_group,
    }


def pairwise_rank_concordance(
    utility: np.ndarray, suffix_group: np.ndarray
) -> float:
    """Fraction of suffix contact pairs ordered correctly by utility."""
    utility = np.asarray(utility, float)
    group = np.asarray(suffix_group, int)
    good = np.isfinite(utility) & (group >= 0)
    idx = np.flatnonzero(good)
    concordant = 0.0
    total = 0
    for pos, i in enumerate(idx):
        for j in idx[pos + 1 :]:
            if group[i] == group[j]:
                continue
            earlier, later = (i, j) if group[i] < group[j] else (j, i)
            diff = utility[earlier] - utility[later]
            concordant += 1.0 if diff > 0 else 0.5 if diff == 0 else 0.0
            total += 1
    return float(concordant / total) if total else float("nan")


@dataclass(frozen=True)
class MarkovBaseline:
    transition: np.ndarray
    support: np.ndarray

    def scores(self, last_set: np.ndarray, recruited: np.ndarray) -> np.ndarray:
        last = np.flatnonzero(np.asarray(last_set, bool))
        if last.size:
            score = np.mean(self.transition[last], axis=0)
        else:
            score = self.support.copy()
        score = np.asarray(score, float)
        score[np.asarray(recruited, bool)] = 0.0
        return np.maximum(score, EPS)


def fit_first_order_markov(
    group_matrix: np.ndarray,
    *,
    laplace_alpha: float = 0.5,
) -> MarkovBaseline:
    """Fit a participation-preserving first-order set transition baseline."""
    groups = np.asarray(group_matrix, int)
    if groups.ndim != 2:
        raise ValueError("group_matrix must have shape [event, contact]")
    n_contacts = groups.shape[1]
    counts = np.full((n_contacts, n_contacts), float(laplace_alpha), dtype=float)
    support = np.full(n_contacts, float(laplace_alpha), dtype=float)
    for event in groups:
        support += event >= 0
        valid = event[event >= 0]
        if valid.size == 0:
            continue
        n_groups = int(np.max(valid) + 1)
        for group_index in range(n_groups - 1):
            left = np.flatnonzero(event == group_index)
            right = np.flatnonzero(event == group_index + 1)
            if left.size and right.size:
                weight = 1.0 / (left.size * right.size)
                counts[np.ix_(left, right)] += weight
    counts /= np.sum(counts, axis=1, keepdims=True)
    support /= np.sum(support)
    return MarkovBaseline(counts, support)


@dataclass(frozen=True)
class EmpiricalTemplateBaseline:
    """Two static calibration-prefix rank templates."""

    template_rank: np.ndarray
    template_support: np.ndarray

    def choose(self, prefix_group: np.ndarray, tau: int) -> int:
        group = np.asarray(prefix_group, int)
        recruited = (group >= 0) & (group < int(tau))
        earliest = group == 0
        scores = []
        for template, support in zip(self.template_rank, self.template_support):
            concordance = []
            idx = np.flatnonzero(recruited & np.isfinite(template))
            for position, left in enumerate(idx):
                for right in idx[position + 1 :]:
                    if group[left] == group[right]:
                        continue
                    earlier, later = (
                        (left, right)
                        if group[left] < group[right]
                        else (right, left)
                    )
                    concordance.append(float(template[earlier] < template[later]))
            if concordance:
                order_score = float(np.mean(concordance))
            elif np.any(earliest & np.isfinite(template)):
                order_score = -float(np.mean(template[earliest & np.isfinite(template)]))
            else:
                order_score = 0.0
            support_score = (
                float(np.mean(support[recruited])) if np.any(recruited) else 0.0
            )
            scores.append(order_score + 0.05 * support_score)
        return int(np.argmax(scores))

    def scores(
        self, prefix_group: np.ndarray, tau: int, *, temperature: float = 0.20
    ) -> Tuple[np.ndarray, np.ndarray]:
        mode = self.choose(prefix_group, tau)
        rank = np.asarray(self.template_rank[mode], float)
        support = np.asarray(self.template_support[mode], float)
        finite_rank = np.where(np.isfinite(rank), rank, 1.0)
        utility = -finite_rank
        score = np.exp((utility - np.max(utility)) / max(float(temperature), EPS))
        score *= np.maximum(support, EPS)
        recruited = (np.asarray(prefix_group, int) >= 0) & (
            np.asarray(prefix_group, int) < int(tau)
        )
        score[recruited] = 0.0
        return np.maximum(score, EPS), utility


def fit_empirical_template_baseline(
    group_matrix: np.ndarray,
    *,
    seed: int = 20260724,
) -> EmpiricalTemplateBaseline:
    """Fit target-free K=2 static templates from calibration events."""
    from sklearn.cluster import KMeans
    from src.lagpat_rank_audit import build_masked_kmeans_features

    groups = np.asarray(group_matrix, int)
    if groups.ndim != 2 or groups.shape[0] < 20:
        raise ValueError("group_matrix must contain at least 20 events")
    participation = groups >= 0
    rank = np.full(groups.shape, np.nan, dtype=float)
    for event_index, event in enumerate(groups):
        valid = event >= 0
        if not np.any(valid):
            continue
        denominator = max(int(np.max(event[valid])), 1)
        rank[event_index, valid] = event[valid] / denominator
    features = build_masked_kmeans_features(
        rank.T,
        participation.T,
        impute="event_median",
    )
    labels = KMeans(n_clusters=2, n_init=20, random_state=int(seed)).fit_predict(
        features
    )
    templates = []
    supports = []
    for mode in range(2):
        selected = labels == mode
        template = np.full(groups.shape[1], np.nan)
        support = np.mean(participation[selected], axis=0)
        for contact in range(groups.shape[1]):
            values = rank[selected, contact]
            values = values[np.isfinite(values)]
            if values.size:
                template[contact] = float(np.median(values))
        templates.append(template)
        supports.append(support)
    return EmpiricalTemplateBaseline(
        np.asarray(templates, float), np.asarray(supports, float)
    )


try:  # Keep non-ML analysis environments able to import the data primitives.
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised in the base analysis env.
    torch = None
    Tensor = object
    nn = None
    F = None


if nn is not None:

    class ContactQueryGRU(nn.Module):
        """Permutation-invariant set-token GRU with contact-query decoders."""

        def __init__(
            self,
            contact_feature_dim: int,
            *,
            hidden_size: int = 32,
            contact_embedding_dim: int = 32,
            contact_encoder_hidden: int = 32,
        ):
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.contact_embedding_dim = int(contact_embedding_dim)
            self.contact_encoder = nn.Sequential(
                nn.Linear(int(contact_feature_dim), int(contact_encoder_hidden)),
                nn.SiLU(),
                nn.Linear(int(contact_encoder_hidden), self.contact_embedding_dim),
                nn.LayerNorm(self.contact_embedding_dim),
            )
            self.initial_state = nn.Linear(2 * self.contact_embedding_dim, self.hidden_size)
            self.gru = nn.GRUCell(self.contact_embedding_dim + 2, self.hidden_size)
            self.next_query = nn.Linear(self.hidden_size, self.contact_embedding_dim)
            self.utility_query = nn.Linear(self.hidden_size, self.contact_embedding_dim)
            self.participation_query = nn.Linear(
                self.hidden_size, self.contact_embedding_dim
            )
            self.next_bias = nn.Linear(int(contact_feature_dim), 1)
            self.utility_bias = nn.Linear(int(contact_feature_dim), 1)
            self.participation_bias = nn.Linear(int(contact_feature_dim), 1)
            self.stop_head = nn.Linear(self.hidden_size, 1)

        @staticmethod
        def _masked_pool(embedding: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
            weight = mask.to(embedding.dtype).unsqueeze(-1)
            mean = (embedding * weight).sum(1) / weight.sum(1).clamp_min(1.0)
            floor = torch.finfo(embedding.dtype).min
            maximum = embedding.masked_fill(~mask.unsqueeze(-1), floor).max(1).values
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            return mean, maximum

        def forward(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            prefix_sets: Tensor,
            step_mask: Tensor,
            recruited_mask: Optional[Tensor] = None,
        ) -> Dict[str, Tensor]:
            if contact_features.ndim != 3:
                raise ValueError("contact_features must be [batch, contact, feature]")
            if prefix_sets.ndim != 3:
                raise ValueError("prefix_sets must be [batch, step, contact]")
            embedding = self.contact_encoder(contact_features)
            mean, maximum = self._masked_pool(embedding, contact_mask)
            hidden = torch.tanh(self.initial_state(torch.cat([mean, maximum], dim=-1)))
            cumulative = torch.zeros_like(contact_mask)
            n_contacts = contact_mask.sum(1).clamp_min(1).to(embedding.dtype)
            for step in range(prefix_sets.shape[1]):
                current = prefix_sets[:, step].bool() & contact_mask
                weight = current.to(embedding.dtype).unsqueeze(-1)
                token = (embedding * weight).sum(1) / weight.sum(1).clamp_min(1.0)
                cumulative = cumulative | current
                progress = cumulative.sum(1).to(embedding.dtype) / n_contacts
                new_fraction = current.sum(1).to(embedding.dtype) / n_contacts
                update = self.gru(
                    torch.cat([token, progress[:, None], new_fraction[:, None]], dim=-1),
                    hidden,
                )
                active = step_mask[:, step].bool().unsqueeze(-1)
                hidden = torch.where(active, update, hidden)
            if recruited_mask is None:
                recruited_mask = prefix_sets.bool().any(1)
            next_query = self.next_query(hidden)
            utility_query = self.utility_query(hidden)
            participation_query = self.participation_query(hidden)
            scale = float(np.sqrt(self.contact_embedding_dim))
            next_logits = torch.einsum("bce,be->bc", embedding, next_query) / scale
            utility = torch.einsum("bce,be->bc", embedding, utility_query) / scale
            participation_logits = (
                torch.einsum("bce,be->bc", embedding, participation_query) / scale
            )
            next_logits = next_logits + self.next_bias(contact_features).squeeze(-1)
            utility = utility + self.utility_bias(contact_features).squeeze(-1)
            participation_logits = participation_logits + self.participation_bias(
                contact_features
            ).squeeze(-1)
            next_valid = contact_mask & ~recruited_mask.bool()
            next_logits = next_logits.masked_fill(~next_valid, -1e9)
            utility = utility.masked_fill(~contact_mask, -1e9)
            participation_logits = participation_logits.masked_fill(~contact_mask, -1e9)
            return {
                "hidden": hidden,
                "next_logits": next_logits,
                "stop_logit": self.stop_head(hidden).squeeze(-1),
                "remaining_participation_logits": participation_logits,
                "suffix_utility": utility,
                "next_valid_mask": next_valid,
            }


    class StaticContactQuery(nn.Module):
        """Contact-query baseline with no recurrent state.

        ``use_last_set=False`` is an unordered DeepSets prefix baseline.
        ``use_last_set=True`` adds only the most recent recruitment set and is
        the matched-capacity feed-forward contact-query baseline. Neither
        model can encode the ordered path through all preceding sets.
        """

        def __init__(
            self,
            contact_feature_dim: int,
            *,
            hidden_size: int = 32,
            contact_embedding_dim: int = 32,
            contact_encoder_hidden: int = 32,
            use_last_set: bool = False,
        ):
            super().__init__()
            self.use_last_set = bool(use_last_set)
            self.contact_embedding_dim = int(contact_embedding_dim)
            self.contact_encoder = nn.Sequential(
                nn.Linear(int(contact_feature_dim), int(contact_encoder_hidden)),
                nn.SiLU(),
                nn.Linear(int(contact_encoder_hidden), self.contact_embedding_dim),
                nn.LayerNorm(self.contact_embedding_dim),
            )
            n_pools = 6 if self.use_last_set else 4
            self.prefix_mlp = nn.Sequential(
                nn.Linear(n_pools * self.contact_embedding_dim + 1, int(hidden_size)),
                nn.SiLU(),
                nn.Linear(int(hidden_size), int(hidden_size)),
            )
            self.next_query = nn.Linear(int(hidden_size), self.contact_embedding_dim)
            self.utility_query = nn.Linear(int(hidden_size), self.contact_embedding_dim)
            self.participation_query = nn.Linear(
                int(hidden_size), self.contact_embedding_dim
            )
            self.next_bias = nn.Linear(int(contact_feature_dim), 1)
            self.utility_bias = nn.Linear(int(contact_feature_dim), 1)
            self.participation_bias = nn.Linear(int(contact_feature_dim), 1)
            self.stop_head = nn.Linear(int(hidden_size), 1)

        @staticmethod
        def _masked_pool(embedding: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
            weight = mask.to(embedding.dtype).unsqueeze(-1)
            mean = (embedding * weight).sum(1) / weight.sum(1).clamp_min(1.0)
            floor = torch.finfo(embedding.dtype).min
            maximum = embedding.masked_fill(~mask.unsqueeze(-1), floor).max(1).values
            maximum = torch.where(
                torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
            )
            return mean, maximum

        def forward(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            prefix_sets: Tensor,
            step_mask: Tensor,
            recruited_mask: Optional[Tensor] = None,
        ) -> Dict[str, Tensor]:
            embedding = self.contact_encoder(contact_features)
            if recruited_mask is None:
                recruited_mask = prefix_sets.bool().any(1)
            recruited_mask = recruited_mask.bool() & contact_mask
            global_mean, global_max = self._masked_pool(embedding, contact_mask)
            prefix_mean, prefix_max = self._masked_pool(embedding, recruited_mask)
            pools = [global_mean, global_max, prefix_mean, prefix_max]
            if self.use_last_set:
                # step_mask is left-aligned by the v0.3 collator.
                last_index = step_mask.long().sum(1).clamp_min(1) - 1
                batch_index = torch.arange(prefix_sets.shape[0], device=prefix_sets.device)
                last_mask = prefix_sets[batch_index, last_index].bool() & contact_mask
                last_mean, last_max = self._masked_pool(embedding, last_mask)
                pools.extend([last_mean, last_max])
            progress = (
                recruited_mask.sum(1).to(embedding.dtype)
                / contact_mask.sum(1).clamp_min(1).to(embedding.dtype)
            )
            hidden = self.prefix_mlp(torch.cat([*pools, progress[:, None]], dim=-1))
            scale = float(np.sqrt(self.contact_embedding_dim))
            next_logits = (
                torch.einsum("bce,be->bc", embedding, self.next_query(hidden)) / scale
            )
            utility = (
                torch.einsum("bce,be->bc", embedding, self.utility_query(hidden))
                / scale
            )
            participation_logits = (
                torch.einsum(
                    "bce,be->bc", embedding, self.participation_query(hidden)
                )
                / scale
            )
            next_logits = next_logits + self.next_bias(contact_features).squeeze(-1)
            utility = utility + self.utility_bias(contact_features).squeeze(-1)
            participation_logits = participation_logits + self.participation_bias(
                contact_features
            ).squeeze(-1)
            next_valid = contact_mask & ~recruited_mask
            return {
                "hidden": hidden,
                "next_logits": next_logits.masked_fill(~next_valid, -1e9),
                "stop_logit": self.stop_head(hidden).squeeze(-1),
                "remaining_participation_logits": participation_logits.masked_fill(
                    ~contact_mask, -1e9
                ),
                "suffix_utility": utility.masked_fill(~contact_mask, -1e9),
                "next_valid_mask": next_valid,
            }


    def contact_query_loss(
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        *,
        loss_weights: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Tensor]:
        """Joint next-set, STOP, remaining-participation and suffix-rank loss."""
        weights = {
            "next_set": 1.0,
            "stop": 0.5,
            "remaining_participation": 0.25,
            "suffix_rank": 0.5,
        }
        if loss_weights:
            weights.update({str(k): float(v) for k, v in loss_weights.items()})

        terminal = batch["terminal"].bool()
        next_logits = outputs["next_logits"]
        next_target = batch["next_set"].bool()
        nonterminal = ~terminal
        if torch.any(nonterminal):
            selected = next_logits[nonterminal]
            target = next_target[nonterminal]
            valid = outputs["next_valid_mask"][nonterminal]
            if not torch.all(target.any(1)):
                raise ValueError("every non-terminal sample needs a non-empty next set")
            numerator = torch.logsumexp(selected.masked_fill(~target, -1e9), dim=1)
            denominator = torch.logsumexp(selected.masked_fill(~valid, -1e9), dim=1)
            loss_next = torch.mean(denominator - numerator)
        else:
            loss_next = next_logits.sum() * 0.0

        loss_stop = F.binary_cross_entropy_with_logits(
            outputs["stop_logit"], terminal.to(outputs["stop_logit"].dtype)
        )
        remaining_mask = batch["contact_mask"].bool() & ~batch["recruited"].bool()
        part_element = F.binary_cross_entropy_with_logits(
            outputs["remaining_participation_logits"],
            batch["remaining"].to(outputs["remaining_participation_logits"].dtype),
            reduction="none",
        )
        loss_part = (part_element * remaining_mask).sum() / remaining_mask.sum().clamp_min(1)

        rank_terms = []
        utility = outputs["suffix_utility"]
        suffix_group = batch["suffix_group"].long()
        for sample in range(utility.shape[0]):
            group = suffix_group[sample]
            valid_idx = torch.where(group >= 0)[0]
            if valid_idx.numel() < 2:
                continue
            gi = group[valid_idx][:, None]
            gj = group[valid_idx][None, :]
            earlier = gi < gj
            if not torch.any(earlier):
                continue
            ui = utility[sample, valid_idx][:, None]
            uj = utility[sample, valid_idx][None, :]
            rank_terms.append(F.softplus(-(ui - uj))[earlier].mean())
        loss_rank = (
            torch.stack(rank_terms).mean() if rank_terms else utility.sum() * 0.0
        )
        total = (
            weights["next_set"] * loss_next
            + weights["stop"] * loss_stop
            + weights["remaining_participation"] * loss_part
            + weights["suffix_rank"] * loss_rank
        )
        return {
            "total": total,
            "next_set": loss_next,
            "stop": loss_stop,
            "remaining_participation": loss_part,
            "suffix_rank": loss_rank,
        }

else:

    class ContactQueryGRU:  # pragma: no cover - informative fallback only.
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ContactQueryGRU")


    class StaticContactQuery:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for StaticContactQuery")


    def contact_query_loss(*args, **kwargs):  # pragma: no cover
        raise ImportError("PyTorch is required for contact_query_loss")
