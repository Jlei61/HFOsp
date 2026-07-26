"""Leakage-safe primitives for the Figure 6 state-conditioned predictor.

This module deliberately separates three contracts:

1. a patient axis is estimated from a chronological interictal calibration prefix;
2. seizure labels are signed projections onto that already frozen axis;
3. the recurrent core only consumes masked-rank event histories.

The low-rank term in :class:`LREICTRNN` is an *effective recurrent interaction*.
Only ``dale=True`` gives the stricter presynaptic-column sign constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-8


def weighted_center(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    w = np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    out = np.full_like(x, np.nan, dtype=float)
    if not np.any(good):
        return out
    wn = w[good] / np.sum(w[good])
    out[good] = x[good] - float(np.sum(wn * x[good]))
    return out


def weighted_inner(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return float("nan")
    wn = w[good] / np.sum(w[good])
    return float(np.sum(wn * x[good] * y[good]))


def weighted_norm(x: np.ndarray, weights: np.ndarray) -> float:
    value = weighted_inner(x, x, weights)
    return float(np.sqrt(max(value, 0.0))) if np.isfinite(value) else float("nan")


def masked_normalized_ranks(ranks: np.ndarray, bools: np.ndarray) -> np.ndarray:
    """Return channel x event local ranks; non-participants remain NaN."""
    from src.lagpat_rank_audit import mask_phantom_ranks

    return mask_phantom_ranks(ranks, bools, normalize=True)


def _cluster_template(
    masked_ranks: np.ndarray,
    bools: np.ndarray,
    event_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    event_indices = np.asarray(event_indices, int)
    n_ch = masked_ranks.shape[0]
    template = np.full(n_ch, np.nan)
    support = np.zeros(n_ch, dtype=float)
    if event_indices.size == 0:
        return template, support
    for ch in range(n_ch):
        vals = masked_ranks[ch, event_indices]
        finite = np.isfinite(vals)
        support[ch] = float(np.mean(bools[ch, event_indices]))
        if np.any(finite):
            template[ch] = float(np.median(vals[finite]))
    return template, support


def _stable_cluster_labels(features: np.ndarray, seed: int, n_seeds: int = 5):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score

    labels = []
    inertias = []
    for offset in range(n_seeds):
        km = KMeans(n_clusters=2, n_init=20, random_state=int(seed + offset))
        labels.append(km.fit_predict(features))
        inertias.append(float(km.inertia_))
    pair = [
        adjusted_mutual_info_score(labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    best = int(np.argmin(inertias))
    return labels[best], float(np.median(pair)) if pair else 1.0


def _orient_pair(
    template_a: np.ndarray,
    support_a: np.ndarray,
    template_b: np.ndarray,
    support_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Give KMeans' arbitrary labels a deterministic, target-free order.

    The first finite contrast in the frozen channel order is positive for A.
    Biological conclusions must remain invariant to this convention; training
    therefore also uses A/B-swap augmentation.
    """
    contrast = np.nan_to_num(template_a - template_b, nan=0.0)
    nz = np.flatnonzero(np.abs(contrast) > 1e-10)
    if nz.size and contrast[nz[0]] < 0:
        return template_b, support_b, template_a, support_a
    return template_a, support_a, template_b, support_b


def _axis_from_templates(
    template_a: np.ndarray,
    support_a: np.ndarray,
    template_b: np.ndarray,
    support_b: np.ndarray,
) -> Dict[str, np.ndarray]:
    q = np.sqrt(np.maximum(support_a, 0) * np.maximum(support_b, 0))
    valid = np.isfinite(template_a) & np.isfinite(template_b) & (q > 0)
    q = np.where(valid, q, 0.0)
    if np.sum(valid) < 4:
        raise ValueError("fewer than four jointly supported template contacts")
    ta = weighted_center(template_a, q)
    tb = weighted_center(template_b, q)
    contrast = ta - tb
    scale = weighted_norm(contrast, q)
    if not np.isfinite(scale) or scale <= EPS:
        raise ValueError("degenerate prefix template contrast")
    direction = contrast / scale
    coord = np.full_like(direction, np.nan)
    finite = np.isfinite(direction) & (q > 0)
    max_abs = float(np.nanmax(np.abs(direction[finite])))
    coord[finite] = direction[finite] / max(max_abs, EPS)
    return {
        "template_a": template_a,
        "template_b": template_b,
        "support_a": support_a,
        "support_b": support_b,
        "support_q": q,
        "direction_basis": direction,
        "axis_coordinate": coord,
    }


def derive_prefix_axis(
    ranks: np.ndarray,
    bools: np.ndarray,
    prefix_event_indices: Sequence[int],
    *,
    seed: int = 20260724,
    min_cluster_fraction: float = 0.10,
) -> Dict[str, object]:
    """Estimate a two-template axis using prefix events only."""
    from src.lagpat_rank_audit import build_masked_kmeans_features

    ranks = np.asarray(ranks, float)
    bools = np.asarray(bools, bool)
    idx = np.asarray(prefix_event_indices, int)
    idx = idx[np.sum(bools[:, idx], axis=0) >= 3]
    if idx.size < 20:
        raise ValueError("too few valid calibration events")
    features = build_masked_kmeans_features(
        ranks[:, idx], bools[:, idx], impute="event_median"
    )
    labels, seed_ami = _stable_cluster_labels(features, seed)
    fractions = np.bincount(labels, minlength=2) / labels.size
    if float(np.min(fractions)) < float(min_cluster_fraction):
        raise ValueError("prefix K=2 contains an undersized cluster")
    masked = masked_normalized_ranks(ranks[:, idx], bools[:, idx])
    local = np.arange(idx.size)
    ta, sa = _cluster_template(masked, bools[:, idx], local[labels == 0])
    tb, sb = _cluster_template(masked, bools[:, idx], local[labels == 1])
    ta, sa, tb, sb = _orient_pair(ta, sa, tb, sb)
    axis = _axis_from_templates(ta, sa, tb, sb)
    axis.update(
        {
            "prefix_event_indices": idx,
            "labels": labels,
            "seed_ami": seed_ami,
            "cluster_fractions": fractions,
        }
    )
    return axis


def axis_split_stability(
    ranks: np.ndarray,
    bools: np.ndarray,
    prefix_event_indices: Sequence[int],
    *,
    seed: int = 20260724,
) -> float:
    """Correlation of independently estimated chronological half-prefix axes."""
    idx = np.asarray(prefix_event_indices, int)
    if idx.size < 40:
        return float("nan")
    mid = idx.size // 2
    left = derive_prefix_axis(ranks, bools, idx[:mid], seed=seed)
    right = derive_prefix_axis(ranks, bools, idx[mid:], seed=seed + 1000)
    q = np.sqrt(left["support_q"] * right["support_q"])
    a = left["direction_basis"]
    b = right["direction_basis"]
    good = np.isfinite(a) & np.isfinite(b) & (q > 0)
    if np.sum(good) < 4 or np.std(a[good]) <= EPS or np.std(b[good]) <= EPS:
        return float("nan")
    # Axis sign is arbitrary; stability concerns the one-dimensional subspace.
    return float(abs(np.corrcoef(a[good], b[good])[0, 1]))


def robust_rebaseline_activation(
    z_trace: np.ndarray,
    rel_time: np.ndarray,
    *,
    onset_rel: float,
    baseline_window: Tuple[float, float] = (-120.0, -90.0),
    target_window: Tuple[float, float] = (0.0, 10.0),
    min_baseline_bins: int = 150,
) -> np.ndarray:
    """Re-reference an existing affine robust-z trace to the frozen baseline.

    Robust-z is affine-equivariant. Re-centering the cached trace over the
    required baseline is therefore algebraically equivalent to re-centering
    the underlying log-power, without target-driven band selection.
    """
    z = np.asarray(z_trace, float)
    rel = np.asarray(rel_time, float) - float(onset_rel)
    bidx = np.where((rel >= baseline_window[0]) & (rel <= baseline_window[1]))[0]
    tidx = np.where((rel >= target_window[0]) & (rel <= target_window[1]))[0]
    out = np.full(z.shape[0], np.nan)
    if bidx.size < min_baseline_bins or tidx.size == 0:
        return out
    for ch in range(z.shape[0]):
        base = z[ch, bidx]
        base = base[np.isfinite(base)]
        target = z[ch, tidx]
        if base.size < min_baseline_bins or not np.any(np.isfinite(target)):
            continue
        med = float(np.median(base))
        mad = float(np.median(np.abs(base - med)))
        if not np.isfinite(mad) or mad <= EPS:
            continue
        out[ch] = float(np.nanmean((target - med) / (1.4826 * mad)))
    return out


def signed_axis_label(
    activation: np.ndarray,
    activation_names: Sequence[str],
    axis_names: Sequence[str],
    direction_basis: np.ndarray,
    support_q: np.ndarray,
    *,
    min_common_contacts: int = 4,
) -> Dict[str, object]:
    lookup = {str(name): i for i, name in enumerate(activation_names)}
    y = np.full(len(axis_names), np.nan)
    for i, name in enumerate(axis_names):
        j = lookup.get(str(name))
        if j is not None:
            y[i] = activation[j]
    q = np.asarray(support_q, float)
    b = np.asarray(direction_basis, float)
    good = np.isfinite(y) & np.isfinite(b) & (q > 0)
    if np.sum(good) < int(min_common_contacts):
        return {"coefficient": float("nan"), "n_common": int(np.sum(good)), "field": y}
    yc = weighted_center(y, np.where(good, q, 0.0))
    coefficient = weighted_inner(yc, b, np.where(good, q, 0.0))
    return {"coefficient": coefficient, "n_common": int(np.sum(good)), "field": y}


def event_feature_matrix(
    ranks: np.ndarray,
    bools: np.ndarray,
    lag_raw: np.ndarray,
    event_times: np.ndarray,
    direction_basis: np.ndarray,
    support_q: np.ndarray,
    axis_coordinate: np.ndarray,
    *,
    frequency_centroid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Sequence[str]]:
    """Build rank-first per-event features; never relabel frequency as energy."""
    ranks = np.asarray(ranks, float)
    bools = np.asarray(bools, bool)
    lag_raw = np.asarray(lag_raw, float)
    event_times = np.asarray(event_times, float)
    masked = masked_normalized_ranks(ranks, bools)
    n_ev = ranks.shape[1]
    out = np.full((n_ev, 7), np.nan, dtype=float)
    if frequency_centroid is None:
        frequency_centroid = np.full_like(ranks, np.nan)
    frequency_centroid = np.asarray(frequency_centroid, float)
    for e in range(n_ev):
        part = bools[:, e] & np.isfinite(masked[:, e])
        if not np.any(part):
            continue
        q = np.where(part, np.maximum(support_q, EPS), 0.0)
        ev = weighted_center(masked[:, e], q)
        out[e, 0] = weighted_inner(ev, direction_basis, q)
        out[e, 1] = float(np.mean(part))
        coord = axis_coordinate[part]
        coord = coord[np.isfinite(coord)]
        out[e, 2] = float(np.mean(coord)) if coord.size else 0.0
        out[e, 3] = float(np.std(coord)) if coord.size else 0.0
        freq = frequency_centroid[:, e]
        freq = freq[part & np.isfinite(freq)]
        out[e, 4] = float(np.median(freq)) if freq.size else np.nan
        lag = lag_raw[:, e]
        lag = lag[part & np.isfinite(lag)]
        out[e, 5] = float(np.max(lag) - np.min(lag)) if lag.size >= 2 else 0.0
    dt = np.diff(event_times, prepend=np.nan)
    out[:, 6] = np.log1p(np.maximum(dt, 0.0))
    names = (
        "signed_direction",
        "participation_fraction",
        "axial_centroid",
        "axial_width",
        "hfo_frequency_centroid",
        "propagation_duration",
        "log_inter_event_interval",
    )
    return out, names


def fit_prefix_standardizer(features: np.ndarray, indices: Sequence[int]):
    values = np.asarray(features, float)[np.asarray(indices, int)]
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    scale = np.where(np.isfinite(mad) & (mad > EPS), 1.4826 * mad, 1.0)
    median = np.where(np.isfinite(median), median, 0.0)
    return median, scale


def apply_standardizer(features: np.ndarray, median: np.ndarray, scale: np.ndarray):
    out = (np.asarray(features, float) - median) / scale
    return np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0)


def swap_ab_features(features: np.ndarray) -> np.ndarray:
    """A/B swap: signed direction and signed axial centroid reverse."""
    out = np.asarray(features, float).copy()
    out[..., 0] *= -1
    out[..., 2] *= -1
    return out


@dataclass(frozen=True)
class NetworkShape:
    n_positions: int = 12
    n_e_per_position: int = 4
    n_i_per_position: int = 1

    @property
    def n_units(self) -> int:
        return self.n_positions * (self.n_e_per_position + self.n_i_per_position)


def _torch_imports():
    import torch
    from torch import nn
    from torch.nn import functional as F

    return torch, nn, F


class LREICTRNN:  # intentionally initialized as nn.Module at runtime below
    """Event-driven 48E/12I continuous-time low-rank recurrent core."""

    def __new__(cls, *args, **kwargs):
        torch, nn, _ = _torch_imports()

        class _Impl(nn.Module):
            def __init__(
                self,
                input_dim: int,
                rank: int,
                *,
                shape: NetworkShape = NetworkShape(),
                dale: bool = False,
                use_local: bool = True,
                use_slow: bool = True,
                use_local_inhibition: bool = True,
                tau_seconds: float = 30.0,
                tau_slow_seconds: float = 900.0,
                max_step_seconds: float = 30.0,
                max_substeps: int = 32,
                state_clip: float = 8.0,
            ):
                super().__init__()
                self.input_dim = int(input_dim)
                self.rank = int(rank)
                self.shape = shape
                self.n_units = shape.n_units
                self.dale = bool(dale)
                self.use_local = bool(use_local)
                self.use_slow = bool(use_slow)
                self.use_local_inhibition = bool(use_local_inhibition)
                self.tau_seconds = float(tau_seconds)
                self.tau_slow_seconds = float(tau_slow_seconds)
                self.max_step_seconds = float(max_step_seconds)
                self.max_substeps = int(max_substeps)
                self.state_clip = float(state_clip)

                unit_types = []
                positions = []
                for pos in range(shape.n_positions):
                    unit_types.extend([1.0] * shape.n_e_per_position + [-1.0] * shape.n_i_per_position)
                    positions.extend([float(pos)] * (shape.n_e_per_position + shape.n_i_per_position))
                self.register_buffer("unit_sign", torch.tensor(unit_types))
                self.register_buffer("unit_position", torch.tensor(positions))
                self.register_buffer("e_mask", torch.tensor(np.asarray(unit_types) > 0))
                self.input = nn.Linear(self.input_dim, self.n_units)
                self.bias = nn.Parameter(torch.zeros(self.n_units))
                self.local_raw = nn.Parameter(torch.tensor([0.35, 0.50, 0.45, 0.25]))
                self.local_log_length = nn.Parameter(torch.log(torch.tensor([3.0, 1.2, 1.5, 1.0])))
                if self.rank > 0:
                    if self.dale:
                        self.factor_a_raw = nn.Parameter(torch.randn(self.n_units, self.rank) * 0.05)
                        self.factor_b_raw = nn.Parameter(torch.randn(self.n_units, self.rank) * 0.05)
                    else:
                        self.factor_m = nn.Parameter(torch.randn(self.n_units, self.rank) * 0.05)
                        self.factor_n = nn.Parameter(torch.randn(self.n_units, self.rank) * 0.05)
                self.slow_beta_raw = nn.Parameter(torch.tensor(0.0))

            def local_matrix(self):
                _, _, F = _torch_imports()
                post_e = self.e_mask[:, None]
                pre_e = self.e_mask[None, :]
                pair = torch.zeros((self.n_units, self.n_units), dtype=self.bias.dtype, device=self.bias.device)
                pair = torch.where(post_e & pre_e, torch.tensor(0, device=pair.device), pair)
                pair = torch.where(post_e & ~pre_e, torch.tensor(1, device=pair.device), pair)
                pair = torch.where(~post_e & pre_e, torch.tensor(2, device=pair.device), pair)
                pair = torch.where(~post_e & ~pre_e, torch.tensor(3, device=pair.device), pair)
                strength = F.softplus(self.local_raw)
                length = F.softplus(self.local_log_length) + 0.25
                distance = torch.abs(self.unit_position[:, None] - self.unit_position[None, :])
                w = strength[pair.long()] * torch.exp(-distance / length[pair.long()])
                if not self.use_local_inhibition:
                    w = torch.where(pre_e, w, torch.zeros_like(w))
                w = w * self.unit_sign[None, :]
                return w / float(self.shape.n_e_per_position + self.shape.n_i_per_position)

            def low_rank_matrix(self):
                _, _, F = _torch_imports()
                if self.rank == 0:
                    return torch.zeros((self.n_units, self.n_units), device=self.bias.device)
                if self.dale:
                    a = F.softplus(self.factor_a_raw)
                    b = F.softplus(self.factor_b_raw)
                    return (a @ b.T) * self.unit_sign[None, :] / float(self.n_units)
                return (self.factor_m @ self.factor_n.T) / float(self.n_units)

            def recurrent_matrix(self):
                local = (
                    self.local_matrix()
                    if self.use_local
                    else torch.zeros(
                        (self.n_units, self.n_units),
                        device=self.bias.device,
                        dtype=self.bias.dtype,
                    )
                )
                return local + self.low_rank_matrix()

            def forward(self, events, delta_t, mask, initial_state=None, return_sequence=True):
                batch, steps, _ = events.shape
                x = (
                    torch.zeros((batch, self.n_units), device=events.device, dtype=events.dtype)
                    if initial_state is None
                    else initial_state
                )
                h = torch.zeros((batch, 1), device=events.device, dtype=events.dtype)
                W = self.recurrent_matrix()
                beta = torch.nn.functional.softplus(self.slow_beta_raw)
                sequence = []
                for t in range(steps):
                    active = mask[:, t].to(events.dtype)[:, None]
                    dt = torch.clamp(delta_t[:, t], min=0.0)
                    n_sub = torch.clamp(
                        torch.ceil(dt / self.max_step_seconds), min=1, max=self.max_substeps
                    )
                    # All samples share a bounded Python loop; inactive samples are masked.
                    max_n = int(torch.max(n_sub).detach().cpu())
                    for sub in range(max_n):
                        take = (n_sub > sub).to(events.dtype)[:, None] * active
                        step_dt = dt[:, None] / n_sub[:, None]
                        alpha = (1.0 - torch.exp(-step_dt / self.tau_seconds)) * take
                        alpha_h = (1.0 - torch.exp(-step_dt / self.tau_slow_seconds)) * take
                        rate = torch.tanh(x)
                        if self.use_slow:
                            gain_i = torch.exp(-beta * torch.clamp(h, -3.0, 3.0))
                            rate = torch.where(self.e_mask[None, :], rate, rate * gain_i)
                        drive = rate @ W.T + self.bias
                        x = x + alpha * (-x + drive)
                        x = torch.clamp(x, -self.state_clip, self.state_clip)
                        e_rate = torch.relu(rate[:, self.e_mask]).mean(dim=1, keepdim=True)
                        if self.use_slow:
                            h = h + alpha_h * (-h + e_rate)
                    impulse = self.input(events[:, t, :])
                    x = x + active * impulse
                    x = torch.clamp(x, -self.state_clip, self.state_clip)
                    sequence.append(torch.cat([x, h], dim=1))
                seq = torch.stack(sequence, dim=1)
                lengths = mask.long().sum(dim=1).clamp(min=1) - 1
                final = seq[torch.arange(batch, device=events.device), lengths]
                return (final, seq) if return_sequence else final

        return _Impl(*args, **kwargs)


class InterictalPretrainer:
    """Self-supervised heads for masked reconstruction and future event dynamics."""

    def __new__(cls, core, input_dim: int):
        torch, nn, _ = _torch_imports()

        class _Impl(nn.Module):
            def __init__(self, core, input_dim):
                super().__init__()
                self.core = core
                hidden = core.n_units + 1
                self.reconstruct = nn.Linear(hidden, input_dim)
                self.next_event = nn.Linear(hidden, 4)
                self.future_balance = nn.Linear(hidden, 3)

            def forward(self, events, delta_t, mask):
                _, seq = self.core(events, delta_t, mask, return_sequence=True)
                return {
                    "sequence": seq,
                    "reconstruct": self.reconstruct(seq),
                    "next_event": self.next_event(seq),
                    "future_balance": self.future_balance(seq),
                }

        return _Impl(core, input_dim)
