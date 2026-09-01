"""``P_local`` and ``P_slow``: session-preserving producers of a frozen state.

What changes relative to v0.1, and why each change is load-bearing:

* **the batch dimension is recorded segments, not artificial slices.**  v0.1 cut
  the TRAIN range into ``n_streams=8`` equal pieces purely for throughput; each
  piece began with a re-initialised state in the middle of a recording.  For a
  next-event model that is a mild handicap, but for a model whose whole claim is
  about hour-scale state it destroys the object being measured.  Here each of the
  ``B`` slots streams one real carry segment at a time, in order, carrying state
  and only detaching the graph at chunk edges (CC 7.2, EI 3).

* **the future-block heads are read at the fixed 5-min anchors**, not at every
  event.  Training them per event would weight a busy hour ten times more than a
  quiet hour of the same length -- exactly the re-weighting CC 5.2 forbids -- and
  would train the heads on a different object from the one they are scored on.

* **the long-horizon heads read the slow state only** (SP 2).  This shapes what
  the objective pushes into ``z_slow``; it does not define the science.  The
  load-bearing evaluation freezes the *whole* state and re-fits a common readout
  on it, so a producer cannot win by naming a latent "slow".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..dataset import SubjectSequence
from ..model import (
    ContinuousState,
    DataShape,
    EncoderConfig,
    EventEncoder,
    InputStats,
    PredictionHeads,
    StateConfig,
    TargetStats,
    gaussian_nll,
    lognormal_nll,
)
from ..train import ENDPOINTS, _data_shape, _load_geometry, estimate_stats
from .scoring import (
    gaussian_nll_from_moments,
    negative_binomial_nll,
    participation_nll_from_counts,
)
from .subject import SubjectTimeline
from .timeline import SPLIT_NAMES


@dataclass
class ProducerConfig:
    """One producer arm.  ``P_local`` and ``P_slow`` share encoder and capacity."""

    name: str
    use_future_heads: bool
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        use_participation=True, use_exact_delay=True, use_tied_groups=True,
        use_legacy_rank=False, use_waveform=True, use_multiband=True,
        use_geometry=True,
    ))
    state: StateConfig = field(default_factory=StateConfig)
    chunk_events: int = 128
    batch_segments: int = 8
    max_epochs: int = 24
    patience: int = 4
    min_epochs: int = 3
    lr_encoder: float = 3e-4
    lr_state: float = 1e-3
    lr_heads: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    max_train_seconds: float = 5400.0

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k not in ("encoder", "state")}
        out["encoder"] = asdict(self.encoder)
        out["state"] = asdict(self.state)
        return out


def build_producer_configs() -> dict[str, ProducerConfig]:
    """The two recurrent producers of the first round (CC 3.2, 3.3)."""

    local = ProducerConfig(name="P_local", use_future_heads=False)
    slow = ProducerConfig(name="P_slow", use_future_heads=True,
                          encoder=local.encoder, state=local.state)
    return {"P_local": local, "P_slow": slow}


# --------------------------------------------------------------------- heads


class FutureBlockHeads(nn.Module):
    """One head per physical horizon, reading the candidate slow state.

    Emits exactly the parameters the shared scoring layer consumes, so the
    producer's own future prediction and the re-fitted readout are scored by the
    same likelihoods.
    """

    def __init__(self, d_slow: int, n_contacts: int, n_dims: int,
                 horizons_seconds: Sequence[float], hidden: int = 128):
        super().__init__()
        self.horizons = tuple(float(h) for h in horizons_seconds)
        self.trunk = nn.ModuleList([
            nn.Sequential(nn.Linear(d_slow, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden), nn.GELU())
            for _ in self.horizons
        ])
        self.count = nn.ModuleList([nn.Linear(hidden, 1) for _ in self.horizons])
        self.log_alpha = nn.Parameter(torch.zeros(len(self.horizons)))
        self.participation = nn.ModuleList(
            [nn.Linear(hidden, n_contacts) for _ in self.horizons]
        )
        self.continuous = nn.ModuleList(
            [nn.Linear(hidden, 2 * n_dims) for _ in self.horizons]
        )
        self.n_dims = int(n_dims)

    @torch.no_grad()
    def initialise_from_targets(self, marginals: Sequence[Mapping[str, np.ndarray]]) -> None:
        for h_i, marg in enumerate(marginals):
            self.count[h_i].bias.copy_(torch.as_tensor(marg["log_mu"], dtype=torch.float32))
            self.count[h_i].weight.mul_(0.01)
            self.log_alpha.data[h_i] = float(marg["log_alpha"][0])
            self.participation[h_i].bias.copy_(
                torch.as_tensor(marg["logit"], dtype=torch.float32)
            )
            self.participation[h_i].weight.mul_(0.01)
            bias = torch.cat([
                torch.as_tensor(marg["cont_mean"], dtype=torch.float32),
                torch.as_tensor(marg["cont_log_sigma"], dtype=torch.float32),
            ])
            self.continuous[h_i].bias.copy_(bias)
            self.continuous[h_i].weight.mul_(0.01)

    def forward(self, z_slow: Tensor, h_i: int) -> dict[str, Tensor]:
        h = self.trunk[h_i](z_slow)
        cont = self.continuous[h_i](h)
        return {
            "log_mu": self.count[h_i](h)[:, 0],
            "log_alpha": self.log_alpha[h_i],
            "logit": self.participation[h_i](h),
            "mu": cont[:, : self.n_dims],
            "log_sigma": cont[:, self.n_dims:].clamp(-6.0, 6.0),
        }


class GroupEventStateProducer(nn.Module):
    def __init__(self, cfg: ProducerConfig, shape: DataShape, geometry: Tensor | None,
                 n_dims: int, horizons: Sequence[float],
                 generator: torch.Generator | None = None,
                 stats: InputStats | None = None, targets: TargetStats | None = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = EventEncoder(cfg.encoder, shape, geometry, stats)
        self.state = ContinuousState(cfg.state, cfg.encoder.d_event, generator)
        self.heads = PredictionHeads(cfg.state.d_fast + cfg.state.d_slow, shape)
        if targets is not None:
            self.heads.initialise_from_targets(targets)
        self.future = (
            FutureBlockHeads(cfg.state.d_slow, shape.n_contacts, n_dims, horizons)
            if cfg.use_future_heads else None
        )


# --------------------------------------------------------------------- streaming


@dataclass(frozen=True)
class EventRange:
    segment_id: int
    lo: int
    hi: int


def split_segment_ranges(tl: SubjectTimeline, split_name: str) -> list[EventRange]:
    """Event index ranges of every carry segment, clipped to one split.

    A segment may straddle a split boundary; the piece inside the split is what
    the trainer streams, so no target and no carried state crosses the boundary.
    """

    lo_edge = -np.inf if split_name == "train" else float(
        tl.split.boundary_epochs[0 if split_name == "val" else 1]
    )
    hi_edge = np.inf if split_name == "test" else float(
        tl.split.boundary_epochs[0 if split_name == "train" else 1]
    )
    out: list[EventRange] = []
    for seg in tl.segments:
        member = np.flatnonzero(tl.event_segment == seg.segment_id)
        if member.size == 0:
            continue
        t = tl.event_times[member]
        keep = (t >= max(seg.start_epoch, lo_edge)) & (t < min(seg.stop_epoch, hi_edge))
        if not keep.any():
            continue
        idx = member[keep]
        out.append(EventRange(seg.segment_id, int(idx[0]), int(idx[-1]) + 1))
    return out


def full_segment_ranges(tl: SubjectTimeline) -> list[EventRange]:
    """Whole carry segments, ignoring split edges.

    Evaluation streams these and scores only the events of the target split, so
    a val/test anchor inherits a state warmed causally from its segment's start.
    Streaming the split slice alone was the v0.1 defect: the warm-up terminal
    state was discarded and every held-out chain began from initialisation.
    """

    out: list[EventRange] = []
    for seg in tl.segments:
        member = np.flatnonzero(tl.event_segment == seg.segment_id)
        if member.size:
            out.append(EventRange(seg.segment_id, int(member[0]), int(member[-1]) + 1))
    return out


def segments_touching_split(tl: SubjectTimeline, split_name: str) -> list[EventRange]:
    """Full ranges of the segments that contain at least one event of a split.

    Evaluation still replays each such segment from its own start, so warm-up is
    unchanged; segments holding no event of the split simply cannot contribute a
    loss term, so replaying them every epoch is pure cost.
    """

    mask = split_event_mask(tl, split_name)
    touched = set(np.unique(tl.event_segment[mask]).tolist())
    return [r for r in full_segment_ranges(tl) if r.segment_id in touched]


def split_event_mask(tl: SubjectTimeline, split_name: str) -> np.ndarray:
    lo = -np.inf if split_name == "train" else float(
        tl.split.boundary_epochs[0 if split_name == "val" else 1]
    )
    hi = np.inf if split_name == "test" else float(
        tl.split.boundary_epochs[0 if split_name == "train" else 1]
    )
    return (tl.event_times >= lo) & (tl.event_times < hi)


class SegmentStreamer:
    """``B`` slots, each streaming one carry segment at a time, in order.

    Segments are shuffled between epochs; chunks inside a segment never are.
    When a slot's segment ends it takes the next one and its state is reset --
    the only place a reset is allowed.
    """

    def __init__(self, ranges: Sequence[EventRange], batch: int, chunk: int,
                 rng: np.random.Generator):
        self.ranges = list(ranges)
        self.batch = max(1, int(batch))
        self.chunk = max(1, int(chunk))
        self.rng = rng

    def epoch(self):
        order = list(self.rng.permutation(len(self.ranges)))
        cursor = 0
        slots: list[tuple[int, int] | None] = [None] * self.batch
        resets = [False] * self.batch
        while True:
            for b in range(self.batch):
                if slots[b] is None and cursor < len(order):
                    r = self.ranges[order[cursor]]
                    slots[b] = (r.lo, r.hi)
                    resets[b] = True
                    cursor += 1
            if all(s is None for s in slots):
                return
            pos = np.zeros((self.batch, self.chunk), dtype=np.int64)
            valid = np.zeros((self.batch, self.chunk), dtype=bool)
            reset = np.array(resets, dtype=bool)
            for b, s in enumerate(slots):
                if s is None:
                    continue
                lo, hi = s
                take = min(self.chunk, hi - lo)
                pos[b, :take] = np.arange(lo, lo + take)
                valid[b, :take] = True
                slots[b] = (lo + take, hi) if lo + take < hi else None
                resets[b] = False
            yield pos, valid, reset
            resets = [False if s is not None else False for s in slots]


# --------------------------------------------------------------------- anchors


@dataclass(frozen=True)
class AnchorTargets:
    """Everything the future heads need for the anchors of one split."""

    anchor_index: np.ndarray         # (A,) into the full grid
    last_event_pos: np.ndarray       # (A,) into the event stream, -1 if none
    segment_start: np.ndarray        # (A,) epoch of the anchor's segment start
    dt: np.ndarray                   # (A,) seconds from that reference to the anchor
    eligible: np.ndarray             # (A, H)
    stats: list[Any]                 # per horizon WindowStats over all A rows


def build_anchor_targets(tl: SubjectTimeline, split_name: str | None) -> AnchorTargets:
    """Anchors of one split, or every anchor when ``split_name`` is None."""

    grid = tl.grid
    idx = (
        np.arange(grid.n_anchors)
        if split_name is None
        else np.flatnonzero(grid.split_mask(split_name))
    )
    seg_start = np.array(
        [tl.segments[s].start_epoch for s in grid.segment_index[idx]], dtype=np.float64
    )
    last = grid.last_event_pos[idx]
    ref = np.where(last >= 0, tl.event_times[np.clip(last, 0, None)], seg_start)
    stats = [
        tl.builder.window_stats(grid.window_lo[idx, h], grid.window_hi[idx, h])
        for h in range(len(grid.horizons_seconds))
    ]
    return AnchorTargets(
        anchor_index=idx,
        last_event_pos=last,
        segment_start=seg_start,
        dt=(grid.t_anchor[idx] - ref).astype(np.float64),
        eligible=grid.eligible[idx],
        stats=stats,
    )


def _anchor_lookup(targets: AnchorTargets) -> tuple[np.ndarray, np.ndarray]:
    """Anchors sorted by their attaching event, for O(log n) chunk lookup."""

    order = np.argsort(targets.last_event_pos, kind="stable")
    return order, targets.last_event_pos[order]


# --------------------------------------------------------------------- losses


def _local_losses(pred, timing_pred, truth, dt, dt_valid, slot_valid):
    from ..train import _endpoint_losses

    return _endpoint_losses(pred, timing_pred, truth, dt, dt_valid, slot_valid)


def _future_loss_for_horizon(
    out: Mapping[str, Tensor], stats, rows: np.ndarray, device: torch.device
) -> tuple[Tensor, dict[str, float]]:
    count = torch.as_tensor(np.asarray(stats.count)[rows], dtype=torch.float64, device=device)
    k = torch.as_tensor(np.asarray(stats.sum_participation)[rows], dtype=torch.float64, device=device)
    nv = torch.as_tensor(np.asarray(stats.n_valid_mark)[rows], dtype=torch.float64, device=device)
    sx = torch.as_tensor(np.asarray(stats.sum_x)[rows], dtype=torch.float64, device=device)
    sx2 = torch.as_tensor(np.asarray(stats.sum_x2)[rows], dtype=torch.float64, device=device)

    n_anchor = max(float(count.numel()), 1.0)
    n_pair = max(float(count.sum() * k.shape[1]), 1.0)
    n_cell = max(float(nv.sum() * sx.shape[1]), 1.0)

    nll_count = negative_binomial_nll(
        count, out["log_mu"].double(), out["log_alpha"].double().expand_as(count)
    ).sum() / n_anchor
    nll_part = participation_nll_from_counts(k, count, out["logit"].double()).sum() / n_pair
    nll_cont = gaussian_nll_from_moments(
        nv, sx, sx2, out["mu"].double(), out["log_sigma"].double()
    ).sum() / n_cell
    total = nll_count + nll_part + nll_cont
    return total, {
        "count": float(nll_count.detach()),
        "participation": float(nll_part.detach()),
        "continuous": float(nll_cont.detach()),
    }


def event_dt_prev(tl: SubjectTimeline) -> np.ndarray:
    """Seconds since the previous event *in the same carry segment*.

    ``SubjectSequence.dt_prev`` resets only at v0.1 recorded-session starts, so
    reusing it would hand the state a real interval across a seizure and its
    postictal exclusion.  NaN marks "no predecessor in this segment".
    """

    dt = np.full(tl.event_times.size, np.nan, dtype=np.float64)
    dt[1:] = np.diff(tl.event_times)
    boundary = np.ones(tl.event_times.size, dtype=bool)
    boundary[1:] = tl.event_segment[1:] != tl.event_segment[:-1]
    dt[boundary] = np.nan
    return dt


# --------------------------------------------------------------------- passes


def _to_device(batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    from ..train import _to_device as _v01_to_device

    return _v01_to_device(batch, device)


class _Accumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def add(self, key: str, total: float, count: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(total)
        self.counts[key] = self.counts.get(key, 0.0) + float(count)

    def means(self) -> dict[str, float]:
        return {
            k: (self.sums[k] / self.counts[k]) if self.counts[k] > 0 else float("nan")
            for k in self.sums
        }


def run_session_pass(
    model: GroupEventStateProducer,
    tl: SubjectTimeline,
    seq: SubjectSequence,
    ranges: Sequence[EventRange],
    targets: AnchorTargets | None,
    device: torch.device,
    cfg: ProducerConfig,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    weights: Mapping[str, float] | None = None,
    rng: np.random.Generator | None = None,
    collect_states: bool = False,
    collect_event_states: bool = False,
    grad_norms: list[float] | None = None,
    score_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """One session-preserving pass over a split.

    Slots stream whole carry segments in order.  ``detach`` at every chunk edge,
    ``reset`` only when a slot takes a new segment (CC 7.2).
    """

    dt_prev = event_dt_prev(tl)
    stream = tl.stream_positions
    horizons = tl.grid.horizons_seconds
    lam = dict(weights or {})
    accum = _Accumulator()
    extra: dict[str, Any] = {"n_events": 0, "n_anchor_terms": 0, "n_nonfinite_steps": 0}
    anchor_state: np.ndarray | None = None
    if collect_states and targets is not None:
        anchor_state = np.zeros(
            (targets.anchor_index.size, cfg.state.d_fast + cfg.state.d_slow),
            dtype=np.float32,
        )
    event_state: np.ndarray | None = None
    if collect_event_states:
        # The state the model reads to predict *this* event: after relaxing over
        # the real interval, before the event itself is encoded.  H2a conditions
        # on exactly this quantity.
        event_state = np.zeros(
            (tl.event_times.size, cfg.state.d_fast + cfg.state.d_slow), dtype=np.float32
        )
    if targets is not None:
        order, sorted_last = _anchor_lookup(targets)
    streamer = SegmentStreamer(
        ranges, cfg.batch_segments, cfg.chunk_events,
        rng if rng is not None else np.random.default_rng(0),
    )
    b_slots = streamer.batch
    fast, slow = model.state.initial(b_slots, device)
    init_f, init_s = model.state.initial(b_slots, device)

    for pos, valid, reset in streamer.epoch():
        n_step = pos.shape[1]
        flat = pos.reshape(-1)
        raw = seq.gather_positions(stream[flat])
        batch = _to_device(raw, device)
        score = valid if score_mask is None else (valid & score_mask[pos])
        slot_valid = torch.from_numpy(score.reshape(-1)).to(device)

        dt_np = np.where(valid, dt_prev[pos], np.nan)
        dt_all = torch.from_numpy(np.nan_to_num(dt_np, nan=0.0).reshape(-1)).float().to(device)
        dt_valid = torch.from_numpy(np.isfinite(dt_np).reshape(-1)).to(device)
        dt_step = dt_all.reshape(b_slots, n_step)

        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=cfg.amp and device.type == "cuda"):
            event_emb, _tokens = model.encoder(batch)
        event_emb = event_emb.float().reshape(b_slots, n_step, -1)

        reset_t = torch.from_numpy(reset).to(device).unsqueeze(-1)
        fast = torch.where(reset_t, init_f, fast)
        slow = torch.where(reset_t, init_s, slow)

        taus = model.state.taus()
        timing_list, content_list, post_fast, post_slow = [], [], [], []
        for step in range(n_step):
            timing_list.append(torch.cat([fast, slow], dim=-1))
            fast_e, slow_e = model.state.evolve(fast, slow, dt_step[:, step], taus)
            content_list.append(torch.cat([fast_e, slow_e], dim=-1))
            fast, slow = model.state.update(fast_e, slow_e, event_emb[:, step])
            post_fast.append(fast)
            post_slow.append(slow)
        timing_states = torch.stack(timing_list, 1).reshape(b_slots * n_step, -1)
        content_states = torch.stack(content_list, 1).reshape(b_slots * n_step, -1)
        if event_state is not None:
            keep = valid.reshape(-1)
            event_state[flat[keep]] = (
                content_states.detach().float().cpu().numpy()[keep]
            )

        pred = model.heads(content_states)
        timing_pred = model.heads(timing_states)
        losses = _local_losses(pred, timing_pred, batch, dt_all, dt_valid, slot_valid)
        loss = torch.zeros((), device=device, dtype=torch.float32)
        for key, (total, count) in losses.items():
            accum.add(f"local.{key}", float(total.detach()), float(count.detach()))
            if float(count) > 0:
                loss = loss + total / count

        future_terms: list[tuple[int, Tensor]] = []
        if targets is not None:
            pf = torch.stack(post_fast, 1)          # (B, T, d_fast)
            ps = torch.stack(post_slow, 1)          # (B, T, d_slow)
            lo = np.searchsorted(sorted_last, flat, side="left")
            hi = np.searchsorted(sorted_last, flat, side="right")
            take = np.flatnonzero((hi > lo) & valid.reshape(-1))
            if take.size:
                slot_of, anchor_rows = [], []
                for j in take:
                    for a in order[lo[j]:hi[j]]:
                        slot_of.append(j)
                        anchor_rows.append(a)
                slot_of = np.asarray(slot_of, dtype=np.int64)
                anchor_rows = np.asarray(anchor_rows, dtype=np.int64)
                b_idx, t_idx = np.divmod(slot_of, n_step)
                sel_f = pf[b_idx, t_idx]
                sel_s = ps[b_idx, t_idx]
                dt_a = torch.from_numpy(
                    targets.dt[anchor_rows].astype(np.float32)
                ).to(device)
                ev_f, ev_s = model.state.evolve(sel_f, sel_s, dt_a, taus)
                if anchor_state is not None:
                    anchor_state[anchor_rows] = (
                        torch.cat([ev_f, ev_s], dim=-1).detach().float().cpu().numpy()
                    )
                for h_i, horizon in enumerate(horizons if model.future is not None else ()):
                    keep = np.flatnonzero(targets.eligible[anchor_rows, h_i])
                    if keep.size == 0:
                        continue
                    out = model.future(ev_s[torch.from_numpy(keep).to(device)], h_i)
                    term, parts = _future_loss_for_horizon(
                        out, targets.stats[h_i], anchor_rows[keep], device
                    )
                    for name, value in parts.items():
                        accum.add(f"future{int(horizon)}.{name}", value, 1.0)
                    future_terms.append((h_i, term))
                    extra["n_anchor_terms"] += int(keep.size)
        for h_i, term in future_terms:
            loss = loss + float(lam.get(f"future_{h_i}", 1.0)) * term.float()

        if train and optimizer is not None:
            if torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if torch.isfinite(norm):
                    optimizer.step()
                if grad_norms is not None:
                    grad_norms.append(float(norm))
            else:
                extra["n_nonfinite_steps"] += 1
        fast = fast.detach()
        slow = slow.detach()
        extra["n_events"] += int(valid.sum())

    extra["anchor_state"] = anchor_state
    extra["event_state"] = event_state
    return accum.means(), extra


# --------------------------------------------------------------------- training


def _future_marginals(tl: SubjectTimeline, targets: AnchorTargets) -> list[dict[str, np.ndarray]]:
    """TRAIN marginals per horizon, used to initialise the future heads."""

    out: list[dict[str, np.ndarray]] = []
    for h_i in range(len(tl.grid.horizons_seconds)):
        keep = np.flatnonzero(targets.eligible[:, h_i])
        stats = targets.stats[h_i]
        n = np.asarray(stats.count, dtype=np.float64)[keep]
        if n.size == 0:
            n = np.zeros(1)
        mu = max(float(n.mean()), 1e-6)
        var = max(float(n.var()), mu * 1.000001)
        k = np.asarray(stats.sum_participation, dtype=np.float64)[keep].sum(0)
        rate = np.clip(k / max(float(n.sum()), 1.0), 1e-4, 1 - 1e-4)
        nv = max(float(np.asarray(stats.n_valid_mark)[keep].sum()), 1.0)
        m = np.asarray(stats.sum_x, dtype=np.float64)[keep].sum(0) / nv
        v = np.maximum(
            np.asarray(stats.sum_x2, dtype=np.float64)[keep].sum(0) / nv - m ** 2, 1e-6
        )
        out.append({
            "log_mu": np.array([math.log(mu)]),
            "log_alpha": np.array([math.log(max((var - mu) / (mu * mu), 1e-6))]),
            "logit": np.log(rate / (1 - rate)),
            "cont_mean": m,
            "cont_log_sigma": 0.5 * np.log(v),
        })
    return out


def balance_future_weights(
    model: GroupEventStateProducer,
    tl: SubjectTimeline,
    seq: SubjectSequence,
    ranges: Sequence[EventRange],
    targets: AnchorTargets,
    device: torch.device,
    cfg: ProducerConfig,
    *,
    n_chunks: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Scale each horizon's loss to the local loss by its initial gradient norm.

    SP 2 fixes the weights this way and then freezes them: they are set on TRAIN
    at initialisation and never touched again, so no development-set number can
    have influenced them.
    """

    if model.future is None:
        return {}
    shared = [p for p in list(model.encoder.parameters()) + list(model.state.parameters())]
    norms: dict[str, list[float]] = {}
    streamer = SegmentStreamer(ranges, cfg.batch_segments, cfg.chunk_events,
                               np.random.default_rng(seed))
    dt_prev = event_dt_prev(tl)
    order, sorted_last = _anchor_lookup(targets)
    seen = 0
    for pos, valid, reset in streamer.epoch():
        if seen >= n_chunks:
            break
        seen += 1
        flat = pos.reshape(-1)
        batch = _to_device(seq.gather_positions(tl.stream_positions[flat]), device)
        n_step = pos.shape[1]
        b_slots = pos.shape[0]
        dt_np = np.where(valid, dt_prev[pos], np.nan)
        dt_step = torch.from_numpy(
            np.nan_to_num(dt_np, nan=0.0)).float().to(device)
        event_emb, _ = model.encoder(batch)
        event_emb = event_emb.float().reshape(b_slots, n_step, -1)
        fast, slow = model.state.initial(b_slots, device)
        taus = model.state.taus()
        timing_list, content_list, post_f, post_s = [], [], [], []
        for step in range(n_step):
            timing_list.append(torch.cat([fast, slow], -1))
            fe, se = model.state.evolve(fast, slow, dt_step[:, step], taus)
            content_list.append(torch.cat([fe, se], -1))
            fast, slow = model.state.update(fe, se, event_emb[:, step])
            post_f.append(fast)
            post_s.append(slow)
        pred = model.heads(torch.stack(content_list, 1).reshape(b_slots * n_step, -1))
        timing_pred = model.heads(torch.stack(timing_list, 1).reshape(b_slots * n_step, -1))
        dt_all = dt_step.reshape(-1)
        losses = _local_losses(
            pred, timing_pred, batch, dt_all,
            torch.from_numpy(np.isfinite(dt_np).reshape(-1)).to(device),
            torch.from_numpy(valid.reshape(-1)).to(device),
        )
        local = torch.zeros((), device=device)
        for _key, (total, count) in losses.items():
            if float(count) > 0:
                local = local + total / count
        norms.setdefault("local", []).append(_grad_norm(local, shared))

        pf, ps = torch.stack(post_f, 1), torch.stack(post_s, 1)
        lo = np.searchsorted(sorted_last, flat, side="left")
        hi = np.searchsorted(sorted_last, flat, side="right")
        take = np.flatnonzero((hi > lo) & valid.reshape(-1))
        if take.size == 0:
            continue
        slot_of = np.concatenate([np.full(hi[j] - lo[j], j) for j in take])
        rows = np.concatenate([order[lo[j]:hi[j]] for j in take])
        b_idx, t_idx = np.divmod(slot_of, n_step)
        dt_a = torch.from_numpy(targets.dt[rows].astype(np.float32)).to(device)
        ev_f, ev_s = model.state.evolve(pf[b_idx, t_idx], ps[b_idx, t_idx], dt_a, taus)
        for h_i, horizon in enumerate(tl.grid.horizons_seconds):
            keep = np.flatnonzero(targets.eligible[rows, h_i])
            if keep.size == 0:
                continue
            out = model.future(ev_s[torch.from_numpy(keep).to(device)], h_i)
            term, _parts = _future_loss_for_horizon(
                out, targets.stats[h_i], rows[keep], device
            )
            norms.setdefault(f"future_{h_i}", []).append(_grad_norm(term.float(), shared))

    base = float(np.median(norms.get("local", [1.0]))) or 1.0
    weights: dict[str, float] = {}
    for key, values in norms.items():
        if key == "local":
            continue
        g = float(np.median(values)) if values else 0.0
        weights[key] = float(np.clip(base / g, 1e-3, 1e3)) if g > 0 else 1.0
    return weights


def _grad_norm(loss: Tensor, params: Sequence[Tensor]) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.detach() ** 2).sum())
    return math.sqrt(total)


def _param_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if v.is_floating_point()}


def _update_magnitude(before: Mapping[str, Tensor], model: nn.Module) -> dict[str, float]:
    after = model.state_dict()
    out: dict[str, float] = {}
    for group in ("encoder", "state", "heads", "future"):
        num = den = 0.0
        for key, value in before.items():
            if not key.startswith(group):
                continue
            num += (after[key].detach().float() - value.float()).norm().item() ** 2
            den += value.float().norm().item() ** 2
        if den > 0:
            out[group] = math.sqrt(num) / math.sqrt(den)
    return out


def _auto_chunk(seq: SubjectSequence, requested: int) -> int:
    from ..train import _auto_chunk as _v01_auto_chunk

    return _v01_auto_chunk(seq, requested)


def extract_anchor_states(
    model: GroupEventStateProducer,
    tl: SubjectTimeline,
    seq: SubjectSequence,
    device: torch.device,
    cfg: ProducerConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """State at every fixed-grid anchor, replayed causally from each segment start.

    This is the frozen artefact the nested readout and Agents B/C consume: one
    row per anchor of ``[z_fast, z_slow]`` after propagating the post-event state
    forward by the real elapsed seconds (CC 5.2).
    """

    targets = build_anchor_targets(tl, None)
    model.eval()
    with torch.no_grad():
        _means, extra = run_session_pass(
            model, tl, seq, full_segment_ranges(tl), targets, device, cfg,
            train=False, collect_states=True, collect_event_states=True,
            rng=np.random.default_rng(0),
        )
    states = extra["anchor_state"]
    # Anchors with no preceding event inside their segment start from the learned
    # initial state, propagated by the time since the segment began.
    orphan = np.flatnonzero(targets.last_event_pos < 0)
    if orphan.size:
        with torch.no_grad():
            f0, s0 = model.state.initial(orphan.size, device)
            dt = torch.from_numpy(targets.dt[orphan].astype(np.float32)).to(device)
            fe, se = model.state.evolve(f0, s0, dt, model.state.taus())
            states[orphan] = torch.cat([fe, se], -1).float().cpu().numpy()
    info = {
        "event_state": extra["event_state"],
        "n_anchors": int(states.shape[0]),
        "n_anchors_from_segment_initial_state": int(orphan.size),
        "state_dim": int(states.shape[1]),
        "d_fast": int(cfg.state.d_fast),
        "d_slow": int(cfg.state.d_slow),
    }
    return states, info


def _val_objective(means: Mapping[str, float], weights: Mapping[str, float],
                   horizons: Sequence[float]) -> float:
    """Pre-registered checkpoint-selection objective (SP 5).

    The training objective itself, evaluated on the chronological inner split:
    the local endpoints (``group_size`` excluded, as in v0.1) plus each horizon's
    future-block terms at their frozen weights.  Nothing from the development
    test split enters it.
    """

    local = float(np.nansum([
        means.get(f"local.{k}", np.nan) for k in ENDPOINTS if k != "group_size"
    ]))
    total = local
    for h_i, horizon in enumerate(horizons):
        terms = [means.get(f"future{int(horizon)}.{f}", np.nan)
                 for f in ("count", "participation", "continuous")]
        value = float(np.nansum(terms))
        if np.isfinite(value):
            total += float(weights.get(f"future_{h_i}", 1.0)) * value
    return total


def train_producer(
    tl: SubjectTimeline,
    seq: SubjectSequence,
    cfg: ProducerConfig,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """Train one (patient, producer, seed) with session-preserving carry."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    cfg = replace(cfg, chunk_events=_auto_chunk(seq, cfg.chunk_events))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape = _data_shape(seq)
    geometry = _load_geometry(seq) if cfg.encoder.use_geometry else None
    train_mask = split_event_mask(tl, "train")
    train_positions = tl.stream_positions[train_mask]
    if train_positions.size == 0:
        raise ValueError(f"{tl.subject}: no TRAIN event after segment filtering")
    input_stats, target_stats = estimate_stats(
        seq, 0, len(seq), seed=seed, positions=train_positions
    )
    model = GroupEventStateProducer(
        cfg, shape, geometry.to(device) if geometry is not None else None,
        tl.n_dims, tl.grid.horizons_seconds, generator, input_stats, target_stats,
    ).to(device)

    train_ranges = split_segment_ranges(tl, "train")
    full_ranges = full_segment_ranges(tl)
    train_targets = build_anchor_targets(tl, "train")
    val_targets = build_anchor_targets(tl, "val")
    if model.future is not None:
        model.future.initialise_from_targets(_future_marginals(tl, train_targets))

    groups = [
        {"params": model.encoder.parameters(), "lr": cfg.lr_encoder},
        {"params": model.state.parameters(), "lr": cfg.lr_state},
        {"params": model.heads.parameters(), "lr": cfg.lr_heads},
    ]
    if model.future is not None:
        groups.append({"params": model.future.parameters(), "lr": cfg.lr_heads})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    weights = balance_future_weights(
        model, tl, seq, train_ranges, train_targets, device, cfg, seed=seed
    ) if model.future is not None else {}
    optimizer.zero_grad(set_to_none=True)

    init_snapshot = _param_snapshot(model)
    history: list[dict[str, Any]] = []
    best = {"objective": float("inf"), "epoch": -1}
    best_state: dict[str, Tensor] | None = None
    started = time.time()
    stop_reason = "max_epochs"
    val_mask = split_event_mask(tl, "val")
    val_ranges = segments_touching_split(tl, "val")
    print(f"{tl.subject}/{cfg.name}/seed{seed}: {len(train_ranges)} train segments, "
          f"{int(train_mask.sum())} train events, {len(val_ranges)} val segments, "
          f"weights={ {k: round(v, 4) for k, v in weights.items()} }", flush=True)

    for epoch in range(cfg.max_epochs):
        model.train()
        grads: list[float] = []
        train_means, train_extra = run_session_pass(
            model, tl, seq, train_ranges, train_targets, device, cfg, train=True,
            optimizer=optimizer, weights=weights,
            rng=np.random.default_rng(seed * 1000 + epoch), grad_norms=grads,
        )
        model.eval()
        with torch.no_grad():
            val_means, _val_extra = run_session_pass(
                model, tl, seq, val_ranges, val_targets, device, cfg, train=False,
                weights=weights, rng=np.random.default_rng(0), score_mask=val_mask,
            )
        objective = _val_objective(val_means, weights, tl.grid.horizons_seconds)
        history.append({
            "epoch": epoch,
            "train": train_means,
            "val": val_means,
            "val_objective": objective,
            "grad_norm_mean": float(np.mean(grads)) if grads else float("nan"),
            "n_train_events": int(train_extra["n_events"]),
            "n_train_anchor_terms": int(train_extra["n_anchor_terms"]),
            "n_nonfinite_steps": int(train_extra["n_nonfinite_steps"]),
            "seconds": round(time.time() - started, 1),
        })
        print(f"  epoch {epoch}: val_objective={objective:.5f} "
              f"grad={history[-1]['grad_norm_mean']:.3f} "
              f"anchors={history[-1]['n_train_anchor_terms']} "
              f"{history[-1]['seconds']:.0f}s", flush=True)
        if objective < best["objective"] - 1e-6:
            best = {"objective": objective, "epoch": epoch}
            best_state = _param_snapshot(model)
        elif epoch - best["epoch"] >= cfg.patience and epoch + 1 >= cfg.min_epochs:
            stop_reason = "early_stopping"
            break
        if time.time() - started > cfg.max_train_seconds and epoch + 1 >= cfg.min_epochs:
            stop_reason = "time_budget"
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    updates = _update_magnitude(init_snapshot, model)

    states, state_info = extract_anchor_states(model, tl, seq, device, cfg)
    event_state = state_info.pop("event_state")
    tmp = out_dir / "anchor_state.npz.tmp"
    with tmp.open("wb") as handle:
        np.savez(handle, state=states.astype(np.float32),
                 t_anchor=tl.grid.t_anchor, split_index=tl.grid.split_index,
                 session_id=tl.grid.session_id)
    os.replace(tmp, out_dir / "anchor_state.npz")

    # Per-event state (post-relaxation, pre-update) -- what H2a conditions on.
    tmp = out_dir / "event_state.npz.tmp"
    with tmp.open("wb") as handle:
        np.savez(handle, state=event_state.astype(np.float32),
                 t_event=tl.event_times, segment=tl.event_segment)
    os.replace(tmp, out_dir / "event_state.npz")

    ckpt_tmp = out_dir / "checkpoint.pt.tmp"
    torch.save({
        "state_dict": model.state_dict(),
        "producer": cfg.name,
        "seed": seed,
        "subject": tl.subject,
        "selected_epoch": best["epoch"],
        "future_loss_weights": weights,
        "config": cfg.as_dict(),
        "timeline_config": tl.config.as_dict(),
    }, ckpt_tmp)
    os.replace(ckpt_tmp, out_dir / "checkpoint.pt")

    return {
        "subject": tl.subject,
        "dataset": tl.dataset,
        "producer": cfg.name,
        "seed": seed,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "chunk_events": cfg.chunk_events,
        "batch_segments": cfg.batch_segments,
        "n_train_segments": len(train_ranges),
        "n_train_events": int(train_mask.sum()),
        "n_train_anchors": int(train_targets.eligible[:, 0].sum()),
        "selected_epoch": best["epoch"],
        "n_epochs_run": len(history),
        "stop_reason": stop_reason,
        "train_seconds": round(time.time() - started, 1),
        "future_loss_weights": weights,
        "param_update_magnitude": updates,
        "state_extraction": state_info,
        "history": history,
        "tau_fast_seconds": [float(v) for v in torch.stack(
            [model.state.taus()[0].min(), model.state.taus()[0].median(),
             model.state.taus()[0].max()]).tolist()],
        "tau_slow_seconds": [float(v) for v in torch.stack(
            [model.state.taus()[1].min(), model.state.taus()[1].median(),
             model.state.taus()[1].max()]).tolist()],
    }
