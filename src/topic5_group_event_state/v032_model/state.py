"""Cross-event state backbones for v0.3.2.

``leaky_bank_trajectory`` is the primary object (design §3.1): every event
writes the same ``K``-vector ``u_e`` into ``T`` fixed-timescale blocks, and the
state between events only decays.  Because the recurrence is linear with no
state-to-state mixing, the whole segment trajectory has a closed form

    S_post[e] = sum_{j <= e, same segment} exp(-(t_e - t_j) / tau) u_j

which is evaluated exactly (float64, rescaled per chunk) and differentiated
without truncation.  The chunk edge is a numerical device; ``detach_chunks``
exists only so the TBPTT audit can show what truncation would have cost.

``RepairedRecurrentState`` is the triage control: the v0.3 gated update with
the same fixed decay but *state-dependent* writes (mixing), with every
LayerNorm removed.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

# A chunk never spans more than this many minimum time constants, so the
# rescaled cumulative weights stay far inside float64 range (exp(30) ~ 1e13).
MAX_CHUNK_IN_MIN_TAUS = 30.0


def _validate_stream(u: Tensor, times: Tensor, segment_ids: Tensor) -> None:
    if u.ndim != 2:
        raise ValueError("u must have shape (n_events, K)")
    n = u.shape[0]
    if times.shape != (n,) or segment_ids.shape != (n,):
        raise ValueError("times and segment_ids must be one-dimensional and match u")
    if n > 1:
        same = segment_ids[1:] == segment_ids[:-1]
        if bool((segment_ids[1:] < segment_ids[:-1]).any()):
            raise ValueError("segment_ids must be non-decreasing (events are time ordered)")
        if bool(((times[1:] - times[:-1]) < 0)[same].any()):
            raise ValueError("event times must be non-decreasing within a segment")


def _chunk_bounds(
    times: np.ndarray, segment_ids: np.ndarray, span_seconds: float
) -> list[tuple[int, int, bool]]:
    """Chronological chunks ``(start, stop, new_segment)`` bounded by real time."""

    out: list[tuple[int, int, bool]] = []
    n = times.size
    start = 0
    while start < n:
        seg = segment_ids[start]
        seg_stop = int(np.searchsorted(segment_ids, seg, side="right"))
        limit = times[start] + span_seconds
        stop = int(np.searchsorted(times[start:seg_stop], limit, side="right")) + start
        stop = max(stop, start + 1)
        new_segment = start == 0 or segment_ids[start - 1] != seg
        out.append((start, stop, bool(new_segment)))
        start = stop
    return out


def leaky_bank_trajectory(
    u: Tensor,
    times: Tensor,
    segment_ids: Tensor,
    taus: Tensor,
    *,
    chunk_seconds: float,
    detach_chunks: bool = False,
) -> tuple[Tensor, Tensor]:
    """Exact pre/post event states of the marked leaky bank.

    Returns ``(state_pre, state_post)`` of shape ``(n_events, T*K)`` in float32
    with tau-major layout ``[tau_0: k_0..k_{K-1}, tau_1: ..., ...]``.
    """

    _validate_stream(u, times, segment_ids)
    n, k = u.shape
    device = u.device
    taus64 = taus.to(device=device, dtype=torch.float64).reshape(-1)
    t_count = taus64.numel()
    u64 = u.to(torch.float64)
    times64 = times.to(device=device, dtype=torch.float64)
    span = float(min(float(chunk_seconds), MAX_CHUNK_IN_MIN_TAUS * float(taus64.min())))
    bounds = _chunk_bounds(
        times64.detach().cpu().numpy(), segment_ids.detach().cpu().numpy(), span
    )
    post_chunks: list[Tensor] = []
    carry = u64.new_zeros((t_count, k))
    carry_time = 0.0
    for start, stop, new_segment in bounds:
        t0 = times64[start]
        t_rel = times64[start:stop] - t0                                  # (n_c,)
        growth = torch.exp(t_rel[:, None] / taus64[None, :])              # (n_c, T)
        decay = torch.exp(-t_rel[:, None] / taus64[None, :])              # (n_c, T)
        weighted = growth[:, :, None] * u64[start:stop, None, :]          # (n_c, T, K)
        running = torch.cumsum(weighted, dim=0)
        post_c = decay[:, :, None] * running
        if not new_segment:
            gap = (t_rel + (t0 - carry_time))[:, None]                    # seconds since carry
            post_c = post_c + torch.exp(-gap / taus64[None, :])[:, :, None] * carry[None]
        post_chunks.append(post_c)
        carry = post_c[-1]
        carry_time = float(times64[stop - 1])
        if detach_chunks:
            carry = carry.detach()
    post64 = torch.cat(post_chunks, dim=0)                                 # (n, T, K)
    pre64 = post64 - u64[:, None, :]
    return (
        pre64.reshape(n, t_count * k).to(torch.float32),
        post64.reshape(n, t_count * k).to(torch.float32),
    )


def anchor_states(
    state_post: Tensor,
    event_times: Tensor,
    t_anchor: Tensor,
    last_event_pos: Tensor,
    taus_full: Tensor,
) -> Tensor:
    """Autonomous decay of the last in-segment post-event state to each anchor."""

    a = int(t_anchor.numel())
    d = int(state_post.shape[1])
    out = state_post.new_zeros((a, d))
    has = last_event_pos >= 0
    if not bool(has.any()):
        return out
    idx = last_event_pos[has].long()
    dt = t_anchor.to(torch.float64)[has] - event_times.to(torch.float64)[idx]
    if bool((dt < -1e-6).any()):
        raise ValueError("anchor precedes its own last event")
    decay = torch.exp(-dt.clamp_min(0.0)[:, None] / taus_full.to(torch.float64)[None, :])
    out = out.clone()
    out[has] = (state_post[idx].to(torch.float64) * decay).to(state_post.dtype)
    return out


def _taus_full(taus_seconds: Sequence[float], channels_per_tau: int) -> Tensor:
    return torch.tensor(list(taus_seconds), dtype=torch.float32).repeat_interleave(
        int(channels_per_tau)
    )


class MarkedLeakyBank(nn.Module):
    """12-dimensional constrained bank: no learnable parameter at all."""

    def __init__(
        self,
        taus_seconds: Sequence[float],
        channels_per_tau: int,
        *,
        chunk_seconds: float,
        detach_chunks: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("taus", torch.tensor(list(taus_seconds), dtype=torch.float32))
        self.register_buffer("taus_full", _taus_full(taus_seconds, channels_per_tau))
        self.channels_per_tau = int(channels_per_tau)
        self.chunk_seconds = float(chunk_seconds)
        self.detach_chunks = bool(detach_chunks)

    @property
    def state_dim(self) -> int:
        return int(self.taus_full.numel())

    def forward(self, u: Tensor, times: Tensor, segment_ids: Tensor) -> tuple[Tensor, Tensor]:
        if u.shape[1] != self.channels_per_tau:
            raise ValueError("write vector width must equal channels_per_tau")
        return leaky_bank_trajectory(
            u, times, segment_ids, self.taus,
            chunk_seconds=self.chunk_seconds, detach_chunks=self.detach_chunks,
        )

    def anchor(
        self, state_post: Tensor, event_times: Tensor, t_anchor: Tensor, last_event_pos: Tensor
    ) -> Tensor:
        return anchor_states(state_post, event_times, t_anchor, last_event_pos, self.taus_full)


class RepairedRecurrentState(nn.Module):
    """v0.3 gated update with fixed decay, no LayerNorm (triage control only)."""

    def __init__(
        self,
        taus_seconds: Sequence[float],
        channels_per_tau: int,
        *,
        event_dim: int,
        hidden: int = 32,
        update_fraction_numerator: float = 2.0,
        update_fraction_cap: float = 0.2,
    ) -> None:
        super().__init__()
        taus_full = _taus_full(taus_seconds, channels_per_tau)
        self.register_buffer("taus_full", taus_full)
        self.register_buffer(
            "update_fraction",
            (float(update_fraction_numerator) / torch.sqrt(taus_full)).clamp(
                max=float(update_fraction_cap)
            ),
        )
        d = int(taus_full.numel())
        self.update_net = nn.Sequential(
            nn.Linear(int(event_dim) + d, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 2 * d),
        )

    @property
    def state_dim(self) -> int:
        return int(self.taus_full.numel())

    def _update(self, state_pre: Tensor, event: Tensor) -> Tensor:
        gate, candidate = self.update_net(torch.cat([state_pre, event], dim=-1)).chunk(2, dim=-1)
        fraction = torch.sigmoid(gate) * self.update_fraction
        return state_pre + fraction * (torch.tanh(candidate) - state_pre)

    def forward(self, e: Tensor, times: Tensor, segment_ids: Tensor) -> tuple[Tensor, Tensor]:
        _validate_stream(e, times, segment_ids)
        n = e.shape[0]
        d = self.state_dim
        device = e.device
        seg_np = segment_ids.detach().cpu().numpy()
        starts = np.flatnonzero(np.r_[True, seg_np[1:] != seg_np[:-1]])
        stops = np.r_[starts[1:], n]
        n_seg = starts.size
        max_len = int((stops - starts).max()) if n_seg else 0
        padded = np.full((n_seg, max_len), -1, dtype=np.int64)
        for s, (a, b) in enumerate(zip(starts, stops)):
            padded[s, : b - a] = np.arange(a, b)
        padded_t = torch.from_numpy(padded).to(device)
        times64 = times.to(device=device, dtype=torch.float64)
        taus64 = self.taus_full.to(torch.float64)
        state = e.new_zeros((n_seg, d))
        prev_time = torch.zeros(n_seg, dtype=torch.float64, device=device)
        rows_all: list[Tensor] = []
        pre_all: list[Tensor] = []
        post_all: list[Tensor] = []
        for step in range(max_len):
            rows = padded_t[:, step]
            active = rows >= 0
            if not bool(active.any()):
                break
            rows = rows[active]
            t_now = times64[rows]
            dt = torch.where(
                torch.tensor(step > 0, device=device), t_now - prev_time[active], torch.zeros_like(t_now)
            )
            decay = torch.exp(-dt[:, None] / taus64[None, :]).to(state.dtype)
            state_pre = state[active] * decay
            state_post = self._update(state_pre, e[rows])
            new_state = state.clone()
            new_state[active] = state_post
            state = new_state
            prev_time = prev_time.clone()
            prev_time[active] = t_now
            rows_all.append(rows)
            pre_all.append(state_pre)
            post_all.append(state_post)
        order = torch.cat(rows_all)
        pre = e.new_zeros((n, d))
        post = e.new_zeros((n, d))
        pre = pre.index_put((order,), torch.cat(pre_all))
        post = post.index_put((order,), torch.cat(post_all))
        return pre, post

    def anchor(
        self, state_post: Tensor, event_times: Tensor, t_anchor: Tensor, last_event_pos: Tensor
    ) -> Tensor:
        return anchor_states(state_post, event_times, t_anchor, last_event_pos, self.taus_full)
