"""C1: functional innovation on the observed trajectory.

For every observed event, the difference between what the model expects of the
next block *after* reading that event and what it would have expected had it only
propagated time and background:

    dS_func(e) = S_func(z_e^+) - S_func(z_tilde_e^-)

``S_func`` is the frozen future-block readout -- expected count, expected
conditional mark -- never a raw latent distance.  Two states with the same
predictions are the same state for this purpose, however far apart their
coordinates happen to be, and a latent-space distance would report the
reparameterisation instead of the physiology.

This layer is descriptive by design.  It locates which events move the forecast
and by how much; it does not establish feedback, because an observer updating its
belief about a shared slow process produces innovation too.  That separation is
what the M0/M1/M2 comparison is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .features import build_block_targets
from .models import H3Model
from .train import SubjectTensors, _segment_inputs, TrainConfig, resolve_window


@dataclass
class InnovationTrace:
    event_rows: np.ndarray
    event_times: np.ndarray
    count_fraction: dict[int, np.ndarray]     # signed change in expected block count
    mark_shift: dict[int, np.ndarray]         # (n, n_groups) signed
    future_count_change: dict[int, np.ndarray]
    producer_source: str

    def as_summary(self, group_names: Sequence[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"producer_source": self.producer_source, "horizons": {}}
        for horizon, values in self.count_fraction.items():
            finite = np.isfinite(values)
            out["horizons"][str(horizon)] = {
                "n_events": int(finite.sum()),
                "median_innovation_count_fraction": float(np.median(values[finite])),
                "fraction_positive": float(np.mean(values[finite] > 0)),
                "median_mark_shift": {
                    name: float(np.median(self.mark_shift[horizon][:, i]))
                    for i, name in enumerate(group_names)
                },
            }
        return out


def collect_states_at(
    model: H3Model,
    data: SubjectTensors,
    cfg: TrainConfig,
    device: torch.device,
    seg: int,
    want_steps: np.ndarray,
    *,
    chunk: int = 256,
) -> Tensor:
    """The causal state arriving at each requested step of one segment.

    ``chunk`` is small on purpose: this pass asks for every event's state, and the
    rollout's weight tensor is (requested rows x chunk x state), so a 1024-step
    chunk with 1024 requested rows would allocate 200 MB per window for no gain.
    """

    cfg = resolve_window(data, cfg)
    tl = data.timelines[seg]
    carry = None
    pieces: list[Tensor] = []
    want_steps = np.asarray(want_steps, dtype=np.int64)
    with torch.no_grad():
        for lo in range(0, tl.n_steps, cfg.window_steps):
            hi = min(lo + cfg.window_steps, tl.n_steps)
            local = want_steps[(want_steps >= lo) & (want_steps < hi)] - lo
            want = torch.from_numpy(local.astype(np.int64)).to(device)
            dt, drive, impulse = _segment_inputs(model, data, seg, lo, hi)
            states, carry = model.rollout(
                dt, drive, impulse, want, state_init=carry, chunk=chunk
            )
            if local.size:
                pieces.append(states)
    return (
        torch.cat(pieces, dim=0)
        if pieces
        else torch.zeros(0, model.cfg.d_state, device=device)
    )


def functional_innovation(
    model: H3Model,
    data: SubjectTensors,
    cfg: TrainConfig,
    device: torch.device,
    horizons: Sequence[int],
    stream_t_abs: np.ndarray,
    features,
    *,
    producer_source: str,
    max_events_per_segment: int = 20000,
    seed: int = 0,
) -> InnovationTrace:
    """Per-event functional innovation, plus the future change it is compared to.

    ``future_count_change`` is what actually happened: the block after the event
    minus the block before it, in events.  Correlating the model's innovation with
    that is the honest version of "does the innovation track anything real"; the
    model's own prediction is not evidence that it did.
    """

    rng = np.random.default_rng(seed)
    rows_all, times_all = [], []
    count_frac: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    mark_shift: dict[int, list[np.ndarray]] = {h: [] for h in horizons}

    model.eval()
    for seg, tl in enumerate(data.timelines):
        event_steps = np.flatnonzero(tl.event_row >= 0)
        if event_steps.size == 0:
            continue
        if event_steps.size > max_events_per_segment:
            event_steps = np.sort(
                rng.choice(event_steps, size=max_events_per_segment, replace=False)
            )
        pre = collect_states_at(model, data, cfg, device, seg, event_steps)
        rows = tl.event_row[event_steps]
        rows_t = torch.from_numpy(rows.astype(np.int64)).to(device)
        with torch.no_grad():
            kick = model.event_impulse(
                data.count_features[rows_t], data.mark_features[rows_t]
            )
            post = pre + kick
            for horizon in horizons:
                before = model.decoder(pre, horizon)
                after = model.decoder(post, horizon)
                count_frac[horizon].append(
                    torch.expm1(after["count_log_mu"] - before["count_log_mu"])
                    .float().cpu().numpy()
                )
                delta = (after["mark_mu"] - before["mark_mu"]).float().cpu().numpy()
                mark_shift[horizon].append(
                    np.stack([delta[:, a:b].mean(1) for _n, (a, b) in data.mark_groups], axis=1)
                )
        rows_all.append(rows)
        times_all.append(tl.step_time[event_steps])

    if not rows_all:
        raise ValueError("no events available for the innovation trace")
    rows = np.concatenate(rows_all)
    times = np.concatenate(times_all)

    future_change: dict[int, np.ndarray] = {}
    for horizon in horizons:
        span = float(horizon) * 60.0
        after = build_block_targets(features, times, span)
        before = build_block_targets(features, times - span, span)
        future_change[horizon] = (after.count - before.count).astype(np.float64)

    return InnovationTrace(
        event_rows=rows,
        event_times=times,
        count_fraction={h: np.concatenate(v) for h, v in count_frac.items()},
        mark_shift={h: np.concatenate(v, axis=0) for h, v in mark_shift.items()},
        future_count_change=future_change,
        producer_source=producer_source,
    )
