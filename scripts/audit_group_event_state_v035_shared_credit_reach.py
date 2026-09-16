#!/usr/bin/env python3
"""How far back does a gradient from the loss actually reach?

Truncating back-propagation at 30 minutes is one limit, but the architecture
itself has a second, softer limit: the state decays as exp(-dt/tau) and every
event moves it by at most ``update_fraction``.  Past some lag the gradient that
reaches an event is numerically negligible, and lengthening the window buys
nothing there.

This probe measures the real reach instead of arguing it.  It runs the shared
recurrence over one long stretch WITHOUT any detach, takes the loss at the end
of the stretch, and reads d(loss)/d(embedding of event t-k) for every k.  The
answer is reported against both event lag and wall-clock lag, so the useful
truncation window can be read off directly.

Reads CALIBRATION/FIT only; writes nothing into the producer tree.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.topic5_group_event_state.v034_spatial_state.we_decoder import (  # noqa: E402
    decoder_tensors, load_frozen_decoder,
)
from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    DECODER_ROOT, V035_DECODER_FITS, atomic_json,
)
from src.topic5_group_event_state.v035.full_mark_state import (  # noqa: E402
    FullMarkStateModel, FullMarkTrainConfig, _to_device, fit_physical_q_baseline,
    load_full_mark_data,
)
from src.topic5_group_event_state.v035.shared_state import (  # noqa: E402
    SharedGrammarHead, SharedProducerConfig, _grid_loss, build_shared_grammar_data,
    fit_grammar_q_baseline,
)

ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", choices=tuple(V035_DECODER_FITS), default="epilepsiae_253")
    ap.add_argument("--family", choices=("S_N", "S_G"), default="S_G")
    ap.add_argument("--decoder-seed", type=int, default=0)
    ap.add_argument("--state-seed", type=int, default=20260903)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config-json", type=Path,
                    default=ROOT / "config/group_event_state_v035_shared_producer_full.json")
    ap.add_argument("--rate-root", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_5_shared/shared_rate"))
    ap.add_argument("--adapter-root", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_5_shared/shared_stepwise_adapter"))
    ap.add_argument("--window-events", type=int, default=4096,
                    help="length of the undetached stretch; must exceed the truncation under test")
    ap.add_argument("--update-rule", choices=("gated", "residual"), default="gated",
                    help="gated = shipped rule state+f*(tanh(c)-state), whose Jacobian carries a "
                         "(1-f) contraction per event; residual = state+f*tanh(c), a leaky "
                         "integrator whose only decay is exp(-dt/tau)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_5_shared/audit/credit_reach"))
    args = ap.parse_args()

    started = time.time()
    raw = json.loads(args.config_json.read_text(encoding="utf-8"))
    overrides = dict(raw.get("base", raw))
    shared_overrides = dict(raw.get("shared", {}))
    overrides.update(objective_family="sn" if args.family == "S_N" else "sg",
                     event_innovation_mode="marked_event_only", seed=args.state_seed,
                     report_selection=False)
    for key in ("state_taus_seconds", "offset_weights", "event_offsets"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    base = FullMarkTrainConfig(**overrides)
    cfg = SharedProducerConfig(family=args.family, base=base, **shared_overrides)

    device = torch.device(args.device)
    fit_name = V035_DECODER_FITS[args.subject]
    bundle = load_frozen_decoder(
        DECODER_ROOT / "formal_units" / fit_name / ARM / f"seed{args.decoder_seed}",
        DECODER_ROOT / "cache" / fit_name, device=device,
    )
    rate = args.rate_root / args.subject / f"seed{args.state_seed}" / "trajectory_and_scores.npz"
    adapter = (args.adapter_root / args.subject
               / f"decoder_seed{args.decoder_seed}_state_seed{args.state_seed}" / "adapter.pt")
    data = load_full_mark_data(args.subject, bundle, rate, event_offsets=base.event_offsets)
    data = replace(data, future_valid=data.future_valid & (data.future_seizure_count == 0))

    torch.manual_seed(args.state_seed); np.random.seed(args.state_seed)
    model = FullMarkStateModel(data, bundle, adapter, base, device).to(device)
    for p in model.expected_from_q.parameters():
        p.requires_grad_(False)
    fit_physical_q_baseline(model, data, base, device)
    grammar = build_shared_grammar_data(data, cfg) if args.family == "S_G" else None
    grammar_head = None
    if grammar is not None:
        grammar_head = SharedGrammarHead(
            data.q_context.shape[1], model.state.cfg.state_dim,
            len(data.physical_horizons_seconds), grammar.dictionary.n_communities,
            grammar.dictionary.n_repertoires,
            grammar.dictionary.event_repertoire_embedding.shape[1],
        ).to(device)
        fit_grammar_q_baseline(grammar_head, data, grammar, base, device)
    decoder_tensors(bundle, device)

    if args.update_rule == "residual":
        # Temporal residual/skip path: drop the multiplicative (1-f) contraction so
        # the carried state is only attenuated by the physical decay exp(-dt/tau).
        import types
        def _residual_update(self, state_pre, event_embedding):
            raw = self.update_net(torch.cat([state_pre, self.event_norm(event_embedding)], dim=-1))
            gate, candidate = raw.chunk(2, dim=-1)
            fraction = torch.sigmoid(gate) * self.update_fraction
            return state_pre + fraction * torch.tanh(candidate)
        model.state.update = types.MethodType(_residual_update, model.state)

    # The state readouts ship zero-initialised, and d(loss)/d(state) = W_state^T . delta,
    # so with W_state == 0 the recurrence receives EXACTLY zero gradient and a reach
    # measurement would read zero everywhere for a trivial reason.  Give them a small
    # random value so what is measured is the credit decay of the recurrence itself.
    # (That the shipped initialisation gates the whole upstream path is itself a
    # finding; it is reported separately, not hidden here.)
    perturbed = []
    with torch.no_grad():
        heads = list(model.physical_head.state_parameters())
        if grammar_head is not None:
            heads += list(grammar_head.state_parameters())
        for p in heads:
            if float(p.abs().sum()) == 0.0:
                p.normal_(0.0, 0.02)
                perturbed.append(int(p.numel()))
    print(f"state-readout parameters given a non-zero probe init: {sum(perturbed):,}")

    # One long uninterrupted stretch inside a single coverage segment.
    fit_rows = np.flatnonzero(np.isin(data.phase, ("CALIBRATION", "FIT")))
    seg = data.event_segment[fit_rows]
    best = max(np.unique(seg), key=lambda s: int((seg == s).sum()))
    rows = fit_rows[seg == best][: args.window_events]
    if rows.size < 64:
        raise ValueError("stretch too short for a reach measurement")

    raw_batch = _to_device(data.seq.gather_positions(data.input_source_position[rows]), device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=base.amp and device.type == "cuda"):
        embedding = model.encode_events(raw_batch)
    embedding = embedding.float().detach().requires_grad_(True)
    q_all = torch.as_tensor(data.q_context[rows], dtype=torch.float32, device=device)

    # Roll the recurrence with NO detach anywhere.
    state = model.state.initial(1, device)
    previous = float(data.event_time[rows[0]])
    post = []
    for local, row in enumerate(rows):
        dt = torch.tensor([max(0.0, float(data.event_time[row]) - previous)], device=device)
        pre = model.state.evolve(state, dt)
        state = model.state.update(pre, model.innovation(embedding[local:local + 1],
                                                         q_all[local:local + 1]))
        post.append(state)
        previous = float(data.event_time[row])

    # Loss built only from the LAST few anchors, so every gradient that reaches
    # an earlier event had to travel through the recurrence.
    grid_rows = np.asarray([], dtype=np.int64)
    tail_len = max(8, rows.size // 100)
    while tail_len <= rows.size:
        tail = rows[-tail_len:]
        grid_rows = np.flatnonzero(np.isin(data.grid_source_event, tail) & (data.grid_source_event >= 0))
        # Keep the scored anchors inside the last tenth of the stretch so that a
        # gradient reaching an early event really had to cross the recurrence.
        if grid_rows.size and tail_len <= max(8, rows.size // 10):
            break
        if grid_rows.size:
            break
        tail_len *= 2
    if grid_rows.size == 0:
        raise ValueError("no fixed-grid anchor attaches to the tail of the stretch")
    print(f"tail={tail_len} events, scored grid anchors={grid_rows.size}")
    source_local = np.searchsorted(rows, data.grid_source_event[grid_rows])
    source_state = torch.cat(post, 0)[torch.as_tensor(source_local, dtype=torch.long, device=device)]
    dt_grid = torch.as_tensor(data.grid_source_dt[grid_rows], dtype=torch.float32, device=device)
    grid_state = model.state.evolve(source_state, dt_grid)
    loss, pieces = _grid_loss(cfg.family, model, grammar_head, grammar, data, grid_rows,
                              grid_state, device, block_grammar_weight=cfg.block_grammar_weight)
    if loss is None:
        raise ValueError("tail anchors produced no loss")
    loss.backward()

    g = embedding.grad.detach()
    per_event = g.norm(dim=1).cpu().numpy().astype(float)
    times = data.event_time[rows].astype(float)
    lag_seconds = float(times[-1]) - times
    total = float(per_event.sum())
    order = np.argsort(lag_seconds)
    lag_sorted = lag_seconds[order]; val_sorted = per_event[order]
    cum = np.cumsum(val_sorted) / max(total, 1e-30)

    def lag_for(frac: float) -> float:
        idx = int(np.searchsorted(cum, frac))
        return float(lag_sorted[min(idx, lag_sorted.size - 1)])

    bands = []
    edges = [0, 300, 900, 1800, 3600, 7200, 14400, 28800, 57600, 1e18]
    labels = ["<5min", "5-15min", "15-30min", "0.5-1h", "1-2h", "2-4h", "4-8h", "8-16h", ">16h"]
    for lo, hi, name in zip(edges[:-1], edges[1:], labels):
        m = (lag_sorted >= lo) & (lag_sorted < hi)
        bands.append({"band": name, "n_events": int(m.sum()),
                      "gradient_share": float(val_sorted[m].sum() / max(total, 1e-30))})

    payload = {
        "format": "group_event_state_v0_3_5_credit_reach_v1",
        "subject": args.subject, "family": args.family,
        "update_rule": args.update_rule,
        "final_state_norm": float(torch.cat(post, 0)[-1].norm()),
        "max_state_norm": float(torch.cat(post, 0).norm(dim=1).max()),
        "window_events": int(rows.size),
        "window_hours": float(lag_seconds.max() / 3600.0),
        "state_taus_hours": [round(float(t) / 3600.0, 3) for t in base.state_taus_seconds],
        "formal_truncation_seconds": float(base.chunk_seconds),
        "loss_components": pieces,
        "probe_init_note": (
            "state readouts are zero-initialised in the shipped model, which makes "
            "d(loss)/d(state) exactly zero at initialisation; this probe gives them a "
            "N(0, 0.02) init so the measurement reflects the recurrence, not that gate"
        ),
        "state_readout_params_perturbed": int(sum(perturbed)),
        "gradient_share_by_lag": bands,
        "lag_hours_capturing": {f"{int(100*f)}%": round(lag_for(f) / 3600.0, 3)
                                for f in (0.5, 0.8, 0.9, 0.95, 0.99)},
        "share_within_formal_truncation": float(
            val_sorted[lag_sorted < base.chunk_seconds].sum() / max(total, 1e-30)),
        "reads_only": "CALIBRATION/FIT of the registered safe prefix",
        "development_targets_read": False, "sealed_partition_opened": False,
        "elapsed_seconds": time.time() - started,
    }
    out_path = args.out / args.subject / args.family / f"credit_reach{args.tag}.json"
    atomic_json(out_path, payload)
    print(json.dumps({k: payload[k] for k in
                      ("subject", "family", "update_rule", "window_hours",
                       "share_within_formal_truncation", "lag_hours_capturing",
                       "max_state_norm")}, indent=2, ensure_ascii=False))
    for b in bands:
        print(f"  {b['band']:>9s}  事件 {b['n_events']:6d}  梯度占比 {100*b['gradient_share']:6.2f}%")
    print("written:", out_path)


if __name__ == "__main__":
    main()
