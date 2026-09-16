#!/usr/bin/env python3
"""Is the shared S_N/S_G producer capacity-limited, or is it not learning at all?

The formal runs show a nearly flat training curve: on E253 the S_G components
move 0.05-0.9 % over 20 epochs (about 4200 optimiser steps).  That is
compatible with two very different situations:

  A. the state genuinely adds nothing on top of the frozen q baseline, or
  B. the state pathway never receives usable gradient, so the run measures the
     instrument rather than the biology.

Only B is fixable by changing the model, and a bigger model is only indicated
if the architecture can actually drive its own training loss down.  This probe
separates the two WITHOUT touching the running queue or its outputs:

  1. per-parameter-group gradient norms on real FIT batches (is the state path
     starved?),
  2. a TRAIN-only tiny-slice overfit with regularisation off and a large
     learning rate (can this architecture fit anything at all?).

It reads CALIBRATION/FIT only, writes nothing into the producer tree, and never
touches INNER or SELECTION.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
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
    FullMarkStateModel, FullMarkTrainConfig, fit_physical_q_baseline, load_full_mark_data,
)
from src.topic5_group_event_state.v035.shared_state import (  # noqa: E402
    SharedGrammarHead, SharedProducerConfig, build_shared_grammar_data,
    fit_grammar_q_baseline, run_shared_phase,
)

ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def _trainable_groups(model, grammar_head, family: str) -> dict[str, list[torch.nn.Parameter]]:
    groups = {
        "event_encoder": [p for p in model.event_encoder.parameters() if p.requires_grad],
        "timing_token": [model.timing_token] if model.timing_token.requires_grad else [],
        "state": [p for p in model.state.parameters() if p.requires_grad],
        "physical_head.state": [p for p in model.physical_head.state_parameters() if p.requires_grad],
    }
    if family == "S_G":
        groups["m_adapter"] = [p for p in model.m_adapter.parameters() if p.requires_grad]
        if grammar_head is not None:
            groups["grammar_head.state"] = [p for p in grammar_head.state_parameters() if p.requires_grad]
    return {k: v for k, v in groups.items() if v}


def _grad_report(groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, params in groups.items():
        norms = [float(p.grad.detach().norm()) for p in params if p.grad is not None]
        weights = [float(p.detach().norm()) for p in params]
        total_g = float(np.sqrt(np.sum(np.square(norms)))) if norms else 0.0
        total_w = float(np.sqrt(np.sum(np.square(weights)))) if weights else 0.0
        out[name] = {
            "grad_norm": total_g,
            "weight_norm": total_w,
            # A step of size lr moves the weights by roughly lr * grad/|w|; this
            # ratio says whether the group can move at all within the budget.
            "relative_grad": total_g / max(total_w, 1e-12),
            "n_params": int(sum(p.numel() for p in params)),
            "n_with_grad": len(norms),
        }
    return out


def _restrict_to_slice(data, n_events: int):
    """Keep the first ``n_events`` CALIBRATION/FIT events as the only FIT phase.

    Everything outside the slice is marked OUTSIDE so ``run_shared_phase`` sees a
    single tiny training set.  No INNER/SELECTION row can enter.
    """
    phase = np.asarray(data.phase).copy()
    usable = np.flatnonzero(np.isin(phase, ("CALIBRATION", "FIT")))
    if usable.size == 0:
        raise ValueError("no CALIBRATION/FIT events for the tiny slice")
    keep = usable[:n_events]
    new_phase = np.full(phase.shape, "OUTSIDE", dtype=phase.dtype)
    new_phase[keep] = "FIT"
    grid_phase = np.asarray(data.grid_phase).copy()
    lo, hi = float(data.event_time[keep[0]]), float(data.event_time[keep[-1]])
    inside = (data.grid_time >= lo) & (data.grid_time <= hi)
    new_grid = np.full(grid_phase.shape, "OUTSIDE", dtype=grid_phase.dtype)
    new_grid[inside] = "FIT"
    return replace(data, phase=new_phase, grid_phase=new_grid), int(keep.size)


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
    ap.add_argument("--chunk-events", type=int, default=256)
    ap.add_argument("--chunk-seconds", type=float,
                    help="override the truncated-BPTT window; the formal runs use 1800 s "
                         "while the state bank reaches 8 h and the targets reach 8 h")
    ap.add_argument("--tag", default="", help="suffix so arms do not overwrite each other")
    ap.add_argument("--update-rule", choices=("gated", "residual"), default="gated",
                    help="gated = shipped state+f*(tanh(c)-state); residual = state+f*tanh(c), "
                         "a temporal skip path whose only decay is exp(-dt/tau)")
    ap.add_argument("--slice-events", type=int, default=512)
    ap.add_argument("--overfit-epochs", type=int, default=40)
    ap.add_argument("--overfit-lr-scale", type=float, default=30.0)
    ap.add_argument("--out", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_5_shared/audit/capacity_probe"))
    args = ap.parse_args()

    started = time.time()
    raw = json.loads(args.config_json.read_text(encoding="utf-8"))
    overrides = dict(raw.get("base", raw))
    shared_overrides = dict(raw.get("shared", {}))
    overrides.update(
        objective_family="sn" if args.family == "S_N" else "sg",
        event_innovation_mode="marked_event_only",
        seed=args.state_seed, chunk_events=args.chunk_events, report_selection=False,
    )
    if args.chunk_seconds is not None:
        overrides["chunk_seconds"] = float(args.chunk_seconds)
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
    # Same primary-layer restriction as the formal producer.
    data = replace(data, future_valid=data.future_valid & (data.future_seizure_count == 0))

    torch.manual_seed(args.state_seed); np.random.seed(args.state_seed)
    model = FullMarkStateModel(data, bundle, adapter, base, device).to(device)
    for parameter in model.expected_from_q.parameters():
        parameter.requires_grad_(False)
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
    if args.family == "S_N":
        for parameter in model.m_adapter.parameters():
            parameter.requires_grad_(False)

    if args.update_rule == "residual":
        import types
        def _residual_update(self, state_pre, event_embedding):
            raw = self.update_net(torch.cat([state_pre, self.event_norm(event_embedding)], dim=-1))
            gate, candidate = raw.chunk(2, dim=-1)
            fraction = torch.sigmoid(gate) * self.update_fraction
            return state_pre + fraction * torch.tanh(candidate)
        model.state.update = types.MethodType(_residual_update, model.state)

    groups = _trainable_groups(model, grammar_head, args.family)
    tensors = decoder_tensors(bundle, device)

    # ---- part 1: gradient health on the real FIT set (a few optimiser steps)
    probe_opt = torch.optim.AdamW(
        [{"params": p, "lr": 0.0} for p in groups.values()], weight_decay=0.0,
    )
    slice_data, n_slice = _restrict_to_slice(data, args.slice_events)
    grad_reports = []
    for _ in range(3):
        probe_opt.zero_grad(set_to_none=True)
        out = run_shared_phase(model, grammar_head, grammar, slice_data, tensors,
                               "FIT", cfg, device, probe_opt)
        grad_reports.append({"loss": out.get("mean_loss"),
                             "components": out.get("components"),
                             "groups": _grad_report(groups)})
    # NOTE: lr is 0 here on purpose, so the weights never move.  The state
    # readouts are zero-initialised, and d(loss)/d(state) = W_state^T . delta,
    # so with W_state pinned at zero the encoder and the recurrent state read
    # EXACTLY zero gradient by construction.  That is a property of this frozen
    # probe, NOT a defect of the training run: after the first real optimiser
    # step W_state is non-zero and the path opens.  The gradient health that
    # actually matters is therefore measured again after real training below.

    # ---- part 2: tiny-slice overfit with regularisation off and a big lr
    overfit_groups = [
        {"params": params, "lr": lr * args.overfit_lr_scale}
        for params, lr in (
            (groups.get("event_encoder", []), base.encoder_lr),
            (groups.get("timing_token", []), base.encoder_lr),
            (groups.get("state", []), base.state_lr),
            (groups.get("physical_head.state", []), base.adapter_lr),
            (groups.get("m_adapter", []), base.adapter_lr),
            (groups.get("grammar_head.state", []), base.adapter_lr),
        ) if params
    ]
    optimizer = torch.optim.AdamW(overfit_groups, weight_decay=0.0)
    trace = []
    for epoch in range(args.overfit_epochs):
        out = run_shared_phase(model, grammar_head, grammar, slice_data, tensors,
                               "FIT", cfg, device, optimizer)
        trace.append({"epoch": epoch, "mean_loss": out.get("mean_loss"),
                      "components": out.get("components")})

    # Gradient health once the zero-initialised readouts have actually moved.
    optimizer.zero_grad(set_to_none=True)
    run_shared_phase(model, grammar_head, grammar, slice_data, tensors,
                     "FIT", cfg, device,
                     torch.optim.AdamW([{"params": p, "lr": 0.0} for p in groups.values()],
                                       weight_decay=0.0))
    grad_after_training = _grad_report(groups)

    first = trace[0]["mean_loss"]
    best = min(t["mean_loss"] for t in trace if t["mean_loss"] is not None)
    drop = None if first in (None, 0) else (first - best) / abs(first)
    comp_first = trace[0].get("components") or {}
    comp_best = {}
    for key in comp_first:
        vals = [t["components"][key] for t in trace
                if t.get("components") and t["components"].get(key) is not None]
        comp_best[key] = {"first": comp_first[key], "best": min(vals) if vals else None,
                          "relative_drop": (None if not vals or comp_first[key] == 0
                                            else (comp_first[key] - min(vals)) / abs(comp_first[key]))}

    verdict = (
        "ARCHITECTURE_CAN_FIT" if drop is not None and drop >= 0.20 else
        "ARCHITECTURE_CANNOT_FIT_TINY_SLICE"
    )
    payload = {
        "format": "group_event_state_v0_3_5_shared_capacity_probe_v1",
        "subject": args.subject, "family": args.family,
        "decoder_seed": args.decoder_seed, "state_seed": args.state_seed,
        "n_slice_events": n_slice,
        "update_rule": args.update_rule,
        "chunk_seconds": float(base.chunk_seconds),
        "chunk_events": int(base.chunk_events),
        "trainable_parameters": {k: int(sum(p.numel() for p in v)) for k, v in groups.items()},
        "total_trainable": int(sum(sum(p.numel() for p in v) for v in groups.values())),
        "gradient_health_at_init_lr0": grad_reports,
        "gradient_health_after_training": grad_after_training,
        "gradient_note": (
            "at initialisation the state readouts are zero, so d(loss)/d(state) is exactly zero and "
            "the encoder/state groups read zero gradient; this is a property of the frozen probe. "
            "gradient_health_after_training is the measurement that speaks to the real run."
        ),
        "tiny_overfit": {
            "lr_scale": args.overfit_lr_scale, "weight_decay": 0.0,
            "epochs": args.overfit_epochs, "first_loss": first, "best_loss": best,
            "relative_drop": drop, "components": comp_best, "trace": trace,
        },
        "verdict": verdict,
        "reading": (
            "ARCHITECTURE_CAN_FIT means the flat formal curve is evidence about the data, "
            "not about capacity, so enlarging the model is not indicated by this probe. "
            "ARCHITECTURE_CANNOT_FIT_TINY_SLICE means the formal run measures the instrument "
            "and no capacity conclusion may be drawn from it."
        ),
        "reads_only": "CALIBRATION/FIT of the registered safe prefix; INNER and SELECTION untouched",
        "development_targets_read": False, "sealed_partition_opened": False,
        "elapsed_seconds": time.time() - started,
    }
    out_path = args.out / args.subject / args.family / f"probe{args.tag}.json"
    atomic_json(out_path, payload)
    print(json.dumps({k: payload[k] for k in
                      ("subject", "family", "n_slice_events", "total_trainable", "verdict")}, indent=2))
    print(json.dumps(payload["tiny_overfit"]["components"], indent=2))
    print("gradient at init (lr=0, zero-init readout -> upstream zero BY CONSTRUCTION):")
    print(json.dumps(grad_reports[-1]["groups"], indent=2))
    print("gradient AFTER training (this is the one that matters):")
    print(json.dumps(grad_after_training, indent=2))
    print("written:", out_path)


if __name__ == "__main__":
    main()
