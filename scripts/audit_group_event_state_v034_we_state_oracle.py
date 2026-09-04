#!/usr/bin/env python3
"""Positive control for the S_P state path: can the same protocol exploit a state that is known to help?

The learned state stage selects its own initialisation on every subject.  That
is either a real negative (the cross-event state carries nothing the frozen
tissue decoder can use) or an instrument failure (the ``h0`` path cannot be
optimised under this protocol).  This audit separates the two by replacing the
encoder-produced state with an *oracle* state that leaks the answer:

    oracle(anchor) = mean participation vector over that anchor's own future events

standardised on the fit anchors.  Everything else -- the frozen decoder, the
``h0 = bias + low-rank(state)`` path, the learning rates, the rolling
inner-validation selection and the patience -- is exactly the protocol used by
``we_state.fit_stage``.  If the oracle arm cannot beat the state-free adapter
under this protocol, the protocol is the limiting factor and no human negative
may be read from it.

DIAGNOSTIC only: the oracle deliberately leaks future information and must
never be reported as a state result.  TRAIN + STATE_SELECTION only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.contracts import TrainConfig, seed_before_model_construction  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.data import load_human_spatial_data, sample_equal_anchor_pairs  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import WEStateScorer, load_frozen_decoder  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_state import WEStateConfig, pair_scores, prepare, weighted  # noqa: E402

DECODER_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/we_decoder")
FITS = {
    "epilepsiae_253": "epilepsiae_253__own_a", "epilepsiae_1146": "epilepsiae_1146__shared",
    "epilepsiae_548": "epilepsiae_548__shared", "epilepsiae_583": "epilepsiae_583__shared",
    "epilepsiae_922": "epilepsiae_922__own_a",
}
ARM = "L3_LOCAL_PLUS_LEARNED_LR"


class OracleScorer(torch.nn.Module):
    """``WEStateScorer`` driven by a fixed per-anchor state table instead of an encoder."""

    def __init__(self, scorer: WEStateScorer, states: torch.Tensor) -> None:
        super().__init__()
        self.scorer = scorer
        self.register_buffer("states", states)

    # the interfaces ``pair_scores`` needs
    @property
    def encoder(self):
        raise AttributeError("oracle arm has no encoder")

    def trajectory(self, *_args, **_kwargs) -> torch.Tensor:
        return self.states


def oracle_states(data, prep, device: torch.device) -> torch.Tensor:
    """Mean participation over each anchor's own future events (leaks the target on purpose)."""

    n_anchor = int(data.anchor_time.size)
    n_contact = int(data.participation.shape[1])
    table = np.zeros((n_anchor, n_contact), dtype=np.float64)
    counts = np.zeros(n_anchor, dtype=np.float64)
    for pairs in (prep.fit_pairs, prep.inner_pairs, prep.selection_pairs):
        rows = pairs.anchor_rows[pairs.pair_anchor]
        np.add.at(table, rows, data.participation[pairs.pair_event].astype(np.float64))
        np.add.at(counts, rows, 1.0)
    table /= np.maximum(counts, 1.0)[:, None]
    fit_rows = prep.fit_pairs.anchor_rows
    centre = table[fit_rows].mean(0)
    scale = table[fit_rows].std(0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return torch.as_tensor((table - centre) / scale, dtype=torch.float32, device=device)


def fit_oracle(model: OracleScorer, prep, config: WEStateConfig, seed: int) -> dict:
    params = list(model.scorer.to_h0.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params": params, "lr": config.lr_h0}], weight_decay=config.weight_decay)
    rng = np.random.default_rng(seed)

    def evaluate(pairs, use_state: bool) -> float:
        model.eval()
        with torch.no_grad():
            sc = pair_scores(model, prep, pairs, model.states if use_state else None, use_bias=True, use_state=use_state)
        return float(weighted(sc["grammar"], pairs))

    adapter_only = evaluate(prep.inner_pairs, False)
    best = evaluate(prep.inner_pairs, True)
    best_step, stale, history = 0, 0, [{"step": 0, "inner_val": best}]
    for step in range(1, config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pairs = sample_equal_anchor_pairs(prep.fit_pairs, rng=rng, n_anchors=config.anchors_per_step,
                                          events_per_anchor=config.events_per_anchor)
        sc = pair_scores(model, prep, pairs, model.states, use_bias=True, use_state=True, grad=True)
        weighted(sc["grammar"], pairs).backward()
        torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
        optimizer.step()
        if step % config.validate_every == 0 or step == config.max_steps:
            value = evaluate(prep.inner_pairs, True)
            history.append({"step": step, "inner_val": value})
            if value < best - 1e-6:
                best, best_step, stale = value, step, 0
            else:
                stale += 1
            if stale >= config.patience_checks:
                break
    return {"adapter_only_inner_val": adapter_only, "best_oracle_inner_val": best, "selected_step": best_step,
            "steps_run": history[-1]["step"], "oracle_gain_over_adapter": adapter_only - best, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=["epilepsiae_548", "epilepsiae_583", "epilepsiae_253", "epilepsiae_1146"])
    ap.add_argument("--decoder-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patience-checks", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--out", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_4/we_state/oracle_control.json"))
    args = ap.parse_args()
    device = torch.device(args.device)
    rows = {}
    for subject in args.subjects:
        seed = 20260903 + args.decoder_seed
        seed_before_model_construction(seed)
        bundle = load_frozen_decoder(DECODER_ROOT / "formal_units" / FITS[subject] / ARM / f"seed{args.decoder_seed}",
                                     DECODER_ROOT / "cache" / FITS[subject], device=device)
        data = load_human_spatial_data(subject, train_config=TrainConfig(max_steps=900, seed=seed))
        config = WEStateConfig(seed=seed, max_steps=args.max_steps, patience_checks=args.patience_checks)
        prep = prepare(data, bundle, config, device)
        states = oracle_states(data, prep, device)
        scorer = WEStateScorer(bundle, state_dim=int(states.shape[1]), rank=min(4, int(states.shape[1])),
                               stop_weight=config.stop_weight).to(device)
        model = OracleScorer(scorer, states).to(device)
        result = fit_oracle(model, prep, config, seed + 77)
        rows[subject] = {**result, "n_fit_anchors": int(prep.fit_pairs.anchor_rows.size),
                         "n_inner_anchors": int(prep.inner_pairs.anchor_rows.size),
                         "state_dim": int(states.shape[1]), "coverage": prep.coverage["coverage_fraction"]}
        print(f"{subject}: adapter {result['adapter_only_inner_val']:.5f} -> oracle {result['best_oracle_inner_val']:.5f} "
              f"(gain {result['oracle_gain_over_adapter']:+.5f}, step {result['selected_step']}/{result['steps_run']})", flush=True)
    atomic_write_json(args.out, {"format": "group_event_state_v0_3_4_we_state_oracle_control_v1",
                                 "interpretation": "DIAGNOSTIC positive control; the oracle state leaks future participation and is never a science arm",
                                 "decoder_seed": args.decoder_seed, "subjects": rows,
                                 "development_targets_read": False, "sealed_partition_opened": False})


if __name__ == "__main__":
    main()
