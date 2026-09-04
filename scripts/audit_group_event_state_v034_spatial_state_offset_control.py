#!/usr/bin/env python3
"""Period-offset / zero-state / block-shift controls for frozen v0.3.4 S_P checkpoints.

For every locked-evaluation (and E253 locked-recipe) seed checkpoint, the
STATE_SELECTION grammar NLL is recomputed on the *full* selection pairs with:

  zero          state = 0 (prefix-only frozen grammar; TRAIN-mean state)
  learned       checkpoint state at the correct time
  period_mean   state replaced by its mean over all STATE_SELECTION anchors
  shifted       same-carry-segment block-circular wrong-time state (v0.3.2 rule)

and, for context, the same zero/learned contrast on the TRAIN pairs.  Seeds
are merged per anchor (median) before a within-segment moving-block bootstrap.
TRAIN + STATE_SELECTION only; nothing here reads development targets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.topic5_group_event_state.v032_model.evaluate import block_bootstrap_mean_ci
from src.topic5_group_event_state.v032_model.shift import block_circular_donor
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json
from src.topic5_group_event_state.v034_spatial_state.contracts import ArchConfig, TrainConfig
from src.topic5_group_event_state.v034_spatial_state.data import load_human_spatial_data
from src.topic5_group_event_state.v034_spatial_state.model import SpatialStateModel
from src.topic5_group_event_state.v034_spatial_state.trainer import _load_trainable_state, _states, _to_device

RUNNER = Path(__file__).resolve().with_name("run_group_event_state_v034_spatial_state.py")
_spec = importlib.util.spec_from_file_location("v034_runner", RUNNER)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
OUTPUT_ROOT = _runner.OUTPUT_ROOT
BOOT = dict(block_len=6, n_boot=1000, seed=0)


def ci(values, segments):
    out = block_bootstrap_mean_ci(np.asarray(values, dtype=np.float64), np.asarray(segments), **BOOT)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


@torch.no_grad()
def grammar_per_anchor(model, tensors, pairs, state_for_anchor, device, batch=4096):
    """Mean grammar NLL per anchor (equal event weight within anchor)."""

    event = torch.as_tensor(pairs.pair_event, dtype=torch.long, device=device)
    owner = torch.as_tensor(pairs.pair_anchor, dtype=torch.long, device=device)
    out = torch.zeros(event.numel(), dtype=torch.float64, device=device)
    for lo in range(0, event.numel(), batch):
        e = event[lo:lo + batch]
        st = state_for_anchor[owner[lo:lo + batch]]
        out[lo:lo + batch] = model.legacy_event_nll(
            tensors["group_ids"][e], tensors["group_count"][e], st
        ).to(torch.float64)
    n = int(pairs.anchor_rows.size)
    sums = torch.zeros(n, dtype=torch.float64, device=device).index_add_(0, owner, out)
    counts = torch.zeros(n, dtype=torch.float64, device=device).index_add_(0, owner, torch.ones_like(out))
    return (sums / counts.clamp_min(1)).cpu().numpy()


def audit_subject(subject, cards, device, out_dir):
    rows = {"zero": [], "learned": [], "period_mean": [], "shifted": [], "train_zero": [], "train_learned": []}
    per_seed = []
    data = None
    for card_path in cards:
        card = json.loads(Path(card_path).read_text())
        seed = int(card["contract"]["train"]["seed"])
        arch_d = card["contract"]["arch"]
        train_cfg = TrainConfig(max_steps=int(card["contract"]["train"]["max_steps"]), seed=seed,
                                burn_in_seconds=float(card["contract"]["train"]["burn_in_seconds"]))
        if data is None:
            data = load_human_spatial_data(subject, train_config=train_cfg)
            tensors = _to_device(data, device)
            decoder, _artifact = _runner._legacy_decoder(subject, device)
            sel = data.selection_pairs
            sel_rows = sel.anchor_rows
            seg = data.event_segment  # carry ids per event
            anchor_seg = np.asarray([int(data.phase.shape[0] and 0)] * 0)  # placeholder
            # carry segment per anchor: use the carry of the last event before the anchor
            last = data.last_event_pos[sel_rows]
            anchor_carry = np.where(last >= 0, seg[np.maximum(last, 0)], -1)
            donor = block_circular_donor(data.anchor_time, np.r_[anchor_carry] if False else _anchor_carry_full(data), sel_rows, horizon=1800.0, fraction=0.5)
            ok = donor >= 0
            boot_seg = _anchor_carry_full(data)[sel_rows]
        arch = ArchConfig(width=int(arch_d["width"]), depth=int(arch_d["depth"]),
                          write_width=int(arch_d["write_width"]), adapter_rank=int(arch_d["adapter_rank"]),
                          residual=bool(arch_d["residual"]), taus_seconds=tuple(float(t) for t in arch_d["taus_seconds"]))
        model = SpatialStateModel(input_dim=data.event_token.shape[1], n_contacts=data.n_contacts,
                                  config=arch, legacy_decoder=decoder).to(device)
        ckpt = torch.load(Path(card_path).with_name("selected_checkpoint.pt"), map_location=device, weights_only=False)
        _load_trainable_state(model, ckpt["state_dict"])
        model.eval()
        states = _states(model, tensors, data.train_pairs)           # standardised by TRAIN anchors
        s_sel = states[torch.as_tensor(sel_rows, dtype=torch.long, device=device)]
        zero = torch.zeros_like(s_sel)
        period = s_sel.mean(0, keepdim=True).expand_as(s_sel).contiguous()
        shifted = s_sel.clone()
        if ok.any():
            shifted[torch.as_tensor(np.flatnonzero(ok), device=device)] = s_sel[torch.as_tensor(donor[ok], device=device)]
        arms = {
            "zero": grammar_per_anchor(model, tensors, sel, zero, device),
            "learned": grammar_per_anchor(model, tensors, sel, s_sel, device),
            "period_mean": grammar_per_anchor(model, tensors, sel, period, device),
            "shifted": grammar_per_anchor(model, tensors, sel, shifted, device),
        }
        arms["shifted"][~ok] = np.nan
        tr = data.train_pairs
        s_tr = states[torch.as_tensor(tr.anchor_rows, dtype=torch.long, device=device)]
        arms["train_zero"] = grammar_per_anchor(model, tensors, tr, torch.zeros_like(s_tr), device)
        arms["train_learned"] = grammar_per_anchor(model, tensors, tr, s_tr, device)
        for k, v in arms.items():
            rows[k].append(v)
        per_seed.append({
            "seed": seed, "selected_step": card["selected_step"], "steps_run": card["steps_run"],
            "gain_zero_minus_learned_full_selection": float(np.mean(arms["zero"] - arms["learned"])),
            "period_offset_gain_zero_minus_period_mean": float(np.mean(arms["zero"] - arms["period_mean"])),
            "beyond_period_period_mean_minus_learned": float(np.mean(arms["period_mean"] - arms["learned"])),
            "shifted_minus_learned": float(np.nanmean(arms["shifted"] - arms["learned"])),
            "train_gain_zero_minus_learned": float(np.mean(arms["train_zero"] - arms["train_learned"])),
            "card_selection_gain_subsample_total": float(card["selection_gain"]),
            "state_selection_mean_abs_standardised": float(s_sel.mean(0).abs().mean().cpu()),
        })
        print(f"  {subject} seed {seed}: full-sel gain {per_seed[-1]['gain_zero_minus_learned_full_selection']:+.4f} | "
              f"period {per_seed[-1]['period_offset_gain_zero_minus_period_mean']:+.4f} | beyond {per_seed[-1]['beyond_period_period_mean_minus_learned']:+.4f} | "
              f"shift {per_seed[-1]['shifted_minus_learned']:+.4f} | train gain {per_seed[-1]['train_gain_zero_minus_learned']:+.4f} | |mean state| {per_seed[-1]['state_selection_mean_abs_standardised']:.2f}", flush=True)
    Z, L, P = (np.median(np.asarray(rows[k]), axis=0) for k in ("zero", "learned", "period_mean"))
    S = np.asarray(rows["shifted"])
    Sm = np.full(Z.shape, np.nan); Sm[ok] = np.median(S[:, ok], axis=0)
    merged = {
        "gain_zero_minus_learned": ci(Z - L, boot_seg),
        "period_offset_zero_minus_period_mean": ci(Z - P, boot_seg),
        "beyond_period_period_mean_minus_learned": ci(P - L, boot_seg),
        "shifted_minus_learned": ci(Sm - L, boot_seg),
        "train_gain_zero_minus_learned_mean": float(np.mean(np.median(np.asarray(rows["train_zero"]), 0) - np.median(np.asarray(rows["train_learned"]), 0))),
        "n_selection_anchors": int(sel_rows.size), "n_valid_donors": int(ok.sum()),
        "n_blocks": int(np.unique(boot_seg).size),
    }
    print(f"{subject}: gain {merged['gain_zero_minus_learned']['mean']:+.4f} [{merged['gain_zero_minus_learned']['ci_low']:+.4f},{merged['gain_zero_minus_learned']['ci_high']:+.4f}] | "
          f"period {merged['period_offset_zero_minus_period_mean']['mean']:+.4f} | beyond {merged['beyond_period_period_mean_minus_learned']['mean']:+.4f} "
          f"[{merged['beyond_period_period_mean_minus_learned']['ci_low']:+.4f},{merged['beyond_period_period_mean_minus_learned']['ci_high']:+.4f}] | "
          f"shift {merged['shifted_minus_learned']['mean']:+.4f} [{merged['shifted_minus_learned']['ci_low']:+.4f},{merged['shifted_minus_learned']['ci_high']:+.4f}] | train gain {merged['train_gain_zero_minus_learned_mean']:+.4f}", flush=True)
    review = {"format": "group_event_state_v0_3_4_spatial_state_offset_control_review_v1", "subject": subject,
              "cards": [str(c) for c in cards], "merged": merged, "per_seed": per_seed,
              "development_targets_read": False, "sealed_partition_opened": False, "seizure_outcomes_read": False}
    atomic_write_json(Path(out_dir) / f"{subject}.json", review)
    return review


def _anchor_carry_full(data):
    last = data.last_event_pos
    seg = data.event_segment
    return np.where(last >= 0, seg[np.maximum(last, 0)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "reports" / "offset_control")
    ap.add_argument("--subjects", nargs="*", default=["epilepsiae_1146", "epilepsiae_583", "epilepsiae_548", "epilepsiae_922", "epilepsiae_253"])
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for subject in args.subjects:
        if subject == "epilepsiae_253":
            cards = sorted((OUTPUT_ROOT / "human" / subject / "rung900").glob("w64_d4_le0.0003_la0.003*/training_card.json"))
        else:
            cards = sorted((OUTPUT_ROOT / "evaluation" / subject / "rung900").glob("*/training_card.json"))
        started = time.time()
        review = audit_subject(subject, cards, device, args.out_dir)
        review["elapsed_seconds"] = time.time() - started
        summary.append({"subject": subject, "merged": review["merged"]})
        atomic_write_json(args.out_dir / "offset_control_summary.json", {"subjects": summary, "development_targets_read": False})


if __name__ == "__main__":
    main()
