#!/usr/bin/env python3
"""Retrain the selected S_N recipes on top of a causally recalibrated H_mark.

Motivation (offset/drift audit, 2026-09-03): the STATE_SELECTION gain of the
frozen S_N candidates is largely reproduced by replacing the state with one
constant vector, and H_mark under-predicts counts in the selection period.
A level miscalibration of the baseline is therefore a competing explanation
for "state information".

This diagnostic replaces log mu_H by

    log mu_H(a) + c(a),   c(a) = log((sum_b sum_bins y_b + 1) / (sum_b sum_bins mu_H,b + 1))

where b ranges over exposed anchors of the same state-carry segment whose
whole target window ended before anchor a (t_b + horizon <= t_a) and within a
trailing window of ``--window-seconds``.  Only past observed counts are used,
so the recalibration is causal.  The selected recipe is then retrained with
the same seeds (learned arm and frozen-random-encoder arm) and scored on
STATE_SELECTION with the same seed-median / block-bootstrap rule as the cards.

Optional arms (2026-09-03 follow-up to the user's "slow variable" question):
``--baseline mark`` keeps the original H_mark instead of the recalibrated one,
and ``--taus`` replaces the recipe's time bank (e.g. 21600 43200 86400 for a
6/12/24 h "slow bank").  A learned slow bank that beats its frozen-random twin
on top of the causal rate recalibration is evidence for a *mark-dependent*
slow component; a tie says the slow component is a rate level only.

TRAIN + STATE_SELECTION only.  No development anchor is read; nothing here is
an H1/H2/H3 result.  Runs are written to a separate directory and never touch
the existing cards or checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.topic5_group_event_state.v032_model.evaluate import block_bootstrap_mean_ci
from src.topic5_group_event_state.v032_model.shift import block_circular_donor
from src.topic5_group_event_state.v033_training_lab.diagnostics import BOOT, _terms
from src.topic5_group_event_state.v033_training_lab.objective import TRAINABLE_REGISTRY
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json
from src.topic5_group_event_state.v033_training_lab.queue import recipe_from_dict
from src.topic5_group_event_state.v033_training_lab.trainer import load_trained, train_recipe
from src.topic5_group_event_state.v033_training_lab.views import view_for_request

FORMAT = "group_event_state_v0_3_3_recalibrated_baseline_arms"
DEFAULT_DATA_ROOT = Path("/data/hfosp_group_event_state_v0_3_3")
EXPECTED_REQUESTS = {
    "epilepsiae_1146": ("agent_b_expansion", "human-sn-r0-1146-trainability-broad-v1"),
    "epilepsiae_1096": ("agent_b_expansion", "human-sn-r0-1096-trainability-broad-v1"),
    "epilepsiae_548": ("agent_b_expansion", "human-sn-r0-548-trainability-broad-v1"),
    "epilepsiae_922": ("agent_b_expansion", "human-sn-r0-922-trainability-broad-v1"),
    "epilepsiae_1125": ("agent_b_expansion", "human-sn-r0-1125-trainability-broad-v1"),
    "epilepsiae_384": ("agent_b_expansion", "human-sn-r0-384-trainability-broad-v1"),
    "epilepsiae_583": ("agent_b_expansion", "human-sn-r0-583-trainability-broad-v1"),
    "epilepsiae_253": ("agent_b", "human-sn-r0-253-trainability-o1a-v1"),
    "epilepsiae_916": ("agent_b", "human-sn-r0-916-trainability-o1a-v1"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
                          capture_output=True).stdout.strip()


def ci(values: np.ndarray, segments: np.ndarray) -> dict[str, Any]:
    out = block_bootstrap_mean_ci(np.asarray(values, dtype=np.float64), np.asarray(segments), **BOOT)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def causal_recalibration(view, *, window_seconds: float, clip: float) -> tuple[np.ndarray, dict[str, Any]]:
    """c(a) from completed past exposed anchors of the same carry segment (strictly causal)."""

    exposed = np.sort(np.r_[view.phase_index["train"], view.phase_index["inner_val"]])
    t = view.t_anchor
    seg = view.anchor_segment
    y_sum = view.counts[exposed].sum(axis=1).astype(np.float64)
    mu_sum = np.exp(view.log_mu_h[exposed]).sum(axis=1)
    if not np.isfinite(mu_sum).all() or (view.counts[exposed] < 0).any():
        raise ValueError("exposed anchors must carry finite H and observed counts")
    t_e, seg_e = t[exposed], seg[exposed]
    c = np.zeros(view.t_anchor.size, dtype=np.float64)
    n_past = np.zeros(view.t_anchor.size, dtype=np.int64)
    for a in exposed:
        past = (seg_e == seg[a]) & (t_e + view.horizon <= t[a]) & (t_e >= t[a] - float(window_seconds))
        n_past[a] = int(past.sum())
        if n_past[a] == 0:
            continue
        c[a] = float(np.clip(np.log((y_sum[past].sum() + 1.0) / (mu_sum[past].sum() + 1.0)), -clip, clip))
    info = {
        "window_seconds": float(window_seconds), "clip": float(clip),
        "n_exposed": int(exposed.size), "fraction_with_past": float((n_past[exposed] > 0).mean()),
        "c_train_mean": float(c[view.phase_index["train"]].mean()),
        "c_inner_val_mean": float(c[view.phase_index["inner_val"]].mean()),
        "c_inner_val_std": float(c[view.phase_index["inner_val"]].std()),
        "rule": "log((sum y + 1)/(sum mu_H + 1)) over completed same-carry-segment exposed anchors in the trailing window",
    }
    return c, info


def run_subject(subject: str, owner: str, request_id: str, *, data_root: Path, out_root: Path, device: torch.device,
                window_seconds: float, clip: float, with_random: bool, baseline: str = "recal",
                taus_override: tuple[float, ...] | None = None) -> dict[str, Any]:
    card_path = data_root / owner / "search" / request_id / "card" / "training_card.json"
    request = json.loads((data_root / "shared" / "job_requests" / f"science_{request_id}.json").read_text())
    card = json.loads(card_path.read_text())
    if card.get("sealed_partition_opened") or card.get("development_evaluation_read"):
        raise ValueError("refusing a card that opened a forbidden partition")
    cfg = recipe_from_dict(card["recipe"])
    if taus_override is not None:
        cfg = cfg.with_overrides(arch=replace(cfg.arch, taus_seconds=tuple(float(t) for t in taus_override)))
    view, _meta = view_for_request(request, release_present=True, scaling=cfg.scaling)
    if view.input_hash != card["input_hash"] or view.split_hash != card["split_hash"]:
        raise ValueError(f"{subject}: input/split hash mismatch")
    trainable = TRAINABLE_REGISTRY[str(request["scientific_target"]["objective"])]()
    c, info = causal_recalibration(view, window_seconds=window_seconds, clip=clip)
    if baseline == "recal":
        log_mu_recal = view.log_mu_h + c[:, None]
        recal_view = replace(view, log_mu_h=log_mu_recal, h_source=f"H_mark_causal_recalibration_{int(window_seconds)}s")
    elif baseline == "mark":
        recal_view = view
        info = {**info, "note": "baseline=mark: H_mark kept; recalibration computed for reference only"}
    else:
        raise ValueError(f"unknown baseline {baseline!r}")
    idx = view.phase_index["inner_val"]
    seg = view.bootstrap_segment(idx)
    tr_idx = view.phase_index["train"]
    h_mark = trainable.h_only_nll(view, "inner_val")
    h_recal = trainable.h_only_nll(recal_view, "inner_val")
    h_mark_tr = trainable.h_only_nll(view, "train")
    h_recal_tr = trainable.h_only_nll(recal_view, "train")
    donor = block_circular_donor(view.t_anchor, view.anchor_segment, idx, horizon=view.horizon, fraction=0.5)
    ok = donor >= 0
    seeds = [int(s) for s in card["seed_dispersion"]["seeds"]]
    subject_root = out_root / subject
    rows_l, rows_p, rows_sh, rows_r, per_seed = [], [], [], [], []
    for seed in seeds:
        run_dir = subject_root / f"seed_{seed}" / "learned"
        result = train_recipe(trainable, recal_view, cfg, seed, device=device, out_dir=run_dir, arm="learned")
        if result["status"] != "complete":
            raise RuntimeError(f"{subject} seed {seed}: learned arm status {result['status']}")
        model = load_trained(run_dir, trainable, recal_view, device)
        correct = _terms(trainable, recal_view, model, "inner_val", device)
        state = correct.state_raw
        period_state = state.mean(dim=0, keepdim=True).expand_as(state).contiguous()
        shifted_state = state.clone()
        if ok.any():
            shifted_state[torch.from_numpy(np.flatnonzero(ok)).to(state.device)] = state[torch.from_numpy(donor[ok]).to(state.device)]
        nll_l = correct.nll.cpu().numpy().astype(np.float64)
        nll_p = _terms(trainable, recal_view, model, "inner_val", device, state_override=period_state).nll.cpu().numpy().astype(np.float64)
        nll_sh = _terms(trainable, recal_view, model, "inner_val", device, state_override=shifted_state).nll.cpu().numpy().astype(np.float64)
        nll_sh[~ok] = np.nan
        nll_r = np.full_like(nll_l, np.nan)
        random_summary = None
        if with_random:
            rdir = subject_root / f"seed_{seed}" / "random_reservoir"
            rres = train_recipe(trainable, recal_view, cfg, seed, device=device, out_dir=rdir, arm="random_reservoir")
            if rres["status"] != "complete":
                raise RuntimeError(f"{subject} seed {seed}: random arm status {rres['status']}")
            rmodel = load_trained(rdir, trainable, recal_view, device)
            nll_r = _terms(trainable, recal_view, rmodel, "inner_val", device).nll.cpu().numpy().astype(np.float64)
            random_summary = {"selected_step": rres.get("selected_step"), "checkpoint_sha256": result and sha256(rdir / "checkpoint.pt")}
        rows_l.append(nll_l); rows_p.append(nll_p); rows_sh.append(nll_sh); rows_r.append(nll_r)
        per_seed.append({
            "seed": seed, "selected_step": result.get("selected_step"), "selected_in_warmup": result.get("selected_in_warmup"),
            "checkpoint_sha256": sha256(run_dir / "checkpoint.pt"), "train_gain_vs_recal_mean": float((h_recal_tr - _terms(trainable, recal_view, model, "train", device).nll.cpu().numpy()).mean()),
            "gain_vs_recal_mean": float((h_recal - nll_l).mean()), "period_offset_gain_mean": float((h_recal - nll_p).mean()),
            "beyond_period_mean": float((nll_p - nll_l).mean()), "shifted_minus_correct_mean": float(np.nanmean(nll_sh - nll_l)),
            "learned_minus_random_mean": None if not with_random else float((nll_l - nll_r).mean()),
            "random": random_summary,
        })
        print(f"  {subject} seed {seed}: step {result.get('selected_step')} gain_vs_recal {per_seed[-1]['gain_vs_recal_mean']:+.4f} "
              f"period {per_seed[-1]['period_offset_gain_mean']:+.4f} beyond {per_seed[-1]['beyond_period_mean']:+.4f} "
              f"shift {per_seed[-1]['shifted_minus_correct_mean']:+.4f} l-r {per_seed[-1]['learned_minus_random_mean']}", flush=True)
    L, P, SH, RD = (np.asarray(r) for r in (rows_l, rows_p, rows_sh, rows_r))
    l_med, p_med = np.median(L, axis=0), np.median(P, axis=0)
    sh_med = np.full(h_mark.shape, np.nan)
    if ok.any():
        sh_med[ok] = np.median(SH[:, ok], axis=0)
    r_med = np.median(RD, axis=0) if with_random else None
    merged = {
        "recalibration_gain_H_mark_minus_H_recal_inner_val": ci(h_mark - h_recal, seg),
        "recalibration_gain_H_mark_minus_H_recal_train_mean": float((h_mark_tr - h_recal_tr).mean()),
        "card_gain_H_mark_minus_learned_original": card["blocked_inner_val_gain"],
        "gain_H_recal_minus_learned": ci(h_recal - l_med, seg),
        "gain_H_mark_minus_learned_on_recal": ci(h_mark - l_med, seg),
        "period_offset_gain_H_recal_minus_period_mean_state": ci(h_recal - p_med, seg),
        "beyond_period_offset_period_mean_minus_learned": ci(p_med - l_med, seg),
        "shifted_minus_correct": ci(sh_med - l_med, seg),
        "learned_minus_random": None if r_med is None else ci(l_med - r_med, seg),
        "n_anchors": int(idx.size), "n_valid_donors": int(ok.sum()), "n_blocks": int(np.unique(view.blocks("inner_val")).size),
        "recalibration": info,
    }
    review = {
        "format": FORMAT, "subject": subject, "request_id": request_id, "card_path": str(card_path), "card_sha256": sha256(card_path),
        "input_hash": view.input_hash, "split_hash": view.split_hash, "recipe_config_hash": cfg.config_hash(),
        "seeds": seeds, "seed_merge_rule": "median per anchor across seeds, then within-target-segment moving-block bootstrap",
        "baseline": baseline, "taus_seconds": list(cfg.arch.taus_seconds),
        "merged": merged, "per_seed": per_seed, "development_evaluation_read": False, "sealed_partition_opened": False,
        "evidence_label": "DIAGNOSTIC", "definition": __doc__,
    }
    atomic_write_json(subject_root / "review.json", review)
    m = merged
    print(f"{subject}: recal_gain(H_mark-H_recal) {m['recalibration_gain_H_mark_minus_H_recal_inner_val']['mean']:+.4f} | "
          f"H_recal-learned {m['gain_H_recal_minus_learned']['mean']:+.4f} [{m['gain_H_recal_minus_learned']['ci_low']:+.4f},{m['gain_H_recal_minus_learned']['ci_high']:+.4f}] | "
          f"beyond_period {m['beyond_period_offset_period_mean_minus_learned']['mean']:+.4f} | shift {m['shifted_minus_correct']['mean']:+.4f} | "
          f"l-r {None if r_med is None else round(m['learned_minus_random']['mean'],4)}", flush=True)
    return review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path,
                        default=DEFAULT_DATA_ROOT / "supervisor_reports" / "trainability_incremental" / "recalibrated_baseline_arms")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--window-seconds", type=float, default=3.0 * 3600.0)
    parser.add_argument("--clip", type=float, default=1.5)
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--baseline", choices=("recal", "mark"), default="recal")
    parser.add_argument("--taus", nargs="*", type=float, default=None,
                        help="override the recipe time bank, e.g. 21600 43200 86400 for a 6/12/24 h slow bank")
    parser.add_argument("--subjects", nargs="*", default=list(EXPECTED_REQUESTS))
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_root / "RUN_NOTE.json", {
        "format": FORMAT, "started_epoch": time.time(), "pid": os.getpid(), "device": str(device),
        "scope": "TRAIN + STATE_SELECTION only; DIAGNOSTIC; no development / seizure / sealed access",
        "source_git_head": git_head(Path(__file__).resolve().parents[1]),
        "producer_sha256": sha256(Path(__file__).resolve()), "window_seconds": args.window_seconds, "clip": args.clip,
        "baseline": args.baseline, "taus_override": args.taus,
    })
    reviews = []
    for subject in args.subjects:
        owner, rid = EXPECTED_REQUESTS[subject]
        started = time.time()
        review = run_subject(subject, owner, rid, data_root=args.data_root, out_root=args.out_root, device=device,
                             window_seconds=args.window_seconds, clip=args.clip, with_random=not args.no_random,
                             baseline=args.baseline, taus_override=None if not args.taus else tuple(args.taus))
        review["elapsed_seconds"] = time.time() - started
        reviews.append({"subject": subject, "merged": review["merged"], "card_sha256": review["card_sha256"]})
        atomic_write_json(args.out_root / "recalibrated_baseline_summary.json", {
            "format": FORMAT, "updated_epoch": time.time(), "subjects": reviews,
            "development_evaluation_read": False, "sealed_partition_opened": False,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
