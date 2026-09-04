#!/usr/bin/env python3
"""Offset / drift control audit for v0.3.3 S_N trainability cards.

Question: how much of the STATE_SELECTION (inner-val) gain of a frozen S_N
candidate is explained by a *constant* period-level offset of the learned
modulation (i.e. the state's mean over the selection period differing from
its TRAIN mean), versus time-resolved state information beyond that constant?

Arms (all scored on STATE_SELECTION anchors with the checkpoint's own frozen
TRAIN statistics and the frozen H_mark dispersion; no development anchor is
read):

  H            explicit history baseline (log mu_H)
  learned      checkpoint state at the correct time
  period_mean  state replaced by its mean over all inner-val anchors (input
               only; one constant vector for the whole selection period)
  segment_mean state replaced by its mean within each target segment of the
               selection period (input only)
  shifted      v0.3.2 same-segment block-circular wrong-time state (card null)
  oracle_offset per-bin constant added to log mu_H, fitted on the inner-val
               targets themselves (upper bound of any constant-offset story;
               an oracle, never a model)

Seeds are merged exactly as the cards do: per-anchor median across the final
seeds, then within-target-segment moving-block bootstrap.  This script does
not alter any card; it writes a separate provenance-rich review.
"""

from __future__ import annotations

import argparse
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
from src.topic5_group_event_state.v033_evaluator import canonical as C
from src.topic5_group_event_state.v033_training_lab.diagnostics import BOOT, _terms
from src.topic5_group_event_state.v033_training_lab.objective import TRAINABLE_REGISTRY
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json
from src.topic5_group_event_state.v033_training_lab.queue import recipe_from_dict
from src.topic5_group_event_state.v033_training_lab.trainer import load_trained
from src.topic5_group_event_state.v033_training_lab.views import view_for_request

FORMAT = "group_event_state_v0_3_3_offset_drift_control_review"
DEFAULT_DATA_ROOT = Path("/data/hfosp_group_event_state_v0_3_3")
EXPECTED_REQUESTS = {
    "epilepsiae_253": ("agent_b", "human-sn-r0-253-trainability-o1a-v1"),
    "epilepsiae_916": ("agent_b", "human-sn-r0-916-trainability-o1a-v1"),
    "epilepsiae_1096": ("agent_b_expansion", "human-sn-r0-1096-trainability-broad-v1"),
    "epilepsiae_1125": ("agent_b_expansion", "human-sn-r0-1125-trainability-broad-v1"),
    "epilepsiae_1146": ("agent_b_expansion", "human-sn-r0-1146-trainability-broad-v1"),
    "epilepsiae_384": ("agent_b_expansion", "human-sn-r0-384-trainability-broad-v1"),
    "epilepsiae_548": ("agent_b_expansion", "human-sn-r0-548-trainability-broad-v1"),
    "epilepsiae_583": ("agent_b_expansion", "human-sn-r0-583-trainability-broad-v1"),
    "epilepsiae_922": ("agent_b_expansion", "human-sn-r0-922-trainability-broad-v1"),
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


def oracle_offset(y: np.ndarray, log_mu_h: np.ndarray, log_r: np.ndarray, steps: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin constant c_b minimising the NB NLL on the given anchors (oracle)."""

    yt = torch.from_numpy(y.astype(np.float32))
    lm = torch.from_numpy(log_mu_h.astype(np.float32))
    c = torch.zeros(y.shape[1], requires_grad=True)
    opt = torch.optim.Adam([c], lr=0.05)
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = sum(C.nb_nll_torch(yt[:, b], lm[:, b] + c[b], torch.tensor(float(log_r[b]))).mean()
                   for b in range(y.shape[1]))
        loss.backward()
        opt.step()
    with torch.no_grad():
        per_anchor = torch.stack([C.nb_nll_torch(yt[:, b], lm[:, b] + c[b], torch.tensor(float(log_r[b])))
                                  for b in range(y.shape[1])], dim=1).sum(dim=1).numpy().astype(np.float64)
    return c.detach().numpy().astype(np.float64), per_anchor


def audit_subject(subject: str, owner: str, request_id: str, *, data_root: Path, device: torch.device) -> dict[str, Any]:
    card_path = data_root / owner / "search" / request_id / "card" / "training_card.json"
    request_path = data_root / "shared" / "job_requests" / f"science_{request_id}.json"
    card = json.loads(card_path.read_text())
    request = json.loads(request_path.read_text())
    if card.get("sealed_partition_opened") or card.get("development_evaluation_read"):
        raise ValueError("refusing a card that opened a forbidden partition")
    cfg = recipe_from_dict(card["recipe"])
    view, _meta = view_for_request(request, release_present=True, scaling=cfg.scaling)
    if view.input_hash != card["input_hash"] or view.split_hash != card["split_hash"]:
        raise ValueError(f"{subject}: input/split hash mismatch between view and card")
    trainable = TRAINABLE_REGISTRY[str(request["scientific_target"]["objective"])]()
    idx = view.phase_index["inner_val"]
    seg = view.bootstrap_segment(idx)
    tr_idx = view.phase_index["train"]
    h = trainable.h_only_nll(view, "inner_val")
    h_train = trainable.h_only_nll(view, "train")
    donor = block_circular_donor(view.t_anchor, view.anchor_segment, idx, horizon=view.horizon, fraction=0.5)
    ok = donor >= 0
    y = view.counts[idx]
    y_train = view.counts[tr_idx]
    mu_h = np.exp(view.log_mu_h[idx])
    mu_h_train = np.exp(view.log_mu_h[tr_idx])

    per_seed_dirs = card["search"]["incumbent"]["seed_dirs"]
    rand_by_seed = {int(r["seed"]): r for r in card["diagnostics"]["multi_seed_diagnostics"]["per_seed"]}
    rows_learned, rows_period, rows_segment, rows_shift, rows_random, rows_train_learned = [], [], [], [], [], []
    per_seed: list[dict[str, Any]] = []
    for seed_dir in per_seed_dirs:
        seed_dir = Path(seed_dir)
        result = json.loads((seed_dir / "result.json").read_text())
        seed = int(result["seed"])
        model = load_trained(seed_dir, trainable, view, device)
        correct = _terms(trainable, view, model, "inner_val", device)
        train_terms = _terms(trainable, view, model, "train", device)
        state = correct.state_raw
        period_state = state.mean(dim=0, keepdim=True).expand_as(state).contiguous()
        seg_state = state.clone()
        for s in np.unique(seg):
            rows = torch.from_numpy(np.flatnonzero(seg == s)).to(state.device)
            seg_state[rows] = state[rows].mean(dim=0, keepdim=True)
        shifted_state = state.clone()
        if ok.any():
            shifted_state[torch.from_numpy(np.flatnonzero(ok)).to(state.device)] = state[torch.from_numpy(donor[ok]).to(state.device)]
        period = _terms(trainable, view, model, "inner_val", device, state_override=period_state)
        segm = _terms(trainable, view, model, "inner_val", device, state_override=seg_state)
        shift = _terms(trainable, view, model, "inner_val", device, state_override=shifted_state)
        rnd = rand_by_seed.get(seed)
        random_nll = None
        if rnd is not None:
            random_model = load_trained(Path(rnd["random_checkpoint"]).parent, trainable, view, device)
            random_nll = _terms(trainable, view, random_model, "inner_val", device).nll.cpu().numpy().astype(np.float64)
        nll_l = correct.nll.cpu().numpy().astype(np.float64)
        nll_p = period.nll.cpu().numpy().astype(np.float64)
        nll_s = segm.nll.cpu().numpy().astype(np.float64)
        nll_sh = shift.nll.cpu().numpy().astype(np.float64)
        nll_sh[~ok] = np.nan
        mod = correct.modulation.cpu().numpy().astype(np.float64)
        mod_tr = train_terms.modulation.cpu().numpy().astype(np.float64)
        rows_learned.append(nll_l)
        rows_period.append(nll_p)
        rows_segment.append(nll_s)
        rows_shift.append(nll_sh)
        rows_random.append(random_nll if random_nll is not None else np.full_like(nll_l, np.nan))
        rows_train_learned.append(train_terms.nll.cpu().numpy().astype(np.float64))
        per_seed.append({
            "seed": seed, "checkpoint_sha256": sha256(seed_dir / "checkpoint.pt"),
            "selected_step": result.get("selected_step"),
            "gain_total_mean": float((h - nll_l).mean()),
            "gain_period_offset_mean": float((h - nll_p).mean()),
            "gain_segment_offset_mean": float((h - nll_s).mean()),
            "gain_beyond_period_mean": float((nll_p - nll_l).mean()),
            "gain_beyond_segment_mean": float((nll_s - nll_l).mean()),
            "shifted_minus_correct_mean": float(np.nanmean(nll_sh - nll_l)),
            "train_gain_mean": float((h_train - rows_train_learned[-1]).mean()),
            "modulation_mean_per_bin_inner_val": mod.mean(axis=0).tolist(),
            "modulation_std_per_bin_inner_val": mod.std(axis=0).tolist(),
            "modulation_mean_per_bin_train": mod_tr.mean(axis=0).tolist(),
            "modulation_std_per_bin_train": mod_tr.std(axis=0).tolist(),
            "per_bin_gain_inner_val": (correct.per_bin_nll.cpu().numpy() * 0 + (
                torch.stack([C.nb_nll_torch(torch.from_numpy(y[:, b].astype(np.float32)),
                                            torch.from_numpy(view.log_mu_h[idx][:, b].astype(np.float32)),
                                            torch.tensor(float(view.log_r_h[b]))) for b in range(view.n_bins)], dim=1).numpy()
                - correct.per_bin_nll.cpu().numpy())).mean(axis=0).tolist(),
        })
    L = np.asarray(rows_learned)
    P = np.asarray(rows_period)
    S = np.asarray(rows_segment)
    SH = np.asarray(rows_shift)
    RD = np.asarray(rows_random)
    l_med, p_med, s_med = np.median(L, axis=0), np.median(P, axis=0), np.median(S, axis=0)
    sh_med = np.full(h.shape, np.nan)
    if ok.any():
        sh_med[ok] = np.median(SH[:, ok], axis=0)
    r_med = np.median(RD, axis=0) if np.isfinite(RD).all() else None
    c_or, nll_or = oracle_offset(y, view.log_mu_h[idx], view.log_r_h)
    c_tr, nll_or_tr = oracle_offset(y_train, view.log_mu_h[tr_idx], view.log_r_h)
    # first-half vs second-half of the selection period (by anchor time)
    order = np.argsort(view.t_anchor[idx], kind="stable")
    half = order.size // 2
    first, second = order[:half], order[half:]
    merged = {
        "gain_total_H_minus_learned": ci(h - l_med, seg),
        "gain_period_offset_H_minus_period_mean_state": ci(h - p_med, seg),
        "gain_segment_offset_H_minus_segment_mean_state": ci(h - s_med, seg),
        "gain_beyond_period_offset_period_mean_minus_learned": ci(p_med - l_med, seg),
        "gain_beyond_segment_offset_segment_mean_minus_learned": ci(s_med - l_med, seg),
        "shifted_minus_correct": ci(sh_med - l_med, seg),
        "learned_minus_random": None if r_med is None else ci(l_med - r_med, seg),
        "oracle_offset_H_minus_oracle": ci(h - nll_or, seg),
        "oracle_offset_per_bin_log_scale_inner_val": c_or.tolist(),
        "oracle_offset_per_bin_log_scale_train": c_tr.tolist(),
        "oracle_offset_gain_train_mean": float((h_train - nll_or_tr).mean()),
        "gain_total_first_half_of_selection": float((h - l_med)[first].mean()),
        "gain_total_second_half_of_selection": float((h - l_med)[second].mean()),
        "h_mark_calibration": {
            "inner_val_mean_count_per_bin": y.mean(axis=0).tolist(),
            "inner_val_mean_mu_h_per_bin": mu_h.mean(axis=0).tolist(),
            "inner_val_count_over_mu_ratio_per_bin": (y.mean(axis=0) / np.maximum(mu_h.mean(axis=0), 1e-9)).tolist(),
            "train_mean_count_per_bin": y_train.mean(axis=0).tolist(),
            "train_mean_mu_h_per_bin": mu_h_train.mean(axis=0).tolist(),
            "train_count_over_mu_ratio_per_bin": (y_train.mean(axis=0) / np.maximum(mu_h_train.mean(axis=0), 1e-9)).tolist(),
            "nll_h_mean_inner_val": float(h.mean()), "nll_h_mean_train": float(h_train.mean()),
        },
        "n_anchors": int(idx.size), "n_valid_donors": int(ok.sum()),
        "n_blocks": int(np.unique(view.blocks("inner_val")).size),
    }
    return {
        "subject": subject, "request_id": request_id, "card_path": str(card_path), "card_sha256": sha256(card_path),
        "input_hash": view.input_hash, "split_hash": view.split_hash, "recipe_config_hash": cfg.config_hash(),
        "n_seeds": len(per_seed), "seed_merge_rule": "median per anchor across final seeds, then within-target-segment moving-block bootstrap",
        "merged": merged, "per_seed": per_seed,
        "definitions": {
            "period_mean": "checkpoint state replaced by its mean over all STATE_SELECTION anchors (one constant vector); input only",
            "segment_mean": "checkpoint state replaced by its mean within each target segment of STATE_SELECTION; input only",
            "oracle_offset": "per-bin constant added to log mu_H fitted on the STATE_SELECTION targets (upper bound; oracle)",
            "gain_beyond_period_offset": "NLL(period_mean) - NLL(learned): the part of the gain a constant period offset cannot explain",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path,
                        default=DEFAULT_DATA_ROOT / "supervisor_reports" / "trainability_incremental" / "offset_drift_control")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subjects", nargs="*", default=list(EXPECTED_REQUESTS))
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out_root.mkdir(parents=True, exist_ok=True)
    reviews = []
    for subject in args.subjects:
        owner, request_id = EXPECTED_REQUESTS[subject]
        started = time.time()
        review = audit_subject(subject, owner, request_id, data_root=args.data_root, device=device)
        review["elapsed_seconds"] = time.time() - started
        atomic_write_json(args.out_root / f"{subject}.json", review)
        m = review["merged"]
        print(f"{subject}: total {m['gain_total_H_minus_learned']['mean']:+.4f} "
              f"[{m['gain_total_H_minus_learned']['ci_low']:+.4f},{m['gain_total_H_minus_learned']['ci_high']:+.4f}] | "
              f"period_offset {m['gain_period_offset_H_minus_period_mean_state']['mean']:+.4f} | "
              f"beyond_period {m['gain_beyond_period_offset_period_mean_minus_learned']['mean']:+.4f} "
              f"[{m['gain_beyond_period_offset_period_mean_minus_learned']['ci_low']:+.4f},{m['gain_beyond_period_offset_period_mean_minus_learned']['ci_high']:+.4f}] | "
              f"segment_offset {m['gain_segment_offset_H_minus_segment_mean_state']['mean']:+.4f} | "
              f"oracle_offset {m['oracle_offset_H_minus_oracle']['mean']:+.4f} | "
              f"calib ratio inner {np.round(m['h_mark_calibration']['inner_val_count_over_mu_ratio_per_bin'],3).tolist()} "
              f"train {np.round(m['h_mark_calibration']['train_count_over_mu_ratio_per_bin'],3).tolist()}", flush=True)
        reviews.append(review)
    summary = {
        "format": FORMAT, "created_epoch": time.time(), "device": str(device),
        "source_git_head": git_head(Path(__file__).resolve().parents[1]),
        "producer_path": str(Path(__file__).resolve()), "producer_sha256": sha256(Path(__file__).resolve()),
        "development_evaluation_read": False, "sealed_partition_opened": False,
        "subjects": [{"subject": r["subject"], "card_sha256": r["card_sha256"], "merged": r["merged"]} for r in reviews],
        "pid": os.getpid(),
    }
    atomic_write_json(args.out_root / "offset_drift_control_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
