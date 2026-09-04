#!/usr/bin/env python3
"""Multi-shift (E3) null for frozen v0.3.3 S_N checkpoints, plus its decomposition.

For each subject / final seed, the STATE_SELECTION anchor states are circularly
shifted within each recording segment by 32 different offsets (v0.3.2
``block_circular_donor`` rule: same segment, >= one horizon away).  Reported:

  * percentile of the correct-time NLL inside the 32-shift null distribution
    (seed-median per anchor, then mean over anchors);
  * decomposition of every shift delta into
        NLL(shift) - NLL(learned) = [NLL(shift) - NLL(period_mean)] + [NLL(period_mean) - NLL(learned)]
    i.e. "misplaced variation hurts" + "right-time variation helps beyond a constant".

TRAIN + STATE_SELECTION only; no development anchor is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.topic5_group_event_state.v032_model.evaluate import block_bootstrap_mean_ci
from src.topic5_group_event_state.v032_model.shift import block_circular_donor
from src.topic5_group_event_state.v033_training_lab.diagnostics import BOOT, _terms
from src.topic5_group_event_state.v033_training_lab.objective import TRAINABLE_REGISTRY
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json
from src.topic5_group_event_state.v033_training_lab.queue import recipe_from_dict
from src.topic5_group_event_state.v033_training_lab.trainer import load_trained
from src.topic5_group_event_state.v033_training_lab.views import view_for_request

DEFAULT_DATA_ROOT = Path("/data/hfosp_group_event_state_v0_3_3")
EXPECTED_REQUESTS = {
    "epilepsiae_1096": ("agent_b_expansion", "human-sn-r0-1096-trainability-broad-v1"),
    "epilepsiae_1125": ("agent_b_expansion", "human-sn-r0-1125-trainability-broad-v1"),
    "epilepsiae_1146": ("agent_b_expansion", "human-sn-r0-1146-trainability-broad-v1"),
    "epilepsiae_253": ("agent_b", "human-sn-r0-253-trainability-o1a-v1"),
    "epilepsiae_384": ("agent_b_expansion", "human-sn-r0-384-trainability-broad-v1"),
    "epilepsiae_548": ("agent_b_expansion", "human-sn-r0-548-trainability-broad-v1"),
    "epilepsiae_583": ("agent_b_expansion", "human-sn-r0-583-trainability-broad-v1"),
    "epilepsiae_916": ("agent_b", "human-sn-r0-916-trainability-o1a-v1"),
    "epilepsiae_922": ("agent_b_expansion", "human-sn-r0-922-trainability-broad-v1"),
}
N_SHIFTS = 32


def ci(values, segments):
    out = block_bootstrap_mean_ci(np.asarray(values, dtype=np.float64), np.asarray(segments), **BOOT)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", type=Path,
                    default=DEFAULT_DATA_ROOT / "supervisor_reports" / "trainability_incremental" / "multi_shift_null")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--subjects", nargs="*", default=list(EXPECTED_REQUESTS))
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out_root.mkdir(parents=True, exist_ok=True)
    fractions = np.linspace(0.03, 0.97, N_SHIFTS)
    summary = []
    for subject in args.subjects:
        owner, rid = EXPECTED_REQUESTS[subject]
        card = json.loads((args.data_root / owner / "search" / rid / "card" / "training_card.json").read_text())
        request = json.loads((args.data_root / "shared" / "job_requests" / f"science_{rid}.json").read_text())
        cfg = recipe_from_dict(card["recipe"])
        view, _ = view_for_request(request, release_present=True, scaling=cfg.scaling)
        trainable = TRAINABLE_REGISTRY[str(request["scientific_target"]["objective"])]()
        idx = view.phase_index["inner_val"]
        seg = view.bootstrap_segment(idx)
        h = trainable.h_only_nll(view, "inner_val")
        donors = []
        for fr in fractions:
            d = block_circular_donor(view.t_anchor, view.anchor_segment, idx, horizon=view.horizon, fraction=float(fr))
            donors.append(d)
        donors = np.stack(donors)                       # (S, A) local donor index or -1
        valid = donors >= 0
        learned_rows, period_rows, trend_rows, shift_rows = [], [], [], []
        t_local = view.t_anchor[idx]
        for seed_dir in card["search"]["incumbent"]["seed_dirs"]:
            model = load_trained(Path(seed_dir), trainable, view, device)
            correct = _terms(trainable, view, model, "inner_val", device)
            state = correct.state_raw
            nll_l = correct.nll.cpu().numpy().astype(np.float64)
            period_state = state.mean(dim=0, keepdim=True).expand_as(state).contiguous()
            nll_p = _terms(trainable, view, model, "inner_val", device, state_override=period_state).nll.cpu().numpy().astype(np.float64)
            # linear-in-time trend per target segment (a slow ramp), input only
            trend_state = state.clone()
            st_np = state.detach().cpu().numpy().astype(np.float64)
            for sg in np.unique(seg):
                rows = np.flatnonzero(seg == sg)
                if rows.size >= 3:
                    tt = t_local[rows] - t_local[rows].mean()
                    A = np.stack([np.ones(rows.size), tt], axis=1)
                    coef, *_ = np.linalg.lstsq(A, st_np[rows], rcond=None)
                    fit = A @ coef
                else:
                    fit = np.repeat(st_np[rows].mean(axis=0, keepdims=True), rows.size, axis=0)
                trend_state[torch.from_numpy(rows).to(state.device)] = torch.from_numpy(fit).to(state.device, state.dtype)
            nll_t = _terms(trainable, view, model, "inner_val", device, state_override=trend_state).nll.cpu().numpy().astype(np.float64)
            per_shift = np.full((N_SHIFTS, idx.size), np.nan)
            for s in range(N_SHIFTS):
                ok = valid[s]
                if not ok.any():
                    continue
                shifted = state.clone()
                shifted[torch.from_numpy(np.flatnonzero(ok)).to(state.device)] = state[torch.from_numpy(donors[s][ok]).to(state.device)]
                nll_s = _terms(trainable, view, model, "inner_val", device, state_override=shifted).nll.cpu().numpy().astype(np.float64)
                nll_s[~ok] = np.nan
                per_shift[s] = nll_s
            learned_rows.append(nll_l); period_rows.append(nll_p); trend_rows.append(nll_t); shift_rows.append(per_shift)
        L = np.median(np.asarray(learned_rows), axis=0)          # (A,)
        P = np.median(np.asarray(period_rows), axis=0)
        T = np.median(np.asarray(trend_rows), axis=0)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S = np.nanmedian(np.asarray(shift_rows), axis=0)         # (S, A) seed-median
        # per-shift anchor means (only anchors valid for that shift)
        shift_means = np.nanmean(S, axis=1)                      # (S,)
        learned_means_matched = np.array([np.nanmean(L[np.isfinite(S[s])]) for s in range(N_SHIFTS)])
        deltas = shift_means - learned_means_matched             # >0 favours correct time
        n_valid_shifts = int(np.isfinite(deltas).sum())
        percentile = float(np.mean(deltas[np.isfinite(deltas)] > 0)) if n_valid_shifts else float("nan")
        # decomposition per shift on that shift's valid anchors, then averaged over shifts
        harms, helps = [], []
        for s_i in range(N_SHIFTS):
            ok = np.isfinite(S[s_i])
            if ok.any():
                harms.append(float(np.mean(S[s_i][ok] - P[ok])))
                helps.append(float(np.mean(P[ok] - L[ok])))
        harm = float(np.mean(harms)) if harms else float("nan")
        help_ = float(np.mean(helps)) if helps else float("nan")
        beyond = ci(P - L, seg)
        beyond_trend = ci(T - L, seg)
        trend_gain = ci(h - T, seg)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_delta_ci = ci(np.nanmean(S, axis=0) - L, seg)
        row = {
            "subject": subject, "n_shifts": N_SHIFTS, "n_valid_shifts": n_valid_shifts,
            "fraction_of_shifts_worse_than_correct_time": percentile,
            "shift_delta_min": float(np.nanmin(deltas)), "shift_delta_median": float(np.nanmedian(deltas)),
            "shift_delta_max": float(np.nanmax(deltas)),
            "mean_over_shifts_delta_ci": mean_delta_ci,
            "decomposition_on_common_anchors": {"misplaced_variation_hurts": harm, "right_time_beyond_constant_helps": help_,
                                                "n_shifts_used": len(harms)},
            "beyond_period_offset_ci": beyond,
            "trend_arm_gain_h_minus_trend_ci": trend_gain,
            "beyond_linear_trend_trend_minus_learned_ci": beyond_trend,
            "gain_total_ci": ci(h - L, seg),
            "n_anchors": int(idx.size), "n_blocks": int(np.unique(view.blocks("inner_val")).size),
            "card_sha256": card.get("config_hash"),
        }
        atomic_write_json(args.out_root / f"{subject}.json", {**row, "per_shift_delta": deltas.tolist(), "fractions": fractions.tolist()})
        summary.append(row)
        print(f"{subject}: shifts worse than correct {100*percentile:.0f}% of {n_valid_shifts} | delta min/med/max "
              f"{row['shift_delta_min']:+.3f}/{row['shift_delta_median']:+.3f}/{row['shift_delta_max']:+.3f} | "
              f"hurt {harm:+.3f} help {help_:+.3f} | beyond const {beyond['mean']:+.3f} [{beyond['ci_low']:+.3f},{beyond['ci_high']:+.3f}] | "
              f"trend arm gain {trend_gain['mean']:+.3f} beyond trend {beyond_trend['mean']:+.3f} [{beyond_trend['ci_low']:+.3f},{beyond_trend['ci_high']:+.3f}]", flush=True)
    atomic_write_json(args.out_root / "multi_shift_null_summary.json",
                      {"format": "group_event_state_v0_3_3_multi_shift_null", "created_epoch": time.time(),
                       "development_evaluation_read": False, "sealed_partition_opened": False, "subjects": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
