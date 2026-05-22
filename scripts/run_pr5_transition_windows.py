"""Template-switching transition_odds in baseline/pre/post seizure windows.

Per-subject per-state next-event transition_odds, pooling across seizure
windows. State definitions match ``SEIZURE_PROXIMITY_CONFIGS`` main:
  baseline = [-4h, -1h], pre = [-1h, -0.25h], post = [+0.25h, +1h]

For each (subject, state):
  - Within each seizure's window, intersect with the recording's block
    coverage so cross-gap pairs are excluded (consistent with PR-7 §3.6).
  - For consecutive event pairs (sorted by time) that fall in the SAME
    block AND the SAME window, count same/opposite transitions.
  - Pool across windows: p_next_opposite_pooled, baseline_odds_pooled,
    transition_odds_pooled. lift = transition_odds / baseline_odds.

This adds the PR-7 §17 "unmeasured form (3) seizure-proximity switching"
case to the descriptive output. NOT a cohort inference claim.

Outputs:
  results/interictal_propagation/pr5_transition_windows.json
  results/interictal_propagation/pr5_transition_windows.csv

Topic 0 Step 5e masked rerun: pass ``--masked-features`` to re-route paths
to ``results/interictal_propagation_masked/``. Labels are read from the
masked per-subject PR-1 JSONs produced by Step 5a — no re-clustering, so
path routing alone is the masked fix.
"""
from __future__ import annotations
import argparse
import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interictal_propagation import (  # noqa: E402
    SEIZURE_PROXIMITY_CONFIGS,
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.event_periodicity import load_seizure_times  # noqa: E402
from scripts.run_interictal_propagation import (  # noqa: E402
    YUQUAN_ROOT,
    EPILEPSIAE_ROOT,
    _subject_dir,
    _has_propagation_inputs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default (legacy / non-masked) paths. `_apply_masked_paths()` swaps these.
RESULTS_DIR = ROOT / "results" / "interictal_propagation"
PR1_SUBJECT_SUMMARY = RESULTS_DIR / "pr1_subject_summary_n40.json"
OUT_JSON = RESULTS_DIR / "pr5_transition_windows.json"
OUT_CSV = RESULTS_DIR / "pr5_transition_windows.csv"


def _apply_masked_paths() -> None:
    """Re-route input + output paths to the `_masked` parallel tree.

    Note: masked PR-1 summary is `pr1_subject_summary.json` (no `_n40`
    suffix) — see scripts/run_pr5b_share_extended.py:_apply_masked_paths.
    """
    global RESULTS_DIR, PR1_SUBJECT_SUMMARY, OUT_JSON, OUT_CSV
    RESULTS_DIR = ROOT / "results" / "interictal_propagation_masked"
    PR1_SUBJECT_SUMMARY = RESULTS_DIR / "pr1_subject_summary.json"
    OUT_JSON = RESULTS_DIR / "pr5_transition_windows.json"
    OUT_CSV = RESULTS_DIR / "pr5_transition_windows.csv"

CFG_NAME = "main"
STATE_RANGES = {
    "baseline": SEIZURE_PROXIMITY_CONFIGS[CFG_NAME]["baseline_hours"],
    "pre": SEIZURE_PROXIMITY_CONFIGS[CFG_NAME]["pre_ictal_hours"],
    "post": SEIZURE_PROXIMITY_CONFIGS[CFG_NAME]["post_ictal_hours"],
}


def _intersect_intervals(
    win: Tuple[float, float],
    blocks: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Return list of (lo, hi) covered intervals inside the window."""
    out = []
    w0, w1 = win
    for b0, b1 in blocks:
        a = max(w0, float(b0))
        b = min(w1, float(b1))
        if b > a:
            out.append((a, b))
    return out


def _accumulate_pairs(
    times: np.ndarray,
    labels: np.ndarray,
    intervals: List[Tuple[float, float]],
) -> Tuple[int, int, int, int]:
    """Return (n_pairs, n_opposite, n0, n1) for events inside any interval.

    Consecutive event pairs are counted only if BOTH events fall in the
    SAME interval (i.e., no gap straddling).
    """
    if times.size == 0 or not intervals:
        return 0, 0, 0, 0

    # Assign each event to interval index (or -1).
    assign = np.full(times.size, -1, dtype=int)
    for k, (lo, hi) in enumerate(intervals):
        m = (times >= lo) & (times < hi)
        assign[m] = k

    in_mask = assign >= 0
    if not in_mask.any():
        return 0, 0, 0, 0

    sub_idx = np.where(in_mask)[0]
    sub_times = times[sub_idx]
    sub_labels = labels[sub_idx]
    sub_assign = assign[sub_idx]
    sort_idx = np.argsort(sub_times, kind="stable")
    sub_times = sub_times[sort_idx]
    sub_labels = sub_labels[sort_idx]
    sub_assign = sub_assign[sort_idx]

    if sub_times.size < 2:
        return 0, 0, int(np.sum(sub_labels == 0)), int(np.sum(sub_labels == 1))

    same_interval = sub_assign[:-1] == sub_assign[1:]
    opp = sub_labels[:-1] != sub_labels[1:]
    n_pairs = int(np.sum(same_interval))
    n_opp = int(np.sum(opp & same_interval))
    n0 = int(np.sum(sub_labels == 0))
    n1 = int(np.sum(sub_labels == 1))
    return n_pairs, n_opp, n0, n1


def _compute_subject(
    dataset: str,
    subject: str,
    base_pr1: Dict[str, Any],
) -> Dict[str, Any] | None:
    ac = base_pr1.get("adaptive_cluster", {})
    stable_k = int(ac.get("stable_k") or 0)
    if stable_k != 2 or "labels" not in ac:
        return {"skip_reason": f"stable_k={stable_k} or labels missing"}

    root = YUQUAN_ROOT if dataset == "yuquan" else EPILEPSIAE_ROOT
    subject_dir = _subject_dir(dataset, root, subject)
    if not _has_propagation_inputs(dataset, subject_dir):
        return {"skip_reason": "lagPat_missing"}

    loaded = load_subject_propagation_events(subject_dir)
    valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
    labels = np.asarray(ac["labels"], dtype=int)
    if valid_events.size != labels.size:
        return {"skip_reason": "label_valid_mismatch"}

    times = loaded["event_abs_times"][valid_events]
    block_ranges = loaded.get("block_time_ranges")
    if block_ranges is None or len(block_ranges) == 0:
        return {"skip_reason": "no_block_ranges"}
    blocks = [(float(lo), float(hi)) for lo, hi in block_ranges]

    seizures = load_seizure_times(subject, dataset)
    if not seizures:
        return {"skip_reason": "no_seizures"}

    # Assign dominant id by global counts (so per-subject A/B labels become
    # dominant/non-dominant; consistent with PR-5-B convention).
    counts = np.bincount(labels, minlength=2)
    dom = int(np.argmax(counts))
    # Map labels so that 0 = dominant, 1 = non-dominant for figure clarity
    relabeled = np.where(labels == dom, 0, 1).astype(int)

    pooled: Dict[str, Dict[str, int]] = {
        s: {"n_pairs": 0, "n_opp": 0, "n0": 0, "n1": 0, "n_seizures_used": 0}
        for s in STATE_RANGES
    }

    for sz_t in seizures:
        for state, (lo_h, hi_h) in STATE_RANGES.items():
            win = (sz_t + lo_h * 3600.0, sz_t + hi_h * 3600.0)
            intervals = _intersect_intervals(win, blocks)
            if not intervals:
                continue
            n_pairs, n_opp, n0, n1 = _accumulate_pairs(times, relabeled, intervals)
            if n_pairs == 0 and (n0 + n1) == 0:
                continue
            pooled[state]["n_pairs"] += n_pairs
            pooled[state]["n_opp"] += n_opp
            pooled[state]["n0"] += n0
            pooled[state]["n1"] += n1
            pooled[state]["n_seizures_used"] += 1

    out_states: Dict[str, Dict[str, Any]] = {}
    eps = 1e-12
    for state, acc in pooled.items():
        if acc["n_pairs"] == 0 or (acc["n0"] + acc["n1"]) < 4:
            out_states[state] = {
                "n_pairs": acc["n_pairs"],
                "n_opp": acc["n_opp"],
                "n0": acc["n0"],
                "n1": acc["n1"],
                "n_seizures_used": acc["n_seizures_used"],
                "p_next_opposite": float("nan"),
                "transition_odds": float("nan"),
                "baseline_odds": float("nan"),
                "lift": float("nan"),
            }
            continue
        p_opp = acc["n_opp"] / acc["n_pairs"]
        trans_odds = p_opp / max(1.0 - p_opp, eps)
        p1 = acc["n1"] / max(acc["n0"] + acc["n1"], 1)
        p_iid_opp = 2.0 * p1 * (1.0 - p1)
        base_odds = p_iid_opp / max(1.0 - p_iid_opp, eps)
        lift = trans_odds / max(base_odds, eps)
        out_states[state] = {
            "n_pairs": acc["n_pairs"],
            "n_opp": acc["n_opp"],
            "n0": acc["n0"],
            "n1": acc["n1"],
            "n_seizures_used": acc["n_seizures_used"],
            "p_next_opposite": float(p_opp),
            "transition_odds": float(trans_odds),
            "baseline_odds": float(base_odds),
            "lift": float(lift),
        }

    deltas = {}
    for a, b in [("post", "baseline"), ("pre", "baseline"), ("post", "pre")]:
        la = out_states[a].get("lift")
        lb = out_states[b].get("lift")
        if la is None or lb is None or not (np.isfinite(la) and np.isfinite(lb)):
            deltas[f"{a}_minus_{b}_lift"] = float("nan")
        else:
            deltas[f"{a}_minus_{b}_lift"] = float(la - lb)

    return {
        "subject_id": subject,
        "dataset": dataset,
        "stable_k": stable_k,
        "dom_global_id": dom,
        "n_seizures": int(len(seizures)),
        "states": out_states,
        "deltas": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--masked-features",
        action="store_true",
        help=(
            "Topic 0 Step 5e masked rerun: read PR-1 inputs and write outputs "
            "to results/interictal_propagation_masked/."
        ),
    )
    args = parser.parse_args()
    if args.masked_features:
        _apply_masked_paths()
        logger.info("Paths auto-routed for --masked-features -> %s", RESULTS_DIR)

    pr1_path = PR1_SUBJECT_SUMMARY
    if not pr1_path.exists():
        raise SystemExit(f"PR-1 subject summary not found: {pr1_path}")
    with open(pr1_path) as f:
        pr1 = json.load(f)

    targets: List[Tuple[str, str]] = []
    for key, val in pr1.items():
        sk = val.get("adaptive_cluster", {}).get("stable_k")
        if sk == 2:
            ds, sub = key.split("/", 1)
            targets.append((ds, sub))
    targets.sort()
    logger.info("Targeting %d stable_k=2 subjects", len(targets))

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for ds, sub in targets:
        base = pr1[f"{ds}/{sub}"]
        res = _compute_subject(ds, sub, base)
        if res is None:
            continue
        if "skip_reason" in res:
            skipped.append({"subject": f"{ds}/{sub}", **res})
            logger.warning("SKIP %s/%s: %s", ds, sub, res["skip_reason"])
            continue
        records.append(res)
        s = res["states"]
        logger.info(
            "OK %s/%s: pairs base/pre/post=%d/%d/%d  lift=%.3f/%.3f/%.3f",
            ds, sub,
            s["baseline"]["n_pairs"], s["pre"]["n_pairs"], s["post"]["n_pairs"],
            s["baseline"]["lift"], s["pre"]["lift"], s["post"]["lift"],
        )

    out_json = OUT_JSON
    payload = {
        "config_name": CFG_NAME,
        "state_ranges_hours": {k: list(v) for k, v in STATE_RANGES.items()},
        "n_targeted": len(targets),
        "n_computed": len(records),
        "skipped": skipped,
        "per_subject": records,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_json)

    out_csv = OUT_CSV
    fields = [
        "dataset", "subject_id", "stable_k", "dom_global_id", "n_seizures",
        "base_pairs", "base_n_opp", "base_lift",
        "pre_pairs", "pre_n_opp", "pre_lift",
        "post_pairs", "post_n_opp", "post_lift",
        "delta_post_minus_base_lift", "delta_pre_minus_base_lift",
        "delta_post_minus_pre_lift",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            s = r["states"]
            d = r["deltas"]
            w.writerow({
                "dataset": r["dataset"],
                "subject_id": r["subject_id"],
                "stable_k": r["stable_k"],
                "dom_global_id": r["dom_global_id"],
                "n_seizures": r["n_seizures"],
                "base_pairs": s["baseline"]["n_pairs"],
                "base_n_opp": s["baseline"]["n_opp"],
                "base_lift": f"{s['baseline']['lift']:.4f}",
                "pre_pairs": s["pre"]["n_pairs"],
                "pre_n_opp": s["pre"]["n_opp"],
                "pre_lift": f"{s['pre']['lift']:.4f}",
                "post_pairs": s["post"]["n_pairs"],
                "post_n_opp": s["post"]["n_opp"],
                "post_lift": f"{s['post']['lift']:.4f}",
                "delta_post_minus_base_lift": f"{d['post_minus_baseline_lift']:.4f}",
                "delta_pre_minus_base_lift": f"{d['pre_minus_baseline_lift']:.4f}",
                "delta_post_minus_pre_lift": f"{d['post_minus_pre_lift']:.4f}",
            })
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
