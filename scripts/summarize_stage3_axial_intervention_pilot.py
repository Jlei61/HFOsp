"""Summarize the Stage 3 axial-intervention pilot: one row per (arm, seed), JSON + CSV only.

Reads the per-arm JSON files written by run_stage3_axial_intervention_probe.py and reduces each to a
compact row (medians + counts + fail-guard fields). Emits NO figures. The cross-arm pilot verdict
(spec §7) is read off these rows by a human; this script does not judge.
"""
import os
import csv
import json
import glob
import argparse

import numpy as np


def _median(events, key):
    vals = [e[key] for e in events if e.get(key) is not None]
    return round(float(np.median(vals)), 4) if vals else None


def summarize_file(data):
    """Reduce one per-arm probe JSON to a summary row (pure). Uses the EXCLUDED-target-contact
    instrument metric (not the raw one) so a read-out-exclusion artifact cannot be hidden."""
    events = data.get("events", [])
    cfg = data.get("config", {}) or {}
    sel = data.get("selected_baseline_event") or {}
    selr = data.get("selected_replay_event") or {}
    return dict(
        arm=data.get("arm"),
        seed=cfg.get("seed", data.get("seed")),
        n_returned=data.get("n_returned"),
        n_neg=data.get("n_neg"), n_pos=data.get("n_pos"),
        n_collision=data.get("n_collision"), n_none=data.get("n_none"),
        collision_rate=data.get("collision_rate"),
        median_oracle_far_ratio=_median(events, "oracle_far_ratio"),
        median_oracle_reach_mm=_median(events, "oracle_reach_mm"),
        median_instr_far_ratio_excl_target_contacts=_median(events, "instr_far_ratio_excl_target_contacts"),
        pre_intervention_parity=data.get("pre_intervention_parity"),
        selected_event_id=sel.get("event_id"),
        selected_source=sel.get("core_source_raw"),
        selected_baseline_far_ratio=sel.get("oracle_far_ratio"),
        selected_replay_far_ratio=selr.get("oracle_far_ratio"),
        selected_baseline_reach_mm=sel.get("oracle_reach_mm"),
        selected_replay_reach_mm=selr.get("oracle_reach_mm"),
    )


def summarize_dir(input_dir):
    """Summarize every probe JSON in input_dir (skips files whose name starts with pilot_summary).
    Returns rows sorted by (arm, seed)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        if os.path.basename(path).startswith("pilot_summary"):
            continue
        try:
            data = json.load(open(path))
        except (ValueError, OSError):
            continue
        if "arm" not in data:
            continue
        rows.append(summarize_file(data))
    return sorted(rows, key=lambda r: (str(r.get("arm")), r.get("seed") if r.get("seed") is not None else -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    rows = summarize_dir(a.input_dir)
    json.dump(rows, open(a.out_prefix + ".json", "w"), indent=2)
    if rows:
        fields = list(rows[0].keys())
        with open(a.out_prefix + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"[summary] {len(rows)} rows -> {a.out_prefix}.json / .csv", flush=True)
    for r in rows:
        print(f"  {r['arm']:<16} s{r['seed']}  far={r['median_oracle_far_ratio']} "
              f"reach={r['median_oracle_reach_mm']} coll={r['collision_rate']} "
              f"parity={r['pre_intervention_parity']} sel_far(base->rep)="
              f"{r['selected_baseline_far_ratio']}->{r['selected_replay_far_ratio']}", flush=True)


if __name__ == "__main__":
    main()
