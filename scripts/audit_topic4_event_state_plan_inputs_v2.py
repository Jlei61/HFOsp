#!/usr/bin/env python3
"""Read completed development artifacts to support the event-state v2 plan.

No simulator, optimizer, evaluator.metrics(), or new patient partition scoring.
The serialized evaluator is trusted local project output; only stored FIT and
historical discovery metadata are inspected after loading.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def quantiles(values):
    return dict(zip(("p10", "p50", "p90"), np.percentile(values, [10, 50, 90]).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-worktree", type=Path, default=Path(
        "/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix"))
    parser.add_argument("--rnn-worktree", type=Path, default=Path(
        "/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.source_worktree / "results/topic4_sef_hfo/initial_state_conditioned_propagation_v1"
    inputs = {}

    def register(path):
        path = Path(path)
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    primary = json.loads(register(result_root / "primary_effect.json").read_text())["final"]
    qualification = json.loads(register(result_root / "qualification.json").read_text())
    events = read_csv(register(result_root / "screen/all_events.csv"))
    runs = read_csv(register(result_root / "screen/run_summary.csv"))
    baseline = [r for r in events if r["arm"] == "B0"]
    eligible = [r for r in baseline if r["primary_eligible"] == "True"]
    by_seed = defaultdict(list)
    for row in baseline:
        by_seed[row["dynamics_seed"]].append(row)
    durations = [float(r["qualifying_end_ms"]) - float(r["qualifying_start_ms"]) for r in eligible]
    first_primary = [min(float(r["event_time_ms"]) for r in group if r["primary_eligible"] == "True")
                     for group in by_seed.values()]
    probes = []
    for seed, group in sorted(by_seed.items()):
        for t in (2000, 5000, 8000, 11000, 14000, 17000, 20000):
            post = sorted((r for r in group if t <= float(r["qualifying_start_ms"]) < t + 2000),
                          key=lambda r: float(r["qualifying_start_ms"]))
            probes.append({
                "parent_seed": int(seed), "clock_ms": t,
                "carryover": any(float(r["qualifying_start_ms"]) < t <= float(r["qualifying_end_ms"])
                                 for r in group),
                "any_new_event": bool(post),
                "first_new_is_primary": bool(post and post[0]["primary_eligible"] == "True"),
                "first_latency_ms": float(post[0]["qualifying_start_ms"]) - t if post else None,
            })

    # Do not call evaluator.metrics(), read cache_probe, or score CAL/PROBE.
    sys.path.insert(0, str(args.source_worktree))
    evaluator_path = register(args.source_worktree / (
        "results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl"))
    with evaluator_path.open("rb") as handle:
        evaluator = pickle.load(handle)
    patient = {
        "fit_event_count": int(len(evaluator.fit)),
        "fit_block_count": int(len(evaluator.partition["FIT"])),
        "fit_mode_counts": dict(Counter(map(int, evaluator.fit_labels))),
        "selected_k": int(evaluator.k),
        "stored_historical_k_scan": [
            {k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()}
            for row in evaluator.k_scan
        ],
        "discovery_caveat": "Historical K stability used PROBE; stored metadata is not fresh validation.",
        "new_non_fit_scoring_performed": False,
    }
    oracle = {}
    for name in ("oracle_control.json", "oracle_control_b.json"):
        path = register(Path("/data/hfosp_group_event_state_v0_3_4/we_state") / name)
        data = json.loads(path.read_text())
        for subject, entry in data["subjects"].items():
            oracle[subject] = {
                "constant_inner_val": entry["adapter_only_inner_val"],
                "oracle_inner_val": entry["best_oracle_inner_val"],
                "gain": entry["adapter_only_inner_val"] - entry["best_oracle_inner_val"],
                "interpretation": data["interpretation"],
            }
    for relative in (
        "src/topic5_group_event_state/v034_spatial_state/we_decoder.py",
        "src/topic5_group_event_state/v035/stepwise_decoder.py",
        "docs/archive/topic5/group_event_state_v0_3_4_we_decoder_state_first_run_2026-09-03.md",
        "docs/archive/topic5/group_event_state_v0_3_7_repaired_closeout_technical_2026-09-05.md",
        "docs/archive/topic5/group_event_state_v0_3_12_independent_scientific_review_2026-09-07.md",
    ):
        register(args.rnn_worktree / relative)
    register("/data/hfosp_rnn_v038_contact_bridge_20260905/contact_bridge_report_zh.md")
    register(args.source_worktree / "src/snn_engine/params.py")

    audit = {
        "schema": "topic4_event_state_plan_input_audit_v2", "date": "2026-09-08",
        "scope": "read-only completed development results and analytical time-scale calculations",
        "new_SNN_simulations": 0, "new_RNN_training": 0,
        "v1": {
            "n_runs": len(runs), "physical_status_counts": dict(Counter(r["physical_status"] for r in runs)),
            "n_detected_windows": len(events),
            "screen": primary["screen"], "external_input_crosscheck": primary["crosscheck"],
            "qualification_status": qualification.get("status"),
            "qualification_checks_passed": sum(qualification["checks"].values()),
            "qualification_checks_total": len(qualification["checks"]),
            "verdict_caveat": "Reported small-effect verdict coexists with unstable=true; original plan gave instability precedence.",
            "all_arm_mode_outside_patient_band": not any(
                primary["quality_conclusion"]["screen"]["run_mean_inside_patient_matched_band"].values()),
        },
        "baseline_time_support": {
            "n_primary_events": len(eligible), "n_parent_runs": len(by_seed),
            "qualifying_duration_ms": quantiles(durations),
            "first_primary_after_cold_reset_ms": quantiles(first_primary),
            "fixed_clock_probes": {
                "interpretation": "Descriptive overlapping parent support, not independent initialization trials; not the v2 null rate.",
                "n_windows": len(probes),
                "carryover_fraction": float(np.mean([r["carryover"] for r in probes])),
                "any_new_event_fraction": float(np.mean([r["any_new_event"] for r in probes])),
                "first_new_event_primary_fraction": float(np.mean([r["first_new_is_primary"] for r in probes])),
                "first_latency_ms": quantiles([r["first_latency_ms"] for r in probes if r["first_latency_ms"] is not None]),
                "windows": probes,
            },
        },
        "analytical_checks_not_patient_time_constant_estimates": {
            "uncoupled_voltage_remaining_at_500ms_tau20ms": float(np.exp(-500 / 20)),
            "proposed_s_remaining_over_median_event_tau1000ms": float(np.exp(-np.median(durations) / 1000)),
        },
        "patient": patient,
        "rnn_v034_leaked_oracle_diagnostic": oracle,
        "source_sha256": inputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(runs), "patient": patient,
                      "baseline_duration_ms": quantiles(durations), "oracle": oracle}, ensure_ascii=False))


if __name__ == "__main__":
    main()
