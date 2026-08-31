#!/usr/bin/env python3
"""Patient-first summary + STATUS.json for Group-Event State v0.1.

Every comparison here is paired across patients with seeds collapsed inside a
patient first, and every effect is printed next to the spread that merely
changing the seed produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.analysis import (  # noqa: E402
    ENDPOINTS,
    load_runs,
    paired_comparison,
    patient_arm_table,
    seed_payload_identity,
    truncation_curve,
    wrong_time_comparison,
)
from src.topic5_group_event_state.source_audit import (  # noqa: E402
    write_csv_atomic,
    write_json_atomic,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"

CORE_ARMS = [
    "a1_static_recent_history",
    "a2_rank_group_state",
    "a3_delay_group_state",
    "a4_full_multimodal_state",
    "a5_full_plus_background",
]
# (arm_a, arm_b, what the difference is evidence about)
KEY_CONTRASTS = [
    ("a4_full_multimodal_state", "a1_static_recent_history", "state vs no state at all"),
    ("a4_full_multimodal_state", "b4_memoryless", "persistent vs same encoder, state reset every event"),
    ("a3_delay_group_state", "a2_rank_group_state", "exact delay + tied groups vs legacy integer rank"),
    ("a4_full_multimodal_state", "a3_delay_group_state", "waveform + multiband on top of exact delay"),
    ("a5_full_plus_background", "a4_full_multimodal_state", "background SEEG correction"),
    ("a4_full_multimodal_state", "b1_no_real_dt", "real elapsed seconds vs an event-count clock"),
    ("a4_full_multimodal_state", "b2_no_waveform", "waveform branch"),
    ("a4_full_multimodal_state", "b3_no_multiband", "multiband branch"),
    ("a4_full_multimodal_state", "b5_no_geometry", "static contact geometry"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=V0_1 / "runs")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--out-dir", type=Path, default=V0_1)
    args = parser.parse_args()

    runs = load_runs(args.runs_root, args.tag)
    if not runs:
        raise SystemExit(f"no runs under {args.runs_root / args.tag}")
    table = patient_arm_table(runs)

    rows = []
    for (subject, arm), entry in sorted(table.items()):
        row = {"subject": subject, "arm": arm, "dataset": entry["dataset"],
               "n_seeds": entry["n_seeds"], "n_events_test": entry["n_events_test"],
               "n_parameters": entry["n_parameters"],
               "selected_epochs": "|".join(str(e) for e in entry["selected_epoch"])}
        for endpoint in ENDPOINTS:
            row[endpoint] = entry[endpoint]
            row[f"{endpoint}__seed_spread"] = entry[f"{endpoint}__seed_spread"]
        rows.append(row)
    write_csv_atomic(rows, args.out_dir / f"patient_arm_table_{args.tag}.csv")

    contrasts = []
    for arm_a, arm_b, meaning in KEY_CONTRASTS:
        for endpoint in ENDPOINTS:
            comparison = paired_comparison(table, arm_a, arm_b, endpoint)
            if comparison.get("n_patients"):
                comparison["meaning"] = meaning
                contrasts.append(comparison)

    h1 = {
        "truncation": {
            endpoint: truncation_curve(runs, "a4_full_multimodal_state", endpoint)
            for endpoint in ("delay", "timing", "participation", "group_size")
        },
        "wrong_time": wrong_time_comparison(runs, "a4_full_multimodal_state"),
    }

    arms_present = sorted({r["arm"] for r in runs})
    subjects = sorted({r["subject"] for r in runs})
    seeds = sorted({int(r["seed"]) for r in runs})
    resource = {
        "n_runs": len(runs),
        "n_subjects": len(subjects),
        "arms": arms_present,
        "seeds": seeds,
        "peak_gpu_bytes_max": int(max(r.get("peak_gpu_bytes", 0) for r in runs)),
        "n_runs_with_oom_retry": int(sum(1 for r in runs if r.get("oom_attempts"))),
        "n_runs_with_nonfinite_steps": int(
            sum(1 for r in runs if any(h.get("n_nonfinite_steps", 0) for h in r.get("history", [])))
        ),
        "median_train_seconds": float(np.median([r["train_seconds"] for r in runs])),
        "stop_reasons": {
            reason: int(sum(1 for r in runs if r["stop_reason"] == reason))
            for reason in {r["stop_reason"] for r in runs}
        },
        "param_update_magnitude_median": {
            group: float(np.median([
                r["param_update_magnitude"][group] for r in runs
                if group in r.get("param_update_magnitude", {})
                and np.isfinite(r["param_update_magnitude"][group])
            ]))
            for group in ("encoder", "state", "heads")
        },
    }
    identity = seed_payload_identity(runs)

    payload = {
        "tag": args.tag,
        "n_runs": len(runs),
        "subjects": subjects,
        "arms": arms_present,
        "seeds": seeds,
        "seed_payload_identity": identity,
        "resource": resource,
        "contrasts": contrasts,
        "h1": h1,
    }
    write_json_atomic(payload, args.out_dir / f"summary_{args.tag}.json")

    print(f"runs={len(runs)} subjects={len(subjects)} arms={len(arms_present)} seeds={seeds}")
    print(f"duplicate-payload seed groups: {identity['n_groups_with_duplicate_payloads']}"
          f"/{identity['n_groups']}")
    print(f"median param update: {resource['param_update_magnitude_median']}")
    print()
    print(f"{'contrast':58s} {'endpoint':16s} {'n':>3s} {'A wins':>7s} {'median Δ':>10s} "
          f"{'seedspread':>10s} {'Δ/noise':>8s} {'sign p':>8s}")
    for c in contrasts:
        if c["endpoint"] not in ("delay", "timing", "participation", "group_size"):
            continue
        name = f"{c['arm_a'].split('_',1)[0]} vs {c['arm_b'].split('_',1)[0]}: {c['meaning']}"
        print(f"{name[:58]:58s} {c['endpoint']:16s} {c['n_patients']:3d} "
              f"{c['n_patients_arm_a_better']:7d} {c['median_delta']:10.5f} "
              f"{c['median_seed_spread']:10.5f} {c['effect_over_seed_noise']:8.2f} "
              f"{c['sign_test_p']:8.4f}")


if __name__ == "__main__":
    main()
