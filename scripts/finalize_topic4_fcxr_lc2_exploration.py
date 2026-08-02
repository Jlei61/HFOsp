#!/usr/bin/env python3
"""Close the bounded FCXR-LC2 closed-loop exploration without widening its claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core",
                   "closed_loop_exploration")
SCRIPT_REL = "scripts/run_topic4_fcxr_lc2_forks.py"
SCRIPT_PATH = os.path.join(ROOT, SCRIPT_REL)


def _load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def _write_json(name, payload):
    path = os.path.join(OUT, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _runtime_source_commit():
    commit = _git("log", "-1", "--format=%H", "--", SCRIPT_REL)
    blob = subprocess.check_output(["git", "show", f"{commit}:{SCRIPT_REL}"], cwd=ROOT)
    blob_hash = hashlib.sha256(blob).hexdigest()
    live_hash = _sha256(SCRIPT_PATH)
    if blob_hash != live_hash:
        raise RuntimeError("runtime fork source differs from its most recent committed blob")
    return commit, live_hash


def _verify_engine_hashes(lock):
    rows = {}
    for rel, expected in lock["engine_hashes"].items():
        got = _sha256(os.path.join(ROOT, rel))
        if got != expected:
            raise RuntimeError(f"blessed engine drift: {rel}: {got} != {expected}")
        rows[rel] = got
    return rows


def build_verdict():
    lock = _load("execution_lock.json")
    r1 = _load("r1_resegmentation_summary.json")
    screen = _load("h_loop_screen.json")
    forks = _load("frozen_fork_map.json")
    e3 = _load("E3_DONE.json")
    e4 = _load("E4_DONE.json")
    e5 = _load("dynamic_pilot_manifest.json")
    e3_watch = _load("E3_WATCHDOG.json")
    chain_watch = _load("CHAIN_WATCHDOG.json")

    if screen.get("n_rows") != 90 or sum(screen["counts"].values()) != 90:
        raise RuntimeError("E3 grid is incomplete or internally inconsistent")
    if e3.get("status") != "COMPLETE" or e3.get("counts") != screen.get("counts"):
        raise RuntimeError("E3 sentinel does not match h_loop_screen.json")
    if forks.get("n_rows") != 12 or len(forks.get("candidate_verdicts", [])) != 2:
        raise RuntimeError("E4 must contain the locked two finalists x six arms")
    if e4.get("status") != "COMPLETE" or e4.get("candidate_verdicts") != forks.get("candidate_verdicts"):
        raise RuntimeError("E4 sentinel does not match frozen_fork_map.json")
    if e5.get("status") != "NOT_UNLOCKED" or e5.get("rows"):
        raise RuntimeError("E5 state is inconsistent with a frozen-geometry negative")
    if e3_watch.get("status") != "TARGET_EXITED" or chain_watch.get("status") != "TARGET_EXITED":
        raise RuntimeError("detached execution watchdog did not observe a clean exit")

    labels = [x["label"] for x in forks["candidate_verdicts"]]
    if labels != ["healthy_baseline_disturbed", "healthy_baseline_disturbed"]:
        raise RuntimeError(f"unexpected frozen-fork outcome: {labels}")
    if any(x.get("x_return_arms") for x in forks["candidate_verdicts"]):
        raise RuntimeError("E5 must not remain locked when an X-return arm exists")

    screen_rss = max(float(x["peak_rss_gib"]) for x in screen["rows"])
    fork_rss = max(float(x["peak_rss_gib"]) for x in forks["rows"])
    numerical_failures = sum(bool(x["numerical_failure"]) for x in forks["rows"])
    if numerical_failures:
        raise RuntimeError("canonical fork set contains a numerical failure")

    low_rows = [x for x in forks["rows"] if x["arm"] == "A_low"]
    c_rows = [x for x in forks["rows"] if x["arm"] == "C"]
    d_rows = [x for x in forks["rows"] if x["arm"] in ("D1", "D2")]
    runtime_source_commit, runtime_source_hash = _runtime_source_commit()
    now = datetime.now(timezone.utc).isoformat()
    return {
        "stage": "E6",
        "status": "COMPLETE_BOUNDED_NEGATIVE",
        "verdict": "H_BOUNDED_HIGH_POSITIVE_ONSET_OFFSET_CONTROL_NEGATIVE",
        "legacy_failure_label": "H_FROZEN_GEOMETRY_NO_GO_HEALTHY_BASELINE_DISTURBED",
        "labels": [
            "SENSOR_CHARACTERIZATION_COMPLETED",
            "H_LOOP_SCREEN_COMPLETED_DEVELOPMENTAL_ONLY",
            "H_BOUNDED_HIGH_STATE_GENERATION_POSITIVE",
            "Z_SUSCEPTIBILITY_SELECTIVE_ONSET_CONTROL_NEGATIVE",
            "H_SELECTIVE_LOW_HIGH_BASIN_NOT_FOUND",
            "X_PHYSIOLOGICAL_LOAD_AMPLITUDE_CONTROL_POSITIVE",
            "X_PHYSIOLOGICAL_LOAD_OFFSET_CONTROL_NEGATIVE",
            "DYNAMIC_ZHX_NOT_UNLOCKED",
            "LIFECYCLE_NOT_TESTED",
        ],
        "component_verdicts": {
            "bounded_high_state_generation": "POSITIVE",
            "susceptibility_selective_onset": "NEGATIVE_FOR_TESTED_H_ARCHITECTURE",
            "low_high_basin_coexistence": "NOT_FOUND",
            "x_amplitude_control": "POSITIVE",
            "x_state_transition_authority": "NEGATIVE_AT_TESTED_LC1_LOADS",
            "dynamic_lifecycle": "NOT_TESTED",
        },
        "scientific_scope": {
            "allowed": (
                "Bounded high-state generation is positive. In the tested post-X local H architecture, "
                "healthy D=0 returning events can ignite the branch, so Z does not have susceptibility-"
                "selective onset control. The two accepted LC1 frozen X loads reduce high-state amplitude "
                "but do not return the tested branch to the interictal workpoint, so they do not have "
                "offset state-transition authority over this branch."
            ),
            "forbidden": [
                "H cannot ever support bistability",
                "X lacks termination authority in general",
                "a Z/H/X lifecycle was tested",
                "a seizure carrier, limit cycle, or E1146 phenotype was established",
            ],
        },
        "r1": {
            "sensor_rows": int(r1["rows_n"]),
            "pareto_rows": int(r1["pareto_n"]),
            "selected_candidates": len(r1["selected_candidates"]),
            "recruited_support_fraction": float(r1["support"]["fraction"]),
            "gap_label": r1["segmentation"]["gap"]["classification"],
            "gap_duration_ms": float(r1["segmentation"]["gap"]["raw_gap_duration_ms"]),
        },
        "e3": {
            "n_rows": int(screen["n_rows"]),
            "counts": screen["counts"],
            "interpretation": (
                "The uniform 2theta high-init probe is gate-saturated at weak k and primarily orders rho; "
                "screen_survivor is not basin evidence."
            ),
        },
        "e4": {
            "canonical_finalists": [x["candidate_run_id"] for x in forks["candidate_verdicts"]],
            "candidate_verdicts": forks["candidate_verdicts"],
            "healthy_A_low_workpoint_labels": [x["workpoint_label"] for x in low_rows],
            "healthy_A_low_tail_rate_hz": [float(x["state_tail_1s"]["rate_mean_hz"]) for x in low_rows],
            "susceptible_C_workpoint_labels": [x["workpoint_label"] for x in c_rows],
            "x_arm_workpoint_labels": [x["workpoint_label"] for x in d_rows],
            "x_arm_tail_rate_hz": [float(x["state_tail_1s"]["rate_mean_hz"]) for x in d_rows],
            "numerical_failures": numerical_failures,
        },
        "e5": {
            "status": "NOT_UNLOCKED",
            "reason": "No H_X_FROZEN_GEOMETRY_CANDIDATE survived E4.",
            "noise_402_opened": False,
        },
        "resources": {
            "screen_peak_rss_gib": screen_rss,
            "fork_peak_rss_gib": fork_rss,
            "e3_workers_max": 4,
            "e4_workers_max": 2,
            "t_ge_20s_workers": 0,
            "e3_elapsed_hours": float(e3_watch["elapsed_hours"]),
            "screen_to_forks_elapsed_hours": float(chain_watch["elapsed_hours"]),
            "max_observed_swap_delta_mib": max(
                float(e3_watch["last"]["swap_used_mib"] -
                      e3_watch["started_baseline"]["swap_used_mib"]),
                float(chain_watch["last"]["swap_used_mib"] -
                      chain_watch["started_baseline"]["swap_used_mib"]),
            ),
            "watchdog_statuses": [e3_watch["status"], chain_watch["status"]],
        },
        "provenance": {
            "design_commit": lock["design_commit"],
            "runtime_fork_source_sha256": runtime_source_hash,
            "runtime_fork_source_committed_as": runtime_source_commit,
            "engine_hashes": _verify_engine_hashes(lock),
        },
        "finished": now,
    }


def write_status(verdict):
    counts = verdict["e3"]["counts"]
    lines = [
        "# FCXR-LC2 closed-loop exploration status",
        "",
        f"- **Status**: `{verdict['status']}`",
        f"- **Canonical verdict**: `{verdict['verdict']}`",
        "- **Stage reached**: E4 frozen H/X forks; E5 dynamic Z/H/X was not unlocked.",
        "",
        "## Plain-language result",
        "",
        "R1 sensor recharacterization completed, and the full 90-cell upper-bound H screen completed. "
        f"The screen yielded {counts['screen_survivor']} developmental survivors and "
        f"{counts['saturated_tonic']} saturated-tonic cells, but this high-initial-condition screen is "
        "not a basin test. In the canonical matched forks, both finalists self-escalated from low H on "
        "the healthy D=0 substrate. The susceptible low/high starts therefore did not establish a "
        "selective bistable window. The tested H loop therefore establishes a finite bounded high state, "
        "but Z has no selective onset control. Both accepted frozen X-load levels reduce its amplitude, "
        "yet neither returns it to the interictal workpoint, so X has no offset state-transition authority "
        "at those loads.",
        "",
        "## Claim boundary",
        "",
        "The formal claim is: bounded high-state generation positive; susceptibility-selective onset and "
        "X-controlled offset negative for the tested architecture and loads. "
        "This is a bounded negative for the two locked finalists and the tested post-X H architecture. "
        "It is not a global impossibility result for H, not a reversal of LC1 X termination authority, "
        "and not a dynamic lifecycle result. M, K, A and ELR were never introduced.",
        "",
        "## Canonical artifacts",
        "",
        "- `r1_resegmentation_summary.json` / `r1_sensor_pareto.csv`",
        "- `h_loop_screen.json` (90/90)",
        "- `frozen_fork_map.json` (2 finalists x 6 matched arms)",
        "- `candidate_verdict.json`",
        "- `figures/` and Chinese `figures/README.md`",
        "",
    ]
    path = os.path.join(OUT, "STATUS.md")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-finalize", action="store_true")
    args = parser.parse_args()
    if not args.confirm_finalize:
        raise SystemExit("refusing to finalize without --confirm-finalize")
    verdict = build_verdict()
    _write_json("candidate_verdict.json", verdict)
    write_status(verdict)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
