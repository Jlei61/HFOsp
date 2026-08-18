#!/usr/bin/env python3
"""Gates for the Z/M ictal-transition round. Every gate fails loudly."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_ictal_transition import load_round_config  # noqa: E402

PARITY_KEYS = ("onsets", "ranks", "event_t_on_ms", "event_t_off_ms", "event_returned",
               "active_fraction", "contact_envelope", "h", "h_I_for_edge",
               "positions_E", "contact_xy_mm", "delta_vtheta", "edge_coefficients")
ARCHIVE = ROOT / ("results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc"
                  "/frozen_substrate_confirmation/workers")


def gate_parity(config, args):
    """Gate A: with Z/M off, checkpoints off and no post-record tail, the new
    rebuild path must reproduce the archived rev11-NLC run bit-for-bit.

    This is an ENGINEERING parity audit. It does not conflict with every formal
    arm running Z/M on -- it proves the substrate the formal arms sit on is the
    frozen one.
    """
    seed = int(config["seeds"]["parity"])
    output_root = ROOT / config["output_root"]
    stem = f"joint_04_control_seed_{seed}_zmoff"
    produced_npz = output_root / "workers" / f"{stem}.npz"
    produced_json = output_root / "workers" / f"{stem}.json"

    if not produced_npz.exists() or args.rerun:
        patched = json.loads(json.dumps(config))
        patched["simulation"]["post_runaway_record_ms"] = 0.0
        temp = output_root / "gate_a_config.json"
        temp.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(patched, str(temp))
        command = [sys.executable,
                   str(ROOT / "scripts/run_topic4_zm_ictal_transition_worker.py"),
                   "--config", str(temp), "--candidate-id", "joint_04_control",
                   "--seed", str(seed), "--zm-mode", "off",
                   "--expected-commit", args.expected_commit]
        subprocess.run(command, cwd=ROOT, check=True)

    mismatched, compared = [], {}
    with np.load(produced_npz, allow_pickle=False) as new, \
            np.load(ARCHIVE / f"joint_04_control_seed_{seed}.npz", allow_pickle=False) as old:
        for key in PARITY_KEYS:
            if key not in old.files:
                continue
            # NaN-aware for float arrays: `onsets`/`ranks` carry NaN sentinels
            # for non-recruited contacts, and NaN != NaN would report a
            # difference where the NaN masks and every finite value agree. This
            # is bit-identity for a sentinel-bearing float array, not a
            # relaxation -- integer/bool/string arrays keep strict equality.
            equal_nan = np.issubdtype(new[key].dtype, np.floating)
            same = (new[key].shape == old[key].shape
                    and np.array_equal(new[key], old[key], equal_nan=equal_nan))
            compared[key] = {"identical": bool(same), "shape": list(new[key].shape)}
            if not same:
                first = None
                if new[key].shape == old[key].shape:
                    diff = np.flatnonzero(np.asarray(new[key] != old[key]).ravel())
                    first = int(diff[0]) if diff.size else None
                compared[key]["first_differing_flat_index"] = first
                compared[key]["nan_masks_identical"] = bool(
                    equal_nan and np.array_equal(np.isnan(new[key]), np.isnan(old[key])))
                mismatched.append(key)

    archived = json.loads((ARCHIVE / f"joint_04_control_seed_{seed}.json").read_text())
    produced = json.loads(produced_json.read_text())
    counts = {
        "n_detected_events": {"new": produced["run"]["n_detected_events"],
                              "archived": archived["run"]["n_common_detector_events"]},
        "n_returned_events": {"new": produced["run"]["n_returned_events"],
                              "archived": archived["run"]["n_returned_events"]},
        "runaway": {"new": produced["run"]["model_ictal_onset_ms"],
                    "archived": archived["run"]["runaway_early_stop_ms"]}}
    for name, pair in counts.items():
        if pair["new"] != pair["archived"]:
            mismatched.append(name)

    verdict = {"gate": "parity", "status": "PASS" if not mismatched else "FAIL",
               "seed": seed, "mismatched_keys": mismatched,
               "compared": compared, "counts": counts,
               "boundary": ("engineering parity audit with Z/M off; the formal arms all "
                            "run Z/M on")}
    atomic_write_json(verdict, str(ROOT / config["output_root"] / "gate_a_parity.json"))
    print(json.dumps({k: verdict[k] for k in ("gate", "status", "mismatched_keys", "counts")},
                     indent=1))
    if mismatched:
        raise SystemExit("Gate A FAILED -- stop the round; do not relax array_equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", required=True, choices=("parity",))
    parser.add_argument("--expected-commit", default="HEAD")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    config = load_round_config(args.config)
    {"parity": gate_parity}[args.gate](config, args)


if __name__ == "__main__":
    main()
