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


# ---------------------------------------------------------------------------
# Phase 1A: the round's ONE new science gate
# ---------------------------------------------------------------------------
def _ema_rate(active_fraction, bin_ms, n_e, tau_ms=20.0):
    """Active fraction -> 20 ms-EMA per-neuron rate in Hz, the engine's units."""
    active = np.asarray(active_fraction, float)
    rate = active * n_e / (float(bin_ms) * 1e-3) / n_e
    alpha = 1.0 - np.exp(-float(bin_ms) / tau_ms)
    out = np.empty_like(rate)
    value = 0.0
    for index, sample in enumerate(rate):
        value += alpha * (sample - value)
        out[index] = value
    return out


def _baseline_statistic(npz_path, window_ms, n_e):
    with np.load(npz_path, allow_pickle=False) as handle:
        active = np.asarray(handle["active_fraction"], float)
        bin_ms = float(handle["active_fraction_bin_ms"])
    ema = _ema_rate(active, bin_ms, n_e)
    lo, hi = (int(round(v / bin_ms)) for v in window_ms)
    segment = ema[lo:hi]
    return float(np.median(segment)) if segment.size else float("nan")


def gate_interictal_baseline(config, args):
    """Three clauses, ALL evaluated and ALL reported -- not only the first.

    Continue when >= 2 of 3 canary networks pass. One network failing is a draw
    of the seed; the formal phase excludes such a network individually. Fewer
    than 2 means the WORK POINT has no interpretable interictal residence
    segment, and the baseline checkpoint is never moved earlier to rescue it.
    """
    output_root = ROOT / config["output_root"]
    workers = output_root / "workers"
    gate = config["interictal_baseline_gate"]
    window = tuple(float(v) for v in gate["baseline_window_ms"])
    joint = config["arms"]["Joint"]

    reference = []
    for seed in config["seeds"]["canary"]:
        path = workers / f"{joint}_seed_{seed}_zmoff.npz"
        if path.exists():
            payload = json.loads((workers / f"{joint}_seed_{seed}_zmoff.json").read_text())
            reference.append(_baseline_statistic(path, window,
                                                 payload["network"]["n_E"]))
    if not reference:
        raise SystemExit("no same-seed Z/M-off canary runs to calibrate against")
    ceiling = float(np.percentile(reference, gate["baseline_rate_percentile"]))
    atomic_write_json({"same_seed_zm_off_medians_hz": reference,
                       "percentile": gate["baseline_rate_percentile"],
                       "ceiling_hz": ceiling, "window_ms": list(window),
                       "n_reference_runs": len(reference)},
                      str(output_root / "zm_off_reference_baseline.json"))

    networks, n_pass = {}, 0
    for seed in config["seeds"]["canary"]:
        path = workers / f"{joint}_seed_{seed}.json"
        if not path.exists():
            networks[str(seed)] = {"status": "MISSING"}
            continue
        payload = json.loads(path.read_text())
        onset = payload["run"]["model_ictal_onset_ms"]
        statistic = _baseline_statistic(workers / f"{joint}_seed_{seed}.npz",
                                        window, payload["network"]["n_E"])
        clauses = {
            "onset_at_least_2500ms": {
                "value": onset,
                "pass": onset is not None and onset >= gate["minimum_onset_ms"]},
            "at_least_3_returned_events_before_onset": {
                "value": payload["run"]["n_returned_events_before_onset"],
                "pass": (payload["run"]["n_returned_events_before_onset"]
                         >= gate["minimum_returned_events_before_onset"])},
            "baseline_window_not_already_elevated": {
                "value": statistic, "ceiling": ceiling,
                "pass": bool(statistic <= ceiling)}}
        passed = all(c["pass"] for c in clauses.values())
        n_pass += int(passed)
        networks[str(seed)] = {"status": "PASS" if passed else "FAIL",
                               "failing_clauses": [k for k, c in clauses.items()
                                                   if not c["pass"]],
                               "clauses": clauses}

    minimum = int(gate["minimum_passing_canary_networks"])
    verdict = {"gate": "interictal-baseline",
               "status": "PASS" if n_pass >= minimum else "FAIL",
               "n_pass": n_pass, "n_required": minimum,
               "networks": networks,
               "reference": "same-seed Z/M-off canary runs",
               "boundary": ("failing means this WORK POINT has no interpretable "
                            "interictal residence segment; the baseline checkpoint "
                            "is not moved earlier to rescue it")}
    atomic_write_json(verdict, str(output_root / "interictal_baseline_gate.json"))
    print(json.dumps({k: verdict[k] for k in ("gate", "status", "n_pass", "n_required")},
                     indent=1))
    for seed, row in networks.items():
        print(f"  seed {seed}: {row['status']}"
              + (f"  failing: {row.get('failing_clauses')}" if row.get("failing_clauses") else ""))
    if verdict["status"] != "PASS":
        raise SystemExit("interictal-baseline gate FAILED -- stop and report")


def gate_cost_projection(config, args):
    """Recompute the schedule from the OBSERVED onsets before Phase 2 launches."""
    output_root = ROOT / config["output_root"]
    workers = output_root / "workers"
    joint = config["arms"]["Joint"]
    onsets, walls = [], []
    for seed in config["seeds"]["canary"]:
        path = workers / f"{joint}_seed_{seed}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload["run"]["model_ictal_onset_ms"] is not None:
            onsets.append(float(payload["run"]["model_ictal_onset_ms"]))
        walls.append(float(payload["simulation"]["wall_seconds_total"]))
    if not onsets:
        raise SystemExit("no transitioned canary network to project from")
    median_onset_s = float(np.median(onsets)) / 1000.0
    seconds_per_sim_second = float(np.median(walls)) / max(median_onset_s, 1e-9)
    n_formal = len(config["seeds"]["formal"])
    grid_sites = int(config["perturbation"]["grid_n"]) ** 2
    pool = int(config["execution"]["max_workers_full_run"])
    probe_pool = int(config["execution"]["max_workers_probe"])
    probe_s = 0.2 * seconds_per_sim_second + 8.0
    projection = {
        "median_onset_ms": float(np.median(onsets)),
        "observed_onsets_ms": onsets,
        "median_full_run_wall_s": float(np.median(walls)),
        "seconds_per_simulated_second": seconds_per_sim_second,
        "phase2_runs_h": 2 * n_formal * np.median(walls) / 3600.0 / pool,
        "phase2_grid_h": (2 * n_formal * 2 * grid_sites * probe_s) / 3600.0 / probe_pool,
        "note": "phase2 covers Joint AND Node; grid is both labels on both arms"}
    projection["phase2_total_h"] = projection["phase2_runs_h"] + projection["phase2_grid_h"]
    atomic_write_json(projection, str(output_root / "cost_projection.json"))
    print(json.dumps(projection, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", required=True,
                        choices=("parity", "interictal-baseline",
                                 "cost-projection"))
    parser.add_argument("--expected-commit", default="HEAD")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    config = load_round_config(args.config)
    {"parity": gate_parity,
     "interictal-baseline": gate_interictal_baseline,
     "cost-projection": gate_cost_projection}[args.gate](config, args)


if __name__ == "__main__":
    main()
