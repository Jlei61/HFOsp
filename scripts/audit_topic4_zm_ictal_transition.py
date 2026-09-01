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


# ---------------------------------------------------------------------------
# Phase 1B gates
# ---------------------------------------------------------------------------
def _dose_rows(output_root, seeds, joint, rung):
    rows = []
    for seed in seeds:
        # `low_activity`, not `baseline`: the state the dose is calibrated on is
        # the window that passes the Z/M-off support rule, not the 2 s clock
        # reading that was called baseline before it was measured.
        path = output_root / "dose" / f"{joint}_seed_{seed}_low_activity_n{rung}.json"
        if path.exists():
            rows.extend(json.loads(path.read_text())["rows"])
    return rows


def gate_dose(config, args):
    """Smallest rung that is measurable, low-activity-safe AND still linear.

    Smallest, not largest: the largest baseline-safe rung is precisely the one
    most likely to leave the sub-event regime once the network becomes more
    excitable at the pre-ictal checkpoint, which would recycle the very
    contamination the descendant metric removed. The ratio clause is the
    linearity check -- the packet doubles between rungs, so a linear regime
    gives ~2; below 1.2 the probe is saturating, above 3.0 it sits near a
    threshold.
    """
    output_root = ROOT / config["output_root"]
    perturbation = config["perturbation"]
    ladder = list(perturbation["dose_ladder_cells"])
    seeds = config["seeds"]["canary"]
    joint = config["arms"]["Joint"]
    floor = float(perturbation["dose_minimum_median_descendant_excess_spikes"])
    ratio_lo, ratio_hi = perturbation["dose_linearity_ratio_range"]

    summary = {}
    for rung in ladder:
        rows = _dose_rows(output_root, seeds, joint, rung)
        if not rows:
            summary[rung] = {"status": "NOT_RUN"}
            continue
        events = sum(r["probe_attributable_event_200ms"] for r in rows)
        ictal = sum(r["reached_model_ictal_200ms"] for r in rows)
        summary[rung] = {
            "status": "OK", "n_units": len(rows),
            "median_descendant_susceptibility": float(np.median(
                [r["susceptibility"] for r in rows])),
            "n_probe_attributable_events": int(events),
            "n_reached_model_ictal": int(ictal),
            "baseline_safe": bool(events == 0 and ictal == 0)}

    selected, reasons = None, {}
    for index, rung in enumerate(ladder):
        row = summary[rung]
        if row.get("status") != "OK":
            reasons[rung] = "not run"; continue
        clauses = {"baseline_safe": row["baseline_safe"],
                   "measurable": row["median_descendant_susceptibility"] >= floor}
        ratio = None
        if index + 1 < len(ladder) and summary[ladder[index + 1]].get("status") == "OK":
            nxt = summary[ladder[index + 1]]["median_descendant_susceptibility"]
            cur = row["median_descendant_susceptibility"]
            ratio = float(nxt / cur) if cur > 0 else float("inf")
            clauses["linear"] = bool(ratio_lo <= ratio <= ratio_hi)
        else:
            clauses["linear"] = False
        row["response_ratio_to_next_rung"] = ratio
        row["clauses"] = clauses
        if all(clauses.values()):
            selected = rung
            break
        reasons[rung] = [k for k, v in clauses.items() if not v]

    verdict = {"gate": "dose",
               "status": "PASS" if selected else "NO_SUBEVENT_PROBE_REGIME",
               "selected_dose_cells": selected, "ladder": ladder,
               "rejection_reasons": reasons, "per_rung": summary,
               "selection_rule": perturbation["dose_selection"],
               "boundary": ("calibrated on BASELINE checkpoints only, blind to any "
                            "pre-ictal or patient-derived quantity")}
    atomic_write_json(verdict, str(output_root / "dose_freeze.json"))
    print(json.dumps({k: verdict[k] for k in
                      ("gate", "status", "selected_dose_cells", "rejection_reasons")},
                     indent=1))
    if not selected:
        raise SystemExit("NO_SUBEVENT_PROBE_REGIME -- stop the round at Phase 1. "
                         "Do not loosen the ignition criterion or drop linearity.")


def gate_repertoire(config, args):
    """Conjunctive claim gate. Not a run blocker; it governs WORDING.

    Two implementation errors were found here by review and are fixed:
      * the shaft-aware embedding must be loaded from the FROZEN SCORING
        CONTRACT (load_scoring_contract), not read off the classifier manifest,
        which has no `center` / `scale` / `components` at all;
      * natural_kmeans returns `direction_balanced_alignment`, not `labels`;
        the old code indexed a missing key inside a broad `except`, so every
        network silently scored NaN and the gate was measuring nothing.
    The broad except is gone: a failure here must be visible, not become a
    quiet "not retained".
    """
    from src.topic4_d6_natural_kmeans import natural_kmeans
    from src.topic4_nlc_pathway_mechanism import formal_mode_assignments
    from scripts.rescore_topic4_rev10_sa_historical_artifacts import (
        load_scoring_contract)

    output_root = ROOT / config["output_root"]
    gate = config["repertoire_claim_gate"]
    reference_dir = ROOT / gate["reference_workers"]
    manifest = json.loads(
        (ROOT / config["inputs"]["frozen_substrate_manifest"]["path"]).read_text())
    classifier = manifest["direction_classifier"]
    contact_names, embedding, _, _ = load_scoring_contract(
        str(ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]),
        str(ROOT / config["inputs"]["shaft_aware_floors"]["path"]),
        "FULL_TIMING", fixed_events_per_mode=6)
    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text())
    groups = {}
    for row in contract["contacts"]:
        groups.setdefault(row["shaft_id"], []).append(row["contact_index"])

    def _measure(npz_path, before_onset_only=False):
        with np.load(npz_path, allow_pickle=False) as handle:
            onsets = np.asarray(handle["onsets"], float)
            returned = np.asarray(handle["event_returned"], bool)
            if before_onset_only and "event_before_onset" in handle.files:
                returned = returned & np.asarray(handle["event_before_onset"], bool)
        if not len(onsets):
            return None
        assigned = formal_mode_assignments(
            onsets, returned, groups=groups, embedding=embedding,
            classifier=classifier)
        labels = np.asarray(assigned["labels"], int)
        clean = np.asarray(assigned["clean"], bool)
        counts = [int(np.sum(clean & (labels == mode))) for mode in (0, 1)]
        ood = (float(np.mean(np.asarray(assigned["ood"], bool)[returned]))
               if returned.any() else float("nan"))
        alignment = float("nan")
        if clean.sum() >= 6:
            km = natural_kmeans(onsets[clean], labels[clean])
            if km.get("status") == "OK":
                alignment = float(km["direction_balanced_alignment"])
        return {"n_returned": int(returned.sum()), "ood_fraction": ood,
                "mode_counts": counts, "balanced_alignment": alignment,
                "n_clean": int(clean.sum())}

    reference = [row for row in
                 (_measure(path) for path in sorted(reference_dir.glob("*.npz")))
                 if row]
    ood_q95 = float(np.nanpercentile([r["ood_fraction"] for r in reference], 95))
    aligned = [r["balanced_alignment"] for r in reference
               if np.isfinite(r["balanced_alignment"])]
    align_q05 = float(np.nanpercentile(aligned, 5)) if aligned else float("nan")
    atomic_write_json({"n_reference_runs": len(reference), "ood_q95": ood_q95,
                       "balanced_alignment_q05": align_q05,
                       "n_reference_with_alignment": len(aligned),
                       "reference_note": "48 archived Z/M-off runs, seeds 1561-1572"},
                      str(output_root / "zm_off_reference_repertoire.json"))

    networks = {}
    for arm_name, candidate in config["arms"].items():
        for seed in config["seeds"]["canary"]:
            npz = output_root / "workers" / f"{candidate}_seed_{seed}.npz"
            if not npz.exists():
                continue
            row = _measure(npz, before_onset_only=True)
            if row is None:
                continue
            clauses = {
                "n_returned_before_onset_at_least_20":
                    row["n_returned"] >= gate["minimum_returned_events_before_onset"],
                "ood_at_most_reference_q95":
                    bool(np.isfinite(row["ood_fraction"])
                         and row["ood_fraction"] <= ood_q95),
                "both_modes_supported":
                    min(row["mode_counts"]) >= gate["minimum_events_per_mode"],
                "kmeans_alignment_at_least_reference_q05":
                    bool(np.isfinite(row["balanced_alignment"])
                         and row["balanced_alignment"] >= align_q05)}
            networks[f"{arm_name}_s{seed}"] = {
                "arm": arm_name, "seed": seed,
                "retained": all(clauses.values()),
                "failing_clauses": [k for k, v in clauses.items() if not v],
                "measures": row, "clauses": clauses}

    by_arm = {}
    for key, row in networks.items():
        by_arm.setdefault(row["arm"], []).append(row["retained"])
    verdict = {
        "gate": "repertoire",
        "retained_by_arm": {k: f"{sum(v)}/{len(v)}" for k, v in by_arm.items()},
        "networks": networks,
        "thresholds": {"ood_q95": ood_q95, "alignment_q05": align_q05,
                       "min_returned": gate["minimum_returned_events_before_onset"],
                       "min_per_mode": gate["minimum_events_per_mode"]},
        "is_run_blocker": False,
        "power_caveat": (
            "an arm that fails only because it has few returned events before "
            "onset is NOT evidence that its modes were abolished. The Joint arm "
            "transitions at ~4.1 s and therefore has an order of magnitude fewer "
            "pre-onset events than the Node arm; a matched-count comparison is "
            "required before any statement about mode loss."),
        "wording": ("retained -> 'data-driven interictal modes to model ictal "
                    "state'; not retained -> 'low-activity background to "
                    "high-activity state', with every mode statement dropped"),
    }
    atomic_write_json(verdict, str(output_root / "repertoire_gate.json"))
    print(json.dumps({k: verdict[k] for k in ("gate", "retained_by_arm")}, indent=1))


def gate_recruitment(config, args):
    """Descriptive: sequential local spread versus near-simultaneous ignition."""
    output_root = ROOT / config["output_root"]
    rows = {}
    for arm_name, candidate in config["arms"].items():
        for seed in config["seeds"]["canary"]:
            path = output_root / "workers" / f"{candidate}_seed_{seed}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if payload.get("recruitment"):
                rows[f"{arm_name}_s{seed}"] = payload["recruitment"]
    verdict = {"gate": "recruitment", "n_networks": len(rows), "networks": rows,
               "reading": ("a long 10-90 % spread duration with a finite axial slope "
                           "is sequential local spread; a near-zero duration is "
                           "near-simultaneous whole-field ignition"),
               "windows": config["recruitment"]}
    atomic_write_json(verdict, str(output_root / "recruitment_audit.json"))
    print(json.dumps({"gate": "recruitment", "n_networks": len(rows)}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", required=True,
                        choices=("parity", "interictal-baseline",
                                 "cost-projection", "dose", "repertoire",
                                 "recruitment"))
    parser.add_argument("--expected-commit", default="HEAD")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    config = load_round_config(args.config)
    {"parity": gate_parity,
     "interictal-baseline": gate_interictal_baseline,
     "cost-projection": gate_cost_projection,
     "dose": gate_dose, "repertoire": gate_repertoire,
     "recruitment": gate_recruitment}[args.gate](config, args)


if __name__ == "__main__":
    main()
