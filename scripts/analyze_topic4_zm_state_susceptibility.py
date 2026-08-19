#!/usr/bin/env python
"""Does the same probe get a bigger answer just before the transition?

Scope, fixed by the round plan and not widened here: Joint arm, 3 canary
networks, 6 representative sites, one dose calibrated on the low-activity state.

This is a SCREEN, not a cohort claim. The statistical unit is the network seed,
and there are three of them. Three networks cannot support a cohort statement no
matter how large the effect looks, so what is reported is:

  * per-seed medians over the six sites (n = 3 networks), and
  * the sign of the difference at each of the 18 site x seed units, with the
    warning that the six sites inside one network are not independent draws.

The decision this feeds is narrow: escalate to the 7 x 7 grid, or not.

TWO endpoints, because either one alone can miss the difference:

  graded    the descendant-only response on units that stayed sub-event at both
            states. This is the intended measurement.
  ignition  how many units the SAME dose ignited at each state. The dose is
            calibrated to be sub-event at the low-activity state, so if it
            ignites at the pre-ictal state that IS the state difference -- the
            most extreme form of it. Excluding those units and reporting only
            the graded endpoint would turn the largest possible effect into
            "no comparable units", a false negative built into the design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_ictal_transition import load_round_config  # noqa: E402

MEASURES = ("susceptibility", "excess_spikes_early", "excess_spikes_late",
            "r90_mm", "contact_excess_energy")


def _rows(path):
    payload = json.loads(Path(path).read_text())
    return {r["site_id"]: r for r in payload["rows"]}, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dose", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--state-label", default="")
    args = ap.parse_args()

    config = load_round_config(args.config)
    root = ROOT / config["output_root"] / "perturbation"
    joint = config["arms"]["Joint"]

    per_seed, missing, units = {}, [], []
    for seed in args.seeds:
        low = root / f"{joint}_seed_{seed}_low_activity_n{args.dose}.json"
        pre = root / f"{joint}_seed_{seed}_pre_ictal_n{args.dose}.json"
        if not (low.exists() and pre.exists()):
            missing.append({"seed": seed, "low_activity": low.exists(),
                            "pre_ictal": pre.exists()})
            continue
        low_rows, low_payload = _rows(low)
        pre_rows, pre_payload = _rows(pre)
        shared = sorted(set(low_rows) & set(pre_rows))
        if not shared:
            missing.append({"seed": seed, "reason": "no site survives at both states"})
            continue
        sites = {}
        for site in shared:
            entry = {}
            for measure in MEASURES:
                a, b = low_rows[site].get(measure), pre_rows[site].get(measure)
                entry[measure] = {"low_activity": a, "pre_ictal": b,
                                  "delta": (None if a is None or b is None
                                            or not np.isfinite(a) or not np.isfinite(b)
                                            else float(b - a))}
            # A probe that ignites the network is no longer a sub-event probe;
            # its susceptibility is not comparable to one that did not.
            entry["probe_attributable_event"] = {
                "low_activity": bool(low_rows[site]["probe_attributable_event_200ms"]),
                "pre_ictal": bool(pre_rows[site]["probe_attributable_event_200ms"])}
            entry["reached_model_ictal"] = {
                "low_activity": bool(low_rows[site]["reached_model_ictal_200ms"]),
                "pre_ictal": bool(pre_rows[site]["reached_model_ictal_200ms"])}
            entry["comparable"] = not (entry["probe_attributable_event"]["low_activity"]
                                       or entry["probe_attributable_event"]["pre_ictal"])
            sites[site] = entry
            d = entry["susceptibility"]["delta"]
            if d is not None:
                units.append({"seed": seed, "site": site, "delta": d,
                              "comparable": entry["comparable"]})
        # Comparable sites ONLY. A site where the probe ignited the network at
        # the pre-ictal state has a response two orders of magnitude larger than
        # a sub-event response; leaving it in the median would manufacture a
        # state difference out of the very contamination the exclusion exists
        # to remove.
        deltas = [sites[s]["susceptibility"]["delta"] for s in shared
                  if sites[s]["comparable"]
                  and sites[s]["susceptibility"]["delta"] is not None]
        all_deltas = [sites[s]["susceptibility"]["delta"] for s in shared
                      if sites[s]["susceptibility"]["delta"] is not None]
        per_seed[seed] = {
            "n_sites": len(shared), "n_comparable_sites": len(deltas), "sites": sites,
            "median_delta_susceptibility": float(np.median(deltas)) if deltas else None,
            "median_delta_susceptibility_including_ignited": (
                float(np.median(all_deltas)) if all_deltas else None),
            "checkpoint_time_ms": {
                "low_activity": low_payload.get("checkpoint_absolute_time_ms"),
                "pre_ictal": pre_payload.get("checkpoint_absolute_time_ms")}}

    ignition = {"low_activity": 0, "pre_ictal": 0, "n_units": 0,
                "per_seed": {}}
    for seed, entry in per_seed.items():
        low_n = sum(1 for v in entry["sites"].values()
                    if v["probe_attributable_event"]["low_activity"])
        pre_n = sum(1 for v in entry["sites"].values()
                    if v["probe_attributable_event"]["pre_ictal"])
        ignition["low_activity"] += low_n
        ignition["pre_ictal"] += pre_n
        ignition["n_units"] += len(entry["sites"])
        ignition["per_seed"][seed] = {"low_activity": low_n, "pre_ictal": pre_n,
                                      "n_sites": len(entry["sites"])}
    ignition["difference"] = ignition["pre_ictal"] - ignition["low_activity"]
    ignition["every_seed_ignites_more_at_pre_ictal"] = bool(
        per_seed and all(v["pre_ictal"] > v["low_activity"]
                         for v in ignition["per_seed"].values()))

    seed_medians = [v["median_delta_susceptibility"] for v in per_seed.values()
                    if v["median_delta_susceptibility"] is not None]
    seeds_with_no_comparable_site = [k for k, v in per_seed.items()
                                     if v["median_delta_susceptibility"] is None]
    comparable = [u for u in units if u["comparable"]]
    n_pos = sum(1 for u in comparable if u["delta"] > 0)

    screen = {
        "seeds_with_both_states": len(seed_medians),
        "seed_median_deltas": seed_medians,
        "all_seed_medians_same_sign": bool(
            seed_medians and (all(v > 0 for v in seed_medians)
                              or all(v < 0 for v in seed_medians))),
        "site_units_comparable": len(comparable),
        "site_units_positive": n_pos,
        "site_units_excluded_probe_attributable": len(units) - len(comparable),
        "seeds_with_no_comparable_site": seeds_with_no_comparable_site,
    }
    graded_established = bool(screen["all_seed_medians_same_sign"]
                              and len(seed_medians) == 3)
    # Either endpoint can establish that the states differ. Requiring the graded
    # one alone would let a dose that ignites at every pre-ictal site read as
    # "no comparable units, nothing found".
    ignition_established = bool(ignition["every_seed_ignites_more_at_pre_ictal"]
                                and len(per_seed) == 3)
    established = graded_established or ignition_established

    report = {
        "status": "ZM_STATE_SUSCEPTIBILITY_SCREEN",
        "scope": "Joint arm, 3 canary networks, 6 representative sites, 1 dose",
        "dose_cells": args.dose,
        "state_label": args.state_label,
        "endpoint": "delta_S(x) = S_pre_ictal(x) - S_low_activity(x), descendant-only",
        "screen": screen,
        "ignition_endpoint": ignition,
        "established_by": {"graded": graded_established,
                           "ignition": ignition_established},
        "state_difference_established_for_escalation": established,
        "escalate_to_grid": established,
        "per_seed": per_seed,
        "missing": missing,
        "claim_boundary": [
            "n = 3 networks. This is a screen for whether a state difference exists "
            "at all, not a cohort claim, and the tier does not change with the size "
            "of the effect.",
            "The six sites inside one network share that network's trajectory and are "
            "not independent draws; the 18 site units are descriptive only.",
            "Sites where the probe itself ignited the network are excluded from the "
            "GRADED comparison rather than counted, because their response is no "
            "longer a sub-event response. The per-seed median is over comparable "
            "sites only; `median_delta_susceptibility_including_ignited` is reported "
            "alongside so the size of that exclusion is visible rather than hidden.",
            "Those same excluded units are the IGNITION endpoint. The dose is "
            "calibrated to be sub-event at the low-activity state, so igniting at "
            "the pre-ictal state is itself the state difference. Escalation follows "
            "either endpoint; `established_by` records which one fired.",
        ],
    }
    out = ROOT / config["output_root"] / "state_susceptibility_screen.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("screen", "ignition_endpoint", "established_by",
                       "state_difference_established_for_escalation",
                       "dose_cells", "state_label", "missing")}, indent=2))


if __name__ == "__main__":
    main()
