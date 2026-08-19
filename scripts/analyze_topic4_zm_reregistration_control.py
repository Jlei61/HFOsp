#!/usr/bin/env python
"""Does the speed-up need the registration the patient's data produced?

Paired within network seed: the same network, the same Joint connectivity, with
the node field and its directed flow coefficients moved together by a D4 element
while the geometry they are registered against stays put.

What r180 does and does not control for, stated because the element matters:
an axis is undirected, so a 180 degree rotation preserves the alignment between
the field's flow and the E->E anisotropy AXIS and reverses the flow's DIRECTION
along it. r180 is therefore the tightest matched control -- identical marginal
geometry -- but it tests direction along a shared axis, not axis identity. The
registered descriptive elements r90 and mx are what move the axis, and they run
on the canary seeds only, so axis identity is NOT formally tested this round.
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
from src.topic4_zm_statistics import (paired_bootstrap,  # noqa: E402
                                      paired_sign_flip_test)


def _onset(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())["run"]["model_ictal_onset_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = load_round_config(args.config)
    output_root = ROOT / config["output_root"]
    workers = output_root / "workers"
    control = config["spatial_reregistration_control"]
    arm = config["arms"][control["arm"]]
    cap_ms = float(config["simulation"]["duration_ms"])
    element = control["formal_element"]
    seeds = list(config["seeds"]["formal"])

    paired, missing, censored = [], [], {"identity": 0, element: 0}
    for seed in seeds:
        plain = _onset(workers / f"{arm}_seed_{seed}.json")
        moved = _onset(workers / f"{arm}_seed_{seed}_ctl_{element}.json")
        if not (workers / f"{arm}_seed_{seed}.json").exists() or \
           not (workers / f"{arm}_seed_{seed}_ctl_{element}.json").exists():
            missing.append(seed)
            continue
        if plain is None:
            censored["identity"] += 1
        if moved is None:
            censored[element] += 1
        # censor, never delete: a control network that fails to transition is
        # the strongest possible evidence about the moved registration
        paired.append((seed,
                       min(float(plain), cap_ms) if plain is not None else cap_ms,
                       min(float(moved), cap_ms) if moved is not None else cap_ms))

    n_transitioned = sum(1 for _, _, m in paired if m < cap_ms)
    report = {
        "status": "ZM_REREGISTRATION_CONTROL",
        "control_name": "matched spatial re-registration control",
        "element": element, "arm": control["arm"],
        "what_r180_controls": (
            "direction of the data-derived flow along a shared undirected axis; "
            "NOT axis identity, which only the descriptive r90 / mx elements move "
            "and which is therefore not formally tested this round"),
        "n_paired": len(paired), "missing_seeds": missing,
        "n_censored": censored,
        "n_control_networks_transitioned": n_transitioned,
        "minimum_required": int(control["minimum_transitioned_control_networks"]),
        "meets_minimum": n_transitioned >= int(control["minimum_transitioned_control_networks"]),
    }

    if len(paired) >= 4:
        plain = np.array([p for _, p, _ in paired], float)
        moved = np.array([m for _, _, m in paired], float)
        delta = moved - plain
        report["per_seed_ms"] = {s: {"identity": p, element: m, "delta": float(m - p)}
                                 for s, p, m in paired}
        report["identity_restricted_mean_ms"] = float(plain.mean())
        report[f"{element}_restricted_mean_ms"] = float(moved.mean())
        report["delta"] = {
            "mean_ms": float(delta.mean()),
            "sign_meaning": (f"positive = moving the registration by {element} makes "
                             "the transition LATER, i.e. the placement the data "
                             "produced was doing work; near zero = the speed-up did "
                             "not need that placement"),
            "n_seeds_later_under_control": int((delta > 0).sum()),
            "bootstrap": paired_bootstrap(moved, plain,
                                          draws=int(config["statistics"]["bootstrap_draws"]),
                                          seed=int(config["statistics"]["bootstrap_seed"])),
            "sign_flip": paired_sign_flip_test(
                delta, draws=int(config["statistics"]["bootstrap_draws"]),
                seed=int(config["statistics"]["bootstrap_seed"]))}
        report["verdict"] = "EVALUATED"
    else:
        report["verdict"] = "NOT_EVALUABLE"
        report["reason"] = f"only {len(paired)} paired networks"

    report["claim_boundary"] = [
        "A null result here does NOT say the connectivity effect is an artefact. "
        "It says the effect does not require the flow direction the patient's data "
        "produced, which is a narrower statement.",
        "The endpoint is time to this round's operational model state, not a "
        "clinical seizure latency.",
    ]
    (output_root / "reregistration_control.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("verdict", "n_paired", "n_control_networks_transitioned",
                       "meets_minimum", "missing_seeds")}, indent=2))
    if "delta" in report:
        d = report["delta"]
        print(json.dumps({"identity_mean_ms": report["identity_restricted_mean_ms"],
                          f"{element}_mean_ms": report[f"{element}_restricted_mean_ms"],
                          "delta_mean_ms": d["mean_ms"],
                          "n_later_under_control": d["n_seeds_later_under_control"],
                          "p": d["sign_flip"].get("p_two_sided"),
                          "ci": [d["bootstrap"].get("q05"), d["bootstrap"].get("q95")]},
                         indent=2))


if __name__ == "__main__":
    main()
