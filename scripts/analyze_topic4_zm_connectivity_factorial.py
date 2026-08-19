#!/usr/bin/env python
"""The 2x2 connectivity factorial, on twelve paired network seeds.

The four arms differ by REDISTRIBUTION under a conserved per-target incoming
budget, not by adding connections: every E cell receives the same total
excitatory and inhibitory weight in every arm (measured: 3600.000 and 2426.760
for all 32000 cells, edge cells included). What changes is where each cell's
OUTGOING weight goes.

Endpoint is the restricted ictal-free time. A run that never transitions within
the cap is CENSORED at the cap, never deleted -- deleting it would compare the
arms on the subset that happened to enter, which is the subset the arms are
supposed to differ on.

The three canary seeds are excluded by construction: they chose the work point
and the figure seed, so reusing them would test the arms on the data that
selected them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_ictal_transition import load_round_config  # noqa: E402
from src.topic4_zm_statistics import factorial_contrasts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    config = load_round_config(args.config)
    output_root = ROOT / config["output_root"]
    workers = output_root / "workers"
    arms = config["arms"]
    seeds = list(config["seeds"]["formal"])
    cap_ms = float(config["simulation"]["duration_ms"])

    arm_values, missing = {}, []
    for arm_name in ("Node", "Node+EE", "Node+EtoI", "Joint"):
        candidate = arms[arm_name]
        values = {}
        for seed in seeds:
            path = workers / f"{candidate}_seed_{seed}.json"
            if not path.exists():
                missing.append({"arm": arm_name, "seed": seed})
                continue
            payload = json.loads(path.read_text())
            values[seed] = payload["run"]["model_ictal_onset_ms"]
        arm_values[arm_name] = values

    complete = sorted(set.intersection(*(set(v) for v in arm_values.values()))) \
        if all(arm_values.values()) else []
    report = {"status": "ZM_CONNECTIVITY_FACTORIAL",
              "seeds_requested": seeds,
              "seeds_complete_in_all_four_arms": complete,
              "missing": missing,
              "canary_seeds_excluded": list(config["seeds"]["canary"]),
              "arms_differ_by": ("redistribution of OUTGOING weight under a conserved "
                                 "per-target incoming budget, not by adding connections")}

    if len(complete) < 4:
        report["verdict"] = "NOT_EVALUABLE"
        report["reason"] = (f"only {len(complete)} seeds have all four arms; a paired "
                            "factorial needs the same networks in every arm")
    else:
        paired = {name: {s: arm_values[name][s] for s in complete}
                  for name in arm_values}
        report["factorial"] = factorial_contrasts(
            paired, cap_ms=cap_ms,
            draws=int(config["statistics"]["bootstrap_draws"]),
            seed=int(config["statistics"]["bootstrap_seed"]))
        report["verdict"] = "EVALUATED"

    report["claim_boundary"] = [
        "The endpoint is time to a model state defined by this round's operational "
        "rule (20 ms EMA of the population E rate >= 120 Hz sustained >= 100 ms). It "
        "is not a clinical seizure latency.",
        "Censored runs are held at the cap, so every contrast involving an arm with "
        "censoring is a bound, not a point estimate. Read entered_fraction with it.",
        "The three canary seeds are not in this analysis. Any number quoted from the "
        "canary runs is exploratory and does not enter the factorial.",
    ]
    out = Path(args.out) if args.out else output_root / "connectivity_factorial.json"
    out.write_text(json.dumps(report, indent=2))
    summary = {k: report[k] for k in ("verdict", "seeds_complete_in_all_four_arms")}
    if "factorial" in report:
        summary["per_arm"] = report["factorial"]["per_arm"]
        summary["contrasts"] = {
            k: {"mean_ms": v["mean_ms"],
                "bootstrap_q05_q95": [v["bootstrap"].get("q05"), v["bootstrap"].get("q95")],
                "p_sign_flip": v["sign_flip"].get("p_two_sided"),
                "n_positive": v["sign_flip"].get("n_positive")}
            for k, v in report["factorial"]["contrasts"].items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
