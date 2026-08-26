#!/usr/bin/env python3
"""Freeze tau_r on the T1/R1 arms before any exposure arm is allowed to run.

Selection uses the one-standard-error rule over the declared tau grid, on
validation loss aggregated seed-wise inside each patient and then across
patients.  Exposure outcomes never enter this choice.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    FROZEN, OUTPUT_ROOT, atomic_write_json, code_revision, package_hash,
)
from src.topic5_epi_prssm.stats import aggregate_seeds  # noqa: E402

RUNS = OUTPUT_ROOT / "exposure_mechanism/runs"
TARGET = OUTPUT_ROOT / "manifests/RESOURCE_TAU_FREEZE.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()

    by_tau: dict[float, list[dict]] = {}
    for path in sorted(RUNS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") != args.cohort or not record.get("arm", "").startswith("t1_r1_tau"):
            continue
        if record.get("evaluation") is None:
            continue
        tau = float(record["spec"]["tau_r_seconds"])
        by_tau.setdefault(tau, []).append(record)
    if not by_tau:
        raise SystemExit("no completed T1/R1 tau-grid runs found; stage 4a must finish first")

    rows = []
    for tau, records in sorted(by_tau.items()):
        per_seed = [{s: v["event_nll"] + v["participation_nll"]
                     for s, v in r["evaluation"]["filtered"].items()} for r in records]
        patient = aggregate_seeds(per_seed)
        values = np.array(list(patient.values()), dtype=float)
        rows.append({"tau_r_seconds": tau, "n_seeds": len(records),
                     "n_patients": int(len(values)), "mean_validation": float(values.mean()),
                     "sem_validation": float(values.std(ddof=1) / np.sqrt(len(values)))})
    best = min(rows, key=lambda r: r["mean_validation"])
    threshold = best["mean_validation"] + best["sem_validation"]
    # one-standard-error rule: the slowest tau whose mean is inside one SE of the
    # best, so an unidentifiable grid resolves to the most conservative constant
    inside = [r for r in rows if r["mean_validation"] <= threshold]
    chosen = max(inside, key=lambda r: r["tau_r_seconds"])
    identifiable = len(inside) == 1
    atomic_write_json(TARGET, {
        "contract": "topic5_epi_prssm_v0_1_resource_tau_freeze",
        "cohort": args.cohort, "grid_seconds": list(FROZEN["resource_tau_grid_seconds"]),
        "rows": rows, "best_mean": best, "one_se_threshold": threshold,
        "tau_r_seconds": chosen["tau_r_seconds"],
        "selection_rule": "one-standard-error over the declared grid, slowest tau inside the band",
        "identifiable": identifiable,
        "identifiable_interval_seconds": [min(r["tau_r_seconds"] for r in inside),
                                          max(r["tau_r_seconds"] for r in inside)],
        "exposure_outcomes_used": False,
        "code_revision": code_revision(), "package_hash": package_hash(),
    })
    print(json.dumps({"tau_r_seconds": chosen["tau_r_seconds"], "identifiable": identifiable,
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
