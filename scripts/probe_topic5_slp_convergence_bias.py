"""How much of the cohort result is an artefact of the epoch budget?

The static baseline hits the epoch ceiling far more often than the recurrent arm
(45 of 63 units against 16 of 63), and it hits it in the direction that would
flatter the recurrent arm.  A caveat is not enough here: this measures the bias
by re-fitting a sample of patients with a budget several times larger, and
reports how much of the reported advantage it accounts for.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"

ARMS = ("STATIC_CONTACT", "ORDINARY_GRU")
LONG_CONFIG = {"epochs_warmup": 0, "epochs_structure": 0,
               "epochs_freeze": 600, "patience": 25}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=[
        "epilepsiae_1146", "epilepsiae_922", "epilepsiae_253",
        "yuquan_pengzihang", "epilepsiae_620", "yuquan_chengshuai",
    ])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    config_path = args.work / "long_budget.json"
    config_path.write_text(json.dumps(LONG_CONFIG))

    from concurrent.futures import ThreadPoolExecutor

    def fit(job):
        subject, arm = job
        cell = args.work / subject / arm
        if not (cell / "DONE.json").exists():
            subprocess.run(
                [PYTHON, str(ROOT / "scripts/train_topic5_slp_unit.py"),
                 "--subject", subject, "--arm", arm, "--seed", "1",
                 "--config", str(config_path), "--out", str(cell)],
                capture_output=True, text=True,
            )
        return cell / "DONE.json"

    jobs = [(s, a) for s in args.subjects for a in ARMS]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fit, jobs))

    rows = []
    for subject in args.subjects:
        entry = {"subject": subject}
        ok = True
        for arm in ARMS:
            short_path = (OUT / "per_subject" / subject / arm / "seed1" / "DONE.json")
            long_path = args.work / subject / arm / "DONE.json"
            if not (short_path.exists() and long_path.exists()):
                ok = False
                break
            short = json.loads(short_path.read_text())
            long_run = json.loads(long_path.read_text())
            entry[arm] = {
                "budget_95_test": short["test_next_bce"],
                "budget_95_converged": short.get("converged"),
                "long_test": long_run["test_next_bce"],
                "long_epochs_run": long_run["epochs_run"],
                "long_converged": long_run.get("converged"),
                "improvement": short["test_next_bce"] - long_run["test_next_bce"],
            }
        if ok:
            entry["advantage_at_budget_95"] = (
                entry["STATIC_CONTACT"]["budget_95_test"]
                - entry["ORDINARY_GRU"]["budget_95_test"])
            entry["advantage_at_long_budget"] = (
                entry["STATIC_CONTACT"]["long_test"]
                - entry["ORDINARY_GRU"]["long_test"])
            entry["advantage_shrinkage"] = (
                entry["advantage_at_budget_95"] - entry["advantage_at_long_budget"])
            rows.append(entry)

    if not rows:
        raise SystemExit("no probe cell completed")

    short_adv = np.array([r["advantage_at_budget_95"] for r in rows])
    long_adv = np.array([r["advantage_at_long_budget"] for r in rows])
    shrink = np.array([r["advantage_shrinkage"] for r in rows])
    fraction = float(np.median(shrink) / np.median(short_adv)) if np.median(short_adv) else float("nan")

    verdict = {
        "contract": "topic5_slp_convergence_bias_probe_v0_1",
        "long_budget": LONG_CONFIG,
        "n_subjects": len(rows),
        "median_advantage_at_budget_95": float(np.median(short_adv)),
        "median_advantage_at_long_budget": float(np.median(long_adv)),
        "median_shrinkage": float(np.median(shrink)),
        "shrinkage_as_fraction_of_reported_advantage": fraction,
        "direction": ("the reported advantage survives the longer budget"
                      if np.median(long_adv) > 0 else
                      "the reported advantage does not survive the longer budget"),
        "subjects": rows,
    }
    (OUT / "convergence_bias_probe.json").write_text(json.dumps(verdict, indent=1))
    print(f"n={len(rows)}  advantage at budget 95: {np.median(short_adv):+.4f}"
          f"   at long budget: {np.median(long_adv):+.4f}"
          f"   shrinkage {np.median(shrink):+.4f} ({fraction:.1%} of the effect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
