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

# The load-bearing negative is the tissue field against an unconstrained
# recurrent model, so the budget probe has to cover that pair, not only the
# static baseline it was first written for.
ARMS = ("STATIC_CONTACT", "ORDINARY_GRU", "LATENT_LEARNED_SPATIAL_RNN")
CONTRASTS = (
    ("ORDINARY_GRU", "STATIC_CONTACT"),
    ("LATENT_LEARNED_SPATIAL_RNN", "ORDINARY_GRU"),
    ("LATENT_LEARNED_SPATIAL_RNN", "STATIC_CONTACT"),
)
LONG_CONFIG = {"epochs_warmup": 10, "epochs_structure": 25,
               "epochs_freeze": 400, "patience": 25}


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
            # Each contrast is written so a positive number means the first arm
            # beats the second, on a loss.
            for better, worse in CONTRASTS:
                key = f"{better}__over__{worse}"
                entry[key] = {
                    "at_budget_95": entry[worse]["budget_95_test"]
                                    - entry[better]["budget_95_test"],
                    "at_long_budget": entry[worse]["long_test"]
                                      - entry[better]["long_test"],
                }
                entry[key]["shrinkage"] = (
                    entry[key]["at_budget_95"] - entry[key]["at_long_budget"])
            rows.append(entry)

    if not rows:
        raise SystemExit("no probe cell completed")

    contrasts = {}
    for better, worse in CONTRASTS:
        key = f"{better}__over__{worse}"
        short_adv = np.array([r[key]["at_budget_95"] for r in rows])
        long_adv = np.array([r[key]["at_long_budget"] for r in rows])
        shrink = np.array([r[key]["shrinkage"] for r in rows])
        denominator = np.median(short_adv)
        contrasts[key] = {
            "median_at_budget_95": float(np.median(short_adv)),
            "median_at_long_budget": float(np.median(long_adv)),
            "median_shrinkage": float(np.median(shrink)),
            "shrinkage_as_fraction": (float(np.median(shrink) / denominator)
                                      if abs(denominator) > 1e-9 else float("nan")),
            "sign_survives_longer_budget": bool(
                np.sign(np.median(long_adv)) == np.sign(np.median(short_adv))),
        }

    verdict = {
        "contract": "topic5_slp_convergence_bias_probe_v0_2",
        "long_budget": LONG_CONFIG,
        "n_subjects": len(rows),
        "means": ("a positive number means the first arm beats the second; the "
                  "question is whether the epoch ceiling, not the model, produced "
                  "the reported gap"),
        "contrasts": contrasts,
        "subjects": rows,
    }
    (OUT / "convergence_bias_probe.json").write_text(json.dumps(verdict, indent=1))
    print(f"n={len(rows)} patients, long budget {LONG_CONFIG['epochs_freeze']} epochs\n")
    for key, block in contrasts.items():
        print(f"{key:52s} budget95={block['median_at_budget_95']:+.4f}  "
              f"long={block['median_at_long_budget']:+.4f}  "
              f"shrink={block['median_shrinkage']:+.4f}  "
              f"sign survives={block['sign_survives_longer_budget']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
