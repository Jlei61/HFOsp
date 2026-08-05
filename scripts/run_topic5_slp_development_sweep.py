"""Milestone G: sequential development sweep, then freeze exactly one config.

Selection reads the development patients' validation partitions only.  Test
partitions stay sealed until the frozen config runs on the cohort.

Not a Cartesian product: microsteps are screened first on the representative
patient, then the wiring economy at the chosen microstep count, then the two
finalists go to the other development patients at two seeds.

The winner is the Pareto knee of validation loss against wiring cost, not the
lowest validation loss -- picking the lowest would select the densest graph
every time and make the whole wiring-economy question vacuous.
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
OUT_ROOT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"

REPRESENTATIVE = "epilepsiae_1146"
OTHER_DEVELOPMENT = ("epilepsiae_958", "yuquan_zhangkexuan")
ARM = "LATENT_LEARNED_SPATIAL_RNN"


def run(subject: str, config: dict, tag: str, work: Path, seed: int = 1) -> dict | None:
    cell = work / tag / subject / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    config_path = cell / "sweep_config.json"
    config_path.write_text(json.dumps(config, indent=1))
    if not (cell / "DONE.json").exists():
        result = subprocess.run(
            [PYTHON, str(ROOT / "scripts/train_topic5_slp_unit.py"),
             "--subject", subject, "--arm", ARM, "--seed", str(seed),
             "--config", str(config_path), "--out", str(cell)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  {tag} {subject} seed{seed} FAILED: {result.stdout[-200:]}")
            return None
    done = json.loads((cell / "DONE.json").read_text())
    done["tag"] = tag
    done["config"] = config
    return done


def describe(row: dict) -> str:
    return (
        f"  {row['tag']:28s} val={row['validation_next_bce']:.4f} "
        f"deg={row.get('mean_degree', float('nan')):5.1f} "
        f"wire={row.get('wiring_cost', float('nan')):8.1f} "
        f"hop={row.get('hop_reachability', float('nan')):.2f} "
        f"conv={row.get('converged')}"
    )


def pareto_knee(rows: list) -> dict:
    """Knee of validation loss against wiring cost, both min-max normalised."""
    loss = np.array([r["validation_next_bce"] for r in rows])
    wire = np.array([r.get("wiring_cost", 0.0) for r in rows])

    def unit(v):
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 0 else np.zeros_like(v)

    distance = np.hypot(unit(loss), unit(wire))
    return rows[int(np.argmin(distance))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT_ROOT / "development")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    record = {"stages": {}}

    print(f"stage 1: microsteps on {REPRESENTATIVE}", flush=True)
    stage1 = []
    for k in (1, 3, 6):
        row = run(REPRESENTATIVE, {"microsteps": k}, f"K{k}", args.work)
        if row:
            stage1.append(row)
            print(describe(row), flush=True)
    if not stage1:
        raise SystemExit("stage 1 produced no usable cell")
    best_k = int(min(stage1, key=lambda r: r["validation_next_bce"])["config"]["microsteps"])
    record["stages"]["microsteps"] = {"rows": stage1, "chosen": best_k}
    print(f"  -> microsteps = {best_k}\n", flush=True)

    print(f"stage 2: wiring economy at K={best_k} on {REPRESENTATIVE}", flush=True)
    stage2 = []
    for strength in (0.03, 0.1, 0.3):
        for budget in (4.0, 6.0):
            config = {"microsteps": best_k, "wiring_strength": strength,
                      "edge_budget": budget}
            row = run(REPRESENTATIVE, config, f"w{strength}_b{budget:.0f}", args.work)
            if row:
                stage2.append(row)
                print(describe(row), flush=True)
    if not stage2:
        raise SystemExit("stage 2 produced no usable cell")
    ranked = sorted(stage2, key=lambda r: r["validation_next_bce"])
    finalists = [pareto_knee(stage2), ranked[0]]
    if finalists[1]["tag"] == finalists[0]["tag"]:
        finalists[1] = ranked[1] if len(ranked) > 1 else ranked[0]
    record["stages"]["wiring"] = {
        "rows": stage2,
        "finalists": [f["tag"] for f in finalists],
        "knee": finalists[0]["tag"],
        "lowest_validation": ranked[0]["tag"],
    }
    print(f"  -> finalists {[f['tag'] for f in finalists]}\n", flush=True)

    print("stage 3: finalists on the other development patients, 2 seeds", flush=True)
    stage3 = []
    for finalist in finalists:
        for subject in OTHER_DEVELOPMENT:
            for seed in (1, 2):
                row = run(subject, finalist["config"], finalist["tag"], args.work, seed)
                if row:
                    row["subject"] = subject
                    stage3.append(row)
                    print(f"  {finalist['tag']:20s} {subject:22s} seed{seed} "
                          f"val={row['validation_next_bce']:.4f}", flush=True)
    record["stages"]["confirmation"] = {"rows": stage3}

    scores = {}
    for finalist in finalists:
        rows_all = [r for r in stage3 if r["tag"] == finalist["tag"]] + [finalist]
        scores[finalist["tag"]] = {
            "median_validation_next_bce": float(np.median(
                [r["validation_next_bce"] for r in rows_all])),
            "median_wiring_cost": float(np.median(
                [r.get("wiring_cost", np.nan) for r in rows_all])),
            "median_mean_degree": float(np.median(
                [r.get("mean_degree", np.nan) for r in rows_all])),
            "median_hop_reachability": float(np.median(
                [r.get("hop_reachability", np.nan) for r in rows_all])),
            "all_converged": bool(all(r.get("converged", True) for r in rows_all)),
            "config": finalist["config"],
        }
    winner = min(scores, key=lambda t: scores[t]["median_validation_next_bce"])
    record["scores"] = scores
    record["frozen"] = {"tag": winner, **scores[winner]}

    (args.out / "SWEEP_SUMMARY.json").write_text(json.dumps(record, indent=1))
    (args.out / "FROZEN_CONFIG.json").write_text(
        json.dumps(dict(scores[winner]["config"]), indent=1)
    )
    print("\nfrozen config:", json.dumps(scores[winner]["config"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
