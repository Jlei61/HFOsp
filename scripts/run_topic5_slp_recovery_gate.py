"""Milestone F1 gate: is node-level structure recoverable at all?

Generates rank events from a known sparse spatial graph, refits the learned
latent arm, and scores two separate layers:

* **edge identity** -- can the fitted adjacency rank the true edges above chance;
* **axis direction** -- does the fitted graph push activity the way the true one
  does, both in cohort mean and per node.

The two layers are scored separately on purpose.  A model can reproduce the
overall direction of travel while getting every individual edge wrong, and those
two facts license very different claims.

Multiple generator seeds and node densities, because a single realisation of
either layer is noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"

# Pre-registered floors.  Edge identity is scored as AUC against the true edge
# set; 0.60 is a deliberately modest bar -- well above chance, well below what a
# usable structural claim would need.
EDGE_AUC_FLOOR = 0.60
FLOW_SIGN_AGREEMENT_FLOOR = 0.80


def edge_auc(fitted: np.ndarray, true: np.ndarray) -> float:
    off = ~np.eye(len(fitted), dtype=bool)
    a, t = np.abs(fitted)[off], (true > 0)[off]
    if t.sum() == 0 or (~t).sum() == 0:
        return float("nan")
    order = np.argsort(a)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(len(a))
    n1, n0 = t.sum(), (~t).sum()
    return float((ranks[t].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def node_flow(weights: np.ndarray, nodes_xy: np.ndarray) -> np.ndarray:
    """Mean +x displacement of each node's outgoing influence."""
    dx = nodes_xy[None, :, 0] - nodes_xy[:, None, 0]
    w = np.abs(weights) * (~np.eye(len(weights), dtype=bool))
    return (w * dx).sum(1) / np.maximum(w.sum(1), 1e-9)


def run_cell(subject: str, n_nodes: int, seed: int, work: Path,
             config: Path) -> dict | None:
    cell = work / f"M{n_nodes}_seed{seed}"
    generate = [
        PYTHON, str(ROOT / "scripts/run_topic5_slp_synthetic_control.py"),
        "--subject", subject, "--n-nodes", str(n_nodes),
        "--seed", str(seed), "--out", str(cell),
    ]
    if subprocess.run(generate, capture_output=True).returncode != 0:
        return None
    fit = [
        PYTHON, str(ROOT / "scripts/train_topic5_slp_unit.py"),
        "--subject", f"synthetic_{subject}", "--arm", "LATENT_LEARNED_SPATIAL_RNN",
        "--seed", "1", "--cache-root", str(cell / "cache"),
        "--config", str(config), "--out", str(cell / "fit"),
    ]
    if subprocess.run(fit, capture_output=True).returncode != 0:
        return None
    if not (cell / "fit" / "graph.npz").exists():
        return None

    truth = np.load(cell / "ground_truth.npz")
    fitted = np.load(cell / "fit" / "graph.npz")["adjacency"]
    done = json.loads((cell / "fit" / "DONE.json").read_text())
    summary = json.loads((cell / "SYNTHETIC_SUMMARY.json").read_text())

    flow_true = node_flow(truth["A_true"], truth["nodes_xy"])
    flow_fit = node_flow(fitted, truth["nodes_xy"])
    return {
        "n_nodes": n_nodes,
        "generator_seed": seed,
        "observation_ratio": summary["observation_ratio"],
        "n_events_usable": summary["n_events_usable"],
        "edge_auc": edge_auc(fitted, truth["A_true"]),
        "flow_true_mean": float(flow_true.mean()),
        "flow_fitted_mean": float(flow_fit.mean()),
        "flow_sign_agrees": bool(np.sign(flow_true.mean()) == np.sign(flow_fit.mean())),
        "flow_node_spearman": float(stats.spearmanr(flow_true, flow_fit).statistic),
        "test_next_bce": done["test_next_bce"],
        "converged": done.get("converged", True),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="epilepsiae_1150")
    parser.add_argument("--node-counts", type=int, nargs="*", default=[12, 18, 36])
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT_ROOT / "synthetic")
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    cells = []
    for n_nodes in args.node_counts:
        for seed in args.seeds:
            cell = run_cell(args.subject, n_nodes, seed, args.work, args.config)
            if cell is None:
                print(f"M={n_nodes} seed={seed}: cell failed to produce a graph")
                continue
            cells.append(cell)
            print(
                f"M={n_nodes:3d} seed={seed:3d} ratio={cell['observation_ratio']:.2f} "
                f"edgeAUC={cell['edge_auc']:.3f} flow_sign={'ok' if cell['flow_sign_agrees'] else 'FLIPPED'} "
                f"flow_rho={cell['flow_node_spearman']:+.3f} bce={cell['test_next_bce']:.4f}"
            )

    if not cells:
        raise SystemExit("no synthetic cell completed; the gate cannot be evaluated")

    aucs = np.array([c["edge_auc"] for c in cells])
    sign_agreement = float(np.mean([c["flow_sign_agrees"] for c in cells]))
    rhos = np.array([c["flow_node_spearman"] for c in cells])

    edge_pass = bool(np.median(aucs) >= EDGE_AUC_FLOOR)
    flow_pass = bool(sign_agreement >= FLOW_SIGN_AGREEMENT_FLOOR)
    verdict = {
        "contract": "topic5_slp_recovery_gate_v0_1",
        "subject": args.subject,
        "n_cells": len(cells),
        "edge_identity": {
            "median_auc": float(np.median(aucs)),
            "min_auc": float(aucs.min()),
            "max_auc": float(aucs.max()),
            "floor": EDGE_AUC_FLOOR,
            "status": "RECOVERABLE" if edge_pass else "NOT_RECOVERABLE",
        },
        "axis_direction": {
            "sign_agreement": sign_agreement,
            "floor": FLOW_SIGN_AGREEMENT_FLOOR,
            "median_node_spearman": float(np.median(rhos)),
            "node_spearman_min": float(rhos.min()),
            "node_spearman_max": float(rhos.max()),
            "status": "RECOVERABLE" if flow_pass else "NOT_RECOVERABLE",
        },
        "consequence": (
            "structure legs (H3 learned-vs-fixed topology, H4 patient-specific "
            "graphs, H5 targeted lesion) are reportable"
            if edge_pass else
            "structure legs H3, H4 and H5 are NOT reportable from this "
            "parameterisation: the fitted adjacency does not rank the true edges "
            "above chance on data generated by a known sparse spatial graph, so "
            "any cohort graph statistic would describe an arbitrary member of a "
            "large equivalence class. Prediction legs H1, H1b and H2 are "
            "unaffected -- they ask whether the field predicts, not which edges "
            "carry it."
        ),
        "cells": cells,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RECOVERY_GATE.json").write_text(json.dumps(verdict, indent=1))
    print("\n" + json.dumps(
        {k: v for k, v in verdict.items() if k != "cells"}, indent=1
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
