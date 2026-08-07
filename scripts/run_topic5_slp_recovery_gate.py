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
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"

# Pre-registered floors.  Edge identity is scored as AUC against the true edge
# set; 0.60 is a deliberately modest bar -- well above chance, well below what a
# usable structural claim would need.
EDGE_AUC_FLOOR = 0.60
FLOW_SIGN_AGREEMENT_FLOOR = 0.80
# The third layer is weaker than either: not which edges exist, not which way the
# field travels overall, but only whether nodes are ordered correctly by how far
# along the axis their influence reaches.  Scored by a sign test across cells.
FLOW_ORDER_BINOMIAL_ALPHA = 0.05


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
             config: Path, readout_radius_mm: float = 0.0,
             true_mean_degree: float = 0.0) -> dict | None:
    tag = f"M{n_nodes}_seed{seed}"
    if readout_radius_mm:
        tag += f"_r{readout_radius_mm:g}"
    cell = work / tag
    generate = [
        PYTHON, str(ROOT / "scripts/run_topic5_slp_synthetic_control.py"),
        "--subject", subject, "--n-nodes", str(n_nodes),
        "--seed", str(seed), "--out", str(cell),
    ]
    if readout_radius_mm:
        generate += ["--readout-radius-mm", str(readout_radius_mm)]
    if true_mean_degree:
        generate += ["--true-mean-degree", str(true_mean_degree)]
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
        "readout_radius_mm": summary.get("readout_radius_mm"),
        "effective_nodes_per_contact": summary.get("effective_nodes_per_contact"),
        # A contact reading fewer than two nodes is a relabelling of one node,
        # not a sharper measurement -- such a cell must not be scored.
        "readout_is_degenerate": summary.get("readout_is_degenerate", False),
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


def score_layers(cells: list[dict]) -> dict:
    """The three layers, scored on one group of cells.

    Pulled out of main because the verdict now has to be computed PER readout
    width. Pooling the widths into a single number would average away the exact
    comparison the sweep exists to make.
    """
    aucs = np.array([c["edge_auc"] for c in cells])
    rhos = np.array([c["flow_node_spearman"] for c in cells])
    sign_agreement = float(np.mean([c["flow_sign_agrees"] for c in cells]))
    n_positive = int((rhos > 0).sum())
    order_p = float(binomtest(n_positive, len(rhos), 0.5, alternative="greater").pvalue)
    edge_pass = bool(np.median(aucs) >= EDGE_AUC_FLOOR)
    flow_pass = bool(sign_agreement >= FLOW_SIGN_AGREEMENT_FLOOR)
    order_pass = bool(order_p < FLOW_ORDER_BINOMIAL_ALPHA)
    return {
        "n_cells": len(cells),
        "edge_identity": {
            "median_auc": float(np.median(aucs)), "floor": EDGE_AUC_FLOOR,
            "status": "RECOVERABLE" if edge_pass else "NOT_RECOVERABLE"},
        "axis_direction": {
            "sign_agreement": sign_agreement, "floor": FLOW_SIGN_AGREEMENT_FLOOR,
            "status": "RECOVERABLE" if flow_pass else "NOT_RECOVERABLE"},
        "flow_ordering": {
            "n_cells_positive": n_positive, "n_cells": len(rhos),
            "median_node_spearman": float(np.median(rhos)), "sign_test_p": order_p,
            "status": "RECOVERABLE" if order_pass else "NOT_RECOVERABLE"},
        "reportable_layers": {"edge_identity": edge_pass,
                              "global_axis_direction": flow_pass,
                              "node_flow_ordering": order_pass},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="epilepsiae_1150")
    parser.add_argument("--node-counts", type=int, nargs="*", default=[12, 18, 36])
    # Readout radius as its own axis. It was previously fixed at half the median
    # contact pitch, which is an interpolation convenience rather than a
    # measurement of how far a contact hears -- and the original verdict was
    # obtained on the subject with the widest footprint in the cohort. 0 means
    # the subject default. A narrow radius needs a dense node cloud to stay
    # meaningful, so node count is held FIXED across radii and the default-radius
    # arm at the same node count is the control that isolates the readout.
    parser.add_argument("--readout-radii", type=float, nargs="*", default=[0.0])
    # Edges per node in the true graph. The original rule kept the top 20% of
    # pairs, so raising the node count raised the degree with it and the recovery
    # question silently became harder for a reason unrelated to the readout.
    parser.add_argument("--true-mean-degree", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT_ROOT / "synthetic")
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    cells = []
    for radius in args.readout_radii:
        for n_nodes in args.node_counts:
            for seed in args.seeds:
                cell = run_cell(args.subject, n_nodes, seed, args.work, args.config,
                                readout_radius_mm=radius,
                                true_mean_degree=args.true_mean_degree)
                if cell is None:
                    print(f"r={radius:g} M={n_nodes} seed={seed}: cell failed to "
                          "produce a graph")
                    continue
                cells.append(cell)
                # A degenerate readout is printed and stored but excluded from the
                # verdict: there a contact reads a single node, which is a
                # different model rather than a sharper look at this one.
                flag = "  [DEGENERATE readout]" if cell["readout_is_degenerate"] else ""
                print(
                    f"r={cell['readout_radius_mm']:4.1f}mm cover="
                    f"{cell['effective_nodes_per_contact']:5.2f} "
                    f"M={n_nodes:3d} seed={seed:3d} ratio={cell['observation_ratio']:.2f} "
                    f"edgeAUC={cell['edge_auc']:.3f} "
                    f"flow_sign={'ok' if cell['flow_sign_agrees'] else 'FLIPPED'} "
                    f"flow_rho={cell['flow_node_spearman']:+.3f} "
                    f"bce={cell['test_next_bce']:.4f}{flag}"
                )

    if not cells:
        raise SystemExit("no synthetic cell completed; the gate cannot be evaluated")


    # Per readout width, degenerate cells excluded from the verdict.
    by_radius = {}
    for c in cells:
        by_radius.setdefault(round(float(c["readout_radius_mm"]), 2), []).append(c)
    per_readout = {}
    for radius, group in sorted(by_radius.items()):
        usable = [c for c in group if not c["readout_is_degenerate"]]
        entry = {
            "readout_radius_mm": radius,
            "median_effective_nodes_per_contact": float(
                np.median([c["effective_nodes_per_contact"] for c in group])),
            "n_cells_total": len(group), "n_cells_degenerate": len(group) - len(usable),
        }
        entry.update(score_layers(usable) if usable else
                     {"status": "ALL_CELLS_DEGENERATE"})
        per_readout[f"{radius:g}mm"] = entry

    aucs = np.array([c["edge_auc"] for c in cells])
    sign_agreement = float(np.mean([c["flow_sign_agrees"] for c in cells]))
    rhos = np.array([c["flow_node_spearman"] for c in cells])

    edge_pass = bool(np.median(aucs) >= EDGE_AUC_FLOOR)
    flow_pass = bool(sign_agreement >= FLOW_SIGN_AGREEMENT_FLOOR)
    n_positive = int((rhos > 0).sum())
    order_p = float(binomtest(n_positive, len(rhos), 0.5, alternative="greater").pvalue)
    order_pass = bool(order_p < FLOW_ORDER_BINOMIAL_ALPHA)
    verdict = {
        "contract": "topic5_slp_recovery_gate_v0_1",
        "subject": args.subject,
        "n_cells": len(cells),
        "true_mean_degree_requested": args.true_mean_degree,
        # The pooled numbers below average over readout widths and are kept only
        # for continuity with the original run. Read per_readout instead.
        "per_readout": per_readout,
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
        "flow_ordering": {
            "n_cells_positive": n_positive,
            "n_cells": len(rhos),
            "median_node_spearman": float(np.median(rhos)),
            "sign_test_p": order_p,
            "status": "RECOVERABLE" if order_pass else "NOT_RECOVERABLE",
            "means": (
                "the relative ordering of how far along the axis each node's "
                "influence reaches is recovered; this is weaker than edge "
                "identity and weaker than the global direction of travel"
            ),
        },
        "reportable_layers": {
            "edge_identity": edge_pass,
            "global_axis_direction": flow_pass,
            "node_flow_ordering": order_pass,
        },
        "consequence": (
            "every structural leg is reportable"
            if edge_pass else
            "claims about which edges exist are NOT reportable from this "
            "parameterisation: the fitted adjacency does not rank the true edges "
            "above chance on data generated by a known sparse spatial graph, so "
            "any cohort graph statistic describes an arbitrary member of a large "
            "equivalence class. That blocks H4 (patient-specific graphs) and H5 "
            "(targeted lesion). It does NOT block H3: whether learning the edges "
            "predicts better than fixing them to nearest neighbours is a "
            "prediction question that never needs the edges to be identified. "
            "H1, H1b and H2 are likewise unaffected."
            + (
                " Node-level flow ORDERING is separately recoverable, so a cohort "
                "comparison of where the field pushes hardest is defensible if it "
                "is framed as relative ordering and never as the graph itself."
                if order_pass else
                " No structural layer survived, including node flow ordering."
            )
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
