"""Does the readout destroy the graph before any fitting is attempted?

The v0.1 gate said the learned adjacency cannot rank the true edges above
chance. That verdict cannot distinguish two very different causes: the fitting
procedure is too weak, or the observation does not carry the information in the
first place. The gate answers the first only by implication and the second not at
all -- and it was run on the subject with the widest read footprint in the
cohort, which makes the second explanation the more likely one.

This asks the second question directly, and costs no training.

Take one node cloud and one set of drives. Draw two DIFFERENT true graphs on it.
Run both through the same observation at a given readout width, and measure how
much the resulting rank sequences differ. If two different graphs produce the
same events, then no estimator of any kind could tell them apart, and the
adjacency was never recoverable regardless of how it was fitted. If they differ
sharply at a narrow readout and converge as it widens, the readout width is the
binding constraint.

Everything except the graph is held identical between the two members of a pair:
same nodes, same contacts, same observation operator, same drive seed. Across
widths, the node cloud and both graphs are also identical -- only the blur moves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_virtual_seeg_operator import (  # noqa: E402
    SUPPORT_SIGMA, build_observation_operator, sample_latent_nodes,
)
from scripts.run_topic5_slp_synthetic_control import (  # noqa: E402
    make_true_graph, simulate_events,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"

# Below this a contact is reading one node rather than a patch, which is a
# different model rather than a sharper measurement of this one.
MIN_EFFECTIVE_NODES = 2.0


def precedence(events: np.ndarray) -> np.ndarray:
    """How often contact i is recruited before contact j, over all events.

    Comparing events one at a time is dominated by the random drive: two runs of
    the SAME graph disagree on most individual ranks. The estimator never sees a
    single event either -- it sees the distribution. This is that distribution in
    the form the model has to reproduce: the tendency of one contact to come
    before another.
    """
    n = events.shape[1]
    before = np.zeros((n, n))
    total = np.zeros((n, n))
    for row in events:
        seen = row >= 0
        idx = np.where(seen)[0]
        if len(idx) < 2:
            continue
        r = row[idx]
        total[np.ix_(idx, idx)] += 1
        before[np.ix_(idx, idx)] += (r[:, None] < r[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(total > 0, before / np.maximum(total, 1), np.nan)
    iu = np.triu_indices(n, 1)
    return p[iu]


def summary_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two precedence profiles."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[ok] - b[ok])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+",
                        default=["yuquan_zhaochenxi", "epilepsiae_1146"])
    parser.add_argument("--n-nodes", type=int, default=288)
    parser.add_argument("--readout-radii", type=float, nargs="+",
                        default=[2.0, 3.0, 4.0, 6.0])
    parser.add_argument("--n-graph-pairs", type=int, default=5)
    parser.add_argument("--n-events", type=int, default=1500)
    parser.add_argument("--out", type=Path,
                        default=OUT_ROOT / "synthetic" / "READOUT_INFORMATION_PROBE.json")
    args = parser.parse_args()

    report = {
        "contract": "topic5_slp_readout_information_probe",
        "question": ("at a given readout width, do two different true graphs "
                     "produce different events -- before any fitting"),
        "held_identical_within_a_pair": ("node cloud, contacts, observation "
                                         "operator, drive seed"),
        "held_identical_across_widths": "node cloud and both graphs",
        "n_nodes": args.n_nodes,
        "subjects": {},
    }

    for subject in args.subjects:
        cache = OUT_ROOT / "cache" / subject
        xy = np.load(cache / "plane_coordinates.npz", allow_pickle=True)["xy_mm"]
        sigma_default = float(np.load(cache / "seeg_operator.npz")["sigma_mm"][0])
        nodes = sample_latent_nodes(xy, args.n_nodes, sigma_default, seed=11)

        graphs = [(make_true_graph(nodes, 0.35, 1.5, 100 + i),
                   make_true_graph(nodes, 0.35, 1.5, 200 + i))
                  for i in range(args.n_graph_pairs)]

        cells = []
        for radius in args.readout_radii:
            H = build_observation_operator(xy, nodes, radius / SUPPORT_SIGMA)
            row = H / H.sum(axis=1, keepdims=True)
            effective = float(np.median(1.0 / np.sum(row ** 2, axis=1)))

            graph_effect, drive_floor = [], []
            for i, (A, B) in enumerate(graphs):
                sim = lambda g, d: precedence(simulate_events(  # noqa: E731
                    g, H, args.n_events, 0.55, 1.9, 0.055, 0.02, d))
                a1, a2 = sim(A, 7000 + i), sim(A, 8000 + i)
                b1 = sim(B, 7000 + i)
                # Changing the graph with the drive held fixed, against changing
                # only the drive. The second is the floor: if the graph moves the
                # profile no further than a reseed does, the observation has not
                # kept which graph it was.
                graph_effect.append(summary_distance(a1, b1))
                drive_floor.append(summary_distance(a1, a2))
            graph_effect = np.array([g for g in graph_effect if np.isfinite(g)])
            drive_floor = np.array([g for g in drive_floor if np.isfinite(g)])
            eff_med = float(np.median(graph_effect)) if len(graph_effect) else None
            floor_med = float(np.median(drive_floor)) if len(drive_floor) else None
            ratio = (eff_med / floor_med) if (eff_med and floor_med) else None

            cells.append({
                "readout_radius_mm": radius,
                "effective_nodes_per_contact": effective,
                "degenerate": bool(effective < MIN_EFFECTIVE_NODES),
                "graph_effect": eff_med,
                "drive_only_floor": floor_med,
                "graph_effect_over_floor": ratio,
                "n_pairs": int(len(graph_effect)),
            })
            flag = "   [degenerate: a contact reads one node]" if cells[-1]["degenerate"] else ""
            print(f"  {subject:20s} radius {radius:4.1f} mm  covers "
                  f"{effective:5.2f} nodes   changing the graph moves the order "
                  f"profile {eff_med:.4f} vs {floor_med:.4f} for a reseed "
                  f"(x{ratio:.2f}){flag}")

        usable = [c for c in cells if not c["degenerate"]
                  and c["graph_effect_over_floor"] is not None]
        entry = {"default_readout_radius_mm": sigma_default * SUPPORT_SIGMA,
                 "n_contacts": int(len(xy)), "cells": cells}
        if len(usable) >= 2:
            narrow, wide = usable[0], usable[-1]
            entry["reading"] = (
                f"changing the graph moves the recruitment-order profile "
                f"{wide['graph_effect_over_floor']:.2f} times as far as a reseed "
                f"does at {wide['readout_radius_mm']:.0f} mm, and "
                f"{narrow['graph_effect_over_floor']:.2f} times at "
                f"{narrow['readout_radius_mm']:.0f} mm")
            entry["narrowing_helps"] = bool(
                narrow["graph_effect_over_floor"] > wide["graph_effect_over_floor"])
        report["subjects"][subject] = entry
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
