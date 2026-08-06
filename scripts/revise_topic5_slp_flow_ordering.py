"""Patient-first flow ordering, with the geometry controls the original lacked.

Two problems with the first pass:

1. it compared 21 within-patient pairs against 70 between-patient pairs with a
   Mann-Whitney U, which treats those 70 as independent samples. They are not --
   every patient appears in many of them. The patient is the unit, so each
   patient gets one number, Delta_p = within_p - median_q!=p rho_pq, and the test
   runs over the 21 Delta_p.

2. it had no control for geometry. Node positions are sampled inside each
   patient's own contact hull with a per-patient seed, and the ordering metric is
   a weighted mean of along-axis displacement, so a node near the low-x edge has
   all its neighbours to one side whatever the weights are. Anything computed on
   those positions is more similar within patient than between, before a single
   gradient step. The control is the same statistic on untrained graphs.

Nothing is retrained. The untrained arm re-instantiates the model at the frozen
config and reads its adjacency at initialisation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_latent_rnn import ModelConfig, SLPModel  # noqa: E402
from src.topic5_virtual_seeg_operator import knn_edge_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
ARM = "LATENT_LEARNED_SPATIAL_RNN"
GRID = 12
BANDS = ((8, 12), (13, 20), (21, 100))


def node_flow(adjacency: np.ndarray, nodes_xy: np.ndarray) -> np.ndarray:
    """How far along the axis each unit's outgoing influence reaches."""
    dx = nodes_xy[None, :, 0] - nodes_xy[:, None, 0]
    w = np.abs(adjacency) * (~np.eye(len(adjacency), dtype=bool))
    return (w * dx).sum(1) / np.maximum(w.sum(1), 1e-9)


def to_common_plane(flow: np.ndarray, nodes_xy: np.ndarray) -> np.ndarray:
    span = nodes_xy.max(axis=0) - nodes_xy.min(axis=0)
    span[span == 0] = 1.0
    unit = (nodes_xy - nodes_xy.min(axis=0)) / span
    axis = np.linspace(0.05, 0.95, GRID)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    return griddata(unit, flow, (gx, gy), method="nearest").ravel()


def delta_per_patient(flows: dict, cache: dict) -> tuple[list[dict], list[str]]:
    """Delta_p = within_p - median over other patients in the same size band."""
    common = {s: {k: to_common_plane(v[0], v[1]) for k, v in seeds.items()}
              for s, seeds in flows.items()}
    rows, skipped = [], []
    for subject, seeds in sorted(flows.items()):
        names = sorted(seeds)
        if len(names) < 2:
            skipped.append(subject)
            continue
        within = float(stats.spearmanr(seeds[names[0]][0], seeds[names[1]][0]).statistic)
        n_contacts = cache[subject]["n_contacts"]
        band = next((b for b in BANDS if b[0] <= n_contacts <= b[1]), None)
        others = [
            float(stats.spearmanr(common[subject][names[0]],
                                  common[other][sorted(common[other])[0]]).statistic)
            for other in common
            if other != subject and band
            and band[0] <= cache[other]["n_contacts"] <= band[1]
        ]
        if not others:
            skipped.append(subject)
            continue
        rows.append({
            "subject": subject,
            "n_contacts": int(n_contacts),
            "within": within,
            "between_median": float(np.median(others)),
            "n_between": len(others),
            "delta": within - float(np.median(others)),
        })
    return rows, skipped


def patient_level_test(deltas: np.ndarray) -> dict:
    rng = np.random.default_rng(20260806)
    boot = np.array([np.median(rng.choice(deltas, len(deltas), replace=True))
                     for _ in range(4000)])
    # Sign-flip is the assumption-light companion to the exact Wilcoxon: under
    # the null a patient's delta is as likely to be negative as positive.
    flips = rng.choice([-1.0, 1.0], size=(10000, len(deltas)))
    null = np.median(flips * deltas, axis=1)
    observed = float(np.median(deltas))
    return {
        "n_patients": int(len(deltas)),
        "median_delta": observed,
        "bootstrap_95ci": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))],
        "n_positive": int((deltas > 0).sum()),
        "wilcoxon_exact_p": float(stats.wilcoxon(deltas, mode="exact").pvalue),
        "sign_flip_p": float((np.abs(null) >= abs(observed)).mean()),
    }


def untrained_flows(cache: dict, config: dict) -> dict:
    """Adjacency at initialisation: same nodes, same wiring prior, no training."""
    import torch

    flows: dict = {}
    for subject, meta in sorted(cache.items()):
        nodes = np.load(OUT / "cache" / subject / "latent_nodes.npz")["nodes_xy"]
        H = np.load(OUT / "cache" / subject / "seeg_operator.npz")["H"]
        d = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
        normalised = d / max(float(np.median(d[d > 0])), 1e-9)
        for seed in (1, 2):
            model = SLPModel(ModelConfig(
                arm=ARM,
                n_contacts=int(meta["n_contacts"]),
                n_nodes=len(nodes),
                microsteps=int(config["microsteps"]),
                edge_budget=float(config["edge_budget"]),
                seed=seed,
                normalised_distance=normalised,
                fixed_edge_mask=knn_edge_mask(nodes, int(config.get("knn_k", 6))),
                observation_operator=H,
            ))
            model.eval()
            with torch.no_grad():
                adjacency = model.graph.adjacency(1.0).numpy()
            flows.setdefault(subject, {})[f"seed{seed}"] = (
                node_flow(adjacency, nodes), nodes
            )
    return flows


def main() -> int:
    cache = {p["subject"]: p for p in
             json.loads((OUT / "cache" / "CACHE_SUMMARY.json").read_text())["patients"]}
    config = json.loads((OUT / "development" / "FROZEN_CONFIG_LONG.json").read_text())

    learned: dict = {}
    for subject in sorted(cache):
        nodes = np.load(OUT / "cache" / subject / "latent_nodes.npz")["nodes_xy"]
        arm_dir = OUT / "per_subject" / subject / ARM
        for seed_dir in sorted(arm_dir.glob("seed*")) if arm_dir.exists() else []:
            graph = seed_dir / "graph.npz"
            if graph.exists():
                learned.setdefault(subject, {})[seed_dir.name] = (
                    node_flow(np.load(graph)["adjacency"], nodes), nodes
                )

    learned_rows, learned_skipped = delta_per_patient(learned, cache)
    untrained_rows, _ = delta_per_patient(untrained_flows(cache, config), cache)

    learned_delta = np.array([r["delta"] for r in learned_rows])
    untrained_delta = np.array([r["delta"] for r in untrained_rows])

    by_subject = {r["subject"]: r["delta"] for r in untrained_rows}
    paired = np.array([
        r["delta"] - by_subject[r["subject"]]
        for r in learned_rows if r["subject"] in by_subject
    ])

    # How much of the ordering is just where the node sits on the axis?
    geometric = []
    for subject, seeds in learned.items():
        flow, nodes = seeds[sorted(seeds)[0]]
        geometric.append(float(abs(stats.spearmanr(flow, nodes[:, 0]).statistic)))

    result = {
        "contract": "topic5_slp_flow_ordering_patient_first_v0_1",
        "supersedes": "flow_ordering.json (pooled 21 within vs 70 between pairs)",
        "why": (
            "the pooled test treated 70 between-patient pairs as independent samples "
            "and had no control for the fact that node positions are patient-specific "
            "by construction"
        ),
        "retrained": False,
        "geometry_status": "RETROSPECTIVE_TEST_INFORMED_GEOMETRY",
        "learned": patient_level_test(learned_delta),
        "untrained_control": patient_level_test(untrained_delta),
        "learned_minus_untrained": patient_level_test(paired),
        "geometry_only_descriptive": {
            "median_abs_spearman_flow_vs_node_axis_coordinate": float(np.median(geometric)),
            "reading": (
                "the ordering is largely a restatement of where each unit sits on the "
                "axis; that part is fixed before training"
            ),
        },
        "skipped_patients": learned_skipped,
        "per_patient": learned_rows,
        "per_patient_untrained": untrained_rows,
    }
    verdict = (
        "PATIENT_SPECIFIC_SUPPORTED"
        if result["learned_minus_untrained"]["wilcoxon_exact_p"] < 0.05
        and result["learned_minus_untrained"]["median_delta"] > 0
        else "NOT_SEPARABLE_FROM_GEOMETRY"
    )
    result["verdict"] = verdict
    result["reading"] = (
        "training raises the within-versus-between gap beyond what the fixed node "
        "geometry already produces"
        if verdict == "PATIENT_SPECIFIC_SUPPORTED" else
        "the within-versus-between gap is not larger than an untrained graph on the "
        "same node positions already gives, so it cannot be called patient-specific "
        "propagation structure"
    )
    (OUT / "flow_ordering_revised.json").write_text(json.dumps(result, indent=1))

    import csv
    with (OUT / "flow_ordering_geometry_control.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "n_contacts", "learned_delta", "untrained_delta",
                    "learned_minus_untrained"])
        for r in learned_rows:
            u = by_subject.get(r["subject"])
            w.writerow([r["subject"], r["n_contacts"], f"{r['delta']:.6f}",
                        "" if u is None else f"{u:.6f}",
                        "" if u is None else f"{r['delta'] - u:.6f}"])

    for name in ("learned", "untrained_control", "learned_minus_untrained"):
        v = result[name]
        print(f"{name:26s} n={v['n_patients']:2d} median={v['median_delta']:+.4f} "
              f"CI[{v['bootstrap_95ci'][0]:+.4f},{v['bootstrap_95ci'][1]:+.4f}] "
              f"pos={v['n_positive']}/{v['n_patients']} "
              f"wilcoxon={v['wilcoxon_exact_p']:.4g} signflip={v['sign_flip_p']:.4g}")
    print(f"\n|rho(flow, node axis coordinate)| median = "
          f"{result['geometry_only_descriptive']['median_abs_spearman_flow_vs_node_axis_coordinate']:.3f}")
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
