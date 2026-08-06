"""The one structural readout the recovery gate leaves open.

That check found the identity of the connections unrecoverable, and the overall
direction of travel unrecoverable, but the *relative ordering* of how far each
tissue unit pushes along the axis recoverable in 7 of 7 synthetic runs.  So this
asks the only structural question that is licensed: is that ordering a property
of the patient, or of the optimiser?

Within patient, the same node positions are reused across training seeds, so the
orderings can be compared directly.  Between patients the node sets differ, so
both are interpolated onto a common normalised plane first, and only patients in
the same contact-count band are compared -- montage size alone would otherwise
separate them.

Everything here is about ordering.  None of it is a statement about which
connections exist, and it must never be written as one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    """Interpolate onto a normalised grid so different patients are comparable."""
    span = nodes_xy.max(axis=0) - nodes_xy.min(axis=0)
    span[span == 0] = 1.0
    unit = (nodes_xy - nodes_xy.min(axis=0)) / span
    axis = np.linspace(0.05, 0.95, GRID)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    values = griddata(unit, flow, (gx, gy), method="nearest")
    return values.ravel()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band-min-patients", type=int, default=3)
    args = parser.parse_args()

    gate = json.loads((OUT / "synthetic" / "RECOVERY_GATE.json").read_text())
    if not gate["reportable_layers"].get("node_flow_ordering"):
        raise SystemExit(
            "the recovery gate did not certify the flow-ordering layer; this "
            "analysis is not licensed"
        )

    cache = {p["subject"]: p for p in
             json.loads((OUT / "cache" / "CACHE_SUMMARY.json").read_text())["patients"]}

    flows: dict = {}
    for subject in sorted(cache):
        nodes_xy = np.load(OUT / "cache" / subject / "latent_nodes.npz")["nodes_xy"]
        for seed_dir in sorted((OUT / "per_subject" / subject / ARM).glob("seed*")) \
                if (OUT / "per_subject" / subject / ARM).exists() else []:
            graph = seed_dir / "graph.npz"
            if not graph.exists():
                continue
            adjacency = np.load(graph)["adjacency"]
            flows.setdefault(subject, {})[seed_dir.name] = (
                node_flow(adjacency, nodes_xy), nodes_xy
            )

    within = []
    for subject, seeds in flows.items():
        names = sorted(seeds)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rho = stats.spearmanr(seeds[names[i]][0], seeds[names[j]][0]).statistic
                within.append({"subject": subject, "pair": f"{names[i]}/{names[j]}",
                               "spearman": float(rho)})

    # Between patients, inside a contact-count band, using the first seed of each.
    between = []
    for low, high in BANDS:
        members = [s for s in flows if low <= cache[s]["n_contacts"] <= high]
        if len(members) < args.band_min_patients:
            continue
        projected = {}
        for s in members:
            first = sorted(flows[s])[0]
            flow, nodes_xy = flows[s][first]
            projected[s] = to_common_plane(flow, nodes_xy)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                rho = stats.spearmanr(projected[members[i]], projected[members[j]]).statistic
                between.append({"band": f"{low}-{high}",
                                "pair": f"{members[i]}/{members[j]}",
                                "spearman": float(rho)})

    verdict = {
        "contract": "topic5_slp_flow_ordering_v0_1",
        "licensed_by": "RECOVERY_GATE.json flow_ordering = RECOVERABLE",
        "scope": ("the relative ordering of how far each tissue unit pushes along "
                  "the axis; NOT which connections exist and NOT the direction of "
                  "travel, both of which the gate found unrecoverable"),
        "n_patients_with_a_graph": len(flows),
        "n_patients_with_two_seeds": sum(1 for v in flows.values() if len(v) >= 2),
        "within_patient": {
            "n_pairs": len(within),
            "median_spearman": float(np.median([w["spearman"] for w in within]))
            if within else None,
            "pairs": within,
        },
        "between_patient": {
            "n_pairs": len(between),
            "median_spearman": float(np.median([b["spearman"] for b in between]))
            if between else None,
            "pairs": between,
        },
    }
    if within and between:
        w = np.array([x["spearman"] for x in within])
        b = np.array([x["spearman"] for x in between])
        stat = stats.mannwhitneyu(w, b, alternative="greater")
        verdict["is_the_ordering_a_property_of_the_patient"] = {
            "within_median": float(np.median(w)),
            "between_median": float(np.median(b)),
            "difference": float(np.median(w) - np.median(b)),
            "mannwhitney_greater_p": float(stat.pvalue),
            "reading": (
                "the ordering is more reproducible within a patient than it is "
                "shared between patients, so it carries something patient-specific"
                if np.median(w) > np.median(b) and stat.pvalue < 0.05 else
                "the ordering is no more reproducible within a patient than "
                "between patients, so it cannot be called patient-specific"
            ),
        }
        print(f"within  n={len(w):3d} median={np.median(w):+.3f}")
        print(f"between n={len(b):3d} median={np.median(b):+.3f}")
        print(f"p={stat.pvalue:.3g}\n{verdict['is_the_ordering_a_property_of_the_patient']['reading']}")
    else:
        print(f"within pairs {len(within)}, between pairs {len(between)}: "
              "not enough to compare")

    (OUT / "flow_ordering.json").write_text(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
