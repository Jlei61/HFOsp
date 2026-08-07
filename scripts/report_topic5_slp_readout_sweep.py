"""Report the readout sweep: three layers per readout width, with the audit.

Assembled from the cell records rather than typed out, and it prints the
sparsification audit next to every number. The first run of this sweep came back
with edge-identity AUC of 0.87 at every width and it was an artefact -- the saved
graph had never been sparsified, so the score measured the wiring prior. Nothing
in the summary said so. Here the edge count sits in the same table as the AUC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"

EDGE_AUC_FLOOR = 0.60
FLOW_SIGN_FLOOR = 0.80


def load_cells(work: Path, verdict: Path) -> list[dict]:
    """Cells from the gate's own verdict, enriched with each fit's edge count."""
    scored = {}
    cells = json.loads(verdict.read_text()).get("cells", [])
    for c in cells:
        scored[(round(float(c["readout_radius_mm"]), 2), c["generator_seed"])] = c
    for cell_dir in sorted(work.iterdir()):
        m = re.match(r"M(\d+)_seed(\d+)_r([\d.]+)", cell_dir.name)
        done = cell_dir / "fit" / "DONE.json"
        if not m or not done.exists():
            continue
        key = (round(float(m.group(3)), 2), int(m.group(2)))
        if key in scored:
            fit = json.loads(done.read_text())
            scored[key]["n_edges"] = fit.get("n_edges")
            scored[key]["best_epoch"] = fit.get("best_epoch")
            scored[key]["n_nodes_fit"] = int(m.group(1))
    return list(scored.values())


def layers(cells: list[dict]) -> dict:
    aucs = np.array([c["edge_auc"] for c in cells])
    rhos = np.array([c["flow_node_spearman"] for c in cells])
    sign = float(np.mean([c["flow_sign_agrees"] for c in cells]))
    pos = int((rhos > 0).sum())
    p = float(binomtest(pos, len(rhos), 0.5, alternative="greater").pvalue)
    return {"edge_auc": float(np.median(aucs)), "edge_ok": bool(np.median(aucs) >= EDGE_AUC_FLOOR),
            "sign": sign, "sign_ok": bool(sign >= FLOW_SIGN_FLOOR),
            "order_rho": float(np.median(rhos)), "order_p": p, "order_ok": bool(p < 0.05),
            "n": len(cells)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+",
                        default=["epilepsiae_1146", "yuquan_zhaochenxi"])
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=OUT_ROOT / "synthetic" / "READOUT_SWEEP_REPORT.json")
    args = parser.parse_args()

    report = {"contract": "topic5_slp_readout_sweep_report",
              "edge_auc_floor": EDGE_AUC_FLOOR,
              "flow_sign_floor": FLOW_SIGN_FLOOR,
              "subjects": {}}

    for subject in args.subjects:
        verdict = OUT_ROOT / "synthetic" / f"readout_sweep_{subject}" / "RECOVERY_GATE.json"
        work = args.work_root / f"gate_{subject}"
        if not verdict.exists():
            print(f"{subject}: no verdict yet")
            continue
        cells = load_cells(work, verdict)
        budget_edges = 6 * cells[0].get("n_nodes_fit", 288) if cells else 0

        print(f"\n===== {subject} =====")
        print(f"{'radius':>7s} {'covers':>7s} {'edges':>7s} {'frozen':>7s} "
              f"{'edgeAUC':>8s} {'>=.60':>6s} {'dir':>5s} {'>=.80':>6s} "
              f"{'order rho':>10s} {'p':>7s}")
        by_radius = {}
        for c in cells:
            by_radius.setdefault(round(float(c["readout_radius_mm"]), 2), []).append(c)
        entry = {}
        for radius, group in sorted(by_radius.items()):
            frozen = [c for c in group
                      if c.get("n_edges") is not None
                      and c["n_edges"] <= 1.5 * budget_edges]
            usable = [c for c in frozen if not c.get("readout_is_degenerate")]
            med_edges = np.median([c.get("n_edges", np.nan) for c in group])
            if not usable:
                print(f"{radius:7.1f} {'':>7s} {med_edges:7.0f} "
                      f"{len(frozen)}/{len(group):<5d} "
                      f"{'no scorable cell':>40s}")
                entry[f"{radius:g}mm"] = {"status": "NO_SCORABLE_CELL",
                                          "n_frozen": len(frozen), "n_total": len(group)}
                continue
            L = layers(usable)
            cover = float(np.median([c["effective_nodes_per_contact"] for c in usable]))
            print(f"{radius:7.1f} {cover:7.2f} {med_edges:7.0f} "
                  f"{len(frozen)}/{len(group):<5d} "
                  f"{L['edge_auc']:8.3f} {'yes' if L['edge_ok'] else 'no':>6s} "
                  f"{L['sign']:5.2f} {'yes' if L['sign_ok'] else 'no':>6s} "
                  f"{L['order_rho']:+10.3f} {L['order_p']:7.4f}")
            entry[f"{radius:g}mm"] = dict(
                L, effective_nodes_per_contact=cover, median_n_edges=float(med_edges),
                n_frozen=len(frozen), n_total=len(group))
        report["subjects"][subject] = entry

        scorable = [k for k, v in entry.items() if v.get("n", 0) > 0]
        if len(scorable) >= 2:
            wide, narrow = entry[scorable[-1]], entry[scorable[0]]
            report["subjects"][subject]["reading"] = (
                f"edge identity {wide['edge_auc']:.3f} at {scorable[-1]} against "
                f"{narrow['edge_auc']:.3f} at {scorable[0]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
