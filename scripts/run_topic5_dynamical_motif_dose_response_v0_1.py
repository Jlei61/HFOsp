#!/usr/bin/env python3
"""Uniform dose-response profile of each motif at the converged solution.

No training happens here.  Each motif parameter is swept on a wide, fixed grid
on top of the layer it extends, and the calibration split alone chooses the best
point; the model-unseen value is then read off at that single chosen point, so
the held-out number is never the maximum over a grid.

This is the shape of the answer the whole study turns on: if the optimum sits at
zero with visible curvature on both sides, the motif is not merely undetected,
it is actively disfavoured at the scale the data can resolve.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_topic5_dynamical_motif_unit_v0_1 import (  # noqa: E402
    DEFAULTS, evaluate, place_tensors, write_json,
)
from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    MotifConfig, MotifRNN, build_motif_event_tensors,
)

THETA_GRID = [i * math.pi / 12.0 for i in range(12)]
SWEEPS = {
    "eta": {"base": "DM0_ISOTROPIC", "model": "DM1_FREE_AXIS", "parameter": "eta_raw",
            "values": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4], "with_theta": True},
    "beta": {"base": "DM1_FREE_AXIS", "model": "DM2_LOCAL_DIRECTIONAL", "parameter": "beta",
             "values": [-3.0, -2.0, -1.5, -1.0, -0.6, -0.3, -0.15, 0.0,
                        0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0], "with_theta": True},
    "gamma": {"base": "DM2_LOCAL_DIRECTIONAL", "model": "DM3_AXIS_FEEDFORWARD_TRANSIENT",
              "parameter": "gamma_raw",
              "values": [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], "with_theta": False},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-rule", default="M2-2RANK")
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    started = time.time()
    cfg = dict(DEFAULTS)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_root = args.out_root / args.tag / args.frame
    subjects = args.subjects or sorted(p.name for p in base_root.iterdir() if p.is_dir())
    rows, selected_rows = [], []

    for subject in subjects:
        unit = load_frame_unit(args.out_root, args.frame, subject)
        tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm,
                                            gate_rule=args.gate_rule)
        tensors, _ = place_tensors(tensors, device)
        calibration, unseen = unit.indices(1), unit.indices(-1)
        for sweep_name, sweep in SWEEPS.items():
            checkpoint = (base_root / subject / sweep["base"] / f"seed{args.seed_index}"
                          / "checkpoint.pt")
            if not checkpoint.exists():
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = MotifConfig(**{**payload["config"], "model_id": sweep["model"]})
            model = MotifRNN(config).to(device)
            model.load_warm_start(payload["model"])
            if sweep["model"] == "DM3_AXIS_FEEDFORWARD_TRANSIENT":
                model.calibrate_shuffle_radius()
            angles = THETA_GRID if sweep["with_theta"] else [float(model.theta)]
            best = None
            for value in sweep["values"]:
                for angle in angles:
                    with torch.no_grad():
                        getattr(model, sweep["parameter"]).fill_(float(value))
                        if sweep["with_theta"]:
                            model.theta.fill_(float(angle))
                    score = evaluate(model, tensors, calibration, device,
                                     cfg["eval_batch"], float(cfg["stop_weight"]))
                    total = score["next_bce"] + float(cfg["stop_weight"]) * score["stop_bce"]
                    rows.append({
                        "frame": args.frame, "subject": subject, "sweep": sweep_name,
                        "parameter": sweep["parameter"], "value": float(value),
                        "theta_rad": float(angle),
                        "calibration_score": total,
                        "calibration_contact_nll": score["contact_nll"],
                        "calibration_next_bce": score["next_bce"],
                        "calibration_stop_bce": score["stop_bce"],
                        "calibration_top1": score["top1"],
                    })
                    if best is None or score["contact_nll"] < best["contact_nll"]:
                        best = {"value": float(value), "theta": float(angle),
                                "contact_nll": score["contact_nll"], "score": total}
            # Read the held-out value at the single calibration-chosen point and
            # at the zero point, so the reported gain is not a grid maximum.
            zero_angle = float(model.theta) if not sweep["with_theta"] else best["theta"]
            report = {}
            for label, value, angle in (("best", best["value"], best["theta"]),
                                        ("zero", 0.0, zero_angle)):
                with torch.no_grad():
                    getattr(model, sweep["parameter"]).fill_(float(value))
                    if sweep["with_theta"]:
                        model.theta.fill_(float(angle))
                unseen_score = evaluate(model, tensors, unseen, device,
                                        cfg["eval_batch"], float(cfg["stop_weight"]))
                report[label] = unseen_score
            selected_rows.append({
                "frame": args.frame, "subject": subject, "sweep": sweep_name,
                "n_contacts": unit.n_contacts, "n_model_unseen": int(unseen.size),
                "geometry_class": unit.provenance.get("geometry_class"),
                "calibration_selected_value": best["value"],
                "calibration_selected_theta_rad": best["theta"],
                "calibration_contact_nll_zero": next(
                    r["calibration_contact_nll"] for r in rows
                    if r["subject"] == subject and r["sweep"] == sweep_name
                    and r["value"] == 0.0 and abs(r["theta_rad"] - best["theta"]) < 1e-9),
                "calibration_contact_nll_best": best["contact_nll"],
                "unseen_contact_nll_zero": report["zero"]["contact_nll"],
                "unseen_contact_nll_best": report["best"]["contact_nll"],
                "unseen_contact_nll_gain": (report["zero"]["contact_nll"]
                                            - report["best"]["contact_nll"]),
                "unseen_score_zero": (report["zero"]["next_bce"]
                                      + report["zero"]["stop_bce"]),
                "unseen_score_best": (report["best"]["next_bce"]
                                      + report["best"]["stop_bce"]),
                "unseen_top1_zero": report["zero"]["top1"],
                "unseen_top1_best": report["best"]["top1"],
                "selected_at_grid_edge": bool(
                    abs(best["value"]) >= max(abs(v) for v in sweep["values"]) - 1e-9),
            })
            print(f"[dose] {subject} {sweep_name}: selected {best['value']:+.3f} "
                  f"held-out gain {selected_rows[-1]['unseen_contact_nll_gain']:+.6f}",
                  flush=True)

    profile = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    profile.to_csv(args.out_root / "DOSE_RESPONSE_PROFILE.csv", index=False)
    selected.to_csv(args.out_root / "DOSE_RESPONSE_PER_PATIENT.csv", index=False)
    summary = {"contract": "topic5_dynamical_motif_dose_response_v0_1",
               "frame": args.frame, "tag": args.tag, "seed_index": args.seed_index,
               "n_patients": int(selected.subject.nunique()) if not selected.empty else 0,
               "grids": {k: v["values"] for k, v in SWEEPS.items()},
               "theta_grid_size": len(THETA_GRID),
               "selection_rule": "calibration split chooses one point; the held-out value is "
                                 "read at that point only",
               "seconds": time.time() - started}
    if not selected.empty:
        for sweep_name, group in selected.groupby("sweep"):
            summary[f"{sweep_name}_summary"] = {
                "n": int(len(group)),
                "n_selected_zero": int((group.calibration_selected_value == 0).sum()),
                "n_selected_at_edge": int(group.selected_at_grid_edge.sum()),
                "median_unseen_contact_nll_gain": float(group.unseen_contact_nll_gain.median()),
                "n_positive_unseen_gain": int((group.unseen_contact_nll_gain > 0).sum()),
            }
    write_json(args.out_root / "DOSE_RESPONSE_SUMMARY.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("summary")
                      or k == "n_patients"}, indent=1))


if __name__ == "__main__":
    main()
