#!/usr/bin/env python3
"""Synthetic identifiability map, in the same currency as the real-data result.

For each cell the isotropic model is fitted from scratch, exactly as on real
patients, and then the motif is swept on the same wide grid used on real data.
The question is therefore not "could an ideal estimator recover the truth" but
"would *this* pipeline, on data of this size and noise, have seen a motif of
this strength at all".

A cell where a known motif is not recovered is a power statement about that
corner of the design.  It is what makes a real-data null readable.
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

from scripts.run_topic5_dynamical_motif_dose_response_v0_1 import SWEEPS, THETA_GRID  # noqa: E402
from scripts.train_topic5_dynamical_motif_unit_v0_1 import (  # noqa: E402
    DEFAULTS, evaluate, place_tensors, train_unit, write_json,
)
from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    MotifConfig, MotifRNN, build_motif_event_tensors,
)

TRUTH_PARAMETER = {"eta": "eta_raw", "beta": "beta", "gamma": "gamma_raw"}


def sweep_cell(unit, tensors, checkpoint: Path, device, cfg, sweep_name: str) -> dict:
    sweep = SWEEPS[sweep_name]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = MotifConfig(**{**payload["config"], "model_id": sweep["model"]})
    model = MotifRNN(config).to(device)
    model.load_warm_start(payload["model"])
    calibration, unseen = unit.indices(1), unit.indices(-1)
    angles = THETA_GRID if sweep["with_theta"] else [float(model.theta)]
    profile, best = [], None
    for value in sweep["values"]:
        for angle in angles:
            with torch.no_grad():
                getattr(model, sweep["parameter"]).fill_(float(value))
                if sweep["with_theta"]:
                    model.theta.fill_(float(angle))
            score = evaluate(model, tensors, calibration, device,
                             cfg["eval_batch"], float(cfg["stop_weight"]))
            profile.append({"value": float(value), "theta_rad": float(angle),
                            "calibration_contact_nll": score["contact_nll"]})
            if best is None or score["contact_nll"] < best["contact_nll"]:
                best = {"value": float(value), "theta": float(angle),
                        "contact_nll": score["contact_nll"]}
    held = {}
    for label, value in (("best", best["value"]), ("zero", 0.0)):
        with torch.no_grad():
            getattr(model, sweep["parameter"]).fill_(float(value))
            if sweep["with_theta"]:
                model.theta.fill_(float(best["theta"]))
        held[label] = evaluate(model, tensors, unseen, device,
                               cfg["eval_batch"], float(cfg["stop_weight"]))
    zero_calibration = min(row["calibration_contact_nll"] for row in profile
                           if row["value"] == 0.0)
    return {
        "selected_value": best["value"], "selected_theta_rad": best["theta"],
        "calibration_contact_nll_zero": zero_calibration,
        "calibration_contact_nll_best": best["contact_nll"],
        "calibration_gain": zero_calibration - best["contact_nll"],
        "unseen_contact_nll_zero": held["zero"]["contact_nll"],
        "unseen_contact_nll_best": held["best"]["contact_nll"],
        "unseen_gain": held["zero"]["contact_nll"] - held["best"]["contact_nll"],
        "profile": profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--tag", default="toy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--shard", default="")
    args = parser.parse_args()

    started = time.time()
    cfg = dict(DEFAULTS)
    cfg["max_epochs"] = args.max_epochs
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache = args.out_root / "frame_cache" / "SYNTHETIC"
    cells = args.cells or sorted(p.name for p in cache.iterdir() if p.is_dir())
    rows, profiles = [], []

    for cell in cells:
        truth = json.loads((cache / cell / "provenance.json").read_text())["ground_truth"]
        checkpoint = (args.out_root / args.tag / "SYNTHETIC" / cell / "DM0_ISOTROPIC"
                      / "seed0" / "checkpoint.pt")
        if not checkpoint.exists():
            try:
                train_unit("SYNTHETIC", cell, "DM0_ISOTROPIC", 0, args.out_root, device, cfg,
                           tag=args.tag)
            except Exception as error:  # noqa: BLE001 - a dead cell is a recorded gap
                rows.append({"cell_id": cell, "status": f"fit_failed:{type(error).__name__}",
                             **{f"truth_{k}": v for k, v in truth.items() if k != "parameters"}})
                print(f"[toy] {cell} fit failed: {error}", flush=True)
                continue
        unit = load_frame_unit(args.out_root, "SYNTHETIC", cell)
        tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm)
        tensors, _ = place_tensors(tensors, device)
        for sweep_name in SWEEPS:
            if sweep_name == "gamma":
                continue          # gamma needs the DM2 layer; covered by the ladder cells
            result = sweep_cell(unit, tensors, checkpoint, device, cfg, sweep_name)
            true_value = float(truth["parameters"].get(TRUTH_PARAMETER[sweep_name], 0.0))
            is_target = (TRUTH_PARAMETER[sweep_name] in truth["parameters"]
                         and abs(true_value) > 1e-9)
            rows.append({
                "cell_id": cell, "status": "ok", "sweep": sweep_name,
                "truth_model": truth["model_id"], "truth_strength": truth["strength"],
                "truth_value": true_value, "is_target_motif": bool(is_target),
                "size_label": truth["size_label"], "noise": truth["noise"],
                "tie_rate": truth["tie_rate"],
                "n_events": int(unit.ranks.shape[0]), "n_contacts": unit.n_contacts,
                "selected_value": result["selected_value"],
                "calibration_gain": result["calibration_gain"],
                "unseen_gain": result["unseen_gain"],
                "recovered_sign": bool(np.sign(result["selected_value"])
                                       == np.sign(true_value)) if is_target else None,
                "recovered_held_out": bool(result["unseen_gain"] > 0) if is_target else None,
                "false_positive": bool(result["unseen_gain"] > 0) if not is_target else None,
                "value_ratio": (result["selected_value"] / true_value) if is_target else None,
            })
            for point in result["profile"]:
                profiles.append({"cell_id": cell, "sweep": sweep_name,
                                 "truth_value": true_value, **point})
            print(f"[toy] {cell} {sweep_name}: truth={true_value:+.2f} "
                  f"selected={result['selected_value']:+.2f} "
                  f"held-out gain={result['unseen_gain']:+.6f}", flush=True)

    table = pd.DataFrame(rows)
    directory = args.out_root / "toy_identifiability"
    directory.mkdir(parents=True, exist_ok=True)
    shard = f"_{args.shard}" if args.shard else ""
    table.to_csv(directory / f"IDENTIFIABILITY_GRID{shard}.csv", index=False)
    pd.DataFrame(profiles).to_csv(directory / f"IDENTIFIABILITY_PROFILE{shard}.csv", index=False)
    summary = {"contract": "topic5_dynamical_motif_identifiability_v0_1",
               "n_cells": int(table.cell_id.nunique()) if not table.empty else 0,
               "seconds": time.time() - started}
    usable = table[table.status == "ok"] if not table.empty else table
    if not usable.empty:
        target = usable[usable.is_target_motif == True]  # noqa: E712
        null = usable[usable.is_target_motif == False]   # noqa: E712
        summary["recovery"] = {
            "n_target_cells": int(len(target)),
            "sign_recovered": int(target.recovered_sign.sum()) if len(target) else 0,
            "held_out_gain_positive": int(target.recovered_held_out.sum()) if len(target) else 0,
            "median_value_ratio": float(target.value_ratio.median()) if len(target) else None,
            "median_held_out_gain": float(target.unseen_gain.median()) if len(target) else None,
        }
        summary["false_positive"] = {
            "n_null_cells": int(len(null)),
            "held_out_gain_positive": int(null.false_positive.sum()) if len(null) else 0,
            "median_held_out_gain": float(null.unseen_gain.median()) if len(null) else None,
        }
        if len(target):
            summary["by_strength"] = (
                target.groupby(["sweep", "truth_strength"])
                .agg(n=("cell_id", "size"),
                     sign=("recovered_sign", "sum"),
                     held_out=("recovered_held_out", "sum"),
                     median_gain=("unseen_gain", "median"))
                .reset_index().to_dict(orient="records"))
    write_json(directory / f"IDENTIFIABILITY_SUMMARY{shard}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_strength"}, indent=1))


if __name__ == "__main__":
    main()
