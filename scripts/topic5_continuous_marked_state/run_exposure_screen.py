#!/usr/bin/env python3
"""Run the full-event H3-S0 screen for one subject and one time scale."""
from __future__ import annotations

import argparse
import json
import os

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.bridge import fit_bridge_arm
from src.topic5_continuous_marked_state.exposure import (
    EXPOSURE_REVISION,
    build_exposure_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    all_subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    parser.add_argument("--subject", required=True, choices=all_subjects)
    parser.add_argument("--tau-minutes", required=True, type=float)
    parser.add_argument("--exposure-kind", choices=("load", "participation"),
                        default="load")
    parser.add_argument("--decay-clock", choices=("physical_time", "event_count"),
                        default="physical_time")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    kind_token = "" if args.exposure_kind == "load" else f"__{args.exposure_kind}"
    name = f"{args.subject}{kind_token}__tau{args.tau_minutes:g}m.json"
    output_folder = (
        "exposure_screen" if args.decay_clock == "physical_time"
        else "exposure_clock_control"
    )
    output = contract.RESULT_ROOT / output_folder / name
    if output.exists() and not args.overwrite:
        old = json.loads(output.read_text())
        if (old.get("contract") == contract.REVISION
                and old.get("exposure_revision") == EXPOSURE_REVISION
                and old.get("fit_revision") == contract.FIT_REVISION
                and old.get("decay_clock", "physical_time") == args.decay_clock):
            print(json.dumps({"status": "SKIPPED", "path": str(output)}))
            return
        raise FileExistsError(
            f"configuration collision at {output}; use --overwrite only after "
            "archiving the incompatible artifact"
        )
    dataset = build_exposure_dataset(
        args.subject, args.tau_minutes, exposure_kind=args.exposure_kind,
        decay_clock=args.decay_clock,
    )
    fits = {
        "history": fit_bridge_arm(dataset.arrays, "b0_history", epochs=args.epochs),
        "real_exposure": fit_bridge_arm(dataset.arrays, "b1_spectral", epochs=args.epochs),
        "causal_delayed_placebo": fit_bridge_arm(dataset.arrays, "b2_raw", epochs=args.epochs),
    }
    for fit in fits.values():
        fit["claim_boundary"] = dataset.metadata["claim_boundary"]
    real = fits["real_exposure"]["validation"]
    placebo = fits["causal_delayed_placebo"]["validation"]
    history = fits["history"]["validation"]
    endpoints = {}
    for key in ("joint_nll", "timing_nll", "mark_nll", "participation_nll",
                "rank_nll", "stop_nll"):
        endpoints[key] = {
            "real_minus_history": float(real[key] - history[key]),
            "placebo_minus_history": float(placebo[key] - history[key]),
            "real_minus_placebo": float(real[key] - placebo[key]),
        }
    result = {
        **dataset.metadata,
        "fit_revision": contract.FIT_REVISION,
        "fits": fits,
        "contrasts": endpoints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, output)
    print(json.dumps({
        "status": "DONE", "path": str(output),
        "joint_real_minus_placebo": endpoints["joint_nll"]["real_minus_placebo"],
        "timing_real_minus_placebo": endpoints["timing_nll"]["real_minus_placebo"],
        "mark_real_minus_placebo": endpoints["mark_nll"]["real_minus_placebo"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
