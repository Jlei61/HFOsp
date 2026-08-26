#!/usr/bin/env python3
"""Run one H3-S0 arm parameterised directly by an event-count memory."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.bridge import fit_bridge_arm
from src.topic5_continuous_marked_state.exposure import (
    EXPOSURE_REVISION,
    build_exposure_dataset,
)


def _train_median_iei_minutes(subject: str) -> float:
    payload = torch.load(
        contract.COHORT_CACHE, map_location="cpu", weights_only=False
    )[subject]
    times = payload["event_time"].numpy().astype(np.float64)
    session = payload["session_index"].numpy().astype(np.int64)
    bound = contract.load_split(subject)
    train = times < bound.train_end_epoch
    pair = train[1:] & train[:-1] & (session[1:] == session[:-1]) & (np.diff(times) > 0)
    intervals = np.diff(times)[pair] / 60.0
    if not len(intervals):
        raise RuntimeError(f"{subject}: no TRAIN intervals")
    return float(np.median(intervals))


def main() -> None:
    parser = argparse.ArgumentParser()
    subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    parser.add_argument("--subject", required=True, choices=subjects)
    parser.add_argument("--memory-events", required=True, type=float)
    parser.add_argument("--exposure-kind", choices=("load", "participation"),
                        default="load")
    parser.add_argument("--decay-clock", choices=("event_count", "physical_time"),
                        default="event_count")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.memory_events <= 0:
        raise ValueError("memory-events must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")

    step = _train_median_iei_minutes(args.subject)
    rate_matched_tau = float(args.memory_events * step)
    kind_token = "" if args.exposure_kind == "load" else "__participation"
    name = f"{args.subject}{kind_token}__N{args.memory_events:g}events.json"
    output_folder = (
        "exposure_event_count_grid" if args.decay_clock == "event_count"
        else "exposure_fixed_memory_physical"
    )
    output = contract.RESULT_ROOT / output_folder / name
    if output.exists() and not args.overwrite:
        old = json.loads(output.read_text())
        if (old.get("contract") == contract.REVISION
                and old.get("fit_revision") == contract.FIT_REVISION
                and old.get("exposure_revision") == EXPOSURE_REVISION
                and old.get("decay_clock") == args.decay_clock
                and old.get("event_count_memory_events") == args.memory_events):
            print(json.dumps({"status": "SKIPPED", "path": str(output)}))
            return
        raise FileExistsError(
            f"configuration collision at {output}; use --overwrite only after "
            "archiving the incompatible artifact"
        )

    dataset = build_exposure_dataset(
        args.subject, rate_matched_tau, exposure_kind=args.exposure_kind,
        decay_clock=args.decay_clock,
    )
    actual_memory = (
        rate_matched_tau
        / float(dataset.metadata["event_count_step_minutes_train_median"])
    )
    if not np.isclose(actual_memory, args.memory_events, rtol=0, atol=1e-10):
        raise RuntimeError("fixed event-count memory drifted from requested value")
    dataset.metadata.update({
        "event_count_memory_events": float(args.memory_events),
        "rate_matched_tau_minutes": rate_matched_tau,
        "clock_parameterisation": (
            "fixed_event_count_across_patients"
            if args.decay_clock == "event_count"
            else "rate_matched_physical_time_for_fixed_event_count"
        ),
    })
    fits = {
        "history": fit_bridge_arm(dataset.arrays, "b0_history", epochs=args.epochs),
        "real_exposure": fit_bridge_arm(
            dataset.arrays, "b1_spectral", epochs=args.epochs
        ),
        "causal_delayed_placebo": fit_bridge_arm(
            dataset.arrays, "b2_raw", epochs=args.epochs
        ),
    }
    for fit in fits.values():
        fit["claim_boundary"] = dataset.metadata["claim_boundary"]
    endpoints = {}
    real = fits["real_exposure"]["validation"]
    placebo = fits["causal_delayed_placebo"]["validation"]
    history = fits["history"]["validation"]
    for endpoint in ("joint_nll", "timing_nll", "mark_nll",
                     "participation_nll", "rank_nll", "stop_nll"):
        endpoints[endpoint] = {
            "real_minus_history": float(real[endpoint] - history[endpoint]),
            "placebo_minus_history": float(placebo[endpoint] - history[endpoint]),
            "real_minus_placebo": float(real[endpoint] - placebo[endpoint]),
        }
    result = {
        **dataset.metadata,
        "fit_revision": contract.FIT_REVISION,
        "fits": fits,
        "contrasts": endpoints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(temporary, output)
    print(json.dumps({"status": "DONE", "path": str(output)}))


if __name__ == "__main__":
    main()
