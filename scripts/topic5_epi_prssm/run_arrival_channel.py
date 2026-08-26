#!/usr/bin/env python3
"""Arrival channel: make the likelihood explain *when* discharges happen.

This is NOT the joint time+mark model.  It fits the arrival times alone, with the
spatial mark model left untouched as a nested control.  The earlier filename said
"joint", which overclaimed: nothing here calls the graph state or the contact
decoder, so it cannot show that one coupled slow state moves both the timing and
the repertoire.  That remains the next step.

v0.1 fed the timestamps in but never asked the likelihood to account for them, so
the state was free to track "which event number is this" -- and it did.  This run
adds the missing channel and asks the two questions that channel exists for:

  t0_exogenous_clock vs renewal_only  does anything slow move the rate at all?
  t1_observer vs t0                   does knowing the past discharges help, i.e. are
                                      they observations of the state?
  t2_physical vs t1_observer          do the discharges have to *push* the state, over
                                      and above informing us about it?  The observer
                                      arm updates on the unpredicted part only, so a
                                      perfectly predicted discharge moves it not at
                                      all; the physical arm still delivers a push.
                                      That difference is the identifiable H3 question.

Only recorded time enters the survival integral.  Interval membership comes from the
metadata-resolved recording blocks, not from the session index: sessions join blocks
across gaps shorter than a threshold, which silently carried 1,070 intervals and
1,337 hours of unrecorded wall time into the integral as evidence of absence.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_json, code_revision,
    dataset_of, input_hash, is_complete, load_tensors, package_hash, resolve_cohort,
    sha256_obj,
)
from src.topic5_epi_prssm.arrival import (  # noqa: E402
    RenewalIntensity, goodness_of_fit)
from src.topic5_epi_prssm.rate_state import ARMS, RateState  # noqa: E402
RECORDED = OUTPUT_ROOT / "recorded_intervals"

GOAL = "goal5_arrival"
OUT = OUTPUT_ROOT / "arrival_channel"
SESSION_JOIN = float(FROZEN["session_join_seconds"])


def patient_arrays(patient) -> dict:
    """Everything the arrival channel needs, in event order."""
    times = np.asarray(patient.event_time, dtype=np.float64)
    order = np.argsort(times)
    times = times[order]
    elapsed = np.diff(times, prepend=times[0])
    segment_start = patient.session_open.numpy().astype(bool)[order].copy()
    segment_start[0] = True
    # Interval membership comes from the recording metadata, never from the session
    # index: a session joins blocks across gaps shorter than its threshold, so the
    # session flag misses most real gaps.  An interval that crosses unrecorded time
    # is evidence in neither direction and is dropped from the integral; the state is
    # also reset there, since it was never observed across the gap.
    cache = RECORDED / f"{patient.subject}.npz"
    if not cache.exists():
        raise SystemExit(f"{cache} missing: run build_recorded_intervals.py first; "
                         "inferring gaps from the session index understates them")
    with np.load(cache) as z:
        spans_gap = z["spans_gap"].astype(bool)
        recorded_seconds = z["recorded"].astype(np.float64)
    if len(spans_gap) != len(times):
        raise SystemExit(f"{patient.subject}: recorded-interval cache has "
                         f"{len(spans_gap)} rows for {len(times)} events")
    segment_start = segment_start | spans_gap
    recorded = ~spans_gap
    # the hazard clock runs only while the recorder is on
    elapsed = np.where(spans_gap, elapsed, recorded_seconds)
    since_open = np.zeros(len(times))
    anchor = times[0]
    for i in range(len(times)):
        if segment_start[i]:
            anchor = times[i]
        since_open[i] = times[i] - anchor
    load = patient.participation.numpy()[order].sum(axis=1).astype(np.float64)
    split = patient.split.numpy()[order]
    elapsed_t = torch.tensor(np.maximum(elapsed, 1e-3), dtype=torch.float32)
    # the previous interval, reset at a segment start where there is no predecessor
    previous = np.concatenate([[np.nan], elapsed[:-1]])
    previous[segment_start] = np.nan
    previous = np.where(np.isfinite(previous), previous, np.nanmedian(elapsed))
    return {
        "previous_elapsed": torch.tensor(np.maximum(previous, 1e-3), dtype=torch.float32),
        "elapsed": elapsed_t,
        "time_of_day": torch.tensor(2 * np.pi * ((times % 86400.0) / 86400.0),
                                    dtype=torch.float32),
        "log_since_open": torch.tensor(np.log1p(since_open), dtype=torch.float32),
        "load": torch.tensor(load, dtype=torch.float32),
        "segment_start": torch.tensor(segment_start),
        "recorded": torch.tensor(recorded),
        "split": torch.tensor(split, dtype=torch.long),
        "n_segments": int(segment_start.sum()),
    }


def channel_nll(rate_state, intensity, arrays, patient_index, mask):
    z = rate_state(arrays["elapsed"], arrays["time_of_day"], arrays["log_since_open"],
                   arrays["load"], arrays["segment_start"])
    keep = arrays["recorded"] & mask
    out = intensity(arrays["elapsed"], z, patient_index, keep,
                    arrays.get("previous_elapsed"))
    n = out["n_recorded"].clamp(min=1.0)
    return out["nll"] / n, out, z, keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--markov-renewal", action="store_true",
                        help="let the intensity depend on the previous interval; without "
                             "it the residuals keep the lag-1 structure the data has")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    patients = load_tensors(resolve_cohort(args.cohort))
    config = {"arm": args.arm, "state_dim": args.state_dim, "epochs": args.max_epochs,
              "lr": args.lr, "session_join_seconds": SESSION_JOIN,
              "markov_renewal": bool(args.markov_renewal)}
    key = JobKey(goal=GOAL, family="arrival", arm=args.arm, seed=args.seed,
                 split="development", cohort=args.cohort,
                 config_hash=sha256_obj(config)[:16],
                 input_hash=input_hash(patients)[:16], code_revision=package_hash()[:16])
    target = OUT / "runs" / f"{key.job_id}.json"
    if target.exists() and is_complete(key) and not args.overwrite:
        print(f"SKIPPED_EXISTING {key.job_id}")
        return

    with JobRunner(key) as record:
        torch.manual_seed(args.seed)
        arrays = [patient_arrays(p) for p in patients]
        index = [torch.full((len(a["elapsed"]),), i, dtype=torch.long)
                 for i, a in enumerate(arrays)]

        rate_state = RateState(args.state_dim, arm=args.arm)
        intensity = RenewalIntensity(len(patients), args.state_dim,
                                     markov_renewal=args.markov_renewal)
        every = torch.cat([a["elapsed"] for a in arrays])
        every_recorded = torch.cat([a["recorded"] for a in arrays])
        intensity.initialise_from(every, every_recorded)

        parameters = list(rate_state.parameters()) + list(intensity.parameters())
        optimiser = torch.optim.Adam(parameters, lr=args.lr)
        # per-patient stepping makes the fit much stronger but also able to diverge;
        # a smoke run went 3.04 -> 5.09 -> 10.52 on validation in two epochs
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=2)
        train_mask = [a["split"] == 0 for a in arrays]
        val_mask = [a["split"] == 1 for a in arrays]

        history, best, best_state, patience = [], float("inf"), None, 0
        # One step per patient, not one per epoch.  Stepping outside the patient loop
        # gave 25 gradient updates for the whole fit, which left every parameter at
        # its initialisation -- the fitted time constants came back bit-identical to
        # the log-spaced init, and the arms with more parameters simply stayed where
        # they started and scored worse.  That produced a monotonically-worse ladder
        # that looked like a negative result and was an optimisation failure.
        for epoch in range(args.max_epochs):
            total = 0.0
            for i in torch.randperm(len(arrays)).tolist():
                optimiser.zero_grad()
                loss, *_ = channel_nll(rate_state, intensity, arrays[i], index[i],
                                       train_mask[i])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimiser.step()
                total += float(loss) / len(arrays)

            with torch.no_grad():
                validation = float(np.mean([
                    float(channel_nll(rate_state, intensity, a, index[i], val_mask[i])[0])
                    for i, a in enumerate(arrays)]))
            schedule.step(validation)
            history.append({"epoch": epoch, "train": total, "validation": validation,
                            "lr": float(optimiser.param_groups[0]["lr"])})
            print(f"[{args.arm} s{args.seed}] epoch {epoch} train {total:.5f} "
                  f"val {validation:.5f}", flush=True)
            if validation < best - 1e-5:
                best, patience = validation, 0
                best_state = ({k: v.detach().clone() for k, v in rate_state.state_dict().items()},
                              {k: v.detach().clone() for k, v in intensity.state_dict().items()})
            else:
                patience += 1
                if patience >= 5:
                    break
        if best_state is not None:
            rate_state.load_state_dict(best_state[0])
            intensity.load_state_dict(best_state[1])

        per_patient, rescaled_all = {}, []
        with torch.no_grad():
            for i, (patient, a) in enumerate(zip(patients, arrays)):
                loss, out, z, keep = channel_nll(rate_state, intensity, a, index[i],
                                                 val_mask[i])
                rescaled = out["compensator"][keep]
                rescaled = rescaled[torch.isfinite(rescaled)]
                gof = goodness_of_fit(rescaled)
                per_patient[patient.subject] = {
                    "arrival_nll_per_event": float(loss),
                    "n_validation_intervals": int(keep.sum()),
                    "n_segments": a["n_segments"],
                    "rescaled_mean": float(rescaled.mean()) if rescaled.numel() else None,
                    "rescaled_sd": float(rescaled.std()) if rescaled.numel() > 1 else None,
                    "state_norm_sd": float(z[val_mask[i]].norm(dim=-1).std())
                                     if z[val_mask[i]].numel() else 0.0,
                    "gof": gof,
                }
                rescaled_all.append(rescaled)

        payload = {
            "contract": "topic5_epi_prssm_v0_2_arrival_run",
            "job_id": key.job_id, "goal": GOAL, "arm": args.arm, "seed": args.seed,
            "cohort": args.cohort, "n_patients": len(patients),
            "subjects": [p.subject for p in patients], "dataset": dataset_of(patients),
            "config": config, "history": history, "best_validation": best,
            "markov_renewal": bool(args.markov_renewal),
            "n_gradient_steps": len(history) * len(patients),
            "per_patient": per_patient,
            "time_constants_seconds": [float(t) for t in rate_state.time_constants()],
            "state_weight_norm": float(intensity.state_weight.weight.norm()),
            "reading": "arrival_nll_per_event is the negative log-likelihood of the "
                       "observed arrival times per recorded interval; lower is better. "
                       "rescaled_mean/sd should both sit near 1 if the intensity is "
                       "correctly specified.",
            "code_revision": code_revision(), "package_hash": package_hash(),
        }
        atomic_write_json(target, payload)
        checkpoint = OUT / "checkpoints" / f"{key.job_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rate_state": rate_state.state_dict(),
                    "intensity": intensity.state_dict(), "config": config,
                    "job_id": key.job_id}, checkpoint)
        record.outputs = {"run_json": str(target), "checkpoint": str(checkpoint)}
        record.metrics = {"best_validation": best, "epochs_run": len(history)}
    print(f"COMPLETE {key.job_id}")


if __name__ == "__main__":
    main()
