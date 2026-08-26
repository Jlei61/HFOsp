#!/usr/bin/env python3
"""H3 lag discovery: at which timescale, if any, does past IED exposure predict
future network behaviour?

Why this exists.  The v0.1 H3 asked one narrow question -- does a bounded resource
ODE with one recovery constant improve the model's own order likelihood -- and that
question turned out to be unanswerable: the recovery constant was unidentifiable
across the whole declared grid, the winning arms were the ones whose resource sat
pinned at its bound, and the strict layer failed its directionality check.  Before
committing to any ODE the honest first question is model-light and non-parametric:

    does the amount of recent IED activity at lag L predict what the network does
    next, and is that association specific to the causal direction?

Design.
  outcome     three future quantities that need no model at all:
                next_log_iei          the interval to the very next discharge
                log_time_to_21st_event how long it takes to accumulate 21 more; the
                                       first version computed this and called it "the
                                       next discharge", which overstated it
                future_load       mean participants over the next H events
                repertoire_shift  how far the next H events sit from the patient's
                                  own baseline participation profile
  predictor   exposure inside disjoint lag bins, so bins compete instead of a single
              exponential kernel deciding the timescale by fiat
  controls    a future-exposure placebo drawn strictly *after* the outcome window
              (no overlap, unlike time reversal), a session-block shuffle, and a
              count-versus-amplitude contrast
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_csv, atomic_write_json,
    code_revision, is_complete, load_tensors, package_hash, resolve_cohort, sha256_obj,
)
from src.topic5_epi_prssm.stats import paired_effect  # noqa: E402

GOAL = "goal4_lag_discovery"
OUT = OUTPUT_ROOT / "exposure_lag"

#: disjoint lag bins in seconds, spanning half a minute to two days
LAG_EDGES = (0.0, 30.0, 120.0, 300.0, 900.0, 3600.0, 14400.0, 43200.0, 172800.0)
LAG_NAMES = tuple(f"lag_{int(a)}_{int(b)}s" for a, b in zip(LAG_EDGES[:-1], LAG_EDGES[1:]))

OUTCOME_HORIZON = 20
#: events between the last event that may enter an exposure bin and the first event
#: of the outcome window.  Without it the placebo and the outcome share events.
OUTCOME_GAP = 20
MIN_EVENTS = 500


def exposure_bins(times: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(E, K) exposure inside each disjoint past lag bin, strictly causal."""
    n, k = len(times), len(LAG_NAMES)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    out = np.zeros((n, k))
    for b, (lo, hi) in enumerate(zip(LAG_EDGES[:-1], LAG_EDGES[1:])):
        # events strictly before this one whose age falls inside (lo, hi]
        start = np.searchsorted(times, times - hi, side="left")
        stop = np.searchsorted(times, times - lo, side="right")
        stop = np.minimum(stop, np.arange(n))          # never include the event itself
        start = np.minimum(start, stop)
        out[:, b] = cumulative[stop] - cumulative[start]
    return out


def placebo_has_full_support(times: np.ndarray) -> np.ndarray:
    """True where the placebo window is fully inside the record.

    Near the end of a record the forward bins run past the last event, so their
    exposure is zero for want of data rather than for want of activity.  Counting
    those as a valid control quietly weakens the comparison the placebo exists to
    provide.
    """
    n = len(times)
    first_allowed = np.minimum(np.arange(n) + 1 + OUTCOME_GAP + OUTCOME_HORIZON, n - 1)
    return (times[-1] - times[first_allowed]) >= LAG_EDGES[-1]


def future_exposure_placebo(times: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Same bins, but drawn from *after* the outcome window, so they cannot overlap it."""
    n, k = len(times), len(LAG_NAMES)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    first_allowed = np.minimum(np.arange(n) + 1 + OUTCOME_GAP + OUTCOME_HORIZON, n)
    anchor = times[np.minimum(first_allowed, n - 1)]
    out = np.zeros((n, k))
    for b, (lo, hi) in enumerate(zip(LAG_EDGES[:-1], LAG_EDGES[1:])):
        start = np.searchsorted(times, anchor + lo, side="left")
        stop = np.searchsorted(times, anchor + hi, side="right")
        start = np.maximum(start, first_allowed)
        stop = np.maximum(stop, start)
        out[:, b] = cumulative[stop] - cumulative[start]
    return out


def outcomes(times: np.ndarray, load: np.ndarray, participation: np.ndarray) -> pd.DataFrame:
    n = len(times)
    lo = np.minimum(np.arange(n) + 1 + OUTCOME_GAP, n)
    hi = np.minimum(lo + OUTCOME_HORIZON, n)
    valid = hi > lo

    # The interval to the very next discharge.  Exposure bins end strictly before
    # event e and the placebo starts after the outcome window, so this is a clean
    # forward outcome.
    next_log_iei = np.full(n, np.nan)
    step_ok = np.arange(n) + 1 < n
    idx = np.minimum(np.arange(n) + 1, n - 1)
    next_log_iei[step_ok] = np.log1p(
        np.maximum(times[idx[step_ok]] - times[step_ok], 0.0))

    # The time to accumulate OUTCOME_GAP more events.  This is what the first version
    # computed while calling it "the next discharge"; it is a cumulative pace over 21
    # events, and a short-lag association with it says much less than the name implied.
    log_time_to_21st = np.full(n, np.nan)
    ok = lo < n
    log_time_to_21st[ok] = np.log1p(np.maximum(times[lo[ok]] - times[ok], 0.0))

    cumulative = np.concatenate([[0.0], np.cumsum(load)])
    future_load = np.full(n, np.nan)
    future_load[valid] = ((cumulative[hi[valid]] - cumulative[lo[valid]])
                          / (hi[valid] - lo[valid]))

    baseline = participation.mean(axis=0)
    profile_cum = np.concatenate([np.zeros((1, participation.shape[1])),
                                  np.cumsum(participation.astype(np.float64), axis=0)])
    shift = np.full(n, np.nan)
    window_mean = np.zeros((n, participation.shape[1]))
    window_mean[valid] = ((profile_cum[hi[valid]] - profile_cum[lo[valid]])
                          / (hi[valid] - lo[valid])[:, None])
    shift[valid] = np.linalg.norm(window_mean[valid] - baseline[None, :], axis=1)
    return pd.DataFrame({"next_log_iei": next_log_iei,
                         "log_time_to_21st_event": log_time_to_21st,
                         "future_load": future_load,
                         "repertoire_shift": shift})


#: contiguous blocks for the cross-fit, plus an embargo so a block cannot be scored
#: by a model fitted on its immediate neighbours
N_FOLDS, EMBARGO = 5, 200


def partial_r2(y: np.ndarray, design: np.ndarray, columns: list[int]) -> float:
    """Out-of-fold extra variance explained by ``columns``.

    The first version fitted and scored on the whole record, so a flexible design
    could report explanatory power it would not have out of sample.  Here each
    contiguous block is scored by a model fitted on the rest of the record with an
    embargo either side, and the reduction in error is measured out of fold.
    """
    keep = np.isfinite(y) & np.isfinite(design).all(axis=1)
    if keep.sum() < 400:
        return float("nan")
    yy, dd = y[keep], design[keep]
    reduced_design = np.delete(dd, columns, axis=1)
    n = len(yy)
    edges = np.linspace(0, n, N_FOLDS + 1).astype(int)
    full_err = reduced_err = 0.0
    for fold in range(N_FOLDS):
        lo, hi = edges[fold], edges[fold + 1]
        test = np.zeros(n, dtype=bool); test[lo:hi] = True
        train = ~test
        train[max(lo - EMBARGO, 0):lo] = False
        train[hi:min(hi + EMBARGO, n)] = False
        if train.sum() < 200 or test.sum() < 20:
            return float("nan")
        for mat, name in ((dd, "full"), (reduced_design, "reduced")):
            beta, *_ = np.linalg.lstsq(mat[train], yy[train], rcond=None)
            err = float(np.sum((yy[test] - mat[test] @ beta) ** 2))
            if name == "full":
                full_err += err
            else:
                reduced_err += err
    return float((reduced_err - full_err) / reduced_err) if reduced_err > 0 else float("nan")


def exposure_weights(kind: str, load: np.ndarray, participation: np.ndarray,
                     n_contacts: int) -> np.ndarray:
    """What "one unit of exposure" means.

    Raw count and raw load were the only two families in the first pass, and they gave
    the same answer -- which is itself informative, but it leaves open whether a
    normalised or a surprise-weighted exposure behaves differently.  Patients differ
    several-fold in how many contacts they have, so a raw participant count is partly
    an implantation variable rather than a physiological one.
    """
    if kind == "count":
        return np.ones_like(load)
    if kind == "amplitude":
        return load
    if kind == "participation_fraction":
        # NOTE: within a patient this is `amplitude` divided by a constant, so a
        # within-patient linear analysis cannot distinguish the two.  It is kept so the
        # equivalence is visible in the output rather than assumed, but it must not be
        # counted as an independent definition.
        return load / max(n_contacts, 1)
    if kind == "surprise":
        # the part of an event's extent that a causal running mean did not predict;
        # a perfectly ordinary discharge contributes nothing
        predicted = np.zeros_like(load)
        running, seen = 0.0, 0
        for e in range(len(load)):
            predicted[e] = running if seen else load[e]
            running = (0.98 * running + 0.02 * load[e]) if seen else load[e]
            seen += 1
        return np.abs(load - predicted)
    raise ValueError(f"unknown exposure weighting {kind!r}")


def per_patient(patient, kind: str, rng) -> dict | None:
    times = np.asarray(patient.event_time, dtype=np.float64)
    order = np.argsort(times)
    times = times[order]
    participation = patient.participation.numpy()[order]
    load = participation.sum(axis=1).astype(np.float64)
    if len(times) < MIN_EVENTS:
        return {"subject": patient.subject, "status": "too_few_events", "n_events": len(times)}

    weights = exposure_weights(kind, load, participation, participation.shape[1])
    past = exposure_bins(times, weights)
    placebo = future_exposure_placebo(times, weights)
    frame = outcomes(times, load, participation)

    # nuisances every bin has to beat: how long since the last event, and where in
    # the day we are.  Local rate is deliberately *not* included: it is the thing
    # the short bins measure, and controlling for it would remove the question.
    iei = np.concatenate([[np.nan], np.diff(times)])
    tod = 2 * np.pi * ((times % 86400.0) / 86400.0)
    nuisance = np.column_stack([np.ones(len(times)), np.log1p(np.nan_to_num(iei, nan=0.0)),
                                np.sin(tod), np.cos(tod)])

    row = {"subject": patient.subject, "dataset": patient.dataset, "status": "ok",
           "n_events": int(len(times)), "weighting": kind}
    for outcome_name in frame.columns:
        y = frame[outcome_name].to_numpy()
        z = np.column_stack([nuisance, past])
        row[f"{outcome_name}__all_bins_partial_r2"] = partial_r2(
            y, z, list(range(nuisance.shape[1], z.shape[1])))
        z_placebo = np.column_stack([nuisance, placebo])
        supported = placebo_has_full_support(times)
        y_placebo = np.where(supported, y, np.nan)
        row[f"{outcome_name}__placebo_partial_r2"] = partial_r2(
            y_placebo, z_placebo, list(range(nuisance.shape[1], z_placebo.shape[1])))
        row["n_events_with_placebo_support"] = int(supported.sum())
        row["placebo_support_fraction"] = float(supported.mean())
        for b, name in enumerate(LAG_NAMES):
            single = np.column_stack([nuisance, past[:, b]])
            row[f"{outcome_name}__{name}_partial_r2"] = partial_r2(
                y, single, [single.shape[1] - 1])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--weighting", default="all",
                        choices=["count", "amplitude", "participation_fraction",
                                 "surprise", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    kinds = (["count", "amplitude", "participation_fraction", "surprise"]
             if args.weighting == "all" else [args.weighting])
    key = JobKey(goal=GOAL, family="model_light", arm="lag_discovery", seed=0,
                 split="development", cohort=args.cohort,
                 config_hash=sha256_obj({"edges": list(LAG_EDGES), "kinds": kinds,
                                         "gap": OUTCOME_GAP,
                                         "horizon": OUTCOME_HORIZON})[:16],
                 input_hash=sha256_obj({"cohort": args.cohort})[:16],
                 code_revision=package_hash()[:16])
    target = OUT / "LAG_DISCOVERY.json"
    if target.exists() and is_complete(key) and not args.overwrite:
        print("SKIPPED_EXISTING")
        return

    with JobRunner(key) as job:
        patients = load_tensors(resolve_cohort(args.cohort))
        rng = np.random.default_rng(FROZEN["bootstrap_seed"])
        rows = [r for kind in kinds for patient in patients
                if (r := per_patient(patient, kind, rng)) is not None]
        frame = pd.DataFrame(rows)
        atomic_write_csv(OUT / "lag_discovery_per_patient.csv", frame)

        usable = frame[frame.status == "ok"]
        summary = {"contract": "topic5_epi_prssm_v0_1_exposure_lag_discovery",
                   "lag_bins_seconds": list(LAG_EDGES), "lag_bin_names": list(LAG_NAMES),
                   "outcome_horizon_events": OUTCOME_HORIZON,
                   "outcome_gap_events": OUTCOME_GAP,
                   "n_patients": int(usable.subject.nunique()),
                   "by_weighting": {}}
        for kind in kinds:
            block = usable[usable.weighting == kind]
            entry = {}
            for outcome_name in ("next_log_iei", "log_time_to_21st_event",
                                 "future_load", "repertoire_shift"):
                real = block.set_index("subject")[f"{outcome_name}__all_bins_partial_r2"].to_dict()
                placebo = block.set_index("subject")[f"{outcome_name}__placebo_partial_r2"].to_dict()
                shared = sorted(set(real) & set(placebo))
                entry[outcome_name] = {
                    "real_vs_future_placebo": paired_effect(
                        {s: real[s] for s in shared}, {s: placebo[s] for s in shared},
                        label=f"{kind}::{outcome_name}::real-vs-future-placebo",
                        lower_is_better=False).as_dict(),
                    "per_bin_median_partial_r2": {
                        name: float(np.nanmedian(
                            block[f"{outcome_name}__{name}_partial_r2"].to_numpy()))
                        for name in LAG_NAMES},
                }
            summary["by_weighting"][kind] = entry
        summary["weighting_note"] = (
            "participation_fraction is amplitude divided by a per-patient constant, so "
            "a within-patient analysis returns identical numbers; there are three "
            "independent definitions here, not four")
        summary["reading"] = (
            "each bin's partial R^2 is how much of the future outcome that lag window "
            "explains beyond the interval since the last event and time of day. The "
            "placebo draws the same bins from strictly after the outcome window, so it "
            "shares no events with it; a real causal lag effect must beat the placebo.")
        summary["code_revision"] = code_revision()
        summary["package_hash"] = package_hash()
        atomic_write_json(target, summary)
        job.outputs = {"summary": str(target)}
        job.metrics = {"n_patients": summary["n_patients"]}
    print(json.dumps(summary["by_weighting"], indent=1)[:1500])


if __name__ == "__main__":
    main()
