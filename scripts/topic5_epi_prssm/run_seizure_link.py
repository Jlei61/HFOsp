#!/usr/bin/env python3
"""Goal 3 sensitivity -- the definite-interictal, long-gap strict arm.

**This is not the primary H2b.**  It runs on the ``dataset_v0_4`` definite-interictal
stream, whose block policy deletes every block that overlaps a seizure or its 120 min
post-ictal guard, that crosses a local day/night boundary, or that sits beside a
recording discontinuity.  The pre-ictal observations an online system would have had
are therefore missing, and the last admissible event before an onset is typically
hours earlier.  What this arm actually asks is:

    can a state inferred hours ago still be read at onset?

which is a strict missing-observation and long-extrapolation control, not the
question H2b was written to ask.  The primary H2b lives in
``run_goal3b_preictal.py``, which observes the pre-ictal IEDs and closes the
observer at a declared lead time.

Original docstring follows.

Goal 3 / H2b -- does the frozen interictal state move before a seizure?

The model is frozen before any label is read (Hard Gate B, enforced in
``seizure_labels.require_freeze``).  At the last interictal event before a target
time the observer is closed and the generator integrates autonomously to that
time on real elapsed time alone; the same probe is applied to matched
pseudo-onsets so that "the state moves" is measured against "the state looks like
this at a matched moment when no seizure followed".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_csv, atomic_write_json,
    code_revision, dataset_of, is_complete, load_tensors, package_hash,
    resolve_cohort, sha256_obj, torch,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.topic5_epi_prssm.evaluate import PROBE_ENDPOINTS, probe_summary  # noqa: E402
from src.topic5_epi_prssm.model import EpiPRSSM, build_cohort_batch  # noqa: E402
from src.topic5_epi_prssm.rollout import cohort_scan  # noqa: E402
from src.topic5_epi_prssm.seizure_labels import load_seizures, require_freeze  # noqa: E402
from src.topic5_epi_prssm.stats import paired_effect  # noqa: E402

GOAL = "goal3_seizure_link"
OUT = OUTPUT_ROOT / "seizure_link"

#: primary, pre-registered in INTERICTAL_MODEL_FREEZE.json
MAX_GAP = FROZEN["max_last_ied_to_onset_seconds"]
#: The interictal stream is built from fail-closed definite-interictal blocks, so
#: blocks near a seizure are excluded by construction and the last interictal event
#: is typically hours before onset.  A wider secondary window is therefore declared
#: on the *gap distribution*, which is a property of the data, not on any outcome.
EXTENDED_GAP_SECONDS = 86400.0
PERI_ICTAL_EXCLUSION = 2 * FROZEN["preictal_window_seconds"]
N_PSEUDO = FROZEN["pseudo_onset_draws"]
RATE_WINDOW_SECONDS = 1800.0
#: A probe whose matched-null spread is at float noise carries no information: at
#: that gap every trajectory, real and null, has already relaxed to the same rest
#: value.  Dividing by that spread would manufacture a z from rounding error, so
#: the probe is marked degenerate and its z is withheld.
NULL_SD_RELATIVE_FLOOR = 1e-4
NULL_SD_ABSOLUTE_FLOOR = 1e-6
#: last-event gap strata, reported separately because the fitted generator time
#: constants are far shorter than the typical gap
GAP_STRATA = ((0.0, 60.0, "le_60s"), (60.0, 300.0, "le_300s"),
              (300.0, 900.0, "le_900s"), (900.0, 3600.0, "le_3600s"))
#: peri-onset probe grid; each offset is probed from its own last interictal event,
#: so a point at -1800 s is not an extrapolation of the point at 0 s
TRAJECTORY_OFFSETS = (-1800.0, -1200.0, -900.0, -600.0, -300.0, -120.0, -60.0, 0.0)
N_TRAJECTORY_PSEUDO = 40


def build_model(entry: dict, feature_dim: int) -> EpiPRSSM:
    payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    model = EpiPRSSM(feature_dim=payload.get("feature_dim", feature_dim), **payload["spec"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def filtered_states(model: EpiPRSSM, patient, chunk: int = 512) -> torch.Tensor:
    """Post-event slow state after every event in the patient's whole stream."""
    batch = build_cohort_batch([patient], [0], [patient.n_events])
    z = model.initial_state(batch)
    states, resources = [], []
    position = 0
    while position < patient.n_events:
        end = min(position + chunk, patient.n_events)
        step = batch.gather(position, end)
        for t in range(end - position):
            z = model.propagate(z, batch, step, t)
            z = model.absorb(z, step, t)
            z, _, _ = model.observe(z, batch, step, t)
            states.append(z.state[0, : patient.n_contacts].clone())
            resources.append(z.resource.clone())
        position = end
    return torch.stack(states), torch.cat(resources)


@torch.no_grad()
def probe_times(model: EpiPRSSM, patient, post_states: torch.Tensor,
                post_resources: torch.Tensor, times: np.ndarray, anchors: np.ndarray,
                batch: int = 512) -> dict[str, np.ndarray]:
    """Integrate autonomously from each anchor event to its target time, then summarise."""
    gaps = times - patient.event_time[anchors]
    out: dict[str, list[np.ndarray]] = {}
    single = build_cohort_batch([patient], [0], [patient.n_events])
    for start in range(0, len(times), batch):
        stop = min(start + batch, len(times))
        index = anchors[start:stop]
        state = post_states[index].clone()
        resource = post_resources[index].clone()
        dt = torch.as_tensor(gaps[start:stop], dtype=torch.float32)
        adjacency = single.adjacency.expand(len(index), -1, -1, -1)
        node_mask = single.node_mask.expand(len(index), -1)
        padded = torch.zeros(len(index), single.n_pad, model.state_dim)
        padded[:, : patient.n_contacts] = state
        resource = model.resource.propagate(resource, padded, dt, node_mask)
        if model.unconstrained_gru:
            moved = padded * torch.exp(-torch.clamp(dt / 300.0, max=40.0)).view(-1, 1, 1)
        else:
            moved = model.generator.propagate(padded, dt, adjacency, resource, node_mask)
        summary = probe_summary(model, patient, moved[:, : patient.n_contacts], resource)
        for key, values in summary.items():
            out.setdefault(key, []).append(values)
    return {k: np.concatenate(v) for k, v in out.items()}


def gap_stratum(gap: float) -> str:
    for low, high, name in GAP_STRATA:
        if low <= gap <= high:
            return name
    return "gt_3600s"


def robust_z(observed: float, null: np.ndarray) -> tuple[float, bool]:
    """z against a matched null, withheld when the null spread is at float noise."""
    mean, sd = float(null.mean()), float(null.std())
    floor = max(NULL_SD_ABSOLUTE_FLOOR, NULL_SD_RELATIVE_FLOOR * abs(mean))
    if not np.isfinite(sd) or sd < floor:
        return float("nan"), True
    return (observed - mean) / sd, False


def anchor_for(times: np.ndarray, event_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last event at or before each target time, and the gap to it."""
    position = np.searchsorted(event_time, times, side="right") - 1
    valid = position >= 0
    gap = np.full(len(times), np.inf)
    gap[valid] = times[valid] - event_time[position[valid]]
    return position, gap


def local_rate(event_time: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Events per hour in the window ending at each target time."""
    lo = np.searchsorted(event_time, times - RATE_WINDOW_SECONDS, side="left")
    hi = np.searchsorted(event_time, times, side="right")
    return (hi - lo) / (RATE_WINDOW_SECONDS / 3600.0)


def pseudo_onsets(patient, onsets: np.ndarray, day_night: np.ndarray, rng) -> dict[int, np.ndarray]:
    """Matched control times inside the same patient, one pool per real onset."""
    event_time = patient.event_time
    grid = np.arange(event_time[0] + 600.0, event_time[-1], 60.0)
    if len(grid) == 0:
        return {}
    far = np.ones(len(grid), dtype=bool)
    for onset in onsets:
        far &= np.abs(grid - onset) > PERI_ICTAL_EXCLUSION
    grid = grid[far]
    if len(grid) < 20:
        return {}
    position, gap = anchor_for(grid, event_time)
    usable = (position >= 0) & (gap <= MAX_GAP)
    grid, gap = grid[usable], gap[usable]
    if len(grid) < 20:
        return {}
    from src.topic5_epi_prssm.seizure_labels import TIMEZONE_OFFSET_HOURS, DAY_START_HOUR, DAY_END_HOUR
    offset = TIMEZONE_OFFSET_HOURS[patient.dataset] * 3600.0
    hours = ((grid + offset) / 3600.0) % 24.0
    grid_day = np.where((hours >= DAY_START_HOUR) & (hours < DAY_END_HOUR), "day", "night")
    grid_rate = local_rate(event_time, grid)
    out: dict[int, np.ndarray] = {}
    for i, onset in enumerate(onsets):
        pool = np.flatnonzero(grid_day == day_night[i])
        if len(pool) < 20:
            pool = np.arange(len(grid))
        target_rate = local_rate(event_time, np.array([onset]))[0]
        _, target_gap = anchor_for(np.array([onset]), event_time)
        cost = (np.abs(grid_rate[pool] - target_rate) / (grid_rate.std() + 1e-6)
                + np.abs(np.log1p(gap[pool]) - np.log1p(target_gap[0])))
        order = pool[np.argsort(cost)]
        take = order[: max(N_PSEUDO, 20)]
        out[i] = grid[take]
    return out


def run(layer: str, cohort: str, *, overwrite: bool = False,
        max_gap_seconds: float | None = None) -> Path:
    global MAX_GAP
    MAX_GAP = float(max_gap_seconds) if max_gap_seconds is not None else MAX_GAP
    tag = "primary" if MAX_GAP == FROZEN["max_last_ied_to_onset_seconds"] else f"gap{int(MAX_GAP)}"
    freeze = require_freeze()
    entry = next((r for r in freeze["representatives"]
                  if r["layer"] == layer and r["status"] == "FROZEN"), None)
    if entry is None:
        raise SystemExit(f"layer {layer!r} is not frozen in INTERICTAL_MODEL_FREEZE.json")

    patients = load_tensors(resolve_cohort(cohort))
    key = JobKey(goal=GOAL, family=entry["arm"], arm=layer, seed=int(entry["seed"]),
                 split="frozen_interictal", cohort=cohort,
                 config_hash=sha256_obj({"layer": layer, "job": entry["job_id"],
                                        "max_gap": MAX_GAP})[:16],
                 input_hash=entry.get("checkpoint_sha256", "")[:16],
                 code_revision=package_hash()[:16])
    target = OUT / "runs" / f"{key.job_id}.json"
    if target.exists() and is_complete(key) and not overwrite:
        print(f"SKIPPED_EXISTING {key.job_id}")
        return target

    with JobRunner(key) as record:
        rng = np.random.default_rng(FROZEN["bootstrap_seed"])
        model = build_model(entry, patients[0].node_features.shape[-1])
        seizure_rows, patient_rows, all_trajectory = [], [], []
        for patient in patients:
            seizures = load_seizures(patient.subject)
            event_time = patient.event_time
            inside = [s for s in seizures if event_time[0] <= s.onset_epoch <= event_time[-1]]
            if not inside:
                continue
            onsets = np.array([s.onset_epoch for s in inside])
            _, gaps = anchor_for(onsets, event_time)
            eligible = gaps <= MAX_GAP
            if eligible.sum() == 0:
                patient_rows.append({
                    "subject": patient.subject, "dataset": patient.dataset,
                    "status": "no_eligible_seizure",
                    "n_seizures_inside_recorded_span": len(inside),
                    "n_seizures_eligible": 0,
                    "median_last_event_gap_all_seconds": float(np.median(gaps[np.isfinite(gaps)]))
                        if np.isfinite(gaps).any() else float("nan"),
                    "fraction_eligible": 0.0})
                continue
            n_inside = len(inside)
            gaps_all = gaps.copy()
            inside = [s for s, ok in zip(inside, eligible) if ok]
            onsets = onsets[eligible]
            day_night = np.array([s.day_night for s in inside])
            post_states, post_resources = filtered_states(model, patient)
            validation_end = float(event_time[int(torch.nonzero(patient.split == 1).max())])

            anchors, gaps = anchor_for(onsets, event_time)
            real = probe_times(model, patient, post_states, post_resources, onsets, anchors)
            pools = pseudo_onsets(patient, onsets, day_night, rng)
            if not pools:
                patient_rows.append({"subject": patient.subject, "status": "no_pseudo_pool",
                                     "n_seizures_eligible": len(onsets)})
                continue
            trajectory_rows = []
            for i, seizure in enumerate(inside):
                pseudo_times = pools[i]
                pseudo_anchor, _ = anchor_for(pseudo_times, event_time)
                pseudo = probe_times(model, patient, post_states, post_resources,
                                     pseudo_times, pseudo_anchor)
                row = {"subject": patient.subject, "dataset": patient.dataset,
                       "seizure_id": seizure.seizure_id, "onset_epoch": seizure.onset_epoch,
                       "onset_kind": seizure.onset_kind, "day_night": seizure.day_night,
                       "last_event_gap_seconds": float(gaps[i]),
                       "n_pseudo": len(pseudo_times),
                       "in_primary_window": bool(seizure.onset_epoch <= validation_end),
                       "local_rate_per_hour": float(local_rate(event_time, np.array([seizure.onset_epoch]))[0])}
                row["gap_stratum"] = gap_stratum(float(gaps[i]))
                for endpoint in PROBE_ENDPOINTS:
                    null = pseudo[endpoint]
                    observed = float(real[endpoint][i])
                    value, degenerate = robust_z(observed, null)
                    row[f"{endpoint}_onset"] = observed
                    row[f"{endpoint}_pseudo_mean"] = float(null.mean())
                    row[f"{endpoint}_pseudo_sd"] = float(null.std())
                    row[f"{endpoint}_z"] = value
                    row[f"{endpoint}_degenerate"] = degenerate
                    row[f"{endpoint}_percentile"] = (float("nan") if degenerate
                                                     else float((null < observed).mean()))
                seizure_rows.append(row)
                # --- peri-onset trajectory, real and matched null ------------
                for offset in TRAJECTORY_OFFSETS:
                    probe_time = np.array([seizure.onset_epoch + offset])
                    anchor, gap = anchor_for(probe_time, event_time)
                    if anchor[0] < 0 or gap[0] > MAX_GAP:
                        continue
                    point = probe_times(model, patient, post_states, post_resources,
                                        probe_time, anchor)
                    null_times = pseudo_times[:N_TRAJECTORY_PSEUDO] + offset
                    null_anchor, null_gap = anchor_for(null_times, event_time)
                    keep = (null_anchor >= 0) & (null_gap <= MAX_GAP)
                    point_row = {"subject": patient.subject, "dataset": patient.dataset,
                                 "seizure_id": seizure.seizure_id,
                                 "offset_seconds": float(offset),
                                 "in_primary_window": bool(seizure.onset_epoch <= validation_end)}
                    if keep.sum() >= 5:
                        null_point = probe_times(model, patient, post_states, post_resources,
                                                 null_times[keep], null_anchor[keep])
                        for endpoint in PROBE_ENDPOINTS:
                            value, degenerate = robust_z(float(point[endpoint][0]),
                                                         null_point[endpoint])
                            point_row[f"{endpoint}"] = float(point[endpoint][0])
                            point_row[f"{endpoint}_null_mean"] = float(null_point[endpoint].mean())
                            point_row[f"{endpoint}_null_sd"] = float(null_point[endpoint].std())
                            point_row[f"{endpoint}_z"] = value
                            point_row[f"{endpoint}_degenerate"] = degenerate
                        point_row["n_null"] = int(keep.sum())
                        trajectory_rows.append(point_row)
            all_trajectory.extend(trajectory_rows)
            patient_rows.append({
                "subject": patient.subject, "dataset": patient.dataset, "status": "ok",
                "n_seizures_inside_recorded_span": int(n_inside),
                "n_seizures_eligible": int(len(onsets)),
                "median_last_event_gap_all_seconds": float(np.median(gaps_all[np.isfinite(gaps_all)]))
                    if np.isfinite(gaps_all).any() else float("nan"),
                "median_last_event_gap_eligible_seconds": float(np.median(gaps)),
                "fraction_eligible": float(len(onsets) / max(n_inside, 1))})

        frame = pd.DataFrame(seizure_rows)
        atomic_write_csv(OUT / f"seizure_aligned_states__{layer}.csv", frame)
        atomic_write_csv(OUT / f"peri_onset_trajectory__{layer}.csv",
                         pd.DataFrame(all_trajectory))
        atomic_write_csv(OUT / f"seizure_denominators__{layer}__{tag}.csv",
                         pd.DataFrame(patient_rows))
        summary = _summarise(frame, layer, entry)
        assert isinstance(target, Path), f"output path was shadowed: {type(target)}"
        atomic_write_json(target, summary)
        record.outputs = {"summary": str(target),
                          "per_seizure_csv": str(OUT / f"seizure_aligned_states__{layer}__{tag}.csv")}
        record.metrics = {"n_seizures": len(frame),
                          "n_patients": int(frame["subject"].nunique()) if len(frame) else 0}
    print(f"COMPLETE {key.job_id}: {len(frame)} seizures")
    return target


def _summarise(frame: pd.DataFrame, layer: str, entry: dict) -> dict:
    out = {
        "contract": "topic5_epi_prssm_v0_1_seizure_link",
        "role": "definite_interictal_long_gap_strict_sensitivity",
        "not_primary_h2b_because": (
            "the definite-interictal block policy deletes the pre-ictal observations, so this "
            "arm measures how long a state survives without observation, not whether the state "
            "moves once the pre-ictal IEDs are observed"),
        "primary_h2b_lives_in": "scripts/topic5_epi_prssm/run_goal3b_preictal.py",
        "layer": layer, "frozen_arm": entry["arm"], "frozen_job_id": entry["job_id"],
        "n_seizures": int(len(frame)),
        "n_patients": int(frame["subject"].nunique()) if len(frame) else 0,
        "code_revision": code_revision(), "package_hash": package_hash(),
        "early_ictal_transfer": {"status": "NOT_RUN",
                                 "reason": "adjudicated per-seizure clinical-onset contacts are "
                                           "0 of 71 and substitutions are forbidden by a LOCKED "
                                           "blinding contract"},
    }
    if len(frame) == 0:
        out["status"] = "NO_ELIGIBLE_SEIZURE"
        return out
    out["degeneracy"] = {
        endpoint: {
            "n_degenerate": int(frame.get(f"{endpoint}_degenerate", pd.Series(dtype=bool)).sum()),
            "n_total": int(len(frame)),
            "note": "a degenerate probe is one whose matched null had no spread left: at that "
                    "last-event gap every trajectory, real and null, had already relaxed to the "
                    "same rest value, so no z exists",
        } for endpoint in PROBE_ENDPOINTS}
    out["by_gap_stratum"] = {}
    if "gap_stratum" in frame:
        for stratum, group in frame.groupby("gap_stratum"):
            block = {"n_seizures": int(len(group)),
                     "n_patients": int(group["subject"].nunique())}
            for endpoint in PROBE_ENDPOINTS:
                column = f"{endpoint}_z"
                values = group[column].dropna() if column in group else pd.Series(dtype=float)
                if values.empty:
                    continue
                per_patient = group.dropna(subset=[column]).groupby("subject")[column].median()
                block[endpoint] = paired_effect(per_patient.to_dict(),
                                                {s: 0.0 for s in per_patient.index},
                                                label=f"{stratum}::{endpoint}",
                                                lower_is_better=False).as_dict()
            out["by_gap_stratum"][stratum] = block
    for window, subset in (("primary_validation_window", frame[frame.in_primary_window]),
                           ("secondary_all_seizures", frame)):
        block = {"n_seizures": int(len(subset)),
                 "n_patients": int(subset["subject"].nunique()) if len(subset) else 0}
        for endpoint in PROBE_ENDPOINTS:
            column = f"{endpoint}_z"
            if column not in subset or subset[column].dropna().empty:
                continue
            per_patient = subset.dropna(subset=[column]).groupby("subject")[column].median().to_dict()
            zeros = {s: 0.0 for s in per_patient}
            effect = paired_effect(per_patient, zeros, label=f"{endpoint}_z",
                                   lower_is_better=False)
            block[endpoint] = effect.as_dict()
            block[f"{endpoint}_leave_seizure_out"] = _leave_seizure_out(subset, column)
        out[window] = block
    out["nuisance_increment"] = _nuisance_increment(frame)
    out["time_in_warning"] = _time_in_warning(frame)
    return out


def _leave_seizure_out(frame: pd.DataFrame, column: str) -> dict:
    """Patient-level effect recomputed with one seizure dropped at a time."""
    medians = []
    for drop in range(len(frame)):
        subset = frame.drop(frame.index[drop])
        if subset.empty:
            continue
        per_patient = subset.groupby("subject")[column].median()
        medians.append(float(per_patient.median()))
    if not medians:
        return {"status": "insufficient"}
    return {"n_folds": len(medians), "min_median": float(np.min(medians)),
            "max_median": float(np.max(medians)), "median_of_medians": float(np.median(medians)),
            "sign_stable": bool(np.sign(np.min(medians)) == np.sign(np.max(medians)))}


def _nuisance_increment(frame: pd.DataFrame) -> dict:
    """Does the state add anything beyond rate, gap and time of day?

    Each seizure already carries a within-patient z against matched pseudo-onsets,
    so the nuisance question is whether that z is explained by the nuisances that
    the matching could not fully equalise.
    """
    out = {}
    covariates = ["local_rate_per_hour", "last_event_gap_seconds"]
    for endpoint in PROBE_ENDPOINTS:
        column = f"{endpoint}_z"
        data = frame[[column, "day_night", *covariates]].dropna()
        if len(data) < 12:
            continue
        design = np.column_stack([
            np.ones(len(data)),
            (data["day_night"] == "day").astype(float).to_numpy(),
            np.log1p(data["local_rate_per_hour"].to_numpy()),
            np.log1p(data["last_event_gap_seconds"].to_numpy()),
        ])
        target = data[column].to_numpy()
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        out[endpoint] = {
            "n": int(len(data)),
            "raw_mean_z": float(target.mean()),
            "intercept_after_nuisance": float(coefficients[0]),
            "residual_mean": float(residual.mean()),
            "day_coefficient": float(coefficients[1]),
            "log_rate_coefficient": float(coefficients[2]),
            "log_gap_coefficient": float(coefficients[3]),
        }
    return out


def _time_in_warning(frame: pd.DataFrame, threshold: float = 0.9) -> dict:
    out = {}
    for endpoint in PROBE_ENDPOINTS:
        column = f"{endpoint}_percentile"
        if column not in frame:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        out[endpoint] = {
            "fraction_onsets_above_threshold": float((values >= threshold).mean()),
            "expected_under_null": float(1.0 - threshold),
            "n": int(len(values)),
            "median_percentile": float(values.median()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", required=True)
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-gap-seconds", type=float, default=None)
    args = parser.parse_args()
    run(args.layer, args.cohort, overwrite=args.overwrite,
        max_gap_seconds=args.max_gap_seconds)


if __name__ == "__main__":
    main()
