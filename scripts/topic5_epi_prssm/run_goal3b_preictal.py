#!/usr/bin/env python3
"""Goal 3b / primary H2b -- does the frozen state move once the pre-ictal IEDs are observed?

One job per patient.  The frozen model is applied causally to the *admissible*
stream: every rebuilt event except those inside a seizure or its 120 min post-ictal
guard.  The onset time is used only to place the window and to score; it never
enters the model.

Three readings are produced per seizure and never pooled:

``filtered_at_onset``   the observer consumed every admissible event up to onset --
                        the most informative reading an online system could hold
``filtered_at_cutoff``  the observer stopped at onset minus the declared lead
``open_loop_at_onset``  the observer stopped at the cut-off and the generator then
                        integrated autonomously to onset

Every reading is z-scored against pseudo-onsets matched inside the same patient on
observation coverage, multi-scale rate, median interval, day/night and session, and
is additionally reported after residualising on those same nuisances -- Topic 2
already shows the event rate itself drifts slowly and rises around seizures, so an
unresidualised effect is not evidence for a spatial-repertoire state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_json, code_revision,
    is_complete, package_hash, sha256_obj, torch,
)

import numpy as np  # noqa: E402

from src.topic5_epi_prssm.evaluate import PROBE_ENDPOINTS, probe_summary  # noqa: E402
from src.topic5_epi_prssm.model import EpiPRSSM, build_cohort_batch  # noqa: E402
from src.topic5_epi_prssm.preictal_stream import (  # noqa: E402
    POST_ICTAL_GUARD_SECONDS, build_admissible_stream, multiscale_rate, observation_coverage,
)
from src.topic5_epi_prssm.seizure_labels import load_seizures, require_freeze  # noqa: E402

GOAL = "goal3b_preictal"
OUT = OUTPUT_ROOT / "seizure_link_preictal"
ADDENDUM = OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE_ADDENDUM_GOAL3B.json"

MEDIAN_IEI_WINDOW = 7200.0
#: pseudo-onsets whose log median interval is further than this many pool standard
#: deviations from the real onset are not admissible partners
MEDIAN_IEI_CALIPER = 0.75
#: A caliper that can never bind is not a caliper.  Requiring the full pseudo-onset
#: count inside it left 72% of seizures on the soft fallback, and the balance did not
#: improve -- the median-interval imbalance stayed at -0.71 z with 22 of 27 patients
#: on the same side.  Accepting a smaller but genuinely tempo-matched set is the point
#: of matching; the reduced count is recorded per seizure.
MEDIAN_IEI_MIN_PSEUDO = 40
COVERAGE_WINDOW = 7200.0
N_PSEUDO = 200
PSEUDO_GRID_SECONDS = 120.0
#: Pseudo cut-offs must sit away from every real seizure.  A fixed wide window is
#: unusable for a patient with dozens of seizures -- the exclusion would cover the
#: whole record -- so the window relaxes in declared steps and the level actually
#: used is written into every row.
PERI_ICTAL_EXCLUSION_LADDER = (4 * 3600.0, 2 * 3600.0, 3600.0, 1800.0)
#: The premise of this arm is that the observer has seen pre-ictal events.  A window
#: with almost none puts us back in the strict arm's situation, so those seizures are
#: reported as their own stratum instead of being pooled or silently dropped.
MIN_LOOKBACK_EVENTS_PRIMARY = 5
NULL_SD_RELATIVE_FLOOR = 1e-4
NULL_SD_ABSOLUTE_FLOOR = 1e-6
NUISANCE_KEYS = ("rate_1800s", "rate_7200s", "rate_14400s", "rate_28800s",
                 "median_iei", "coverage", "log_anchor_gap")


def require_addendum() -> dict:
    if not ADDENDUM.exists():
        raise SystemExit(f"{ADDENDUM} missing: write the Goal 3b freeze addendum first")
    return json.loads(ADDENDUM.read_text())


def robust_z(observed: float, null: np.ndarray) -> tuple[float, bool]:
    mean, sd = float(np.mean(null)), float(np.std(null))
    floor = max(NULL_SD_ABSOLUTE_FLOOR, NULL_SD_RELATIVE_FLOOR * abs(mean))
    if not np.isfinite(sd) or sd < floor:
        return float("nan"), True
    return (observed - mean) / sd, False


@torch.no_grad()
def causal_states(model: EpiPRSSM, patient, chunk: int = 1024):
    """Post-event slow state after every admissible event."""
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
    return torch.stack(states), torch.cat(resources), batch


@torch.no_grad()
def integrate_to(model: EpiPRSSM, patient, batch, state: torch.Tensor,
                 resource: torch.Tensor, elapsed: np.ndarray) -> dict[str, np.ndarray]:
    """Autonomous integration by ``elapsed`` seconds from an observer-updated state."""
    n = len(elapsed)
    padded = torch.zeros(n, batch.n_pad, model.state_dim)
    padded[:, : patient.n_contacts] = state
    dt = torch.as_tensor(np.maximum(elapsed, 0.0), dtype=torch.float32)
    node_mask = batch.node_mask.expand(n, -1)
    adjacency = batch.adjacency.expand(n, -1, -1, -1)
    moved_resource = model.resource.propagate(resource.clone(), padded, dt, node_mask)
    if model.unconstrained_gru:
        moved = padded * torch.exp(-torch.clamp(dt / 300.0, max=40.0)).view(-1, 1, 1)
    else:
        moved = model.generator.propagate(padded, dt, adjacency, moved_resource, node_mask)
    return probe_summary(model, patient, moved[:, : patient.n_contacts], moved_resource)


def nuisance_at(event_time: np.ndarray, probe: float, anchor_gap: float) -> dict[str, float]:
    row = multiscale_rate(event_time, probe)
    row["coverage"] = observation_coverage(event_time, probe, COVERAGE_WINDOW)
    row["log_anchor_gap"] = float(np.log1p(max(anchor_gap, 0.0)))
    return row


def matched_pseudo(event_time: np.ndarray, session_index: np.ndarray, onsets: np.ndarray,
                   target_time: float, target_nuisance: dict, target_session: int,
                   target_day_night: str, dataset: str, lead: float,
                   rng) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pseudo cut-offs inside the same patient, matched on observability and rate."""
    from src.topic5_epi_prssm.seizure_labels import (
        DAY_END_HOUR, DAY_START_HOUR, TIMEZONE_OFFSET_HOURS)
    base = np.arange(event_time[0] + lead + 600.0, event_time[-1], PSEUDO_GRID_SECONDS)
    if len(base) == 0:
        return np.zeros(0), np.zeros(0, dtype=int), {"exclusion_seconds": None}
    grid, exclusion = np.zeros(0), None
    for candidate in PERI_ICTAL_EXCLUSION_LADDER:
        far = np.ones(len(base), dtype=bool)
        for onset in onsets:
            far &= np.abs(base - onset) > candidate
        if far.sum() >= 60:
            grid, exclusion = base[far], candidate
            break
    if exclusion is None:
        return np.zeros(0), np.zeros(0, dtype=int), {"exclusion_seconds": None}
    provenance = {"exclusion_seconds": exclusion,
                  "n_grid_candidates": int(len(grid)),
                  "relaxed": bool(exclusion != PERI_ICTAL_EXCLUSION_LADDER[0])}
    anchor = np.searchsorted(event_time, grid, side="right") - 1
    keep = anchor >= 0
    grid, anchor = grid[keep], anchor[keep]
    gap = grid - event_time[anchor]
    offset = TIMEZONE_OFFSET_HOURS[dataset] * 3600.0
    hours = ((grid + offset) / 3600.0) % 24.0
    day_night = np.where((hours >= DAY_START_HOUR) & (hours < DAY_END_HOUR), "day", "night")
    same_session = session_index[anchor] == target_session
    pool = np.flatnonzero((day_night == target_day_night) & same_session)
    if len(pool) < 20:
        pool = np.flatnonzero(day_night == target_day_night)
    if len(pool) < 20:
        pool = np.arange(len(grid))
    cost = np.zeros(len(pool))
    for key in ("rate_1800s", "rate_7200s", "rate_14400s", "rate_28800s"):
        window = float(key.split("_")[1].rstrip("s"))
        lo = np.searchsorted(event_time, grid[pool] - window, side="left")
        hi = np.searchsorted(event_time, grid[pool], side="right")
        values = (hi - lo) / (window / 3600.0)
        scale = np.std(values) + 1e-6
        cost += np.abs(values - target_nuisance[key]) / scale
    coverage = np.array([observation_coverage(event_time, t, COVERAGE_WINDOW) for t in grid[pool]])
    cost += 3.0 * np.abs(coverage - target_nuisance["coverage"])
    cost += np.abs(np.log1p(gap[pool]) - target_nuisance["log_anchor_gap"])

    # The frozen addendum matches on the typical interval as well as on rate.  It was
    # missing from the cost, and it is exactly the quantity that already differs at
    # onset -- the interval shortens before a seizure in 22 of 27 patients -- so
    # leaving it out lets a pre-ictal difference in tempo masquerade as a state effect.
    median_iei = np.full(len(pool), np.nan)
    for i, t in enumerate(grid[pool]):
        lo = np.searchsorted(event_time, t - MEDIAN_IEI_WINDOW, side="left")
        hi = np.searchsorted(event_time, t, side="right")
        if hi - lo >= 3:
            median_iei[i] = float(np.median(np.diff(event_time[lo:hi])))
    target_iei = target_nuisance.get("median_iei")
    iei_ok = np.isfinite(median_iei) & np.isfinite(np.asarray(target_iei, dtype=float))
    if target_iei is not None and np.isfinite(target_iei) and iei_ok.any():
        log_iei = np.log1p(np.where(iei_ok, median_iei, np.nan))
        scale = np.nanstd(log_iei) + 1e-6
        penalty = np.abs(log_iei - np.log1p(target_iei)) / scale
        # a caliper, not only a soft cost: a pseudo-onset whose tempo is more than
        # this far from the real one is not an admissible match at all
        inside = np.where(np.isfinite(penalty), penalty <= MEDIAN_IEI_CALIPER, False)
        provenance["median_iei_matched"] = True
        provenance["n_inside_iei_caliper"] = int(inside.sum())
        if inside.sum() >= MEDIAN_IEI_MIN_PSEUDO:
            pool, cost = pool[inside], cost[inside]
            penalty = penalty[inside]
            provenance["median_iei_caliper_applied"] = True
            provenance["n_pseudo_after_caliper"] = int(inside.sum())
        else:
            # too few admissible partners: fall back to the soft cost and say so, rather
            # than silently matching on a quantity that was supposed to be balanced
            provenance["median_iei_caliper_applied"] = False
            provenance["median_iei_fallback_reason"] = (
                f"only {int(inside.sum())} of {len(pool)} pseudo-onsets fell inside the "
                f"caliper, fewer than the {MEDIAN_IEI_MIN_PSEUDO} required")
        cost = cost + np.nan_to_num(penalty, nan=float(np.nanmax(penalty) if
                                                       np.isfinite(penalty).any() else 0.0))
    else:
        provenance["median_iei_matched"] = False
        provenance["median_iei_fallback_reason"] = "target or pool interval undefined"

    order = pool[np.argsort(cost)][:N_PSEUDO]
    provenance["pool_after_day_night_and_session"] = int(len(pool))
    provenance["n_selected"] = int(len(order))
    return grid[order], anchor[order], provenance


def run_subject(subject: str, layer: str, lead_minutes: float, *,
                overwrite: bool = False) -> Path:
    base = require_freeze()
    require_addendum()
    entry = next((r for r in base["representatives"]
                  if r["layer"] == layer and r["status"] == "FROZEN"), None)
    if entry is None:
        raise SystemExit(f"layer {layer!r} is not frozen")
    lead = float(lead_minutes) * 60.0
    key = JobKey(goal=GOAL, family=entry["arm"], arm=f"{layer}_lead{int(lead_minutes)}m",
                 seed=int(entry["seed"]), split="frozen_interictal", cohort=subject,
                 config_hash=sha256_obj({"lead": lead, "layer": layer})[:16],
                 input_hash=entry.get("checkpoint_sha256", "")[:16],
                 code_revision=package_hash()[:16])
    target = OUT / "per_subject" / f"{subject}__{layer}__lead{int(lead_minutes)}m.json"
    if target.exists() and is_complete(key) and not overwrite:
        print(f"SKIPPED_EXISTING {key.job_id}")
        return target

    with JobRunner(key) as record:
        payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        model = EpiPRSSM(**payload["spec"], feature_dim=payload.get("feature_dim", 6))
        model.load_state_dict(payload["state_dict"])
        model.eval()

        seizures = load_seizures(subject)
        intervals = [(s.onset_epoch, s.offset_epoch) for s in seizures]
        stream = build_admissible_stream(subject, intervals)
        patient = stream.tensors
        event_time = stream.event_time
        session_index = np.asarray(patient.meta["session_index"])
        inside = [s for s in seizures
                  if event_time[0] + lead < s.onset_epoch <= event_time[-1]]
        result = {
            "contract": "topic5_epi_prssm_v0_1_goal3b_preictal",
            "subject": subject, "dataset": patient.dataset, "layer": layer,
            "frozen_arm": entry["arm"], "frozen_job_id": entry["job_id"],
            "lead_minutes": lead_minutes,
            "stream": {
                "n_events_full": stream.n_events_full,
                "n_events_admissible": stream.n_events_admissible,
                "n_events_removed_ictal_or_postictal": stream.n_events_removed_ictal_or_postictal,
                "n_events_beyond_definite_interictal": stream.n_events_beyond_definite_interictal,
                "post_ictal_guard_seconds": POST_ICTAL_GUARD_SECONDS,
            },
            "n_seizures_total": len(seizures), "n_seizures_in_span": len(inside),
            "code_revision": code_revision(), "package_hash": package_hash(),
        }
        if not inside:
            result["status"] = "NOT_OBSERVABLE_FROM_CURRENT_STREAM"
            result["reason"] = "no seizure onset falls inside the admissible event span"
            atomic_write_json(target, result)
            record.outputs = {"per_subject": str(target)}
            return target

        states, resources, batch = causal_states(model, patient)
        # Every known seizure, not only the ones whose onset falls inside the
        # admissible event span.  The ladder below uses these to keep pseudo
        # cut-offs away from ictal time; restricting it to in-span onsets lets a
        # seizure just outside the span leave genuinely pre-ictal moments in the
        # matched null, which biases the effect toward zero.
        onsets = np.array([s.onset_epoch for s in seizures])
        rng = np.random.default_rng(FROZEN["bootstrap_seed"])
        rows, n_eligible = [], 0
        for i, seizure in enumerate(inside):
            onset = seizure.onset_epoch
            cutoff = onset - lead
            anchor_cut = int(np.searchsorted(event_time, cutoff, side="right") - 1)
            anchor_onset = int(np.searchsorted(event_time, onset, side="right") - 1)
            if anchor_cut < 0:
                continue
            gap = float(cutoff - event_time[anchor_cut])
            nuisance = nuisance_at(event_time, cutoff, gap)
            pseudo_times, pseudo_anchor, matching = matched_pseudo(
                event_time, session_index, onsets, cutoff, nuisance,
                int(session_index[anchor_cut]), seizure.day_night, patient.dataset, lead, rng)
            if len(pseudo_times) < 20:
                continue
            n_eligible += 1
            real = {
                "filtered_at_cutoff": integrate_to(
                    model, patient, batch, states[[anchor_cut]], resources[[anchor_cut]],
                    np.array([gap])),
                "open_loop_at_onset": integrate_to(
                    model, patient, batch, states[[anchor_cut]], resources[[anchor_cut]],
                    np.array([onset - event_time[anchor_cut]])),
                "filtered_at_onset": integrate_to(
                    model, patient, batch, states[[anchor_onset]], resources[[anchor_onset]],
                    np.array([onset - event_time[anchor_onset]])),
            }
            pseudo_gap = pseudo_times - event_time[pseudo_anchor]
            pseudo_onset_equivalent = pseudo_times + lead
            pseudo_anchor_onset = np.searchsorted(
                event_time, pseudo_onset_equivalent, side="right") - 1
            valid_onset_anchor = pseudo_anchor_onset >= 0
            null = {
                "filtered_at_cutoff": integrate_to(
                    model, patient, batch, states[pseudo_anchor], resources[pseudo_anchor],
                    pseudo_gap),
                "open_loop_at_onset": integrate_to(
                    model, patient, batch, states[pseudo_anchor], resources[pseudo_anchor],
                    pseudo_gap + lead),
                "filtered_at_onset": integrate_to(
                    model, patient, batch, states[pseudo_anchor_onset[valid_onset_anchor]],
                    resources[pseudo_anchor_onset[valid_onset_anchor]],
                    pseudo_onset_equivalent[valid_onset_anchor]
                    - event_time[pseudo_anchor_onset[valid_onset_anchor]]),
            }
            row = {
                "subject": subject, "dataset": patient.dataset,
                "seizure_id": seizure.seizure_id, "onset_epoch": onset,
                "onset_kind": seizure.onset_kind, "day_night": seizure.day_night,
                "lead_minutes": lead_minutes,
                "anchor_gap_to_cutoff_seconds": gap,
                "n_events_in_lookback_2h": int(
                    np.searchsorted(event_time, cutoff, "right")
                    - np.searchsorted(event_time, cutoff - COVERAGE_WINDOW, "left")),
                "n_pseudo": int(len(pseudo_times)),
                "pseudo_exclusion_seconds": matching["exclusion_seconds"],
                "pseudo_exclusion_relaxed": matching["relaxed"],
                "n_pseudo_grid_candidates": matching["n_grid_candidates"],
                # the caliper's own evidence, persisted so a downstream reader can
                # check that the balance was enforced rather than assume it
                "median_iei_matched": bool(matching.get("median_iei_matched", False)),
                "median_iei_caliper_applied": bool(
                    matching.get("median_iei_caliper_applied", False)),
                "n_inside_iei_caliper": matching.get("n_inside_iei_caliper"),
                "median_iei_fallback_reason": matching.get("median_iei_fallback_reason"),
                **{f"nuisance_{k}": float(nuisance[k]) for k in NUISANCE_KEYS},
            }
            for reading, values in real.items():
                for endpoint in PROBE_ENDPOINTS:
                    observed = float(values[endpoint][0])
                    z, degenerate = robust_z(observed, null[reading][endpoint])
                    row[f"{reading}__{endpoint}"] = observed
                    row[f"{reading}__{endpoint}_z"] = z
                    row[f"{reading}__{endpoint}_degenerate"] = degenerate
                    row[f"{reading}__{endpoint}_null_sd"] = float(
                        np.std(null[reading][endpoint]))
            # the nuisance set's own discriminability, so a state claim must beat it
            for key in ("rate_1800s", "rate_7200s", "rate_14400s", "rate_28800s",
                        "median_iei", "coverage"):
                # The same pseudo set the endpoint z uses.  matched_pseudo
                # returns partners sorted by matching cost, so taking the first
                # 60 gave the nuisance benchmark a tighter, lower-variance null
                # than the state readout got and inflated its |z|; the two nulls
                # have to be the same set for the comparison to mean anything.
                pseudo_values = np.array([
                    nuisance_at(event_time, t, g)[key]
                    for t, g in zip(pseudo_times, pseudo_gap)])
                z, degenerate = robust_z(nuisance[key], pseudo_values)
                row[f"nuisanceonly__{key}_z"] = z
                row[f"nuisanceonly__{key}_degenerate"] = degenerate
            n_lookback = row["n_events_in_lookback_2h"]
            row["lookback_stratum"] = ("ge20" if n_lookback >= 20 else
                                       "5to19" if n_lookback >= 5 else
                                       "1to4" if n_lookback >= 1 else "none")
            row["preictal_observation_premise_met"] = bool(
                n_lookback >= MIN_LOOKBACK_EVENTS_PRIMARY and gap <= lead)
            rows.append(row)

        result["status"] = "ok" if rows else "NOT_OBSERVABLE_FROM_CURRENT_STREAM"
        if not rows:
            result["reason"] = ("no seizure had both an admissible anchor before the cut-off and "
                                "at least 20 matched pseudo cut-offs")
        result["n_seizures_eligible"] = n_eligible
        result["n_seizures_premise_met"] = int(sum(
            1 for r in rows if r.get("preictal_observation_premise_met")))
        result["premise_rule"] = (
            f"at least {MIN_LOOKBACK_EVENTS_PRIMARY} admissible events in the 2 h before the "
            "cut-off and an anchor no further from the cut-off than the lead itself; a seizure "
            "that fails this is reported in its own stratum, not dropped")
        result["per_seizure"] = rows
        atomic_write_json(target, result)
        record.outputs = {"per_subject": str(target)}
        record.metrics = {"n_seizures_eligible": n_eligible,
                          "n_events_admissible": stream.n_events_admissible}
    print(f"COMPLETE {subject}: {n_eligible} eligible seizures "
          f"({stream.n_events_beyond_definite_interictal} events beyond the frozen stream)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--lead-minutes", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_subject(args.subject, args.layer, args.lead_minutes, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
