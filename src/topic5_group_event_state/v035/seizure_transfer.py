"""Frozen interictal-state transfer to seizure distance and early ictal fields.

No seizure outcome updates the event encoder, state transition or contact
decoder.  Seizure labels are opened only here, after a W3 checkpoint and its
trajectory have been frozen.  All outcomes remain before the registered 80 %
boundary; the later development and sealed partitions are untouched.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import DATASET_ROOT, FORMAT_PREFIX, INPUT_ROOT, atomic_json
from .functional_readouts import (
    RIDGES, Endpoint, _block_shift, _design, _fit_masked_ridge, _fit_scaler,
    _score, _states_at_grid,
)
from .long_windows import (
    exposure_and_gap_count,
    exposure_seconds,
    matched_wrong_time_donors,
)


LEADS_SECONDS = (21600.0, 7200.0, 1800.0, 300.0)
RISK_HORIZONS_SECONDS = (300.0, 900.0, 1800.0, 3600.0, 7200.0)
HAZARD_BIN_SECONDS = 300.0
HAZARD_MAX_SECONDS = 21600.0
STATE_SHIFT_MIN_SECONDS = 1800.0
ICTAL_TARGET_ROOT = Path(os.environ.get(
    "HFOSP_RESULTS_ROOT", Path(__file__).resolve().parents[3] / "results"
)) / "topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"


def _phase(t: float, bounds: dict[str, float]) -> str:
    if t < bounds["20pct"]: return "CALIBRATION"
    if t < bounds["60pct"]: return "FIT"
    if t < bounds["70pct"]: return "INNER"
    if t < bounds["80pct"]: return "SELECTION"
    return "OUTSIDE"


def _clinical_feature(anchor_time: np.ndarray, seizures: list[dict[str, Any]]) -> np.ndarray:
    offsets = np.sort(np.asarray([float(s["offset_epoch"]) for s in seizures], dtype=np.float64))
    pos = np.searchsorted(offsets, anchor_time, side="right") - 1
    since = np.full(anchor_time.size, 30 * 86400.0, dtype=np.float64)
    ok = pos >= 0
    since[ok] = np.maximum(anchor_time[ok] - offsets[pos[ok]], 0.0)
    return np.column_stack((np.log1p(since), (since < 1800.0).astype(np.float64)))


def _hazard_rows(anchor_time: np.ndarray, segment: np.ndarray, phase: np.ndarray,
                 segment_bounds: np.ndarray, phase_hi: dict[str, float],
                 onsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors, bins, labels = [], [], []
    n_bins = int(HAZARD_MAX_SECONDS // HAZARD_BIN_SECONDS)
    for i, (t, seg, ph) in enumerate(zip(anchor_time, segment, phase)):
        if ph not in phase_hi:
            continue
        follow = min(float(segment_bounds[int(seg), 1]), phase_hi[str(ph)]) - float(t)
        if follow <= 0:
            continue
        nxt = np.searchsorted(onsets, t, side="right")
        delta = float(onsets[nxt] - t) if nxt < onsets.size else math.inf
        stop_bins = min(n_bins, int(math.ceil(min(follow, HAZARD_MAX_SECONDS) / HAZARD_BIN_SECONDS)))
        for k in range(stop_bins):
            left, right = k * HAZARD_BIN_SECONDS, (k + 1) * HAZARD_BIN_SECONDS
            # Bins use (left, right].  A seizure onset commonly defines the
            # right edge of an otherwise observed interictal segment; it is an
            # observed transition, not censoring.  The old half-open test
            # silently discarded every such transition and left all H2b
            # fitting labels at zero.
            event_in_bin = left < delta <= min(right, follow) + 1e-9
            # An incompletely observed final bin supplies no no-event evidence
            # unless its observed endpoint is the seizure itself.
            if follow < right - 1e-9 and not event_in_bin:
                break
            anchors.append(i); bins.append(k); labels.append(float(event_in_bin))
            if event_in_bin:
                break
    return np.asarray(anchors, dtype=np.int64), np.asarray(bins, dtype=np.int64), np.asarray(labels, dtype=np.float32)


def _hazard_rows_observed_support(
    anchor_time: np.ndarray, phase: np.ndarray, support_segments: np.ndarray,
    phase_hi: dict[str, float], onsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build right-censored bins while giving missing time zero weight."""

    anchors, bins, labels, weights = [], [], [], []
    n_bins = int(HAZARD_MAX_SECONDS // HAZARD_BIN_SECONDS)
    for i, (t_raw, ph_raw) in enumerate(zip(anchor_time, phase)):
        t, ph = float(t_raw), str(ph_raw)
        if ph not in phase_hi:
            continue
        nxt = np.searchsorted(onsets, t, side="right")
        delta = float(onsets[nxt] - t) if nxt < onsets.size else math.inf
        for k in range(n_bins):
            left, right = k * HAZARD_BIN_SECONDS, (k + 1) * HAZARD_BIN_SECONDS
            bin_lo = t + left
            bin_hi = min(t + right, float(phase_hi[ph]))
            if bin_hi <= bin_lo:
                break
            event_in_bin = left < delta <= (bin_hi - t) + 1e-9
            observed = exposure_seconds(support_segments, bin_lo, bin_hi)
            if event_in_bin:
                anchors.append(i); bins.append(k); labels.append(1.0); weights.append(1.0)
                break
            if observed > 0:
                anchors.append(i); bins.append(k); labels.append(0.0)
                weights.append(min(1.0, observed / HAZARD_BIN_SECONDS))
    return (
        np.asarray(anchors, dtype=np.int64), np.asarray(bins, dtype=np.int64),
        np.asarray(labels, dtype=np.float32), np.asarray(weights, dtype=np.float64),
    )


class DiscreteHazard(torch.nn.Module):
    def __init__(self, width: int, n_bins: int = 72) -> None:
        super().__init__()
        self.bin_logit = torch.nn.Parameter(torch.full((n_bins,), -5.0))
        self.beta = torch.nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        return self.bin_logit[bins] + x @ self.beta


def _fit_hazard(x: np.ndarray, anchor_rows: np.ndarray, bins: np.ndarray, y: np.ndarray,
                phase: np.ndarray, l2: float, weights: np.ndarray | None = None) -> tuple[DiscreteHazard, float]:
    rows = np.flatnonzero(phase[anchor_rows] == "FIT")
    inner = np.flatnonzero(phase[anchor_rows] == "INNER")
    if rows.size == 0 or np.sum(y[rows]) == 0:
        raise ValueError("no fitting seizure transition for discrete hazard")
    xt = torch.as_tensor(x, dtype=torch.float64)
    ar = torch.as_tensor(anchor_rows, dtype=torch.long)
    bt = torch.as_tensor(bins, dtype=torch.long)
    yt = torch.as_tensor(y, dtype=torch.float64)
    wt = torch.ones_like(yt) if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    model = DiscreteHazard(
        x.shape[1], n_bins=int(HAZARD_MAX_SECONDS // HAZARD_BIN_SECONDS)
    ).double()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=160,
                                  line_search_fn="strong_wolfe")
    ridx = torch.as_tensor(rows, dtype=torch.long)
    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = model(xt[ar[ridx]], bt[ridx])
        raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[ridx], reduction="none")
        loss = (raw * wt[ridx]).sum() / wt[ridx].sum().clamp_min(1e-8)
        loss = loss + float(l2) * model.beta.square().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        score_rows = inner if inner.size else rows
        si = torch.as_tensor(score_rows, dtype=torch.long)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            model(xt[ar[si]], bt[si]), yt[si], reduction="none",
        )
        score = (raw * wt[si]).sum() / wt[si].sum().clamp_min(1e-8)
    return model, float(score)


def _hazard_probabilities(model: DiscreteHazard, x: np.ndarray) -> np.ndarray:
    xt = torch.as_tensor(x, dtype=torch.float64)
    with torch.no_grad():
        logits = model.bin_logit[None] + xt @ model.beta[:, None]
        hazard = torch.sigmoid(logits).cpu().numpy()
    return 1.0 - np.cumprod(1.0 - hazard, axis=1)


def _person_period_log_score(
    model: DiscreteHazard, x: np.ndarray, anchor_rows: np.ndarray,
    bins: np.ndarray, y: np.ndarray, phase: np.ndarray, split: str,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score the right-censored likelihood without selecting on outcomes."""

    rows = np.flatnonzero(phase[anchor_rows] == split)
    if rows.size == 0:
        return {"status": "NOT_ESTIMABLE", "n_person_period_rows": 0}
    xt = torch.as_tensor(x, dtype=torch.float64)
    ar = torch.as_tensor(anchor_rows[rows], dtype=torch.long)
    bt = torch.as_tensor(bins[rows], dtype=torch.long)
    yt = torch.as_tensor(y[rows], dtype=torch.float64)
    wt = (
        torch.ones_like(yt) if weights is None
        else torch.as_tensor(np.asarray(weights)[rows], dtype=torch.float64)
    )
    with torch.no_grad():
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            model(xt[ar], bt), yt, reduction="none",
        )
        score = (raw * wt).sum() / wt.sum().clamp_min(1e-8)
    return {
        "status": "ESTIMATED", "log_score": float(score),
        "n_person_period_rows": int(rows.size), "n_seizure_transitions": int(np.sum(y[rows])),
        "contract": "all observed at-risk bins through seizure or censoring; no outcome-selected anchor filter",
    }


def _risk_scores(prob: np.ndarray, anchor_time: np.ndarray, rows: np.ndarray,
                 onsets: np.ndarray, segment: np.ndarray, segment_bounds: np.ndarray,
                 observation_hi: float | None = None) -> dict[str, Any]:
    out = {}
    for horizon in RISK_HORIZONS_SECONDS:
        k = int(horizon // HAZARD_BIN_SECONDS) - 1
        next_index = np.searchsorted(onsets, anchor_time[rows], side="right")
        delta = np.full(rows.size, np.inf)
        has_next = next_index < onsets.size
        delta[has_next] = onsets[next_index[has_next]] - anchor_time[rows[has_next]]
        follow_hi = segment_bounds[segment[rows], 1].astype(np.float64, copy=True)
        if observation_hi is not None:
            follow_hi = np.minimum(follow_hi, float(observation_hi))
        full_followup = anchor_time[rows] + horizon <= follow_hi + 1e-9
        observed_event = (
            (delta > 0.0) & (delta <= horizon + 1e-9)
            & (anchor_time[rows] + delta <= follow_hi + 1e-9)
        )
        eligible = rows[full_followup | observed_event]
        key = f"{int(horizon // 60)}min"
        if eligible.size == 0:
            out[key] = {"status": "NOT_ESTIMABLE", "reason": "no anchor with a determined outcome"}
            continue
        nxt = np.searchsorted(onsets, anchor_time[eligible], side="right")
        delta = np.full(eligible.size, np.inf)
        ok = nxt < onsets.size; delta[ok] = onsets[nxt[ok]] - anchor_time[eligible[ok]]
        y = (delta <= horizon).astype(np.float64)
        p = np.clip(prob[eligible, k], 1e-6, 1 - 1e-6)
        record = {
            "brier": float(np.mean((p - y) ** 2)),
            "log_score": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "n_anchors": int(eligible.size), "n_positive": int(y.sum()),
            "n_full_followup": int(full_followup.sum()),
            "n_event_only": int((observed_event & ~full_followup).sum()),
        }
        # An anchor enters through ``observed_event`` only because a seizure was
        # seen: that branch is outcome-dependent.  While full-follow-up anchors
        # dominate this is harmless, but once the horizon exceeds the remaining
        # observation in the segment/phase every survivor is a positive and the
        # Brier score is no longer a forecast score.  Withhold it and keep the
        # raw value for audit (review 2026-09-04).
        record["outcome_dependent_eligibility"] = bool(record["n_event_only"] > 0)
        if record["n_full_followup"] == 0:
            out[key] = {"status": "NOT_ESTIMABLE",
                        "reason": "every eligible anchor qualified only because its seizure was observed; "
                                  "no anchor has follow-up covering the horizon, so the sample has no structural negatives",
                        "withheld_brier": record.pop("brier"),
                        "withheld_log_score": record.pop("log_score"), **record}
        else:
            record["status"] = (
                "DESCRIPTIVE_RIGHT_CENSORED_BINARY_NOT_PRIMARY"
                if record["n_event_only"] > 0 else "ESTIMATED_COMPLETE_FOLLOWUP"
            )
            out[key] = record
    return out


def _run_hazard(anchor_time: np.ndarray, segment: np.ndarray, phase: np.ndarray,
                q: np.ndarray, state: np.ndarray, segment_bounds: np.ndarray,
                bounds: dict[str, float], seizures: list[dict[str, Any]],
                *, observed_support: bool = False,
                observed_support_bounds: np.ndarray | None = None) -> dict[str, Any]:
    onsets = np.sort(np.asarray([float(s["onset_epoch"]) for s in seizures if float(s["onset_epoch"]) < bounds["80pct"]]))
    clinical = _clinical_feature(anchor_time, seizures)
    xq = np.concatenate((clinical, q), axis=1)
    phase_hi = {"CALIBRATION": bounds["20pct"], "FIT": bounds["60pct"],
                "INNER": bounds["70pct"], "SELECTION": bounds["80pct"]}
    if observed_support:
        support = (
            segment_bounds if observed_support_bounds is None
            else np.asarray(observed_support_bounds, dtype=np.float64)
        )
        ar, bins, y, row_weights = _hazard_rows_observed_support(
            anchor_time, phase, support, phase_hi, onsets,
        )
    else:
        ar, bins, y = _hazard_rows(
            anchor_time, segment, phase, segment_bounds, phase_hi, onsets,
        )
        row_weights = np.ones(y.shape, dtype=np.float64)
    selection = np.flatnonzero(phase == "SELECTION")
    matched_donors = None
    if observed_support:
        donor_pool = np.flatnonzero(np.isin(phase, ("FIT", "INNER", "SELECTION")))
        exposure, _ = exposure_and_gap_count(
            support, anchor_time, anchor_time + float(HAZARD_MAX_SECONDS)
        )
        matched_donors = matched_wrong_time_donors(
            anchor_time, selection, donor_pool,
            minimum_time_separation=float(HAZARD_MAX_SECONDS),
            recent_rate=q[:, 0],
            exposure_fraction=exposure / float(HAZARD_MAX_SECONDS),
            n_donors=5,
        )
        matched_ok = np.all(matched_donors >= 0, axis=1)
        matched_rows = selection[matched_ok]
        shift_valid = np.zeros(anchor_time.size, dtype=bool)
        shift_valid[matched_rows] = True
        shifted = state.copy()
    else:
        shifted, shift_valid = _block_shift(
            state, anchor_time, segment, selection, STATE_SHIFT_MIN_SECONDS
        )
        matched_rows = selection[shift_valid[selection]]
    arms = {}
    fitted = {}
    # DiscreteHazard already owns one baseline logit per survival bin.  Do not
    # add the generic ridge intercept from ``_design`` here: it is exactly
    # collinear with those logits and makes the clinical/state coefficients
    # non-identifiable.
    designs = (
        ("clinical_only", clinical),
        ("q_clinical", xq),
        ("mark_state_clinical", np.concatenate((clinical, state), axis=1)),
        ("q_clinical_plus_state", np.concatenate((xq, state), axis=1)),
    )
    for name, x in designs:
        best = None
        for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            try: model, inner = _fit_hazard(x, ar, bins, y, phase, l2, row_weights)
            except ValueError: continue
            if best is None or inner < best[0]: best = (inner, l2, model)
        if best is None:
            arms[name] = {"status": "NOT_ESTIMABLE"}; continue
        prob = _hazard_probabilities(best[2], x)
        arms[name] = {"status": "ESTIMATED", "l2": best[1], "inner_person_period_logloss": best[0],
                      "selection_censored_likelihood": _person_period_log_score(
                          best[2], x, ar, bins, y, phase, "SELECTION", row_weights
                      ),
                      "selection": _risk_scores(prob, anchor_time, selection, onsets, segment,
                                                segment_bounds, bounds["80pct"])}
        fitted[name] = best[2]
    if "q_clinical_plus_state" in fitted:
        model = fitted["q_clinical_plus_state"]
        valid_rows = matched_rows
        if observed_support and matched_donors is not None and valid_rows.size:
            phase_support = phase.copy()
            phase_support[(phase == "SELECTION") & ~shift_valid] = "OUTSIDE"
            donor_scores = []
            donor_matrix = matched_donors[np.all(matched_donors >= 0, axis=1)]
            for donor_column in range(donor_matrix.shape[1]):
                donor_state = state.copy()
                donor_state[valid_rows] = state[donor_matrix[:, donor_column]]
                x_shift = np.concatenate((xq, donor_state), axis=1)
                score = _person_period_log_score(
                    model, x_shift, ar, bins, y, phase_support, "SELECTION", row_weights
                )
                if score.get("status") == "ESTIMATED":
                    donor_scores.append(float(score["log_score"]))
            correct_score = _person_period_log_score(
                model, np.concatenate((xq, state), axis=1), ar, bins, y,
                phase_support, "SELECTION", row_weights,
            )
            arms["matched_wrong_time_state"] = {
                "status": "ESTIMATED" if donor_scores else "NOT_ESTIMABLE",
                "selection_censored_likelihood": (
                    {"status": "NOT_ESTIMABLE"} if not donor_scores else {
                        "status": "ESTIMATED", "log_score": float(np.mean(donor_scores)),
                        "n_matched_anchor_rows": int(valid_rows.size),
                        "donors_per_anchor": int(donor_matrix.shape[1]),
                        "contract": "mean right-censored log score across five real matched wrong-time states",
                    }
                ),
            }
            arms["correct_state_on_matched_support"] = {
                "status": correct_score.get("status", "NOT_ESTIMABLE"),
                "selection_censored_likelihood": correct_score,
            }
        elif not observed_support:
            x_shift = np.concatenate((xq, shifted), axis=1)
            prob = _hazard_probabilities(model, x_shift)
            arms["block_shift_state"] = {"status": "ESTIMATED", "selection":
                _risk_scores(prob, anchor_time, valid_rows, onsets, segment, segment_bounds,
                             bounds["80pct"])}
            prob = _hazard_probabilities(model, np.concatenate((xq, state), axis=1))
            arms["correct_state_on_shift_support"] = {"status": "ESTIMATED", "selection":
                _risk_scores(prob, anchor_time, valid_rows, onsets, segment, segment_bounds,
                             bounds["80pct"])}
        else:
            arms["matched_wrong_time_state"] = {
                "status": "NOT_ESTIMABLE",
                "selection_censored_likelihood": {"status": "NOT_ESTIMABLE"},
            }
            arms["correct_state_on_matched_support"] = {
                "status": "NOT_ESTIMABLE",
                "selection_censored_likelihood": {"status": "NOT_ESTIMABLE"},
            }
        mean = np.nanmean(state[phase == "FIT"], axis=0, keepdims=True)
        prob = _hazard_probabilities(model, np.concatenate((xq, np.broadcast_to(mean, state.shape)), axis=1))
        arms["fit_period_mean_state"] = {"status": "ESTIMATED", "selection":
            _risk_scores(prob, anchor_time, selection, onsets, segment, segment_bounds,
                         bounds["80pct"])}
    def censored_gain(base: str, full: str) -> float | None:
        a = arms.get(base, {}).get("selection_censored_likelihood", {})
        b = arms.get(full, {}).get("selection_censored_likelihood", {})
        if a.get("status") != "ESTIMATED" or b.get("status") != "ESTIMATED":
            return None
        return float(a["log_score"] - b["log_score"])
    arms["registered_contrasts"] = {
        "primary_metric": "right-censored selection person-period log score",
        "dynamic_rate_increment": censored_gain("clinical_only", "q_clinical"),
        "mark_state_without_rate": censored_gain("clinical_only", "mark_state_clinical"),
        "mark_state_increment_over_rate": censored_gain("q_clinical", "q_clinical_plus_state"),
        "correct_time_increment_over_matched_wrong": censored_gain(
            "matched_wrong_time_state", "correct_state_on_matched_support"
        ),
    }
    seizure_phase = [_phase(float(t), bounds) for t in onsets]
    return {"arms": arms, "n_person_period_rows": int(ar.size),
            "seizures_by_phase": {p: int(seizure_phase.count(p)) for p in ("FIT", "INNER", "SELECTION")},
            "model": f"single coherent {int(HAZARD_BIN_SECONDS // 60)}-min discrete hazard; horizon risks derived from survival product",
            "observation_contract": (
                "bins may cross excluded intervals; no-event likelihood is weighted by actually observed seconds"
                if observed_support else "bins end at the current target-coverage segment"
            )}


def _feature_at_times(times: np.ndarray, segment_bounds: np.ndarray,
                      anchor_time: np.ndarray, anchor_segment: np.ndarray, q: np.ndarray,
                      trajectory: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segs = np.full(times.size, -1, dtype=np.int64)
    for seg, (lo, hi) in enumerate(segment_bounds):
        segs[(times >= lo) & (times < hi)] = seg
    state = _states_at_grid(trajectory, times, segs)
    qout = np.zeros((times.size, q.shape[1]), dtype=np.float32)
    valid = segs >= 0
    for seg in np.unique(segs[valid]):
        tr = np.flatnonzero(segs == seg); ar = np.flatnonzero(anchor_segment == seg)
        pos = np.searchsorted(anchor_time[ar], times[tr], side="right") - 1
        ok = pos >= 0; qout[tr[ok]] = q[ar[pos[ok]]]; valid[tr[~ok]] = False
    return qout, state, valid


def _field_targets(index: dict[str, Any], subject: str) -> dict[int, dict[str, Endpoint]]:
    root = DATASET_ROOT / subject
    scalars = dict(np.load(root / "scalars.npz"))
    part = np.load(root / index["arrays"]["participation"]["file"], mmap_mode="r")
    delay = np.load(root / index["arrays"]["relative_delay"]["file"], mmap_mode="r")
    t = np.asarray(scalars["t_abs"], dtype=np.float64)
    labels = np.asarray([str(c["detector_label"]) for c in index["contacts"]])
    out: dict[int, dict[str, Endpoint]] = {}
    exact_path = ICTAL_TARGET_ROOT / f"{subject}.npz"
    exact = np.load(exact_path, allow_pickle=False) if exact_path.exists() else None
    exact_lookup = ({str(v): i for i, v in enumerate(exact["channels"].astype(str))} if exact is not None else {})
    join = np.asarray([exact_lookup.get(v, -1) for v in labels], dtype=np.int64)
    for si, seizure in enumerate(index.get("seizures", [])):
        onset = float(seizure["onset_epoch"])
        rows = np.flatnonzero((t >= onset) & (t < onset + 10.0))
        current: dict[str, Endpoint] = {}
        if rows.size:
            pp = np.asarray(part[rows], dtype=np.float32)
            burden = pp.mean(0, keepdims=True)
            current["early_ictal_group_event_field"] = Endpoint(
                "early_ictal_group_event_field", burden, np.ones_like(burden, dtype=bool),
            )
            arrival = np.full((1, labels.size), np.nan, dtype=np.float32)
            for c in range(labels.size):
                use = rows[np.asarray(part[rows, c], dtype=bool)]
                if use.size:
                    arrival[0, c] = float(np.min(t[use] - onset + np.asarray(delay[use, c])))
            current["early_ictal_arrival_field"] = Endpoint(
                "early_ictal_arrival_field", arrival, np.isfinite(arrival),
            )
        # The ictal cache is keyed by the seizure's index in the unfiltered
        # source inventory.  Keep that identity explicit: using the position
        # in a filtered list silently attaches the wrong spatial field as soon
        # as an excluded seizure is not a chronological suffix.
        source_index = int(seizure.get("_source_index", si))
        key = f"bb150_auc__{source_index}"
        if exact is not None and key in exact.files and np.sum(join >= 0) >= 3:
            values = np.full((1, labels.size), np.nan, dtype=np.float32)
            ok = join >= 0; values[0, ok] = np.asarray(exact[key], dtype=np.float32)[join[ok]]
            current["early_ictal_bb150_energy_field"] = Endpoint(
                "early_ictal_bb150_energy_field", values, np.isfinite(values),
            )
        if current: out[si] = current
    if exact is not None: exact.close()
    return out


def _seizure_field_readouts(subject: str, index: dict[str, Any], bounds: dict[str, float],
                            segment_bounds: np.ndarray, anchor_time: np.ndarray,
                            anchor_segment: np.ndarray, q: np.ndarray,
                            trajectory: dict[str, np.ndarray]) -> dict[str, Any]:
    targets = _field_targets(index, subject)
    results = {}
    for lead in LEADS_SECONDS:
        samples = []
        for si, endpoint_map in targets.items():
            seizure = index["seizures"][si]; onset = float(seizure["onset_epoch"])
            ph = _phase(onset, bounds)
            if ph not in ("FIT", "INNER", "SELECTION"): continue
            samples.append((si, onset - lead, ph, endpoint_map))
        if not samples:
            results[f"lead_{int(lead // 60)}min"] = {"status": "NOT_ESTIMABLE"}; continue
        times = np.asarray([v[1] for v in samples], dtype=np.float64)
        qq, ss, valid_time = _feature_at_times(times, segment_bounds, anchor_time, anchor_segment, q, trajectory)
        clinical = _clinical_feature(times, index["seizures"])
        qq_full = np.concatenate((clinical, qq), axis=1)
        lead_result = {}
        endpoint_names = sorted(set().union(*(v[3].keys() for v in samples)))
        for endpoint_name in endpoint_names:
            example = next(v[3][endpoint_name] for v in samples if endpoint_name in v[3])
            y = np.full((len(samples), example.values.shape[1]), np.nan, dtype=np.float32)
            valid = np.zeros_like(y, dtype=bool)
            for i, sample in enumerate(samples):
                if endpoint_name in sample[3] and valid_time[i]:
                    y[i] = sample[3][endpoint_name].values[0]
                    valid[i] = sample[3][endpoint_name].valid[0]
            phases = np.asarray([v[2] for v in samples])
            fit, inner, sel = (np.flatnonzero(phases == p) for p in ("FIT", "INNER", "SELECTION"))
            if fit.size < 5 or sel.size == 0:
                lead_result[endpoint_name] = {"status": "NOT_ESTIMABLE", "n_fit_seizures": int(fit.size),
                                              "n_selection_seizures": int(sel.size)}; continue
            endpoint = Endpoint(endpoint_name, y, valid)
            shifted, shift_valid = _block_shift(ss, times, np.zeros(times.size, int), sel, max(lead, 1800.0))
            # Seizure samples are sparse; if a strict time shift has no donor it is reported, never zero-filled.
            result = {}
            fitted: dict[str, np.ndarray] = {}
            centre, scale = _fit_scaler(y, valid, fit, False)
            designs = (
                ("clinical_only", _design(clinical, None)),
                ("q_clinical", _design(qq_full, None)),
                ("mark_state_clinical", _design(clinical, ss)),
                ("q_clinical_plus_state", _design(qq_full, ss)),
            )
            for name, x in designs:
                best = None
                for alpha in RIDGES:
                    coef = _fit_masked_ridge(x, y, valid, fit, alpha, centre, scale)
                    rows = inner if inner.size else fit
                    loss, n = _score(x @ coef, y, valid, rows, centre, scale, False)
                    if loss is not None and (best is None or loss < best[0]): best = (loss, alpha, coef)
                if best is None: result[name] = {"status": "NOT_ESTIMABLE"}; continue
                loss, n = _score(x @ best[2], y, valid, sel, centre, scale, False)
                result[name] = {"selection_loss": loss, "n_values": n, "alpha": best[1]}
                fitted[name] = best[2]
            if "q_clinical_plus_state" in fitted:
                coef = fitted["q_clinical_plus_state"]
                shifted_rows = sel[shift_valid[sel]]
                shifted_loss, shifted_n = _score(
                    _design(qq_full, shifted) @ coef, y, valid, shifted_rows,
                    centre, scale, False,
                )
                correct_support_loss, correct_support_n = _score(
                    _design(qq_full, ss) @ coef, y, valid, shifted_rows,
                    centre, scale, False,
                )
                result["block_shift_state"] = {
                    "selection_loss": shifted_loss,
                    "n_values": shifted_n,
                    "alpha": result["q_clinical_plus_state"]["alpha"],
                }
                result["correct_state_on_shift_support"] = {
                    "selection_loss": correct_support_loss,
                    "n_values": correct_support_n,
                    "alpha": result["q_clinical_plus_state"]["alpha"],
                }
                valid_fit = fit[valid_time[fit]]
                mean_state = (
                    np.nanmean(ss[valid_fit], axis=0, keepdims=True)
                    if valid_fit.size else np.zeros((1, ss.shape[1]), dtype=ss.dtype)
                )
                constant_state = np.broadcast_to(mean_state, ss.shape)
                constant_loss, constant_n = _score(
                    _design(qq_full, constant_state) @ coef, y, valid, sel,
                    centre, scale, False,
                )
                result["fit_period_mean_state"] = {
                    "selection_loss": constant_loss,
                    "n_values": constant_n,
                    "alpha": result["q_clinical_plus_state"]["alpha"],
                }
            qloss = result.get("q_clinical", {}).get("selection_loss")
            sloss = result.get("q_clinical_plus_state", {}).get("selection_loss")
            result["state_gain_over_q"] = None if qloss is None or sloss is None else qloss - sloss
            shift_loss = result.get("block_shift_state", {}).get("selection_loss")
            correct_support_loss = result.get("correct_state_on_shift_support", {}).get("selection_loss")
            period_loss = result.get("fit_period_mean_state", {}).get("selection_loss")
            result["correct_time_gain_over_shift"] = (
                None if shift_loss is None or correct_support_loss is None
                else shift_loss - correct_support_loss
            )
            result["mark_gain_over_period_mean"] = (
                None if period_loss is None or sloss is None else period_loss - sloss
            )
            clinical_loss = result.get("clinical_only", {}).get("selection_loss")
            q_only_loss = result.get("q_clinical", {}).get("selection_loss")
            mark_only_loss = result.get("mark_state_clinical", {}).get("selection_loss")
            result["registered_contrasts"] = {
                "rate_gain_over_clinical": None if clinical_loss is None or q_only_loss is None else clinical_loss - q_only_loss,
                "mark_gain_over_clinical": None if clinical_loss is None or mark_only_loss is None else clinical_loss - mark_only_loss,
                "mark_gain_over_rate": None if q_only_loss is None or sloss is None else q_only_loss - sloss,
            }
            result["support"] = {"n_fit_seizures": int(fit.size), "n_inner_seizures": int(inner.size),
                                 "n_selection_seizures": int(sel.size),
                                 "block_shift_donors": int(shift_valid[sel].sum())}
            lead_result[endpoint_name] = result
        results[f"lead_{int(lead // 60)}min"] = lead_result
    return results


def run_seizure_transfer(subject: str, trajectory_path: Path, rate_path: Path,
                         *, out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite: return json.loads(card_path.read_text(encoding="utf-8"))
    index = json.loads((DATASET_ROOT / subject / "index.json").read_text(encoding="utf-8"))
    manifest = json.loads((INPUT_ROOT / subject / "manifest_v3.json").read_text(encoding="utf-8"))
    bounds = {k: float(v) for k, v in manifest["report"]["phase_boundaries_epoch"].items()}
    with np.load(manifest["input_path"], allow_pickle=False) as z:
        segment_bounds = np.asarray(z["target_segment_bounds"], dtype=np.float64)
    with np.load(rate_path, allow_pickle=False) as z:
        anchor_time = np.asarray(z["anchor_time"], dtype=np.float64)
        segment = np.asarray(z["segment"], dtype=np.int64)
        phase = np.asarray(z["phase"]).astype(str)
        q = np.asarray(z["q_standardized"], dtype=np.float32)
        window_contract = (
            str(np.asarray(z["window_contract"]).item())
            if "window_contract" in z.files else "same_segment_complete"
        )
        if "segment_bounds" in z.files:
            segment_bounds = np.asarray(z["segment_bounds"], dtype=np.float64)
        observed_support_bounds = (
            np.asarray(z["observed_support_bounds"], dtype=np.float64)
            if "observed_support_bounds" in z.files else segment_bounds
        )
        if "phase_boundaries_json" in z.files:
            bounds = {
                k: float(v) for k, v in
                json.loads(str(np.asarray(z["phase_boundaries_json"]).item())).items()
            }
    # Do not even materialise outcome targets beyond the registered 80 %
    # boundary.  Filtering only after loading the ictal fields would violate
    # the nested-time contract even if those rows were later discarded.
    analysis_index = dict(index)
    analysis_index["seizures"] = [
        {**seizure, "_source_index": source_index}
        for source_index, seizure in enumerate(index.get("seizures", []))
        if float(seizure["onset_epoch"]) < bounds["80pct"]
    ]
    with np.load(trajectory_path, allow_pickle=False) as z:
        trajectory = {k: np.asarray(z[k]) for k in z.files}
    state = _states_at_grid(trajectory, anchor_time, segment)
    hazard = _run_hazard(
        anchor_time, segment, phase, q, state, segment_bounds, bounds,
        analysis_index["seizures"], observed_support=window_contract == "observed_support",
        observed_support_bounds=observed_support_bounds,
    )
    fields = _seizure_field_readouts(subject, analysis_index, bounds, segment_bounds, anchor_time,
                                     segment, q, trajectory)
    card = {
        "format": f"{FORMAT_PREFIX}_frozen_seizure_transfer_v1", "subject": subject,
        "state_trajectory": str(trajectory_path), "rate_trajectory": str(rate_path),
        "distance_survival": hazard, "early_ictal_field_and_path": fields,
        "frozen_contract": "event encoder, state dynamics and contact decoder never see seizure outcomes",
        "outcome_boundary": "seizure onsets before registered 80pct boundary only",
        "n_seizures_materialised": len(analysis_index["seizures"]),
        "long_horizon_contract": {
            "risk_horizons_seconds": list(RISK_HORIZONS_SECONDS),
            "early_field_leads_seconds": list(LEADS_SECONDS),
            "hazard_bin_seconds": HAZARD_BIN_SECONDS,
            "hazard_max_seconds": HAZARD_MAX_SECONDS,
            "state_shift_min_seconds": STATE_SHIFT_MIN_SECONDS,
        },
        "development_targets_read": False, "sealed_partition_opened": False,
        "seizure_outcomes_read": True,
    }
    atomic_json(card_path, card); return card
