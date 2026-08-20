"""MODEL_ICTAL_ELIGIBLE_V2: did the model enter the intended dynamical endpoint?

This layer answers one question and reads no patient quantity at all. It is a
versioned addition: ``src.topic4_runaway_morphology.classify_sustained_runaway``
stays exactly as it was, because every historical candidate table was produced
under that contract and rewriting it would silently restate old results.

Three differences from the historical classifier are deliberate and are the
reason a new function exists:

* the recruitment clause is JOINT and duty-based. The historical rule asked for
  ``q05(F_E) >= 0.5`` and ``q05(F_sheet) >= 0.5`` separately, which is a 95% duty
  on each margin; V2 asks that 80% of windows satisfy both AT THE SAME TIME.
* the rate and frequency references are ``t_base = [500,1000] ms``, not the
  500 ms immediately before onset. With ``tau_z = 5000 ms`` the pre-onset window
  already sits inside the slow buildup, so referencing it understates the change.
* the population-frequency clause is dropped from the verdict and kept as a
  diagnostic.

Spectral estimation, paired-window construction and the recruitment traces are
taken from the frozen morphology module rather than re-derived here.
"""
from __future__ import annotations

import numpy as np

from src.topic4_runaway_morphology import (
    contact_oscillation_metrics, population_rate_frequency_metrics)

ELIGIBLE = "MODEL_ICTAL_ELIGIBLE_V2"
NOT_ELIGIBLE = "MODEL_ICTAL_NOT_ELIGIBLE_V2"
NOT_EVALUABLE = "NOT_EVALUABLE_FROM_EXISTING_ARTIFACTS"

RECRUITMENT_WINDOW_MS = 20.0


class NotEvaluableError(RuntimeError):
    """The artifact cannot answer the clause; this is missing evidence."""


def sheet_bin_occupancy(positions_e, *, bin_mm, sheet_l_mm):
    """E neurons per square bin, binned exactly as the frozen recruitment trace.

    ``rolling_full_field_recruitment`` calls a bin "occupied" when it holds at
    least one E neuron. The spec asks for at least 20. Whether those two rules
    differ is a property of the frozen geometry, not of any trajectory, so it is
    decided here once and reused by every candidate.
    """
    positions = np.asarray(positions_e, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_e must be (n_neurons, 2)")
    n = max(1, int(round(float(sheet_l_mm) / float(bin_mm))))
    ix = np.clip((positions[:, 0] / float(sheet_l_mm) * n).astype(int), 0, n - 1)
    iy = np.clip((positions[:, 1] / float(sheet_l_mm) * n).astype(int), 0, n - 1)
    occupancy = np.bincount(ix * n + iy, minlength=n * n).astype(float)
    occupied = occupancy > 0
    return {
        "bin_mm": float(bin_mm),
        "n_bins": int(n * n),
        "n_occupied": int(occupied.sum()),
        "minimum_occupancy": float(occupancy[occupied].min()) if occupied.any() else 0.0,
        "median_occupancy": float(np.median(occupancy[occupied])) if occupied.any() else 0.0,
        "occupancy": occupancy,
    }


def occupancy_rule_is_inert(occupancy_audit, minimum_occupancy):
    """True when 'occupied' and 'at least ``minimum_occupancy``' select the same bins.

    When this holds, a trace computed with the permissive rule IS the trace the
    spec asks for, and no sparse bin can contribute to ``F_sheet``.
    """
    return bool(float(occupancy_audit["minimum_occupancy"])
                >= float(minimum_occupancy))


def time_landmarks(operational_onset_ms, config):
    """Spec 5.1. Every window is derived from the unchanged detector time."""
    spec = config["model_ictal_v2"]
    t_op = float(operational_onset_ms)
    t_ictal = t_op + float(spec["t_ictal_offset_from_operational_onset_ms"])
    def _window(key):
        low, high = spec[key]
        return (t_ictal + float(low), t_ictal + float(high))
    return {
        "t_base_ms": [float(v) for v in spec["t_base_ms"]],
        "t_op_ms": t_op,
        "t_ictal_ms": t_ictal,
        "w_pre_ms": list(_window("w_pre_ms_relative_to_t_ictal")),
        "w_early_ms": list(_window("w_early_ms_relative_to_t_ictal")),
        "w_freq_ms": list(_window("w_freq_ms_relative_to_t_ictal")),
    }


def _fully_contained(time_ms, window, *, window_ms):
    """Select rolling windows that lie entirely inside ``window``.

    The recruitment trace stamps each sample with the END of its 20 ms window, so
    an end-time test alone would count windows whose first milliseconds precede
    ``W_early``. Full containment removes that ambiguity.
    """
    time_ms = np.asarray(time_ms, float)
    return (time_ms - float(window_ms) >= float(window[0])) & (time_ms <= float(window[1]))


def _slice(values, dt_ms, window):
    values = np.asarray(values, float)
    time = np.arange(len(values)) * float(dt_ms)
    return values[(time >= float(window[0])) & (time < float(window[1]))]


def _covers(n_samples, dt_ms, window):
    return float(n_samples) * float(dt_ms) >= float(window[1])


def joint_recruitment_duty(f_e, f_sheet, time_ms, window, *, activity_threshold,
                           recruitment_window_ms=RECRUITMENT_WINDOW_MS):
    """Spec 5.2 clause 2: both fractions must clear the bar in the SAME window."""
    selected = _fully_contained(time_ms, window, window_ms=recruitment_window_ms)
    n = int(selected.sum())
    if n == 0:
        raise NotEvaluableError("no recruitment window lies inside W_early")
    e_ok = np.asarray(f_e, float)[selected] >= float(activity_threshold)
    sheet_ok = np.asarray(f_sheet, float)[selected] >= float(activity_threshold)
    return {
        "n_windows": n,
        "joint_duty": float(np.mean(e_ok & sheet_ok)),
        "f_e_duty": float(np.mean(e_ok)),
        "f_sheet_duty": float(np.mean(sheet_ok)),
        "median_f_e": float(np.median(np.asarray(f_e, float)[selected])),
        "median_f_sheet": float(np.median(np.asarray(f_sheet, float)[selected])),
        "activity_threshold": float(activity_threshold),
    }


def _paired_two_windows(values, dt_ms, first, second):
    """Concatenate two equal-length windows so the frozen spectral helper can
    treat them as a pre/post pair with an artificial onset in the middle."""
    a = _slice(values, dt_ms, first)
    b = _slice(values, dt_ms, second)
    n = min(len(a), len(b))
    if n < 2:
        raise NotEvaluableError("spectral comparison windows are empty")
    a, b = a[-n:], b[:n]
    if a.ndim == 1:
        paired = np.concatenate([a, b])
    else:
        paired = np.concatenate([a, b], axis=0)
    return paired, n * float(dt_ms)


def contact_frequency_against_base(contact_trace, dt_ms, landmarks, *, band_hz):
    """Spec 5.2 clause 4, referenced to ``t_base`` rather than the pre-onset window.

    The spec writes ``f_contact(W_freq) - f_contact(t_base)`` with ``f_contact``
    defined as "the median contact spectral centroid". Read literally that is a
    difference of medians, which is what ``primary_shift_hz`` reports. The frozen
    helper's ``median_spectral_centroid_shift_hz`` is the median of per-contact
    differences, which the historical classifier used; it is reported beside the
    primary value because the two are not equal in general.
    """
    trace = np.asarray(contact_trace, float)
    paired, half_ms = _paired_two_windows(
        trace, dt_ms, landmarks["t_base_ms"], landmarks["w_freq_ms"])
    metrics = contact_oscillation_metrics(
        paired, dt_ms=dt_ms, onset_ms=half_ms, pre_ms=half_ms, post_ms=half_ms,
        frequency_band_hz=tuple(float(v) for v in band_hz))
    base = float(metrics["median_spectral_centroid_pre_hz"])
    early = float(metrics["median_spectral_centroid_post_hz"])
    return {
        "reference_window_ms": list(landmarks["t_base_ms"]),
        "analysis_window_ms": list(landmarks["w_freq_ms"]),
        "band_hz": [float(band_hz[0]), float(band_hz[1])],
        "frequency_resolution_hz": float(metrics["frequency_resolution_hz"]),
        "median_centroid_base_hz": base,
        "median_centroid_early_hz": early,
        "primary_shift_hz": early - base,
        "primary_ratio": early / max(base, 1e-12),
        "median_of_per_contact_shifts_hz": float(
            metrics["median_spectral_centroid_shift_hz"]),
        "median_peak_frequency_base_hz": float(
            metrics["median_peak_frequency_pre_hz"]),
        "median_peak_frequency_early_hz": float(
            metrics["median_peak_frequency_post_hz"]),
        "median_band_rms_ratio": float(
            metrics["median_band_rms_ratio_post_over_pre"]),
        "median_high_envelope_duty": float(
            metrics["median_post_high_envelope_duty"]),
    }


def population_rate_summary(rate_hz, dt_ms, landmarks):
    """Spec 5.2 clause 3 plus the broad-band rate-envelope diagnostics."""
    base = _slice(rate_hz, dt_ms, landmarks["t_base_ms"])
    early = _slice(rate_hz, dt_ms, landmarks["w_early_ms"])
    pre = _slice(rate_hz, dt_ms, landmarks["w_pre_ms"])
    if not len(base) or not len(early):
        raise NotEvaluableError("population-rate windows are empty")
    median_base = float(np.median(base))
    summary = {
        "median_rate_base_hz": median_base,
        "median_rate_early_hz": float(np.median(early)),
        "median_rate_pre_hz": float(np.median(pre)) if len(pre) else None,
        "ratio_early_over_base": float(np.median(early)) / max(median_base, 1e-12),
        "ratio_early_over_pre": (float(np.median(early))
                                 / max(float(np.median(pre)), 1e-12)
                                 if len(pre) else None),
        "envelope": {
            name: {"q05": float(np.quantile(values, 0.05)),
                   "median": float(np.median(values)),
                   "q95": float(np.quantile(values, 0.95)),
                   "max": float(np.max(values))}
            for name, values in (("t_base", base), ("w_pre", pre),
                                 ("w_early", early)) if len(values)},
    }
    try:
        paired, half_ms = _paired_two_windows(
            rate_hz, dt_ms, landmarks["t_base_ms"], landmarks["w_freq_ms"])
        summary["population_frequency_diagnostic"] = population_rate_frequency_metrics(
            paired, dt_ms=dt_ms, onset_ms=half_ms, pre_ms=half_ms, post_ms=half_ms)
    except (NotEvaluableError, ValueError) as error:
        summary["population_frequency_diagnostic"] = {
            "status": NOT_EVALUABLE, "reason": str(error)}
    return summary


def _finite_through(values, dt_ms, window):
    values = np.asarray(values, float)
    time = np.arange(len(values)) * float(dt_ms)
    inside = values[time <= float(window[1])]
    return bool(np.all(np.isfinite(inside)))


def qualify_model_ictal_v2(
        *, operational_onset_ms, recruitment_time_ms, f_e, f_sheet,
        f_sheet_provenance, occupancy_audit, rate_hz, rate_dt_ms,
        contact_trace, contact_dt_ms, config, simulator_error=False,
        onset_shift_ms=0.0):
    """Return the fixed V2 verdict plus the raw quantities that produced it."""
    spec = config["model_ictal_v2"]
    if operational_onset_ms is None:
        return {"status": NOT_ELIGIBLE, "eligible": False,
                "clauses": {"operational_detector_reached": False},
                "reason": "the unchanged operational detector was never reached",
                "onset_shift_ms": float(onset_shift_ms)}

    landmarks = time_landmarks(
        float(operational_onset_ms) + float(onset_shift_ms), config)
    early = landmarks["w_early_ms"]
    freq = landmarks["w_freq_ms"]
    missing = []
    if len(recruitment_time_ms) and float(np.max(recruitment_time_ms)) < early[1]:
        missing.append("recruitment trace stops before W_early ends")
    if not _covers(len(rate_hz), rate_dt_ms, early):
        missing.append("population rate stops before W_early ends")
    if not _covers(len(contact_trace), contact_dt_ms, freq):
        missing.append("contact trace stops before W_freq ends")
    if missing:
        return {"status": NOT_EVALUABLE, "eligible": None,
                "landmarks": landmarks, "missing_evidence": missing,
                "onset_shift_ms": float(onset_shift_ms)}

    requested_bin = float(spec["sheet_bin_mm"])
    requested_fraction = float(spec["sheet_recruited_bin_fraction"])
    requested_occupancy = float(spec["sheet_minimum_bin_occupancy"])
    sheet_rule = {
        "requested_bin_mm": requested_bin,
        "trace_bin_mm": float(f_sheet_provenance["bin_mm"]),
        "requested_recruited_bin_fraction": requested_fraction,
        "trace_recruited_bin_fraction": float(
            f_sheet_provenance["recruited_bin_fraction"]),
        "requested_minimum_occupancy": requested_occupancy,
        "trace_minimum_occupancy_applied": float(
            f_sheet_provenance["minimum_bin_occupancy_applied"]),
        "observed_minimum_bin_occupancy": float(
            occupancy_audit["minimum_occupancy"]),
    }
    sheet_rule["bin_matches"] = sheet_rule["trace_bin_mm"] == requested_bin
    sheet_rule["fraction_matches"] = (
        sheet_rule["trace_recruited_bin_fraction"] == requested_fraction)
    sheet_rule["occupancy_admissible"] = bool(
        sheet_rule["trace_minimum_occupancy_applied"] >= requested_occupancy
        or occupancy_rule_is_inert(occupancy_audit, requested_occupancy))
    sheet_rule["occupancy_rule_inert"] = occupancy_rule_is_inert(
        occupancy_audit, requested_occupancy)
    if not (sheet_rule["bin_matches"] and sheet_rule["fraction_matches"]
            and sheet_rule["occupancy_admissible"]):
        return {"status": NOT_EVALUABLE, "eligible": None,
                "landmarks": landmarks, "sheet_rule": sheet_rule,
                "missing_evidence": [
                    "the stored F_sheet trace was built under a bin/occupancy "
                    "rule that does not satisfy the primary rule, and cannot be "
                    "recomputed without per-spike data"],
                "onset_shift_ms": float(onset_shift_ms)}

    duty = joint_recruitment_duty(
        f_e, f_sheet, recruitment_time_ms, early,
        activity_threshold=float(spec["activity_threshold"]))
    rate = population_rate_summary(rate_hz, rate_dt_ms, landmarks)
    contact = contact_frequency_against_base(
        contact_trace, contact_dt_ms, landmarks,
        band_hz=spec["contact_centroid_band_hz"])
    finite = (_finite_through(rate_hz, rate_dt_ms, early)
              and _finite_through(np.asarray(f_e, float), 1.0,
                                  (0.0, float(len(f_e))))
              and bool(np.all(np.isfinite(np.asarray(contact_trace, float)))))

    resolution = max(float(contact["frequency_resolution_hz"]),
                     float(spec["contact_centroid_shift_min_hz"]))
    clauses = {
        "operational_detector_reached": True,
        "joint_broad_recruitment_duty": (
            duty["joint_duty"] >= float(spec["duty_threshold"])),
        "population_rate_ratio": (
            rate["ratio_early_over_base"] >= float(spec["population_rate_ratio_min"])),
        "contact_frequency_increased": (
            contact["primary_shift_hz"] >= resolution
            and contact["primary_ratio"] >= float(spec["contact_centroid_ratio_min"])),
        "numerically_safe": bool(finite and not simulator_error),
    }
    eligible = all(clauses.values())
    return {
        "status": ELIGIBLE if eligible else NOT_ELIGIBLE,
        "eligible": bool(eligible),
        "onset_shift_ms": float(onset_shift_ms),
        "landmarks": landmarks,
        "clauses": clauses,
        "failing_clauses": [name for name, value in clauses.items() if not value],
        "thresholds": {
            "duty": float(spec["duty_threshold"]),
            "activity": float(spec["activity_threshold"]),
            "population_rate_ratio_min": float(spec["population_rate_ratio_min"]),
            "contact_centroid_shift_min_hz": resolution,
            "contact_centroid_ratio_min": float(spec["contact_centroid_ratio_min"]),
        },
        "recruitment": duty,
        "population_rate": rate,
        "contact_frequency": contact,
        "sheet_rule": sheet_rule,
        "boundary": ("model-state endpoint only; this is not a clinical seizure "
                     "classifier and reads no patient quantity"),
    }


def qualification_sensitivities(*, operational_onset_ms, recruitment_time_ms,
                                f_e, f_sheet, f_sheet_provenance, occupancy_audit,
                                rate_hz, rate_dt_ms, contact_trace, contact_dt_ms,
                                config, extra_occupancy_audits=None):
    """Spec 5.3. Reported beside every verdict; never substituted for it."""
    spec = config["model_ictal_v2"]
    grid = spec["sensitivities"]
    landmarks = time_landmarks(operational_onset_ms, config)
    early = landmarks["w_early_ms"]

    activity = {}
    for threshold in grid["activity"]:
        try:
            row = joint_recruitment_duty(
                f_e, f_sheet, recruitment_time_ms, early,
                activity_threshold=float(threshold))
        except NotEvaluableError as error:
            activity[f"{threshold:g}"] = {"status": NOT_EVALUABLE,
                                          "reason": str(error)}
            continue
        row["passes_duty"] = {
            f"{duty:g}": bool(row["joint_duty"] >= float(duty))
            for duty in grid["duty"]}
        activity[f"{threshold:g}"] = row

    occupancy = {}
    audits = {float(occupancy_audit["bin_mm"]): occupancy_audit}
    audits.update({float(row["bin_mm"]): row
                   for row in (extra_occupancy_audits or [])})
    for bin_mm in grid["bin_mm"]:
        entry = audits.get(float(bin_mm))
        if entry is None:
            occupancy[f"{bin_mm:g}mm"] = {
                "status": NOT_EVALUABLE,
                "reason": "bin geometry not supplied"}
            continue
        recomputable = float(bin_mm) == float(f_sheet_provenance["bin_mm"])
        occupancy[f"{bin_mm:g}mm"] = {
            "n_bins": entry["n_bins"], "n_occupied": entry["n_occupied"],
            "minimum_occupancy": entry["minimum_occupancy"],
            "median_occupancy": entry["median_occupancy"],
            "occupancy_rule_inert": {
                f"{minimum:g}": occupancy_rule_is_inert(entry, minimum)
                for minimum in grid["minimum_bin_occupancy"]},
            "f_sheet_recomputable": recomputable,
            "status": ("EVALUABLE" if recomputable else NOT_EVALUABLE),
            "reason": (None if recomputable else
                       "F_sheet at this bin size needs per-spike positions, "
                       "which the stored artifacts do not contain"),
        }

    onset = {}
    for shift in grid["onset_shift_ms"]:
        onset[f"{shift:+g}ms"] = qualify_model_ictal_v2(
            operational_onset_ms=operational_onset_ms,
            recruitment_time_ms=recruitment_time_ms, f_e=f_e, f_sheet=f_sheet,
            f_sheet_provenance=f_sheet_provenance,
            occupancy_audit=occupancy_audit, rate_hz=rate_hz,
            rate_dt_ms=rate_dt_ms, contact_trace=contact_trace,
            contact_dt_ms=contact_dt_ms, config=config,
            onset_shift_ms=float(shift))["status"]
    return {"activity_and_duty": activity, "bin_and_occupancy": occupancy,
            "onset_shift": onset}


def require_model_ictal_eligible(verdict):
    """Gate for any consumer that wants to present a candidate as model ictal.

    A consumer must call this rather than reading ``status`` itself, so that
    "not evaluable" can never be silently rendered as "eligible".
    """
    if verdict.get("status") != ELIGIBLE or verdict.get("eligible") is not True:
        raise NotEvaluableError(
            f"candidate is not MODEL_ICTAL_ELIGIBLE_V2 (status="
            f"{verdict.get('status')!r}, failing="
            f"{verdict.get('failing_clauses', verdict.get('missing_evidence'))!r})")
    return verdict
