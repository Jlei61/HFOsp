"""Pure adjudication contracts for the FCXR-LC4 functional-selectivity gate."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


BASELINE_RATIO_BANDS = {
    "event_rate": (0.80, 1.25),
    "iei_cv": (2.0 / 3.0, 1.50),
    "duration": (0.75, 4.0 / 3.0),
    "participation": (0.80, 1.25),
}


def force_matched_candidates(separation: dict, *, ns=(6, 8), dose_frac=0.20,
                             recurrent_scale=272.75518960107513,
                             base_n=4) -> list[dict]:
    """Build curve-shape candidates while holding the executed ictal current fixed."""
    row = separation["by_tau"]["1000"]
    hill = row["hill"]
    key0 = str(int(base_n))
    target = float(dose_frac) * float(recurrent_scale) * float(hill[key0]["ictal_mean"])
    out = []
    for n in ns:
        key = str(int(n))
        if key not in hill:
            raise ValueError(f"missing measured Hill row n={n}")
        a_ictal = float(hill[key]["ictal_mean"])
        if not (0.0 < a_ictal <= 1.0):
            raise ValueError(f"invalid ictal activation for n={n}: {a_ictal}")
        out.append(dict(
            name=f"hill_n{int(n)}_slow", n=float(n), K=float(row["K_midgap"]),
            tau_adp_ms=1000.0, tau_a_on_ms=100.0, tau_a_off_ms=10000.0,
            g_m_max=target / a_ictal, ictal_activation=a_ictal,
            predicted_interictal_activation=float(hill[key]["interictal_mean"]),
            matched_ictal_current=target,
        ))
    return out


def _cv(values: Iterable[float]) -> float | None:
    a = np.asarray(list(values), dtype=float)
    if a.size < 2 or not np.all(np.isfinite(a)):
        return None
    mean = float(np.mean(a))
    return float(np.std(a, ddof=1) / mean) if mean > 0.0 else None


def summarize_returning_events(events: list[dict], *, start_ms: float, end_ms: float) -> dict:
    """Functional interictal summary over one explicitly matched observation window."""
    if not (np.isfinite(start_ms) and np.isfinite(end_ms) and end_ms > start_ms):
        raise ValueError("event-summary window must be finite and non-empty")
    rows = [e for e in events
            if bool(e.get("returned")) and start_ms <= float(e["t_on"]) < end_ms]
    onsets = np.asarray([float(e["t_on"]) for e in rows], dtype=float)
    iei = np.diff(onsets)
    return dict(
        n_returning=len(rows),
        event_rate_hz=float(len(rows) / ((end_ms - start_ms) * 1e-3)),
        iei_cv=_cv(iei),
        median_duration_ms=(float(np.median([float(e["dur_ms"]) for e in rows]))
                            if rows else None),
        median_participation=(float(np.median([float(e["peak_ext"]) for e in rows]))
                              if rows else None),
        window_ms=[float(start_ms), float(end_ms)],
    )


def _safe_ratio(value, reference):
    if value is None or reference is None:
        return None
    value, reference = float(value), float(reference)
    if not (np.isfinite(value) and np.isfinite(reference) and reference > 0.0):
        return None
    return value / reference


def baseline_gate(candidate: dict, control: dict, *, numerical_safe: bool,
                  sustained_bout: bool, max_current: float,
                  recurrent_scale: float = 272.75518960107513) -> dict:
    """Adjudicate the paired functional baseline without substituting offline activation."""
    ratios = {
        "event_rate": _safe_ratio(candidate.get("event_rate_hz"), control.get("event_rate_hz")),
        "iei_cv": _safe_ratio(candidate.get("iei_cv"), control.get("iei_cv")),
        "duration": _safe_ratio(candidate.get("median_duration_ms"),
                                control.get("median_duration_ms")),
        "participation": _safe_ratio(candidate.get("median_participation"),
                                     control.get("median_participation")),
    }
    clauses = {
        "numerical_safe": bool(numerical_safe),
        "no_sustained_bout": not bool(sustained_bout),
        "at_least_three_returning": int(candidate.get("n_returning", 0)) >= 3,
        "current_leakage": bool(np.isfinite(max_current) and max_current >= 0.0
                                and max_current / float(recurrent_scale) <= 0.001),
    }
    for key, (lo, hi) in BASELINE_RATIO_BANDS.items():
        r = ratios[key]
        clauses[f"{key}_ratio"] = bool(r is not None and lo <= r <= hi)
    passed = all(clauses.values())
    return dict(
        passed=passed,
        verdict="BASELINE_PRESERVED" if passed else "BASELINE_DISTURBED",
        ratios=ratios, ratio_bands=BASELINE_RATIO_BANDS,
        current_fraction=float(max_current / float(recurrent_scale)), clauses=clauses,
    )


def select_candidate(rows: list[dict]) -> dict | None:
    """Prefer the least singular passing curve; outcomes never tune a new parameter."""
    passed = [r for r in rows if bool(r.get("gate", {}).get("passed"))]
    return min(passed, key=lambda r: float(r["candidate"]["n"])) if passed else None


def onset_surface_gate(rows: list[dict], *, field_order=("D_healthy", "D10", "D30", "D50")) -> dict:
    """Require a live positive control, a stable healthy field and a departing candidate row."""
    controls = [r for r in rows if r.get("role") == "positive_control"]
    candidates = {r["d_label"]: r for r in rows if r.get("role") == "candidate"}
    positive = bool(len(controls) == 1 and controls[0].get("d_label") == "D10"
                    and controls[0].get("departed"))
    healthy = candidates.get("D_healthy")
    healthy_stable = bool(healthy is not None and not healthy.get("departed"))
    first = next((label for label in field_order[1:]
                  if label in candidates and candidates[label].get("departed")), None)
    clauses = dict(positive_control_D10=positive, candidate_Dhealthy_stable=healthy_stable,
                   candidate_departure_through_D50=first is not None)
    passed = all(clauses.values())
    if not positive:
        verdict = "ONSET_INSTRUMENT_INVALID"
    elif not healthy_stable:
        verdict = "BASELINE_FIELD_DEPARTS"
    elif first is None:
        verdict = "ONSET_SURFACE_UNREACHABLE_IN_TESTED_RANGE"
    else:
        verdict = "ONSET_SURFACE_RETAINED"
    return dict(passed=passed, verdict=verdict, first_departing_field=first,
                field_order=list(field_order), clauses=clauses)
