"""Estimability by endpoint × horizon from real coverage segments (plan Task 7, G1-G4).

Support counts come from the real window builder objects (target segments,
partition, anchor grid) -- never from session counts or sliding-window totals.
Required block counts come from the medium-effect power curve; when a curve is
missing the endpoint is flagged ``power_curve_pending`` rather than guessed.
A patient can leave a denominator for support only, never for a result.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.topic5_group_event_state.v032_eval.partition import EVAL_PHASES

DEVELOPMENT_PHASES = ("dev_val", "dev_test")
ENDPOINTS: tuple[dict[str, Any], ...] = (
    {"endpoint": "count_profile", "view": "count_profile", "horizons": (1800,), "support_unit": "independent_blocks", "exploratory": False},
    {"endpoint": "conditional_grammar", "view": "grammar", "horizons": (300, 1800), "support_unit": "independent_blocks", "exploratory": False},
    {"endpoint": "count_120min_exploratory", "view": "count", "horizons": (7200,), "support_unit": "independent_blocks", "exploratory": True},
    {"endpoint": "h2a_event_anchor", "view": None, "horizons": (None,), "support_unit": "positive_k_events", "exploratory": False},
    {"endpoint": "h2b_seizure_risk", "view": None, "horizons": (None,), "support_unit": "seizures_in_development_phases", "exploratory": False},
)


def _blocks_by_phase(segments: Sequence[Any], partition: Any, horizon: float) -> dict[str, int]:
    out: dict[str, int] = {}
    for phase in EVAL_PHASES:
        lo, hi = partition.bounds(phase)
        total = 0
        for seg in segments:
            a, b = max(float(seg.start_epoch), lo), min(float(seg.stop_epoch), hi)
            if b > a:
                total += int(math.floor((b - a) / float(horizon)))
        out[phase] = total
    out["development"] = sum(out[p] for p in DEVELOPMENT_PHASES)
    return out


def subject_support_from_arrays(*, segments: Sequence[Any], partition: Any, grid: Any, event_times: np.ndarray,
                                group_count: np.ndarray, seizures: Sequence[Mapping[str, Any]],
                                horizons: Sequence[float]) -> dict[str, Any]:
    t_anchor = np.asarray(grid.t_anchor, dtype=np.float64)
    event_labels = partition.labels_of(np.asarray(event_times, dtype=np.float64))
    gc = np.asarray(group_count, dtype=np.int64)
    blocks, anchors, positive = {}, {}, {}
    for h_i, h in enumerate(horizons):
        key = str(int(h))
        blocks[key] = _blocks_by_phase(segments, partition, float(h))
        a_phase, p_phase = {}, {}
        for phase in EVAL_PHASES:
            rows = np.flatnonzero(partition.mask_for_phase(t_anchor, phase) & grid.eligible[:, h_i]
                                  & partition.window_within_phase(t_anchor, float(h)))
            counts = (grid.window_hi[rows, h_i] - grid.window_lo[rows, h_i]) if rows.size else np.zeros(0, np.int64)
            a_phase[phase] = int(rows.size)
            p_phase[phase] = int((counts > 0).sum())
        a_phase["development"] = sum(a_phase[p] for p in DEVELOPMENT_PHASES)
        p_phase["development"] = sum(p_phase[p] for p in DEVELOPMENT_PHASES)
        anchors[key], positive[key] = a_phase, p_phase
    events = {name: int((event_labels == i).sum()) for i, name in enumerate(EVAL_PHASES)}
    positive_k = {name: int(((event_labels == i) & (gc >= 2)).sum()) for i, name in enumerate(EVAL_PHASES)}
    events["development"] = sum(events[p] for p in DEVELOPMENT_PHASES)
    positive_k["development"] = sum(positive_k[p] for p in DEVELOPMENT_PHASES)
    sz_phase = {name: 0 for name in EVAL_PHASES}
    for sz in seizures:
        sz_phase[partition.phase_of(float(sz["onset_epoch"]))] += 1
    return {
        "n_sessions": int(len({int(s.session_id) for s in segments})),
        "n_target_segments": int(len(segments)),
        "recorded_seconds_by_phase": dict(partition.recorded_seconds),
        "blocks": blocks, "anchors": anchors, "grammar_positive_anchors": positive,
        "events": events, "h2a_positive_k_events": positive_k,
        "seizures": {"n_total": len(seizures), "by_phase": sz_phase,
                     "development": sum(sz_phase[p] for p in DEVELOPMENT_PHASES)},
        "support_source": "real window builder: build_carry_segments target segments ∩ recorded-time partition",
    }


def subject_support(tl: Any) -> dict[str, Any]:
    return subject_support_from_arrays(segments=tl.segments, partition=tl.partition, grid=tl.grid,
                                       event_times=tl.event_times, group_count=tl.group_count, seizures=tl.seizures,
                                       horizons=tl.horizons_seconds)


def eligibility_rows(subject: str, support: Mapping[str, Any],
                     requirements: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ENDPOINTS:
        for horizon in spec["horizons"]:
            row: dict[str, Any] = {"subject": subject, "endpoint": spec["endpoint"], "view": spec["view"],
                                   "horizon_seconds": horizon, "support_unit": spec["support_unit"],
                                   "exploratory": spec["exploratory"], "reasons": []}
            if spec["support_unit"] == "independent_blocks":
                key = str(int(horizon))
                row["available_development_blocks"] = support["blocks"][key]["development"]
                row["available_blocks_by_phase"] = dict(support["blocks"][key])
                row["available_development_anchors"] = support["anchors"][key]["development"]
                if spec["view"] == "grammar":
                    row["available_development_positive_anchors"] = support["grammar_positive_anchors"][key]["development"]
                req = requirements.get((spec["view"], int(horizon)))
                if req is None or req.get("required_blocks") is None:
                    row.update({"required_blocks": None, "requirement_source": None, "estimable": None,
                                "status": "power_curve_pending"})
                    row["reasons"].append("no medium-effect power curve for this view/horizon yet")
                else:
                    need = int(req["required_blocks"])
                    ok = row["available_development_blocks"] >= need
                    row.update({"required_blocks": need, "requirement_source": req.get("source"),
                                "requirement_tier": req.get("tier"), "estimable": bool(ok),
                                "status": "estimable" if ok else "not_estimable"})
                    if not ok:
                        row["reasons"].append(f"development_blocks={row['available_development_blocks']}<{need}")
            elif spec["support_unit"] == "positive_k_events":
                row.update({"available_development_positive_k_events": support["h2a_positive_k_events"]["development"],
                            "available_by_phase": dict(support["h2a_positive_k_events"]),
                            "required_blocks": None, "estimable": None, "status": "support_described_only"})
            else:
                row.update({"available_development_seizures": support["seizures"]["development"],
                            "available_by_phase": dict(support["seizures"]["by_phase"]),
                            "n_seizures_total": support["seizures"]["n_total"],
                            "required_blocks": None, "estimable": None, "status": "support_described_only"})
            rows.append(row)
    return rows


def requirements_from_power_curves(curves: Mapping[str, Any], *, tier: str = "medium") -> dict[tuple[str, int], dict[str, Any]]:
    """Conservative requirements from all declared calibration scaffolds.

    A single scaffold must never silently overwrite another scaffold.  For
    each endpoint/horizon, the planning requirement is the maximum finite
    Level-0 requirement across every declared scaffold.  If any scaffold has
    no finite requirement, the endpoint remains power-pending.
    """

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in curves.get("curves", []):
        tiers = entry.get("effect_tiers", {})
        chosen = tiers.get(tier)
        if not chosen:
            continue
        key = (entry["view"], int(entry["horizon_seconds"]))
        grouped.setdefault(key, []).append({
            "subject": entry.get("subject"),
            "required_blocks": chosen.get("required_blocks_level0"),
            "required_blocks_by_level": chosen.get("required_blocks_by_level"),
            "oracle_gain_median": chosen.get("oracle_gain_median"),
        })
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, scaffold_rows in grouped.items():
        finite = [row["required_blocks"] for row in scaffold_rows if row.get("required_blocks") is not None]
        all_finite = len(finite) == len(scaffold_rows)
        out[key] = {
            "required_blocks": int(max(finite)) if finite and all_finite else None,
            "required_blocks_by_level": None,
            "source": (
                f"{curves.get('format')} @ {curves.get('source_commit', 'unknown')[:10]} "
                f"conservative max across {len(scaffold_rows)} declared scaffolds"
            ),
            "tier": tier,
            "aggregation_rule": "max finite required_blocks across all scaffolds; any nonfinite => pending",
            "calibration_scaffolds": scaffold_rows,
        }
    return out
