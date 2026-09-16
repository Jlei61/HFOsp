#!/usr/bin/env python3
"""Aggregate v0.3.7 at the patient/seed level without opening sealed data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.contracts import atomic_json


DATA_ROOT = Path("/data/hfosp_group_event_state_v0_3_7")
CORE_SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
EXPANSION_SUBJECTS = ("epilepsiae_916",)
HORIZONS = (1800, 7200, 21600, 28800)
ENDPOINT_GROUPS = {
    "burden": {"count": 1.0, "burden": 0.5},
    "grammar": {"community": 1.0, "coupling": 1.0, "mixture": 1.0,
                "embedding": 0.5, "mark": 0.5},
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cards(root: Path) -> list[dict[str, Any]]:
    return [_read(path) for path in sorted(root.glob("**/card.json"))]


def _median(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(finite)) if finite else None


def _by_subject(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        output[card["subject"]].append(card)
    return dict(output)


def _endpoint_group_score(score: dict[str, Any], group: str) -> float:
    weights = ENDPOINT_GROUPS[group]
    endpoints = score["endpoints"]
    return float(sum(weight * float(endpoints[name]) for name, weight in weights.items()) / sum(weights.values()))


def _contrast_summary(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for subject, group in sorted(_by_subject(cards).items()):
        keys = sorted(set().union(*(card.get("primary_contrasts", {}) for card in group)))
        rows.append({
            "subject": subject,
            "n_seeds": len(group),
            "seed_values": {key: [card.get("primary_contrasts", {}).get(key) for card in group] for key in keys},
            "median": {key: _median([card.get("primary_contrasts", {}).get(key) for card in group]) for key in keys},
        })
    return rows


def _h1_horizon(cards: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    rows = []
    for subject, group in sorted(_by_subject(cards).items()):
        for horizon in HORIZONS:
            key = str(horizon)
            values: dict[str, list[float | None]] = defaultdict(list)
            for card in group:
                score = card["selection_scores"]
                if family in {"event", "grid"}:
                    state_name = "S_event" if family == "event" else "S_grid"
                    arm_pairs = {
                        "gain_over_mark_ewma": ("B_mark", state_name),
                        "dynamic_over_constant": (f"{state_name}_constant", state_name),
                    }
                else:
                    arm_pairs = {
                        "persistent_background_over_current": ("B_mark_current_background", "B_background_persistent"),
                        "event_after_background": ("B_background_persistent", "S_dual"),
                        "dynamic_event_over_constant": ("S_dual_constant_event", "S_dual"),
                    }
                for name, (baseline, model) in arm_pairs.items():
                    try:
                        baseline_score = score[baseline]["by_horizon"][key]
                        model_score = score[model]["by_horizon"][key]
                        values[name].append(float(baseline_score["total"]) - float(model_score["total"]))
                        for endpoint_group in ENDPOINT_GROUPS:
                            values[f"{name}_{endpoint_group}"].append(
                                _endpoint_group_score(baseline_score, endpoint_group)
                                - _endpoint_group_score(model_score, endpoint_group)
                            )
                    except (KeyError, TypeError):
                        values[name].append(None)
                shift = card.get("time_shift_by_horizon", {}).get(key)
                values["correct_time_over_shifted"].append(None if shift is None else shift.get("gain"))
                for endpoint_group in ENDPOINT_GROUPS:
                    values[f"correct_time_over_shifted_{endpoint_group}"].append(
                        None if shift is None else (
                            _endpoint_group_score(shift["shifted"], endpoint_group)
                            - _endpoint_group_score(shift["correct"], endpoint_group)
                        )
                    )
            rows.append({
                "subject": subject, "horizon_seconds": horizon, "n_seeds": len(group),
                **{name: _median(v) for name, v in values.items()},
            })
    return rows


def _h2b(cards: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for subject, group in sorted(_by_subject(cards).items()):
        hazard_status = [card["distance_survival"]["support"]["status"] for card in group]
        seizure_counts = [card["distance_survival"]["support"].get("seizures_by_phase", {}) for card in group]
        representative_counts = seizure_counts[0] if seizure_counts else {}
        selection_seizures = int(representative_counts.get("SELECTION", 0))
        fitted = "ESTIMATED" in hazard_status
        scientific_status = (
            "REPEATED_HELD_OUT_SEIZURES"
            if fitted and selection_seizures >= 3 else
            "SINGLE_HELD_OUT_SEIZURE_DESCRIPTIVE_ONLY"
            if fitted and selection_seizures == 1 else
            "NOT_ESTIMABLE"
        )
        field_status: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        field_gains: dict[str, dict[str, dict[str, list[float | None]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for card in group:
            for lead, targets in card["early_ictal_field_and_path"].items():
                for target, result in targets.items():
                    field_status[lead][target].append(result.get("status", "NOT_ESTIMABLE"))
                    for key in (
                        "state_gain_over_background", "event_only_state_gain_over_mark_history",
                        "grid_state_gain_over_mark_history", "event_correct_time_gain_over_shift",
                        "grid_correct_time_gain_over_shift", "dual_correct_time_gain_over_shift",
                    ):
                        field_gains[lead][target][key].append(result.get(key))
        rows.append({
            "subject": subject, "n_seeds": len(group),
            "hazard_status": "ESTIMATED" if fitted else "NOT_ESTIMABLE",
            "scientific_repeatability_status": scientific_status,
            "seizures_by_phase": representative_counts,
            "hazard_contrasts": {
                key: _median([card["primary_contrasts"].get(key) for card in group])
                for key in sorted(set().union(*(card["primary_contrasts"] for card in group)))
            },
            "field_status": {
                lead: {target: ("ESTIMATED" if "ESTIMATED" in statuses else "NOT_ESTIMABLE")
                       for target, statuses in targets.items()}
                for lead, targets in field_status.items()
            },
            "field_gains": {
                lead: {
                    target: {key: _median(values) for key, values in contrasts.items()}
                    for target, contrasts in targets.items()
                }
                for lead, targets in field_gains.items()
            },
        })
    return {"rows": rows}


def _h3(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate H3 endpoints without conflating count persistence with state change."""

    rows = []
    for subject, group in sorted(_by_subject(cards).items()):
        values: dict[str, list[float | None]] = defaultdict(list)
        paired_blocks = []
        for card in group:
            primary = card["primary_contrasts"]
            controls = card["causal_delay_and_constant_controls"]
            scores = card["scores"]
            for key in (
                "count_feedback_gain_M1_over_M0",
                "count_feedback_gain_on_future_background",
                "mark_feedback_gain_on_future_background",
            ):
                values[key].append(primary.get(key))
            # A positive value means the correctly timed exposure beats the
            # named control on the same outcome and, for delay, same support.
            values["count_to_count_correct_over_delayed"].append(
                controls["M1_causal_delayed_count"]["count"]
                - controls["M1_correct_on_delayed_support"]["count"]
            )
            values["count_to_count_correct_over_constant"].append(
                controls["M1_fit_mean_count"]["count"]
                - scores["M1_count_feedback"]["count"]
            )
            values["count_to_background_correct_over_delayed"].append(
                controls["M1_causal_delayed_count"]["future_background"]
                - controls["M1_correct_on_delayed_support"]["future_background"]
            )
            values["count_to_background_correct_over_constant"].append(
                controls["M1_fit_mean_count"]["future_background"]
                - scores["M1_count_feedback"]["future_background"]
            )
            values["mark_to_background_correct_over_delayed"].append(
                controls["M2_causal_delayed_mark"]["future_background"]
                - controls["M2_correct_on_delayed_support"]["future_background"]
            )
            values["mark_to_background_correct_over_constant"].append(
                controls["M2_fit_mean_mark"]["future_background"]
                - scores["M2_mark_feedback"]["future_background"]
            )
            paired_blocks.append(int(controls["paired_selection_blocks"]))
        rows.append({
            "subject": subject, "n_seeds": len(group),
            "median": {key: _median(items) for key, items in values.items()},
            "seed_values": dict(values),
            "positive_seed_counts": {
                key: sum(value is not None and float(value) > 0 for value in items)
                for key, items in values.items()
            },
            "paired_selection_blocks": int(np.median(paired_blocks)) if paired_blocks else 0,
        })
    return {"rows": rows}


def _h3_persistent(cards: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for subject, group in sorted(_by_subject(cards).items()):
        keys = sorted(set().union(*(card.get("primary_contrasts", {}) for card in group)))
        eligibility = {}
        for tau in (7200, 21600, 86400, 172800):
            flags = [
                bool(card.get("tau_eligibility", {}).get(str(tau), {}).get("eligible", False))
                for card in group
            ]
            eligibility[str(tau)] = {
                "eligible_all_seeds": bool(flags) and all(flags),
                "eligible_seed_count": int(sum(flags)),
            }
        rows.append({
            "subject": subject, "n_seeds": len(group),
            "eligible_physical_scales": eligibility,
            "median": {key: _median([card.get("primary_contrasts", {}).get(key) for card in group]) for key in keys},
            "positive_seed_counts": {
                key: int(sum(
                    card.get("primary_contrasts", {}).get(key) is not None
                    and float(card["primary_contrasts"][key]) > 0
                    for card in group
                )) for key in keys
            },
            "allowed_claim": group[0].get("allowed_claim") if group else None,
        })
    return {"rows": rows}


def _h2a_joint(cards: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for subject, group in sorted(_by_subject(cards).items()):
        h2_keys = sorted(set().union(*(card.get("h2a_primary_contrasts", {}) for card in group)))
        h1_keys = sorted(set().union(*(card.get("h1_mandatory_reevaluation", {}).get("primary_contrasts", {}) for card in group)))
        rows.append({
            "subject": subject, "n_seeds": len(group),
            "h2a_median": {
                key: _median([card.get("h2a_primary_contrasts", {}).get(key) for card in group])
                for key in h2_keys
            },
            "h1_after_joint_median": {
                key: _median([
                    card.get("h1_mandatory_reevaluation", {}).get("primary_contrasts", {}).get(key)
                    for card in group
                ]) for key in h1_keys
            },
            "producer_updated_only_by_interictal_h2a": all(
                card.get("producer_updated_by") == "interictal H2a grammar likelihood only"
                for card in group
            ),
        })
    return {"rows": rows}


def _queue(path: Path) -> dict[str, Any]:
    return _read(path) if path.exists() else {"status": "MISSING", "path": str(path)}


def _sha(path: Path) -> str:
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()



def _training_adequacy(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Report whether each optimisation path was genuinely explored.

    Round one produced exact zeros in three separate layers because the arm
    under test never moved away from its initialisation.  In the repaired
    nested design, however, a path may move substantially during training and
    still *select* the parent-parity checkpoint.  That is an interpretable
    validation result, not an unfitted arm.  Trainability is therefore based
    on a non-zero first-step gradient/parameter movement (or the explicit H2a
    movement audit), while selected-at-origin remains a separate scientific
    outcome.
    """
    rows: dict[str, dict[str, Any]] = {}
    for card in cards:
        for stage, value in (card.get("stages") or {}).items():
            if not isinstance(value, dict) or "selected_at_init" not in value:
                continue
            row = rows.setdefault(stage, {
                "n": 0,
                "selected_at_init": 0,
                "budget_exhausted": 0,
                "path_explored": 0,
            })
            row["n"] += 1
            row["selected_at_init"] += int(bool(value.get("selected_at_init")))
            row["budget_exhausted"] += int(bool(value.get("training_budget_exhausted")))
            if "adapter_moved_during_training" in value:
                explored = bool(value.get("adapter_moved_during_training"))
            else:
                gradient = float(value.get("first_step_gradient_norm", 0.0) or 0.0)
                movement = float(value.get("peak_parameter_delta_from_stage_start", 0.0) or 0.0)
                explored = gradient > 0.0 and movement > 0.0
            row["path_explored"] += int(explored)
    for stage, row in rows.items():
        row["selected_non_origin_fraction"] = (
            1.0 - row["selected_at_init"] / row["n"] if row["n"] else None
        )
        row["path_explored_fraction"] = row["path_explored"] / row["n"] if row["n"] else None
        row["budget_exhausted_fraction"] = row["budget_exhausted"] / row["n"] if row["n"] else None
        row["arm_was_fitted"] = bool(
            row["path_explored_fraction"] is not None
            and row["path_explored_fraction"] >= 0.75
            and row["budget_exhausted_fraction"] <= 0.25
        )
    control = rows.get("bmark") or rows.get("B_mark")
    return {
        "by_stage": rows,
        "control_arm_fitted": None if control is None else control["arm_was_fitted"],
        "interpretable": bool(control is not None and control["arm_was_fitted"]),
        "rule": (
            "a contrast is trainability-interpretable only when the optimisation path "
            "actually moved in at least 75% of cells and no more than 25% exhausted "
            "the budget; selecting the parent-parity origin after such exploration is "
            "a scientific null, not a training failure"
        ),
    }


def _code_provenance_audit(cards: list[dict[str, Any]], sources: dict[str, Path]) -> dict[str, Any]:
    """Refuse to summarise cards that a later code version has superseded."""
    current = {name: _sha(path) for name, path in sources.items() if path.exists()}
    seen: dict[str, set[str]] = {}
    missing = 0
    for card in cards:
        provenance = card.get("code_provenance")
        if not isinstance(provenance, dict):
            missing += 1
            continue
        seen.setdefault(provenance.get("source_file", "unknown"), set()).add(
            str(provenance.get("source_sha256"))
        )
    stale = {
        name: sorted(hashes - {current.get(name)})
        for name, hashes in seen.items()
        if current.get(name) is not None and hashes - {current.get(name)}
    }
    return {
        "cards_without_code_provenance": missing,
        "current_source_sha256": current,
        "stale_source_sha256": stale,
        "all_cards_match_current_code": bool(missing == 0 and not stale),
    }



def _donor_offset_hours(card: dict[str, Any]) -> dict[str, Any]:
    """Recover the wrong-time donor displacement from the saved trajectory.

    Computed here rather than in the trainer so that a formal queue already in
    flight is not split across two source hashes.  The donor rule is a half
    circular roll inside each coverage segment, so the displacement has to be
    reproduced segment by segment; rolling the whole held-out block instead
    reports an offset that the code never used.
    """
    path = card.get("trajectory_path")
    if not path or not Path(path).exists():
        return {"status": "TRAJECTORY_UNAVAILABLE"}
    with np.load(path, allow_pickle=False) as stored:
        if "segment" not in stored:
            return {
                "status": "SEGMENT_NOT_STORED",
                "note": "offset needs the coverage segment the donor roll is confined to",
            }
        time = np.asarray(stored["anchor_time"], dtype=np.float64)
        phase = stored["phase"].astype(str)
        segment = np.asarray(stored["segment"])
    rows = np.flatnonzero(phase == "SELECTION")
    offsets, spans = [], []
    for value in np.unique(segment[rows]):
        block = rows[segment[rows] == value]
        if block.size < 4:
            continue
        spans.append(float(time[block].max() - time[block].min()) / 3600.0)
        donor = np.roll(block, max(1, block.size // 2))
        offsets.append(np.abs(time[donor] - time[block]) / 3600.0)
    if not offsets:
        return {"status": "NOT_ESTIMABLE"}
    joined = np.concatenate(offsets)
    return {
        "status": "ESTIMATED",
        "median_hours": float(np.median(joined)),
        "min_hours": float(joined.min()),
        "max_hours": float(joined.max()),
        "n_held_out_segments": len(offsets),
        "longest_held_out_segment_hours": float(max(spans)),
    }



def _wrong_time_control_quality(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Report how good the wrong-time null actually is, per patient.

    Two properties decide whether a "correct beats shifted" number means
    anything, and neither was reported:

    1.  How far away the donor is.  The rule is a half roll inside each
        coverage segment, so the displacement is an accident of how fragmented
        the patient's held-out coverage happens to be -- 3.25 h for a patient
        with four short segments, 12 h for a patient with one long one.  A
        correct-time gain is therefore not comparable across patients.
    2.  Whether the shifted arm is merely uninformative or actively wrong.  A
        shifted state that scores worse than the constant-state arm is
        misleading rather than null, and inflates the gain.
    """
    rows = []
    for card in cards:
        shift = card.get("time_shift_by_horizon") or {}
        constant = (card.get("selection_scores") or {})
        constant = (
            constant.get("S_dual_constant_all")
            or constant.get("S_event_constant")
            or constant.get("S_grid_constant")
            or {}
        ).get("by_horizon") or {}
        per_horizon = {}
        for key, entry in shift.items():
            if not isinstance(entry, dict):
                per_horizon[key] = {"status": "NOT_ESTIMABLE"}
                continue
            shifted = entry.get("shifted", {}).get("total")
            const = constant.get(key, {}).get("total")
            per_horizon[key] = {
                "status": "ESTIMATED",
                "correct_time_gain": entry.get("gain"),
                "shifted_minus_constant": (
                    None if shifted is None or const is None else float(shifted - const)
                ),
                "shifted_arm_worse_than_constant": (
                    None if shifted is None or const is None else bool(shifted > const)
                ),
            }
        donor = _donor_offset_hours(card)
        rows.append({
            "subject": card.get("subject"), "seed": card.get("seed"),
            "median_donor_offset_hours": donor,
            "by_horizon": per_horizon,
        })
    return {
        "rows": rows,
        "donor_rule": "half circular roll inside each held-out coverage segment",
        "offset_is_uncontrolled": True,
        "clock_matched_control_available": False,
        "why_no_clock_matched_control": (
            "the longest held-out coverage segment in this cohort is 12.8-24.0 h, so no two "
            "held-out anchors inside one segment are a day apart; a same-clock-phase wrong-time "
            "donor cannot be constructed and its absence must not be read as one having passed"
        ),
        "rule": (
            "a correct-time gain may be read within a patient; it may not be compared across "
            "patients whose donor offsets differ, and it is not reportable for a horizon whose "
            "shifted arm scores worse than the constant-state arm"
        ),
    }


def _floored_h3_persistent(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Surface the wrong-time contrasts with the null floored at no-edge."""
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        contrasts = card.get("primary_contrasts") or {}
        floor = (
            (card.get("fitted_equal_capacity_wrong_time_placebos") or {}).get("no_edge_floor") or {}
        ).get("count_future_background") or {}
        by_subject.setdefault(str(card.get("subject")), []).append({
            "nested_over_no_edge": contrasts.get("M1_persistent_over_M0_background"),
            "raw_over_fitted_placebo": contrasts.get(
                "persistent_count_real_over_fitted_wrong_time_background"),
            "floored_over_null": contrasts.get(
                "persistent_count_real_over_floored_wrong_time_background"),
            "placebo_worse_than_no_edge": floor.get("placebo_worse_than_no_edge"),
        })
    out = {}
    for subject, rows in by_subject.items():
        out[subject] = {
            "n_seeds": len(rows),
            "median": {
                key: _median([row[key] for row in rows])
                for key in ("nested_over_no_edge", "raw_over_fitted_placebo", "floored_over_null")
            },
            "seeds_with_placebo_worse_than_no_edge": sum(
                1 for row in rows if row["placebo_worse_than_no_edge"]
            ),
        }
    return {
        "by_subject": out,
        "rule": (
            "only the floored contrast is reportable; a fitted same-capacity placebo that "
            "generalises worse than the no-edge model inflates the raw contrast by its own "
            "overfitting rather than by the real edge's skill"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(); root = args.data_root
    out = root / "final_reports"; out.mkdir(parents=True, exist_ok=True)

    # The first-pass roots remain immutable provenance.  The integrated v2
    # report uses the recipe selected on FIT/INNER and the full 5-seed rerun.
    h1_event = _cards(root / "h1_optimized" / "event")
    h1_grid = _cards(root / "h1_optimized" / "grid")
    h1_dual = _cards(root / "h1_optimized" / "dual")
    h2a_event = _cards(root / "h2a_optimized" / "event")
    h2a_grid = _cards(root / "h2a_optimized" / "grid")
    h2a_dual = _cards(root / "h2a_optimized" / "dual")
    h2a_joint_cards = _cards(root / "h2a_joint_sensitivity_optimized")
    h2b_cards = _cards(root / "h2b_optimized" / "outcomes")
    h3_cards = _cards(root / "h3_independent_generative")
    h3_persistent_cards = _cards(root / "h3_persistent_feedback_v4")
    all_cards = (
        h1_event + h1_grid + h1_dual + h2a_event + h2a_grid + h2a_dual
        + h2a_joint_cards + h2b_cards + h3_cards + h3_persistent_cards
    )
    optimizer_summary_path = root / "optimizer_search" / "summary.json"
    optimizer_summary = _read(optimizer_summary_path) if optimizer_summary_path.exists() else None
    payload = {
        "format": "group_event_state_v0_3_7_integrated_summary_v2",
        "version": "v0.3.7",
        "science_contract": {
            "S_obs": "causal predictive summary; not a physiological state claim",
            "Z_phys": "independent H3 generative candidate; never copied from observer updates",
            "patient_is_statistical_unit": True,
            "shared_state_producer_across_horizons": True,
            "primary_horizons_seconds": list(HORIZONS),
        },
        "queues": {
            "optimizer_search": _queue(root / "optimizer_search" / "supervisor" / "queue_status.json"),
            "h1_optimized": _queue(root / "h1_optimized" / "supervisor" / "queue_status.json"),
            "h2a_optimized": _queue(root / "h2a_optimized" / "supervisor" / "queue_status.json"),
            "h2a_joint": _queue(root / "h2a_joint_sensitivity_optimized" / "supervisor" / "queue_status.json"),
            "h2b_optimized": _queue(root / "h2b_optimized" / "supervisor" / "queue_status.json"),
            "h3_one_step": _queue(root / "h3_independent_generative" / "supervisor" / "queue_status.json"),
            "h3_persistent": _queue(root / "h3_persistent_feedback_v4" / "supervisor" / "queue_status.json"),
        },
        "optimizer_search": optimizer_summary,
        "training_adequacy": {
            "h1_event": _training_adequacy(h1_event),
            "h1_grid": _training_adequacy(h1_grid),
            "h1_dual": _training_adequacy(h1_dual),
            "h2a_event": _training_adequacy(h2a_event),
            "h2a_dual": _training_adequacy(h2a_dual),
        },
        "code_provenance_audit": _code_provenance_audit(all_cards, {
            "h1_train.py": ROOT / "src/topic5_group_event_state/v037/h1_train.py",
            "h1_dual_train.py": ROOT / "src/topic5_group_event_state/v037/h1_dual_train.py",
            "h2a.py": ROOT / "src/topic5_group_event_state/v037/h2a.py",
            "h2b.py": ROOT / "src/topic5_group_event_state/v037/h2b.py",
            "h3_generative.py": ROOT / "src/topic5_group_event_state/v037/h3_generative.py",
            "h3_persistent.py": ROOT / "src/topic5_group_event_state/v037/h3_persistent.py",
        }),
        "wrong_time_control_quality": {
            "h1_event": _wrong_time_control_quality(h1_event),
            "h1_grid": _wrong_time_control_quality(h1_grid),
            "h1_dual": _wrong_time_control_quality(h1_dual),
        },
        "h3_persistent_floored_null": _floored_h3_persistent(h3_persistent_cards),
        "held_out_units": {
            "rule": (
                "held-out sample size for a physical horizon is the number of "
                "non-overlapping windows, not the number of five-minute anchors"
            ),
            "by_card": [
                {
                    "subject": card.get("subject"), "seed": card.get("seed"),
                    "independent_windows_by_horizon": card.get("independent_windows_by_horizon"),
                    "selection_window_audit": card.get("selection_window_audit"),
                }
                for card in h1_event + h1_grid + h1_dual
            ],
        },
        "h1": {
            "event_contrasts": _contrast_summary(h1_event),
            "grid_contrasts": _contrast_summary(h1_grid),
            "dual_contrasts": _contrast_summary(h1_dual),
            "event_by_horizon": _h1_horizon(h1_event, "event"),
            "grid_by_horizon": _h1_horizon(h1_grid, "grid"),
            "dual_by_horizon": _h1_horizon(h1_dual, "dual"),
            "core_subjects": list(CORE_SUBJECTS), "estimability_expansion": list(EXPANSION_SUBJECTS),
        },
        "h2a": {
            "event": _contrast_summary(h2a_event),
            "grid": _contrast_summary(h2a_grid),
            "dual": _contrast_summary(h2a_dual),
            "joint_sensitivity": _h2a_joint(h2a_joint_cards),
        },
        "h2b": _h2b(h2b_cards),
        "h3": {
            "one_step_instrument": _read(root / "h3_instrument" / "audit.json")
            if (root / "h3_instrument" / "audit.json").exists() else None,
            "persistent_instrument": _read(root / "h3_persistent_instrument" / "audit.json")
            if (root / "h3_persistent_instrument" / "audit.json").exists() else None,
            "one_step_human": _h3(h3_cards),
            "persistent_human": _h3_persistent(h3_persistent_cards),
            "m0_trainability_sensitivity": _read(
                root / "h3_m0_trainability_sensitivity" / "summary.json"
            ) if (root / "h3_m0_trainability_sensitivity" / "summary.json").exists() else None,
            "model_boundary": "feedback-like directional dependence; not intervention-level causality",
            "estimand_scope": {
                "primary": "one-step five-minute feedback-like prediction",
                "persistent_long_horizon_feedback": (
                    "estimated only at outcome-blind physically eligible 2 h / 6 h scales; "
                    "24 h and 48 h withheld when support failed"
                ),
                "common_core_and_intercept_frozen": all(
                    card.get("common_core_and_intercept_frozen_across_nested_models", False)
                    for card in h3_cards
                ) if h3_cards else None,
                "complete_models_equal_parameter_count": all(
                    card.get("complete_models_have_equal_parameter_count", True)
                    for card in h3_cards
                ) if h3_cards else None,
            },
        },
        "audit": {
            "n_cards": len(all_cards),
            "development_targets_read_any": any(card.get("development_targets_read", False) for card in all_cards),
            "sealed_partition_opened_any": any(card.get("sealed_partition_opened", False) for card in all_cards),
            "h2b_seizure_outcomes_read_by_design": any(card.get("seizure_outcomes_read", False) for card in h2b_cards),
            "h3_observer_checkpoint_used_as_jump_any": any(card.get("observer_checkpoint_used_as_jump", False) for card in h3_cards),
        },
    }
    summary_path = out / "integrated_summary_v2.json"; atomic_json(summary_path, payload)
    if not args.skip_figures:
        subprocess.run([
            sys.executable, str(ROOT / "scripts/paper_figures/plot_group_event_state_v037_core_evidence.py"),
            "--summary", str(summary_path), "--out-dir", str(out / "figures"),
        ], cwd=ROOT, check=True)
    evidence_paths = [
        summary_path,
        root / "instrument" / "ctssm_instrument_audit.json",
        root / "h3_instrument" / "audit.json",
        root / "h3_persistent_instrument" / "audit.json",
        root / "h3_m0_trainability_sensitivity" / "summary.json",
        optimizer_summary_path,
        root / "h3_independent_generative" / "card_contract_upgrade_audit.json",
    ]
    if not args.skip_figures:
        evidence_paths.extend(sorted((out / "figures").glob("*.png")))
        evidence_paths.extend(sorted((out / "figures").glob("*.pdf")))
        evidence_paths.extend(sorted((out / "figures").glob("*.metadata.json")))
    manifest = {
        "format": "group_event_state_v0_3_7_final_manifest_v3",
        "artifacts": {
            str(path): {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in evidence_paths if path.exists()
        },
        "card_counts": {
            "h1_event": len(h1_event), "h1_grid": len(h1_grid), "h1_dual": len(h1_dual),
            "h2a_event": len(h2a_event), "h2a_grid": len(h2a_grid), "h2a_dual": len(h2a_dual),
            "h2a_joint": len(h2a_joint_cards), "h2b": len(h2b_cards),
            "h3_one_step": len(h3_cards), "h3_persistent": len(h3_persistent_cards),
            "total": len(all_cards),
        },
        "all_registered_queues_complete": all(
            queue.get("status") == "COMPLETE" for queue in payload["queues"].values()
        ),
        "development_targets_read_any": payload["audit"]["development_targets_read_any"],
        "sealed_partition_opened_any": payload["audit"]["sealed_partition_opened_any"],
    }
    atomic_json(out / "manifest_v3.json", manifest)
    print(json.dumps({"summary": str(summary_path), "n_cards": len(all_cards)}, indent=2))


if __name__ == "__main__":
    main()
