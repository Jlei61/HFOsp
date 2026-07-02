"""Topic 4 M3-v2.2 approach-criticality config loader (Task 0).

Loads the config-of-record `config/topic4_criticality.yaml`: operator units,
verdict thresholds + threshold-sweep, quality-gate floors, branching policy,
mode-selection policy, finite-time-gain horizons, the slow_to_ratefield entry
terminology lock, slow_sensitivity finite-difference steps, atlas grid, and
the virtual_seeg estimator-reuse contract.

This module will be heavily extended by later tasks (spec
docs/superpowers/specs/2026-07-02-topic4-m3v2-2-approach-criticality-design.md);
kept to the config loader only for now.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "topic4_criticality.yaml"


def load_crit_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the topic4 criticality config YAML as a dict.

    path=None resolves to config/topic4_criticality.yaml relative to the repo root.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# M3A-v2.2 -> M3B-R2 interface export (Task 1): fail-closed handoff wiring       #
# --------------------------------------------------------------------------- #
# The v2.2 approach-to-criticality sim feeds the SAME canonical M3A->M3B handoff
# contract as A2 (src/sef_hfo_m3_interface.py + src/sef_hfo_m3a_export.py). The
# real export is EXPECTED to refuse the phase-map overlay because the slow->rate
# mapping for this sim is NOT calibrated -- that refusal is a science outcome, not
# an adapter bug. export_fixture_handoff isolates "the machinery works" from "real
# data legitimately refuses" by feeding a hand-built sign-calibrated mapping that
# passes all four overlay conditions.


def _fixture_calibrated_mapping_and_ranges(mapping_id: str):
    """A hand-built SIGN-calibrated mapping (+ranges) that passes all four overlay
    conditions -- the KNOWN-GOOD control for export_fixture_handoff.

    Starts from the canonical uncalibrated placeholder (schema-valid, physically sensible
    reciprocal-affine transforms) and flips only the two on-axis coords to
    calibration_status='passed' with a passing sign_test. The transforms are strictly
    monotone over their calibrated input domain, so check_sign_direction holds.
    """
    from src.sef_hfo_m3_interface import ON_AXIS_COORDS
    from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges
    mapping, ranges = default_precalib_mapping_and_ranges(mapping_id)
    for coord in ON_AXIS_COORDS:
        c = mapping["coordinates"][coord]
        t = c["transform"]
        c["calibration_status"] = "passed"
        c["sign_tests"] = [{
            "name": f"{coord}_sign_cal", "coord": coord, "input_var": t["input_var"],
            "expected_direction": t["expected_direction"],
            "observed_slope_sign": ("negative" if t["expected_direction"] == "decreasing_in_input"
                                    else "positive"),
            "passed": True, "engine_sha": "fixture",
        }]
    return mapping, ranges


def _fixture_landmark_rows() -> list:
    """Landmark rows whose slow values sit solidly inside the calibrated input domain
    (q in [0.25,1] -> phase in [0,1]) with canonical event_stages, so every trajectory
    row is phase_coord_valid and in-range (satisfies cond3)."""
    return [
        {"time_ms": 0.0, "event_id": 0, "event_stage": "onset", "q_core": 0.90, "q_global": 0.90, "g_K": 0.10},
        {"time_ms": 10.0, "event_id": 0, "event_stage": "peak", "q_core": 0.55, "q_global": 0.65, "g_K": 0.30},
        {"time_ms": 20.0, "event_id": 0, "event_stage": "end", "q_core": 0.40, "q_global": 0.50, "g_K": 0.50},
    ]


def _fixture_passing_summary(mapping_id: str) -> dict:
    """Summary satisfying cond4 (STRICT A2 gate): gate_A PASS + rate_matched passed +
    rate_matched_group recorded."""
    return {
        "slow_to_rate_mapping_id": mapping_id,
        "gate_A_trajectory": "PASS",
        "gate_B_seizure_like": "INCONCLUSIVE",
        "trajectory_robustness": "robust",
        "rate_matched_control": "passed",
        "rate_matched_group": {"n": 8, "source": "fixture"},
        "out_of_range_fraction": 0.0,
        "forbidden_claims": [],
    }


def export_fixture_handoff(out_dir) -> str:
    """Write a KNOWN-GOOD handoff (calibrated mapping + passing phenotype summary) and
    return its overlay_verdict. Guaranteed 'phase_map_trajectory' -- proves the M3A->M3B
    interface machinery is wired correctly, so a 'refused' verdict on real data is a
    science outcome, not an adapter bug."""
    from src.sef_hfo_m3a_export import write_handoff_artifacts
    mapping_id = "m3a_v2_2_fixture"
    mapping, ranges = _fixture_calibrated_mapping_and_ranges(mapping_id)
    audit = write_handoff_artifacts(
        str(out_dir),
        landmark_rows=_fixture_landmark_rows(),
        mapping=mapping, ranges=ranges,
        summary=_fixture_passing_summary(mapping_id),
    )
    return audit["overlay_verdict"]


def export_v2_2_handoff(out_dir, cfg: Dict[str, Any]) -> str:
    """Run the v2.2 transition sim and write the fail-closed M3A->M3B handoff artifacts.

    Uses the DEFAULT uncalibrated mapping (build_handoff_from_sim mapping/ranges=None) so the
    self-audit legitimately REFUSES the phase-map overlay: the slow->rate mapping for this
    approach-to-criticality sim is not calibrated and the phenotype gate is INCONCLUSIVE.
    The mapping is NOT weakened to force a pass -- refusal here is the honest verdict.
    Returns the overlay_verdict (expected 'refused' / 'mechanism_candidate_only').
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.sef_hfo_transition_sim import run_transition, sim_dict_for_handoff
    from src.sef_hfo_m3a_export import build_handoff_from_sim, write_handoff_artifacts
    res = run_transition(cfg)
    h = build_handoff_from_sim(
        sim_dict_for_handoff(res), res["events"], res["dt_ms"],
        mapping_id="m3a_v2_2_approach", gk_enabled=cfg["use_gK"],
    )
    audit = write_handoff_artifacts(str(out_dir), **h)
    return audit["overlay_verdict"]
