"""Freeze one rev10-ZM1.1 tau_adp phase before its network realizations."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROLE = "development_only_data_driven_h_zm_tau_adp_calibration"
STATUS_BY_PHASE = {
    "fit": "REV10ZM1_1_H_ZM_TAU_FIT_LIBRARY_FROZEN",
    "selection": "REV10ZM1_1_H_ZM_TAU_SELECTION_LIBRARY_FROZEN",
    "confirmation": "REV10ZM1_1_H_ZM_TAU_CONFIRMATION_LIBRARY_FROZEN",
}
CONTROL_ID = "h_spou_slow_off"


def candidate_id(tau_adp_ms):
    return f"h_spou_zm_tau{int(round(float(tau_adp_ms))):04d}"


def _source_candidate(config):
    source_path = ROOT / config["inputs"]["source_d5_2_manifest"]["path"]
    verdict_path = ROOT / config["inputs"]["source_d5_2_verdict"]["path"]
    source = json.loads(source_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    if source.get("status") != (
            "REV10D5_2_SPATIAL_OU_CONFIRMATION_LIBRARY_FROZEN"):
        raise RuntimeError("frozen D5.2 source manifest is invalid")
    source_id = verdict.get("selected_local_candidate_id")
    candidates = {
        row["candidate_id"]: row for row in source["candidate_set"]["candidates"]
    }
    if source_id not in candidates:
        raise RuntimeError("frozen D5.2 source candidate is absent")
    candidate = candidates[source_id]
    if candidate.get("spatial_ou", {}).get("mode") != "local":
        raise RuntimeError("ZM1.1 must retain the frozen local OU process")
    if not all(value == 0.0 for value in candidate["coefficients"]):
        raise RuntimeError("ZM1.1 requires exact no-op edge coefficients")
    return source_id, candidate, source


def _decision_candidate_ids(config, phase, decision_path):
    all_ids = [
        candidate_id(value)
        for value in config["mz_calibration"]["candidate_tau_adp_ms"]
    ]
    if phase == "fit":
        return all_ids, None
    if decision_path is None:
        raise RuntimeError(f"{phase} requires an upstream decision")
    decision_path = Path(decision_path).resolve()
    decision = json.loads(decision_path.read_text())
    if phase == "selection":
        selected = list(decision.get("shortlisted_candidate_ids", []))
    else:
        selected_id = decision.get("selected_candidate_id")
        selected = [] if selected_id is None else [selected_id]
    if not selected or any(value not in all_ids for value in selected):
        raise RuntimeError(f"invalid {phase} candidate set from upstream decision")
    return selected, {
        "path": str(decision_path.relative_to(ROOT)),
        "sha256": _sha256(decision_path),
        "status": decision.get("status"),
    }


def build_manifest(config_path, expected_commit, decision_path=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    phase = config["search"]["phase"]
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("ZM1.1 scientific role changed")
    if phase not in STATUS_BY_PHASE:
        raise RuntimeError("ZM1.1 phase is invalid")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    source_id, source_candidate, source_manifest = _source_candidate(config)
    active_ids, upstream = _decision_candidate_ids(
        config, phase, decision_path,
    )
    slow_off = copy.deepcopy(source_candidate)
    slow_off["candidate_id"] = CONTROL_ID
    slow_off["mz"] = {
        "mode": "off", "use_z": False, "use_m": False,
        "trace_stride_steps": int(config["mz_calibration"]["trace_stride_steps"]),
    }
    by_id = {}
    fixed = config["mz_calibration"]
    for tau in fixed["candidate_tau_adp_ms"]:
        active = copy.deepcopy(source_candidate)
        active["candidate_id"] = candidate_id(tau)
        active["mz"] = {
            "mode": "z_plus_m", "use_z": True, "use_m": True,
            "I_th_EI": float(fixed["I_th_EI"]),
            "tau_z": float(fixed["tau_z"]),
            "tau_adp": float(tau),
            "eta_m": float(fixed["eta_m"]),
            "trace_stride_steps": int(fixed["trace_stride_steps"]),
        }
        by_id[active["candidate_id"]] = active
    candidates = [slow_off] + [by_id[value] for value in active_ids]

    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if (config_dirty or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("ZM1.1 freezer runtime is not frozen")

    seed_key = f"{phase}_network_seeds"
    return {
        "status": STATUS_BY_PHASE[phase],
        "scientific_role": EXPECTED_ROLE,
        "candidate_set": {
            "candidates": candidates,
            "paired_candidate_ids": [CONTROL_ID, *active_ids],
        },
        "selection_freeze": {
            "selected_nonzero_candidate_id": active_ids[0],
            "paired_control_candidate_id": CONTROL_ID,
            "selected_before_active_phase_networks": True,
            "active_phase_networks_were_read": False,
            "upstream_decision": upstream,
        },
        "source_freeze": {
            "d5_2_candidate_id": source_id,
            "d5_2_manifest": config["inputs"]["source_d5_2_manifest"],
            "d5_2_verdict": config["inputs"]["source_d5_2_verdict"],
            "mz_fixed_parameters": {
                key: fixed[key] for key in (
                    "I_th_EI", "tau_z", "eta_m", "trace_stride_steps",
                )
            },
            "varied_parameter_only": "tau_adp",
        },
        "direction_classifier": source_manifest["direction_classifier"],
        "fixed_contract": {
            "phase": phase,
            "network_seeds": config["search"][seed_key],
            "simulation": config["search"]["simulation"],
            "same_network_and_dynamics_seeds_across_arms": True,
            "spatial_ou_is_identical_across_arms": True,
            "static_edge_coefficients": "all exact zero",
            "beta": "closed",
        },
        "claim_boundary": config["claim_boundary"],
        "inputs": config["inputs"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--upstream-decision")
    parser.add_argument("--out")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(
        args.out or ROOT / config["output_root"] / "candidate_manifest.json"
    )
    payload = build_manifest(
        args.config, args.expected_commit, args.upstream_decision,
    )
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "candidates": payload["candidate_set"]["paired_candidate_ids"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
