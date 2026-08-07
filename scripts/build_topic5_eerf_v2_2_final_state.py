#!/usr/bin/env python3
"""Build the fail-closed final acceptance state for EERF v2.2."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/topic5_event_indexed_evolving_rank_field/development"
V21 = ROOT / "results/topic5_stable_interaction_graph/development/SIG_V2_1_ACCEPTANCE.json"
PILOT_AUDIT = RESULT_ROOT / "input_audit/EVENT_INDEXED_INPUT_AUDIT_PILOT.json"
ALL_AUDIT = RESULT_ROOT / "input_audit/EVENT_INDEXED_INPUT_AUDIT_ALL_INVENTORY.json"
PHASE0 = RESULT_ROOT / "phase0_observability/EERF_V2_2_PHASE0_STATE.json"
PHASE1 = RESULT_ROOT / "phase1_event_history/EERF_V2_2_PHASE1_STATE.json"
OUTPUT = RESULT_ROOT / "EERF_V2_2_FINAL_STATE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_dependency_hashes(state: dict[str, Any], source: Path, module: Path, config: Path) -> None:
    require(state["source_sha256"] == sha256(source), f"source hash drift: {source}")
    require(state["module_sha256"] == sha256(module), f"module hash drift: {module}")
    require(state["config_sha256"] == sha256(config), f"config hash drift: {config}")


def main() -> None:
    v21 = load(V21)
    pilot = load(PILOT_AUDIT)
    inventory = load(ALL_AUDIT)
    phase0 = load(PHASE0)
    phase1 = load(PHASE1)
    require(v21.get("status") == "ACCEPTED_AND_FROZEN", "v2.1 is not frozen")
    require(pilot.get("n_subjects") == 6 and pilot.get("n_pass") == 6, "pilot input audit failed")
    require(inventory.get("n_subjects") == 34 and inventory.get("n_pass") == 34, "inventory audit failed")
    require(phase0.get("status") == "COMPLETE_PHASE0_DEVELOPMENT", "Phase 0 incomplete")
    require(phase0.get("n_elr_authorized") == 2, "unexpected Phase-0 eligibility count")
    require(phase1.get("status") == "COMPLETE_DEVELOPMENT_PHASE1", "Phase 1 incomplete")
    require(phase1.get("n_phase1_pass") == 0, "Phase-1 result is not the frozen bounded negative")
    require(phase1.get("decision") == "STOP_EVENT_DRIVEN_ELR", "Phase-1 stop decision drift")
    verify_dependency_hashes(
        phase0,
        ROOT / "scripts/run_topic5_event_indexed_observability_v2_2.py",
        ROOT / "src/topic5_event_indexed_evolving_rank_field.py",
        ROOT / "config/topic5_event_indexed_evolving_rank_field_v2_2.yaml",
    )
    verify_dependency_hashes(
        phase1,
        ROOT / "scripts/run_topic5_eerf_v2_2_phase1.py",
        ROOT / "src/topic5_event_history_increment.py",
        ROOT / "config/topic5_event_indexed_evolving_rank_field_v2_2_phase1.yaml",
    )
    for state in (phase0, phase1):
        require(not state.get("old_heldout20_entered_into_analysis"), "old heldout20 leakage flag")
        require(not state.get("snn_inputs_read"), "SNN leakage flag")
        require(not state.get("forbidden_labels_read"), "forbidden label leakage flag")
    payload = {
        "contract": "topic5_event_indexed_evolving_rank_field_v2_2",
        "status": "ACCEPTED_BOUNDED_NEGATIVE_EVENT_HISTORY_INCREMENT",
        "decision": "DO_NOT_IMPLEMENT_EVENT_DRIVEN_ELR_RNN",
        "v2_1_status": v21["status"],
        "input_audit": {"pilot": "6/6 PASS", "all_inventory": "34/34 PASS"},
        "phase0": {
            "n_subjects": phase0["n_subjects"],
            "n_block_reliable": phase0["n_block_reliable"],
            "n_g0_pass": phase0["n_g0_pass"],
            "n_temporal_structure_supportive": phase0["n_temporal_structure_supportive"],
            "n_phase1_eligible": phase0["n_elr_authorized"],
        },
        "phase1": {
            "n_eligible": phase1["n_phase0_eligible"],
            "n_pass": phase1["n_phase1_pass"],
            "decision": phase1["decision"],
        },
        "safe_claim": (
            "Most development pilots show observable block-wise variation around a "
            "patient-specific propagation field, but chronology-sensitive event-history "
            "features do not add stable next-block information beyond autonomous/state controls."
        ),
        "forbidden_claims": [
            "interictal events causally shape the pathological network",
            "an evolving pathological graph was identified",
            "IEI defines a recovery or plasticity time constant",
            "independent cohort confirmation",
            "human RNN interpretation depends on an SNN gate",
        ],
        "next_action": (
            "Stop this model branch. Do not expand to 34 patients or add GRU, hidden "
            "dimensions, process noise, or a general contact graph as rescue analyses."
        ),
        "dependencies": {
            "v2_1_acceptance": {"path": str(V21), "sha256": sha256(V21)},
            "pilot_input_audit": {"path": str(PILOT_AUDIT), "sha256": sha256(PILOT_AUDIT)},
            "inventory_input_audit": {"path": str(ALL_AUDIT), "sha256": sha256(ALL_AUDIT)},
            "phase0_state": {"path": str(PHASE0), "sha256": sha256(PHASE0)},
            "phase1_state": {"path": str(PHASE1), "sha256": sha256(PHASE1)},
        },
        "old_heldout20_entered_into_analysis": False,
        "snn_inputs_read": False,
        "forbidden_labels_read": False,
        "builder_sha256": sha256(Path(__file__)),
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
