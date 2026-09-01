"""Finalize a completed ZM1.1 decision after a controller-only failure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DECISION_BY_PHASE = {
    "fit": "tau_adp_fit_decision.json",
    "selection": "tau_adp_selection_decision.json",
    "confirmation": "tau_adp_confirmation_decision.json",
}


def finalize(config_path, expected_runtime_commit, source_decision_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    phase = config["search"]["phase"]
    root = ROOT / config["output_root"]
    decision_path = root / DECISION_BY_PHASE[phase]
    decision = json.loads(decision_path.read_text())
    if decision.get("phase") != phase:
        raise RuntimeError("decision phase does not match config")
    if decision["inputs"]["config"]["sha256"] != _sha256(config_path):
        raise RuntimeError("decision config hash changed")
    for key in ("manifest", "summary"):
        record = decision["inputs"][key]
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"decision {key} input hash changed")
    source = decision["provenance"]
    if (source.get("expected_git_commit") != source_decision_commit
            or source.get("runtime_modules_dirty")
            or not source.get("runtime_modules_match_expected_commit")):
        raise RuntimeError("source decision provenance is not frozen")
    runtime = _runtime_provenance(expected_runtime_commit)
    if (runtime["runtime_modules_dirty"]
            or not runtime["runtime_modules_match_expected_commit"]):
        raise RuntimeError("finalizer runtime is not frozen")
    no_advancement = decision["status"].endswith(
        "NO_SAFE_EVALUABLE_CANDIDATE"
    )
    payload = {
        "status": (
            f"REV10ZM1_1_TAU_{phase.upper()}_NO_ADVANCEMENT_COMPLETE"
            if no_advancement
            else f"REV10ZM1_1_TAU_{phase.upper()}_FINALIZED"
        ),
        "phase": phase,
        "decision_status": decision["status"],
        "advance_to_next_phase": not no_advancement,
        "decision": {
            "path": str(decision_path.relative_to(ROOT)),
            "sha256": _sha256(decision_path),
            "source_expected_commit": source_decision_commit,
        },
        "finalizer_provenance": runtime,
    }
    atomic_write_json(payload, root / "DONE.json")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-runtime-commit", required=True)
    parser.add_argument("--source-decision-commit", required=True)
    args = parser.parse_args()
    payload = finalize(
        args.config, args.expected_runtime_commit, args.source_decision_commit,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
