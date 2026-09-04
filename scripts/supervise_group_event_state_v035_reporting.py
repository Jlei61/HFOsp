#!/usr/bin/env python3
"""Wait for every registered v0.3.5 queue, then build the final evidence pack."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import OUTPUT_ROOT, atomic_json  # noqa: E402

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
DEPENDENCIES = {
    "rate_search": OUTPUT_ROOT / "dynamic_rate_search_supervisor" / "queue_done.json",
    "search": OUTPUT_ROOT / "full_mark_search_supervisor" / "queue_done.json",
    "final": OUTPUT_ROOT / "final_supervisor" / "queue_done.json",
    "future_oracle": OUTPUT_ROOT / "stepwise_oracle_supervisor" / "queue_done.json",
    "background": OUTPUT_ROOT / "background_rate_final_extra_supervisor" / "queue_done.json",
}


def main() -> None:
    root = OUTPUT_ROOT / "reporting_supervisor"
    root.mkdir(parents=True, exist_ok=True)
    while True:
        missing = [name for name, path in DEPENDENCIES.items() if not path.exists()]
        if missing:
            atomic_json(root / "queue_state.json", {"format":"group_event_state_v0_3_5_reporting_queue_v1",
                        "status":"WAITING", "missing":missing, "updated_epoch":time.time()})
            time.sleep(30); continue
        payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in DEPENDENCIES.items()}
        failures = {name: payload.get("failed", []) for name, payload in payloads.items() if payload.get("failed")}
        if failures:
            atomic_json(root / "queue_state.json", {"format":"group_event_state_v0_3_5_reporting_queue_v1",
                        "status":"WAITING_FOR_REPAIR", "failures":failures, "updated_epoch":time.time()})
            time.sleep(60); continue
        break
    atomic_json(root / "queue_state.json", {"format":"group_event_state_v0_3_5_reporting_queue_v1",
                "status":"FINALIZING", "updated_epoch":time.time()})
    # Recompute the availability/provenance table after second-wave decoders
    # have been built.  This is read-only with respect to scientific targets.
    subprocess.run([str(PYTHON), str(ROOT / "scripts/audit_group_event_state_v035_scope_and_estimability.py")],
                   cwd=ROOT, check=True)
    subprocess.run([str(PYTHON), str(ROOT / "scripts/audit_group_event_state_v035_q_trajectory_equivalence.py")],
                   cwd=ROOT, check=True)
    subprocess.run([str(PYTHON), str(ROOT / "scripts/finalize_group_event_state_v035.py")],
                   cwd=ROOT, check=True)
    scope_path = OUTPUT_ROOT / "scope_manifest.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    evidence = {
        "W0":[OUTPUT_ROOT/"audit/timescale_estimability.csv", OUTPUT_ROOT/"audit/split_and_decoder_provenance.json",
              OUTPUT_ROOT/"audit/q_trajectory_equivalence.json"],
        "W1":list((OUTPUT_ROOT/"dynamic_rate_final").glob("**/card.json"))+list((OUTPUT_ROOT/"background_rate_final").glob("**/card.json"))+[OUTPUT_ROOT/"dynamic_rate_search/selected_recipe.json"],
        "W2":list((OUTPUT_ROOT/"stepwise_decoder").glob("**/card.json"))+list((OUTPUT_ROOT/"stepwise_oracle").glob("**/card.json")),
        "W3":list((OUTPUT_ROOT/"full_mark_final").glob("**/card.json")),
        "W4":list((OUTPUT_ROOT/"functional_readouts_final").glob("**/card.json"))+list((OUTPUT_ROOT/"stepwise_auxiliary_final").glob("**/card.json")),
        "W5":list((OUTPUT_ROOT/"seizure_transfer_final").glob("**/card.json")),
        "W6":list((OUTPUT_ROOT/"feedback_models_final").glob("**/card.json")),
        "REPORT":list((OUTPUT_ROOT/"final_reports").glob("*.json"))+list((OUTPUT_ROOT/"final_reports/figures").glob("*.png")),
    }
    # Completion is an execution claim, not a formatting convenience.  The
    # registered final cohort has seven trainable subjects x three seeds; E922
    # is separately retained as NOT_ESTIMABLE because its mature decoder has
    # no event in the registered evaluation window.  Do not let an empty or
    # partial result directory become a COMPLETE work package.
    expected_final_units = 7 * 3
    required_counts = {
        "W1": 30 + 30,  # dynamic q plus fixed-clock background, incl. extra seeds
        "W2": 21 + 9,  # human stepwise units plus future-oracle sensitivity units
        "W3": expected_final_units,
        "W4": expected_final_units * 2,
        "W5": expected_final_units,
        "W6": expected_final_units,
    }
    actual_counts = {name: len(paths) for name, paths in evidence.items()}
    short = {
        "W1": actual_counts["W1"], "W2": actual_counts["W2"],
        "W3": actual_counts["W3"], "W4": actual_counts["W4"],
        "W5": actual_counts["W5"], "W6": actual_counts["W6"],
    }
    incomplete = {
        name: {"expected": required_counts[name], "actual": short[name]}
        for name in required_counts if short[name] < required_counts[name]
    }
    if incomplete:
        atomic_json(root / "queue_state.json", {
            "format": "group_event_state_v0_3_5_reporting_queue_v1",
            "status": "WAITING_FOR_COMPLETE_REGISTERED_OUTPUTS",
            "incomplete": incomplete, "updated_epoch": time.time(),
        })
        raise RuntimeError(f"registered v0.3.5 outputs incomplete: {incomplete}")
    for name, paths in evidence.items():
        scope["work_packages"][name]["status"] = "COMPLETE"
        scope["work_packages"][name]["evidence"] = [str(p) for p in sorted(paths)]
        scope["work_packages"][name]["n_evidence_files"] = len(paths)
    # Patient-specific assay limitations remain explicitly visible even though
    # every registered unit has been attempted and the package is complete.
    scope["work_packages"]["W3"]["not_estimable"] = [str(p) for p in sorted((OUTPUT_ROOT/"full_mark_final").glob("*/NOT_ESTIMABLE.json"))]
    scope["overall_status"] = "COMPLETE"
    scope["completion_note"] = "all W0-W6 attempted; patient-endpoint NOT_ESTIMABLE records retained"
    atomic_json(scope_path, scope)
    atomic_json(root / "queue_done.json", {"format":"group_event_state_v0_3_5_reporting_done_v1",
                "status":"COMPLETE", "scope_manifest":str(scope_path),
                "final_reports":str(OUTPUT_ROOT/"final_reports")})


if __name__ == "__main__":
    main()
