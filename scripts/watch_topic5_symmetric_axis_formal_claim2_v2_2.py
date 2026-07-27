#!/usr/bin/env python3
"""Poll formal Claim-2 runs and invoke the frozen analyzer at completion."""
from __future__ import annotations

import json
import hashlib
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "results/topic5_symmetric_axis_propagation_state_v2_2/formal"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot() -> dict:
    cohort = json.loads(
        (
            ROOT
            / "results/topic5_symmetric_axis_propagation_state_v2_2/input_audit/"
            "physical_axis_formal_cohort.json"
        ).read_text(encoding="utf-8")
    )["subjects"]
    cohort_set = set(cohort)
    states = {"COMPLETE": 0, "RUNNING": 0, "FAILED": 0, "MISSING": 0}
    failures = []
    contract_violations = []
    core_hash = sha256(
        ROOT / "src/topic5_symmetric_axis_propagation_state_v2_2.py"
    )
    trainer_hash = sha256(
        ROOT / "scripts/train_topic5_symmetric_axis_formal_claim2_v2_2.py"
    )
    target_gate = json.loads(
        (
            ROOT
            / "results/topic5_symmetric_axis_propagation_state_v2_2/"
            "target_audit/TARGET_METADATA_GATE.json"
        ).read_text(encoding="utf-8")
    )
    if (
        target_gate.get("energy_values_read")
        or target_gate.get("recruitment_values_read")
        or target_gate.get("early_ictal_transfer_allowed")
    ):
        contract_violations.append("target seal or transfer blocker drifted")
    for subject in cohort:
        for seed in (17, 29, 43):
            run = BASE / "claim2_runs" / subject / f"seed_{seed}"
            path = run / "run_state.json"
            if not path.is_file():
                states["MISSING"] += 1
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            status = str(record.get("status", "MISSING"))
            states[status] = states.get(status, 0) + 1
            if status == "FAILED":
                failures.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "error": record.get("error"),
                    }
                )
            resolved_path = run / "resolved_config.json"
            if not resolved_path.is_file():
                contract_violations.append(
                    f"{subject}/seed_{seed}: missing resolved config"
                )
                continue
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            training = set(resolved.get("shared_training_subjects", []))
            checks = {
                "objective_h3": (
                    resolved.get("selected_objective")
                    == "next_plus_rollout_h3"
                    and resolved.get("H_train") == 3
                ),
                "exact_loso": (
                    len(training) == 21
                    and subject not in training
                    and training == cohort_set - {subject}
                ),
                "target_not_read": resolved.get("target_values_read") is False,
                "core_hash": resolved.get("core_sha256") == core_hash,
                "formal_trainer_hash": (
                    resolved.get("formal_trainer_sha256") == trainer_hash
                ),
                "bias_inventory_22": len(
                    resolved.get("node_bias_sha256", {})
                )
                == 22,
            }
            if not all(checks.values()):
                contract_violations.append(
                    f"{subject}/seed_{seed}: {checks}"
                )
            metrics_path = run / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if (
                    not metrics.get("full_control_bias_identical")
                    or metrics.get("target_values_read")
                ):
                    contract_violations.append(
                        f"{subject}/seed_{seed}: completed metric contract drift"
                    )
    return {
        "unix_time": time.time(),
        "states": states,
        "failures": failures,
        "contract_violations": contract_violations,
        "contract_status": "PASS" if not contract_violations else "FAIL",
    }


def main() -> None:
    log = BASE / "claim2_watch.jsonl"
    while True:
        status = snapshot()
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(status, ensure_ascii=False) + "\n")
        print(json.dumps(status), flush=True)
        if status["states"]["FAILED"]:
            raise SystemExit("formal Claim-2 grid has failed runs")
        if status["contract_violations"]:
            raise SystemExit("formal Claim-2 runtime contract drifted")
        if status["states"]["COMPLETE"] == 66:
            subprocess.run(
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "cuda_env",
                    "python",
                    "scripts/analyze_topic5_symmetric_axis_formal_claim2_v2_2.py",
                ],
                cwd=ROOT,
                check=True,
            )
            print("formal Claim-2 analysis complete", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
