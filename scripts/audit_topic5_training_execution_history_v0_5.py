#!/usr/bin/env python3
"""Persist the complete v0.5 training-attempt history without rewriting logs.

The final phase tables describe the successful immutable execution, whereas
``runner.log`` files can also retain attempts from an earlier fail-closed
launch.  This audit keeps those two facts separate.  It is read-only with
respect to checkpoints and never imports or reads an early-ictal target.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("training execution-history audit must be frozen before target unseal")

    phase_paths = sorted(out.glob("PHASE_*_EXECUTION.csv"))
    phase = pd.concat([pd.read_csv(path) for path in phase_paths], ignore_index=True)
    rows = []
    for log in sorted((out / "formal_units").glob("*/*/seed*/runner.log")):
        text = log.read_text(errors="replace")
        relative = log.relative_to(out / "formal_units")
        parts = relative.parts
        rows.append({
            "fit_id": parts[0],
            "arm": parts[1],
            "seed": int(parts[2].replace("seed", "")),
            "path": str(log),
            "sha256": sha256_file(log),
            "attempt_markers_all_launches": text.count("ATTEMPT "),
            "tracebacks_all_launches": text.count("Traceback (most recent call last):"),
            "keyerror_n_contacts_all_launches": text.count("KeyError: 'n_contacts'"),
            "cuda_oom_all_launches": text.lower().count("cuda out of memory"),
            "out_of_memory_all_launches": text.lower().count("out of memory"),
            "terminal_success_payloads": text.count('"target_values_read": false'),
        })
    history = pd.DataFrame(rows)
    destination = out / "TRAINING_EXECUTION_HISTORY_AUDIT.csv"
    history.to_csv(destination, index=False)

    final_key = phase.assign(
        internal_arm=phase.arm.map({
            "L0": "L0_LOCAL_ONLY",
            "L1": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
            "L2m": "L2M_MACRO_MATCHED_RANDOM_LR",
            "L3": "L3_LOCAL_PLUS_LEARNED_LR",
            "C-suffix": "C_L3_ORDER_SHUFFLED",
        })
    )[["fit_id", "internal_arm", "seed", "status", "returncode", "attempt"]]
    joined = history.merge(
        final_key,
        left_on=["fit_id", "arm", "seed"],
        right_on=["fit_id", "internal_arm", "seed"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    failures = []
    if len(history) != 531 or history[["fit_id", "arm", "seed"]].duplicated().any():
        failures.append("HISTORY_UNIT_COVERAGE")
    if not joined._merge.eq("both").all():
        failures.append("FINAL_PHASE_HISTORY_JOIN")
    if not phase.status.astype(str).eq("DONE").all() or not phase.returncode.astype(int).eq(0).all():
        failures.append("FINAL_PHASE_NOT_ALL_DONE")
    if int(history.cuda_oom_all_launches.sum()) != 0:
        failures.append("CUDA_OOM_PRESENT")
    if int(history.out_of_memory_all_launches.sum()) != 0:
        failures.append("OOM_PRESENT")
    if not history.terminal_success_payloads.ge(1).all():
        failures.append("MISSING_TERMINAL_SUCCESS_PAYLOAD")

    payload = {
        "contract": "topic5_v0_5_complete_training_attempt_history",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_RECORDED_PRELAUNCH_INCIDENT" if not failures else "FAIL",
        "target_values_read": False,
        "formal_units": int(len(history)),
        "final_phase_rows": int(len(phase)),
        "final_phase_done": int(phase.status.astype(str).eq("DONE").sum()),
        "final_phase_nonzero_returncode": int(phase.returncode.astype(int).ne(0).sum()),
        "final_phase_retry_attempt_gt_zero": int(phase.attempt.astype(int).gt(0).sum()),
        "units_with_historical_traceback": int(history.tracebacks_all_launches.gt(0).sum()),
        "historical_tracebacks": int(history.tracebacks_all_launches.sum()),
        "units_with_prelaunch_n_contacts_keyerror": int(
            history.keyerror_n_contacts_all_launches.gt(0).sum()
        ),
        "prelaunch_n_contacts_keyerrors": int(history.keyerror_n_contacts_all_launches.sum()),
        "cuda_oom_occurrences_all_launches": int(history.cuda_oom_all_launches.sum()),
        "oom_occurrences_all_launches": int(history.out_of_memory_all_launches.sum()),
        "unresolved_failed_units": int(
            (~phase.status.astype(str).eq("DONE") | phase.returncode.astype(int).ne(0)).sum()
        ),
        "history_table": str(destination),
        "history_table_sha256": sha256_file(destination),
        "phase_table_hashes": {path.name: sha256_file(path) for path in phase_paths},
        "producer_script": str(Path(__file__).resolve()),
        "producer_script_sha256": sha256_file(Path(__file__).resolve()),
        "incident_boundary": (
            "RUNNER_LOGS_RETAIN_A_FAIL_CLOSED_PRELAUNCH_WITH_LEGACY_N_CONTACTS_KEY;_"
            "THE_FINAL_FROZEN_531_UNIT_EXECUTION_COMPLETED_WITH_ATTEMPT_0_AND_NO_OOM"
        ),
        "peak_vram_telemetry": "NOT_RECORDED_PER_UNIT",
        "failures": failures,
    }
    write_json(out / "TRAINING_EXECUTION_HISTORY_AUDIT.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
