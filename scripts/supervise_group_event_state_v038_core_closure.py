#!/usr/bin/env python3
"""Keep the v0.3.8 H1/H2 scientific closure queues continuously occupied.

This supervisor never uses seizure outcomes to select a state producer.  It
first freezes the long and medium H1 trajectories, then releases frozen H2a
and H2b consumers.  Long and medium cohorts remain separate because they use
different, predeclared physical-horizon contracts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
BASE = Path("/data/hfosp_group_event_state_v0_3_8_core_expansion")
SEARCH = Path("/data/hfosp_group_event_state_v0_3_7/repaired_v5/optimizer_search/summary.json")
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)
LONG = ("epilepsiae_1096", "epilepsiae_253", "epilepsiae_958",
        "epilepsiae_1077", "epilepsiae_1125", "epilepsiae_916")
LONG_H2A = tuple(subject for subject in LONG if subject != "epilepsiae_916")
MEDIUM = ("epilepsiae_1146", "epilepsiae_384", "epilepsiae_548",
          "epilepsiae_583", "epilepsiae_922")


def _read(path: Path) -> dict:
    if not path.exists():
        return {"status": "PENDING"}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _launch(name: str, command: list[str]) -> int:
    log = BASE / "supervisor" / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    handle.close()
    return int(process.pid)


def main() -> None:
    h1_long = BASE / "h1_long_8h"
    h1_medium = BASE / "h1_medium_2h"
    h2a_long = BASE / "h2a_long_8h"
    h2a_medium = BASE / "h2a_medium_2h"
    h2b_long = BASE / "h2b_long_8h"
    h2b_medium = BASE / "h2b_medium_2h"
    paths = {
        "h1_long": h1_long / "supervisor/queue_status.json",
        "decoder_long": BASE / "decoder_strict/supervisor/queue_status.json",
        "decoder_medium": BASE / "decoder_strict_medium/supervisor/queue_status.json",
        "h1_medium": h1_medium / "supervisor/queue_status.json",
        "trained_credit_long": BASE / "trained_credit_long/queue_status.json",
        "dual_random_background_long": BASE / "dual_random_background_long_v2/queue_status.json",
        "dual_random_background_medium": BASE / "dual_random_background_medium_v2/queue_status.json",
        "h2a_long": h2a_long / "supervisor/queue_status.json",
        "h2a_medium": h2a_medium / "supervisor/queue_status.json",
        "h2b_long": h2b_long / "supervisor/queue_status.json",
        "h2b_medium": h2b_medium / "supervisor/queue_status.json",
    }
    launched: dict[str, int] = {}
    while True:
        status = {name: _read(path) for name, path in paths.items()}
        failed = {name: row for name, row in status.items() if row.get("status") == "FAILED"}
        if failed:
            _atomic(BASE / "supervisor/queue_status.json", {
                "status": "FAILED", "failed_stages": failed, "stages": status,
                "launched_pids": launched,
            })
            raise SystemExit(1)

        if (status["h1_long"].get("status") == "COMPLETE"
                and status["h1_medium"].get("status") == "PENDING"
                and "h1_medium" not in launched):
            launched["h1_medium"] = _launch("h1_medium", [
                str(PYTHON), "scripts/supervise_group_event_state_v037_optimized_h1.py",
                "--workers-per-gpu", "2", "--gpus", "0,1",
                "--out-root", str(h1_medium), "--search-summary", str(SEARCH),
                "--models", "dual", "event", "grid", "--seeds", *(str(v) for v in SEEDS),
                "--horizons-hours", "0.5", "2.0", "--subjects", *MEDIUM,
            ])

        long_frozen = status["h1_long"].get("status") == "COMPLETE"
        if (long_frozen and status["trained_credit_long"].get("status") == "PENDING"
                and "trained_credit_long" not in launched):
            launched["trained_credit_long"] = _launch("trained_credit_long", [
                str(PYTHON), "scripts/audit_group_event_state_v038_trained_credit.py",
                "--h1-root", str(h1_long), "--out-root", str(BASE / "trained_credit_long"),
                "--device", "cuda:0", "--maximum-anchors", "8",
                "--seeds", *(str(v) for v in SEEDS), "--subjects", *LONG,
            ])
        if (long_frozen and status["dual_random_background_long"].get("status") == "PENDING"
                and "dual_random_background_long" not in launched):
            launched["dual_random_background_long"] = _launch("dual_random_background_long", [
                str(PYTHON), "scripts/audit_group_event_state_v038_dual_random_background.py",
                "--h1-root", str(h1_long),
                "--out-root", str(BASE / "dual_random_background_long_v2"),
                "--device", "cuda:1", "--seeds", *(str(v) for v in SEEDS),
                "--subjects", *LONG,
            ])
        if (long_frozen and status["decoder_long"].get("status") == "COMPLETE"
                and status["h2a_long"].get("status") == "PENDING"
                and "h2a_long" not in launched):
            launched["h2a_long"] = _launch("h2a_long", [
                str(PYTHON), "scripts/supervise_group_event_state_v037_optimized_h2a.py",
                "--workers-per-gpu", "2", "--gpus", "0,1", "--h1-root", str(h1_long),
                "--decoder-root", str(BASE / "decoder_strict"), "--out-root", str(h2a_long),
                "--models", "dual", "event", "grid", "--seeds", *(str(v) for v in SEEDS),
                "--subjects", *LONG_H2A,
            ])
        both_h1_frozen = long_frozen and status["h1_medium"].get("status") == "COMPLETE"
        if (both_h1_frozen and status["h2b_long"].get("status") == "PENDING"
                and "h2b_long" not in launched):
            launched["h2b_long"] = _launch("h2b_long", [
                str(PYTHON), "scripts/supervise_group_event_state_v037_optimized_h2b.py",
                "--workers", "4", "--h1-root", str(h1_long), "--out-root", str(h2b_long),
                "--phase", "freeze", "--seeds", *(str(v) for v in SEEDS), "--subjects", *LONG,
            ])

        medium_frozen = status["h1_medium"].get("status") == "COMPLETE"
        if (medium_frozen and status["dual_random_background_medium"].get("status") == "PENDING"
                and "dual_random_background_medium" not in launched):
            launched["dual_random_background_medium"] = _launch("dual_random_background_medium", [
                str(PYTHON), "scripts/audit_group_event_state_v038_dual_random_background.py",
                "--h1-root", str(h1_medium),
                "--out-root", str(BASE / "dual_random_background_medium_v2"),
                "--device", "cuda:1", "--seeds", *(str(v) for v in SEEDS),
                "--subjects", *MEDIUM,
            ])
        if (medium_frozen and status["decoder_medium"].get("status") == "COMPLETE"
                and status["h2a_medium"].get("status") == "PENDING"
                and "h2a_medium" not in launched):
            launched["h2a_medium"] = _launch("h2a_medium", [
                str(PYTHON), "scripts/supervise_group_event_state_v037_optimized_h2a.py",
                "--workers-per-gpu", "2", "--gpus", "0,1", "--h1-root", str(h1_medium),
                "--decoder-root", str(BASE / "decoder_strict_medium"), "--out-root", str(h2a_medium),
                "--models", "dual", "event", "grid", "--seeds", *(str(v) for v in SEEDS),
                "--subjects", *MEDIUM,
            ])
        if (both_h1_frozen and status["h2b_medium"].get("status") == "PENDING"
                and "h2b_medium" not in launched):
            launched["h2b_medium"] = _launch("h2b_medium", [
                str(PYTHON), "scripts/supervise_group_event_state_v037_optimized_h2b.py",
                "--workers", "4", "--h1-root", str(h1_medium), "--out-root", str(h2b_medium),
                "--phase", "freeze", "--seeds", *(str(v) for v in SEEDS), "--subjects", *MEDIUM,
            ])

        globally_frozen = (
            status["h2b_long"].get("status") == "FEATURES_FROZEN"
            and status["h2b_medium"].get("status") == "FEATURES_FROZEN"
        )
        global_marker = BASE / "supervisor/ALL_INTERICTAL_FEATURES_FROZEN"
        if globally_frozen and not global_marker.exists():
            global_marker.write_text(
                "Long and medium observer features frozen before any seizure outcome access.\n",
                encoding="utf-8",
            )
        for name, output, h1, subjects in (
            ("h2b_long", h2b_long, h1_long, LONG),
            ("h2b_medium", h2b_medium, h1_medium, MEDIUM),
        ):
            launch_marker = output / "supervisor/OUTCOMES_LAUNCHED"
            if globally_frozen and not launch_marker.exists():
                launch_marker.write_text("released by global freeze marker\n", encoding="utf-8")
                launched[f"{name}_outcomes"] = _launch(f"{name}_outcomes", [
                    str(PYTHON), "scripts/supervise_group_event_state_v037_optimized_h2b.py",
                    "--workers", "4", "--h1-root", str(h1), "--out-root", str(output),
                    "--phase", "outcomes", "--global-freeze-marker", str(global_marker),
                    "--seeds", *(str(v) for v in SEEDS), "--subjects", *subjects,
                ])

        status = {name: _read(path) for name, path in paths.items()}
        downstream = ("h1_long", "decoder_long", "decoder_medium", "h1_medium", "trained_credit_long",
                      "dual_random_background_long", "dual_random_background_medium",
                      "h2a_long", "h2a_medium", "h2b_long", "h2b_medium")
        complete = all(status[name].get("status") == "COMPLETE" for name in downstream)
        _atomic(BASE / "supervisor/queue_status.json", {
            "format": "group_event_state_v0_3_8_core_closure_queue_v1",
            "status": "COMPLETE" if complete else "RUNNING",
            "stages": {name: row.get("status", "PENDING") for name, row in status.items()},
            "launched_pids": launched,
            "long_subjects": list(LONG), "medium_subjects": list(MEDIUM),
            "h2a_long_subjects": list(LONG_H2A), "seeds": list(SEEDS),
            "state_selection_used_seizure_outcomes": False,
            "global_interictal_feature_freeze_before_any_outcome": global_marker.exists(),
            "development_targets_read": False, "sealed_partition_opened": False,
        })
        if complete:
            return
        time.sleep(30.0)


if __name__ == "__main__":
    main()
