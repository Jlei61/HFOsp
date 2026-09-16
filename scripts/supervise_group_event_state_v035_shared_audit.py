#!/usr/bin/env python3
"""Independent watcher that audits the shared-state run after finalization."""

from __future__ import annotations

from collections import Counter
import glob
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/data/hfosp_group_event_state_v0_3_5_shared")
STATUS = OUT / "supervisor/audit_watcher_status.json"
PRIMARY = OUT / "supervisor/queue_status.json"
POST = OUT / "supervisor/post_status.json"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, STATUS)


def snapshot(status: str) -> dict:
    primary, post = read(PRIMARY), read(POST)
    return {
        "format": "group_event_state_v0_3_5_shared_audit_watcher_v1",
        "status": status,
        "updated_epoch": time.time(),
        "primary_status": primary.get("status"),
        "primary_unit_counts": dict(Counter(primary.get("units", {}).values())),
        "primary_stages": primary.get("stages", {}),
        "post_status": post.get("status"),
        "post_unit_counts": dict(Counter(post.get("units", {}).values())),
        "producer_cards": len(glob.glob(str(OUT / "shared_producer/**/card.json"), recursive=True)),
        "evaluator_cards": len(glob.glob(str(OUT / "frozen_evaluator/**/card.json"), recursive=True)),
        "same_prefix_cards": len(glob.glob(str(OUT / "same_prefix/**/card.json"), recursive=True)),
        "h2b_bindings": len(glob.glob(str(OUT / "shared_h2b/**/shared_state_binding.json"), recursive=True)),
        "cross_evaluator_cards": len(glob.glob(str(OUT / "cross_evaluator/**/card.json"), recursive=True)),
        "development_targets_read": False,
        "sealed_partition_opened": False,
    }


def main() -> None:
    while True:
        post = read(POST)
        state = snapshot("WAITING_FOR_FINALIZER")
        atomic(state)
        if post.get("status") in {"COMPLETE", "FAILED_FINALIZER"}:
            break
        time.sleep(60)
    log = OUT / "logs/shared_machine_audit.log"
    with log.open("a", encoding="utf-8") as handle:
        rc = subprocess.run(
            [PYTHON, "scripts/audit_group_event_state_v035_shared.py"],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
    state = snapshot("COMPLETE" if rc == 0 else "FAILED_AUDIT")
    state["audit_returncode"] = rc
    atomic(state)


if __name__ == "__main__":
    main()
