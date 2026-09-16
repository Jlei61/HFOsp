#!/usr/bin/env python3
"""Wait for shared producers, run cross-transfer, audit, and finalize reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
OUT = Path("/data/hfosp_group_event_state_v0_3_5_shared")
PRIMARY = OUT / "supervisor/queue_status.json"
STATUS = OUT / "supervisor/post_status.json"
SUBJECTS = ("epilepsiae_253", "epilepsiae_1096", "epilepsiae_1125")
SEEDS = (20260903, 20260904, 20260905)


def atomic(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, STATUS)


def unit(subject: str, family: str, seed: int) -> Path:
    matches = list((OUT / "shared_producer" / subject / family).glob(f"*_state_seed{seed}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {subject}/{family}/{seed} producer, got {len(matches)}")
    return matches[0]


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + " ".join(cmd) + "\n"); handle.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def main() -> None:
    state = {"format": "group_event_state_v0_3_5_shared_post_supervisor_v1",
             "status": "WAITING_FOR_PRIMARY", "units": {},
             "development_targets_read": False, "sealed_partition_opened": False}
    atomic(state)
    while True:
        if PRIMARY.exists():
            primary = json.loads(PRIMARY.read_text())
            if primary.get("status") == "COMPLETE":
                break
        time.sleep(60)
    failed = [key for key, value in primary.get("units", {}).items() if "FAILED" in str(value)]
    state["primary_failed_units"] = failed
    state["status"] = "RUNNING_CROSS_TRANSFER"; atomic(state)
    for subject in SUBJECTS:
        for decoder_seed, seed in zip((0, 1, 2), SEEDS):
            sn, sg = unit(subject, "S_N", seed), unit(subject, "S_G", seed)
            specs = (
                ("sn_to_sg", sn, "S_G", None),
                ("sg_to_sn", sg, "S_N", None),
                ("combined_to_sn", sn, "S_N", sg),
                ("combined_to_sg", sn, "S_G", sg),
            )
            for name, source, target, extra in specs:
                out = OUT / "cross_evaluator" / subject / f"seed{seed}" / name
                key = f"{subject}::{seed}::{name}"
                if (out / "card.json").exists():
                    state["units"][key] = "COMPLETE"; atomic(state); continue
                cmd = [PYTHON, "scripts/run_group_event_state_v035_frozen_shared_evaluator.py",
                       "--producer-unit", str(source), "--decoder-seed", str(decoder_seed),
                       "--rate-root", str(OUT / "shared_rate"), "--out-dir", str(out),
                       "--target-family", target]
                if extra is not None:
                    cmd += ["--extra-producer-unit", str(extra)]
                rc = run(cmd, OUT / "logs" / f"cross_{subject}_{seed}_{name}.log")
                state["units"][key] = "COMPLETE" if rc == 0 and (out / "card.json").exists() else f"FAILED_RC_{rc}"
                atomic(state)
    state["status"] = "FINALIZING"; atomic(state)
    rc = run([PYTHON, "scripts/finalize_group_event_state_v035_shared.py"],
             OUT / "logs/shared_finalize.log")
    state["finalizer_returncode"] = rc
    state["status"] = "COMPLETE" if rc == 0 else "FAILED_FINALIZER"
    atomic(state)


if __name__ == "__main__":
    main()
