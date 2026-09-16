#!/usr/bin/env python3
"""Wait for formal dual H1, run a sensitivity smoke, then release dual H2a."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
H1_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/h1_dual_budget_complete")
OUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/h2a_frozen_decoder_dual")
SMOKE_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/smoke/h2a_frozen_decoder_dual")


def _h1_complete() -> bool:
    path = H1_ROOT / "supervisor/queue_status.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETE"
    except json.JSONDecodeError:
        return False


def main() -> None:
    supervisor = OUT_ROOT / "supervisor"; supervisor.mkdir(parents=True, exist_ok=True)
    while not _h1_complete():
        (supervisor / "release_status.json").write_text(
            json.dumps({"status": "WAITING_FOR_FORMAL_DUAL_H1", "released": False}, indent=2) + "\n",
            encoding="utf-8",
        )
        time.sleep(20)
    smoke_card = SMOKE_ROOT / "epilepsiae_1077/seed20260903/card.json"
    if not smoke_card.exists():
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"; env.setdefault("OMP_NUM_THREADS", "2")
        subprocess.run(
            [str(PYTHON), str(ROOT / "scripts/run_group_event_state_v037_h2a.py"),
             "--subject", "epilepsiae_1077", "--seed", "20260903", "--device", "cuda:0",
             "--max-epochs", "2", "--h1-root", str(H1_ROOT), "--out-root", str(SMOKE_ROOT),
             "--state-family", "dual"],
            cwd=ROOT, env=env, check=True,
        )
    card = json.loads(smoke_card.read_text(encoding="utf-8"))
    oracle = card.get("primary_contrasts", {}).get("oracle_sensitivity_grammar_gain")
    if oracle is None or not float(oracle) > 0.0:
        raise RuntimeError(f"dual H2a positive-control smoke failed: oracle gain={oracle}")
    (supervisor / "RELEASED").write_text(
        f"dual H1 complete; leaked-future oracle sensitivity={float(oracle):.8g}\n",
        encoding="utf-8",
    )
    (supervisor / "release_status.json").write_text(
        json.dumps({"status": "RELEASED_AFTER_SMOKE", "released": True,
                    "smoke_card": str(smoke_card), "oracle_gain": float(oracle)}, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [str(PYTHON), str(ROOT / "scripts/supervise_group_event_state_v037_h2a.py"),
         "--workers-per-gpu", "2", "--h1-root", str(H1_ROOT), "--out-root", str(OUT_ROOT),
         "--state-family", "dual"],
        cwd=ROOT, check=True,
    )


if __name__ == "__main__":
    main()

