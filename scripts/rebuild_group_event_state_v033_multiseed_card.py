#!/usr/bin/env python3
"""Rebuild completed v0.3.3 pilot cards with seed-first reduction.

The original search artifacts and cards are preserved.  Each corrected card is
written under ``card_multiseed_v2`` and consumes every final-seed checkpoint
selected by the already-frozen inner-validation search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.paths import AGENT_B_ROOT, payload_hash, release_status
from src.topic5_group_event_state.v033_training_lab.queue import EXIT_OK, Unit, execute_unit


def rebuild(request_id: str, *, device: str, agent_root: Path) -> Path:
    request_path = agent_root / "requests" / request_id / "request.json"
    state_path = agent_root / "search" / request_id / "driver_state.json"
    if not request_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"missing stored request or driver state for {request_id}")
    request = json.loads(request_path.read_text())
    state = json.loads(state_path.read_text())
    incumbent = state.get("incumbent") or {}
    seed_dirs = list(incumbent.get("seed_dirs") or [])
    if len(seed_dirs) < 2:
        raise ValueError(f"{request_id}: final incumbent has fewer than two completed seed directories")
    out_dir = agent_root / "search" / request_id / "card_multiseed_v2"
    search_summary = {
        "incumbent": {k: v for k, v in incumbent.items() if k != "recipe"},
        "stop_reason": state.get("stop_reason"),
        "n_batches": len(state.get("batches") or []),
    }
    params = {
        "request": request,
        "recipe": incumbent["recipe"],
        "seed": int(request.get("seed_policy", {}).get("base_seed", 0)),
        "seed_dirs": seed_dirs,
        "t0_path": state["t0_path"],
        "search_summary": search_summary,
        "scaling": incumbent["recipe"].get("scaling", "zscore"),
    }
    unit = Unit(
        unit_id=f"{request_id}:card_multiseed_v2",
        unit_type="card",
        request_id=request_id,
        job_key=payload_hash({"request_id": request_id, "kind": "card_multiseed_v2", "seed_dirs": seed_dirs}),
        params=params,
        out_dir=str(out_dir),
        workload_class="gpu_train_fixed_leaky" if device.startswith("cuda") else "cpu_train_fixed_leaky",
    )
    code = execute_unit(unit, device=device, release_present=bool(release_status()["present"]))
    if code != EXIT_OK:
        raise RuntimeError(f"{request_id}: corrected card failed with exit code {code}")
    result = json.loads((out_dir / "unit_result.json").read_text())
    return Path(result["card_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request_id", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--agent-root", type=Path, default=AGENT_B_ROOT)
    args = parser.parse_args()
    for request_id in args.request_id:
        path = rebuild(request_id, device=args.device, agent_root=args.agent_root)
        print(json.dumps({"request_id": request_id, "corrected_card": str(path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
