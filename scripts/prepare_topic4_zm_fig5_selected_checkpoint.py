#!/usr/bin/env python3
"""Materialize the exact replay checkpoint selected for Figure 5 C/D."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.prepare_topic4_zm_fig5_state_contrast import (  # noqa: E402
    _apply_workpoint,
    _continuation_slice,
)
from scripts.run_topic4_zm_perturbation_worker import _continue  # noqa: E402
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_ictal_transition import build_substrate, load_round_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--mode-snapshot", required=True)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    replay_path = Path(args.replay).resolve()
    replay_meta = json.loads(replay_path.with_suffix(".json").read_text())
    with np.load(replay_path, allow_pickle=False) as handle:
        replay_rate = np.asarray(handle["rate_E_hz"], np.float32)
    snapshot_path = Path(args.mode_snapshot).resolve()
    snapshot_meta = json.loads(snapshot_path.with_suffix(".json").read_text())
    with np.load(snapshot_path, allow_pickle=False) as handle:
        target_ms = float(handle["selected_time_ms"])
    if int(snapshot_meta["seed"]) != int(replay_meta["seed"]):
        raise RuntimeError("mode snapshot and replay use different seeds")

    parameters = replay_meta["workpoint_parameters"]
    config = _apply_workpoint(load_round_config(args.config), parameters)
    seed = int(replay_meta["seed"])
    substrate = build_substrate(
        config, replay_meta["candidate_id"], seed,
        cache_dir=str(ROOT / config["output_root"] / "network_cache"),
        ee_dose=float(parameters["E_to_E_dose"]),
        etoi_dose=float(parameters["E_to_I_dose"]),
    )
    dt_ms = float(substrate.engine["dt"])
    start_state = ckpt.load(args.start_checkpoint)
    start_ms = float(start_state["absolute_time_ms"])
    if target_ms < start_ms - 1e-9:
        raise RuntimeError("selected snapshot precedes the supplied checkpoint")
    target_step = int(round(target_ms / dt_ms))
    duration_ms = target_ms - start_ms + dt_ms
    captured: dict[int, dict] = {}
    continuation, _ = _continue(
        substrate, config, start_state, duration_ms=duration_ms,
        checkpoint_steps=[target_step],
        checkpoint_sink=lambda step, state: captured.setdefault(step, state),
    )
    if target_step not in captured:
        raise RuntimeError("continuation missed the selected snapshot checkpoint")
    replay_slice = _continuation_slice(start_ms, duration_ms, dt_ms)
    exact = np.array_equal(
        np.asarray(continuation["rate_E"], np.float32), replay_rate[replay_slice])
    if not exact:
        raise RuntimeError("selected checkpoint continuation diverged from replay")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sha256 = ckpt.save(captured[target_step], out)
    summary = {
        "status": "ZM_FIG5_SELECTED_CHECKPOINT_COMPLETE",
        "seed": seed,
        "selected_time_ms": target_ms,
        "start_time_ms": start_ms,
        "continuation_rate_exact": True,
        "checkpoint_sha256": sha256,
        "mode_snapshot": str(snapshot_path),
        "checkpoint": str(out),
    }
    atomic_write_json(summary, str(out.with_suffix(".json")))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
