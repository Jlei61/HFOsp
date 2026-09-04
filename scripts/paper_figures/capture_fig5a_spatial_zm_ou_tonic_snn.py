#!/usr/bin/env python3
"""Replay the locked Fig5A seed and capture read-only SNN movie frames.

The scientific runner intentionally does not archive the full 21,972 x 32,000
E-spike boolean matrix.  This wrapper leaves that runner and its RNG order
untouched, but observes its returned spike matrix and the already-updated slow
state.  It stores only 10-ms active-neuron masks every 20 ms plus q/M spatial
frames.  The replay is accepted only when the locked float32 rate, LFP, OU and
slow traces are bit-identical to the archived seed-1842 trajectory.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "src" / "snn_engine"
for path in (ROOT, ENGINE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

ARCHIVED_NPZ = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_node_local_connectivity_plus_zm/spatial_zm_ou/"
    "tonic_confirmation_v2/tonic_b0_v2_s1842.npz"
)
ARCHIVED_SHA256 = "283ee32711a2c5388f065d9e2faa9a54390f788bb3c7496c7e8cd4ea993a7248"
CAPTURE_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_node_local_connectivity_plus_zm/spatial_zm_ou/"
    "snn_gif_capture"
)
REPLAY_BASE = CAPTURE_ROOT / "tonic_b0_v2_s1842_replay"
CAPTURE_PATH = CAPTURE_ROOT / "tonic_b0_v2_s1842_snn_frames.npz"
CAPTURE_META = CAPTURE_ROOT / "tonic_b0_v2_s1842_snn_frames_metadata.json"
FRAME_DT_MS = 20.0
ACTIVITY_WINDOW_MS = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if sha256(ARCHIVED_NPZ) != ARCHIVED_SHA256:
        raise RuntimeError("locked seed-1842 NPZ hash changed")

    import kick_probe
    import scripts.run_topic4_spatial_zm_ou_transition as runner
    import src.topic4_spatial_zm_qigk as slow_module

    captured_result = {}
    slow_frames = {"time_ms": [], "q_grid": [], "m_grid": []}
    slow_ref = {"value": None}
    original_simulate = kick_probe.simulate_kick
    original_step = slow_module.SpatialZMQIGKSlowVars.step

    def observed_step(self, spk, labels, dt):
        original_step(self, spk, labels, dt)
        slow_ref["value"] = self
        stride = max(1, int(round(FRAME_DT_MS / float(dt))))
        if self._step_index % stride != 0:
            return
        n_grid = int(self.cfg.n_grid)
        slow_frames["time_ms"].append(float(self._step_index * float(dt)))
        slow_frames["q_grid"].append(np.asarray(self.q_I, np.float32).copy())
        slow_frames["m_grid"].append(np.asarray(
            slow_module._grid_mean(
                self.m[:self.nE], self._ixE, self._iyE, n_grid
            ),
            np.float32,
        ))

    def observed_simulate(*args, **kwargs):
        result = original_simulate(*args, **kwargs)
        captured_result["value"] = result
        return result

    slow_module.SpatialZMQIGKSlowVars.step = observed_step
    kick_probe.simulate_kick = observed_simulate
    runner.SpatialZMQIGKSlowVars.step = observed_step

    REPLAY_BASE.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "run_topic4_spatial_zm_ou_transition.py",
        "--config", "config/topic4_data_driven_zm_ictal_transition_v1.json",
        "--seed", "1842",
        "--run-role", "confirmation",
        "--parameter-set-id", "tonic_b0_v2_snn_replay",
        "--mode", "hybrid",
        "--duration-ms", "9000",
        "--post-onset-ms", "1600",
        "--k-q", "0.001",
        "--q-a50", "0.004",
        "--q-hill-n", "8",
        "--q-min", "0.775",
        "--tau-m", "12.5",
        "--eta-m", "0.02",
        "--m-spatial-mix", "0.0",
        "--out", str(REPLAY_BASE),
    ]
    runner.main()

    result = captured_result.get("value")
    slow = slow_ref.get("value")
    if result is None or slow is None:
        raise RuntimeError("read-only capture hooks did not observe the simulation")

    replay_path = REPLAY_BASE.with_suffix(".npz")
    comparison_keys = [
        "time_ms", "rate_E_hz", "rate_I_hz", "lfp_trace",
        "ou_time_ms", "ou_spatial_mean_rate_per_ms",
        "ou_spatial_sd_rate_per_ms", "slow_time_ms", "slow_q_mean",
        "slow_q_core_mean", "slow_q_surround_mean", "slow_m_mean",
        "slow_adaptation_current_mean", "slow_spike_count_E",
    ]
    comparisons = {}
    with np.load(ARCHIVED_NPZ, allow_pickle=False) as archived, np.load(
        replay_path, allow_pickle=False
    ) as replay:
        for key in comparison_keys:
            same_shape = archived[key].shape == replay[key].shape
            exact = bool(same_shape and np.array_equal(archived[key], replay[key]))
            max_abs = (
                float(np.max(np.abs(
                    archived[key].astype(np.float64) - replay[key].astype(np.float64)
                ))) if same_shape and archived[key].size else 0.0
            )
            comparisons[key] = {
                "same_shape": bool(same_shape),
                "bit_identical": exact,
                "max_abs_difference": max_abs,
            }
    if not all(row["bit_identical"] for row in comparisons.values()):
        atomic_json(CAPTURE_META, {
            "status": "REPLAY_MISMATCH_REJECTED",
            "comparisons": comparisons,
        })
        raise RuntimeError("seed-1842 replay diverged from locked trajectory")

    spikes = np.asarray(result["E_spk_bool"], bool)
    dt_ms = 0.1
    duration_ms = float(spikes.shape[0] * dt_ms)
    frame_time_ms = [0.0] + list(slow_frames["time_ms"])
    q_frames = [np.asarray(slow.q_init_grid, np.float32)] + slow_frames["q_grid"]
    zero_m = np.zeros_like(slow_frames["m_grid"][0], dtype=np.float32)
    m_frames = [zero_m] + slow_frames["m_grid"]
    if frame_time_ms[-1] < duration_ms - 1e-9:
        frame_time_ms.append(duration_ms)
        q_frames.append(np.asarray(slow.q_I, np.float32).copy())
        m_frames.append(np.asarray(slow_module._grid_mean(
            slow.m[:slow.nE], slow._ixE, slow._iyE, int(slow.cfg.n_grid)
        ), np.float32))

    active_masks = []
    half_window_steps = max(1, int(round(ACTIVITY_WINDOW_MS / dt_ms)))
    for time_ms in frame_time_ms:
        hi = min(spikes.shape[0], max(1, int(round(time_ms / dt_ms))))
        lo = max(0, hi - half_window_steps)
        active_masks.append(np.any(spikes[lo:hi], axis=0))
    active_masks = np.asarray(active_masks, bool)

    atomic_npz(
        CAPTURE_PATH,
        frame_time_ms=np.asarray(frame_time_ms, np.float32),
        active_E_packbits=np.packbits(active_masks, axis=1),
        n_E=np.asarray(spikes.shape[1], np.int32),
        activity_window_ms=np.asarray(ACTIVITY_WINDOW_MS, np.float32),
        q_grid=np.asarray(q_frames, np.float32),
        m_grid=np.asarray(m_frames, np.float32),
    )
    replay_json = json.loads(REPLAY_BASE.with_suffix(".json").read_text())
    atomic_json(CAPTURE_META, {
        "status": "LOCKED_TRAJECTORY_REPLAY_BIT_IDENTICAL",
        "source_commit": "1292e064d5092ca72a9711a8c3dfc37b78558c1b",
        "seed": 1842,
        "scientific_onset_ms": replay_json["scientific_onset_ms"],
        "trajectory_duration_ms": replay_json["trajectory_duration_ms"],
        "archived_npz": str(ARCHIVED_NPZ),
        "archived_npz_sha256": ARCHIVED_SHA256,
        "replay_npz": str(replay_path),
        "replay_npz_sha256": sha256(replay_path),
        "capture_npz": str(CAPTURE_PATH),
        "capture_npz_sha256": sha256(CAPTURE_PATH),
        "frame_dt_ms": FRAME_DT_MS,
        "activity_window_ms": ACTIVITY_WINDOW_MS,
        "n_frames": len(frame_time_ms),
        "read_only_hooks": [
            "observe returned E_spk_bool after simulate_kick",
            "copy q grid and spatially binned M after slow.step",
            "no RNG draw, current, threshold, edge, delay or state update changed",
        ],
        "comparisons": comparisons,
    })
    print(json.dumps({
        "status": "LOCKED_TRAJECTORY_REPLAY_BIT_IDENTICAL",
        "capture": str(CAPTURE_PATH),
        "metadata": str(CAPTURE_META),
        "frames": len(frame_time_ms),
        "capture_sha256": sha256(CAPTURE_PATH),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
