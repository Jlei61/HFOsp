#!/usr/bin/env python3
"""FCXR-LC5v2 finite-horizon calibration and sequential U2 exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle,
    SparseSpikeStream,
    json_sanitize,
    load_sparse_spike_stream,
    replay_sparse_loads,
)
from src.topic4_fcxr_lc5_finite_episode import (  # noqa: E402
    ACTIVATION_SAMPLE_MS,
    CALIBRATION_DT_MS,
    array_sha256,
    calibrate_episode_dose,
    coarsen_sparse_stream,
    estimate_shrunken_p0,
    replay_finite_load,
    solve_a_for_window_target,
)


OLD = ROOT / "results/topic4_sef_hfo/fcxr_lc5_episode_pump"
SOURCE = OLD / "u1_capture"
OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc5v2_finite_episode"
CAL = OUT / "finite_calibration"
SPEC = ROOT / "docs/superpowers/specs/2026-08-11-topic4-fcxr-lc5v2-finite-horizon-episode-load-design.md"
PLAN = ROOT / "docs/superpowers/plans/2026-08-11-topic4-fcxr-lc5v2-finite-horizon-episode-load.md"
DT_SOURCE_MS = 0.05
TAUS_MS = (3000.0, 8000.0, 15000.0)
GAMMAS = (0.10, 0.25, 0.40)
ONSET_MS = 11000.0
BASELINE_MS = (7000.0, 11000.0)
EARLY_MS = (12000.0, 14000.0)
REPLAY_STOP_MS = 15000.0


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(json_sanitize(value), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def stage_audit():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "lineage_lock.json"
    if path.is_file():
        return json.loads(path.read_text())
    required_status = json.loads((OLD / "u1_prefix_validation.json").read_text())
    carrier = json.loads((OLD / "u1_carrier_adjudication.json").read_text())
    if required_status["status"] != "CAPTURE_DOES_NOT_CONTAMINATE_TRAJECTORY":
        raise SystemExit("LC5v2 requires a prefix-clean U1 source")
    if carrier["source_type"] != "escalating_saturated_source":
        raise SystemExit("LC5v2 is registered against the escalating U1 source")
    sources = [
        SPEC, PLAN, SOURCE / "u1_capture_summary.json", SOURCE / "u1_sparse_spikes.npz",
        SOURCE / "u1_rate_fields.npz", SOURCE / "states/onset.npz",
        ROOT / "src/topic4_fcxr_lc5_finite_episode.py", Path(__file__),
    ]
    payload = {
        "status": "LC5V2_LINEAGE_LOCKED",
        "git_head": _git_head(),
        "source_hashes": {str(p): _sha(p) for p in sources},
        "source_type": carrier["source_type"],
        "prefix_status": required_status["status"],
        "old_scale_stop_is_provenance_not_v2_gate": str(OLD / "U_SCALE_NOT_IDENTIFIABLE.json"),
        "calibration": {
            "dt_ms": CALIBRATION_DT_MS,
            "activation_sample_ms": ACTIVATION_SAMPLE_MS,
            "baseline_ms": list(BASELINE_MS),
            "early_episode_ms": list(EARLY_MS),
            "target_activation": 0.5,
            "tau_ms": list(TAUS_MS),
            "gammas": list(GAMMAS),
        },
    }
    _write_json(path, payload)
    return payload


def _assert_lock_current():
    lock = stage_audit()
    for name, want in lock["source_hashes"].items():
        got = _sha(name)
        if got != want:
            raise SystemExit(f"LC5v2 source drift: {name}: {got} != {want}")
    return lock


def _step(ms, dt=CALIBRATION_DT_MS, *, last=False):
    x = int(round(float(ms) / float(dt)))
    return x - 1 if last else x


def stage_calibrate():
    if CAL.is_dir():
        return json.loads((CAL / "finite_episode_calibration.json").read_text())
    lock = _assert_lock_current()
    summary = json.loads((SOURCE / "u1_capture_summary.json").read_text())
    if float(summary["onset_ms"]) != ONSET_MS:
        raise SystemExit("source onset drifted from the locked LC5v2 windows")

    full = load_sparse_spike_stream(SOURCE / "u1_sparse_spikes.npz")
    fine_stop = int(round(1000.0 / DT_SOURCE_MS))
    fine_right = int(np.searchsorted(full.steps, fine_stop, side="left"))
    fine_1s = SparseSpikeStream(
        full.steps[:fine_right].copy(), full.cells[:fine_right].copy(), fine_stop, full.n_cells
    )
    coarse = coarsen_sparse_stream(
        full, source_dt_ms=DT_SOURCE_MS, target_dt_ms=CALIBRATION_DT_MS,
        stop_ms=REPLAY_STOP_MS,
    )
    coarse_1s = coarsen_sparse_stream(
        full, source_dt_ms=DT_SOURCE_MS, target_dt_ms=CALIBRATION_DT_MS,
        stop_ms=1000.0,
    )
    del full

    blocks = {
        "base0": (_step(7000), _step(8000)),
        "base1": (_step(8000), _step(9000)),
        "base2": (_step(9000), _step(10000)),
        "base3": (_step(10000), _step(11000)),
        "early": (_step(12000), _step(14000)),
    }
    snapshots = {
        _step(ONSET_MS, last=True): "onset",
        _step(ONSET_MS + 1000.0, last=True): "onset_plus_1s",
        _step(ONSET_MS + 4000.0, last=True): "onset_plus_4s",
    }
    fields = np.load(SOURCE / "u1_rate_fields.npz")
    baseline_rate = np.asarray(fields["baseline_rate_hz"], float)
    block_index = np.asarray(fields["recurrent_block_index"], int)
    force_blocks = np.asarray(fields["recurrent_effective_force_mean"], float)
    want_blocks = [int(EARLY_MS[0] // 1000), int(EARLY_MS[1] // 1000) - 1]
    rows = [int(np.where(block_index == b)[0][0]) for b in want_blocks]
    force_integral = force_blocks[rows].sum(axis=0) * 1000.0

    payload = {
        "status": "FINITE_EPISODE_CALIBRATION_COMPLETE",
        "lineage_git_head": lock["git_head"],
        "analysis_git_head": _git_head(),
        "source_spike_sha256": summary["spike_stream_sha256"],
        "source_config_sha256": summary["config_sha256"],
        "dt_source_ms": DT_SOURCE_MS,
        "dt_calibration_ms": CALIBRATION_DT_MS,
        "activation_sample_ms": ACTIVATION_SAMPLE_MS,
        "baseline_window_ms": list(BASELINE_MS),
        "early_episode_window_ms": list(EARLY_MS),
        "target_activation": 0.5,
        "tau": {},
    }
    arrays = {
        "baseline_rate_hz": baseline_rate.astype(np.float32),
        "recurrent_force_integral_ms": force_integral.astype(np.float32),
    }
    sample_every = int(round(ACTIVATION_SAMPLE_MS / CALIBRATION_DT_MS))

    for tau in TAUS_MS:
        solved = solve_a_for_window_target(
            coarse, dt_ms=CALIBRATION_DT_MS, tau_ms=tau,
            target_window=blocks["early"], target=0.5, sample_every_steps=sample_every,
        )
        a = float(solved["a_load"])
        first = replay_finite_load(
            coarse, dt_ms=CALIBRATION_DT_MS, tau_ms=tau, a_load=a, blocks=blocks,
            target_block="early", sample_every_steps=sample_every, snapshot_steps=snapshots,
        )
        phi_blocks = np.vstack([first.block_phi_mean[f"base{k}"] for k in range(4)])
        p0_fit = estimate_shrunken_p0(phi_blocks, baseline_rate)
        p0 = np.asarray(p0_fit.pop("p0"), float)
        second = replay_finite_load(
            coarse, dt_ms=CALIBRATION_DT_MS, tau_ms=tau, a_load=a,
            blocks={"early": blocks["early"]}, p0=p0, excess_block="early",
        )
        dose = calibrate_episode_dose(
            unit_excess_integral_ms=second.excess_integral_ms,
            recurrent_force_integral_ms=force_integral, gammas=GAMMAS,
        )

        exact_1s = replay_sparse_loads(
            fine_1s, candidates={"x": {"a_load": a, "tau_ms": tau, "h": 3}},
            dt_ms=DT_SOURCE_MS,
        )["x"]["u_final"]
        coarse_audit = replay_finite_load(
            coarse_1s, dt_ms=CALIBRATION_DT_MS, tau_ms=tau, a_load=a,
        ).u_final
        delta = np.abs(exact_1s - coarse_audit)
        key = f"tau{int(tau)}"
        payload["tau"][key] = {
            "tau_ms": tau,
            "a_load": a,
            "achieved_target": solved["achieved_target"],
            "bisection_iterations": solved["iterations"],
            "bisection_bracket": solved["bracket"],
            "p0": p0_fit,
            "dose": dose,
            "coarse_vs_exact_first_1s": {
                "max_abs_u": float(delta.max()),
                "q99_abs_u": float(np.quantile(delta, 0.99)),
                "mean_abs_u": float(delta.mean()),
                "exact_u_sha256": array_sha256(exact_1s),
                "coarse_u_sha256": array_sha256(coarse_audit),
            },
            "u_onset_sha256": array_sha256(first.snapshots["onset"]),
            "u_onset_plus_1s_sha256": array_sha256(first.snapshots["onset_plus_1s"]),
            "u_onset_plus_4s_sha256": array_sha256(first.snapshots["onset_plus_4s"]),
        }
        arrays[f"p0_{key}"] = p0.astype(np.float32)
        arrays[f"u_onset_{key}"] = first.snapshots["onset"].astype(np.float32)
        arrays[f"u_onset_plus_1s_{key}"] = first.snapshots["onset_plus_1s"].astype(np.float32)
        arrays[f"u_onset_plus_4s_{key}"] = first.snapshots["onset_plus_4s"].astype(np.float32)
        arrays[f"unit_excess_integral_ms_{key}"] = second.excess_integral_ms.astype(np.float32)
        arrays[f"baseline_phi_blocks_{key}"] = phi_blocks.astype(np.float32)

    primary = payload["tau"]["tau8000"]
    prelock = {
        "status": "LC5V2_CANDIDATE_PRELOCK",
        "tau_ms": 8000.0,
        "a_load": primary["a_load"],
        "h": 3,
        "p0_key": "p0_tau8000",
        "u_onset_key": "u_onset_tau8000",
        "Imax_by_gamma": primary["dose"]["Imax_by_gamma"],
        "ordered_u2a": [0.0, 0.25, 0.10, 0.40],
        "source_state": str(SOURCE / "states/onset.npz"),
        "calibration_arrays": str(CAL / "u_fields_tau3_8_15.npz"),
    }

    with AtomicStageBundle(CAL) as bundle:
        _write_json(bundle.path("finite_episode_calibration.json"), payload)
        _npz_atomic(bundle.path("u_fields_tau3_8_15.npz"), **arrays)
        _write_json(bundle.path("candidate_prelock.json"), prelock)
        bundle.commit(required=[
            "finite_episode_calibration.json", "u_fields_tau3_8_15.npz", "candidate_prelock.json",
        ])
    _write_json(OUT / "CALIBRATION_DONE.json", {
        "status": "DONE", "bundle": str(CAL), "candidate": prelock,
    })
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("audit", "calibrate"), required=True)
    args = ap.parse_args()
    out = stage_audit() if args.stage == "audit" else stage_calibrate()
    print(json.dumps(json_sanitize(out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
