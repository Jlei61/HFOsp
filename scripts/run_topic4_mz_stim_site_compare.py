#!/usr/bin/env python3
"""Run the frozen MZ z_i+m_i candidate under endpoint versus middle suppression.

This is a model-side visual diagnostic for the E1146 registered shared-axis
substrate.  It intentionally does not use the retired q_I/J_K spatial-field
path.  Both arms use the frozen MZ V2 candidate and the same noise seed; the
only difference is which four ICL contacts define the E-cell threshold clamp.

The simulation is expensive and therefore gated by ``--confirm-run``.  The
output is an analysis artifact consumed by
``scripts/paper_figures/plot_fig_mz_stim_site_near_runaway.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_m4_dynamic_qi as M4  # noqa: E402
import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_mz_slowvars as MZR  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_onset_dynamics import MZOnsetProbe  # noqa: E402


DT_MS = 0.1
FROZEN_BASELINE_T120_MS = 12956.2
FROZEN_CANDIDATE = MZSlowVarsConfig(
    use_z=True,
    use_m=True,
    I_th_EI=95.19851312666987,
    tau_z=5000.0,
    tau_adp=500.0,
    eta_m=0.007451594355587098,
)
FROZEN_CANDIDATE_LABEL = "zA_q75_tz5000__mA0p001_tau500"
EXPECTED_MONTAGE = [
    "SCL6", "SCL7", "SCL8", "SCL9",
    "ICL1", "ICL2", "ICL3", "ICL4", "ICL5", "ICL6", "ICL7", "ICL8",
    "ICL9", "ICL10", "ICL11",
]
GUARDED_ENGINE = (
    "kick_probe.py", "params.py", "model.py", "connectivity.py", "connectivity_rot.py", "lfp.py",
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _git_sha() -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip()


def _engine_shas() -> dict[str, str]:
    engine = ROOT / "src" / "snn_engine"
    return {
        name: hashlib.sha256((engine / name).read_bytes()).hexdigest()[:12]
        for name in GUARDED_ENGINE
    }


def select_icl_contacts(
    names: list[str],
    contacts: np.ndarray,
    anchor: np.ndarray,
    n_contacts: int = 4,
) -> np.ndarray:
    """Return the nearest ICL contacts to ``anchor`` in physical sheet space."""
    pool = [index for index, name in enumerate(names) if str(name).upper().startswith("ICL")]
    if len(pool) < n_contacts:
        raise ValueError(f"need {n_contacts} ICL contacts, found {len(pool)}")
    ranked = sorted(pool, key=lambda index: float(np.linalg.norm(contacts[index] - anchor)))
    return np.asarray(sorted(ranked[:n_contacts]), dtype=int)


def electrode_target_mask(
    pos_e: np.ndarray,
    stim_contacts: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """E-indexed target mask: cells within ``radius_mm`` of any selected contact."""
    distances = np.linalg.norm(
        np.asarray(pos_e, float)[:, None, :] - np.asarray(stim_contacts, float)[None, :, :],
        axis=2,
    )
    return np.any(distances <= float(radius_mm), axis=1)


def choose_capture_snapshot(
    snapshots: dict[str, dict],
    t_run_ms: float,
    lead_ms: float,
) -> tuple[str, dict, float]:
    """Choose the latest registered snapshot not later than ``t_run-lead``."""
    target = float(t_run_ms) - float(lead_ms)
    choices = []
    for label, payload in snapshots.items():
        time_ms = float(payload["step"]) * DT_MS
        if time_ms <= target + 1e-9:
            choices.append((time_ms, str(label), payload))
    if not choices:
        raise RuntimeError(f"no snapshot at or before t_run-lead={target:.1f} ms")
    time_ms, label, payload = max(choices, key=lambda item: item[0])
    return label, payload, time_ms


def _snapshot_schedule(start_ms: float, stop_ms: float, spacing_ms: float) -> dict[int, str]:
    times = np.arange(float(start_ms), float(stop_ms) + 0.5 * spacing_ms, float(spacing_ms))
    return {
        int(round(time_ms / DT_MS)): f"t{time_ms:.1f}ms"
        for time_ms in times
    }


def _window_spike_payload(spikes: np.ndarray, capture_ms: float, window_ms: float) -> tuple[np.ndarray, np.ndarray]:
    hi = int(round(float(capture_ms) / DT_MS)) + 1
    lo = max(0, hi - int(round(float(window_ms) / DT_MS)))
    window = np.asarray(spikes[lo:hi], bool)
    counts = window.sum(axis=0).astype(np.uint16)
    first = np.full(window.shape[1], np.nan, dtype=np.float32)
    active = window.any(axis=0)
    if np.any(active):
        first[active] = np.argmax(window[:, active], axis=0).astype(np.float32) * DT_MS
    return counts, first


def _run_arm(
    substrate: dict,
    *,
    arm: str,
    stim_indices: np.ndarray,
    stim_on_ms: float,
    stim_off_ms: float,
    stim_radius_mm: float,
    clamp_delta_mv: float,
    t_ms: float,
    snapshot_spacing_ms: float,
    capture_lead_ms: float,
    activity_window_ms: float,
    runaway_stop_dur_ms: float,
) -> tuple[dict, dict]:
    p = dataclasses.replace(substrate["p"], T=float(t_ms))
    montage = substrate["reg"]["montage_sheet"]
    contacts = np.asarray(montage.contacts, float)
    names = [str(name) for name in montage.names]
    stim_contacts = contacts[np.asarray(stim_indices, int)]
    target_e = electrode_target_mask(substrate["posE"], stim_contacts, stim_radius_mm)
    if int(target_e.sum()) == 0:
        raise RuntimeError(f"{arm}: stimulation target has no E cells")

    snapshots = _snapshot_schedule(stim_on_ms, t_ms, snapshot_spacing_ms)
    slow = MZOnsetProbe(
        substrate["N"],
        18.0,
        FROZEN_CANDIDATE,
        NE=substrate["NE"],
        core_mask_E=MZR.build_core_masks(substrate),
        snapshot_steps=snapshots,
    )
    slow.set_suppression(
        lo=int(round(stim_on_ms / DT_MS)),
        hi=int(round(stim_off_ms / DT_MS)),
        target_E=target_e,
        delta=float(clamp_delta_mv),
    )
    recorder = LFPRecorder(
        p,
        substrate["net"]["pos"],
        substrate["net"]["labels"],
        sites=contacts,
    )
    substrate["net"]["rng"] = np.random.default_rng(substrate["seed"])
    started = time.time()
    result = simulate_kick(
        p,
        substrate["net"],
        0.0,
        slow=slow,
        kick_center=list(substrate["src_xy"]),
        r_kick=PP.R_KICK,
        t_kick=1e9,
        V_th_per_neuron=substrate["vth"],
        # Keep a real post-threshold segment while avoiding a prohibitively long
        # integration in the sustained high-firing regime.  The renderer uses a
        # broken post-stimulation axis and must never pad the truncated trace.
        early_stop_runaway=True,
        es_dur_ms=float(runaway_stop_dur_ms),
        lfp_recorder=recorder,
        verbose=True,
    )
    wall_s = time.time() - started

    smoothed = M4._smooth(np.asarray(result["rate_E"], float), DT_MS)
    t_run = M4._first_sustained(smoothed, DT_MS)
    if t_run is None:
        raise RuntimeError(f"{arm}: operational runaway not reached by {t_ms:.1f} ms")
    snap_label, snapshot, capture_ms = choose_capture_snapshot(
        slow.snapshots, float(t_run), capture_lead_ms
    )
    spike_count, first_spike = _window_spike_payload(
        result["E_spk_bool"], capture_ms, activity_window_ms
    )

    arrays = {
        "times": np.asarray(result["times"], np.float32),
        "lfp_trace": np.asarray(result["lfp_trace"], np.float32),
        "rate_e_hz": np.asarray(result["rate_E"], np.float32),
        "rate_e_smooth_hz": np.asarray(smoothed, np.float32),
        "z_mean": np.asarray(slow.trace_z_mean, np.float32),
        "z_core_mean": np.asarray(slow.trace_z_core_mean, np.float32),
        "adaptation_current_mean": np.asarray(slow.trace_adap_current, np.float32),
        "z_snapshot_e": np.asarray(snapshot["z_E"], np.float32),
        "m_snapshot_e": np.asarray(snapshot["m_E"], np.float32),
        "spike_count_window": spike_count,
        "first_spike_rel_ms": first_spike,
        "pos_e": np.asarray(substrate["posE"], np.float32),
        "contacts": contacts.astype(np.float32),
        "contact_names": np.asarray(names, dtype=object),
        "axis_unit": np.asarray(substrate["axis_unit"], np.float32),
        "center": np.asarray(substrate["center"], np.float32),
        "source_xy": np.asarray(substrate["src_xy"], np.float32),
        "sink_xy": np.asarray(substrate["snk_xy"], np.float32),
        "stim_contact_indices": np.asarray(stim_indices, np.int16),
        "stim_target_e": np.asarray(target_e, bool),
    }
    metadata = {
        "arm": arm,
        "seed": int(substrate["seed"]),
        "candidate": FROZEN_CANDIDATE_LABEL,
        "candidate_cfg": dataclasses.asdict(FROZEN_CANDIDATE),
        "stim_contact_names": [names[index] for index in stim_indices],
        "stim_contact_indices": [int(index) for index in stim_indices],
        "n_stim_target_e": int(target_e.sum()),
        "stim_radius_mm": float(stim_radius_mm),
        "clamp_delta_mv": float(clamp_delta_mv),
        "stim_on_ms": float(stim_on_ms),
        "stim_off_ms": float(stim_off_ms),
        "t_run_ms": float(t_run),
        "baseline_t_run_ms": FROZEN_BASELINE_T120_MS,
        "delay_vs_frozen_baseline_ms": float(t_run - FROZEN_BASELINE_T120_MS),
        "early_stop_ms": result.get("runaway_early_stop_ms"),
        "runaway_stop_dur_ms": float(runaway_stop_dur_ms),
        "simulation_stop_ms": float(np.asarray(result["times"], float)[-1]),
        "post_runaway_recorded_ms": float(
            np.asarray(result["times"], float)[-1] - float(t_run)
        ),
        "capture_snapshot_label": snap_label,
        "capture_time_ms": float(capture_ms),
        "capture_lead_realized_ms": float(t_run - capture_ms),
        "activity_window_ms": float(activity_window_ms),
        "snapshot_spacing_ms": float(snapshot_spacing_ms),
        "n_steps": int(len(result["times"])),
        "wall_s": float(wall_s),
    }
    return arrays, metadata


def _save_arm(output_dir: Path, arm: str, arrays: dict, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{arm}.npz", **arrays)
    (output_dir / f"{arm}.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--T", type=float, default=20000.0)
    parser.add_argument("--stim-on", type=float, default=8000.0)
    parser.add_argument("--stim-off", type=float, default=14000.0)
    parser.add_argument("--stim-radius", type=float, default=2.0)
    parser.add_argument("--n-stim-contacts", type=int, default=4)
    parser.add_argument("--clamp-delta-mv", type=float, default=82.0)
    parser.add_argument("--snapshot-spacing-ms", type=float, default=20.0)
    parser.add_argument("--capture-lead-ms", type=float, default=20.0)
    parser.add_argument("--activity-window-ms", type=float, default=20.0)
    parser.add_argument(
        "--arm",
        choices=("both", "endpoint", "middle"),
        default="both",
        help="rerun both arms or refresh one arm while reusing the other saved artifact",
    )
    parser.add_argument("--runaway-stop-dur-ms", type=float, default=100.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/topic4_sef_hfo/mz_stim_site_compare",
    )
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("Simulation is gated: rerun with --confirm-run")
    if not (0.0 <= args.stim_on < args.stim_off <= args.T):
        raise ValueError("require 0 <= stim_on < stim_off <= T")

    substrate = PP.build_substrate(args.seed)
    montage = substrate["reg"]["montage_sheet"]
    names = [str(name) for name in montage.names]
    contacts = np.asarray(montage.contacts, float)
    if names != EXPECTED_MONTAGE:
        raise ValueError(f"E1146 montage drift: {names}")

    endpoint_indices = select_icl_contacts(
        names,
        contacts,
        np.asarray(substrate["src_xy"], float),
        args.n_stim_contacts,
    )
    middle_indices = select_icl_contacts(
        names,
        contacts,
        np.asarray(substrate["center"], float),
        args.n_stim_contacts,
    )
    if np.array_equal(endpoint_indices, middle_indices):
        raise RuntimeError("endpoint and middle policies selected identical contacts")

    all_specs = {"endpoint": endpoint_indices, "middle": middle_indices}
    selected_arms = list(all_specs) if args.arm == "both" else [args.arm]
    arm_metadata = {}
    arm_arrays = {}
    for arm in all_specs:
        if arm in selected_arms:
            continue
        json_path = args.output_dir / f"{arm}.json"
        npz_path = args.output_dir / f"{arm}.npz"
        if not (json_path.exists() and npz_path.exists()):
            raise FileNotFoundError(
                f"--arm {args.arm} requires reusable {arm} artifacts in {args.output_dir}"
            )
        arm_metadata[arm] = json.loads(json_path.read_text(encoding="utf-8"))
        arm_arrays[arm] = np.load(npz_path, allow_pickle=True)

    for arm in selected_arms:
        indices = all_specs[arm]
        print(f"[{arm}] contacts={[names[index] for index in indices]}", flush=True)
        arrays, metadata = _run_arm(
            substrate,
            arm=arm,
            stim_indices=indices,
            stim_on_ms=args.stim_on,
            stim_off_ms=args.stim_off,
            stim_radius_mm=args.stim_radius,
            clamp_delta_mv=args.clamp_delta_mv,
            t_ms=args.T,
            snapshot_spacing_ms=args.snapshot_spacing_ms,
            capture_lead_ms=args.capture_lead_ms,
            activity_window_ms=args.activity_window_ms,
            runaway_stop_dur_ms=args.runaway_stop_dur_ms,
        )
        arm_arrays[arm] = arrays
        arm_metadata[arm] = metadata
        _save_arm(args.output_dir, arm, arrays, metadata)

    pre = int(round(args.stim_on / DT_MS))
    parity_rate = bool(np.array_equal(
        arm_arrays["endpoint"]["rate_e_hz"][:pre],
        arm_arrays["middle"]["rate_e_hz"][:pre],
    ))
    parity_lfp = bool(np.array_equal(
        arm_arrays["endpoint"]["lfp_trace"][:pre],
        arm_arrays["middle"]["lfp_trace"][:pre],
    ))
    if not (parity_rate and parity_lfp):
        raise RuntimeError(f"pre-stim parity failed: rate={parity_rate}, lfp={parity_lfp}")

    endpoint_post_off = float(arm_metadata["endpoint"]["t_run_ms"] - args.stim_off)
    middle_post_off = float(arm_metadata["middle"]["t_run_ms"] - args.stim_off)
    middle_advantage = float(
        arm_metadata["middle"]["t_run_ms"] - arm_metadata["endpoint"]["t_run_ms"]
    )

    summary = {
        "schema_id": "topic4_mz_stim_site_compare_v1",
        "status": "visual diagnostic; model-only external E-threshold clamp",
        "scientific_model": "frozen per-neuron postsynaptic inhibitory efficacy z_i plus adaptation m_i",
        "forbidden_model_paths": ["q_I spatial field", "J_K/g_K spatial field"],
        "candidate": FROZEN_CANDIDATE_LABEL,
        "seed": int(args.seed),
        "pre_stim_parity": {"rate_e_bit_identical": parity_rate, "lfp_bit_identical": parity_lfp},
        "montage_contract": {
            "source": "run_m4_phaseplane.build_substrate(seed)::reg.montage_sheet",
            "names": names,
            "layout": "E1146 registered plane; rendered in centered TA shared-axis coordinates",
        },
        "stimulation_contract": {
            "endpoint_policy": "four nearest ICL contacts to the frozen source core",
            "middle_policy": "four nearest ICL contacts to the registered sheet center",
            "mechanism": "raise E-cell firing threshold within radius during a finite window",
            "claim_boundary": "external model intervention; not a biophysical or clinical stimulation protocol",
        },
        "comparison": {
            "status": "single-seed descriptive comparison",
            "endpoint_runaway_after_stim_off_ms": endpoint_post_off,
            "middle_runaway_after_stim_off_ms": middle_post_off,
            "middle_minus_endpoint_runaway_ms": middle_advantage,
            "interpretation_boundary": (
                "runaway timing supports a site effect; timing alone does not establish "
                "that altered propagation order is the mediating mechanism"
            ),
        },
        "arms": arm_metadata,
        "provenance": {
            "git_sha": _git_sha(),
            "engine_shas": _engine_shas(),
            "producer": "scripts/run_topic4_mz_stim_site_compare.py",
            "argv": sys.argv,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary["arms"]), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
