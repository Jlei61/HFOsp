#!/usr/bin/env python3
"""Small Joint-only Z/M calibration against the frozen runaway morphology."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.snn_engine.mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_runaway_morphology import (  # noqa: E402
    classify_sustained_runaway,
    contact_oscillation_metrics,
    population_rate_frequency_metrics,
    rolling_full_field_recruitment,
    summarize_runaway_morphology,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
    make_external_drive,
)
from src.topic4_zm_slow_vars import ZMTracedSlowVars  # noqa: E402


def _paired_windows(values, dt_ms, pre_window, post_window):
    values = np.asarray(values)
    time_ms = np.arange(len(values)) * float(dt_ms)
    pre = values[(time_ms >= pre_window[0]) & (time_ms < pre_window[1])]
    post = values[(time_ms >= post_window[0]) & (time_ms < post_window[1])]
    n = min(len(pre), len(post))
    if n < 2:
        raise RuntimeError("morphology comparison windows are empty")
    return np.concatenate([pre[-n:], post[:n]], axis=0), n * float(dt_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--i-th-ei", type=float, required=True)
    parser.add_argument("--eta-m", type=float, required=True)
    parser.add_argument("--tau-z", type=float, default=None)
    parser.add_argument("--tau-adp", type=float, default=None)
    parser.add_argument("--ee-dose", type=float, default=1.0)
    parser.add_argument("--etoi-dose", type=float, default=1.0)
    parser.add_argument("--duration-ms", type=float, default=10000.0)
    parser.add_argument("--post-runaway-ms", type=float, default=2000.0)
    parser.add_argument("--zm-mode", choices=("z_plus_m", "off"),
                        default="z_plus_m")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_round_config(args.config)
    config["simulation"] = dict(config["simulation"])
    config["simulation"]["duration_ms"] = float(args.duration_ms)
    candidate = config["arms"]["Joint"]
    output_root = ROOT / config["output_root"]
    started = time.time()
    substrate = build_substrate(
        config, candidate, args.seed, cache_dir=str(output_root / "network_cache"),
        ee_dose=float(args.ee_dose), etoi_dose=float(args.etoi_dose))
    dt = float(substrate.engine["dt"])
    tau_z = float(config["zm"]["tau_z"] if args.tau_z is None else args.tau_z)
    tau_adp = float(
        config["zm"]["tau_adp"] if args.tau_adp is None else args.tau_adp)

    slow = None
    if args.zm_mode == "z_plus_m":
        slow = ZMTracedSlowVars(
            substrate.n_e + substrate.n_i,
            substrate.params.V_th,
            MZSlowVarsConfig(
                use_z=True,
                use_m=True,
                I_th_EI=float(args.i_th_ei),
                tau_z=tau_z,
                tau_adp=tau_adp,
                eta_m=float(args.eta_m),
                trace_stride_steps=int(config["zm"]["trace_stride_steps"]),
            ),
            NE=substrate.n_e,
            core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
            trace_weights_E=substrate.h_e,
        )
    drive = make_external_drive(substrate, config["spatial_ou"], args.seed)

    from kick_probe import simulate_kick
    from lfp import LFPRecorder

    recorder = LFPRecorder(
        substrate.params,
        substrate.net["pos"],
        substrate.net["labels"],
        sites=substrate.contact_xy,
    )
    substrate.net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        substrate.params,
        substrate.net,
        KICK_BOOST=0.0,
        t_kick=1e9,
        V_th_per_neuron=substrate.vtheta,
        slow=slow,
        lfp_recorder=recorder,
        early_stop_runaway=True,
        es_thresh_hz=float(config["simulation"]["es_thresh_hz"]),
        es_dur_ms=float(config["simulation"]["es_dur_ms"]),
        post_runaway_record_ms=float(args.post_runaway_ms),
        external_e_rate_drive=drive,
    )
    operational_onset = result["runaway_early_stop_ms"]
    payload = {
        "status": "ZM_JOINT_MORPHOLOGY_CANARY_COMPLETE",
        "candidate_id": candidate,
        "seed": int(args.seed),
        "zm_mode": args.zm_mode,
        "parameters": {
            "I_th_EI": float(args.i_th_ei),
            "tau_z": tau_z,
            "tau_adp": tau_adp,
            "eta_m": float(args.eta_m),
            "eta_tau_product": float(args.eta_m) * tau_adp,
            "E_to_E_dose": float(args.ee_dose),
            "E_to_I_dose": float(args.etoi_dose),
        },
        "final_joint_eligible": bool(args.ee_dose > 0.0 and args.etoi_dose > 0.0),
        "operational_onset_ms": operational_onset,
        "interictal_dwell_min_ms": 2000.0,
        "wall_seconds": time.time() - started,
    }
    arrays = {
        "rate_E_hz": np.asarray(result["rate_E"], np.float32),
        "lfp_trace": np.asarray(result["lfp_trace"], np.float32),
        "lfp_dt_ms": np.asarray(dt, float),
        "contact_names": np.asarray(substrate.contact_names, dtype="U16"),
        "shaft_ids": np.asarray(substrate.shaft_ids, dtype="U8"),
    }
    if slow is not None:
        arrays.update({
            f"slow_{name}": np.asarray(values, np.float32)
            for name, values in slow.trace_arrays().items()
        })
        payload["slow_state_summary"] = slow.summary()
    if operational_onset is None:
        payload["verdict"] = "NO_TRANSITION_WITHIN_CANARY"
    else:
        onset = float(operational_onset) - float(config["simulation"]["es_dur_ms"])
        recruitment = rolling_full_field_recruitment(
            result["E_spk_bool"],
            substrate.positions_e,
            dt_ms=dt,
            sheet_l_mm=float(substrate.engine["L"]),
        )
        # Frequency and contact persistence are read after the initial crossing,
        # while the reference remains the last 500 ms of the interictal state.
        pre_window = (onset - 500.0, onset)
        stable_window = (onset + 500.0, onset + 1000.0)
        lfp_pair, pair_ms = _paired_windows(
            result["lfp_trace"], dt, pre_window, stable_window)
        rate_pair, _ = _paired_windows(
            result["rate_E"], dt, pre_window, stable_window)
        oscillation = contact_oscillation_metrics(
            lfp_pair, dt_ms=dt, onset_ms=pair_ms,
            pre_ms=pair_ms, post_ms=pair_ms)
        population = population_rate_frequency_metrics(
            rate_pair, dt_ms=dt, onset_ms=pair_ms,
            pre_ms=pair_ms, post_ms=pair_ms)
        morphology = summarize_runaway_morphology(
            recruitment, oscillation, onset_ms=onset, post_ms=1500.0,
            population_frequency=population)
        morphology["classification"] = classify_sustained_runaway(morphology)
        morphology["operational_detection_ms"] = float(operational_onset)
        morphology["scientific_onset_ms"] = onset
        morphology["frequency_reference_window_ms"] = list(pre_window)
        morphology["frequency_stable_window_ms"] = list(stable_window)
        dwell = onset >= 2000.0
        passed = bool(morphology["classification"]["all_checks_pass"] and dwell)
        payload["interictal_dwell_pass"] = dwell
        payload["runaway_morphology"] = morphology
        payload["verdict"] = (
            "JOINT_SUSTAINED_HIGH_OSCILLATORY_STATE_CANARY_PASS"
            if passed else "JOINT_ICTAL_MORPHOLOGY_CANARY_FAIL")
        arrays.update({
            "full_field_time_ms": np.asarray(recruitment["time_ms"], np.float32),
            "active_neuron_fraction_20ms": np.asarray(
                recruitment["active_neuron_fraction"], np.float32),
            "recruited_spatial_fraction_1mm": np.asarray(
                recruitment["recruited_spatial_fraction"], np.float32),
        })

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["wall_seconds"] = time.time() - started
    _atomic_npz(out.with_suffix(".npz"), **arrays)
    atomic_write_json(payload, str(out.with_suffix(".json")))
    print(json.dumps({"verdict": payload["verdict"], "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
