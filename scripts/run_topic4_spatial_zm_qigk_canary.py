#!/usr/bin/env python3
"""Run one full-edge spatial Z/qI--M development canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_global_recruited_oscillation import (  # noqa: E402
    classify_global_recruited_oscillation,
    contact_rhythm_metrics,
    detect_sustained_high_state_onset,
    recruitment_duty_metrics,
    state_rate_metrics,
)
from src.topic4_runaway_morphology import rolling_full_field_recruitment  # noqa: E402
from src.topic4_spatial_zm_qigk import (  # noqa: E402
    SpatialZMQIGKConfig,
    SpatialZMQIGKSlowVars,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
    make_external_drive,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_endpoint_contact_centers(substrate, side="union"):
    """Return the declared source-then-sink contact centres without refitting."""
    placement = substrate.extras["placement"]
    source_names = [str(name) for name in placement["source_names"]]
    sink_names = [str(name) for name in placement["sink_names"]]
    if side == "source":
        endpoint_names = source_names
    elif side == "sink":
        endpoint_names = sink_names
    elif side == "union":
        endpoint_names = [*source_names, *sink_names]
    else:
        raise ValueError("endpoint side must be source, sink, or union")
    contact_map = {
        str(name): np.asarray(xy, float)
        for name, xy in zip(substrate.contact_names, substrate.contact_xy)
    }
    if len(contact_map) != len(substrate.contact_names):
        raise RuntimeError("frozen contact contract contains duplicate names")
    missing = [name for name in endpoint_names if name not in contact_map]
    if missing:
        raise RuntimeError(f"frozen endpoint contacts missing: {missing}")
    return endpoint_names, np.stack([contact_map[name] for name in endpoint_names])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument(
        "--run-role",
        choices=("discovery", "confirmation", "ablation"),
        default="discovery",
        help="Predeclared role used by the image-blind family aggregator.",
    )
    parser.add_argument(
        "--parameter-set-id",
        default=None,
        help="Frozen parameter-family identifier; required for confirmation runs.",
    )
    parser.add_argument("--mode", choices=("hybrid", "q_only", "m_only", "slow_off"),
                        default="hybrid")
    parser.add_argument("--duration-ms", type=float, default=8000.0)
    parser.add_argument("--post-onset-ms", type=float, default=1600.0)
    parser.add_argument("--n-grid", type=int, default=64)
    parser.add_argument("--field-update-ms", type=float, default=1.0)
    parser.add_argument("--k-q", type=float, default=0.02)
    parser.add_argument("--tau-q", type=float, default=5000.0)
    parser.add_argument("--q-min", type=float, default=0.05)
    parser.add_argument("--q-init", type=float, default=1.0)
    parser.add_argument("--q-init-h-gain", type=float, default=0.0)
    parser.add_argument("--q-endpoint-gain", type=float, default=0.0)
    parser.add_argument("--q-source-gain", type=float, default=0.0)
    parser.add_argument("--q-sink-gain", type=float, default=0.0)
    parser.add_argument("--q-endpoint-sigma", type=float, default=2.0)
    parser.add_argument(
        "--q-endpoint-side", choices=("union", "source", "sink"),
        default="union",
    )
    parser.add_argument("--freeze-q", action="store_true")
    parser.add_argument("--q-a0", type=float, default=0.0)
    parser.add_argument("--q-a50", type=float, default=1.0)
    parser.add_argument("--q-hill-n", type=float, default=1.0)
    parser.add_argument("--sigma-q", type=float, default=1.5)
    parser.add_argument("--tau-m", type=float, default=62.5)
    parser.add_argument("--m-build-gain", type=float, default=1.0)
    parser.add_argument("--eta-m", type=float, default=0.05961275484469678)
    parser.add_argument("--m-current-threshold", type=float, default=0.0)
    parser.add_argument("--m-current-saturation-width", type=float, default=0.0)
    parser.add_argument("--m-current-hill-n", type=float, default=1.0)
    parser.add_argument("--m-state-ceiling", type=float, default=0.0)
    parser.add_argument("--m-spatial-mix", type=float, default=0.0)
    parser.add_argument("--sigma-m", type=float, default=0.5)
    parser.add_argument("--k-q-h-gain", type=float, default=0.0)
    parser.add_argument("--q-floor-h-gain", type=float, default=0.0)
    parser.add_argument("--eta-m-h-gain", type=float, default=0.0)
    parser.add_argument("--eta-m-source-add", type=float, default=0.0)
    parser.add_argument("--eta-m-sink-add", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.run_role == "confirmation" and not args.parameter_set_id:
        parser.error("--parameter-set-id is required for confirmation runs")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    round_config = load_round_config(str(config_path))
    round_config["simulation"] = dict(round_config["simulation"])
    round_config["simulation"]["duration_ms"] = float(args.duration_ms)
    output_root = ROOT / round_config["output_root"]
    candidate = round_config["arms"]["Joint"]
    started = time.time()
    substrate = build_substrate(
        round_config,
        candidate,
        int(args.seed),
        cache_dir=str(output_root / "network_cache"),
        ee_dose=1.0,
        etoi_dose=1.0,
    )
    dt = float(substrate.engine["dt"])
    endpoint_names, endpoint_centers_xy = _frozen_endpoint_contact_centers(
        substrate, side=args.q_endpoint_side)
    source_names, source_centers_xy = _frozen_endpoint_contact_centers(
        substrate, side="source")
    sink_names, sink_centers_xy = _frozen_endpoint_contact_centers(
        substrate, side="sink")
    hybrid_config = SpatialZMQIGKConfig(
        n_grid=int(args.n_grid),
        sigma_r_mm=0.5,
        tau_rate_ms=20.0,
        field_update_ms=float(args.field_update_ms),
        tau_q_ms=float(args.tau_q),
        k_q_per_ms=(0.0 if args.mode in {"m_only", "slow_off"}
                    else float(args.k_q)),
        q_min=float(args.q_min),
        q_init=float(args.q_init),
        q_init_h_gain=float(args.q_init_h_gain),
        q_endpoint_gain=float(args.q_endpoint_gain),
        q_source_gain=float(args.q_source_gain),
        q_sink_gain=float(args.q_sink_gain),
        q_endpoint_sigma_mm=float(args.q_endpoint_sigma),
        freeze_q=bool(args.freeze_q),
        sigma_q_mm=float(args.sigma_q),
        q_a0=float(args.q_a0),
        q_a50=float(args.q_a50),
        q_hill_n=float(args.q_hill_n),
        tau_m_ms=float(args.tau_m),
        m_build_gain=float(args.m_build_gain),
        eta_m=(0.0 if args.mode in {"q_only", "slow_off"}
               else float(args.eta_m)),
        m_current_threshold=float(args.m_current_threshold),
        m_current_saturation_width=float(args.m_current_saturation_width),
        m_current_hill_n=float(args.m_current_hill_n),
        m_state_ceiling=float(args.m_state_ceiling),
        m_spatial_mix=float(args.m_spatial_mix),
        sigma_m_mm=float(args.sigma_m),
        k_q_h_gain=float(args.k_q_h_gain),
        q_floor_h_gain=float(args.q_floor_h_gain),
        eta_m_h_gain=float(args.eta_m_h_gain),
        eta_m_source_add=float(args.eta_m_source_add),
        eta_m_sink_add=float(args.eta_m_sink_add),
        trace_stride_steps=max(1, int(round(1.0 / dt))),
    )
    slow = None
    if args.mode != "slow_off":
        slow = SpatialZMQIGKSlowVars(
            substrate.n_e + substrate.n_i,
            substrate.params.V_th,
            substrate.positions_e,
            substrate.positions_i,
            float(substrate.engine["L"]),
            substrate.h_e,
            core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
            endpoint_centers_xy=endpoint_centers_xy,
            source_centers_xy=source_centers_xy,
            sink_centers_xy=sink_centers_xy,
            cfg=hybrid_config,
        )
    drive = make_external_drive(substrate, round_config["spatial_ou"], args.seed)

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
        es_thresh_hz=120.0,
        es_dur_ms=100.0,
        post_runaway_record_ms=float(args.post_onset_ms),
        external_e_rate_drive=drive,
        verbose=True,
    )
    operational = result["runaway_early_stop_ms"]
    scientific_onset = detect_sustained_high_state_onset(
        result["rate_E"], dt_ms=dt)
    placement = substrate.extras["placement"]
    spatial_basis_contract = {
        "version": "frozen_patient_propagation_endpoints_v1",
        "source": (
            "stage-config narrow top-3 template source/sink contact sets, "
            "registered once to the frozen shared SNN plane"
        ),
        "source_contact_names": list(placement["source_names"]),
        "sink_contact_names": list(placement["sink_names"]),
        "active_endpoint_side": args.q_endpoint_side,
        "endpoint_contact_order": endpoint_names,
        "endpoint_centers_xy_mm": endpoint_centers_xy.tolist(),
        "source_centers_xy_mm": source_centers_xy.tolist(),
        "sink_centers_xy_mm": sink_centers_xy.tolist(),
        "q_field_components": {
            "legacy_endpoint_side": args.q_endpoint_side,
            "legacy_endpoint_gain": float(args.q_endpoint_gain),
            "source_gain": float(args.q_source_gain),
            "sink_gain": float(args.q_sink_gain),
        },
        "endpoint_field_rule": (
            "maximum of periodic Gaussian fields centred on the declared "
            "active endpoint contact set; sigma is declared in "
            "hybrid_config.q_endpoint_sigma_mm"
        ),
        "fitted_to_current_dynamics_result": False,
        "per_neuron_random_parameter_field": False,
    }
    payload = {
        "status": "SPATIAL_ZQIM_HYBRID_CANARY_COMPLETE",
        "candidate_id": candidate,
        "run_role": args.run_role,
        "parameter_set_id": args.parameter_set_id,
        "seed": int(args.seed),
        "mode": args.mode,
        "full_edge_contract": {
            "E_to_E_dose": 1.0,
            "E_to_I_dose": 1.0,
            "learned_edges_modified": False,
        },
        "hybrid_config": asdict(hybrid_config),
        "spatial_basis_contract": spatial_basis_contract,
        "protocol_contract": {
            "round_config_path": str(config_path.relative_to(ROOT)),
            "round_config_sha256": _sha256(config_path),
            "duration_ms": float(args.duration_ms),
            "post_onset_record_ms": float(args.post_onset_ms),
            "early_stop_rate_hz": 120.0,
            "early_stop_hold_ms": 100.0,
            "scientific_onset_rule": (
                "first >=120-Hz block inside the earliest 300-ms forward "
                "window whose median 20-ms block rate is >=120 Hz"
            ),
        },
        "operational_detection_ms": operational,
        "scientific_onset_ms": scientific_onset,
        "scientific_onset_contract": {
            "version": "oscillatory_median_v1",
            "threshold_hz": 120.0,
            "block_ms": 20.0,
            "forward_window_ms": 300.0,
            "isolated_bursts_are_onsets": False,
            "oscillatory_troughs_are_allowed": True,
        },
        "network": substrate.network_cache,
        "wall_seconds": time.time() - started,
    }
    recruitment = rolling_full_field_recruitment(
        result["E_spk_bool"],
        substrate.positions_e,
        dt_ms=dt,
        sheet_l_mm=float(substrate.engine["L"]),
    )
    if scientific_onset is None:
        payload["verdict"] = "NO_SUSTAINED_HIGH_STATE_WITHIN_CANARY"
    else:
        try:
            rates = state_rate_metrics(
                result["rate_E"], dt_ms=dt, onset_ms=scientific_onset)
            rec_metrics = recruitment_duty_metrics(
                recruitment, onset_ms=scientific_onset)
            rhythm = contact_rhythm_metrics(
                result["lfp_trace"], dt_ms=dt, onset_ms=scientific_onset)
            classification = classify_global_recruited_oscillation(
                onset_ms=scientific_onset,
                rates=rates,
                recruitment=rec_metrics,
                rhythm=rhythm,
            )
            payload["state_rate"] = rates
            payload["global_recruitment"] = rec_metrics
            payload["contact_rhythm"] = _json_safe(rhythm)
            payload["classification"] = classification
            payload["verdict"] = classification["status"]
        except ValueError as error:
            payload["verdict"] = "INCOMPLETE_POST_ONSET_WINDOW"
            payload["metric_error"] = str(error)
    if slow is not None:
        payload["slow_state_summary"] = slow.summary()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "time_ms": np.asarray(result["times"], np.float32),
        "rate_E_hz": np.asarray(result["rate_E"], np.float32),
        "rate_I_hz": np.asarray(result["rate_I"], np.float32),
        "lfp_trace": np.asarray(result["lfp_trace"], np.float32),
        "lfp_dt_ms": np.asarray(dt, float),
        "contact_names": np.asarray(substrate.contact_names, dtype="U16"),
        "shaft_ids": np.asarray(substrate.shaft_ids, dtype="U8"),
        "contact_xy_mm": np.asarray(substrate.contact_xy, np.float32),
        "full_field_time_ms": np.asarray(recruitment["time_ms"], np.float32),
        "active_neuron_fraction_20ms": np.asarray(
            recruitment["active_neuron_fraction"], np.float32),
        "recruited_spatial_fraction_1mm": np.asarray(
            recruitment["recruited_spatial_fraction"], np.float32),
        "positions_E": np.asarray(substrate.positions_e, np.float32),
        "h_E": np.asarray(substrate.h_e, np.float32),
    }
    if slow is not None:
        arrays.update({f"slow_{name}": np.asarray(values, np.float32)
                       for name, values in slow.trace_arrays().items()})
        arrays.update({
            "q_grid_final": np.asarray(slow.q_I, np.float32),
            "q_grid_initial": np.asarray(slow.q_init_grid, np.float32),
            "m_E_final": np.asarray(slow.m[:slow.nE], np.float32),
            "q_floor_grid": np.asarray(slow.q_floor_grid, np.float32),
            "k_q_grid": np.asarray(slow.k_q_grid, np.float32),
            "h_grid": np.asarray(slow.h_grid, np.float32),
            "endpoint_field": np.asarray(slow.endpoint_field, np.float32),
            "endpoint_centers_xy_mm": np.asarray(
                slow.endpoint_centers_xy, np.float32),
            "source_field": np.asarray(slow.source_field, np.float32),
            "sink_field": np.asarray(slow.sink_field, np.float32),
            "source_centers_xy_mm": np.asarray(
                slow.source_centers_xy, np.float32),
            "sink_centers_xy_mm": np.asarray(
                slow.sink_centers_xy, np.float32),
            "eta_m_E": np.asarray(slow.eta_m_E, np.float32),
        })
    _atomic_npz(out.with_suffix(".npz"), **arrays)
    payload["wall_seconds"] = time.time() - started
    atomic_write_json(_json_safe(payload), str(out.with_suffix(".json")))
    print(json.dumps({
        "verdict": payload["verdict"],
        "onset_ms": scientific_onset,
        "wall_seconds": round(float(payload["wall_seconds"]), 1),
        "out": str(out),
    }), flush=True)


if __name__ == "__main__":
    main()
