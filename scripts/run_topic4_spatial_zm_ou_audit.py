#!/usr/bin/env python3
"""Stage A: prove the spatial OU drive runs, and qualify the slow-off low state.

Two questions, one trajectory, no slow variables:

1. **Runtime.** Is the declared spatial OU field actually queried on every
   membrane step, and are its measured mean / SD / autocorrelation time /
   correlation length the declared ones?  A config entry is not evidence.
2. **Low-state eligibility.** With Z/M off, does this OU working point give
   intermittent local events, no sustained global high state, and enough
   readable events to call the pre-transition segment an interictal residence?

The working point is chosen on clause 2 alone.  Nothing in this script looks at
post-transition rhythm, so the OU choice cannot be tuned on the outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_global_recruited_oscillation import (  # noqa: E402
    detect_sustained_high_state_onset,
)
from src.topic4_ou_runtime_audit import (  # noqa: E402
    OUAuditProxy,
    spatial_correlation_length_mm,
    stationarity_report,
    temporal_autocorrelation_time_ms,
)
from src.topic4_runaway_morphology import rolling_full_field_recruitment  # noqa: E402
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


def low_state_qualification(*, onset_ms, rate_hz, dt_ms, events,
                            recruitment, min_events=6):
    """Frozen slow-off eligibility clauses, declared before any run.

    The clauses mirror the Fig5A pre-transition requirements so that a low state
    accepted here can serve as the pre-window of a candidate trajectory.
    """
    from scipy.ndimage import uniform_filter1d

    rate = np.asarray(rate_hz, float)
    smooth = uniform_filter1d(
        rate, size=max(1, int(round(20.0 / float(dt_ms)))), mode="nearest")
    returned = int(sum(bool(row["returned"]) for row in events))
    joint = np.asarray(recruitment["active_neuron_fraction"], float)
    sheet = np.asarray(recruitment["recruited_spatial_fraction"], float)
    sustained_global = float(np.mean((joint >= 0.5) & (sheet >= 0.5)))
    clauses = {
        "no_sustained_global_high_state": {
            "onset_ms": onset_ms,
            "pass": onset_ms is None,
        },
        "low_state_rate_bounded": {
            "median_hz": float(np.median(smooth)),
            "q95_hz": float(np.quantile(smooth, 0.95)),
            "pass": bool(float(np.median(smooth)) <= 60.0
                         and float(np.quantile(smooth, 0.95)) < 120.0),
        },
        "intermittent_not_continuous_recruitment": {
            "joint_global_recruitment_duty": sustained_global,
            "pass": bool(sustained_global <= 0.10),
        },
        "enough_readable_events": {
            "n_detected_events": int(len(events)),
            "n_returned_events": returned,
            "minimum_required": int(min_events),
            "pass": bool(returned >= int(min_events)),
        },
    }
    return {
        "status": ("SLOW_OFF_LOW_STATE_ELIGIBLE" if all(
            clause["pass"] for clause in clauses.values())
            else "SLOW_OFF_LOW_STATE_NOT_ELIGIBLE"),
        "all_clauses_pass": bool(all(c["pass"] for c in clauses.values())),
        "clauses": clauses,
        "selection_rule": (
            "OU working point is selected on these slow-off clauses only; "
            "post-transition rhythm scores are never consulted"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--duration-ms", type=float, default=6000.0)
    parser.add_argument("--ou-sigma-rate-per-ms", type=float, default=None)
    parser.add_argument("--ou-tau-ms", type=float, default=None)
    parser.add_argument("--ou-ell-mm", type=float, default=None)
    parser.add_argument("--ou-mode", default=None)
    parser.add_argument("--snapshot-interval-ms", type=float, default=2.0)
    parser.add_argument("--min-returned-events", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    round_config = load_round_config(str(config_path))
    round_config["simulation"] = dict(round_config["simulation"])
    round_config["simulation"]["duration_ms"] = float(args.duration_ms)
    ou_cfg = dict(round_config["spatial_ou"])
    declared = dict(ou_cfg)
    if args.ou_mode is not None:
        ou_cfg["mode"] = args.ou_mode
    if args.ou_sigma_rate_per_ms is not None:
        ou_cfg["sigma_rate_per_ms"] = float(args.ou_sigma_rate_per_ms)
    if args.ou_tau_ms is not None:
        ou_cfg["tau_ms"] = float(args.ou_tau_ms)
    if args.ou_ell_mm is not None:
        ou_cfg["ell_mm"] = float(args.ou_ell_mm)

    output_root = ROOT / round_config["output_root"]
    started = time.time()
    substrate = build_substrate(
        round_config, round_config["arms"]["Joint"], int(args.seed),
        cache_dir=str(output_root / "network_cache"), ee_dose=1.0, etoi_dose=1.0)
    dt = float(substrate.engine["dt"])
    nsteps = int(round(float(args.duration_ms) / dt))

    drive = make_external_drive(substrate, ou_cfg, args.seed)
    if drive is None:
        raise RuntimeError("stage A requires an active spatial OU drive")
    proxy = OUAuditProxy(drive, dt_ms=dt,
                         snapshot_interval_ms=float(args.snapshot_interval_ms))

    from kick_probe import simulate_kick
    from lfp import LFPRecorder

    recorder = LFPRecorder(substrate.params, substrate.net["pos"],
                           substrate.net["labels"], sites=substrate.contact_xy)
    substrate.net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None, lfp_recorder=recorder,
        early_stop_runaway=False, external_e_rate_drive=proxy, verbose=True)

    onset_ms = detect_sustained_high_state_onset(result["rate_E"], dt_ms=dt)
    recruitment = rolling_full_field_recruitment(
        result["E_spk_bool"], substrate.positions_e, dt_ms=dt,
        sheet_l_mm=float(substrate.engine["L"]))

    cmrun = substrate.extras["cmrun"]
    from src.sef_hfo_events import detect_events
    active, active_dt = cmrun.active_fraction(
        np.asarray(result["E_spk_bool"], bool), dt, cmrun.BIN_MS)
    detected = detect_events(active, active_dt,
                             event_on_frac=substrate.detector_threshold)
    events = [{"t_on_ms": float(row["t_on"]), "t_off_ms": float(row["t_off"]),
               "duration_ms": float(row["dur_ms"]),
               "peak_active_fraction": float(row["peak_ext"]),
               "returned": bool(row["returned"])} for row in detected]

    ou_trace = drive.trace_arrays()
    snapshots = proxy.snapshot_arrays()
    grid = snapshots["ou_grid_snapshots"]
    neuron = proxy.neuron_snapshots()
    temporal = temporal_autocorrelation_time_ms(
        grid, float(args.snapshot_interval_ms))
    spatial = spatial_correlation_length_mm(grid, drive.grid_spacing_mm)
    stationarity = stationarity_report(
        neuron, proxy.snapshot_times_ms,
        onset_ms if onset_ms is not None else float(args.duration_ms) / 2.0)
    qualification = low_state_qualification(
        onset_ms=onset_ms, rate_hz=result["rate_E"], dt_ms=dt, events=events,
        recruitment=recruitment, min_events=int(args.min_returned_events))

    payload = {
        "status": "SPATIAL_ZM_OU_AUDIT_COMPLETE",
        "stage": "A",
        "mode": "slow_off",
        "seed": int(args.seed),
        "duration_ms": float(args.duration_ms),
        "protocol_contract": {
            "round_config_path": str(config_path.relative_to(ROOT)),
            "round_config_sha256": _sha256(config_path),
            "early_stop_runaway": False,
            "slow_variables": "absent (slow=None)",
            "scientific_onset_rule_version": "oscillatory_median_v1",
        },
        "declared_spatial_ou": declared,
        "applied_spatial_ou": ou_cfg,
        "ou_runtime_evidence": proxy.runtime_evidence(nsteps),
        "ou_negative_rate_clipping": result["external_e_rate_drive"],
        "ou_measured_temporal": {
            "tau_hat_ms": temporal["tau_hat_ms"],
            "declared_tau_ms": float(ou_cfg["tau_ms"]),
            "n_lags_used_in_fit": temporal["n_lags_used_in_fit"],
        },
        "ou_measured_spatial": {
            "correlation_length_mm_1_over_e":
                spatial["correlation_length_mm_1_over_e"],
            "declared_ell_mm": float(ou_cfg["ell_mm"]),
            "grid_spacing_mm": float(drive.grid_spacing_mm),
        },
        "ou_stationarity": stationarity,
        "ou_trace_sha256": hashlib.sha256(
            np.ascontiguousarray(ou_trace["spatial_sd_rate_per_ms"]).tobytes()
        ).hexdigest(),
        "low_state_qualification": qualification,
        "events": events,
        "network": substrate.network_cache,
        "boundary": (
            "model-state audit only; the OU field is a stationary environmental "
            "process, not a stimulus, and no clinical claim follows from it"),
    }
    payload["verdict"] = qualification["status"]

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
        "active_fraction": np.asarray(active, np.float32),
        "active_fraction_dt_ms": np.asarray(active_dt, float),
        "ou_lag_ms": np.asarray(temporal["lag_ms"], np.float32),
        "ou_lag_correlation": np.asarray(temporal["lag_correlation"], np.float32),
        "ou_distance_mm": np.asarray(spatial["distance_mm"], np.float32),
        "ou_spatial_correlation": np.asarray(spatial["correlation"], np.float32),
    }
    arrays.update({f"ou_{name}": np.asarray(values, np.float32)
                   for name, values in ou_trace.items()})
    arrays.update(snapshots)
    _atomic_npz(out.with_suffix(".npz"), **arrays)
    payload["wall_seconds"] = time.time() - started
    atomic_write_json(_json_safe(payload), str(out.with_suffix(".json")))
    print(json.dumps({
        "verdict": payload["verdict"],
        "onset_ms": onset_ms,
        "n_returned_events":
            qualification["clauses"]["enough_readable_events"]["n_returned_events"],
        "median_rate_hz":
            qualification["clauses"]["low_state_rate_bounded"]["median_hz"],
        "tau_hat_ms": temporal["tau_hat_ms"],
        "ell_hat_mm": spatial["correlation_length_mm_1_over_e"],
        "wall_seconds": round(float(payload["wall_seconds"]), 1),
        "out": str(out),
    }), flush=True)


if __name__ == "__main__":
    main()
