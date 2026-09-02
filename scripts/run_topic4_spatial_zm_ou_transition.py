#!/usr/bin/env python3
"""Stage B/C: one full dynamic spatial Z/M trajectory under persistent OU noise.

This is the round's primary instrument.  Unlike the frozen-q atlas it never
freezes the slow state: q(x,t) and m_i(t) evolve from their initial conditions
under the same stationary spatial OU field that runs before, during and after
the transition.  Everything the frozen Fig5A gate needs is computed here from
that single continuous trajectory, together with the evidence that makes the
result auditable:

* OU runtime evidence and before/after stationarity, so "the noise did not
  change across the transition" is measured rather than asserted;
* a numerical-stability screen, so a diverged integration cannot pass;
* per-contact rhythm diagnostics, so a failure can be attributed to a contact
  and a mechanism rather than to an aggregate score.

Learned E->E and E->I doses stay at 1.0 and no edge is modified.
"""
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
from scripts.run_topic4_spatial_zm_qigk_canary import (  # noqa: E402
    _frozen_endpoint_contact_centers,
    _frozen_gk_support_centers,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_global_recruited_oscillation import (  # noqa: E402
    classify_global_tonic_runaway,
    classify_global_recruited_oscillation,
    contact_rhythm_metrics,
    detect_sustained_high_state_onset,
    recruitment_duty_metrics,
    state_rate_metrics,
)
from src.topic4_ou_runtime_audit import (  # noqa: E402
    OUAuditProxy,
    OUProtocolProxy,
    spatial_correlation_length_mm,
    stationarity_report,
    temporal_autocorrelation_time_ms,
)
from src.topic4_runaway_morphology import rolling_full_field_recruitment  # noqa: E402
from src.topic4_tonic_fixed_point import (  # noqa: E402
    classify_tonic_fixed_point,
)
from src.topic4_spatial_zm_qigk import (  # noqa: E402
    SpatialZMQIGKConfig,
    SpatialZMQIGKSlowVars,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
    make_external_drive,
)

SETTLE_MS = 300.0
POST_GATE_MS = 1000.0
PRE_GATE_MS = 500.0


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


def numerical_stability(result, slow, *, tau_ref_e_ms):
    """Reject a diverged or unphysical integration before any gate is applied."""
    rate_e = np.asarray(result["rate_E"], float)
    rate_i = np.asarray(result["rate_I"], float)
    lfp = np.asarray(result["lfp_trace"], float)
    ceiling_hz = 1000.0 / float(tau_ref_e_ms)
    checks = {
        "rates_finite": bool(np.all(np.isfinite(rate_e))
                             and np.all(np.isfinite(rate_i))),
        "lfp_finite": bool(np.all(np.isfinite(lfp))),
        "rate_below_refractory_ceiling": bool(
            float(np.max(rate_e)) <= ceiling_hz + 1e-6),
    }
    detail = {
        "max_rate_E_hz": float(np.max(rate_e)),
        "refractory_ceiling_hz": float(ceiling_hz),
        "max_abs_lfp": float(np.max(np.abs(lfp))),
    }
    if slow is not None:
        q_grid = np.asarray(slow.q_I, float)
        m_e = np.asarray(slow.m[:slow.nE], float)
        checks["slow_state_finite"] = bool(np.all(np.isfinite(q_grid))
                                           and np.all(np.isfinite(m_e)))
        checks["q_within_declared_bounds"] = bool(
            float(np.min(q_grid)) >= float(np.min(slow.q_floor_grid)) - 1e-9
            and float(np.max(q_grid)) <= 1.0 + 1e-9)
        detail.update({
            "q_grid_min": float(np.min(q_grid)),
            "q_grid_max": float(np.max(q_grid)),
            "m_max": float(np.max(m_e)),
        })
    return {"all_checks_pass": bool(all(checks.values())), "checks": checks,
            "detail": detail}


def per_contact_diagnosis(*, rhythm, spikes, positions_e, contact_names,
                          contact_xy, shaft_ids, dt_ms, onset_ms, radius_mm):
    """Attribute each contact's verdict to local firing and to its spectrum.

    ``radius_mm`` is the LFP forward model's spatial summation cutoff, so the
    local rate reported here is measured over exactly the tissue the contact
    sees.  Reporting the per-window peak frequencies alongside makes the "20 Hz
    tonic versus 40-46 Hz rhythm" switching visible per contact rather than as
    an aggregate.
    """
    spikes = np.asarray(spikes, bool)
    time = np.arange(spikes.shape[0], dtype=float) * float(dt_ms)
    pre = (time >= float(onset_ms) - PRE_GATE_MS) & (time < float(onset_ms))
    post_start = float(onset_ms) + SETTLE_MS
    post = (time >= post_start) & (time < post_start + POST_GATE_MS)
    rows = []
    peak_hz = np.asarray(rhythm["per_window_contact_peak_hz"], float)
    fraction = np.asarray(rhythm["per_window_contact_peak_power_fraction"], float)
    ratio = np.asarray(rhythm["per_window_contact_band_power_ratio"], float)
    passing = np.asarray(rhythm["per_contact_consistently_rhythmic"], bool)
    for index, name in enumerate(contact_names):
        distance = np.linalg.norm(
            np.asarray(positions_e, float) - np.asarray(contact_xy[index], float),
            axis=1)
        local = distance <= float(radius_mm)
        n_local = int(np.sum(local))
        if n_local == 0:
            local = distance <= float(np.min(distance)) + 1e-9
            n_local = int(np.sum(local))
        pre_rate = float(spikes[pre][:, local].mean() / (dt_ms * 1e-3))
        post_rate = float(spikes[post][:, local].mean() / (dt_ms * 1e-3))
        window_pass = ((peak_hz[:, index] >= 30.0) & (peak_hz[:, index] <= 80.0)
                       & (fraction[:, index] >= 0.20) & (ratio[:, index] >= 2.0))
        failed = []
        if not np.all((peak_hz[:, index] >= 30.0) & (peak_hz[:, index] <= 80.0)):
            failed.append("peak_outside_30_80_hz")
        if not np.all(fraction[:, index] >= 0.20):
            failed.append("peak_power_fraction_below_0.20")
        if not np.all(ratio[:, index] >= 2.0):
            failed.append("band_power_ratio_below_2")
        rows.append({
            "contact": str(name),
            "shaft": str(shaft_ids[index]),
            "xy_mm": [float(contact_xy[index][0]), float(contact_xy[index][1])],
            "n_local_E_neurons": n_local,
            "local_rate_pre_hz": pre_rate,
            "local_rate_post_hz": post_rate,
            "local_rate_ratio_post_over_pre": float(
                post_rate / max(pre_rate, 1e-12)),
            "per_window_peak_hz": peak_hz[:, index].tolist(),
            "per_window_peak_power_fraction": fraction[:, index].tolist(),
            "per_window_band_power_ratio": ratio[:, index].tolist(),
            "n_windows_passing": int(np.sum(window_pass)),
            "consistently_rhythmic": bool(passing[index]),
            "failing_clauses": failed,
        })
    return rows


def tonic_contact_recruitment_diagnosis(
    *, spikes, positions_e, contact_names, contact_xy, shaft_ids, dt_ms,
    onset_ms, radius_mm,
):
    """Measure local tonic recruitment without invoking a spectrum gate.

    The tonic endpoint only needs the local pre/post firing-rate step.  Keeping
    this calculation independent of ``contact_rhythm_metrics`` prevents an
    incomplete or irrelevant spectral window from making a valid plateau
    unscorable.  The post window is the same onset+300..1300-ms interval used
    by the population-rate and recruitment summaries.
    """
    spikes = np.asarray(spikes, bool)
    time = np.arange(spikes.shape[0], dtype=float) * float(dt_ms)
    pre = (time >= max(0.0, float(onset_ms) - PRE_GATE_MS)) & (
        time < float(onset_ms))
    post_start = float(onset_ms) + SETTLE_MS
    post = (time >= post_start) & (time < post_start + POST_GATE_MS)
    if not np.any(pre):
        raise ValueError("tonic contact pre window is empty")
    expected_post = int(round(POST_GATE_MS / float(dt_ms)))
    if int(np.sum(post)) < expected_post - 1:
        raise ValueError("tonic contact post window is incomplete")
    rows = []
    for index, name in enumerate(contact_names):
        distance = np.linalg.norm(
            np.asarray(positions_e, float) - np.asarray(contact_xy[index], float),
            axis=1)
        local = distance <= float(radius_mm)
        n_local = int(np.sum(local))
        if n_local == 0:
            local = distance <= float(np.min(distance)) + 1e-9
            n_local = int(np.sum(local))
        pre_rate = float(spikes[pre][:, local].mean() / (dt_ms * 1e-3))
        post_rate = float(spikes[post][:, local].mean() / (dt_ms * 1e-3))
        rows.append({
            "contact": str(name),
            "shaft": str(shaft_ids[index]),
            "xy_mm": [float(contact_xy[index][0]), float(contact_xy[index][1])],
            "n_local_E_neurons": n_local,
            "local_rate_pre_hz": pre_rate,
            "local_rate_post_hz": post_rate,
            "local_rate_ratio_post_over_pre": float(
                post_rate / max(pre_rate, 1e-12)),
        })
    return rows


def _post_window_active_fraction(recruitment, onset_ms):
    time = np.asarray(recruitment["time_ms"], float)
    start = float(onset_ms) + SETTLE_MS
    selected = (time >= start) & (time < start + POST_GATE_MS)
    return np.asarray(
        recruitment["active_neuron_fraction"], float)[selected]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--run-role", choices=("discovery", "confirmation",
                                               "ablation"), default="discovery")
    parser.add_argument("--parameter-set-id", default=None)
    parser.add_argument("--mode", choices=("hybrid", "q_only", "m_only",
                                           "slow_off"), default="hybrid")
    parser.add_argument("--duration-ms", type=float, default=9000.0)
    parser.add_argument("--post-onset-ms", type=float, default=2200.0)
    parser.add_argument("--n-grid", type=int, default=64)
    parser.add_argument("--field-update-ms", type=float, default=1.0)
    # --- inhibitory-resource / qI field ---
    parser.add_argument("--k-q", type=float, default=0.02)
    parser.add_argument("--tau-q", type=float, default=5000.0)
    parser.add_argument("--q-min", type=float, default=0.05)
    parser.add_argument("--q-init", type=float, default=1.0)
    parser.add_argument("--q-init-h-gain", type=float, default=0.0)
    parser.add_argument("--q-floor-h-gain", type=float, default=0.0)
    parser.add_argument("--k-q-h-gain", type=float, default=0.0)
    parser.add_argument("--q-a0", type=float, default=0.0)
    parser.add_argument("--q-a50", type=float, default=1.0)
    parser.add_argument("--q-hill-n", type=float, default=1.0)
    parser.add_argument("--sigma-q", type=float, default=1.5)
    parser.add_argument("--freeze-q", action="store_true",
                        help="Hold q at its initial field; fast-subsystem probe "
                             "only, never a transition result.")
    parser.add_argument("--local-rate-radii-mm", default="",
                        help="Comma-separated radii; records per-contact local "
                             "E spike counts so local and global modulation "
                             "depth can be compared directly.")
    # --- adaptation / gK field ---
    parser.add_argument("--tau-m", type=float, default=12.5)
    parser.add_argument("--m-build-gain", type=float, default=1.0)
    parser.add_argument("--eta-m", type=float, default=0.0)
    parser.add_argument("--eta-m-h-gain", type=float, default=0.0)
    parser.add_argument("--m-current-threshold", type=float, default=0.0)
    parser.add_argument("--m-current-saturation-width", type=float, default=0.0)
    parser.add_argument("--m-current-hill-n", type=float, default=1.0)
    parser.add_argument("--m-state-ceiling", type=float, default=0.0)
    parser.add_argument("--m-spatial-mix", type=float, default=0.0)
    parser.add_argument("--sigma-m", type=float, default=0.5)
    # --- environment ---
    parser.add_argument("--ou-sigma-rate-per-ms", type=float, default=None)
    parser.add_argument("--ou-tau-ms", type=float, default=None)
    parser.add_argument("--ou-ell-mm", type=float, default=None)
    parser.add_argument("--ou-seed-offset", type=int, default=None,
                        help="Change only the noise realisation, not its statistics.")
    parser.add_argument("--snapshot-interval-ms", type=float, default=4.0)
    # --- Stage C causal controls on the environment (off by default) ---
    parser.add_argument("--ou-reseed-at-ms", type=float, default=None,
                        help="Swap the innovation stream at this time; every "
                             "declared OU statistic is unchanged.")
    parser.add_argument("--ou-reseed-seed", type=int, default=None)
    parser.add_argument("--ou-dip-start-ms", type=float, default=None)
    parser.add_argument("--ou-dip-duration-ms", type=float, default=None)
    parser.add_argument("--ou-dip-factor", type=float, default=1.0,
                        help="Amplitude scale inside the dip; 0 removes the "
                             "noise entirely for that bounded interval.")
    parser.add_argument("--out", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.run_role == "confirmation" and not args.parameter_set_id:
        raise SystemExit("--parameter-set-id is required for confirmation runs")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    round_config = load_round_config(str(config_path))
    round_config["simulation"] = dict(round_config["simulation"])
    round_config["simulation"]["duration_ms"] = float(args.duration_ms)
    ou_cfg = dict(round_config["spatial_ou"])
    if args.ou_sigma_rate_per_ms is not None:
        ou_cfg["sigma_rate_per_ms"] = float(args.ou_sigma_rate_per_ms)
    if args.ou_tau_ms is not None:
        ou_cfg["tau_ms"] = float(args.ou_tau_ms)
    if args.ou_ell_mm is not None:
        ou_cfg["ell_mm"] = float(args.ou_ell_mm)
    if args.ou_seed_offset is not None:
        ou_cfg["seed_offset"] = int(args.ou_seed_offset)

    output_root = ROOT / round_config["output_root"]
    started = time.time()
    substrate = build_substrate(
        round_config, round_config["arms"]["Joint"], int(args.seed),
        cache_dir=str(output_root / "network_cache"), ee_dose=1.0, etoi_dose=1.0)
    dt = float(substrate.engine["dt"])
    endpoint_names, endpoint_centers_xy = _frozen_endpoint_contact_centers(
        substrate, side="union")
    _, source_centers_xy = _frozen_endpoint_contact_centers(
        substrate, side="source")
    _, sink_centers_xy = _frozen_endpoint_contact_centers(substrate, side="sink")
    gk_names, gk_centers_xy = _frozen_gk_support_centers(
        substrate, support="downstream")

    hybrid_config = SpatialZMQIGKConfig(
        n_grid=int(args.n_grid), sigma_r_mm=0.5, tau_rate_ms=20.0,
        field_update_ms=float(args.field_update_ms),
        tau_q_ms=float(args.tau_q),
        k_q_per_ms=(0.0 if args.mode in {"m_only", "slow_off"}
                    else float(args.k_q)),
        q_min=float(args.q_min), q_init=float(args.q_init),
        q_init_h_gain=float(args.q_init_h_gain),
        q_floor_h_gain=float(args.q_floor_h_gain),
        k_q_h_gain=float(args.k_q_h_gain),
        q_a0=float(args.q_a0), q_a50=float(args.q_a50),
        q_hill_n=float(args.q_hill_n), sigma_q_mm=float(args.sigma_q),
        freeze_q=bool(args.freeze_q),
        tau_m_ms=float(args.tau_m), m_build_gain=float(args.m_build_gain),
        eta_m=(0.0 if args.mode in {"q_only", "slow_off"}
               else float(args.eta_m)),
        eta_m_h_gain=float(args.eta_m_h_gain),
        m_current_threshold=float(args.m_current_threshold),
        m_current_saturation_width=float(args.m_current_saturation_width),
        m_current_hill_n=float(args.m_current_hill_n),
        m_state_ceiling=float(args.m_state_ceiling),
        m_spatial_mix=float(args.m_spatial_mix), sigma_m_mm=float(args.sigma_m),
        trace_stride_steps=max(1, int(round(1.0 / dt))))
    slow = None
    if args.mode != "slow_off":
        slow = SpatialZMQIGKSlowVars(
            substrate.n_e + substrate.n_i, substrate.params.V_th,
            substrate.positions_e, substrate.positions_i,
            float(substrate.engine["L"]), substrate.h_e,
            core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
            endpoint_centers_xy=endpoint_centers_xy,
            source_centers_xy=source_centers_xy,
            sink_centers_xy=sink_centers_xy,
            gk_centers_xy=gk_centers_xy, cfg=hybrid_config)

    drive = make_external_drive(substrate, ou_cfg, args.seed)
    if drive is None:
        raise RuntimeError("this round requires a persistent spatial OU drive")
    protocol_requested = (args.ou_reseed_at_ms is not None
                          or args.ou_dip_start_ms is not None)
    if protocol_requested:
        proxy = OUProtocolProxy(
            drive, dt_ms=dt,
            snapshot_interval_ms=float(args.snapshot_interval_ms),
            reseed_at_ms=args.ou_reseed_at_ms,
            reseed_seed=args.ou_reseed_seed,
            dip_start_ms=args.ou_dip_start_ms,
            dip_duration_ms=args.ou_dip_duration_ms,
            dip_factor=float(args.ou_dip_factor))
    else:
        proxy = OUAuditProxy(
            drive, dt_ms=dt,
            snapshot_interval_ms=float(args.snapshot_interval_ms))

    from kick_probe import simulate_kick
    from lfp import LFPRecorder

    recorder = LFPRecorder(substrate.params, substrate.net["pos"],
                           substrate.net["labels"], sites=substrate.contact_xy)
    substrate.net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow, lfp_recorder=recorder,
        early_stop_runaway=True, es_thresh_hz=120.0, es_dur_ms=100.0,
        post_runaway_record_ms=float(args.post_onset_ms),
        external_e_rate_drive=proxy, verbose=True)

    operational = result["runaway_early_stop_ms"]
    onset_ms = detect_sustained_high_state_onset(result["rate_E"], dt_ms=dt)
    recruitment = rolling_full_field_recruitment(
        result["E_spk_bool"], substrate.positions_e, dt_ms=dt,
        sheet_l_mm=float(substrate.engine["L"]))
    stability = numerical_stability(
        result, slow, tau_ref_e_ms=float(substrate.params.tau_ref_E))

    ou_trace = drive.trace_arrays()
    snapshots = proxy.snapshot_arrays()
    temporal = temporal_autocorrelation_time_ms(
        snapshots["ou_grid_snapshots"], float(args.snapshot_interval_ms))
    spatial = spatial_correlation_length_mm(
        snapshots["ou_grid_snapshots"], drive.grid_spacing_mm)
    duration = float(len(result["rate_E"])) * dt
    stationarity = stationarity_report(
        proxy.neuron_snapshots(), proxy.snapshot_times_ms,
        onset_ms if onset_ms is not None else duration / 2.0)

    payload = {
        "status": "SPATIAL_ZM_OU_TRANSITION_COMPLETE",
        "stage": "B/C",
        "candidate_id": round_config["arms"]["Joint"],
        "run_role": args.run_role,
        "parameter_set_id": args.parameter_set_id,
        "seed": int(args.seed),
        "mode": args.mode,
        "full_edge_contract": {"E_to_E_dose": 1.0, "E_to_I_dose": 1.0,
                               "learned_edges_modified": False},
        "hybrid_config": asdict(hybrid_config),
        "applied_spatial_ou": ou_cfg,
        "protocol_contract": {
            "round_config_path": str(config_path.relative_to(ROOT)),
            "round_config_sha256": _sha256(config_path),
            "duration_ms": float(args.duration_ms),
            "post_onset_record_ms": float(args.post_onset_ms),
            "early_stop_rate_hz": 120.0,
            "early_stop_hold_ms": 100.0,
            "scientific_onset_rule": (
                "first >=120-Hz block inside the earliest 300-ms forward "
                "window whose median 20-ms block rate is >=120 Hz"),
            "persistent_noise_contract": (
                "one stationary spatial OU field drives the E afferent rate on "
                "every membrane step, before during and after the transition; "
                "no timed pulse train, onset-triggered drive, periodic noise or "
                "injected 30-80 Hz signal is used"),
        },
        "scientific_onset_contract": {
            "version": "oscillatory_median_v1", "threshold_hz": 120.0,
            "block_ms": 20.0, "forward_window_ms": 300.0,
            "isolated_bursts_are_onsets": False,
            "oscillatory_troughs_are_allowed": True},
        "operational_detection_ms": operational,
        "scientific_onset_ms": onset_ms,
        "trajectory_duration_ms": duration,
        "numerical_stability": stability,
        "ou_runtime_evidence": proxy.runtime_evidence(len(result["rate_E"])),
        "ou_negative_rate_clipping": result["external_e_rate_drive"],
        "ou_measured_temporal": {"tau_hat_ms": temporal["tau_hat_ms"],
                                 "declared_tau_ms": float(ou_cfg["tau_ms"])},
        "ou_measured_spatial": {
            "correlation_length_mm_1_over_e":
                spatial["correlation_length_mm_1_over_e"],
            "declared_ell_mm": float(ou_cfg["ell_mm"])},
        "ou_stationarity_across_transition": stationarity,
        "ou_protocol_control": (proxy.protocol_evidence()
                                if protocol_requested else None),
        "ou_realisation_sha256": hashlib.sha256(
            np.ascontiguousarray(ou_trace["spatial_sd_rate_per_ms"]).tobytes()
        ).hexdigest(),
        "network": substrate.network_cache,
    }

    contact_rows = None
    if onset_ms is None:
        payload["verdict"] = "NO_SUSTAINED_HIGH_STATE_WITHIN_RUN"
    elif not stability["all_checks_pass"]:
        payload["verdict"] = "NUMERICALLY_UNSTABLE"
    else:
        # The tonic endpoint is evaluated independently of the legacy rhythm
        # branch.  A missing 500-ms spectral baseline must not erase a valid
        # tonic plateau or its local-contact recruitment evidence.
        try:
            rates = state_rate_metrics(result["rate_E"], dt_ms=dt,
                                       onset_ms=onset_ms)
            rec_metrics = recruitment_duty_metrics(recruitment,
                                                   onset_ms=onset_ms)
            contact_rows = tonic_contact_recruitment_diagnosis(
                spikes=result["E_spk_bool"],
                positions_e=substrate.positions_e,
                contact_names=substrate.contact_names,
                contact_xy=substrate.contact_xy,
                shaft_ids=substrate.shaft_ids, dt_ms=dt, onset_ms=onset_ms,
                radius_mm=float(substrate.params.Rr))
            payload["state_rate"] = rates
            payload["global_recruitment"] = rec_metrics
            tonic = classify_tonic_fixed_point(
                result["rate_E"], dt_ms=dt, onset_ms=onset_ms,
                active_fraction_20ms=_post_window_active_fraction(
                    recruitment, onset_ms))
            payload["tonic_global_runaway"] = classify_global_tonic_runaway(
                onset_ms=onset_ms,
                observed_post_transition_ms=(
                    float(len(result["rate_E"])) * dt - float(onset_ms)),
                rates=rates,
                recruitment=rec_metrics,
            )
            payload["criterion10_tonic_exclusion"] = tonic
            payload["per_contact_diagnosis"] = contact_rows
            payload["tonic_verdict"] = payload["tonic_global_runaway"]["status"]
        except ValueError as error:
            payload["tonic_verdict"] = "INCOMPLETE_TONIC_METRIC_WINDOW"
            payload["tonic_metric_error"] = str(error)

        # Preserve the old oscillatory endpoint as a parallel diagnostic.  It
        # may legitimately be unscorable when onset occurs before its 500-ms
        # pre-spectrum window; that does not alter the tonic verdict above.
        try:
            rhythm = contact_rhythm_metrics(result["lfp_trace"], dt_ms=dt,
                                            onset_ms=onset_ms)
            classification = classify_global_recruited_oscillation(
                onset_ms=onset_ms, rates=rates, recruitment=rec_metrics,
                rhythm=rhythm)
            rhythmic_rows = per_contact_diagnosis(
                rhythm=rhythm, spikes=result["E_spk_bool"],
                positions_e=substrate.positions_e,
                contact_names=substrate.contact_names,
                contact_xy=substrate.contact_xy,
                shaft_ids=substrate.shaft_ids, dt_ms=dt, onset_ms=onset_ms,
                radius_mm=float(substrate.params.Rr))
            payload["classification"] = classification
            payload["contact_rhythm"] = _json_safe(rhythm)
            payload["per_contact_diagnosis"] = rhythmic_rows
            contact_rows = rhythmic_rows
            payload["n_rhythmic_contacts"] = int(sum(
                row["consistently_rhythmic"] for row in rhythmic_rows))
            payload["nine_clause_lfp_gate_pass"] = bool(
                classification["all_checks_pass"])
            payload["fig5a_full_gate_pass"] = bool(
                classification["all_checks_pass"] and tonic["all_checks_pass"])
            payload["verdict"] = (
                classification["status"] if tonic["all_checks_pass"]
                else "TONIC_HIGH_RATE_FIXED_POINT_WITH_RIPPLE")
        except (NameError, ValueError) as error:
            payload["verdict"] = "OSCILLATORY_ENDPOINT_UNSCORABLE"
            payload["oscillatory_metric_error"] = str(error)
    if slow is not None:
        payload["slow_state_summary"] = slow.summary()
    payload["boundary"] = (
        "model-state morphology screen under a stationary random environment; "
        "not a clinical seizure, not a patient-mechanism identification")

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
        "ou_lag_ms": np.asarray(temporal["lag_ms"], np.float32),
        "ou_lag_correlation": np.asarray(temporal["lag_correlation"], np.float32),
        "ou_distance_mm": np.asarray(spatial["distance_mm"], np.float32),
        "ou_spatial_correlation": np.asarray(spatial["correlation"], np.float32),
    }
    arrays.update({f"ou_{name}": np.asarray(values, np.float32)
                   for name, values in ou_trace.items()})
    if slow is not None:
        arrays.update({f"slow_{name}": np.asarray(values, np.float32)
                       for name, values in slow.trace_arrays().items()})
        arrays.update({
            "q_grid_final": np.asarray(slow.q_I, np.float32),
            "q_grid_initial": np.asarray(slow.q_init_grid, np.float32),
            "q_floor_grid": np.asarray(slow.q_floor_grid, np.float32),
            "k_q_grid": np.asarray(slow.k_q_grid, np.float32),
            "h_grid": np.asarray(slow.h_grid, np.float32),
            "m_E_final": np.asarray(slow.m[:slow.nE], np.float32),
            "eta_m_E": np.asarray(slow.eta_m_E, np.float32),
        })
    radii = [float(value) for value in str(args.local_rate_radii_mm).split(",")
             if value.strip()]
    if radii:
        spikes = np.asarray(result["E_spk_bool"], bool)
        positions = np.asarray(substrate.positions_e, float)
        contacts = np.asarray(substrate.contact_xy, float)
        for radius in radii:
            counts = np.empty((spikes.shape[0], len(contacts)), np.float32)
            sizes = np.empty(len(contacts), np.int32)
            for index, centre in enumerate(contacts):
                local = np.linalg.norm(positions - centre, axis=1) <= radius
                sizes[index] = int(np.sum(local))
                counts[:, index] = spikes[:, local].sum(axis=1)
            key = f"local_spike_count_r{radius:g}mm".replace(".", "p")
            arrays[key] = counts
            arrays[f"{key}_n_neurons"] = sizes
        payload["local_rate_radii_mm"] = radii
    if contact_rows is not None and "contact_rhythm" in payload:
        arrays["per_window_contact_peak_hz"] = np.asarray(
            payload["contact_rhythm"]["per_window_contact_peak_hz"], np.float32)
    _atomic_npz(out.with_suffix(".npz"), **arrays)
    payload["wall_seconds"] = time.time() - started
    atomic_write_json(_json_safe(payload), str(out.with_suffix(".json")))
    print(json.dumps({
        "verdict": payload["verdict"],
        "onset_ms": onset_ms,
        "n_rhythmic_contacts": payload.get("n_rhythmic_contacts"),
        "median_post_hz": (payload.get("state_rate") or {}).get(
            "median_post_hz"),
        "median_peak_hz": (payload.get("contact_rhythm") or {}).get(
            "median_contact_peak_hz"),
        "wall_seconds": round(float(payload["wall_seconds"]), 1),
        "out": str(out),
    }), flush=True)


if __name__ == "__main__":
    main()
