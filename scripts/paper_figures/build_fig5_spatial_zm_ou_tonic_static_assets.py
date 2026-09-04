#!/usr/bin/env python3
"""Build the missing static-panel assets for the tonic Fig. 5.

This is an observation/probe wrapper around the locked seed-1842 trajectory.
It captures two exact simulator checkpoints without changing an update or RNG
draw, verifies the replay against every archived scientific trace, and then
runs the same 16-cell perturbation at 16 frozen stratified-random locations in
the low and early-runaway states.
"""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "src" / "snn_engine"
for search_path in (ROOT, ENGINE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

ARCHIVE = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_node_local_connectivity_plus_zm/spatial_zm_ou"
)
SOURCE_NPZ = ARCHIVE / "tonic_confirmation_v2/tonic_b0_v2_s1842.npz"
SOURCE_JSON = SOURCE_NPZ.with_suffix(".json")
SOURCE_SHA256 = "283ee32711a2c5388f065d9e2faa9a54390f788bb3c7496c7e8cd4ea993a7248"
OUT_DIR = ARCHIVE / "paper_ready_fig5_static"
REPLAY_BASE = OUT_DIR / "seed1842_static_asset_replay"
ASSET_NPZ = OUT_DIR / "seed1842_static_panels.npz"
ASSET_JSON = ASSET_NPZ.with_suffix(".json")
EVENT_NPZ = OUT_DIR / "seed1842_static_event_and_energy.npz"
EVENT_JSON = EVENT_NPZ.with_suffix(".json")
LOW_CHECKPOINT = OUT_DIR / "seed1842_checkpoint_t0200ms.npz"
HIGH_CHECKPOINT = OUT_DIR / "seed1842_checkpoint_t0600ms.npz"

LOW_TIME_MS = 200.0
HIGH_TIME_MS = 600.0
EVENT_HALF_WINDOW_MS = 50.0
EARLY_ENERGY_WINDOW_MS = 100.0
PROBE_WINDOW_MS = 200.0
PROBE_SPLIT_MS = 50.0
PROBE_DOSE_CELLS = 16
SITE_SEED = 20260820
N_SITE_SIDE = 4
SITE_MARGIN_MM = 1.2


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


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def patch_spatial_checkpoint_support():
    """Teach the existing checkpoint helper about this spatial slow object.

    The simulation equations are untouched.  The extra fields are only the
    mutable arrays needed for an exact continuation of SpatialZMQIGKSlowVars.
    """
    import checkpoint as ckpt
    from src.topic4_ou_runtime_audit import OUAuditProxy
    from src.topic4_spatial_zm_qigk import SpatialZMQIGKSlowVars

    extra_arrays = (
        "q_I", "qdriver_rE", "qdriver_rI", "field_count_E",
        "field_count_I", "last_m_drive_E", "last_q_drive",
    )
    ckpt._SLOW_ARRAYS = tuple(dict.fromkeys(ckpt._SLOW_ARRAYS + extra_arrays))
    original_capture = ckpt.capture
    original_restore = ckpt.restore_slow

    def capture_with_spatial(**kwargs):
        state = original_capture(**kwargs)
        slow = kwargs.get("slow")
        if isinstance(slow, SpatialZMQIGKSlowVars):
            state["slow"].update({
                "q_I": np.array(slow.q_I, copy=True),
                "qdriver_rE": np.array(slow._qdriver.rE, copy=True),
                "qdriver_rI": np.array(slow._qdriver.rI, copy=True),
                "field_count_E": np.array(slow._field_count_E, copy=True),
                "field_count_I": np.array(slow._field_count_I, copy=True),
                "last_m_drive_E": np.array(slow._last_m_drive_E, copy=True),
                "last_q_drive": np.array(slow._last_q_drive, copy=True),
                "field_steps_seen": int(slow._field_steps_seen),
                "field_steps_per_update": (
                    None if slow._field_steps_per_update is None
                    else int(slow._field_steps_per_update)),
            })
        return state

    def restore_with_spatial(state, slow):
        original_restore(state, slow)
        payload = state.get("slow")
        if isinstance(slow, SpatialZMQIGKSlowVars) and payload is not None:
            slow.q_I[:] = payload["q_I"]
            slow._qdriver.rE[:] = payload["qdriver_rE"]
            slow._qdriver.rI[:] = payload["qdriver_rI"]
            slow._field_count_E[:] = payload["field_count_E"]
            slow._field_count_I[:] = payload["field_count_I"]
            slow._last_m_drive_E[:] = payload["last_m_drive_E"]
            slow._last_q_drive[:] = payload["last_q_drive"]
            slow._field_steps_seen = int(payload["field_steps_seen"])
            steps = payload["field_steps_per_update"]
            slow._field_steps_per_update = None if steps is None else int(steps)

    def proxy_getattr(self, name):
        return getattr(self.drive, name)

    ckpt.capture = capture_with_spatial
    ckpt.restore_slow = restore_with_spatial
    OUAuditProxy.__getattr__ = proxy_getattr
    return ckpt


def build_context(round_config, source, *, substrate=None):
    from scripts.run_topic4_spatial_zm_qigk_canary import (
        _frozen_endpoint_contact_centers,
        _frozen_gk_support_centers,
    )
    from src.topic4_spatial_zm_qigk import (
        SpatialZMQIGKConfig,
        SpatialZMQIGKSlowVars,
    )
    from src.topic4_zm_ictal_transition import build_substrate, make_external_drive

    seed = int(source["seed"])
    if substrate is None:
        output_root = ROOT / round_config["output_root"]
        substrate = build_substrate(
            round_config, round_config["arms"]["Joint"], seed,
            cache_dir=str(output_root / "network_cache"), ee_dose=1.0,
            etoi_dose=1.0)
    _, endpoint_centers = _frozen_endpoint_contact_centers(
        substrate, side="union")
    _, source_centers = _frozen_endpoint_contact_centers(
        substrate, side="source")
    _, sink_centers = _frozen_endpoint_contact_centers(substrate, side="sink")
    _, gk_centers = _frozen_gk_support_centers(
        substrate, support="downstream")
    config = SpatialZMQIGKConfig(**source["hybrid_config"])
    slow = SpatialZMQIGKSlowVars(
        substrate.n_e + substrate.n_i, substrate.params.V_th,
        substrate.positions_e, substrate.positions_i,
        float(substrate.engine["L"]), substrate.h_e,
        core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
        endpoint_centers_xy=endpoint_centers,
        source_centers_xy=source_centers,
        sink_centers_xy=sink_centers,
        gk_centers_xy=gk_centers, cfg=config)
    drive = make_external_drive(
        substrate, source["applied_spatial_ou"], seed)
    return substrate, slow, drive


def continue_from_state(substrate, round_config, source, state, *, duration_ms,
                        packet=None):
    from kick_probe import simulate_kick
    from params import Params

    substrate, slow, drive = build_context(
        round_config, source, substrate=substrate)
    engine = substrate.engine
    dt_ms = float(engine["dt"])
    params = Params(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=float(duration_ms), dt=dt_ms,
        nu_ext_ratio=substrate.params.nu_ext_ratio,
        seed=int(substrate.params.seed))
    substrate.net["rng"] = np.random.default_rng(int(source["seed"]))
    forced = {}
    if packet is not None:
        full = np.zeros(substrate.n_e + substrate.n_i, bool)
        full[:substrate.n_e] = np.asarray(packet, bool)
        forced = {
            "forced_spike_mask": full,
            "forced_spike_ms": float(state["absolute_time_ms"]),
        }
    result = simulate_kick(
        params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow,
        early_stop_runaway=False,
        external_e_rate_drive=drive,
        resume_state=copy.deepcopy(state),
        time_offset_ms=float(state["absolute_time_ms"]),
        **forced)
    return substrate, result


def probe_state(round_config, source, state, sites, *, site_indices=None):
    from src.topic4_zm_perturbation import (
        in_window_ignition,
        response_metrics,
        select_packet,
    )

    substrate, _, _ = build_context(round_config, source)
    _, sham = continue_from_state(
        substrate, round_config, source, state, duration_ms=PROBE_WINDOW_MS)
    dt_ms = float(substrate.engine["dt"])
    cmrun = substrate.extras["cmrun"]
    sham_active, active_dt = cmrun.active_fraction(
        np.asarray(sham["E_spk_bool"], bool), dt_ms, cmrun.BIN_MS)
    rows = []
    early_fields = []
    indices = (list(range(len(sites))) if site_indices is None
               else sorted(set(int(value) for value in site_indices)))
    for site_index in indices:
        xy = np.asarray(sites[int(site_index)], float)
        packet = select_packet(
            substrate.positions_e, xy, n_cells=PROBE_DOSE_CELLS,
            radius_mm=float(round_config["perturbation"]["packet_radius_mm"]))
        _, probe = continue_from_state(
            substrate, round_config, source, state,
            duration_ms=PROBE_WINDOW_MS, packet=packet)
        probe_active, _ = cmrun.active_fraction(
            np.asarray(probe["E_spk_bool"], bool), dt_ms, cmrun.BIN_MS)
        metrics = response_metrics(
            probe, sham, dt_ms=dt_ms, positions_e=substrate.positions_e,
            packet_mask=packet, packet_xy=xy,
            envelope_probe=np.zeros((15, 1)),
            envelope_sham=np.zeros((15, 1)), envelope_dt_ms=2.0,
            inject_step=0, split_ms=PROBE_SPLIT_MS,
            window_ms=PROBE_WINDOW_MS)
        regime = in_window_ignition(
            probe_active, sham_active, active_dt_ms=float(active_dt),
            detector_threshold=substrate.detector_threshold,
            inject_ms=0.0, window_ms=PROBE_WINDOW_MS,
            probe_rate_hz=np.asarray(probe["rate_E"], float), dt_ms=dt_ms,
            es_thresh_hz=120.0, es_dur_ms=100.0)
        rows.append({
            "site_index": int(site_index),
            "site_xy_mm": [float(xy[0]), float(xy[1])],
            "dose_cells": PROBE_DOSE_CELLS,
            "susceptibility": float(metrics["susceptibility"]),
            "excess_spikes_early": float(metrics["excess_spikes_early"]),
            "excess_spikes_late": float(metrics["excess_spikes_late"]),
            "r90_mm": float(metrics["r90_mm"]),
            **json_safe(regime),
        })
        early_fields.append(np.asarray(
            metrics["excess_per_neuron_early"], np.float32))
    return substrate, sham, rows, np.asarray(early_fields, np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    if sha256(SOURCE_NPZ) != SOURCE_SHA256:
        raise RuntimeError("locked seed-1842 source hash changed")
    source = json.loads(SOURCE_JSON.read_text())
    if not (
        int(source["seed"]) == 1842
        and source["tonic_global_runaway"]["all_checks_pass"]
        and source["hybrid_config"]["use_SG"] is False
    ):
        raise RuntimeError("source is not the accepted tonic Z/M representative")

    ckpt = patch_spatial_checkpoint_support()
    import kick_probe
    import scripts.run_topic4_spatial_zm_ou_transition as runner
    from src.topic4_zm_fig5 import stratified_random_sites
    from src.topic4_zm_ictal_transition import load_round_config

    dt_ms = 0.1
    checkpoint_steps = {
        int(round(LOW_TIME_MS / dt_ms)),
        int(round(HIGH_TIME_MS / dt_ms)),
    }
    captured_states = {}
    captured_result = {}
    original_simulate = kick_probe.simulate_kick

    def observed_simulate(*args, **kwargs):
        kwargs["checkpoint_steps"] = sorted(checkpoint_steps)
        kwargs["checkpoint_sink"] = (
            lambda step, state: captured_states.setdefault(int(step), state))
        result = original_simulate(*args, **kwargs)
        captured_result["value"] = result
        return result

    kick_probe.simulate_kick = observed_simulate
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "run_topic4_spatial_zm_ou_transition.py",
        "--config", "config/topic4_data_driven_zm_ictal_transition_v1.json",
        "--seed", "1842",
        "--run-role", "confirmation",
        "--parameter-set-id", "tonic_b0_v2_static_panel_replay",
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
    if result is None or set(captured_states) != checkpoint_steps:
        raise RuntimeError("replay did not return the result and both checkpoints")

    comparison_keys = [
        "time_ms", "rate_E_hz", "rate_I_hz", "lfp_trace",
        "ou_time_ms", "ou_spatial_mean_rate_per_ms",
        "ou_spatial_sd_rate_per_ms", "slow_time_ms", "slow_q_mean",
        "slow_q_core_mean", "slow_q_surround_mean", "slow_m_mean",
        "slow_adaptation_current_mean", "slow_spike_count_E",
    ]
    comparisons = {}
    with np.load(SOURCE_NPZ, allow_pickle=False) as archived, np.load(
            REPLAY_BASE.with_suffix(".npz"), allow_pickle=False) as replay:
        for key in comparison_keys:
            same = bool(np.array_equal(archived[key], replay[key]))
            comparisons[key] = {
                "bit_identical": same,
                "max_abs_difference": (
                    0.0 if same else float(np.max(np.abs(
                        archived[key].astype(np.float64)
                        - replay[key].astype(np.float64)))))
            }
    if not all(row["bit_identical"] for row in comparisons.values()):
        raise RuntimeError("static-panel replay diverged from the locked trajectory")

    low_state = captured_states[int(round(LOW_TIME_MS / dt_ms))]
    high_state = captured_states[int(round(HIGH_TIME_MS / dt_ms))]
    low_sha = ckpt.save(low_state, LOW_CHECKPOINT)
    high_sha = ckpt.save(high_state, HIGH_CHECKPOINT)

    spikes = np.asarray(result["E_spk_bool"], bool)
    rates = np.asarray(result["rate_E"], float)
    onset_ms = float(source["scientific_onset_ms"])
    smooth = uniform_filter1d(
        rates, size=max(1, int(round(20.0 / dt_ms))), mode="nearest")
    peak_indices, properties = find_peaks(
        smooth, distance=int(round(50.0 / dt_ms)), prominence=10.0)
    eligible = np.asarray([
        index for index in peak_indices
        if 100.0 <= index * dt_ms
        and index * dt_ms + EVENT_HALF_WINDOW_MS < onset_ms
    ], dtype=int)
    if eligible.size == 0:
        raise RuntimeError("no complete low-state population event is available")
    event_peak = int(eligible[np.argmax(smooth[eligible])])
    event_peak_ms = event_peak * dt_ms
    event_start_ms = event_peak_ms - EVENT_HALF_WINDOW_MS
    event_stop_ms = event_peak_ms + EVENT_HALF_WINDOW_MS
    event_lo = int(round(event_start_ms / dt_ms))
    event_hi = int(round(event_stop_ms / dt_ms))
    event_spikes = spikes[event_lo:event_hi]
    active_event = np.any(event_spikes, axis=0)
    first_spike_ms = np.full(spikes.shape[1], np.nan, np.float32)
    first_spike_ms[active_event] = (
        event_lo + np.argmax(event_spikes[:, active_event], axis=0)
    ) * dt_ms

    round_config = load_round_config(
        str(ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json"))
    substrate, _, _ = build_context(round_config, source)
    contact_first_ms = np.full(len(substrate.contact_xy), np.nan, np.float32)
    for index, contact in enumerate(substrate.contact_xy):
        distance = np.linalg.norm(substrate.positions_e - contact, axis=1)
        local = (distance <= float(substrate.params.Rr)) & np.isfinite(first_spike_ms)
        if np.any(local):
            contact_first_ms[index] = np.median(first_spike_ms[local])

    energy_start_ms = onset_ms
    energy_stop_ms = onset_ms + EARLY_ENERGY_WINDOW_MS
    energy_lo = int(round(energy_start_ms / dt_ms))
    energy_hi = int(round(energy_stop_ms / dt_ms))
    early_rate_hz = (
        spikes[energy_lo:energy_hi].sum(axis=0)
        / (EARLY_ENERGY_WINDOW_MS * 1e-3))
    early_activity_energy = np.square(early_rate_hz).astype(np.float32)

    atomic_npz(
        EVENT_NPZ,
        positions_E=np.asarray(substrate.positions_e, np.float32),
        h_E=np.asarray(substrate.h_e, np.float32),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float32),
        contact_names=np.asarray(substrate.contact_names, dtype="U16"),
        shaft_ids=np.asarray(substrate.shaft_ids, dtype="U8"),
        sample_event_peak_ms=np.asarray(event_peak_ms, float),
        sample_event_start_ms=np.asarray(event_start_ms, float),
        sample_event_stop_ms=np.asarray(event_stop_ms, float),
        sample_first_spike_ms=first_spike_ms,
        sample_contact_first_spike_ms=contact_first_ms,
        early_activity_energy=early_activity_energy,
        early_activity_energy_start_ms=np.asarray(energy_start_ms, float),
        early_activity_energy_stop_ms=np.asarray(energy_stop_ms, float),
    )
    event_payload = {
        "status": "FIG5_STATIC_EVENT_AND_ENERGY_COMPLETE",
        "seed": 1842,
        "source": str(SOURCE_NPZ),
        "source_sha256": SOURCE_SHA256,
        "replay_comparisons": comparisons,
        "checkpoints": {
            "low": {"time_ms": LOW_TIME_MS, "path": str(LOW_CHECKPOINT),
                    "sha256": low_sha},
            "early_runaway": {"time_ms": HIGH_TIME_MS,
                               "path": str(HIGH_CHECKPOINT),
                               "sha256": high_sha},
        },
        "event_selection": (
            "highest 20-ms-smoothed population-rate local maximum with "
            "prominence >=10 Hz, peak >=100 ms, and complete fixed +/-50-ms "
            "window before the frozen 480-ms onset; image pixels unused"),
        "event_peak_ms": event_peak_ms,
        "event_window_ms": [event_start_ms, event_stop_ms],
        "n_event_active_E": int(np.sum(active_event)),
        "n_contacts_with_local_first_spike": int(np.sum(
            np.isfinite(contact_first_ms))),
        "event_order_measure": "exact first E-spike time in the event window",
        "early_activity_energy_window_ms": [energy_start_ms, energy_stop_ms],
        "early_activity_energy_measure": "squared per-neuron firing rate",
        "output_npz": str(EVENT_NPZ),
    }
    atomic_json(EVENT_JSON, event_payload)
    if args.replay_only:
        print(json.dumps({
            "status": event_payload["status"],
            "event_asset": str(EVENT_NPZ),
            "event_asset_sha256": sha256(EVENT_NPZ),
            "event_peak_ms": event_peak_ms,
            "replay_bit_identical": True,
        }, indent=2), flush=True)
        return 0

    sites = stratified_random_sites(
        n_side=N_SITE_SIDE, extent_mm=(0.0, 20.0),
        margin_mm=SITE_MARGIN_MM, seed=SITE_SEED)
    low_substrate, low_sham, low_rows, low_fields = probe_state(
        round_config, source, low_state, sites)
    high_substrate, high_sham, high_rows, high_fields = probe_state(
        round_config, source, high_state, sites)
    if not np.array_equal(low_substrate.positions_e, high_substrate.positions_e):
        raise RuntimeError("low/high probes use different neuron sheets")

    exact_continuations = {}
    with np.load(SOURCE_NPZ, allow_pickle=False) as archived:
        archived_rate = np.asarray(archived["rate_E_hz"], np.float32)
    for label_name, offset, sham in (
        ("low", LOW_TIME_MS, low_sham),
        ("early_runaway", HIGH_TIME_MS, high_sham),
    ):
        start = int(round(offset / dt_ms))
        observed = np.asarray(sham["rate_E"], np.float32)
        expected = archived_rate[start:start + len(observed)]
        exact_continuations[label_name] = bool(np.array_equal(observed, expected))
    if not all(exact_continuations.values()):
        raise RuntimeError("a sham continuation is not bit-identical to the replay")

    atomic_npz(
        ASSET_NPZ,
        positions_E=np.asarray(substrate.positions_e, np.float32),
        h_E=np.asarray(substrate.h_e, np.float32),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float32),
        contact_names=np.asarray(substrate.contact_names, dtype="U16"),
        shaft_ids=np.asarray(substrate.shaft_ids, dtype="U8"),
        sample_event_peak_ms=np.asarray(event_peak_ms, float),
        sample_event_start_ms=np.asarray(event_start_ms, float),
        sample_event_stop_ms=np.asarray(event_stop_ms, float),
        sample_first_spike_ms=first_spike_ms,
        sample_contact_first_spike_ms=contact_first_ms,
        early_activity_energy=early_activity_energy,
        early_activity_energy_start_ms=np.asarray(energy_start_ms, float),
        early_activity_energy_stop_ms=np.asarray(energy_stop_ms, float),
        site_xy_mm=np.asarray(sites, np.float32),
        low_response_early=low_fields,
        early_runaway_response_early=high_fields,
        low_response_early_mean=np.mean(low_fields, axis=0).astype(np.float32),
        early_runaway_response_early_mean=np.mean(
            high_fields, axis=0).astype(np.float32),
    )
    payload = {
        "status": "FIG5_SPATIAL_ZM_OU_TONIC_STATIC_ASSETS_COMPLETE",
        "seed": 1842,
        "source": str(SOURCE_NPZ),
        "source_sha256": SOURCE_SHA256,
        "replay": str(REPLAY_BASE.with_suffix(".npz")),
        "replay_sha256": sha256(REPLAY_BASE.with_suffix(".npz")),
        "replay_comparisons": comparisons,
        "checkpoint_contract": {
            "support": (
                "read-only extension captures q grid, q-driver rate fields, "
                "partial 1-ms accumulators, M spatial drive, and OU state"),
            "low": {"time_ms": LOW_TIME_MS, "path": str(LOW_CHECKPOINT),
                    "sha256": low_sha},
            "early_runaway": {"time_ms": HIGH_TIME_MS,
                               "path": str(HIGH_CHECKPOINT),
                               "sha256": high_sha},
            "sham_continuation_rate_bit_identical": exact_continuations,
        },
        "panel_C": {
            "event_selection": (
                "highest 20-ms-smoothed population-rate local maximum with "
                "prominence >=10 Hz, peak >=100 ms, and complete fixed +/-50-ms "
                "window before the frozen 480-ms onset; image pixels unused"),
            "event_peak_ms": event_peak_ms,
            "event_window_ms": [event_start_ms, event_stop_ms],
            "n_event_active_E": int(np.sum(active_event)),
            "n_contacts_with_local_first_spike": int(np.sum(
                np.isfinite(contact_first_ms))),
            "event_order_measure": "exact first E-spike time in the event window",
            "early_activity_energy_window_ms": [energy_start_ms, energy_stop_ms],
            "early_activity_energy_measure": "squared per-neuron firing rate",
        },
        "panel_D": {
            "state_times_ms": {"low_activity": LOW_TIME_MS,
                               "early_runaway": HIGH_TIME_MS},
            "site_contract": {
                "kind": "one uniform random point per square stratum",
                "n_side": N_SITE_SIDE, "n_total": len(sites),
                "seed": SITE_SEED, "sheet_extent_mm": [0.0, 20.0],
                "edge_margin_mm": SITE_MARGIN_MM,
            },
            "dose_cells": PROBE_DOSE_CELLS,
            "dose_origin": (
                "frozen 16-cell weak dose inherited from the accepted Fig5 "
                "low-state assay; not selected from this response map"),
            "response": "paired probe-minus-sham descendant spikes, 0-50 ms",
            "aggregation": "equal-weight mean over 16 paired stratified-random sites",
            "low_n_e1_evaluable": int(sum(row["e1_evaluable"] for row in low_rows)),
            "early_runaway_n_e1_evaluable": int(sum(
                row["e1_evaluable"] for row in high_rows)),
            "low_n_probe_attributable_event": int(sum(
                row["probe_attributable_event_200ms"] for row in low_rows)),
            "early_runaway_n_probe_attributable_event": int(sum(
                row["probe_attributable_event_200ms"] for row in high_rows)),
            "low_rows": low_rows,
            "early_runaway_rows": high_rows,
        },
        "output_npz": str(ASSET_NPZ),
        "claim_boundary": (
            "single frozen network and one inherited weak dose; state-contrast "
            "mechanistic assay, not a population estimate across network seeds"),
    }
    atomic_json(ASSET_JSON, payload)
    print(json.dumps({
        "status": payload["status"],
        "asset": str(ASSET_NPZ),
        "asset_sha256": sha256(ASSET_NPZ),
        "event_peak_ms": event_peak_ms,
        "low_evaluable": payload["panel_D"]["low_n_e1_evaluable"],
        "high_evaluable": payload["panel_D"]["early_runaway_n_e1_evaluable"],
        "continuations_exact": exact_continuations,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
