#!/usr/bin/env python3
"""Export the accepted E1146 M3 q_I build-up-to-runaway trajectory.

This runner reuses the exact continuous M3A-v2.1 protocol used by
``plot_fig_m3a_v2_1_qigk_runaway_transition_gif.py``.  It adds no new model
dynamics; it only materializes the continuous virtual-SEEG trace and two
contact-space summaries plus their neuron-resolved substrates needed by the
early-recruitment figure:

* a single pre-runaway event from the same endpoint source as the last pulse
  before runaway, with 1..N contact order derived from its 30--80 Hz
  burst-envelope peak latency;
* mean excess virtual-LFP energy from runaway onset until the next scheduled
  external pulse (capped at 100 ms by the accepted protocol).

The neuron layer is taken directly from ``E_spk_bool``: single-event first-spike
latency for the interictal field and per-neuron firing rate in the early-runaway
window. Multi-event medians remain exported for audit but are not plotted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.paper_figures.plot_fig_m3a_v2_1_qigk_runaway_transition_gif as M3  # noqa: E402


DEFAULT_OUT = ROOT / "results/topic4_sef_hfo/early_recruitment_readout"
DEFAULT_GEOMETRY = (
    Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/")
    / "field_swap_subject_snn/figdata_epilepsiae_1146_twoend_equal_tsrc_s3.npz"
)
RUNAWAY_WINDOW_CAP_MS = 100.0
PULSE_READOUT_MS = 85.0
BURST_BAND_HZ = (30.0, 80.0)
LOCAL_RECRUIT_RADIUS_MM = 1.5
LOCAL_RECRUIT_FRACTION = 0.05


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _runaway_window(runaway_start_ms, pulse_t0_ms, trace_end_ms, cap_ms=RUNAWAY_WINDOW_CAP_MS):
    onset = float(runaway_start_ms)
    later_pulses = np.asarray(pulse_t0_ms, float)
    later_pulses = later_pulses[later_pulses > onset + 1e-9]
    next_pulse = float(np.min(later_pulses)) if later_pulses.size else float(trace_end_ms)
    end = min(onset + float(cap_ms), next_pulse, float(trace_end_ms))
    if end <= onset:
        raise ValueError(f"empty runaway energy window: onset={onset}, end={end}")
    return onset, end, next_pulse


def _normalized_lfp(res, runaway_start_ms):
    times = np.asarray(res["times"], float)
    lfp = np.abs(np.asarray(res["lfp_trace"], float))
    pre = times < float(runaway_start_ms)
    if pre.sum() < 2:
        raise ValueError("fewer than two pre-runaway LFP samples")
    base = np.median(lfp[pre], axis=0)
    pre_scale = np.maximum(np.percentile(lfp[pre], 99, axis=0) - base, 1e-9)
    full_scale = np.maximum(np.percentile(lfp, 99, axis=0) - base, 1e-9)
    scale = np.maximum(pre_scale, 0.35 * full_scale)
    z = (lfp - base[None, :]) / scale[None, :]
    return lfp, base, scale, z


def _burst_readout(res, runaway_start_ms):
    """Signed 30--80 Hz trace and its positive-excess analytic envelope."""
    times = np.asarray(res["times"], float)
    lfp = np.asarray(res["lfp_trace"], float)
    dt_ms = float(np.median(np.diff(times)))
    fs_hz = 1000.0 / dt_ms
    sos = butter(4, BURST_BAND_HZ, btype="bandpass", fs=fs_hz, output="sos")
    burst = sosfiltfilt(sos, lfp, axis=0)
    envelope = np.abs(hilbert(burst, axis=0))
    pre = times < float(runaway_start_ms)
    baseline = np.median(envelope[pre], axis=0)
    excess = np.maximum(envelope - baseline[None, :], 0.0)
    scale = np.percentile(excess[pre], 95.0, axis=0)
    finite_positive = scale[np.isfinite(scale) & (scale > 1e-12)]
    if finite_positive.size == 0:
        raise ValueError("pre-runaway burst envelope is constant")
    scale = np.maximum(scale, 0.15 * float(np.median(finite_positive)))
    return burst, envelope, baseline, scale, excess / scale[None, :]


def _ordinal_contact_rank(latency):
    """Deterministic 1..N recruitment order over finite contact latencies."""
    latency = np.asarray(latency, float)
    rank = np.full(latency.shape, np.nan, float)
    valid_idx = np.flatnonzero(np.isfinite(latency))
    order = valid_idx[np.argsort(latency[valid_idx], kind="mergesort")]
    rank[order] = np.arange(1, order.size + 1, dtype=float)
    return rank


def _qualifying_pulse_rows(metrics, runaway_start_ms):
    rows = []
    for row in metrics["pulse_rows"]:
        qualifies = (
            bool(row["before_runaway"])
            and float(row["t0"]) < float(runaway_start_ms) - 20.0
            and 5.0 <= float(row["peak_hz"]) <= 120.0
            and float(row["active_frac"]) >= 0.02
        )
        rows.append({**row, "qualifies_local": bool(qualifies)})
    return rows


def _template_latency(times, z_lfp, pulse_rows, source):
    """Median peak latency for the pre-runaway events matching ``source``."""
    per_event = []
    used_t0 = []
    times = np.asarray(times, float)
    for row in pulse_rows:
        if not row["qualifies_local"] or row["source"] != source:
            continue
        t0 = float(row["t0"])
        mask = (times >= t0) & (times <= t0 + PULSE_READOUT_MS)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue
        peak_idx = idx[np.argmax(z_lfp[mask], axis=0)]
        peak_amp = z_lfp[peak_idx, np.arange(z_lfp.shape[1])]
        latency = times[peak_idx] - t0
        latency[peak_amp < 0.06] = np.nan
        per_event.append(latency)
        used_t0.append(t0)
    if not per_event:
        raise ValueError(f"no qualifying pre-runaway {source} local events")
    stack = np.asarray(per_event, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_latency = np.nanmedian(stack, axis=0)
    support = np.sum(np.isfinite(stack), axis=0)
    # The static propagation field keeps only contacts readable in every
    # reference event, so a one-off current-volume peak cannot define its edge.
    min_support = int(stack.shape[0])
    median_latency[support < min_support] = np.nan
    return median_latency, stack, np.asarray(used_t0, float), support


def _neuron_template_latency(times, E_spk_bool, pulse_rows, source):
    """Median first-spike latency for neurons active in every reference event."""
    times = np.asarray(times, float)
    spikes = np.asarray(E_spk_bool, bool)
    if spikes.ndim != 2 or spikes.shape[0] != times.size:
        raise ValueError("E_spk_bool must be time x E-neuron and align with times")
    per_event = []
    used_t0 = []
    for row in pulse_rows:
        if not row["qualifies_local"] or row["source"] != source:
            continue
        t0 = float(row["t0"])
        mask = (times >= t0) & (times <= t0 + PULSE_READOUT_MS)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue
        event_spikes = spikes[idx]
        fired = np.any(event_spikes, axis=0)
        latency = np.full(spikes.shape[1], np.nan, float)
        if np.any(fired):
            first_local = np.argmax(event_spikes[:, fired], axis=0)
            latency[fired] = times[idx[first_local]] - t0
        per_event.append(latency)
        used_t0.append(t0)
    if not per_event:
        raise ValueError(f"no qualifying neuron-resolved {source} local events")
    stack = np.asarray(per_event, float)
    support = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_latency = np.nanmedian(stack, axis=0)
    median_latency[support < stack.shape[0]] = np.nan
    return median_latency, stack, np.asarray(used_t0, float), support


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--geometry-npz", type=Path, default=DEFAULT_GEOMETRY)
    ap.add_argument("--runaway-window-ms", type=float, default=RUNAWAY_WINDOW_CAP_MS)
    ap.add_argument("--k-q", type=float, default=0.10)
    ap.add_argument("--ee-ar", type=float, default=None)
    ap.add_argument("--pulse-first-source", choices=["tempA", "tempB"], default="tempA")
    ap.add_argument(
        "--core-radius-scale",
        type=float,
        default=1.0,
        help="multiply the accepted subject-specific low-threshold core radius",
    )
    ap.add_argument(
        "--core-transverse-scale",
        type=float,
        default=None,
        help="independently multiply the transverse radius; default inherits --core-radius-scale",
    )
    args = ap.parse_args()
    if not args.geometry_npz.exists():
        raise FileNotFoundError(f"accepted E1146 geometry missing: {args.geometry_npz}")

    os.chdir(ROOT)
    # The accepted producer resolves this constant relative to its checkout.
    # Override it with the canonical artifact path so results-light worktrees
    # reuse the exact accepted geometry rather than guessing a replacement.
    M3.SUBJECT1146_FIGDATA = args.geometry_npz.resolve()
    cfg = M3.ProtocolConfig(
        k_q=args.k_q,
        q_min=0.05,
        kick_boost=5.0,
        r_kick=0.6,
        T=1500.0,
        layout="subject1146",
        core_radius_scale=args.core_radius_scale,
        core_transverse_scale=args.core_transverse_scale,
        ee_ar_override=args.ee_ar,
        pulse_first_source=args.pulse_first_source,
        fig_name="fig_m3a_v2_1_qigk_runaway_transition_epilepsiae_1146",
    )
    S, res, metrics = M3.run_one(cfg, record_gif=True)
    onset = metrics.get("runaway_start_ms")
    if onset is None:
        raise RuntimeError("accepted M3 protocol did not produce runaway")

    pulse_rows = _qualifying_pulse_rows(metrics, onset)
    pulse_t0 = np.asarray([float(row["t0"]) for row in pulse_rows], float)
    start, end, next_pulse = _runaway_window(
        onset, pulse_t0, float(res["times"][-1]), args.runaway_window_ms)
    lfp_abs, lfp_base, lfp_scale, z_lfp = _normalized_lfp(res, onset)
    burst_trace, burst_envelope, burst_base, burst_scale, z_burst = _burst_readout(
        res, onset
    )
    energy_mask = (res["times"] >= start) & (res["times"] < end)
    if energy_mask.sum() < 2:
        raise RuntimeError("runaway energy window has fewer than two samples")
    positive_excess = np.maximum(lfp_abs[energy_mask] - lfp_base[None, :], 0.0)
    runaway_energy = np.mean(positive_excess ** 2, axis=0)
    energy_den = max(float(np.max(runaway_energy)), 1e-15)
    runaway_energy_norm = runaway_energy / energy_den

    pre_pulses = [row for row in pulse_rows if float(row["t0"]) < float(onset)]
    transition_source = str(max(pre_pulses, key=lambda row: float(row["t0"]))["source"])
    template_latency, latency_events, latency_t0, latency_support = _template_latency(
        res["times"], z_burst, pulse_rows, transition_source)
    display_event_idx = int(len(latency_t0) - 1)
    display_event_t0 = float(latency_t0[display_event_idx])
    display_latency = np.asarray(latency_events[display_event_idx], float)
    display_rank = _ordinal_contact_rank(display_latency)
    layout_scale = float(S["layout"]["scale"])
    contacts_reference_mm = np.asarray(res["contacts"], float) / layout_scale
    neuron_positions_reference_mm = np.asarray(S["posE"], float) / layout_scale
    template_neuron_latency, neuron_latency_events, neuron_latency_t0, neuron_latency_support = (
        _neuron_template_latency(
            res["times"], res["E_spk_bool"], pulse_rows, transition_source
        )
    )
    if not np.array_equal(neuron_latency_t0, latency_t0):
        raise RuntimeError("contact and neuron interictal reference events diverged")
    display_neuron_latency = np.asarray(
        neuron_latency_events[display_event_idx], float
    )
    neuron_runaway_spike_count = np.sum(
        np.asarray(res["E_spk_bool"], bool)[energy_mask], axis=0, dtype=np.int32
    )
    neuron_runaway_rate_hz = (
        neuron_runaway_spike_count.astype(float) / ((end - start) / 1000.0)
    )
    q_frames = np.asarray(res["q_frames"], float)
    q_frame_steps = np.asarray(res["q_frame_steps"], int)
    q_frame_times = np.asarray(res["times"], float)[q_frame_steps]
    contact_names = [str(name) for name in res["names"]]
    event_contact_mask = np.isfinite(display_rank)
    scl_mask = np.asarray([name.startswith("SCL") for name in contact_names], bool)
    icl_mask = np.asarray([name.startswith("ICL") for name in contact_names], bool)
    source_reference_mm = (
        np.asarray(M3._source_xy(S, transition_source), float) / layout_scale
    )
    source_distance_mm = np.linalg.norm(
        contacts_reference_mm - source_reference_mm[None, :], axis=1
    )
    icl_event = event_contact_mask & icl_mask
    source_distance_rank_rho = (
        float(spearmanr(source_distance_mm[icl_event], display_rank[icl_event]).statistic)
        if icl_event.sum() >= 3
        else None
    )
    effective_core_parallel_sim_mm, effective_core_transverse_sim_mm = (
        M3._effective_core_radii(S, cfg)
    )
    effective_core_radius_sim_mm = float(effective_core_parallel_sim_mm)
    effective_core_radius_reference_mm = effective_core_radius_sim_mm / layout_scale
    effective_core_transverse_reference_mm = (
        float(effective_core_transverse_sim_mm) / layout_scale
    )
    display_neuron_active = np.isfinite(display_neuron_latency)
    local_total = np.zeros(len(contact_names), int)
    local_active = np.zeros(len(contact_names), int)
    for ci, contact in enumerate(contacts_reference_mm):
        local = (
            np.linalg.norm(neuron_positions_reference_mm - contact[None, :], axis=1)
            <= LOCAL_RECRUIT_RADIUS_MM
        )
        local_total[ci] = int(local.sum())
        local_active[ci] = int(np.sum(local & display_neuron_active))
    local_fraction = local_active / np.maximum(local_total, 1)
    locally_recruited = local_fraction >= LOCAL_RECRUIT_FRACTION

    args.out.mkdir(parents=True, exist_ok=True)
    npz_path = args.out / "m3_runaway_readout.npz"
    np.savez_compressed(
        npz_path,
        times_ms=np.asarray(res["times"], float),
        lfp_trace=np.asarray(res["lfp_trace"], float),
        lfp_abs=lfp_abs,
        lfp_baseline=lfp_base,
        lfp_scale=lfp_scale,
        lfp_normalized=z_lfp,
        lfp_burst_30_80_hz=np.asarray(burst_trace, float),
        lfp_burst_envelope=np.asarray(burst_envelope, float),
        lfp_burst_envelope_baseline=np.asarray(burst_base, float),
        lfp_burst_envelope_scale=np.asarray(burst_scale, float),
        rate_E_hz=np.asarray(res["rate_E"], float),
        contact_names=np.asarray(contact_names, object),
        contacts_reference_mm=contacts_reference_mm,
        neuron_positions_reference_mm=neuron_positions_reference_mm,
        runaway_start_ms=np.asarray(float(onset)),
        runaway_energy_start_ms=np.asarray(start),
        runaway_energy_end_ms=np.asarray(end),
        runaway_energy=np.asarray(runaway_energy, float),
        runaway_energy_norm=np.asarray(runaway_energy_norm, float),
        interictal_reference_source=np.asarray(transition_source),
        interictal_reference_latency_ms=np.asarray(display_latency, float),
        interictal_event_contact_rank=np.asarray(display_rank, float),
        interictal_display_event_t0_ms=np.asarray(display_event_t0),
        interictal_display_event_t1_ms=np.asarray(display_event_t0 + PULSE_READOUT_MS),
        interictal_event_local_active_neuron_count=np.asarray(local_active, int),
        interictal_event_local_neuron_count=np.asarray(local_total, int),
        interictal_event_local_active_fraction=np.asarray(local_fraction, float),
        interictal_event_locally_recruited=np.asarray(locally_recruited, bool),
        interictal_template_median_latency_ms=np.asarray(template_latency, float),
        interictal_reference_latency_per_event_ms=np.asarray(latency_events, float),
        interictal_reference_event_t0_ms=latency_t0,
        interictal_reference_contact_support=np.asarray(latency_support, int),
        neuron_interictal_reference_latency_ms=np.asarray(display_neuron_latency, float),
        neuron_interictal_template_median_latency_ms=np.asarray(
            template_neuron_latency, float
        ),
        neuron_interictal_reference_latency_per_event_ms=np.asarray(
            neuron_latency_events, float
        ),
        neuron_interictal_reference_support=np.asarray(neuron_latency_support, int),
        neuron_runaway_spike_count=np.asarray(neuron_runaway_spike_count, np.int32),
        neuron_runaway_rate_hz=np.asarray(neuron_runaway_rate_hz, float),
        pulse_t0_ms=pulse_t0,
        pulse_t1_ms=np.asarray([float(row["t0"]) + cfg.pulse_duration for row in pulse_rows]),
        pulse_source=np.asarray([row["source"] for row in pulse_rows], object),
        pulse_qualifies_local=np.asarray([row["qualifies_local"] for row in pulse_rows], bool),
        q_frame_times_ms=q_frame_times,
        q_mean=np.mean(q_frames, axis=tuple(range(1, q_frames.ndim))),
        q_min=np.min(q_frames, axis=tuple(range(1, q_frames.ndim))),
    )

    summary = {
        "schema_id": "topic4_m3_runaway_readout_v2",
        "status": "M3A-v2.1 core-size variant with single-event onset-locked readout",
        "producer": "scripts/run_topic4_m3_runaway_readout.py",
        "upstream_producer": "scripts/paper_figures/plot_fig_m3a_v2_1_qigk_runaway_transition_gif.py",
        "geometry_npz": str(args.geometry_npz.resolve()),
        "config": asdict(cfg),
        "effective_core_radius": {
            "parallel_simulation_mm": effective_core_radius_sim_mm,
            "transverse_simulation_mm": float(effective_core_transverse_sim_mm),
            "parallel_reference_E1146_plane_mm": effective_core_radius_reference_mm,
            "transverse_reference_E1146_plane_mm": effective_core_transverse_reference_mm,
            "parallel_scale_vs_accepted_subject_core": float(args.core_radius_scale),
            "transverse_scale_vs_accepted_subject_core": float(
                args.core_radius_scale
                if args.core_transverse_scale is None
                else args.core_transverse_scale
            ),
        },
        "metrics": _jsonable(metrics),
        "runaway_onset": {
            "time_ms": float(onset),
            "operational_definition": "first 100-ms interval with >=80% samples above 120-Hz 20-ms-smoothed E rate",
            "separatrix_boundary": "q_I-depletion-driven transition; no independently solved analytic q_I* in this M3 artifact",
        },
        "runaway_energy_window": {
            "start_ms": start,
            "end_ms": end,
            "duration_ms": end - start,
            "next_external_pulse_ms": next_pulse,
            "rule": "from operational runaway onset, capped at 100 ms and ending before the next scheduled pulse",
            "signal": "mean squared positive excess absolute virtual-LFP relative to pre-runaway median",
        },
        "interictal_reference": {
            "source": transition_source,
            "selection_rule": (
                "last qualifying local event from the same endpoint source as the last "
                "scheduled pulse before runaway onset"
            ),
            "event_t0_ms": latency_t0.tolist(),
            "n_events": int(len(latency_t0)),
            "display_event_t0_ms": display_event_t0,
            "display_event_t1_ms": display_event_t0 + PULSE_READOUT_MS,
            "quantity": (
                "1..N contact recruitment rank from the 30-80-Hz burst-envelope peak "
                "latency in one displayed event"
            ),
            "template_quantity_audit": (
                "median peak latency across all qualifying same-source events, exported but "
                "not displayed"
            ),
            "qualification": (
                "event before onset-20 ms; 5<=peak_hz<=120; active_frac>=0.02; "
                "displayed contact finite in the selected single event"
            ),
            "neuron_grain": (
                "first-spike latency for E neurons firing in the identical displayed event"
            ),
            "display_contact_n": int(event_contact_mask.sum()),
            "display_SCL_contact_n": int(np.sum(event_contact_mask & scl_mask)),
            "display_SCL_contacts": [
                name for name, keep in zip(contact_names, event_contact_mask & scl_mask) if keep
            ],
            "local_recruitment_rule": (
                f">={LOCAL_RECRUIT_FRACTION:.0%} of E neurons within "
                f"{LOCAL_RECRUIT_RADIUS_MM:g} mm of the contact fire in the displayed event"
            ),
            "locally_recruited_SCL_contact_n": int(
                np.sum(locally_recruited & scl_mask)
            ),
            "locally_recruited_SCL_contacts": [
                name for name, keep in zip(contact_names, locally_recruited & scl_mask) if keep
            ],
            "SCL_local_active_fraction": {
                name: float(frac)
                for name, frac in zip(contact_names, local_fraction)
                if name.startswith("SCL")
            },
            "ICL_source_distance_vs_rank_spearman": source_distance_rank_rho,
        },
        "neuron_layer": {
            "n_E": int(neuron_positions_reference_mm.shape[0]),
            "interictal_display_event_active_n": int(
                np.isfinite(display_neuron_latency).sum()
            ),
            "interictal_template_repeated_support_n": int(
                np.isfinite(template_neuron_latency).sum()
            ),
            "runaway_active_n": int(np.sum(neuron_runaway_spike_count > 0)),
            "runaway_quantity": "per-neuron firing rate in the identical onset-locked energy window",
        },
        "outputs": {"npz": str(npz_path)},
        "claim_boundary": [
            "continuous M3 visual diagnostic, not a seizure/recovery mechanism proof",
            "runaway onset is operational, not an independently localized analytic separatrix",
            "virtual-LFP energy is a model readout proxy, not clinical broadband SEEG power",
        ],
    }
    json_path = args.out / "m3_runaway_readout.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {npz_path}")
    print(f"wrote {json_path}")
    print(json.dumps({
        "runaway_start_ms": onset,
        "energy_window_ms": [start, end],
        "next_external_pulse_ms": next_pulse,
        "interictal_reference_source": transition_source,
        "interictal_reference_events": latency_t0.tolist(),
        "display_event_t0_ms": display_event_t0,
        "display_contact_n": int(event_contact_mask.sum()),
        "display_SCL_contacts": [
            name for name, keep in zip(contact_names, event_contact_mask & scl_mask) if keep
        ],
        "locally_recruited_SCL_contacts": [
            name for name, keep in zip(contact_names, locally_recruited & scl_mask) if keep
        ],
        "effective_core_radius_reference_mm": effective_core_radius_reference_mm,
        "effective_core_transverse_reference_mm": effective_core_transverse_reference_mm,
        "wall_s": round(float(res["wall_s"]), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
