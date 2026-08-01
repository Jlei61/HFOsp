#!/usr/bin/env python3
"""Online multi-objective analysis for the seed-1 joint lifecycle sprint."""
from __future__ import annotations

import csv
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_topic4_zm_fast_lifecycle_development as A  # noqa: E402


IN_ROOT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/seed1"
OUT_ROOT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
BIN_MS = 25.0
RETURN_REFERENCE = (
    ROOT / "results/topic4_sef_hfo/zm_branch_decision/anchors/seed1"
)


def contact_rms_from_baseline(raw, fs_hz, baseline_bins, *, bin_ms=BIN_MS):
    x = np.asarray(raw, float)
    if x.ndim == 1:
        x = x[:, None]
    bs = max(1, int(round(float(bin_ms) * float(fs_hz) / 1000.0)))
    nb = x.shape[0] // bs
    x = x[:nb * bs]
    baseline_bins = np.asarray(baseline_bins, bool)[:nb]
    if baseline_bins.sum() < 4:
        return None, "insufficient_event_free_baseline"
    samples = x.reshape(nb, bs, x.shape[1])
    baseline_mean = np.mean(samples[baseline_bins], axis=(0, 1))
    rms = np.sqrt(np.mean((samples - baseline_mean[None, None, :]) ** 2, axis=1))
    return rms, "ok"


def event_free_baseline_bins(core_rate, *, max_ms=1200.0, bin_ms=BIN_MS):
    core = np.asarray(core_rate, float)
    n = min(core.size, int(round(float(max_ms) / float(bin_ms))))
    if n < 4:
        return np.zeros(core.size, bool)
    x = core[:n]
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    threshold = max(10.0, med + 2.5 * mad)
    mask = np.zeros(core.size, bool)
    mask[:n] = x <= threshold
    return mask


def detect_episode(core_rate, baseline_bins, *, bin_ms=BIN_MS):
    core = np.asarray(core_rate, float)
    base = core[np.asarray(baseline_bins, bool)]
    if base.size < 4:
        return {"status": "insufficient_baseline", "onset_bin": None, "offset_bin": None}
    med = float(np.median(base))
    mad = float(1.4826 * np.median(np.abs(base - med)))
    onset_threshold = max(50.0, med + 3.0 * mad)
    recovery_threshold = max(25.0, med + 1.5 * mad)
    smooth = A._moving_mean(core, int(round(500.0 / bin_ms)))
    onset = A._first_sustained(
        smooth >= onset_threshold, int(round(250.0 / bin_ms)),
        start=int(round(500.0 / bin_ms)),
    )
    offset = None
    transient_offsets = []
    rapid_reentries = []
    status = "no_onset"
    if onset is not None:
        status = "onset_persistent"
        need_low = int(round(500.0 / bin_ms))
        need_high = int(round(250.0 / bin_ms))
        min_tail = int(round(1000.0 / bin_ms))
        rapid_horizon = int(round(2000.0 / bin_ms))
        search = onset + int(round(1000.0 / bin_ms))
        while search < core.size:
            candidate = A._first_sustained(
                smooth <= recovery_threshold, need_low, start=search,
            )
            if candidate is None:
                break
            reentry = A._first_sustained(
                smooth >= onset_threshold, need_high,
                start=candidate + need_low,
            )
            if reentry is not None and reentry <= candidate + rapid_horizon:
                transient_offsets.append(int(candidate))
                rapid_reentries.append(int(reentry))
                search = reentry + need_high
                continue
            if core.size - candidate < min_tail:
                status = "offset_unconfirmed_at_trace_end"
                break
            offset = int(candidate)
            status = "onset_durable_offset"
            break
    return {
        "status": status,
        "baseline_core_median_hz": med,
        "baseline_core_mad_hz": mad,
        "onset_threshold_hz": onset_threshold,
        "recovery_threshold_hz": recovery_threshold,
        "onset_bin": onset,
        "offset_bin": offset,
        "onset_ms": None if onset is None else onset * bin_ms,
        "offset_ms": None if offset is None else offset * bin_ms,
        "transient_offset_bins": transient_offsets,
        "rapid_reentry_bins": rapid_reentries,
        "offset_confirmation_rule": "500ms low, no 250ms high re-entry within 2s, >=1s observed tail",
    }


def returning_event_windows(
    core_rate, *, threshold_hz, lo_bin=0, hi_bin=None, smooth_ms=0.0,
):
    """Segment discrete events above a threshold frozen outside the candidate.

    The caller must supply the threshold from the seed-1 pre-escalation anchor.
    A post-offset fragment is therefore not allowed to define its own amplitude
    and promote itself to a returning event.
    """
    core = np.asarray(core_rate, float)
    hi = core.size if hi_bin is None else min(core.size, int(hi_bin))
    lo = max(0, int(lo_bin))
    work = core
    if float(smooth_ms) > 0:
        work = A._moving_mean(core, max(1, int(round(float(smooth_ms) / BIN_MS))))
    on = work >= float(threshold_hz)
    windows = []
    start = None
    for index in range(lo, hi):
        if on[index] and start is None:
            start = index
        elif not on[index] and start is not None:
            windows.append((start, index))
            start = None
    if start is not None:
        windows.append((start, hi))
    return windows


def returning_event_features(
    core_rate, surround_rate, active_fraction, kymo_axial, contact_rms,
    windows, *, bin_ms=BIN_MS,
):
    """Extract morphology and geometry for candidate returning events."""
    core = np.asarray(core_rate, float)
    surround = np.asarray(surround_rate, float)
    active = np.asarray(active_fraction, float)
    kymo = np.asarray(kymo_axial, float)
    rms = None if contact_rms is None else np.asarray(contact_rms, float)
    feats = []
    axis = np.arange(kymo.shape[0], dtype=float)
    for lo, hi in windows:
        lo, hi = int(lo), int(hi)
        if hi <= lo or lo >= core.size:
            continue
        hi = min(hi, core.size)
        mode = np.mean(kymo[:, lo:hi], axis=1) if kymo.size else np.zeros(1)
        centroids = []
        for column in kymo[:, lo:hi].T:
            total = float(np.sum(column))
            centroids.append(float(axis @ column / total) if total > 0 else np.nan)
        valid = np.asarray(centroids, float)
        valid = valid[np.isfinite(valid)]
        direction = 0
        if valid.size >= 2:
            delta = float(np.mean(valid[-max(1, valid.size // 2):])
                          - np.mean(valid[:max(1, valid.size // 2)]))
            direction = int(np.sign(delta))
        contact_order = []
        if rms is not None and rms.ndim == 2 and lo < rms.shape[0]:
            local = rms[lo:min(hi, rms.shape[0])]
            if local.size:
                peak_bins = np.argmax(local, axis=0)
                contact_order = np.argsort(
                    np.argsort(peak_bins, kind="stable"), kind="stable"
                ).astype(int).tolist()
        feats.append({
            "onset_ms": float(lo * bin_ms),
            "duration_ms": float((hi - lo) * bin_ms),
            "peak_core_hz": float(np.max(core[lo:hi])),
            "active_fraction": float(np.mean(active[lo:hi])),
            "core_surround_ratio": float(
                np.mean(core[lo:hi]) / max(np.mean(surround[lo:hi]), 1e-9)
            ),
            "spatial_mode": np.asarray(mode, float).tolist(),
            "axial_direction": direction,
            "contact_order": contact_order,
        })
    return feats


def _corr(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape or left.size < 2:
        return None
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def match_returning_events(reference, post, *, min_post=3, tol_frac=0.6):
    """Separate one plausible returning event from distribution-level recovery.

    This is a development diagnostic.  `single_event_candidate` is the sprint's
    first-return target; `distribution_recovered` is deliberately stronger and
    requires at least three post-offset events with matched cadence and geometry.
    """
    reference, post = list(reference), list(post)
    if not reference:
        return {
            "status": "reference_unavailable", "n_reference": 0,
            "n_post": len(post), "single_event_candidate": False,
            "distribution_recovered": False,
        }
    keys = ("duration_ms", "peak_core_hz", "active_fraction", "core_surround_ratio")
    ref_med = {key: float(np.median([row[key] for row in reference])) for key in keys}
    ref_mode = np.mean([np.asarray(row["spatial_mode"], float) for row in reference], axis=0)
    ref_dirs = {int(row["axial_direction"]) for row in reference if int(row["axial_direction"]) != 0}
    ref_orders = [row["contact_order"] for row in reference if row.get("contact_order")]

    individual = []
    for row in post:
        ratios = {
            key: float(row[key]) / max(ref_med[key], 1e-12) for key in keys
        }
        morphology_ok = all(1.0 - tol_frac <= value <= 1.0 + tol_frac
                            for value in ratios.values())
        mode = np.asarray(row["spatial_mode"], float)
        mode_cos = float(ref_mode @ mode / (
            np.linalg.norm(ref_mode) * np.linalg.norm(mode) + 1e-12
        ))
        contact_corrs = [
            value for value in (_corr(order, row.get("contact_order", []))
                                for order in ref_orders) if value is not None
        ]
        best_contact_corr = max(contact_corrs) if contact_corrs else None
        direction_ok = (not ref_dirs or int(row["axial_direction"]) in ref_dirs)
        geometry_ok = (
            mode_cos >= 0.5 and direction_ok
            and (best_contact_corr is None or best_contact_corr >= 0.3)
        )
        individual.append({
            "onset_ms": row["onset_ms"], "ratios": ratios,
            "spatial_mode_cosine": mode_cos,
            "best_contact_order_correlation": best_contact_corr,
            "direction_ok": bool(direction_ok),
            "matches": bool(morphology_ok and geometry_ok),
        })
    single = any(row["matches"] for row in individual)
    match_fraction = (
        float(np.mean([row["matches"] for row in individual])) if individual else 0.0
    )

    per_metric = {}
    for key in keys:
        post_med = float(np.median([row[key] for row in post])) if post else None
        ratio = None if post_med is None else post_med / max(ref_med[key], 1e-12)
        per_metric[key] = {
            "reference_median": ref_med[key], "post_median": post_med,
            "ratio": ratio,
            "ok": bool(ratio is not None and 1.0 - tol_frac <= ratio <= 1.0 + tol_frac),
        }
    ref_iei = np.diff([row["onset_ms"] for row in reference])
    post_iei = np.diff([row["onset_ms"] for row in post])
    iei_ratio = None
    if ref_iei.size and post_iei.size:
        iei_ratio = float(np.median(post_iei) / max(np.median(ref_iei), 1e-12))
    per_metric["iei"] = {
        "reference_median_ms": float(np.median(ref_iei)) if ref_iei.size else None,
        "post_median_ms": float(np.median(post_iei)) if post_iei.size else None,
        "ratio": iei_ratio,
        "ok": bool(iei_ratio is not None and 1.0 - tol_frac <= iei_ratio <= 1.0 + tol_frac),
    }
    recovered = (
        len(post) >= int(min_post) and match_fraction >= 0.5
        and all(row["ok"] for row in per_metric.values())
    )
    return {
        "status": "ok", "n_reference": len(reference), "n_post": len(post),
        "single_event_candidate": bool(single),
        "distribution_recovered": bool(recovered),
        "matched_event_fraction": match_fraction,
        "per_metric": per_metric, "individual": individual,
        "claim_boundary": (
            "single event is a development candidate; distribution recovery requires >=3 "
            "events and is not a multi-seed lifecycle claim"
        ),
    }


@lru_cache(maxsize=1)
def load_returning_reference():
    """Load the frozen 15-event pre-escalation reference from the parent trajectory."""
    anchor = json.loads((RETURN_REFERENCE / "anchor.json").read_text())
    hi = int(anchor["selection"]["eligibility"]["escalation_bin"])
    with np.load(RETURN_REFERENCE / "anchor_traces.npz", allow_pickle=False) as data:
        core = np.asarray(data["r_core"], float)
        surround = np.asarray(data["r_surround"], float)
        active = np.asarray(data["A_active"], float)
        kymo = np.asarray(data["kymo_axial"], float)
        baseline = event_free_baseline_bins(core, max_ms=hi * BIN_MS)
        rms, status = contact_rms_from_baseline(
            data["lfp"], float(data["lfp_fs"]), baseline
        )
    smooth = A._moving_mean(core[:hi], int(round(100.0 / BIN_MS)))
    base = float(np.percentile(smooth, 20))
    threshold = float(base + 0.5 * (np.max(smooth) - base))
    windows = returning_event_windows(
        core, threshold_hz=threshold, hi_bin=hi, smooth_ms=100.0
    )
    feats = returning_event_features(
        core, surround, active, kymo, rms, windows
    )
    return {
        "status": status, "threshold_hz": threshold,
        "n_events": len(feats), "events": feats,
        "source": str((RETURN_REFERENCE / "anchor_traces.npz").relative_to(ROOT)),
    }


def returning_recovery(arrays, contact_rms, offset_bin):
    reference = load_returning_reference()
    if offset_bin is None:
        return {
            "status": "no_offset", "single_event_candidate": False,
            "distribution_recovered": False,
            "reference": {key: value for key, value in reference.items() if key != "events"},
        }
    start = int(offset_bin) + int(round(500.0 / BIN_MS))
    core = np.asarray(arrays["coarse_core_rate_hz"], float)
    windows = returning_event_windows(
        core, threshold_hz=reference["threshold_hz"], lo_bin=start,
        smooth_ms=100.0,
    )
    post = returning_event_features(
        core, arrays["coarse_surround_rate_hz"], arrays["coarse_active_fraction"],
        arrays["coarse_kymo_axial"], contact_rms, windows,
    )
    match = match_returning_events(reference["events"], post)
    match.update({
        "post_offset_search_start_ms": float(start * BIN_MS),
        "post_event_candidates": post,
        "reference": {key: value for key, value in reference.items() if key != "events"},
    })
    return match


def baseline_referenced_intensity(rms, baseline_bins, episode_slice):
    if rms is None:
        return {"status": "unavailable"}
    # RMS is an amplitude.  Square it before using the 10*log10 power rule;
    # applying 10*log10 directly to RMS would halve every dB gain and turn the
    # +6 dB threshold from a 2x RMS ratio into an incorrect 4x ratio.
    power = np.asarray(rms, float) ** 2
    base = power[np.asarray(baseline_bins, bool)[:len(power)]]
    cand = power[episode_slice]
    if base.size == 0 or cand.size == 0:
        return {"status": "unavailable"}
    # Normalize after integrating power within each contact.  Dividing every
    # time bin by a near-zero event-free value produces arbitrarily large
    # ratios for quiet contacts and rewards sparse bursts.  Contacts with an
    # exactly zero baseline are explicitly unavailable rather than epsilon-
    # promoted to infinite gain.
    baseline_power = np.mean(base, axis=0)
    candidate_power = np.mean(cand, axis=0)
    floor = max(float(np.max(baseline_power)) * 1e-12, np.finfo(float).tiny)
    valid = baseline_power > floor
    if not np.any(valid):
        return {
            "status": "zero_power_event_free_baseline",
            "n_valid_baseline_contacts": 0,
        }
    gain_db = 10.0 * np.log10(candidate_power[valid] / baseline_power[valid])
    threshold = baseline_power[valid] * (10.0 ** (6.0 / 10.0))
    above = cand[:, valid] > threshold[None, :]
    normalized_integral = float(
        np.sum(candidate_power[valid]) / np.sum(baseline_power[valid])
    )
    return {
        "status": "ok",
        "median_gain_db_across_contacts": float(np.median(gain_db)),
        "max_gain_db_across_contacts": float(np.max(gain_db)),
        "active_contact_fraction_6db": float(np.mean(gain_db >= 6.0)),
        "occupancy_above_6db": float(np.mean(above)),
        "normalized_integrated_energy_per_s": normalized_integral,
        "candidate_energy_median": float(np.median(candidate_power[valid])),
        "baseline_energy_median": float(np.median(baseline_power[valid])),
        "n_valid_baseline_contacts": int(np.sum(valid)),
        "n_total_contacts": int(valid.size),
    }


def _spectral_entropy(x, dt_s):
    x = np.asarray(x, float)
    if x.size < 8 or np.std(x) <= 1e-12:
        return 0.0
    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    power = power[1:]
    if power.sum() <= 0:
        return 0.0
    p = power / power.sum()
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(p)))


def analyze_one(root):
    summary = json.loads((root / "summary.json").read_text())
    with np.load(root / "traces.npz", allow_pickle=False) as data:
        a = {key: np.asarray(data[key]) for key in data.files}
    core = np.asarray(a["coarse_core_rate_hz"], float)
    baseline = event_free_baseline_bins(core)
    episode = detect_episode(core, baseline)
    rms, baseline_status = contact_rms_from_baseline(
        a["lfp_raw_synaptic_proxy"], float(a["lfp_fs_hz"]), baseline
    )
    onset = episode["onset_bin"]
    offset = episode["offset_bin"]
    e0 = 0 if onset is None else int(onset)
    e1 = len(core) if offset is None else int(offset)
    intensity = baseline_referenced_intensity(rms, baseline, slice(e0, e1))
    recovery = returning_recovery(a, rms, offset)
    post0 = min(e1, e0 + int(round(1000.0 / BIN_MS)))
    post = core[post0:e1]
    post_cv = float(np.std(post) / max(np.mean(post), 1e-12)) if post.size else None
    post_entropy = _spectral_entropy(post, BIN_MS / 1000.0) if post.size else None
    kymo = np.asarray(a["coarse_kymo_axial"], float)[:, e0:e1]
    spatial = A._post_entry_spatial_metrics(kymo, skip_ms=1000.0)
    relay = A._packet_axial_relay(
        kymo[:, int(round(1000.0 / BIN_MS)):],
        core[e0:e1][int(round(1000.0 / BIN_MS)):],
    ) if e1 - e0 > int(round(1000.0 / BIN_MS)) else {"packet_axial_relay_R": 0.0}
    corr = A._conditional_corr(
        np.asarray(a["coarse_core_rate_hz"])[post0:e1],
        np.asarray(a["coarse_surround_rate_hz"])[post0:e1],
    ) if e1 > post0 else {"status": "insufficient", "value": None}
    active = np.asarray(a["coarse_active_fraction"], float)[post0:e1]
    active_slope = (
        float(np.polyfit(np.arange(active.size) * BIN_MS / 1000.0, active, 1)[0])
        if active.size >= 4 else None
    )
    runaway = summary.get("runaway_early_stop_ms") is not None
    if runaway:
        phenotype = "runaway"
    elif onset is None:
        phenotype = "no_onset"
    elif post_cv is not None and post_cv < 0.15 and active_slope is not None and active_slope > 0.002:
        phenotype = "spreading_plateau"
    elif post_cv is not None and post_cv < 0.15:
        phenotype = "tonic_patch"
    elif intensity.get("occupancy_above_6db", 0.0) < 0.20:
        phenotype = "weak_or_fragmented"
    elif spatial.get("spatial_effective_rank", 1.0) >= 2.0 and post_cv < 1.0:
        phenotype = "structured_dynamic_candidate"
    else:
        phenotype = "relaxation_burst_train"
    components = {}
    if "lfp_exc_synaptic_proxy" in a and "lfp_inh_synaptic_proxy" in a:
        exc_rms, _ = contact_rms_from_baseline(
            a["lfp_exc_synaptic_proxy"], float(a["lfp_fs_hz"]), baseline
        )
        inh_rms, _ = contact_rms_from_baseline(
            a["lfp_inh_synaptic_proxy"], float(a["lfp_fs_hz"]), baseline
        )
        components = {
            "exc_candidate_median": float(np.median(exc_rms[e0:e1])),
            "inh_candidate_median": float(np.median(inh_rms[e0:e1])),
        }
    return {
        "stem": root.name,
        "mechanism": summary["mechanism"],
        "finite_control": summary.get("finite_control"),
        "terminal_status": "scientific_early_stop" if runaway else "success",
        "phenotype": phenotype,
        "baseline_status": baseline_status,
        "n_event_free_baseline_bins": int(baseline.sum()),
        "episode": episode,
        "intensity": intensity,
        "recovery": recovery,
        "post_entry_core_cv": post_cv,
        "post_entry_spectral_entropy": post_entropy,
        "post_entry_active_fraction_slope_s": active_slope,
        "core_surround_correlation": corr,
        "within_episode_spatial": spatial,
        "within_episode_relay": relay,
        "readout_components": components,
        "summary_path": str((root / "summary.json").relative_to(ROOT)),
        "trace_path": str((root / "traces.npz").relative_to(ROOT)),
    }


def _flat(row):
    slow = row["mechanism"].get("dynamic_slow_flow", {})
    control = row.get("finite_control") or {}
    return {
        "stem": row["stem"],
        "terminal_status": row["terminal_status"],
        "phenotype": row["phenotype"],
        "tau_D_ms": row["mechanism"].get("i2e_depression", {}).get("tau_D_ms"),
        "d_star": row["mechanism"].get("i2e_depression", {}).get("d_star_nominal"),
        "tau_I_ms": row["mechanism"].get("i_adaptation", {}).get("tau_aI_ms"),
        "f_I": row["mechanism"].get("i_adaptation", {}).get("f_aI"),
        "strength_scale": row["mechanism"].get("strength_scale"),
        "g_M": slow.get("g_M"),
        "tau_M_ms": slow.get("tau_M_ms"),
        "g_Z": slow.get("g_Z"),
        "control_target": control.get("target"),
        "control_t0_ms": control.get("t0_ms"),
        "control_duration_ms": control.get("duration_ms"),
        "control_uplift_mV": control.get("uplift_mV"),
        "onset_ms": row["episode"].get("onset_ms"),
        "offset_ms": row["episode"].get("offset_ms"),
        "energy_gain_db": row["intensity"].get("median_gain_db_across_contacts"),
        "energy_6db_occupancy": row["intensity"].get("occupancy_above_6db"),
        "integrated_energy": row["intensity"].get("normalized_integrated_energy_per_s"),
        "returning_event_candidate": row["recovery"].get("single_event_candidate"),
        "returning_event_distribution_recovered": row["recovery"].get("distribution_recovered"),
        "n_post_offset_event_candidates": row["recovery"].get("n_post"),
        "core_cv": row["post_entry_core_cv"],
        "spatial_rank": row["within_episode_spatial"].get("spatial_effective_rank"),
        "centroid_excursion": row["within_episode_spatial"].get("centroid_excursion_bins"),
        "pc1": row["within_episode_spatial"].get("common_mode_pc1_fraction"),
        "summary_path": row["summary_path"],
    }


def main():
    roots = [p for p in sorted(IN_ROOT.glob("*")) if (p / "summary.json").is_file()]
    rows = [analyze_one(root) for root in roots]
    adaptation_path = OUT_ROOT / "batch1_adaptation_decisions.json"
    adaptation = (
        json.loads(adaptation_path.read_text()).get("decisions", [])
        if adaptation_path.is_file() else []
    )
    payload = {
        "schema": "topic4_zm_lifecycle_sprint_phase_map_v1_2026-08-02",
        "semantic_scope": "seed1_development_multiobjective_not_acceptance",
        "n_runs": len(rows),
        "adaptation_decisions": adaptation,
        "rows": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "batch1_phase_map.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    flat = [_flat(row) for row in rows]
    if flat:
        with (OUT_ROOT / "batch1_phase_map.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    print(json.dumps({
        "n_runs": len(rows),
        "phenotypes": {name: sum(r["phenotype"] == name for r in rows)
                       for name in sorted({r["phenotype"] for r in rows})},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
