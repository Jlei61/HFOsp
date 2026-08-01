#!/usr/bin/env python3
"""Online multi-objective analysis for the seed-1 joint lifecycle sprint."""
from __future__ import annotations

import csv
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
    if onset is not None:
        offset = A._first_sustained(
            smooth <= recovery_threshold, int(round(500.0 / bin_ms)),
            start=onset + int(round(1000.0 / bin_ms)),
        )
    return {
        "status": "onset" if onset is not None else "no_onset",
        "baseline_core_median_hz": med,
        "baseline_core_mad_hz": mad,
        "onset_threshold_hz": onset_threshold,
        "recovery_threshold_hz": recovery_threshold,
        "onset_bin": onset,
        "offset_bin": offset,
        "onset_ms": None if onset is None else onset * bin_ms,
        "offset_ms": None if offset is None else offset * bin_ms,
    }


def baseline_referenced_intensity(rms, baseline_bins, episode_slice):
    if rms is None:
        return {"status": "unavailable"}
    base = rms[np.asarray(baseline_bins, bool)[:len(rms)]]
    cand = rms[episode_slice]
    if base.size == 0 or cand.size == 0:
        return {"status": "unavailable"}
    eps = np.finfo(float).eps
    baseline_median = np.median(base, axis=0)
    candidate_median = np.median(cand, axis=0)
    gain_db = 10.0 * np.log10((candidate_median + eps) / (baseline_median + eps))
    threshold = baseline_median * (10.0 ** (6.0 / 10.0))
    above = cand > threshold[None, :]
    normalized_integral = float(np.mean(cand / np.maximum(baseline_median, eps)))
    return {
        "status": "ok",
        "median_gain_db_across_contacts": float(np.median(gain_db)),
        "max_gain_db_across_contacts": float(np.max(gain_db)),
        "active_contact_fraction_6db": float(np.mean(gain_db >= 6.0)),
        "occupancy_above_6db": float(np.mean(above)),
        "normalized_integrated_energy_per_s": normalized_integral,
        "candidate_energy_median": float(np.median(cand)),
        "baseline_energy_median": float(np.median(base)),
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
        "terminal_status": "scientific_early_stop" if runaway else "success",
        "phenotype": phenotype,
        "baseline_status": baseline_status,
        "n_event_free_baseline_bins": int(baseline.sum()),
        "episode": episode,
        "intensity": intensity,
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
    return {
        "stem": row["stem"],
        "terminal_status": row["terminal_status"],
        "phenotype": row["phenotype"],
        "tau_D_ms": row["mechanism"].get("i2e_depression", {}).get("tau_D_ms"),
        "d_star": row["mechanism"].get("i2e_depression", {}).get("d_star_nominal"),
        "tau_I_ms": row["mechanism"].get("i_adaptation", {}).get("tau_aI_ms"),
        "f_I": row["mechanism"].get("i_adaptation", {}).get("f_aI"),
        "strength_scale": row["mechanism"].get("strength_scale"),
        "onset_ms": row["episode"].get("onset_ms"),
        "offset_ms": row["episode"].get("offset_ms"),
        "energy_gain_db": row["intensity"].get("median_gain_db_across_contacts"),
        "energy_6db_occupancy": row["intensity"].get("occupancy_above_6db"),
        "integrated_energy": row["intensity"].get("normalized_integrated_energy_per_s"),
        "core_cv": row["post_entry_core_cv"],
        "spatial_rank": row["within_episode_spatial"].get("spatial_effective_rank"),
        "centroid_excursion": row["within_episode_spatial"].get("centroid_excursion_bins"),
        "pc1": row["within_episode_spatial"].get("common_mode_pc1_fraction"),
        "summary_path": row["summary_path"],
    }


def main():
    roots = [p for p in sorted(IN_ROOT.glob("*")) if (p / "summary.json").is_file()]
    rows = [analyze_one(root) for root in roots]
    payload = {
        "schema": "topic4_zm_lifecycle_sprint_phase_map_v1_2026-08-02",
        "semantic_scope": "seed1_development_multiobjective_not_acceptance",
        "n_runs": len(rows),
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
