#!/usr/bin/env python3
"""Summarise the seed-1 dynamic-threshold discovery matrix.

This is a development readout: it ranks visible dynamical phenotypes and does
not certify reachability, offset, recovery, or a lifecycle.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import topic4_zm_phasec_phenotype as PH  # noqa: E402


IN_ROOT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/discovery/seed1"
RACE_ROOT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/race/seed1"
DYNAMIC_ROOT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/dynamic/seed1"
OUT_ROOT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development"
STATES = (
    "bounded_mid__rising", "bounded_mid__peak",
    "bounded_late__rising", "bounded_late__peak",
)
PARAMS = ((60, .15), (60, .30), (100, .15), (100, .30), (160, .15), (160, .30))
PHENOTYPES = (
    "silence", "tonic", "burst_train", "whole_sheet_oscillation",
    "metastable_carrier_like", "spatially_relayed_carrier", "runaway",
    "technical_invalid",
)
COLORS = (
    "#D9D9D9", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#1B9E77", "#B2182B", "#111111",
)


def _safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def _vseeg_energy(lfp, fs_hz):
    x = np.asarray(lfp, float)
    if x.ndim == 1:
        x = x[:, None]
    bs = max(1, int(round(0.025 * float(fs_hz))))
    n = x.shape[0] // bs * bs
    if n == 0:
        return np.zeros(0), {"energy_floor_fraction": None, "energy_gap_fraction": None}
    x = x[:n] - np.mean(x[:n], axis=0, keepdims=True)
    rms = np.sqrt(np.mean(x.reshape(-1, bs, x.shape[1]) ** 2, axis=1))
    envelope = np.mean(rms, axis=1)
    p95 = float(np.percentile(envelope, 95))
    return envelope, {
        "energy_floor_fraction": float(np.median(envelope) / max(p95, 1e-12)),
        "energy_gap_fraction": float(np.mean(envelope < 0.20 * max(p95, 1e-12))),
        "energy_modulation_depth": float(
            (np.percentile(envelope, 95) - np.percentile(envelope, 5))
            / max(np.mean(envelope), 1e-12)
        ),
    }


def _race_energy(lfp, fs_hz):
    """The locked race readout: 25-ms vSEEG RMS and Q10/Q90 support."""
    x = np.asarray(lfp, float)
    if x.ndim == 1:
        x = x[:, None]
    bs = max(1, int(round(0.025 * float(fs_hz))))
    n = x.shape[0] // bs * bs
    if n == 0:
        return np.zeros(0), {"energy_floor_F": None, "deep_gap_G": None}
    x = x[:n] - np.mean(x[:n], axis=0, keepdims=True)
    rms = np.sqrt(np.mean(x.reshape(-1, bs, x.shape[1]) ** 2, axis=1))
    envelope = np.mean(rms, axis=1)
    q10, q90 = np.percentile(envelope, [10, 90])
    return envelope, {
        "energy_q10": float(q10),
        "energy_q90": float(q90),
        "energy_floor_F": float(q10 / max(q90, 1e-12)),
        "deep_gap_G": float(np.mean(envelope < 0.25 * max(q90, 1e-12))),
    }


def _merged_episodes(mask, gap_bins):
    idx = np.flatnonzero(np.asarray(mask, bool))
    if not idx.size:
        return []
    out, start, previous = [], int(idx[0]), int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value - previous > int(gap_bins) + 1:
            out.append((start, previous + 1))
            start = value
        previous = value
    out.append((start, previous + 1))
    return out


def _packet_axial_relay(kymograph_axis_time, core_rate, *, bin_ms=25.0):
    """Continuous packet-wise first-passage score; flashes score near zero."""
    K = np.asarray(kymograph_axis_time, float).T
    rate = np.asarray(core_rate, float).ravel()
    if K.ndim != 2 or K.shape[0] != rate.size or K.shape[1] < 4:
        return {"packet_axial_relay_R": 0.0, "n_packets": 0, "n_valid_packets": 0}
    r10, r90 = np.percentile(rate, [10, 90])
    active = rate >= r10 + 0.25 * max(r90 - r10, 1e-12)
    episodes = _merged_episodes(active, gap_bins=max(1, int(round(50.0 / bin_ms))))
    lo = np.percentile(K, 10, axis=0)
    hi = np.percentile(K, 95, axis=0)
    threshold = lo + 0.35 * (hi - lo)
    scores, signs = [], []
    for i0, i1 in episodes:
        packet = K[i0:i1]
        if packet.shape[0] < 2:
            continue
        crossed = packet >= threshold[None, :]
        idx = np.flatnonzero(np.any(crossed, axis=0))
        if idx.size < 4:
            continue
        first = np.array([np.flatnonzero(crossed[:, j])[0] for j in idx], float)
        temporal_span = float(np.ptp(first))
        if temporal_span < 1.0:
            scores.append(0.0)
            continue
        rho = float(np.corrcoef(np.arange(idx.size, dtype=float), first)[0, 1])
        if not np.isfinite(rho):
            continue
        spatial_span = float(np.ptp(idx) / max(1, K.shape[1] - 1))
        temporal_score = min(1.0, temporal_span / 2.0)
        earliest = int(np.min(first))
        flash_fraction = float(np.mean(crossed[earliest, idx]))
        scores.append(abs(rho) * spatial_span * temporal_score * (1.0 - flash_fraction))
        signs.append(int(np.sign(rho)))
    if not scores:
        value, consistency = 0.0, 0.0
    else:
        nonzero = [s for s in signs if s]
        consistency = (
            max(nonzero.count(-1), nonzero.count(1)) / len(nonzero)
            if nonzero else 0.0
        )
        value = float(np.mean(scores) * consistency)
    return {
        "packet_axial_relay_R": value,
        "n_packets": len(episodes),
        "n_valid_packets": len(scores),
        "direction_consistency": float(consistency),
        "packet_scores": [float(x) for x in scores],
    }


def _minmax(rows, key):
    values = np.array([float(row[key]) for row in rows], float)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo <= 1e-12:
        return np.full(values.size, 0.5)
    return (values - lo) / (hi - lo)


def analyze_race():
    """Rank all active bounded mechanism arms without turning this into a gate."""
    rows = []
    for root in sorted(RACE_ROOT.glob("*")):
        summary_path, trace_path = root / "summary.json", root / "traces.npz"
        if not summary_path.is_file() or not trace_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        with np.load(trace_path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        envelope, energy = _race_energy(
            arrays["lfp_raw_synaptic_proxy"], float(arrays["lfp_fs_hz"])
        )
        core = np.asarray(arrays["fine_core_rate_hz"], float)
        surround = np.asarray(arrays["fine_surround_rate_hz"], float)
        coarse = np.asarray(arrays["coarse_core_rate_hz"], float)
        q10, q90 = np.percentile(coarse, [10, 90])
        episodes = _merged_episodes(
            coarse >= q10 + 0.25 * max(q90 - q10, 1e-12), gap_bins=2
        )
        episode_occupancy = (
            max((i1 - i0) for i0, i1 in episodes) / coarse.size if episodes else 0.0
        )
        relay = _packet_axial_relay(arrays["coarse_kymo_axial"], coarse)
        corr = _safe_corr(core, surround)
        runaway = summary.get("runaway_early_stop_ms") is not None
        silent = float(summary["core_modulation"]["mean_hz"]) < 2.0
        rows.append({
            "stem": root.name,
            "mechanism": summary.get("mechanism"),
            "excluded": bool(runaway or silent),
            "exclusion_reason": "runaway" if runaway else ("silence" if silent else None),
            "core_mean_hz": float(summary["core_modulation"]["mean_hz"]),
            "all_E_mean_hz": float(summary["all_E_modulation"]["mean_hz"]),
            "peak_active_fraction": float(summary["peak_active_fraction"]),
            "core_surround_correlation": float(corr) if corr is not None else 1.0,
            "episode_occupancy_O": float(episode_occupancy),
            **energy,
            **relay,
            "summary_path": str(summary_path.relative_to(ROOT)),
            "trace_path": str(trace_path.relative_to(ROOT)),
            "_arrays": arrays,
            "_envelope": envelope,
        })
    valid = [row for row in rows if not row["excluded"]]
    if not valid:
        raise SystemExit("mechanism race has no active bounded arms")
    desirability = {
        "F_norm": _minmax(valid, "energy_floor_F"),
        "one_minus_G_norm": _minmax(
            [{"v": 1.0 - row["deep_gap_G"]} for row in valid], "v"
        ),
        "one_minus_rho_norm": _minmax(
            [{"v": 1.0 - row["core_surround_correlation"]} for row in valid], "v"
        ),
        "R_axis_norm": _minmax(valid, "packet_axial_relay_R"),
        "O_episode_norm": _minmax(valid, "episode_occupancy_O"),
    }
    for i, row in enumerate(valid):
        for name, values in desirability.items():
            row[name] = float(values[i])
        row["J"] = float(
            1.5 * row["F_norm"]
            + row["one_minus_G_norm"]
            + row["one_minus_rho_norm"]
            + 1.5 * row["R_axis_norm"]
            + 0.5 * row["O_episode_norm"]
        )
    for row in rows:
        if row["excluded"]:
            row["J"] = None
    ranking = sorted(valid, key=lambda row: row["J"], reverse=True)
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    payload = {
        "schema": "zm_fast_lifecycle_mechanism_race_v1_2026-08-01",
        "semantic_scope": "seed1_frozen_branch_continuous_ranking_not_lifecycle",
        "score": "J=1.5F+1(1-G)+1(1-rho)+1.5R_axis+0.5O_episode after arm-wise min-max normalization",
        "exclusion": "runaway_or_core_mean_below_2Hz_only",
        "n_expected": 21 if any(
            row["stem"].startswith("combined__") for row in rows
        ) else 17,
        "n_observed": len(rows),
        "n_ranked": len(ranking),
        "ranking": [public(row) for row in ranking],
        "excluded": [public(row) for row in rows if row["excluded"]],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "mechanism_race_ranking.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    fields = sorted({k for row in payload["ranking"] + payload["excluded"] for k in row})
    with (OUT_ROOT / "mechanism_race_ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["ranking"] + payload["excluded"])
    _plot_race(ranking)
    return payload


def _plot_race(ranking):
    n = min(6, len(ranking))
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 2.2 * n), squeeze=False, constrained_layout=True)
    for i, row in enumerate(ranking[:n]):
        arrays, env = row["_arrays"], row["_envelope"]
        t = 1.0 + np.asarray(arrays["fine_time_ms"]) / 1000.0
        axes[i, 0].plot(t, arrays["fine_core_rate_hz"], color="#B2182B", lw=.55)
        axes[i, 0].plot(t, arrays["fine_surround_rate_hz"], color="#2166AC", lw=.45)
        axes[i, 0].set_ylabel(f"#{i+1}  J={row['J']:.2f}\nHz")
        te = 1.0 + np.arange(env.size) * .025
        axes[i, 1].plot(te, env, color="#E66101", lw=.65)
        axes[i, 2].imshow(
            arrays["coarse_kymo_axial"], origin="lower", aspect="auto", cmap="magma",
            extent=(1, 1 + arrays["coarse_kymo_axial"].shape[1] * .025, 0, 24),
        )
        axes[i, 0].set_title(row["stem"], loc="left", fontsize=9)
        axes[i, 1].set_title(
            f"F={row['energy_floor_F']:.2f}, G={row['deep_gap_G']:.2f}, rho={row['core_surround_correlation']:.2f}",
            loc="left", fontsize=8,
        )
        axes[i, 2].set_title(
            f"Raxis={row['packet_axial_relay_R']:.2f}, O={row['episode_occupancy_O']:.2f}",
            loc="left", fontsize=8,
        )
    for ax in axes[-1]:
        ax.set_xlabel("time after frozen checkpoint (s)")
    axes[0, 0].set_title("core/surround · " + axes[0, 0].get_title(), loc="left", fontsize=9)
    fig.suptitle("Seed-1 fast inhibitory mechanism race — diagnostic ranking", fontweight="bold")
    figures = OUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "fig_mechanism_race_diagnostic.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _moving_mean(x, bins):
    x = np.asarray(x, float)
    if bins <= 1:
        return x.copy()
    return np.convolve(x, np.ones(int(bins)) / int(bins), mode="same")


def _first_sustained(mask, bins, start=0):
    x = np.asarray(mask, bool)
    n = int(bins)
    if n < 1 or x.size < n:
        return None
    run = np.convolve(x.astype(np.int16), np.ones(n, np.int16), mode="valid")
    hits = np.flatnonzero((run == n) & (np.arange(run.size) >= int(start)))
    return None if not hits.size else int(hits[0])


def _dynamic_semantics(summary, arrays):
    """Descriptive onset/offset/recovery semantics, deliberately not a new gate."""
    bin_ms = 25.0
    core = np.asarray(arrays["coarse_core_rate_hz"], float)
    surround = np.asarray(arrays["coarse_surround_rate_hz"], float)
    nbase = min(core.size, int(round(float(summary["equilibration_ms"]) / bin_ms)))
    baseline = core[:max(4, nbase)]
    med = float(np.median(baseline))
    mad = float(1.4826 * np.median(np.abs(baseline - med)))
    ictal_threshold = max(50.0, med + 3.0 * mad)
    smooth = _moving_mean(core, int(round(500.0 / bin_ms)))
    onset_bin = _first_sustained(
        smooth >= ictal_threshold, int(round(250.0 / bin_ms)), start=nbase // 2
    )
    offset_bin = None
    recovery_threshold = max(25.0, med + 1.5 * mad)
    if onset_bin is not None:
        offset_bin = _first_sustained(
            smooth <= recovery_threshold,
            int(round(1000.0 / bin_ms)),
            start=onset_bin + int(round(500.0 / bin_ms)),
        )
    tail_start = offset_bin if offset_bin is not None else max(0, core.size - int(5000 / bin_ms))
    tail = core[tail_start:]
    event_mask = tail >= max(35.0, med + 2.0 * mad)
    tail_events = _merged_episodes(event_mask, gap_bins=2)
    tail_events = [ep for ep in tail_events if (ep[1] - ep[0]) * bin_ms >= 25.0]
    envelope, energy_all = _race_energy(
        arrays["lfp_raw_synaptic_proxy"], float(arrays["lfp_fs_hz"])
    )
    baseline_energy_q90 = float(np.percentile(envelope[:max(4, nbase)], 90))
    e0 = 0 if onset_bin is None else onset_bin
    e1 = len(envelope) if offset_bin is None else min(len(envelope), offset_bin)
    _, energy_episode = _race_energy(
        arrays["lfp_raw_synaptic_proxy"][
            int(e0 * bin_ms / 0.1): int(e1 * bin_ms / 0.1)
        ],
        float(arrays["lfp_fs_hz"]),
    ) if e1 > e0 else (np.zeros(0), {"energy_floor_F": None, "deep_gap_G": None})
    if summary.get("runaway_early_stop_ms") is not None:
        label = "onset_to_runaway"
    elif onset_bin is None:
        label = "no_detected_onset"
    elif offset_bin is None:
        label = "onset_persistent_to_end"
    elif tail_events:
        label = "onset_offset_returning_events_candidate"
    else:
        label = "onset_offset_quiet_tail"
    return {
        "trajectory_semantics": label,
        "baseline_core_median_hz": med,
        "baseline_core_mad_hz": mad,
        "ictal_threshold_hz": ictal_threshold,
        "recovery_threshold_hz": recovery_threshold,
        "onset_ms_after_warm_start": None if onset_bin is None else onset_bin * bin_ms,
        "offset_ms_after_warm_start": None if offset_bin is None else offset_bin * bin_ms,
        "episode_duration_ms": (
            None if onset_bin is None or offset_bin is None
            else (offset_bin - onset_bin) * bin_ms
        ),
        "tail_returning_event_count": len(tail_events),
        "tail_core_mean_hz": float(np.mean(tail)) if tail.size else None,
        "core_surround_correlation": _safe_corr(core, surround),
        "whole_trace_energy": energy_all,
        "candidate_episode_energy": energy_episode,
        "baseline_energy_q90": baseline_energy_q90,
        "candidate_floor_gain_over_baseline_q90": (
            None if energy_episode.get("energy_q10") is None
            else float(energy_episode["energy_q10"] / max(baseline_energy_q90, 1e-12))
        ),
        "candidate_q90_gain_over_baseline_q90": (
            None if energy_episode.get("energy_q90") is None
            else float(energy_episode["energy_q90"] / max(baseline_energy_q90, 1e-12))
        ),
        "vseeg_envelope": envelope,
    }


def analyze_dynamic():
    rows = []
    for root in sorted(DYNAMIC_ROOT.glob("*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        with np.load(tp, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        semantics = _dynamic_semantics(summary, arrays)
        rows.append({
            "stem": root.name,
            "mechanism": summary.get("mechanism"),
            "observed_ms": summary["observed_ms"],
            "z_initial_core_mean": float(arrays["trace_z_core_mean"][0]),
            "z_final_core_mean": float(arrays["trace_z_core_mean"][-1]),
            "m_initial_core_mean": float(arrays["trace_m_core_mean"][0]),
            "m_final_core_mean": float(arrays["trace_m_core_mean"][-1]),
            **{k: v for k, v in semantics.items() if k != "vseeg_envelope"},
            "summary_path": str(sp.relative_to(ROOT)),
            "trace_path": str(tp.relative_to(ROOT)),
            "_arrays": arrays,
            "_envelope": semantics["vseeg_envelope"],
        })
    if not rows:
        raise SystemExit("no dynamic lifecycle prototypes found")
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    payload = {
        "schema": "zm_fast_lifecycle_dynamic_prototype_v1_2026-08-01",
        "semantic_scope": "seed1_descriptive_prototype_not_lifecycle_acceptance",
        "n_runs": len(rows),
        "runs": [public(row) for row in rows],
    }
    (OUT_ROOT / "dynamic_lifecycle_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _plot_dynamic(rows)
    return payload


def _plot_dynamic(rows):
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.0 * len(rows)), squeeze=False,
                             constrained_layout=True)
    for i, row in enumerate(rows):
        a, env = row["_arrays"], row["_envelope"]
        t = np.asarray(a["fine_time_ms"]) / 1000.0
        axes[i, 0].plot(t, a["fine_core_rate_hz"], color="#B2182B", lw=.5)
        axes[i, 0].plot(t, a["fine_surround_rate_hz"], color="#2166AC", lw=.4)
        axes[i, 0].set_title(row["stem"] + "\n" + row["trajectory_semantics"], loc="left", fontsize=8)
        te = np.arange(env.size) * .025
        axes[i, 1].plot(te, env, color="#E66101", lw=.6)
        axes[i, 1].set_title("virtual-SEEG energy", loc="left", fontsize=8)
        axes[i, 2].imshow(
            a["coarse_kymo_axial"], origin="lower", aspect="auto", cmap="magma",
            extent=(0, a["coarse_kymo_axial"].shape[1] * .025, 0, 24),
        )
        axes[i, 2].set_title("pathological-axis activity", loc="left", fontsize=8)
        stride = max(1, len(a["trace_z_core_mean"]) // 3000)
        ts = np.arange(len(a["trace_z_core_mean"]))[::stride] * .0001
        axes[i, 3].plot(ts, a["trace_z_core_mean"][::stride], color="#1B9E77", lw=.7, label="z core")
        m = a["trace_m_core_mean"][::stride]
        axes[i, 3].plot(ts, m / max(1.0, float(np.max(m))), color="#6A3D9A", lw=.7, label="m core / max")
        axes[i, 3].legend(frameon=False, fontsize=7)
        axes[i, 3].set_title("dynamic Z/M slow flow", loc="left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time after pre-entry warm start (s)")
    fig.suptitle("Full dynamic Z/M lifecycle prototypes — seed 1 diagnostic", fontweight="bold")
    path = OUT_ROOT / "figures/fig_dynamic_lifecycle_prototypes.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def diagnose(summary, arrays):
    core = np.asarray(arrays["fine_core_rate_hz"], float)
    surround = np.asarray(arrays["fine_surround_rate_hz"], float)
    area = np.asarray(arrays["fine_active_fraction"], float)
    bin_ms = float(np.median(np.diff(arrays["fine_time_ms"])))
    rho80 = float(summary["core_rho80_active_fraction"])
    gate = PH.common_bounded_gate(
        core,
        bin_ms=bin_ms,
        active_area_fraction=area,
        runaway_early_stop_ms=summary.get("runaway_early_stop_ms"),
        saturation_fraction=float(summary["peak_active_fraction"]),
        refractory_fraction=rho80,
    )
    periodic = PH._peak_train(
        core, bin_ms=bin_ms,
        min_period_ms=max(2 * bin_ms, 1000 / 150), max_period_ms=200,
    )
    clonic = PH._peak_train(
        core, bin_ms=bin_ms, min_period_ms=150, max_period_ms=2000,
        lowpass_hz=5.0,
    )
    kymo = np.asarray(arrays["coarse_kymo_axial"], float).T
    relay = PH.spatial_relay_modifier(
        kymo, np.arange(kymo.shape[1], dtype=float), bin_ms=25.0,
        n_perm=199, rng_seed=0,
    )
    envelope, energy = _vseeg_energy(
        arrays["lfp_raw_synaptic_proxy"], float(arrays["lfp_fs_hz"])
    )
    common_corr = _safe_corr(core, surround)
    modulation = float(summary["core_modulation"]["depth"])
    active = core >= float(gate["activity_threshold_hz"])
    starts = np.flatnonzero(np.diff(np.r_[False, active].astype(np.int8)) == 1)
    interburst_ms = np.diff(starts) * bin_ms

    if summary.get("runaway_early_stop_ms") is not None or gate["status"] == "runaway":
        phenotype = "runaway"
    elif gate["status"] == "rest" or gate["source_mean_hz"] < 2.0:
        phenotype = "silence"
    elif gate["status"] == "hfo_like_train":
        phenotype = "burst_train"
    elif modulation < 0.20:
        phenotype = "tonic"
    elif (
        common_corr is not None and common_corr >= 0.90
        and gate["median_active_area_fraction"] >= 0.50
    ):
        phenotype = "whole_sheet_oscillation"
    elif (
        relay["is_spatial_relay"] and gate["active_occupancy"] >= 0.80
        and gate["longest_rest_dwell_ms"] < 100
    ):
        phenotype = "spatially_relayed_carrier"
    elif (
        gate["active_occupancy"] >= 0.80
        and gate["longest_rest_dwell_ms"] < 100
        and (periodic["n_cycles"] >= 10 or clonic["n_cycles"] >= 5)
    ):
        phenotype = "metastable_carrier_like"
    elif gate["n_active_episodes"] >= 4:
        phenotype = "burst_train"
    elif gate["active_occupancy"] >= 0.20:
        phenotype = "metastable_carrier_like"
    else:
        phenotype = "silence"

    candidate_score = (
        2.0 * (phenotype in {"spatially_relayed_carrier", "metastable_carrier_like"})
        + 1.0 * (modulation >= 0.20)
        + 1.0 * (gate["active_occupancy"] >= 0.80)
        + 1.0 * relay["is_spatial_relay"]
        + 1.0 * ((energy["energy_floor_fraction"] or 0) >= 0.50)
        + 1.0 * (rho80 <= 0.20)
        - 2.0 * (phenotype in {"silence", "runaway", "whole_sheet_oscillation"})
    )
    return {
        "phenotype": phenotype,
        "candidate_score": float(candidate_score),
        "core_surround_correlation": common_corr,
        "periodic_cycles": int(periodic["n_cycles"]),
        "clonic_cycles": int(clonic["n_cycles"]),
        "median_interburst_ms": (
            float(np.median(interburst_ms)) if interburst_ms.size else None
        ),
        "phi_core_mean_mV": float(np.mean(arrays["trace_phi_core_mean"])),
        "phi_core_p95_mV": float(np.percentile(arrays["trace_phi_core_mean"], 95)),
        "S_G_mean": float(np.mean(arrays["trace_S_G"])),
        "relay": relay,
        "bounded_gate": gate,
        "energy": energy,
        "vseeg_envelope": envelope,
    }


def load_rows():
    rows = []
    for state in STATES:
        for tau, fraction in PARAMS:
            stem = f"{state}__tau{tau:g}__f{fraction:g}"
            root = IN_ROOT / stem
            summary_path, trace_path = root / "summary.json", root / "traces.npz"
            base = {"state": state, "tau_phi_ms": tau, "fraction": fraction, "stem": stem}
            if not summary_path.is_file() or not trace_path.is_file():
                rows.append({**base, "phenotype": "technical_invalid", "reason": "missing_output"})
                continue
            try:
                summary = json.loads(summary_path.read_text())
                with np.load(trace_path, allow_pickle=False) as data:
                    arrays = {key: np.asarray(data[key]) for key in data.files}
                diag = diagnose(summary, arrays)
            except Exception as exc:
                rows.append({**base, "phenotype": "technical_invalid", "reason": repr(exc)})
                continue
            row = {
                **base,
                "phenotype": diag["phenotype"],
                "candidate_score": diag["candidate_score"],
                "core_mean_hz": summary["core_modulation"]["mean_hz"],
                "core_modulation_depth": summary["core_modulation"]["depth"],
                "all_E_mean_hz": summary["all_E_modulation"]["mean_hz"],
                "rho80": summary["core_rho80_active_fraction"],
                "active_occupancy": diag["bounded_gate"]["active_occupancy"],
                "longest_rest_dwell_ms": diag["bounded_gate"]["longest_rest_dwell_ms"],
                "n_active_episodes": diag["bounded_gate"]["n_active_episodes"],
                "core_surround_correlation": diag["core_surround_correlation"],
                "spatial_relay": diag["relay"]["is_spatial_relay"],
                "relay_reason": diag["relay"]["reason"],
                "periodic_cycles": diag["periodic_cycles"],
                "clonic_cycles": diag["clonic_cycles"],
                "median_interburst_ms": diag["median_interburst_ms"],
                "phi_core_mean_mV": diag["phi_core_mean_mV"],
                "phi_core_p95_mV": diag["phi_core_p95_mV"],
                "S_G_mean": diag["S_G_mean"],
                **diag["energy"],
                "summary_path": str(summary_path.relative_to(ROOT)),
                "trace_path": str(trace_path.relative_to(ROOT)),
            }
            row["_arrays"] = arrays
            row["_envelope"] = diag["vseeg_envelope"]
            rows.append(row)
    return rows


def _public(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_outputs(rows):
    valid = [row for row in rows if row["phenotype"] != "technical_invalid"]
    ranking = sorted(valid, key=lambda row: row.get("candidate_score", -99), reverse=True)
    promoted = [
        _public(row) for row in ranking
        if row["phenotype"] in {"spatially_relayed_carrier", "metastable_carrier_like"}
    ][:2]
    payload = {
        "schema": "zm_fast_lifecycle_stageA_phenotype_matrix_v1_2026-08-01",
        "semantic_scope": "seed1_branch_intervention_discovery_not_reachability",
        "n_expected": 24,
        "n_valid": len(valid),
        "phenotype_counts": {
            phenotype: sum(row["phenotype"] == phenotype for row in rows)
            for phenotype in PHENOTYPES
        },
        "selected_stageB_candidates": promoted,
        "cells": [_public(row) for row in rows],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "stageA_phenotype_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    fields = sorted({key for row in payload["cells"] for key in row})
    with (OUT_ROOT / "stageA_phenotype_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["cells"])
    return payload, ranking


def plot(rows, ranking):
    matrix = np.full((len(STATES), len(PARAMS)), PHENOTYPES.index("technical_invalid"))
    lookup = {(row["state"], row["tau_phi_ms"], row["fraction"]): row for row in rows}
    for iy, state in enumerate(STATES):
        for ix, (tau, fraction) in enumerate(PARAMS):
            matrix[iy, ix] = PHENOTYPES.index(lookup[state, tau, fraction]["phenotype"])
    representative = ranking[0]
    a = representative["_arrays"]
    envelope = representative["_envelope"]
    fig = plt.figure(figsize=(13.4, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=(1.25, 1.75), height_ratios=(1, .75, .85, 1.15))
    axm = fig.add_subplot(gs[:, 0])
    axr = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1])
    axp = fig.add_subplot(gs[2, 1])
    axk = fig.add_subplot(gs[3, 1])
    axm.imshow(matrix, cmap=ListedColormap(COLORS), vmin=-.5, vmax=len(PHENOTYPES)-.5,
               interpolation="nearest", aspect="auto")
    axm.set_xticks(range(len(PARAMS)), [f"{tau} ms\n{fraction:.2f}" for tau, fraction in PARAMS])
    axm.set_yticks(range(len(STATES)), [state.replace("bounded_", "").replace("__", " · ") for state in STATES])
    axm.set_xlabel(r"$\tau_\phi$ and target threshold fraction")
    axm.set_title("a  Fast-phenotype map", loc="left", fontweight="bold")
    for iy in range(len(STATES)):
        for ix in range(len(PARAMS)):
            row = lookup[STATES[iy], *PARAMS[ix]]
            axm.text(ix, iy, row["phenotype"].replace("spatially_relayed_", "relay\n")
                     .replace("metastable_", "meta\n").replace("whole_sheet_", "global\n")
                     .replace("_oscillation", "").replace("_carrier_like", ""),
                     ha="center", va="center", fontsize=7,
                     color="white" if matrix[iy, ix] in (3, 6, 7) else "black")
    t = 1.0 + np.asarray(a["fine_time_ms"]) / 1000
    axr.plot(t, a["fine_core_rate_hz"], color="#B2182B", lw=.8, label="core E")
    axr.plot(t, a["fine_surround_rate_hz"], color="#2166AC", lw=.7, label="surround E")
    axr.set_ylabel("rate (Hz)")
    axr.set_title(f"b  Representative: {representative['phenotype']}", loc="left", fontweight="bold")
    axr.legend(frameon=False, ncol=2, loc="upper right")
    axr.set_xlim(0, 6)
    te = 1.0 + np.arange(len(envelope)) * .025
    axe.plot(te, envelope, color="#E66101", lw=.9)
    axe.fill_between(te, 0, envelope, color="#FDB863", alpha=.35, linewidth=0)
    axe.set_ylabel("vSEEG RMS")
    axe.set_title("c  Continuous-energy readout", loc="left", fontweight="bold")
    axe.set_xlim(0, 6)
    tp = np.arange(len(a["trace_phi_core_mean"])) * .1 / 1000
    axp.plot(tp, a["trace_phi_core_mean"], color="#6A3D9A", lw=.8, label=r"core $\phi$")
    axp.plot(tp, a["trace_S_G"], color="#1B9E77", lw=.8, label=r"$S_G$")
    axp.set_ylabel("slow feedback")
    axp.legend(frameon=False, ncol=2, loc="upper right")
    axp.set_xlim(0, 6)
    im = axk.imshow(a["coarse_kymo_axial"], origin="lower", aspect="auto", cmap="magma",
                    extent=(1, 1 + a["coarse_kymo_axial"].shape[1] * .025, 0, 24))
    axk.set_xlabel("time after checkpoint (s)")
    axk.set_ylabel("pathological axis")
    axk.set_title("d  Axial recruitment", loc="left", fontweight="bold")
    axk.set_xlim(0, 6)
    fig.colorbar(im, ax=axk, label="spikes / bin", pad=.01)
    fig.suptitle("Dynamic threshold on frozen Z/M tonic branches — seed 1 discovery", fontweight="bold")
    figures = OUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "fig_stageA_phi_phenotype_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    (figures / "README.md").write_text(
        "### fig_stageA_phi_phenotype_matrix.png\n\n"
        "左侧展示四个原生 Z/M 高态检查点在六组 E-only 动态阈值参数下的 seed-1 表型。右侧展示当前排序最高的代表轨迹，包括核区/外围放电率、动态阈值与共享抑制，以及沿病理轴的时空活动。该图只回答旧 tonic branch 被改造成了什么，不证明该状态可从间期到达或能够终止恢复。\n\n"
        "**关注点**：优先寻找持续有界、调制明显、非全场同步且沿病理轴存在接力的区域；离散高频 burst train 不能当作 ictal carrier。\n"
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--race", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()
    if args.dynamic:
        payload = analyze_dynamic()
        print(json.dumps({
            "n_runs": payload["n_runs"],
            "semantics": {
                row["stem"]: row["trajectory_semantics"] for row in payload["runs"]
            },
        }, sort_keys=True))
        return
    if args.race:
        payload = analyze_race()
        print(json.dumps({
            "n_observed": payload["n_observed"],
            "n_ranked": payload["n_ranked"],
            "top": [row["stem"] for row in payload["ranking"][:5]],
        }, sort_keys=True))
        return
    rows = load_rows()
    payload, ranking = write_outputs(rows)
    if not ranking:
        raise SystemExit("no valid discovery cells")
    figure = plot(rows, ranking)
    print(json.dumps({
        "n_valid": payload["n_valid"],
        "phenotype_counts": payload["phenotype_counts"],
        "selected": [row["stem"] for row in payload["selected_stageB_candidates"]],
        "figure": str(figure.relative_to(ROOT)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
