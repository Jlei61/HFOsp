#!/usr/bin/env python3
"""Summarise the seed-1 dynamic-threshold discovery matrix.

This is a development readout: it ranks visible dynamical phenotypes and does
not certify reachability, offset, recovery, or a lifecycle.
"""
from __future__ import annotations

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
