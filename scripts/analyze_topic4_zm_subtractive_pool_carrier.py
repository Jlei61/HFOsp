#!/usr/bin/env python3
"""Does the pool's subtractive component give the sustained regime a rhythm?

Two findings are kept apart on purpose.  Breaking the tonic fixed point is one
result; clearing every gate while doing so is a different and stronger one, and
collapsing them would let a strength that trades away the energy floor read as
a carrier.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_topic4_zm_mode_h_pilot import _row  # noqa: E402
from scripts.analyze_topic4_zm_conductance_homotopy import credible_carrier  # noqa: E402
from scripts.analyze_topic4_zm_pv_som_carrier import (  # noqa: E402
    realized_removal_ratio, sustained_core_cv,
)


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/smoke/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
SUBSTRATE_LEVELS = (0.0, 0.32)
# Locked by the spec: below the first band the sustained rate is a flat line,
# above the second it is unambiguously modulated, and between them the panel
# cannot tell a rhythm from drift.
TONIC_CEILING = 0.10
CLEAN_FLOOR = 0.25


def _close(a, b):
    return a is not None and np.isclose(float(a), float(b))


def arm_key(summary):
    """(subtractive strength, persistent conductance, SOM wiring) or None."""
    mech = summary.get("mechanism", {})
    subtype = mech.get("pv_som_inhibitory_subtypes")
    mode = mech.get("state_selective_mode_H") or {}
    if not subtype or summary.get("state") != "bounded_late__peak":
        return None
    g = float(mode.get("mode_H_persistent_g_max", 0.0))
    if not (
        _close(summary.get("T_ms"), 2500.0)
        and _close(mode.get("rho_mode_H"), 0.0)
        and _close(mode.get("mode_H_persistent_e_exc"), 60.0)
        and _close(mode.get("tau_mode_H_down"), 250.0)
        and _close(mode.get("mode_H_common_subtraction"), 0.0)
        and _close(subtype.get("tau_d_som_ms"), 60.0)
        and _close(subtype.get("som_source_fraction_realized"), 0.25)
        and _close(subtype.get("som_slow_integrated_budget_fraction"), 0.35)
        and _close(subtype.get("som_recruit_delay_scale"), 3.0)
        and subtype.get("slow_membrane_mode") != "shunt"
        and any(np.isclose(g, level) for level in SUBSTRATE_LEVELS)
    ):
        return None
    beta = float(mech.get("subtractive_pool", {}).get("beta_SG", 0.0))
    return beta, g, int(subtype.get("seed", 1))


def modulation_band(cv):
    """Three-band reading; an arm can never be both passing and ambiguous."""
    if cv is None:
        return None
    if cv < TONIC_CEILING:
        return "tonic"
    return "ambiguous" if cv <= CLEAN_FLOOR else "clean"


def spectral_peak(sustained_rate, *, fs, band=(2.0, 200.0)):
    """Peak frequency of the sustained rate and how far it stands above the band.

    Reported, never gated on: variability alone cannot separate a rhythm from
    drift, and this is the quantity that says whether one frequency dominates.
    """
    rate = np.asarray(sustained_rate, float)
    rate = rate - rate.mean()
    nperseg = min(rate.size, 1024)
    freqs, power = welch(rate, fs=fs, nperseg=nperseg)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    if not inside.any() or power[inside].max() <= 0.0:
        return None, 0.0
    median = float(np.median(power[inside]))
    peak = int(np.argmax(power[inside]))
    prominence = float(power[inside][peak] / median) if median > 0.0 else float("inf")
    return float(freqs[inside][peak]), prominence


def adjudicate(rows):
    """Separate "the fixed point broke" from "a carrier cleared every gate"."""
    main = [r for r in rows if r["som_seed"] == 1 and r["persistent_g"] == 0.32]
    bare = [r for r in rows if r["som_seed"] == 1 and r["persistent_g"] == 0.0]
    broke = any(
        r["beta_SG"] > 0.0 and modulation_band(r["sustained_core_cv"]) == "clean"
        for r in main
    )
    candidates = [
        {"beta_SG": r["beta_SG"], "som_seed": r["som_seed"]}
        for r in rows
        if r["beta_SG"] > 0.0 and r["persistent_g"] == 0.32
        and r["credible_carrier"]
        and modulation_band(r["sustained_core_cv"]) == "clean"
    ]
    bare_carries = any(
        r["beta_SG"] > 0.0 and r["credible_carrier"]
        and modulation_band(r["sustained_core_cv"]) == "clean"
        for r in bare
    )
    replication = {}
    for r in rows:
        if r["beta_SG"] <= 0.0 or r["persistent_g"] != 0.32:
            continue
        entry = replication.setdefault(
            f"{r['beta_SG']:g}", {"seeds_tested": [], "seeds_passing": []}
        )
        entry["seeds_tested"].append(r["som_seed"])
        if r["credible_carrier"] and (
            modulation_band(r["sustained_core_cv"]) == "clean"
        ):
            entry["seeds_passing"].append(r["som_seed"])
    for entry in replication.values():
        entry["seeds_tested"].sort()
        entry["seeds_passing"].sort()

    if candidates and bare_carries:
        headline = "SUBTRACTIVE_POOL_CARRIES_WITHOUT_THE_EXCITATION"
        coordinate = "drop the persistent conductance; the subtractive term alone carries"
    elif candidates:
        headline = "SUBTRACTIVE_POOL_MODULATED_CANDIDATE"
        coordinate = "12-s durability, spectrum and DMD on the weakest candidate"
    elif broke:
        headline = "SUBTRACTIVE_POOL_BREAKS_THE_FIXED_POINT_BELOW_THE_ENERGY_FLOOR"
        coordinate = "the rhythm and the energy floor were not simultaneously reachable here"
    else:
        headline = "SUBTRACTIVE_POOL_LEAVES_THE_FIXED_POINT_INTACT"
        coordinate = "no tested strength made the sustained rate move"
    return {
        "verdict": headline,
        "candidate_arms": candidates,
        "broke_the_fixed_point": broke,
        "bare_substrate_carries": bare_carries,
        "wiring_replication": replication,
        "next_coordinate": coordinate,
        "claim_boundary": "seed-1 frozen-Z/M 2.5-s subtractive factorial",
    }


def main():
    found = {}
    for root in sorted(IN.glob("*pvSOM*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        key = arm_key(summary)
        if key is not None:
            if key in found:
                raise RuntimeError(f"duplicate subtractive arm: {key}")
            found[key] = (root, summary)
    if not found:
        raise RuntimeError("no subtractive-panel arms found")

    rows = {}
    for key, (root, summary) in sorted(found.items()):
        beta, g, seed = key
        row, array = _row(f"b{beta:g}_g{g:g}_s{seed}", root.resolve(), summary)
        row["core_mean_hz"] = float(summary["core_modulation"]["mean_hz"])
        row["core_rho80_active_fraction"] = float(summary["core_rho80_active_fraction"])
        row["credible_carrier"] = credible_carrier(row)
        row["beta_SG"], row["persistent_g"], row["som_seed"] = beta, g, seed
        rate = np.asarray(array["fine_core_rate_hz"], float)
        row["sustained_core_cv"] = sustained_core_cv(rate)
        row["modulation_band"] = modulation_band(row["sustained_core_cv"])
        peak_hz, prominence = spectral_peak(rate[-500:], fs=500.0)
        row["sustained_peak_hz"], row["sustained_peak_prominence"] = peak_hz, prominence
        sub = array.get("trace_Isub_mean")
        post = array.get("trace_Irec_postdiv_mean")
        if sub is not None and post is not None:
            tail = slice(sub.size - 10000, None)
            row["realized_removal_ratio"] = realized_removal_ratio(
                sub[tail], post[tail]
            )
        else:
            row["realized_removal_ratio"] = 0.0
        rows[key] = row

    verdict = adjudicate(list(rows.values()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subtractive_pool_carrier_summary.json").write_text(
        json.dumps(
            {
                "schema": "topic4_zm_subtractive_pool_carrier_v1_2026-08-04",
                "verdict": verdict,
                "rows": {f"b{b:g}_g{g:g}_s{s}": r for (b, g, s), r in rows.items()},
            },
            indent=2, sort_keys=True, allow_nan=False,
        ) + "\n"
    )

    panels = (
        ("sustained_core_cv", "sustained core-rate CV", (TONIC_CEILING, CLEAN_FLOOR)),
        ("post_onset_deep_gap_fraction", "deep-gap fraction", (0.20,)),
        ("energy_occupancy_6db", "energy occupancy", (0.50,)),
        ("median_vseeg_gain_db", "virtual-SEEG gain (dB)", (20.0,)),
        ("spatial_pc1", "spatial PC1 fraction", (0.95,)),
        ("sustained_peak_prominence", "spectral peak prominence (no gate)", ()),
    )
    colors = {(0.0, 1): "#8a8a8a", (0.32, 1): "#d95f45",
              (0.32, 2): "#2b7a5b", (0.32, 3): "#4a6fb5"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for ax, (field, label, gates) in zip(axes.ravel(), panels):
        for (g, seed), color in colors.items():
            pts = sorted(
                (r["beta_SG"], r) for r in rows.values()
                if r["persistent_g"] == g and r["som_seed"] == seed
            )
            if not pts:
                continue
            xs = [b for b, _ in pts]
            ys = [r[field] if r[field] is not None else np.nan for _, r in pts]
            ok = [r["credible_carrier"] for _, r in pts]
            ax.plot(xs, ys, "-", color=color, lw=1.1,
                    label=f"g={g:g}, wiring {seed}")
            ax.scatter(xs, ys, s=[46 if k else 16 for k in ok],
                       facecolors=[color if k else "white" for k in ok],
                       edgecolors=color, zorder=3)
        for gate in gates:
            ax.axhline(gate, color="#333", ls=":", lw=1)
        ax.set(xlabel="subtractive pool strength", ylabel=label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(verdict["verdict"], fontsize=13)
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "subtractive_pool_dose_response.png", dpi=170)
    plt.close(fig)

    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    marker = "### subtractive_pool_dose_response.png"
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "把判据分别对共享抑制池的减法强度作图，灰线是不加慢兴奋的衬底、"
            "红线是加了慢兴奋的衬底，绿蓝两条是同强度换连接布线的复现。"
            "实心点表示该点过了原有七条判据。第一格是末段放电起伏"
            "（两条虚线分别是「算平线」与「算干净有节律」的分界），"
            "最后一格不是判据，是频谱峰的突出度——起伏大也可能只是漂移，"
            "这一格才说明是不是有一个频率在主导。\n\n"
            "**关注点**：红线的起伏抬起来时，深间隙、能量占空、场增益是不是"
            "还同时守得住；灰线是否始终填不平间隙（那说明连续性只能由慢兴奋提供）。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
