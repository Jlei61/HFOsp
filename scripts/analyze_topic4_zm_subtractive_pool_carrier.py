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
from scipy.signal import detrend, welch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_topic4_zm_mode_h_pilot import _row  # noqa: E402
from scripts.analyze_topic4_zm_conductance_homotopy import (  # noqa: E402
    _validate_short_prefix, credible_carrier,
)
from scripts.analyze_topic4_zm_pv_som_carrier import (  # noqa: E402
    realized_removal_ratio, sustained_core_cv,
)


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/smoke/seed1"
LONG = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
SUBSTRATE_LEVELS = (0.0, 0.32)
# Locked by the spec: below the first band the sustained rate is a flat line,
# above the second it is unambiguously modulated, and between them the panel
# cannot tell a rhythm from drift.
TONIC_CEILING = 0.10
CLEAN_FLOOR = 0.25


def _close(a, b):
    return a is not None and np.isclose(float(a), float(b))


def _arm_key(summary, duration_ms):
    """(subtractive strength, persistent conductance, SOM wiring) or None."""
    mech = summary.get("mechanism", {})
    subtype = mech.get("pv_som_inhibitory_subtypes")
    mode = mech.get("state_selective_mode_H") or {}
    if not subtype or summary.get("state") != "bounded_late__peak":
        return None
    g = float(mode.get("mode_H_persistent_g_max", 0.0))
    if not (
        _close(summary.get("T_ms"), duration_ms)
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


def arm_key(summary):
    """Arms of the 2.5 s factorial."""
    return _arm_key(summary, 2500.0)


def long_arm_key(summary):
    """Arms of the 12 s durability ladder.

    Not every long arm has a short counterpart — the strengths that bracket the
    branch change were only ever worth running long — and keying the long panel
    off the short one would silently drop them.
    """
    return _arm_key(summary, 12000.0)


def modulation_band(cv):
    """Three-band reading; an arm can never be both passing and ambiguous."""
    if cv is None:
        return None
    if cv < TONIC_CEILING:
        return "tonic"
    return "ambiguous" if cv <= CLEAN_FLOOR else "clean"


def spectral_peak(sustained_rate, *, fs, band=(1.0, 200.0), min_cycles=5.0):
    """Peak frequency of the sustained rate and how far it stands above the band.

    Reported, never gated on.  A ramp has no period, so the trend is removed
    before transforming; and a frequency the window holds only a cycle or two
    of is not a measurement, so the band starts at `min_cycles` per window.
    """
    rate = np.asarray(sustained_rate, float)
    if rate.size < 8:
        return None, 0.0
    scale = max(1.0, float(np.abs(rate).mean()))
    rate = detrend(rate, type="linear")
    # A pure ramp detrends to float residue, whose spectrum is arbitrary; a
    # residual this far below the signal scale is not a rhythm.
    if float(rate.std()) <= 1e-9 * scale:
        return None, 0.0
    nperseg = min(rate.size, 2048)
    freqs, power = welch(rate, fs=fs, nperseg=nperseg)
    low = max(band[0], min_cycles * fs / rate.size)
    inside = (freqs >= low) & (freqs <= band[1])
    if not inside.any() or power[inside].max() <= 0.0:
        return None, 0.0
    median = float(np.median(power[inside]))
    peak = int(np.argmax(power[inside]))
    prominence = float(power[inside][peak] / median) if median > 0.0 else float("inf")
    return float(freqs[inside][peak]), prominence


def cv_block_profile(rate, *, fs, block_ms):
    """Per-block relative variability, in order, over the whole trace.

    A 2.5 s window cannot tell a sustained rhythm from a transient relaxing
    back onto the fixed point.  This is the shape that can.
    """
    rate = np.asarray(rate, float)
    width = int(round(block_ms * fs / 1000.0))
    if width <= 1:
        raise ValueError("block shorter than one sample")
    out = []
    for start in range(0, rate.size - width + 1, width):
        block = rate[start:start + width]
        mean = float(block.mean())
        out.append(float(block.std() / mean) if mean > 0.0 else 0.0)
    return out


def modulation_amplitude_hz(rate):
    """Absolute size of the sustained modulation, in Hz.

    Relative peak prominence is misleading on its own: a nearly flat trace can
    still put a large peak above its own low-amplitude residual.  This says how
    big the swing actually is.
    """
    rate = np.asarray(rate, float)
    return float(detrend(rate, type="linear").std()) if rate.size > 1 else 0.0


def long_run_class(row):
    """Which of the branches the twelve-second arm actually settled into.

    Collapsing these would misreport the high-strength arms, which hold their
    modulation for the whole run rather than relaxing back onto the point.
    """
    profile = list(row["cv_block_profile"])
    if not profile:
        return "no_profile"
    started = modulation_band(max(profile)) == "clean"
    ended = modulation_band(profile[-1]) == "clean"
    if not ended:
        return "decays_to_tonic_fixed_point" if started else "tonic_throughout"
    if row["post_onset_deep_gap_fraction"] > 0.20:
        return "persistent_deep_gap_burst_train"
    return (
        "continuous_modulated_carrier" if row["credible_carrier"]
        else "persistent_modulated_below_gate"
    )


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

    long_rows = {}
    for root in sorted(LONG.glob("*pvSOM*")):
        if not (root / "summary.json").is_file() or not (root / "traces.npz").is_file():
            continue
        summary = json.loads((root / "summary.json").read_text())
        key = long_arm_key(summary)
        if key is None:
            continue
        short_root = found.get(key, (None,))[0]
        if short_root is not None:
            # No stitching and no clock shift: where a short arm exists it must
            # be the literal opening of the long one, or they are not the same
            # trajectory.
            _validate_short_prefix(short_root, root.resolve())
        beta, g, seed = key
        row, array = _row(f"long_b{beta:g}_g{g:g}_s{seed}", root.resolve(), summary)
        row["core_mean_hz"] = float(summary["core_modulation"]["mean_hz"])
        row["core_rho80_active_fraction"] = float(
            summary["core_rho80_active_fraction"]
        )
        row["credible_carrier"] = credible_carrier(row)
        row["beta_SG"], row["persistent_g"], row["som_seed"] = beta, g, seed
        rate = np.asarray(array["fine_core_rate_hz"], float)
        # Spec: the twelve-second arm reads clause 8 on its final two seconds.
        row["sustained_core_cv"] = sustained_core_cv(rate, window_ms=2000.0)
        row["modulation_band"] = modulation_band(row["sustained_core_cv"])
        row["cv_block_profile"] = cv_block_profile(rate, fs=500.0, block_ms=2000.0)
        peak_hz, prominence = spectral_peak(rate[-4000:], fs=500.0)
        row["sustained_peak_hz"], row["sustained_peak_prominence"] = peak_hz, prominence
        row["sustained_modulation_hz"] = modulation_amplitude_hz(rate[-4000:])
        row["long_run_class"] = long_run_class(row)
        row["short_arm_is_bit_exact_prefix"] = short_root is not None
        long_rows[key] = row

    verdict = adjudicate(list(rows.values()))
    verdict["durability"] = {
        f"b{b:g}_g{g:g}_s{s}": {
            "long_run_class": r["long_run_class"],
            "credible_carrier": r["credible_carrier"],
            "post_onset_deep_gap_fraction": r["post_onset_deep_gap_fraction"],
            "sustained_core_cv": r["sustained_core_cv"],
            "sustained_modulation_hz": r["sustained_modulation_hz"],
            "cv_block_profile": r["cv_block_profile"],
            "tail_label": r["tail"]["label"],
            "dmd_leading_hz": (
                None if not r["dmd"] else r["dmd"]["leading_mode"]["frequency_hz"]
            ),
        }
        for (b, g, s), r in sorted(long_rows.items())
    }
    # The twelve-second arms override the short verdict, and they do not all
    # land in the same place: weak strengths relax back onto the point while
    # strong ones hold their modulation as the deep-gap train the substrate
    # already had.  Reporting one label for both would misstate half the panel.
    # The short panel's candidates were read on a 2.5 s window; where the same
    # arm has a long run that settles flat, the candidacy is refuted by its own
    # trajectory and must not be left standing in the artifact.
    verdict["candidate_arms_refuted_by_long_run"] = [
        {"beta_SG": b, "som_seed": s, "long_run_class": long_rows[(b, g, s)]["long_run_class"]}
        for (b, g, s) in sorted(long_rows)
        if any(c["beta_SG"] == b and c["som_seed"] == s for c in verdict["candidate_arms"])
        and long_rows[(b, g, s)]["long_run_class"] != "continuous_modulated_carrier"
    ]
    main_long = {
        key: r for key, r in long_rows.items() if key[1] == 0.32 and key[2] == 1
    }
    if main_long:
        classes = {r["long_run_class"] for r in main_long.values()}
        verdict["long_run_classes_seen"] = sorted(classes)
        if "continuous_modulated_carrier" in classes:
            verdict["verdict"] = "SUBTRACTIVE_POOL_CONTINUOUS_MODULATED_CARRIER"
            verdict["next_coordinate"] = "replicate the carrier, then release Z/M"
        elif {"decays_to_tonic_fixed_point", "tonic_throughout"} & classes and (
            "persistent_deep_gap_burst_train" in classes
        ):
            verdict["verdict"] = "SUBTRACTIVE_POOL_SWITCHES_BRANCH_WITHOUT_AN_INTERMEDIATE_CARRIER"
            verdict["next_coordinate"] = (
                "weak strengths relax onto the tonic point and strong ones restore the "
                "deep-gap train; no strength held both continuity and modulation"
            )
        elif "persistent_deep_gap_burst_train" in classes:
            verdict["verdict"] = "SUBTRACTIVE_POOL_RESTORES_THE_DEEP_GAP_BURST_TRAIN"
            verdict["next_coordinate"] = "the modulation it sustains is the gappy train, not a carrier"
        else:
            verdict["verdict"] = "SUBTRACTIVE_POOL_MODULATION_IS_A_DECAYING_TRANSIENT"
            verdict["next_coordinate"] = (
                "every tested strength relaxes back onto the fixed point within 12 s"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subtractive_pool_carrier_summary.json").write_text(
        json.dumps(
            {
                "schema": "topic4_zm_subtractive_pool_carrier_v1_2026-08-04",
                "verdict": verdict,
                "rows": {f"b{b:g}_g{g:g}_s{s}": r for (b, g, s), r in rows.items()},
                "long_rows": {
                    f"b{b:g}_g{g:g}_s{s}": r for (b, g, s), r in long_rows.items()
                },
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
