#!/usr/bin/env python3
"""Compare the targeted PV/SOM carrier arms under one locked metric contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_topic4_zm_mode_h_pilot import _row  # noqa: E402
from scripts.analyze_topic4_zm_conductance_homotopy import credible_carrier  # noqa: E402


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/smoke/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
BASE_ORDER = (
    "current_H250",
    "current_SOM30",
    "current_H1500",
    "contrast_H1500",
    "SOM_shunt",
)


def _close(a, b):
    return a is not None and np.isclose(float(a), float(b))


def _label(summary):
    mech = summary.get("mechanism", {})
    subtype = mech.get("pv_som_inhibitory_subtypes")
    mode = mech.get("state_selective_mode_H") or {}
    if not subtype or not (
        summary.get("state") == "bounded_late__peak"
        and _close(summary.get("T_ms"), 2500.0)
        and _close(subtype.get("som_source_fraction_realized"), 0.25)
        and _close(subtype.get("som_slow_integrated_budget_fraction"), 0.35)
        and _close(subtype.get("som_recruit_delay_scale"), 3.0)
    ):
        return None
    down = float(mode.get("tau_mode_H_down", mode.get("tau_mode_H", 250.0)))
    common = float(mode.get("mode_H_common_subtraction", 0.0))
    som_tau_d = float(subtype.get("tau_d_som_ms", 60.0))
    seed = int(subtype.get("seed", 1))
    # Runs predating the persistent-conductance commit carry no such key; 0 is
    # the literal parity path, so the default is the correct historical value.
    g_max = float(mode.get("mode_H_persistent_g_max", 0.0))
    # rho=0 is the persistent-conductance dose series; g=0 is its matched
    # control, where the H sensor runs but neither slow-excitation path couples.
    if _close(mode.get("rho_mode_H"), 0.0):
        if not (
            _close(mode.get("mode_H_persistent_e_exc"), 60.0)
            and _close(down, 250.0)
            and _close(common, 0.0)
            and _close(som_tau_d, 60.0)
            and subtype.get("slow_membrane_mode") != "shunt"
        ):
            return None
        return f"persistent_g{g_max:g}" + ("" if seed == 1 else f"__som{seed}")
    # The multiplicative comparison panel only ever ran on the seed-1 wiring.
    if seed != 1 or not _close(mode.get("rho_mode_H"), 0.5):
        return None
    if subtype.get("slow_membrane_mode") == "shunt":
        return "SOM_shunt" if _close(down, 250.0) and _close(som_tau_d, 60.0) else None
    if _close(down, 250.0) and _close(common, 0.0) and _close(som_tau_d, 30.0):
        return "current_SOM30"
    if _close(down, 250.0) and _close(common, 0.0) and _close(som_tau_d, 60.0):
        return "current_H250"
    if _close(down, 1500.0) and _close(common, 0.0) and _close(som_tau_d, 60.0):
        return "current_H1500"
    if _close(down, 1500.0) and _close(common, 1.0) and _close(som_tau_d, 60.0):
        return "contrast_H1500"
    return None


def sustained_core_cv(fine_core_rate_hz, window_ms=1000.0, bin_ms=2.0):
    """Relative variability of the core rate over the final second.

    The locked gate reads a perfectly flat rate as perfectly continuous, so a
    fixed point and a gap-free carrier score identically on it.  This is the
    diagnostic that tells them apart; it is reported, never gated on.
    """
    rate = np.asarray(fine_core_rate_hz, float)[-int(window_ms / bin_ms):]
    mean = float(rate.mean()) if rate.size else 0.0
    return float(rate.std() / mean) if mean > 0.0 else None


def _split_arm(label):
    """Recover (persistent conductance dose, SOM wiring seed) from an arm label."""
    dose, _, seed = label.removeprefix("persistent_g").partition("__som")
    return float(dose), int(seed) if seed else 1


def _gap_spatial_class(row):
    """Mechanical readout of the two carrier failure axes; no hand-assigned label."""
    if row["runaway"]:
        return "runaway"
    if row["post_onset_deep_gap_fraction"] is None or row["spatial_pc1"] is None:
        return "no_episode"
    filled = row["post_onset_deep_gap_fraction"] <= 0.20
    common = row["spatial_pc1"] > 0.95
    if filled and common:
        return "common_mode_plateau"
    if filled:
        return "gaps_filled_spatially_distributed"
    return "common_mode_fragmented" if common else "fragmented_spatially_distributed"


def adjudicate(rows):
    """Adjudicate the locked panel; a pass on one wiring is not yet a carrier."""
    persistent = sorted(
        (key for key in rows if key.startswith("persistent_g")), key=_split_arm
    )
    passing = [
        key for key in list(BASE_ORDER) + persistent
        if key in rows and rows[key]["credible_carrier"]
    ]
    replication = {}
    for key in persistent:
        dose, seed = _split_arm(key)
        entry = replication.setdefault(
            f"g{dose:g}",
            {"seeds_tested": [], "seeds_passing_gate": [], "classes": {}},
        )
        entry["seeds_tested"].append(seed)
        if rows[key]["credible_carrier"]:
            entry["seeds_passing_gate"].append(seed)
        entry["classes"][str(seed)] = rows[key]["gap_spatial_class"]

    passing_doses = sorted(
        {_split_arm(key)[0] for key in passing if key.startswith("persistent_g")}
    )
    dosed = [key for key in persistent if _split_arm(key)[0] > 0.0]
    wirings = sorted({_split_arm(key)[1] for key in dosed})
    # A wiring carries if some dose clears the gate on it, which is a weaker
    # and different claim from one dose clearing the gate on every wiring.
    passing_dose_per_wiring = {}
    for key in passing:
        if not key.startswith("persistent_g"):
            continue
        dose, seed = _split_arm(key)
        passing_dose_per_wiring[str(seed)] = min(
            dose, passing_dose_per_wiring.get(str(seed), dose)
        )
    unsupported = [w for w in wirings if str(w) not in passing_dose_per_wiring]
    if passing_doses:
        # One wiring makes "passes on every tested wiring" vacuously true.
        transferable = [] if len(wirings) < 2 else [
            dose for dose in passing_doses
            if len(replication[f"g{dose:g}"]["seeds_passing_gate"]) == len(wirings)
        ]
        if transferable:
            headline = "PERSISTENT_SLOW_EXCITATION_CARRIER_REPLICATES_ACROSS_SUBSTRATES"
            coordinate = "none; advance to 12-s durability at the weakest replicated strength"
        elif unsupported:
            headline = "PERSISTENT_SLOW_EXCITATION_CARRIER_IS_SUBSTRATE_DEPENDENT"
            coordinate = "at least one wiring has no passing dose; do not advance to M"
        elif len(wirings) > 1:
            headline = (
                "PERSISTENT_SLOW_EXCITATION_CARRIER_REPLICATES_AT_A_WIRING_SPECIFIC_DOSE"
            )
            coordinate = "every wiring carries but the passing dose moves; durability decides"
        else:
            headline = "PERSISTENT_SLOW_EXCITATION_CARRIER_CANDIDATE"
            coordinate = "none; advance to 12-s durability at the weakest passing strength"
    elif passing:
        headline = "PV_SOM_CARRIER_CANDIDATE"
        coordinate = "none; advance to durability/M"
    elif dosed and all(rows[key]["gap_spatial_class"] == "common_mode_plateau"
                       for key in dosed):
        headline = "PERSISTENT_SLOW_EXCITATION_FILLS_GAPS_AS_COMMON_MODE"
        coordinate = "stop this mechanism; gap filling and spatial freedom are anti-correlated here"
    else:
        headline = "PV_SOM_SPATIAL_PATTERN_WITH_TEMPORAL_GAPS"
        coordinate = "the PV/SOM relaxation-cycle timescale, not H amplitude or membrane form"
    return {
        "verdict": headline,
        "passing_arms": passing,
        "weakest_passing_dose": passing_doses[0] if passing_doses else None,
        "passing_dose_per_wiring": passing_dose_per_wiring,
        "wirings_without_a_passing_dose": unsupported,
        "seed_replication": replication,
        "persistent_arm_classes": {
            key: rows[key]["gap_spatial_class"] for key in persistent
        },
        "next_coordinate": coordinate,
        "claim_boundary": "seed-1 frozen-Z/M 2.5-s mechanism panel",
    }


def _dose_response_figure(rows, path):
    """Plot each gate against dose, so the surviving window is visible at all."""
    series = {}
    for key in rows:
        if not key.startswith("persistent_g"):
            continue
        dose, seed = _split_arm(key)
        series.setdefault(seed, []).append((dose, rows[key]))
    for points in series.values():
        points.sort()
    panels = (
        ("post_onset_deep_gap_fraction", "deep-gap fraction", 0.20, "below"),
        ("median_vseeg_gain_db", "virtual-SEEG gain (dB)", 20.0, "above"),
        ("energy_occupancy_6db", "energy occupancy", 0.50, "above"),
        ("spatial_pc1", "spatial PC1 fraction", 0.95, "below"),
        # Not a gate: the gate above scores a fixed point as perfectly
        # continuous, so this is what separates a carrier from a frozen state.
        ("sustained_core_cv", "sustained core-rate CV (no gate)", None, None),
    )
    colors = {1: "#d95f45", 2: "#2b7a5b", 3: "#4a6fb5"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), constrained_layout=True)
    axes.ravel()[-1].axis("off")
    for ax, (field, label, gate, side) in zip(axes.ravel(), panels):
        for seed, points in sorted(series.items()):
            doses = [dose for dose, _ in points]
            values = [row[field] for _, row in points]
            passing = [row["credible_carrier"] for _, row in points]
            ax.plot(doses, values, "-", color=colors.get(seed, "#777"), lw=1.1,
                    label=f"SOM wiring {seed}")
            ax.scatter(doses, values, s=[46 if ok else 16 for ok in passing],
                       facecolors=[colors.get(seed, "#777") if ok else "white"
                                   for ok in passing],
                       edgecolors=colors.get(seed, "#777"), zorder=3)
        if gate is not None:
            ax.axhline(gate, color="#333", ls=":", lw=1)
            ax.annotate(f"gate ({'≤' if side == 'below' else '≥'} {gate:g})",
                        xy=(0.99, gate), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom" if side == "above" else "top",
                        fontsize=8, color="#333")
        ax.set(xlabel="persistent slow excitatory conductance g", ylabel=label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "filled marker = clears every gate simultaneously", fontsize=12
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _run_lengths(mask):
    x = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.pad(x, (1, 1)))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return stops - starts


def main():
    found = {}
    for root in sorted(IN.glob("*pvSOM*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        label = _label(summary)
        if label is not None:
            if label in found:
                raise RuntimeError(f"duplicate PV/SOM arm: {label}")
            found[label] = (root, summary)
    missing = sorted(set(BASE_ORDER).difference(found))
    if missing:
        raise RuntimeError(f"PV/SOM panel incomplete: {missing}")
    persistent = sorted(
        (key for key in found if key.startswith("persistent_g")), key=_split_arm
    )
    if len(persistent) < 2:
        raise RuntimeError(f"persistent-H panel incomplete: {persistent}")
    # Replicate substrates are adjudicated but not drawn; the panel figure
    # compares mechanisms on one wiring so its rows stay like-for-like.
    order = [key for key in persistent if _split_arm(key)[1] == 1]
    order = list(BASE_ORDER) + order

    rows, arrays = {}, {}
    for label in list(BASE_ORDER) + persistent:
        root, summary = found[label]
        row, array = _row(label, root, summary)
        row["core_mean_hz"] = float(summary["core_modulation"]["mean_hz"])
        row["core_rho80_active_fraction"] = float(summary["core_rho80_active_fraction"])
        row["credible_carrier"] = credible_carrier(row)
        active = np.asarray(array["fine_core_rate_hz"]) >= 50.0
        on = _run_lengths(active) * 2.0
        off = _run_lengths(~active) * 2.0
        row["median_core_on_ms_at_50hz"] = float(np.median(on)) if on.size else 0.0
        row["median_core_off_ms_at_50hz"] = float(np.median(off)) if off.size else 0.0
        for field, key in (
            ("persistent_g_mean_peak", "trace_mode_H_persistent_g_mean"),
            ("persistent_g_max_peak", "trace_mode_H_persistent_g_max"),
            ("persistent_g_core_mean_peak", "trace_mode_H_persistent_g_core_mean"),
        ):
            row[field] = float(np.max(array.get(key, np.zeros(1))))
        row["gap_spatial_class"] = _gap_spatial_class(row)
        row["sustained_core_cv"] = sustained_core_cv(array["fine_core_rate_hz"])
        row["sustained_centroid_speed_bins_s"] = row["tail"][
            "centroid_median_speed_bins_s"
        ]
        rows[label], arrays[label] = row, array

    verdict = adjudicate(rows)
    payload = {
        "schema": "topic4_zm_pv_som_carrier_v1_2026-08-03",
        "verdict": verdict,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pv_som_carrier_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(
        len(order), 3, figsize=(15, 2.6 * len(order)), constrained_layout=True
    )
    for ir, label in enumerate(order):
        row, a = rows[label], arrays[label]
        axes[ir, 0].plot(a["fine_time_ms"] / 1000.0, a["fine_core_rate_hz"],
                         color="#d95f45", lw=0.7)
        axes[ir, 0].set(xlabel="time (s)", ylabel=f"{label}\ncore Hz")
        axes[ir, 0].set_title(
            f"occ {row['energy_occupancy_6db']:.2f}; gap {row['post_onset_deep_gap_fraction']:.2f}; "
            f"off50 {row['median_core_off_ms_at_50hz']:.0f} ms"
        )
        kymo = a["coarse_kymo_axial"]
        axes[ir, 1].imshow(kymo, origin="lower", aspect="auto", cmap="magma",
                           extent=[0, .025 * kymo.shape[1], 0, kymo.shape[0]])
        axes[ir, 1].set(xlabel="time (s)", ylabel="axis bin",
                        title=f"PC1 {row['spatial_pc1']:.3f}; rank {row['spatial_effective_rank']:.2f}")
        t = np.arange(a["trace_mode_H_gain_core_mean"].size) * 0.0001
        axes[ir, 2].plot(t, a["trace_mode_H_gain_core_mean"], label="H gain core")
        axes[ir, 2].plot(t, a["trace_S_G"], label="S_G", alpha=.8)
        axes[ir, 2].set(xlabel="time (s)", ylabel="feedback state")
        twin = axes[ir, 2].twinx()
        g_core = a.get("trace_mode_H_persistent_g_core_mean", np.zeros(t.size))
        twin.plot(t[:g_core.size], g_core, color="#2b7a5b", ls="--", lw=.9,
                  label="persistent g core")
        twin.set_ylabel("slow exc. g", color="#2b7a5b")
        twin.tick_params(axis="y", labelcolor="#2b7a5b")
        if ir == 0:
            axes[ir, 2].legend(frameon=False, fontsize=8, loc="upper left")
            twin.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(verdict["verdict"], fontsize=15)
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "pv_som_carrier_panel.png", dpi=170)
    plt.close(fig)
    _dose_response_figure(rows, fig_dir / "persistent_dose_response.png")

    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    marker = "### pv_som_carrier_panel.png"
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "在 PV/SOM 独立抑制亚群衬底上统一比较两类局部慢兴奋机制："
            "前五行是乘性 H（原 H、单点 SOM 衰减缩短、慢衰减 H、去 common-mode H、SOM conductance），"
            "后面各行是新的局部慢兴奋性 conductance（乘性项关闭，只靠亚阈值记忆填补 burst 间隙）。"
            "左列量化 burst/gap，中央展示轴向空间自由度，"
            "右列同时给出 H 增益、共享抑制状态与慢兴奋 conductance（右轴虚线）。\n\n"
            "**关注点**：填平深间隙与保住低 PC1 是否能同时成立；"
            "只把间隙填平却把空间结构压成共模平台，不算 ictal carrier。\n"
        )
    marker = "### persistent_dose_response.png"
    prior = readme.read_text()
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "前四格把四条判据分别对慢兴奋强度作图，每条连接种子一条线，"
            "实心点表示该强度同时满足全部判据，虚线是各自的判据线。"
            "第五格不是判据，是末段放电的相对起伏——"
            "判据把一条完全平的放电读成\"完美连续\"，只有这一格能把"
            "真正还在动的 carrier 和已经停住的不动点分开。\n\n"
            "**关注点**：时间连续性要求强度往上走，而场增益和能量占空要求强度往下走，"
            "先看这两侧把可行区间夹成多宽、不同连接种子的可行区间是否重叠；"
            "然后看所有实心点是不是都落在第五格的贴地段——"
            "若是，则\"填平间隙\"其实是靠消灭振荡换来的。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
