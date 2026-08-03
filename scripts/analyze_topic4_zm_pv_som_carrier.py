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
ORDER = ("current_H250", "current_H1500", "contrast_H1500", "SOM_shunt")


def _close(a, b):
    return a is not None and np.isclose(float(a), float(b))


def _label(summary):
    mech = summary.get("mechanism", {})
    subtype = mech.get("pv_som_inhibitory_subtypes")
    mode = mech.get("state_selective_mode_H") or {}
    if not subtype or not (
        summary.get("state") == "bounded_late__peak"
        and _close(summary.get("T_ms"), 2500.0)
        and _close(mode.get("rho_mode_H"), 0.5)
        and _close(subtype.get("som_source_fraction_realized"), 0.25)
        and _close(subtype.get("som_slow_integrated_budget_fraction"), 0.35)
        and _close(subtype.get("som_recruit_delay_scale"), 3.0)
    ):
        return None
    down = float(mode.get("tau_mode_H_down", mode.get("tau_mode_H", 250.0)))
    common = float(mode.get("mode_H_common_subtraction", 0.0))
    if subtype.get("slow_membrane_mode") == "shunt":
        return "SOM_shunt" if _close(down, 250.0) else None
    if _close(down, 250.0) and _close(common, 0.0):
        return "current_H250"
    if _close(down, 1500.0) and _close(common, 0.0):
        return "current_H1500"
    if _close(down, 1500.0) and _close(common, 1.0):
        return "contrast_H1500"
    return None


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
    missing = sorted(set(ORDER).difference(found))
    if missing:
        raise RuntimeError(f"PV/SOM panel incomplete: {missing}")

    rows, arrays = {}, {}
    for label in ORDER:
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
        rows[label], arrays[label] = row, array

    passing = [key for key, row in rows.items() if row["credible_carrier"]]
    verdict = {
        "verdict": "PV_SOM_CARRIER_CANDIDATE" if passing else "PV_SOM_SPATIAL_PATTERN_WITH_TEMPORAL_GAPS",
        "passing_arms": passing,
        "next_coordinate": (
            "none; advance to durability/M" if passing else
            "the PV/SOM relaxation-cycle timescale, not H amplitude or membrane form"
        ),
        "claim_boundary": "seed-1 frozen-Z/M 2.5-s mechanism panel",
    }
    payload = {
        "schema": "topic4_zm_pv_som_carrier_v1_2026-08-03",
        "verdict": verdict,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pv_som_carrier_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(4, 3, figsize=(15, 10.5), constrained_layout=True)
    for ir, label in enumerate(ORDER):
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
        if ir == 0:
            axes[ir, 2].legend(frameon=False, fontsize=8)
    fig.suptitle(verdict["verdict"], fontsize=15)
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "pv_som_carrier_panel.png", dpi=170)
    plt.close(fig)

    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    marker = "### pv_som_carrier_panel.png"
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "比较 PV/SOM 独立抑制亚群下的原 H、慢衰减 H、去 common-mode H 与 SOM conductance。"
            "左列量化 burst/gap，中央展示轴向空间自由度，右列展示 H 与共享抑制状态。\n\n"
            "**关注点**：PV/SOM 是否在保持低 PC1 的同时把事件间深间隙缩短到连续 ictal-energy 范围。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
