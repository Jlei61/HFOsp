#!/usr/bin/env python3
"""Adjudicate the three targeted fast-modal alternatives after scalar NO-GO."""
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
ORDER = ("common_H", "delay_dispersion", "dual_GABA", "contrast_H")


def _close(a, b):
    return a is not None and np.isclose(float(a), float(b))


def _label(summary: dict) -> str | None:
    mech = summary.get("mechanism", {})
    mode = mech.get("state_selective_mode_H") or {}
    dep = mech.get("i2e_depression") or {}
    delay = mech.get("i2e_delay_rescaling") or {}
    slow = mech.get("dynamic_slow_flow") or {}
    if not (
        mech.get("arm") == "i2e"
        and _close(dep.get("tau_D_ms"), 300.7)
        and _close(dep.get("d_star_nominal"), 0.7281)
        and _close(summary.get("T_ms"), 2500.0)
    ):
        return None
    dual = mech.get("dual_scale_i2e_gaba")
    common = float(mode.get("mode_H_common_subtraction", 0.0))
    source_cv = float(delay.get("source_delay_cv_requested", 0.0))
    if (
        dual
        and summary.get("state") == "bounded_late__peak"
        and bool(summary.get("freeze_policy", {}).get("freeze_z", False))
        and _close(mode.get("rho_mode_H"), 0.5)
    ):
        return "dual_GABA"
    hom = mech.get("state_dependent_conductance_homotopy")
    if not (
        hom
        and _close(delay.get("scale"), 3.0)
        and _close(slow.get("g_M"), 0.0)
        and float(mode.get("m_mode_half", 0.0)) > 1e8
    ):
        return None
    if _close(mode.get("rho_mode_H"), 2.0) and _close(common, 0.0):
        return "delay_dispersion" if _close(source_cv, 0.5) else "common_H"
    if (
        _close(mode.get("rho_mode_H"), 3.5)
        and _close(common, 1.0)
        and _close(source_cv, 0.0)
    ):
        return "contrast_H"
    return None


def load_rows():
    found = {}
    for root in sorted(IN.glob("*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        label = _label(summary)
        if label is not None:
            if label in found:
                raise RuntimeError(f"duplicate modal arm: {label}")
            found[label] = (root, summary)
    missing = sorted(set(ORDER).difference(found))
    if missing:
        raise RuntimeError(f"modal fast panel incomplete: {missing}")
    rows, arrays = {}, {}
    for label in ORDER:
        root, summary = found[label]
        rows[label], arrays[label] = _row(label, root, summary)
        rows[label]["core_mean_hz"] = float(summary["core_modulation"]["mean_hz"])
        rows[label]["core_rho80_active_fraction"] = float(
            summary["core_rho80_active_fraction"]
        )
        rows[label]["credible_carrier"] = credible_carrier(rows[label])
    return rows, arrays


def adjudicate(rows):
    passed = [label for label, row in rows.items() if row["credible_carrier"]]
    return {
        "verdict": (
            "FAST_MODAL_CARRIER_CANDIDATE_SEED1"
            if passed else "NO_CREDIBLE_CARRIER_IN_TARGETED_FAST_MODAL_PANEL"
        ),
        "passing_arms": passed,
        "gate": {
            "gain_db_min": 20.0,
            "occupancy_min": 0.50,
            "deep_gap_max": 0.20,
            "spatial_pc1_max": 0.95,
            "core_mean_hz_max": 250.0,
            "core_rho80_max": 0.25,
            "runaway_forbidden": True,
        },
        "claim_boundary": (
            "seed-1 targeted 2.5-s carrier screen; even a pass would require "
            "a durable run, locked seed and M-causal offset before lifecycle language"
        ),
    }


def main():
    rows, arrays = load_rows()
    verdict = adjudicate(rows)
    payload = {
        "schema": "topic4_zm_targeted_fast_modal_panel_v1_2026-08-03",
        "verdict": verdict,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "targeted_fast_modal_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(4, 3, figsize=(15, 10.5), constrained_layout=True)
    for ir, label in enumerate(ORDER):
        a, row = arrays[label], rows[label]
        axes[ir, 0].plot(a["fine_time_ms"] / 1000.0, a["fine_core_rate_hz"],
                         lw=0.7, color="#d95f45")
        axes[ir, 0].set(xlabel="time (s)", ylabel=f"{label}\ncore Hz")
        axes[ir, 0].set_title(
            f"gain {row['median_vseeg_gain_db']:.1f} dB; occ {row['energy_occupancy_6db']:.2f}; "
            f"gap {row['post_onset_deep_gap_fraction']:.2f}"
        )
        kymo = a["coarse_kymo_axial"]
        axes[ir, 1].imshow(kymo, origin="lower", aspect="auto", cmap="magma",
                           extent=[0, 0.025 * kymo.shape[1], 0, kymo.shape[0]])
        axes[ir, 1].set(xlabel="time (s)", ylabel="axis bin",
                        title=f"PC1 {row['spatial_pc1']:.3f}; rank {row['spatial_effective_rank']:.2f}")
        values = [row["core_mean_hz"], 100 * row["core_rho80_active_fraction"]]
        axes[ir, 2].bar(["core Hz", "rho80 %"], values,
                        color=["#4c78a8", "#72b7b2"])
        axes[ir, 2].axhline(250, color="0.5", ls="--", lw=0.8)
        axes[ir, 2].set_title("PASS" if row["credible_carrier"] else "FAIL")
    fig.suptitle(verdict["verdict"], fontsize=15)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "targeted_fast_modal_mechanisms.png", dpi=170)
    plt.close(fig)

    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    marker = "### targeted_fast_modal_mechanisms.png"
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "统一比较原 common-mode H、抑制到达相位分散、局部快/宽程慢双 GABA 和去 common-mode 的 contrast-H。"
            "每一行同时展示核心放电、轴向时空图和饱和指标，并使用同一 carrier gate 判定。\n\n"
            "**关注点**：高能量、时间连续、非 rank-1 空间结构和非饱和必须同时成立；单项改善不算 ictal carrier。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
