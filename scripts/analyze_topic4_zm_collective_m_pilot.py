#!/usr/bin/env python3
"""Adjudicate the fixed-anchor collective-M divisor pilot.

This is intentionally a small mechanism-balance panel, not a broad sweep:
two H strengths by three M-divisor strengths, plus an H-off prevention control.
"""
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


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
ORDER = (
    "H05_mdiv2", "H05_mdiv3", "H05_mdiv4",
    "H1_mdiv2", "H1_mdiv3", "H1_mdiv4",
    "noH_mdiv4",
)


def _close(x, y) -> bool:
    return x is not None and np.isclose(float(x), float(y))


def _label(summary: dict) -> str | None:
    if not _close(summary.get("T_ms"), 12000.0):
        return None
    mech = summary.get("mechanism", {})
    dep = mech.get("i2e_depression", {})
    slow = mech.get("dynamic_slow_flow", {})
    div = mech.get("collective_mode_M_divisive")
    if not div or not (
        mech.get("arm") == "i2e"
        and _close(dep.get("tau_D_ms"), 300.7)
        and _close(dep.get("d_star_nominal"), 0.7281)
        and _close(mech.get("strength_scale"), 1.0)
        and _close(slow.get("g_M"), 1.0)
        and _close(slow.get("tau_M_ms"), 500.0)
        and _close(div.get("m_mode_div_ref"), 30.0)
        and _close(div.get("m_mode_div_power"), 4.0)
        and _close(div.get("m_mode_div_hill_power"), 4.0)
        and not bool(div.get("use_mode_M_memory", False))
    ):
        return None
    kappa = float(div["kappa_mode_M"])
    mode = mech.get("state_selective_mode_H")
    if mode is None and _close(kappa, 4.0):
        return "noH_mdiv4"
    if mode and _close(mode.get("m_mode_half"), 30.0):
        if not _close(mode.get("tau_mode_H_down", 250.0), 250.0):
            return None
        rho = float(mode.get("rho_mode_H"))
        if any(_close(kappa, value) for value in (2.0, 3.0, 4.0)):
            if _close(rho, 0.5):
                return f"H05_mdiv{int(kappa)}"
            if _close(rho, 1.0):
                return f"H1_mdiv{int(kappa)}"
    return None


def _entered(row: dict) -> bool:
    return row["episode_status"] not in {"no_onset", "no_sustained_onset"}


def _offset(row: dict) -> bool:
    return row.get("offset_ms") is not None and not row.get("runaway", False)


def _credible_carrier(row: dict) -> bool:
    """Macro onset is necessary but cannot override energy/spatial semantics."""
    gain = row.get("median_vseeg_gain_db")
    occupancy = row.get("energy_occupancy_6db")
    gap = row.get("post_onset_deep_gap_fraction")
    pc1 = row.get("spatial_pc1")
    return bool(
        _entered(row)
        and gain is not None and gain >= 20.0
        and occupancy is not None and occupancy >= 0.50
        and gap is not None and gap <= 0.20
        and pc1 is not None and pc1 <= 0.95
    )


def _verdict(rows: dict[str, dict]) -> dict:
    h_rows = [row for key, row in rows.items() if key != "noH_mdiv4"]
    macro_onset = [row for row in h_rows if _entered(row)]
    carrier = [row for row in h_rows if _credible_carrier(row)]
    offset = [row for row in carrier if _offset(row)]
    returned = [
        row for row in offset
        if row["returning_event"] or row["returning_distribution"]
    ]
    if not macro_onset:
        verdict = "COLLECTIVE_M_PREVENTS_ENTRY"
    elif not carrier:
        verdict = "NO_CREDIBLE_ICTAL_CARRIER"
    elif returned:
        verdict = "LIFECYCLE_CANDIDATE_SEED1"
    elif offset:
        verdict = "NATIVE_OFFSET_WITHOUT_INTERICTAL_RETURN"
    elif carrier:
        verdict = "ENTRY_WITHOUT_NATIVE_OFFSET"
    else:
        raise AssertionError("unreachable collective-M verdict state")
    return {
        "verdict": verdict,
        "n_H_arms_macro_onset": len(macro_onset),
        "n_H_arms_credible_carrier": len(carrier),
        "n_H_arms_offset": len(offset),
        "n_H_arms_returned": len(returned),
        "noH_control_entered": _entered(rows["noH_mdiv4"]),
        "claim_boundary": (
            "seed-1 seven-arm targeted mechanism panel; lifecycle needs a data-consistent carrier, durable offset, returning "
            "interictal statistics, healthy specificity, and locked-seed replication"
        ),
    }


def main() -> None:
    found = {}
    for root in sorted(IN.glob("*mdiv*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        label = _label(summary)
        if label is None:
            continue
        with np.load(tp, allow_pickle=False) as arrays:
            if "trace_mode_M_divisor" not in arrays.files:
                continue
        if label in found:
            raise RuntimeError(f"duplicate collective-M pilot arm: {label}")
        found[label] = (root, summary)
    missing = sorted(set(ORDER).difference(found))
    if missing:
        raise RuntimeError(f"collective-M pilot incomplete: {missing}")

    rows, arrays = {}, {}
    for label in ORDER:
        rows[label], arrays[label] = _row(label, *found[label])
        a = arrays[label]
        rows[label]["M_pool_peak"] = float(np.max(a["trace_mode_M_pool"]))
        rows[label]["M_divisor_peak"] = float(np.max(a["trace_mode_M_divisor"]))
        rows[label]["M_divisor_final"] = float(a["trace_mode_M_divisor"][-1])
    verdict = _verdict(rows)
    payload = {
        "schema": "topic4_zm_collective_M_divisor_pilot_v1_2026-08-03",
        "verdict": verdict,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "collective_M_divisor_pilot_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(len(ORDER), 4, figsize=(17, 2.7 * len(ORDER)), constrained_layout=True)
    for ir, label in enumerate(ORDER):
        row, a = rows[label], arrays[label]
        tf = a["fine_time_ms"] / 1000.0
        axes[ir, 0].plot(tf, a["fine_core_rate_hz"], lw=0.8, color="#d95f45")
        axes[ir, 0].set(ylabel=f"{label}\ncore Hz", xlabel="time (s)")
        axes[ir, 0].set_title(f"{row['episode_status']}; offset={row['offset_ms']}")
        kymo = a["coarse_kymo_axial"]
        axes[ir, 1].imshow(
            kymo, origin="lower", aspect="auto", cmap="magma",
            extent=[0, kymo.shape[1] * 0.025, 0, kymo.shape[0]],
        )
        axes[ir, 1].set(xlabel="time (s)", ylabel="axis bin", title=f"PC1={row['spatial_pc1']:.3f}")
        ts = np.arange(a["trace_z_core_mean"].size) * float(found[label][1]["dt_ms"]) / 1000.0
        axes[ir, 2].plot(ts, a["trace_z_core_mean"], label="z core", color="#2878b5")
        axes[ir, 2].plot(ts, a["trace_m_core_mean"] / 100.0, label="m core /100", color="#d95f45")
        if "trace_mode_H_max" in a:
            axes[ir, 2].plot(ts, a["trace_mode_H_max"], label="H max", color="#15803d")
        axes[ir, 2].set(xlabel="time (s)", ylabel="slow state")
        axes[ir, 3].plot(ts, a["trace_mode_M_divisor"], label="M divisor", color="#7e22ce")
        if "trace_mode_H_gain_max" in a:
            axes[ir, 3].plot(ts, a["trace_mode_H_gain_max"] + 1.0, label="1 + max H gain", color="#15803d")
        axes[ir, 3].set(xlabel="time (s)", ylabel="recurrent factor")
        if ir == 0:
            axes[ir, 2].legend(frameon=False, fontsize=8)
            axes[ir, 3].legend(frameon=False, fontsize=8)
    fig.suptitle(f"collective M recurrent brake: {verdict['verdict']}", fontsize=15)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "collective_M_divisor_pilot.png", dpi=170)
    plt.close(fig)
    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    marker = "### collective_M_divisor_pilot.png"
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "固定 fast anchor 上比较两档 H、三档 collective-M 及 H-off prevention 对照。"
            "四列依次为核心放电、病理轴时空图、Z/M/H 慢状态和 recurrent-E 的 M 分母。\n\n"
            "**关注点**：只有先进入有界高态、随后内生退出并恢复间期事件，才能升级为 lifecycle candidate；"
            "单纯阻止 onset 或降低瞬时放电都不算退出。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
