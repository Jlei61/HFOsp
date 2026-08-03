#!/usr/bin/env python3
"""Adjudicate the targeted Z-gated conductance/H/M lifecycle prototype."""
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


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
SMOKE_ORDER = ("noH", "H_M30", "H_M45", "H_M60", "H_Moff")


def _close(a, b):
    return a is not None and np.isclose(float(a), float(b))


def _label(summary: dict) -> str | None:
    mech = summary.get("mechanism", {})
    hom = mech.get("state_dependent_conductance_homotopy")
    dep = mech.get("i2e_depression", {})
    delay = mech.get("i2e_delay_rescaling", {})
    slow = mech.get("dynamic_slow_flow", {})
    if not hom or not (
        mech.get("arm") == "i2e"
        and _close(dep.get("tau_D_ms"), 300.7)
        and _close(dep.get("d_star_nominal"), 0.7281)
        and _close(delay.get("scale"), 3.0)
        and _close(hom.get("cond_homotopy_z_native"), 0.52)
        and _close(hom.get("cond_homotopy_z_conductance"), 0.48)
        and _close(hom.get("cond_gamma"), 1.0 / 6.0)
    ):
        return None
    mode = mech.get("state_selective_mode_H")
    if mode is None:
        return "noH" if _close(slow.get("g_M"), 0.0) else None
    if not (
        _close(mode.get("rho_mode_H"), 2.0)
        and _close(mode.get("tau_mode_H"), 250.0)
        and _close(mode.get("tau_mode_H_down"), 1500.0)
    ):
        return None
    half = float(mode["m_mode_half"])
    if half > 1e8 and _close(slow.get("g_M"), 0.0):
        return "H_Moff"
    if _close(slow.get("g_M"), 1.0) and any(
        _close(half, value) for value in (30.0, 45.0, 60.0)
    ):
        return f"H_M{int(half)}"
    return None


def credible_carrier(row: dict) -> bool:
    """Locked macro, energy, spatial and non-saturation carrier contract."""
    return bool(
        row.get("episode_status") not in {"no_onset", "no_sustained_onset"}
        and not row.get("runaway", False)
        and row.get("median_vseeg_gain_db") is not None
        and row["median_vseeg_gain_db"] >= 20.0
        and row.get("energy_occupancy_6db") is not None
        and row["energy_occupancy_6db"] >= 0.50
        and row.get("post_onset_deep_gap_fraction") is not None
        and row["post_onset_deep_gap_fraction"] <= 0.20
        and row.get("spatial_pc1") is not None
        and row["spatial_pc1"] <= 0.95
        and row.get("core_mean_hz", np.inf) <= 250.0
        and row.get("core_rho80_active_fraction", np.inf) <= 0.25
    )


def adjudicate(rows: dict[str, dict], long_row: dict | None) -> dict:
    credible_smoke = [key for key, row in rows.items() if credible_carrier(row)]
    if long_row is None:
        verdict = (
            "CARRIER_CANDIDATE_AWAITS_LONG_RUN"
            if credible_smoke else "NO_CREDIBLE_CARRIER_IN_SHORT_PANEL"
        )
    elif not credible_carrier(long_row):
        verdict = "NO_DURABLE_CREDIBLE_ICTAL_CARRIER"
    elif long_row.get("offset_ms") is None:
        verdict = "DURABLE_CARRIER_WITHOUT_NATIVE_OFFSET"
    elif long_row.get("returning_event") or long_row.get("returning_distribution"):
        verdict = "LIFECYCLE_CANDIDATE_SEED1"
    else:
        verdict = "NATIVE_OFFSET_WITHOUT_INTERICTAL_RETURN"
    return {
        "verdict": verdict,
        "credible_short_arms": credible_smoke,
        "long_run_present": long_row is not None,
        "claim_boundary": (
            "seed-1 targeted mechanism screen; a lifecycle claim additionally "
            "requires M-causal offset, postictal return and locked-seed replication"
        ),
    }


def _load_panel():
    found = {}
    for root in sorted((IN / "smoke/seed1").glob("*condhom_z0.52to0.48*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        if not _close(summary.get("T_ms"), 2500.0):
            continue
        label = _label(summary)
        if label is not None:
            found[label] = (root, summary)
    missing = sorted(set(SMOKE_ORDER).difference(found))
    if missing:
        raise RuntimeError(f"homotopy short panel incomplete: {missing}")
    rows, arrays = {}, {}
    for label in SMOKE_ORDER:
        root, summary = found[label]
        rows[label], arrays[label] = _row(label, root, summary)
        rows[label]["core_mean_hz"] = float(summary["core_modulation"]["mean_hz"])
        rows[label]["core_rho80_active_fraction"] = float(
            summary["core_rho80_active_fraction"]
        )
        for key in ("trace_cond_lambda_core_mean", "trace_mode_H_gain_core_mean"):
            values = arrays[label].get(key, np.zeros(1))
            rows[label][f"{key}_peak"] = float(np.max(values))
            rows[label][f"{key}_final"] = float(values[-1])
    return found, rows, arrays


def _load_long():
    matches = []
    for root in sorted((IN / "lifecycle_sprint/seed1").glob("*condhom_z0.52to0.48*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        if _close(summary.get("T_ms"), 12000.0) and _label(summary) == "H_M45":
            matches.append((root, summary))
    if len(matches) > 1:
        raise RuntimeError("duplicate long H_M45 homotopy run")
    if not matches:
        return None, None
    root, summary = matches[0]
    row, arrays = _row("H_M45_long", root, summary)
    row["core_mean_hz"] = float(summary["core_modulation"]["mean_hz"])
    row["core_rho80_active_fraction"] = float(
        summary["core_rho80_active_fraction"]
    )
    return row, arrays


def main():
    _, rows, arrays = _load_panel()
    long_row, long_arrays = _load_long()
    verdict = adjudicate(rows, long_row)
    payload = {
        "schema": "topic4_zm_conductance_homotopy_v1_2026-08-03",
        "verdict": verdict,
        "short_rows": rows,
        "long_row": long_row,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conductance_homotopy_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    labels = list(SMOKE_ORDER) + (["H_M45_long"] if long_arrays is not None else [])
    all_arrays = {**arrays, **({"H_M45_long": long_arrays} if long_arrays is not None else {})}
    all_rows = {**rows, **({"H_M45_long": long_row} if long_row is not None else {})}
    fig, axes = plt.subplots(len(labels), 4, figsize=(17, 2.5 * len(labels)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for ir, label in enumerate(labels):
        a, row = all_arrays[label], all_rows[label]
        tf = a["fine_time_ms"] / 1000.0
        axes[ir, 0].plot(tf, a["fine_core_rate_hz"], lw=.65, color="#d95f45")
        axes[ir, 0].set(xlabel="time (s)", ylabel=f"{label}\ncore Hz")
        axes[ir, 0].set_title(
            f"occ={row['energy_occupancy_6db']:.3f}; gap={row['post_onset_deep_gap_fraction']:.3f}"
        )
        kymo = a["coarse_kymo_axial"]
        axes[ir, 1].imshow(kymo, origin="lower", aspect="auto", cmap="magma",
                           extent=[0, .025 * kymo.shape[1], 0, kymo.shape[0]])
        axes[ir, 1].set(xlabel="time (s)", ylabel="axis bin",
                        title=f"PC1={row['spatial_pc1']:.3f}; rank={row['spatial_effective_rank']:.2f}")
        dt_s = 0.0001
        ts = np.arange(a["trace_z_core_mean"].size) * dt_s
        axes[ir, 2].plot(ts, a["trace_z_core_mean"], label="z core")
        axes[ir, 2].plot(ts, a["trace_m_core_mean"] / 250.0, label="m core/250")
        axes[ir, 2].set(xlabel="time (s)", ylabel="native slow state")
        axes[ir, 3].plot(ts, a["trace_cond_lambda_core_mean"], label="lambda core")
        if "trace_mode_H_gain_core_mean" in a:
            axes[ir, 3].plot(ts, a["trace_mode_H_gain_core_mean"], label="H gain core")
        axes[ir, 3].set(xlabel="time (s)", ylabel="state-selective gain")
        if ir == 0:
            axes[ir, 2].legend(frameon=False, fontsize=8)
            axes[ir, 3].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Z-gated conductance/H/M pilot: {verdict['verdict']}", fontsize=15)
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "conductance_homotopy_lifecycle.png", dpi=170)
    plt.close(fig)
    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    marker = "### conductance_homotopy_lifecycle.png"
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "比较 Z 门控 conductance、局部 H 与不同 M 关闭强度下的核心放电、轴向时空图和慢变量。"
            "短臂用于定位碎片态与 common-mode plateau 之间的平衡，长臂用于检验内生退出和间期返回。\n\n"
            "**关注点**：持续能量、低深间隙、非共模空间结构和非饱和必须同时成立；单独高放电或 PC1 下降均不算 lifecycle。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
