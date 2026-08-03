#!/usr/bin/env python3
"""Adjudicate the targeted I->E phase-lag lifecycle panel."""
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
from scripts.analyze_topic4_zm_collective_m_pilot import (  # noqa: E402
    _credible_carrier,
    _entered,
    _offset,
)


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
ORDER = ("delay2_H_Mmem", "delay3_H_Mmem", "delay3_H_noM", "delay3_noH_noM")


def _close(x, y) -> bool:
    return x is not None and np.isclose(float(x), float(y))


def _label(summary: dict) -> str | None:
    if not _close(summary.get("T_ms"), 12000.0):
        return None
    mech = summary.get("mechanism", {})
    dep, slow = mech.get("i2e_depression", {}), mech.get("dynamic_slow_flow", {})
    delay = mech.get("i2e_delay_rescaling")
    if not delay or not (
        mech.get("arm") == "i2e"
        and _close(dep.get("tau_D_ms"), 300.7)
        and _close(dep.get("d_star_nominal"), 0.7281)
        and _close(dep.get("tau_recovery_cv", 0.0), 0.0)
        and _close(slow.get("g_M"), 1.0)
        and _close(slow.get("tau_M_ms"), 500.0)
    ):
        return None
    scale = float(delay["scale"])
    mode = mech.get("state_selective_mode_H")
    div = mech.get("collective_mode_M_divisive")
    has_locked_h = bool(
        mode
        and _close(mode.get("rho_mode_H"), 2.0)
        and _close(mode.get("tau_mode_H"), 250.0)
        and _close(mode.get("tau_mode_H_down"), 1500.0)
        and _close(mode.get("m_mode_half"), 30.0)
    )
    has_locked_mem = bool(
        div
        and _close(div.get("kappa_mode_M"), 4.0)
        and bool(div.get("use_mode_M_memory", False))
        and _close(div.get("tau_mode_M_memory_up"), 3000.0)
        and _close(div.get("tau_mode_M_memory_down"), 8000.0)
    )
    if _close(scale, 2.0) and has_locked_h and has_locked_mem:
        return "delay2_H_Mmem"
    if _close(scale, 3.0) and has_locked_h and has_locked_mem:
        return "delay3_H_Mmem"
    if _close(scale, 3.0) and has_locked_h and div is None:
        return "delay3_H_noM"
    if _close(scale, 3.0) and mode is None and div is None:
        return "delay3_noH_noM"
    return None


def adjudicate(rows: dict[str, dict]) -> dict:
    credible = [key for key, row in rows.items() if _credible_carrier(row)]
    target = rows["delay3_H_Mmem"]
    open_m = rows["delay3_H_noM"]
    if not credible:
        verdict = "NO_CREDIBLE_ICTAL_CARRIER"
    elif _credible_carrier(target) and _offset(target):
        if target["returning_event"] or target["returning_distribution"]:
            verdict = "PHASE_LAG_LIFECYCLE_CANDIDATE_SEED1"
        else:
            verdict = "PHASE_LAG_NATIVE_OFFSET_NO_INTERICTAL_RETURN"
        if not (_credible_carrier(open_m) and not _offset(open_m)):
            verdict += "_M_ABLATION_UNRESOLVED"
    elif _credible_carrier(open_m) and not _credible_carrier(target):
        verdict = "M_MEMORY_PREVENTS_OR_FRAGMENTS_CARRIER"
    else:
        verdict = "CARRIER_WITHOUT_NATIVE_OFFSET"
    return {
        "verdict": verdict,
        "credible_carrier_arms": credible,
        "macro_onset_arms": [key for key, row in rows.items() if _entered(row)],
        "claim_boundary": (
            "seed-1 phase-lag mechanism panel; no lifecycle claim without credible energy/spatial carrier, "
            "causal M offset, returning interictal distribution, and locked-seed replication"
        ),
    }


def main() -> None:
    found = {}
    for root in sorted(IN.glob("*i2edelay*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        label = _label(summary)
        if label is None:
            continue
        if label in found:
            raise RuntimeError(f"duplicate phase-lag arm: {label}")
        found[label] = (root, summary)
    missing = sorted(set(ORDER).difference(found))
    if missing:
        raise RuntimeError(f"phase-lag panel incomplete: {missing}")
    rows, arrays = {}, {}
    for label in ORDER:
        rows[label], arrays[label] = _row(label, *found[label])
        a = arrays[label]
        rows[label]["M_memory_peak"] = float(np.max(a.get("trace_mode_M_memory", np.zeros(1))))
        rows[label]["M_memory_final"] = float(a.get("trace_mode_M_memory", np.zeros(1))[-1])
    verdict = adjudicate(rows)
    payload = {
        "schema": "topic4_zm_i2e_phase_lag_lifecycle_v1_2026-08-03",
        "verdict": verdict,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "i2e_phase_lag_lifecycle_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(len(ORDER), 4, figsize=(17, 11), constrained_layout=True)
    for ir, label in enumerate(ORDER):
        row, a = rows[label], arrays[label]
        tf = a["fine_time_ms"] / 1000.0
        axes[ir, 0].plot(tf, a["fine_core_rate_hz"], lw=0.7, color="#d95f45")
        axes[ir, 0].set(ylabel=f"{label}\ncore Hz", xlabel="time (s)")
        axes[ir, 0].set_title(f"{row['episode_status']}; off={row['offset_ms']}")
        kymo = a["coarse_kymo_axial"]
        axes[ir, 1].imshow(kymo, origin="lower", aspect="auto", cmap="magma",
                           extent=[0, kymo.shape[1] * .025, 0, kymo.shape[0]])
        axes[ir, 1].set(xlabel="time (s)", ylabel="axis bin",
                        title=f"PC1={row['spatial_pc1']:.3f}; rank={row['spatial_effective_rank']:.2f}")
        dt_s = float(found[label][1]["dt_ms"]) / 1000.0
        ts = np.arange(a["trace_z_core_mean"].size) * dt_s
        axes[ir, 2].plot(ts, a["trace_z_core_mean"], label="z core")
        axes[ir, 2].plot(ts, a["trace_m_core_mean"] / 100.0, label="m core/100")
        if "trace_mode_H_max" in a:
            axes[ir, 2].plot(ts, a["trace_mode_H_max"], label="H max")
        axes[ir, 2].set(xlabel="time (s)", ylabel="slow state")
        if "trace_mode_M_memory" in a:
            axes[ir, 3].plot(ts, a["trace_mode_M_memory"], label="M memory", color="#7e22ce")
            axes[ir, 3].plot(ts, a["trace_mode_M_pool"], label="instant M drive", alpha=.55)
        axes[ir, 3].set(xlabel="time (s)", ylabel="exit coordinate")
        if ir == 0:
            axes[ir, 2].legend(frameon=False, fontsize=8)
            axes[ir, 3].legend(frameon=False, fontsize=8)
    fig.suptitle(f"I-to-E phase-lag lifecycle pilot: {verdict['verdict']}", fontsize=15)
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "i2e_phase_lag_lifecycle.png", dpi=170)
    plt.close(fig)
    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    marker = "### i2e_phase_lag_lifecycle.png"
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "固定 E→E scaffold 上比较 I→E 延迟 2×/3×，并对 3× 条件消融 H 与慢 M-memory。"
            "四列依次显示核心放电、轴向时空模式、Z/M/H 和独立的退出记忆。\n\n"
            "**关注点**：PC1 下降只有与持续能量、内生 offset 和 returning interictal event 同时出现时，才构成 lifecycle 证据。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
