#!/usr/bin/env python3
"""Summarise the four-state matched frozen Z/M mode-H probes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_topic4_zm_fast_lifecycle_development import (  # noqa: E402
    _post_entry_spatial_metrics,
)


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/smoke/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
STATES = (
    "pre_entry__natural", "bounded_mid__rising",
    "bounded_mid__peak", "bounded_late__peak",
)
ARMS = ("no_H", "rho0.5", "rho1")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _arm(summary: dict) -> str | None:
    mode = summary.get("mechanism", {}).get("state_selective_mode_H")
    if not mode:
        return "no_H"
    rho = float(mode["rho_mode_H"])
    half = float(mode["m_mode_half"])
    if not np.isclose(half, 30.0):
        return None
    if np.isclose(rho, 0.5):
        return "rho0.5"
    if np.isclose(rho, 1.0):
        return "rho1"
    return None


def main() -> None:
    found: dict[tuple[str, str], tuple[Path, dict]] = {}
    for root in sorted(IN.glob("*freeze*")):
        sp = root / "summary.json"
        tp = root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        if not np.isclose(float(summary.get("T_ms", 0.0)), 2500.0):
            continue
        state, arm = summary.get("state"), _arm(summary)
        if state not in STATES or arm not in ARMS:
            continue
        with np.load(tp, allow_pickle=False) as arrays:
            if arm != "no_H" and "trace_mode_H_rate_max_hz" not in arrays.files:
                continue
        key = (state, arm)
        if key in found:
            raise RuntimeError(f"duplicate frozen probe {key}")
        found[key] = (root, summary)
    missing = [(state, arm) for state in STATES for arm in ARMS if (state, arm) not in found]
    if missing:
        raise RuntimeError(f"frozen probe panel incomplete: {missing}")

    rows = []
    arrays_by_key = {}
    for state in STATES:
        for arm in ARMS:
            root, summary = found[state, arm]
            with np.load(root / "traces.npz", allow_pickle=False) as data:
                a = {key: np.asarray(data[key], float) for key in data.files}
            arrays_by_key[state, arm] = a
            core, alle = a["coarse_core_rate_hz"], a["coarse_all_e_rate_hz"]
            tail_bins = min(20, core.size)
            spatial = _post_entry_spatial_metrics(a["coarse_kymo_axial"], skip_ms=0.0)
            rows.append({
                "state": state,
                "arm": arm,
                "core_mean_hz": float(core.mean()),
                "core_tail_mean_hz": float(core[-tail_bins:].mean()),
                "core_peak_hz": float(core.max()),
                "all_E_tail_mean_hz": float(alle[-tail_bins:].mean()),
                "pc1_fraction": spatial.get("common_mode_pc1_fraction"),
                "spatial_effective_rank": spatial.get("spatial_effective_rank"),
                "H_peak": float(a.get("trace_mode_H_max", np.zeros(1)).max()),
                "gain_local_peak": float(a.get("trace_mode_H_gain_max", np.zeros(1)).max()),
                "z_drift": summary["z_max_abs_drift"],
                "m_drift": summary["m_max_abs_drift"],
                "runaway_ms": summary.get("runaway_early_stop_ms"),
                "summary_path": str((root / "summary.json").relative_to(ROOT)),
                "trace_path": str((root / "traces.npz").relative_to(ROOT)),
                "trace_sha256": _sha(root / "traces.npz"),
            })
    by_key = {(row["state"], row["arm"]): row for row in rows}
    for state in STATES:
        base = by_key[state, "no_H"]
        for arm in ("rho0.5", "rho1"):
            row = by_key[state, arm]
            row["delta_tail_core_hz_vs_no_H"] = row["core_tail_mean_hz"] - base["core_tail_mean_hz"]
            row["delta_pc1_vs_no_H"] = row["pc1_fraction"] - base["pc1_fraction"]
            row["delta_effective_rank_vs_no_H"] = (
                row["spatial_effective_rank"] - base["spatial_effective_rank"]
            )

    h_rows = [row for row in rows if row["arm"] != "no_H"]
    low_h_rows = [row for row in h_rows if row["core_tail_mean_hz"] < 25.0]
    best_rank = max(h_rows, key=lambda row: row["spatial_effective_rank"])
    verdict = (
        "FROZEN_LOW_OR_RECOVERY_REGION_OBSERVED"
        if low_h_rows else "NO_RECOVERY_REGION_IN_VISITED_FROZEN_STATES"
    )
    output = {
        "schema": "topic4_zm_mode_H_frozen_probes_v1_2026-08-03",
        "verdict": verdict,
        "rows": rows,
        "best_spatial_rank_H_arm": best_rank,
        "matched_direction": {
            "H_increased_tail_core_rate_in_all_visited_states": all(
                row["delta_tail_core_hz_vs_no_H"] > 0.0
                for row in h_rows
            ),
            "minimum_H_tail_increase_hz": min(
                row["delta_tail_core_hz_vs_no_H"] for row in h_rows
            ),
        },
        "claim_boundary": (
            "four visited seed-1 slow states, 2.5-s continuations; directional topology only, not a full bifurcation map"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state_selective_H_frozen_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    matrices = {}
    for field in ("core_tail_mean_hz", "pc1_fraction", "spatial_effective_rank"):
        matrices[field] = np.array([
            [by_key[state, arm][field] for arm in ARMS] for state in STATES
        ])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, field, title, cmap in (
        (axes[0, 0], "core_tail_mean_hz", "tail core rate (Hz)", "magma"),
        (axes[0, 1], "pc1_fraction", "axial PC1 fraction", "viridis"),
        (axes[1, 0], "spatial_effective_rank", "axial effective rank", "cividis"),
    ):
        image = ax.imshow(matrices[field], aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(ARMS)), ARMS)
        ax.set_yticks(range(len(STATES)), [s.replace("__", "\n") for s in STATES])
        ax.set_title(title)
        for i in range(len(STATES)):
            for j in range(len(ARMS)):
                ax.text(j, i, f"{matrices[field][i,j]:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax)
    best_key = (best_rank["state"], best_rank["arm"])
    kymo = arrays_by_key[best_key]["coarse_kymo_axial"]
    axes[1, 1].imshow(
        kymo, origin="lower", aspect="auto", cmap="magma",
        extent=[0, kymo.shape[1] * 0.025, 0, kymo.shape[0]],
    )
    axes[1, 1].set(
        xlabel="time (s)", ylabel="pathological-axis bin",
        title=f"highest-rank H arm: {best_key[0]}, {best_key[1]}",
    )
    fig.suptitle(f"frozen Z/M mode-H probes: {verdict}", fontsize=15)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "state_selective_H_frozen_probes.png", dpi=170)
    plt.close(fig)
    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    section = (
        "\n### state_selective_H_frozen_probes.png\n\n"
        "在四个自然轨迹上已访问的 Z/M 状态中，比较 matched no-H、rho=0.5 和 rho=1 的 2.5 秒冻结延续。"
        "前三格分别显示尾段核心放电、轴向 PC1 和有效空间秩；右下角展示空间秩最高的 H 条件。\n\n"
        "**关注点**：空间秩升高只有在尾段同时离开高活动流形时才可解释为恢复拓扑；单独降低 PC1 只是 pattern-generator 线索。\n"
    )
    if "### state_selective_H_frozen_probes.png" not in prior:
        readme.write_text(prior.rstrip() + "\n" + section)
    print(json.dumps({"verdict": verdict, "best": best_key, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
