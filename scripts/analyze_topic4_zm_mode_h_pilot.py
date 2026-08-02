#!/usr/bin/env python3
"""Compare the fixed-anchor state-selective-H causal pilot.

This script intentionally analyses only the baseline and five predeclared H
arms.  It does not glob a parameter surface or turn the pilot into a new grid.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_topic4_zm_lifecycle_m_panel as MP  # noqa: E402
from scripts import analyze_topic4_zm_lifecycle_sprint as LS  # noqa: E402
from src.topic4_zm_empirical_modes import fit_axial_dmd  # noqa: E402
from src.topic4_zm_mode_h_pilot import adjudicate_mode_h_pilot  # noqa: E402


IN_ROOT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
LABELS = (
    "baseline", "rho025_gate", "rho05_gate", "rho05_mc30", "rho05_nomgate",
    "rho1_gate", "rho1_mc30", "rho1_nomgate",
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _close(left, right, tol=1e-8) -> bool:
    return left is not None and abs(float(left) - float(right)) <= tol


def _label(summary: dict) -> str | None:
    if not _close(summary.get("T_ms"), 12000.0):
        return None
    mech = summary.get("mechanism", {})
    dep = mech.get("i2e_depression", {})
    slow = mech.get("dynamic_slow_flow", {})
    if (
        mech.get("arm") != "i2e"
        or not _close(dep.get("tau_D_ms"), 300.7)
        or not _close(dep.get("d_star_nominal"), 0.7281)
        or not _close(mech.get("strength_scale"), 1.0)
        or not _close(slow.get("g_M"), 1.0)
        or not _close(slow.get("tau_M_ms"), 500.0)
    ):
        return None
    mode = mech.get("state_selective_mode_H")
    if not mode:
        return "baseline"
    rho, half = float(mode["rho_mode_H"]), float(mode["m_mode_half"])
    if _close(rho, 0.25) and _close(half, 45.0):
        return "rho025_gate"
    if _close(rho, 0.5) and _close(half, 45.0):
        return "rho05_gate"
    if _close(rho, 0.5) and _close(half, 30.0):
        return "rho05_mc30"
    if _close(rho, 0.5) and half > 1e8:
        return "rho05_nomgate"
    if _close(rho, 1.0) and _close(half, 45.0):
        return "rho1_gate"
    if _close(rho, 1.0) and _close(half, 30.0):
        return "rho1_mc30"
    if _close(rho, 1.0) and half > 1e8:
        return "rho1_nomgate"
    return None


def _post_onset_gap(core: np.ndarray, onset_bin: int | None) -> float | None:
    start = 0 if onset_bin is None else min(core.size, int(onset_bin) + 40)
    x = np.asarray(core[start:], float)
    if x.size < 20:
        return None
    threshold = max(5.0, 0.10 * float(np.percentile(x, 95)))
    return float(np.mean(x <= threshold))


def _row(label: str, root: Path, summary: dict) -> tuple[dict, dict]:
    analysis = LS.analyze_one(root.resolve())
    tail = MP._tail_state_metrics(root.resolve())
    trace = MP._trace_metrics(
        root.resolve(), analysis["episode"], dt_ms=float(summary["dt_ms"]),
    )
    with np.load(root / "traces.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key], float) for key in data.files}
    h_max = arrays.get("trace_mode_H_max", np.zeros(1))
    h_mean = arrays.get("trace_mode_H_mean", np.zeros(1))
    gain = arrays.get("trace_mode_H_gain_mean", np.zeros(1))
    gain_max = arrays.get("trace_mode_H_gain_max", np.zeros(1))
    gain_core = arrays.get("trace_mode_H_gain_core_mean", np.zeros(1))
    onset = analysis["episode"].get("onset_ms")
    offset = analysis["episode"].get("offset_ms")
    dmd_start = max(1000.0, float(onset or 0.0) + 1000.0)
    dmd_end = None
    if offset is not None and float(offset) - dmd_start >= 1500.0:
        dmd_end = float(offset)
    dmd = None
    if arrays["coarse_kymo_axial"].shape[1] * 25.0 - dmd_start >= 1500.0:
        dmd = fit_axial_dmd(
            arrays["coarse_kymo_axial"], dt_ms=25.0,
            start_ms=dmd_start, end_ms=dmd_end,
        )
    row = {
        "label": label,
        "stem": root.name,
        "phenotype": analysis["phenotype"],
        "episode_status": analysis["episode"]["status"],
        "onset_ms": onset,
        "offset_ms": offset,
        "runaway": summary.get("runaway_early_stop_ms") is not None,
        "runaway_ms": summary.get("runaway_early_stop_ms"),
        "returning_event": bool(analysis["recovery"].get("single_event_candidate")),
        "returning_distribution": bool(analysis["recovery"].get("distribution_recovered")),
        "median_vseeg_gain_db": analysis["intensity"].get("median_gain_db_across_contacts"),
        "energy_occupancy_6db": analysis["intensity"].get("occupancy_above_6db"),
        "post_onset_deep_gap_fraction": _post_onset_gap(
            arrays["coarse_core_rate_hz"], analysis["episode"].get("onset_bin"),
        ),
        "spatial_pc1": analysis["within_episode_spatial"].get("common_mode_pc1_fraction"),
        "spatial_effective_rank": analysis["within_episode_spatial"].get("spatial_effective_rank"),
        "tail": tail,
        "H_peak": float(np.max(h_max)),
        "H_mean_peak": float(np.max(h_mean)),
        "gain_population_mean_peak": float(np.max(gain)),
        "gain_local_max_peak": float(np.max(gain_max)),
        "gain_core_mean_peak": float(np.max(gain_core)),
        "m_peak": trace.get("m_peak"),
        "m_final": trace.get("m_final"),
        "z_final": trace.get("z_core_final"),
        "z_minimum": trace.get("z_core_minimum"),
        "z_post_offset_recovery": trace.get("z_core_post_offset_recovery"),
        "dmd": None if dmd is None else {
            "pc1_fraction": dmd["pc1_fraction"],
            "heldout_relative_error": dmd["heldout_relative_error"],
            "leading_mode": dmd["leading_mode"],
            "pathological_mode_candidate": dmd["pathological_mode_candidate"],
            "claim_boundary": dmd["claim_boundary"],
        },
        "summary_path": str((root / "summary.json").relative_to(ROOT)),
        "trace_path": str((root / "traces.npz").relative_to(ROOT)),
        "summary_sha256": _sha(root / "summary.json"),
        "trace_sha256": _sha(root / "traces.npz"),
    }
    return row, arrays


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    found: dict[str, tuple[Path, dict]] = {}
    for root in sorted(IN_ROOT.glob("*")):
        summary_path = root / "summary.json"
        if not summary_path.is_file() or not (root / "traces.npz").is_file():
            continue
        summary = json.loads(summary_path.read_text())
        label = _label(summary)
        if label is None:
            continue
        if label in found:
            raise RuntimeError(f"duplicate fixed-pilot arm: {label}")
        found[label] = (root, summary)
    missing = sorted(set(LABELS).difference(found))
    if missing:
        raise RuntimeError(f"fixed H pilot incomplete: {missing}")

    rows, arrays_by_label = {}, {}
    for label in LABELS:
        rows[label], arrays_by_label[label] = _row(label, *found[label])
    verdict = adjudicate_mode_h_pilot({key: rows[key] for key in (
        "baseline", "rho05_gate", "rho05_mc30", "rho05_nomgate",
        "rho1_gate", "rho1_mc30", "rho1_nomgate",
    )})
    summary = {
        "schema": "topic4_zm_state_selective_H_fixed_pilot_v1_2026-08-02",
        "git_sha_analysis": _git_sha(),
        "design": "one fixed fast anchor; two H strengths x M-gate on/off plus weak directional arm",
        "verdict": verdict,
        "rows": rows,
        "claim_boundary": (
            "seed-1 mechanism pilot; no lifecycle claim without causal exit, interictal return, healthy specificity, and locked-seed replication"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state_selective_H_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(len(LABELS), 3, figsize=(15, 3.0 * len(LABELS)), constrained_layout=True)
    for ir, label in enumerate(LABELS):
        row, a = rows[label], arrays_by_label[label]
        t = np.asarray(a["fine_time_ms"]) / 1000.0
        axes[ir, 0].plot(t, a["fine_core_rate_hz"], color="#d95f45", lw=0.8)
        axes[ir, 0].set(ylabel=f"{label}\ncore Hz", xlabel="time (s)")
        axes[ir, 0].set_title(
            f"{row['episode_status']}; gap={row['post_onset_deep_gap_fraction']!s}"
        )
        kymo = a["coarse_kymo_axial"]
        axes[ir, 1].imshow(
            kymo, origin="lower", aspect="auto", cmap="magma",
            extent=[0, kymo.shape[1] * 0.025, 0, kymo.shape[0]],
        )
        axes[ir, 1].set(xlabel="time (s)", ylabel="axis bin", title=f"PC1={row['spatial_pc1']}")
        dt_s = float(found[label][1]["dt_ms"]) / 1000.0
        slow_t = np.arange(a["trace_z_core_mean"].size) * dt_s
        axes[ir, 2].plot(slow_t, a["trace_z_core_mean"], label="z core", color="#2878b5")
        axes[ir, 2].plot(slow_t, a["trace_m_core_mean"] / 100.0, label="m core / 100", color="#d95f45")
        if "trace_mode_H_max" in a:
            axes[ir, 2].plot(slow_t, a["trace_mode_H_max"], label="H max", color="#15803d")
        axes[ir, 2].set(xlabel="time (s)", ylabel="slow coordinate", ylim=(-0.02, 1.05))
        if ir == 0:
            axes[ir, 2].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"state-selective H fixed-anchor pilot: {verdict['verdict']}", fontsize=15)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "state_selective_H_fixed_pilot.png", dpi=170)
    plt.close(fig)
    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    section = (
        "\n### state_selective_H_fixed_pilot.png\n\n"
        "在同一个 Z/M 快抑制 anchor 上比较无 H、弱 H，以及两档 H 强度的 M 关闭门 on/off。"
        "每行依次给出核心放电、24-bin 病理轴活动和 z/m/H 慢变量；这里只裁决 H 是否接合、"
        "是否填补间隙，以及 M 关闭 H 后是否出现相对 no-M-gate 臂的因果退出。\n\n"
        "**关注点**：退出必须在 M-gated 臂发生而 matched no-M-gate 臂保持高态；短暂降幅、"
        "tonic plateau 或单纯静默都不能算完整 lifecycle。\n"
    )
    if "### state_selective_H_fixed_pilot.png" not in prior:
        readme.write_text(prior.rstrip() + "\n" + section)
    print(json.dumps({"verdict": verdict, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
