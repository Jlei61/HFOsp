#!/usr/bin/env python3
"""Figure E -- H3, does IED exposure update the functional state?

asset_id: epi_prssm_exposure_mechanism
This is an independent extension: a negative here changes nothing about B, C or D.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, code_revision, package_hash
from src.topic5_epi_prssm.figure_style import (
    COLOR, DOUBLE_COLUMN_MM, FS_AXIS, FS_TICK, FS_TITLE, LW_MAIN, LW_REFERENCE,
    figure, panel_letter, save_asset, zero_line,
)

ASSET = "epi_prssm_exposure_mechanism"
EXPO = OUTPUT_ROOT / "exposure_mechanism"
FIG_ROOT = OUTPUT_ROOT / "figures"


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a table that a stage may legitimately have left empty."""
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        pass
    return pd.DataFrame(columns=columns or [])



def panel_a(ax, ladder: pd.DataFrame):
    ax.set_title("R0-R3 nested ladder", fontsize=FS_TITLE, loc="left", pad=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    rungs = [("R0", "no resource", 0.80, COLOR["G2"]),
             ("R1", "autonomous recovery\nand consumption", 0.57, COLOR["G3"]),
             ("R2", "+ single-event\ndepletion", 0.34, COLOR["exposure"]),
             ("R3", "+ integrated\nexposure", 0.11, COLOR["exposure"])]
    for name, body, y, colour in rungs:
        ax.add_patch(FancyBboxPatch((0.03, y), 0.34, 0.17, boxstyle="round,pad=0.012",
                                    facecolor="white", edgecolor=colour, lw=1.0))
        ax.text(0.055, y + 0.115, name, fontsize=FS_TICK + 0.4, fontweight="bold", color=colour)
        ax.text(0.40, y + 0.085, body, fontsize=FS_TICK - 0.6, va="center", color="#333333")
    for y0, y1 in ((0.80, 0.74), (0.57, 0.51), (0.34, 0.28)):
        ax.add_patch(FancyArrowPatch((0.20, y0), (0.20, y1), arrowstyle="-|>",
                                     mutation_scale=7, color="#666666", lw=0.9))
    ax.add_patch(FancyArrowPatch((0.20, 0.34), (0.20, 0.28), arrowstyle="-|>",
                                 mutation_scale=7, color=COLOR["exposure"], lw=1.1))
    ax.text(0.03, 0.02, r"the only rust arrow is the exposure forcing $\gamma_x$",
            fontsize=FS_TICK - 0.8, color=COLOR["exposure"])


def panel_b(ax, ladder: pd.DataFrame, tau_freeze: dict | None):
    ax.set_title("Frozen and fitted time constants", fontsize=FS_TITLE, loc="left", pad=5)
    frame = ladder[ladder.status == "COMPLETE"] if not ladder.empty else ladder
    if tau_freeze:
        rows = pd.DataFrame(tau_freeze["rows"])
        ax.errorbar(rows["tau_r_seconds"], rows["mean_validation"],
                    yerr=rows["sem_validation"], fmt="-o", ms=3.4, lw=LW_MAIN,
                    color=COLOR["G3"], capsize=1.8)
        ax.axhline(tau_freeze["one_se_threshold"], color="#8A8A8A", lw=LW_REFERENCE,
                   ls=(0, (3, 2)))
        ax.axvline(tau_freeze["tau_r_seconds"], color=COLOR["exposure"], lw=1.0)
        ax.text(tau_freeze["tau_r_seconds"], ax.get_ylim()[1],
                f"  frozen $\\tau_r$ = {tau_freeze['tau_r_seconds']:.0f} s",
                fontsize=FS_TICK - 0.6, va="top", color=COLOR["exposure"])
        interval = tau_freeze["identifiable_interval_seconds"]
        ax.text(0.02, 0.04, "identifiable" if tau_freeze["identifiable"] else
                            f"unidentifiable band {interval[0]:.0f}-{interval[1]:.0f} s",
                transform=ax.transAxes, fontsize=FS_TICK - 0.6, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau_r$ (s), frozen before any exposure arm")
    ax.set_ylabel("held-out event NLL")


def panel_c(ax, curve: pd.DataFrame):
    ax.set_title("Exposure timescale curve", fontsize=FS_TITLE, loc="left", pad=5)
    if curve.empty:
        ax.text(0.5, 0.5, "no timescale sweep", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_TICK)
        ax.set_xticks([]); ax.set_yticks([])
        return
    subset = curve[curve.endpoint == "order_nll"]
    clock = subset[subset.kernel == "clock"].sort_values("scale")
    events = subset[subset.kernel == "event_count"].sort_values("scale")
    if not clock.empty:
        ax.plot(clock["scale"], clock["median_delta_vs_base"], "-o", ms=3.4, lw=LW_MAIN,
                color=COLOR["exposure"], label="clock kernel")
        ax.fill_between(clock["scale"], clock["ci_low"], clock["ci_high"],
                        color=COLOR["exposure"], alpha=0.14, lw=0)
    zero_line(ax)
    ax.set_xscale("log")
    ax.set_xlabel(r"clock $\tau_x$ (s)")
    ax.set_ylabel("Δ masked-order NLL vs base arm")
    if not events.empty:
        twin = ax.twiny()
        twin.plot(events["scale"], events["median_delta_vs_base"], "-s", ms=3.0,
                  lw=LW_MAIN, color="#8A6A5A", label="event-count kernel")
        twin.set_xscale("log")
        twin.set_xlabel("event-count kernel (events)", fontsize=FS_TICK, color="#8A6A5A")
        twin.tick_params(axis="x", labelsize=FS_TICK - 1.0, colors="#8A6A5A")
    ax.legend(loc="best", fontsize=FS_TICK - 0.8, handlelength=1.0, borderpad=0.2)


def panel_d(ax, effects: pd.DataFrame):
    ax.set_title("Non-load endpoint, per patient", fontsize=FS_TITLE, loc="left", pad=5)
    subset = effects[effects.endpoint == "order_nll"] if not effects.empty else effects
    if subset.empty:
        ax.text(0.5, 0.5, "no completed exposure contrast", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_TICK)
        ax.set_xticks([]); ax.set_yticks([])
        return
    arms = [a for a in ("t1_r0", "t2_r2", "t2_r3_clock1800", "t2_r3_events20")
            if a in set(subset.arm)]
    if not arms:
        arms = sorted(set(subset.arm))[:4]
    y = np.arange(len(arms))
    for i, arm in enumerate(arms):
        row = subset[subset.arm == arm].iloc[0]
        colour = COLOR["exposure"] if arm.startswith("t2_") else COLOR["G3"]
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=colour, lw=1.2)
        ax.plot(row["median_delta"], i, "o", ms=4.6, color=colour, mew=0)
        ax.text(row["ci_high"], i + 0.24,
                f"{int(row['n_favourable'])}/{int(row['n_patients'])}",
                fontsize=FS_TICK - 0.8, color="#333333", va="bottom")
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels(arms, fontsize=FS_TICK - 0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Δ masked-order NLL vs the matched base arm")


def panel_e(ax, innovation: dict | None):
    ax.set_title("Innovation and directionality", fontsize=FS_TITLE, loc="left", pad=5)
    if not innovation:
        ax.text(0.5, 0.5, "innovation controls not run", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS_TICK)
        return
    taus = sorted(innovation["by_tau"], key=float)
    controls = ["state_matched_shuffle", "time_reversal", "event_count_kernel",
                "session_block_shuffle"]
    labels = ["state-matched\nshuffle", "time\nreversal", "event-count\nkernel",
              "session block\nshuffle"]
    width = 0.8 / max(len(taus), 1)
    for j, tau in enumerate(taus):
        block = innovation["by_tau"][tau]
        values = [block[f"real_minus_{c}"]["median_delta"] for c in controls]
        low = [block[f"real_minus_{c}"]["ci_low"] for c in controls]
        high = [block[f"real_minus_{c}"]["ci_high"] for c in controls]
        x = np.arange(len(controls)) + (j - (len(taus) - 1) / 2) * width
        colour = [COLOR["exposure"], "#C08A72", "#D8B4A2"][j % 3]
        ax.bar(x, values, width=width * 0.9, color=colour, label=f"$\\tau_x$={float(tau):.0f} s")
        ax.errorbar(x, values, yerr=[np.array(values) - np.array(low),
                                     np.array(high) - np.array(values)],
                    fmt="none", ecolor="#3A3A3A", lw=0.8, capsize=1.4)
    zero_line(ax)
    ax.set_xticks(range(len(controls)))
    ax.set_xticklabels(labels, fontsize=FS_TICK - 1.0)
    ax.set_ylabel("real − control (Spearman)")
    ax.legend(loc="best", fontsize=FS_TICK - 1.0, handlelength=0.8, borderpad=0.2)


def panel_f(ax, h3a: dict, h3b: dict, ladder: pd.DataFrame):
    ax.set_title("H3a and H3b evidence cards", fontsize=FS_TITLE, loc="left", pad=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.plot([0.48, 0.48], [0.02, 0.98], color="#DDDDDD", lw=0.8)
    ax.text(0.0, 0.96, "H3a  interictal updating", fontsize=FS_TICK, fontweight="bold",
            color=COLOR["exposure"])
    predictive = h3a.get("predictive_leg", {})
    lines = []
    for key, effect in list(predictive.items())[:4]:
        arm = key.split("::")[-1]
        lines.append(f"{arm}: {effect['median_delta']:+.4f}\n"
                     f"  [{effect['ci_low']:+.4f}, {effect['ci_high']:+.4f}]  "
                     f"{effect['n_favourable']}/{effect['n_patients']}")
    ax.text(0.0, 0.88, "\n".join(lines) or "not available", fontsize=FS_TICK - 1.0,
            va="top", color="#333333")
    health = h3a.get("resource_health", {})
    ax.text(0.0, 0.22, f"collapsed resource runs: {health.get('n_collapsed_runs', '?')}\n"
                       f"static resource runs: {health.get('n_static_runs', '?')}",
            fontsize=FS_TICK - 1.0, va="top", color="#666666")
    ax.text(0.52, 0.96, "H3b  transition consistency", fontsize=FS_TICK, fontweight="bold",
            color=COLOR["onset"])
    ax.text(0.52, 0.88, f"status: {h3b.get('status')}\n\n{h3b.get('reason', h3b.get('requires', ''))}",
            fontsize=FS_TICK - 1.0, va="top", color="#333333", wrap=True)
    ax.text(0.52, 0.14, "H3b is read-only: it never gates H1, H2a, H2b or H3a",
            fontsize=FS_TICK - 1.0, va="top", color="#666666", style="italic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()
    ladder = _read_csv(EXPO / "resource_ladder.csv",
                       ["arm", "resource_arm", "seed", "status", "best_validation",
                        "tau_r_seconds", "tau_x_seconds", "exposure_kind",
                        "resource_boundary_occupancy", "resource_collapsed"])
    effects = _read_csv(EXPO / "t1_t2_patient_effects.csv",
                        ["endpoint", "arm", "reference", "n_patients", "median_delta",
                         "ci_low", "ci_high", "n_favourable", "sign_test_p", "wilcoxon_p"])
    curve = _read_csv(EXPO / "exposure_timescale_curve.csv",
                      ["endpoint", "arm", "kernel", "scale", "median_delta_vs_base",
                       "ci_low", "ci_high", "n_favourable", "n_patients", "sign_test_p"])
    h3a = json.loads((EXPO / "H3A_EVIDENCE_CARD.json").read_text())
    h3b = json.loads((EXPO / "H3B_EVIDENCE_CARD.json").read_text())
    tau_path = OUTPUT_ROOT / "manifests/RESOURCE_TAU_FREEZE.json"
    tau_freeze = json.loads(tau_path.read_text()) if tau_path.exists() else None
    innovation_path = EXPO / "innovation_controls_summary.json"
    innovation = json.loads(innovation_path.read_text()) if innovation_path.exists() else None

    fig, axes = figure(DOUBLE_COLUMN_MM, 128.0, nrows=2, ncols=3)
    fig.subplots_adjust(left=0.070, right=0.985, top=0.90, bottom=0.105, wspace=0.36, hspace=0.48)
    panel_a(axes[0, 0], ladder)
    panel_b(axes[0, 1], ladder, tau_freeze)
    panel_c(axes[0, 2], curve)
    panel_d(axes[1, 0], effects)
    panel_e(axes[1, 1], innovation)
    panel_f(axes[1, 2], h3a, h3b, ladder)
    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        panel_letter(ax, letter, dx=-0.16, dy=1.19)

    files = save_asset(fig, ASSET, FIG_ROOT, metadata={
        "asset_id": ASSET, "provisional_role": "Figure E", "hypothesis": "H3a and H3b",
        "status": "EXPLORATORY", "reference_arm": h3a.get("reference_arm"),
        "tau_r_freeze": tau_freeze, "resource_health": h3a.get("resource_health"),
        "denominators": h3a.get("denominators"),
        "innovation_leg_present": innovation is not None,
        "claim_boundary": h3a["claim_boundary"],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }, readme_entries=[{
        "filename": f"{ASSET}.png",
        "body": "H3 的独立证据卡，阴性同样有效，不影响 B、C、D。A 是 R0-R3 嵌套阶梯，"
                "唯一的锈色箭头是暴露驱动项。B 是先在 T1/R1 上冻结的资源恢复时间常数，"
                "竖线是被选中的值，若一个标准误带内有多个格点则标为不可辨识区间。"
                "C 是暴露时间尺度曲线，主轴是时钟核、上轴是事件计数对照核。"
                "D 是与匹配基臂相比的逐患者非负荷端点效应（掩蔽顺序似然）。"
                "E 把真实创新与四个对照（状态匹配打乱、时间反转、事件计数核、按段打乱）并列。"
                "F 左右分开写 H3a 与 H3b。",
        "focus": "D 与 E 的零线是承重的；若 F 中 collapsed/static resource 计数不为零，"
                 "则相应臂的比较应读作资源没有携带信息，而不是资源不存在。",
    }])
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
