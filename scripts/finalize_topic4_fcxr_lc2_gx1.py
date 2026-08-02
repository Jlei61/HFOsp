#!/usr/bin/env python3
"""Finalize FCXR-LC2-GX1 frozen entry/offset diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core",
                   "gx1_entry_offset_diagnostics")
ARCHIVE = os.path.join(ROOT, "docs", "archive", "topic4", "sef_hfo",
                       "fcxr_lc2_gx1_entry_offset_diagnostics_2026-08-02.md")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def _write_json(name, payload):
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text.rstrip() + "\n")
    os.replace(tmp, path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def choose_next_hypothesis(strip_verdict, x_verdict):
    natural = strip_verdict == "NATURAL_SELECTIVITY_WINDOW_CANDIDATE"
    reachable = x_verdict in (
        "X_PATH_REACHABLE_RANGE_INSUFFICIENT",
        "X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH",
    )
    bypass = x_verdict == "H_ACTUATOR_BYPASSES_X_AT_MAXIMAL_SHUTDOWN"
    if natural and reachable:
        return "KEEP_H_EQUATION_CALIBRATE_X_RANGE"
    if natural and bypass:
        return "SHARED_PATH_X_H_COUPLING_ONLY"
    if strip_verdict == "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP" and reachable:
        return "LOCAL_D_DEPENDENT_H_GAIN_ONLY_X_RANGE_SEPARATE"
    if strip_verdict == "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP" and bypass:
        return "CAUSAL_2X2_D_GATE_BY_SHARED_X_H_PATH"
    return "MEASUREMENT_REPAIR_NO_STRUCTURAL_CLAIM"


def build_candidate_verdict(strip, xmap):
    rows = [a for p in strip["point_rows"] for a in p["arms"]] + list(xmap["rows"])
    safe = sum(not bool(r.get("numerical_failure", True)) for r in rows)
    return dict(
        status="COMPLETE",
        scientific_tier="frozen_component_diagnostic",
        canonical_verdict="GX1_ENTRY_OFFSET_DIAGNOSTIC_COMPLETE",
        selectivity_strip_verdict=strip["verdict"],
        x_authority_verdict=xmap["verdict"],
        authorized_next_hypothesis=choose_next_hypothesis(strip["verdict"], xmap["verdict"]),
        n_strip_rows=int(strip["n_rows"]),
        n_strip_points=int(strip["n_points"]),
        n_strip_pass=int(strip["n_pass"]),
        n_strip_window_points=int(strip["n_window_points"]),
        n_x_rows=int(xmap["n_rows"]),
        numerical_safe_rows=int(safe),
        numerical_total_rows=len(rows),
        dynamic_lifecycle_tested=False,
        morphology_tested=False,
        forbidden_claims=[
            "spontaneous seizure lifecycle",
            "bistability or hysteresis from the frozen strip alone",
            "physiological validity of x=0",
            "patient-like ictal morphology",
        ],
        finalized_at=_now(),
    )


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "savefig.dpi": 220,
    })


def plot_strip(strip, path):
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3), constrained_layout=True)
    families = ("H1", "H6")
    rhos = (0.025, 0.05, 0.075)
    thetas = (1.0, 1.25)
    labels = {"INTERICTAL_WORKPOINT": 0, "ELEVATED_EVENT_TRAIN": 1,
              "FINITE_HIGH_FIXED": 2, "FINITE_HIGH_ORBIT": 2}
    cmap = ListedColormap(["#4c78a8", "#f2cf5b", "#d1495b", "#777777"])
    for iy, family in enumerate(families):
        pts = {p["point_id"]: p for p in strip["point_rows"] if p["family"] == family}
        for ix, arm in enumerate(("healthy_low", "susceptible_low", "susceptible_high")):
            z = np.full((len(thetas), len(rhos)), 3.0)
            rates = np.full_like(z, np.nan)
            for ti, ts in enumerate(thetas):
                for ri, rho in enumerate(rhos):
                    p = next((q for q in pts.values()
                              if q["theta_scale"] == ts and q["rho_fraction"] == rho), None)
                    if p is None:
                        continue
                    a = next(q for q in p["arms"] if q["arm"] == arm)
                    z[ti, ri] = labels.get(a.get("workpoint_label"), 3)
                    rates[ti, ri] = float(a["state_tail_1s"]["rate_mean_hz"])
            ax = axes[iy, ix]
            ax.imshow(z, vmin=-0.5, vmax=3.5, cmap=cmap, aspect="auto")
            for ti in range(len(thetas)):
                for ri in range(len(rhos)):
                    if np.isfinite(rates[ti, ri]):
                        ax.text(ri, ti, f"{rates[ti, ri]:.1f}", ha="center", va="center",
                                color="white" if z[ti, ri] in (0, 2, 3) else "#222222",
                                fontsize=8, fontweight="bold")
            ax.set_xticks(range(len(rhos)), [f"{v:.3f}" for v in rhos])
            ax.set_yticks(range(len(thetas)), [f"{v:.2f}" for v in thetas])
            ax.set_xlabel(r"$\rho_H/g_{sat}$")
            if ix == 0:
                ax.set_ylabel(f"{family}\n" + r"$\theta$ scale")
            else:
                ax.set_ylabel(r"$\theta$ scale")
            if iy == 0:
                ax.set_title(arm.replace("_", " "))
    fig.suptitle(f"GX1 selectivity strip: {strip['verdict']}\ncell text = final-1 s rate (Hz)",
                 fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _rolling(x, n):
    x = np.asarray(x, float)
    if n <= 1:
        return x
    if x.size < n:
        return np.full_like(x, np.nan)
    y = np.convolve(x, np.ones(n) / n, mode="valid")
    return np.r_[np.full(n - 1, np.nan), y]


def plot_x(xmap, path):
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.1), sharex=True,
                             constrained_layout=True)
    colors = {1.0: "#4c78a8", 0.5: "#59a14f", 0.1: "#f28e2b", 0.0: "#d1495b"}
    for row in sorted(xmap["rows"], key=lambda r: -float(r["x_availability"])):
        x = float(row["x_availability"])
        dt = float(row["trace_dt_ms"])
        t = np.arange(len(row["rate_trace"])) * dt / 1000.0
        axes[0].plot(t, _rolling(row["rate_trace"], max(1, int(round(300.0 / dt)))),
                     lw=1.4, color=colors[x], label=f"availability={x:g}")
        axes[1].plot(t, row["h_trace"], lw=1.4, color=colors[x])
    axes[0].axhline(20.0, color="#555555", ls="--", lw=0.8, label="high guide")
    axes[0].set_ylabel("300-ms rate (Hz)")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].set_ylabel("mean H")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"GX1 maximal-X authority: {xmap['verdict']}", fontsize=11,
                 fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_logic(strip, xmap, path):
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    ax.axis("off")
    natural = strip["verdict"] == "NATURAL_SELECTIVITY_WINDOW_CANDIDATE"
    reachable = xmap["verdict"] in ("X_PATH_REACHABLE_RANGE_INSUFFICIENT",
                                    "X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH")
    cells = [
        (0.05, 0.55, "Natural selectivity\n+ X path reachable",
         "Keep H; calibrate X range", natural and reachable),
        (0.52, 0.55, "Natural selectivity\n+ maximal-X bypass",
         "Shared X/H path only", natural and not reachable),
        (0.05, 0.08, "No natural window\n+ X path reachable",
         "Local D-gate only", (not natural) and reachable),
        (0.52, 0.08, "No natural window\n+ maximal-X bypass",
         "Causal 2x2: D gate x shared path", (not natural) and (not reachable)),
    ]
    for x, y, title, action, active in cells:
        ax.add_patch(plt.Rectangle((x, y), 0.43, 0.35,
                                   fc="#f28e2b" if active else "#edf0f2",
                                   ec="#333333" if active else "#aaaaaa", lw=1.6 if active else 0.8))
        ax.text(x + 0.215, y + 0.23, title, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + 0.215, y + 0.09, action, ha="center", va="center", fontsize=8)
    ax.text(0.5, 0.98, "GX1 conditional mechanism logic", ha="center", va="top",
            fontsize=12, fontweight="bold")
    ax.text(0.5, 0.01, f"Observed: {strip['verdict']} | {xmap['verdict']}",
            ha="center", va="bottom", fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _status(verdict):
    return f"""# FCXR-LC2-GX1 status

Status: **COMPLETE — frozen component diagnostic**

- S1: `{verdict['selectivity_strip_verdict']}`
- X1: `{verdict['x_authority_verdict']}`
- Authorized next hypothesis: `{verdict['authorized_next_hypothesis']}`
- Numerical safety: {verdict['numerical_safe_rows']}/{verdict['numerical_total_rows']} rows
- Dynamic lifecycle: **not tested**
- M/K/A/ELR: **not used**

GX1 distinguishes entry selectivity from offset authority. It does not establish a spontaneous
interictal-ictal-interictal lifecycle or patient-like ictal morphology.
"""


def _archive(verdict, strip, xmap):
    return f"""# FCXR-LC2-GX1 frozen entry/offset diagnostics — 2026-08-02

## 一句话结论

GX1 在不改方程、不接动态慢变量的条件下，分别检验现有 H 方程是否自带易感性选择窗，以及 X
理论最大关断是否有权把 H 高态拉回间期。正式结果是：

- S1：`{strip['verdict']}`（{strip['n_pass']}/{strip['n_points']} 点通过，
  {strip['n_window_points']} 个点属于相邻窗）；
- X1：`{xmap['verdict']}`；
- 下一条获准检验的结构假说：`{verdict['authorized_next_hypothesis']}`。

## 测了什么

S1 固定 connection seed 1 / noise 401，在 H1/H6 两个既有家族上扫描低于旧下界的三个 H 增益
和两个阈值尺度。每点同时要求健康低初值、易感低初值保持间期，且易感高初值保持有限高态。相邻
两点同时通过才算自然参数窗。

X1 从同一个解析高 H 初值出发，把 recurrent relay availability 冻结为 1、0.5、0.1、0，检验
现有 X 路径的理论最大终止权限。x=0 只是一条结构性因果探针，不是生理参数。

## 科学边界

本轮只允许说明 frozen entry/offset component control。没有接 dynamic Z/X，没有跑无 kick
lifecycle，没有测试 M 形态、K 招募、A/ELR，也没有比较真实 E1146 ictal morphology。因此不能称为
迟滞、双稳态、极限环或可恢复发作闭环。

## 工程与资源

- strip trajectories: {strip['n_rows']}; X trajectories: {xmap['n_rows']};
- numerical safe: {verdict['numerical_safe_rows']}/{verdict['numerical_total_rows']};
- blessed engine files were checked by the execution lock;
- long stages used setsid/nohup, exact PID watchdogs, stage locks and sentinels;
- final commit and test counts are recorded in `run_manifest.json`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-finalize", action="store_true")
    args = ap.parse_args()
    if not args.confirm_finalize:
        raise SystemExit("--confirm-finalize is required")
    strip = _load("selectivity_strip.json")
    xmap = _load("x_authority_map.json")
    if strip.get("status") != "COMPLETE" or xmap.get("status") != "COMPLETE":
        raise SystemExit("S1 and X1 must both be complete")
    verdict = build_candidate_verdict(strip, xmap)
    figures = os.path.join(OUT, "figures")
    os.makedirs(figures, exist_ok=True)
    plot_strip(strip, os.path.join(figures, "selectivity_strip.png"))
    plot_x(xmap, os.path.join(figures, "x_authority.png"))
    plot_logic(strip, xmap, os.path.join(figures, "failure_logic.png"))
    readme = """### selectivity_strip.png

两行分别是 H1/H6，三列分别检验健康低初值、易感低初值和易感高初值；格内数字是末 1 秒平均率。
颜色区分间期、升高事件串、有限高态和未解析结果。
**关注点**：是否存在同一家族内相邻两点同时满足“两个低态不点燃、易感高态能维持”。

### x_authority.png

同一个易感高 H 初值下，比较四档 frozen relay availability 的 300 ms 平滑率与 H 轨迹。
x=0 是理论最大权限因果探针，不代表生理可实现值。
**关注点**：最大关断是否让末段连续至少 2 秒回到间期，而不只是出现短 trough。

### failure_logic.png

把 S1 的自然选择窗结论与 X1 的路径权限结论组成决策表，橙色格表示当前数据落点。
**关注点**：下一步获准检验的是参数范围、局部 D gate、共享 X/H 路径，还是两者的因果 2×2。
"""
    _write_text(os.path.join(figures, "README.md"), readme)
    _write_json("candidate_verdict.json", verdict)
    _write_text(os.path.join(OUT, "STATUS.md"), _status(verdict))
    _write_text(ARCHIVE, _archive(verdict, strip, xmap))
    manifest = dict(status="FINALIZED", head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        candidate_verdict=verdict, artifacts={}, finalized_at=_now())
    for rel in ("execution_lock.json", "selectivity_strip_manifest.json",
                "selectivity_strip.json", "x_authority_manifest.json",
                "x_authority_map.json", "candidate_verdict.json", "STATUS.md",
                "figures/selectivity_strip.png", "figures/x_authority.png",
                "figures/failure_logic.png", "figures/README.md"):
        path = os.path.join(OUT, rel)
        manifest["artifacts"][rel] = dict(path=path, sha256=_sha256(path))
    _write_json("run_manifest.json", manifest)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
