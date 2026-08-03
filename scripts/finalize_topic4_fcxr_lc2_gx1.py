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
from matplotlib.patches import Patch
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


def summarize_entry_geometry(strip):
    """Describe the strongest locked-strip component without upgrading it to a basin."""
    points = list(strip.get("point_rows", []))
    selective = []
    for point in points:
        arms = {a["arm"]: a for a in point.get("arms", []) if "arm" in a}
        if set(arms) != {"healthy_low", "susceptible_low", "susceptible_high"}:
            continue
        healthy_low = arms["healthy_low"].get("workpoint_label") == "INTERICTAL_WORKPOINT"
        susceptible_low_high = arms["susceptible_low"].get("workpoint_label") in (
            "FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
        susceptible_high_high = arms["susceptible_high"].get("workpoint_label") in (
            "FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
        if healthy_low and susceptible_low_high and susceptible_high_high:
            selective.append(point["point_id"])
    return dict(
        component_label=("D_SELECTIVE_ONE_WAY_IGNITION_WITHOUT_DUAL_BASIN"
                         if selective else "NO_D_SELECTIVE_IGNITION_COMPONENT"),
        selective_one_way_points=selective,
        natural_dual_basin_window=bool(strip.get("n_window_points", 0)),
        explicit_d_gate_status=(
            "AUTHORIZED_AS_FALSIFIABLE_HYPOTHESIS_NOT_PROVEN_SUFFICIENT"
            if strip.get("verdict") == "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP"
            else "NOT_AUTHORIZED"),
    )


def summarize_x_authority(xmap):
    rows = sorted((r for r in xmap.get("rows", []) if "x_availability" in r),
                  key=lambda r: float(r["x_availability"]), reverse=True)
    returning = sorted(float(v) for v in xmap.get("returning_availabilities", []))
    nonreturning = sorted(
        float(r["x_availability"]) for r in rows
        if r.get("required_low_workpoint_label") != "INTERICTAL_WORKPOINT")
    smallest_nonreturning_above = None
    if returning:
        ret_hi = max(returning)
        above = [v for v in nonreturning if v > ret_hi]
        smallest_nonreturning_above = min(above) if above else None
    return dict(
        current_x_path_reachable=bool(returning),
        h_actuator_bypasses_x=False if returning else None,
        largest_tested_returning_availability=max(returning) if returning else None,
        smallest_tested_nonreturning_availability_above_return=(
            smallest_nonreturning_above),
        experimental_return_bracket=(
            [max(returning), smallest_nonreturning_above]
            if returning and smallest_nonreturning_above is not None else None),
        archived_lc1_availability_levels=[0.872, 0.786],
        archived_lc1_range_status="INSUFFICIENT_FOR_THIS_H_BRANCH",
        physiological_validity_of_returning_probe="NOT_ESTABLISHED",
    )


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
        entry_geometry=summarize_entry_geometry(strip),
        x_authority=summarize_x_authority(xmap),
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
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.7), constrained_layout=True)
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
    fig.legend(handles=[
        Patch(facecolor="#4c78a8", label="interictal workpoint"),
        Patch(facecolor="#f2cf5b", label="elevated event train"),
        Patch(facecolor="#d1495b", label="finite high state"),
        Patch(facecolor="#777777", label="unresolved"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=4,
       frameon=False, fontsize=8)
    fig.suptitle("GX1 entry strip: no natural low/high dual-basin window\n"
                 "cell text = final-1 s rate (Hz)", fontsize=11, fontweight="bold")
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
         "Test local D-gate; calibrate X separately", (not natural) and reachable),
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
    observed = ("Observed: no dual-basin window | X path reachable"
                if (not natural) and reachable else
                f"Observed: {strip['verdict']} | {xmap['verdict']}")
    ax.text(0.5, 0.01, observed, ha="center", va="bottom", fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _status(verdict):
    entry = verdict["entry_geometry"]
    xauth = verdict["x_authority"]
    return f"""# FCXR-LC2-GX1 status

Status: **COMPLETE — frozen component diagnostic**

- S1: `{verdict['selectivity_strip_verdict']}`
- X1: `{verdict['x_authority_verdict']}`
- Authorized next hypothesis: `{verdict['authorized_next_hypothesis']}`
- Numerical safety: {verdict['numerical_safe_rows']}/{verdict['numerical_total_rows']} rows
- Entry component: `{entry['component_label']}`
- Natural low/high dual-basin window: **no**
- X path reachable: **{str(xauth['current_x_path_reachable']).lower()}**; tested return bracket:
  `{xauth['experimental_return_bracket']}`
- Dynamic lifecycle: **not tested**
- M/K/A/ELR: **not used**

GX1 distinguishes entry selectivity from offset authority. It does not establish a spontaneous
interictal-ictal-interictal lifecycle or patient-like ictal morphology.
"""


def _archive(verdict, strip, xmap):
    entry = verdict["entry_geometry"]
    xauth = verdict["x_authority"]
    xrows = {float(row["x_availability"]): row for row in xmap["rows"]}
    return f"""# FCXR-LC2-GX1 frozen entry/offset diagnostics — 2026-08-02

## 一句话结论

GX1 在不改方程、不接动态慢变量的条件下，分别检验现有 H 方程是否自带易感性选择窗，以及 X
理论最大关断是否有权把 H 高态拉回间期。正式结果是：

- S1：`{strip['verdict']}`（{strip['n_pass']}/{strip['n_points']} 点通过，
  {strip['n_window_points']} 个点属于相邻窗）；
- X1：`{xmap['verdict']}`；
- 下一条获准检验的结构假说：`{verdict['authorized_next_hypothesis']}`。

这不是一个笼统的双阴性。S1 在 H1、`theta_scale=1.25` 的三个 rho 点上都看到了同一分解：
健康 `D=0/H_low` 保持约 4.2 Hz 的间期工作点，而易感 `D=0.15/H_low` 已经升到
54.8--91.5 Hz；易感高初值也维持 58.7--87.6 Hz。也就是说，现有方程已经出现
**D 选择性的单向点火**，但同一个易感 D 下低初值和高初值都落到高态，因此不是目标中的低/高
双盆地。显式 D gate 只被授权为下一条可证伪假说，并未被本轮证明充分或唯一必要。

X1 则给出清楚的权限括号：availability=1.0/0.5 仍为高态（尾段
{xrows[1.0]['state_tail_1s']['rate_mean_hz']:.1f}/{xrows[0.5]['state_tail_1s']['rate_mean_hz']:.1f} Hz），
0.1/0.0 在末段连续 2 s 回到间期（尾段
{xrows[0.1]['state_tail_1s']['rate_mean_hz']:.3f}/{xrows[0.0]['state_tail_1s']['rate_mean_hz']:.3f} Hz）。
所以 H 没有结构性绕过 X；当前路径可以终止，已观察 LC1 availability 0.872/0.786 的动态范围
对这条 H 分支不足。0.1 只是理论实验臂，不具有生理标定资格。

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

## 下一结构的授权边界

预注册决策表落在“no natural window + X path reachable”。因此只授权将**局部 D-dependent H
gain**作为 entry 几何的下一条独立假说，同时把 X 动态范围作为另一条独立校准问题。完整的
`D gate × shared X/H path` 2×2 只在“no window + maximal-X bypass”时才有资格执行；本轮已经
否定了 bypass 前提，所以随附 GX2 2×2 spec/plan 仅作条件性预案，当前不得执行。

## 工程与资源

- strip trajectories: {strip['n_rows']}; X trajectories: {xmap['n_rows']};
- numerical safe: {verdict['numerical_safe_rows']}/{verdict['numerical_total_rows']};
- blessed engine files were checked by the execution lock;
- long stages used setsid/nohup, exact PID watchdogs, stage locks and sentinels;
- S1 watchdog elapsed 5.933 h; X1 watchdog elapsed 1.008 h;
- peak single-cell RSS 11.236 GiB; swap delta 0 MiB;
- final commit and test counts are recorded in `run_manifest.json` after sign-off.
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
**关注点**：没有一个点满足“两个低态不点燃、易感高态能维持”；H1 高阈值行显示的是健康低态
保留、易感低态也点燃的单向 D 选择性，不要误读成双稳态。

### x_authority.png

同一个易感高 H 初值下，比较四档 frozen relay availability 的 300 ms 平滑率与 H 轨迹。
x=0 是理论最大权限因果探针，不代表生理可实现值。
**关注点**：availability=0.1 已让末段连续至少 2 秒回到间期，说明通路可达；0.5 仍维持高态，
故当前证据指向动态范围不足而不是 H 绕过 X。

### failure_logic.png

把 S1 的自然选择窗结论与 X1 的路径权限结论组成决策表，橙色格表示当前数据落点。
**关注点**：当前落点只授权局部 D gate 作为 entry 假说，并把 X 范围单独处理；共享路径与完整
2×2 当前未获授权。
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
