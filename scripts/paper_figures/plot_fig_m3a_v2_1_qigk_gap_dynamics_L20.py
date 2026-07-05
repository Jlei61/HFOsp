"""M3A-v2.1 q_I+g_K gap-sweep dynamics, L=20 variant (Topic-4 SNN four-panel style).

Same canonical grammar and the SAME representative cases as the L=10
``plot_fig_m3a_v2_1_qigk_gap_dynamics.py`` -- this only scales the sheet to L=20 (the
quantified gap sweep ran at L=10; the user asked to also eyeball the dynamics on the
larger L=20 sheet). It REUSES every painter + helper + the CASES from the L=10 script
unchanged; the only differences are: L=20 substrate, a per-process build cache (an L=20
build is ~100 s and all cases share one (substrate, seed)), a wider contact pitch so the
∥ shaft spans the L=20 axis ends, and a distinct output directory.

    mechanism | tempA source | tempB source | electrode readout

Visual diagnostic, NOT a new statistical sweep. tempA/tempB + readout shading are SOURCE
IDENTITY, never propagation/seizure direction. Source of truth stays the gap-sweep
per_run.jsonl + REPORT.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENG = ROOT / "src" / "snn_engine"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "paper_figures"))
sys.path.insert(0, str(ENG))

import run_m3a_v2_step2_qI as S2  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
# reuse the L=10 script's locked painters, helpers, and representative CASES verbatim
import plot_fig_m3a_v2_1_qigk_gap_dynamics as G  # noqa: E402
from plot_fig_m3a_v2_1_qigk_gap_dynamics import (  # noqa: E402
    CASES, _plot_mechanism, _plot_event, _plot_readout,
    _source_xy, _source_vth, _make_field, _event_window, _onset_from_window, _source_metrics,
)

FIG_NAME = "fig_m3a_v2_1_qigk_gap_dynamics_L20"
OUT = ROOT / "results" / "paper-ready-figure" / FIG_NAME / "figures"

L_SHEET = 20.0
N_CONTACTS_PER_SHAFT = 7
CONTACT_PITCH = 2.0      # mm; spans the L=20 axis ends (source ends sit at +-0.6*half along axis)

_BUILD: dict = {}


def _build_cached(sub_id: str, seed: int):
    key = (sub_id, int(seed))
    if key not in _BUILD:
        _BUILD[key] = S2.build(S2.SUBSTRATES[sub_id], int(seed), L=L_SHEET)
    return _BUILD[key]


def _contacts(S: dict):
    center = np.asarray(S["center"], float)
    u = np.asarray(S["axis_unit"], float)
    p = np.array([-u[1], u[0]])
    half = (N_CONTACTS_PER_SHAFT - 1) / 2.0
    offsets = (np.arange(N_CONTACTS_PER_SHAFT) - half) * CONTACT_PITCH
    a = np.array([center + d * u for d in offsets])
    b = np.array([center + d * p for d in offsets])
    names = [f"A{i}" for i in range(N_CONTACTS_PER_SHAFT)] + [f"B{i}" for i in range(N_CONTACTS_PER_SHAFT)]
    return np.vstack([a, b]), names


def _run_probe(case: dict, source: str):
    base = _build_cached(case["substrate"], case["seed"])
    S = dict(base)                                   # shallow copy: per-source vth/core_xy override
    S["core_xy"] = _source_xy(S, source)
    S["vth"] = _source_vth(S, source)
    contacts, names = _contacts(S)
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["labels"], sites=contacts)
    field = _make_field(S, case)
    seed_offset = 0 if source == "tempA" else 1000
    S["net"]["rng"] = np.random.default_rng(int(case["seed"]) + seed_offset)
    res = simulate_kick(S["p"], S["net"], KICK_BOOST=S2.KICK, slow=field, kick_center=S["core_xy"],
                        r_kick=0.3, t_kick=S2.T_KICK, V_th_per_neuron=S["vth"], lfp_recorder=rec)
    t_on, t_off, sl = _event_window(res["rate_E"], S["p"].dt)
    onset = _onset_from_window(res["E_spk_bool"], S["p"].dt, t_on, min(t_off, S["p"].T - S["p"].dt))
    metrics = _source_metrics(res, S, onset, sl)
    return dict(S=S, field=field, res=res, contacts=contacts, names=names, t_on=t_on, t_off=t_off,
                onset=onset, metrics=metrics, source=source, case=case)


def _compose_case(case: dict):
    runA = _run_probe(case, "tempA")
    runB = _run_probe(case, "tempB")
    fig = plt.figure(figsize=(18.0, 4.45), facecolor="white")
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 2.75],
                           left=0.045, right=0.992, bottom=0.16, top=0.82, wspace=0.075)
    _plot_mechanism(fig.add_subplot(gs[0, 0]), runA, runB)
    _plot_event(fig.add_subplot(gs[0, 1]), runA, "tempA source")
    _plot_event(fig.add_subplot(gs[0, 2]), runB, "tempB source")
    _plot_readout(fig.add_subplot(gs[0, 3]), runA, runB)
    fig.text(0.012, 0.925, "A", fontsize=19, fontweight="bold")
    fig.text(0.50, 0.93, f"{case['label']}  (L=20)", fontsize=10.0, fontweight="bold", ha="center")
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{case['case_id']}.png"
    pdf = OUT / f"{case['case_id']}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {case['case_id']}: tempA {runA['metrics']}  |  tempB {runB['metrics']}", flush=True)
    return dict(case=case, outputs={"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT))},
                tempA_metrics=runA["metrics"], tempB_metrics=runB["metrics"])


def _write_readme(summaries):
    text = """# M3A-v2.1 q_I+g_K gap-sweep dynamics (L=20)

L=20 版本，与 L=10 版 (`fig_m3a_v2_1_qigk_gap_dynamics/`) 同样的代表性 case、同样的四列画法，只是把基底放大到 L=20 让动力学在更大的片子上目视。每行一次仿真，四列：左=机制（兴奋性底物 plasma + E→E 长轴带 + 两端 source 区 A/B + 虚拟电极 A/B 杆）；中两格=同一基底分别从 A 端、B 端作为 source 点一下的代表事件（点云颜色=传播起始相对时间 viridis 早→晚，星号=该事件实际 source 端）；右=同一虚拟 SEEG montage 的 readout，暖/浅蓝 shading 标 **source identity**（tempA-source probe / tempB-source probe），**不是**方向、**不是**发作方向。

**tempA/tempB 一律是 source identity（哪一端点火），不得读成方向或发作方向。** 视觉诊断，非统计主张；定量口径以 gap-sweep `per_run.jsonl` + `REPORT.md` 为准。

### baseline_axial.png
slow field 关掉的参照。两端 source 各自沿 E→E 长轴自限传播、能回到低活动。**关注点**：L=20 上的轴向自限基线长相。

### returned_axis_only_clean.png
旁边冻松（q_off=0.5）、不加疲劳的"最接近成功"一类。**关注点**：尽管旁边更松，活动仍主要沿轴、能平息——"松"单独不足以改道离轴。

### metric_edge_small_suppress.png
小footprint suppress（核心保护、轴向中等疲劳）。**关注点**：主轴读数可能偏低，但事件其实又小又能平息——提醒不要只看 S_axis。

### dynamic_gk_suppress.png
轴向强疲劳 g_K（核心保护）。**关注点**：g_K 把事件缩小/提前结束，是"压"不是"改道"，离轴没被招募。

### dynamic_gk_runaway.png
旁边很松（q_off=0.3）+ 轴向疲劳。**关注点**：离轴确实被招募，但以**不返回的 runaway**形式；轴向疲劳收不住它——"离轴只在 runaway 里出现"的 L=20 可视化。
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(text)


def main():
    os.chdir(ROOT)
    summaries = []
    for case in CASES:
        print(f"[plot L=20] {case['case_id']}", flush=True)
        summaries.append(_compose_case(case))
    _write_readme(summaries)
    meta = dict(
        figure=FIG_NAME, L=L_SHEET, status="visual diagnostic, not a new statistical sweep",
        reuses="plot_fig_m3a_v2_1_qigk_gap_dynamics.py (same CASES + painters; L=20 + build cache + wider pitch)",
        source_of_truth=["results/topic4_m3a_v2_1_qigk_gap_sweep/<ts>/per_run.jsonl",
                         "results/topic4_m3a_v2_1_qigk_gap_sweep/<ts>/REPORT.md"],
        notes=[
            "L=20 mirror of the single-core probe at the two scaffold ends; preserves the four-panel source-identity standard.",
            "Same representative q_I/g_K clamp cases as the L=10 figure; clamp values not re-tuned -- the L=20 metrics are reported as-run.",
            "Shading + tempA/tempB labels are SOURCE IDENTITY, not propagation direction or seizure direction.",
            "Mechanism-panel axis is scaffold ORIENTATION only (non-directional, no arrowhead).",
        ],
        cases=summaries,
    )
    (OUT / f"{FIG_NAME}_metadata.json").write_text(json.dumps(meta, indent=2))
    for s in summaries:
        print(f"wrote {ROOT / s['outputs']['png']} and {ROOT / s['outputs']['pdf']}", flush=True)
    print(f"wrote {OUT / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
