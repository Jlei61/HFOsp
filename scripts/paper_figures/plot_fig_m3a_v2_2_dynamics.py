"""M3A-v2.2 key-dynamics visual diagnostic (Topic-4 four-panel grammar).

Visual diagnostic ONLY (not a new statistical sweep): renders the v2.2 PILOT scenarios
(slow-off baseline + q_I+g_K sustained carrier + Exp-0 r_hold ladder rungs) under the
sustained ramp+HOLD protocol, in the same `mechanism | tempA source | tempB source |
electrode readout` standard as Fig5 / the Step-4 diagnostic. The mechanism-panel axis is
NON-DIRECTIONAL (scaffold orientation, no arrowhead -- reused from the fixed Step-4 panel).
Science gates live in the pilot JSON + tests; this figure adds no statistical claim.

Render is heavy (real SNN sims) -- run manually:
    python scripts/paper_figures/plot_fig_m3a_v2_2_dynamics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

# reuse the (non-directional) four-panel helpers + the sustained-protocol pilot runs
import plot_fig_m3a_v2_step4_dynamics as STEP4  # noqa: E402  panels: _plot_mechanism (non-directional) / _plot_event / _plot_readout
import run_m3a_v2_2_pilot as PILOT  # noqa: E402  _drive / _event_window / _participation / _segment_and_classify / S2
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402

STATUS = "visual diagnostic, not a new statistical sweep"
FIG_NAME = "fig_m3a_v2_2_dynamics"
OUT = ROOT / "results" / "paper-ready-figure" / FIG_NAME / "figures"

# Representative cases from the 3184-sim sweep (results/topic4_m3a_v2_2_explore/): the protocol's
# fail-closed tonic (slow-off and q_I+g_K) + the ONLY clean event found (backup r=0.85, seed 22).
CASES = [
    {"case_id": "slow_off_tonic", "label": "slow-off (sustained) — fail-closed tonic",
     "substrate": "primary", "slow": None, "r_hold": 0.65, "seed": 1},
    {"case_id": "qI_gK_tonic", "label": "q_I + g_K carrier — still fail-closed tonic",
     "substrate": "primary", "slow": "qI_gK", "r_hold": 0.60, "seed": 1},
    {"case_id": "backup_clean_blip", "label": "backup r=0.85 — the only clean event (tiny axial blip)",
     "substrate": "backup", "slow": None, "r_hold": 0.85, "seed": 22},
]


def _make_field(S, case):
    """The q_I+g_K carrier field (pilot cfg) for the 'qI_gK' case; None (slow-off) otherwise."""
    if case["slow"] != "qI_gK":
        return None
    cfg = SpatialSlowFieldConfig(use_qI=True, use_gK=True, use_hG=False,
                                 k_q=0.3, sigma_q=1.5, q_min=0.25,
                                 k_K=1.0, sigma_K=0.5, eta_K=1.0, tau_a=20.0)
    return SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)


def _run_probe(case, source, T=500.0):
    """Mirror the single-core probe at one scaffold end (source identity), under the SUSTAINED
    ramp+HOLD protocol (no kick: KICK_BOOST=0). Builds the `run` dict the Step-4 panels expect."""
    seed = int(case["seed"])
    S = PILOT.S2.build(PILOT.S2.SUBSTRATES[case["substrate"]], seed, T=T)
    S["core_xy"] = STEP4._source_xy(S, source)
    S["vth"] = STEP4._source_vth(S, source)
    contacts, names = STEP4._contacts(S)
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["labels"], sites=contacts)
    field = _make_field(S, case)
    seed_offset = 0 if source == "tempA" else 1000
    S["net"]["rng"] = np.random.default_rng(seed + seed_offset)        # paired per source
    nu_fn = PILOT._drive(S, case["r_hold"])
    res = simulate_kick(S["p"], S["net"], KICK_BOOST=0.0, slow=field, nu_signal_fn=nu_fn,
                        kick_center=S["core_xy"], r_kick=0.3, t_kick=50.0,
                        V_th_per_neuron=S["vth"], lfp_recorder=rec)
    dt = S["p"].dt
    seg = PILOT._event_window(res["rate_E"], dt)
    if seg is None:
        i_on, i_off = int(50.0 / dt), int(min(S["p"].T - dt, 90.0) / dt)
    else:
        i_on, i_off = seg[0], seg[1]
    _, onset = PILOT._participation(res["E_spk_bool"], i_on, i_off, dt)
    return {"S": S, "field": field, "res": res, "contacts": contacts, "names": names,
            "t_on": float(i_on * dt), "t_off": float(i_off * dt), "onset": onset,
            "metrics": PILOT._segment_and_classify(res, S), "source": source, "case": case}


def _compose_case(case):
    runA = _run_probe(case, "tempA")
    runB = _run_probe(case, "tempB")
    fig = plt.figure(figsize=(18.0, 4.45), facecolor="white")
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 2.75],
                           left=0.045, right=0.992, bottom=0.16, top=0.82, wspace=0.075)
    STEP4._plot_mechanism(fig.add_subplot(gs[0, 0]), runA, runB)       # non-directional axis
    STEP4._plot_event(fig.add_subplot(gs[0, 1]), runA, "tempA source")
    STEP4._plot_event(fig.add_subplot(gs[0, 2]), runB, "tempB source")
    STEP4._plot_readout(fig.add_subplot(gs[0, 3]), runA, runB)
    cls = runA["metrics"]["class_label"]
    fig.text(0.012, 0.925, "A", fontsize=19, fontweight="bold")
    fig.text(0.50, 0.925, f"{case['label']}  ·  tempA: {cls}", fontsize=10.0, fontweight="bold", ha="center")
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{case['case_id']}.png"
    pdf = OUT / f"{case['case_id']}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"case": case, "outputs": {"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT))},
            "tempA_metrics": runA["metrics"], "tempB_metrics": runB["metrics"]}


def _write_readme(summaries):
    lines = ["# M3A-v2.2 key-dynamics visual diagnostic\n",
             "> **当前渲染口径**：在当前采样下，5 个 case 全部读出 fail-closed 的 "
             "`TONIC_OR_MULTIBURST` / `INSUFFICIENT_FOR_EVENT_PHENOTYPE`——这只说明 fail-closed 正常工作，"
             "**不构成 “部分填充” 或 “恢复轨道” 候选**。要画真正的 key-dynamics 候选图，必须等正式 pilot "
             "跑出 clean single-event / returned anchor 之后再画。\n",
             "这组图是 **visual diagnostic**，不是新的统计 sweep。它把 v2.2 pilot 的关键动力学场景",
             "（持续 ramp+HOLD 驱动下的 slow-off 基线、`q_I+g_K` 载体、Exp-0 的 r_hold ladder）",
             "用 Topic-4 四列标准画出来：`mechanism | tempA source | tempB source | electrode readout`。",
             "mechanism 面板的轴线只表示 scaffold 朝向（**无箭头、非传播/发作方向**）；",
             "shading 标的是 source identity，不是方向。统计口径在 pilot JSON + tests，本图不下结论。\n"]
    blurbs = {
        "slow_off_baseline": ("持续驱动下不加任何慢变量的基线。看它是否仍是全或无 / 不回到基线（C1-A），"
                              "还是协议本身就变温和（C1-B）。", "事件是否回落、是否全场招募。"),
        "qI_gK_sustained": ("同一持续协议 + `q_I`（推）+ `g_K`（刹）载体。看刹车在有积分窗口时能否把蔓延"
                            "限在中途、造出部分填充。", "范围是否仍铺满全场、轴向是否仍主导。"),
        "ladder_low": ("Exp-0 标定 ladder 低档（r_hold=0.50）slow-off 锚点。", "属于 returned-axial 还是 runaway。"),
        "ladder_mid": ("Exp-0 标定 ladder 中档（r_hold=0.60）slow-off 锚点。", "属于 returned-axial 还是 runaway。"),
        "ladder_high": ("Exp-0 标定 ladder 高档（r_hold=0.75）slow-off 锚点。", "属于 returned-axial 还是 runaway。"),
    }
    for s in summaries:
        cid = s["case"]["case_id"]
        body, focus = blurbs.get(cid, ("v2.2 pilot 动力学场景。", "事件范围 / 轴向 / 是否回落。"))
        lines += [f"### {cid}.png\n", body, f"\n**关注点**：{focus}\n"]
    (OUT / "README.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for case in CASES:
        print(f"[plot] {case['case_id']}", flush=True)
        summaries.append(_compose_case(case))
    _write_readme(summaries)
    (OUT / "m3a_v2_2_dynamics_metadata.json").write_text(json.dumps(PILOT._json_safe(
        {"figure": FIG_NAME, "status": STATUS,
         "source_of_truth": ["results/topic4_m3a_v2_2_explore/<stamp>/per_run.jsonl (3184-sim sweep)",
                             "results/paper-ready-figure/fig_m3a_v2_2_explore_summary/ (result summary)"],
         "note_on_cases": "representative cases re-simulated for the four-panel view; the statistical "
                          "verdict lives in the sweep + the explore_summary figure, not in these single seeds",
         "notes": ["Mechanism-panel axis is scaffold ORIENTATION only (non-directional, no arrowhead).",
                   "Shading labels source identity, not propagation/seizure direction.",
                   "Carrier + pilot scenarios only; closed-loop h_G ideal-orbit figure is deferred.",
                   "CURRENT RENDER = fail-closed tonic/multiburst readout under the sustained protocol; "
                   "it is NOT a partial-fill or recovery candidate. A key-dynamics candidate figure "
                   "requires a real pilot producing clean single-event / returned anchors first."],
         "cases": summaries}), indent=2, allow_nan=False))   # strict JSON (NaN -> null)
    for s in summaries:
        print(f"wrote {ROOT / s['outputs']['png']}", flush=True)
    print(f"wrote {OUT / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
