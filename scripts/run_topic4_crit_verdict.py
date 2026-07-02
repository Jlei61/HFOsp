"""Assembly (Task 3a-5b): the REAL M3A-v2.2 approach-criticality verdict + Figure 1.

Runs the v2.2 transition SNN ONCE, evaluates the frozen-Jacobian read-out on the actual 3-D slow
trajectory (q_I depletion + h_G recovery; g_K only if the sim coupled it), classifies it into the
pre-registered verdict {smooth_CSD, hard_jump_no_CSD, unresolved_operating_point}, and writes:

    <out-dir>/trajectory_verdict.json   (verdict_source="actual_trajectory" -- NOT the 2-D atlas)
    <out-dir>/STATUS.md                 (branch-aware frozen-Jacobian PRELIMINARY verdict framing)
    <out-dir>/figures/trajectory_criticality_verdict.png  (+ figures/README.md, 中文)
    <out-dir>/handoff/                  (the fail-closed overlay audit, reused from the SAME sim)

The phase-map overlay is drawn ONLY IF the M3A->M3B interface returns overlay_verdict==
phase_map_trajectory; for the real uncalibrated v2.2 it is REFUSED (mechanism_candidate_only), so no
overlay is drawn -- the refusal + reason are recorded in STATUS.md. This is a PRELIMINARY model-side
probe pending Milestone-2 spot-check / attribution / controls; it never claims the model proves CSD.

    python scripts/run_topic4_crit_verdict.py --out-dir results/topic4_criticality
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ENUM = ("smooth_CSD", "hard_jump_no_CSD", "unresolved_operating_point")

# discrete mode-class palette (self-contained figure legend; no §X / cluster_id codes).
_MODE_COLORS = {
    "stable": "#4c72b0", "local": "#dd8452", "axial": "#c44e52", "mixed": "#8172b3",
    "global": "#55a868", "runaway": "#000000", "unresolved": "#b0b0b0",
}


def _sanitize(obj):
    """Recursively replace non-finite floats (inf from an alpha1>=0 adiabatic sentinel, NaN) with
    None so the written JSON is strict-parser safe; the in-memory payload keeps the raw values."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def build_and_write_verdict(out_dir, *, layout: str = "subject1146", top: str = "qI") -> dict:
    """Run the v2.2 sim once, build the trajectory verdict, write all artifacts, return the payload.

    Importable by the integration test (asserts verdict_source + the 3-enum on the written JSON)."""
    from src.sef_hfo_transition_sim import run_transition, default_transition_config, sim_dict_for_handoff
    from src.sef_hfo_m3a_export import (
        default_precalib_mapping_and_ranges, build_handoff_from_sim, write_handoff_artifacts)
    from src.topic4_criticality import build_trajectory_verdict, load_crit_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tcfg = default_transition_config(layout=layout, top=top)
    sim = run_transition(tcfg)                                    # ONE SNN run (T2 review: reuse it)
    sim["use_gK"] = bool(tcfg["use_gK"])                          # annotate coupling for the evaluator
    sim["use_hG"] = bool(tcfg["use_hG"])

    mapping, _ranges = default_precalib_mapping_and_ranges("m3a_v2_2_approach")
    cfg = load_crit_config()
    payload, points = build_trajectory_verdict(sim, mapping, cfg)

    # overlay gate -- from the SAME sim (no re-run): the M3A->M3B interface's honest verdict.
    h = build_handoff_from_sim(sim_dict_for_handoff(sim), sim["events"], sim["dt_ms"],
                               mapping_id="m3a_v2_2_approach", gk_enabled=sim["use_gK"])
    audit = write_handoff_artifacts(str(out_dir / "handoff"), **h)
    overlay_verdict = audit["overlay_verdict"]
    overlay_drawn = overlay_verdict == "phase_map_trajectory"
    onset = float(sim["events"][0]["t_on"]) if sim["events"] else None
    payload["provenance"]["runaway_onset_ms"] = onset
    payload["overlay_verdict"] = overlay_verdict
    payload["overlay_drawn"] = bool(overlay_drawn)
    payload["overlay_reason"] = (
        "phase_map overlay drawn (interface calibrated)" if overlay_drawn else
        f"phase_map overlay REFUSED (overlay_verdict={overlay_verdict}); uncalibrated slow->rate "
        "mapping -> mechanism_candidate_only, no atlas overlay (Hard-QC #7)")

    (out_dir / "trajectory_verdict.json").write_text(
        json.dumps(_sanitize(payload), indent=1), encoding="utf-8")
    _write_status(out_dir, payload)
    _plot_verdict(out_dir / "figures", payload, onset)
    return payload


def _write_status(out_dir: Path, payload: dict) -> None:
    """STATUS.md -- plain-language (CLAUDE.md §8) + the mandated PRELIMINARY framing line."""
    v = payload["verdict"]
    prov = payload["provenance"]
    nq = payload["n_qualified_points"]
    nland = prov["n_landmarks"]
    a1 = payload.get("last_stable_alpha1")
    a1s = f"{a1:.4g}" if isinstance(a1, (int, float)) else "n/a"
    injected = ", ".join(prov["slow_vars_injected"])
    lines = [
        "# M3A-v2.2 approach-to-criticality — frozen-Jacobian verdict (PRELIMINARY)",
        "",
        "**Output framing:** branch-aware frozen-Jacobian PRELIMINARY verdict, pending Milestone-2 "
        "spot-check / attribution / controls. This does NOT claim the model proves CSD exists or is absent.",
        "",
        "## 测了什么",
        "我们把一次会“跑飞”的仿真（抑制资源 q_I 慢慢耗尽、全局恢复变量 h_G 参与）沿时间抽了若干快照，"
        "在每个快照把网络当下的慢状态冻住，问一句：如果现在轻轻推一下，这个网络回弹得快不快？回弹越慢，"
        "说明它越靠近“一推就失稳”的临界点。",
        "",
        "## 怎么测的",
        f"每个快照都在一个小的降维率场模型上求解当下的静止工作点（低放电支），再算冻结雅可比的主特征值 "
        f"α₁（越接近 0 越临界；τ=−1/α₁ 是回弹时间）。只有通过质量门（收敛、非饱和、准静态、率不失配）的"
        f"低支点才算数。抽了 {nland} 个快照，其中 {nq} 个合格；注入的慢变量：{injected}。",
        "",
        "## 揭示了什么",
        {
            "smooth_CSD": f"看起来像：合格点上的 α₁ 随失抑制单调抬升并平滑逼近 0（末点 α₁≈{a1s}），"
                          "τ 同步拉长——像“临界慢化”式的软着陆。",
            "hard_jump_no_CSD": f"看起来不像软着陆：合格低支点的 α₁ 一直离 0 有明显余量（末点 α₁≈{a1s}），"
                                "随后系统在很短窗口内直接跳进饱和/跑飞；分支延续核验确认中间没有被跳过的 α₁≈0 低支点。",
            "unresolved_operating_point": "没看清：合格低支点太少、分支身份不干净、或系统已被 ramp 拖着跑而非"
                                          "准静态小扰动恢复——在当前口径下判不了软着陆还是硬跳。",
        }.get(v, ""),
        "",
        "## Overlay 决策",
        f"- overlay_verdict = `{payload.get('overlay_verdict')}`；overlay_drawn = "
        f"`{payload.get('overlay_drawn')}`。",
        f"- {payload.get('overlay_reason')}",
        "",
        "## 关键字段（内部归档代号，括号补注）",
        f"- verdict = `{v}`（∈ smooth_CSD / hard_jump_no_CSD / unresolved_operating_point）",
        f"- verdict_source = `{payload.get('verdict_source')}`（来自真实 3-D 轨迹，不是 2-D atlas）",
        f"- operator_type = `{payload.get('operator_type')}`，alpha_units = `{payload.get('alpha_units')}`",
        f"- tier = `{payload.get('tier')}`",
        "",
        "阈值敏感性、每点 α₁/τ、mode-class、非正规放大（numerical_abscissa / directional_gain）见 "
        "`trajectory_verdict.json`；诊断图见 `figures/trajectory_criticality_verdict.png`。",
        "",
    ]
    (out_dir / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_verdict(fig_dir: Path, payload: dict, onset) -> None:
    """Figure 1 (DIAGNOSTIC): (A) α₁ vs disinhibition on the qualified-low-branch mask + α₁=0 line +
    mode-class colour; (B) relaxation time τ=−1/α₁ vs time (critical-slowing signature)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig_dir.mkdir(parents=True, exist_ok=True)
    points = payload["points"]
    v = payload["verdict"]
    tol = 0.002  # alpha_near_zero_tol_per_ms (config-of-record default; band width only)

    qual = [p for p in points if p.get("qualified") and p.get("branch_id") == "low_branch"]
    unqual_a = [p for p in points if not p.get("qualified") and p.get("alpha1") is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # --- Panel A: alpha1 vs sheet-mean q_I (disinhibition control axis) ---
    ax1.axhline(0.0, color="#c44e52", ls="--", lw=1.4, zorder=2)          # the alpha1=0 "contour"
    ax1.axhspan(-tol, tol, color="#c44e52", alpha=0.10, zorder=1)         # near-zero tolerance band
    for p in unqual_a:                                                    # context: not on the mask
        ax1.scatter(p["q_global"], p["alpha1"], s=22, facecolors="none",
                    edgecolors="#b0b0b0", lw=0.8, zorder=3)
    for p in qual:                                                        # the qualified-low-branch mask
        ax1.scatter(p["q_global"], p["alpha1"], s=42,
                    color=_MODE_COLORS.get(p.get("mode_class"), "#333333"),
                    edgecolors="k", lw=0.4, zorder=5)
    if qual:
        xs = [p["q_global"] for p in qual]
        ax1.plot(xs, [p["alpha1"] for p in qual], color="#555555", lw=0.8, alpha=0.5, zorder=4)
    ax1.invert_xaxis()                                                    # depletion (approach) reads left->right
    ax1.set_xlabel("sheet-mean $q_I$ (inhibition efficacy) —  more disinhibited →")
    ax1.set_ylabel(r"leading eigenvalue $\alpha_1$ (per ms)")
    ax1.set_title("α₁ approach vs disinhibition", fontsize=10, fontweight="bold")
    ax1.grid(alpha=0.25)

    # --- Panel B: relaxation time tau = -1/alpha1 (alpha1<0) vs time ---
    tau_pts = [(p["time_ms"], -1.0 / p["alpha1"], p.get("mode_class")) for p in qual
               if p.get("alpha1") is not None and p["alpha1"] < 0]
    if tau_pts:
        ax2.plot([t for t, _, _ in tau_pts], [u for _, u, _ in tau_pts],
                 color="#555555", lw=0.8, alpha=0.5, zorder=3)
        for t, u, mc in tau_pts:
            ax2.scatter(t, u, s=42, color=_MODE_COLORS.get(mc, "#333333"),
                        edgecolors="k", lw=0.4, zorder=5)
        ax2.set_yscale("log")
    else:
        ax2.text(0.5, 0.5, "no qualified low-branch points\n(verdict: unresolved)",
                 ha="center", va="center", transform=ax2.transAxes, color="#777777")
    if onset is not None:
        ax2.axvline(onset, color="#000000", ls=":", lw=1.3, zorder=2,
                    label=f"runaway onset ≈ {onset:.0f} ms")
        ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel(r"relaxation time $\tau=-1/\alpha_1$ (ms)")
    ax2.set_title("critical slowing (τ growth)", fontsize=10, fontweight="bold")
    ax2.grid(alpha=0.25, which="both")

    # --- shared mode-class legend (only classes actually PLOTTED) + overlay annotation ---
    plotted = [p.get("mode_class") for p in (qual + unqual_a)]
    present = [m for m in _MODE_COLORS if m in plotted]
    handles = [Line2D([0], [0], marker="o", ls="", color=_MODE_COLORS[m], mec="k", mew=0.4,
                      ms=7, label=m) for m in present]
    if unqual_a:
        handles.append(Line2D([0], [0], marker="o", ls="", mfc="none", mec="#b0b0b0", ms=7,
                              label="unqualified (context)"))
    fig.legend(handles=handles, loc="lower center", ncol=max(len(handles), 1),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    # saturated / runaway landmarks carry no alpha1 (no low branch) -> cannot appear on an
    # alpha1-vs-x or tau-vs-t axis; state their count so the figure is honest about what is omitted.
    n_sat = sum(1 for p in points if p.get("saturated"))
    overlay_note = ("phase-map overlay: DRAWN" if payload.get("overlay_drawn")
                    else "phase-map overlay: REFUSED — uncalibrated slow→rate mapping")
    fig.suptitle(f"M3A-v2.2 approach-to-criticality — frozen-Jacobian verdict: {v}   (PRELIMINARY)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.90, f"{payload['n_qualified_points']}/{payload['provenance']['n_landmarks']} "
             f"qualified low-branch landmarks   ·   {n_sat} saturated (no α₁, not plotted)   ·   {overlay_note}",
             ha="center", fontsize=8.5, color="#444444")

    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    fig.savefig(fig_dir / "trajectory_criticality_verdict.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _write_fig_readme(fig_dir, payload)


def _write_fig_readme(fig_dir: Path, payload: dict) -> None:
    v = payload["verdict"]
    txt = (
        "### trajectory_criticality_verdict.png\n"
        "这张图诊断 M3A-v2.2 那条“抑制耗尽→跑飞”的真实轨迹是不是“临界慢化式软着陆”。左图把每个"
        "合格快照的主特征值 α₁ 画在失抑制轴上（红虚线是 α₁=0 临界线，灰点是不合格点仅作背景，颜色是"
        "mode-class）；右图把回弹时间 τ=−1/α₁ 沿时间画出来（对数轴），看它有没有在跑飞前拉长。"
        f"当前判定：{v}。overlay 是否叠加见标题注记（未标定映射时 REFUSED，不画 atlas 背景）。\n"
        "**关注点**：左图 α₁ 是平滑逼近 0（软着陆）还是一直离 0 有余量后直接跳（硬跳）；右图 τ 有没有临界式发散。\n"
    )
    (fig_dir / "README.md").write_text(txt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/topic4_criticality",
                    help="directory for trajectory_verdict.json / STATUS.md / figures/")
    ap.add_argument("--layout", default="subject1146", choices=["stage5", "subject1146"])
    ap.add_argument("--top", default="qI", choices=["hG", "qI"])
    args = ap.parse_args()
    payload = build_and_write_verdict(args.out_dir, layout=args.layout, top=args.top)
    assert payload["verdict"] in _ENUM, payload["verdict"]
    print(f"verdict={payload['verdict']}  verdict_source={payload['verdict_source']}  "
          f"n_qualified={payload['n_qualified_points']}/{payload['provenance']['n_landmarks']}  "
          f"overlay={payload['overlay_verdict']}  out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
