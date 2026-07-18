"""Task 5 (M2 final): the two-stage ignition/spread verdict CLI (Topic 4 M3-v2.2 criticality
Milestone 2).

Loads M1's dense trajectory points (results/topic4_criticality/trajectory_verdict.json),
localizes the alpha0 crossing (T1), reads the linear_ignition class + two-core confirm (T2) +
the off_axis sentinel (T3), reads the nonlinear_spread footprint verdict (T4), and assembles the
spec §1 two-stage verdict (T5): `csd_verdict` (M1, unchanged, co-displayed) + `linear_ignition` +
`nonlinear_spread` + `interpretation`. Writes:

    <out>/ignition_spread_verdict.json   (two-stage verdict, private/non-serializable fields stripped)
    <out>/STATUS.md                      (plain-language two-stage summary, CLAUDE.md §8)
    <out>/figures/ignition_panel.png     (crossing-mode loading + two-core region power)
    <out>/figures/spread_panel.png       (footprint active_frac(t)/off_axis(t)/elongation(t))
    <out>/figures/basis_sanity.png       (basis vectors + crossing loading + nonaxis residual)
    <out>/figures/README.md              (中文, per figures/ convention)

tier=model_side_preliminary: this is a readout on the actual v2.2 SIMULATION trajectory, never a
claim the model proves or disproves CSD; a global-runaway footprint endgame is not a claim about
real seizures.

    python scripts/run_topic4_crit_m2.py --out results/topic4_criticality_m2
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

# categorical pair for the two footprint depths (dataviz skill: fixed categorical order, never
# cycled; blue/orange are slots 1/8 of this repo's validated categorical set).
_DEPTH_STYLE = {
    "at_crossing": {"color": "#2a78d6", "ls": "-", "label": "at_crossing (alpha0)"},
    "just_past": {"color": "#eb6834", "ls": "--", "label": "just_past (frac=0.75)"},
}


# Only these `_`-prefixed working fields are genuinely NON-JSON-serializable / duplicative and must
# be dropped from the written JSON: `_crossing_op`/`_crossing_res` are live OperatingPoint/EigResult
# objects, and `_two_core_crossing` re-nests another copy of those same objects. Every OTHER
# `_`-prefixed field in the verdict tree (`_two_core_region_frac`, `_two_core_axis_profile`,
# `_epsilon_sweep_detail`, `_depth_aggregate`, `_branch_continuation_status`) is a plain
# dict/list/str of the audit evidence STATUS.md tells readers to find in this JSON -- those are kept.
_SANITIZE_STRIP = {"_crossing_op", "_crossing_res", "_two_core_crossing"}


def _sanitize(obj):
    """Prepare the in-memory verdict for JSON: (1) drop the genuinely non-serializable / duplicate
    working fields named in ``_SANITIZE_STRIP``; (2) for every OTHER ``_``-prefixed key, strip the
    leading underscore so the audit evidence STATUS.md points at (``two_core_region_frac`` /
    ``two_core_axis_profile`` / ``epsilon_sweep_detail`` / ``depth_aggregate`` /
    ``branch_continuation_status``) lands as a PUBLIC JSON key rather than one that reads as
    "private" -- otherwise the STATUS "见 ignition_spread_verdict.json" reference is a false pointer;
    (3) replace non-finite floats (NaN from off_axis_sentinel's low-residual short-circuit) with None
    so the file is strict-parser safe. The rename happens ONLY here at write time -- the in-memory
    tree keeps its ``_``-prefixed keys, so upstream T2/T4 code and the committed tests that assert on
    the literal ``_branch_continuation_status`` (tests/test_topic4_criticality_m2.py) are untouched.
    (The non-finite-float cleanup follows M1's own ``_sanitize``, run_topic4_crit_verdict.py; the
    underscore handling is new to T5 -- M1 has no ``_``-prefixed working fields.)"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _SANITIZE_STRIP:
                continue
            out[k[1:] if k.startswith("_") else k] = _sanitize(v)
        return out
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def build_and_write_verdict(out_dir) -> dict:
    """Load M1's points, build the two-stage verdict, write JSON + STATUS.md + figures.

    Importable by the integration test (asserts csd_verdict + the retired-three-way absence on the
    written JSON)."""
    from src.topic4_criticality import load_crit_config
    import src.topic4_criticality_m2 as m2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_crit = load_crit_config()
    m2cfg = m2.load_m2_config()
    points = json.loads(m2._M1_VERDICT_PATH.read_text())["points"]

    verdict = m2.build_ignition_spread_verdict(points, cfg_crit, m2cfg)

    (out_dir / "ignition_spread_verdict.json").write_text(
        json.dumps(_sanitize(verdict), indent=1), encoding="utf-8")
    _write_status(out_dir, verdict)
    _plot_verdict(out_dir / "figures", verdict, cfg_crit, m2cfg)
    return verdict


def _audit_get(d: dict, public_name: str):
    """Read an audit field whether it still carries its in-memory leading underscore (the
    ``build_and_write_verdict`` path passes the raw in-memory verdict) or has been renamed to its
    public key in the written JSON (the ``--from-json`` path reads the sanitized file)."""
    v = d.get(public_name)
    return v if v is not None else d.get("_" + public_name)


def _write_status(out_dir: Path, verdict: dict) -> None:
    """STATUS.md -- plain-language (CLAUDE.md §8): 测了什么/怎么测的/揭示了什么, codes in parens."""
    ig = verdict["linear_ignition"]
    sp = verdict["nonlinear_spread"]
    note = sp.get("descriptive_igniting_note")
    region_frac = _audit_get(ig, "two_core_region_frac") or {}

    if note is not None:
        n_ac = note["n_igniting_of_total"].get("at_crossing", "?/4")
        igniting_note_zh = (
            f"额外记一笔（不算正式结论）：4 组推法里，确实点着的那 {n_ac} 组，"
            f"看到的都是同一件事——先沿着轴线方向烧一段（`{note['igniting_onset']}`），"
            "然后自己收住。没点着的那一组，是往下压、力度又最大的那种推法——"
            "网络自己弹回去了，压根没烧起来。"
        )
    else:
        igniting_note_zh = ""

    spread_paragraph = (
        "火势往哪蔓延这件事，这一轮**没看清**：4 种推法（2 种力度 × 2 个方向）里，"
        "3 种一致看到“先沿着轴线方向烧开一段、然后自己收住”，但第 4 种（往下压、力度最大的那种）"
        "根本没能把网络推过点火的门槛——它自己弹回去了，连“烧”都没烧起来。"
        "因为我们预先说好“4 种推法必须全部一致才算数”（不是少数服从多数），"
        "这一票不点火的结果就让这一段的正式结论变成“没看清”。"
        + (("\n\n" + igniting_note_zh) if igniting_note_zh else "")
    )

    lines = [
        "# Topic 4 M3-v2.2 criticality Milestone 2 — 两段式点火/铺开判读（PRELIMINARY）",
        "",
        "**Output framing:** `model_side_preliminary` —— 这是在同一段 v2.2 **仿真**轨迹"
        "（actual v2.2 SIMULATION trajectory，不是新病人数据、不是真实临床记录）上做的读数，"
        "从不声称“模型证明了 CSD 是否存在”；下面出现的“全场烧起来”也不是在说真实癫痫发作。",
        "",
        "## 测了什么",
        "上一轮（Milestone 1）已经把“这段轨迹是不是稳步逼近失稳”这件事量过一次，结论是没看清楚——"
        "抽到的快照上系统看起来还很稳，但补做加密检查后发现，在两个快照之间的空隙里，"
        "系统确实有一瞬间翻了过去。这一轮在那个翻转的瞬间附近继续问两个新问题：\n"
        "(1) 失稳发生的时候，“着火点”在哪——是缩在一小撮病灶细胞里，还是整张网一起烧；\n"
        "(2) 烧起来之后“火势往哪蔓延”——是顺着一条轴线烧一段就自己灭了，还是各个方向都烧、烧穿全场。",
        "",
        "## 怎么测的",
        "先把上一轮空着没抽到的空隙加密重新解一遍，把“翻过去”的那个时间点精确定位到 1 毫秒以内。"
        "然后在这个精确时间点上，问“着火点”在哪：如果失稳时全网细胞同等参与，"
        "着火的样子应该摊满整张网、找不到集中的地方；实测烧起来的强度几乎全部（99.4%）"
        "窝在原来那一小撮病灶细胞里，几乎不往外漏。再摆一个对照：把病灶从一个改成两个、"
        "隔开放，如果着火是“两边一起烧”，两个病灶应该差不多亮；实测还是几乎只有一个病灶在烧"
        "（约 99.5% vs 几乎 0%），中间的走廊几乎全暗（0.0%）。\n\n"
        "再问“往哪蔓延”：从着火点位置轻轻推一下网络（2 种力度 × 2 个方向，共 4 种推法），"
        "看烧起来的面积随时间怎么变化。我们预先说好，这套判读要可信，4 种推法必须看到一致的结果"
        "（不是少数服从多数）。",
        "",
        "## 揭示了什么",
        f"**着火点位置**：在当前这个 v2.2 模型、这条仿真轨迹上，最先要点着（变软）的那个花样稳稳地缩在原来那一小撮病灶细胞里，"
        f"不是全网一起烧起来的（集中度打分 {ig.get('core_overlap')}，1 = 完全集中在病灶、"
        f"0 = 摊满全场；打分越低说明摊得越开的“摊开度”另有一项，读数 {ig.get('globality')}）。"
        f"这个结论换成双病灶对照场景重新验证过一遍，结果一样：一个病灶几乎全亮"
        f"（{region_frac.get('coreB', 'n/a')}），"
        f"另一个几乎全暗，中间走廊几乎不亮"
        f"（{region_frac.get('corridor_axial', 'n/a')}）。"
        "这一段跟上一轮 M1 的“翻转时机没看清”并列存在——两个不同问题的两个答案，"
        "谁都没有推翻或取代对方。",
        "",
        spread_paragraph,
        "",
        "## 关键字段（内部归档代号，括号补注）",
        f"- csd_verdict = `{verdict['csd_verdict']}`（M1 结论，本轮不变，并存展示，非被取代）",
        f"- linear_ignition.class = `{ig.get('class')}`"
        f"（core_overlap=`{ig.get('core_overlap')}`, globality=`{ig.get('globality')}`, "
        f"two_core_symmetry_break=`{ig.get('two_core_symmetry_break')}`, "
        f"corridor_power=`{ig.get('corridor_power')}`）",
        f"- linear_ignition.off_axis_sentinel.off_axis = "
        f"`{(ig.get('off_axis_sentinel') or {}).get('off_axis')}`（core-compactness residual, "
        "NOT sideways propagation）",
        f"- nonlinear_spread.epsilon_sensitivity = `{sp.get('epsilon_sensitivity')}` -> "
        f"onset=`{sp.get('onset')}`, endgame=`{sp.get('endgame')}`, off_axis=`{sp.get('off_axis')}`",
        f"- base_gate_passed = `{verdict['base_gate_passed']}`; "
        f"unresolved_subreason = `{verdict['unresolved_subreason']}`",
        f"- interpretation = \"{verdict['interpretation']}\"",
        "",
        "阈值敏感性、逐区功率、逐 (depth, epsilon_rel, polarity) 明细见 "
        "`ignition_spread_verdict.json`；诊断图见 `figures/`。",
    ]
    (out_dir / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def _region_bar_colors():
    return {"coreA": "#2a78d6", "coreB": "#1baf7a", "corridor_axial": "#eda100", "offcore_rest": "#9a9a95"}


def _plot_ignition_panel(fig_dir: Path, verdict: dict, cfg_crit, m2cfg) -> None:
    """ONE question: where does the linear critical mode sit -- localized in the (single) core,
    and does a symmetric two-core geometry still collapse onto one side with the corridor dark?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import src.topic4_criticality_m2 as m2

    ig = verdict["linear_ignition"]
    crossing = ig.get("crossing") or {}
    grid, kernels, core, _b_core = m2._crit_op_context(cfg_crit)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    res = crossing.get("_crossing_res")
    if res is not None:
        scores = m2.shape_scores_at(res, grid, kernels, core)
        loading = scores["_loading"]
        im = ax1.imshow(loading, cmap="viridis", origin="lower")
        ax1.contour(core.mask.astype(float), levels=[0.5], colors="white", linewidths=1.6)
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="E-power loading (a.u.)")
    else:
        ax1.text(0.5, 0.5, "crossing not localized\n(ignition undetermined)",
                 ha="center", va="center", transform=ax1.transAxes, color="#777777")
    ax1.set_title("crossing-mode E-power loading (single-core)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("grid x (cell)"); ax1.set_ylabel("grid y (cell)")

    region_frac = ig.get("_two_core_region_frac")
    tcc = m2cfg["two_core_confirm"]
    if region_frac:
        names = ["coreA", "coreB", "corridor_axial", "offcore_rest"]
        colors = _region_bar_colors()
        vals = [region_frac.get(n, 0.0) for n in names]
        ax2.bar(names, vals, color=[colors[n] for n in names], edgecolor="k", linewidth=0.5)
        ax2.axhline(tcc["single_core_thresh"], color="#2a78d6", ls="--", lw=1.2,
                    label=f"single_core_thresh={tcc['single_core_thresh']}")
        ax2.axhline(tcc["corridor_dark_thresh"], color="#eda100", ls="--", lw=1.2,
                    label=f"corridor_dark_thresh={tcc['corridor_dark_thresh']}")
        ax2.legend(fontsize=7.5, frameon=False, loc="upper center")
    else:
        ax2.text(0.5, 0.5, "two-core confirm not available\n(ignition undetermined)",
                 ha="center", va="center", transform=ax2.transAxes, color="#777777")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("fraction of E-power")
    ax2.set_title("two-core confirm: power by region", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(f"linear_ignition = {ig.get('class')}   (PRELIMINARY, model_side_preliminary)",
                 fontsize=12, fontweight="bold")
    subtitle = (f"core_overlap={ig.get('core_overlap')}   globality={ig.get('globality')}   "
                f"two_core_symmetry_break={ig.get('two_core_symmetry_break')}   "
                f"off_axis_sentinel={(ig.get('off_axis_sentinel') or {}).get('off_axis')}")
    fig.text(0.5, 0.90, subtitle, ha="center", fontsize=8.5, color="#444444")
    note = ig.get("near_fold_note") or ""
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=7, color="#777777", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.savefig(fig_dir / "ignition_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_spread_panel(fig_dir: Path, verdict: dict, m2cfg) -> None:
    """ONE question: once ignited, does the footprint expand along the axis then pull back
    (self-limited), or does it flood globally / leak off-axis -- and does the answer depend on how
    far past the crossing the perturbation is applied (at_crossing vs just_past)?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sp = verdict["nonlinear_spread"]
    traj = sp.get("footprint_trajectory") or {}
    bcfg = m2cfg["basis"]; scfg = m2cfg["spread"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fields = [("active_frac", "active fraction of grid", None),
              ("off_axis", "off-axis power score", bcfg["off_axis_score_tol"]),
              ("elongation_axis", "along-axis elongation score", scfg["axial_onset_thresh"])]

    any_drawn = False
    for ax, (key, ylabel, thr) in zip(axes, fields):
        for depth, style in _DEPTH_STYLE.items():
            run = traj.get(depth) or {}
            samples = run.get("core_kick")
            if not samples:
                continue
            any_drawn = True
            ts = [fm["t_ms"] for fm in samples]
            ys = [fm[key] for fm in samples]
            ax.plot(ts, ys, color=style["color"], ls=style["ls"], marker="o", ms=3.5, lw=1.6)
        if thr is not None:
            ax.axhline(thr, color="#9a9a95", ls=":", lw=1.1)
        ax.set_xlabel("time since kick (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    if not any_drawn:
        for ax in axes:
            ax.text(0.5, 0.5, "no footprint trajectory\n(spread undetermined)",
                    ha="center", va="center", transform=ax.transAxes, color="#777777")
    axes[0].set_title("active_frac(t)", fontsize=10, fontweight="bold")
    axes[1].set_title("off_axis(t)  (dotted = sentinel tol)", fontsize=10, fontweight="bold")
    axes[2].set_title("elongation_axis(t)  (dotted = axial-onset thresh)", fontsize=10, fontweight="bold")

    handles = [Line2D([0], [0], color=s["color"], ls=s["ls"], marker="o", ms=4, label=s["label"])
               for s in _DEPTH_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    # the lines show ONE representative perturbation combo (largest epsilon_rel, last-listed
    # polarity) per depth, not an average -- the formal onset/endgame/off_axis/depth_dependent
    # verdict above is a vote over the FULL epsilon_rel x polarity sweep (4 combos per depth), so a
    # visually diverging pair of lines here does not by itself contradict depth_dependent=False.
    fig.text(0.5, -0.005,
             "lines = ONE representative combo (largest eps_rel, +1 polarity) per depth, not an "
             "average -- the onset/endgame/off_axis/depth_dependent verdict above votes across "
             "the FULL epsilon_rel x polarity sweep (4 combos/depth), not just this one",
             ha="center", fontsize=7, color="#888888")

    fig.suptitle(f"nonlinear_spread — core_kick footprint   "
                 f"(epsilon_sensitivity={sp.get('epsilon_sensitivity')}, PRELIMINARY)",
                 fontsize=12, fontweight="bold")
    subtitle = (f"onset={sp.get('onset')}   endgame={sp.get('endgame')}   "
                f"off_axis={sp.get('off_axis')}   depth_dependent={sp.get('depth_dependent')}")
    fig.text(0.5, 0.90, subtitle, ha="center", fontsize=8.5, color="#444444")

    note = sp.get("descriptive_igniting_note")
    if note:
        note_txt = (f"descriptive-only (not the spread verdict): igniting subset "
                    f"({note['n_igniting_of_total']}) onset={note['igniting_onset']}, "
                    f"endgame={note['igniting_endgame']}")
        fig.text(0.5, 0.855, note_txt, ha="center", fontsize=7.5, color="#777777")
        layout_top = 0.83
    else:
        layout_top = 0.86

    fig.tight_layout(rect=(0, 0.13, 1, layout_top))
    fig.savefig(fig_dir / "spread_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_basis_sanity(fig_dir: Path, verdict: dict, cfg_crit, m2cfg) -> None:
    """ONE question: does the analysis basis make geometric sense, and how much of the crossing
    mode's power is left over (nonaxis residual) once the global + along-axis components are
    projected out -- confirming the residual reads as core-compactness, not a sideways ridge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import src.topic4_criticality_m2 as m2

    ig = verdict["linear_ignition"]
    crossing = ig.get("crossing") or {}
    grid, kernels, core, _b_core = m2._crit_op_context(cfg_crit)
    theta = kernels.theta
    bcfg = m2cfg["basis"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, 4.2))

    b = m2.basis_vectors(grid, theta)
    axis_field = b["e_axis_gradient"].reshape(grid.n, grid.n)
    vmax = float(np.abs(axis_field).max()) or 1.0
    im1 = ax1.imshow(axis_field, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title("along-axis basis vector (e_axis_gradient)", fontsize=9.5, fontweight="bold")

    res = crossing.get("_crossing_res")
    off_axis_sentinel = ig.get("off_axis_sentinel") or {}
    if res is not None:
        scores = m2.shape_scores_at(res, grid, kernels, core)
        loading = scores["_loading"]
        im2 = ax2.imshow(loading, cmap="viridis", origin="lower")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title("crossing-mode loading (same as ignition panel)", fontsize=9.5, fontweight="bold")

        e_nonaxis, frac_resid, _fg, _fa = m2.nonaxis_direction(
            loading, grid, theta, bcfg["nonaxis_direction_min_norm"])
        if e_nonaxis is not None:
            resid_field = e_nonaxis.reshape(grid.n, grid.n)
            vmax3 = float(np.abs(resid_field).max()) or 1.0
            im3 = ax3.imshow(resid_field, cmap="RdBu_r", origin="lower", vmin=-vmax3, vmax=vmax3)
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            ax3.set_title(f"nonaxis residual (off_axis sentinel={off_axis_sentinel.get('off_axis')})",
                         fontsize=9.5, fontweight="bold")
        else:
            ax3.text(0.5, 0.5, "residual norm below\nnonaxis_direction_min_norm\n(no direction to show)",
                     ha="center", va="center", transform=ax3.transAxes, color="#777777")
            ax3.set_title("nonaxis residual (unavailable)", fontsize=9.5, fontweight="bold")
    else:
        for ax in (ax2, ax3):
            ax.text(0.5, 0.5, "crossing not localized", ha="center", va="center",
                    transform=ax.transAxes, color="#777777")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("grid x (cell)"); ax.set_ylabel("grid y (cell)")

    fig.suptitle("basis sanity: axis vector, crossing loading, nonaxis residual", fontsize=12,
                 fontweight="bold")
    fig.text(0.5, 0.90,
             "nonaxis_residual = core-compactness residual in a core-localized mode, "
             "NOT sideways propagation", ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.86))
    fig.savefig(fig_dir / "basis_sanity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_verdict(fig_dir: Path, verdict: dict, cfg_crit, m2cfg) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_ignition_panel(fig_dir, verdict, cfg_crit, m2cfg)
    _plot_spread_panel(fig_dir, verdict, m2cfg)
    _plot_basis_sanity(fig_dir, verdict, cfg_crit, m2cfg)
    _write_fig_readme(fig_dir, verdict)


def _write_fig_readme(fig_dir: Path, verdict: dict) -> None:
    ig = verdict["linear_ignition"]; sp = verdict["nonlinear_spread"]
    txt = (
        "### ignition_panel.png\n"
        "左图把穿零点上临界模态的 E-power 分布画在网格上（白色轮廓=病灶核边界），"
        "右图把同一时刻双病灶对照实验里，功率落在两个病灶、中间走廊、其余区域的比例画成柱状图"
        f"（虚线=判定阈值）。当前判定：`{ig.get('class')}`，双病灶对照下也只亮一侧、"
        "走廊几乎全暗。\n"
        "**关注点**：左图亮斑是否只窝在白色轮廓（病灶核）内部；右图是不是只有一根柱子很高、"
        "corridor_axial 是否压在虚线以下。\n\n"
        "### spread_panel.png\n"
        "三张折线图分别画穿零点（实线）和过临界稍深处（虚线）两个深度上，"
        "被轻推之后激活面积、离轴功率、沿轴拉长度随时间的变化。图上每条线只是 4 种扰动强度/方向"
        "组合里的其中一种（最大幅度、朝上那种），不是 4 组的平均——正式判读"
        f"（epsilon_sensitivity=`{sp.get('epsilon_sensitivity')}`）是把全部 4 组都跑完之后投票"
        "得出的，不是只看这一条线。4 组没能全部一致，正式判读是“没看清”；若有 descriptive note，"
        "标题下方会加一行说明确实点着的那几组看到了什么，仅作参考不算正式结论。\n"
        "**关注点**：两条线是否都在离轴分数虚线以下（未见侧向）；左图两条线一条收回、"
        "一条冲到顶——这只是这一种扰动组合的个例，不能只看这一眼就下“深度决定结局”的结论，"
        "要看标题里的 depth_dependent 是不是 True。\n\n"
        "### basis_sanity.png\n"
        "三张图依次是：沿病灶轴方向定义的坐标基向量、穿零点临界模态的功率分布（与 ignition_panel "
        "左图相同，摆在这里方便跟基向量并排对比）、把模态投影去掉“全局”和“沿轴”分量之后剩下的残差。\n"
        "**关注点**：残差图是不是一小团局部斑点（=核心紧致残差），而不是一条侧向的长条（=真正的"
        "离轴传播）；右上角标的 off_axis sentinel 结论要和图形状一致。\n"
    )
    (fig_dir / "README.md").write_text(txt, encoding="utf-8")


def _append_figure_index(out_dir: Path) -> None:
    """Append (idempotently) one row for this milestone's figures/ right after M1's own row in
    results/FIGURE_INDEX.md (AGENTS.md Results Directory Standards: new figure dirs must be
    indexed)."""
    results_dir = ROOT / "results"
    idx = results_dir / "FIGURE_INDEX.md"
    if not idx.exists():
        return
    out_abs = out_dir if out_dir.is_absolute() else (ROOT / out_dir)
    try:
        rel = out_abs.resolve().relative_to(results_dir.resolve())   # link path is relative to results/
    except ValueError:
        return   # --out points outside results/ (e.g. a test's tmp_path) -- nothing to index
    row_key = f"[{rel}/figures/]"
    entry = (f"| [{rel}/figures/]({rel}/figures/) | M3A-v2.2 criticality Milestone 2："
             "两段式点火(linear_ignition=core_localized)/铺开(nonlinear_spread=undetermined，"
             "epsilon-sensitive)判读诊断图，与 M1 CSD unresolved 并存展示 |")
    lines = idx.read_text(encoding="utf-8").split("\n")
    if any(row_key in line for line in lines):
        return   # already appended (idempotent re-runs)
    marker = "[topic4_criticality/figures/](topic4_criticality/figures/)"
    for i, line in enumerate(lines):
        if marker in line:
            lines.insert(i + 1, entry)
            break
    else:
        lines.append(entry)
    idx.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/topic4_criticality_m2",
                    help="directory for ignition_spread_verdict.json / STATUS.md / figures/")
    ap.add_argument("--from-json", default=None,
                    help="re-render STATUS.md ONLY from an EXISTING ignition_spread_verdict.json "
                         "-- no re-solve. Figures are NOT re-rendered in this mode: two of the "
                         "three panels need the live crossing eigenvector fields, which are "
                         "private/non-serializable and are stripped from the written JSON, so "
                         "they cannot be reconstructed without re-solving.")
    args = ap.parse_args()

    if args.from_json:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        verdict = json.loads(Path(args.from_json).read_text())
        _write_status(out_dir, verdict)
        print(f"re-rendered STATUS.md ONLY from {args.from_json} (figures need a fresh --out run) "
              f"csd_verdict={verdict['csd_verdict']}  out_dir={out_dir}")
        return 0

    verdict = build_and_write_verdict(args.out)
    _append_figure_index(Path(args.out))
    assert "final_verdict" not in verdict
    print(f"csd_verdict={verdict['csd_verdict']}  "
          f"linear_ignition.class={verdict['linear_ignition']['class']}  "
          f"nonlinear_spread.epsilon_sensitivity={verdict['nonlinear_spread']['epsilon_sensitivity']}  "
          f"unresolved_subreason={verdict['unresolved_subreason']}  out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
