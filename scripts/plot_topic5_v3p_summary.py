#!/usr/bin/env python
"""Topic 5 V3p preictal trajectory -- result figure (Task 10, integration).

Three independent questions (CLAUDE.md Sec 7: one construct per ROW; the two
COLUMNS inside a row are a units split, not a second question -- non-axial
flux and mode-shift density live in incommensurable units, exactly the
precedent V3a's own summary figure documents for splitting H3b/H3c the same
way):

  (A) -- per-SUBJECT summary. Does each subject's own preictal rate-of-rise
  (a Theil-Sen slope fit across P0..P3) look elevated in raw units, and which
  subjects individually clear their OWN label-permutation null? One point per
  subject for each co-primary endpoint (H3p-b non-axial-flux surplus slope,
  H3p-c mode-shift-density surplus slope), grouped narrow (primary) vs broad
  (replication, = broad_expanded). Circles = narrow, or broad's curated
  core-9; triangles = broad's admitted-candidate-only subjects
  (``in_broad_core==False``) -- so a reader sees at a glance how much of any
  broad signal leans on the 4 non-curated admissions. A black ring marks a
  subject whose own label-permutation p (``p_label_slope_{b,c}``) clears
  0.05. A thick horizontal tick is each block's median.

  (B) -- pooled WINDOW-level trajectory. Averaged across every window, every
  seizure, every subject in a cohort, does the RAW metric's shape actually
  climb from P0 to P3, or could (A)'s slope summary be hiding a flat /
  non-monotone reality? Mean +/- IQR band per phase bin (P0..P3 ONLY --
  preictal-only; O/I1/I2/I3 never enter this figure), computed directly from
  ``v3p_window_detail.csv`` by binning each window's ``t_center`` into its
  fixed P0..P3 bin. This is genuinely independent of (A): (A) is a
  per-subject FITTED-RATE summary; (B) is the pooled RAW-trajectory SHAPE --
  a flat (A) could still hide a real bend in (B), and vice versa.

  (C, optional) -- the SAME per-subject slopes as (A), rescaled onto one
  shared, dimensionless null-relative z-scale ((obs - null median) / null
  MAD) with +/-1.96 reference guides. (A) necessarily splits H3p-b/H3p-c onto
  two different-unit y-axes, so it cannot show whether the two endpoints are
  SIMILARLY far from their own null for the same subject; (C) can, because z
  is unit-free (an effect-size-vs-significance pairing, not a rotation of
  (A)). Skipped entirely (figure degrades to 2 rows) if no subject in either
  cohort has a finite slope-z in either column.

Reads exactly 3 files per cohort under ``--indir`` (default: canonical
results path) -- no mounted-data / no heavy src imports, so it renders
identically whether ``--indir`` points at the real pipeline output or a
synthetic dev fixture:
  ``v3p_trajectory_subject.csv`` (Task 7/8 subject-level co-primary slopes),
  ``v3p_window_detail.csv``      (Task 7 per-window raw metrics),
  ``v3p_cohort_tier.json``       (Task 9 cohort Holm-p / tier verdict --
                                   caption only, purely descriptive, never
                                   re-derived here).

See docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md
Task 10. Visual style mirrors scripts/plot_topic5_v3_summary.py (Topic 5
V3a's most recent summary-figure redesign) per this task's brief --
docs/figure_style_guide.md Sec "Topic 5" notes the canonical style is NOT
locked yet (exploratory, per-case), so this matches V3a's look without
over-polishing; a unified restyle happens later once Topic 5's figure
language settles.
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INDIR = _ROOT / "results/topic5_ictal_recruitment/v3p_preictal_trajectory"
_PNG_NAME = "v3p_preictal_trajectory_summary.png"
_ALPHA = 0.05

# House V3a palette reused verbatim (see module docstring: Topic 5 has no
# locked canonical style yet, so this mirrors the most recent V3a summary
# figure rather than inventing a new one).
COHORT_COLOR = {"narrow": "#c0603a", "broad": "#3b6fb0"}
COHORT_ROLE = {"narrow": "primary", "broad": "replication"}

PHASES = ["P0", "P1", "P2", "P3"]
# Fixed eeg-onset-anchored preictal bins (config/topic5_v3p.yaml
# preictal.span_full_rel == [-120, -10]); NEVER O/I1/I2/I3 -- V3p is
# preictal-only (brief Hard QC: "no O/I1/I2/I3").
PHASE_BOUNDS = {"P0": (-120.0, -90.0), "P1": (-90.0, -60.0), "P2": (-60.0, -30.0), "P3": (-30.0, -10.0)}


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _b(x) -> bool:
    return x is True or str(x) == "True"


def _read_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _load_subject_rows(indir: Path) -> dict:
    return {c: _read_csv_rows(indir / c / "v3p_trajectory_subject.csv") for c in ("narrow", "broad")}


def _load_window_rows(indir: Path) -> dict:
    return {c: _read_csv_rows(indir / c / "v3p_window_detail.csv") for c in ("narrow", "broad")}


def _load_tier_payload(indir: Path) -> dict | None:
    """Task-9 tier JSON, written identically under narrow/ and broad/ --
    prefer narrow (primary). Returns ``None`` (never raises) if neither
    exists yet, so the figure can still render pre-Task-9 -- the caption
    just degrades to a generic exploratory line instead of a p-value."""
    for cohort in ("narrow", "broad"):
        p = indir / cohort / "v3p_cohort_tier.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


# ---------------------------------------------------------------------------
# Panels A + C share one per-subject grouped-scatter renderer -- only the
# value / significance COLUMN and the axis text differ between calls.
# ---------------------------------------------------------------------------
def _subject_points(rows_by_cohort: dict, value_col: str, sig_col: str) -> dict:
    """``{cohort: [(subject, value, in_broad_core, is_label_sig), ...]}``,
    sorted by subject; a non-finite ``value_col`` row is dropped (a skipped /
    geometry_insufficient subject contributes no point, never a fake 0)."""
    out = {}
    for cohort, rows in rows_by_cohort.items():
        items = []
        for r in rows:
            v = _f(r.get(value_col))
            if not np.isfinite(v):
                continue
            p_sig = _f(r.get(sig_col))
            items.append((r.get("subject", "?"), v, _b(r.get("in_broad_core", False)),
                          bool(np.isfinite(p_sig) and p_sig < _ALPHA)))
        items.sort(key=lambda t: t[0])
        out[cohort] = items
    return out


def _plot_subject_panel(ax, rows_by_cohort: dict, value_col: str, sig_col: str, ylabel: str, title: str) -> None:
    """One panel: per-subject points grouped narrow-then-broad, a zero
    reference line, and a thick cohort-median tick per block (brief: "a zero
    reference line ... cohort-median bars"). Marker SHAPE encodes
    narrow-vs-broad-core (circle) vs broad-admitted-candidate-only
    (triangle); marker EDGE encodes the subject's own label-null
    significance -- the two encodings are orthogonal so they combine without
    conflict (mirrors V3a's color=cohort / edge=robustness convention)."""
    groups = _subject_points(rows_by_cohort, value_col, sig_col)
    ax.axhline(0.0, color="0.55", lw=1.1, ls="--", zorder=1)
    cursor = 0.0
    gap = 1.2
    xtick_pos: list = []
    xtick_lab: list = []
    for cohort in ("narrow", "broad"):
        items = groups[cohort]
        n = len(items)
        color = COHORT_COLOR[cohort]
        if n:
            xs = cursor + np.arange(n, dtype=float)
            for x, (_subj, v, is_core, is_sig) in zip(xs, items):
                marker = "o" if (cohort == "narrow" or is_core) else "^"
                ax.scatter([x], [v], s=64 if is_sig else 50, marker=marker, color=color,
                           alpha=0.85, edgecolor=("black" if is_sig else "white"),
                           linewidth=(1.7 if is_sig else 0.6), zorder=4 if is_sig else 3)
            med = float(np.median([v for _, v, _, _ in items]))
            ax.plot([xs[0] - 0.4, xs[-1] + 0.4], [med, med], color=color, lw=2.8, zorder=5)
            block_center = float(xs.mean())
            cursor = xs[-1] + 1.0
        else:
            block_center = cursor
        xtick_pos.append(block_center)
        xtick_lab.append(f"{cohort} ({COHORT_ROLE[cohort]})\nn={n}")
        cursor += gap
        if cohort == "narrow":
            ax.axvline(cursor - gap / 2, color="0.82", lw=1.1, zorder=0)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=9.4)
    ax.set_xlim(-0.8, max(cursor - gap + 0.8, 1.0))
    ax.margins(y=0.22)
    ax.set_ylabel(ylabel, fontsize=10.6)
    ax.set_title(title, fontsize=10.8, loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Panel B: pooled window-level trajectory (mean +/- IQR per P0..P3 bin).
# ---------------------------------------------------------------------------
def _bin_phase(t_center: float) -> str | None:
    for p, (lo, hi) in PHASE_BOUNDS.items():
        if lo <= t_center <= hi:
            return p
    return None


def _compute_trajectory(window_rows_by_cohort: dict, value_col: str) -> dict:
    """``{cohort: {phase: {"mean","q25","q75","n"}}}``, pooling every
    (subject, seizure, window) row whose ``t_center`` falls in that phase's
    fixed bin FLATLY across seizures/subjects (brief: "mean +/- IQR band
    across seizures/subjects") -- not a per-subject-then-cohort hierarchy. A
    phase bin with zero finite values is simply absent from the output (the
    plotting helper skips it rather than drawing a fake point)."""
    out = {}
    for cohort, rows in window_rows_by_cohort.items():
        by_phase = {p: [] for p in PHASES}
        for r in rows:
            t = _f(r.get("t_center"))
            phase = _bin_phase(t) if np.isfinite(t) else None
            if phase is None:
                continue
            v = _f(r.get(value_col))
            if np.isfinite(v):
                by_phase[phase].append(v)
        out[cohort] = {
            p: {"mean": float(np.mean(vs)), "q25": float(np.percentile(vs, 25)),
                "q75": float(np.percentile(vs, 75)), "n": len(vs)}
            for p, vs in by_phase.items() if vs
        }
    return out


def _plot_trajectory_panel(ax, traj_by_cohort: dict, ylabel: str, title: str) -> None:
    for cohort in ("narrow", "broad"):
        d = traj_by_cohort.get(cohort, {})
        xs = [i for i, p in enumerate(PHASES) if p in d]
        if not xs:
            continue
        means = [d[PHASES[i]]["mean"] for i in xs]
        los = [d[PHASES[i]]["q25"] for i in xs]
        his = [d[PHASES[i]]["q75"] for i in xs]
        color = COHORT_COLOR[cohort]
        ax.fill_between(xs, los, his, color=color, alpha=0.16, lw=0, zorder=2)
        ax.plot(xs, means, "-o", color=color, lw=2.2, ms=6.5, mec="white", mew=0.7, zorder=4)
    ax.set_xticks(range(len(PHASES)))
    ax.set_xticklabels(PHASES, fontsize=10.2)
    ax.set_xlim(-0.4, len(PHASES) - 0.6)
    ax.set_xlabel("preictal window relative to EEG onset: P0 [-120,-90s] .. P3 [-30,-10s]", fontsize=8.8)
    ax.set_ylabel(ylabel, fontsize=10.6)
    ax.set_title(title, fontsize=10.8, loc="left", fontweight="bold")


def _panel_c_available(subject_rows: dict) -> bool:
    for rows in subject_rows.values():
        for r in rows:
            if (np.isfinite(_f(r.get("net_offaxis_flux_slope_z")))
                    or np.isfinite(_f(r.get("mode_shift_density_slope_z")))):
                return True
    return False


# ---------------------------------------------------------------------------
# caption (read live from the tier JSON -- never hardcoded numbers) + figure
# ---------------------------------------------------------------------------
def _fmt_p(x) -> str:
    x = _f(x)
    return "n/a" if not np.isfinite(x) else (f"{x:.3f}" if x >= 1e-3 else f"{x:.1e}")


def _caption(tier_payload: dict | None) -> str:
    lead = "EXPLORATORY, preictal-only (P0-P3, no O/I1-I3), no forecasting."
    if not tier_payload:
        text = lead + " Cohort Holm-p / tier verdict not available yet for this render (v3p_cohort_tier.json missing)."
        return "\n".join(textwrap.wrap(text, width=170))
    nb, bb = tier_payload["narrow"], tier_payload["broad"]
    text = (
        lead + f" Primary cohort (narrow, n={nb['n_eligible']}): cohort Holm p = {_fmt_p(nb['p_holm_b'])} "
        f"(non-axial flux) / {_fmt_p(nb['p_holm_c'])} (mode-shift); {nb['n_subject_support']}/{nb['n_eligible']} "
        f"subjects individually robust. Replication (broad, n={bb['n_eligible']}): p = {_fmt_p(bb['p_holm_b'])} / "
        f"{_fmt_p(bb['p_holm_c'])}. (internal bookkeeping: evidence tier {tier_payload['tier']}/4, "
        f"supported={tier_payload['state_v3p_supported']})"
    )
    return "\n".join(textwrap.wrap(text, width=170))


def _build_figure(subject_rows: dict, window_rows: dict, tier_payload: dict | None, panel_c_on: bool):
    n_rows = 3 if panel_c_on else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(13.0, 5.4 * n_rows))

    _plot_subject_panel(
        axes[0, 0], subject_rows, "net_offaxis_flux_surplus_slope", "p_label_slope_b",
        "non-axial flux surplus slope\n(null-corrected, per s)",
        "(A1) non-axial flow — per-subject rate-of-rise",
    )
    _plot_subject_panel(
        axes[0, 1], subject_rows, "mode_shift_density_surplus_slope", "p_label_slope_c",
        "mode-shift density surplus slope\n(null-corrected, per s)",
        "(A2) mode-shift direction — per-subject rate-of-rise",
    )

    traj_flux = _compute_trajectory(window_rows, "net_offaxis_flux_lag1")
    traj_mode = _compute_trajectory(window_rows, "mode_shift_density")
    _plot_trajectory_panel(
        axes[1, 0], traj_flux, "non-axial flux\n(observed, mean ± IQR)",
        "(B1) non-axial flow — pooled P0→P3 trajectory",
    )
    _plot_trajectory_panel(
        axes[1, 1], traj_mode, "mode-shift density\n(observed, mean ± IQR)",
        "(B2) mode-shift direction — pooled P0→P3 trajectory",
    )

    if panel_c_on:
        _plot_subject_panel(
            axes[2, 0], subject_rows, "net_offaxis_flux_slope_z", "p_label_slope_b",
            "non-axial flux slope, null-relative z",
            "(C1) non-axial flow — null-relative z",
        )
        _plot_subject_panel(
            axes[2, 1], subject_rows, "mode_shift_density_slope_z", "p_label_slope_c",
            "mode-shift slope, null-relative z",
            "(C2) mode-shift direction — null-relative z",
        )
        for ax in axes[2]:
            ax.axhline(1.96, color="0.35", lw=1.1, ls=":", zorder=1)
            ax.axhline(-1.96, color="0.35", lw=1.1, ls=":", zorder=1)
        axes[2, 1].text(0.985, 0.05, "guides: ±1.96 (approx. 2-sided reference)",
                         transform=axes[2, 1].transAxes, fontsize=7.4, ha="right", color="0.35", style="italic")

    # Reserved top/bottom space (suptitle+caption / legend) is a roughly FIXED
    # number of inches, not a fixed fraction -- the figure's total height
    # scales with n_rows (2 vs 3, panel C on/off), so a hardcoded fraction
    # would leave a growing blank band as rows are added (caught by eyeball:
    # a 3-row render left a visibly oversized gap under a fraction tuned for
    # 2 rows).
    fig_h = 5.4 * n_rows
    top_rect = 1.0 - 1.05 / fig_h
    bottom_rect = 0.85 / fig_h
    fig.tight_layout(rect=(0.02, bottom_rect, 0.98, top_rect))

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COHORT_COLOR["narrow"],
                   markeredgecolor=COHORT_COLOR["narrow"], markersize=8, label="narrow (primary)"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COHORT_COLOR["broad"],
                   markeredgecolor=COHORT_COLOR["broad"], markersize=8, label="broad, curated core"),
        plt.Line2D([0], [0], marker="^", linestyle="none", markerfacecolor=COHORT_COLOR["broad"],
                   markeredgecolor=COHORT_COLOR["broad"], markersize=8, label="broad, admitted candidate"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.5", markeredgecolor="black",
                   markeredgewidth=1.7, markersize=8, label="subject's own label-null p < 0.05"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8.7, bbox_to_anchor=(0.5, 0.35 / fig_h), columnspacing=1.4, handletextpad=0.4)

    fig.text(0.5, 1.0 - 0.62 / fig_h, _caption(tier_payload), ha="center", va="top", fontsize=7.9)

    fig.suptitle(
        "Topic 5 V3p — preictal non-axial trajectory, P0→P3 before seizure onset (EXPLORATORY)",
        fontsize=13.0, fontweight="bold", y=1.0 - 0.28 / fig_h,
    )
    return fig


def _fmt_z(x) -> str:
    x = _f(x)
    return "n/a" if not np.isfinite(x) else f"{x:+.3f}"


def _label_sig_subjects(rows: list, sig_col: str) -> list:
    """Subjects whose OWN label-permutation p on this one co-primary leg
    clears p<0.05 -- same column/threshold ``_subject_points`` uses for the
    black marker ring. Re-derived from the live rows on every call (never a
    hardcoded id list), so a README callout naming "isolated nominal hits"
    can never go stale relative to the actual render."""
    return sorted(
        r.get("subject", "?") for r in rows
        if np.isfinite(_f(r.get(sig_col))) and _f(r.get(sig_col)) < _ALPHA
    )


# Fixed description of the pre-registered hard-gate stack -- this is METHOD,
# not a per-render result (unlike the numbers below), so it is not
# re-derived live: H3p-b additionally requires the rate-preserving null +
# lag1-vs-lag0 (real delayed flow, not "just a synchronous burst"); H3p-c
# additionally requires the phase + block surrogates (real mode shift, not a
# spectral/smoothing artifact); both also require the dual onset-span guard
# (a hit that only survives on the span closest to onset is downgraded).
# See docs/superpowers/specs/2026-07-03-topic5-v3p-preictal-trajectory-design.md
# L1b/L4b/L4d.
_GATE_EXPLANATION = (
    "但这些孤立命中一进入后续几道预先设定好的、专门排除假象的关卡——是否保持了原本的放电/参与节律而不是单纯"
    "数量变多、是否真的隔了一步才轮到非轴向而不是同时一起爆、换成打乱相位或分块的方式重测是否依然站得住、"
    "把窗口往前挪一段、离发作起点更远后是否依然成立——就没有一项能一起挺住，所以只能算零散的、未经把关的"
    "提示，够不成稳健信号。"
)


def _cohort_block_sentence(label: str, blk: dict) -> str:
    """One sentence reporting a single cohort block's numbers -- shared
    phrasing so narrow / broad_expanded / broad_core differ only in the
    label text and the numbers actually read from that block."""
    return (
        f"{label}可用 {blk.get('n_eligible', 'n/a')} 人，其中 {blk.get('n_subject_support', 'n/a')} "
        "人个体同时通过全部稳健性检验；按被试标签置换、Holm 校正后的队列级 p 值：非轴向流 "
        f"{_fmt_p(blk.get('p_holm_b'))}（斜率中位数 z={_fmt_z(blk.get('median_slope_z_b'))}），"
        f"模态转移方向 {_fmt_p(blk.get('p_holm_c'))}（斜率中位数 z={_fmt_z(blk.get('median_slope_z_c'))}）。"
    )


def _verdict_sentence(rows: list) -> str:
    """Honest conclusion for ONE cohort's rows (CLAUDE.md Sec 8: state what
    was measured / how / what it shows, not just a verdict word). A
    complete-hard-gate negative (no cohort direction, no subject support) is
    NOT the same claim as "no preictal non-axial change" -- if any subject's
    own single-leg label-permutation p happens to clear 0.05, name it, then
    say why it still doesn't count as robust support."""
    b_hits = _label_sig_subjects(rows, "p_label_slope_b")
    c_hits = _label_sig_subjects(rows, "p_label_slope_c")
    lead = (
        "这是一次完整的多重把关阴性，不是『发作前完全没有非轴向变化的迹象』：队列层面没有测到方向一致、"
        "经得起 Holm 校正的爬升信号，也没有一个人同时通过全部预先定好的稳健性关卡。"
    )
    if not b_hits and not c_hits:
        return lead + "这批人里，连最基础的一步——按轴/非轴标签打乱重算斜率——也没有谁单独冒出过 p<0.05 的巧合命中。"
    bits = []
    if b_hits:
        bits.append(f"非轴向流这条腿单独冒出 p<0.05 的是 {'/'.join(b_hits)}")
    if c_hits:
        bits.append(f"模态转移方向这条腿单独冒出 p<0.05 的是 {'/'.join(c_hits)}")
    if b_hits and c_hits and not (set(b_hits) & set(c_hits)):
        bits[-1] += "，两条腿从没有在同一个人身上同时冒出过"
    return lead + f"不过在最基础的一步——按轴/非轴标签打乱重算斜率——上，{'；'.join(bits)}。" + _GATE_EXPLANATION


def _write_readme(outdir: Path, tier_payload: dict | None, panel_c_on: bool,
                   cohort: str | None, subject_rows: dict) -> Path:
    """Chinese ``figures/README.md`` (AGENTS.md format: ``### filename`` + a
    few sentences + a trailing ``**关注点**：`` line), written AFTER the PNG
    so every number quoted here matches THIS exact render. ``cohort`` pins
    which block(s) of the (shared, narrow+broad+broad_core) tier JSON THIS
    directory's numbers paragraph reports -- narrow/ only ever reports the
    narrow block, broad/ only ever reports broad_expanded + broad_core
    (never narrow's numbers -- that mismatch is the bug this fixes)."""
    if not tier_payload:
        stat_txt = "本次渲染没有找到队列汇总 JSON，图上不显示 Holm p 值/tier 标注（仅展示原始点位与轨迹）。"
    elif cohort == "narrow":
        stat_txt = (
            "当前这次渲染读到的队列：" + _cohort_block_sentence("narrow（主力）", tier_payload["narrow"]) +
            f"（内部记账：evidence tier {tier_payload['tier']}/4，"
            f"formally supported={tier_payload['state_v3p_supported']}）。" +
            _verdict_sentence(subject_rows.get("narrow", []))
        )
    elif cohort == "broad":
        bc = tier_payload.get("broad_core")
        core_txt = _cohort_block_sentence("broad_core（复制队列核心，去掉 4 个候选补录）", bc) if bc else ""
        tier_bc_txt = (f"，broad_core 口径 tier {tier_payload['tier_broad_core']}/4"
                       if "tier_broad_core" in tier_payload else "")
        stat_txt = (
            "当前这次渲染读到的队列：" +
            _cohort_block_sentence("broad_expanded（复制队列，含全部候选）", tier_payload["broad"]) +
            core_txt +
            f"（内部记账：evidence tier {tier_payload['tier']}/4{tier_bc_txt}，"
            f"formally supported={tier_payload['state_v3p_supported']}）。" +
            _verdict_sentence(subject_rows.get("broad", []))
        )
    else:
        # Explicit --outdir (dev/eyeball render or a test): no single cohort
        # dir to attribute the render to -- report whichever block(s) are
        # present at their plain numbers, no per-cohort verdict prose (that
        # honest-negative framing is the narrow/broad production-dir
        # contract, see the two ``elif`` branches above).
        bits = [_cohort_block_sentence(f"{c}（{'主力' if c == 'narrow' else '复制'}）", tier_payload[c])
                for c in ("narrow", "broad") if c in tier_payload]
        stat_txt = (
            "当前这次渲染读到的队列：" + "".join(bits) +
            f"（内部记账：evidence tier {tier_payload['tier']}/4，"
            f"formally supported={tier_payload['state_v3p_supported']}）。"
        )
    c_txt = (
        "**下排**把上排同样的斜率换算成相对于随机置换基线的标准化 z 值（±1.96 是常规两侧参考线），"
        "让『非轴向流』和『模态转移』两个量纲不同的指标能在同一把尺子上比较谁的信号更强、更一致。\n\n"
        if panel_c_on else ""
    )
    body = (
        f"### {_PNG_NAME}\n\n"
        "这张图检验：发作真正开始前的最后两分钟里（从 P0 到 P3，越往后越接近发作），系统里连锁扩散的活动是不是"
        "逐渐、稳定地往病人间期就走熟的固定高频通路**之外**挪，以及最容易被放大的那个活动方向是不是也逐渐"
        "往通路外偏。**上排**是每个病人这两分钟里『非轴向流』『模态转移方向』各自的爬升斜率——每个点一个病人，"
        "圆圈=narrow（主力）或 broad 的复制核心 9 人，三角=broad 里额外纳入的候选人，黑色描边=该病人自己按"
        "标签置换检验显著（p<0.05）。**中排**把同样两个量画成从 P0 到 P3 的完整轨迹（线=均值，色带=四分位"
        f"区间），直接看整段是不是真的在爬升，而不只是看斜率这一个汇总数字。\n\n{c_txt}"
        f"{stat_txt}\n\n"
        "**关注点**：看上排的点是不是整体偏离 0 线、中排的线是不是从 P0 到 P3 单调爬升、黑色描边的点"
        "（个体显著）有多少——三者一致才支持『发作前活动确实在往通路外搬』，任何一环看起来平的都要谨慎解读。\n"
    )
    readme_path = outdir / "README.md"
    readme_path.write_text(body, encoding="utf-8")
    return readme_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=str(_DEFAULT_INDIR),
                     help="parent dir with narrow/ and broad/ subdirs holding v3p_trajectory_subject.csv, "
                          "v3p_window_detail.csv, v3p_cohort_tier.json (default: canonical results path).")
    ap.add_argument("--outdir", default=None,
                     help="default: write into BOTH <indir>/narrow/figures/ and <indir>/broad/figures/ "
                          "(matching how the Task-9 tier JSON is already duplicated there); pass an "
                          "explicit --outdir to write to ONE location only (dev/eyeball render or a test).")
    args = ap.parse_args(argv)

    indir = Path(args.indir)
    subject_rows = _load_subject_rows(indir)
    window_rows = _load_window_rows(indir)
    tier_payload = _load_tier_payload(indir)

    if not any(subject_rows.values()):
        raise FileNotFoundError(
            f"no v3p_trajectory_subject.csv under {indir}/{{narrow,broad}} -- "
            "run scripts/run_topic5_v3p_trajectory.py first"
        )

    panel_c_on = _panel_c_available(subject_rows)
    fig = _build_figure(subject_rows, window_rows, tier_payload, panel_c_on)

    outdirs = ([(None, Path(args.outdir))] if args.outdir
               else [(c, indir / c / "figures") for c in ("narrow", "broad")])
    out_paths = []
    for cohort, outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        out_png = outdir / _PNG_NAME
        fig.savefig(out_png, dpi=170, bbox_inches="tight")
        print(f"[fig] -> {out_png}", flush=True)
        out_paths.append(out_png)
        readme_path = _write_readme(outdir, tier_payload, panel_c_on, cohort, subject_rows)
        print(f"[fig] -> {readme_path}", flush=True)
    plt.close(fig)
    return out_paths


if __name__ == "__main__":
    main()
