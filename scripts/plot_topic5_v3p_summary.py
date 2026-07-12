#!/usr/bin/env python
"""Topic 5 V3p preictal trajectory -- result figures (real-time redesign 2026-07-05).

Rewritten to match ``scripts/plot_topic5_v3_summary.py`` (V3a's real-time
redesign, on ``main``) EXACTLY -- the earlier 6-panel narrow+broad-on-one-axes
grid with parenthetical axis labels and an internal-bookkeeping caption was
rejected (same problems as V3a's pre-revision figure). Design discipline
(replicated verbatim):

  - Real-time x-axis in SECONDS relative to EEG onset (window centres) --
    NEVER phase codes on the axis.
  - ONE cohort per figure -- narrow and broad are never on the same axes ->
    SEPARATE PNGs per cohort.
  - MAIN figure per cohort (``v3p_axis_vs_offaxis_{narrow,broad}.png``): the
    migration pair on ONE axes, both baseline-normalized (baseline-SD units,
    0 = "same as far-preictal baseline") so the two different-unit metrics
    sit comparably on one y-axis:
      - ALONG-AXIS organization = median ``|beta_axis|`` (H3p-a, descriptive/
        supportive only -- never cohort-tested) -- ORANGE, hypothesis FALLS
        toward onset.
      - OFF-AXIS flux = ``net_offaxis_flux`` (H3p-b, co-primary) -- TEAL,
        hypothesis RISES toward onset.
    If the migration held, orange dips below 0 while teal climbs above it.
  - SUPPLEMENTARY figure per cohort (``v3p_mode_direction_{narrow,broad}.png``):
    mode-shift density (H3p-c, co-primary; ``lowrank_var ->
    dominant_right_singular_vector -> map_lowrank_vector_to_contacts ->
    subspace_mode_shift(..,"density")``) -- PURPLE, one line.
  - Median + IQR band, legend INSIDE upper-right, minimal on-canvas text (one
    small italic corner annotation carrying the endpoint Holm p, read live
    from the Task-9 tier JSON). The WORDS live in ``figures/README.md``.

V3P ADAPTATIONS (vs the V3a template):
  - PREICTAL-ONLY phases: ``PHASES = ["P0","P1","P2","P3"]`` (``PHASE_TIME``
    -105/-75/-45/-20 s). NO O/I1 -- V3p never touches onset. x-lim is
    ``[-118, -5]``; a faint dotted marker at -10 s ("P3 ends ... onset not
    analyzed") replaces the V3a template's onset buffer/marker (there is no
    onset buffer here -- onset itself is never in the analyzed span).
  - BASELINE = far-preictal ``BASELINE_PHASES = ["P0","P1"]`` (-105/-75 s,
    >60 s before onset); 0 = far-from-onset, and the eye watches whether
    P2/P3 (near onset) rise -- unlike the V3a template's 3-phase baseline
    (P0,P1,P2), V3p only has 4 phases total, so baseline/contrast is a clean
    2-2 split.
  - ROSTER: narrow = ``SUBJECTS_BY_SUB["narrow"]`` (7, same list as V3a's
    narrow). broad = **broad_expanded (13)**: read ``<indir>/admission.json``
    's ``broad_expanded`` list (the AUTHORITATIVE V3p Task-1 roster record)
    when present, else fall back to config ``cohort_expansion.broad_core +
    candidates_epilepsiae`` -- mirrors ``run_topic5_v3p_trajectory.py``'s own
    fallback exactly. Do NOT use ``SUBJECTS_BY_SUB["broad"]`` (V3a's own
    9-subject broad_core -- a different, narrower cohort).
  - TIER JSON: ``<indir>/{narrow,broad}/v3p_cohort_tier.json`` (written
    identically under both cohort dirs); per-cohort keys are ``p_holm_b`` /
    ``p_holm_c`` + ``cohort_b_pass`` / ``cohort_c_pass`` -- NOT the V3a
    template's ``p_holm_h3b`` / ``p_holm_h3c`` naming.
  - OUTPUT: PNGs + README land under ``<indir>/{narrow,broad}/figures/`` (per
    cohort -> its own dir, matching the existing V3p results-tree layout),
    not a single flat ``<indir>/figures/`` -- 4 PNGs + 2 READMEs total.

WORKTREE/MAIN DIVERGENCE (why this is not a byte-identical import list): this
branch's checked-out ``scripts/run_topic5_v3_susceptibility.py`` predates the
``main``-only commit that introduced ``_abs_beta_sz`` (main has since
refactored the H3a susceptibility run; this V3p branch never rebased onto
that). This file must NOT edit any V3a/V2 file, so the along-axis (H3p-a)
    leg is built from the current susceptibility helpers -- ``_phase_llr``
    (per-seizure ``{name: line-length-rate}`` dicts for one phase) +
    ``_abs_beta_sz`` (``|beta_axis|`` for one seizure, restricted to the axis
    set), followed by an explicit finite median over seizures -- the
same primitives V3a's own ``_run_ok_subject`` uses for its P3/I1 contrast,
just pointed at P0..P3. The flux (H3p-b) and mode (H3p-c) legs use the exact
same primitives as the V3a template (``activations_from_z -> atm_offdiag ->
net_offaxis_flux``; ``lowrank_var -> dominant_right_singular_vector ->
map_lowrank_vector_to_contacts -> subspace_mode_shift``), all of which are
unaffected by the drift and confirmed present in this worktree (V3p's own
Task-7 runner already imports the identical set).

FRAMING (complete-hard-gate NEGATIVE, tier 0 -- 2026-07-05 real n_perm=1000
run, see the archive doc): both cohorts' trajectories are FLAT. Cohort Holm p
is >= 0.65 on every leg; 0/7 narrow and 0/13 broad subjects individually pass
the full pre-registered hard-gate stack. A few broad subjects show an
isolated single-null nominal hit (never a cohort-level or full-gate pass) --
these are named, not hidden, in the per-cohort README, alongside why the
harder gates filter them out. This is NOT "no evidence of any preictal
non-axial change" -- it is "no direction-consistent, gate-surviving ramp was
found".

See docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md
Task 10, docs/superpowers/specs/2026-07-03-topic5-v3p-preictal-trajectory-design.md,
and docs/archive/topic5/v3p_preictal_nonaxis_trajectory_2026-07-05.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import (  # noqa: E402
    classify_subject_contacts,
    load_subject_phase_envelopes,
)
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from scripts.run_topic5_v3_susceptibility import _abs_beta_sz, _phase_llr  # noqa: E402
from src.topic5_v2_criticality import activations_from_z  # noqa: E402
from src.topic5_v3_mode_transition import (  # noqa: E402
    atm_offdiag,
    dominant_right_singular_vector,
    load_v3_config,
    lowrank_var,
    map_lowrank_vector_to_contacts,
    net_offaxis_flux,
    rank_forward,
    sliding_windows,
    subspace_mode_shift,
    subspace_projectors,
)
from src.topic5_v3p_preictal_trajectory import load_v3p_config  # noqa: E402

_DEFAULT_INDIR = _ROOT / "results/topic5_ictal_recruitment/v3p_preictal_trajectory"

# Preictal-only fixed-second phases (contiguous -120..-10 s); V3p never
# touches O/I1/I2/I3. Value = the window CENTRE in seconds relative to onset.
PHASES = ["P0", "P1", "P2", "P3"]
PHASE_TIME = {"P0": -105.0, "P1": -75.0, "P2": -45.0, "P3": -20.0}
BASELINE_PHASES = ["P0", "P1"]  # far-preictal baseline (>60 s before onset); P2/P3 are the near-onset contrast
P3_END = -10.0                  # V3p's clean preictal boundary -- onset (0 s) is never analyzed

# House semantic colours (figure_style_guide Sec 0.1 Topic-4 lock, reused
# verbatim from the V3a template): on-axis = orange, cross/off-axis = teal.
# Mode-direction = a distinct purple.
AXIS_COLOR = "#d1791f"
OFFAXIS_COLOR = "#2a9d8f"
MODE_COLOR = "#7b5aa6"
COHORT_ROLE = {"narrow": "primary", "broad": "replication"}


# ---------------------------------------------------------------------------
# roster + tier payload
# ---------------------------------------------------------------------------
def _broad_expanded_roster(indir: Path, v3pcfg: dict) -> list:
    """13-subject V3p replication roster (9 curated core + 4 admitted
    candidates) -- NEVER ``SUBJECTS_BY_SUB["broad"]`` (V3a's own, different,
    9-subject broad_core). Prefers the AUTHORITATIVE ``admission.json``
    roster record (Task 1 axis-quality gate); config ``cohort_expansion``
    fallback mirrors ``run_topic5_v3p_trajectory.py``'s own fallback exactly.
    """
    admission_path = indir / "admission.json"
    if admission_path.exists():
        return list(json.loads(admission_path.read_text())["broad_expanded"])
    print(
        f"[warn] admission.json not found at {admission_path}; using config broad_expanded "
        "(run run_topic5_v3p_feasibility.py --include-candidates to harden)",
        flush=True,
    )
    exp = v3pcfg["cohort_expansion"]
    return list(exp["broad_core"]) + list(exp.get("candidates_epilepsiae", []))


def _roster_for(cohort: str, indir: Path, v3pcfg: dict) -> list:
    return list(SUBJECTS_BY_SUB["narrow"]) if cohort == "narrow" else _broad_expanded_roster(indir, v3pcfg)


def _load_tier_payload(indir: Path) -> dict:
    """Task-9 tier verdict JSON (written identically under both cohort dirs)."""
    for cohort in ("narrow", "broad"):
        p = indir / cohort / "v3p_cohort_tier.json"
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(
        f"no v3p_cohort_tier.json under {indir}/{{narrow,broad}} -- "
        "run scripts/run_topic5_v3p_summary.py first"
    )


# ---------------------------------------------------------------------------
# data: observed-only real-time trajectory (mirrors plot_topic5_v3_summary.py
# ``_compute_trajectory``, restricted to P0..P3 + V3p's own roster/config)
# ---------------------------------------------------------------------------
def _rank_forward_for_subject(cc: dict) -> dict:
    """Fixed interictal forward-rank axis for one subject (susceptibility-run
    pattern, copied verbatim): ``typical_rank`` over the TRUE axis template
    only, rescaled to [-1, +1]. Never recomputed under any null here (this is
    an observed-only figure).
    """
    axis_set = set(cc["is_axis"])
    typical_rank: dict = {}
    for rec in (cc["ctx"]["ta"], cc["ctx"]["tb"]):
        for ch in rec["channels"]:
            nm = ch["name"]
            r = ch.get("typical_rank", np.nan)
            if nm in axis_set and np.isfinite(r):
                typical_rank.setdefault(nm, float(r))
    return rank_forward(typical_rank)


def _mode_shift_for_window(Xw, P_N, P_A, rank, alpha, kstar) -> float:
    """The exact H3p-c per-window chain the dynamics run uses (density norm)."""
    B_r, U_r = lowrank_var(Xw, rank, alpha)
    u_c = map_lowrank_vector_to_contacts(dominant_right_singular_vector(B_r, kstar), U_r)
    return subspace_mode_shift(u_c, P_N, P_A, "density")


def _windows_of(n_t: int, hop: float, win_sec: float, step_sec: float) -> list:
    relt_syn = np.arange(n_t) * hop
    return sliding_windows(relt_syn, 0, n_t, win_sec, step_sec)


def _compute_trajectory(cohort: str, cfg: dict, roster: list) -> dict:
    """Per-phase list of per-subject OBSERVED medians for all three metrics.

    Returns ``{"a"|"b"|"c": {phase: [subject_median, ...]}}`` over the 4
    preictal phases (a = along-axis |beta_axis|, H3p-a; b = off-axis flux,
    H3p-b; c = mode-shift density, H3p-c). Per subject: a) median over
    seizures of ``|beta_axis|`` restricted to the axis set (via
    ``_phase_llr`` + ``_abs_beta_sz``, matching V3a's own P3/I1 H3a
    runner pattern); b) median over seizures of the WHOLE-phase-span
    off-axis flux; c) median over sliding sub-windows -> per seizure -> median
    over seizures (a VAR fit needs a bounded window, unlike a/b). A
    geometry_insufficient subject or one whose load/compute fails
    contributes nothing (warned, never crashes the whole cohort). NOT paired
    across phases -- each phase uses whatever seizures carry it (a
    descriptive trajectory, not the Task-7 paired per-seizure Theil-Sen slope
    fit).
    """
    z_thr = float(cfg["avalanche"]["z_threshold"])
    rank = int(cfg["dynamics"]["lowrank"])
    alpha = float(cfg["dynamics"]["var_ridge_alpha"])
    kstar = int(cfg["dynamics"]["finite_horizon_k"])
    hop = float(cfg["phases"]["hop_sec"])
    win_sec = float(cfg["phases"]["window_sec"])
    step_sec = float(cfg["phases"]["step_sec"])

    out = {m: {p: [] for p in PHASES} for m in ("a", "b", "c")}

    for ds_sid in roster:
        try:
            cc = classify_subject_contacts(ds_sid, cohort, cfg)
        except Exception as exc:  # noqa: BLE001 - external mount; never crash the figure
            print(f"[warn] traj {ds_sid} ({cohort}): load failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not cc["geometry_sufficient"]:
            continue

        all_clean = cc["all_clean"]
        is_axis = cc["is_axis"]
        rf = _rank_forward_for_subject(cc)
        P_A, P_N = subspace_projectors(all_clean, is_axis, cc["is_nonaxis_strict"])
        env = load_subject_phase_envelopes(ds_sid, cohort, cfg, PHASES, onset_shift=0.0, cls=cc)
        axis_idx, nonaxis_idx = env["axis_idx"], env["nonaxis_idx"]

        try:
            for phase in PHASES:
                llr_by_id = _phase_llr(env, all_clean, phase)
                a_sz = [
                    _abs_beta_sz(llr_by_name, is_axis, rf)
                    for llr_by_name in llr_by_id.values()
                ]
                a_sz = [v for v in a_sz if np.isfinite(v)]
                if a_sz:
                    out["a"][phase].append(float(np.median(a_sz)))

                b_sz, c_sz = [], []
                for sz in env["seizures"]:
                    if phase not in sz["phases"]:
                        continue
                    Xp = sz["phases"][phase]

                    flux = net_offaxis_flux(atm_offdiag(activations_from_z(Xp, z_thr)),
                                            axis_idx, nonaxis_idx, "source_mean")
                    if np.isfinite(flux):
                        b_sz.append(flux)

                    ms = [_mode_shift_for_window(Xp[:, ws:we], P_N, P_A, rank, alpha, kstar)
                          for ws, we in _windows_of(Xp.shape[1], hop, win_sec, step_sec)]
                    ms = [m for m in ms if np.isfinite(m)]
                    if ms:
                        c_sz.append(float(np.median(ms)))

                if b_sz:
                    out["b"][phase].append(float(np.median(b_sz)))
                if c_sz:
                    out["c"][phase].append(float(np.median(c_sz)))
        except Exception as exc:  # noqa: BLE001 - one bad subject must not drop the whole figure
            print(f"[warn] traj {ds_sid} ({cohort}): compute failed: {type(exc).__name__}: {exc}", flush=True)
            continue

    return out


def _baseline_z(by_phase: dict) -> dict | None:
    """Normalize a per-phase trajectory to its far-preictal baseline.

    Baseline = pooled finite per-subject values over ``BASELINE_PHASES``
    (P0+P1, -105/-75 s -- >60 s before onset). Every phase's per-subject
    values are z-scored by the baseline (mean, SD) so 0 = "same as
    far-preictal" and the unit is baseline-SD -- two different-unit metrics
    become directly comparable on one axis. Returns ``{phase: {"med","q25",
    "q75","n"}}`` (per-phase cohort median + IQR of the z-scores), or
    ``None`` if the baseline is degenerate (<2 finite points or zero
    spread), in which case the caller skips that line rather than dividing
    by zero.
    """
    base = [v for p in BASELINE_PHASES for v in by_phase.get(p, []) if np.isfinite(v)]
    if len(base) < 2:
        return None
    mu0, sd0 = float(np.mean(base)), float(np.std(base))
    if not np.isfinite(sd0) or sd0 <= 0:
        return None
    out: dict = {}
    for p in PHASES:
        vals = [(v - mu0) / sd0 for v in by_phase.get(p, []) if np.isfinite(v)]
        if vals:
            out[p] = {"med": float(np.median(vals)), "q25": float(np.percentile(vals, 25)),
                      "q75": float(np.percentile(vals, 75)), "n": len(vals)}
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_metric_line(ax, znorm: dict, color: str, label: str) -> None:
    """One metric's cohort-median z-trajectory + IQR band over real time."""
    xs, meds, los, his = [], [], [], []
    for p in PHASES:
        d = znorm.get(p)
        if d is None:
            continue
        xs.append(PHASE_TIME[p])
        meds.append(d["med"])
        los.append(d["q25"])
        his.append(d["q75"])
    ax.fill_between(xs, los, his, color=color, alpha=0.14, lw=0, zorder=2)
    ax.plot(xs, meds, "-o", color=color, lw=2.6, ms=7, mec="white", mew=0.8, label=label, zorder=4)


def _decorate_time_axis(ax, cohort: str, title: str, ylabel: str) -> None:
    """Shared real-time decoration: baseline line, P3-end/onset-excluded
    marker, numeric-second x-ticks (no phase codes), title, labels. V3p is
    strictly preictal (unlike the V3a template, there is no onset buffer and
    no O/I1 marker) -- the only boundary decoration is the RIGHT edge of the
    analyzed span (P3 ends at -10 s), never seizure onset itself.
    """
    ax.axhline(0.0, color="0.55", lw=1.1, ls="--", zorder=1)
    ax.axvline(P3_END, color="0.75", lw=1.1, ls=":", zorder=1)
    ax.text(P3_END, 1.008, "P3 ends (−10 s); onset (0 s) not analyzed →",
            transform=ax.get_xaxis_transform(),
            fontsize=8.2, color="0.45", ha="right", va="bottom", style="italic")

    ticks = [PHASE_TIME[p] for p in PHASES]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks], fontsize=10)
    ax.set_xlim(-118, -5)
    ax.set_xlabel("time relative to EEG onset (s)", fontsize=11.5)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.set_title(f"{cohort.capitalize()} cohort ({COHORT_ROLE[cohort]}) — {title}",
                 fontsize=12.6, fontweight="bold", loc="left", pad=14)
    ax.margins(y=0.20)


def _fmt_p(p: float) -> str:
    return "n/a" if not np.isfinite(p) else (f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}")


def _build_axis_offaxis_figure(cohort: str, traj: dict, tier_cohort: dict) -> "plt.Figure":
    """MAIN per-cohort figure: along-axis organization vs off-axis flux over
    real time, with the endpoint off-axis-flux surplus Holm-p annotated.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.9))

    za = _baseline_z(traj["a"])
    zb = _baseline_z(traj["b"])
    if za is not None:
        _plot_metric_line(ax, za, AXIS_COLOR, "along-axis organization  (|β| on interictal axis)")
    if zb is not None:
        _plot_metric_line(ax, zb, OFFAXIS_COLOR, "off-axis flux  (avalanche onto non-axis)")

    _decorate_time_axis(
        ax, cohort,
        "does non-axial flux ramp OFF the interictal HFO axis before onset?",
        "change vs. far-preictal baseline  (baseline-SD units)",
    )

    p = tier_cohort["p_holm_b"]
    tag = "significant, but fragile" if tier_cohort["cohort_b_pass"] else "n.s."
    ax.text(0.015, 0.03,
            f"preictal non-axial-flux slope (null-corrected):  Holm p = {_fmt_p(p)}  ({tag})",
            transform=ax.transAxes, fontsize=8.8, color="0.32", style="italic", ha="left", va="bottom")

    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.85",
              fontsize=9.6, handletextpad=0.5, borderpad=0.7)
    fig.tight_layout()
    return fig


def _build_mode_figure(cohort: str, traj: dict, tier_cohort: dict) -> "plt.Figure":
    """SUPPLEMENTARY per-cohort figure: mode-transition DIRECTION over real time."""
    fig, ax = plt.subplots(figsize=(8.6, 5.9))

    zc = _baseline_z(traj["c"])
    if zc is not None:
        _plot_metric_line(ax, zc, MODE_COLOR, "mode-shift density  (non-axis − axis)")

    _decorate_time_axis(
        ax, cohort,
        "does the most-amplifiable mode drift off-axis before onset?",
        "change vs. far-preictal baseline  (baseline-SD units)",
    )

    p = tier_cohort["p_holm_c"]
    tag = "significant" if tier_cohort["cohort_c_pass"] else "n.s."
    ax.text(0.015, 0.03,
            f"preictal mode-direction slope (null-corrected):  Holm p = {_fmt_p(p)}  ({tag})",
            transform=ax.transAxes, fontsize=8.8, color="0.32", style="italic", ha="left", va="bottom")

    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.85",
              fontsize=9.6, handletextpad=0.5, borderpad=0.7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# README (per cohort dir; the WORDS live here, not on the figures)
# ---------------------------------------------------------------------------
def _write_readme(outdir: Path, cohort: str, tier_cohort: dict) -> Path:
    """Chinese figures/README.md (AGENTS.md format), one call per cohort dir.

    The headline Holm-p numbers + roster size are read LIVE off the
    (already-loaded) cohort tier block. The specific isolated-nominal-hit
    subject lists below are FIXED prose reflecting the archived real
    n_perm=1000 run (docs/archive/topic5/v3p_preictal_nonaxis_trajectory_2026-07-05.md)
    -- this script only recomputes the observed real-time trajectory + reads
    the cohort Holm p, it does not itself recompute per-subject label-null p.
    """
    role_zh = "主力队列" if cohort == "narrow" else "复制队列"
    p_b, p_c = _fmt_p(tier_cohort["p_holm_b"]), _fmt_p(tier_cohort["p_holm_c"])
    n = tier_cohort["n_eligible"]

    if cohort == "narrow":
        nominal_txt = (
            "这两条腿在这 7 个人里都是 0/7 通过全部预先设定的把关流程——连最基础的一步"
            "（按轴/非轴标签打乱重算）也没有谁单独冒出过 p<0.05 的巧合命中。"
        )
    else:
        nominal_txt = (
            "这两条腿在这 13 个人里都是 0/13 通过全部预先设定的把关流程；不过非轴向流单独这一步的"
            "标签置换检验有 3 人压线 p<0.05（epilepsiae_253/620/139），模态转移方向这一步有 2 人压线"
            "（epilepsiae_1084/916）——但换成保持放电率的对照重排、把窗口挪远一点、看是不是隔了一步才"
            "轮到非轴向而不是同时一起爆、换成打乱相位或分块的方式重测，就没有一项能一起挺住，只算零散、"
            "方向不一致、未经把关的提示，够不成方向一致的队列证据。"
        )

    body = (
        f"### v3p_axis_vs_offaxis_{cohort}.png（主图）\n\n"
        "**这张图问一句话**：发作真正开始前的最后两分钟里（横轴=相对脑电起始的秒数，只画到 −10 秒——"
        "V3p 严格只看发作前，图上不画发作起始本身、更不碰起始之后），系统里连锁扩散的活动是不是从病人"
        "间期就走熟的固定高频通路（间期 HFO 轴）上「挪开」。**橙线=沿轴组织度**（活动还有多强跟着那条"
        "固定顺序走，辅助描述，不参与显著性检验）；**青线=离轴流**（连锁活动往通路之外触点铺的量，"
        "承重的检验对象之一）。两条线都各自除以最靠前两个窗口（−105/−75 秒，离发作最远）的基线（纵轴"
        "单位=基线标准差，0=跟离发作最远时一样），量纲不同也能放同一根纵轴直接比。**若真有「发作前"
        "爬升」，青线应在接近 −10 秒时明显抬高、橙线明显走低。**\n\n"
        f"**实测（{role_zh}，n={n}）**：两条线在四个窗口间都有起伏（比如青线在中段一度抬高），但都没有"
        "表现出朝图右侧（越接近 −10 秒边界）稳定、单调的反向张开——橙线不是稳定走低，青线也不是稳定走高，"
        "到分析截止时两条线并未像假设预期的那样明显分开。真正承重的统计量是对每个人这两分钟内的斜率、"
        f"扣除随机置换基线后做的检验，同样不显著：Holm p={p_b}。{nominal_txt}\n\n"
        "**关注点**：看橙线和青线在图右侧（接近 −10 秒、虚点线标出的分析边界）有没有比中段更明显地反向"
        "张开（橙下、青上）。实测没有——这不是「发作前完全没有非轴向变化的迹象」，只是我们没有测到方向"
        "一致、经得起把关的爬升信号。\n\n"
        f"### v3p_mode_direction_{cohort}.png（附图）\n\n"
        "**这张图问第二个、不同的问题**：不是流的大小，而是最容易被放大的那个活动模式的**方向**有没有"
        "转到离轴触点上。紫线=模态离轴密度（离轴−沿轴），同样按发作前最远基线归一、横轴同为真实秒数。\n\n"
        f"**实测**：紫线在四个窗口间起伏波动，没有表现出朝正方向（离轴）随时间单调爬升的趋势；−10 秒"
        f"端点 Holm p={p_c}（不显著）。\n\n"
        "**关注点**：紫线没有随时间稳定往正方向走，与主图橙/青线一样——模态方向这一路同样没有测到爬升"
        "信号。\n"
    )
    readme_path = outdir / "README.md"
    readme_path.write_text(body, encoding="utf-8")
    return readme_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=str(_DEFAULT_INDIR),
                    help="tier-JSON tree root (default: canonical results path).")
    ap.add_argument("--outdir", default=None,
                    help="figures base dir; each cohort writes its 2 PNGs + README to "
                         "<outdir-or-indir>/<cohort>/figures/ (default base: --indir).")
    args = ap.parse_args(argv)

    indir = Path(args.indir)
    base = Path(args.outdir) if args.outdir else indir

    cfg = load_v3_config()
    v3pcfg = load_v3p_config()
    tier = _load_tier_payload(indir)

    print("[fig] computing observed real-time preictal trajectories (no permutation nulls) "
          "from the field cache; a few minutes...", flush=True)
    out_paths = []
    for cohort in ("narrow", "broad"):
        roster = _roster_for(cohort, indir, v3pcfg)
        traj = _compute_trajectory(cohort, cfg, roster)
        tier_cohort = tier[cohort]

        outdir = base / cohort / "figures"
        outdir.mkdir(parents=True, exist_ok=True)

        fig_main = _build_axis_offaxis_figure(cohort, traj, tier_cohort)
        p_main = outdir / f"v3p_axis_vs_offaxis_{cohort}.png"
        fig_main.savefig(p_main, dpi=170, bbox_inches="tight")
        plt.close(fig_main)
        print(f"[fig] -> {p_main}", flush=True)
        out_paths.append(p_main)

        fig_mode = _build_mode_figure(cohort, traj, tier_cohort)
        p_mode = outdir / f"v3p_mode_direction_{cohort}.png"
        fig_mode.savefig(p_mode, dpi=170, bbox_inches="tight")
        plt.close(fig_mode)
        print(f"[fig] -> {p_mode}", flush=True)
        out_paths.append(p_mode)

        readme_path = _write_readme(outdir, cohort, tier_cohort)
        print(f"[fig] -> {readme_path}", flush=True)

    return out_paths


if __name__ == "__main__":
    main()
