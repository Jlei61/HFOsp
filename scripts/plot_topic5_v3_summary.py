#!/usr/bin/env python
"""Topic 5 V3a mode-transition -- result figures (real-time trajectory redesign 2026-07-04).

ONE scientific question per figure, read off a REAL-TIME x-axis (seconds
relative to EEG onset -- never phase codes P/O/I on the axis), WITHIN one
cohort at a time (narrow and broad are never drawn on the same axes), legend
INSIDE the axes at upper-right, minimal text (the words live in README.md, not
on the canvas):

  MAIN -- ``v3_axis_vs_offaxis_{narrow,broad}.png``: the V3a mechanism pair on
  ONE axes so a non-expert sees at a glance whether, across onset, activity
  moves OFF the fixed interictal HFO axis. Two lines, both normalized to their
  own PRE-ICTAL baseline (mean over -120..-30 s, in baseline-SD units) so the
  two different-unit metrics are directly comparable and 0 = "same as before":
    - ALONG-AXIS organization  = ``|beta_axis|`` (H3a; hypothesis: FALLS into
      the seizure) -- orange, the house on-axis colour (figure_style_guide Sec
      0.1 Topic-4 lock: on-axis = orange).
    - OFF-AXIS flux            = ``net_offaxis_flux`` (H3b; hypothesis: RISES)
      -- teal, the house cross/off-axis colour.
  If V3a held, orange dips below 0 while teal climbs above it around onset.
  The observed (raw) trajectory is honest about our fragile reality: the two
  lines do NOT cleanly diverge (the H3b endpoint is significant only
  null-relative, i.e. above a rate-preserving null, which a raw trajectory
  cannot draw -- so the null-corrected surplus endpoint p is annotated, not
  plotted). One small top-left annotation carries the paired -20->+20 s surplus
  Holm p; no wall-of-text caption.

  SUPPLEMENTARY -- ``v3_mode_direction_{narrow,broad}.png``: the SECOND
  co-primary (H3c, the DIRECTION of the most-amplifiable mode, not flux
  magnitude) on the same real-time axis, same normalization, one line (purple).
  It is flat -> the mode-direction endpoint is null. Same axes/legend/minimal-
  text discipline as the main figure.

Only P0..I1 map to a FIXED second offset from onset (-105,-75,-45,-20,0,+20 s,
contiguous) and can honestly sit on a real-time axis; I2/I3 are seizure-
FRACTION windows (variable absolute time) and are excluded from the trajectory.
The onset buffer O (-10..+10 s) is shaded, never part of the primary contrast.

All trajectory metrics are OBSERVED-only (no permutation nulls) and read
straight from the ictal-field long cache via
``scripts._topic5_v3_io.load_subject_phase_envelopes`` + the SAME pure metric
chains the run scripts use (H3a: ``_line_length_rate -> _abs_beta_sz`` from the
susceptibility runner; H3b: ``activations_from_z -> atm_offdiag ->
net_offaxis_flux``; H3c: ``lowrank_var -> dominant_right_singular_vector(k*) ->
map_lowrank_vector_to_contacts -> subspace_mode_shift(..,"density")``), so the
lines match the endpoints they summarize. The endpoint Holm-p annotations are
read from the Task-10 tier JSON under ``--indir``.

See docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md Task 11 and
docs/archive/topic5/v3a_mode_transition_2026-07-04.md. This real-time redesign
supersedes the 1x2 per-subject-endpoint layout (2026-07-04: user brief -- use a
real time axis not P/O/I codes; compare off-axis vs axis WITHIN each cohort as
two separate figures; legend inside upper-right; let the picture, not text,
carry the message).
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
from scripts.run_topic5_v3_susceptibility import _abs_beta_sz, _line_length_rate  # noqa: E402
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

_DEFAULT_INDIR = _ROOT / "results/topic5_ictal_recruitment/v3_mode_transition"

# Only the fixed-second phases (contiguous -120..+30 s); I2/I3 are seizure-
# fraction windows with no fixed onset offset and are excluded from a real-
# time axis. Value = the window CENTRE in seconds relative to EEG onset.
PHASE_TIME = {"P0": -105.0, "P1": -75.0, "P2": -45.0, "P3": -20.0, "O": 0.0, "I1": 20.0}
PHASES = ["P0", "P1", "P2", "P3", "O", "I1"]
BASELINE_PHASES = ["P0", "P1", "P2"]  # pre-ictal baseline: everything before the P3 (-20 s) contrast window
O_BUFFER = (-10.0, 10.0)              # onset buffer (shaded); never part of the primary -20->+20 s contrast

# House semantic colours (figure_style_guide Sec 0.1 Topic-4 lock reused):
# on-axis = orange, cross/off-axis = teal. Mode-direction = a distinct purple.
AXIS_COLOR = "#d1791f"
OFFAXIS_COLOR = "#2a9d8f"
MODE_COLOR = "#7b5aa6"
COHORT_ROLE = {"narrow": "primary", "broad": "replication"}


# ---------------------------------------------------------------------------
# data: observed-only real-time trajectory + endpoint Holm-p annotations
# ---------------------------------------------------------------------------
def _load_tier_payload(indir: Path) -> dict:
    """Task-10 tier verdict JSON (written identically under both cohort dirs)."""
    for cohort in ("narrow", "broad"):
        p = indir / cohort / "v3_cohort_tier.json"
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(
        f"no v3_cohort_tier.json under {indir}/{{narrow,broad}} -- "
        "run scripts/run_topic5_v3_summary.py first"
    )


def _rank_forward_for_subject(cc: dict) -> dict:
    """Fixed interictal forward-rank axis for one subject (susceptibility-run
    pattern, copied verbatim): ``typical_rank`` over the TRUE axis template
    only, rescaled to [-1, +1]. Never recomputed under any null here (this is
    an observed-only figure), so a plain restriction to the axis set is all we
    need to match the run's ``beta_axis`` inputs.
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


def _h3c_mode_shift_for_window(Xw, P_N, P_A, rank, alpha, kstar) -> float:
    """The exact H3c per-window chain the dynamics run uses (density norm)."""
    B_r, U_r = lowrank_var(Xw, rank, alpha)
    u_c = map_lowrank_vector_to_contacts(dominant_right_singular_vector(B_r, kstar), U_r)
    return subspace_mode_shift(u_c, P_N, P_A, "density")


def _windows_of(n_t: int, hop: float, win_sec: float, step_sec: float) -> list:
    relt_syn = np.arange(n_t) * hop
    return sliding_windows(relt_syn, 0, n_t, win_sec, step_sec)


def _compute_trajectory(cohort: str, cfg: dict) -> dict:
    """Per-phase list of per-subject OBSERVED medians for all three metrics.

    Returns ``{"h3a"|"h3b"|"h3c": {phase: [subject_median, ...]}}`` over the 6
    fixed-second phases. Per subject: H3a/H3b median over seizures carrying the
    phase; H3c median over sliding windows -> per seizure -> median over
    seizures. A geometry_insufficient subject (no axis/non-axis subspace) or a
    subject whose load/compute fails contributes nothing (warned, matching the
    run scripts). NOT paired across phases -- each phase uses whatever seizures
    carry it (a descriptive trajectory, not the paired P3->I1 endpoint).
    """
    z_thr = float(cfg["avalanche"]["z_threshold"])
    rank = int(cfg["dynamics"]["lowrank"])
    alpha = float(cfg["dynamics"]["var_ridge_alpha"])
    kstar = int(cfg["dynamics"]["finite_horizon_k"])
    hop = float(cfg["phases"]["hop_sec"])
    win_sec = float(cfg["phases"]["window_sec"])
    step_sec = float(cfg["phases"]["step_sec"])

    out = {m: {p: [] for p in PHASES} for m in ("h3a", "h3b", "h3c")}

    for ds_sid in SUBJECTS_BY_SUB[cohort]:
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
                a_sz, b_sz, c_sz = [], [], []
                for sz in env["seizures"]:
                    if phase not in sz["phases"]:
                        continue
                    Xp = sz["phases"][phase]

                    ba = _abs_beta_sz(dict(zip(all_clean, _line_length_rate(Xp))), is_axis, rf)
                    if np.isfinite(ba):
                        a_sz.append(ba)

                    flux = net_offaxis_flux(atm_offdiag(activations_from_z(Xp, z_thr)),
                                            axis_idx, nonaxis_idx, "source_mean")
                    if np.isfinite(flux):
                        b_sz.append(flux)

                    ms = [_h3c_mode_shift_for_window(Xp[:, ws:we], P_N, P_A, rank, alpha, kstar)
                          for ws, we in _windows_of(Xp.shape[1], hop, win_sec, step_sec)]
                    ms = [m for m in ms if np.isfinite(m)]
                    if ms:
                        c_sz.append(float(np.median(ms)))

                if a_sz:
                    out["h3a"][phase].append(float(np.median(a_sz)))
                if b_sz:
                    out["h3b"][phase].append(float(np.median(b_sz)))
                if c_sz:
                    out["h3c"][phase].append(float(np.median(c_sz)))
        except Exception as exc:  # noqa: BLE001 - one bad subject must not drop the whole figure
            print(f"[warn] traj {ds_sid} ({cohort}): compute failed: {type(exc).__name__}: {exc}", flush=True)
            continue

    return out


def _baseline_z(by_phase: dict) -> dict | None:
    """Normalize a per-phase trajectory to its pre-ictal baseline.

    Baseline = pooled finite per-subject values over ``BASELINE_PHASES`` (the
    -120..-30 s pre-ictal region, before the P3 contrast window). Every phase's
    per-subject values are z-scored by the baseline (mean, SD) so 0 = "same as
    pre-ictal" and the unit is baseline-SD -- two different-unit metrics become
    directly comparable on one axis. Returns ``{phase: {"med","q25","q75","n"}}``
    (per-phase cohort median + IQR of the z-scores), or ``None`` if the baseline
    is degenerate (<2 finite points or zero spread), in which case the caller
    skips that line rather than dividing by zero.
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
    """Shared real-time decoration: baseline line, onset marker, buffer shade,
    numeric-second x-ticks (no P/O/I codes), title, labels.
    """
    ax.axhline(0.0, color="0.55", lw=1.1, ls="--", zorder=1)
    ax.axvspan(O_BUFFER[0], O_BUFFER[1], color="0.90", zorder=0)
    ax.axvline(0.0, color="0.30", lw=1.5, zorder=1)
    ax.text(0.0, 1.008, "seizure onset", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color="0.30", ha="center", va="bottom", style="italic")

    ticks = [PHASE_TIME[p] for p in PHASES]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:+.0f}".replace("+0", "0") for t in ticks], fontsize=10)
    ax.set_xlim(-118, 30)
    ax.set_xlabel("time relative to EEG onset (s)", fontsize=11.5)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.set_title(f"{cohort.capitalize()} cohort ({COHORT_ROLE[cohort]}) — {title}",
                 fontsize=12.6, fontweight="bold", loc="left", pad=14)
    ax.margins(y=0.20)


def _fmt_p(p: float) -> str:
    return "n/a" if not np.isfinite(p) else (f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}")


def _build_axis_offaxis_figure(cohort: str, traj: dict, tier: dict) -> "plt.Figure":
    """MAIN per-cohort figure: along-axis organization vs off-axis flux over
    real time, with the paired -20->+20 s off-axis surplus Holm-p annotated.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.9))

    za = _baseline_z(traj["h3a"])
    zo = _baseline_z(traj["h3b"])
    if za is not None:
        _plot_metric_line(ax, za, AXIS_COLOR, "along-axis organization  (|β| on interictal axis)")
    if zo is not None:
        _plot_metric_line(ax, zo, OFFAXIS_COLOR, "off-axis flux  (avalanche onto non-axis)")

    _decorate_time_axis(
        ax, cohort,
        "does seizure onset pull activity OFF the interictal HFO axis?",
        "change vs. pre-ictal baseline  (baseline-SD units)",
    )

    p = tier[cohort]["p_holm_h3b"]
    tag = "significant, but fragile" if tier[cohort]["cohort_h3b_pass"] else "n.s."
    ax.text(0.015, 0.03,
            f"−20→+20 s off-axis surplus (null-corrected):  Holm p = {_fmt_p(p)}  ({tag})",
            transform=ax.transAxes, fontsize=8.8, color="0.32", style="italic", ha="left", va="bottom")

    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.85",
              fontsize=9.6, handletextpad=0.5, borderpad=0.7)
    fig.tight_layout()
    return fig


def _build_mode_figure(cohort: str, traj: dict, tier: dict) -> "plt.Figure":
    """SUPPLEMENTARY per-cohort figure: mode-transition DIRECTION over real time."""
    fig, ax = plt.subplots(figsize=(8.6, 5.9))

    zc = _baseline_z(traj["h3c"])
    if zc is not None:
        _plot_metric_line(ax, zc, MODE_COLOR, "mode-shift density  (non-axis − axis)")

    _decorate_time_axis(
        ax, cohort,
        "does the most-amplifiable mode point off-axis?",
        "change vs. pre-ictal baseline  (baseline-SD units)",
    )

    p = tier[cohort]["p_holm_h3c"]
    tag = "significant" if tier[cohort]["cohort_h3c_pass"] else "null"
    ax.text(0.015, 0.03,
            f"−20→+20 s mode-direction shift (null-corrected):  Holm p = {_fmt_p(p)}  ({tag})",
            transform=ax.transAxes, fontsize=8.8, color="0.32", style="italic", ha="left", va="bottom")

    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.85",
              fontsize=9.6, handletextpad=0.5, borderpad=0.7)
    fig.tight_layout()
    return fig


def _write_readme(outdir: Path, tier: dict) -> Path:
    """Chinese figures/README.md (AGENTS.md format): the WORDS live here, not on
    the figures. One section per figure family, honest-fragile framing.
    """
    nb, bb = tier["narrow"], tier["broad"]
    body = (
        "### v3_axis_vs_offaxis_narrow.png / v3_axis_vs_offaxis_broad.png（主图，每队列一张）\n\n"
        "**这张图问一句话**：发作真正开始前后（横轴=相对脑电起始的秒数，0=起始），系统的活动是不是从"
        "病人平时就走熟的固定高频通路（间期 HFO 轴）上「挪开」？两条线各答一半：**橙线=沿轴组织度**"
        "（活动还有多强跟着那条固定顺序走，H3a，假设发作时**下降**）；**青线=离轴流**（连锁活动往通路"
        "之外触点铺的量，H3b，假设发作时**上升**）。两条线都各自除以自己发作前 2 分钟的基线（纵轴单位="
        "基线标准差，0=跟发作前一样），所以量纲不同也能放同一根纵轴直接比。**若 V3a 成立，橙线应在起始"
        "附近掉到 0 以下、青线爬到 0 以上、两线张开。**\n\n"
        "**实测（诚实）**：两条线并没有干净地张开——离轴流的原始轨迹没有稳定抬升。承重的统计结论是"
        f"「扣掉放电率随机基线后的离轴流增量」在 −20→+20 秒配对检验里达到队列显著（narrow Holm "
        f"p={_fmt_p(nb['p_holm_h3b'])}、broad p={_fmt_p(bb['p_holm_h3b'])}，见图左上角标注），但这是"
        "**相对随机基线**的偏高、不是绝对上升，原始轨迹画不出来，所以只标注 p、不画进曲线。这是一个"
        "**脆弱的候选信号**，不是「轴→非轴转移成立」。\n\n"
        "**关注点**：看橙线和青线在 0 秒（灰色起始缓冲带）前后有没有**反向张开**（橙下、青上）。实测没有"
        "明显张开——这正是「原始信号里看不出干净转移、显著性只存在于扣基线之后」的直接视觉证据。\n\n"
        "### v3_mode_direction_narrow.png / v3_mode_direction_broad.png（附图，每队列一张）\n\n"
        "**这张图问第二个、不同的问题**：不只是「流的大小」，而是「最容易被放大的那个模态的**方向**」有没有"
        "转到离轴触点上（H3c）。紫线=模态离轴密度（离轴−沿轴），同样按发作前基线归一、横轴同为真实秒数。\n\n"
        f"**实测**：紧贴 0 线、几乎不动——两队列都不显著（narrow Holm p={_fmt_p(nb['p_holm_h3c'])}、"
        f"broad p={_fmt_p(bb['p_holm_h3c'])}）。即「方向真的搬家了」这条**没看到**。\n\n"
        "**关注点**：紫线整条基本压在 0 线上，与主图橙/青线（至少在扣基线后有一点信号）形成对比——"
        "模态方向这一路是干净的阴性。\n"
    )
    readme_path = outdir / "README.md"
    readme_path.write_text(body, encoding="utf-8")
    return readme_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=str(_DEFAULT_INDIR),
                    help="tier-JSON tree root (default: canonical results path).")
    ap.add_argument("--outdir", default=None, help="default: <indir>/figures")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_v3_config()
    tier = _load_tier_payload(indir)

    print("[fig] computing observed real-time trajectories (no permutation nulls) from the "
          "field cache; ~2-4 min...", flush=True)
    outs = []
    for cohort in ("narrow", "broad"):
        traj = _compute_trajectory(cohort, cfg)

        fig_main = _build_axis_offaxis_figure(cohort, traj, tier)
        p_main = outdir / f"v3_axis_vs_offaxis_{cohort}.png"
        fig_main.savefig(p_main, dpi=170, bbox_inches="tight")
        plt.close(fig_main)
        print(f"[fig] -> {p_main}", flush=True)
        outs.append(p_main)

        fig_mode = _build_mode_figure(cohort, traj, tier)
        p_mode = outdir / f"v3_mode_direction_{cohort}.png"
        fig_mode.savefig(p_mode, dpi=170, bbox_inches="tight")
        plt.close(fig_mode)
        print(f"[fig] -> {p_mode}", flush=True)
        outs.append(p_mode)

    out_readme = _write_readme(outdir, tier)
    print(f"[fig] -> {out_readme}", flush=True)
    return outs[0]


if __name__ == "__main__":
    main()
