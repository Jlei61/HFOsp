#!/usr/bin/env python
"""Topic 5 V3a mode-transition — result figure (Task 11, integration).

Plain-language questions (EXPLORATORY; this figure must read honestly as a
TIER-2 / NOT-SUPPORTED result, not a positive finding):

  Panel A (top row) - "Does the non-axis co-primary Delta move P3->I1?" One
  point per subject for each co-primary endpoint (H3b net-off-axis-flux
  surplus, H3c mode-shift density), grouped into the narrow (PRIMARY) block
  and the broad (REPLICATION-ONLY) block, with a zero line and a cohort
  median tick. Subjects whose own ``module_support_flag`` is True (passed
  direction + both/three of that endpoint's own null gates) get a
  black-outlined marker so the reader can see how few subjects actually
  carry the story.

  Panel B (bottom row) - "Is there peri-ictal structure across the WHOLE
  trajectory (P0..I3), not just the P3->I1 endpoints?" Cohort-median OBSERVED
  (no permutation null) trajectory of the two raw metrics
  (``net_offaxis_flux``, ``mode_shift_density``) across all 8 phase bins,
  narrow vs broad, with the onset buffer window (O, +-10 s) shaded — O is
  descriptive/buffer only, never part of the primary P3->I1 contrast.

H3b and H3c live in incommensurable units (flux is O(0.01-0.8), density is
O(0.0001-0.05)), so each row gets two sub-axes (one per endpoint) rather than
one shared y-axis — this is a units split, not a second scientific question
(CLAUDE.md #7): Panel A always asks "did the Delta move", Panel B always asks
"is there trajectory structure", regardless of which endpoint's column you
read.

Panel A reads the co-primary CSVs from ``--indir`` (default: the canonical
``results/topic5_ictal_recruitment/v3_mode_transition`` tree, so the
committed script always re-renders on whatever the pipeline last wrote there
-- point ``--indir`` at a frozen snapshot for a race-free dev/eyeball render
while a background rerun is overwriting the canonical CSVs).

Panel B is computed OBSERVED-ONLY (no permutation nulls) directly from the
ictal-field long cache via ``scripts._topic5_v3_io.load_subject_phase_envelopes``
plus the SAME pure metric chains the run scripts use:
  H3b: ``activations_from_z -> atm_offdiag -> net_offaxis_flux``, median over
       seizures carrying that phase.
  H3c: per sliding window, ``lowrank_var -> dominant_right_singular_vector(k*)
       -> map_lowrank_vector_to_contacts -> subspace_mode_shift(..., "density")``,
       median over windows -> per seizure -> median over seizures.
This always reads the (race-free) field cache and the fixed
``SUBJECTS_BY_SUB`` cohort lists, never ``--indir`` -- so Panel B is already
"final" regardless of which permutation rerun is in flight for Panel A.

See docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md Task 11
and docs/superpowers/specs/2026-07-02-topic5-v3a-mode-transition-design.md.
"""
from __future__ import annotations

import argparse
import csv
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
from src.topic5_v2_criticality import activations_from_z  # noqa: E402
from src.topic5_v3_mode_transition import (  # noqa: E402
    atm_offdiag,
    dominant_right_singular_vector,
    load_v3_config,
    lowrank_var,
    map_lowrank_vector_to_contacts,
    net_offaxis_flux,
    sliding_windows,
    subspace_mode_shift,
    subspace_projectors,
)

_DEFAULT_INDIR = _ROOT / "results/topic5_ictal_recruitment/v3_mode_transition"
COHORT_COLOR = {"narrow": "#c0603a", "broad": "#3b6fb0"}
COHORT_ROLE = {"narrow": "primary", "broad": "replication-only"}
PHASES = ["P0", "P1", "P2", "P3", "O", "I1", "I2", "I3"]


# ---------------------------------------------------------------------------
# Panel A: read the already-computed co-primary subject CSVs (--indir)
# ---------------------------------------------------------------------------
def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv_rows(path: Path) -> list:
    return list(csv.DictReader(path.open())) if path.exists() else []


def _panel_a_data(indir: Path, cohort: str) -> dict:
    """``{"h3b": {subject: (delta, module_support_flag)}, "h3c": {...}}``.

    Reads each endpoint's OWN CSV and OWN ``module_support_flag`` column
    (avalanche's for H3b, dynamics' for H3c) -- not the summary's combined
    ``subject_support`` -- per the brief ("subjects whose module_support_flag
    == True"). A row with a non-finite delta (skipped/geometry-insufficient
    subject) is kept in the dict but filtered out by the plotting helper, so
    a subject absent from one endpoint's CSV never silently disappears from
    the other endpoint's sub-axis.
    """
    av = _read_csv_rows(indir / cohort / "v3_avalanche_subject.csv")
    dyn = _read_csv_rows(indir / cohort / "v3_dynamics_subject.csv")
    h3b = {r["subject"]: (_f(r["delta_net_offaxis_flux_surplus"]), r.get("module_support_flag") == "True")
           for r in av}
    h3c = {r["subject"]: (_f(r["delta_mode_shift_density"]), r.get("module_support_flag") == "True")
           for r in dyn}
    return {"h3b": h3b, "h3c": h3c}


def _load_tier_payload(indir: Path) -> dict:
    """The Task-10 tier verdict JSON is written IDENTICALLY under both
    ``narrow/`` and ``broad/`` (same merged payload, both cohort blocks
    inside) -- read whichever exists, preferring narrow (primary).
    """
    for cohort in ("narrow", "broad"):
        p = indir / cohort / "v3_cohort_tier.json"
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(
        f"no v3_cohort_tier.json under {indir}/{{narrow,broad}} -- "
        "run scripts/run_topic5_v3_summary.py first"
    )


# ---------------------------------------------------------------------------
# Panel B: observed-only (no perms) phase trajectory straight from the field
# cache. Always uses the canonical SUBJECTS_BY_SUB list and the (race-free)
# ictal_field_long_cache -- independent of --indir.
# ---------------------------------------------------------------------------
def _windows_of(n_t: int, hop: float, win_sec: float, step_sec: float) -> list:
    """Same synthetic-relt sliding-window helper the dynamics run script uses
    (``run_topic5_v3_dynamics.py::_windows_of``): a phase envelope has no
    ``relt`` of its own (it is already a plain (n_contacts, n_t) slice), so a
    synthetic evenly-spaced time axis at ``hop`` spacing recovers the sample
    index windows ``sliding_windows`` expects.
    """
    relt_syn = np.arange(n_t) * hop
    return sliding_windows(relt_syn, 0, n_t, win_sec, step_sec)


def _h3c_mode_shift_for_window(Xw: np.ndarray, P_N: np.ndarray, P_A: np.ndarray, rank: int, alpha: float, kstar: int) -> float:
    """lowrank_var -> dominant_right_singular_vector(k*) ->
    map_lowrank_vector_to_contacts -> subspace_mode_shift(..., "density") --
    the EXACT H3c chain ``run_topic5_v3_dynamics.py::_observed`` uses per
    window (density normalization only; no raw/2D-consistency descriptive
    siblings needed for a trajectory panel).
    """
    B_r, U_r = lowrank_var(Xw, rank, alpha)
    u_c = map_lowrank_vector_to_contacts(dominant_right_singular_vector(B_r, kstar), U_r)
    return subspace_mode_shift(u_c, P_N, P_A, "density")


def _compute_observed_trajectory(cohort: str, cfg: dict) -> dict:
    """Per-phase, per-subject OBSERVED median (no nulls) for H3b/H3c, then
    the caller cohort-medians across subjects.

    Returns ``{"h3b": {phase: [subject_medians]}, "h3c": {phase: [subject_medians]}}``.
    Aggregation order (per the brief): H3b = median over seizures carrying
    that phase; H3c = median over sliding windows -> per seizure -> median
    over seizures carrying that phase. UNLIKE the H3b/H3c run scripts, this
    is NOT paired by seizure across phases (P0..I3 span 8 phases, not a
    single P3->I1 delta) -- each phase's subject value uses every seizure
    that happens to carry THAT phase, independently of which phases its
    other seizures carry. A geometry_insufficient subject (undefined
    axis/non-axis subspace) or a subject whose context/lagPat load fails
    contributes nothing (skipped with a warning), matching how the run
    scripts treat the same subjects.
    """
    z_thr = float(cfg["avalanche"]["z_threshold"])
    rank = int(cfg["dynamics"]["lowrank"])
    alpha = float(cfg["dynamics"]["var_ridge_alpha"])
    kstar = int(cfg["dynamics"]["finite_horizon_k"])
    hop = float(cfg["phases"]["hop_sec"])
    win_sec = float(cfg["phases"]["window_sec"])
    step_sec = float(cfg["phases"]["step_sec"])

    h3b_by_phase = {p: [] for p in PHASES}
    h3c_by_phase = {p: [] for p in PHASES}

    for ds_sid in SUBJECTS_BY_SUB[cohort]:
        try:
            cc = classify_subject_contacts(ds_sid, cohort, cfg)
        except Exception as exc:  # noqa: BLE001 - external mount; never crash the whole figure
            print(f"[warn] panel B {ds_sid} ({cohort}): load failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not cc["geometry_sufficient"]:
            continue

        P_A, P_N = subspace_projectors(cc["all_clean"], cc["is_axis"], cc["is_nonaxis_strict"])
        env = load_subject_phase_envelopes(ds_sid, cohort, cfg, PHASES, onset_shift=0.0, cls=cc)
        axis_idx, nonaxis_idx = env["axis_idx"], env["nonaxis_idx"]

        try:
            for phase in PHASES:
                h3b_sz, h3c_sz = [], []
                for sz in env["seizures"]:
                    if phase not in sz["phases"]:
                        continue
                    Xp = sz["phases"][phase]

                    flux = net_offaxis_flux(atm_offdiag(activations_from_z(Xp, z_thr)),
                                             axis_idx, nonaxis_idx, "source_mean")
                    if np.isfinite(flux):
                        h3b_sz.append(flux)

                    ms_windows = [
                        _h3c_mode_shift_for_window(Xp[:, ws:we], P_N, P_A, rank, alpha, kstar)
                        for ws, we in _windows_of(Xp.shape[1], hop, win_sec, step_sec)
                    ]
                    ms_windows = [m for m in ms_windows if np.isfinite(m)]
                    if ms_windows:
                        h3c_sz.append(float(np.median(ms_windows)))

                if h3b_sz:
                    h3b_by_phase[phase].append(float(np.median(h3b_sz)))
                if h3c_sz:
                    h3c_by_phase[phase].append(float(np.median(h3c_sz)))
        except Exception as exc:  # noqa: BLE001 - a compute failure must not drop the whole figure
            print(f"[warn] panel B {ds_sid} ({cohort}): compute failed: {type(exc).__name__}: {exc}", flush=True)
            continue

    return {"h3b": h3b_by_phase, "h3c": h3c_by_phase}


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_panel_a(ax, cohort_data: dict, key: str, ylabel: str, subtitle: str) -> None:
    """One sub-axis: per-subject P3->I1 Delta for ONE co-primary endpoint.

    x = subjects grouped into a narrow (primary) block then a broad
    (replication-only) block, separated by a vertical divider; each subject
    is one point; a dashed zero line; a thick cohort-median tick spanning
    each block; subjects with module_support_flag==True get a black-outlined
    marker (larger + black edge) so the reader sees exactly how many/which
    subjects carry the story, not just the cohort median.
    """
    ax.axhline(0.0, color="0.55", lw=1.1, ls="--", zorder=1)
    cursor = 0.0
    gap = 1.3
    xtick_pos: list = []
    xtick_lab: list = []
    for cohort in ("narrow", "broad"):
        items = sorted(cohort_data[cohort][key].items())
        finite = [(s, d, flg) for s, (d, flg) in items if np.isfinite(d)]
        n = len(finite)
        color = COHORT_COLOR[cohort]
        if n:
            xs = cursor + np.arange(n, dtype=float)
            plain = [(x, d) for x, (_, d, flg) in zip(xs, finite) if not flg]
            strong = [(x, d) for x, (_, d, flg) in zip(xs, finite) if flg]
            if plain:
                px, py = zip(*plain)
                ax.scatter(px, py, s=46, color=color, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
            if strong:
                sx, sy = zip(*strong)
                ax.scatter(sx, sy, s=78, color=color, edgecolor="black", linewidth=1.6, zorder=4)
            med = float(np.median([d for _, d, _ in finite]))
            ax.plot([xs[0] - 0.4, xs[-1] + 0.4], [med, med], color=color, lw=2.8, zorder=5)
            block_center = float(xs.mean())
            cursor = xs[-1] + 1.0
        else:
            block_center = cursor
        xtick_pos.append(block_center)
        xtick_lab.append(f"{cohort}\n({COHORT_ROLE[cohort]}, n={n})")
        cursor += gap
        if cohort == "narrow":
            ax.axvline(cursor - gap / 2, color="0.82", lw=1.1, zorder=0)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=8.3)
    ax.set_xlim(-0.8, cursor - gap + 0.8)
    ax.set_ylabel(ylabel, fontsize=8.8)
    ax.set_title(subtitle, fontsize=9.8, loc="left")


def _plot_panel_b(ax, traj_by_cohort: dict, key: str, ylabel: str, subtitle: str) -> None:
    """One sub-axis: cohort-median OBSERVED trajectory of one raw metric
    across all 8 phases, narrow vs broad, with the onset buffer (O) shaded.

    The per-cohort subject count goes in the TITLE, not as in-axes text --
    the data's own y-range (and therefore where the zero line sits) varies
    per metric, so any fixed in-axes text position risks colliding with the
    zero line or a data point for one metric even after being tuned for the
    other (this happened once already: a fixed axes-fraction y collided with
    the zero line in one sub-axis while looking fine in its sibling). The
    title area has no data in it, so it can never collide.
    """
    o_i = PHASES.index("O")
    ax.axvspan(o_i - 0.5, o_i + 0.5, color="0.90", zorder=0)
    ax.axhline(0.0, color="0.55", lw=1.0, ls="--", zorder=1)
    xs = np.arange(len(PHASES))
    n_range = {}
    for cohort in ("narrow", "broad"):
        by_phase = traj_by_cohort[cohort][key]
        ns = [len(by_phase[p]) for p in PHASES]
        meds = [float(np.median(by_phase[p])) if by_phase[p] else np.nan for p in PHASES]
        color = COHORT_COLOR[cohort]
        ax.plot(xs, meds, "-o", color=color, lw=2.0, ms=5.0, zorder=3)
        n_lo, n_hi = min(ns), max(ns)
        n_range[cohort] = f"{n_lo}" if n_lo == n_hi else f"{n_lo}-{n_hi}"

    ax.text(o_i, 1.0, "onset\nbuffer", transform=ax.get_xaxis_transform(), fontsize=6.8,
            color="0.4", ha="center", va="top", style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels(PHASES, fontsize=8.5)
    ax.set_xlim(-0.5, len(PHASES) - 0.5)
    ax.set_xlabel("seizure phase (P0-P3 preictal -> O onset buffer -> I1-I3 ictal)", fontsize=7.6)
    ax.set_ylabel(ylabel, fontsize=8.8)
    ax.set_title(
        f"{subtitle}\n(n subjects/phase: narrow {n_range['narrow']}, broad {n_range['broad']})",
        fontsize=9.8, loc="left",
    )


def _fmt_p(x) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.3f}"


_ENDPOINT_PLAIN = {"h3b": "the off-axis-flux endpoint", "h3c": "the mode-transition endpoint"}


def _tier_caption(tier_payload: dict) -> str:
    """Build the honesty caption FROM the live tier JSON's booleans (not a
    fixed prose template) so the sentence stays correct whichever of the 4
    qualitative regimes (narrow pass/fail x broad replicates/not) the FINAL
    n_perm=1000 rerun lands in -- only the NUMBERS are expected to move a
    little; this function's branch must not silently mis-describe a
    different regime if the pattern itself shifts.

    Plain language first, internal plan bookkeeping (the tier number /
    state_v3_supported flag) only as a trailing parenthetical -- this is a
    reader-facing figure, not an archive doc (style guide Sec 0.2 / CLAUDE.md
    Sec 8: no bare internal codenames in axis/legend/title/caption text).
    """
    nb, bb = tier_payload["narrow"], tier_payload["broad"]
    tier = tier_payload["tier"]
    supported = tier_payload["state_v3_supported"]
    narrow_pass = tier_payload["narrow_cohort_pass"]
    broad_pass = tier_payload["broad_cohort_pass"]
    broad_replicates = tier_payload["broad_replicates"]

    lead = "EXPLORATORY, data-side only (no forecasting)."
    narrow_txt = (
        f" Primary cohort (narrow, n={nb['n_geometry_sufficient']}): neither endpoint clears cohort-level "
        f"significance after multiple-comparison correction (p={_fmt_p(nb['p_holm_h3b'])} for the off-axis-flux "
        f"endpoint, p={_fmt_p(nb['p_holm_h3c'])} for the mode-transition endpoint; both above the 0.05 threshold)"
        if not narrow_pass else
        f" Primary cohort (narrow, n={nb['n_geometry_sufficient']}): at least one endpoint DOES clear "
        f"cohort-level significance (p={_fmt_p(nb['p_holm_h3b'])} / p={_fmt_p(nb['p_holm_h3c'])})"
    )
    narrow_txt += (
        f"; only {nb['n_subject_support']} of {nb['n_geometry_sufficient']} subjects individually pass every "
        "robustness check for either endpoint."
    )
    broad_txt = (
        f" Replication-only cohort (broad, n={bb['n_geometry_sufficient']}, never combined with narrow): "
        f"p={_fmt_p(bb['p_holm_h3b'])} / p={_fmt_p(bb['p_holm_h3c'])}."
    )
    if not narrow_pass:
        lean = ""
        if broad_pass:
            which = _ENDPOINT_PLAIN["h3b" if bb["cohort_h3b_pass"] else "h3c"]
            lean = (f" Broad alone clearing {which} is a one-cohort lean, NOT a replicated finding, "
                    "since the primary cohort did not clear it first.")
        verdict_txt = (
            " Net verdict: no robust evidence that seizure onset moves activity off the interictal HFO pathway "
            "in the primary cohort." + lean
        )
    elif broad_replicates:
        verdict_txt = (
            " Net verdict: the replication cohort moves the SAME endpoint in the SAME direction as the primary "
            "cohort -- the strongest evidence tier this design can reach."
        )
    else:
        verdict_txt = (
            " Net verdict: the primary cohort shows a real effect, but the replication cohort does not move the "
            "same endpoint in the same direction, so this stops short of a replicated finding."
        )
    bookkeeping = f" (internal bookkeeping: evidence tier {tier}/4, formally supported={supported})"
    return lead + narrow_txt + broad_txt + verdict_txt + bookkeeping


def _build_figure(panel_a: dict, panel_b: dict, tier_payload: dict) -> "plt.Figure":
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.0))
    (axA_b, axA_c), (axB_b, axB_c) = axes

    # Sub-axis titles name the QUANTITY only (no H3b/H3c plan-internal
    # codename -- style guide Sec 0.2 / CLAUDE.md Sec 8: internal labels
    # never go in reader-facing title/axis/legend text). The two columns
    # stay visually paired top-to-bottom by repeating the same quantity name.
    _plot_panel_a(axA_b, panel_a, "h3b",
                  "Δ net off-axis flux surplus\n(null-corrected, I1 minus P3)",
                  "off-axis flux surplus")
    _plot_panel_a(axA_c, panel_a, "h3c",
                  "Δ mode-shift density\n(I1 minus P3)",
                  "mode-transition density")
    _plot_panel_b(axB_b, panel_b, "h3b",
                  "net off-axis flux\n(observed, source-normalized)",
                  "off-axis flux -- phase trajectory (observed)")
    _plot_panel_b(axB_c, panel_b, "h3c",
                  "mode-shift density\n(observed, non-axis minus axis)",
                  "mode-transition density -- phase trajectory (observed)")

    fig.tight_layout(rect=(0.01, 0.065, 0.99, 0.905), h_pad=7.0)

    # The "B" row banner must sit in the gap between row A's FULL rendered
    # extent (axes box + its two-line x-tick labels) and row B's FULL extent
    # (its own title) -- get_position() alone only covers the axes frame and
    # would let the banner collide with row A's tick labels (as it did before
    # this fix). get_tightbbox() includes tick labels/titles, so the gap
    # midpoint computed from it is robust to font-size/label-length changes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_a = axA_b.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    bbox_b = axB_b.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    row_gap_y = (bbox_a.y0 + bbox_b.y1) / 2.0

    fig.text(0.5, 0.955,
              "A - does the non-axis co-primary Δ move P3->I1? (per-subject points, cohort-median bar, zero line)",
              ha="center", fontsize=10.5, fontweight="bold")
    fig.text(0.5, row_gap_y,
              "B - is there peri-ictal structure across the WHOLE trajectory (P0..I3), not just P3/I1? "
              "(observed cohort-median, no permutation nulls)",
              ha="center", fontsize=10.5, fontweight="bold")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COHORT_COLOR["narrow"],
                    markersize=8, label="narrow (primary cohort)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COHORT_COLOR["broad"],
                    markersize=8, label="broad (replication-only, never pooled)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.5", markeredgecolor="black",
                    markeredgewidth=1.6, markersize=9, label="subject individually passes its own endpoint's checks"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.5", markeredgecolor="white",
                    markeredgewidth=0.6, markersize=7, label="subject does not"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False,
               fontsize=8.6, bbox_to_anchor=(0.5, 1.012))

    caption = _tier_caption(tier_payload)
    fig.text(0.5, 0.006, caption, ha="center", va="bottom", fontsize=7.9, wrap=True,
             bbox={"boxstyle": "round", "facecolor": "0.96", "edgecolor": "0.8"})

    supported = tier_payload["state_v3_supported"]
    fig.suptitle(
        "Topic 5 V3a -- seizure axis-to-non-axis mode transition, P3->I1 (EXPLORATORY; "
        f"{'SUPPORTED' if supported else 'NOT supported'} in the primary cohort)",
        fontsize=12.5, y=1.075, fontweight="bold",
    )
    return fig


def _write_readme(outdir: Path, tier_payload: dict) -> Path:
    """Chinese ``figures/README.md`` per AGENTS.md format (``### filename`` +
    2-4 sentences + trailing ``**关注点**：`` line), written AFTER the PNG so
    every number quoted here always matches THIS exact render (CLAUDE.md Sec
    8 plain-language-first discipline: what was measured / how / what it
    shows, with the tier/state_v3_supported bookkeeping only in a trailing
    parenthetical -- never the primary language).
    """
    nb, bb = tier_payload["narrow"], tier_payload["broad"]
    tier = tier_payload["tier"]
    supported = tier_payload["state_v3_supported"]
    lean_txt = ""
    if not tier_payload["narrow_cohort_pass"] and tier_payload["broad_cohort_pass"]:
        lean_txt = "；复制队列里有一个指标看到了显著偏移，但因为主力队列没有过，这不能算复制成功"
    body = (
        "### v3_mode_transition_summary.png\n\n"
        "这张图检验：发作真正开始前后（发作前 30~10 秒到发作后 10~30 秒），系统里最容易被放大的"
        "活动方向/连锁流向，是不是从一条病人自己间期就走熟的固定电极通路，转移到了通路之外的电极和"
        f"方向上。上排是每个病人这段时间前后的变化量（narrow 是主力队列 n={nb['n_geometry_sufficient']}，"
        f"broad 只作复制 n={bb['n_geometry_sufficient']}、从不与主力队列合并）；下排把同样两个指标画成"
        "发作前 2 分钟到发作后的完整时间线，发作起始前后 10 秒用灰色底纹标出，只作缓冲、不进主结论。\n\n"
        "实测下来：主力队列里两个指标都没有整体、稳健地偏离 0，只有 "
        f"{nb['n_subject_support']}/{nb['n_geometry_sufficient']} 个病人在个体层面同时过了全部稳健性检验"
        f"{lean_txt}。总体是一次探索性的偏阴性结果：没有看到稳健证据表明发作开始把活动系统性地搬出间期"
        f"通路（内部记账：evidence tier {tier}/4，formally supported={supported}）。\n\n"
        "**关注点**：看两行四张子图的点/线是不是整体贴着 0（没有系统性偏移），以及黑色描边的点（个体"
        "通过全部稳健性检验的病人）数量很少——这就是偏阴性结论的直接视觉来源。\n"
    )
    readme_path = outdir / "README.md"
    readme_path.write_text(body, encoding="utf-8")
    return readme_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=str(_DEFAULT_INDIR),
                    help="Panel-A co-primary CSV tree root (default: canonical results path). "
                         "Panel B always reads the canonical field cache + SUBJECTS_BY_SUB, "
                         "independent of --indir.")
    ap.add_argument("--outdir", default=None, help="default: <indir>/figures")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_v3_config()

    panel_a = {c: _panel_a_data(indir, c) for c in ("narrow", "broad")}
    tier_payload = _load_tier_payload(indir)

    print("[fig] computing Panel B observed-only phase trajectories from the field cache "
          "(race-free w.r.t. any Panel-A rerun; ~1-2 min)...", flush=True)
    panel_b = {c: _compute_observed_trajectory(c, cfg) for c in ("narrow", "broad")}

    fig = _build_figure(panel_a, panel_b, tier_payload)
    out_png = outdir / "v3_mode_transition_summary.png"
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    print(f"[fig] -> {out_png}", flush=True)

    out_readme = _write_readme(outdir, tier_payload)
    print(f"[fig] -> {out_readme}", flush=True)
    return out_png


if __name__ == "__main__":
    main()
