#!/usr/bin/env python
"""Topic 5 V3a mode-transition -- result figures (redesigned 2026-07-04).

Two PNGs, both written by ``main()``:

  ``v3_mode_transition_summary.png`` (MAIN) -- the ONE story this figure must
  make obvious, honest-fragile: the off-axis-flux endpoint (H3b) is
  cohort-significant but FRAGILE (raw flux mostly decreases -> null-relative;
  mostly common-drive; weak individual robustness); the mode-direction
  endpoint (H3c) is NULL. A clean 1x2, one endpoint per panel (CLAUDE.md
  Sec 7): left = per-subject P3->I1 Delta in off-axis flux (FILLED =
  null-corrected surplus, the endpoint; OPEN = raw, same color, hollow);
  right = per-subject P3->I1 Delta in mode-shift density (FILLED only, no
  raw). ONE shared legend (<=4 entries), ONE short caption -- replacing the
  old 2x2 (2 endpoint + 2 trajectory panels), its two legends (a top 4-item
  legend + a per-panel mini-legend) and its 5-line wall-of-text caption. No
  internal codenames (H3b/H3c/tier/module_support) in titles/axes/legend --
  plain quantity names only (figure_style_guide.md Sec 0.2 / CLAUDE.md Sec
  7-8); the tier/supported bookkeeping lives in the archive doc, not this
  figure. The old per-subject "passes its own robustness checks" black-ring
  marker is removed entirely -- that fact ("only 1/7 individually robust")
  is caption TEXT now, not a marker.

  ``v3_mode_transition_trajectory.png`` (SUPPLEMENTARY) -- the peri-ictal
  phase trajectory (P0..I3, onset buffer O shaded) moved OUT of the main
  figure into its own file: cohort-median OBSERVED (no permutation nulls)
  trajectory of the same two raw metrics, so "is there structure across the
  WHOLE window, not just the P3/I1 endpoints" is still answerable without
  cluttering the main result. Reuses ``_plot_panel_b`` and
  ``_compute_observed_trajectory`` UNCHANGED (only the figure they are
  assembled into is new).

Main-figure data comes from the co-primary CSVs under ``--indir`` (default:
the canonical ``results/topic5_ictal_recruitment/v3_mode_transition`` tree,
so the committed script always re-renders on whatever the pipeline last
wrote there -- point ``--indir`` at a frozen snapshot for a race-free
dev/eyeball render while a background rerun is overwriting the canonical
CSVs). The supplementary trajectory is computed OBSERVED-ONLY (no
permutation nulls) directly from the ictal-field long cache via
``scripts._topic5_v3_io.load_subject_phase_envelopes`` plus the SAME pure
metric chains the run scripts use:
  H3b: ``activations_from_z -> atm_offdiag -> net_offaxis_flux``, median over
       seizures carrying that phase.
  H3c: per sliding window, ``lowrank_var -> dominant_right_singular_vector(k*)
       -> map_lowrank_vector_to_contacts -> subspace_mode_shift(..., "density")``,
       median over windows -> per seizure -> median over seizures.
This always reads the (race-free) field cache and the fixed
``SUBJECTS_BY_SUB`` cohort lists, never ``--indir`` -- so it is already
"final" regardless of which permutation rerun is in flight for the main
figure.

See docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md Task 11
and docs/superpowers/specs/2026-07-02-topic5-v3a-mode-transition-design.md.
This redesign supersedes the original Task-11 2x2 layout (2026-07-04 figure
cleanup: unclear main message, misused legend, wall-of-text caption, wrong
layout -- see docs/archive/topic5/v3a_mode_transition_2026-07-04.md for the
full honest-fragile writeup this figure summarizes).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
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
COHORT_ROLE = {"narrow": "primary", "broad": "replication"}
PHASES = ["P0", "P1", "P2", "P3", "O", "I1", "I2", "I3"]


# ---------------------------------------------------------------------------
# Main-figure data: read the already-computed co-primary subject CSVs (--indir)
# ---------------------------------------------------------------------------
def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv_rows(path: Path) -> list:
    return list(csv.DictReader(path.open())) if path.exists() else []


def _endpoint_subject_deltas(indir: Path, cohort: str) -> dict:
    """``{"h3b": {subject: delta}, "h3c": {subject: delta}, "h3b_raw": {...}}``.

    Reads each endpoint's OWN CSV (avalanche for H3b, dynamics for H3c). A
    row with a non-finite delta (skipped/geometry-insufficient subject) is
    kept in the dict but filtered out by the plotting helper, so a subject
    absent from one endpoint's CSV never silently disappears from the other
    endpoint's panel. The redesigned figure no longer marks individual
    ``module_support_flag`` subjects (that fact is cohort-level caption
    text now, not a per-point marker), so this no longer reads that column.
    """
    av = _read_csv_rows(indir / cohort / "v3_avalanche_subject.csv")
    dyn = _read_csv_rows(indir / cohort / "v3_dynamics_subject.csv")
    h3b = {r["subject"]: _f(r["delta_net_offaxis_flux_surplus"]) for r in av}
    # raw (uncorrected) I1-P3 flux per subject — overlaid as open markers so the
    # reader sees the raw flux mostly DECREASES while the null-corrected surplus
    # is positive (the "amplification" is relative to the rate baseline).
    h3b_raw = {r["subject"]: _f(r.get("delta_net_offaxis_flux_raw", "nan")) for r in av}
    h3c = {r["subject"]: _f(r["delta_mode_shift_density"]) for r in dyn}
    return {"h3b": h3b, "h3c": h3c, "h3b_raw": h3b_raw}


def _h3b_caveats(indir: Path) -> dict:
    """Per-cohort honesty stats for the caption: how many ok-status subjects
    have a NEGATIVE raw flux Δ (so the surplus is null-relative, not absolute)
    and how many are ``common_drive_sensitive`` (lag1~lag0 co-activation, not
    directed propagation). Read straight from the avalanche CSV so they always
    match the rendered points.
    """
    out: dict = {}
    for cohort in ("narrow", "broad"):
        rows = [r for r in _read_csv_rows(indir / cohort / "v3_avalanche_subject.csv")
                if r.get("status") == "ok"]
        raw = [_f(r.get("delta_net_offaxis_flux_raw", "nan")) for r in rows]
        out[cohort] = {
            "n_ok": len(rows),
            "n_raw_neg": sum(1 for v in raw if np.isfinite(v) and v < 0),
            "n_common_drive": sum(1 for r in rows if r.get("common_drive_sensitive") == "True"),
        }
    return out


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
# Supplementary-figure data: observed-only (no perms) phase trajectory
# straight from the field cache. Always uses the canonical SUBJECTS_BY_SUB
# list and the (race-free) ictal_field_long_cache -- independent of --indir.
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
# plotting -- main figure (endpoint panels) + supplementary figure (trajectory)
# ---------------------------------------------------------------------------
def _p_annotation(nb: dict, bb: dict, key: str, ndec: int) -> str:
    """"cohort Holm p: narrow <p> / broad <p> (<0.05|n.s.|mixed)" -- the ONE
    top-left per-panel annotation that replaces the removed per-subject
    robustness marker. ``ndec`` controls decimal places (3 for the
    borderline-significant off-axis-flux endpoint, 2 for the clearly-null
    mode-direction endpoint -- more precision where the significance
    boundary actually matters).
    """
    def fmt(x: float) -> str:
        return "n/a" if not np.isfinite(x) else f"{x:.{ndec}f}"

    p_n, p_b = nb[f"p_holm_{key}"], bb[f"p_holm_{key}"]
    both_pass = bool(nb[f"cohort_{key}_pass"] and bb[f"cohort_{key}_pass"])
    both_ns = bool(not nb[f"cohort_{key}_pass"] and not bb[f"cohort_{key}_pass"])
    tag = "<0.05" if both_pass else ("n.s." if both_ns else "mixed")
    return f"cohort Holm p: narrow {fmt(p_n)} / broad {fmt(p_b)} ({tag})"


def _plot_endpoint_panel(ax, cohort_data: dict, key: str, ylabel: str, title: str,
                          p_annotation: str, show_raw: bool) -> None:
    """ONE main-figure panel: per-subject P3->I1 Delta for ONE endpoint.

    x = subjects grouped into a narrow (primary) block then a broad
    (replication) block, separated by a thin vertical divider; a dashed
    zero line; a thick cohort-median tick spanning each block. FILLED
    marker = the endpoint value itself (colored by cohort) -- for the
    off-axis-flux panel (``show_raw=True``) an OPEN marker of the same
    color overlays the raw (uncorrected) Delta, the direct visual evidence
    that the surplus is elevated relative to a raw baseline that mostly
    DECREASES, not an absolute rise. No per-subject "passes its own
    robustness checks" marker (removed in the 2026-07-04 redesign -- that
    fact is cohort-level caption text, not a marker) and no per-panel
    legend (the ONE shared bottom legend explains filled/open once for the
    whole figure).
    """
    ax.axhline(0.0, color="0.55", lw=1.1, ls="--", zorder=1)
    cursor = 0.0
    gap = 1.2
    xtick_pos: list = []
    xtick_lab: list = []
    for cohort in ("narrow", "broad"):
        items = sorted(cohort_data[cohort][key].items())
        finite = [(s, d) for s, d in items if np.isfinite(d)]
        n = len(finite)
        color = COHORT_COLOR[cohort]
        if n:
            xs = cursor + np.arange(n, dtype=float)
            ys = [d for _, d in finite]
            ax.scatter(xs, ys, s=55, color=color, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
            med = float(np.median(ys))
            ax.plot([xs[0] - 0.4, xs[-1] + 0.4], [med, med], color=color, lw=2.8, zorder=4)
            if show_raw:
                raw = cohort_data[cohort].get("h3b_raw", {})
                rys = [raw.get(s, np.nan) for s, _ in finite]
                ax.scatter(xs, rys, s=38, facecolor="none", edgecolor=color, linewidth=1.2, alpha=0.9, zorder=2)
            block_center = float(xs.mean())
            cursor = xs[-1] + 1.0
        else:
            block_center = cursor
        xtick_pos.append(block_center)
        xtick_lab.append(f"{cohort} ({COHORT_ROLE[cohort]})\nn={n}")
        cursor += gap
        if cohort == "narrow":
            ax.axvline(cursor - gap / 2, color="0.8", lw=1.1, zorder=0)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=9.8)
    ax.set_xlim(-0.8, cursor - gap + 0.8)
    ax.margins(y=0.22)
    ax.set_ylabel(ylabel, fontsize=11.0)
    ax.set_title(title, fontsize=12.2, loc="left", fontweight="bold")
    ax.text(0.02, 0.95, p_annotation, transform=ax.transAxes, fontsize=8.6,
            style="italic", color="0.3", ha="left", va="top")


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


def _short_caption(tier_payload: dict, caveats: dict) -> str:
    """The ONE short caption for the MAIN figure -- replaces the old 5-line
    wall-of-text caption. Plain quantity names only, no tier/module_support
    codenames (those stay in the archive doc); the two fraction numbers are
    read live from the per-subject CSVs (via ``tier_payload``/``caveats``)
    so they always match this render. Wrapped to a fixed character width
    (not matplotlib's ``wrap=True``, which under-wraps centered fig.text)
    so it reliably renders as 2 lines.
    """
    nb = tier_payload["narrow"]
    nc = caveats["narrow"]
    text = (
        "Paired within-seizure P3→I1, n_perm=1000. The off-axis-flux surplus (filled) clears cohort "
        "significance in both cohorts, but it is FRAGILE: the raw flux (open) mostly decreases "
        f"({nc['n_raw_neg']}/{nc['n_ok']} narrow), is mostly common-drive co-activation, only "
        f"{nb['n_subject_support']}/{nb['n_geometry_sufficient']} narrow subjects are individually "
        "robust, and the PRIMARY-cohort significance is knife-edge (Holm p=0.031 — drops >0.05 if any "
        "single narrow subject bar one is removed; only the replication cohort is drop-robust); the "
        "mode-direction endpoint is null. → a fragile data-side candidate signal, not an established "
        "axis→non-axis transition (detail: README)."
    )
    return "\n".join(textwrap.wrap(text, width=220))


def _build_main_figure(endpoint_data: dict, tier_payload: dict, caveats: dict) -> "plt.Figure":
    """The ONE main result figure: a clean 1x2, one endpoint per panel
    (CLAUDE.md Sec 7), ONE shared legend (<=4 entries), ONE short caption --
    replacing the old 2x2 (2 endpoint + 2 trajectory panels, two legends, a
    5-line caption, no single clear takeaway).
    """
    fig, (ax_flux, ax_mode) = plt.subplots(1, 2, figsize=(13.0, 6.3))

    nb, bb = tier_payload["narrow"], tier_payload["broad"]
    _plot_endpoint_panel(
        ax_flux, endpoint_data, "h3b",
        "Δ off-axis flux (I1−P3)",
        "① off-axis flux — surplus significant, but raw mostly decreases",
        _p_annotation(nb, bb, "h3b", 3),
        show_raw=True,
    )
    _plot_endpoint_panel(
        ax_mode, endpoint_data, "h3c",
        "Δ mode-shift density (I1−P3)",
        "② mode-transition direction — null",
        _p_annotation(nb, bb, "h3c", 2),
        show_raw=False,
    )

    fig.tight_layout(rect=(0.02, 0.17, 0.98, 0.94))

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COHORT_COLOR["narrow"],
                   markeredgecolor=COHORT_COLOR["narrow"], markersize=8, label="narrow"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COHORT_COLOR["broad"],
                   markeredgecolor=COHORT_COLOR["broad"], markersize=8, label="broad"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.4", markeredgecolor="white",
                   markersize=8, label="● filled = null-corrected surplus (endpoint)"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="0.4",
                   markersize=8, label="○ open = raw Δ"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.2, bbox_to_anchor=(0.5, 0.095), columnspacing=1.6, handletextpad=0.4)

    caption = _short_caption(tier_payload, caveats)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8.0)

    fig.suptitle(
        "Topic 5 V3a — seizure axis→non-axis (paired P3→I1, EXPLORATORY): "
        "a fragile off-axis-flux signal; mode-direction null",
        fontsize=12.8, fontweight="bold", y=0.975,
    )
    return fig


def _build_trajectory_figure(trajectory_data: dict) -> "plt.Figure":
    """SUPPLEMENTARY figure (``v3_mode_transition_trajectory.png``): the
    peri-ictal phase trajectory moved OUT of the main figure (2026-07-04
    redesign). A clean 1x2, reusing ``_plot_panel_b`` UNCHANGED -- one panel
    per raw metric, narrow vs broad, O-buffer shaded; its own legend/
    caption/suptitle since this is now a standalone file, not a row of the
    main figure.
    """
    fig, (ax_flux, ax_mode) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    _plot_panel_b(ax_flux, trajectory_data, "h3b",
                  "net off-axis flux\n(observed, source-normalized)",
                  "off-axis flux -- phase trajectory (observed)")
    _plot_panel_b(ax_mode, trajectory_data, "h3c",
                  "mode-shift density\n(observed, non-axis minus axis)",
                  "mode-transition density -- phase trajectory (observed)")

    fig.tight_layout(rect=(0.02, 0.15, 0.98, 0.92))

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=COHORT_COLOR["narrow"], lw=2.0, markersize=6,
                   label="narrow (primary)"),
        plt.Line2D([0], [0], marker="o", color=COHORT_COLOR["broad"], lw=2.0, markersize=6,
                   label="broad (replication)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.06))

    caption = "\n".join(textwrap.wrap(
        "Supplementary to v3_mode_transition_summary.png: cohort-median OBSERVED trajectory (no "
        "permutation nulls) across all 8 seizure phases (P0–P3 preictal → O onset buffer "
        "→ I1–I3 ictal); O is a descriptive buffer only, never part of the primary "
        "P3→I1 contrast in the main figure (detail: README).",
        width=190,
    ))
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8.0)

    fig.suptitle(
        "Topic 5 V3a supplementary — peri-ictal phase trajectory (observed, no permutation nulls)",
        fontsize=12.3, fontweight="bold", y=0.965,
    )
    return fig


def _write_readme(outdir: Path, tier_payload: dict, caveats: dict) -> Path:
    """Chinese ``figures/README.md`` (AGENTS.md format): one ``### filename``
    section per PNG, written AFTER both PNGs so every number matches THIS
    render. Honest-fragile framing throughout (2026-07-04 redesign: clean
    1x2 main figure + a separate supplementary trajectory PNG).
    """
    nb, bb = tier_payload["narrow"], tier_payload["broad"]
    nc = caveats["narrow"]
    tier = tier_payload["tier"]
    supported = tier_payload["state_v3_supported"]
    body = (
        "### v3_mode_transition_summary.png\n\n"
        "这张图检验：发作真正开始前后（发作前 30~10 秒 → 发作后 10~30 秒，**每次发作按它自己的脑电起始"
        "配对**），系统里最容易被放大的连锁流向/活动方向，是不是从病人间期就走熟的固定高频通路，转移到"
        "通路之外的电极上。**左图**是每个病人『非轴向净流』的前后变化量，**右图**是『模态转移方向』的前后"
        f"变化量（narrow 主力队列 n={nb['n_geometry_sufficient']}，broad 只作复制 "
        f"n={bb['n_geometry_sufficient']}、从不与主力合并）；每张子图里**实心点**=扣除放电率随机基线后的"
        "增量（承重端点），**空心点**（只出现在左图）=未扣基线的原始变化量。\n\n"
        "实测（配对、正式 n_perm=1000）：扣掉每触点放电率随机基线后的『非轴向净流增量』（左图实心点）在"
        "主力+复制队列都达到了队列级显著（判读机械上到顶档），但这是一个**很脆的阳性**，不能读成『轴→非轴"
        f"模态转移成立』——① 左图空心点（未扣基线的原始流）大多在下降（{nc['n_raw_neg']}/{nc['n_ok']} 个"
        f"主力病人）：所谓『放大』是相对随机基线、不是绝对上升；② {nc['n_common_drive']}/{nc['n_ok']} 个"
        "主力病人以同时共激活为主（lag1≈lag0），不是定向传导；③ 只有 "
        f"{nb['n_subject_support']}/{nb['n_geometry_sufficient']} 个病人个体层面过了全部稳健性检验（这条"
        "只写进文字说明，图上不再画黑色描边的『个体稳健』标记）；④ 右图『模态转移方向』端点是阴的（两队列 "
        "Holm p 都远高于 0.05）。总体是一个数据侧候选信号，待机制侧(V3b)与敏感性检验，不是确立的支持"
        f"（内部记账：evidence tier {tier}/4，formally supported={supported}）。\n\n"
        "**关注点**：左图里**空心点（原始流）大多落在 0 以下、实心点（扣基线后的增量）落在 0 以上**——这"
        "就是『相对基线偏高、绝对在降』的直接视觉证据；右图的点整体紧贴 0 线，两个队列的中位线都几乎不偏离"
        "零，与左图的偏移形成对比。\n\n"
        "### v3_mode_transition_trajectory.png\n\n"
        "这张图是补充材料：不只看发作前后两个端点，而是把发作前 2 分钟到发作后的整段时间线（P0…P3 发作前 "
        "→ O 起始缓冲 → I1…I3 发作后）都画出来，看『非轴向净流』（左）和『模态转移方向』（右）在这条完整"
        "时间线上有没有结构，而不只是两个端点之间的一个变化量。两条线（narrow/broad）都是队列中位数，"
        "**没有做置换检验**（纯描述性的『实测长什么样』，不是承重统计）；起始前后 10 秒（O，灰色底纹）只是"
        "缓冲窗口，不算进主图 P3→I1 的承重对比。\n\n"
        "**关注点**：两条队列曲线在灰色缓冲窗口前后都没有明显的单调爬升或断崖式变化，更像是围绕各自基线的"
        "轻微波动——这不影响主图『P3→I1 两个端点之间确实有队列级变化』的结论，只是说明这个变化在更细的时间"
        "颗粒度上没有呈现出贯穿全程的清楚趋势。\n"
    )
    readme_path = outdir / "README.md"
    readme_path.write_text(body, encoding="utf-8")
    return readme_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=str(_DEFAULT_INDIR),
                    help="Main-figure co-primary CSV tree root (default: canonical results path). "
                         "The supplementary trajectory figure always reads the canonical field cache "
                         "+ SUBJECTS_BY_SUB, independent of --indir.")
    ap.add_argument("--outdir", default=None, help="default: <indir>/figures")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_v3_config()

    endpoint_data = {c: _endpoint_subject_deltas(indir, c) for c in ("narrow", "broad")}
    caveats = _h3b_caveats(indir)
    tier_payload = _load_tier_payload(indir)

    fig_main = _build_main_figure(endpoint_data, tier_payload, caveats)
    out_png = outdir / "v3_mode_transition_summary.png"
    fig_main.savefig(out_png, dpi=170, bbox_inches="tight")
    print(f"[fig] -> {out_png}", flush=True)

    print("[fig] computing supplementary phase trajectory (observed-only, no permutation nulls) "
          "from the field cache (race-free w.r.t. any main-figure rerun; ~1-2 min)...", flush=True)
    trajectory_data = {c: _compute_observed_trajectory(c, cfg) for c in ("narrow", "broad")}
    fig_traj = _build_trajectory_figure(trajectory_data)
    out_traj_png = outdir / "v3_mode_transition_trajectory.png"
    fig_traj.savefig(out_traj_png, dpi=170, bbox_inches="tight")
    print(f"[fig] -> {out_traj_png}", flush=True)

    out_readme = _write_readme(outdir, tier_payload, caveats)
    print(f"[fig] -> {out_readme}", flush=True)
    return out_png


if __name__ == "__main__":
    main()
