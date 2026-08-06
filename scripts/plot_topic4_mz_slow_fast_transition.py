"""Topic 4 MZ slow–fast dynamical transition — four-panel candidate figure (design §9).

Each panel answers ONE question (CLAUDE.md §7):
  A  natural trajectories: how do R_E, D, a approach onset for the four conditions?
  B  frozen-fast escape:   does P_runaway step across a boundary as the natural slow state (D) advances?
  C  ignition + recovery:  do epsilon_c and tau_rec show a boundary (epsilon_c->0, tau_rec up) near it?
  D  M/Z counterfactuals:  is proximity to the boundary set by z (disinhibition) or m (adaptation)?

Reuses the Topic 4 palette/typography from plot_topic4_mz_onset_dynamics.py. Reads the runner outputs under
results/topic4_sef_hfo/mz_slow_fast_transition/. Renders gracefully from partial data (mid-run eyeballing).
No spatial maps (design §11). PNG + PDF.
"""
import glob
import json
import os
import sys

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slow_fast_transition")
FIGDIR = os.path.join(OUT, "figures")

DT = 0.1
COND_ORDER = ["z_only", "mz_runaway", "mz_edge", "mz_plateau"]
COND_COLOR = {"z_only": "#c0392b", "mz_runaway": "#e67e22", "mz_edge": "#8e44ad", "mz_plateau": "#2c6fbb"}
COND_LABEL = {"z_only": "z-only (A=0)", "mz_runaway": "runaway (τ=500)",
              "mz_edge": "edge (τ=1000)", "mz_plateau": "plateau (τ=2000)"}
STATE_ORDER = ["baseline_1000ms", "mid_fraction", "pre_onset_2000ms", "pre_onset_1000ms",
               "pre_onset_500ms", "pre_onset_200ms", "pre_onset_100ms", "first_crossing"]
CF_ORDER = ["native_zm", "native_z_reset_m", "reset_z_native_m", "late_z_early_m", "early_z_late_m"]
CF_LABEL = {"native_zm": "native z+m", "native_z_reset_m": "m→0", "reset_z_native_m": "z→1",
            "late_z_early_m": "late z / early m", "early_z_late_m": "early z / late m"}
SEEDS = [1, 3, 4]


# ------------------------------------------------------------------ loaders
def _load_natural(cond, seed):
    p = os.path.join(OUT, "per_state", f"{cond}_seed{seed}_natural.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    cr = float(d["crossing_ms"])
    return dict(t=d["t_ms"], D=d["D"], a=d["a"], rate=d["rate_E_hz"], O_s=float(d["onset_anchor_ms"]),
               crossing=(None if not np.isfinite(cr) else cr))


def _load_states(cond, seed):
    out = {}
    for st in STATE_ORDER:
        p = os.path.join(OUT, "per_state", f"{cond}_seed{seed}_{st}.json")
        if os.path.exists(p):
            out[st] = json.load(open(p))
    return out


def _load_cf(cond, seed):
    p = os.path.join(OUT, "counterfactual", f"{cond}_seed{seed}.json")
    return {r["branch"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else {}


def _load_matched_d(cond, seed):
    p = os.path.join(OUT, "matched_d", f"{cond}_seed{seed}.json")
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def _labels():
    p = os.path.join(OUT, "slow_fast_transition_summary.json")
    if not os.path.exists(p):
        return {}
    return {c: v.get("consensus") for c, v in json.load(open(p)).get("per_condition", {}).items()}


# ------------------------------------------------------------------ Panel A: natural trajectories
def panel_A(sub):
    grid = np.arange(-3000.0, 250.0, 5.0)
    axes = [plt.subplot(sub[i]) for i in range(3)]
    series = [("rate", "E rate (Hz)"), ("D", "disinhibition\nD = 1 − z̄"), ("a", "adaptation\na = η$_m$ m̄ / I$_{EE}$")]
    any_data = False
    for cond in COND_ORDER:
        col = COND_COLOR[cond]
        stacks = {k: [] for k, _ in series}
        for seed in SEEDS:
            nat = _load_natural(cond, seed)
            if nat is None:
                continue
            any_data = True
            x = nat["t"] - nat["O_s"]
            for k, _ in series:
                stacks[k].append(np.interp(grid, x, nat[k], left=np.nan, right=np.nan))
        for ax, (k, _) in zip(axes, series):
            if not stacks[k]:
                continue
            M = np.vstack(stacks[k])
            mean = np.nanmean(M, axis=0)
            ax.plot(grid / 1000.0, mean, color=col, lw=1.5, zorder=4)
            if M.shape[0] > 1:
                ax.fill_between(grid / 1000.0, np.nanmin(M, 0), np.nanmax(M, 0), color=col, alpha=0.12, lw=0)
    for ax, (k, ylab) in zip(axes, series):
        ax.axvline(0.0, color="#888", ls=":", lw=0.9, zorder=1)
        ax.set_ylabel(ylab, fontsize=8.3)
        ax.tick_params(labelsize=7.5)
        if k != "a":
            ax.set_xticklabels([])
    axes[0].set_title("A · natural approach to onset  (onset-aligned, mean±seed-range)",
                      fontsize=9.5, loc="left", weight="bold")
    axes[2].set_xlabel("time relative to z-only onset (s)", fontsize=8.5)
    if not any_data:
        axes[1].text(0.5, 0.5, "no natural trajectories yet", ha="center", va="center", transform=axes[1].transAxes)


# ------------------------------------------------------------------ Panel B: P_runaway vs D
def panel_B(ax):
    # NO connecting line (it implied a continuous step through unsampled D — misleading, esp. for MZ whose own
    # onset region is 1.6-3.8 s past the last matched-time point). Show only the ACTUAL sampled points:
    # filled o = matched-time checkpoints; hollow s = matched-D checkpoints (these sample the mid-D range);
    # star = first crossing. + Wilson CI bars.
    for cond in COND_ORDER:
        col = COND_COLOR[cond]
        for seed in SEEDS:
            st = _load_states(cond, seed)
            for s in STATE_ORDER:
                if s not in st or st[s].get("p_runaway") is None:
                    continue
                d, p = st[s]["D"], st[s]["p_runaway"]
                ci = st[s].get("p_runaway_ci") or [None, None]
                if ci[0] is not None:
                    ax.plot([d, d], [ci[0], ci[1]], color=col, lw=0.7, alpha=0.4, zorder=2)
                if s == "first_crossing":
                    ax.scatter([d], [p], marker="*", s=110, color=col, edgecolor="white", lw=0.7, zorder=6)
                else:
                    ax.scatter([d], [p], marker="o", s=18, color=col, alpha=0.8, zorder=4)
            for r in _load_matched_d(cond, seed):   # mid-D samples (the "middle" of the curve)
                if r.get("p_runaway") is None:
                    continue
                ax.scatter([r["D"]], [r["p_runaway"]], marker="s", s=22, facecolor="none",
                           edgecolor=col, lw=1.0, alpha=0.8, zorder=5)
    ax.set_xlabel("natural slow state   D = 1 − z̄", fontsize=9)
    ax.set_ylabel("P(operational runaway)  frozen fast system", fontsize=9)
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("B · perturbation-free escape probability  (points only — no interpolation)",
                 fontsize=9.5, loc="left", weight="bold")
    ax.tick_params(labelsize=7.5)
    ax.legend(handles=[Line2D([0], [0], color=COND_COLOR[c], lw=0, marker="o", ms=5, label=COND_LABEL[c])
                       for c in COND_ORDER] +
                      [Line2D([0], [0], marker="s", color="#555", markerfacecolor="none", lw=0, ms=6, label="matched-D"),
                       Line2D([0], [0], marker="*", color="#555", lw=0, ms=10, label="first crossing")],
              fontsize=6.8, loc="center right", frameon=True, framealpha=0.9, edgecolor="#ccc")


# ------------------------------------------------------------------ Panel C: epsilon_c + tau_rec vs D
def panel_C(ax):
    ax2 = ax.twinx()
    for cond in COND_ORDER:
        col = COND_COLOR[cond]
        eps_pts, tau_pts = [], []
        for seed in SEEDS:
            st = _load_states(cond, seed)
            for s in STATE_ORDER:
                if s not in st:
                    continue
                D = st[s]["D"]
                ec = st[s].get("epsilon_c")
                if ec is not None:
                    eps_pts.append((D, ec))
                tr = st[s].get("tau_rec")
                if tr is not None:
                    tau_pts.append((D, tr))
        if eps_pts:
            eps_pts.sort()
            ax.plot([p[0] for p in eps_pts], [p[1] for p in eps_pts], color=col, lw=1.2, marker="o", ms=3, zorder=4)
        if tau_pts:
            tau_pts.sort()
            ax2.plot([p[0] for p in tau_pts], [p[1] for p in tau_pts], color=col, lw=1.0, ls="--", marker="s",
                     ms=2.6, alpha=0.7, zorder=3)
    ax.set_xlabel("natural slow state   D = 1 − z̄", fontsize=9)
    ax.set_ylabel("ε$_c$  global ignition threshold (vth-gap)", fontsize=9)
    ax2.set_ylabel("τ$_{rec}$  recovery time (ms, dashed)", fontsize=9)
    ax.set_title("C · ignition threshold + recovery time", fontsize=9.5, loc="left", weight="bold")
    ax.tick_params(labelsize=7.5); ax2.tick_params(labelsize=7.5)
    ax.annotate("solid = ε$_c$ (→0 = spontaneous)\ndashed = τ$_{rec}$ (↑ = critical slowing)\ncensored points omitted",
                xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top", fontsize=6.6, color="#555")


# ------------------------------------------------------------------ Panel D: counterfactuals
def panel_D(ax):
    conds = [c for c in COND_ORDER if any(_load_cf(c, s) for s in SEEDS)]
    if not conds:
        ax.text(0.5, 0.5, "no counterfactuals yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D · state-matched M/Z counterfactuals", fontsize=9.5, loc="left", weight="bold")
        return
    n = len(CF_ORDER)
    width = 0.8 / n
    cf_hatch = {"native_zm": "", "native_z_reset_m": "//", "reset_z_native_m": "..",
                "late_z_early_m": "xx", "early_z_late_m": "\\\\"}
    for ci, cond in enumerate(conds):
        vals = {br: [] for br in CF_ORDER}
        for seed in SEEDS:
            cf = _load_cf(cond, seed)
            for br in CF_ORDER:
                if br in cf and cf[br].get("p_runaway") is not None:
                    vals[br].append(cf[br]["p_runaway"])
        for bi, br in enumerate(CF_ORDER):
            if not vals[br]:
                continue
            x = ci + (bi - (n - 1) / 2) * width
            m = float(np.mean(vals[br]))
            ax.bar(x, m, width=width * 0.92, color=COND_COLOR[cond], edgecolor="#333", lw=0.5,
                   hatch=cf_hatch[br], alpha=(1.0 if br == "native_zm" else 0.55))
            if len(vals[br]) > 1:
                ax.plot([x, x], [min(vals[br]), max(vals[br])], color="#333", lw=0.7)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABEL[c].split(" ")[0] for c in conds], fontsize=8)
    ax.set_ylabel("P(operational runaway)  at pre-onset (100 ms)", fontsize=8.6)
    ax.set_ylim(0, 1.04)
    ax.set_title("D · state-matched M/Z counterfactuals", fontsize=9.5, loc="left", weight="bold")
    ax.tick_params(labelsize=7.5)
    ax.text(0.5, 0.60, "UNINFORMATIVE NULL (not evidence about m):\n"
                       "branch = pre-onset 100 ms (z-only anchor) is far below the\n"
                       "boundary for the MZ conditions → nothing tips (P≈0); AND the\n"
                       "replays are NOT common-noise-paired (z-only native vs m→0 are\n"
                       "physically identical yet gave P=0 vs 0.25). Cannot judge z vs m here.",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.4, color="#555",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fbf3f3", ec="#e0c0c0", lw=0.8))
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor="#bbb", markeredgecolor="#333",
                              markersize=9, label=CF_LABEL[br]) for br in CF_ORDER],
              fontsize=6.6, loc="upper right", frameon=True, framealpha=0.9, edgecolor="#ccc", ncol=1)


# ------------------------------------------------------------------ compose
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    labels = _labels()
    fig = plt.figure(figsize=(14.5, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22, left=0.06, right=0.955, top=0.9, bottom=0.07)
    panel_A(GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 0], hspace=0.12))
    panel_B(plt.subplot(gs[0, 1]))
    panel_C(plt.subplot(gs[1, 0]))
    panel_D(plt.subplot(gs[1, 1]))
    _sc = {"z_only": "z-only", "mz_runaway": "runaway", "mz_edge": "edge", "mz_plateau": "plateau"}
    _sl = {"dynamical_tipping": "tipping", "finite_amplitude_escape": "finite-amp",
           "noise_driven_escape": "noise", "smooth_crossover": "smooth", "seed-inconsistent": "seed-incons"}
    verdict = " · ".join(f"{_sc[c]}:{_sl.get(labels.get(c, ''), labels.get(c, '—'))}" for c in COND_ORDER) \
        if labels else "run aggregate for labels"
    fig.suptitle("MZ slow–fast dynamical transition — frozen fast system across the natural slow-state drift\n"
                 "E1146; operational runaway 120 Hz / 100 ms; model-side mechanism, NOT seizure    "
                 f"class: {verdict}",
                 fontsize=9.5, weight="bold", x=0.06, ha="left")
    base = os.path.join(FIGDIR, "mz_slow_fast_transition")
    fig.savefig(base + ".png", dpi=150)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"[plot] wrote {base}.png / .pdf")


if __name__ == "__main__":
    main()
