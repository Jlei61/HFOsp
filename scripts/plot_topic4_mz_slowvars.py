"""M4-MZ discovery figures (design §13). Reads the runner artifacts and renders 3 self-contained
figures into results/topic4_sef_hfo/mz_slowvars/figures/. Plotting only -- no simulation.

  mz_phenotype_map.png      -- parameter-sweep phenotype map (arms A/B/C, 3x3 each). Q: what does each
                               z / m / z+m configuration produce?
  mz_mechanism_traces.png   -- slow-off vs z-only vs m-only vs z+m: rate + z + adaptation traces.
                               Q: what is the dynamical mechanism (does z push, does m brake)?
  mz_spatial_recruitment.png-- peak E-active spatial footprint + source/sink cores + axis.
                               Q: where does recruitment go (does z+m stay axial or fill the sheet)?

Categorical phenotype colors; activity magnitude uses viridis (style guide: sequential=viridis).
Labels are plain-language (no q50 / zA_* code-words on the axes).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slowvars")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# phenotype -> (color, short label). Green = the wanted bounded+recovered outcome; red = runaway failure.
PHENO = {
    "interictal_like": ("#4E79A7", "inter"),
    "expanded_bounded": ("#F28E2B", "exp-B"),
    "expanded_returned": ("#59A14F", "exp-R"),
    "fragment": ("#B07AA1", "frag"),
    "suppress": ("#79706E", "supp"),
    "runaway": ("#E15759", "run"),
    "insufficient": ("#D3D3D3", "insuf"),
}


def _load(name):
    p = os.path.join(OUT_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _grid_cell_text(r):
    return f"{PHENO.get(r['phenotype'], ('', '?'))[1]}\nn={r['n_events']}"


def _draw_arm(ax, rows, row_keys, col_keys, row_labels, col_labels, title, keyfn):
    """One arm's 3x3 phenotype grid. keyfn(row)->(row_key,col_key)."""
    by = {keyfn(r): r for r in rows}
    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            r = by.get((rk, ck))
            color = PHENO.get(r["phenotype"], ("#ffffff", "?"))[0] if r else "#ffffff"
            ax.add_patch(plt.Rectangle((j, len(row_keys) - 1 - i), 1, 1, facecolor=color,
                                       edgecolor="white", linewidth=2))
            if r:
                ax.text(j + 0.5, len(row_keys) - 1 - i + 0.5, _grid_cell_text(r),
                        ha="center", va="center", fontsize=8,
                        color="white" if r["phenotype"] in ("runaway", "suppress", "interictal_like") else "black")
    ax.set_xlim(0, len(col_keys)); ax.set_ylim(0, len(row_keys))
    ax.set_xticks(np.arange(len(col_keys)) + 0.5); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(len(row_keys)) + 0.5); ax.set_yticklabels(row_labels[::-1], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig_phenotype_map(summary, cal):
    rows = summary["rows"]
    ith = cal["I_th_EI"]
    A = [r for r in rows if r["arm"] == "A"]
    B = [r for r in rows if r["arm"] == "B"]
    Cc = [r for r in rows if r["arm"] == "C"]

    def kA(r):
        l = r["label"].split("_"); return (l[1], int(l[2][2:]))          # (q50, tau_z)

    def kB(r):
        l = r["label"].split("_"); return (int(l[1][2:]), int(l[2][1:]))  # (tau_adp, frac%)

    def kC(r):
        l = r["label"].split("_"); return (l[1][1:], l[2][1:])            # (z level, m level)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    _draw_arm(axes[0], A, ["q50", "q75", "q90"], [2500, 5000, 10000],
              [f"strong\nI_th={ith['q50']:.1f}", f"mid\nI_th={ith['q75']:.0f}", f"weak\nI_th={ith['q90']:.0f}"],
              ["2.5 s", "5 s", "10 s"], "Arm A — disinhibition only (z)", kA)
    axes[0].set_ylabel("inhibitory-efficacy depletion strength", fontsize=10)
    axes[0].set_xlabel("z recovery time  τ_z", fontsize=10)
    _draw_arm(axes[1], B, [500, 2000, 5000], [5, 10, 20],
              ["0.5 s", "2 s", "5 s"], ["low\n5%", "mid\n10%", "high\n20%"],
              "Arm B — adaptation only (m)", kB)
    axes[1].set_ylabel("adaptation time  τ_adp", fontsize=10)
    axes[1].set_xlabel("adaptation current level (η_m × peak m / excit. scale)", fontsize=10)
    _draw_arm(axes[2], Cc, ["w", "m", "s"], ["w", "m", "s"],
              ["z weak", "z mid", "z strong"], ["m weak", "m mid", "m strong"],
              "Arm C — z + m combined", kC)
    axes[2].set_ylabel("realized disinhibition", fontsize=10)
    axes[2].set_xlabel("realized adaptation", fontsize=10)

    present = [p for p in PHENO if any(r["phenotype"] == p for r in rows)]
    handles = [Patch(facecolor=PHENO[p][0], edgecolor="gray", label=f"{p}") for p in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"M4-MZ phenotype map — E1146, spontaneous, seed {summary['seed']}, "
                 f"{summary['T'] / 1000:.0f} s   (slow-off baseline: {summary['baseline']['n_events']} "
                 f"returning interictal events)", fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "mz_phenotype_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _tags(cap):
    order = ["slow_off"]
    for arm in ("A", "B", "C"):
        order += sorted([k[:-7] for k in cap.files if k.endswith("__rate") and k.startswith(f"arm{arm}_")])
    return [t for t in order if f"{t}__rate" in cap.files]


def _title_for(cap, tag):
    ph = str(cap[f"{tag}__pheno"])
    if tag == "slow_off":
        return "slow-off (baseline)"
    arm = tag[3]
    kind = {"A": "z-only", "B": "m-only", "C": "z+m"}[arm]
    return f"{kind}\n{ph}"


def fig_mechanism_traces(cap, summary):
    tags = _tags(cap)
    T = float(summary["T"]) / 1000.0
    n = len(tags)
    fig, axes = plt.subplots(3, n, figsize=(3.4 * n, 8), sharex=True)
    if n == 1:
        axes = axes[:, None]
    for j, tag in enumerate(tags):
        rate = cap[f"{tag}__rate"]; t = np.linspace(0, T, len(rate))
        axes[0, j].plot(t, rate, color="#222222", lw=0.8)
        axes[0, j].set_title(_title_for(cap, tag), fontsize=10, fontweight="bold")
        axes[0, j].axhline(120, color="#E15759", ls=":", lw=1)                # runaway line (120 Hz)
        z = cap[f"{tag}__z_mean"]; zc = cap[f"{tag}__z_core"]
        tz = np.linspace(0, T, len(z))
        axes[1, j].plot(tz, z, color="#4E79A7", lw=1.0, label="z mean (all E)")
        axes[1, j].plot(np.linspace(0, T, len(zc)), zc, color="#1f3b5c", lw=1.2, label="z core")
        axes[1, j].set_ylim(-0.05, 1.05)
        ad = cap[f"{tag}__adap"]
        axes[2, j].plot(np.linspace(0, T, len(ad)), ad, color="#E15759", lw=1.0, label="η_m·m (mean)")
        axes[2, j].set_xlabel("time (s)", fontsize=9)
    axes[0, 0].set_ylabel("E population rate (Hz)", fontsize=10)
    axes[1, 0].set_ylabel("inhibitory efficacy z", fontsize=10)
    axes[2, 0].set_ylabel("adaptation current (mV)", fontsize=10)
    axes[1, -1].legend(fontsize=7, loc="lower left", frameon=False)
    axes[2, -1].legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("M4-MZ mechanism decomposition — does disinhibition (z) push events up, "
                 "does adaptation (m) brake them?", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mz_mechanism_traces.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_spatial_recruitment(cap, summary):
    tags = _tags(cap)
    L = float(cap["L"]); core_r = float(cap["core_r"])
    src = cap["src_xy"]; snk = cap["snk_xy"]
    n = len(tags)
    vmax = max((cap[f"{t}__movie"].max(axis=0).max() for t in tags if f"{t}__movie" in cap.files), default=1.0)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
    if n == 1:
        axes = [axes]
    im = None
    for j, tag in enumerate(tags):
        mv = cap[f"{tag}__movie"]
        foot = mv.max(axis=0) if mv.size else np.zeros((24, 24))              # peak recruited footprint
        im = axes[j].imshow(foot, extent=[0, L, 0, L], origin="lower", cmap="viridis",
                            vmin=0, vmax=vmax, aspect="equal")
        for c in (src, snk):
            axes[j].add_patch(Circle((c[0], c[1]), core_r, fill=False, edgecolor="#E15759", lw=1.6))
        axes[j].plot([src[0], snk[0]], [src[1], snk[1]], color="white", ls="--", lw=1.0)
        axes[j].set_title(_title_for(cap, tag), fontsize=10, fontweight="bold")
        axes[j].set_xlabel("x (mm)", fontsize=9)
        if j == 0:
            axes[j].set_ylabel("y (mm)", fontsize=9)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("peak E-active fraction", fontsize=9)
    fig.suptitle("M4-MZ spatial recruitment footprint — red circles = source/sink cores, "
                 "dashed line = patient propagation axis", fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIG_DIR, "mz_spatial_recruitment.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summary = _load("discovery_summary.json")
    cal = _load("calibration.json")
    if summary is None or "rows" not in summary:
        print("no discovery_summary.json with rows -> nothing to plot", file=sys.stderr)
        sys.exit(1)
    fig_phenotype_map(summary, cal)
    print("wrote mz_phenotype_map.png")
    cap_path = os.path.join(OUT_DIR, "figure_capture.npz")
    if os.path.exists(cap_path):
        cap = np.load(cap_path, allow_pickle=True)
        fig_mechanism_traces(cap, summary)
        print("wrote mz_mechanism_traces.png")
        fig_spatial_recruitment(cap, summary)
        print("wrote mz_spatial_recruitment.png")
    else:
        print("no figure_capture.npz -> run `capture-figures` first for traces + spatial figs", file=sys.stderr)


if __name__ == "__main__":
    main()
