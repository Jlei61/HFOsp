"""Audit view of computed periodic folds; does not extrapolate stability labels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/topic4_sef_hfo/fig5_zm_rate_synchronized_20260917"
AUDIT = RESULT / "recheck_20260917"
PERIODIC = RESULT / "periodic_completion"


def read(path):
    return json.loads(path.read_text())


def main():
    rows = [read(p) for p in sorted((PERIODIC / "orbits").glob("burstUp_*N512.json"))]
    extra = [read(p) for p in sorted((AUDIT / "orbits").glob("LPC_small_*N512.json"))]
    small = sorted(rows[27:35] + extra, key=lambda q: q["mean_rates_hz"][1])
    folds = [read(AUDIT / "orbits" / f"LPC_small_{kind}_N512.json") for kind in ["max", "min"]]
    mainfold = read(PERIODIC / "orbits/LPC_burst_N1024.json")
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    fig.subplots_adjust(left=.09, right=.98, top=.93, bottom=.27, wspace=.30)
    for ax, points in zip(axes, [rows[:27] + small + rows[35:], small]):
        ax.plot([q["D"] for q in points], [q["global_mean_hz"] for q in points],
                ".:", color="#606060", ms=4, lw=.9)
        ax.set_xlabel(r"$D=1-\langle Z_E\rangle$")
        ax.set_ylabel("Global E period mean (Hz / neuron)")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.7f"))
        ax.tick_params(axis="x", rotation=20)
    axes[0].set(xlim=(.18845, .18866), ylim=(16.58, 17.03),
                xticks=[.18845, .18855, .18865])
    axes[1].set(xlim=(.18852392, .18852434), ylim=(16.795, 16.86),
                xticks=[.1885240, .1885241, .1885243])
    for i, fold in enumerate(folds):
        for ax in axes:
            ax.plot(fold["D"], fold["global_mean_hz"], "*", color="#b23245", ms=12, zorder=5)
        offset = (-57, 6) if i == 0 else (14, 10)
        axes[1].annotate(f"LPC {i+1}", (fold["D"], fold["global_mean_hz"]),
                         xytext=offset, textcoords="offset points", color="#922638")
    axes[0].plot(mainfold["D"], mainfold["global_mean_hz"], "*", color="#b23245", ms=12)
    axes[0].annotate("LPC 3", (mainfold["D"], mainfold["global_mean_hz"]),
                     xytext=(-54, -32), textcoords="offset points", color="#922638")
    axes[0].annotate("LPC 1, 2", (folds[0]["D"], folds[0]["global_mean_hz"]),
                     xytext=(-64, 30), textcoords="offset points", color="#922638",
                     arrowprops={"arrowstyle": "-", "lw": .7, "color": "#922638"})
    handles = [Line2D([], [], marker=".", ls=":", color="#606060",
                      label="Periodic geometry; stability not assigned"),
               Line2D([], [], marker="*", ls="none", color="#b23245", ms=11,
                      label="LPC: fold of periodic orbits")]
    for name, ax in zip("AB", axes):
        ax.text(-.14, 1.03, name, transform=ax.transAxes, fontsize=17, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
               frameon=False, ncol=1, fontsize=11)
    out = AUDIT / "figures"
    out.mkdir(exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"fig_periodic_fold_coverage_audit.{ext}", dpi=220)
    plt.close(fig)
    (out / "fig_periodic_fold_coverage_audit_metadata.json").write_text(json.dumps({
        "model": "Fixed g20 / 935-group Z-held, M-dynamic rate DDE",
        "meaning": "Audit of fold geometry; no interpolation of Floquet stability",
        "LPC_numbering": "Local audit labels 1, 2, 3 in continuation order; prior main LPC is LPC 3",
        "fold_sources": [str(AUDIT / "orbits" / f"LPC_small_{k}_N512.json") for k in ["max", "min"]]
                        + [str(PERIODIC / "orbits/LPC_burst_N1024.json")],
        "human_visual_acceptance": "PENDING"
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
