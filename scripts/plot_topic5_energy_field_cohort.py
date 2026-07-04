"""Topic5 能量场外推 cohort 图：per-subject F_core_only vs 逐通道基线 + null 显著性。

每频段一张：x=baseline(C1/C2)、y=F_core_only；对角线上方=场赢基线；
红=F_core_only 过 channel null。配 FDR 表摘要。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/topic5_ictal_recruitment/field_extrapolation")
PDIR = OUT / "cohort_per_subject"


def _short(sid):
    return sid.replace("epilepsiae_", "E").replace("yuquan_", "Y:")


def plot_band(band):
    rows = [json.load(open(f)) for f in sorted(PDIR.glob(f"*__{band}.json"))]
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        print(f"{band}: no ok subjects"); return
    vals = [v for r in ok for v in (r["F_core_only"], r["C1"], r["C2"]) if np.isfinite(v)]
    vmax = max(0.85, (max(vals) + 0.05)) if vals else 0.85   # 动态上限, 防裁点(E916 hfa=0.842)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, base in zip(axes, ["C1", "C2"]):
        for r in ok:
            f, c = r["F_core_only"], r[base]
            sig = np.isfinite(r["null_channel_p"]) and r["null_channel_p"] < 0.05
            ax.scatter(c, f, c="crimson" if sig else "lightgray", s=90,
                       edgecolors="k", linewidths=0.6, zorder=5)
            ax.annotate(_short(r["subject"]), (c, f), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        lim = [0, vmax]
        ax.plot(lim, lim, "k--", alpha=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        nm = "own interictal order (C1)" if base == "C1" else "own energy fingerprint (C2)"
        ax.set_xlabel(f"baseline: {nm}  (|rho| median)")
        ax.set_ylabel("F_core_only  (core-only field, |rho| median)")
        ax.set_title(f"{band}: F_core_only vs {base}\nred=F_core passes channel null; above diag=field beats baseline")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT / "figures" / f"cohort_energy_F_core_vs_baselines_{band}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="*", default=["bb_auc", "hfa_auc"])
    args = ap.parse_args()
    for b in args.bands:
        plot_band(b)
    fj = OUT / "energy_field_extrapolation_FINAL.json"
    if fj.exists():
        print("\n=== FDR table ===")
        for r in json.load(open(fj)):
            q = r.get("fdr_q")
            print(f"{r['band']:8} {r['hypothesis']:22} {r['kind']:16} "
                  f"n={r['n_subjects']:>2} p={r['cohort_p']:.4f} q={q if q is None else round(q,4)}")


if __name__ == "__main__":
    main()
