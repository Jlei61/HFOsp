"""E958 negative diagnostic (field-swap plan §3D / review P1-3): is the no-direction result a
sparse-sampling effect or a registration/coverage artifact? Compare plane-fit (all contacts valid)
vs core-anchored (some contacts off-sheet). If events are local (n_part small) even at FULL
coverage (plane-fit), the failure is sampling, not coverage.
"""
import sys
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
RUN_DIR = "results/topic4_sef_hfo/field_swap_subject_snn"


def load(tag):
    return (json.load(open(os.path.join(RUN_DIR, f"readout_{tag}.json"))),
            np.load(os.path.join(RUN_DIR, f"figdata_{tag}.npz"), allow_pickle=True))


def contact_panel(ax, fd, title):
    C = np.asarray(fd["contacts"]); valid = np.asarray(fd["valid"]); reg = fd["reg"].item()
    L = reg["L"]
    ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, ec="0.6", ls=":"))
    ax.scatter(C[valid, 0], C[valid, 1], s=60, c="#1f77b4", edgecolors="k", label=f"valid ({valid.sum()})", zorder=3)
    ax.scatter(C[~valid, 0], C[~valid, 1], s=60, c="none", edgecolors="0.6", label=f"off-sheet/invalid ({(~valid).sum()})", zorder=3)
    ax.scatter(*reg["source_centroid"], marker="X", s=180, c="#d62728", edgecolors="k", zorder=5, label="source core")
    ax.scatter(*reg["sink_centroid"], marker="X", s=180, c="#1f77b4", edgecolors="k", zorder=5, label="sink core")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel("sheet x (mm)"); ax.set_ylabel("y (mm)")


def _stats(out):
    nps = [e["n_part"] for e in out["events"]]
    signs = [e.get("sign") for e in out["events"]]
    return (max(nps) if nps else 0,
            sum(1 for s in signs if s and s > 0), sum(1 for s in signs if s and s < 0), nps)


def npart_panel(ax, out, title):
    mx, fwd, rev, nps = _stats(out)
    ax.hist(nps, bins=range(0, 12), color="0.5", edgecolor="k", align="left")
    ax.axvline(6, color="r", ls="--", label="min for direction (2*k_dir, k=3->6 / k=2->4)")
    ax.set_xlabel("n participating contacts per event"); ax.set_ylabel("events")
    ax.set_title(f"{title}\nmax n_part={mx}, dir fwd/rev={fwd}/{rev}", fontsize=9)
    ax.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planefit-tag", default="e958_planefit_T3k")
    ap.add_argument("--anchored-tag", default="epilepsiae_958_source_anc12_m17.0_kd2_s1")
    ap.add_argument("--label", default="epilepsiae_958 (subdural grid)")
    a = ap.parse_args()
    po, pfd = load(a.planefit_tag); ao, afd = load(a.anchored_tag)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    contact_panel(axes[0, 0], pfd, f"plane-fit: all contacts valid ({np.asarray(pfd['valid']).sum()}/{len(pfd['valid'])})")
    npart_panel(axes[0, 1], po, "plane-fit events")
    contact_panel(axes[1, 0], afd, f"core-anchored: reduced coverage ({np.asarray(afd['valid']).sum()}/{len(afd['valid'])})")
    npart_panel(axes[1, 1], ao, "core-anchored events")
    fig.suptitle(f"NEGATIVE diagnostic: {a.label}\n"
                 f"events stay local (n_part 3-4) even at FULL plane-fit coverage -> no readable direction; "
                 f"sparse grid sampling, not just a coverage artifact", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(RUN_DIR, "figures", "negative_epilepsiae_958.png")
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    pm, pf, pr, _ = _stats(po); am, af_, ar, _ = _stats(ao)
    print(f"plane-fit: valid={np.asarray(pfd['valid']).sum()}/{len(pfd['valid'])} max_n_part={pm} dir={pf}/{pr}")
    print(f"anchored:  valid={np.asarray(afd['valid']).sum()}/{len(afd['valid'])} max_n_part={am} dir={af_}/{ar}")
    print(f"[written] {fp}")


if __name__ == "__main__":
    main()
