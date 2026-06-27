"""M3A-A2 exploration summary + figures. Reads readout_*.json + a2_trace_*.npz from an explore dir;
prints a per-anchor k_use->regime table + frozen-q boundary calibration, and writes 4 figures
(one question each, §7). Usage: python scripts/plot_a2_pilot.py <explore_dir>"""
import os, sys, json, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = sys.argv[1]
FIG = os.path.join(DIR, "figures"); os.makedirs(FIG, exist_ok=True)
_RORDER = ["R0", "R1", "R2", "R3", "R4a", "R4b"]
ANCHOR_LGR = {"l0_g1.0": 1.00, "l1_g1.3": 1.04, "l2_g1.6": 1.16}


def _load():
    runs = []
    for f in sorted(glob.glob(os.path.join(DIR, "readout_*.json"))):
        r = json.load(open(f)); tag = os.path.basename(f)[8:-5]
        a2 = r.get("a2", {}); act = r.get("activity", {})
        anchor = tag.split("_")[0] + "_" + tag.split("_")[1]
        kind = "base" if "_base_" in tag else ("frozen" if "_frzq" in tag else "dyn")
        k_use = a2.get("k_use"); qfrz = None
        mk = re.search(r"_dyn_k([0-9.]+)_", tag); mf = re.search(r"_frzq([0-9.]+)_", tag)
        if mf: qfrz = float(mf.group(1))
        rc = [e.get("R_class", "R0") for e in r.get("events", [])]
        maxR = max(rc, key=lambda x: _RORDER.index(x)) if rc else "R0"
        runs.append(dict(tag=tag, anchor=anchor, kind=kind, k_use=(float(mk.group(1)) if mk else k_use),
                         qfrz=qfrz, lgr=a2.get("rho_static"), rho_peak=a2.get("rho_peak"),
                         rho_p95=a2.get("rho_p95"), q_core_min=a2.get("q_core_min"),
                         q_core_end=a2.get("q_core_end"), a_core_mean=a2.get("a_core_mean"),
                         tail=act.get("tail_to_baseline_ratio"), gr=act.get("global_E_rate_mean_hz"),
                         coreR=act.get("core_E_rate_mean_hz"), collis=act.get("collision_rate_returned_sidecar"),
                         r95=act.get("r95_mm"), n_events=r.get("n_events"), maxR=maxR,
                         II_IE=a2.get("I_I_over_I_E_core"), npz=f.replace("readout_", "a2_trace_").replace(".json", ".npz")))
    return runs


def main():
    runs = _load()
    by_anchor = {}
    for r in runs:
        by_anchor.setdefault(r["anchor"], []).append(r)

    print("\n=== Task-0 baselines (k_use=0): clean interictal + a_bar ===")
    for anc in sorted(by_anchor):
        bs = [r for r in by_anchor[anc] if r["kind"] == "base"]
        if bs:
            ab = np.nanmedian([r["a_core_mean"] for r in bs]); tl = np.nanmedian([r["tail"] for r in bs])
            ne = np.nanmedian([r["n_events"] for r in bs]); ii = np.nanmedian([r["II_IE"] for r in bs])
            print(f"  {anc} lgr={ANCHOR_LGR.get(anc)}: a_bar={ab:.5f} tail={tl:.2f} evt={ne:.0f} "
                  f"I_I/I_E_core={ii:.2f} clean={(tl<=1.5 and ne>0)}")

    print("\n=== Task-0b frozen-q: does the A1b 1.35/1.86 boundary hold? (phenotype vs frozen rho) ===")
    for anc in sorted(by_anchor):
        fz = sorted([r for r in by_anchor[anc] if r["kind"] == "frozen"], key=lambda r: r["qfrz"] or 1)
        for r in fz:
            rho = ANCHOR_LGR.get(anc, 1) / r["qfrz"]
            print(f"  {anc} q={r['qfrz']} -> rho={rho:.2f}: collis={r['collis']} r95={r['r95']} "
                  f"coreR={r['coreR']:.0f} tail={r['tail']:.2f} maxR={r['maxR']} (A1b: rho1.35=seizure, 1.86=runaway)")

    print("\n=== Dynamic core_only k_use sweep: rho excursion + return + phenotype ===")
    for anc in sorted(by_anchor):
        dy = sorted([r for r in by_anchor[anc] if r["kind"] == "dyn"], key=lambda r: r["k_use"] or 0)
        for r in dy:
            print(f"  {anc} k={r['k_use']}: q_core_min={r['q_core_min']} rho_peak={r['rho_peak']} "
                  f"rho_p95={r['rho_p95']} tail={r['tail']:.2f} coreR={r['coreR']:.0f} evt={r['n_events']} maxR={r['maxR']}")

    # ---- Fig 1: k_use -> rho_peak / q_core_min response per anchor (the regime transition) ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for anc in sorted(by_anchor):
        dy = sorted([r for r in by_anchor[anc] if r["kind"] == "dyn" and r["k_use"]], key=lambda r: r["k_use"])
        if not dy:
            continue
        ks = [r["k_use"] for r in dy]
        ax[0].plot(ks, [r["rho_peak"] for r in dy], "o-", label=anc)
        ax[1].plot(ks, [r["q_core_min"] for r in dy], "o-", label=anc)
    ax[0].axhline(1.35, ls="--", c="orange", label="A1b seizure 1.35"); ax[0].axhline(1.86, ls="--", c="red", label="runaway 1.86")
    ax[0].set_xlabel("k_use"); ax[0].set_ylabel("rho_peak"); ax[0].set_xscale("log"); ax[0].set_title("dynamic rho excursion vs depletion rate"); ax[0].legend(fontsize=7)
    ax[1].axhline(0.25, ls=":", c="gray", label="q_min floor"); ax[1].set_xlabel("k_use"); ax[1].set_ylabel("q_core_min"); ax[1].set_xscale("log"); ax[1].set_title("core tank minimum vs depletion rate"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "kuse_response.png"), dpi=110); plt.close(fig)

    # ---- Fig 2: representative dynamic trace (rho + q_core + rate) for each anchor's mid k_use ----
    fig, axs = plt.subplots(len(by_anchor), 1, figsize=(10, 2.6 * len(by_anchor)), squeeze=False)
    for i, anc in enumerate(sorted(by_anchor)):
        dy = [r for r in by_anchor[anc] if r["kind"] == "dyn" and r["k_use"]]
        if not dy:
            continue
        r = sorted(dy, key=lambda r: abs((r["rho_peak"] or 0) - 1.5))[0]   # the one nearest a real excursion
        if not os.path.exists(r["npz"]):
            continue
        z = np.load(r["npz"]); t = np.arange(len(z["rho_bin"])) * 0.1 / 1000.0  # s (dt=0.1ms)
        ax = axs[i][0]; ax.plot(t, z["rho_bin"], "purple", lw=0.8, label="rho(t)")
        ax.axhline(1.35, ls="--", c="orange", lw=0.7); ax.axhline(1.86, ls="--", c="red", lw=0.7)
        ax2 = ax.twinx(); ax2.plot(t, z["q_core_bin"], "teal", lw=0.8, label="q_core"); ax2.set_ylim(0, 1.05)
        ax.set_title(f"{anc} k_use={r['k_use']} (rho purple / q_core teal)", fontsize=9)
        ax.set_ylabel("rho"); ax2.set_ylabel("q_core"); ax.set_xlabel("s")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "rep_trace.png"), dpi=110); plt.close(fig)

    # ---- Fig 3: R-class (maxR) vs k_use per anchor ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ymap = {r: i for i, r in enumerate(_RORDER)}
    for anc in sorted(by_anchor):
        dy = sorted([r for r in by_anchor[anc] if r["kind"] == "dyn" and r["k_use"]], key=lambda r: r["k_use"])
        if dy:
            ax.plot([r["k_use"] for r in dy], [ymap[r["maxR"]] for r in dy], "s-", label=anc)
    ax.set_yticks(range(len(_RORDER))); ax.set_yticklabels(_RORDER); ax.set_xscale("log")
    ax.set_xlabel("k_use"); ax.set_title("max event R-class vs depletion rate"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "rclass_vs_kuse.png"), dpi=110); plt.close(fig)

    # ---- Fig 4: frozen-q boundary calibration (collision/r95 vs frozen rho) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for anc in sorted(by_anchor):
        allr = [r for r in by_anchor[anc] if r["kind"] in ("frozen", "base")]
        pts = []
        for r in allr:
            rho = ANCHOR_LGR.get(anc, 1) / (r["qfrz"] if r["qfrz"] else 1.0)
            pts.append((rho, r["collis"] if r["collis"] is not None else 0))
        pts = sorted(pts)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=anc)
    ax.axvline(1.35, ls="--", c="orange", label="A1b seizure 1.35"); ax.axvline(1.86, ls="--", c="red", label="runaway 1.86")
    ax.set_xlabel("frozen rho = lgr / q"); ax.set_ylabel("two-core collision rate"); ax.set_title("Task-0b: does the A1b boundary transfer to A2?"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "frozen_boundary.png"), dpi=110); plt.close(fig)
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
