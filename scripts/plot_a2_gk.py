"""M3A-A2 g_K figures (P1-4 fix): (1) honestly-titled burst-cycle candidate trace; (2) frozen-q ACTIVITY
boundary (rho vs core rate — the real evidence, NOT collision which is all-zero in the readout activity);
(3) g_K window map (k_use x gk_max -> regime). Run: python scripts/plot_a2_gk.py"""
import glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); from src.sef_hfo_a2 import detect_bouts
ROOT = "results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg"
ANCHOR_LGR = {"l0_g1.0": 1.00, "l1_g1.3": 1.04, "l2_g1.6": 1.16}


# ---- (1) headline trace, honest title ----
def trace_fig():
    d = ROOT + "/explore3_gk"; FIG = d + "/figures"; os.makedirs(FIG, exist_ok=True)
    z = np.load(d + "/a2_trace_l0_gk_k0.2_gk0.03_tk2000_s1.npz")
    rho, qc, gk, rate = z["rho_bin"], z["q_core_bin"], z["gk_bin"], z["rate_E_hz"]
    t = np.arange(len(rho)) * 0.1 / 1000.0
    fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, rho, "purple", lw=0.7); ax[0].axhline(1.35, ls="--", c="orange", label="rho=1.35 (model coord)")
    ax[0].axhline(1.86, ls="--", c="red", lw=0.7, label="rho=1.86")
    ax[0].fill_between(t, 1.35, 1.86, where=(rho >= 1.35), color="orange", alpha=0.13)
    ax[0].set_ylabel("rho(t)\n(model coord)"); ax[0].set_ylim(0.9, 2.0); ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("M3A-A2 g_K: CANDIDATE slow-fast population-burst cycle with rho excursions\n"
                    "(l0_g1.0, depletion k=0.2 + g_K sAHP gk=0.03 tau_k=2000)  —  PROPAGATION PHENOTYPE PENDING:\n"
                    "rho-cross + rate-up is NOT verified interictal<->seizure (events still R3 not R4a; no virtual-SEEG / rate-matched gate)",
                    fontsize=9)
    ax[1].plot(t, qc, "teal", lw=0.7); ax[1].set_ylabel("q_core\n(inhib tank)"); ax[1].set_ylim(0.6, 1.02)
    ax[2].plot(t, gk, "firebrick", lw=0.7); ax[2].set_ylabel("g_K\n(sAHP brake)")
    ax[3].plot(t, rate, "k", lw=0.5); ax[3].set_ylabel("E rate (Hz)"); ax[3].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(FIG + "/gk_two_state_trace.png", dpi=120); plt.close(fig)
    print("trace ->", FIG + "/gk_two_state_trace.png")


# ---- (2) frozen-q ACTIVITY boundary (replaces broken collision figure) ----
def frozen_fig():
    d = ROOT + "/explore1"; FIG = d + "/figures"
    pts = {}
    for f in glob.glob(d + "/readout_*frzq*.json") + glob.glob(d + "/readout_*base*.json"):
        r = json.load(open(f)); tag = os.path.basename(f)[8:-5]
        anc = tag.split("_")[0] + "_" + tag.split("_")[1]
        mq = re.search(r"frzq([0-9.]+)", tag); q = float(mq.group(1)) if mq else 1.0
        rho = ANCHOR_LGR.get(anc, 1) / q
        pts.setdefault(anc, []).append((rho, r["activity"]["core_E_rate_mean_hz"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for anc in sorted(pts):
        p = sorted(pts[anc]); ax.plot([x[0] for x in p], [x[1] for x in p], "o-", label=anc)
    ax.axvline(1.35, ls="--", c="orange", label="A1b seizure 1.35"); ax.axvline(1.86, ls="--", c="red", label="A1b runaway 1.86")
    ax.set_yscale("log"); ax.set_xlabel("frozen rho = lgr / q_core"); ax.set_ylabel("core E rate (Hz, log)")
    ax.set_title("Task-0b frozen-q boundary: core activity vs frozen rho\n(real evidence; the collision-based figure was empty)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG + "/frozen_q_activity.png", dpi=120); plt.close(fig)
    old = FIG + "/frozen_boundary.png"
    if os.path.exists(old): os.remove(old)
    print("frozen ->", FIG + "/frozen_q_activity.png", "(removed broken frozen_boundary.png)")


# ---- (3) g_K window map: k_use x gk_max -> regime (tau_k=2000 slice) ----
def window_fig():
    FIG = ROOT + "/explore3_gk/figures"
    rows = []
    for d in ("explore3_gk", "explore4_osc"):
        for f in glob.glob(ROOT + "/" + d + "/readout_*.json"):
            tag = os.path.basename(f)[8:-5]
            if "tk2000" not in tag and "_gk" not in tag:
                continue
            r = json.load(open(f)); a2 = r["a2"]
            mtk = re.search(r"tk(\d+)", tag); tk = int(mtk.group(1)) if mtk else 5000
            if tk != 2000:
                continue
            npz = f.replace("readout_", "a2_trace_").replace(".json", ".npz")
            if not os.path.exists(npz):
                continue
            rho = np.load(npz)["rho_bin"]; nb = len(detect_bouts(rho, 1.35))
            inter = float((rho < 1.35).mean()); band = float(((rho >= 1.35) & (rho < 1.86)).mean()); run = float((rho >= 1.86).mean())
            if nb >= 3 and inter > 0.3 and band > 0.05 and run < 0.3:
                reg = "candidate cycle"
            elif run > 0.4:
                reg = "runaway"
            elif band < 0.02:
                reg = "quiet"
            else:
                reg = "marginal"
            rows.append((a2["k_use"], a2["gk_max"], reg))
    cmap = {"candidate cycle": "green", "runaway": "red", "quiet": "gray", "marginal": "orange"}
    fig, ax = plt.subplots(figsize=(7, 5))
    seen = set()
    for k, gk, reg in rows:
        ax.scatter(k, gk, c=cmap[reg], s=140, edgecolor="k",
                   label=reg if reg not in seen else None); seen.add(reg)
    ax.set_xlabel("k_use (depletion strength)"); ax.set_ylabel("gk_max (recovery strength)")
    ax.set_title("g_K window (tau_k=2000): regime vs (depletion, recovery)\n"
                 "candidate cycle only in a narrow knife-edge; seed-fragile (same point -> 3 outcomes)")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout(); fig.savefig(FIG + "/gk_window.png", dpi=120); plt.close(fig)
    print("window ->", FIG + "/gk_window.png", f"({len(rows)} pts)")


if __name__ == "__main__":
    trace_fig(); frozen_fig(); window_fig()
