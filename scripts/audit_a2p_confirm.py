"""A2-P propagation gate — CONFIRMATION on the full source-space instrument (batch5, T=20000).

Supersedes the T=8000 first pass (audit_a2p_propagation.py) for the spatial questions: the
source-space readout (r95_mm spread, reach_axis_mm, n_fired_E recruitment, centroid_x) +
4-16x more events (n=113/260/71 vs 17) overturn TWO first-pass claims and refine the verdict.

RETRACTED (small-n / coarse-grid artifacts of the first pass):
  - "best-point events spatially identical to / smaller than baseline" -> FALSE. Source-space
    median spread is ~12 mm (best) vs ~5.6 mm (baseline), at the RUNAWAY's scale (~12.3).
  - "pure rate/timing oscillation, no spatial change" -> FALSE. Permissivity gates source-space
    spread rate-independently (partial Spearman rho->r95_mm | n_fired_E = +0.3..+0.7).

CONFIRMED & REFINED verdict (three independent questions, §7):
  Q1 Does the slow variable gate event spatial EXTENT?  YES — high-permissivity bouts spread
     far along the inter-core axis (reach 22.8 vs baseline 7.9), centered between the cores
     (71% mid vs baseline 11%); permissivity gates reach rate-independently.
  Q2 Is the large state a traveling SEIZURE or a synchronous BURST?  SYNCHRONOUS BURST —
     direction-readability ~0.12 for best AND baseline AND runaway (none are clean waves);
     collision 0.0 (best) vs 0.12 (baseline) vs 0.5 (runaway); no R4a. The substrate makes
     synchronous bursts, not traveling waves (cf. axial-intervention pilot STOP).
  Q3 Where do bouts sit?  Big like runaway in EXTENT, unreadable like baseline in DIRECTION.

NET: the slow variable produces a real small-local <-> big-SYNCHRONOUS extent two-state, but
NOT an interictal-propagation <-> seizure-propagation two-state. The bottleneck is the
substrate (no spontaneous traveling waves at L=20), not just the slow variable.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr, rankdata, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/explore5_fullfield"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

BEST = ["l0_gk_best_s1", "l0_gk_best_s2", "l0_gk_best_s3"]
BASE = "l0_base_s1"
RUN = "l0_runaway_k0.4_s1"
PROBE = ["l1_gk_best_s1", "l2_gk_best_s1"]


def rd(t): return json.load(open(f"{OUT}/readout_{t}.json"))
def ff(t): return json.load(open(f"{OUT}/fullfield_{t}.json"))["events"]
def ffa(t, k): return np.array([e[k] for e in ff(t) if e.get(k) is not None], float)
def coll(t):
    p = f"{OUT}/sidecar_{t}.json"
    return json.load(open(p)).get("collision_rate") if os.path.exists(p) else None
def sign_readable_frac(t): return float(np.mean([e.get("sign") is not None for e in rd(t)["events"]]))
def rho_pre(t):
    r = rd(t); T = r["config"]["T"]
    z = np.load(f"{OUT}/a2_trace_{t}.npz"); rho = z["rho_bin"]; dt = T / len(rho)
    return np.array([rho[max(0, int(e["t_on"] / dt) - int(200 / dt)):max(1, int(e["t_on"] / dt) - int(50 / dt))].mean() for e in ff(t)])


def partial_spearman(x, y, z):
    """Spearman partial corr of x,y controlling z (rank residuals)."""
    xr, yr, zr = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]; coef, *_ = np.linalg.lstsq(b1, a, rcond=None); return a - b1 @ coef
    r, p = pearsonr(resid(xr, zr), resid(yr, zr))
    return float(r), float(p)


# ---- Q1: permissivity gates spatial extent (rate-controlled) ----------------
q1 = []
for t in BEST:
    rp = rho_pre(t); r95 = ffa(t, "r95_mm"); nfe = ffa(t, "n_fired_E")
    s = spearmanr(rp, r95); ps = partial_spearman(rp, r95, nfe)
    q1.append(dict(tag=t, n=len(rp), rho_vs_r95mm=[float(s[0]), float(s[1])],
                   rho_vs_r95mm_rate_controlled=ps, median_r95mm=float(np.median(r95)),
                   median_reach_axis=float(np.median(ffa(t, "reach_axis_mm")))))

# ---- Q2: traveling wave vs synchronous burst --------------------------------
q2 = dict(
    direction_readable_frac=dict(baseline=sign_readable_frac(BASE),
                                 best=[sign_readable_frac(t) for t in BEST],
                                 runaway=sign_readable_frac(RUN)),
    collision_rate=dict(baseline=coll(BASE), best=[coll(t) for t in BEST], runaway=coll(RUN)),
    n_R4a_best=sum(e.get("R_class") == "R4a" for t in BEST for e in rd(t)["events"]),
    reading="best/baseline/runaway all ~0.12 direction-readable => none are traveling waves; "
            "the big g_K bouts are global synchronous bursts (collision 0, no R4a)",
)

# ---- Q3: extent vs recruitment placement ------------------------------------
q3 = dict(median_r95mm=dict(baseline=float(np.median(ffa(BASE, "r95_mm"))),
                            best=[float(np.median(ffa(t, "r95_mm"))) for t in BEST],
                            runaway=float(np.median(ffa(RUN, "r95_mm")))),
          median_reach_axis=dict(baseline=float(np.median(ffa(BASE, "reach_axis_mm"))),
                                 best=[float(np.median(ffa(t, "reach_axis_mm"))) for t in BEST],
                                 runaway=float(np.median(ffa(RUN, "reach_axis_mm")))),
          frac_centroid_mid=dict(baseline=float(np.mean((ffa(BASE, "centroid_x") > 7.5) & (ffa(BASE, "centroid_x") < 12.5))),
                                 best=float(np.mean((ffa(BEST[0], "centroid_x") > 7.5) & (ffa(BEST[0], "centroid_x") < 12.5))),
                                 runaway=float(np.mean((ffa(RUN, "centroid_x") > 7.5) & (ffa(RUN, "centroid_x") < 12.5)))))

probe = {t: dict(n=len(rd(t)["events"]), coll=coll(t),
                 median_r95mm=float(np.median(ffa(t, "r95_mm"))),
                 direction_readable_frac=sign_readable_frac(t)) for t in PROBE}

verdict = dict(
    Q1_permissivity_gates_extent=q1,
    Q2_wave_or_synchronous_burst=q2,
    Q3_placement=q3,
    substrate_probe=probe,
    GATE_propagation_two_state="FAIL",
    retracted_first_pass_claims=[
        "best events spatially identical to / smaller than baseline (FALSE: ~12mm vs 5.6mm, runaway-scale)",
        "pure rate/timing oscillation with no spatial change (FALSE: rho gates r95_mm rate-independently +0.3..+0.7)",
    ],
    interpretation=(
        "Refined FAIL. The slow variable DOES gate a real, rate-independent spatial-extent "
        "two-state: low-permissivity small-local events (median spread ~5.6mm, on the cores) "
        "<-> high-permissivity large bouts (~12mm, axis-spanning, middle-centered, runaway-scale). "
        "BUT the large state is a GLOBAL SYNCHRONOUS BURST, not a traveling seizure: direction is "
        "unreadable (~0.12, same as baseline AND runaway), collision is 0 (< baseline 0.12 << "
        "runaway 0.5), no R4a. The substrate at L=20 produces synchronous bursts not traveling "
        "waves (cf. axial-intervention pilot STOP), so an interictal-PROPAGATION <-> seizure-"
        "PROPAGATION two-state is not expressible here regardless of the slow variable. More "
        "excitable anchors don't help (l1 stays small/interictal, l2 silenced by the brake)."),
)
json.dump(verdict, open(f"{OUT}/gate_confirm_verdict.json", "w"), indent=2, default=float)

# ---- figure -----------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

# Q1 pooled within-seed z scatter
xs, ys = [], []
for t in BEST:
    rp = rho_pre(t); v = ffa(t, "r95_mm")
    xs.append((rp - rp.mean()) / rp.std()); ys.append((v - v.mean()) / v.std())
xs = np.concatenate(xs); ys = np.concatenate(ys); pr = spearmanr(xs, ys)
ax[0].scatter(xs, ys, s=14, alpha=0.5, c="#1456c4")
ax[0].axhline(0, color="0.7", lw=.6); ax[0].axvline(0, color="0.7", lw=.6)
ax[0].set_xlabel("permissivity rho_pre  (z within seed)")
ax[0].set_ylabel("source-space spread r95_mm  (z within seed)")
ax[0].set_title("Q1  Permissivity gates EXTENT?\npooled n=%d Spearman=%.2f -> YES (rate-controlled +0.3..+0.7)" % (len(xs), pr[0]))

# Q2 readability + collision: synchronous burst not wave
g = ["baseline", "best g_K\n(3 seeds)", "runaway"]
rdb = [q2["direction_readable_frac"]["baseline"], np.mean(q2["direction_readable_frac"]["best"]), q2["direction_readable_frac"]["runaway"]]
cl = [q2["collision_rate"]["baseline"], np.mean(q2["collision_rate"]["best"]), q2["collision_rate"]["runaway"]]
x = np.arange(3)
ax[1].bar(x - 0.18, rdb, width=0.36, color="#7a3", label="direction-readable frac")
ax[1].bar(x + 0.18, cl, width=0.36, color="#c44", label="collision rate")
ax[1].set_xticks(x); ax[1].set_xticklabels(g, fontsize=8); ax[1].set_ylim(0, 0.6)
ax[1].legend(fontsize=7, loc="upper left")
ax[1].set_title("Q2  Traveling wave or synchronous burst?\nall direction-unreadable ~0.12; best collision 0 -> SYNCHRONOUS BURST")

# Q3 extent vs recruitment clouds
ax[2].scatter(ffa(BASE, "r95_mm"), ffa(BASE, "reach_axis_mm"), c="0.55", s=18, alpha=.6, label="baseline interictal")
ax[2].scatter(ffa(RUN, "r95_mm"), ffa(RUN, "reach_axis_mm"), c="#c44", s=9, marker="x", alpha=.4, label="runaway")
for t in BEST:
    ax[2].scatter(ffa(t, "r95_mm"), ffa(t, "reach_axis_mm"), c="#1456c4", s=13, alpha=.5)
ax[2].scatter([], [], c="#1456c4", s=30, label="best-point bouts")
ax[2].set_xlabel("source-space spread r95 (mm)")
ax[2].set_ylabel("reach along inter-core axis (mm)")
ax[2].set_title("Q3  Where do bouts sit?\nbig like runaway in EXTENT, unreadable like baseline in DIRECTION")
ax[2].legend(fontsize=7, loc="lower right")

fig.suptitle("A2-P gate CONFIRM (T=20000, source-space): slow var gates EXTENT (small<->big-synchronous) but NOT propagation — big state = synchronous burst, not traveling seizure. GATE=FAIL",
             fontsize=9.8, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIG}/a2p_gate_confirm.png", dpi=130)
print("GATE (propagation two-state) =", verdict["GATE_propagation_two_state"])
print(json.dumps(verdict, indent=2, default=float)[-1500:])
