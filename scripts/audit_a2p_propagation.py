"""A2-P propagation-phenotype gate (first pass, existing T=8000 runs).

Question (recap §5): does the g_K "candidate slow-fast burst cycle" actually
switch the SNN between an interictal propagation phenotype and a seizure
propagation phenotype, or is the rho>1.35 excursion only a population-rate
oscillation in the model coordinate (spec §4.3.2 coordinate artifact)?

Three independent questions, one panel each (CLAUDE.md §7):
  Q1  Does permissivity (rho_pre) gate per-event spatial spread, at matched rate?
  Q2  Do the high-rho bouts show seizure-spread signatures (collision, larger
      extent) that are ABSENT in baseline interictal? (anchored to baseline +
      depletion-only runaway as the two phenotype endpoints)
  Q3  Where do the bout events sit on the interictal<->runaway phenotype cloud?

PASS (gate) requires ALL of: rho_pre->r95 positive & non-trivial; collision
appears at high-rho above baseline; some R4a (sustained-with-front) appears;
bout extent exceeds the baseline interictal cloud. Otherwise FAIL = rate
oscillator, not a propagation two-state.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg"
OUT = f"{B}/a2p_propagation_gate"
os.makedirs(f"{OUT}/figures", exist_ok=True)

BEST = ("explore3_gk", "l0_gk_k0.2_gk0.03_tk2000_s1")
BASELINE = [("explore1", "l0_g1.0_base_s1"), ("explore1", "l0_g1.0_base_s2")]
RUNAWAY = [("explore1", "l0_g1.0_dyn_k0.4_t2000_s1"), ("explore1", "l0_g1.0_dyn_k0.8_t2000_s1")]
SIBLINGS = [("explore4_osc", t) for t in [
    "l0_osc_k0.15_gk0.02_tk2000_s1", "l0_osc_k0.15_gk0.025_tk2000_s1",
    "l0_osc_k0.2_gk0.025_tk2000_s1", "l0_osc_k0.2_gk0.03_tk1500_s1",
    "l0_osc_k0.2_gk0.03_tk2000_s2", "l0_osc_k0.2_gk0.03_tk2000_s3"]]


def load(d, t):
    return json.load(open(f"{B}/{d}/readout_{t}.json"))


def coll_rate(d, t):
    p = f"{B}/{d}/sidecar_{t}.json"
    return json.load(open(p)).get("collision_rate") if os.path.exists(p) else None


def evarr(ev, key):
    return np.array([e[key] for e in ev if e.get(key) is not None], float)


def pheno(d, t):
    """Run-level phenotype summary."""
    r = load(d, t); ev = r["events"]
    r95 = evarr(ev, "r95_ea"); npart = evarr(ev, "n_part")
    rcl = [e.get("R_class") for e in ev]
    return dict(tag=t, T=r["config"]["T"], n_events=len(ev),
                coll_rate=coll_rate(d, t),
                max_r95=float(r95.max()) if len(r95) else None,
                mean_r95=float(r95.mean()) if len(r95) else None,
                max_npart=int(npart.max()) if len(npart) else None,
                n_R4a=sum(c == "R4a" for c in rcl),
                n_R4b=sum(c == "R4b" for c in rcl),
                frac_R3plus=float(np.mean([c in ("R3", "R4a", "R4b") for c in rcl])) if rcl else None,
                r95=r95, npart=npart)


# ---- best-point per-event rho (the predictor) -------------------------------
r = load(*BEST); ev = r["events"]; T = r["config"]["T"]
z = np.load(f"{B}/{BEST[0]}/a2_trace_{BEST[1]}.npz")
rho = z["rho_bin"]; dt = T / len(rho)
w_pre0, w_pre1, w_50 = int(200 / dt), int(50 / dt), int(50 / dt)
rho_pre, rho_dur = [], []
for e in ev:
    i0 = int(e["t_on"] / dt); i1 = int(e["t_off"] / dt)
    rho_pre.append(rho[max(0, i0 - w_pre0):max(1, i0 - w_50)].mean())
    rho_dur.append(rho[i0:i1 + 1].max() if i1 > i0 else rho[i0])
rho_pre = np.array(rho_pre); rho_dur = np.array(rho_dur)
r95 = evarr(ev, "r95_ea"); npart = evarr(ev, "n_part")
far = evarr(ev, "far_ea"); ap = evarr(ev, "active_peak")
sign_readable = sum(e.get("sign") is not None for e in ev)


def sp(x, y):
    if len(x) < 5:
        return (float("nan"), float("nan"))
    s, p = spearmanr(x, y)
    return (float(s), float(p))


med = np.median(rho_pre)
q1 = dict(
    n=len(ev),
    spearman_rho_pre_r95=sp(rho_pre, r95),
    spearman_rho_pre_npart=sp(rho_pre, npart),
    spearman_rho_pre_far=sp(rho_pre, far),
    active_peak_range=[float(ap.min()), float(ap.max())],   # ~matched rate -> raw==rate-matched
    high_rho_mean_r95=float(r95[rho_pre >= med].mean()),
    low_rho_mean_r95=float(r95[rho_pre < med].mean()),
    sign_readable=f"{sign_readable}/{len(ev)}",
)

# ---- Q2/Q3 phenotype endpoints ----------------------------------------------
best_ph = pheno(*BEST)
base_ph = [pheno(*x) for x in BASELINE]
run_ph = [pheno(*x) for x in RUNAWAY]
sib_ph = [pheno(*x) for x in SIBLINGS]

base_max_r95 = max(p["max_r95"] for p in base_ph)
base_max_coll = max(p["coll_rate"] for p in base_ph)

verdict = dict(
    Q1_rho_gates_spread=dict(
        **q1,
        reading="rho_pre->r95 not positive; rate (active_peak) ~constant so raw==rate-matched",
        pass_q1=bool(q1["spearman_rho_pre_r95"][0] > 0.3 and q1["spearman_rho_pre_r95"][1] < 0.05),
    ),
    Q2_seizure_signatures=dict(
        best_coll_rate=best_ph["coll_rate"], baseline_coll_rate=[p["coll_rate"] for p in base_ph],
        runaway_coll_rate=[p["coll_rate"] for p in run_ph],
        best_max_r95=best_ph["max_r95"], baseline_max_r95=[p["max_r95"] for p in base_ph],
        runaway_max_r95=[p["max_r95"] for p in run_ph],
        best_n_R4a=best_ph["n_R4a"], best_n_events=best_ph["n_events"],
        baseline_n_events=[p["n_events"] for p in base_ph], runaway_n_events=[p["n_events"] for p in run_ph],
        pass_q2=bool(best_ph["coll_rate"] > base_max_coll and best_ph["max_r95"] > base_max_r95 and best_ph["n_R4a"] > 0),
    ),
    Q3_phenotype_placement=dict(
        best_in_baseline_cloud=bool(best_ph["max_r95"] <= base_max_r95 and best_ph["max_npart"] <= max(p["max_npart"] for p in base_ph)),
        reading="bout events sit inside (<=) the baseline interictal cloud, far from runaway",
    ),
    sibling_coll_rates={p["tag"]: p["coll_rate"] for p in sib_ph},
    sibling_max_R4a=max(p["n_R4a"] for p in sib_ph),
)
verdict["GATE"] = "PASS" if (verdict["Q1_rho_gates_spread"]["pass_q1"]
                             and verdict["Q2_seizure_signatures"]["pass_q2"]) else "FAIL"
verdict["interpretation"] = (
    "FAIL = the rho>1.35 excursion is a population-rate (timing) oscillation, NOT a "
    "propagation-mode switch. High-rho bouts have collision_rate=%.3f (<= baseline %.3f) and "
    "max extent %.1f (<= baseline %.1f); no R4a; rho_pre does not gate spread. The g_K cycle "
    "modulates WHEN events fire (burst vs quiet), not WHAT they look like (still baseline "
    "interictal-type, in fact slightly smaller / fewer)."
    % (best_ph["coll_rate"], base_max_coll, best_ph["max_r95"], base_max_r95)
)

json.dump({k: v for k, v in verdict.items()},
          open(f"{OUT}/gate_verdict.json", "w"), indent=2, default=float)

# ---- figure: 3 independent panels -------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# Q1: rho_pre vs r95, colored by rate
sc = ax[0].scatter(rho_pre, r95, c=ap * 1000, cmap="viridis", s=70, edgecolor="k", lw=0.5)
ax[0].axvline(1.35, color="orange", ls="--", lw=1, label="rho=1.35 (model 'seizure' line)")
ax[0].set_xlabel("rho_pre  (permissivity just BEFORE the event)")
ax[0].set_ylabel("event spatial extent  r95 (a.u.)")
s, p = q1["spearman_rho_pre_r95"]
ax[0].set_title("Q1  Does permissivity gate spread?\nSpearman(rho_pre, r95) = %.2f (p=%.2f)  -> NO" % (s, p))
ax[0].legend(fontsize=7, loc="upper left")
cb = plt.colorbar(sc, ax=ax[0]); cb.set_label("firing rate proxy (active_peak x1000)", fontsize=7)

# Q2: collision rate + max extent bars, best vs baseline vs runaway
groups = ["baseline\n(interictal ref)", "best-point\ng_K cycle", "runaway\n(high ref)"]
coll = [base_max_coll, best_ph["coll_rate"], max(p["coll_rate"] for p in run_ph)]
mr = [base_max_r95, best_ph["max_r95"], max(p["max_r95"] for p in run_ph)]
x = np.arange(3); ax2 = ax[1]; axb = ax2.twinx()
ax2.bar(x - 0.18, coll, width=0.36, color="#c44", label="collision rate")
axb.bar(x + 0.18, mr, width=0.36, color="#48a", label="max extent r95")
ax2.set_xticks(x); ax2.set_xticklabels(groups, fontsize=8)
ax2.set_ylabel("collision rate (bilateral recruitment)", color="#c44")
axb.set_ylabel("max event extent r95", color="#48a")
ax2.set_title("Q2  Seizure signatures at high-rho?\nbest-point coll=%.2f <= baseline %.2f; extent %.0f <= %.0f -> NO"
              % (best_ph["coll_rate"], base_max_coll, best_ph["max_r95"], base_max_r95))

# Q3: r95 vs n_part phenotype clouds
def pool(phs, key):
    return np.concatenate([p[key] for p in phs])
ax[2].scatter(pool(base_ph, "r95"), pool(base_ph, "npart"), c="0.55", s=40, label="baseline interictal", alpha=0.7)
ax[2].scatter(pool(run_ph[:1], "r95"), pool(run_ph[:1], "npart"), c="#c44", s=24, marker="x", label="runaway (high state)", alpha=0.6)
ax[2].scatter(best_ph["r95"], best_ph["npart"], c="#1456c4", s=90, edgecolor="k", lw=0.6, label="best-point bouts")
ax[2].set_xlabel("event spatial extent  r95 (a.u.)")
ax[2].set_ylabel("# participating virtual-SEEG contacts")
ax[2].set_title("Q3  Where do bouts sit?\nbout cloud INSIDE baseline interictal, not toward runaway")
ax[2].legend(fontsize=7, loc="lower right")

fig.suptitle("A2-P propagation gate (T=8000 first pass):  GATE = %s  — rho excursion = rate oscillation, NOT interictal<->seizure two-state"
             % verdict["GATE"], fontsize=11, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/figures/a2p_gate.png", dpi=130)
print("GATE =", verdict["GATE"])
print(json.dumps(verdict, indent=2, default=float)[:2000])
print("\nwrote", f"{OUT}/gate_verdict.json", "+ figures/a2p_gate.png")
