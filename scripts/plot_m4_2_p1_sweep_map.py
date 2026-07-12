"""M4-2 P1 sweep figure: (A) termination-class map over (ee_std_u x ee_std_tau) at the aG16 bounded
op-point (seed=1, coarse+low-u), (B) example activity envelopes for persist / fragment / suppress.

The claim: adding E->E short-term depression (STD) to the M4 bounded persistent state goes
persist -> fragment -> suppress with NO clean single-event termination (terminate_clean absent) ->
STD alone cannot terminate the attractor into a re-triggerable interictal state. Seeds 3 and 4 reproduce.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "topic4_m4_dynamic_p1_sweep", "figures")
os.makedirs(OUT, exist_ok=True)

# categorical, colorblind-safe
CLASS_COLOR = {
    "persist": "#4C6EDB",          # blue  - bounded, never terminates
    "fragment": "#E8833A",         # orange- burst train
    "suppress": "#BFC4CC",         # gray  - killed
    "terminate_clean": "#2E9E5B",  # green - the TARGET (absent here)
    "rebound": "#C44E9D", "fade": "#8A6D3B", "runaway": "#C0392B", "ERROR": "#000000",
}


def _load(dirn):
    p = os.path.join(RES, dirn, "p1_sweep_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _cls(rows, u, tau):
    for r in rows:
        if abs(r["ee_std_u"] - u) < 1e-9 and abs(r["ee_std_tau_ms"] - tau) < 1e-9:
            return r.get("termination_class")
    return None


# assemble seed=1 full grid (Arm0 u=0 + low-u + coarse)
coarse = _load("topic4_m4_dynamic_p1_sweep")["rows"]
lowu = _load("topic4_m4_dynamic_p1_sweep_lowu")["rows"]
rows1 = coarse + [r for r in lowu if r["ee_std_u"] > 0]
us = [0.0, 0.05, 0.08, 0.11, 0.15, 0.30, 0.50]
taus = [1000.0, 2500.0, 5000.0]
grid = np.full((len(us), len(taus)), -1, int)
classes_seen = []
for i, u in enumerate(us):
    for j, tau in enumerate(taus):
        c = "persist" if u == 0.0 else _cls(rows1, u, tau)   # Arm0 has one tau slot; show across the row
        if u == 0.0:
            c = _cls(rows1, 0.0, taus[0]) or "persist"
        if c is None:
            continue
        if c not in classes_seen:
            classes_seen.append(c)
        grid[i, j] = list(CLASS_COLOR).index(c)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.05, 1.25]))

# ---- Panel A: map ----
palette = list(CLASS_COLOR.values())
cmap = matplotlib.colors.ListedColormap(palette)
disp = np.ma.masked_less(grid, 0)
axA.imshow(disp, cmap=cmap, vmin=0, vmax=len(palette) - 1, aspect="auto", origin="lower")
axA.set_xticks(range(len(taus))); axA.set_xticklabels([f"{int(t)}" for t in taus])
axA.set_yticks(range(len(us))); axA.set_yticklabels([("0 (no STD)" if u == 0 else f"{u:g}") for u in us])
axA.set_xlabel("STD recovery  ee_std_tau_ms  (ms)")
axA.set_ylabel("STD depletion  ee_std_u")
axA.set_title("A  termination class over (u, tau)  @ k_q=0.10, alpha_G=16")
for i, u in enumerate(us):
    for j, tau in enumerate(taus):
        if grid[i, j] >= 0:
            nm = list(CLASS_COLOR)[grid[i, j]]
            axA.text(j, i, nm[:4], ha="center", va="center", fontsize=7.5,
                     color="white" if nm in ("persist", "fragment") else "#222")
present = [c for c in CLASS_COLOR if c in classes_seen] + ["terminate_clean"]
axA.legend(handles=[Patch(facecolor=CLASS_COLOR[c], edgecolor="#555",
                          label=c + (" (TARGET — absent)" if c == "terminate_clean" else ""))
                    for c in dict.fromkeys(present)],
           loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)

# ---- Panel B: example envelopes ----
z = np.load(os.path.join(RES, "topic4_m4_dynamic_p1_sweep", "p1_sweep_traces.npz"))
zl = np.load(os.path.join(RES, "topic4_m4_dynamic_p1_sweep_lowu", "p1_sweep_traces.npz"))


def env(store, lbl, k=200):
    a = np.asarray(store[f"{lbl}__af"], float)
    n = (len(a) // k) * k
    return np.arange(n // k) * k / 1000.0, a[:n].reshape(-1, k).mean(1)  # 200ms envelope, x in s


for lbl, store, name, col in [
    ("p1_arm0", z, "persist  (u=0)", CLASS_COLOR["persist"]),
    ("p1_u0.15_tau1000", z, "fragment  (u=0.15, tau=1000)", CLASS_COLOR["fragment"]),
    ("p1_u0.5_tau5000", z, "suppress  (u=0.5, tau=5000)", CLASS_COLOR["suppress"]),
]:
    t, e = env(store, lbl)
    axB.plot(t, e, color=col, lw=1.6, label=name)
axB.set_xlabel("time (s)"); axB.set_ylabel("active fraction (200 ms envelope)")
axB.set_title("B  example envelopes — none is a sustained event + clean offset")
axB.legend(fontsize=8, frameon=False, loc="upper right")
axB.spines[["top", "right"]].set_visible(False)

fig.suptitle("M4-2 P1 (k_q=0.10, alpha_G=16): E->E STD does not cleanly terminate the M4 bounded state — "
             "persist -> fragment -> suppress, no terminate_clean   [map = seed 1; seed 3/4 replicate the no-go]",
             fontsize=10, y=1.02)
fig.tight_layout()
out = os.path.join(OUT, "m4_2_p1_sweep_map.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
