"""M4 bound-window bifurcation diagnostic (empirical eigenmode): the aG16 'third state' is a STABLE ~5Hz
limit cycle; strengthening the pool (aG20/24) makes the oscillation GROW (Hopf-like) -> runaway; weakening
it (aG12) is a q_I-drain (non-oscillatory) -> runaway. So the bound window is bracketed by a saddle-node
below and a delayed-divisive-feedback Hopf above. Smoothed per-neuron rate(t) per regime. Plotting-only."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/paper-ready-figure/fig_m4_dynamic_qi/figures"; DT=0.1
def load(d,l): z=np.load(ROOT/f"results/{d}/dynamic_qi_traces.npz",allow_pickle=True); return np.asarray(z[l+"__rate"],float)
def sm(x,w=30.0): n=int(w/DT); return np.convolve(x,np.ones(n)/n,mode="same")
cells=[("topic4_m4_dynamic_confirm","kq0.10_aG12.0","aG12 — q_I-drain (non-osc) → runaway 2235ms","#c1272d",2235),
       ("topic4_m4_dynamic_longconfirm","kq0.10_aG16.0","aG16 — STABLE ~5Hz limit cycle (bounded)","#2e7d32",None),
       ("topic4_m4_dynamic_longconfirm","kq0.10_aG20.0","aG20 — GROWING oscillation (Hopf) → runaway 6813ms","#e8873a",6813),
       ("topic4_m4_dynamic_delay","kq0.10_aG24.0","aG24 — growing oscillation","#8b3a8b",None)]
fig,ax=plt.subplots(4,1,figsize=(11,8.5),sharex=True)
for a,(d,l,ttl,c,run) in zip(ax,cells):
    r=sm(load(d,l)); t=np.arange(len(r))*DT
    a.plot(t,r,color=c,lw=0.7); a.axhline(120,color="0.6",lw=0.7,ls="--")
    if run: a.axvline(run,color="crimson",lw=1.0,ls=":"); a.text(run,a.get_ylim()[1]*0.9,"runaway",fontsize=7,color="crimson",ha="right")
    a.set_title(ttl,fontsize=10,loc="left"); a.set_ylabel("rate (Hz)",fontsize=8.5); a.set_xlim(0,min(len(r)*DT,15000))
ax[-1].set_xlabel("time (ms)")
fig.suptitle("M4 bound window = stable limit cycle bracketed by a q_I-drain (below) and a Hopf (above): "
             "the 'third state' is a bounded ~5Hz oscillation, destabilized by too-strong pool",fontsize=11,y=1.0)
fig.tight_layout(rect=[0,0,1,0.97]); out=OUT/"fig_m4_oscillation_diag.png"
fig.savefig(out,dpi=150,bbox_inches="tight"); plt.close(fig); print(f"wrote {out}")
