"""M4 pass-1 boundary-continuity and empirical-mode diagnostic.

For each available seed x alpha_G fine-scan cell, extract a pre-escape or quasi-stationary
window and measure: (i) descriptive oscillation amplitude and period, and (ii) a rate-channel
delay-embedded empirical mode (growth rate sigma, frequency, complex vs real).

This script is a continuity diagnostic for the upper boundary of the bounded window. It can
support or reject "Hopf-like delayed-feedback oscillation" wording, but it is not a full
network Jacobian analysis and does not prove a Hopf bifurcation.
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DT = 0.1
OUT = ROOT / "results/paper-ready-figure/fig_m4_dynamic_qi/figures"
SEED_DIR = {1: "finescan_seed1", 2: "finescan_seed2", 3: "finescan_seed3", 4: "finescan_seed4"}
SEED_COL = {1: "#2e7d32", 2: "#1f6feb", 3: "#e8873a", 4: "#8b3a8b"}


def _sm(x, w_ms=30.0):
    n = max(1, int(w_ms / DT)); return np.convolve(x, np.ones(n) / n, mode="same")


def _period_ms(x):
    x = x - x.mean(); ac = np.correlate(x, x, "full")[len(x) - 1:]
    if ac[0] <= 0: return None
    ac = ac / ac[0]; below = np.where(ac < 0)[0]
    if below.size == 0: return None
    i0 = below[0]; seg = ac[i0:i0 + 3000]
    if seg.size < 3: return None
    pk = i0 + int(np.argmax(seg)); return pk * DT if pk > i0 else None


def _thirds(x):
    n = len(x) // 3; return [float(x[k * n:(k + 1) * n].ptp()) for k in range(3)]


def _rate_eigenmode(rate, run, T):
    """⑤ empirical leading mode from the rate channel ALONE (delay-embedded), on the pre-escape LINEAR window
    (excludes the nonlinear blow-up + the saturating q_I/S_G channels that confound a 3-channel fit). Returns
    the leading complex pair's growth rate sigma [1/ms] + frequency f [Hz]. NOT a model Jacobian -> empirical."""
    r = _sm(rate, 10.0); lo = int(1500 / DT)
    if run:
        cut = run - 300
        over = np.where(r[lo:] > 130)[0]                                # cut before it crosses toward runaway
        if over.size: cut = min(cut, 1500 + over[0] * DT)
        hi = int(cut / DT)
    else:
        hi = min(int(T / DT), len(r))
    if hi - lo < int(1500 / DT): return None
    ds = 50; s = r[lo:hi][::ds]; d = 10                                 # downsample 5ms, delay 50ms, dim 3
    if len(s) < 3 * d + 10: return None
    Y = np.column_stack([s[2 * d:], s[d:-d], s[:-2 * d]])
    Z = np.hstack([Y[:-1], np.ones((len(Y) - 1, 1))])
    M, *_ = np.linalg.lstsq(Z, Y[1:], rcond=None); A = M[:-1].T
    lam = np.log(np.linalg.eigvals(A).astype(complex)) / (DT * ds)
    lam = lam[np.argsort(-lam.real)]; comp = lam[np.abs(lam.imag) > 1e-9]
    lead = comp[0] if len(comp) else lam[0]
    return dict(sigma=float(lead.real), f_hz=float(abs(lead.imag) / (2 * np.pi) * 1e3),
                is_complex=bool(len(comp)), win_ms=int((hi - lo) * DT))


def analyze_seed(seed):
    d = ROOT / f"results/topic4_m4_dynamic_{SEED_DIR[seed]}"
    f = d / "dynamic_qi_summary.json"
    if not f.exists(): return None
    with open(f) as fh:
        s = json.load(fh)
    T = s["meta"]["T"]
    rows = []
    with np.load(d / "dynamic_qi_traces.npz", allow_pickle=True) as z:
        for r in s["rows"]:
            lab = r["label"]; ag = float(lab.split("_aG")[-1]); run = r.get("runaway_ms")
            rate = np.asarray(z[lab + "__rate"], float)
            lo = int(1000 / DT); hi = int((run - 500) / DT) if run else min(int(T / DT), len(rate))
            rec = dict(alpha_G=ag, runaway_ms=run, bounded=(run is None), verdict=r["verdict"])
            if hi - lo > int(2000 / DT):                                # descriptive amplitude/period (④)
                rs = _sm(rate[lo:hi]); rec["period_ms"] = _period_ms(rs); rec["amp_thirds"] = [round(a, 1) for a in _thirds(rs)]
            else:
                rec["period_ms"] = None; rec["amp_thirds"] = None
            em = _rate_eigenmode(rate, run, T)                          # rate-only pre-escape empirical mode
            if em: rec.update(sigma=em["sigma"], f_hz=em["f_hz"], is_complex=em["is_complex"])
            else: rec.update(sigma=None, f_hz=None, is_complex=None)
            rows.append(rec)
    return sorted(rows, key=lambda r: r["alpha_G"])


def main():
    data = {sd: analyze_seed(sd) for sd in [1, 2, 3, 4]}
    data = {k: v for k, v in data.items() if v}
    print(f"=== M4-1 ④+⑤ continuity: seeds available = {sorted(data)} ===")
    for sd, rows in data.items():
        print(f"\n-- seed{sd} --")
        for r in rows:
            amp = r.get("amp_thirds"); grow = (amp and amp[0] > 0 and amp[2] / max(amp[0], 1e-9) > 1.8)
            print(f"  aG{r['alpha_G']:4.1f} {'BND' if r['bounded'] else 'run@%d' % r['runaway_ms']:9} "
                  f"sigma={r['sigma']:+.5f}/ms f={r['f_hz']}Hz cplx={r['is_complex']} "
                  f"amp={amp} {'GROWING' if grow else ''}" if r.get("sigma") is not None
                  else f"  aG{r['alpha_G']:4.1f} {'BND' if r['bounded'] else 'run@%d' % r['runaway_ms']:9} (window too short)")
    if not data: return
    # 3-panel figure: (1) boundary location, (2) sigma(alpha_G) Hopf test, (3) f(alpha_G)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for sd, rows in data.items():
        c = SEED_COL[sd]; ag = [r["alpha_G"] for r in rows]
        rm = [r["runaway_ms"] if r["runaway_ms"] else np.nan for r in rows]
        bnd = [r["alpha_G"] for r in rows if r["bounded"]]
        ax[0].plot(ag, rm, "o-", color=c, lw=1, ms=4, label=f"seed{sd}")
        ax[0].plot(bnd, [15500] * len(bnd), "s", color=c, ms=7, clip_on=False)
        sig = [r.get("sigma") for r in rows]; cx = [r.get("is_complex") for r in rows]
        ax[1].plot(ag, [s if s is not None else np.nan for s in sig], "-", color=c, lw=1)
        for r in rows:
            if r.get("sigma") is not None:
                ax[1].plot(r["alpha_G"], r["sigma"], "o" if r["is_complex"] else "x", color=c, ms=5)
        fh = [r.get("f_hz") if (r.get("is_complex")) else np.nan for r in rows]
        ax[2].plot(ag, fh, "o-", color=c, lw=1, ms=4)
    ax[0].axhline(15500, color="0.7", lw=0.6, ls=":"); ax[0].set_ylim(0, 16000)
    ax[0].set_title("① boundary: runaway time vs pool strength\n(square @top = bounded to T)", fontsize=9, loc="left")
    ax[0].set_xlabel("alpha_G"); ax[0].set_ylabel("runaway_ms"); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="crimson", lw=0.8, ls="--")
    ax[1].set_title("② ⑤ empirical growth rate sigma(alpha_G)\n(o=complex/oscillatory, x=real; cross 0 = Hopf-like test)", fontsize=9, loc="left")
    ax[1].set_xlabel("alpha_G"); ax[1].set_ylabel("sigma (1/ms)")
    ax[2].axhline(5.0, color="0.6", lw=0.6, ls=":")
    ax[2].set_title("③ leading-mode frequency f(alpha_G)\n(complex modes only; ~5Hz line)", fontsize=9, loc="left")
    ax[2].set_xlabel("alpha_G"); ax[2].set_ylabel("f (Hz)")
    fig.suptitle("M4-1 ④ boundary continuity + ⑤ empirical eigenmode (EMPIRICAL data-driven linearization; "
                 "supports/rejects 'Hopf-like', not a proof of Hopf)", fontsize=10, y=1.02)
    fig.tight_layout()
    out = OUT / "fig_m4_continuity_eigenmode.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
