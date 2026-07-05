#!/usr/bin/env python3
"""Aggregate the M3 static-μ tiny pilot (B spontaneous + C basin + D h-controls) into the
STATUS report answering the 6 questions (spec m3_static_mu_pilot_2026-06-24 v2 §8). OFFLINE.

B: spontaneous_summary.json per μ -> event rate / size / duration / R-class vs μ.
C: basin per_seed_metrics.csv per μ -> P_return/P_escape(K) from RAW returned/runaway (NOT
   differenced — high-μ core_only self-ignition is phenotype), Kmin_return/escape(μ).
D: spontaneous_summary.json per (h-mode, μ) -> does core-h give more R3/R4a than uniform/shuffled.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = "results/topic4_sef_hfo/m3_static_mu"
MUS = [0, 0.5, 0.9]   # dir names are mu0 / mu0.5 / mu0.9 (str(0)=='0', matches the launcher)


def _load_spont(d):
    p = os.path.join(d, "spontaneous_summary.json")
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def _basin_PK(d):
    """P_return(K), P_escape(K) from RAW core_kick returned/runaway (per K, mean over seeds)."""
    p = os.path.join(d, "per_seed_metrics.csv")
    if not os.path.exists(p):
        return {}
    def _t(v):
        return str(v).strip() in ("1", "True", "true") or _f(v) > 0.5
    def _f(v):
        try:
            return float(v)
        except ValueError:
            return 0.0
    by = {}
    for r in csv.DictReader(open(p)):
        k = round(float(r["kick_boost"]), 3)
        seed = int(float(r["seed"]))
        by.setdefault(k, {})[seed] = (_t(r.get("returned", 0)), _t(r.get("runaway", 0)))
    out = {}
    for k, sd in by.items():
        ret = np.mean([v[0] for v in sd.values()])
        run = np.mean([v[1] for v in sd.values()])
        out[k] = {"P_return": float(ret), "P_escape": float(run), "n": len(sd)}
    return out


def _kmin(pk, key, thresh=0.5):
    ks = sorted(k for k in pk if pk[k][key] >= thresh)
    return ks[0] if ks else float("inf")


def main():
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    # --- B spontaneous ---
    B = {mu: _load_spont(f"{ROOT}/spontaneous/mu{mu}") for mu in MUS}
    # --- C basin ---
    C = {mu: _basin_PK(f"{ROOT}/basin/mu{mu}") for mu in MUS}
    Ck = {mu: {"Kmin_return": _kmin(C[mu], "P_return"), "Kmin_escape": _kmin(C[mu], "P_escape")}
          for mu in MUS if C[mu]}
    # --- D h controls ---
    D = {}
    for h in ("core_susceptibility", "uniform", "shuffled"):
        for mu in (0.5, 0.9):
            d = (f"{ROOT}/spontaneous/mu{mu}" if h == "core_susceptibility"
                 else f"{ROOT}/h_controls/{h}_mu{mu}")
            D[(h, mu)] = _load_spont(d)

    def rfrac(s, k):
        return (s or {}).get("R_fractions", {}).get(k, 0.0)

    # --- STATUS answers ---
    rates = {mu: (B[mu]["event_rate_hz_per_seed"] if B[mu] else None) for mu in MUS}
    sizes = {mu: (B[mu]["size_active_bins"]["median"] if B[mu] else None) for mu in MUS}
    durs = {mu: (B[mu]["duration_ms"]["median"] if B[mu] else None) for mu in MUS}
    q1 = (rates[0.9] is not None and rates[0.0] is not None and rates[0.9] > rates[0.0])
    q2 = any(B[mu] and rfrac(B[mu], "R3") > 0 for mu in MUS)
    r4a = max((rfrac(B[mu], "R4a") for mu in MUS if B[mu]), default=0.0)
    r4b = max((rfrac(B[mu], "R4b") for mu in MUS if B[mu]), default=0.0)
    q3 = "R4a present" if r4a > 0 else ("only R4b tonic" if r4b > 0 else "no R4")
    q4 = (len(Ck) >= 2 and Ck.get(0.9, {}).get("Kmin_return", np.inf)
          <= Ck.get(0.0, {}).get("Kmin_return", np.inf))
    cs_hi = rfrac(D.get(("core_susceptibility", 0.9)), "R3") + rfrac(D.get(("core_susceptibility", 0.9)), "R4a")
    un_hi = rfrac(D.get(("uniform", 0.9)), "R3") + rfrac(D.get(("uniform", 0.9)), "R4a")
    sh_hi = rfrac(D.get(("shuffled", 0.9)), "R3") + rfrac(D.get(("shuffled", 0.9)), "R4a")
    q5 = (cs_hi >= un_hi and cs_hi >= sh_hi)

    status = {
        "Q1_rate_size_duration_change": {"answer": bool(q1), "event_rate_hz": rates,
                                         "size_median_bins": sizes, "duration_median_ms": durs},
        "Q2_R3_appears": {"answer": bool(q2),
                          "R3_frac": {mu: rfrac(B[mu], "R3") for mu in MUS}},
        "Q3_R4a_vs_R4b": {"answer": q3, "R4a_max": r4a, "R4b_max": r4b,
                          "R4a_frac": {mu: rfrac(B[mu], "R4a") for mu in MUS},
                          "R4b_frac": {mu: rfrac(B[mu], "R4b") for mu in MUS}},
        "Q4_Kmin_decreases": {"answer": bool(q4), "basin": Ck},
        "Q5_core_h_beats_controls": {"answer": bool(q5),
                                     "R3+R4a@μ0.9": {"core": cs_hi, "uniform": un_hi, "shuffled": sh_hi}},
    }
    json.dump(status, open(f"{ROOT}/status_static_mu_pilot.json", "w"), indent=1)

    # --- figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # fig 1: event rate + R-class stack vs μ
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    mus = [m for m in MUS if B[m]]
    ax[0].plot(mus, [B[m]["event_rate_hz_per_seed"] for m in mus], "o-", color="tab:red")
    ax[0].set_xlabel("μ"); ax[0].set_ylabel("spontaneous event rate (Hz/seed)")
    ax[0].set_title("Q1: event rate vs μ")
    bottom = np.zeros(len(mus))
    for cls, col in [("R2", "tab:green"), ("R3", "tab:orange"), ("R4a", "tab:red"), ("R4b", "0.4"),
                     ("R1", "tab:blue"), ("R0", "0.85")]:
        vals = [rfrac(B[m], cls) for m in mus]
        ax[1].bar([str(m) for m in mus], vals, bottom=bottom, label=cls, color=col)
        bottom += vals
    ax[1].set_xlabel("μ"); ax[1].set_ylabel("R-class fraction"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Q2/Q3: regime fractions vs μ"); ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{ROOT}/figures/spontaneous_vs_mu.png", dpi=130); plt.close(fig)

    # fig 2: basin P_escape(K) per μ + Kmin
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for mu in MUS:
        if C[mu]:
            ks = sorted(C[mu]); ax.plot(ks, [C[mu][k]["P_escape"] for k in ks], "o-", label=f"μ={mu}")
    ax.set_xlabel("kick K"); ax.set_ylabel("P_escape (raw runaway)")
    ax.set_title("Q4: basin escape vs K, per μ"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{ROOT}/figures/basin_escape_vs_mu.png", dpi=130); plt.close(fig)

    # fig 3: h-control R3+R4a at μ0.9
    fig, ax = plt.subplots(figsize=(6, 4))
    hs = ["core_susceptibility", "uniform", "shuffled"]
    ax.bar(hs, [cs_hi, un_hi, sh_hi], color=["tab:red", "tab:grey", "tab:blue"])
    ax.set_ylabel("R3+R4a fraction @ μ=0.9"); ax.set_title("Q5: core-h vs controls")
    fig.tight_layout(); fig.savefig(f"{ROOT}/figures/h_control_mu09.png", dpi=130); plt.close(fig)

    # --- STATUS markdown ---
    with open(f"{ROOT}/status_static_mu_pilot.md", "w") as f:
        f.write("# M3 static-μ tiny pilot — STATUS (spec v2 §8)\n\n")
        f.write("**TINY pilot (L20, seeds 3 spontaneous / 6 basin, μ={0,0.5,0.9}, core-h). NOT a "
                "formal grid. Goal = does the spontaneous phenotype shift with μ.**\n\n")
        f.write(f"1. **spontaneous μ changes rate/size/duration?** {q1}. event_rate(Hz/seed)="
                f"{rates}; size_median_bins={sizes}; duration_median_ms={durs}\n")
        f.write(f"2. **R3 appears?** {q2}. R3_frac={ {m: round(rfrac(B[m],'R3'),3) for m in mus} }\n")
        f.write(f"3. **R4a vs R4b?** {q3}. R4a_max={r4a:.3f}, R4b_max={r4b:.3f}\n")
        f.write(f"4. **Kmin_return/escape decrease with μ?** {q4}. {Ck}\n")
        f.write(f"5. **core-h beats uniform/shuffled?** {q5}. R3+R4a@μ0.9: core={cs_hi:.3f} "
                f"uniform={un_hi:.3f} shuffled={sh_hi:.3f}\n")
        f.write("6. **worth a formal L20 μ grid?** — see judgment below.\n\n")
        f.write("figures: spontaneous_vs_mu.png, basin_escape_vs_mu.png, h_control_mu09.png\n")
    print("[static-μ pilot] STATUS:")
    for k, v in status.items():
        print(f"  {k}: {v.get('answer')}")
    print(f"[static-μ pilot] wrote -> {ROOT}/status_static_mu_pilot.{{md,json}} + figures/")


if __name__ == "__main__":
    main()
