"""FCXR-HEO2 Phase-1 analysis + figure — classify each arm and compare to the real E1146 spectrum.

Splits each arm at m_enable_ms (pre = established 16Hz state, post = with adaptation), computes the
Phase-0 metrics on the post window, applies phase1_verdict, writes broadband_diagnostic/phase1_verdict.json,
and renders figures/phase1_arms.png (per-arm rate+m traces with the enable line + six-band ΔdB vs real).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import src.topic4_mz_fcxr_heo2 as H2  # noqa: E402
from topic4_mz_fcxr_heo1 import build_baseline_reference, band_db_field, decimate_to_work, oscillation_probe, BANDS  # noqa: E402
from run_topic4_heo2_phase0 import _coverage  # noqa: E402

HEO1 = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                    "high_energy_oscillatory_branch")
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay", "broadband_diagnostic")
DT = 0.05
VCOL = dict(transformed_broadband_spiky="#2ca02c", stalled="#dd8452", collapsed_sparse="#e8c547",
            silenced="#9aa0a6", unchanged_16Hz="#4c72b0", unsafe="#7a1f1f", no_high_state="#9aa0a6")


def _metrics(lfp, rate, ref):
    ldec, fs = decimate_to_work(lfp, DT); rdec, _ = decimate_to_work(rate, DT)
    ddb = np.median(band_db_field(lfp, DT, ref), axis=0)
    l2, cos = H2.spectral_distance_to_real(ddb)
    return dict(dominant_hz=float(H2.dominant_2s(rdec, fs)), event_ipi_hz=float(H2.event_ipi_hz(rdec, fs)),
                mean_rate=float(rate.mean()), coverage=_coverage(lfp, ref, DT), dist_to_real=float(l2),
                six_band_ddb=[float(x) for x in ddb], coherence=float(oscillation_probe(lfp, rate, DT)["coherence_med"]))


def main():
    d0 = np.load(os.path.join(HEO1, "baseline_lfp_seed1.npz"), allow_pickle=True)
    ref = build_baseline_reference(np.asarray(d0["lfp_trace"], float), np.asarray(d0["rate_E"], float), DT)
    bm = json.load(open(os.path.join(OUT, "phase1_arms.json")))
    labels = [r["label"] for r in bm["arms"]]

    arms = {}
    for lab in labels:
        d = np.load(os.path.join(OUT, "arms", lab + "_trace.npz"), allow_pickle=True)
        j = json.load(open(os.path.join(OUT, "arms", lab + ".json")))
        rate = np.asarray(d["rate_E"], float); lfp = np.asarray(d["lfp_trace"], float)
        m_mean = np.asarray(d["m_mean"], float); men = float(d["m_enable_ms"])
        k = int(men / DT) if men > 0 else 0
        pre = _metrics(lfp[:k], rate[:k], ref) if k > 32 else _metrics(lfp, rate, ref)
        post = _metrics(lfp[k:], rate[k:], ref)
        arms[lab] = dict(rate=rate, lfp=lfp, m_mean=m_mean, men=men, k=k, pre=pre, post=post, safety=j)

    m_off = arms["m_off"]["post"]
    verdicts = {}
    for lab, a in arms.items():
        v = H2.phase1_verdict(dict(mean_rate=a["pre"]["mean_rate"], coherence=a["pre"]["coherence"]), a["post"],
                              a["m_mean"][a["k"]:], m_off["dist_to_real"], m_off["coverage"],
                              dict(numerical_unsafe=a["safety"]["numerical_unsafe"],
                                   runaway_early_stop_ms=a["safety"]["runaway_early_stop_ms"]))
        verdicts[lab] = dict(verdict=v["verdict"], criteria=v["criteria"], post=a["post"], pre_mean_rate=a["pre"]["mean_rate"])
    json.dump(verdicts, open(os.path.join(OUT, "phase1_verdict.json"), "w"), indent=1)
    print("=== PHASE-1 VERDICTS ===")
    for lab in labels:
        p = arms[lab]["post"]
        print(f'  {lab}: {verdicts[lab]["verdict"]} | post dom {p["dominant_hz"]:.1f}Hz ipi {p["event_ipi_hz"]:.1f} '
              f'rate {p["mean_rate"]:.1f} cov {p["coverage"]}/15 dist {p["dist_to_real"]:.1f} ddb {[round(x,1) for x in p["six_band_ddb"]]}')

    # figure: per-arm (rate + m_mean, enable line) ; last panel = six-band ΔdB vs real
    n = len(labels)
    fig, ax = plt.subplots(n + 1, 1, figsize=(11.0, 2.0 * (n + 1)))
    for i, lab in enumerate(labels):
        a = arms[lab]
        rdec, fs = decimate_to_work(a["rate"], DT)
        t = np.arange(len(rdec)) / fs
        ax[i].plot(t, rdec, lw=0.5, color="#333", label="rate_E")
        ax2 = ax[i].twinx()
        md = a["m_mean"][::int(len(a["m_mean"]) / len(rdec)) or 1][:len(rdec)]
        ax2.plot(t[:len(md)], md, lw=1.0, color="#c44e52", label="m_mean")
        if a["men"] > 0:
            ax[i].axvline(a["men"] / 1000.0, ls="--", c="0.4", lw=1)
        ax[i].set_ylabel("rate", fontsize=7); ax2.set_ylabel("m", fontsize=7, color="#c44e52")
        ax[i].set_title(f'{lab} → {verdicts[lab]["verdict"]}  (post {a["post"]["dominant_hz"]:.1f}Hz, '
                        f'rate {a["post"]["mean_rate"]:.0f}, cov {a["post"]["coverage"]}/15, dist {a["post"]["dist_to_real"]:.0f})',
                        fontsize=8.5, color=VCOL.get(verdicts[lab]["verdict"], "k"))
        ax[i].set_xlim(0, t[-1])
    ax[n].set_xlabel("time (s)")
    # six-band ΔdB comparison
    axb = ax[n]; x = np.arange(6); w = 0.8 / (n + 1)
    axb.clear()
    axb.bar(x - 0.4 + 0 * w, H2.REAL_E1146_DDB, w, label="real E1146", color="#c44e52")
    for i, lab in enumerate(labels):
        axb.bar(x - 0.4 + (i + 1) * w, arms[lab]["post"]["six_band_ddb"], w, label=lab)
    axb.axhline(0, lw=0.6, c="0.6"); axb.set_xticks(x); axb.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in BANDS], fontsize=7)
    axb.set_ylabel("ΔdB"); axb.set_title("post six-band ΔdB vs real", fontsize=8.5)
    axb.legend(fontsize=6.5, ncol=n + 1, loc="upper center")
    fig.text(0.5, 0.004, "FCXR-HEO2 Phase 1 — delayed-adaptation wedge diagnostic", ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "figures", "phase1_arms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[phase1] wrote phase1_verdict.json + figures/phase1_arms.png")


if __name__ == "__main__":
    main()
