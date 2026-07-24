"""FCXR-HEO2 Phase 0 — re-map the 48 HEO1 cells with fixed estimators + 4-class state map (zero compute).

Reads the existing high_energy_oscillatory_branch/screen_cells/*_trace.npz + baseline_lfp_seed1.npz;
writes broadband_diagnostic/phase0_state_map.json + figures/phase0_state_map.png. No simulation.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import src.topic4_mz_fcxr_heo2 as H2  # noqa: E402
from topic4_mz_fcxr_heo1 import (  # noqa: E402
    build_baseline_reference, band_db_field, band_power_spectrogram, decimate_to_work,
    oscillation_probe, Z_GATE, N_BANDS_GATE, BROADBAND_IDX, DB_GAIN_GATE)

HEO1 = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                    "high_energy_oscillatory_branch")
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                   "broadband_diagnostic")
CLASS_COL = dict(sparse_event_train="#dd8452", transitional="#9aa0a6",
                 tonic_16Hz_cycle="#4c72b0", target_like_spiky="#2ca02c")


def _coverage(lfp, ref, dt=0.05):
    ldec, fs = decimate_to_work(lfp, dt)
    bp, _ = band_power_spectrogram(ldec, fs)
    logbp = np.log10(np.maximum(bp, 1e-300))
    denom = 1.4826 * ref["mad_log"]
    z = np.where(denom[None] > 0, (logbp - ref["med_log"][None]) / np.where(denom[None] > 0, denom[None], 1), -np.inf)
    bpass = (z >= Z_GATE) & (bp >= ref["q99_power"][None])
    med_db = np.median(10 * np.log10(np.maximum(bp, 1e-300) / np.maximum(ref["med_power"][None], 1e-300)), axis=2)
    ch = (bpass.sum(2) >= N_BANDS_GATE) & bpass[:, :, BROADBAND_IDX[0]] & bpass[:, :, BROADBAND_IDX[1]] & (med_db >= DB_GAIN_GATE)
    return int(ch.sum(1).max())


def main():
    d0 = np.load(os.path.join(HEO1, "baseline_lfp_seed1.npz"), allow_pickle=True)
    ref = build_baseline_reference(np.asarray(d0["lfp_trace"], float), np.asarray(d0["rate_E"], float), 0.05)
    rows = []
    for f in sorted(glob.glob(os.path.join(HEO1, "screen_cells", "gq*_trace.npz"))):
        lab = os.path.basename(f).replace("_trace.npz", "")
        j = json.load(open(os.path.join(HEO1, "screen_cells", lab + ".json")))
        dc = np.load(f, allow_pickle=True)
        lfp = np.asarray(dc["lfp_trace"], float); rate = np.asarray(dc["rate_E"], float)
        rdec, fs = decimate_to_work(rate, 0.05)
        ddb = np.median(band_db_field(lfp, 0.05, ref), axis=0)
        l2, cos = H2.spectral_distance_to_real(ddb)
        cov = _coverage(lfp, ref)
        coh = oscillation_probe(lfp, rate, 0.05)["coherence_med"]
        met = dict(label=lab, D=j["D"], A_c=j["A_c"], gq=j["gate_quantile"], ic=j["ic"],
                   mean_rate_hz=round(j["mean_rate_hz"], 1),
                   dominant_hz=round(H2.dominant_2s(rdec, fs), 2), event_ipi_hz=round(H2.event_ipi_hz(rdec, fs), 2),
                   spikiness=round(H2.spikiness(rdec), 2), spectral_entropy=round(H2.spectral_entropy(rdec, fs), 2),
                   bw90=round(H2.bw90(rdec, fs), 1), duty_cycle=round(H2.duty_cycle(rdec, fs), 3),
                   max_gap_ms=round(H2.max_silence_gap_ms(rdec, fs), 0), coverage=cov, coherence=round(float(coh), 3),
                   six_band_ddb=[round(float(x), 1) for x in ddb], dist_to_real=round(l2, 1), cos_to_real=round(cos, 3))
        met["class"] = H2.classify_state(dict(dominant_hz=met["dominant_hz"], duty_cycle=met["duty_cycle"],
                                              coverage=cov, six_band_ddb=ddb, coherent=coh >= 0.9))
        rows.append(met)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "phase0_state_map.json"), "w"), indent=1)

    import collections
    tally = collections.Counter(r["class"] for r in rows)
    print("=== CLASS TALLY:", dict(tally), "===")
    print("any target_like_spiky:", [r["label"] for r in rows if r["class"] == "target_like_spiky"] or "NONE")
    print("closest to real (min dist_to_real):")
    for r in sorted(rows, key=lambda x: x["dist_to_real"])[:5]:
        print(f'  {r["label"]}: dist {r["dist_to_real"]} cos {r["cos_to_real"]} dom {r["dominant_hz"]}Hz '
              f'ipi {r["event_ipi_hz"]}Hz duty {r["duty_cycle"]} cov {r["coverage"]}/15 class {r["class"]} '
              f'ddb {r["six_band_ddb"]}')
    for a in ("gq0.9999_A1_D0.15_nokick", "gq0.9999_A4_D0.15_nokick"):
        r = next((x for x in rows if x["label"] == a), None)
        if r:
            print(f'  ANCHOR {a}: dom {r["dominant_hz"]}Hz ipi {r["event_ipi_hz"]}Hz duty {r["duty_cycle"]} '
                  f'cov {r["coverage"]}/15 dist {r["dist_to_real"]} class {r["class"]}')

    # figure: (a) dominant vs duty (size=coverage, color=class), (b) dist_to_real sorted
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for r in rows:
        ax[0].scatter(r["dominant_hz"], r["duty_cycle"], s=30 + 18 * r["coverage"],
                      c=CLASS_COL[r["class"]], edgecolors="k", lw=0.5, alpha=0.85)
    ax[0].axvspan(3, 8, color="#2ca02c", alpha=0.06); ax[0].axhline(0.6, ls=":", c="0.5", lw=1)
    ax[0].set_xlabel("dominant freq (Hz, 2s Welch)"); ax[0].set_ylabel("duty cycle")
    ax[0].set_title("48-cell state map (size=coverage/15; green band=3-8Hz target)", fontsize=10)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", label=k)
                          for k, c in CLASS_COL.items()], fontsize=8, loc="upper left")
    srt = sorted(rows, key=lambda x: x["dist_to_real"])
    ax[1].barh(range(len(srt)), [r["dist_to_real"] for r in srt],
               color=[CLASS_COL[r["class"]] for r in srt])
    ax[1].set_yticks([]); ax[1].invert_yaxis(); ax[1].set_xlabel("L2 distance to real E1146 ΔdB (dB)")
    ax[1].set_title("closest→farthest from real seizure spectrum (top=closest)", fontsize=10)
    fig.text(0.5, 0.005, "FCXR-HEO2 Phase 0 — state map diagnostic", ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "figures", "phase0_state_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[phase0] wrote phase0_state_map.json + figures/phase0_state_map.png")


if __name__ == "__main__":
    main()
