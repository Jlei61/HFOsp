"""FCXR-HEO3 Stage H3.0 part A — joint sliding-window audit of the 8 existing arms (compute-free).

Answers the question HEO2.1 could not: do "broadband", "desynchronized" and "high energy" hold AT THE
SAME TIME in the fast-τ/10% precursor, and for how long consecutively? Whole-window summaries said
broadband 7/15 + coherence 0.54; this reports per-250ms-window joint target occupancy + longest run,
which is the quantity HEO3's 2x2 has to improve. Writes heo3/stage0_joint_windows.json + a figure.
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
import src.topic4_mz_fcxr_heo3 as H3  # noqa: E402
from topic4_mz_fcxr_heo1 import build_baseline_reference, decimate_to_work  # noqa: E402

DT = 0.05
WIN_MS, HOP_MS = 1000.0, 100.0   # 250ms cannot measure 1-4Hz (empty band) and leaks 16Hz into 8-13
MZ = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay")
HEO1 = os.path.join(MZ, "high_energy_oscillatory_branch")
BD = os.path.join(MZ, "broadband_diagnostic")
OUT = os.path.join(MZ, "heo3")
ARMS = ["m_off", "dyn_tau250_frac0.05", "dyn_tau250_frac0.1", "dyn_tau750_frac0.05",
        "dyn_tau750_frac0.1", "static_K", "mean_static_K", "dyn_tau750_frac0.1_ext"]


def audit(lab, ref):
    t = np.load(os.path.join(BD, "arms", lab + "_trace.npz"), allow_pickle=True)
    rate = np.asarray(t["rate_E"], float); lfp = np.asarray(t["lfp_trace"], float)
    men = float(t["m_enable_ms"]); k = int(men / DT) if men > 0 else 0
    rate, lfp = rate[k:], lfp[k:]                                    # post-enable only
    ldec, fs = decimate_to_work(lfp, DT); rdec, _ = decimate_to_work(rate, DT)
    bp, tcen = H3.band_power_windows(ldec, fs, WIN_MS, HOP_MS)
    dt_dec_ms = 1000.0 / fs
    active = rdec >= 20.0                                            # ACTIVE samples only
    phi = H3.band_phase(ldec, fs)
    plv_w = H3.pairwise_plv_windows(phi, tcen, dt_dec_ms, WIN_MS, active=active)
    R_w = H3.resample_to_windows(H3.phase_order_parameter(ldec, fs), tcen, dt_dec_ms, gate=active)
    rate_w = H3.resample_to_windows(rdec, tcen, dt_dec_ms)
    fs_win = 1000.0 / HOP_MS
    g = H3.joint_target_windows(bp, ref["med_power"], rate_w, plv_w, fs_win)
    g["order_R"] = R_w
    return g, tcen


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    d0 = np.load(os.path.join(HEO1, "baseline_lfp_seed1.npz"), allow_pickle=True)
    ref = build_baseline_reference(np.asarray(d0["lfp_trace"], float), np.asarray(d0["rate_E"], float), DT)

    rows, series = {}, {}
    print(f"=== H3.0 joint sliding-window audit ({WIN_MS:g}ms win / {HOP_MS:g}ms hop) ===")
    print(f"{'arm':24s} {'target%':>8s} {'run_ms':>8s} | per-criterion %: recruit broadband desync energy | medPLV")
    for lab in ARMS:
        g, tcen = audit(lab, ref)
        c = g["frac_by_criterion"]
        rows[lab] = dict(frac_target=g["frac_target"], longest_run_ms=g["longest_run_ms"],
                         frac_by_criterion=c, n_windows=int(len(g["target"])),
                         median_recruit=float(np.median(g["recruit"])),
                         median_broadband=float(np.median(g["broadband"])),
                         median_plv=float(np.nanmedian(g["plv"])),
                         median_order_R=float(np.nanmedian(g["order_R"])))
        series[lab] = dict(t=tcen.tolist(), recruit=g["recruit"].tolist(),
                           broadband=g["broadband"].tolist(), plv=np.asarray(g["plv"]).tolist(),
                           order_R=np.asarray(g["order_R"]).tolist(),
                           rate=np.asarray(g["rate"]).tolist(), target=g["target"].astype(int).tolist())
        print(f"{lab:24s} {100*g['frac_target']:7.1f}% {g['longest_run_ms']:7.0f} | "
              f"{100*c['recruited']:6.0f} {100*c['broadband']:9.0f} {100*c['desynchronized']:9.0f} "
              f"{100*c['high_energy']:6.0f} | PLV {np.nanmedian(g['plv']):.2f}")
    json.dump(dict(win_ms=WIN_MS, hop_ms=HOP_MS, arms=rows), open(os.path.join(OUT, "stage0_joint_windows.json"), "w"), indent=1)
    json.dump(series, open(os.path.join(OUT, "stage0_joint_series.json"), "w"))

    # figure: the precursor arm's four criteria over time + joint target band
    focus = ["dyn_tau250_frac0.1", "m_off", "mean_static_K"]
    fig, axes = plt.subplots(len(focus), 1, figsize=(11, 2.3 * len(focus)), sharex=True)
    for ax, lab in zip(np.atleast_1d(axes), focus):
        s = series[lab]; t = np.asarray(s["t"])
        ax.plot(t, s["recruit"], lw=1.0, color="#4c72b0", label="recruit /15")
        ax.plot(t, s["broadband"], lw=1.2, color="#e8a33d", label="broadband /15")
        ax.plot(t, 15 * np.asarray(s["plv"], float), lw=1.2, color="#c44e52", label="pairwise PLV (×15)")
        ax.axhline(12, ls=":", c="#4c72b0", lw=0.7); ax.axhline(8, ls=":", c="#e8a33d", lw=0.7)
        ax.axhline(15 * 0.60, ls=":", c="#c44e52", lw=0.7)
        tg = np.asarray(s["target"], bool)
        ax.fill_between(t, 0, 16, where=tg, color="#2ca02c", alpha=0.22, step="mid", label="joint target")
        ax.set_ylim(0, 16); ax.set_ylabel("contacts /15", fontsize=8)
        ax.set_title(f'{lab} — joint target {100*rows[lab]["frac_target"]:.1f}% of windows, '
                     f'longest run {rows[lab]["longest_run_ms"]:.0f} ms', fontsize=9)
        if ax is np.atleast_1d(axes)[0]:
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.035), frameon=False)
    np.atleast_1d(axes)[-1].set_xlabel("time after adaptation enable (s)")
    fig.text(0.5, 0.004, "FCXR-HEO3 H3.0 — joint sliding-window gate (recruited ∧ broadband ∧ desynchronized ∧ high-energy)",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "stage0_joint_windows.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[h3.0] wrote heo3/stage0_joint_windows.json + figures/stage0_joint_windows.png")


if __name__ == "__main__":
    main()
