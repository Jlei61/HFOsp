"""FCXR-HEO2.1 unified figure — the de-conflated spatial readouts (review P1-a/b/c).
Panel 1: Phase-0 48 cells in (broadband_coverage_1_80 × phase_coherence), size∝duty, colour=class —
         all recruit ~15/15, so coverage is NOT the gap; the families split on broadband-shape × coherence
         × duty. Panel 2: per-arm de-conflated readouts (recruit / broadband / coherence) for the 8 arms
         incl. both static-K controls — shows fast-τ/10% partially broadens+desyncs (dynamics, since the
         mean-matched constant K stays synchronous narrowband). Panel 3: the slow-τ/10% arm extended to 9 s
         (1 s-bin activity) — extinguish then sparse late re-ignition = long-period intermittent, not
         permanent termination. Also writes the 2 control arms' readouts into phase1_controls.json.
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
from topic4_mz_fcxr_heo1 import build_baseline_reference, band_db_field, oscillation_probe  # noqa: E402

DT = 0.05
MZ = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay")
HEO1 = os.path.join(MZ, "high_energy_oscillatory_branch"); OUT = os.path.join(MZ, "broadband_diagnostic")
CLASS_COL = dict(sparse_event_train="#e8a33d", tonic_16Hz_cycle="#4c72b0", transitional="#55a868")


def _readout(lfp, rate, ref, men):
    k = int(men / DT) if men > 0 else 0
    plfp = lfp[k:] if lfp[k:].shape[0] > 20000 else lfp
    prate = rate[k:] if rate[k:].size > 20000 else rate
    ddb = band_db_field(plfp, DT, ref)
    return (H2.active_recruitment(ddb), H2.broadband_coverage_1_80(ddb),
            round(float(oscillation_probe(plfp, prate, DT)["coherence_med"]), 3))


def main():
    d0 = np.load(os.path.join(HEO1, "baseline_lfp_seed1.npz"), allow_pickle=True)
    ref = build_baseline_reference(np.asarray(d0["lfp_trace"], float), np.asarray(d0["rate_E"], float), DT)
    cells = json.load(open(os.path.join(OUT, "phase0_state_map.json")))
    verd = json.load(open(os.path.join(OUT, "phase1_verdict.json")))

    # control-arm readouts -> augment phase1_controls.json
    ctrl = json.load(open(os.path.join(OUT, "phase1_controls.json")))
    cro = {}
    for lab in ["mean_static_K", "dyn_tau750_frac0.1_ext"]:
        t = np.load(os.path.join(OUT, "arms", lab + "_trace.npz"), allow_pickle=True)
        rate = np.asarray(t["rate_E"], float); lfp = np.asarray(t["lfp_trace"], float); men = float(t["m_enable_ms"])
        rec, bb, coh = _readout(lfp, rate, ref, men)
        seg = H2.segment_state_label(rate, 1000.0 / DT, m_enable_ms=(men or None), dt=DT)
        cro[lab] = dict(active_recruitment=rec, broadband_coverage_1_80=bb, phase_coherence=coh, segment_label=seg)
    ctrl["readouts"] = cro
    json.dump(ctrl, open(os.path.join(OUT, "phase1_controls.json"), "w"), indent=1)

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.5, 1.0], wspace=0.32)

    # --- Panel 1: Phase-0 families ---
    ax = fig.add_subplot(gs[0, 0])
    for cls in ("sparse_event_train", "tonic_16Hz_cycle", "transitional"):
        sub = [c for c in cells if c["class"] == cls]
        ax.scatter([c["broadband_coverage_1_80"] for c in sub], [c["phase_coherence"] for c in sub],
                   s=[20 + 180 * c["duty_cycle"] for c in sub], c=CLASS_COL[cls], alpha=0.7,
                   edgecolors="0.3", linewidths=0.4, label=cls.replace("_", " "))
    ax.set_xlabel("broadband coverage (contacts, 1–80 Hz)"); ax.set_ylabel("phase coherence")
    ax.set_title("Phase 0: all recruit ~15/15\nfamilies split on broadband × coherence × duty", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left"); ax.set_xlim(-1, 16); ax.set_ylim(0, 1.05)
    ax.text(0.98, 0.03, "size ∝ duty", transform=ax.transAxes, ha="right", fontsize=6.5, color="0.4")

    # --- Panel 2: per-arm de-conflated readouts ---
    ax = fig.add_subplot(gs[0, 1])
    arms = [("m_off", "ref 16Hz"), ("dyn_tau250_frac0.05", "τ250/5%"), ("dyn_tau250_frac0.1", "τ250/10%"),
            ("dyn_tau750_frac0.05", "τ750/5%"), ("dyn_tau750_frac0.1", "τ750/10%"),
            ("static_K", "static-K\n(peak)"), ("mean_static_K", "static-K\n(mean)"),
            ("dyn_tau750_frac0.1_ext", "τ750/10%\next 9s")]
    def _ro(lab):
        s = verd.get(lab) or cro.get(lab)
        return (s["active_recruitment"], s["broadband_coverage_1_80"], s["phase_coherence"])
    x = np.arange(len(arms)); w = 0.27
    rec = [_ro(l)[0] for l, _ in arms]; bb = [_ro(l)[1] for l, _ in arms]; coh = [_ro(l)[2] for l, _ in arms]
    ax.bar(x - w, rec, w, color="#4c72b0", label="recruit /15")
    ax.bar(x, bb, w, color="#e8a33d", label="broadband /15")
    ax.bar(x + w, [c * 15 for c in coh], w, color="#c44e52", label="coherence (×15)")
    ax.axvspan(4.5, 7.5, color="0.92", zorder=0)                # controls shaded
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in arms], fontsize=6.8, rotation=25, ha="right")
    ax.set_ylabel("contacts /15  (coherence ×15)"); ax.set_ylim(0, 16)
    ax.set_title("Per-arm de-conflated readouts (post-enable)\nfast-τ/10% broadens+desyncs; mean-K stays synchronous → dynamics", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)

    # --- Panel 3: extended termination ---
    ax = fig.add_subplot(gs[0, 2])
    t = np.load(os.path.join(OUT, "arms", "dyn_tau750_frac0.1_ext_trace.npz"), allow_pickle=True)
    rate = np.asarray(t["rate_E"], float); nps = int(1000.0 / DT)
    binned = [rate[s * nps:(s + 1) * nps].mean() for s in range(int(len(rate) / nps))]
    ax.plot(np.arange(len(binned)) + 0.5, binned, "-", color="#c44e52", marker="o", ms=3, label="9s extended")
    ax.axvline(1.0, ls=":", c="0.5", lw=1)                       # adaptation enabled
    ax.axvline(5.0, ls="--", c="0.4", lw=1.2)                    # where the original 5s window ended (in silence)
    ax.text(5.1, 100, "5s window\nended here", fontsize=6.2, color="0.4", va="top")
    ax.set_xlabel("time (s)"); ax.set_ylabel("mean rate (Hz, 1s bins)")
    ax.set_title("slow-τ/10%: extinguish → sparse late\nre-ignition = intermittent, not terminated", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")

    fig.text(0.5, 0.005, "FCXR-HEO2.1 — de-conflated spatial readouts (review P1-a/b/c closeout)", ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "heo2p1_deconflated.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("=== control readouts ===")
    for lab, r in cro.items():
        print(f"  {lab:24s} recruit {r['active_recruitment']}/15 broadband {r['broadband_coverage_1_80']}/15 coh {r['phase_coherence']} seg={r['segment_label']}")
    print("[heo2.1] wrote figures/heo2p1_deconflated.png + phase1_controls.json readouts")


if __name__ == "__main__":
    main()
