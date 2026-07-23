#!/usr/bin/env python
"""Diagnostic figure for the Z/M ictal-carrier gate (run_topic4_zm_ictal_carrier.py). ONE arm, 6 panels
(spec §6.6). The title + caption answer the ONE question directly: is this a train of separated HFO-like
bursts, or one continuous ictal high-frequency-energy macroepisode? NOT paper-ready until a carrier passes.

  (1) core / surround / all-E rate + active-area      [the event itself, source space]
  (2) stacked virtual-SEEG (15 contacts)              [what the electrode sees]
  (3) band-energy dB envelopes (30-80/80-150/1-150)   [is high-freq energy SUSTAINED or intermittent?]
  (4) axis + transverse kymographs (mm)               [local onset -> recruitment, or whole-field flash?]
  (5) z / m / S_G                                      [the slow drivers]
  (6) source-space snapshots: pre-onset/onset/carrier/late
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(d, arm, seed):
    npz = np.load(os.path.join(d, f"{arm}_seed{seed}.npz"), allow_pickle=True)
    jp = os.path.join(d, f"{arm}_seed{seed}.json")
    meta = json.load(open(jp)) if os.path.exists(jp) else {}
    return npz, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", default="sg")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d, meta = _load(a.dir, a.arm, a.seed)
    sm = meta.get("source_metrics", {}); om = meta.get("observed_metrics", {})
    verdict = meta.get("ictal_carrier_verdict", "?"); life = meta.get("lifecycle_verdict", "?")
    is_carrier = verdict in ("candidate_source_only", "candidate_observed_carrier")

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 12, hspace=0.42, wspace=0.9)
    bin_ms = float(d["rate_bin_ms"]); t_r = np.arange(len(d["core_rate"])) * bin_ms

    # (1) rates + active area
    ax = fig.add_subplot(gs[0, 0:4])
    ax.plot(t_r, d["all_rate"], lw=0.6, color="0.6", label="all-E")
    ax.plot(t_r, d["core_rate"], lw=0.9, color="#d62728", label="core")
    ax.plot(t_r, d["surr_rate"], lw=0.8, color="#1f77b4", label="surround")
    ax.set_ylabel("E rate (Hz)"); ax.set_xlabel("t (ms)"); ax.legend(fontsize=6, loc="upper right")
    ax2 = ax.twinx(); ax2.plot(t_r, d["active_frac"], lw=0.8, color="#2ca02c", alpha=0.6)
    ax2.set_ylabel("active frac", color="#2ca02c"); ax2.set_ylim(0, max(0.05, float(d["active_frac"].max()) * 1.1))
    if sm.get("onset_ms") is not None:
        ax.axvline(sm["onset_ms"], color="k", ls=":", lw=0.8)
    ax.set_title(f"(1) source rates + active area\ncore_peak={sm.get('core_peak_hz',0):.0f}Hz "
                 f"all_mean={sm.get('all_mean_hz',0):.2f}Hz", fontsize=8)

    # (2) stacked virtual-SEEG
    ax = fig.add_subplot(gs[0, 4:8])
    lfp = np.asarray(d["lfp"]); fs = float(d["lfp_fs"]); names = [str(x) for x in d["contact_names"]]
    t_l = np.arange(lfp.shape[0]) / fs * 1000.0
    for c in range(lfp.shape[1]):
        v = lfp[:, c].astype(float); v = (v - v.mean()) / (v.std() + 1e-9)
        ax.plot(t_l, v + c * 6.0, lw=0.25, color="k", rasterized=True)
    ax.set_yticks(np.arange(len(names)) * 6.0); ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("t (ms)"); ax.set_title("(2) virtual-SEEG (z-scored, stacked)", fontsize=8)

    # (3) band-energy dB envelopes (best contact + cohort mean)
    ax = fig.add_subplot(gs[0, 8:12])
    ft = np.asarray(d["frame_times_ms"]); bc = names.index(om.get("best_contact", names[0])) if om.get("best_contact") in names else 0
    for key, col, lab in (("lowgamma_db", "#d62728", "30-80"), ("highfreq_db", "#9467bd", "80-150"),
                          ("broadband_db", "#7f7f7f", "1-150")):
        env = np.asarray(d[key])
        ax.plot(ft, env[:, bc], lw=0.9, color=col, label=f"{lab}Hz (best {names[bc]})")
        ax.plot(ft, env.mean(axis=1), lw=0.6, color=col, ls="--", alpha=0.5)
    ax.axhline(6.0, color="k", ls=":", lw=0.8)   # ENH_DB
    ax.set_ylabel("dB re pre-onset"); ax.set_xlabel("t (ms)"); ax.legend(fontsize=5, loc="upper right")
    ax.set_title(f"(3) band power: sustained or intermittent?\nsustained contacts={om.get('n_sustained_contacts',0)}",
                 fontsize=8)

    # (4) kymographs
    for col, key, ekey, lab in ((slice(0, 3), "kymo_axis", "kymo_axis_edges", "axial"),
                                (slice(3, 6), "kymo_transverse", "kymo_transverse_edges", "transverse")):
        ax = fig.add_subplot(gs[1, col])
        ky = np.asarray(d[key]); edg = np.asarray(d[ekey]); kt = np.asarray(d["kymo_t_ms"])
        ax.imshow(ky, aspect="auto", origin="lower", cmap="magma",
                  extent=[kt[0], kt[-1], edg[0], edg[-1]], vmax=np.percentile(ky, 99.5) + 1e-6)
        ax.set_xlabel("t (ms)"); ax.set_ylabel(f"{lab} pos (mm)")
        ax.set_title(f"(4) {lab} kymograph", fontsize=8)

    # (5) z / m / S_G
    ax = fig.add_subplot(gs[1, 6:9])
    zc = np.asarray(d["z_core"]); DTs = 0.1
    if zc.size:
        tz = np.arange(zc.size) * DTs
        ax.plot(tz, zc, lw=0.9, color="#ff7f0e", label="z core")
        ax.plot(tz, np.asarray(d["z_surround"]), lw=0.8, color="#2ca02c", label="z surround")
        ax.plot(tz, np.asarray(d["z_min"]), lw=0.6, color="0.5", ls="--", label="z min")
        ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=6, loc="lower left")
    ax.set_ylabel("z"); ax.set_xlabel("t (ms)")
    ax3 = ax.twinx()
    for key, col, lab in (("m_core", "#9467bd", "m core"), ("SG", "#d62728", "S_G")):
        v = np.asarray(d[key])
        if v.size and float(np.max(v)) > 0:
            ax3.plot(np.arange(v.size) * DTs, v, lw=0.8, color=col, ls=":", label=lab)
    ax3.set_ylim(bottom=0); h, l = ax3.get_legend_handles_labels()
    if h:
        ax3.legend(fontsize=5, loc="upper right")
    ax.set_title("(5) slow drivers z / m / S_G", fontsize=8)

    # (6) snapshots
    snaps = np.asarray(d["snapshots"]); sms = np.asarray(d["snapshot_ms"])
    labs = ["pre-onset", "onset/recruit", "carrier?", "late"]
    vmax = np.percentile(snaps, 99.5) + 1e-6
    for i in range(4):
        ax = fig.add_subplot(gs[2, i * 3:(i + 1) * 3])
        ax.imshow(snaps[i], origin="lower", cmap="inferno", vmax=vmax)
        ax.set_title(f"(6) {labs[i]}\nt={sms[i]:.0f}ms", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    q = "SUSTAINED ictal high-frequency-energy macroepisode" if is_carrier else "HFO-like burst train / no sustained carrier"
    fig.suptitle(f"Z/M ictal-carrier gate — {meta.get('arm', a.arm)} seed{a.seed}   "
                 f"ictal_carrier_verdict = {verdict}   (lifecycle = {life})\n"
                 f"→ {q}    [src macro dur={sm.get('macro',{}).get('duration_ms',0):.0f}ms "
                 f"occ={sm.get('macro',{}).get('occupancy',0):.2f}, "
                 f"obs sustained contacts={om.get('n_sustained_contacts',0)}]", fontsize=11)
    out = a.out or os.path.join(a.dir, "figures", f"carrier_diagnostic_{a.arm}_seed{a.seed}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
