"""FCXR pump-lifecycle diagnostic figures (spec §11 items 1, 3, 5).

These are GATE DIAGNOSTICS, not the paper-ready mechanism figure: the lifecycle candidate figure is
only generated once I-a + T + C + S all pass, and the four-column paper figure with a susceptibility
panel only once E passes and I-b supports a response-mode claim.

One panel = one independent question (CLAUDE.md §7). Missing inputs are skipped with a message
rather than drawn as placeholders.

    instrument_baseline_equivalence.png   is the instrument valid, and does the pump leave the
                                          interictal baseline where it was?
    virtual_seeg_component_audit.png      is the readout pump-separable, and does the direct pump
                                          term carry the band power?
    frozen_topology_and_slow_flow.png     is there a selective exit corridor in the frozen Z x P
                                          plane, and does the high branch drift into it?

Usage: python scripts/plot_topic4_mz_fcxr_pump.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-mz-fcxr-pump")

import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import src.topic4_mz_fcxr_pump as PUMP  # noqa: E402
import src.topic4_mz_fcxr_pump_lifecycle as LC  # noqa: E402

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                   "pump_lifecycle")
FIG = os.path.join(OUT, "figures")
C_OK, C_BAD, C_MID, C_GREY = "#2a7f62", "#b5432f", "#c9922e", "#7a7a7a"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight"})


def _load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_instrument():
    calib = _load("pump_baseline_calibration.json")
    equiv = _load("pump_baseline_equivalence.json")
    if calib is None:
        print("[fig1] no calibration artifact - skipped")
        return
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))

    # (a) which load candidates make an ordinary event visibly move the load without pinning it
    rows = calib["candidate_grid"]
    taus = sorted({r["tau_N"] for r in rows})
    marks = {500.0: "o", 1000.0: "s", 2000.0: "^"}
    for t in taus:
        rr = [r for r in rows if r["tau_N"] == t]
        x = [r["phi_q99"] for r in rr]
        y = [r["visibility_ratio"] for r in rr]
        col = [C_OK if r["admissible"] else C_BAD for r in rr]
        ax[0].scatter(x, y, c=col, marker=marks.get(t, "o"), s=46, edgecolor="k", linewidth=0.4,
                      label=f"tau_N = {t:.0f} ms", zorder=3)
    ax[0].axhline(3.0, color=C_GREY, ls="--", lw=0.9)
    ax[0].axvline(0.90, color=C_GREY, ls="--", lw=0.9)
    ax[0].text(0.02, 3.4, "visibility floor (3x)", color=C_GREY, fontsize=6.5)
    ax[0].text(0.905, ax[0].get_ylim()[1] * 0.55, "pump\npinned", color=C_GREY, fontsize=6.5)
    ch = calib.get("chosen_candidate")
    if ch:
        ax[0].scatter([ch["phi_q99"]], [ch["visibility_ratio"]], s=190, facecolor="none",
                      edgecolor="k", linewidth=1.3, zorder=4)
        ax[0].annotate(f"selected\ntau_N={ch['tau_N']:.0f} ms\na_load={ch['a_load']:.4f}",
                       (ch["phi_q99"], ch["visibility_ratio"]), textcoords="offset points",
                       xytext=(-12, -34), fontsize=6.5, ha="center")
    ax[0].set_xlabel("99th pct of per-cell pump activation")
    ax[0].set_ylabel("event rise / matched quiet change")
    ax[0].set_title("(a) load candidates: visible, not pinned", fontsize=8.5, loc="left")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    ax[0].set_xlim(0, 1.05)

    # (b) held-out equivalence:每 metric 的 (on - off) 相对其预锁 margin
    if equiv is None:
        ax[1].text(0.5, 0.5, "held-out equivalence not run yet", ha="center", va="center",
                   transform=ax[1].transAxes, color=C_GREY)
        ax[1].set_axis_off()
    else:
        per = equiv["equivalence"]["per_metric"]
        keys = [k for k in sorted(per) if "delta" in per[k]]
        rel = [per[k]["delta"] / per[k]["margin"] if per[k]["margin"] > 0 else 0.0 for k in keys]
        cols = [(C_MID if per[k].get("underpowered") else (C_OK if per[k]["within"] else C_BAD))
                for k in keys]
        yy = np.arange(len(keys))
        ax[1].hlines(yy, 0, rel, color=cols, lw=2.2)
        ax[1].scatter(rel, yy, c=cols, s=26, zorder=3)
        ax[1].axvline(0, color="k", lw=0.7)
        for s in (-1, 1):
            ax[1].axvline(s, color=C_GREY, ls="--", lw=0.9)
        ax[1].set_yticks(yy)
        ax[1].set_yticklabels([k.replace("bandpower_1_80_", "band ").replace("_", " ") for k in keys],
                              fontsize=6.3)
        ax[1].set_xlabel("(pump on - pump off) / pre-locked margin")
        n_up = equiv["equivalence"].get("n_underpowered", 0)
        ax[1].set_title(f"(b) held-out baseline equivalence  ({n_up} underpowered)",
                        fontsize=8.5, loc="left")
        lim = max(1.6, 1.15 * max(abs(np.array(rel))) if rel else 1.6)
        ax[1].set_xlim(-lim, lim)

    # (c) did the load stay equilibrated and the pump current stay centred on zero?
    tp = os.path.join(OUT, "heldout_traces_noise202.npz")
    if not os.path.exists(tp):
        ax[2].text(0.5, 0.5, "held-out traces not written yet", ha="center", va="center",
                   transform=ax[2].transAxes, color=C_GREY)
        ax[2].set_axis_off()
    else:
        d = np.load(tp, allow_pickle=True)
        dt = float(d["dt"])
        u = np.asarray(d["u_mean_on"], float)
        ex = np.asarray(d["pump_excess_mean_on"], float)
        t = np.arange(u.size) * dt / 1000.0
        ax[2].plot(t, u, color=C_OK, lw=0.9, label="mean load u")
        ax[2].set_ylabel("mean load u", color=C_OK)
        ax[2].tick_params(axis="y", labelcolor=C_OK)
        ax2 = ax[2].twinx()
        ax2.plot(np.arange(ex.size) * dt / 1000.0, ex, color=C_BAD, lw=0.7, alpha=0.85)
        ax2.axhline(0.0, color="k", lw=0.6, ls=":")
        ax2.set_ylabel("mean pump excess current", color=C_BAD)
        ax2.tick_params(axis="y", labelcolor=C_BAD)
        ax2.spines["top"].set_visible(False)
        ax[2].set_xlabel("time (s)")
        ax[2].set_title("(c) held-out: load equilibrated, pump centred", fontsize=8.5, loc="left")
    fig.savefig(os.path.join(FIG, "instrument_baseline_equivalence.png"))
    plt.close(fig)
    print("[fig1] instrument_baseline_equivalence.png")


def fig_components():
    tp = os.path.join(OUT, "heldout_traces_noise202.npz")
    if not os.path.exists(tp):
        print("[fig2] no held-out traces - skipped")
        return
    d = np.load(tp, allow_pickle=True)
    dt = float(d["dt"])
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))

    # (a) what the separated signed components actually look like (one contact, 400 ms)
    n0 = int(3000 / dt)
    n1 = n0 + int(400 / dt)
    t = (np.arange(n0, n1) * dt - n0 * dt)
    for key, col, lab in (("on_seeg_legacy_abs", C_GREY, "legacy |I_E|+|I_I|"),
                          ("on_seeg_no_direct_pump", C_OK, "no_direct_pump (signed)"),
                          ("on_seeg_pump", C_BAD, "pump component")):
        y = np.asarray(d[key], float)[n0:n1, 0]
        ax[0].plot(t, y, color=col, lw=0.8, label=lab)
    ax[0].axhline(0, color="k", lw=0.5, ls=":")
    ax[0].set_xlabel("time within window (ms)")
    ax[0].set_ylabel("virtual-SEEG proxy (a.u.)")
    ax[0].set_title("(a) signed component separation, contact 0", fontsize=8.5, loc="left")
    ax[0].legend(frameon=False, fontsize=6.3)

    # (b) does the DIRECT pump term carry the 1-80 Hz power? (contamination check)
    comps = ["legacy_abs", "no_direct_pump", "pump", "all_components"]
    w = 0.38
    xx = np.arange(len(comps))
    for k, (pre, col, lab) in enumerate((("off", C_GREY, "pump off"), ("on", C_OK, "pump on"))):
        vals = [PUMP.band_power(np.asarray(d[f"{pre}_seeg_{c}"], float), dt, (1.0, 80.0))
                for c in comps]
        ax[1].bar(xx + (k - 0.5) * w, np.maximum(vals, 1e-18), w, color=col, label=lab)
    ax[1].set_yscale("log")
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels([c.replace("_", "\n") for c in comps], fontsize=6.5)
    ax[1].set_ylabel("1-80 Hz power (proxy units)")
    ax[1].set_title("(b) where the band power lives", fontsize=8.5, loc="left")
    ax[1].legend(frameon=False, fontsize=6.5)
    fig.savefig(os.path.join(FIG, "virtual_seeg_component_audit.png"))
    plt.close(fig)
    print("[fig2] virtual_seeg_component_audit.png")


def fig_topology():
    maps = sorted(glob.glob(os.path.join(OUT, "frozen_topology_map_*.json")))
    if not maps:
        print("[fig3] no topology map - skipped")
        return
    cells = []
    for m in maps:
        cells += json.load(open(m))["cells"]
    fields = sorted({c["field"] for c in cells})
    fig, ax = plt.subplots(1, len(fields), figsize=(4.6 * len(fields), 3.6), squeeze=False)
    style = {"INTERICTAL_WORKPOINT": (C_OK, "o"), "ELEVATED_EVENT_TRAIN": (C_OK, "s"),
             "METASTABLE_TRANSIENT": (C_MID, "v"), "FINITE_HIGH_ORBIT": (C_BAD, "^"),
             "FINITE_HIGH_FIXED": (C_BAD, "D")}
    for j, fk in enumerate(fields):
        a = ax[0][j]
        rows = [c for c in cells if c["field"] == fk]
        for c in rows:
            if c["ic"] != "high":
                continue
            col, mk = style.get(c["label"], (C_GREY, "x"))
            a.scatter([c["P"]], [c["D"]], color=col, marker=mk, s=95, edgecolor="k", linewidth=0.4,
                      zorder=3)
            f = c["slow_flow"]
            a.annotate("", xy=(c["P"] + 0.06 * np.sign(f["dP_dt"]), c["D"]), xytext=(c["P"], c["D"]),
                       arrowprops=dict(arrowstyle="->", color="k", lw=0.7, alpha=0.6))
        for c in rows:
            if c["ic"] == "low" and not LC._is_low(c["label"]):
                a.scatter([c["P"]], [c["D"]], facecolor="none", edgecolor=C_BAD, s=210,
                          linewidth=1.4, zorder=2)
        v = LC.adjudicate_gate_T(cells, field=fk)
        if v.get("exit"):
            a.axvline(v["exit"]["P"], color=C_GREY, ls="--", lw=1.0)
        a.set_xlabel("mean excess pump activation  P = mean[phi(u) - p0]")
        a.set_ylabel("frozen inhibition-depletion scale D" if j == 0 else "")
        a.set_title(f"{fk} field -> {v['status']}", fontsize=8.5, loc="left")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markeredgecolor="k",
                          markeredgewidth=0.4, label=k.replace("_", " ").lower())
               for k, (c, m) in style.items()]
    handles.append(plt.Line2D([], [], color=C_BAD, marker="o", ls="", markerfacecolor="none",
                              markersize=9, label="low branch NOT preserved"))
    ax[0][-1].legend(handles=handles, frameon=False, fontsize=6.2, loc="center left",
                     bbox_to_anchor=(1.02, 0.5))
    fig.savefig(os.path.join(FIG, "frozen_topology_and_slow_flow.png"))
    plt.close(fig)
    print("[fig3] frozen_topology_and_slow_flow.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_instrument()
    fig_components()
    fig_topology()


if __name__ == "__main__":
    main()
