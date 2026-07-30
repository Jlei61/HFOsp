"""FCXR-ION B0-B2 gate diagnostics.

Three figures, one per gate that has actually produced output.  Missing input -> the figure is
SKIPPED, never drawn as a placeholder (plan §12).

These are GATE DIAGNOSTICS, not SNN mechanism figures: the four-column
`mechanism | tempA source | tempB source | electrode readout` standard in
docs/figure_style_guide.md §Topic 4 explicitly exempts diagnostic figures, and nothing here is a
mechanism or paper-ready claim.  Lifecycle candidate figures and the four-column paper figure are
NOT generated -- B3/B4 are not authorised.

Panel discipline (CLAUDE.md §7): every panel answers one independent question; quantities that are
exactly 0 or exactly 3.0/2.0 are annotated rather than given a bar of their own.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                   "ion_homeostasis")
FIG = os.path.join(OUT, "figures")
OK, BAD, MID = "#2a7f62", "#b5292f", "#4a6fa5"


def _load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_b0():
    feas, units = _load("b0_analytic_feasibility.json"), _load("b0_voltage_unit_audit.json")
    if not (feas and units):
        print("[plot] skip b0_feasibility_and_units (missing input)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw=dict(width_ratios=[1.05, 1]))

    # Q1: how large is the potassium effect at each candidate scale, interictal vs high rate?
    rates = ["interictal", "20hz", "50hz"]
    xlab = [f"interictal\n({feas['r0_hz']:.2f} Hz)", "sustained\n20 Hz", "sustained\n50 Hz"]
    col = {0.25: "0.72", 0.5: "#7fb3d5", 1.0: BAD, 2.0: "#1f4e79", 4.0: "0.72"}
    for r in feas["rows"]:
        y = [r["dE_K_interictal_mV"], r["dE_K_20hz_mV"], r["dE_K_50hz_mV"]]
        cand, prim = r["in_candidate_set"], r["is_primary"]
        ax[0].plot(range(3), y, marker="o", color=col[r["f_prime"]],
                   lw=2.8 if prim else 1.8, ms=8 if prim else 5.5,
                   ls="-" if cand else "--", zorder=3 if prim else 2,
                   label=f"f'={r['f_prime']}" + (" (primary)" if prim else
                                                 ("" if cand else " (reference only)")))
    ax[0].axhline(0.3 * 18.0, color="k", ls=":", lw=1.2)
    ax[0].text(0.02, 0.3 * 18.0 + 0.4, "30% of V_th", fontsize=8, color="k")
    ax[0].set_xticks(range(3), xlab)
    ax[0].set_ylabel(r"$\Delta E_K$  (mV)")
    ax[0].set_title("potassium reversal shift vs firing rate", fontsize=11)
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    sec = ax[0].secondary_yaxis("right", functions=(lambda v: 100 * v / 18.0,
                                                    lambda v: v * 18.0 / 100))
    sec.set_ylabel("% of V_th")
    ax[0].grid(alpha=0.25)
    ax[0].text(0.98, 0.03,
               f"$\\tau_{{Na}}$={feas['tau_Na_s']:.1f} s   $\\tau_{{K_o}}$="
               f"{feas['tau_Ko_s']:.3f} s   ratio {feas['tau_ratio']:.0f}x",
               transform=ax[0].transAxes, ha="right", fontsize=8.5,
               bbox=dict(fc="white", ec="0.7", alpha=0.9))

    # Q2: is injecting an mV-dimensioned term into `drive` self-consistent?
    ax[1].axis("off")
    ax[1].set_title(f"engine voltage-unit chain — {units['status']}", fontsize=11)
    import textwrap
    for i, c in enumerate(units["chain"]):
        y = 0.98 - i * 0.138
        ax[1].text(0.0, y, "PASS" if c["ok"] else "FAIL", color=OK if c["ok"] else BAD,
                   fontsize=8.5, fontweight="bold", family="monospace")
        ax[1].text(0.10, y, c["step"], fontsize=8.5, va="baseline")
        ax[1].text(0.10, y - 0.038, "\n".join(textwrap.wrap(c["evidence"], 80)[:3]),
                   fontsize=6.3, color="0.35", va="top", linespacing=1.3)
    ax[1].text(0.0, -0.075, "Dimension only: g_K_ion = 1 is a declared normalization\n"
                           "(calibrated in B3), NOT a conclusion of this audit.",
               fontsize=8, style="italic", color="0.25")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "b0_feasibility_and_units.png"), dpi=190)
    plt.close(fig)
    print("[plot] b0_feasibility_and_units.png")


def fig_gate_h():
    gh = _load("gate_H.json")
    if not gh:
        print("[plot] skip gate_H_homeostasis (missing input)")
        return
    c = gh["per_network"][gh["primary_network"]]["checks"]
    r = c["heterogeneous_init_residual"]
    fig, ax = plt.subplots(1, 3, figsize=(14.8, 5.0))

    # Q1: is rest a fixed point -- including on an empty voxel -- or only by cancellation?
    labels = ["resting\nfixed point", "EMPTY-voxel\nfixed point"]
    got = [max(c["resting_fixed_point"]["max_abs_Na_drift"],
               c["resting_fixed_point"]["max_abs_K_drift"]),
           c["empty_voxel_fixed_point"]["max_abs_K_drift"]]
    broken = c["empty_voxel_fixed_point"]["broken_form_would_give_mM_s"]
    floor = 1e-18
    ax[0].bar(labels, [max(g, floor) for g in got], color=OK, width=0.5, label="rev4 deviation form")
    ax[0].axhline(broken, color=BAD, ls="--", lw=1.8,
                  label=f"broken forms: {broken:.5f} mM/s")
    ax[0].set_yscale("log")
    ax[0].set_ylim(floor / 3, broken * 4e4)
    ax[0].set_ylabel("|drift| after 400 ion blocks  (mM  /  mM·s$^{-1}$)")
    ax[0].set_title("rest is a fixed point by construction", fontsize=11)
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    ax[0].text(0.5, 0.06, "measured: exactly 0.0", transform=ax[0].transAxes,
               ha="center", fontsize=8.5, color=OK, fontweight="bold")
    ax[0].text(0.5, 0.40, f"K budget closure rel. err   "
                          f"{c['k_budget_closure']['relative_error']:.1e}\n"
                          f"zero-flux net flux            "
                          f"{c['zero_flux_boundary']['diffusion_net_flux']:.1e}\n"
                          f"pump stoichiometry            "
                          f"{c['pump_stoichiometry']['Na_coefficient']:.1f} : "
                          f"{c['pump_stoichiometry']['K_coefficient_over_beta']:.1f}·β",
               transform=ax[0].transAxes, ha="center", fontsize=7.6, family="monospace",
               bbox=dict(fc="white", ec="0.7", alpha=0.95))

    # Q2: does the heterogeneous initializer actually remove what the scalar one leaves?
    stats = ["q95", "q99", "max"]
    het = [r[f"{s}_abs_dNa_dt"] for s in stats]
    hetk = [r[f"{s}_abs_dKo_dt"] for s in stats]
    x = np.arange(3)
    ax[1].bar(x - 0.2, het, 0.38, color=MID, label=r"heterogeneous  $|dNa_i/dt|$")
    ax[1].bar(x + 0.2, hetk, 0.38, color=OK, label=r"heterogeneous  $|dK_{o,g}/dt|$")
    ax[1].axhline(r["threshold"], color="k", ls=":", lw=1.4, label="gate  1e-6 mM/s")
    ax[1].axhline(r["scalar_init_q99_abs_dNa_dt"], color=BAD, ls="--", lw=1.8,
                  label=f"scalar init q99: {r['scalar_init_q99_abs_dNa_dt']:.2e}")
    ax[1].set_xticks(x, stats)
    ax[1].set_yscale("log")
    ax[1].set_ylim(min(het + hetk) / 40, r["scalar_init_q99_abs_dNa_dt"] * 3e3)
    ax[1].set_ylabel("residual  (mM/s)")
    ax[1].set_title("initialization residual — per cell / per voxel", fontsize=11)
    ax[1].legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
    ax[1].set_xlabel("population means are reported but never a pass criterion",
                     fontsize=8, style="italic", color="0.4")

    # Q3: is the discretisation converged in time and in space?
    conv = c["dt_ion_convergence"]
    ax[1].set_axisbelow(True)
    ax[2].plot([2.0, 1.0, 0.5], [conv["coarse"], conv["mid"], conv["fine"]], "o-",
               color=MID, lw=2, ms=7)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].invert_xaxis()
    ax[2].set_ylabel("max |K difference| vs next-finer step (mM)")
    ax[2].set_title("time-step and grid convergence", fontsize=11)
    ax[2].grid(alpha=0.25, which="both")
    g = c["grid_consistency"]
    ax[2].text(0.03, 0.06, f"grid {g['grids'][0]} vs {g['grids'][1]}:\n"
                           f"  total K content  {g['total_K_rel_diff']:.1e}\n"
                           f"  coarse-grained field  {g['coarse_grained_max_rel_dev']:.1e}\n"
                           f"checkpoint/restart: bit-identical",
               transform=ax[2].transAxes, fontsize=8,
               bbox=dict(fc="white", ec="0.7", alpha=0.92))
    ax[2].set_xlabel(r"$dt_{ion}$ (ms), coarser $\rightarrow$ finer" + "\n"
                     + "numerics only — dt-halving agreement is not evidence\nthe equations are right",
                     fontsize=8.5)
    fig.suptitle(f"Gate H = {gh['status']}   (primary tier {gh['primary_network']}; "
                 f"{gh['primary_network']} and n1000 both {gh['status']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "gate_H_homeostasis.png"), dpi=190)
    plt.close(fig)
    print("[plot] gate_H_homeostasis.png")


def fig_gate_b():
    gb = _load("gate_B.json")
    if not gb:
        print("[plot] skip gate_B_interictal_substrate (missing input)")
        return
    per = gb["per_trajectory"]
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # Q1: did the ion substrate come back to the accepted interictal working point?
    names = [k for k in per[0]["tolerance"] if not per[0]["tolerance"][k]["underpowered"]]
    x = np.arange(len(names))
    for i, n in enumerate(names):
        m = per[0]["tolerance"][n]["margin"]
        ax[0].add_patch(plt.Rectangle((i - 0.4, -1), 0.8, 2, fc="0.88", ec="none", zorder=1))
        vals = [p["tolerance"][n]["delta"] / m for p in per]
        ax[0].scatter([i] * len(vals), vals, s=42, zorder=3,
                      color=[OK if abs(v) <= 1 else BAD for v in vals])
    ax[0].axhline(0, color="k", lw=0.9)
    ax[0].axhline(1, color="0.4", ls=":", lw=1)
    ax[0].axhline(-1, color="0.4", ls=":", lw=1)
    ax[0].set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=7.5)
    ax[0].set_ylabel("deviation from the accepted arm / tolerance")
    ax[0].set_title("interictal metrics vs the pump-off arm", fontsize=11)
    ax[0].text(0.98, 0.03, "binding metrics only;\nUNDERPOWERED ones excluded",
               transform=ax[0].transAxes, ha="right", fontsize=7.5, style="italic", color="0.35")

    # Q2: do events still start at BOTH registered cores?
    tags = [p["tag"] or f"{p['conn_seed']}/{p['noise_seed']}" for p in per]
    fa = [p["direction"]["frac_A"] for p in per]
    fb = [p["direction"]["frac_B"] for p in per]
    y = np.arange(len(per))
    ax[1].barh(y, fa, 0.6, color=MID, label="core_A")
    ax[1].barh(y, fb, 0.6, left=fa, color="#d9a441", label="core_B")
    ax[1].axvline(gb["thresholds"]["min_frac"], color=BAD, ls="--", lw=1.5)
    ax[1].axvline(1 - gb["thresholds"]["min_frac"], color=BAD, ls="--", lw=1.5)
    for i, p in enumerate(per):
        ax[1].text(1.02, i, f"n={p['direction']['n_scoreable']}", va="center", fontsize=7.5,
                   color=OK if p["direction"]["ok"] else BAD)
    ax[1].set_yticks(y, tags, fontsize=7.5)
    ax[1].set_xlim(0, 1.14)
    ax[1].set_xlabel("share of scoreable events")
    ax[1].set_title("initiation site — both registered cores", fontsize=11)
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")

    # Q3: is there a slow ion countdown hiding under a flat population mean?
    q99na = [p["ion"]["q99_abs_dNa_dt"] for p in per]
    q99k = [p["ion"]["q99_abs_dKo_dt"] for p in per]
    net = [p["ion"]["net_Na_drift_mM_s"] for p in per]
    ax[2].scatter(q99na, net, s=52, color=MID, label="per-cell q99 vs net drift")
    ax[2].scatter(q99k, net, s=52, marker="^", color=OK, label="per-voxel q99 vs net drift")
    ax[2].axvline(gb["thresholds"]["ion_block_drift_max"], color=BAD, ls="--", lw=1.5,
                  label="block-drift gate")
    ax[2].axhline(gb["thresholds"]["ion_net_drift_max"], color="0.4", ls=":", lw=1.4,
                  label="net-drift gate")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("inter-block |d/dt| q99  (mM/s)")
    ax[2].set_ylabel("net Na drift over the window  (mM/s)")
    ax[2].set_title("no slow countdown", fontsize=11)
    ax[2].legend(fontsize=7.5, frameon=False, loc="best")
    ax[2].grid(alpha=0.25, which="both")
    ax[2].text(0.5, 1.06, "11 s = 0.2 $\\tau_{Na}$: stability near the initialization point, "
                          "not steady state", transform=ax[2].transAxes, ha="center",
               fontsize=7.5, style="italic", color="0.4")
    fig.suptitle(f"Gate B = {gb['status']}   "
                 f"(B-real {gb['b_real']['n_direction_passing']}/"
                 f"{gb['b_real']['n_trajectories']} trajectories; B-model ok="
                 f"{gb['b_model']['ok']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "gate_B_interictal_substrate.png"), dpi=190)
    plt.close(fig)
    print("[plot] gate_B_interictal_substrate.png")


def fig_t7():
    """T7 sensor diagnostic.  This is the run that stopped the sprint, so it gets a figure even
    though it is not one of the three gate figures plan §12 lists -- the plan's list assumed T7
    would not be the terminal result."""
    fs = _load("b1_f_selection_v2.json") or _load("b1_f_selection.json")
    aud = _load("b1_gate_reference_audit.json")
    tp = os.path.join(OUT, "b1_f_selection_traces.npz")
    if not (fs and aud and os.path.exists(tp)):
        print("[plot] skip t7_sensor_diagnostic (missing input)")
        return
    z = np.load(tp)
    G = fs.get("hard_gates") or fs["gates_definition"]
    v2 = "hard_gates" in fs
    rows = sorted(fs["rows"], key=lambda r: r["f_prime"])
    fps = [r["f_prime"] for r in rows]
    col = {0.5: "#7fb3d5", 1.0: BAD, 2.0: "#1f4e79"}
    fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.7))

    # Q1: is there a usable amplitude window, and is any candidate inside it?
    dK = [r["measured"]["dK_peak_single_mM"] for r in rows]
    floor = (G["measurable_abs_floor_mM"] if v2 else rows[0]["gates"]["measurable"]["threshold"])
    ax[0].axhspan(floor, G["safe_ceiling_mM"], color=OK, alpha=0.16, zorder=0)
    ax[0].axhline(floor, color="k", ls=":", lw=1.4)
    ax[0].axhline(G["safe_ceiling_mM"], color="k", ls="--", lw=1.4)
    for r, v in zip(rows, dK):
        per = r["measured"]["dK_peak_per_event_mM"]
        ax[0].scatter([r["f_prime"]] * len(per), per, s=16, color=col[r["f_prime"]], alpha=0.4,
                      zorder=2)
        ax[0].scatter([r["f_prime"]], [v], s=140, color=col[r["f_prime"]], zorder=3,
                      edgecolor="k", linewidth=0.8)
    ax[0].text(0.52, floor * 1.06, f"measurable floor {floor:.2f}"
               + (" (absolute only, T7.1)" if v2 else ""), fontsize=7.5)
    ax[0].text(0.52, G["safe_ceiling_mM"] * 1.03, f"safe ceiling {G['safe_ceiling_mM']}",
               fontsize=7.5)
    ax[0].set_xticks(fps)
    ax[0].set_xlabel("f'")
    ax[0].set_ylabel(r"single-event peak $\Delta K_o$  (mM)")
    ax[0].set_title("amplitude window (dots = 22 real events)", fontsize=10.5)
    ax[0].grid(alpha=0.22)

    # Q2: is the Na gate's reference in the right place?
    wp = {r["f_prime"]: r for r in aud["rows"]}
    x = np.arange(len(fps))
    meas = [r["measured"]["na_excess_decay_frac_20s"] for r in rows]
    pred = [wp[f]["decay_20s_at_working_point"] for f in fps]
    ax[1].bar(x - 0.28, meas, 0.26, color=[col[f] for f in fps], label="measured (coupled)")
    clamp = [r["measured"].get("na_decay_frac_k_clamped", np.nan) for r in rows]
    ax[1].bar(x, clamp, 0.26, color="none", edgecolor=OK, lw=1.6,
              label=r"measured, $K_o$ clamped at $K_o^*(f')$")
    pred2 = [r["measured"].get("na_decay_pred_coupled", np.nan) for r in rows]
    ax[1].bar(x + 0.28, pred2 if v2 else pred, 0.26, color="none", edgecolor="k", hatch="///",
              label="coupled-Jacobian prediction")
    if not v2:
        ax[1].axhspan(*G["na_decay_band"], color=BAD, alpha=0.14)
    ax[1].axhline(aud["gate_reference"]["decay_20s"], color=BAD, ls="--", lw=1.8,
                  label="retired band centre (linearised at REST)")
    if v2:
        ax[1].axhline(G["na_net_decay_min_frac"], color=OK, ls=":", lw=1.6,
                      label=f"T7.1: net decay >= {G['na_net_decay_min_frac']}")
    ax[1].set_xticks(x, [f"f'={f}" for f in fps])
    ax[1].set_ylabel("event-induced Na excess decayed in 20 s")
    ax[1].set_title("Na recovery: reference and the K-clamp control", fontsize=10.5)
    ax[1].legend(fontsize=6.2, frameon=False, loc="upper center", ncol=2, columnspacing=0.8,
                 bbox_to_anchor=(0.5, 1.02))
    ax[1].set_ylim(0, 2.55)

    # Q3: why does the monotonicity clause fail?
    st = float(z["f1.0_trace_stride_blocks"]) * float(z["f1.0_dt_ion_ms"]) * 1e-3
    for f in fps:
        e = z[f"f{f}_na_excess"]
        ax[2].plot(np.arange(e.size) * st, e, color=col[f], lw=1.3, label=f"f'={f}")
        kc = f"f{f}_na_excess_k_clamped"
        if kc in z:
            ec = z[kc]
            ax[2].plot(np.arange(ec.size) * st, ec, color=col[f], lw=1.0, ls=":",
                       label=f"f'={f}, $K_o$ clamped")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("time after the event (s)")
    ax[2].set_ylabel("median event-induced Na excess (mM)")
    ax[2].set_title("Na excess: decay vs the monotonicity clause", fontsize=10.5)
    ax[2].legend(fontsize=6.2, frameon=False, loc="upper right", ncol=2, columnspacing=0.8)
    ax[2].grid(alpha=0.22)
    e1 = z["f1.0_na_excess"]
    rough = float(np.mean(np.diff(e1) > 0))
    ax[2].text(0.97, 0.42, f"f'=1.0: {100*rough:.0f}% of samples step UP; largest up-step\n"
                           f"is 16% of the peak (background events in the\nreplay land on the same "
                           f"cells). Smoothing to 1/2/5/10 s\ndoes NOT remove them. But NO candidate "
                           f"shows a net\nrise from peak to 20 s -- the clause fails on its\n"
                           f"zero tolerance, not on re-accumulation.",
               transform=ax[2].transAxes, ha="right", va="top", fontsize=6.6,
               bbox=dict(fc="white", ec="0.7", alpha=0.92))

    # Q4: does potassium accumulate across 200 ms-spaced events?
    tr = z["f1.0_cluster_K"]
    ax[3].plot(np.arange(tr.size) * float(z["f1.0_dt_ion_ms"]) * 1e-3, tr, color=BAD, lw=1.3)
    pk = rows[fps.index(1.0)]["measured"]["integration_peaks_mM"]
    lin = [pk[0] * sum(np.exp(-0.2 * j / 0.5704) for j in range(k + 1)) for k in range(5)]
    for k in range(5):
        ax[3].plot(0.2 * k + 0.1, pk[k], "o", color=BAD, ms=7, zorder=3)
        ax[3].plot(0.2 * k + 0.1, lin[k], "^", color="0.35", ms=7, zorder=3)
    ax[3].plot([], [], "o", color=BAD, label="measured peak")
    ax[3].plot([], [], "^", color="0.35", label=r"linear superposition ($\tau_{K_o}$ at workpoint)")
    ax[3].set_xlabel("time (s), five identical events 200 ms apart")
    ax[3].set_ylabel(r"$\Delta K_o$ at the event voxel (mM)")
    ax[3].set_title("K accumulation: SUB-linear (open loop, non-blocking)", fontsize=10.5)
    ax[3].legend(fontsize=7.5, frameon=False, loc="lower right")
    ax[3].grid(alpha=0.22)
    ax[3].text(0.03, 0.95, "increments 0.235 / 0.139 / 0.080 / 0.045\n"
                           "linear would give 0.451 / 0.318 / 0.224 / 0.158",
               transform=ax[3].transAxes, va="top", fontsize=7,
               bbox=dict(fc="white", ec="0.7", alpha=0.92))

    fig.suptitle(f"T7 sensor diagnostic — {fs['status']} "
                 f"(g_K_ion = 0: the K→E_K→firing loop is CUT, so panels 1 and 4 characterise the "
                 f"clearance side only)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIG, "t7_sensor_diagnostic.png"), dpi=185)
    plt.close(fig)
    print("[plot] t7_sensor_diagnostic.png")


def fig_b2_1():
    """B2.1 calibration-instrument repair.  Four independent questions (CLAUDE.md 7):

    1  does the self-consistent iteration converge?          -> convergence quantity + r_E
    2  is the ion field stationary?                          -> signed slope ladder vs bounds
    3  does the K feedback change the 2nd event in TIME?     -> hot-voxel excess K, both arms
    4  does it change the 2nd event in SPACE?                -> per-voxel excess-K difference map
    """
    sc, mc = _load("b2_1_selfconsistent.json"), _load("b2_1_matched_control.json")
    if sc is None or mc is None or "spatial_extent" not in mc:
        print("[plot] b2_1_calibration_repair.png SKIPPED (inputs missing)")
        return
    import src.topic4_fcxr_ion as ION                                        # noqa: E402
    import run_topic4_fcxr_ion as RUN                                        # noqa: E402
    h = sc["history"]
    it = [x["iteration"] for x in h]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    ax = axes.ravel()

    # 1 -- does the iteration converge?
    a = ax[0]
    for key, lab, c, ls in (("max_rel", "max  (spec 2.2 gate statistic)", BAD, "-"),
                            ("q99_rel", "q99", MID, "--"), ("q95_rel", "q95", "#8a8a8a", ":")):
        a.plot(it, [x["change"][key] for x in h], "o" + ls, color=c, lw=1.8, ms=5, label=lab)
    a.axhline(ION.B2_1_RATE_REL_TOL, color=OK, lw=1.5, ls="-.",
              label=f"convergence gate {ION.B2_1_RATE_REL_TOL}")
    a.set_yscale("log"); a.set_xticks(it)
    a.set_ylim(bottom=ION.B2_1_RATE_REL_TOL / 3.6)
    a.set_xlabel("update  k"); a.set_ylabel(r"damped step  $|r^{(k+1)}-r^{(k)}|$ / mean(r)")
    a.legend(fontsize=7.2, loc="lower center", ncol=2, framealpha=0.93)
    b = a.twinx()
    b.plot(it, [x["mean_rate_E_hz"] for x in h], "s-", color="#c98a20", lw=1.7, ms=6)
    b.set_ylabel(r"population $r_E$  (Hz)  — orange squares", color="#c98a20")
    b.tick_params(axis="y", colors="#c98a20")
    lo = min(x["mean_rate_E_hz"] for x in h); hi = max(x["mean_rate_E_hz"] for x in h)
    b.set_ylim(-0.30 * hi, 1.22 * hi)
    a.set_title(f"1  self-consistent iteration — {sc['status']}\n"
                f"even q95 stays far above the gate, so this is not a sparse-tail artefact;\n"
                fr"$r_E$ swings {lo:.2f}–{hi:.2f} Hz at frozen bias", fontsize=9)

    # 2 -- is the ion field stationary?
    a = ax[1]
    floor = ION.B2_1_SLOPE_BOUND_K / 30.0
    for i, (spec, bound, c) in enumerate((("slope_Na", ION.B2_1_SLOPE_BOUND_NA, "#3b6ea5"),
                                          ("slope_K", ION.B2_1_SLOPE_BOUND_K, "#a5533b"))):
        x = np.arange(len(it)) + (i - 0.5) * 0.34
        nm = spec.split("_")[1]
        a.vlines(x, [e[spec]["q95_abs"] for e in h], [e[spec]["max_abs"] for e in h],
                 color=c, lw=6, alpha=0.30)
        a.plot(x, [e[spec]["q99_abs"] for e in h], "o", color=c, ms=6.5)
        a.plot(x, [max(abs(e[spec]["mean_signed"]), floor) for e in h], "x", color=c, ms=8, mew=2)
        a.axhline(bound, color=c, ls="--", lw=1.4)
        a.text(len(it) - 0.62, bound * 1.18, f"{nm} stationarity bound", color=c, fontsize=7)
    a.set_yscale("log")
    a.set_ylim(floor * 0.6, 4.5 * max(max(e[s2]["max_abs"] for e in h)
                                      for s2 in ("slope_Na", "slope_K")))
    a.set_xticks(np.arange(len(it))); a.set_xticklabels(it)
    a.set_xlabel("update  k"); a.set_ylabel("|signed secular slope|  (mM/s)")
    a.text(0.5, 0.985, "bar = q95 to max        \u25cf per-cell q99        "
                       "\u2715 |population mean|", transform=a.transAxes, ha="center", va="top",
           fontsize=7.6, bbox=dict(fc="white", ec="0.75", alpha=0.93))
    a.set_title("2  ion stationarity — corrected estimator, still FAILS\n"
                "the per-cell distribution sits ABOVE the bound while the population mean sits\n"
                "BELOW it: cells drift in opposite directions and cancel", fontsize=9)

    # 3 -- does the feedback change the 2nd window's burst in time?
    a = ax[2]
    t_k, sp_ms = RUN.CL_PROBE_T_KICK_MS, RUN.CL_PROBE_SPACING_MS
    bs = mc.get("burst_structure", {})
    for tag, c in (("closed", BAD), ("open", MID)):
        arm = mc["arms"][tag]
        a.plot(np.asarray(arm["k_trace_t_ms"]), arm["k_trace_hot"], color=c, lw=1.7,
               label=f"{tag}   window-2 max {arm['peak2_mM']:.4f} mM")
        for w in bs.get(tag, {}).get("windows", []):
            for r in w["rises"]:
                a.plot(r["t_start_ms"], r["peak_mM"] - r["climb_mM"], "^", color=c, ms=6,
                       clip_on=False)
    a.set_xlim(t_k - 130.0, t_k + 2.2 * sp_ms)
    a.axhline(0.0, color="0.8", lw=0.9)
    for j in (0, 1):
        a.axvline(t_k + j * sp_ms, color="0.55", lw=1.1, ls=":")
        a.text(t_k + j * sp_ms, a.get_ylim()[1], f" kick{j + 1}", color="0.4", fontsize=7.4,
               va="top")
    a.axvline(t_k + 2 * sp_ms, color="0.75", lw=0.9, ls=":")
    a.axvline(t_k - 50.0, color="#c98a20", lw=1.3, ls="--")
    a.text(t_k - 54.0, a.get_ylim()[1], "freeze\n(open arm) ", color="#c98a20", fontsize=7.4,
           va="top", ha="right")
    a.set_xlabel("time (ms)"); a.set_ylabel("K$_o$ at the hot voxel, minus its pre-kick mean (mM)")
    a.legend(fontsize=7.8, loc="lower right", framealpha=0.93)
    a.set_title(f"3  same hot voxel ({mc['arms']['closed']['hot_voxel']}) in both arms.  "
                "\u25b2 = burst onset\n"
                "the 200 ms windows are NOT clean kick responses: window 1 holds two bursts,\n"
                "window 2 one that starts 117 ms (open) / 165 ms (closed) after kick 2",
                fontsize=8.6)

    # 4 -- does it change window 2 in space?
    a = ax[3]
    ng = int(round(np.sqrt(len(mc["arms"]["closed"]["dk_map_per_kick"][1]))))
    d = (np.asarray(mc["arms"]["closed"]["dk_map_per_kick"][1]).reshape(ng, ng)
         - np.asarray(mc["arms"]["open"]["dk_map_per_kick"][1]).reshape(ng, ng))
    v = float(np.abs(d).max()) or 1.0
    im = a.imshow(d, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).set_label(
        "window-2 peak excess K$_o$:  closed − open  (mM)", fontsize=8)
    sp = mc["spatial_extent"]
    a.set_xticks([]); a.set_yticks([])
    rows = [f"{'':10}{'vox>25%':>9}{'vox>50%':>9}{'radius mm':>11}{'occupied':>10}"]
    for t in ("closed", "open"):
        for j, nm in ((0, "win1"), (1, "win2")):
            e = sp[t][j]
            rows.append(f"{t[:6] + ' ' + nm:<12}{e['active_voxels_25pct']:>7}"
                        f"{e['active_voxels_50pct']:>9}{e['recruit_radius_mm']:>11.2f}"
                        f"{e['participant_voxels']:>10}")
    a.text(0.0, -0.05, "\n".join(rows), transform=a.transAxes, va="top", fontsize=7.4,
           family="monospace")
    a.set_title("4  spatial extent in window 2 — broader and flatter with live feedback\n"
                "radius +24%, occupied voxels +22%, but the same count at 25% of each\n"
                "event's OWN peak  (descriptive: one seed, one kick pair, no null)", fontsize=8.6)

    fig.suptitle("FCXR-ION B2.1 calibration-instrument repair — "
                 f"self-consistency {sc['status']}, matched control {mc['status']}.  "
                 "Gate B NOT adjudicated.", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    fig.savefig(os.path.join(FIG, "b2_1_calibration_repair.png"), dpi=175)
    plt.close(fig)
    print("[plot] b2_1_calibration_repair.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_b0()
    fig_gate_h()
    fig_t7()
    fig_gate_b()
    fig_b2_1()
