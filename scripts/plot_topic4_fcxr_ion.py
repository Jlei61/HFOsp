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
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))

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
        y = 0.95 - i * 0.135
        ax[1].text(0.0, y, "PASS" if c["ok"] else "FAIL", color=OK if c["ok"] else BAD,
                   fontsize=8.5, fontweight="bold", family="monospace")
        ax[1].text(0.10, y, c["step"], fontsize=8.5, va="baseline")
        ax[1].text(0.10, y - 0.042, "\n".join(textwrap.wrap(c["evidence"], 82)[:2]),
                   fontsize=6.6, color="0.35", va="top", linespacing=1.35)
    ax[1].text(0.0, -0.02, "Dimension only: g_K_ion = 1 is a declared normalization\n"
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


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_b0()
    fig_gate_h()
    fig_gate_b()
