"""Rate Step3b — non-spatial cell-heterogeneity SENSITIVITY MATRIX (spec §3 Θ_cell).

The first-round V_th-only result (optpoint/patch/margin.json) is a sub-conclusion: narrowing
the THRESHOLD distribution alone does not move the rate-model safety margin. This screen asks
the broader question WITHOUT leaving the rate model and WITHOUT going spatial: if "cells become
more alike" is not just V_th but the whole input→output curve, which cell parameter's
distribution-narrowing actually has LEVERAGE on the effective curve?

For each parameter P in {V_th, tau_m, tau_ref, sigma} we narrow its across-cell spread (a fixed
relative reduction, mean-matched to the self-consistent rest rate nuE so we isolate "spread
narrowing" from "working point moved" — spec §5.0 Contract 2) and report how the effective
transfer's slope (gain), curvature, and the 2×2 closed-loop margin change. Direction is
COMPUTED, not preset (spec §7); the goal is to FIND leverage, not to tune a desired sign.
E_L/baseline drive is included only as a working-point-shift CONTROL (mu_shift row), never read
as a heterogeneity effect.

Screen only. The finite-pulse / spatial patch (the real gate, expensive) is run LATER and ONLY
for parameter packs that move the margin here — else the spatial layer just wastes time.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_lif import (  # noqa: E402
    mean_field, lif_gains, lif_rate, closed_loop_leading, TAU_ME, TREF_E, V_TH,
)
from src.sef_hfo_heterogeneity import phi_eff_param, mean_match_param, phi_eff_multi  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

OUT = Path("results/topic4_sef_hfo/heterogeneity/sensitivity_matrix.json")
COMBO_OUT = Path("results/topic4_sef_hfo/heterogeneity/combo.json")

# A fixed RELATIVE narrowing (coefficient of variation) applied to every parameter so the
# leverage comparison is fair across parameters with different units/scales. The V_th values
# 0.08/0.027 reproduce the locked wide=1.5 / narrow=0.5 (1.5/18, 0.5/18).
CV_WIDE = 0.08
CV_NARROW = 0.027
PARAMS = ("v_th", "tau_m", "tau_ref", "sigma")


def _nominal_mean(param, op):
    return {"v_th": V_TH, "tau_m": TAU_ME, "tau_ref": TREF_E, "sigma": op["sE"]}[param]


def _slope_curv(muE, sE, param, mean, std, h=1e-2):
    """rate, slope (∂Φ_eff/∂μ), curvature (∂²Φ_eff/∂μ²) at the input muE, holding the
    parameter's distribution (mean, std) fixed. μ is the common input drive."""
    f = lambda m: phi_eff_param(m, sE, TAU_ME, TREF_E, param=param, mean=mean, std=std)
    f0, fp, fm = f(muE), f(muE + h), f(muE - h)
    return f0, (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / (h * h)


def _layer(muE, sE, param, mean, std, gI):
    rate, slope, curv = _slope_curv(muE, sE, param, mean, std)
    gE_eff = slope * TAU_ME                       # effective dimensionless E gain
    cl = closed_loop_leading(gE_eff, gI)          # dispersion's tau_m held nominal (screen approx)
    return dict(mean=mean, std=std, rate=rate, slope=slope, curvature=curv, gE_eff=gE_eff,
                closed_loop_re_max=cl["re_max"], closed_loop_regime=cl["regime"],
                closed_loop_converged=cl["converged"])


def screen_param(param, op, gI):
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    nom = _nominal_mean(param, op)
    std_wide, std_narrow = CV_WIDE * nom, CV_NARROW * nom
    # Mean-match each spread to the rest rate nuE. If the rate is INSENSITIVE to this
    # parameter at this op, the match is not bracketed — that is itself a 'no leverage'
    # screen result (record it, fall back to the nominal mean, do not crash).
    rate_insensitive = False
    try:
        m_wide = mean_match_param(nuE, muE, sE, TAU_ME, TREF_E, param=param, std=std_wide)
        m_narrow = mean_match_param(nuE, muE, sE, TAU_ME, TREF_E, param=param, std=std_narrow)
    except ValueError:
        rate_insensitive = True
        m_wide = m_narrow = nom
    wide = _layer(muE, sE, param, m_wide, std_wide, gI)
    narrow = _layer(muE, sE, param, m_narrow, std_narrow, gI)
    base_slope = wide["slope"]
    return dict(
        param=param, nominal_mean=nom, cv_wide=CV_WIDE, cv_narrow=CV_NARROW,
        std_wide=std_wide, std_narrow=std_narrow, rate_insensitive=rate_insensitive,
        baseline_wide=wide, mean_matched_narrow=narrow,
        d_slope_pure=narrow["slope"] - wide["slope"],
        d_slope_pure_pct=(100.0 * (narrow["slope"] - wide["slope"]) / base_slope
                          if base_slope != 0 else float("nan")),
        d_curvature_pure=narrow["curvature"] - wide["curvature"],
        d_closed_loop_re_max_pure=narrow["closed_loop_re_max"] - wide["closed_loop_re_max"])


def mu_shift_control(op, gI, dmu=1.0):
    """CONTROL (not a heterogeneity row): shift the mean input drive by dmu (working point
    moves; NOT mean-matched). Confirms the screen's metrics DO respond to a known working-
    point nudge — so a flat d_slope for an intrinsic parameter means 'no leverage', not
    'screen is dead'."""
    muE, sE = op["muE"], op["sE"]
    base = _layer(muE, sE, "v_th", V_TH, 0.0, gI)             # homogeneous baseline at the op
    shifted = _layer(muE + dmu, sE, "v_th", V_TH, 0.0, gI)    # working point shifted by +dmu
    return dict(dmu=dmu, baseline=base, shifted=shifted,
                d_rate=shifted["rate"] - base["rate"],
                d_slope=shifted["slope"] - base["slope"],
                d_closed_loop_re_max=shifted["closed_loop_re_max"] - base["closed_loop_re_max"])


def _combo_layer(op, gI, combo, cv, n_nodes, h=1e-2):
    """Narrow ALL params in `combo` by relative factor `cv` (point mass if cv==0), restore
    the rest rate nuE via a COMMON input-drive offset δ (Contract 2 control that does not
    privilege any one intrinsic param), then report slope/curvature in that common drive
    and the closed-loop margin. δ is the rate-matching knob; the slope axis is the same δ."""
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    specs = [(p, _nominal_mean(p, op), cv * _nominal_mean(p, op)) for p in combo]
    f = lambda d: phi_eff_multi(muE + d, sE, TAU_ME, TREF_E, specs=specs, n_nodes=n_nodes)
    # mean-match: common drive δ restoring nuE
    delta = brentq(lambda d: f(d) - nuE, -12.0, 12.0, xtol=1e-7)
    f0, fp, fm = f(delta), f(delta + h), f(delta - h)
    slope = (fp - fm) / (2 * h)
    curv = (fp - 2 * f0 + fm) / (h * h)
    gE_eff = slope * TAU_ME
    cl = closed_loop_leading(gE_eff, gI)
    return dict(cv=cv, delta_match=float(delta), rate=f0, slope=slope, curvature=curv,
                gE_eff=gE_eff, closed_loop_re_max=cl["re_max"], closed_loop_regime=cl["regime"],
                closed_loop_converged=cl["converged"])


def screen_combo(op, gI, combo, n_nodes):
    wide = _combo_layer(op, gI, combo, CV_WIDE, n_nodes)
    narrow = _combo_layer(op, gI, combo, CV_NARROW, n_nodes)
    base_slope = wide["slope"]
    return dict(
        combo=list(combo), n_nodes=n_nodes, baseline_wide=wide, mean_matched_narrow=narrow,
        d_slope_pure=narrow["slope"] - wide["slope"],
        d_slope_pure_pct=(100.0 * (narrow["slope"] - wide["slope"]) / base_slope
                          if base_slope != 0 else float("nan")),
        d_curvature_pure=narrow["curvature"] - wide["curvature"],
        d_closed_loop_re_max_pure=narrow["closed_loop_re_max"] - wide["closed_loop_re_max"])


def run_combo():
    op = mean_field(1.0)
    gI = lif_gains(op)["I"]
    combos = {"v_th+sigma": (("v_th", "sigma"), 48),
              "v_th+tau_m+tau_ref": (("v_th", "tau_m", "tau_ref"), 22)}
    rows = {name: screen_combo(op, gI, combo, n) for name, (combo, n) in combos.items()}
    res = dict(operating_point=dict(muE=op["muE"], sE=op["sE"], nuE=op["nuE"], gI=gI),
               cv=dict(wide=CV_WIDE, narrow=CV_NARROW), combos=rows,
               note="Combo axes (spec §5.2 'cells more alike' = several cell params narrowed "
                    "TOGETHER). Each combo narrows all listed params by the same relative "
                    "factor; rate restored to nuE via a COMMON drive offset δ. d_*_pure = "
                    "narrow − wide. Direction computed not preset (spec §7). closed_loop "
                    "dispersion holds tau_m nominal (screen approx).")
    COMBO_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMBO_OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {COMBO_OUT}")
    for name, r in rows.items():
        print(f"  {name:20s} d_slope={r['d_slope_pure_pct']:+6.1f}%  "
              f"d_cl_reMax={r['d_closed_loop_re_max_pure']:+.4f}")


def run():
    op = mean_field(1.0)
    gI = lif_gains(op)["I"]
    rows = {p: screen_param(p, op, gI) for p in PARAMS}
    control = mu_shift_control(op, gI)
    res = dict(
        operating_point=dict(muE=op["muE"], sE=op["sE"], nuE=op["nuE"], gI=gI),
        cv=dict(wide=CV_WIDE, narrow=CV_NARROW),
        params=rows, mu_shift_control=control,
        note="Non-spatial leverage screen (spec §3 Θ_cell, Step3b). Each parameter's spread "
             "narrowed by the same relative factor (CV wide→narrow), mean-matched to nuE "
             "(Contract 2). d_*_pure = mean_matched_narrow − baseline_wide = the PURE spread "
             "effect. Direction computed not preset (spec §7). closed_loop dispersion holds "
             "tau_m nominal (screen approximation). Leverage = |d_slope_pure_pct| and "
             "|d_closed_loop_re_max_pure| materially above the V_th baseline; only "
             "leverage-positive packs proceed to the (expensive) spatial finite-pulse gate.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    print(f"  op: nuE={op['nuE']*1000:.3f} Hz, muE={op['muE']:.2f}, sE={op['sE']:.2f}")
    print(f"  {'param':8s} {'d_slope%':>9s} {'d_curv':>11s} {'d_cl_reMax':>11s}  insensitive")
    for p in PARAMS:
        r = rows[p]
        print(f"  {p:8s} {r['d_slope_pure_pct']:+8.1f}% {r['d_curvature_pure']:+.3e} "
              f"{r['d_closed_loop_re_max_pure']:+.4f}   {r['rate_insensitive']}")
    c = control
    print(f"  [control mu+{c['dmu']}]: d_rate={c['d_rate']:+.3e} d_slope={c['d_slope']:+.3e} "
          f"d_cl_reMax={c['d_closed_loop_re_max']:+.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "combo":
        run_combo()
    else:
        run()
