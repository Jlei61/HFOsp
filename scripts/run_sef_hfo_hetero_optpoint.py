"""Track E level 1 (spec §5.2): operating-point forced chain for narrowing Var(V_th,E).

Reports slope/curvature of Φ_eff and the FULL 2×2 E/I closed-loop stability (max_k Re λ,
spec §2C — NOT an E→E proxy) for three layers — baseline (wide var), raw narrow, and
mean-matched narrow. DIRECTION IS COMPUTED, NOT PRESET (spec §7): the JSON records the
signs; this script asserts nothing about them.

HARDENING (2026-06-07 user review — close API traps at the source, not in docs):
- Every layer carries the closed-loop ``converged`` / ``n_converged`` / ``n_modes`` flags,
  and ``_layer`` RAISES if the leading mode did not converge — we refuse to emit a
  stability regime read off a failed root search (re_max=-inf sentinel).
- The locked std (wide=1.5, narrow=0.5) and the reset-knee gate are validated ONLY at the
  default op (ratio=1, w_ee_mult=1). For any non-default op we re-run the knee diagnostic
  at THIS op and RAISE if the locked std leave the clean band — rather than silently
  reporting a forced chain dominated by reset-floor saturation.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_lif import (  # noqa: E402
    mean_field, lif_gains, lif_rate, closed_loop_leading, V_RESET, TAU_ME, TREF_E,
)
from src.sef_hfo_heterogeneity import eff_gain_curvature, mean_match_vth  # noqa: E402

OUT = Path("results/topic4_sef_hfo/heterogeneity/optpoint.json")

# reset-knee gate: fraction of effective rate from cells within [V_RESET, KNEE_HI]
KNEE_HI = 13.0
KNEE_GATE = 0.05


def _knee_share(muE, sE, vth_mean, vth_std):
    """Fraction of the truncated-Gaussian-weighted effective rate contributed by the
    near-reset saturation knee [V_RESET, KNEE_HI]. High => Φ_eff measures reset-floor
    saturation of a near-spiking minority, not collective gain."""
    hi = vth_mean + 8.0 * vth_std
    pdf = lambda v: np.exp(-0.5 * ((v - vth_mean) / vth_std) ** 2)
    rate_w = lambda v: lif_rate(muE, sE, TAU_ME, TREF_E, v_th=v) * pdf(v)
    den = quad(rate_w, V_RESET, hi, limit=200)[0]
    knee = quad(rate_w, V_RESET, KNEE_HI, limit=200)[0]
    return knee / den


def _layer(muE, sE, vth_mean, vth_std, w_ee_mult, gI):
    g = eff_gain_curvature(muE, sE, TAU_ME, TREF_E, vth_mean=vth_mean, vth_std=vth_std)
    gE_eff = g["slope"] * TAU_ME                       # effective dimensionless E gain
    cl = closed_loop_leading(gE_eff, gI, w_ee_mult=w_ee_mult)   # FULL 2×2 E/I closed loop (spec §2C)
    if not cl["converged"]:
        raise RuntimeError(
            f"closed_loop_leading did not converge at vth_mean={vth_mean:.4g}, "
            f"vth_std={vth_std:.4g} (n_converged={cl['n_converged']}/{cl['n_modes']}); "
            f"refusing to report a stability regime read off a failed root search.")
    return dict(vth_mean=vth_mean, vth_std=vth_std, rate=g["rate"],
                slope=g["slope"], curvature=g["curvature"], gE_eff=gE_eff,
                closed_loop_re_max=cl["re_max"], closed_loop_k_star=cl["k_star"],
                closed_loop_regime=cl["regime"], closed_loop_converged=cl["converged"],
                closed_loop_n_converged=cl["n_converged"], closed_loop_n_modes=cl["n_modes"])


def analyze_optpoint(ratio=1.0, w_ee_mult=1.0, vth_std_wide=1.5, vth_std_narrow=0.5):
    op = mean_field(ratio, w_ee_mult=w_ee_mult)
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    gI = lif_gains(op)["I"]                            # I gain is homogeneous (no threshold spread)
    is_default = (ratio == 1.0 and w_ee_mult == 1.0)
    # Anchor every layer to the self-consistent rest nuE (review fix 2026-06-06): mean_field
    # solved nuE with bare lif_rate (vth_std=0), so the wide-var transfer must be mean-matched
    # to nuE at muE or it starts off its own fixed point.
    vm_base = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_wide)
    vm_match = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_narrow)
    # Per-op knee re-check (precondition). The locked std are pre-validated ONLY at the
    # default op; for any other op, confirm the distributions stay out of the reset knee
    # at THIS op before trusting the forced chain (else we'd report saturation, not gain).
    knee_wide = _knee_share(muE, sE, vm_base, vth_std_wide)
    knee_narrow = _knee_share(muE, sE, vm_match, vth_std_narrow)
    if not is_default and max(knee_wide, knee_narrow) >= KNEE_GATE:
        raise RuntimeError(
            f"reset-knee share {max(knee_wide, knee_narrow):.3f} >= {KNEE_GATE} at non-default "
            f"op (ratio={ratio}, w_ee_mult={w_ee_mult}); the locked std (wide={vth_std_wide}, "
            f"narrow={vth_std_narrow}) leave the clean band here. Re-pick std for this op "
            f"(re-run the knee diagnostic) before trusting the forced chain.")
    baseline = _layer(muE, sE, vm_base, vth_std_wide, w_ee_mult, gI)        # at nuE
    raw = _layer(muE, sE, vm_base, vth_std_narrow, w_ee_mult, gI)           # same mean, narrowed → Jensen shift
    matched = _layer(muE, sE, vm_match, vth_std_narrow, w_ee_mult, gI)      # narrowed AND re-matched → at nuE
    d_slope_pure = matched["slope"] - baseline["slope"]
    return dict(
        operating_point=dict(muE=muE, sE=sE, nuE=nuE, gI=gI, ratio=ratio,
                             w_ee_mult=w_ee_mult, is_default=is_default,
                             knee_share_wide=knee_wide, knee_share_narrow=knee_narrow),
        params=dict(vth_std_wide=vth_std_wide, vth_std_narrow=vth_std_narrow,
                    knee_gate=KNEE_GATE),
        baseline=baseline, raw_narrow=raw, mean_matched=matched,
        interpretation=dict(
            note="All layers referenced to the self-consistent rest nuE (baseline & "
                 "mean_matched sit AT nuE; raw_narrow carries the Jensen shift). "
                 "closed_loop_re_max is the FULL 2×2 E/I stability (spec §2C), not an "
                 "E→E proxy. Direction computed, not preset (spec §7). mean_matched − "
                 "baseline = PURE variance effect; raw − mean_matched = the Jensen mean "
                 "shift alone. closed_loop_converged MUST be true for re_max to be "
                 "meaningful (a failed search yields regime='unresolved').",
            d_slope_pure=d_slope_pure,
            d_slope_pure_pct=100.0 * d_slope_pure / baseline["slope"],
            d_curvature_pure=matched["curvature"] - baseline["curvature"],
            d_closed_loop_re_max_pure=matched["closed_loop_re_max"] - baseline["closed_loop_re_max"]))


def run():
    res = analyze_optpoint()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    op = res["operating_point"]
    print(f"  op: nuE={op['nuE']*1000:.3f} Hz, muE={op['muE']:.2f}, sE={op['sE']:.2f}, "
          f"knee_share wide/narrow={op['knee_share_wide']:.4f}/{op['knee_share_narrow']:.4f}")
    for k in ("baseline", "raw_narrow", "mean_matched"):
        v = res[k]
        print(f"  {k:13s} slope={v['slope']:.4e} curv={v['curvature']:.4e} "
              f"cl_re_max={v['closed_loop_re_max']:+.4f} ({v['closed_loop_regime']}, "
              f"conv={v['closed_loop_n_converged']}/{v['closed_loop_n_modes']})")
    it = res["interpretation"]
    print(f"  PURE variance effect: Δslope={it['d_slope_pure']:+.4e} ({it['d_slope_pure_pct']:+.1f}%), "
          f"Δcurv={it['d_curvature_pure']:+.4e}, Δ(cl max Re λ)={it['d_closed_loop_re_max_pure']:+.4f}")


if __name__ == "__main__":
    run()
