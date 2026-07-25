# src/topic4_zm_field_verdict.py
"""Pure adjudicator for the reduced-field screen (spec 2026-07-24 rev3 §8-§9). Thresholds locked; every
missing / NaN metric fails CLOSED. The transverse taxonomy already excludes the DC mode upstream."""
from __future__ import annotations
import math

TH = dict(occ=0.80, area=0.50, osc=0.50, R=0.50, corr=0.50, R_global=0.80, p_lo=0.5, p_hi=2.0,
          P95_min=0.1, seeds_required=3, levels_required=3,
          # spec §6.2: a growth rate is only sign-resolvable ABOVE the discretisation error floor.
          # Measured Euler-vs-exact monodromy error 2e-5..1.3e-3 and dt-halving scatter ~5e-4 (local) /
          # 1.3e-3 (global) at dt=0.25 -> 2e-3 is the conservative floor. |lam| <= floor = indeterminate.
          lam_floor=2e-3)

def _num(d, k):
    v = d.get(k, None) if isinstance(d, dict) else None
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v

def _seed_passes(m, global_period_ms):
    need = dict(occupancy=lambda v: v >= TH["occ"], P95=lambda v: v >= TH["P95_min"],
                active_area_frac=lambda v: v >= TH["area"], osc_frac=lambda v: v >= TH["osc"],
                median_R_phase=lambda v: v < TH["R"], mean_pair_corr=lambda v: v < TH["corr"])
    for k, ok in need.items():
        v = _num(m, k)
        if v is None or not ok(v):
            return False, k
    per = _num(m, "median_local_period_ms")
    if per is None or global_period_ms in (None, 0):
        return False, "median_local_period_ms"
    if not (TH["p_lo"] * global_period_ms <= per <= TH["p_hi"] * global_period_ms):
        return False, "period_band"
    return True, None

def level_arm_passes(metrics_list, global_period_ms):
    n, why = 0, []
    for m in metrics_list:
        ok, k = _seed_passes(m if isinstance(m, dict) else {}, global_period_ms)
        n += int(ok); why.append(k)
    return n, dict(failed_on=why)

def level_is_valid(global_metrics):
    """The dual_global control must remain a SYNCHRONISED oscillation, else this level has no matched
    comparison (excluded, NOT counted as 'global failed to synchronise')."""
    R = _num(global_metrics or {}, "median_R_phase"); osc = _num(global_metrics or {}, "osc_frac")
    return bool(R is not None and osc is not None and R >= TH["R_global"] and osc >= TH["osc"])

def _taxonomy(lam_local, lam_global):
    def cls(v):
        if v is None or abs(v) <= TH["lam_floor"]:
            return "indet"                      # below the discretisation noise floor -> sign not resolvable
        return "unstable" if v > 0 else "stable"
    l, g = cls(lam_local), cls(lam_global)
    if "indet" in (l, g): return "indeterminate_below_noise_floor"
    if g == "unstable" and l == "unstable": return "both_unstable"
    if g == "unstable": return "global_unstable_local_stable"
    if l == "unstable": return "global_stable_local_unstable"
    return "both_stable"

def adjudicate_field_screen(summary, lock):
    levels = summary.get("levels", {})
    order = [str(x) for x in lock.get("I0_levels", sorted(levels))]
    passing, excluded, tax_votes, floquet_ok = [], [], [], []
    for key in order:
        lv = levels.get(key)
        if not lv:
            continue
        arms = lv.get("arms", {})
        g = arms.get("dual_global", {}); gm = (g.get("metrics") or [{}])[0]
        if not level_is_valid(gm):
            excluded.append(key); continue
        gper = _num(g, "period_ms")
        lam_g = _num(g, "lambda_perp_max")
        best = None
        for arm in ("dual_local", "dual_mixed"):
            a = arms.get(arm)
            if not a:
                continue
            n, _ = level_arm_passes(a.get("metrics") or [], gper)
            lam_l = _num(a, "lambda_perp_max")
            tax_votes.append(_taxonomy(lam_l, lam_g))
            if n >= TH["seeds_required"]:
                best = arm
                floquet_ok.append(lam_l is not None and lam_g is not None
                                  and lam_l > TH["lam_floor"] and lam_g < -TH["lam_floor"])
        if best:
            passing.append(key)
    # longest run of CONSECUTIVE passing levels in the locked order
    run = best_run = 0; window = []
    cur = []
    for key in order:
        if key in passing:
            cur.append(key); run += 1
            if run > best_run:
                best_run, window = run, list(cur)
        else:
            run = 0; cur = []
    taxonomy = max(set(tax_votes), key=tax_votes.count) if tax_votes else "both_stable"
    if best_run >= TH["levels_required"]:
        verdict = "GO" if any(floquet_ok) else "subcritical_finite_amplitude_candidate"
    else:
        verdict = {"global_unstable_local_stable": "reverse_global_unstable_local_stable",
                   "both_stable": "both_stable", "both_unstable": "both_unstable"}.get(taxonomy, "no_go")
    return dict(verdict=verdict, taxonomy=taxonomy, passing_levels=passing, window=window,
                reasons=dict(excluded_levels=excluded, longest_consecutive=best_run))
