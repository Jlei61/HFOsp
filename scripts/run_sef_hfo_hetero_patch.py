"""Track E level 2 (spec §5.2): spatial patch finite-pulse + event analysis.

Narrowed Var(V_th,E) core vs baseline surround; raw vs mean-matched control. Reports
nucleation concentration (peak-field mass fraction in the patch) and the self-limit
label. DIRECTION IS COMPUTED, NOT PRESET (spec §7); surround is mandatory (don't tune
core alone).

This is the SPATIAL finite-pulse layer the linear closed-loop null (Task 7) handed the
existence question to — the framework's 2026-06-03 correction says the LIF op is robustly
stable but nonlinearly excitable, so the finite-pulse/nucleation margin (here), not the
linear margin (Task 7), is the real gate.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_lif import mean_field, classify_response, TAU_ME, TREF_E  # noqa: E402
from src.sef_hfo_heterogeneity import integrate_hetero_field, mean_match_vth  # noqa: E402
from src.sef_hfo_field import _grid  # noqa: E402

OUT = Path("results/topic4_sef_hfo/heterogeneity/patch.json")
_STIM_X0 = -3.0   # off-center seed x (Step-0b validated stimulus)


def _offcenter_pulse(A=8.0, T=30.0, R=2.0, x0=_STIM_X0, n=96, L=12.0):
    X, Y = _grid(n, L)
    disk = ((X - x0) ** 2 + Y ** 2) <= R ** 2
    return lambda t: (A * disk if t < T else 0.0)


def _one(op, x_patch, r_patch, vmc, vsc, vms, vss, t_max):
    ext, front, peak = integrate_hetero_field(
        op, _offcenter_pulse(), x_patch=x_patch, r_patch=r_patch,
        vth_mean_core=vmc, vth_std_core=vsc,
        vth_mean_surround=vms, vth_std_surround=vss,
        t_max=t_max, return_peak_field=True)
    label, info = classify_response(ext, front)
    # nucleation concentration: peak-field deviation mass fraction inside the patch
    X, Y = _grid(96, 12.0)
    core = ((X - x_patch) ** 2 + Y ** 2) <= r_patch ** 2
    dev = np.clip(peak - op["nuE"], 0, None)
    frac_in_patch = float(dev[core].sum() / max(dev.sum(), 1e-12))
    return dict(label=label, max_ext=info["max_ext"], adv_mm=info["adv_mm"],
                dur_ms=info["dur_ms"], returned=info["returned"],
                frac_mass_in_patch=frac_in_patch)


def analyze_patch(ratio=1.0, x_patch=0.0, r_patch=2.0, vth_std_wide=1.5,
                  vth_std_narrow=0.5, t_max=200.0):
    op = mean_field(ratio)
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    # Mean-match the SURROUND (the bulk that sets the rest) and the matched core to the
    # self-consistent rest nuE, so the whole field starts at rest — else a stimulus-free
    # transient contaminates every layer (review fix 2026-06-06).
    vm_surround = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_wide)
    vm_core_matched = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_narrow)
    # baseline: uniform wide var at nuE (core == surround) → field at rest.
    base = _one(op, x_patch, r_patch, vm_surround, vth_std_wide, vm_surround, vth_std_wide, t_max)
    # raw_narrow: core variance narrowed, SAME mean as surround → carries Jensen core shift.
    raw = _one(op, x_patch, r_patch, vm_surround, vth_std_narrow, vm_surround, vth_std_wide, t_max)
    # mean_matched: core narrowed AND core mean re-solved to nuE → pure variance change.
    matched = _one(op, x_patch, r_patch, vm_core_matched, vth_std_narrow, vm_surround, vth_std_wide, t_max)
    return dict(operating_point=dict(nuE=nuE, muE=muE, sE=sE),
                patch=dict(x_patch=x_patch, r_patch=r_patch, stim_x0=_STIM_X0,
                           vth_std_wide=vth_std_wide, vth_std_narrow=vth_std_narrow,
                           vth_mean_surround=vm_surround, vth_mean_core_matched=vm_core_matched),
                layers=dict(baseline=base, raw_narrow=raw, mean_matched=matched),
                interpretation=dict(
                    note="Whole field referenced to self-consistent rest nuE (surround, "
                         "baseline & matched-core sit AT nuE). Direction computed not preset "
                         "(spec §7). mean_matched vs baseline = PURE variance effect on "
                         "nucleation/self-limit; raw carries the Jensen core shift.",
                    d_frac_in_patch_pure=matched["frac_mass_in_patch"] - base["frac_mass_in_patch"],
                    d_max_ext_pure=matched["max_ext"] - base["max_ext"]))


MARGIN_OUT = Path("results/topic4_sef_hfo/heterogeneity/margin.json")
_RUNAWAY = ("runaway", "global_synchronous")


def _label_and_peak(op, A, x_patch, r_patch, core_vm, core_std, surr_vm, surr_std, t_max):
    X, Y = _grid(96, 12.0)
    pulse = lambda t: (A * (((X - x_patch) ** 2 + Y ** 2) <= r_patch ** 2) if t < 30.0 else 0.0)
    ext, front, peak = integrate_hetero_field(
        op, pulse, x_patch=x_patch, r_patch=r_patch,
        vth_mean_core=core_vm, vth_std_core=core_std,
        vth_mean_surround=surr_vm, vth_std_surround=surr_std,
        t_max=t_max, return_peak_field=True)
    label, info = classify_response(ext, front)
    return label, info["max_ext"], float(peak.max())


def analyze_margin(ratio=1.0, x_patch=-3.0, r_patch=2.0, vth_std_wide=1.5,
                   vth_std_narrow=0.5, A_grid=(8, 24, 48, 80, 200), t_max=120.0):
    """Finite-pulse safety margin (spec §2C order parameter). Vary ONLY stimulus
    amplitude A; find A_runaway (first A with a runaway/global label) for baseline
    (uniform wide) vs mean_matched (narrow core in wide surround), both mean-matched to
    nuE, co-located patch+seed. S compression = A_runaway_narrow < A_runaway_wide (the
    Rich direction the linear map + fixed-A label cannot detect). The A_grid spans into
    the saturation plateau (high A) so a None A_runaway means UNREACHABLE, not just
    out-of-grid. Direction computed not preset (spec §7)."""
    op = mean_field(ratio)
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    vm_wide = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_wide)
    vm_narrow = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_narrow)

    def sweep(core_vm, core_std):
        rows, a_run = [], None
        for A in A_grid:
            lbl, mx, pk = _label_and_peak(op, float(A), x_patch, r_patch,
                                          core_vm, core_std, vm_wide, vth_std_wide, t_max)
            rows.append(dict(A=float(A), label=lbl, max_ext=mx, peak_rE_max=pk))
            if a_run is None and lbl in _RUNAWAY:
                a_run = float(A)
        # saturation flag: top-two A give the same max_ext => response saturated => a None
        # A_runaway is genuinely unreachable, not merely beyond the grid.
        saturated = (len(rows) >= 2 and abs(rows[-1]["max_ext"] - rows[-2]["max_ext"]) < 1e-3)
        return dict(A_runaway=a_run, saturated_at_high_A=saturated, sweep=rows)

    wide = sweep(vm_wide, vth_std_wide)
    narrow = sweep(vm_narrow, vth_std_narrow)
    ar_w, ar_n = wide["A_runaway"], narrow["A_runaway"]
    return dict(
        operating_point=dict(nuE=nuE, muE=muE, sE=sE),
        params=dict(vth_std_wide=vth_std_wide, vth_std_narrow=vth_std_narrow, x_patch=x_patch,
                    r_patch=r_patch, vm_wide=vm_wide, vm_narrow=vm_narrow,
                    A_grid=list(A_grid), t_max=t_max),
        baseline_wide=wide, mean_matched_narrow=narrow,
        interpretation=dict(
            note="Vary ONLY A; A_runaway = first A with runaway/global label. Both None + "
                 "saturated_at_high_A True => the finite-pulse margin is UNBOUNDED (no kick "
                 "amplitude triggers runaway; response saturates) for that layer. S "
                 "compression = A_runaway_narrow < A_runaway_wide. Direction computed not "
                 "preset (spec §7).",
            A_runaway_wide=ar_w, A_runaway_narrow=ar_n,
            margin_compressed=(ar_n is not None and ar_w is not None and ar_n < ar_w),
            both_unbounded=(ar_w is None and ar_n is None
                            and wide["saturated_at_high_A"] and narrow["saturated_at_high_A"])))


def run():
    # Two patch placements: co-located with the seed (does narrowing make the SEED site
    # the nucleation hotspot?) and downstream at center (does a propagating wave get
    # concentrated/trapped by a narrowed-variance patch it encounters?).
    res = {
        "colocated_x-3": analyze_patch(x_patch=_STIM_X0),
        "downstream_x0": analyze_patch(x_patch=0.0),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    for place, r in res.items():
        print(f"\n[{place}]  patch x={r['patch']['x_patch']}, r={r['patch']['r_patch']}")
        for k, v in r["layers"].items():
            print(f"  {k:13s} label={v['label']:28s} frac_in_patch={v['frac_mass_in_patch']:.3f} "
                  f"max_ext={v['max_ext']:.3f}")
        it = r["interpretation"]
        print(f"  PURE variance effect: Δfrac_in_patch={it['d_frac_in_patch_pure']:+.3f}, "
              f"Δmax_ext={it['d_max_ext_pure']:+.4f}")


def run_margin():
    """Finite-pulse safety-margin S sweep (spec §2C real gate) — writes margin.json."""
    res = analyze_margin()
    MARGIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    MARGIN_OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {MARGIN_OUT}")
    it = res["interpretation"]
    for tag in ("baseline_wide", "mean_matched_narrow"):
        L = res[tag]
        plateau = "  ".join(f"A{r['A']:.0f}:{r['max_ext']:.3f}" for r in L["sweep"])
        print(f"  {tag:20s} A_runaway={L['A_runaway']} saturated={L['saturated_at_high_A']}  [{plateau}]")
    print(f"  margin_compressed={it['margin_compressed']}  both_unbounded={it['both_unbounded']}")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "margin":
        run_margin()
    else:
        run()
