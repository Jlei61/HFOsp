"""Increment-3a detection-front-end SENSITIVITY (the dispositive contract test, 2026-06-07).

The participation margin (10%) and the event-window source (field-ext vs contact-aggregate)
were CHANGED after seeing the four-control fail. That is exactly the "tune the gate after
seeing the result" anti-pattern. This harness discharges that concern the only way that
counts: it shows the SCIENTIFIC OUTPUT (recovered axis_err) is INVARIANT to these knobs —
the knobs move only COVERAGE (whether a window exists / whether n_part≥7), not the read.

If axis_err is flat across the knob while coverage changes, the knob is a detection
front-end, not a result-manufacturing dial (advisor 2026-06-07). If axis_err were to MOVE
with the knob, that is a genuine problem — reported, not papered over.

Two sweeps:
  (A) margin ∈ {0.10, 0.20, 0.30}, window=field. Table axis_err + n_part per condition.
      Expect axis_err flat; n_part rises as margin drops.
  (B) window ∈ {field, contact}, margin=0.10. Table axis_err per condition.
      Expect ~identical WHERE BOTH yield a window. Honest caveat: C3 rot30/rot90 have NO
      contact-aggregate window (a shaft lies parallel to the wave → contacts stay lit →
      the aggregate never "returns"); field-ext is the only thing that brackets them, so
      their PASS rests on the inductive argument from the cases where both windows exist.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_sef_hfo_obs_increment3a import run_four_controls   # noqa: E402

OUT = "results/topic4_sef_hfo/observation_layer/increment3a_rate_parity"


def _flatten(res):
    """condition-key -> (axis_err, n_part, win_exists) for all 10 conditions."""
    out = {}
    for grp in ("C1_connectivity", "C2_kicktrack", "C3_shaft_invariance"):
        for k, r in res[grp].items():
            out[f"{grp.split('_')[0]}:{k}"] = (r["axis_err"], r["n_part"], r["win"] is not None)
    c4 = res["C4_iso"]
    out["C4:iso"] = (c4["axis_err"], c4["n_part"], c4["win"] is not None)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- (A) margin sweep (window=field) ----
    margin_runs = {}
    for mf in (0.10, 0.20, 0.30):
        print(f"[margin sweep] margin_frac={mf} window=field ...", flush=True)
        margin_runs[mf] = _flatten(run_four_controls(margin_frac=mf, window_source="field"))

    # ---- (B) window-source comparison (margin=0.10) ----
    print("[window sweep] window=contact margin=0.10 ...", flush=True)
    win_field   = margin_runs[0.10]                                  # reuse
    win_contact = _flatten(run_four_controls(margin_frac=0.10, window_source="contact"))

    conds = list(margin_runs[0.10].keys())

    # ---- assemble + invariance check ----
    report = {"margin_sweep": {}, "window_sweep": {}, "invariance": {}}
    for c in conds:
        report["margin_sweep"][c] = {
            f"m{int(mf*100)}": dict(axis_err=margin_runs[mf][c][0], n_part=margin_runs[mf][c][1])
            for mf in (0.10, 0.20, 0.30)}
        report["window_sweep"][c] = {
            "field":   dict(axis_err=win_field[c][0],   n_part=win_field[c][1],   win=win_field[c][2]),
            "contact": dict(axis_err=win_contact[c][0], n_part=win_contact[c][1], win=win_contact[c][2])}

    # ---- meaningful invariance (advisor 2026-06-07) ----
    # The dispositive question is NOT "does axis_err stay bit-identical" but "does the
    # front-end knob MANUFACTURE the verdict". Two checks:
    #  (1) window source must not MOVE the read where both windows exist (axis_err diff ~0);
    #  (2) margin must never flip a condition pass->FAIL — only pass->INSUFFICIENT (coverage
    #      loss is honest). axis_err MAY drift via coverage->accuracy (fewer contacts ->
    #      noisier centroid); we report that drift honestly but it stays < the 25° gate
    #      wherever n_part>=7, and 10% is the MAXIMAL-coverage end (not a cherry-picked optimum).
    from scripts.run_sef_hfo_obs_increment3a import AXIS_ERR_MAX, PART_MIN   # noqa: E402

    def _cond_verdict(axis_err, n_part, is_iso=False):
        # n_part<7 = insufficient (honest coverage loss) for ALL conditions.
        if n_part < PART_MIN:
            return "insufficient"
        if is_iso:
            # iso negative control: axis=None (no readable direction) IS the pass.
            return "pass" if axis_err is None else "fail"
        if axis_err is None or axis_err >= AXIS_ERR_MAX:
            return "fail"
        return "pass"

    def _spread(vals):
        xs = [v for v in vals if v is not None]
        return None if len(xs) < 2 else round(max(xs) - min(xs), 2)

    for c in conds:
        is_iso = c.startswith("C4")
        both_win = win_field[c][2] and win_contact[c][2]
        win_errs = [win_field[c][0], win_contact[c][0]] if both_win else []
        margin_verdicts = {f"m{int(mf*100)}": _cond_verdict(*margin_runs[mf][c][:2], is_iso=is_iso)
                           for mf in (0.10, 0.20, 0.30)}
        report["invariance"][c] = dict(
            margin_verdicts=margin_verdicts,
            any_fail_over_margin=any(v == "fail" for v in margin_verdicts.values()),
            axis_err_field_vs_contact=(_spread(win_errs) if both_win else "contact_has_no_window"),
            both_windows_exist=both_win)

    # (1) window invariance: axis_err diff field-vs-contact where both exist
    window_spreads = [report["invariance"][c]["axis_err_field_vs_contact"]
                      for c in conds
                      if isinstance(report["invariance"][c]["axis_err_field_vs_contact"], (int, float))]
    # (2) margin never flips pass->fail (only pass->insufficient)
    any_fail_anywhere = any(report["invariance"][c]["any_fail_over_margin"] for c in conds)
    # axis_err drift among conditions with n_part>=7 at ALL three margins (apples-to-apples)
    stable = [c for c in conds if all(margin_runs[mf][c][1] >= PART_MIN for mf in (0.10, 0.20, 0.30))]
    drift = {c: _spread([margin_runs[mf][c][0] for mf in (0.10, 0.20, 0.30)]) for c in stable}
    drift_vals = [d for d in drift.values() if d is not None]

    report["summary"] = dict(
        window_axis_err_invariant=bool(not window_spreads or max(window_spreads) <= 5.0),
        max_axis_err_field_vs_contact=(max(window_spreads) if window_spreads else None),
        margin_never_flips_to_fail=bool(not any_fail_anywhere),
        max_axis_err_drift_over_margin=(max(drift_vals) if drift_vals else None),
        axis_err_drift_per_stable_condition=drift,
        n_conditions_contact_window_missing=sum(
            1 for c in conds if report["invariance"][c]["axis_err_field_vs_contact"] == "contact_has_no_window"),
        # the meaningful verdict: front-end knobs change COVERAGE, not the READ / the pass-fail decision
        VERDICT_ROBUST=bool(not any_fail_anywhere and (not window_spreads or max(window_spreads) <= 5.0)))

    with open(os.path.join(OUT, "rate_parity_sensitivity.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: None)

    # ---- console tables ----
    print("\n=== (A) MARGIN SWEEP (window=field): axis_err° / n_part ===")
    print(f"{'condition':<22} {'m10':>14} {'m20':>14} {'m30':>14}")
    for c in conds:
        cells = []
        for mf in (0.10, 0.20, 0.30):
            e, nn = margin_runs[mf][c][0], margin_runs[mf][c][1]
            cells.append(f"{(str(e)+'°'):>7}/{nn:<3}")
        print(f"{c:<22} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")

    print("\n=== (B) WINDOW SOURCE (margin=0.10): axis_err° / n_part ===")
    print(f"{'condition':<22} {'field':>16} {'contact':>16}")
    for c in conds:
        fe, fn, fw = win_field[c]
        ce, cn, cw = win_contact[c]
        fcell = f"{(str(fe)+'°'):>7}/{fn:<3}" if fw else "no_window"
        ccell = f"{(str(ce)+'°'):>7}/{cn:<3}" if cw else "no_window"
        print(f"{c:<22} {fcell:>16} {ccell:>16}")

    s = report["summary"]
    print("\n=== INVARIANCE SUMMARY (meaningful: do front-end knobs MANUFACTURE the verdict?) ===")
    print(f"  (1) window source: max axis_err diff field-vs-contact where BOTH exist = "
          f"{s['max_axis_err_field_vs_contact']}°  -> invariant={s['window_axis_err_invariant']}")
    print(f"      (conditions where contact-aggregate has NO window: "
          f"{s['n_conditions_contact_window_missing']} — geometry-fragile, field-only)")
    print(f"  (2) margin 10/20/30%: never flips pass->FAIL = {s['margin_never_flips_to_fail']} "
          f"(only pass->insufficient via coverage loss)")
    print(f"      axis_err drift over margin (stable conds, n_part>=7 at all 3) = "
          f"{s['max_axis_err_drift_over_margin']}° (coverage->accuracy; 10% = max coverage)")
    print(f"  VERDICT_ROBUST (knobs move coverage, not the read/decision): {s['VERDICT_ROBUST']}")
    print("wrote", os.path.join(OUT, "rate_parity_sensitivity.json"), flush=True)


if __name__ == "__main__":
    main()
