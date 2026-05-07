"""PR-7 addendum: P3 cohort-level equivalence test (TOST + bootstrap CI).

Framework: docs/paper1_framework_sba.md v1.1.2 §5.3
Lock δ_excess = 0.05 (scientific equivalence margin, not anchored to PR-7 data).

Inputs:
- results/interictal_propagation/template_pairing/per_subject/<dataset>_<sid>.json
- results/interictal_propagation/template_pairing/per_subject_burst/<dataset>_<sid>.json

Cohort: 6 forward/reverse-reproduced subjects (PR-7 H1 cohort).

Tests (all on N2 main null per framework):
  T1 — cohort excess(Δt) for Δt ∈ {10, 30, 60, 1800} s:
       robust median + bootstrap 95% CI; TOST(δ=0.05); leave-one-out + leave-548-out
  T2 — cohort lag1_same_excess vs N2 null:
       bootstrap CI; TOST(equivalence vs 0, δ=0.05)
  T3 — cohort run_length_lift vs N2 null:
       bootstrap CI; TOST(equivalence vs 1, δ=0.05)

Verdict logic per v1.1.2 §5.3:
  PASS:           T1 + T2 + T3 全部 cohort-level equivalence 满足
  INCONCLUSIVE:   cohort median 落 ±δ 内但 bootstrap CI 跨 ±δ
  SENSITIVITY:    cohort 主 INCONCLUSIVE 但 leave-one-out / leave-548-out 满足 PASS — archive only
  NULL:           cohort robust median |excess| > δ 且 leave-one-out 仍 > δ
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/honglab/leijiaxin/HFOsp")
PER_SUBJECT_DIR = REPO / "results/interictal_propagation/template_pairing/per_subject"
BURST_DIR = REPO / "results/interictal_propagation/template_pairing/per_subject_burst"
OUT_DIR = REPO / "results/interictal_propagation/template_pairing"

DELTA_EXCESS = 0.05            # framework v1.1.2 lock
WINDOWS = [10.0, 30.0, 60.0, 1800.0]
N_BOOT = 10000
RNG_SEED = 0
NULL_KEY = "N2"                # main null per framework
SUBJECT_548 = "epilepsiae_548"
ALPHA = 0.05


def load_subject_metrics() -> list[dict[str, Any]]:
    out = []
    for fname in sorted(os.listdir(PER_SUBJECT_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(PER_SUBJECT_DIR / fname) as fh:
            ps = json.load(fh)
        subj = f"{ps['dataset']}_{ps['subject_id']}"
        n_a, n_b, n_used = ps["n_T_a"], ps["n_T_b"], ps["n_events_used"]
        p_fwd = n_a / n_used

        lift_n2 = ps["pairing_with_nulls"]["lift"][NULL_KEY]
        excess = {w: float(lift_n2[str(w)]["excess"]) for w in WINDOWS}

        bp = BURST_DIR / fname
        if bp.exists():
            with open(bp) as fh:
                bd = json.load(fh)["burst_diagnostic"]
            lag1_same_excess = float(bd["lag1_same_excess"][NULL_KEY])
            run_length_lift = float(bd["lift"][NULL_KEY]["run_length_lift"])
            lag1_same_emp = float(bd["empirical"]["lag1_same"])
        else:
            lag1_same_excess = float("nan")
            run_length_lift = float("nan")
            lag1_same_emp = float("nan")

        out.append(
            {
                "subject": subj,
                "n_used": int(n_used),
                "n_a": int(n_a),
                "n_b": int(n_b),
                "p_fwd": float(p_fwd),
                "p_rev": float(1 - p_fwd),
                "lag1_marg_iid": float(p_fwd**2 + (1 - p_fwd) ** 2),
                "lag1_same_emp": lag1_same_emp,
                "lag1_same_excess": lag1_same_excess,
                "run_length_lift": run_length_lift,
                **{f"excess_{int(w)}s": excess[w] for w in WINDOWS},
            }
        )
    return out


def tost_equivalence(values: np.ndarray, target: float, delta: float, n_boot: int, seed: int) -> dict:
    """TOST equivalence test on cohort median via bootstrap.

    Equivalence iff bootstrap p_lower < α AND p_upper < α AND CI ⊂ (target±δ).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    medians = np.median(values[idx], axis=1)
    p_lower = float(np.mean(medians <= target - delta))   # frac claiming "below margin"
    p_upper = float(np.mean(medians >= target + delta))   # frac claiming "above margin"
    p_tost = max(p_lower, p_upper)
    obs_median = float(np.median(values))
    ci_lo = float(np.percentile(medians, 2.5))
    ci_hi = float(np.percentile(medians, 97.5))
    equivalence_pass = (p_tost < ALPHA) and (ci_lo > target - delta) and (ci_hi < target + delta)
    inside_band = (target - delta <= obs_median <= target + delta)
    return {
        "median_obs": obs_median,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "tost_p_lower": p_lower,
        "tost_p_upper": p_upper,
        "tost_p": p_tost,
        "equivalence_pass": bool(equivalence_pass),
        "median_inside_band": bool(inside_band),
        "ci_inside_band": bool(ci_lo > target - delta and ci_hi < target + delta),
        "target": target,
        "delta": delta,
    }


def leave_one_out(values: np.ndarray, names: list[str], target: float, delta: float, seed: int) -> dict:
    """For each subject, drop and recompute cohort median + CI."""
    out = {}
    for i, name in enumerate(names):
        sub = np.delete(values, i)
        seed_i = seed + i + 1
        out[f"drop_{name}"] = tost_equivalence(sub, target, delta, n_boot=N_BOOT, seed=seed_i)
    return out


def _run_test_block(
    vals: np.ndarray,
    names: list[str],
    target: float,
    seed_base: int,
) -> dict:
    """Cohort TOST + leave-one-out + leave-548-out, packaged for results dict."""
    main_test = tost_equivalence(vals, target=target, delta=DELTA_EXCESS, n_boot=N_BOOT, seed=seed_base)
    loo = leave_one_out(vals, names, target=target, delta=DELTA_EXCESS, seed=seed_base + 100)
    idx_548 = names.index(SUBJECT_548)
    l548_vals = np.delete(vals, idx_548)
    l548_test = tost_equivalence(l548_vals, target=target, delta=DELTA_EXCESS, n_boot=N_BOOT, seed=seed_base + 200)
    return {
        "values_per_subject": dict(zip(names, vals.tolist())),
        "cohort_main": main_test,
        "leave_one_out": loo,
        "leave_548_out": l548_test,
    }


def _print_block(label: str, block: dict, *, with_l548: bool = False) -> None:
    m = block["cohort_main"]
    print(f"{label}: median={m['median_obs']:+.4f}  CI95=[{m['ci95_lo']:+.4f},{m['ci95_hi']:+.4f}]  TOST p={m['tost_p']:.4f}  equiv_pass={m['equivalence_pass']}")
    if with_l548:
        l = block["leave_548_out"]
        print(f"   leave-548-out: median={l['median_obs']:+.4f}  CI95=[{l['ci95_lo']:+.4f},{l['ci95_hi']:+.4f}]  equiv_pass={l['equivalence_pass']}")


def main() -> None:
    print(f"[PR-7 addendum] δ_excess = {DELTA_EXCESS}, n_boot = {N_BOOT}, null = {NULL_KEY}")
    subjects = load_subject_metrics()
    names = [s["subject"] for s in subjects]
    n = len(subjects)
    print(f"Cohort N = {n}: {names}")
    print()

    results: dict[str, Any] = {
        "framework_version": "v1.1.2",
        "delta_excess": DELTA_EXCESS,
        "alpha": ALPHA,
        "n_boot": N_BOOT,
        "rng_seed": RNG_SEED,
        "null_key": NULL_KEY,
        "cohort_size": n,
        "subjects": subjects,
        "tests": {},
    }

    # T1 — excess at each window (target = 0.0)
    for w in WINDOWS:
        key = f"excess_{int(w)}s"
        vals = np.array([s[key] for s in subjects])
        block = _run_test_block(vals, names, target=0.0, seed_base=RNG_SEED + int(w) * 1000)
        results["tests"][key] = block
        _print_block(f"T1 excess({int(w)}s)", block, with_l548=True)

    # T2 — lag1_same_excess (target = 0.0)
    vals = np.array([s["lag1_same_excess"] for s in subjects])
    block = _run_test_block(vals, names, target=0.0, seed_base=RNG_SEED + 100_000)
    results["tests"]["lag1_same_excess"] = block
    _print_block("T2 lag1_same_excess", block)

    # T3 — run_length_lift (target = 1.0)
    vals = np.array([s["run_length_lift"] for s in subjects])
    block = _run_test_block(vals, names, target=1.0, seed_base=RNG_SEED + 200_000)
    results["tests"]["run_length_lift"] = block
    _print_block("T3 run_length_lift ", block)

    # Overall verdict per v1.1.2 §5.3
    main_excess_passes = [results["tests"][f"excess_{int(w)}s"]["cohort_main"]["equivalence_pass"] for w in WINDOWS]
    t2_pass = results["tests"]["lag1_same_excess"]["cohort_main"]["equivalence_pass"]
    t3_pass = results["tests"]["run_length_lift"]["cohort_main"]["equivalence_pass"]
    all_main_pass = all(main_excess_passes) and t2_pass and t3_pass

    main_excess_inside = [
        results["tests"][f"excess_{int(w)}s"]["cohort_main"]["median_inside_band"] for w in WINDOWS
    ]
    t2_inside = results["tests"]["lag1_same_excess"]["cohort_main"]["median_inside_band"]
    t3_inside = results["tests"]["run_length_lift"]["cohort_main"]["median_inside_band"]
    all_median_inside = all(main_excess_inside) and t2_inside and t3_inside

    # leave-548-out PASS
    l548_excess_passes = [
        results["tests"][f"excess_{int(w)}s"]["leave_548_out"]["equivalence_pass"] for w in WINDOWS
    ]
    l548_t2 = results["tests"]["lag1_same_excess"]["leave_548_out"]["equivalence_pass"]
    l548_t3 = results["tests"]["run_length_lift"]["leave_548_out"]["equivalence_pass"]
    l548_all_pass = all(l548_excess_passes) and l548_t2 and l548_t3

    # NULL check: cohort median > delta AND leave-one-out still > delta
    cohort_null = []
    for w in WINDOWS:
        cm = results["tests"][f"excess_{int(w)}s"]["cohort_main"]
        if abs(cm["median_obs"]) > DELTA_EXCESS:
            loo_dict = results["tests"][f"excess_{int(w)}s"]["leave_one_out"]
            all_loo_above = all(abs(v["median_obs"]) > DELTA_EXCESS for v in loo_dict.values())
            cohort_null.append(all_loo_above)
        else:
            cohort_null.append(False)
    is_null = any(cohort_null)

    if all_main_pass:
        verdict = "PASS"
    elif is_null:
        verdict = "NULL"
    elif l548_all_pass and all_median_inside:
        verdict = "SENSITIVITY-only"
    else:
        verdict = "INCONCLUSIVE"

    results["verdict"] = {
        "label": verdict,
        "all_main_pass": bool(all_main_pass),
        "all_median_inside_band": bool(all_median_inside),
        "leave_548_out_pass": bool(l548_all_pass),
        "any_null_window": bool(is_null),
    }

    print()
    print(f"=== VERDICT: {verdict} ===")
    print(f"   all main equivalence pass: {all_main_pass}")
    print(f"   all median inside ±δ band: {all_median_inside}")
    print(f"   leave-548-out all pass:    {l548_all_pass}")
    print(f"   any window NULL:           {is_null}")

    out_path = OUT_DIR / "pr7_addendum_p3.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
