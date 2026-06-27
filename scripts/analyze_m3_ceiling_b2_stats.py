#!/usr/bin/env python3
"""Ceiling B2 statistics — seed-level K_min + the logit model artifact (P1, 2026-06-23).

OFFLINE over the ceiling per_seed_metrics.csv (bare + n17.6). Runs NO SNN.

Two reviewer P1 fixes:
  (1) Seed-level K_min: the cohort number "K_min(n17.6)=1.1 vs bare=1.6" is the
      P_EA>=0.7 crossing of the PROPORTION curve, NOT a seed-level threshold that
      dropped 0.5. Report the per-seed first-EA-local-returned kick with a paired
      sign test, a censored survival view, and the bootstrap median shift. The
      honest wording is "core makes the cohort EA-local probability curve cross 0.7
      earlier", and per-seed "most seeds advance ~one kick grid step, none later".
  (2) The OR=4.49 logit had no model artifact. Dump the full coefficient table
      (input summary, model type, cluster dim, coef/SE/CI/p, OR+CI, naive AND
      cluster-robust) plus a seed-block permutation + cluster bootstrap sensitivity
      (12 seed clusters is small, so cluster-robust SE alone is anti-conservative).

Outputs (--out-dir):
  seed_first_EA_local_kick.csv  — per (substrate, seed): first EA-local-returned kick (inf=never)
  seed_kmin_shift.json          — paired sign test / median shift / survival / censoring
  logit_substrate_kick.json     — full logit coefficient table + permutation/bootstrap sensitivity
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_m3_finite_event_robustness import (  # noqa: E402
    _read_csv, _fnum, R95_LOCAL_CAP_MM, FARFIELD_NOISE_FRAC, ROBUST_FRAC,
)


# --------------------------------------------------------------------------- #
# Pure functions (TDD)                                                        #
# --------------------------------------------------------------------------- #
def first_crossing_kick(flags_by_kick: Dict[float, int], kicks: Sequence[float]) -> float:
    """First kick (ascending) at which the EA-local-returned flag is 1; inf if never."""
    for k in sorted(kicks):
        if flags_by_kick.get(k, 0) == 1:
            return float(k)
    return math.inf


def paired_sign_counts(core: Dict[int, float], bare: Dict[int, float]) -> Tuple[int, int, int]:
    """(n core-earlier, n same, n core-later) over seeds present in both, inf-aware."""
    n_earlier = n_same = n_later = 0
    for s in sorted(set(core) & set(bare)):
        c, b = core[s], bare[s]
        if c < b:
            n_earlier += 1
        elif c > b:
            n_later += 1
        else:                       # equal (incl. both inf)
            n_same += 1
    return n_earlier, n_same, n_later


def binomial_sign_p(n_earlier: int, n_later: int) -> float:
    """Exact two-sided sign test on DISCORDANT pairs only (ties excluded)."""
    n = n_earlier + n_later
    if n == 0:
        return 1.0
    k = min(n_earlier, n_later)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _ea_flags(run_dir: str) -> Dict[int, Dict[float, int]]:
    """seed -> {kick -> EA-local-returned 0/1} from one ceiling run dir (win-indep EA)."""
    rows = [r for r in _read_csv(os.path.join(run_dir, "per_seed_metrics.csv"))
            if _fnum(r, "win_lo") == 22.0]      # any single window; EA is window-independent
    out: Dict[int, Dict[float, int]] = {}
    for r in rows:
        s = int(_fnum(r, "seed"))
        k = _fnum(r, "kick_boost")
        ea = int(_fnum(r, "r95_mm_ea") <= R95_LOCAL_CAP_MM
                 and _fnum(r, "far_field_frac_ea") <= FARFIELD_NOISE_FRAC
                 and _fnum(r, "returned") >= 1)
        out.setdefault(s, {})[k] = ea
    return out


def cohort_crossing_kick(first_cross: Dict[int, float], kicks: Sequence[float],
                         frac: float = ROBUST_FRAC) -> float:
    """Lowest kick at which >= frac of seeds have EVER crossed (cumulative view)."""
    n = len(first_cross)
    for k in sorted(kicks):
        if sum(1 for v in first_cross.values() if v <= k) / n >= frac:
            return float(k)
    return math.inf


def p_ea_crossing_kick(flags: Dict[int, Dict[float, int]], kicks: Sequence[float],
                       frac: float = ROBUST_FRAC) -> float:
    """Lowest kick where the CURRENTLY-LOCAL fraction P_EA(k) >= frac.

    This is the lens the recap's headline '1.1 vs 1.6' came from. It differs from
    cohort_crossing_kick (cumulative-ever) because EA-local is non-monotonic in kick
    (a seed can be local at 1.1, not 1.2, local again at 1.3); the P_EA lens is the
    MOST threshold-sensitive of the cohort views.
    """
    n = len(flags)
    for k in sorted(kicks):
        if sum(1 for f in flags.values() if f.get(k, 0) == 1) / n >= frac:
            return float(k)
    return math.inf


# --------------------------------------------------------------------------- #
# Artifacts                                                                    #
# --------------------------------------------------------------------------- #
def seed_kmin(base: str, kicks: Sequence[float]) -> Tuple[List[dict], dict]:
    core_flags = _ea_flags(os.path.join(base, "kick_ceiling_n17.6"))
    bare_flags = _ea_flags(os.path.join(base, "kick_ceiling_bare"))
    core_fc = {s: first_crossing_kick(f, kicks) for s, f in core_flags.items()}
    bare_fc = {s: first_crossing_kick(f, kicks) for s, f in bare_flags.items()}

    rows: List[dict] = []
    for s in sorted(set(core_fc) | set(bare_fc)):
        rows.append({"substrate": "n17.6", "seed": s,
                     "first_EA_local_kick": core_fc.get(s, math.inf)})
        rows.append({"substrate": "bare", "seed": s,
                     "first_EA_local_kick": bare_fc.get(s, math.inf)})

    n_earlier, n_same, n_later = paired_sign_counts(core_fc, bare_fc)
    p_sign = binomial_sign_p(n_earlier, n_later)
    finite_deltas = [core_fc[s] - bare_fc[s] for s in set(core_fc) & set(bare_fc)
                     if math.isfinite(core_fc[s]) and math.isfinite(bare_fc[s])]
    # cluster bootstrap of the median finite-pair shift
    rng = np.random.default_rng(7)
    med = float(statistics.median(finite_deltas)) if finite_deltas else float("nan")
    if finite_deltas:
        boot = [float(np.median(rng.choice(finite_deltas, len(finite_deltas), replace=True)))
                for _ in range(10000)]
        med_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    else:
        med_ci = [float("nan"), float("nan")]
    summary = {
        "paired_sign": {"core_earlier": n_earlier, "same": n_same, "core_later": n_later,
                        "binomial_two_sided_p": round(p_sign, 4)},
        "per_seed_shift_finite_pairs": {
            "n_finite_pairs": len(finite_deltas),
            "median_kick_shift_core_minus_bare": round(med, 3),
            "median_shift_boot95CI": [round(med_ci[0], 3), round(med_ci[1], 3)],
            "note": "shift is ~one kick grid step (0.1), NOT 0.5",
        },
        "cohort_proportion_view": {
            "crossing_frac": ROBUST_FRAC,
            "p_ea_currently_local_crossing_bare": p_ea_crossing_kick(bare_flags, kicks),
            "p_ea_currently_local_crossing_n17.6": p_ea_crossing_kick(core_flags, kicks),
            "cumulative_ever_crossed_bare": cohort_crossing_kick(bare_fc, kicks),
            "cumulative_ever_crossed_n17.6": cohort_crossing_kick(core_fc, kicks),
            "note": ("the recap headline '1.1 vs 1.6' is the P_EA (currently-local fraction) "
                     ">=0.7 crossing — the MOST threshold-sensitive lens, amplified by a "
                     "non-monotonic dip in the bare curve. Under the cumulative-ever-crossed "
                     "lens the gap nearly vanishes (1.1 vs 1.1). The robust seed-level signal "
                     "is the paired sign test (7 earlier / 0 later, ~0.1 kick), not a 0.5 drop."),
        },
        "censoring": {
            "bare_never_cross_by_max": sum(1 for v in bare_fc.values() if math.isinf(v)),
            "n17.6_never_cross_by_max": sum(1 for v in core_fc.values() if math.isinf(v)),
            "max_kick": max(kicks),
        },
    }
    return rows, summary


def logit_artifact(base: str) -> dict:
    """Full logit EA_local ~ substrate + kick with naive + cluster-robust SE + permutation/bootstrap."""
    import statsmodels.api as sm
    def load(tag):
        rows = [r for r in _read_csv(os.path.join(base, f"kick_ceiling_{tag}/per_seed_metrics.csv"))
                if _fnum(r, "win_lo") == 22.0]
        return [(_fnum(r, "kick_boost"), int(_fnum(r, "seed")),
                 int(_fnum(r, "r95_mm_ea") <= R95_LOCAL_CAP_MM
                     and _fnum(r, "far_field_frac_ea") <= FARFIELD_NOISE_FRAC
                     and _fnum(r, "returned") >= 1)) for r in rows]
    data = [("bare", x) for x in load("bare")] + [("n17.6", x) for x in load("n17.6")]
    y = np.array([d[1][2] for d in data], float)
    sub = np.array([1.0 if d[0] == "n17.6" else 0.0 for d in data])
    kick = np.array([d[1][0] for d in data])
    seed = np.array([d[1][1] for d in data])
    X = sm.add_constant(np.column_stack([sub, kick]))
    names = ["const", "substrate_core", "kick"]

    def coef_table(model):
        ci = model.conf_int()
        return {names[i]: {"coef": float(model.params[i]), "se": float(model.bse[i]),
                           "z": float(model.tvalues[i]), "p": float(model.pvalues[i]),
                           "ci95": [float(ci[i][0]), float(ci[i][1])]} for i in range(len(names))}

    m_naive = sm.Logit(y, X).fit(disp=0)
    m_clu = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": seed})
    or_ci = m_clu.conf_int()[1]

    # seed-block permutation + cluster bootstrap on a clean paired statistic:
    # per-seed mean(EA-local rate) difference n17.6 - bare, averaged over seeds.
    seeds = sorted(set(seed.astype(int)))
    per_seed_diff = []
    for s in seeds:
        c = y[(sub == 1) & (seed == s)].mean()
        b = y[(sub == 0) & (seed == s)].mean()
        per_seed_diff.append(c - b)
    per_seed_diff = np.array(per_seed_diff)
    obs = float(per_seed_diff.mean())
    rng = np.random.default_rng(11)
    perm = np.array([float((per_seed_diff * rng.choice([1, -1], len(per_seed_diff))).mean())
                     for _ in range(20000)])
    perm_p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    boot = np.array([float(rng.choice(per_seed_diff, len(per_seed_diff), replace=True).mean())
                     for _ in range(20000)])
    boot_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    return {
        "model": "Logit  EA_local_returned ~ const + substrate_core + kick",
        "outcome": "EA_local_returned (r95_ea<=6mm AND far_ea<=0.5 AND returned), per (seed,kick)",
        "n_obs": int(len(y)), "n_seed_clusters": len(seeds),
        "ea_local_rate": {"bare": float(y[sub == 0].mean()), "n17.6": float(y[sub == 1].mean())},
        "coefficients_naive": coef_table(m_naive),
        "coefficients_cluster_robust_by_seed": coef_table(m_clu),
        "substrate_odds_ratio": {"OR": float(np.exp(m_clu.params[1])),
                                 "ci95": [float(np.exp(or_ci[0])), float(np.exp(or_ci[1]))],
                                 "cov_type": "cluster (groups=seed)"},
        "pseudo_r2": float(m_clu.prsquared),
        "small_cluster_sensitivity": {
            "statistic": "mean over seeds of per-seed (n17.6 - bare) EA-local rate diff",
            "observed": round(obs, 4),
            "seed_block_sign_permutation_p": round(perm_p, 4),
            "cluster_bootstrap_95CI": [round(boot_ci[0], 4), round(boot_ci[1], 4)],
            "note": "12 seed clusters is small; cluster-robust SE alone is anti-conservative",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="kick_calibration_explore dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kicks", nargs="+", type=float, default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.6])
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    rows, kmin_summary = seed_kmin(args.base, args.kicks)
    with open(os.path.join(args.out_dir, "seed_first_EA_local_kick.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["substrate", "seed", "first_EA_local_kick"])
        for r in rows:
            v = r["first_EA_local_kick"]
            w.writerow([r["substrate"], r["seed"], "inf" if math.isinf(v) else f"{v:g}"])
    with open(os.path.join(args.out_dir, "seed_kmin_shift.json"), "w") as fh:
        json.dump(kmin_summary, fh, indent=2)

    logit = logit_artifact(args.base)
    with open(os.path.join(args.out_dir, "logit_substrate_kick.json"), "w") as fh:
        json.dump(logit, fh, indent=2)

    ps = kmin_summary["paired_sign"]
    print(f"seed-level K_min: core_earlier={ps['core_earlier']} same={ps['same']} "
          f"core_later={ps['core_later']} sign_p={ps['binomial_two_sided_p']}")
    cv = kmin_summary["cohort_proportion_view"]
    print(f"  per-seed median shift = {kmin_summary['per_seed_shift_finite_pairs']['median_kick_shift_core_minus_bare']} kick (NOT 0.5)")
    print(f"  P_EA(currently-local)>=0.7 crossing: bare={cv['p_ea_currently_local_crossing_bare']} "
          f"n17.6={cv['p_ea_currently_local_crossing_n17.6']}  [headline lens, threshold-sensitive]")
    print(f"  cumulative-ever-crossed>=0.7: bare={cv['cumulative_ever_crossed_bare']} "
          f"n17.6={cv['cumulative_ever_crossed_n17.6']}  [robust lens, gap nearly vanishes]")
    print(f"logit: OR={logit['substrate_odds_ratio']['OR']:.2f} "
          f"CI{[round(x,2) for x in logit['substrate_odds_ratio']['ci95']]} "
          f"cluster-robust p={logit['coefficients_cluster_robust_by_seed']['substrate_core']['p']:.4f}; "
          f"seed-block perm p={logit['small_cluster_sensitivity']['seed_block_sign_permutation_p']}")
    print(f"outputs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
