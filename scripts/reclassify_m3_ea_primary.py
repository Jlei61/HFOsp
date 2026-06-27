#!/usr/bin/env python3
"""EA-primary reclassification + cap sensitivity + n17.6-vs-bare paired stats.

OFFLINE over already-dumped per_seed_metrics.csv. Runs NO SNN.

Reviewer reframe (2026-06-23): the event-aligned (EA) readout is PRIMARY; the
fixed windows are SENSITIVITY ONLY. A "fixed-local but EA-non-local" cell (e.g.
bare kick=1.2 late window) is window-sensitive and CANNOT be primary W_event
evidence. The EA fields (r95_mm_ea, far_field_frac_ea, t0_ms) and the raw-trace
flags (returned, runaway) are window-INDEPENDENT (verified: identical across the
three fixed windows for a given kick+seed), so the EA readout is one value per
(substrate, kick, seed).

Definition (EA primary):
  seed is EA-local-returned  <=>  r95_ea <= R95_CAP  AND  far_ea <= FAR_CAP  AND returned==1
  P_EA(substrate,kick) = mean over seeds of that 0/1 flag
  EA-local cell        <=>  P_EA >= ROBUST_FRAC (0.7)  AND  n_seeds >= MIN_SEEDS (6)

Outputs (--out-dir):
  finescan_ea_primary_reclass.csv      — per (substrate,kick): fixed vs EA verdict + primary class
  cap_sensitivity_r95_5p0_to_7p0.csv   — EA-local kicks per substrate over a grid of r95/far caps,
                                          plus n17.6 core-specific wins (EA-local where bare is not)
  fixed_vs_ea_discordance_table.csv     — every (substrate,kick,window) fixed-vs-EA locality (dis)agreement
  paired_stats.json                     — n17.6 vs bare paired bootstrap CI + permutation + effect sizes
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_m3_finite_event_robustness import (  # noqa: E402
    _read_csv, _fnum, _median, ROBUST_FRAC, MIN_SEEDS,
    R95_LOCAL_CAP_MM, FARFIELD_NOISE_FRAC,
)

# default primary caps (mirror Lane A) — cap sensitivity sweeps around these
R95_CAP = R95_LOCAL_CAP_MM          # 6.0
FAR_CAP = FARFIELD_NOISE_FRAC       # 0.5
CORE_BG_RATIO = 1.5                 # core_only downstream above this x bare = self-active
CORE_BG_MARGIN = 5.0

R95_CAP_GRID = (5.0, 5.5, 6.0, 6.5, 7.0)
FAR_CAP_GRID = (0.03, 0.05, 0.08, 0.10, 0.50)   # incl. 0.50 (current loose cap) as reference


def _per_seed_ea(run_dir: str) -> Dict[float, List[dict]]:
    """One EA record per (kick, seed) — EA fields are window-independent, so collapse windows."""
    rows = _read_csv(os.path.join(run_dir, "per_seed_metrics.csv"))
    seen: Dict[Tuple[float, float], dict] = {}
    for r in rows:
        kick = _fnum(r, "kick_boost")
        seed = _fnum(r, "seed")
        key = (kick, seed)
        if key in seen:
            continue                       # EA identical across windows; keep first
        seen[key] = {
            "kick": kick, "seed": seed,
            "r95_ea": _fnum(r, "r95_mm_ea"),
            "far_ea": _fnum(r, "far_field_frac_ea"),
            "returned": _fnum(r, "returned", 0.0),
            "runaway": _fnum(r, "runaway", 0.0),
            "t0": _fnum(r, "t0_ms"),
            "core_only_ds": _fnum(r, "core_only_downstream_resp"),
            "bare_bg_ds": _fnum(r, "no_core_no_kick_downstream"),
        }
    by_kick: Dict[float, List[dict]] = {}
    for rec in seen.values():
        by_kick.setdefault(rec["kick"], []).append(rec)
    return by_kick


def _ea_local_flag(rec: dict, r95cap: float, farcap: float) -> int:
    return int(rec["r95_ea"] <= r95cap and rec["far_ea"] <= farcap and rec["returned"] >= 1)


def _p_ea(recs: List[dict], r95cap: float, farcap: float) -> float:
    if not recs:
        return float("nan")
    return sum(_ea_local_flag(r, r95cap, farcap) for r in recs) / len(recs)


def _fixed_p_best(run_dir: str, kick: float) -> Tuple[float, bool]:
    """Best fixed-window P_local_returned over the 3 windows (sensitivity layer)."""
    rows = _read_csv(os.path.join(run_dir, "per_seed_metrics.csv"))
    by_win: Dict[Tuple[float, float], List[float]] = {}
    for r in rows:
        if abs(_fnum(r, "kick_boost") - kick) > 1e-9:
            continue
        w = (_fnum(r, "win_lo"), _fnum(r, "win_hi"))
        by_win.setdefault(w, []).append(_fnum(r, "seed_local_returned", 0.0))
    best = 0.0
    for vals in by_win.values():
        best = max(best, sum(vals) / len(vals) if vals else 0.0)
    return best, best >= ROBUST_FRAC


def _core_confounded(recs: List[dict]) -> bool:
    """core_only no-kick downstream materially above the bare sheet => self-active substrate."""
    co = _median([r["core_only_ds"] for r in recs])
    bg = _median([r["bare_bg_ds"] for r in recs])
    if co != co or bg != bg:
        return False
    return co > CORE_BG_RATIO * bg + CORE_BG_MARGIN


def _primary_class(p_ea: float, fixed_local: bool, p_ret: float, p_run: float,
                   confounded: bool) -> str:
    if confounded:
        return "spontaneous_confounded"
    if p_run >= 0.5 or p_ret < 0.5:
        return "runaway_not_returned"
    if p_ea >= ROBUST_FRAC:
        return "EA_local_returned"
    if fixed_local:
        return "fixed_only_local"
    if p_ret >= 0.5:
        return "nonlocal_returned"
    return "subthreshold"


def build_reclass(run_dirs: Dict[str, str]) -> List[dict]:
    out: List[dict] = []
    for name, d in run_dirs.items():
        by_kick = _per_seed_ea(d)
        for kick in sorted(by_kick):
            recs = by_kick[kick]
            n = len(recs)
            p_ea = _p_ea(recs, R95_CAP, FAR_CAP)
            p_ret = sum(r["returned"] for r in recs) / n
            p_run = sum(r["runaway"] for r in recs) / n
            r95_ea = _median([r["r95_ea"] for r in recs])
            far_ea = _median([r["far_ea"] for r in recs])
            fixed_best, fixed_local = _fixed_p_best(d, kick)
            confounded = _core_confounded(recs)
            cls = _primary_class(p_ea, fixed_local, p_ret, p_run, confounded)
            out.append({
                "substrate": name, "kick": kick, "n_seeds": n,
                "fixed_P_best": round(fixed_best, 3), "fixed_local": int(fixed_local),
                "P_EA": round(p_ea, 3), "EA_local": int(p_ea >= ROBUST_FRAC and n >= MIN_SEEDS),
                "r95_EA": round(r95_ea, 3), "farfield_EA": round(far_ea, 3),
                "P_returned": round(p_ret, 3), "P_runaway": round(p_run, 3),
                "core_confounded": int(confounded), "primary_class": cls,
            })
    return out


def cap_sensitivity(run_dirs: Dict[str, str]) -> List[dict]:
    """For each (r95cap, farcap): EA-local kicks per substrate + n17.6 core-specific wins vs bare."""
    per = {name: _per_seed_ea(d) for name, d in run_dirs.items()}
    rows: List[dict] = []
    for r95cap in R95_CAP_GRID:
        for farcap in FAR_CAP_GRID:
            ea_local: Dict[str, List[float]] = {}
            for name, by_kick in per.items():
                ks = [k for k, recs in by_kick.items()
                      if len(recs) >= MIN_SEEDS and _p_ea(recs, r95cap, farcap) >= ROBUST_FRAC]
                ea_local[name] = sorted(ks)
            bare = set(ea_local.get("bare", []))
            core_specific = sorted(set(ea_local.get("n17.6", [])) - bare)
            rows.append({
                "r95_cap": r95cap, "far_cap": farcap,
                **{f"EAlocal_{n}": ";".join(f"{k:g}" for k in ea_local.get(n, [])) or "-"
                   for n in run_dirs},
                "n17.6_core_specific_wins": ";".join(f"{k:g}" for k in core_specific) or "-",
                "n_core_specific": len(core_specific),
            })
    return rows


def discordance(run_dirs: Dict[str, str]) -> List[dict]:
    """Per (substrate, kick, window): fixed-window local vs EA local agreement."""
    rows: List[dict] = []
    for name, d in run_dirs.items():
        per_seed = _read_csv(os.path.join(d, "per_seed_metrics.csv"))
        grouped: Dict[Tuple[float, float, float], List[dict]] = {}
        for r in per_seed:
            key = (_fnum(r, "kick_boost"), _fnum(r, "win_lo"), _fnum(r, "win_hi"))
            grouped.setdefault(key, []).append(r)
        for (kick, wlo, whi) in sorted(grouped):
            g = grouped[(kick, wlo, whi)]
            fix_r95 = _median([_fnum(r, "r95_mm") for r in g])
            fix_far = _median([_fnum(r, "far_field_frac") for r in g])
            ea_r95 = _median([_fnum(r, "r95_mm_ea") for r in g])
            ea_far = _median([_fnum(r, "far_field_frac_ea") for r in g])
            fl = int(fix_r95 <= R95_CAP and fix_far <= FAR_CAP)
            el = int(ea_r95 <= R95_CAP and ea_far <= FAR_CAP)
            if fl == el:
                continue                    # only keep DIScordant cells
            rows.append({
                "substrate": name, "kick": kick, "win_lo": wlo, "win_hi": whi,
                "fixed_local": fl, "EA_local": el,
                "fix_r95": round(fix_r95, 2), "fix_far": round(fix_far, 2),
                "ea_r95": round(ea_r95, 2), "ea_far": round(ea_far, 2),
            })
    return rows


def paired_stats(run_dirs: Dict[str, str], kicks: Sequence[float],
                 n_boot: int = 10000, n_perm: int = 10000) -> dict:
    """n17.6 vs bare, paired by seed index, on EA-local-returned + r95_ea/far_ea effect sizes."""
    rng = np.random.default_rng(12345)
    core = _per_seed_ea(run_dirs["n17.6"])
    bare = _per_seed_ea(run_dirs["bare"])
    res: Dict[str, dict] = {}
    for kick in kicks:
        c = sorted(core.get(kick, []), key=lambda r: r["seed"])
        b = sorted(bare.get(kick, []), key=lambda r: r["seed"])
        # pair by seed
        bseed = {r["seed"]: r for r in b}
        pairs = [(rc, bseed[rc["seed"]]) for rc in c if rc["seed"] in bseed]
        if len(pairs) < 3:
            res[f"{kick:g}"] = {"n_pairs": len(pairs), "note": "too few pairs"}
            continue
        cf = np.array([_ea_local_flag(rc, R95_CAP, FAR_CAP) for rc, _ in pairs], float)
        bf = np.array([_ea_local_flag(rb, R95_CAP, FAR_CAP) for _, rb in pairs], float)
        dP = float(cf.mean() - bf.mean())
        # paired bootstrap CI on dP
        idx = np.arange(len(pairs))
        boot = np.array([(cf[s].mean() - bf[s].mean())
                         for s in (rng.integers(0, len(pairs), len(pairs)) for _ in range(n_boot))])
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        # paired permutation: swap labels within each seed
        diff = cf - bf
        perm = np.array([float((diff * rng.choice([1, -1], len(diff))).mean())
                         for _ in range(n_perm)])
        p_perm = float((np.abs(perm) >= abs(dP) - 1e-12).mean())
        # effect sizes on continuous EA spatial metrics (bare - core; positive = core tighter)
        cr95 = np.array([rc["r95_ea"] for rc, _ in pairs]); br95 = np.array([rb["r95_ea"] for _, rb in pairs])
        cfar = np.array([rc["far_ea"] for rc, _ in pairs]); bfar = np.array([rb["far_ea"] for _, rb in pairs])
        def _dz(a, b):  # paired Cohen's dz
            d = b - a
            sd = d.std(ddof=1)
            return float(d.mean() / sd) if sd > 0 else float("nan")
        res[f"{kick:g}"] = {
            "n_pairs": len(pairs),
            "P_EA_core": round(float(cf.mean()), 3), "P_EA_bare": round(float(bf.mean()), 3),
            "deltaP": round(dP, 3), "deltaP_boot95CI": [round(ci[0], 3), round(ci[1], 3)],
            "paired_perm_p": round(p_perm, 4),
            "r95_ea_median_core": round(float(np.median(cr95)), 2),
            "r95_ea_median_bare": round(float(np.median(br95)), 2),
            "r95_ea_dz_bare_minus_core": round(_dz(cr95, br95), 3),
            "far_ea_median_core": round(float(np.median(cfar)), 3),
            "far_ea_median_bare": round(float(np.median(bfar)), 3),
            "far_ea_dz_bare_minus_core": round(_dz(cfar, bfar), 3),
        }
    return res


def _write_csv(rows: List[dict], path: str, cols: Sequence[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="kick_calibration_explore dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--substrates", nargs="+",
                    default=["bare", "n17.6", "n17.8", "n18.0", "w18.0"])
    ap.add_argument("--prefix", default="finescan_",
                    help="run-dir basename prefix (e.g. 'kick_ceiling_' for the ceiling run)")
    ap.add_argument("--paired-kicks", nargs="+", type=float, default=[0.85, 1.0, 1.2])
    args = ap.parse_args(argv)

    run_dirs = {s: os.path.join(args.base, f"{args.prefix}{s}") for s in args.substrates}
    run_dirs = {s: d for s, d in run_dirs.items()
                if os.path.isfile(os.path.join(d, "per_seed_metrics.csv"))}
    os.makedirs(args.out_dir, exist_ok=True)

    reclass = build_reclass(run_dirs)
    _write_csv(reclass, os.path.join(args.out_dir, "finescan_ea_primary_reclass.csv"),
               ["substrate", "kick", "n_seeds", "fixed_P_best", "fixed_local", "P_EA",
                "EA_local", "r95_EA", "farfield_EA", "P_returned", "P_runaway",
                "core_confounded", "primary_class"])

    caps = cap_sensitivity(run_dirs)
    _write_csv(caps, os.path.join(args.out_dir, "cap_sensitivity_r95_5p0_to_7p0.csv"),
               list(caps[0].keys()))

    disc = discordance(run_dirs)
    _write_csv(disc, os.path.join(args.out_dir, "fixed_vs_ea_discordance_table.csv"),
               ["substrate", "kick", "win_lo", "win_hi", "fixed_local", "EA_local",
                "fix_r95", "fix_far", "ea_r95", "ea_far"])

    pstats = paired_stats(run_dirs, args.paired_kicks) if "n17.6" in run_dirs and "bare" in run_dirs else {}
    with open(os.path.join(args.out_dir, "paired_stats.json"), "w") as fh:
        json.dump({"caps": {"r95": R95_CAP, "far": FAR_CAP, "robust_frac": ROBUST_FRAC},
                   "n17.6_vs_bare": pstats}, fh, indent=2)

    # console summary
    print("EA-primary reclassification (P_EA = fraction of seeds EA-local-returned):")
    for r in reclass:
        if r["kick"] >= 0.75:
            print(f"  {r['substrate']:6s} kick={r['kick']:<4g} P_EA={r['P_EA']:.3f} "
                  f"fixed_best={r['fixed_P_best']:.3f} class={r['primary_class']}")
    print("\nn17.6 core-specific EA-local wins across cap grid (EA-local where bare is not):")
    for r in caps:
        if r["n_core_specific"] > 0:
            print(f"  r95cap={r['r95_cap']} far_cap={r['far_cap']}: wins at kick {r['n17.6_core_specific_wins']}")
    print(f"\noutputs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
