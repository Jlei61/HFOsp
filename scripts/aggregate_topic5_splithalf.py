"""Topic5 A-line §3.1 split-half robustness aggregation (leg ii: alignment robustness).

Reads the 8 per-(mode,half)x(metric) alignment outputs (sh_{mode}_half{h}_{act}_B1000.json) and
asks: does the ictal-vs-interictal axis alignment survive rebuilding the interictal axis from only
HALF the events? Three readouts per (mode, metric):

  1. per-half cohort: does each half still beat the null? broadband -> channel layer (binomial over
     EVALUABLE subjects, attrition to 18 reported); hfa -> joint layer (Wilcoxon one-sided p<0.05
     AND patient-level effect-size 95% CI>0 over ADEQUATE subjects).
  2. half1-vs-half2 paired stability: across subjects present in both halves, Spearman(real_h1,
     real_h2) of |alignment| -> gate >= 0.5 (strong-aligning subjects align strongly in both halves).
  3. attrition: per layer, how many of 18 are non-evaluable (status!=ok, n_matched<6, or joint
     effective_shuffle_n<4). Non-evaluable counts in the denominator but never as a pass.

Denominator is LOCKED at 18 (the A-line primary cohort); halves that cannot form a frame are
non-evaluable, not silently dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

SPLIT = Path("results/topic5_ictal_recruitment/axis_alignment/splithalf")
N_COHORT = 18
RNG_SEED = 20260615


def _load(mode, half, act):
    f = SPLIT / f"sh_{mode}_half{half}_{act}_B1000.json"
    if not f.exists():
        return {}
    j = json.load(open(f))
    return {r["subject_id"]: r for r in j["per_subject"] if r.get("dataset") == "epilepsiae"}


def _adequate_joint(r):
    if r.get("status") != "ok":
        return False
    eff = (r.get("effective_shuffle_n") or {}).get("joint")
    return r.get("n_matched_channels", 0) >= 6 and eff is not None and eff >= 4


def _effect_ci(diffs, seed=RNG_SEED, n_boot=10000):
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    if d.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    boot = [np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(n_boot)]
    return float(np.median(d)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _wilcox(diffs):
    d = [x for x in diffs if np.isfinite(x)]
    if not any(v != 0 for v in d):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="greater").pvalue)
    except ValueError:
        return 1.0


def _half_cohort(rows_by_sid, layer):
    """layer in {channel, joint}. Returns dict with n_pass / n_eval / binom_p / wilcox_p /
    effect+CI / attrition over the locked 18 denominator."""
    passk = f"pass_{layer}_null"
    medk = f"{layer}_null_median"
    ok = [r for r in rows_by_sid.values() if r.get("status") == "ok"]
    if layer == "joint":
        evaluable = [r for r in ok if _adequate_joint(r)]
    else:
        evaluable = [r for r in ok if r.get(passk) is not None and r.get(medk) is not None]
    npass = sum(bool(r.get(passk)) for r in evaluable)
    diffs = [r["real_median_abs_corr"] - r[medk] for r in evaluable if r.get(medk) is not None]
    eff, lo, hi = _effect_ci(diffs)
    binom_p = (binomtest(npass, len(evaluable), 0.05, alternative="greater").pvalue
               if evaluable else None)
    return {
        "layer": layer, "n_denominator": N_COHORT, "n_evaluable": len(evaluable),
        "n_nonevaluable": N_COHORT - len(evaluable), "n_pass": npass,
        "binom_p": round(float(binom_p), 5) if binom_p is not None else None,
        "wilcoxon_p": round(_wilcox(diffs), 5) if diffs else None,
        "effect_size": round(eff, 5) if eff is not None else None,
        "effect_ci": [round(lo, 5), round(hi, 5)] if lo is not None else None,
    }


def main():
    summary = {"note": "Topic5 A-line split-half robustness (leg ii). Denominator locked at 18.",
               "results": []}
    for mode in ["first_second", "odd_even"]:
        for act, layer in [("broadband", "channel"), ("hfa", "joint")]:
            h1, h2 = _load(mode, 1, act), _load(mode, 2, act)
            if not h1 or not h2:
                print(f"[splithalf-agg] MISSING {mode} {act} (h1={len(h1)} h2={len(h2)})", flush=True)
                continue
            c1, c2 = _half_cohort(h1, layer), _half_cohort(h2, layer)
            # half1 vs half2 paired |alignment| stability across subjects ok in both halves
            common = sorted(set(h1) & set(h2))
            pairs = [(h1[s]["real_median_abs_corr"], h2[s]["real_median_abs_corr"]) for s in common
                     if h1[s].get("status") == "ok" and h2[s].get("status") == "ok"]
            rho = (float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic)
                   if len(pairs) >= 3 else None)
            rec = {"mode": mode, "metric": act, "layer": layer,
                   "half1": c1, "half2": c2,
                   "half1_vs_half2_spearman": round(rho, 4) if rho is not None else None,
                   "n_paired": len(pairs)}
            summary["results"].append(rec)
            print(f"[{mode} {act} {layer}] h1 pass {c1['n_pass']}/{c1['n_evaluable']} (binom {c1['binom_p']}) "
                  f"| h2 pass {c2['n_pass']}/{c2['n_evaluable']} (binom {c2['binom_p']}) "
                  f"| h1-h2 rho={rec['half1_vs_half2_spearman']} (n={len(pairs)}) "
                  f"| nonEval h1={c1['n_nonevaluable']} h2={c2['n_nonevaluable']}", flush=True)
    out = SPLIT / "splithalf_summary.json"
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
