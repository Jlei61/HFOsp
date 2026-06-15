"""Topic 5 A-line — finalize the sweep into ONE tiered, FDR-controlled table.

Reads results/.../axis_alignment/axis_alignment_<metric>_B<B>.json (4 metrics x 2 B, each with
4 nulls) and writes a single summary that:
  - tiers the metrics: broadband = PRIMARY (pre-registered); HFA / ramp = sensitivity;
    EI-like = exploratory (so a strong HFA result is NOT read as the primary conclusion);
  - reports, per (metric, B, null): n_pass/n, binomial p, Wilcoxon p, leave-one-subject-out
    worst-case Wilcoxon p, and n_adequate (subjects whose null actually shuffled >= MIN_EFF
    channels — degenerate nulls on small subjects are not counted as real controls);
  - applies Benjamini-Hochberg FDR across the whole EXPLORATORY family (every metric x null
    Wilcoxon p), since this is a multi-metric multi-null exploration, not one pre-set test.

Wording is locked to 'interictal axis aligns with ictal early-activation spatial gradient'
(sign-free / collinear) — never 'path replay'.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

OUT = Path("results/topic5_ictal_recruitment/axis_alignment")
MIN_EFF = 4          # a null must actually shuffle >= this many channels to count for a subject
NULLS = ["channel", "within_shaft", "anchor_matched", "joint"]
TIER = {"broadband": "PRIMARY", "hfa": "sensitivity", "ramp": "sensitivity", "ei": "exploratory"}


def _bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _effect_stats(diffs, seed=0, n_boot=10000):
    """Patient-level effect size + bootstrap 95% CI + Wilcoxon rank-biserial.

    `diffs` = per-subject (real_median_abs_corr - <null>_null_median) values.
    Pure / deterministic for a fixed `seed` (uses np.random.default_rng, not the
    global RNG). Returns {effect_size, ci_lo, ci_hi, rank_biserial}.
      - effect_size  = median over subjects of diff
      - ci_lo/ci_hi  = 2.5/97.5 pct of the bootstrapped subject-median (resample
                       n subjects with replacement, n_boot times)
      - rank_biserial = (W+ - W-)/(W+ + W-) from the signed-rank statistic, in [-1,1]
                        (sign-aware effect size accompanying the Wilcoxon test)
    """
    d = np.asarray(diffs, float)
    n = d.size
    if n == 0:
        return {"effect_size": None, "ci_lo": None, "ci_hi": None, "rank_biserial": None}
    effect = float(np.median(d))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_meds = np.median(d[idx], axis=1)
    ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])
    # Wilcoxon signed-rank rank-biserial: rank |diff| over nonzero diffs (wilcox
    # zero-method drops exact zeros), split rank mass by the sign of the diff.
    nz = d[d != 0]
    if nz.size == 0:
        rb = 0.0
    else:
        from scipy.stats import rankdata
        ranks = rankdata(np.abs(nz))
        w_pos = ranks[nz > 0].sum()
        w_neg = ranks[nz < 0].sum()
        rb = float((w_pos - w_neg) / (w_pos + w_neg))
    return {"effect_size": effect, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "rank_biserial": rb}


def _adequate_n(summ, null):
    """# Epilepsiae subjects whose `null` actually shuffled >= MIN_EFF channels."""
    c = 0
    for r in summ["per_subject"]:
        if r.get("status") != "ok" or r["dataset"] != "epilepsiae":
            continue
        eff = (r.get("effective_shuffle_n") or {}).get(null)
        if eff is not None and eff >= MIN_EFF:
            c += 1
    return c


# per-null (pass-key, null-median-key) — same keys as runner _cohort_stats
_NULL_KEYS = {"channel": ("pass_channel_null", "channel_null_median"),
              "within_shaft": ("pass_within_shaft_null", "within_shaft_null_median"),
              "anchor_matched": ("pass_anchor_matched_null", "anchor_matched_null_median"),
              "joint": ("pass_joint_null", "joint_null_median")}


def _subject_diffs(summ, null, adequate_only=False):
    """Patient-level (real - null) diffs over the Epilepsiae primary cohort, using the
    same per-subject filter as run_topic5_axis_alignment._cohort_stats.

    adequate_only=True ADDITIONALLY drops subjects whose `null` shuffled < MIN_EFF channels
    (degenerate null on small/few-shaft subjects). REQUIRED for the joint layer: with only
    ~13 adequate subjects, an effect-size CI over all 18 mixes in 5 degenerate nulls whose
    (real - null_median) is not a meaningful effect. The HFA-joint gate uses adequate-only."""
    passk, medk = _NULL_KEYS[null]
    rows = [r for r in summ["per_subject"]
            if r.get("status") == "ok" and r["dataset"] == "epilepsiae"]
    sub = [r for r in rows if r.get(passk) is not None and r.get(medk) is not None]
    if adequate_only:
        sub = [r for r in sub
               if (r.get("effective_shuffle_n") or {}).get(null) is not None
               and (r["effective_shuffle_n"][null] >= MIN_EFF)]
    return [r["real_median_abs_corr"] - r[medk] for r in sub]


def _wilcox_greater(diffs):
    """One-sided Wilcoxon signed-rank p (real > null) over `diffs`. Recomputed in the
    aggregate so the joint gate's Wilcoxon is over the SAME adequate set as its effect CI."""
    from scipy.stats import wilcoxon
    d = [x for x in diffs if np.isfinite(x)]
    if not any(v != 0 for v in d):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="greater").pvalue)
    except ValueError:
        return 1.0


def main():
    files = sorted(glob.glob(str(OUT / "axis_alignment_*_B*.json")))
    if not files:
        print("no axis_alignment_*.json found", flush=True)
        return
    rows, wilcox_ps = [], []
    for f in files:
        summ = json.load(open(f))
        m, B = summ["activation"], summ["B"]
        c = summ["epilepsiae_primary"]
        for null in NULLS:
            if c.get(f"n_pass_{null}") is None:
                continue
            diffs_adeq = _subject_diffs(summ, null, adequate_only=True)
            diffs_all = _subject_diffs(summ, null, adequate_only=False)
            es = _effect_stats(diffs_adeq, seed=20260615, n_boot=10000)     # adequate-only = gate-relevant
            es_all = _effect_stats(diffs_all, seed=20260615, n_boot=10000)  # all-subjects = transparency
            row = {"metric": m, "tier": TIER.get(m, "exploratory"), "B": B, "null": null,
                   "n": c["n"], "n_pass": c[f"n_pass_{null}"],
                   "n_adequate_eff": _adequate_n(summ, null),
                   "binom_p": c.get(f"binom_p_{null}"), "wilcoxon_p": c.get(f"wilcoxon_p_{null}"),
                   "wilcoxon_p_adequate": round(_wilcox_greater(diffs_adeq), 5),
                   "loso_worst_wilcoxon_p": c.get(f"loso_wilcoxon_max_p_{null}"),
                   "effect_n_adequate": len(diffs_adeq), "effect_n_all": len(diffs_all),
                   "effect_size": round(es["effect_size"], 5) if es["effect_size"] is not None else None,
                   "effect_ci_lo": round(es["ci_lo"], 5) if es["ci_lo"] is not None else None,
                   "effect_ci_hi": round(es["ci_hi"], 5) if es["ci_hi"] is not None else None,
                   "rank_biserial": round(es["rank_biserial"], 4) if es["rank_biserial"] is not None else None,
                   "effect_size_all": round(es_all["effect_size"], 5) if es_all["effect_size"] is not None else None,
                   "effect_ci_all_lo": round(es_all["ci_lo"], 5) if es_all["ci_lo"] is not None else None,
                   "effect_ci_all_hi": round(es_all["ci_hi"], 5) if es_all["ci_hi"] is not None else None}
            rows.append(row)
            wilcox_ps.append(row["wilcoxon_p"] if row["wilcoxon_p"] is not None else 1.0)

    q = _bh_fdr(wilcox_ps)
    for row, qv in zip(rows, q):
        row["wilcoxon_fdr_q"] = round(float(qv), 5)

    out = {"note": "interictal axis aligns with ictal early-activation spatial gradient "
                   "(sign-free / collinear); NOT path replay. Exploratory family, FDR-controlled.",
           "MIN_EFF": MIN_EFF, "evidence_tier": TIER, "table": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "axis_alignment_FINAL.json", "w"), indent=2, ensure_ascii=False)

    # markdown table for eyeballing
    lines = ["# Topic5 A-line — final tiered, FDR-controlled table (Epilepsiae primary cohort)",
             "",
             "interictal axis vs ictal early-activation SPATIAL GRADIENT (sign-free / collinear).",
             "`n_pass`/`n` beat that null's 95th-pct; `n_adequate` = subjects whose null actually",
             f"shuffled >= {MIN_EFF} channels. FDR = Benjamini-Hochberg over all metric x null Wilcoxon p.",
             "",
             "`eff` / `eff 95%CI` / `rb` are over the **n_adequate** subjects (= `adeq` column;",
             "the joint layer drops degenerate nulls, so these are NOT over all 18). `Wadeq` =",
             "Wilcoxon p recomputed over the same adequate set. `Wilcox p` + `FDR q` stay over the",
             "runner's full filter. The HFA-joint gate uses `Wadeq` + adequate `eff 95%CI`, NOT the",
             "all-subjects CI. bootstrap n_boot=10000 fixed seed; `rb` = rank-biserial in [-1,1].",
             "",
             "| metric | tier | B | null | n_pass/n | adeq | binom p | Wilcox p | Wadeq | FDR q | LOSO-worst p | eff | eff 95%CI | rb |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['metric']} | {r['tier']} | {r['B']} | {r['null']} | "
                     f"{r['n_pass']}/{r['n']} | {r['n_adequate_eff']} | {r['binom_p']} | "
                     f"{r['wilcoxon_p']} | {r['wilcoxon_p_adequate']} | {r['wilcoxon_fdr_q']} | "
                     f"{r['loso_worst_wilcoxon_p']} | "
                     f"{r['effect_size']} | [{r['effect_ci_lo']}, {r['effect_ci_hi']}] | {r['rank_biserial']} |")
    (OUT / "axis_alignment_FINAL.md").write_text("\n".join(lines))
    print(f"wrote {OUT/'axis_alignment_FINAL.json'} + .md ({len(rows)} metric x null rows)", flush=True)
    # quick console summary of the PRIMARY metric (broadband)
    print("\n=== PRIMARY (broadband) ===")
    for r in rows:
        if r["metric"] == "broadband":
            print(f"  B={r['B']} {r['null']:14s}: {r['n_pass']}/{r['n']} (adeq {r['n_adequate_eff']}) "
                  f"Wilcox p={r['wilcoxon_p']} FDR q={r['wilcoxon_fdr_q']} LOSO-worst={r['loso_worst_wilcoxon_p']}")


if __name__ == "__main__":
    main()
