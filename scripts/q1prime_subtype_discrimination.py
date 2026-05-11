"""Stage D: Feature × subtype discrimination analysis.

Per subject (where topic5 status=ok + n_subtypes >= 2):
  For each feature: KW (k>=3) or MW (k=2) on (feature × subtype_label, excluding -1)
  Effect size: epsilon^2 (KW) or rank-biserial r (MW)

Cohort sign-test: for each feature, count subjects where p < 0.05 vs binomial(n, 0.05).

Output:
  results/topic1_topic5_bridge/q1prime_feature_subtype_discrimination.csv
  results/topic1_topic5_bridge/q1prime_subtype_discrimination_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

REPO = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(REPO))

from src.topic1_topic5_bridge import (  # noqa: E402
    _kruskal_wallis_with_effect,
    _mann_whitney_with_effect,
)

RESULTS = REPO / "results"
FEATURES_CSV = RESULTS / "topic1_topic5_bridge" / "q1prime_features.csv"
OUT_DIR = RESULTS / "topic1_topic5_bridge"

FEATURE_COLS = [
    "n_active",
    "active_fraction",
    "onset_spread_sec",
    "median_onset_latency_sec",
]
MIN_PER_SUBTYPE = 2  # minimum seizures per subtype for test


def per_subject_subtype_test(df: pd.DataFrame) -> pd.DataFrame:
    """Run per-subject feature × subtype test."""
    rows = []
    for (ds, sid), subdf in df.groupby(["dataset", "subject"]):
        topic5_status = subdf["topic5_status"].iloc[0]
        n_subtypes = subdf["subtype_label"].nunique()

        # Skip: insufficient_n or no multi-subtype structure
        # Include subtype=-1 data in table but exclude from tests
        main_sub = subdf[subdf["subtype_label"] >= 0]  # exclude outlier=-1
        unique_subtypes = sorted(main_sub["subtype_label"].dropna().unique().tolist())

        if topic5_status != "ok" or len(unique_subtypes) < 2:
            for feat in FEATURE_COLS:
                rows.append({
                    "dataset": ds, "subject": sid, "feature": feat,
                    "topic5_status": topic5_status, "n_subtypes": len(unique_subtypes),
                    "n_total_eligible": len(main_sub),
                    "pval": float("nan"), "effect": float("nan"),
                    "effect_type": None, "direction": None,
                    "skip_reason": f"topic5_status={topic5_status}" if topic5_status != "ok" else "single_subtype"
                })
            continue

        for feat in FEATURE_COLS:
            groups = []
            group_ns = []
            for st in unique_subtypes:
                g = main_sub.loc[main_sub["subtype_label"] == st, feat].values.astype(float)
                groups.append(g[np.isfinite(g)])
                group_ns.append(len(g[np.isfinite(g)]))

            # Check minimum per-subtype counts
            if any(n < MIN_PER_SUBTYPE for n in group_ns):
                rows.append({
                    "dataset": ds, "subject": sid, "feature": feat,
                    "topic5_status": topic5_status, "n_subtypes": len(unique_subtypes),
                    "n_total_eligible": len(main_sub),
                    "pval": float("nan"), "effect": float("nan"),
                    "effect_type": None, "direction": None,
                    "skip_reason": f"min_per_subtype<{MIN_PER_SUBTYPE}: {group_ns}"
                })
                continue

            if len(unique_subtypes) == 2:
                p, eff = _mann_whitney_with_effect(groups[0], groups[1])
                eff_type = "rank_biserial_r"
                direction = "pos" if eff > 0 else ("neg" if eff < 0 else "zero")
            else:
                p, eff = _kruskal_wallis_with_effect(groups)
                eff_type = "epsilon_squared"
                # For KW, direction = subtype 0 vs (1,...) median difference
                m0 = float(np.nanmedian(groups[0])) if groups[0].size > 0 else float("nan")
                m1_all = np.concatenate([g for g in groups[1:]]) if len(groups) > 1 else np.array([])
                m1 = float(np.nanmedian(m1_all)) if m1_all.size > 0 else float("nan")
                direction = "pos" if m0 > m1 else ("neg" if m0 < m1 else "zero")

            rows.append({
                "dataset": ds, "subject": sid, "feature": feat,
                "topic5_status": topic5_status, "n_subtypes": len(unique_subtypes),
                "n_total_eligible": len(main_sub),
                "pval": float(p), "effect": float(eff),
                "effect_type": eff_type, "direction": direction,
                "skip_reason": None,
                "group_ns": str(group_ns),
            })
    return pd.DataFrame(rows)


def cohort_sign_test(ps_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """For each feature, count subjects with p < alpha; compare to binomial(n, alpha)."""
    out = {}
    eligible = ps_df[ps_df["skip_reason"].isna()]
    for feat in FEATURE_COLS:
        fe = eligible[eligible["feature"] == feat]
        n_total = len(fe)
        n_sig = (fe["pval"] < alpha).sum()
        n_strong = (fe["effect"].abs() > 0.3).sum()
        # Binomial test: how surprising is n_sig out of n_total at p_null=alpha?
        if n_total > 0:
            binom_p = float(sp_stats.binomtest(int(n_sig), int(n_total), alpha, alternative="greater").pvalue)
        else:
            binom_p = float("nan")

        # Direction sign test (signed effects only)
        signs = fe.loc[fe["direction"].notna() & (fe["direction"] != "zero"), "direction"].tolist()
        n_pos_sign = sum(1 for s in signs if s == "pos")
        n_neg_sign = sum(1 for s in signs if s == "neg")
        if n_pos_sign + n_neg_sign > 0:
            sign_p = float(sp_stats.binomtest(n_pos_sign, n_pos_sign + n_neg_sign, 0.5, alternative="two-sided").pvalue)
        else:
            sign_p = float("nan")

        out[feat] = {
            "n_eligible_subjects": int(n_total),
            "n_sig_p05": int(n_sig),
            "n_strong_effect_03": int(n_strong),
            "binom_p_against_null": float(binom_p),
            "median_abs_effect": float(fe["effect"].abs().median()) if n_total > 0 else float("nan"),
            "median_pval": float(fe["pval"].median()) if n_total > 0 else float("nan"),
            "direction_n_pos": n_pos_sign,
            "direction_n_neg": n_neg_sign,
            "direction_sign_test_p": sign_p,
        }
    return out


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} rows")

    # Run per-subject tests
    ps_df = per_subject_subtype_test(df)
    out_csv = OUT_DIR / "q1prime_feature_subtype_discrimination.csv"
    ps_df.to_csv(out_csv, index=False)
    print(f"Wrote per-subject discrimination CSV: {out_csv}")

    # Cohort sign test
    sign_results = cohort_sign_test(ps_df)

    print("\n=== Cohort Sign-Test Results ===")
    for feat, s in sign_results.items():
        print(f"\n{feat}:")
        print(f"  n_eligible: {s['n_eligible_subjects']}, n_sig (p<0.05): {s['n_sig_p05']}, n_strong (|eff|>0.3): {s['n_strong_effect_03']}")
        print(f"  Binomial p (against null): {s['binom_p_against_null']:.4f}")
        print(f"  Median |effect|: {s['median_abs_effect']:.3f}, Median p: {s['median_pval']:.3f}")
        print(f"  Direction: {s['direction_n_pos']} pos / {s['direction_n_neg']} neg, sign_p={s['direction_sign_test_p']:.3f}")

    # Show significant per-subject results
    eligible = ps_df[ps_df["skip_reason"].isna()]
    sig = eligible[eligible["pval"] < 0.05]
    print(f"\n=== Significant per-subject results (p<0.05), n={len(sig)} ===")
    for _, row in sig.iterrows():
        print(f"  {row['feature']} × {row['dataset']}_{row['subject']}: p={row['pval']:.4f}, eff={row['effect']:.3f} ({row['effect_type']}), dir={row['direction']}, n={row['n_total_eligible']}")

    # Skip reasons summary
    skipped = ps_df[ps_df["skip_reason"].notna()]
    print(f"\n=== Skip reasons ({len(skipped)} feature×subject entries) ===")
    print(skipped.groupby("skip_reason")["subject"].count().to_string())

    # Write summary JSON
    payload = {
        "schema_version": 1,
        "features": FEATURE_COLS,
        "min_per_subtype": MIN_PER_SUBTYPE,
        "alpha": 0.05,
        "cohort_sign_test": sign_results,
        "n_sig_any_feature": int(len(sig)),
    }
    out_json = OUT_DIR / "q1prime_subtype_discrimination_summary.json"
    with out_json.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nWrote summary JSON: {out_json}")


if __name__ == "__main__":
    main()
