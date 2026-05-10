"""Stage C: Feature × delta_rho cross-correlation analysis.

Per-subject Spearman(feature_i, delta_rho) across seizures within subject.
Cohort-pooled: pool all seizures, compute Spearman(feature_i, delta_rho) [flagged caveat].
Optional: statsmodels mixed-effects if available.

Output:
  results/topic1_topic5_bridge/q1prime_feature_delta_rho_correlation.csv
  (feature × subject matrix of rho + p values + directional consistency)
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

RESULTS = REPO / "results"
FEATURES_CSV = RESULTS / "topic1_topic5_bridge" / "q1prime_features.csv"
OUT_DIR = RESULTS / "topic1_topic5_bridge"

# Features to correlate with delta_rho (exclude degenerate ones)
FEATURE_COLS = [
    "n_active",
    "active_fraction",
    "onset_spread_sec",
    "median_onset_latency_sec",
    # fast_recruit_fraction excluded — mostly 0 (degenerate)
]

DELTA_RHO_COLS = ["delta_rho_swap", "delta_rho_full"]
MIN_PAIRS_PER_SUBJECT = 4  # minimum valid (feature, delta_rho) pairs for Spearman


def per_subject_spearman(
    df: pd.DataFrame, feature: str, delta_rho_col: str
) -> pd.DataFrame:
    """Compute Spearman(feature, delta_rho_col) per subject.

    Returns DataFrame with columns:
      subject, dataset, n_valid, rho, pval, direction
    """
    rows = []
    for (ds, sid), subdf in df.groupby(["dataset", "subject"]):
        valid = subdf[[feature, delta_rho_col]].dropna()
        valid = valid[valid[feature].apply(np.isfinite) & valid[delta_rho_col].apply(np.isfinite)]
        n = len(valid)
        if n < MIN_PAIRS_PER_SUBJECT:
            rows.append({
                "dataset": ds, "subject": sid, "feature": feature,
                "delta_rho_col": delta_rho_col, "n_valid": n,
                "rho": float("nan"), "pval": float("nan"), "direction": None,
                "skip_reason": f"n<{MIN_PAIRS_PER_SUBJECT}"
            })
            continue
        try:
            r, p = sp_stats.spearmanr(valid[feature].values, valid[delta_rho_col].values)
        except Exception as e:
            rows.append({
                "dataset": ds, "subject": sid, "feature": feature,
                "delta_rho_col": delta_rho_col, "n_valid": n,
                "rho": float("nan"), "pval": float("nan"), "direction": None,
                "skip_reason": str(e)
            })
            continue
        rows.append({
            "dataset": ds, "subject": sid, "feature": feature,
            "delta_rho_col": delta_rho_col, "n_valid": n,
            "rho": float(r) if np.isfinite(r) else float("nan"),
            "pval": float(p) if np.isfinite(p) else float("nan"),
            "direction": "pos" if r > 0 else ("neg" if r < 0 else "zero"),
            "skip_reason": None
        })
    return pd.DataFrame(rows)


def cohort_pooled_spearman(
    df: pd.DataFrame, feature: str, delta_rho_col: str
) -> dict:
    """Pool all subjects, compute Spearman. CAVEAT: mixes T0/T1 conventions."""
    valid = df[[feature, delta_rho_col]].dropna()
    valid = valid[valid[feature].apply(np.isfinite) & valid[delta_rho_col].apply(np.isfinite)]
    n = len(valid)
    if n < 3:
        return {"n": n, "rho": float("nan"), "pval": float("nan"), "caveat": "insufficient_n"}
    try:
        r, p = sp_stats.spearmanr(valid[feature].values, valid[delta_rho_col].values)
    except Exception as e:
        return {"n": n, "rho": float("nan"), "pval": float("nan"), "caveat": str(e)}
    return {
        "n": n,
        "rho": float(r),
        "pval": float(p),
        "caveat": "CROSS-SUBJECT T0/T1 CONVENTION MIXED — pooled rho is indicative only"
    }


def directional_consistency(rho_values: list[float]) -> dict:
    """Given a list of per-subject rho values (finite only), compute directional stats."""
    vals = [v for v in rho_values if np.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "sign_test_p": float("nan"), "median_rho": float("nan")}
    n_pos = sum(1 for v in vals if v > 0)
    n_neg = sum(1 for v in vals if v < 0)
    # Sign test: binomial test (n_pos vs n/2)
    from scipy.stats import binomtest
    if n_pos + n_neg > 0:
        sign_p = float(binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided").pvalue)
    else:
        sign_p = float("nan")
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "sign_test_p": sign_p,
        "median_rho": float(np.median(vals)),
        "median_abs_rho": float(np.median([abs(v) for v in vals])),
    }


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} rows from {FEATURES_CSV}")

    all_per_subject_rows = []
    cohort_summary = {}
    directional_summary = {}

    for delta_col in DELTA_RHO_COLS:
        cohort_summary[delta_col] = {}
        directional_summary[delta_col] = {}
        for feat in FEATURE_COLS:
            ps = per_subject_spearman(df, feat, delta_col)
            all_per_subject_rows.append(ps)

            # Cohort-pooled
            pooled = cohort_pooled_spearman(df, feat, delta_col)
            cohort_summary[delta_col][feat] = pooled

            # Directional consistency across subjects with n_valid >= MIN_PAIRS_PER_SUBJECT
            eligible = ps[ps["skip_reason"].isna()]
            dircons = directional_consistency(eligible["rho"].tolist())
            directional_summary[delta_col][feat] = dircons

            # Print summary
            n_sig = (eligible["pval"] < 0.05).sum()
            n_strong = (eligible["rho"].abs() > 0.3).sum()
            print(f"\n{delta_col} × {feat}:")
            print(f"  Subjects with n>={MIN_PAIRS_PER_SUBJECT}: {len(eligible)}/{ps['subject'].nunique()}")
            print(f"  Median |ρ|: {dircons['median_abs_rho']:.3f}, Median ρ: {dircons['median_rho']:.3f}")
            print(f"  n_sig (p<0.05): {n_sig}, n_strong (|ρ|>0.3): {n_strong}")
            print(f"  Sign: {dircons['n_pos']} pos / {dircons['n_neg']} neg, binomial p={dircons['sign_test_p']:.3f}")
            print(f"  Pooled ρ: {pooled['rho']:.3f} (p={pooled['pval']:.4f}, n={pooled['n']}) [CAVEAT: mixed conventions]")

    # Write per-subject correlation CSV
    all_ps_df = pd.concat(all_per_subject_rows, ignore_index=True)
    out_corr = OUT_DIR / "q1prime_feature_delta_rho_correlation.csv"
    all_ps_df.to_csv(out_corr, index=False)
    print(f"\nWrote per-subject correlation CSV: {out_corr}")

    # Write cohort summary JSON
    payload = {
        "schema_version": 1,
        "features": FEATURE_COLS,
        "delta_rho_cols": DELTA_RHO_COLS,
        "min_pairs_per_subject": MIN_PAIRS_PER_SUBJECT,
        "cohort_pooled": cohort_summary,
        "directional_consistency": directional_summary,
        "caveat": "Cohort-pooled Spearman mixes subject-specific T0/T1 conventions — use directional consistency as primary readout",
    }
    out_json = OUT_DIR / "q1prime_feature_correlation_summary.json"
    with out_json.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"Wrote correlation summary JSON: {out_json}")

    # Ranking features by directional consistency
    print("\n=== FEATURE RANKING by directional consistency (delta_rho_swap) ===")
    col = "delta_rho_swap"
    ranking = []
    for feat in FEATURE_COLS:
        d = directional_summary[col][feat]
        ranking.append((feat, d["n_strong"] if "n_strong" in d else 0, d["median_abs_rho"], d["median_rho"], d["sign_test_p"]))

    # Re-compute n_strong for ranking
    print(f"{'Feature':<35} {'median|ρ|':>10} {'medianρ':>10} {'sign_p':>10} {'n_pos':>6} {'n_neg':>6}")
    for feat in FEATURE_COLS:
        d = directional_summary[col][feat]
        eligible = all_ps_df[
            (all_ps_df["feature"] == feat) &
            (all_ps_df["delta_rho_col"] == col) &
            (all_ps_df["skip_reason"].isna())
        ]
        n_strong = (eligible["rho"].abs() > 0.3).sum()
        print(f"{feat:<35} {d['median_abs_rho']:>10.3f} {d['median_rho']:>10.3f} {d['sign_test_p']:>10.3f} {d['n_pos']:>6} {d['n_neg']:>6}  n_strong={n_strong}")

    # Top features by n subjects with |ρ|>0.3
    print("\n=== Per-subject strong correlations (|ρ|>0.3) by feature × delta_rho_swap ===")
    for feat in FEATURE_COLS:
        eligible = all_ps_df[
            (all_ps_df["feature"] == feat) &
            (all_ps_df["delta_rho_col"] == "delta_rho_swap") &
            (all_ps_df["skip_reason"].isna())
        ]
        strong = eligible[eligible["rho"].abs() > 0.3]
        for _, row in strong.iterrows():
            print(f"  {feat} × {row['dataset']}_{row['subject']}: ρ={row['rho']:.3f}, p={row['pval']:.3f}, n={row['n_valid']}")


if __name__ == "__main__":
    main()
