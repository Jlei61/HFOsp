"""Topic 5 V2 Phase-1-v2 / Workstream W2 ("who?") — Task 2.1.

Continuous per-subject phenotype artifact. Pure post-processing of existing
Phase-1 artifacts (n_perm=1000): NO new nulls, NO simulation, NO re-running
alignment. One output row per (subject x substrate) across BOTH substrates.

Tier LOCK (global): ceiling is exploratory *candidate scaffold refinement*.
The cohort-level early-ictal spatial recruitment scaffold is an AGGREGATE;
per-subject multi-band consistency is weak (this artifact quantifies exactly
that spread). This file must never assert "formal Gate A passed", spatial
random-field exceedance, HFO/LVFA/ripple specificity, timing-order replay,
criticality, mechanism, or "every subject is stable".

Inputs per substrate dir (READ-ONLY):
  phase1_null_raw_subject_summary.csv       (spatial/order self-null per band)
  phase1_alignment_raw_subject_summary.csv  (per-subject aligned maxAB per band)
  phase1_alignment_raw_seizure_summary.csv  (per-seizure aligned maxAB per band)

Output: results/topic5_ictal_recruitment/v2_band_scan/phase1_v2_subject_phenotype.csv
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# --- Band contract (authoritative; see Task 2.1 brief "Band contract") --------
PRIMARY_BANDS = [
    "delta_HYP_slow",
    "theta_preictal_PAC",
    "alpha_sharp_leq13",
    "beta_LVFA_low",
    "gamma_LVFA",
    "hg_low_ripple",
    "ripple_high",
]
LOW_BAND = ["delta_HYP_slow", "theta_preictal_PAC", "alpha_sharp_leq13"]   # 1-13 Hz
LVFA_BAND = ["beta_LVFA_low", "gamma_LVFA"]                                # 13-80 Hz
HFA_RIPPLE = ["hg_low_ripple", "ripple_high"]                             # 80-250 Hz

# Subjects whose Phase-1 alignment anchored on eeg_onset (all others clin_onset).
EEG_ONSET_SUBJECTS = {"xuxinyi", "zhangkexuan"}

# Output column order (subject/substrate first, then features, then provenance).
_OUTPUT_COLUMNS = [
    "subject",
    "substrate",
    "mean_delta_7bands",
    "median_delta_7bands",
    "n_positive_delta_7bands",
    "n_sig_7bands",
    "band_genericity_index",
    "low_band_score",
    "LVFA_band_score",
    "HFA_ripple_score",
    "HF_minus_low",
    "ripple_rank",
    "profile_entropy",
    "within_subject_seizure_consistency",
    "n_sz",
    "n_contacts",
    "maxab_primary",
    "spatial_strength",
    "order_strength",
    "anchor",
    "tier",
]


# --- small helpers ------------------------------------------------------------
def _mode(series: pd.Series):
    """Most common non-null value; NaN if the series is all-null/empty."""
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan


def _scalar_int(series: pd.Series) -> int:
    """Median of a per-band integer-valued column collapsed to one scalar.

    n_seizures can vary by band (HFA bands drop seizures with too few contacts);
    the median over the 7 primary bands equals the full distinct-seizure count.
    """
    vals = series.dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return 0
    return int(round(float(np.median(vals))))


def _descending_rank(deltas: np.ndarray, primary_bands, target: str):
    """Rank of `target` band's delta among the 7 (rank 1 = largest; min-rank ties)."""
    idx = primary_bands.index(target)
    v = deltas[idx]
    if not np.isfinite(v):
        return float("nan")
    finite = deltas[np.isfinite(deltas)]
    return int(1 + np.sum(finite > v))


def _positive_profile_entropy(deltas: np.ndarray) -> float:
    """Shannon entropy of the normalized POSITIVE part of the delta vector,
    normalized to [0, 1] by dividing by ln(n_bands).

    Interpretation: how evenly the (positive) alignment advantage is spread
    across bands. 1.0 = spread evenly over all bands (generic multi-band);
    near 0 = concentrated in a single band. If every delta <= 0 -> 0.0.
    """
    n = len(deltas)
    pos = np.clip(deltas, 0.0, None)
    pos = pos[np.isfinite(pos)]
    total = float(pos.sum())
    if total <= 0.0:
        return 0.0
    q = pos[pos > 0.0] / total
    h = float(-np.sum(q * np.log(q)))
    return h / np.log(n)


def _seizure_consistency(seizure_df: pd.DataFrame, primary_bands) -> float:
    """Cross-seizure stability of the aligned maxAB scaffold.

    Per seizure: median align_abs_maxab over the 7 primary bands -> one scalar.
    Across seizures: 1 - IQR/|median|.  NaN if <2 usable seizures or |median|~0.
    """
    sz = seizure_df[seizure_df["band"].isin(primary_bands)]
    if sz.empty:
        return float("nan")
    per_sz = sz.groupby("seizure")["align_abs_maxab"].median().to_numpy(dtype=float)
    per_sz = per_sz[np.isfinite(per_sz)]
    if per_sz.size < 2:
        return float("nan")
    med = float(np.median(per_sz))
    if abs(med) < 1e-9:
        return float("nan")
    q1, q3 = np.percentile(per_sz, [25.0, 75.0])
    iqr = float(q3 - q1)
    return float(1.0 - iqr / abs(med))


def assign_tier(n_sig_7bands: int, n_positive_delta_7bands: int) -> str:
    """Three-tier subject label (Task 2.2 brief; thresholds + precedence LOCKED).

    strong:       n_sig_7bands >= 4
    directional:  n_positive_delta_7bands >= 5  AND  n_sig_7bands < 4
    weak_absent:  n_positive_delta_7bands < 5

    Precedence: `strong` wins whenever n_sig_7bands >= 4, regardless of
    n_positive_delta_7bands; else `directional`; else `weak_absent`. Subject-level
    significance is very sensitive to seizure count -- a subject with only 2-3
    seizures can fail every band's p<0.05 test even with all 7 bands positive.
    `directional` names that "consistent direction but underpowered" case
    separately from truly weak/absent subjects (brief "Why three tiers").
    """
    if n_sig_7bands >= 4:
        return "strong"
    if n_positive_delta_7bands >= 5:
        return "directional"
    return "weak_absent"


# --- Task 2.3: descriptive band-gradient bucket -------------------------------
def assign_band_profile_group(hf_minus_low: float, band_genericity_index: float,
                              tier: str) -> str:
    """Descriptive band-gradient bucket for ONE subject (Task 2.3 brief §Step 2).

    This is a DESCRIPTIVE bucketing label, NOT a KMeans / statistical subtype
    claim and the numbers below are NOT thresholds of any test -- they are
    hand-picked descriptive cutoffs on the signed HFA-minus-low gradient.

        weak_absent   : tier == 'weak_absent'  (assigned FIRST -- takes precedence)
        low_leaning   : HF_minus_low < -0.05
        hf_leaning    : HF_minus_low > +0.05
        flat_generic  : |HF_minus_low| <= 0.05 AND band_genericity_index >= 0.6

    The three non-weak_absent buckets split "the rest": in both real pools every
    non-weak_absent subject with a flat (|HF_minus_low| <= 0.05) gradient is also
    band-generic (>= 0.6), so `flat_lowgeneric` never occurs -- it exists only so
    the function is total (never returns None) for hypothetical inputs.
    """
    if tier == "weak_absent":
        return "weak_absent"
    if hf_minus_low < -0.05:
        return "low_leaning"
    if hf_minus_low > 0.05:
        return "hf_leaning"
    if band_genericity_index >= 0.6:
        return "flat_generic"
    return "flat_lowgeneric"  # flat gradient but not band-generic (empty in real data)


# --- Task 2.3: Spearman phenotype-hunt gate -----------------------------------
# Candidate predictors screened against multi-band positivity (brief §Step 1).
PHENOTYPE_TARGETS = ["n_sig_7bands", "band_genericity_index"]
PHENOTYPE_PREDICTORS = [
    "n_sz", "n_contacts", "maxab_primary", "within_subject_seizure_consistency",
    "profile_entropy", "HF_minus_low", "low_band_score", "LVFA_band_score",
    "HFA_ripple_score",
]
# LOCKED gate: only call a feature a "predictor" if |r| > GATE_R AND p < GATE_P.
GATE_R = 0.4
GATE_P = 0.05


def spearman_phenotype_gate(x, y, r_thresh: float = GATE_R, p_thresh: float = GATE_P):
    """Spearman correlation + the LOCKED phenotype-hunt gate for two feature vectors.

    Returns (r, p, n, passes). NaN pairs (a NaN in either vector) are dropped
    first; `n` is the finite-pair count. `passes` is True ONLY when
    |r| > r_thresh AND p < p_thresh (both required -- the AND is load-bearing:
    a high |r| that is not significant at small n does NOT pass). A constant
    input (<2 distinct values) or n < 3 yields NaN r/p and passes=False.
    This is a descriptive report filter, NOT a formal test threshold.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    xv, yv = x[mask], y[mask]
    if n < 3 or np.unique(xv).size < 2 or np.unique(yv).size < 2:
        return (float("nan"), float("nan"), n, False)
    r, p = spearmanr(xv, yv)
    passes = bool(np.isfinite(r) and np.isfinite(p)
                  and abs(r) > r_thresh and p < p_thresh)
    return (float(r), float(p), n, passes)


# --- core: one subject --------------------------------------------------------
def subject_profile(null_df: pd.DataFrame, align_df: pd.DataFrame,
                    seizure_df: pd.DataFrame, primary_bands) -> dict:
    """Compute the continuous phenotype features for ONE subject.

    Each input frame must be pre-filtered to a single subject. Returns a dict
    of feature values (excluding `substrate`, which the caller attaches).
    """
    subj_ids = null_df["subject"].astype(str).unique()
    assert len(subj_ids) == 1, f"subject_profile expects one subject, got {subj_ids!r}"
    subject = str(subj_ids[0])

    # spatial null frame over the 7 primary bands (spatial is the primary endpoint)
    spat = null_df[(null_df["feature"] == "raw")
                   & (null_df["null_type"] == "spatial")
                   & (null_df["band"].isin(primary_bands))]
    delta_by_band = spat.groupby("band")["delta"].median()
    p_by_band = spat.groupby("band")["empirical_p"].median()

    deltas = np.array([float(delta_by_band.get(b, np.nan)) for b in primary_bands])
    ps = np.array([float(p_by_band.get(b, np.nan)) for b in primary_bands])

    positive = deltas > 0.0                       # NaN -> False
    n_positive = int(np.sum(positive))
    # n_sig LOCK: spatial delta>0 AND empirical_p<0.05 over the 7 primary bands.
    n_sig = int(np.sum(positive & (ps < 0.05)))   # NaN comparisons -> False

    def _grp_median(bands):
        return float(np.nanmedian([float(delta_by_band.get(b, np.nan)) for b in bands]))

    low_score = _grp_median(LOW_BAND)
    lvfa_score = _grp_median(LVFA_BAND)
    hfa_score = _grp_median(HFA_RIPPLE)

    # align-subject frame over the 7 primary bands (feature=='raw' only)
    al = align_df[(align_df["feature"] == "raw") & (align_df["band"].isin(primary_bands))]
    maxab_vals = al["align_abs_maxab"].to_numpy(dtype=float)
    if maxab_vals.size == 0 or not np.isfinite(maxab_vals).any():
        maxab_primary = float("nan")  # empty / all-NaN align -> clean NaN, no RuntimeWarning
    else:
        maxab_primary = float(np.nanmedian(maxab_vals))

    # per-seizure frame (feature=='raw' only); n_sz = distinct-seizure count, robust to
    # HFA bands dropping seizures. _seizure_consistency filters further to primary bands.
    sz_raw = seizure_df[seizure_df["feature"] == "raw"]
    n_sz = int(sz_raw["seizure"].nunique())

    return {
        "subject": subject,
        "mean_delta_7bands": float(np.nanmean(deltas)),
        "median_delta_7bands": float(np.nanmedian(deltas)),
        "n_positive_delta_7bands": n_positive,
        "n_sig_7bands": n_sig,
        "band_genericity_index": n_positive / len(primary_bands),
        "low_band_score": low_score,
        "LVFA_band_score": lvfa_score,
        "HFA_ripple_score": hfa_score,
        "HF_minus_low": hfa_score - low_score,
        "ripple_rank": _descending_rank(deltas, primary_bands, "ripple_high"),
        "profile_entropy": _positive_profile_entropy(deltas),
        "within_subject_seizure_consistency": _seizure_consistency(sz_raw, primary_bands),
        "n_sz": n_sz,
        "n_contacts": _scalar_int(al["n_contacts"]),
        "maxab_primary": maxab_primary,
        "spatial_strength": _mode(spat["spatial_null_strength"]),
        "order_strength": _mode(spat["order_null_strength"]),
        "anchor": "eeg_onset" if subject in EEG_ONSET_SUBJECTS else "clin_onset",
    }


# --- io + cohort assembly -----------------------------------------------------
def load_substrate_frames(substrate_dir):
    """Load the three per-substrate Phase-1 CSVs (subject ids coerced to str)."""
    substrate_dir = Path(substrate_dir)
    null_df = pd.read_csv(substrate_dir / "phase1_null_raw_subject_summary.csv")
    align_df = pd.read_csv(substrate_dir / "phase1_alignment_raw_subject_summary.csv")
    seizure_df = pd.read_csv(substrate_dir / "phase1_alignment_raw_seizure_summary.csv")
    for d in (null_df, align_df, seizure_df):
        d["subject"] = d["subject"].astype(str)
    return null_df, align_df, seizure_df


def build_phenotype_table(null_df, align_df, seizure_df, substrate, primary_bands):
    """One phenotype row per subject in a substrate; adds the `substrate` and `tier` columns."""
    rows = []
    for subject in sorted(null_df["subject"].astype(str).unique()):
        nd = null_df[null_df["subject"].astype(str) == subject]
        ad = align_df[align_df["subject"].astype(str) == subject]
        sd = seizure_df[seizure_df["subject"].astype(str) == subject]
        prof = subject_profile(nd, ad, sd, primary_bands)
        prof["substrate"] = substrate
        prof["tier"] = assign_tier(prof["n_sig_7bands"], prof["n_positive_delta_7bands"])
        rows.append(prof)
    df = pd.DataFrame(rows)
    return df.reindex(columns=_OUTPUT_COLUMNS)


def main(argv=None):
    project_root = Path(__file__).resolve().parents[1]
    default_root = project_root / "results" / "topic5_ictal_recruitment" / "v2_band_scan"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=default_root,
                    help="v2_band_scan root containing narrow/ and broad/ subdirs")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: <root>/phase1_v2_subject_phenotype.csv)")
    args = ap.parse_args(argv)

    out_path = args.out or (args.root / "phase1_v2_subject_phenotype.csv")
    tables = []
    for substrate in ("narrow", "broad"):
        frames = load_substrate_frames(args.root / substrate)
        tables.append(build_phenotype_table(*frames, substrate, PRIMARY_BANDS))
    out_df = pd.concat(tables, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Median convention pinned here (downstream 2.2/2.3 build on this): the narrow pool
    # is even-n (20) so np.median = 2.5 while statistics.median_low = 2; the Phase-1
    # archive "2" is median_low of the identical n_sig distribution (a median
    # convention, NOT a data change). count(n_sig>=4) [n_ge4/n_total] is the
    # convention-free, load-bearing statistic. All three are reported per pool.
    for substrate in ("narrow", "broad"):
        sub = out_df[out_df["substrate"] == substrate]
        n_sig = sub["n_sig_7bands"].astype(int).tolist()
        n_ge4 = int(sum(v >= 4 for v in n_sig))
        ge4 = sorted(sub.loc[sub["n_sig_7bands"] >= 4, "subject"].astype(str))
        print(f"[{substrate}] n_total={len(n_sig)} "
              f"median_n_sig={float(np.median(n_sig)):.2f} "
              f"median_n_sig_low={statistics.median_low(n_sig)} "
              f"n_ge4={n_ge4}/{len(n_sig)} ge4={ge4}")
    print(f"wrote {out_path} ({len(out_df)} rows)")
    return out_df


if __name__ == "__main__":
    main()
