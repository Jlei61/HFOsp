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
from pathlib import Path

import numpy as np
import pandas as pd

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

    # align-subject frame over primary bands
    al = align_df[align_df["band"].isin(primary_bands)]
    maxab_primary = float(np.nanmedian(al["align_abs_maxab"].to_numpy(dtype=float)))

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
        "within_subject_seizure_consistency": _seizure_consistency(seizure_df, primary_bands),
        "n_sz": _scalar_int(al["n_seizures"]),
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
    """One phenotype row per subject in a substrate; adds the `substrate` column."""
    rows = []
    for subject in sorted(null_df["subject"].astype(str).unique()):
        nd = null_df[null_df["subject"].astype(str) == subject]
        ad = align_df[align_df["subject"].astype(str) == subject]
        sd = seizure_df[seizure_df["subject"].astype(str) == subject]
        prof = subject_profile(nd, ad, sd, primary_bands)
        prof["substrate"] = substrate
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

    for substrate in ("narrow", "broad"):
        sub = out_df[out_df["substrate"] == substrate]
        n_sig = sub["n_sig_7bands"].astype(int)
        ge4 = sorted(sub.loc[n_sig >= 4, "subject"].astype(str))
        print(f"[{substrate}] n_subj={len(sub)} "
              f"median_n_sig={float(np.median(n_sig)):.1f} "
              f"count(>=4)={int((n_sig >= 4).sum())} ge4={ge4}")
    print(f"wrote {out_path} ({len(out_df)} rows)")
    return out_df


if __name__ == "__main__":
    main()
