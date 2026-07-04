"""Topic 5 V2 Phase-1-v2 / Workstream W3 ("when?") — Task 3.2.

Peri-ictal scaffold-score TRAJECTORY + subject-level SIGN-FLIP test.

Answers "when?": is the interictal-HFO-geometry / peri-ictal-energy-field alignment
(the candidate early-ictal spatial recruitment scaffold) PRE-ICTALLY present (static /
anatomy-like) or ONSET-TRIGGERED? Pure post-processing of the Step-0 dual-anchor
peri-ictal window-long CSVs (NO new nulls, NO simulation, NO re-running alignment).

Aggregation LOCK (statistical unit = SUBJECT; window -> seizure -> subject -> cohort):
  per window:               bg = median align_abs_maxab over the 7 PRIMARY bands
  per (subject,seizure,bin): median of bg over that bin's windows
  per (subject,bin):        median over that subject's seizures
  cohort_bin:               median over subjects
bins = the 5 epoch_region labels (far_pre, mid_pre, near_pre, peri_onset, early_post).

Main test = subject-level SIGN-FLIP permutation on the per-subject PAIRED contrast diffs
(statistic = median(d_i); null symmetric about 0; two-sided; exact 2^n enumeration for
n<=14 else n_perm>=10000 random). NOT window-label shuffle (window autocorrelation would
inflate). Wilcoxon signed-rank is a secondary check.

Dual anchor: EEG onset (PRIMARY, results/.../periictal/) + clinical onset (SENSITIVITY,
results/.../periictal_clin/). If they disagree, EEG-onset is the primary physiological read.

Tier LOCK (global): ceiling is exploratory *candidate scaffold refinement*. The "when?"
answer is the DESCRIPTIVE interpretation of trajectory shape + sign-flip p -- NOT
"criticality", NOT "mechanism", NOT "approach to bifurcation". band-generic / "NOT
ripple-specific" is allowed; "formal Gate A passed" / mechanism / criticality forbidden
even negated.

Outputs (results/topic5_ictal_recruitment/v2_band_scan/):
  phase1_v2_alignment_trajectory.csv    per (anchor, pool, subject, bin): subject_bin + n_seizures
  phase1_v2_trajectory_contrasts.csv    per (anchor, pool, contrast): cohort median diff,
                                        sign-flip p, wilcoxon p, n_subjects
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_topic5_v2_subject_phenotype import PRIMARY_BANDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/topic5_ictal_recruitment/v2_band_scan"

# Anchor -> peri-ictal output dir (Step 0). EEG = PRIMARY, clin = SENSITIVITY.
ANCHOR_DIRS = {"eeg": V2 / "periictal", "clin": V2 / "periictal_clin"}
POOLS = ["narrow", "broad"]

# The 5 peri-ictal bins in temporal order, with the figure bin centers (s rel EEG onset).
BIN_ORDER = ["far_pre", "mid_pre", "near_pre", "peri_onset", "early_post"]
BIN_CENTERS = {"far_pre": -80.0, "mid_pre": -45.0, "near_pre": -20.0,
               "peri_onset": 0.0, "early_post": 15.0}

# The 3 per-subject paired contrasts (name, higher bin, lower bin) -> diff = subject_bin[hi]-subject_bin[lo].
CONTRASTS = [
    ("near_pre_minus_far_pre", "near_pre", "far_pre"),
    ("post_minus_far_pre", "early_post", "far_pre"),
    ("post_minus_near_pre", "early_post", "near_pre"),
]

DEFAULT_N_PERM = 20000
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# aggregation: window -> seizure -> subject
# ---------------------------------------------------------------------------
def compute_subject_bins(df: pd.DataFrame, primary_bands) -> pd.DataFrame:
    """window(median over the PRESENT primary bands) -> seizure_bin(median over windows in bin)
    -> subject_bin(median over seizures). Returns long: [subject, epoch_region, subject_bin,
    n_seizures, median_n_bands].

    Only raw feature + the 7 primary bands + the 5 named epoch_region bins participate. A window is
    keyed by (subject, seizure, win_start_rel, win_end_rel) (bit-identical across that window's bands);
    epoch_region is constant within a window. Non-finite align_abs_maxab is dropped before the medians.

    Band-set transparency: the per-window score is the median over the primary bands PRESENT in that
    window. ~7% of windows carry only 5 of the 7 primary bands -- the two dropped are always the
    80-250 Hz HFA bands (hg_low_ripple, ripple_high), which get skipped together per window. Median-
    over-present is deliberate (a strict all-7 gate would further thin the sparser far-from-onset
    windows); n_bands is carried up as median_n_bands per (subject,bin) so the mix is auditable."""
    d = df[df["feature"] == "raw"]
    d = d[d["band"].isin(list(primary_bands))]
    d = d[d["epoch_region"].isin(BIN_ORDER)]
    d = d[np.isfinite(pd.to_numeric(d["align_abs_maxab"], errors="coerce"))].copy()
    if d.empty:
        return pd.DataFrame(columns=["subject", "epoch_region", "subject_bin",
                                     "n_seizures", "median_n_bands"])
    d["align_abs_maxab"] = d["align_abs_maxab"].astype(float)

    # per window: scaffold score = median over present primary bands; n_bands = # primary bands present
    win = d.groupby(["subject", "seizure", "win_start_rel", "win_end_rel"], as_index=False).agg(
        bg=("align_abs_maxab", "median"), n_bands=("band", "nunique"),
        epoch_region=("epoch_region", "first"))
    # per (subject, seizure, bin) = median of bg (+ median n_bands) over that bin's windows
    sz = win.groupby(["subject", "seizure", "epoch_region"], as_index=False).agg(
        bg=("bg", "median"), n_bands=("n_bands", "median"))
    # per (subject, bin) = median over that subject's seizures (+ #seizures, + median n_bands)
    subj = sz.groupby(["subject", "epoch_region"], as_index=False).agg(
        subject_bin=("bg", "median"), n_seizures=("bg", "size"), median_n_bands=("n_bands", "median"))
    return subj


def subject_bin_wide(subj_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long per-(subject,bin) table to subject x bin (columns reindexed to BIN_ORDER;
    a subject missing a bin gets NaN in that column -> excluded from any contrast touching it)."""
    if subj_long.empty:
        return pd.DataFrame(columns=BIN_ORDER)
    wide = subj_long.pivot(index="subject", columns="epoch_region", values="subject_bin")
    return wide.reindex(columns=BIN_ORDER)


def cohort_trajectory(subj_wide: pd.DataFrame) -> pd.DataFrame:
    """Per bin: cohort median + subject IQR (q25/q75) + n_subjects contributing (finite subject_bin)."""
    rows = []
    for b in BIN_ORDER:
        vals = subj_wide[b].to_numpy(dtype=float) if b in subj_wide else np.array([])
        vals = vals[np.isfinite(vals)]
        rows.append(dict(
            epoch_region=b, bin_center=BIN_CENTERS[b], n_subjects=int(vals.size),
            cohort_median=float(np.median(vals)) if vals.size else float("nan"),
            q25=float(np.percentile(vals, 25)) if vals.size else float("nan"),
            q75=float(np.percentile(vals, 75)) if vals.size else float("nan")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# subject-level sign-flip permutation (MAIN test)
# ---------------------------------------------------------------------------
def signflip_test(diffs, n_perm: int = DEFAULT_N_PERM, seed: int = DEFAULT_SEED) -> dict:
    """Subject-level sign-flip permutation on per-subject paired diffs d_i.

    Statistic = median(d_i). Null: each d_i is equally likely +/- (symmetric about 0). For n<=14
    ENUMERATE all 2^n sign vectors exactly (the identity all-+ vector is included, so the observed
    statistic is in the reference set); for n>14 draw n_perm>=10000 random +/-1 sign vectors under a
    fixed seed. Two-sided p:
       exact  : #{|perm_stat| >= |obs|} / 2^n            (identity is enumerated -> p>0)
       random : (1 + #{|perm_stat| >= |obs|}) / (1 + n_perm)   (Monte-Carlo, +1 for the observed)
    A tiny tolerance keeps float ties on the |obs| boundary counted (conservative). Returns
    median_diff, p_signflip, n, method."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return dict(median_diff=float("nan"), p_signflip=float("nan"), n=0, method="none")
    obs = float(np.median(d))
    tol = 1e-12
    if n <= 14:
        idx = np.arange(2 ** n)[:, None]
        bits = np.arange(n)[None, :]
        signs = 1.0 - 2.0 * ((idx >> bits) & 1)               # (2^n, n) of +/-1
        stats = np.median(d[None, :] * signs, axis=1)
        count = int(np.sum(np.abs(stats) + tol >= abs(obs)))
        p = count / (2 ** n)
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
        stats = np.median(d[None, :] * signs, axis=1)
        count = int(np.sum(np.abs(stats) + tol >= abs(obs)))
        p = (1 + count) / (1 + n_perm)
        method = "random"
    return dict(median_diff=obs, p_signflip=float(p), n=n, method=method)


def _wilcoxon_p(diffs) -> float:
    """Two-sided Wilcoxon signed-rank p on the paired diffs (secondary check). NaN if it cannot be
    computed (n==0, or all-zero diffs)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or np.allclose(d, 0.0):
        return float("nan")
    try:
        return float(wilcoxon(d)[1])
    except ValueError:
        return float("nan")


def compute_contrasts(subj_wide: pd.DataFrame, n_perm: int = DEFAULT_N_PERM,
                      seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """The 3 paired contrasts. Per contrast: subject diffs = subject_bin[hi]-subject_bin[lo] over
    subjects with BOTH bins finite (paired exclusion); sign-flip p (main) + wilcoxon p (secondary)."""
    rows = []
    for name, hi, lo in CONTRASTS:
        if hi in subj_wide and lo in subj_wide:
            diffs = (subj_wide[hi] - subj_wide[lo]).to_numpy(dtype=float)
            diffs = diffs[np.isfinite(diffs)]
        else:
            diffs = np.array([])
        sf = signflip_test(diffs, n_perm=n_perm, seed=seed)
        rows.append(dict(
            contrast=name, higher_bin=hi, lower_bin=lo,
            cohort_median_diff=sf["median_diff"], p_signflip=sf["p_signflip"],
            p_wilcoxon=_wilcoxon_p(diffs), n_subjects=sf["n"], signflip_method=sf["method"],
            n_perm=(n_perm if sf["method"] == "random" else 2 ** sf["n"] if sf["n"] else 0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _load_window_long(anchor: str, pool: str) -> pd.DataFrame:
    """Read one anchor x pool peri-ictal window-long CSV. subject MUST be read as str (the chunked
    low-memory reader otherwise splits a numeric subject id across dtypes -> phantom extra subject)."""
    path = ANCHOR_DIRS[anchor] / pool / "phase1_alignment_raw_window_long.csv"
    return pd.read_csv(path, dtype={"subject": str}, low_memory=False)


def build_tables(n_perm: int = DEFAULT_N_PERM, seed: int = DEFAULT_SEED):
    """Compute the per-(anchor,pool) trajectory + contrast tables. Returns (traj_df, contrast_df)."""
    traj_rows, con_rows = [], []
    for anchor in ANCHOR_DIRS:
        for pool in POOLS:
            df = _load_window_long(anchor, pool)
            subj = compute_subject_bins(df, PRIMARY_BANDS)
            wide = subject_bin_wide(subj)
            # per subject x bin trajectory
            merged = subj.copy()
            merged.insert(0, "anchor", anchor)
            merged.insert(1, "pool", pool)
            merged["bin_center"] = merged["epoch_region"].map(BIN_CENTERS)
            traj_rows.append(merged[["anchor", "pool", "subject", "epoch_region",
                                     "bin_center", "subject_bin", "n_seizures", "median_n_bands"]])
            # contrasts
            con = compute_contrasts(wide, n_perm=n_perm, seed=seed)
            con.insert(0, "anchor", anchor)
            con.insert(1, "pool", pool)
            con_rows.append(con)
    traj = pd.concat(traj_rows, ignore_index=True)
    # deterministic bin order in the trajectory csv
    traj["_o"] = traj["epoch_region"].map({b: i for i, b in enumerate(BIN_ORDER)})
    traj = traj.sort_values(["anchor", "pool", "subject", "_o"]).drop(columns="_o").reset_index(drop=True)
    return traj, pd.concat(con_rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Topic5 V2 W3 Task 3.2 peri-ictal trajectory + sign-flip.")
    ap.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM,
                    help="sign-flip permutations for the random path (n_subjects>14). Exact 2^n for n<=14.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    if args.n_perm < 10000:
        ap.error("--n-perm must be >= 10000 (brief LOCK)")

    traj, con = build_tables(n_perm=args.n_perm, seed=args.seed)
    traj_out = V2 / "phase1_v2_alignment_trajectory.csv"
    con_out = V2 / "phase1_v2_trajectory_contrasts.csv"
    traj.to_csv(traj_out, index=False)
    con.to_csv(con_out, index=False)
    print("wrote", traj_out)
    print("wrote", con_out)

    # console summary (cohort trajectory + contrasts, both anchors, both pools)
    for anchor in ANCHOR_DIRS:
        for pool in POOLS:
            wide = subject_bin_wide(
                traj[(traj.anchor == anchor) & (traj.pool == pool)]
                [["subject", "epoch_region", "subject_bin"]])
            coh = cohort_trajectory(wide)
            traj_str = "  ".join(f"{r.epoch_region}={r.cohort_median:.3f}" for r in coh.itertuples())
            print(f"\n[{anchor} {pool}] cohort trajectory: {traj_str}")
            sub = con[(con.anchor == anchor) & (con.pool == pool)]
            for r in sub.itertuples():
                print(f"    {r.contrast:24s} n={int(r.n_subjects):2d} med={r.cohort_median_diff:+.4f} "
                      f"signflip_p={r.p_signflip:.4f} wilcoxon_p={r.p_wilcoxon:.4f} ({r.signflip_method})")


if __name__ == "__main__":
    main()
