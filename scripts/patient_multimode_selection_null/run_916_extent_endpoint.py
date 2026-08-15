#!/usr/bin/env python
"""`epilepsiae_916` recruitment-extent endpoint, defined on train blocks only.

Question
--------
The 2026-08-15 audit found that `epilepsiae_916` is the one K>2 subject with a
non-degenerate recruitment contrast: one of its modes recruits five of six
contacts and pulls in the two-contact AM shaft, while the others recruit three.
That observation used the frozen K=4 labels on all blocks.  This script asks the
harder question: **is there a low-extent / high-extent stratification that is
defined without ever seeing the held-out blocks, and that still separates them?**

Design (every choice fixed before any held-out number was computed)
------------------------------------------------------------------
* The frozen K=4 labels are NOT used to merge or seed anything here.  The only
  frozen inputs reused are the event eligibility rule, the contact order and the
  recording-block ids.
* Blocks are split with the pipeline's own frozen rule and seed,
  `split_by_block(block_ids, frac=0.3, seed=20260815)` from
  `src/topic4_core_field_profile.py`, so this split is the same construction the
  34-subject cohort uses (916 itself is not in that cohort — it is K=4, and the
  cohort denominator is the 34 masked stable-K=2 subjects).
* Strata are fit on TRAIN events only, frozen, then transformed onto held-out.
    - PRIMARY: one-dimensional 2-means on the event's recruited-contact
      fraction.  One number, one threshold, nothing else.
    - SENSITIVITY: 2-means on [recruited fraction, AM participation,
      AH participation, both-shaft indicator].
* Names are `low-extent` / `high-extent`, assigned by which centroid has the
  larger recruited fraction.  No A/B, no TA/TB, no pathological reading.

Circularity control (this is the whole difficulty of the endpoint)
------------------------------------------------------------------
The PRIMARY split is a threshold on recruited fraction, so a held-out difference
in recruited fraction is **definitional, not evidence**, and is flagged as such.

The first version of this script used the raw AM-shaft participation difference
as the out-of-definition readout.  That was wrong.  `epilepsiae_916` has 4 AH and
2 AM contacts and the frozen threshold falls at 4.38 contacts, so the high-extent
stratum is exactly n_participating in {5, 6}; with only 4 AH contacts available,
every such event must include an AM contact.  P(AM) = 1.000 there by pigeonhole,
and the apparent +0.671 cross-shaft difference is forced by the electrode
montage.  That run is retained at `extent_endpoint_916_v1_defective_am_gate.json`.

The gate now uses the **size-matched AM excess**: observed AM contacts minus
p * n_AM / n_contacts, the count a uniformly random p-subset of the contacts
would supply.  This removes the mechanical extent coupling, so a non-zero
difference is a real cross-shaft preference rather than a counting identity.
The correction makes the gate strictly harder to pass; it is not a threshold
retune to rescue a failing endpoint.

Pre-registered status rule
--------------------------
`EXTENT_ENDPOINT_REPRODUCIBLE` requires ALL of, on held-out blocks:
  (a) both strata hold at least 10% of held-out events;
  (b) the held-out difference in **size-matched AM excess** between high- and
      low-extent has a block-bootstrap 95% interval excluding zero AND the same
      sign as the train difference;
  (c) the held-out high-extent proportion is within 0.15 of the train one, i.e.
      the frozen rule transfers rather than re-partitioning a different regime.
Otherwise `EXTENT_ENDPOINT_NOT_REPRODUCIBLE`, the 916 SNN arm stops, and no
threshold is retuned to rescue it.

Reported separately, never used to gate the status:
  `same_direction_only_extent_differs` = the held-out common-contact rank
  prototypes of the two strata correlate positively with a bootstrap interval
  excluding zero, i.e. the two strata order their shared contacts the same way
  and differ in how far the recruitment reaches.

SAFETY: reads only.  `src/topic4_core_field_profile.py` and
`src/interictal_propagation.py` are inside the runtime-module set that the
concurrently running formal cohort worker hashes against commit 96618174;
this script imports them and never writes to `src/` or `scripts/`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.topic4_core_field_profile import split_by_block, assert_block_disjoint  # noqa: E402
import run_multimode_grammar_audit as M  # noqa: E402

SUBJECT = "epilepsiae_916"
HELDOUT_FRACTION = 0.3          # pipeline frozen value
SPLIT_SEED = 20260815           # pipeline frozen value
KMEANS_SEED = 20260816
KMEANS_N_INIT = 20
N_BOOT = 2000
BOOT_SEED = 20260816
MIN_STRATUM_FRACTION = 0.10     # status rule (a)
MAX_PROPORTION_DRIFT = 0.15     # status rule (c)


def _boot_ci(values, lo=2.5, hi=97.5):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < 50:
        return None
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def _stratum_descriptors(masked, bools, sel, shaft_of, n_ch):
    """Participation profile, recruitment, prototype and pairwise precedence."""
    if sel.sum() == 0:
        return None
    b = bools[:, sel]
    m = masked[:, sel]
    profile = b.mean(axis=1)                                    # participation rate
    with np.errstate(invalid="ignore"):
        proto = np.array([np.nanmean(m[c][b[c]]) if b[c].any() else np.nan
                          for c in range(n_ch)])
    npart = b.sum(axis=0)
    am = np.array([s == "AM" for s in shaft_of])
    ah = np.array([s == "AH" for s in shaft_of])
    prec = {}
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            both = b[i] & b[j]
            prec[f"{i}<{j}"] = float(np.mean(m[i][both] < m[j][both])) if both.any() else np.nan
    n_am = int(am.sum())
    am_count = b[am].sum(axis=0).astype(float)
    size_matched_expected_am = npart.astype(float) * (n_am / float(n_ch))
    am_excess = float(np.mean(am_count - size_matched_expected_am))
    # which AH contacts, given how many: out-of-definition shape of the AH set
    ah_count = b[ah].sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ah_shape = np.where(ah_count[None, :] > 0, b[ah] / np.maximum(ah_count[None, :], 1), np.nan)
    ah_profile_shape = np.nanmean(ah_shape, axis=1)
    return {
        "n_events": int(sel.sum()),
        "participation_profile": profile.tolist(),
        "am_excess_vs_size_matched": am_excess,
        "am_observed_over_size_matched_chance": float(
            np.mean(b[am].any(axis=0)) /
            max(float(np.mean(1.0 - np.array([
                (math.comb(n_ch - n_am, int(pp)) / math.comb(n_ch, int(pp)))
                if pp <= n_ch - n_am else 0.0 for pp in npart]))), 1e-12)),
        "size_matched_chance_am_participation": float(np.mean(1.0 - np.array([
            (math.comb(n_ch - n_am, int(pp)) / math.comb(n_ch, int(pp)))
            if pp <= n_ch - n_am else 0.0 for pp in npart]))),
        "ah_profile_shape": ah_profile_shape.tolist(),
        "_ah_shape": ah_profile_shape,
        "prototype_masked_rank": [None if not np.isfinite(v) else float(v) for v in proto],
        "recruited_fraction_mean": float(np.mean(npart) / n_ch),
        "recruited_fraction_median": float(np.median(npart) / n_ch),
        "am_shaft_participation": float(np.mean(b[am].any(axis=0))),
        "ah_shaft_participation": float(np.mean(b[ah].any(axis=0))),
        "both_shaft_fraction": float(np.mean(b[am].any(axis=0) & b[ah].any(axis=0))),
        "am_contact_fraction": float(np.mean(b[am].sum(axis=0)) / int(am.sum())),
        "pairwise_precedence": prec,
        "_proto": proto,
        "_prec": np.array([prec[k] for k in sorted(prec)], float),
    }


def _contrast(hi, lo, n_ch):
    """high-extent minus low-extent, plus the shared-direction readouts."""
    if hi is None or lo is None:
        return None
    a, b = hi["_proto"], lo["_proto"]
    ok = np.isfinite(a) & np.isfinite(b)
    rho = (float(stats.spearmanr(a[ok], b[ok]).statistic)
           if ok.sum() >= 3 and not (np.all(a[ok] == a[ok][0]) or np.all(b[ok] == b[ok][0]))
           else float("nan"))
    pa, pb = hi["_prec"], lo["_prec"]
    pok = np.isfinite(pa) & np.isfinite(pb)
    prec_r = (float(stats.pearsonr(pa[pok], pb[pok]).statistic)
              if pok.sum() >= 3 else float("nan"))
    # Pearson over 15 probabilities that all sit inside a narrow band is noise
    # dominated; the direction-level statistic is whether the two strata put the
    # same contact first in each pair.
    prec_sign = (float(np.mean((pa[pok] > 0.5) == (pb[pok] > 0.5)))
                 if pok.sum() >= 3 else float("nan"))
    prec_mad = float(np.mean(np.abs(pa[pok] - pb[pok]))) if pok.sum() >= 3 else float("nan")
    sa, sb = hi["_ah_shape"], lo["_ah_shape"]
    sok = np.isfinite(sa) & np.isfinite(sb)
    return {
        "d_am_excess_vs_size_matched": hi["am_excess_vs_size_matched"] - lo["am_excess_vs_size_matched"],
        "d_size_matched_chance_am": (hi["size_matched_chance_am_participation"]
                                     - lo["size_matched_chance_am_participation"]),
        "max_abs_d_ah_profile_shape": (float(np.max(np.abs(sa[sok] - sb[sok])))
                                       if sok.any() else float("nan")),
        "d_recruited_fraction_mean": hi["recruited_fraction_mean"] - lo["recruited_fraction_mean"],
        "d_am_shaft_participation": hi["am_shaft_participation"] - lo["am_shaft_participation"],
        "d_ah_shaft_participation": hi["ah_shaft_participation"] - lo["ah_shaft_participation"],
        "d_both_shaft_fraction": hi["both_shaft_fraction"] - lo["both_shaft_fraction"],
        "d_am_contact_fraction": hi["am_contact_fraction"] - lo["am_contact_fraction"],
        "max_abs_d_participation_profile": float(np.max(np.abs(
            np.asarray(hi["participation_profile"]) - np.asarray(lo["participation_profile"])))),
        "common_contact_prototype_spearman": rho,
        "n_common_contacts": int(ok.sum()),
        "pairwise_precedence_pearson": prec_r,
        "pairwise_precedence_sign_agreement": prec_sign,
        "pairwise_precedence_mean_abs_diff": prec_mad,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "extent_endpoint_916"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = M.replay_and_audit(SUBJECT)
    ve = rec["_valid_events"]
    n_ch = rec["n_channels"]
    names = rec["channel_names"]
    shaft_of = [M.CHANNEL_RE.fullmatch(c).group(1) for c in names]
    ranks_v = rec["_loaded"]["ranks"][:, ve]
    bools_v = rec["_loaded"]["bools"][:, ve]
    masked = mask_phantom_ranks(ranks_v, bools_v, normalize=True)
    block_ids = rec["_block_ids_valid"]

    tr_idx, ho_idx = split_by_block(block_ids, HELDOUT_FRACTION, SPLIT_SEED)
    assert_block_disjoint(block_ids, tr_idx, ho_idx)

    am = np.array([s == "AM" for s in shaft_of])
    ah = np.array([s == "AH" for s in shaft_of])
    npart = bools_v.sum(axis=0).astype(float)
    feat_rec = (npart / n_ch)[:, None]
    feat_full = np.column_stack([
        npart / n_ch,
        bools_v[am].any(axis=0).astype(float),
        bools_v[ah].any(axis=0).astype(float),
        (bools_v[am].any(axis=0) & bools_v[ah].any(axis=0)).astype(float),
    ])

    results = {}
    for name, feat in (("primary_1d_recruited_fraction", feat_rec),
                       ("sensitivity_4d_extent", feat_full)):
        km = KMeans(n_clusters=2, n_init=KMEANS_N_INIT, random_state=KMEANS_SEED)
        km.fit(feat[tr_idx])                                  # TRAIN ONLY
        centers = km.cluster_centers_
        high_id = int(np.argmax(centers[:, 0]))               # larger recruited fraction
        assign_all = km.predict(feat)                         # frozen transform
        is_high = assign_all == high_id

        split_out = {}
        for split_name, idx in (("train", tr_idx), ("heldout", ho_idx)):
            sel_hi = np.zeros(bools_v.shape[1], bool); sel_hi[idx[is_high[idx]]] = True
            sel_lo = np.zeros(bools_v.shape[1], bool); sel_lo[idx[~is_high[idx]]] = True
            hi = _stratum_descriptors(masked, bools_v, sel_hi, shaft_of, n_ch)
            lo = _stratum_descriptors(masked, bools_v, sel_lo, shaft_of, n_ch)
            con = _contrast(hi, lo, n_ch)

            # ---- recording-block bootstrap inside this split ----
            rng = np.random.default_rng(BOOT_SEED)
            blocks = np.unique(block_ids[idx])
            per_block = {b: idx[block_ids[idx] == b] for b in blocks}
            boot = {k: [] for k in ("d_recruited_fraction_mean", "d_am_shaft_participation",
                                    "d_both_shaft_fraction", "d_am_contact_fraction",
                                    "d_am_excess_vs_size_matched", "max_abs_d_ah_profile_shape",
                                    "common_contact_prototype_spearman",
                                    "pairwise_precedence_sign_agreement",
                                    "high_extent_fraction")}
            for _ in range(N_BOOT):
                pick = rng.choice(blocks, size=blocks.size, replace=True)
                m_idx = np.concatenate([per_block[b] for b in pick])
                hb = is_high[m_idx]
                boot["high_extent_fraction"].append(float(hb.mean()))
                s_hi = np.zeros(bools_v.shape[1], bool); s_lo = np.zeros(bools_v.shape[1], bool)
                s_hi[m_idx[hb]] = True; s_lo[m_idx[~hb]] = True
                if s_hi.sum() < 20 or s_lo.sum() < 20:
                    continue
                h2 = _stratum_descriptors(masked, bools_v, s_hi, shaft_of, n_ch)
                l2 = _stratum_descriptors(masked, bools_v, s_lo, shaft_of, n_ch)
                c2 = _contrast(h2, l2, n_ch)
                for kk in boot:
                    if kk != "high_extent_fraction":
                        boot[kk].append(c2[kk])
            split_out[split_name] = {
                "n_events": int(idx.size),
                "n_blocks": int(blocks.size),
                "high_extent_fraction": float(is_high[idx].mean()),
                "high_extent": {k: v for k, v in hi.items() if not k.startswith("_")} if hi else None,
                "low_extent": {k: v for k, v in lo.items() if not k.startswith("_")} if lo else None,
                "contrast_high_minus_low": con,
                "block_bootstrap_ci": {k: _boot_ci(v) for k, v in boot.items()},
            }

        tr, ho = split_out["train"], split_out["heldout"]
        ci_am = ho["block_bootstrap_ci"]["d_am_excess_vs_size_matched"]
        d_am_tr = tr["contrast_high_minus_low"]["d_am_excess_vs_size_matched"]
        d_am_ho = ho["contrast_high_minus_low"]["d_am_excess_vs_size_matched"]
        cond_a = (min(ho["high_extent_fraction"], 1 - ho["high_extent_fraction"])
                  >= MIN_STRATUM_FRACTION)
        cond_b = bool(ci_am is not None and (ci_am[0] > 0 or ci_am[1] < 0)
                      and np.sign(d_am_ho) == np.sign(d_am_tr) and d_am_tr != 0)
        cond_c = abs(ho["high_extent_fraction"] - tr["high_extent_fraction"]) <= MAX_PROPORTION_DRIFT
        status = ("EXTENT_ENDPOINT_REPRODUCIBLE" if (cond_a and cond_b and cond_c)
                  else "EXTENT_ENDPOINT_NOT_REPRODUCIBLE")
        ci_rho = ho["block_bootstrap_ci"]["common_contact_prototype_spearman"]
        same_dir = bool(ci_rho is not None and ci_rho[0] > 0)

        results[name] = {
            "kmeans_train_centers": centers.tolist(),
            "high_centroid_index": high_id,
            "feature_names": (["recruited_fraction"] if feat.shape[1] == 1 else
                              ["recruited_fraction", "am_shaft", "ah_shaft", "both_shaft"]),
            "threshold_recruited_fraction_midpoint": (
                float(np.mean(np.sort(centers[:, 0]))) if feat.shape[1] == 1 else None),
            "splits": split_out,
            "status_conditions": {
                "a_both_strata_ge_10pct_heldout": bool(cond_a),
                "b_heldout_am_excess_vs_size_matched_ci_excludes_zero_and_sign_matches_train": bool(cond_b),
                "c_proportion_drift_le_0.15": bool(cond_c),
            },
            "status": status,
            "train_to_heldout_sign_consistency": {
                k: bool(np.sign(tr["contrast_high_minus_low"][k]) ==
                        np.sign(ho["contrast_high_minus_low"][k]))
                for k in ("d_recruited_fraction_mean", "d_am_shaft_participation",
                          "d_both_shaft_fraction", "d_am_contact_fraction",
                          "d_am_excess_vs_size_matched")
            },
            "same_direction_only_extent_differs": same_dir,
            "circularity_note": (
                "d_recruited_fraction_mean is DEFINITIONAL for this split and is not "
                "evidence; the load-bearing held-out readout is d_am_shaft_participation, "
                "which never enters the primary split."
                if feat.shape[1] == 1 else
                "both d_recruited_fraction_mean and the shaft terms are definitional for "
                "this sensitivity split; it is reported only to show the primary split is "
                "not an artefact of using one feature."),
        }

    payload = {
        "provenance": {
            "git_commit": M._git_commit(),
            "split_rule": "src.topic4_core_field_profile.split_by_block",
            "heldout_block_fraction": HELDOUT_FRACTION,
            "split_seed": SPLIT_SEED,
            "kmeans_seed": KMEANS_SEED, "kmeans_n_init": KMEANS_N_INIT,
            "n_block_bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED,
            "status_rule_fixed_before_first_run": True,
            "status_rule_corrected_after_validity_defect": (
                "Condition (b) originally used the raw AM-shaft participation difference. "
                "That term is NOT out-of-definition for this montage: with 4 AH and 2 AM "
                "contacts and a frozen threshold at 4.38 contacts, the high-extent stratum "
                "is n_participating in {5,6}, so P(AM) = 1 by pigeonhole. The gate now uses "
                "the size-matched AM excess (observed AM contacts minus p * n_AM / n_contacts), "
                "which removes the mechanical extent coupling. This makes the gate STRICTER; "
                "it is not a threshold retune to rescue a failing endpoint. The superseded "
                "run is kept at extent_endpoint_916_v1_defective_am_gate.json."),
            "frozen_k4_labels_used": False,
            "safety": "reads only; never writes src/ or scripts/",
        },
        "subject_id": SUBJECT,
        "n_channels": n_ch,
        "channel_names": names,
        "shaft_of_channel": shaft_of,
        "n_valid_events": int(ve.size),
        "n_train_events": int(tr_idx.size), "n_heldout_events": int(ho_idx.size),
        "n_train_blocks": int(np.unique(block_ids[tr_idx]).size),
        "n_heldout_blocks": int(np.unique(block_ids[ho_idx]).size),
        "input_json_sha256": rec["input_json_sha256"],
        "variants": results,
        "primary_status": results["primary_1d_recruited_fraction"]["status"],
    }
    out = out_dir / "extent_endpoint_916.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    for name, r in results.items():
        tr, ho = r["splits"]["train"], r["splits"]["heldout"]
        print(f"[{name}] status={r['status']}")
        print(f"   train  n={tr['n_events']:6d} blocks={tr['n_blocks']:3d} high={tr['high_extent_fraction']:.3f} "
              f"dRec={tr['contrast_high_minus_low']['d_recruited_fraction_mean']:+.3f} "
              f"dAMraw={tr['contrast_high_minus_low']['d_am_shaft_participation']:+.3f} "
              f"dAMexcess={tr['contrast_high_minus_low']['d_am_excess_vs_size_matched']:+.3f} "
              f"rho={tr['contrast_high_minus_low']['common_contact_prototype_spearman']:+.3f}")
        print(f"   heldout n={ho['n_events']:6d} blocks={ho['n_blocks']:3d} high={ho['high_extent_fraction']:.3f} "
              f"dRec={ho['contrast_high_minus_low']['d_recruited_fraction_mean']:+.3f} "
              f"dAMraw={ho['contrast_high_minus_low']['d_am_shaft_participation']:+.3f} "
              f"dAMexcess={ho['contrast_high_minus_low']['d_am_excess_vs_size_matched']:+.3f} "
              f"CI={ho['block_bootstrap_ci']['d_am_excess_vs_size_matched']} "
              f"rho={ho['contrast_high_minus_low']['common_contact_prototype_spearman']:+.3f}")
        print(f"   conditions={r['status_conditions']} same_direction={r['same_direction_only_extent_differs']}")
    print(f"\nPRIMARY STATUS: {payload['primary_status']} -> {out}")


if __name__ == "__main__":
    main()
