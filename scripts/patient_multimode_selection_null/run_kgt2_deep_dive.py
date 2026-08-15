#!/usr/bin/env python
"""K>2 deep dive: superfamily stability, frozen alternative-K scan, mode geometry.

The main audit (``run_multimode_grammar_audit.py``) reports a single best 2-way
split of each subject's mode prototypes into "direction superfamilies".  A
single best split is not evidence: with k modes there are 2^(k-1)-1 candidate
splits and the maximiser is guaranteed to look clean on any prototype set.  This
script therefore asks the only question that matters for the discriminant --
**is that split stable when the recording blocks are resampled?** -- and reports
the frozen adaptive-K scan alongside it so the reader can see how the chosen K
compares with the alternatives the producer already evaluated.

CONTRACT CLAUSES honoured here (same numbering as the main audit):
  C1 block bootstrap resamples whole recording blocks; no cross-block mixing.
  C2 ``chosen_k`` is read from the artifact.  The frozen ``adaptive_cluster.scan``
     is REPORTED for alternative k; no k is re-selected and no KMeans is re-fit.
  C3 all rank quantities go through ``mask_phantom_ranks``.
  C4 per-cluster valid participation mask from raw ``eventsBool``.
  C9 commit / hashes / seed recorded in the output.

Output: kgt2_deep_dive.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
import run_multimode_grammar_audit as M  # noqa: E402

N_BOOT = 2000
SEED = 20260815


def canonical_split(assign: np.ndarray) -> tuple:
    """Label-invariant key for a 2-way partition of the modes."""
    a = np.asarray(assign, dtype=int)
    if a[0] == 1:
        a = 1 - a
    return tuple(a.tolist())


def main() -> None:
    per_subject_in = REPO / "results/interictal_propagation_masked/per_subject"
    targets = []
    for p in sorted(per_subject_in.glob("*.json")):
        doc = json.load(open(p))
        if int(doc["adaptive_cluster"]["chosen_k"]) > 2:
            targets.append(p.stem)

    out = {
        "provenance": {
            "git_commit": M._git_commit(),
            "seed": SEED,
            "n_boot_blocks": N_BOOT,
            "note": "chosen_k is frozen; the k-scan below is REPORTED from the "
                    "artifact, never re-selected.",
        },
        "subjects": {},
    }

    for si, sid in enumerate(targets):
        rng = np.random.default_rng(SEED + 1000 + si)
        rec = M.replay_and_audit(sid)
        loaded, labels, ve = rec["_loaded"], rec["_labels"], rec["_valid_events"]
        k, n_ch = rec["chosen_k"], rec["n_channels"]
        ranks_v, bools_v = loaded["ranks"][:, ve], loaded["bools"][:, ve]
        masked = mask_phantom_ranks(ranks_v, bools_v, normalize=True)   # C3
        block_ids = rec["_block_ids_valid"]
        uniq_blocks = np.unique(block_ids)
        n_blk = uniq_blocks.size
        blk_dense = np.searchsorted(uniq_blocks, block_ids)

        # per-block sufficient statistics (C1 + C4)
        sum_rank = np.zeros((n_blk, k, n_ch))
        cnt_part = np.zeros((n_blk, k, n_ch))
        cnt_ev = np.zeros((n_blk, k))
        masked0 = np.where(bools_v, np.nan_to_num(masked, nan=0.0), 0.0)
        for ci in range(k):
            sel = labels == ci
            bd = blk_dense[sel]
            for ch in range(n_ch):
                sum_rank[:, ci, ch] = np.bincount(bd, weights=masked0[ch, sel], minlength=n_blk)
                cnt_part[:, ci, ch] = np.bincount(
                    bd, weights=bools_v[ch, sel].astype(float), minlength=n_blk)
            cnt_ev[:, ci] = np.bincount(bd, minlength=n_blk)

        def protos(w):
            sr = np.tensordot(w, sum_rank, axes=(0, 0))
            cp = np.tensordot(w, cnt_part, axes=(0, 0))
            ne = w @ cnt_ev
            with np.errstate(invalid="ignore", divide="ignore"):
                pr = np.where(cp > 0, sr / np.maximum(cp, 1e-12), np.nan)
                fr = np.where(ne[:, None] > 0, cp / np.maximum(ne[:, None], 1e-12), 0.0)
            vd = (cp >= M.MIN_PROTO_COUNT) & (fr >= M.MIN_PROTO_FRAC)
            return np.where(vd, pr, np.nan), vd

        pr_obs, vd_obs = protos(np.ones(n_blk))
        sf_obs = M._direction_superfamily(pr_obs, vd_obs, k)
        obs_key = canonical_split(sf_obs["assignment"]) if sf_obs.get("applicable") else None

        # ---- random-ordering null for the superfamily reading -------------
        # The best 2-way split is a MAXIMISER over 2^(k-1)-1 candidate splits.
        # On any set of roughly uncorrelated prototypes the maximiser will
        # essentially always return within_rho > 0 and between_rho < 0, so
        # "two opposite families" is not evidence on its own.  The null keeps
        # each mode's valid-contact mask and randomises only the ORDER of its
        # contacts, then re-runs the identical maximiser.
        rho_obs = np.array(
            [[np.nan if v is None else v for v in row] for row in sf_obs["rho_matrix"]]
        ) if sf_obs.get("rho_matrix") else None
        iu = np.triu_indices(k, 1)
        obs_pair_rho = rho_obs[iu] if rho_obs is not None else np.array([])
        obs_sep = sf_obs.get("separation", np.nan)
        obs_min_rho = float(np.nanmin(obs_pair_rho)) if obs_pair_rho.size else np.nan
        obs_n_fr = int(np.nansum(obs_pair_rho < -0.5))    # producer's r < -0.5 rule

        null_sep, null_min, null_nfr, null_pair = [], [], [], []
        rng_null = np.random.default_rng(SEED + 5000 + si)
        for _ in range(N_BOOT):
            pr_n = np.full((k, n_ch), np.nan)
            for ci in range(k):
                idx = np.flatnonzero(vd_obs[ci])
                if idx.size:
                    pr_n[ci, idx] = rng_null.permutation(idx.size).astype(float)
            sf_n = M._direction_superfamily(pr_n, vd_obs, k)
            if not sf_n.get("applicable"):
                continue
            rn = np.array([[np.nan if v is None else v for v in row] for row in sf_n["rho_matrix"]])
            pr_vals = rn[iu]
            null_sep.append(sf_n["separation"])
            null_min.append(float(np.nanmin(pr_vals)))
            null_nfr.append(int(np.nansum(pr_vals < -0.5)))
            null_pair.extend([v for v in pr_vals if np.isfinite(v)])
        null_sep = np.asarray(null_sep, dtype=float)
        null_min = np.asarray(null_min, dtype=float)
        null_nfr = np.asarray(null_nfr, dtype=float)
        null_pair = np.asarray(null_pair, dtype=float)

        random_ordering_null = {
            "n_draws": int(null_sep.size),
            "observed_best_split_separation": (None if not np.isfinite(obs_sep) else float(obs_sep)),
            "null_separation_mean": float(null_sep.mean()) if null_sep.size else None,
            "null_separation_p95": float(np.percentile(null_sep, 95)) if null_sep.size else None,
            "p_separation": (float((np.sum(null_sep >= obs_sep) + 1) / (null_sep.size + 1))
                             if null_sep.size and np.isfinite(obs_sep) else None),
            "frac_null_draws_two_opposite_families": (
                float(np.mean(null_sep > 0)) if null_sep.size else None),
            "observed_min_pair_rho": (None if not np.isfinite(obs_min_rho) else obs_min_rho),
            "null_min_pair_rho_mean": float(null_min.mean()) if null_min.size else None,
            "p_min_pair_rho": (float((np.sum(null_min <= obs_min_rho) + 1) / (null_min.size + 1))
                               if null_min.size and np.isfinite(obs_min_rho) else None),
            "observed_n_pairs_rho_lt_-0.5": obs_n_fr,
            "null_n_pairs_rho_lt_-0.5_mean": float(null_nfr.mean()) if null_nfr.size else None,
            "p_n_pairs_rho_lt_-0.5": (float((np.sum(null_nfr >= obs_n_fr) + 1) / (null_nfr.size + 1))
                                      if null_nfr.size else None),
            "chance_rate_single_pair_rho_lt_-0.5": (
                float(np.mean(null_pair < -0.5)) if null_pair.size else None),
            "null_pair_rho_sd": float(null_pair.std(ddof=1)) if null_pair.size > 1 else None,
            "null_separation_percentiles": (
                {str(q): float(np.percentile(null_sep, q))
                 for q in (2.5, 25, 50, 75, 97.5)} if null_sep.size else None),
            "null_separation_samples": (
                null_sep[:: max(1, null_sep.size // 500)].tolist() if null_sep.size else []),
        }

        keys, within, between, opposite = [], [], [], []
        for _ in range(N_BOOT):
            w = rng.multinomial(n_blk, np.full(n_blk, 1.0 / n_blk)).astype(float)
            pr_i, vd_i = protos(w)
            sf = M._direction_superfamily(pr_i, vd_i, k)
            if not sf.get("applicable"):
                continue
            keys.append(canonical_split(sf["assignment"]))
            within.append(sf["within_mean_rho"])
            between.append(sf["between_mean_rho"])
            opposite.append(bool(sf["two_opposite_families"]))

        cnt = Counter(keys)
        modal, modal_n = (cnt.most_common(1)[0] if cnt else (None, 0))
        n_possible_splits = 2 ** (k - 1) - 1

        # frozen alternative-K scan, reported verbatim (C2)
        scan = rec["_doc"]["adaptive_cluster"]["scan"]
        scan_rep = [
            {
                "k": int(s["k"]),
                "viable": bool(s.get("viable")),
                "median_silhouette": s.get("median_silhouette"),
                "median_ami": s.get("median_ami"),
                "worst_min_cluster_fraction": s.get("worst_min_cluster_fraction"),
                "passes_both": bool(s.get("passes_both")),
            }
            for s in scan
        ]
        passing = [s for s in scan_rep if s["passes_both"]]
        sil = {s["k"]: s["median_silhouette"] for s in scan_rep}
        sil_k2 = sil.get(2)
        sil_chosen = sil.get(k)

        out["subjects"][sid] = {
            "chosen_k": k,
            "n_channels": n_ch,
            "n_shafts": int(np.unique(
                [M.CHANNEL_RE.fullmatch(c).group(1) for c in rec["channel_names"]]).size),
            "channel_names": rec["channel_names"],
            "n_blocks": int(n_blk),
            "observed_superfamily": {kk: vv for kk, vv in sf_obs.items() if kk != "rho_matrix"},
            "rho_matrix": sf_obs.get("rho_matrix"),
            "random_ordering_null": random_ordering_null,
            "superfamily_stability": {
                "n_boot_usable": len(keys),
                "n_possible_2way_splits": n_possible_splits,
                "chance_rate_if_uniform": (1.0 / n_possible_splits) if n_possible_splits else None,
                "observed_split": list(obs_key) if obs_key else None,
                "modal_split": list(modal) if modal else None,
                "modal_split_frequency": (modal_n / len(keys)) if keys else None,
                "observed_split_frequency": (cnt.get(obs_key, 0) / len(keys)) if keys else None,
                "frac_draws_two_opposite_families": (float(np.mean(opposite)) if opposite else None),
                "within_rho_ci": ([float(np.percentile(within, 2.5)),
                                   float(np.percentile(within, 97.5))] if len(within) >= 50 else None),
                "between_rho_ci": ([float(np.percentile(between, 2.5)),
                                    float(np.percentile(between, 97.5))] if len(between) >= 50 else None),
                "split_frequency_table": {str(list(kk)): vv / len(keys) for kk, vv in cnt.most_common()},
            },
            "frozen_k_scan": scan_rep,
            "frozen_k_scan_summary": {
                "k_values_passing_both_gates": [s["k"] for s in passing],
                "median_silhouette_at_k2": sil_k2,
                "median_silhouette_at_chosen_k": sil_chosen,
                "silhouette_gain_chosen_over_k2": (
                    None if (sil_k2 is None or sil_chosen is None) else float(sil_chosen - sil_k2)),
            },
            "input_json_sha256": rec["input_json_sha256"],
        }
        st = out["subjects"][sid]["superfamily_stability"]
        no = random_ordering_null
        print(
            f"{sid:24s} k={k} nch={n_ch} | split {obs_key} boot-freq="
            f"{st['observed_split_frequency']:.3f} (chance {1.0/n_possible_splits:.3f}) | "
            f"sep={no['observed_best_split_separation']:.3f} vs random-order null mean "
            f"{no['null_separation_mean']:.3f} p={no['p_separation']:.3f} | "
            f"null draws with 'two opposite families' = "
            f"{no['frac_null_draws_two_opposite_families']:.3f} | "
            f"min rho {no['observed_min_pair_rho']:.2f} p={no['p_min_pair_rho']:.3f} | "
            f"pairs rho<-0.5: obs {no['observed_n_pairs_rho_lt_-0.5']} vs null "
            f"{no['null_n_pairs_rho_lt_-0.5_mean']:.2f} p={no['p_n_pairs_rho_lt_-0.5']:.3f}",
            flush=True)

    with open(HERE / "kgt2_deep_dive.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {HERE/'kgt2_deep_dive.json'} for {len(out['subjects'])} subjects.")


if __name__ == "__main__":
    main()
