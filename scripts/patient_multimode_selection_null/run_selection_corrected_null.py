#!/usr/bin/env python
"""Selection-corrected null for mode-internal ordering concentration (K frozen).

Question
--------
The 2026-08-15 audit left one gap open: each mode of the six K>2 subjects does
concentrate on a dominant firing order (20-47% of the mode's events, 3-86x the
uniform-over-its-own-orderings reference), but **no null was run**, so that
number could not distinguish a genuine template from "fixed-K KMeans cutting a
small discrete lattice into tight groups".  This script builds that null.

Null construction (the contract)
--------------------------------
PRIMARY (`order`): for every valid event, keep
  * the recruitment mask (exactly which contacts participated) — untouched
  * the event count and the recording-block membership — untouched
  * the contact order and the frozen K — untouched
and randomise ONLY the onset/rank order among the participating contacts, drawn
uniformly over the p! orderings.  Then re-run the *identical* clustering the
producer ran (`_kmeans_stability_for_k` semantics: 10 seeds, `n_init=10`,
silhouette on a 2000-row subsample from `default_rng(0)`, best seed = argmax
silhouette) at the frozen K, and recompute the statistics on those null labels.

SENSITIVITY (`marginal`): destroy the within-event cross-contact coordination
while keeping each contact's marginal rank distribution as close as possible.
Within each participation-SET stratum, each contact's normalised rank values are
permuted independently across the events of that stratum, then each event is
re-ranked so it is a valid ordering again (which is what keeps the features on
the same lattice the primary null lives on).  The re-rank repair perturbs the
marginals slightly, so the realised marginal drift is measured and reported
(`marginal_drift_total_variation`), not assumed.

What is frozen and never re-derived here
----------------------------------------
K (`adaptive_cluster.chosen_k`), event eligibility
(`_valid_event_indices(min_participating=3)`), recording blocks, the masked
feature construction (`build_masked_kmeans_features(impute="event_median")`),
and the contact order.  Nothing in this script re-picks K.

SAFETY: this script only READS `src/interictal_propagation.py` and
`src/lagpat_rank_audit.py`.  Both sit in (or next to) the frozen runtime-module
set that the concurrently running formal cohort worker hashes against commit
96618174; modifying either would abort every newly launched formal worker.
Nothing here writes to `src/` or `scripts/`.

Statistics (computed identically on observed and on every null draw)
--------------------------------------------------------------------
  * per-mode top-1 ordering share
  * equal-mode-weighted dominant-order concentration  (unweighted mean over the
    K modes, so a large mode cannot carry the statistic)
  * per-mode ordering entropy (nats) and effective ordering count exp(H)
  * equal-mode-weighted mean effective ordering count

Reported as observed, null mean/sd/q95, observed-minus-null, and a Monte-Carlo
interval on the null mean.  Exceeding q95 is an EXPLORATORY per-subject state;
it is not a blocker and not a multiple-comparison family.

Usage:
  python run_selection_corrected_null.py --subject epilepsiae_916 --n-perm 256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402
import run_multimode_grammar_audit as M  # noqa: E402

MAX_SILHOUETTE_EVENTS = 2000       # producer `_MAX_SILHOUETTE_EVENTS`
N_KMEANS_SEEDS = 10                # producer `n_stability_seeds`
KMEANS_N_INIT = 10                 # producer `KMeans(n_init=10)`
IMPUTE_FILL = 0.5                  # `build_masked_kmeans_features(impute="event_median")`
SEED = 20260816


# ---------------------------------------------------------------------------
# producer-identical clustering
# ---------------------------------------------------------------------------
def producer_kmeans(features: np.ndarray, k: int):
    """Mirror `src.interictal_propagation._kmeans_stability_for_k` at one k.

    Replicated clause by clause so the null sees the same estimator the frozen
    labels came from: ten seeds `range(10)`, `KMeans(n_init=10, random_state=seed)`,
    silhouette on at most 2000 rows drawn from a `default_rng(0)` that is created
    once and advances across seeds, and best seed = argmax silhouette.
    """
    n = features.shape[0]
    sil_rng = np.random.default_rng(0)
    all_labels, silhouettes = [], []
    for seed in range(N_KMEANS_SEEDS):
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed)
        lab = km.fit_predict(features)
        all_labels.append(lab)
        if np.unique(lab).size >= 2:
            if n > MAX_SILHOUETTE_EVENTS:
                idx = sil_rng.choice(n, MAX_SILHOUETTE_EVENTS, replace=False)
                silhouettes.append(float(silhouette_score(features[idx], lab[idx])))
            else:
                silhouettes.append(float(silhouette_score(features, lab)))
        else:
            silhouettes.append(np.nan)
    sil = np.asarray(silhouettes, float)
    best = int(np.nanargmax(sil)) if np.any(np.isfinite(sil)) else 0
    return all_labels[best], best, sil


# ---------------------------------------------------------------------------
# vectorised randomisation
# ---------------------------------------------------------------------------
class OrderRandomiser:
    """Pre-groups the valid events so a draw costs O(n_events) numpy work.

    `mask_phantom_ranks` walks events in Python; calling it 256 times on 93k
    events would dominate the runtime.  Because the primary null replaces the
    ordering wholesale, the masked normalised rank of a null event is simply a
    uniform random permutation of {0, 1/(p-1), ..., 1} scattered onto that
    event's participating contacts, which is exact and fully vectorised.
    """

    def __init__(self, bools_valid: np.ndarray, masked_valid: np.ndarray):
        self.n_ch, self.n_ev = bools_valid.shape
        self.bools = bools_valid
        npart = bools_valid.sum(axis=0).astype(int)
        self.npart = npart
        self.p_groups = []
        for p in np.unique(npart):
            if p < 1:
                continue
            ev = np.flatnonzero(npart == p)
            contacts = np.array([np.flatnonzero(bools_valid[:, e]) for e in ev], dtype=int)
            self.p_groups.append((int(p), ev, contacts))
        # participation-SET strata for the marginal-preserving null
        codes = np.packbits(bools_valid.T.astype(np.uint8), axis=1)
        _, inv = np.unique(codes, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        self.set_strata = []
        for s in np.unique(inv):
            ev = np.flatnonzero(inv == s)
            p = int(npart[ev[0]])
            if p < 2:
                continue
            contacts = np.flatnonzero(bools_valid[:, ev[0]])
            vals = masked_valid[np.ix_(contacts, ev)].T.copy()   # (n_s, p)
            self.set_strata.append((ev, contacts, vals))

    def draw_order(self, rng) -> np.ndarray:
        """(n_ev, n_ch) feature matrix; uniform random ordering per event."""
        X = np.full((self.n_ch, self.n_ev), IMPUTE_FILL, dtype=float)
        for p, ev, contacts in self.p_groups:
            if p == 1:
                X[contacts[:, 0], ev] = IMPUTE_FILL
                continue
            base = np.tile(np.arange(p, dtype=float), (ev.size, 1))
            perm = rng.permuted(base, axis=1) / float(p - 1)
            X[contacts.ravel(), np.repeat(ev, p)] = perm.ravel()
        return X.T

    def draw_marginal(self, rng) -> tuple[np.ndarray, float]:
        """(n_ev, n_ch) features; per-contact marginals shuffled across events.

        Returns the features and the realised marginal drift (mean total
        variation between the observed and the drawn per-contact rank histogram),
        so the "as close as possible" clause is measured rather than asserted.
        """
        X = np.full((self.n_ch, self.n_ev), IMPUTE_FILL, dtype=float)
        obs_vals = {c: [] for c in range(self.n_ch)}
        new_vals = {c: [] for c in range(self.n_ch)}
        for ev, contacts, vals in self.set_strata:
            p = vals.shape[1]
            shuffled = np.empty_like(vals)
            for j in range(p):
                shuffled[:, j] = rng.permutation(vals[:, j])
            # repair: re-rank each event so it is a valid ordering again
            repaired = np.argsort(np.argsort(shuffled, axis=1, kind="stable"),
                                  axis=1, kind="stable").astype(float) / float(p - 1)
            X[np.ix_(contacts, ev)] = repaired.T
            for j, c in enumerate(contacts):
                obs_vals[c].append(vals[:, j])
                new_vals[c].append(repaired[:, j])
        drifts = []
        edges = np.linspace(-1e-9, 1 + 1e-9, 11)
        for c in range(self.n_ch):
            if not obs_vals[c]:
                continue
            a = np.histogram(np.concatenate(obs_vals[c]), bins=edges)[0].astype(float)
            b = np.histogram(np.concatenate(new_vals[c]), bins=edges)[0].astype(float)
            if a.sum() == 0:
                continue
            drifts.append(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())
        return X.T, float(np.mean(drifts)) if drifts else float("nan")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def ordering_stats(labels: np.ndarray, order_id: np.ndarray, k: int) -> dict:
    """Dominant-order concentration and ordering entropy, per mode and pooled."""
    top1, eff, ent, sizes = [], [], [], []
    for m in range(k):
        sub = order_id[labels == m]
        sizes.append(int(sub.size))
        if sub.size == 0:
            top1.append(np.nan); eff.append(np.nan); ent.append(np.nan)
            continue
        c = np.bincount(sub)
        c = c[c > 0].astype(float)
        p = c / c.sum()
        top1.append(float(p.max()))
        h = float(-np.sum(p * np.log(p)))
        ent.append(h)
        eff.append(float(np.exp(h)))
    return {
        "per_mode_top1_ordering_share": top1,
        "per_mode_ordering_entropy_nats": ent,
        "per_mode_effective_ordering_count": eff,
        "per_mode_n_events": sizes,
        "equal_mode_weighted_dominant_order_concentration": float(np.nanmean(top1)),
        "equal_mode_weighted_effective_ordering_count": float(np.nanmean(eff)),
        "equal_mode_weighted_ordering_entropy_nats": float(np.nanmean(ent)),
    }


def _mc(values: np.ndarray) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "mc_se_of_mean": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
        "q05": float(np.percentile(v, 5)),
        "q50": float(np.percentile(v, 50)),
        "q95": float(np.percentile(v, 95)),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--n-perm", type=int, default=256)
    ap.add_argument("--out-dir", default=str(HERE / "selection_corrected_null"))
    ap.add_argument("--max-attempt-factor", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = args.subject
    t0 = time.time()

    rec = M.replay_and_audit(sid)                       # frozen replay + C1-C6
    labels_frozen = rec["_labels"]
    k = rec["chosen_k"]
    ve = rec["_valid_events"]
    ranks_v = rec["_loaded"]["ranks"][:, ve]
    bools_v = rec["_loaded"]["bools"][:, ve]

    X_obs = build_masked_kmeans_features(ranks_v, bools_v, impute="event_median")
    # ordering identity = the distinct masked feature vector
    _, order_id = np.unique(np.round(X_obs, 9), axis=0, return_inverse=True)
    order_id = np.asarray(order_id).ravel()
    n_orderings = int(order_id.max()) + 1

    observed = ordering_stats(labels_frozen, order_id, k)

    # sanity: does the producer-identical estimator reproduce the frozen labels?
    lab_refit, best_seed, sil = producer_kmeans(X_obs, k)
    refit = {
        "ami_vs_frozen_labels": float(adjusted_mutual_info_score(labels_frozen, lab_refit)),
        "best_seed": best_seed,
        "median_silhouette": float(np.nanmedian(sil)),
        "stats": ordering_stats(lab_refit, order_id, k),
    }

    masked_valid = np.where(bools_v, X_obs.T, np.nan)
    rnd = OrderRandomiser(bools_v, masked_valid)

    results = {}
    for null_kind in ("order", "marginal"):
        # deterministic across processes: Python's str hash is salted per process
        _sid_key = int.from_bytes(hashlib.sha256(sid.encode()).digest()[:4], "big") % 10_000
        rng = np.random.default_rng(SEED + _sid_key + (0 if null_kind == "order" else 1))
        draws, drifts, n_invalid = [], [], 0
        max_attempts = int(args.n_perm * args.max_attempt_factor)
        attempts = 0
        while len(draws) < args.n_perm and attempts < max_attempts:
            attempts += 1
            if null_kind == "order":
                Xn = rnd.draw_order(rng); drift = None
            else:
                Xn, drift = rnd.draw_marginal(rng)
            lab_n, _, _ = producer_kmeans(Xn, k)
            if np.unique(lab_n).size != k:
                n_invalid += 1
                continue
            _, oid_n = np.unique(np.round(Xn, 9), axis=0, return_inverse=True)
            draws.append(ordering_stats(lab_n, np.asarray(oid_n).ravel(), k))
            if drift is not None:
                drifts.append(drift)
        conc = np.array([d["equal_mode_weighted_dominant_order_concentration"] for d in draws])
        effc = np.array([d["equal_mode_weighted_effective_ordering_count"] for d in draws])
        entc = np.array([d["equal_mode_weighted_ordering_entropy_nats"] for d in draws])
        per_mode = np.array([d["per_mode_top1_ordering_share"] for d in draws])  # (n, k)
        obs_c = observed["equal_mode_weighted_dominant_order_concentration"]
        obs_e = observed["equal_mode_weighted_effective_ordering_count"]
        results[null_kind] = {
            "n_valid_randomisations": len(draws),
            "n_invalid_randomisations": n_invalid,
            "n_attempts": attempts,
            "concentration_null": _mc(conc),
            "effective_ordering_count_null": _mc(effc),
            "ordering_entropy_null": _mc(entc),
            "observed_minus_null_concentration": float(obs_c - conc.mean()) if conc.size else None,
            "observed_minus_null_effective_count": float(obs_e - effc.mean()) if effc.size else None,
            "observed_exceeds_null_q95_concentration": (
                bool(obs_c > np.percentile(conc, 95)) if conc.size else None),
            "empirical_p_concentration_ge": (
                float((np.sum(conc >= obs_c) + 1) / (conc.size + 1)) if conc.size else None),
            "per_mode_sorted_top1_null": {
                "note": "modes sorted descending within each draw so mode identity, "
                        "which KMeans does not preserve across draws, never enters",
                **{f"rank_{i+1}": _mc(np.sort(per_mode, axis=1)[:, ::-1][:, i])
                   for i in range(k)},
            },
            "marginal_drift_total_variation": (_mc(np.array(drifts)) if drifts else None),
            # full draw arrays: keeps every quantile and the empirical p
            # recomputable without re-running, and lets the figure show the
            # realised null distribution rather than three stored quantiles
            "draws_concentration": conc.tolist(),
            "draws_effective_ordering_count": effc.tolist(),
            "draws_ordering_entropy": entc.tolist(),
        }

    payload = {
        "provenance": {
            "git_commit": M._git_commit(),
            "seed": SEED,
            "seed_derivation": "SEED + sha256(subject_id)[:4] % 10000 + null_index",
            "n_perm_requested": args.n_perm,
            "kmeans": {"n_seeds": N_KMEANS_SEEDS, "n_init": KMEANS_N_INIT,
                       "silhouette_subsample": MAX_SILHOUETTE_EVENTS,
                       "rule": "producer-identical `_kmeans_stability_for_k` at frozen k"},
            "frozen": ["chosen_k", "valid-event eligibility (min_participating=3)",
                       "recording blocks", "masked feature construction", "contact order"],
            "elapsed_seconds": None,
            "safety": "reads only; never writes src/ or scripts/ (formal-cohort frozen modules)",
        },
        "subject_id": sid,
        "chosen_k": k,
        "n_channels": rec["n_channels"],
        "n_valid_events": rec["n_valid_events"],
        "n_distinct_orderings": n_orderings,
        "input_json_sha256": rec["input_json_sha256"],
        "labels_sha256": rec["labels_sha256"],
        "observed": observed,
        "producer_refit_sanity": refit,
        "nulls": results,
    }
    payload["provenance"]["elapsed_seconds"] = round(time.time() - t0, 1)
    out = out_dir / f"{sid}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    o = results["order"]
    print(f"{sid}: K={k} obs_conc={observed['equal_mode_weighted_dominant_order_concentration']:.4f} "
          f"null={o['concentration_null']['mean']:.4f} q95={o['concentration_null']['q95']:.4f} "
          f"delta={o['observed_minus_null_concentration']:+.4f} "
          f"exceeds_q95={o['observed_exceeds_null_q95_concentration']} "
          f"p={o['empirical_p_concentration_ge']:.4f} "
          f"(n={o['n_valid_randomisations']}, refit AMI={refit['ami_vs_frozen_labels']:.3f}, "
          f"{payload['provenance']['elapsed_seconds']:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
