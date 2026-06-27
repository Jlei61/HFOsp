"""mini-W_event pilot helpers (Topic 4 M3, Step D / B1 verification).

Design: docs/archive/topic4/sef_hfo/m3_mini_w_event_design_2026-06-23.md §2-§4.
PILOT SCOPE ONLY: K_min(q) extraction + center-source W_shape reproducibility (B1a).
B1b (axis fit), B1c (ordering predictivity), B1d (matched shape) are post-review step 4
and deliberately NOT implemented here.

REUSES the canonical upstream flag definitions — do NOT redefine (CLAUDE.md §6, §6.1):
  - reclassify_m3_ea_primary: _ea_local_flag (EA-local = r95_ea<=6 AND far_ea<=0.5 AND
    returned>=1), _per_seed_ea, _p_ea, R95_CAP, FAR_CAP, ROBUST_FRAC (0.7), MIN_SEEDS (6).
  - audit_m3_core_only_seed_confounds: spontaneous_ignition_flag, _per_seed_core_only
    (per-seed max core_only over kicks/windows vs global bare-bg median — Step B).

The ea_net_bins.npz raw ingredient is produced by run_m3_kick_calibration.py --emit-ea-bins;
W_shape CONSTRUCTION lives here (design §0: "不混进 runner").
"""
import os
import sys

import numpy as np

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from reclassify_m3_ea_primary import (  # noqa: E402  (reuse, don't reinvent)
    _ea_local_flag, _per_seed_ea, _p_ea, R95_CAP, FAR_CAP, ROBUST_FRAC, MIN_SEEDS,
)
from audit_m3_core_only_seed_confounds import (  # noqa: E402
    spontaneous_ignition_flag, _per_seed_core_only,
)

__all__ = [
    "extract_kmin", "extract_k50", "success_seeds_at_kick",
    "build_w_shape", "w_shape_reproducibility", "load_run_dir",
]


# --------------------------------------------------------------------------- #
# K_min / K50 (C8)                                                             #
# --------------------------------------------------------------------------- #
def extract_kmin(kicks, p_ea, n_seeds, thresh=ROBUST_FRAC, min_seeds=MIN_SEEDS):
    """Smallest kick with P_EA-local-returned >= thresh AND n_seeds >= min_seeds.

    Matches reclassify_m3_ea_primary's EA-local cell rule (P_EA>=0.7 AND n>=6).
    Returns inf when no kick qualifies (right-censored).
    """
    for k, p, n in sorted(zip(kicks, p_ea, n_seeds)):
        if n >= min_seeds and not np.isnan(p) and p >= thresh:
            return float(k)
    return float("inf")


def extract_k50(kicks, p_ea, level=0.5):
    """Kick where the P_EA curve linearly crosses `level` (0.5). NaN if never reached.

    Used to pick the near-threshold kick at which W_shape is estimated (design §3(2):
    compare "how it spreads once ignited", not "where it ignites easiest").
    """
    pts = sorted((float(k), float(p)) for k, p in zip(kicks, p_ea) if not np.isnan(p))
    for (k0, p0), (k1, p1) in zip(pts, pts[1:]):
        if (p0 < level <= p1) or (p0 >= level > p1):
            if p1 == p0:
                return float(k0)
            return float(k0 + (level - p0) * (k1 - k0) / (p1 - p0))
    return float("nan")


# --------------------------------------------------------------------------- #
# Success-seed selection (C6) — EA-local AND NOT spontaneous-ignition           #
# --------------------------------------------------------------------------- #
def success_seeds_at_kick(recs, spont_seeds, r95cap=R95_CAP, farcap=FAR_CAP):
    """Seeds at one kick that are EA-local-returned (reused _ea_local_flag) AND not in the
    Step-B spontaneous-ignition set. Both gates required (design §3(2))."""
    spont = {int(s) for s in spont_seeds}
    return {int(r["seed"]) for r in recs
            if _ea_local_flag(r, r95cap, farcap) and int(r["seed"]) not in spont}


# --------------------------------------------------------------------------- #
# W_shape construction (C7) — source-excluded, per-seed normalized, success-only #
# --------------------------------------------------------------------------- #
def build_w_shape(ea_matrix, success_seeds, src_bin_idx, normalize="l1"):
    """W_event_shape(p|q) from the per-seed event-aligned per-bin matrix.

    ea_matrix : (n_seed, n_bins) per-bin early-window differenced response (ea_net_bins.npz).
    success_seeds : seeds that are EA-local-returned AND not spontaneous (success_seeds_at_kick).

    Drops the source bin (the kick lands there — not propagation), L1/L2-normalizes each
    seed's shape, and averages over success seeds only. NaN (no-event) and empty (all-zero,
    un-normalizable) rows are excluded even if listed as success. Raises if none survive.

    Returns (per_seed_w_shapes (n_used, n_bins-1), mean_w_shape (n_bins-1,), used_seeds).
    """
    ea_matrix = np.asarray(ea_matrix, dtype=float)
    n_seed, n_bins = ea_matrix.shape
    nonsrc = [b for b in range(n_bins) if b != src_bin_idx]
    rows, used = [], []
    for s in sorted(int(x) for x in success_seeds):
        v = ea_matrix[s, nonsrc]
        if np.any(~np.isfinite(v)):
            continue                       # no-event NaN row -> exclude
        tot = v.sum() if normalize == "l1" else float(np.linalg.norm(v))
        if tot <= 0:
            continue                       # empty shape, cannot normalize -> exclude
        rows.append(v / tot)
        used.append(s)
    if not rows:
        raise ValueError(
            "build_w_shape: zero successful seeds with a finite, non-empty shape")
    per_seed = np.vstack(rows)
    return per_seed, per_seed.mean(axis=0), used


# --------------------------------------------------------------------------- #
# B1a reproducibility (C9) — observed cross-seed similarity vs bin-shuffle null  #
# --------------------------------------------------------------------------- #
def _pairwise_mean_sim(M, metric):
    n = M.shape[0]
    if n < 2:
        return float("nan")
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i], M[j]
            if metric == "cosine":
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                sims.append(float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0)
            elif metric == "spearman":
                from scipy.stats import spearmanr
                sims.append(float(spearmanr(a, b).correlation))
            else:
                raise ValueError(f"unknown metric {metric!r}")
    return float(np.mean(sims))


def w_shape_reproducibility(per_seed, n_null=1000, metric="cosine", rng_seed=0):
    """B1a: do different seeds agree on the W_shape beyond chance bin-alignment?

    observed = mean pairwise similarity across the per-seed W_shapes.
    null = same statistic after INDEPENDENTLY permuting the spatial bins of each seed
    (destroys cross-seed spatial correspondence while preserving each shape's value set).
    pass iff observed >= null p95 (design §4 B1a draft threshold). Deterministic rng.
    """
    per_seed = np.asarray(per_seed, dtype=float)
    observed = _pairwise_mean_sim(per_seed, metric)
    rng = np.random.default_rng(rng_seed)
    n_seed, n_bins = per_seed.shape
    null = np.empty(n_null)
    for t in range(n_null):
        shuffled = np.vstack([per_seed[i, rng.permutation(n_bins)] for i in range(n_seed)])
        null[t] = _pairwise_mean_sim(shuffled, metric)
    null_p95 = float(np.nanpercentile(null, 95))
    return {
        "observed": observed, "null_p95": null_p95, "n_seeds": int(n_seed),
        "metric": metric, "n_null": int(n_null), "pass": bool(observed >= null_p95),
    }


# --------------------------------------------------------------------------- #
# Loader (integration) — assemble one (substrate, source) run_dir               #
# --------------------------------------------------------------------------- #
EXPECTED_N_BINS = 25   # the 5x5 ceiling working point; 4x4 (16) puts the center at a junction


def load_run_dir(run_dir, expected_n_bins=EXPECTED_N_BINS):
    """Assemble the pilot inputs for one runner output dir (one substrate, one source).

    Reuses _per_seed_ea (per-(kick,seed) EA recs + P_EA) and _per_seed_core_only +
    spontaneous_ignition_flag (Step-B spontaneous set). Loads ea_net_bins.npz (--emit-ea-bins).

    FAIL CLOSED against stale/mixed artifacts (P1-2, 2026-06-24) — the 4x4->5x5 bug slipped
    in because n_bins is in thresholds.json, not config.sweep_parameters. Raises ValueError on:
      - thresholds.json n_bins != expected_n_bins (e.g. a stale 4x4 run)
      - npz n_bins != thresholds n_bins (mixed artifacts)
      - ea_net_bins.shape[2] != n_bins
      - npz kicks != sorted CSV kicks
      - src_bin_idx != argmin(bin_centers, kick_xy)

    Returns dict: kicks, p_ea (per kick), n_seeds (per kick), spont_seeds (set),
    ea_net_bins (n_kick,n_seed,n_bins), src_bin_idx, bin_centers, recs_by_kick, npz_kicks.
    """
    import json
    by_kick = _per_seed_ea(run_dir)
    kicks = sorted(by_kick)
    npz = np.load(os.path.join(run_dir, "ea_net_bins.npz"))

    # --- fail-closed guards -------------------------------------------------- #
    thr_path = os.path.join(run_dir, "thresholds.json")
    thr_n_bins = int(json.load(open(thr_path))["n_bins"]) if os.path.exists(thr_path) else None
    npz_n_bins = int(npz["n_bins"]) if "n_bins" in npz.files else None
    if thr_n_bins is not None and thr_n_bins != expected_n_bins:
        raise ValueError(f"{run_dir}: thresholds.json n_bins={thr_n_bins} != expected "
                         f"{expected_n_bins} (stale grid? 4x4 puts the center on a junction)")
    if npz_n_bins is not None and thr_n_bins is not None and npz_n_bins != thr_n_bins:
        raise ValueError(f"{run_dir}: npz n_bins={npz_n_bins} != thresholds n_bins="
                         f"{thr_n_bins} (mixed artifacts)")
    ea = npz["ea_net_bins"]
    ref_n_bins = npz_n_bins if npz_n_bins is not None else expected_n_bins
    if ea.shape[2] != ref_n_bins:
        raise ValueError(f"{run_dir}: ea_net_bins.shape[2]={ea.shape[2]} != n_bins="
                         f"{ref_n_bins} (mixed/stale artifacts)")
    npz_kicks = [float(k) for k in npz["kicks"]]
    if npz_kicks != [float(k) for k in kicks]:
        raise ValueError(f"{run_dir}: npz kicks {npz_kicks} != sorted CSV kicks "
                         f"{[float(k) for k in kicks]} (mixed artifacts)")
    bin_centers = npz["bin_centers"]
    src = int(npz["src_bin_idx"])
    if "kick_xy" in npz.files and np.all(np.isfinite(npz["kick_xy"])):
        kx = np.asarray(npz["kick_xy"], dtype=float)
        expect_src = int(np.argmin(np.linalg.norm(bin_centers - kx[None, :], axis=1)))
        if src != expect_src:
            raise ValueError(f"{run_dir}: src_bin_idx={src} != argmin(bin_centers, kick_xy)="
                             f"{expect_src} (kick_xy={kx.tolist()}; stale/mismatched src bin)")

    p_ea = [_p_ea(by_kick[k], R95_CAP, FAR_CAP) for k in kicks]
    n_seeds = [len(by_kick[k]) for k in kicks]
    co_by_seed, bg_med = _per_seed_core_only(run_dir)
    spont = {int(s) for s, v in co_by_seed.items()
             if spontaneous_ignition_flag(v, bg_med)}
    return {
        "kicks": kicks, "p_ea": p_ea, "n_seeds": n_seeds, "spont_seeds": spont,
        "ea_net_bins": ea, "npz_kicks": npz["kicks"],
        "src_bin_idx": src, "bin_centers": bin_centers,
        "recs_by_kick": by_kick,
    }
