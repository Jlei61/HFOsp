"""SEF-HFO SOZ-localization: HFO firing-rate vs propagation-geometry head-to-head.

Pure analysis functions consumed by scripts/{build_soz_localization_cohort,
run_sef_hfo_soz_localization}.py. masked-only geometry.

Contract:
  docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md (v3)
  docs/archive/topic4/sef_hfo/channel_universe_montage_diagnostic_2026-06-06.md

Narrowed claim (v3): within the HFO-active region, propagation ORDER (source/sink
endpoints) is a more sampling-stable SOZ fingerprint than firing COUNT. Geometry lives
on high-rate channels (a propagation order requires participation); SOZ channels it
misses are low-rate -> a shared blind spot of both readouts, not geometry's failure.
"""
from __future__ import annotations

from typing import Optional, Sequence

# §3.0 / Task 1 thresholds (locked v3)
MIN_CH_EVENTS = 30   # a channel is HFO-active iff its (bridged) rate channel has >=30 events
MIN_HOURS = 12       # subject quality gate (drops broken pengzihang, total_hours=2.0)
MIN_UNIVERSE = 5     # comparison-A eligibility: |U| >= 5
MIN_SOZ_IN_U = 2     # comparison-A eligibility: |SOZ_core ∩ U| >= 2


def comparison_a_eligibility(
    n_universe: int,
    n_soz_in_u: int,
    min_universe: int = MIN_UNIVERSE,
    min_soz_in_u: int = MIN_SOZ_IN_U,
) -> tuple:
    """Single source of truth for comparison-A eligibility (used by build_cohort AND
    comparison_a_subject so the frozen flag matches what the runner computes).

    AUC-computable iff: |SOZ∩U| >= min_soz_in_u, |U| >= min_universe, AND there is at
    least one non-SOZ channel in U (|SOZ∩U| < |U|; else no negatives -> AUC undefined).
    Returns (eligible: bool, reason: str).
    """
    if n_soz_in_u < min_soz_in_u:
        return False, f"|SOZ∩U|={n_soz_in_u}<{min_soz_in_u}"
    if n_universe < min_universe:
        return False, f"|U|={n_universe}<{min_universe}"
    if n_soz_in_u >= n_universe:
        return False, "no non-SOZ in U (AUC undefined)"
    return True, ""


def classify_montage(channel_metrics: Sequence[dict], hyphen_frac_threshold: float = 0.8) -> str:
    """yuquan rate naming varies per subject: bipolar pairs ('E9-E10') vs single
    contacts ('A6'). Classify by the fraction of names containing '-'."""
    names = [c["ch_name"] for c in channel_metrics]
    if not names:
        return "single"
    frac = sum("-" in n for n in names) / len(names)
    return "bipolar" if frac >= hyphen_frac_threshold else "single"


def _first_contact(bipolar_name: str) -> str:
    """'E11-E12' -> 'E11' (split on first '-'; contact labels may carry primes, e.g. A'7)."""
    return bipolar_name.split("-", 1)[0]


def build_rate_lookup(
    channel_metrics: Sequence[dict],
    montage: str,
    min_ch_events: int = MIN_CH_EVENTS,
) -> dict:
    """Map a geometry single-contact name -> full-recording event_rate, via the montage bridge.

    single  : exact rate name == contact.
    bipolar : contact X -> the UNIQUE bipolar pair whose FIRST contact is X ('X-next'),
              the verified lagPat labeling convention (geom ⊆ first-contacts ∧ ⊄ second-contacts;
              see diagnostic doc §2). No unique first-contact pair -> contact absent (missing, not 0).

    Channels whose n_events < min_ch_events are excluded (no zero-fill).
    """
    lookup: dict = {}
    if montage == "single":
        for c in channel_metrics:
            if c["n_events"] >= min_ch_events:
                lookup[c["ch_name"]] = c["event_rate"]
        return lookup
    # bipolar first-contact bridge (gate before mapping so uniqueness is over passing pairs)
    first_map: dict = {}
    for c in channel_metrics:
        name = c["ch_name"]
        if "-" in name and c["n_events"] >= min_ch_events:
            first_map.setdefault(_first_contact(name), []).append(c)
    for contact, pairs in first_map.items():
        if len(pairs) == 1:  # unique first-contact pair only; ambiguous (>=2) -> drop
            lookup[contact] = pairs[0]["event_rate"]
    return lookup


def geom_valid_contacts(geom_pair: dict) -> list:
    """Contacts with joint_valid=True in the primary geometry pair (pairs[0]).

    §6.1 alignment: joint_valid is aligned to pair['channel_names']; callers must NOT
    assume pair order == top-level channel_names order (re-align by name elsewhere).
    """
    names = list(geom_pair["channel_names"])
    valid = list(geom_pair["joint_valid"])
    if len(names) != len(valid):
        raise ValueError(
            f"channel_names ({len(names)}) != joint_valid ({len(valid)}) length mismatch"
        )
    return [n for n, v in zip(names, valid) if v]


def compute_channel_universe(
    geom_pair: dict,
    channel_metrics: Sequence[dict],
    min_ch_events: int = MIN_CH_EVENTS,
) -> dict:
    """U = geometry-valid contacts whose bridged rate channel is HFO-active (>=min events).

    Contacts that are geometry-valid but have no bridged rate (or below the event gate)
    are reported in geom_missing_no_rate and EXCLUDED from U (not zero-filled, §3.0/§6).
    """
    montage = classify_montage(channel_metrics)
    rate_lookup = build_rate_lookup(channel_metrics, montage, min_ch_events)
    geom_valid = geom_valid_contacts(geom_pair)
    universe = [c for c in geom_valid if c in rate_lookup]
    missing = [c for c in geom_valid if c not in rate_lookup]
    return {
        "montage": montage,
        "rate_lookup": rate_lookup,
        "geom_valid": geom_valid,
        "universe": universe,
        "geom_missing_no_rate": missing,
    }


def geom_scores(channels: Sequence[str], geom_pair: dict) -> dict:
    """Per-channel propagation-geometry scores, aligned to `channels` (§3.1).

    Dense ranks r_a/r_b are 0..n_valid-1 (0=source, n_valid-1=sink), aligned to
    geom_pair['channel_names']. §6.1: `channels` may be in a different order than the
    pair — look each up by name. A channel absent from the pair, or with joint_valid=False,
    scores NaN (missing, not 0).

      endpoint(c) = mean( 2|r_a/(n-1)-0.5| , 2|r_b/(n-1)-0.5| )   # source OR sink -> 1, middle -> 0
      source(c)   = 1 - r_a/(n-1)                                  # earliest -> 1
      sink(c)     = r_a/(n-1)                                      # latest -> 1
    where n = n_valid. Degenerate n_valid < 2 -> all NaN.
    """
    import numpy as _np

    names = list(geom_pair["channel_names"])
    valid = list(geom_pair["joint_valid"])
    ra = list(geom_pair["rank_a_dense_full"])
    rb = list(geom_pair["rank_b_dense_full"])
    if not (len(names) == len(valid) == len(ra) == len(rb)):
        raise ValueError("geom_pair arrays length mismatch (channel_names/joint_valid/rank_a/rank_b)")

    idx_of = {n: i for i, n in enumerate(names)}
    n_valid = sum(bool(v) for v in valid)
    nan = float("nan")
    endpoint, source, sink = [], [], []
    denom = (n_valid - 1) if n_valid >= 2 else None
    for c in channels:
        i = idx_of.get(c)
        if i is None or not valid[i] or denom is None:
            endpoint.append(nan); source.append(nan); sink.append(nan)
            continue
        a = ra[i] / denom
        b = rb[i] / denom
        endpoint.append(0.5 * (2 * abs(a - 0.5) + 2 * abs(b - 0.5)))
        source.append(1.0 - a)
        sink.append(a)
    return {"endpoint": _np.asarray(endpoint, dtype=float),
            "source": _np.asarray(source, dtype=float),
            "sink": _np.asarray(sink, dtype=float)}


def align_rate_and_soz(
    universe: Sequence[str],
    rate_in_universe: dict,
    soz_core: Sequence[str],
) -> tuple:
    """Align the (montage-bridged) rate and the SOZ truth onto the universe (§3.2).

    Consumes the frozen cohort outputs (universe + rate_in_universe + soz_core) — the
    bridge was applied once in build_cohort (single source of truth). Returns
    (rate_vec, y) as float / bool arrays aligned to `universe` order, where
    y[i] = (universe[i] in SOZ_core). sum(y) == |SOZ_core ∩ U|; SOZ channels outside U
    never appear, and every universe channel must carry a rate (no 0-fill -> raise).
    """
    import numpy as _np

    missing = [c for c in universe if c not in rate_in_universe]
    if missing:
        raise ValueError(f"universe channels without bridged rate (would 0-fill): {missing}")
    soz_set = set(soz_core)
    rate_vec = _np.asarray([rate_in_universe[c] for c in universe], dtype=float)
    y = _np.asarray([c in soz_set for c in universe], dtype=bool)
    return rate_vec, y


def comparison_a_subject(
    scores: dict,
    y,
    min_universe: int = MIN_UNIVERSE,
    min_soz_in_u: int = MIN_SOZ_IN_U,
) -> dict:
    """Static SOZ-localization per subject (§4): ROC-AUC + top-k overlap for each score.

    `scores`: {name -> per-channel score array aligned to the universe} (e.g. rate, endpoint,
    source, sink, swap). `y`: boolean SOZ membership aligned to the universe.

    AUC = roc_auc_score(y, score). top-k overlap = |top-k-by-score ∩ SOZ| / k at k=|SOZ∩U|.
    Degenerate (insufficient, no AUC reported): |SOZ∩U| < min_soz_in_u, OR |U| < min_universe,
    OR no non-SOZ channel in U (AUC undefined).
    """
    import numpy as _np
    from sklearn.metrics import roc_auc_score

    y = _np.asarray(y, dtype=bool)
    n_u = int(y.size)
    n_soz = int(y.sum())
    base = {"n_universe": n_u, "n_soz_in_u": n_soz, "auc": {}, "topk_overlap": {}}
    eligible, reason = comparison_a_eligibility(n_u, n_soz, min_universe, min_soz_in_u)
    if not eligible:
        return {**base, "insufficient": True, "reason": reason}

    soz_idx = set(_np.where(y)[0].tolist())
    k = n_soz
    auc, topk = {}, {}
    for name, sc in scores.items():
        sc = _np.asarray(sc, dtype=float)
        if not _np.all(_np.isfinite(sc)):
            raise ValueError(f"score '{name}' has non-finite values in U (universe must be finite)")
        auc[name] = float(roc_auc_score(y, sc))
        order = _np.argsort(-sc, kind="mergesort")  # descending, stable on ties
        topk[name] = len(set(order[:k].tolist()) & soz_idx) / k
    return {"n_universe": n_u, "n_soz_in_u": n_soz, "insufficient": False, "reason": "",
            "auc": auc, "topk_overlap": topk}


# =============================================================================
# v4 MAIN analysis — SOZ-internal readout stability over sampling (post-review)
# Within SOZ∩U: under short/noisy sampling, which readout keeps its top-k closest to the
# full-recording top-k — firing participation COUNT, or a propagation-geometry readout?
# Geometry readouts: source (reversal-SENSITIVE), sink, ENDPOINT/axis (reversal-INVARIANT),
# template-stratified source_fwd/source_rev. count-matched null separates few-events
# sampling noise from real temporal drift. (plan v4 §5 reframe; review fixes E1/E2/S1/S2)
# =============================================================================

# readout name -> (base vector key, pick-largest?). source/sink are reversal-SENSITIVE;
# endpoint is reversal-INVARIANT; *_fwd/_rev require per-event template labels.
_READOUT_SPECS = {
    "rate": ("rate", True),
    "source": ("mean_rank", False),
    "sink": ("mean_rank", True),
    "endpoint": ("endpoint", True),
    "source_fwd": ("mean_rank_fwd", False),
    "source_rev": ("mean_rank_rev", False),
}


def _nanmean_rows(vals):
    import numpy as _np
    with _np.errstate(invalid="ignore"):
        return _np.array([_np.nanmean(r) if _np.any(~_np.isnan(r)) else _np.nan for r in vals],
                         dtype=float)


def compute_readouts(masked, bools, ch_idx, ev_idx, labels=None, fwd_id: int = 0, rev_id: int = 1):
    """Per-channel base readout vectors over a set of events, aligned to ch_idx.

    rate       = participation count (never NaN; 0 if absent).
    mean_rank  = mean masked propagation rank over participated events (low=source, high=sink);
                 NaN if no participation. Reversal-SENSITIVE.
    endpoint   = mean of 2|masked-0.5| over participated events (high=extreme=source OR sink).
                 Reversal-INVARIANT (a channel extreme in either template stays extreme).
    mean_rank_fwd/_rev (only if `labels` given) = mean_rank restricted to template fwd_id / rev_id
                 events — removes the source/sink blur that mixing reversing templates causes.
    """
    import numpy as _np

    ch_idx = _np.asarray(ch_idx, dtype=int)
    ev_idx = _np.asarray(ev_idx, dtype=int)
    sub_b = _np.asarray(bools)[_np.ix_(ch_idx, ev_idx)].astype(bool)
    sub_m = _np.asarray(masked, dtype=float)[_np.ix_(ch_idx, ev_idx)]
    vals = _np.where(sub_b, sub_m, _np.nan)
    out = {
        "rate": sub_b.sum(axis=1).astype(float),
        "mean_rank": _nanmean_rows(vals),
        "endpoint": _nanmean_rows(2.0 * _np.abs(vals - 0.5)),
    }
    if labels is not None:
        lab = _np.asarray(labels)[ev_idx]
        out["mean_rank_fwd"] = _nanmean_rows(vals[:, lab == fwd_id]) if _np.any(lab == fwd_id) \
            else _np.full(len(ch_idx), _np.nan)
        out["mean_rank_rev"] = _nanmean_rows(vals[:, lab == rev_id]) if _np.any(lab == rev_id) \
            else _np.full(len(ch_idx), _np.nan)
    return out


def topk_indices(values, k: int, largest: bool = True) -> set:
    """Indices of the top-k values (largest for rate/sink/endpoint, smallest for source rank).

    NaN is treated as worst (never selected). Stable tie-break by index. Note: this does NOT
    guard against <k finite values — callers needing that must use _topk_or_none (E2 fix)."""
    import numpy as _np

    v = _np.asarray(values, dtype=float)
    worst = -_np.inf if largest else _np.inf
    v = _np.where(_np.isnan(v), worst, v)
    order = _np.argsort(-v if largest else v, kind="stable")
    k = min(int(k), v.size)
    return set(int(i) for i in order[:k])


def _topk_or_none(values, k: int, largest: bool):
    """top-k set, or None if fewer than k FINITE values exist (E2: no NaN back-fill)."""
    import numpy as _np
    v = _np.asarray(values, dtype=float)
    if int(_np.sum(_np.isfinite(v))) < k:
        return None
    return topk_indices(v, k, largest=largest)


def jaccard(a: set, b: set) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def soz_internal_source_stability(
    ranks, bools, event_times, channel_names, soz_u_channels,
    M_grid, k: int, n_null: int = 200, n_starts: int = 8, seed: int = 0,
    labels=None, fwd_id: int = 0, rev_id: int = 1, readouts=None,
):
    """Window-vs-full reproducibility of the SOZ-internal top-k target, per readout.

    For each event budget M: n_starts time-CONTIGUOUS windows of M events; recompute each
    readout's top-k within soz_u_channels; Jaccard + Spearman to the full-recording target.
    NULL draws M events at random (time-scrambled) -> sampling-noise floor; observed < null
    => real temporal drift beyond sampling. A window/draw is skipped for a readout if it has
    <k finite channels (E2). Masking reuses lagpat_rank_audit.mask_phantom_ranks.

    `labels` (per-event template id) enables the reversal-robust source_fwd/source_rev readouts.
    """
    import numpy as _np
    from scipy.stats import spearmanr
    from src.lagpat_rank_audit import mask_phantom_ranks

    masked = mask_phantom_ranks(_np.asarray(ranks, dtype=float), _np.asarray(bools) > 0, normalize=True)
    name_idx = {n: i for i, n in enumerate(channel_names)}
    ch_idx = [name_idx[c] for c in soz_u_channels]
    n_ev = int(len(event_times))
    time_order = _np.argsort(_np.asarray(event_times, dtype=float), kind="stable")

    if readouts is None:
        readouts = ["rate", "source", "sink", "endpoint"]
        if labels is not None:
            readouts += ["source_fwd", "source_rev"]

    def _vecs(ev):
        return compute_readouts(masked, bools, ch_idx, ev, labels=labels, fwd_id=fwd_id, rev_id=rev_id)

    def _spearman(win_v, full_v):
        ok = _np.isfinite(win_v) & _np.isfinite(full_v)
        if ok.sum() < 3:
            return _np.nan
        rho = spearmanr(win_v[ok], full_v[ok]).correlation
        return float(rho) if rho == rho else _np.nan

    full_vecs = _vecs(_np.arange(n_ev))
    full_topk, full_target_names = {}, {}
    for r in readouts:
        vkey, largest = _READOUT_SPECS[r]
        s = _topk_or_none(full_vecs[vkey], k, largest)
        full_topk[r] = s
        full_target_names[r] = None if s is None else sorted(soz_u_channels[i] for i in s)  # E1 fix

    rng = _np.random.default_rng(seed)
    curves = {}
    for M in M_grid:
        Meff = min(int(M), n_ev)
        n_pos = n_ev - Meff
        starts = _np.unique(_np.linspace(0, n_pos, num=min(n_starts, n_pos + 1), dtype=int))
        win_evs = [time_order[st:st + Meff] for st in starts]
        null_evs = [rng.choice(n_ev, size=Meff, replace=False) for _ in range(n_null)]
        win_vecs = [_vecs(ev) for ev in win_evs]
        null_vecs = [_vecs(ev) for ev in null_evs]

        cur = {"M_eff": Meff, "n_windows": int(len(starts))}
        for r in readouts:
            vkey, largest = _READOUT_SPECS[r]
            full_set = full_topk[r]
            if full_set is None:
                cur[r] = {"jaccard_obs": float("nan"), "jaccard_null": float("nan"),
                          "spearman_obs": float("nan"), "spearman_null": float("nan"),
                          "drift_frac": float("nan"), "n_valid_windows": 0, "n_valid_null": 0,
                          "insufficient": True}
                continue
            j_obs, sp_obs = [], []
            for wv in win_vecs:
                ws = _topk_or_none(wv[vkey], k, largest)
                if ws is None:
                    continue
                j_obs.append(jaccard(ws, full_set))
                sp_obs.append(_spearman(wv[vkey], full_vecs[vkey]))
            j_null, sp_null = [], []
            for wv in null_vecs:
                ws = _topk_or_none(wv[vkey], k, largest)
                if ws is None:
                    continue
                j_null.append(jaccard(ws, full_set))
                sp_null.append(_spearman(wv[vkey], full_vecs[vkey]))
            nm = lambda a: float(_np.nanmean(a)) if len(a) and _np.any(_np.isfinite(a)) else float("nan")
            j_obs_m = nm(j_obs)
            j_null_a = _np.asarray(j_null, dtype=float)
            cur[r] = {
                "jaccard_obs": j_obs_m,
                "jaccard_null": float(_np.mean(j_null_a)) if j_null_a.size else float("nan"),
                "spearman_obs": nm(sp_obs), "spearman_null": nm(sp_null),
                "drift_frac": (float(_np.mean(j_null_a <= j_obs_m))
                               if (j_null_a.size and j_obs_m == j_obs_m) else float("nan")),
                "n_valid_windows": len(j_obs), "n_valid_null": len(j_null),
                "insufficient": len(j_obs) == 0,
            }
        curves[M] = cur

    return {"k": k, "n_events": n_ev, "n_soz_u": len(ch_idx),
            "readouts": readouts, "full_targets": full_target_names, "curves": curves}


def aggregate_comparison_a(per_subject: Sequence[dict], primary_geom: str = "endpoint") -> dict:
    """Cohort-level comparison-A: paired geometry-vs-rate AUC over eligible subjects (§4).

    per_subject records: {subject, dataset, soz_coverage, insufficient, auc:{rate, endpoint, ...}}.
    Insufficient subjects are excluded from the paired test (reported in n_insufficient).

    For each geometry score present alongside 'rate': one-sided paired Wilcoxon (geom AUC >=
    rate AUC), median ΔAUC (geom - rate), count(geom >= rate). Also median AUC per score and
    median SOZ coverage per dataset (the honest-limitation readout).
    """
    import numpy as _np
    from scipy.stats import wilcoxon

    eligible = [r for r in per_subject if not r.get("insufficient")]
    geom_names = [k for k in (eligible[0]["auc"].keys() if eligible else []) if k != "rate"]

    delta, n_ge, wilco, median_auc = {}, {}, {}, {}
    rate_aucs = _np.array([r["auc"]["rate"] for r in eligible], dtype=float)
    median_auc["rate"] = float(_np.median(rate_aucs)) if rate_aucs.size else float("nan")
    for g in geom_names:
        g_aucs = _np.array([r["auc"][g] for r in eligible], dtype=float)
        diffs = g_aucs - rate_aucs
        median_auc[g] = float(_np.median(g_aucs)) if g_aucs.size else float("nan")
        delta[g] = float(_np.median(diffs)) if diffs.size else float("nan")
        n_ge[g] = int(_np.sum(diffs >= 0))
        # one-sided paired Wilcoxon (g >= rate); needs >=1 nonzero diff
        if _np.any(diffs != 0):
            try:
                wilco[g] = float(wilcoxon(g_aucs, rate_aucs, alternative="greater",
                                          zero_method="wilcox").pvalue)
            except ValueError:
                wilco[g] = float("nan")
        else:
            wilco[g] = float("nan")

    cov_by_ds = {}
    for ds in sorted({r["dataset"] for r in per_subject}):
        covs = [r["soz_coverage"] for r in per_subject if r["dataset"] == ds]
        cov_by_ds[ds] = float(_np.median(covs)) if covs else float("nan")

    return {
        "n_total": len(per_subject),
        "n_eligible": len(eligible),
        "n_insufficient": len(per_subject) - len(eligible),
        "primary_geom": primary_geom,
        "median_auc": median_auc,
        "median_delta_auc": delta,
        "n_geom_ge_rate": n_ge,
        "wilcoxon_p_endpoint_ge_rate": wilco.get(primary_geom, float("nan")),
        "wilcoxon_p_by_geom": wilco,
        "median_soz_coverage": cov_by_ds,
    }


def build_cohort(
    candidates: Sequence[dict],
    min_hours: int = MIN_HOURS,
    min_ch_events: int = MIN_CH_EVENTS,
) -> dict:
    """Three-source intersection -> per-subject channel universe + SOZ coverage.

    Each candidate: {dataset, subject, total_hours, soz_core[list], channel_metrics[list],
    geom_pair[dict|None]}. A subject is KEPT iff it has all three sources, SOZ_core is
    non-empty, and total_hours >= min_hours. Excluded subjects carry a reason.

    Per kept subject: channel universe U (montage-bridged), soz_coverage = |SOZ∩U|/|SOZ_core|,
    and comparison_a_eligible = (|SOZ∩U| >= MIN_SOZ_IN_U) and (|U| >= MIN_UNIVERSE).
    """
    kept: list = []
    excluded: list = []
    for cand in candidates:
        ds = cand["dataset"]
        subj = cand["subject"]
        soz = list(cand.get("soz_core") or [])
        cm = cand.get("channel_metrics")
        pair = cand.get("geom_pair")
        th = cand.get("total_hours")

        if not soz:
            excluded.append({"dataset": ds, "subject": subj, "reason": "empty_soz"})
            continue
        if not cm:
            excluded.append({"dataset": ds, "subject": subj, "reason": "no_rate"})
            continue
        if pair is None:
            excluded.append({"dataset": ds, "subject": subj, "reason": "no_geometry"})
            continue
        if th is None or th < min_hours:
            excluded.append(
                {"dataset": ds, "subject": subj, "reason": f"total_hours<{min_hours} (={th})"}
            )
            continue

        u = compute_channel_universe(pair, cm, min_ch_events)
        universe = u["universe"]
        rate_in_universe = {c: u["rate_lookup"][c] for c in universe}  # §6: exactly U, no phantom keys
        soz_in_u = sorted(set(universe) & set(soz))
        coverage = len(soz_in_u) / len(soz)
        eligible, _ = comparison_a_eligibility(len(universe), len(soz_in_u), MIN_UNIVERSE, MIN_SOZ_IN_U)
        kept.append(
            {
                "dataset": ds,
                "subject": subj,
                "total_hours": th,
                "montage": u["montage"],
                "universe": universe,
                "n_universe": len(universe),
                "soz_core": soz,
                "n_soz_core": len(soz),
                "soz_in_universe": soz_in_u,
                "n_soz_in_universe": len(soz_in_u),
                "soz_coverage": coverage,
                "comparison_a_eligible": eligible,
                "geom_missing_no_rate": u["geom_missing_no_rate"],
                "rate_in_universe": rate_in_universe,
            }
        )

    return {
        "kept": kept,
        "excluded": excluded,
        "meta": {
            "min_hours": min_hours,
            "min_ch_events": min_ch_events,
            "min_universe": MIN_UNIVERSE,
            "min_soz_in_universe": MIN_SOZ_IN_U,
        },
    }
