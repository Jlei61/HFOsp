"""Stage 4-long PRIMARY endpoint (reviewer 2026-06-17): does a STABLE k=2 read-out template pair —
gated by the validated AMI-stability null, NOT a raw KMeans-centroid correlation — emerge AND
reproduce as events accumulate in ONE long recording? (template-level forward/reverse line-advantage
from a single extended patch + the fixed anisotropic E->E axis — the layer that maps to the real-data
two-template structure, which the event-level Phase-2 sweep never tested).

§6.1 NOTE: a naive "two cluster-centroids' Spearman r" is NOT a valid metric — KMeans k=2 centroids are
ALWAYS anti-correlated (they straddle the global mean), so pure noise scores inter_corr ≈ -0.7. We
therefore REUSE the canonical null-bearing pipeline: `compute_adaptive_cluster_stereotypy`
(AMI-stability gate -> `stable_k`; noise -> stable_k=None, no fwd/rev pair) + the candidate
forward/reverse pair it annotates, and `compute_time_split_reproducibility` (Hungarian cross-split
match). NO sim. NO conclusion written here — reports the accumulation + reproducibility for a human.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.getcwd())
from src.interictal_propagation import (load_subject_propagation_events,                 # noqa: E402
                                        compute_adaptive_cluster_stereotypy,
                                        compute_time_split_reproducibility, _valid_event_indices)

S_OFF, S_EDGE, AX_CLEAN = 1.0, 4.5, 25.0


def _stereotypy(ranks, bools, names):
    """Canonical AMI-stability-gated k-scan. The forward/reverse pair is taken from the function's own
    `candidate_forward_reverse_pairs` (which only exist when a stable k>=2 clustering is found)."""
    res = compute_adaptive_cluster_stereotypy(np.asarray(ranks, float), np.asarray(bools, bool),
                                              list(names), use_masked_features=True)
    pairs = res.get("candidate_forward_reverse_pairs") or []
    fr = min((p["spearman_r"] for p in pairs), default=None)   # most-negative reversed pair, if any
    # §6.1: the pair annotation is computed from WHATEVER clustering (incl. the unstable fallback),
    # and KMeans centroids are always anti-correlated -> noise gets a "pair". It only COUNTS when the
    # k=2 clustering is AMI-STABLE. Gate on stable_k==2.
    stable_pair = bool(pairs) and (res.get("stable_k") == 2)
    return dict(stable_k=res.get("stable_k"), chosen_k=res.get("chosen_k"),
                chosen_reason=res.get("chosen_reason"), has_fwd_rev_pair=stable_pair,
                fwd_rev_spearman=(None if fr is None else round(float(fr), 3)), _res=res)


def prefix_stereotypy(ranks, bools, names, prefix_points=(10, 20, 40, 80)):
    """Accumulation curve: does a STABLE k=2 + a forward/reverse pair appear as the FIRST N events
    accumulate? Reports (stable_k, has_fwd_rev_pair, fwd_rev_spearman) at each N (+ the full set)."""
    n_ev = np.asarray(ranks).shape[1]
    pts = sorted({min(N, n_ev) for N in prefix_points} | {n_ev})
    out = []
    for N in pts:
        s = _stereotypy(np.asarray(ranks)[:, :N], np.asarray(bools)[:, :N], names)
        out.append(dict(n=int(N), stable_k=s["stable_k"], has_fwd_rev_pair=s["has_fwd_rev_pair"],
                        fwd_rev_spearman=s["fwd_rev_spearman"]))
    return out


def reproducibility(ev):
    """Canonical split-half + odd-even cross-time reproducibility (Hungarian template match) on the
    full set, at the stereotypy's chosen_k. Returns the grade + per-split forward_reverse_reproduced.
    `ev` = load_subject_propagation_events(record_dir)."""
    ranks = np.asarray(ev["ranks"], float); bools = np.asarray(ev["bools"], bool)
    s = _stereotypy(ranks, bools, list(ev["channel_names"]))
    res = s["_res"]; ck = res.get("chosen_k"); labels = res.get("labels")
    if ck is None or labels is None:
        return dict(error="no_clustering", stable_k=s["stable_k"])
    vidx = _valid_event_indices(bools, min_participating=3)
    rep = compute_time_split_reproducibility(
        ranks, bools, np.asarray(ev["event_abs_times"], float), np.asarray(ev["block_ids"]),
        int(ck), np.asarray(labels, int), vidx, use_masked_features=True)
    grade = rep.get("reproducibility_grade") or rep.get("grade")
    splits = rep.get("splits") or {}
    fr = {name: (splits.get(name, {}) or {}).get("forward_reverse_reproduced")
          for name in ("first_half_second_half", "odd_even_block")}
    return dict(stable_k=s["stable_k"], chosen_k=int(ck), grade=grade, forward_reverse_reproduced=fr)


def _stratified_readability(readout):
    strat = {"central(<1)": [0, 0], "mid(1-4.5)": [0, 0], "edge(>4.5)": [0, 0]}
    for e in readout.get("events", []):
        nuc = e.get("nucleation") or {}; s = nuc.get("s_nuc")
        if s is None:
            continue
        key = "central(<1)" if abs(s) <= S_OFF else ("mid(1-4.5)" if abs(s) <= S_EDGE else "edge(>4.5)")
        strat[key][1] += 1
        if e["sign"] in (1, -1) and e["axis_err"] is not None and e["axis_err"] < AX_CLEAN:
            strat[key][0] += 1
    return {k: dict(readable=v[0], total=v[1], frac=(round(v[0] / v[1], 3) if v[1] else None))
            for k, v in strat.items()}


def main(tag, base):
    rec_dir = Path(base) / "record" / tag
    if not list(rec_dir.glob("*_lagPat_withFreqCent.npz")):
        print(f"no record for {tag}"); return
    ev = load_subject_propagation_events(rec_dir)
    ranks, bools, names = np.asarray(ev["ranks"], float), np.asarray(ev["bools"], bool), list(ev["channel_names"])
    readout = json.load(open(os.path.join(base, f"readout_{tag}.json")))
    t0p = os.path.join(base, f"t0_gate_{tag}.json")
    t0 = json.load(open(t0p))["hotspot_degeneracy"]["verdict"] if os.path.exists(t0p) else "no_gate"
    res = dict(tag=tag, n_returned_events=int(ranks.shape[1]), t0_verdict=t0,
               config={k: readout["config"].get(k) for k in ("T", "core_mean", "core_std", "core_r",
                                                              "pitch", "k_dir", "patch_elongation")},
               prefix_stereotypy=prefix_stereotypy(ranks, bools, names),
               reproducibility=reproducibility(ev),
               stratified_readability=_stratified_readability(readout))
    out = os.path.join(base, f"stage4_long_template_{tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2, ensure_ascii=False)); print(f"\n[wrote] {out}")
    return res


if __name__ == "__main__":
    b = "results/topic4_sef_hfo/observation_layer/stage4_search"
    for t in (sys.argv[1:] or ["stage4_long_s3"]):
        main(t, b)
