"""Pool spontaneous FORWARD (-end lesion) + REVERSE (+end lesion) records into ONE synthetic
bidirectional subject in the legacy lagPat format, then run the REAL masked PR-2 / PR-2.5 /
rank-displacement to ask: does the real pipeline auto-recover TWO opposite templates from a fixed
connectivity axis with foci at the two endpoints? (reviewer 2026-06-10 step 3-4).

Faithful framing (reviewer 4): the EE connectivity long axis is FIXED; the lesion sits at one
endpoint (forward train) or the other (reverse train); pooling = a subject whose two foci fire at
DIFFERENT times. This is NOT a claim that one network spontaneously alternates (the two-foci-same-
network design was messier — scoped separately). Events are interleaved in time so split-half /
odd-even both see both templates.

Output: a synthetic subject dir (real-loader-readable) + a summary JSON with stable_k,
inter-cluster correlation, candidate forward/reverse pair, time-split reproducibility, and the
rank-displacement swap sweep on the two cluster templates.
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

sys.path.insert(0, os.getcwd())
from src.interictal_propagation import (                                   # noqa: E402
    load_subject_propagation_events, compute_adaptive_cluster_stereotypy,
    compute_time_split_reproducibility, build_cluster_templates, _valid_event_indices)
from src.rank_displacement import compute_swap_score_sweep                 # noqa: E402

REC = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/record"
OUTDIR = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/pooled_bidir"


def _find(tag):
    hits = glob.glob(os.path.join(REC, tag, "*_lagPat_withFreqCent.npz"))
    if not hits:
        raise FileNotFoundError(f"no record for {tag} under {REC}")
    return hits[0]


def _pool(fwd_npz, rev_npz):
    """Concatenate the two on-disk legacy records, interleaving event times so split-half /
    odd-even each contain both templates. Returns (lagPatRank, eventsBool, lagPatRaw, chnNames,
    packedTimes) in the legacy on-disk convention (seconds)."""
    f, r = np.load(fwd_npz, allow_pickle=True), np.load(rev_npz, allow_pickle=True)
    cf = [str(c) for c in f["chnNames"]]; cr = [str(c) for c in r["chnNames"]]
    assert cf == cr, "channel name/montage mismatch between forward and reverse records"
    rank = np.concatenate([f["lagPatRank"], r["lagPatRank"]], axis=1)
    bools = np.concatenate([f["eventsBool"], r["eventsBool"]], axis=1)
    raw = np.concatenate([f["lagPatRaw"], r["lagPatRaw"]], axis=1)
    nf, nr = f["lagPatRank"].shape[1], r["lagPatRank"].shape[1]
    # interleave times: forward at even slots, reverse at odd (units arbitrary, seconds-ish)
    tf = np.arange(nf) * 2.0
    tr = np.arange(nr) * 2.0 + 1.0
    starts = np.concatenate([tf, tr])
    packed = np.column_stack([starts, starts + 0.05])
    return rank, bools, raw, cf, packed, nf, nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", default="oneend_neg_s1")
    ap.add_argument("--rev", default="oneend_pos_s1")
    ap.add_argument("--tag", default="model_spont_bidir")
    a = ap.parse_args()
    sub_dir = os.path.join(OUTDIR, a.tag)
    os.makedirs(sub_dir, exist_ok=True)

    rank, bools, raw, chn, packed, nf, nr = _pool(_find(a.fwd), _find(a.rev))
    base = os.path.join(sub_dir, a.tag)
    np.savez(base + "_lagPat_withFreqCent.npz", lagPatRank=rank, eventsBool=bools,
             lagPatRaw=raw, chnNames=np.array(chn), start_t=0.0)
    np.save(base + "_packedTimes_withFreqCent.npy", packed)
    print(f"[pool] forward={nf} + reverse={nr} = {rank.shape[1]} events, {rank.shape[0]} channels "
          f"-> {sub_dir}", flush=True)

    # --- load via the REAL loader (lagpat 同款) ---
    ev = load_subject_propagation_events(sub_dir)
    R = np.asarray(ev["ranks"], float); B = np.asarray(ev["bools"], bool)
    names = list(ev["channel_names"])
    eat = np.asarray(ev["event_abs_times"], float)
    bid = np.asarray(ev["block_ids"]) if "block_ids" in ev else np.zeros(R.shape[1], int)

    # --- PR-2: masked adaptive cluster ---
    print("[PR-2] compute_adaptive_cluster_stereotypy (masked) ...", flush=True)
    pr2 = compute_adaptive_cluster_stereotypy(R, B, names, use_masked_features=True)
    icc = np.asarray(pr2["inter_cluster_corr_matrix"], float)
    off = icc[np.triu_indices_from(icc, k=1)] if icc.size > 1 else np.array([np.nan])
    labels = np.array(pr2.get("labels", []), dtype=int)
    chosen_k = int(pr2["chosen_k"])

    # --- PR-2.5: masked time-split reproducibility ---
    print("[PR-2.5] compute_time_split_reproducibility (masked) ...", flush=True)
    valid_ev = _valid_event_indices(B, min_participating=3)
    try:
        pr25 = compute_time_split_reproducibility(R, B, eat, bid, chosen_k, labels, valid_ev,
                                                  use_masked_features=True)
    except Exception as e:
        pr25 = {"error": repr(e)}

    # --- rank-displacement: swap sweep on the two cluster templates ---
    swap = {"skipped": "chosen_k != 2"}
    if chosen_k == 2 and labels.size:
        templates = build_cluster_templates(R, B, labels, 2)        # (2, n_ch)
        t0, t1 = templates[0], templates[1]
        vm0 = np.isfinite(t0) & (t0 >= 0); vm1 = np.isfinite(t1) & (t1 >= 0)
        try:
            swap = compute_swap_score_sweep(t0, t1, vm0, vm1, n_perm=1000, seed=0)
        except Exception as e:
            swap = {"error": repr(e)}

    summary = dict(
        inputs=dict(forward=a.fwd, reverse=a.rev, n_forward=nf, n_reverse=nr,
                    n_events=int(R.shape[1]), n_channels=int(R.shape[0])),
        pr2=dict(stable_k=pr2.get("stable_k"), chosen_k=chosen_k,
                 chosen_reason=pr2.get("chosen_reason"),
                 inter_cluster_corr_offdiag=[round(float(x), 3) for x in off],
                 candidate_forward_reverse_pairs=pr2.get("candidate_forward_reverse_pairs"),
                 cluster_sizes=[int((labels == c).sum()) for c in range(chosen_k)] if labels.size else []),
        pr25=dict(forward_reverse_reproduced=_dig_fwd_rev(pr25), raw=_short(pr25)),
        rank_displacement=dict(swap_class=swap.get("swap_class") if isinstance(swap, dict) else None,
                               decision_k=swap.get("decision_k") if isinstance(swap, dict) else None,
                               exit_reason=swap.get("exit_reason") if isinstance(swap, dict) else None,
                               raw=_short(swap)))
    json.dump(summary, open(os.path.join(sub_dir, "masked_pipeline_summary.json"), "w"), indent=2,
              default=lambda o: None)
    print("\n=== MASKED PIPELINE ON POOLED BIDIRECTIONAL RECORD ===")
    print(f"  PR-2   stable_k={pr2.get('stable_k')} chosen_k={chosen_k} "
          f"inter-cluster corr(offdiag)={summary['pr2']['inter_cluster_corr_offdiag']} "
          f"cand_fwd_rev_pairs={bool(pr2.get('candidate_forward_reverse_pairs'))} "
          f"cluster_sizes={summary['pr2']['cluster_sizes']}")
    print(f"  PR-2.5 forward_reverse_reproduced={summary['pr25']['forward_reverse_reproduced']}")
    print(f"  rank-displacement swap_class={summary['rank_displacement']['swap_class']} "
          f"decision_k={summary['rank_displacement']['decision_k']} "
          f"exit={summary['rank_displacement']['exit_reason']}")


def _short(d):
    if not isinstance(d, dict):
        return None
    return {k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in d.items() if isinstance(v, (int, float, str, bool))}


def _dig_fwd_rev(pr25):
    if not isinstance(pr25, dict):
        return None
    sp = pr25.get("splits", {})
    out = {}
    for k, v in (sp.items() if isinstance(sp, dict) else []):
        if isinstance(v, dict) and "forward_reverse_reproduced" in v:
            out[k] = v["forward_reverse_reproduced"]
    return out or pr25.get("forward_reverse_reproduced")


if __name__ == "__main__":
    main()
