#!/usr/bin/env python3
"""Validate broad-template endpoint GEOMETRY under TWO definitions (user
2026-06-07): do they agree, and do both place endpoints beyond the SOZ?

  def-a  rank endpoints  = top-3 source ∪ top-3 sink by propagation rank
                           (dominant broad cluster).
  def-b  rank-displacement swap-k nodes = top-k ∪ bottom-k of the forward/reverse
                           broad template pair at decision_k (PR-6 swap_sweep;
                           src.rank_displacement). Carries swap_class/p_fw.

Per subject (masked features): both endpoint sets, Jaccard(a,b) overlap (geometry
robust to definition iff high), and SOZ inside/outside for each. Run on any broad
dir (default results/lagpat_broad; pass --broad-dir for the dynamic-top_n dir).
"""
from __future__ import annotations
import sys, json, re, csv, glob, os, argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.interictal_propagation import (  # noqa: E402
    load_subject_propagation_events, compute_adaptive_cluster_stereotypy)
from src.rank_displacement import compute_swap_score_sweep, derive_swap_endpoint  # noqa: E402

SOZ = json.load(open(REPO / "results" / "yuquan_soz_core_channels.json"))
ENDPOINT_K = 3

def base(ch):
    m = re.match(r"([A-Z]+'?)(\d+)", str(ch).replace("’", "'"))
    if not m:
        return {str(ch)}
    el, n = m.group(1), int(m.group(2))
    return {f"{el}{n}", f"{el}{n+1}"}

def in_soz(ch, soz):
    return bool(base(ch) & set(soz))

def two_clusters(ac):
    cl = ac.get("clusters")
    if isinstance(cl, dict):
        cl = list(cl.values())
    if not isinstance(cl, list) or len(cl) < 2:
        return None
    cl = sorted(cl, key=lambda c: c.get("n_events", 0), reverse=True)
    return cl[0], cl[1]

def rank_endpoints(cluster, ch):
    tr = cluster.get("template_rank"); vm = cluster.get("template_valid_mask")
    pairs = [(r, c) for r, c, m in zip(tr, ch, (vm or [True] * len(ch)))
             if m and r is not None]
    pairs.sort()
    return set([c for _, c in pairs[:ENDPOINT_K]] + [c for _, c in pairs[-ENDPOINT_K:]])

def swap_endpoints(ca, cb, ch):
    ra = np.asarray(ca["template_rank"], float)
    rb = np.asarray(cb["template_rank"], float)
    vma = np.asarray(ca.get("template_valid_mask", [True] * len(ch)), bool)
    vmb = np.asarray(cb.get("template_valid_mask", [True] * len(ch)), bool)
    sweep = compute_swap_score_sweep(ra, rb, vma, vmb, n_perm=500, seed=0)
    dk = sweep.get("decision_k")
    joint = vma & vmb
    n_valid = int(joint.sum())
    # dk saturates at floor(n_valid/2) when pool too small -> def-b degenerate (=all)
    saturated = bool(dk is not None and int(dk) >= (n_valid // 2))
    out = dict(swap_class=sweep.get("swap_class"), decision_k=dk, n_valid=n_valid,
               dk_saturated=saturated,
               p_fw=round(float(sweep.get("p_fw", float("nan"))), 4),
               T_obs=round(float(sweep.get("T_obs", float("nan"))), 3))
    if not dk:
        out["endpoints"] = set(); out["note"] = "no_decision_k"
        return out
    vnames = [c for c, m in zip(ch, joint) if m]
    ra_v = ra[joint]
    if 2 * int(dk) > len(vnames):
        out["endpoints"] = set(); out["note"] = "dk_too_large"
        return out
    out["endpoints"] = set(derive_swap_endpoint(vnames, ra_v, int(dk)))
    out["note"] = ""
    return out

def jaccard(a, b):
    if not a and not b:
        return None
    return round(len(a & b) / len(a | b), 3)

def analyze(subj, broad_dir):
    d = load_subject_propagation_events(broad_dir / subj)
    ch = [str(c) for c in d["channel_names"]]
    ranks = np.asarray(d["ranks"], float)
    bools = np.asarray(d["bools"]) > 0
    ac = compute_adaptive_cluster_stereotypy(ranks, bools, ch, use_masked_features=True)
    cc = two_clusters(ac)
    if cc is None:
        return dict(subject=subj, status="lt2_clusters")
    ca, cb = cc
    ep_a = rank_endpoints(ca, ch)              # def-a
    sw = swap_endpoints(ca, cb, ch)            # def-b
    ep_b = sw["endpoints"]
    soz = SOZ.get(subj, [])
    return dict(
        subject=subj, status="ok", stable_k=int(ac.get("stable_k", -1)),
        n_broad=len(ch), swap_class=sw["swap_class"], decision_k=sw["decision_k"],
        n_valid=sw.get("n_valid"), dk_saturated=sw.get("dk_saturated"), p_fw=sw["p_fw"],
        defA_rank_endpoints=";".join(sorted(ep_a)),
        defB_swap_nodes=";".join(sorted(ep_b)),
        jaccard_AB=jaccard(ep_a, ep_b),
        n_overlap=len(ep_a & ep_b),
        defA_in_soz=sum(in_soz(c, soz) for c in ep_a), defA_n=len(ep_a),
        defB_in_soz=sum(in_soz(c, soz) for c in ep_b), defB_n=len(ep_b),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broad-dir", default=str(REPO / "results" / "lagpat_broad"))
    ap.add_argument("subjects", nargs="?", default=None)
    args = ap.parse_args()
    bdir = Path(args.broad_dir)
    only = args.subjects.split(",") if args.subjects else None
    rows = []
    subj_dirs = sorted(d for d in glob.glob(str(bdir / "*")) if os.path.isdir(d)
                       and glob.glob(os.path.join(d, "*_lagPat.npz")))
    for sd in subj_dirs:
        subj = os.path.basename(sd)
        if (only and subj not in only) or subj not in SOZ:
            continue
        try:
            r = analyze(subj, bdir)
        except Exception as e:
            print(f"[err] {subj}: {str(e)[:90]}"); continue
        rows.append(r)
        if r["status"] == "ok":
            print(f"{subj:<14} swap={str(r['swap_class']):<10} dk={r['decision_k']} "
                  f"jaccard(A,B)={r['jaccard_AB']} (overlap {r['n_overlap']}) "
                  f"A_inSOZ={r['defA_in_soz']}/{r['defA_n']} B_inSOZ={r['defB_in_soz']}/{r['defB_n']}")
    if rows:
        cols = [c for c in rows[0].keys()]
        for r in rows:
            for c in cols:
                r.setdefault(c, "")
        out = bdir / "geometry_validation.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {out} ({len(rows)} subjects)")

if __name__ == "__main__":
    main()
