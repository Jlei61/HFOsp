#!/usr/bin/env python3
"""Broad-lagPat KMeans/stable-template + SOZ inside/outside analysis (user plan
2026-06-07 steps 4-5). Run on eligible broad subjects after re-pack + QC.

Per subject (masked features throughout — never phantom path):
  - re-run compute_adaptive_cluster_stereotypy on broad lagPat -> stable_k +
    per-cluster template_rank (+ valid_mask).
  - endpoint = source (k lowest valid rank) ∪ sink (k highest valid rank) of the
    dominant cluster.
  - SOZ readouts: #endpoint in-SOZ vs out-SOZ; added channels (broad-narrow) in
    documented clinical network; broad-vs-narrow endpoint SOZ-coverage
    (does broad extend all-in-SOZ endpoints into SOZ-core + peri/extra-SOZ?).
  - rate (SENSITIVITY only): Spearman(channel event-rate, median propagation
    position) — high = broad mainly extended high-firing network (still valuable).

Narrow vs broad = subject-level only (event sets differ). Outputs
results/lagpat_broad/broad_template_soz.{csv,json}.
"""
from __future__ import annotations
import sys, json, re, csv, glob, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.interictal_propagation import (  # noqa: E402
    load_subject_propagation_events, compute_adaptive_cluster_stereotypy)

BROAD = REPO / "results" / "lagpat_broad"
NARROW = REPO / "results" / "interictal_propagation_masked" / "per_subject"
SOZ = json.load(open(REPO / "results" / "yuquan_soz_core_channels.json"))
NETS = json.load(open(BROAD / "yuquan_clinical_networks.json")) if (BROAD / "yuquan_clinical_networks.json").exists() else {}
ENDPOINT_K = 3

def base(ch):
    m = re.match(r"([A-Z]+'?)(\d+)", str(ch).replace("’", "'"))
    if not m:
        return {str(ch)}
    el, n = m.group(1), int(m.group(2))
    return {f"{el}{n}", f"{el}{n+1}"}

def in_set(ch, s):
    return bool(base(ch) & set(s))

def dominant_cluster(ac):
    cl = ac.get("clusters")
    if isinstance(cl, list) and cl:
        return max(cl, key=lambda c: c.get("n_events", 0))
    if isinstance(cl, dict) and cl:
        return max(cl.values(), key=lambda c: c.get("n_events", 0))
    return None

def endpoints(dom, ch):
    tr = dom.get("template_rank"); vm = dom.get("template_valid_mask")
    pairs = [(r, c) for r, c, m in zip(tr, ch, (vm or [True]*len(ch)))
             if m and r is not None]
    pairs.sort()
    src = [c for _, c in pairs[:ENDPOINT_K]]
    sink = [c for _, c in pairs[-ENDPOINT_K:]]
    return src, sink

def median_position(ranks, bools):
    """per-channel median rank-fraction over participating events."""
    pos = np.full(ranks.shape[0], np.nan)
    for i in range(ranks.shape[0]):
        ev = bools[i] > 0
        if ev.sum() < 5:
            continue
        vals = []
        for e in np.where(ev)[0]:
            part = bools[:, e] > 0
            npart = int(part.sum())
            if npart < 2:
                continue
            r = ranks[part, e]
            vals.append((ranks[i, e] - r.min()) / (npart - 1))
        if vals:
            pos[i] = np.median(vals)
    return pos

def analyze(subj):
    d = load_subject_propagation_events(BROAD / subj)
    ch = [str(c) for c in d["channel_names"]]
    ranks = np.asarray(d["ranks"], float)
    bools = np.asarray(d["bools"]) > 0
    ac = compute_adaptive_cluster_stereotypy(ranks, bools, ch, use_masked_features=True)
    dom = dominant_cluster(ac)
    if dom is None:
        return dict(subject=subj, status="no_cluster"), None
    src, sink = endpoints(dom, ch)
    ep = list(dict.fromkeys(src + sink))
    soz = SOZ.get(subj, [])
    net = NETS.get(subj, {}).get("network", [])
    narrow = None
    nf = NARROW / f"yuquan_{subj}.json"
    if nf.exists():
        narrow = {str(c) for c in json.load(open(nf))["channel_names"]}
    added = sorted(set(ch) - narrow) if narrow else sorted(ch)
    # rate sensitivity
    rate = bools.sum(axis=1).astype(float)  # participation count as rate proxy
    pos = median_position(ranks, bools)
    ok = ~np.isnan(pos)
    rho = float(spearmanr(rate[ok], pos[ok]).statistic) if ok.sum() > 3 else None
    s = dict(
        subject=subj, status="ok",
        stable_k_broad=int(ac.get("stable_k", -1)),
        n_broad_ch=len(ch),
        endpoint_channels=";".join(ep),
        n_endpoint_in_soz=sum(in_set(c, soz) for c in ep),
        n_endpoint_out_soz=sum(not in_set(c, soz) for c in ep),
        endpoint_in_clinical_net=sum(in_set(c, net) for c in ep),
        n_added=len(added),
        n_added_in_soz=sum(in_set(c, soz) for c in added),
        n_added_out_soz=sum(not in_set(c, soz) for c in added),
        n_added_in_clinical_net=sum(in_set(c, net) for c in added),
        rate_position_spearman=round(rho, 3) if rho is not None else None,
        n_soz_channels=len(soz),
    )
    return s, ac

def main():
    rows = []
    full = {}
    subj_dirs = sorted(d for d in glob.glob(str(BROAD / "*")) if os.path.isdir(d)
                       and glob.glob(os.path.join(d, "*_lagPat.npz")))
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    for sd in subj_dirs:
        subj = os.path.basename(sd)
        if only and subj not in only:
            continue
        if subj not in SOZ:
            print(f"[skip] {subj}: no SOZ labels")
            continue
        try:
            s, ac = analyze(subj)
        except Exception as e:
            print(f"[err] {subj}: {str(e)[:80]}")
            continue
        rows.append(s)
        if ac is not None:
            full[subj] = {k: ac.get(k) for k in ("stable_k", "chosen_reason")}
        print(f"{subj:<14} k={s.get('stable_k_broad')} ep_inSOZ={s.get('n_endpoint_in_soz')}/{len(s.get('endpoint_channels','').split(';'))} "
              f"added_inSOZ={s.get('n_added_in_soz')}/{s.get('n_added')} added_inNet={s.get('n_added_in_clinical_net')} "
              f"rho(rate,pos)={s.get('rate_position_spearman')}")
    if rows:
        cols = list(rows[0].keys())
        with open(BROAD / "broad_template_soz.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        json.dump(rows, open(BROAD / "broad_template_soz.json", "w"), indent=2)
        print(f"\nwrote {BROAD/'broad_template_soz.csv'} ({len(rows)} subjects)")

if __name__ == "__main__":
    main()
