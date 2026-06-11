#!/usr/bin/env python3
"""Epilepsiae broad lagPat re-pack — multi-strategy version.

Three channel-selection strategies (all from full _refineGpu.npz universe):
  --top-n N        top-N by event count (default 20; already running)
  --pick-k K       mean + K*std threshold (std-based; K<1.0 -> more channels
                   than legacy K=1.0; K=0.0 = mean; K=-1.0 = below mean)
  --dynamic        per-subject max(20, narrow_n + 15), mirrors Yuquan --dynamic

Each strategy writes to its own output dir so runs never overwrite each other:
  top-n 20   -> results/lagpat_broad_epilepsiae/          (original)
  top-n 40   -> results/lagpat_broad_epilepsiae_topn40/
  pick-k 0.5 -> results/lagpat_broad_epilepsiae_k05/
  pick-k 0.0 -> results/lagpat_broad_epilepsiae_k0/
  pick-k -1  -> results/lagpat_broad_epilepsiae_km1/
  dynamic    -> results/lagpat_broad_epilepsiae_dyn/
"""
from __future__ import annotations
import sys, argparse, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import scripts.run_epilepsiae_lagpat_backfill as ebf  # noqa: E402

NARROW_DIR = REPO / "results" / "interictal_propagation_masked" / "per_subject"

COHORT = [
    "139","384","583","635","1146",
    "1073","1077","1084","1096","1125",
    "1150","253","442","548","590",
    "620","818","922","958","916",
]

def _narrow_n(subject: str) -> int | None:
    f = NARROW_DIR / f"epilepsiae_{subject}.json"
    if f.exists():
        return len(json.load(open(f))["channel_names"])
    return None

def _make_top_n_selector(top_n: int):
    def _sel(subject: str, strategy: str = "mean_plus_std"):
        z = np.load(ebf._refine_path_for_subject(subject), allow_pickle=True)
        names = [str(c) for c in z["chns_names"]]
        ec = np.asarray(z["events_count"], dtype=float)
        idx = sorted(np.argsort(ec)[::-1][:top_n].tolist())
        return tuple(names[i] for i in idx)
    return _sel

def _make_pick_k_selector(k: float):
    def _sel(subject: str, strategy: str = "mean_plus_std"):
        z = np.load(ebf._refine_path_for_subject(subject), allow_pickle=True)
        names = np.asarray([str(c) for c in z["chns_names"]])
        ec = np.asarray(z["events_count"], dtype=float)
        thr = ec.mean() + k * ec.std()
        idx = np.where(ec > thr)[0]
        if len(idx) < 2:          # fallback: at least top-5
            idx = np.argsort(ec)[::-1][:5]
        return tuple(names[idx].tolist())
    return _sel

def _make_dynamic_selector(floor: int = 20, margin: int = 15):
    def _sel(subject: str, strategy: str = "mean_plus_std"):
        nn = _narrow_n(subject)
        top_n = max(floor, (nn + margin) if nn else floor)
        z = np.load(ebf._refine_path_for_subject(subject), allow_pickle=True)
        names = [str(c) for c in z["chns_names"]]
        ec = np.asarray(z["events_count"], dtype=float)
        idx = sorted(np.argsort(ec)[::-1][:top_n].tolist())
        return tuple(names[i] for i in idx)
    return _sel

def _default_out(args) -> Path:
    if args.top_n is not None:
        if args.top_n == 20:
            return REPO / "results" / "lagpat_broad_epilepsiae"
        return REPO / "results" / f"lagpat_broad_epilepsiae_topn{args.top_n}"
    if args.pick_k is not None:
        tag = str(args.pick_k).replace("-","m").replace(".","")
        return REPO / "results" / f"lagpat_broad_epilepsiae_k{tag}"
    if args.dynamic:
        return REPO / "results" / "lagpat_broad_epilepsiae_dyn"
    return REPO / "results" / "lagpat_broad_epilepsiae"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", action="append", default=None,
                    help="subject id(s); default=all 20 cohort")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--top-n", type=int, default=None,
                     help="top-N by event count (e.g. 20, 40)")
    grp.add_argument("--pick-k", type=float, default=None,
                     help="mean + K*std threshold (e.g. 0.5, 0.0, -1.0)")
    grp.add_argument("--dynamic", action="store_true", default=False,
                     help="per-subject max(20, narrow_n+15)")
    args = ap.parse_args()

    subjects = args.subject if args.subject else COHORT
    out = Path(args.out_dir) if args.out_dir else _default_out(args)

    if args.top_n is not None:
        ebf.load_refine_chns_for_subject = _make_top_n_selector(args.top_n)
        tag = f"top_n={args.top_n}"
    elif args.pick_k is not None:
        ebf.load_refine_chns_for_subject = _make_pick_k_selector(args.pick_k)
        tag = f"pick_k={args.pick_k}"
    else:
        ebf.load_refine_chns_for_subject = _make_dynamic_selector()
        tag = "dynamic(max(20,narrow+15))"

    ebf.OUTPUT_ROOT = out
    ebf.NEW_GPU_ROOT = REPO / "results" / "hfo_detection"
    out.mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        print(f"=== epilepsiae re-pack: {subj} ({tag}) -> {out.name} ===", flush=True)
        if args.smoke:
            recs = ebf._discover_records(subj)
            if not recs:
                print(f"  {subj}: no records"); continue
            stem = recs[0]["stem"]
            r = ebf.process_one_record(subj, stem, force=True)
            print(f"  smoke {subj}/{stem}: n_ch={r.get('n_channels')} n_ev={r.get('n_events')}")
        else:
            try:
                log = ebf.process_subject(subj, force=True)
                ebf._print_subject_summary(subj, log)
            except Exception as e:
                print(f"  [ERROR] {subj}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        sys.stdout.flush()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
