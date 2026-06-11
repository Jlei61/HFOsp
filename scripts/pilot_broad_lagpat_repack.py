#!/usr/bin/env python3
"""PILOT (2 subjects): broaden the lagPat channel pool to test whether interictal
propagation structure extends beyond the rate/SOZ-restricted core.

Re-packs group events with a permissive channel pick (top-N by event count)
instead of the default mean+k*std, writing a parallel broad lagPat. Channel
selection is the ONLY change; the legacy-faithful centroid/rank code path is
reused verbatim via run_yuquan_lagpat_backfill.run_subject.

Subjects: zhaojinrui (4-ch core, documented F6-7->C/D/K spread),
          chenziyang (10-ch core, onset E/D + multi-region spread).

Output: results/lagpat_broad_pilot/<subject>/  (dry-run; never touches raw tree)

This script ONLY produces the broad lagPat. Reproducibility + anatomical-match
analysis (the PASS/FAIL gates) is a separate step.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.run_yuquan_lagpat_backfill as bf  # noqa: E402

import json  # noqa: E402
BROAD_PICK_K = -2.0   # permissive: lets ~all alive channels pass, then cap by top_n
BROAD_TOP_N = 20      # static broad pool target (~20 highest-count channels)
HARD_FLOOR = 20       # dynamic mode: every subject gets >= this many channels
MARGIN = 15           # dynamic mode: and >= narrow_n + MARGIN (guarantees expansion)
DEFAULT_OUT = REPO / "results" / "lagpat_broad"
NARROW_DIR = REPO / "results" / "interictal_propagation_masked" / "per_subject"
DYNAMIC = False

_orig = bf.resolve_subject_pack_params

def _narrow_n(subject: str):
    f = NARROW_DIR / f"yuquan_{subject}.json"
    if f.exists():
        return len(json.load(open(f))["channel_names"])
    return None

def _patched(subject: str):
    p = dict(_orig(subject))
    p["pick_k"] = BROAD_PICK_K
    if DYNAMIC:
        nn = _narrow_n(subject)
        p["pack_top_n"] = max(HARD_FLOOR, (nn + MARGIN) if nn else HARD_FLOOR)
    else:
        p["pack_top_n"] = BROAD_TOP_N
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", action="append", required=True)
    ap.add_argument("--records", type=str, default=None,
                    help="comma-separated record stems (smoke test); default=all")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--dynamic", action="store_true",
                    help="per-subject top_n = max(%d, narrow_n + %d)" % (HARD_FLOOR, MARGIN))
    args = ap.parse_args()
    only = tuple(args.records.split(",")) if args.records else None
    OUT = Path(args.out_dir)
    global DYNAMIC
    DYNAMIC = args.dynamic

    bf.resolve_subject_pack_params = _patched  # inject broad pick (no shared-config edit)
    OUT.mkdir(parents=True, exist_ok=True)
    for subj in args.subject:
        tn = _patched(subj)["pack_top_n"]
        print(f"=== broad re-pack: {subj} (pick_k={BROAD_PICK_K}, top_n={tn}"
              f"{' dynamic' if DYNAMIC else ''}) ===", flush=True)
        summary, manifest = bf.run_subject(subj, dry_run_out_dir=OUT, only_records=only)
        n_ok = sum(1 for r in summary.get("records", []) if r.get("wrote_lagpat"))
        print(f"  {subj}: {n_ok} records packed -> {OUT/subj}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
