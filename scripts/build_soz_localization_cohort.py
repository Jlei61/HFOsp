#!/usr/bin/env python3
"""Task 1: freeze the SEF-HFO SOZ-localization cohort + channel universe + timezone.

Three-source intersection (SOZ truth ∩ per-channel rate ∩ masked propagation geometry),
subject quality gate (total_hours>=12), per-subject channel universe U with the montage
bridge, SOZ coverage, comparison-A eligibility. Writes the single source of truth
results/topic4_sef_hfo/soz_localization/cohort.json. Pure read-only; no detection re-run.

Contract: docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md (v3)
          docs/archive/topic4/sef_hfo/channel_universe_montage_diagnostic_2026-06-06.md
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sef_hfo_soz_localization import build_cohort, MIN_HOURS, MIN_CH_EVENTS

RESULTS = _ROOT / "results"
OUT_DIR = RESULTS / "topic4_sef_hfo" / "soz_localization"
SOZ_FILES = {"yuquan": RESULTS / "yuquan_soz_core_channels.json",
             "epilepsiae": RESULTS / "epilepsiae_soz_core_channels.json"}
PERCHANNEL = RESULTS / "spatial_modulation" / "per_channel_metrics"
GEOM_DIR = RESULTS / "interictal_propagation_masked" / "rank_displacement" / "per_subject"
YUQUAN_ARTIFACT_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")

# D-daynight (v3): time-of-day proxy timezone, frozen here.
DAYNIGHT_TZ = {"epilepsiae": "Europe/Berlin", "yuquan": "Asia/Shanghai"}


def _load_geom_pair(path: Path):
    """Return the primary geometry pair (pairs[0]) or None if unavailable."""
    d = json.loads(path.read_text())
    pairs = d.get("pairs") or []
    return pairs[0] if pairs else None


def _assemble_candidates() -> list:
    """One candidate per SOZ-truth subject; attach rate + geometry if present."""
    candidates = []
    for ds, soz_path in SOZ_FILES.items():
        soz = json.loads(soz_path.read_text())
        for subj, core in sorted(soz.items()):
            rf = PERCHANNEL / ds / f"{subj}_perchannel.json"
            gf = GEOM_DIR / f"{ds}_{subj}.json"
            cm, total_hours = None, None
            if rf.exists():
                rd = json.loads(rf.read_text())
                cm = rd.get("channel_metrics")
                total_hours = rd.get("total_hours")
            geom_pair = _load_geom_pair(gf) if gf.exists() else None
            candidates.append({
                "dataset": ds, "subject": subj, "total_hours": total_hours,
                "soz_core": core, "channel_metrics": cm, "geom_pair": geom_pair,
            })
    return candidates


def _print_yuquan_packedtimes_span():
    """Sanity print: first/last event time of one yuquan packedTimes (for the tz record)."""
    cands = sorted(glob.glob(str(YUQUAN_ARTIFACT_ROOT / "chengshuai" / "*_packedTimes_withFreqCent.npy")))
    if not cands:
        print("  (no yuquan packedTimes found for tz sanity print)")
        return
    arr = np.load(cands[0], allow_pickle=True)
    flat = np.concatenate([np.ravel(a) for a in arr if np.size(a)]) if arr.dtype == object else np.ravel(arr)
    flat = flat[np.isfinite(flat)]
    print(f"  yuquan packedTimes sample: {os.path.basename(cands[0])}  "
          f"min={flat.min():.1f}s max={flat.max():.1f}s span={(flat.max()-flat.min())/3600:.1f}h")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = _assemble_candidates()
    cohort = build_cohort(candidates, min_hours=MIN_HOURS, min_ch_events=MIN_CH_EVENTS)
    kept, excl = cohort["kept"], cohort["excluded"]

    cohort["meta"]["daynight_tz"] = DAYNIGHT_TZ
    cohort["meta"]["raw_intersection"] = sum(
        1 for c in candidates if c["soz_core"] and c["channel_metrics"] and c["geom_pair"] is not None
    )
    cohort["meta"]["bridge"] = ("single: exact name; bipolar: first-contact X->X-next unique "
                                "(verified lagPat convention), else missing")
    cohort["meta"]["claim"] = ("within HFO-active region, propagation ORDER more sampling-stable "
                               "than firing COUNT (narrowed v3)")

    out_path = OUT_DIR / "cohort.json"
    out_path.write_text(json.dumps(cohort, indent=2, ensure_ascii=False))

    # ----- report -----
    by_ds = lambda pred: {ds: sum(1 for r in kept if r["dataset"] == ds and pred(r))
                          for ds in ("yuquan", "epilepsiae")}
    n_elig = by_ds(lambda r: r["comparison_a_eligible"])
    raw = cohort["meta"]["raw_intersection"]
    print(f"\n=== SOZ-localization cohort (Task 1) ===")
    print(f"raw 3-source intersection: {raw}   kept (total_hours>={MIN_HOURS}): {len(kept)}   "
          f"comparison-A eligible: {sum(r['comparison_a_eligible'] for r in kept)}")
    print(f"  kept by dataset: yuquan={sum(r['dataset']=='yuquan' for r in kept)}, "
          f"epilepsiae={sum(r['dataset']=='epilepsiae' for r in kept)}")
    print(f"  A-eligible by dataset: {n_elig}")
    for ds in ("yuquan", "epilepsiae"):
        covs = [r["soz_coverage"] for r in kept if r["dataset"] == ds]
        if covs:
            print(f"  median joint-U SOZ coverage ({ds}): {np.median(covs):.2f}")
    print(f"  excluded: {[(e['subject'], e['reason']) for e in excl]}")
    print()
    print(f"{'subject':<13}{'ds':<5}{'hrs':>5}{'montage':>8}{'|U|':>4}{'nSOZ':>5}{'soz∩U':>6}{'covU':>6}{'A-elig':>7}")
    for r in kept:
        print(f"{r['subject']:<13}{r['dataset']:<5}{r['total_hours']:>5.0f}{r['montage']:>8}"
              f"{r['n_universe']:>4}{r['n_soz_core']:>5}{r['n_soz_in_universe']:>6}"
              f"{r['soz_coverage']:>6.2f}{'Y' if r['comparison_a_eligible'] else 'no':>7}")
    print()
    print(f"frozen daynight_tz: {DAYNIGHT_TZ}")
    _print_yuquan_packedtimes_span()
    print(f"\nwrote {out_path}")

    # Cohort expectation (report, do not hard-fail on a single magic number)
    assert raw == 29, f"expected raw intersection 29, got {raw}"
    assert len(kept) == 28, f"expected 28 kept after hours gate, got {len(kept)}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
