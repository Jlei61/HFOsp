"""Topic5 A-line §3.2 — v2 window-cache attrition audit (run BEFORE the window sweep stats).

The v2 trace cache extends the pre window to >=130s (to cover the distal [-120,-90] negative
control). A longer pre can push a seizure before its recording-block start, so the v2 cache may
hold FEWER seizures than the original 0-10s cache (observed: epilepsiae_1077 sz5 load-skipped, 8
vs 9). We must NOT keep claiming "354 seizures" unchanged. This audit writes v2_cache_attrition.csv:
per subject expected (original cache eligible) / cached (v2 has bb_zt) / missing idx, and parses the
build log for the skip reason. The window sweep's patient unit stays 18; seizure-precision N is
reported from ACTUAL cached + per-window finite counts, not the original 354.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ORIG = Path("results/topic5_ictal_recruitment/t0_feature_cache")
V2 = Path("results/topic5_ictal_recruitment/t0_feature_cache_v2_windows")
BUILD_LOG = Path("results/run_logs/t0_v2_windows_build.log")
OUT = Path("results/topic5_ictal_recruitment/axis_alignment/v2_cache_attrition.csv")


def _skip_reasons():
    """(sid, idx) -> reason text parsed from the v2 build log skip lines."""
    reasons = {}
    if not BUILD_LOG.exists():
        return reasons
    for ln in BUILD_LOG.read_text(errors="ignore").splitlines():
        m = re.search(r"\[(\S+) sz(\d+)\].*(skip|error|mismatch)(.*)", ln, re.I)
        if m:
            reasons[(m.group(1), int(m.group(2)))] = ln.strip()
    return reasons


def main():
    reasons = _skip_reasons()
    rows = [("subject", "expected", "cached", "missing_idx", "reason")]
    tot_exp = tot_cached = 0
    for orig_meta in sorted(ORIG.glob("*.json")):
        sid = orig_meta.stem
        if not sid.startswith("epilepsiae"):
            continue
        expected = list(json.load(open(orig_meta)).get("eligible_idxs", []))
        v2_npz = V2 / f"{sid}.npz"
        if not v2_npz.exists():
            rows.append((sid, len(expected), 0, ";".join(map(str, expected)), "v2 npz not built"))
            tot_exp += len(expected)
            continue
        data = np.load(v2_npz, allow_pickle=True)
        cached = [i for i in expected if f"bb_zt__{i}" in data.files]
        missing = [i for i in expected if i not in cached]
        reason = " | ".join(reasons.get((sid, i), f"idx{i} absent") for i in missing) if missing else ""
        rows.append((sid, len(expected), len(cached), ";".join(map(str, missing)), reason))
        tot_exp += len(expected)
        tot_cached += len(cached)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"\n=== v2 cache attrition: expected {tot_exp} seizures, cached {tot_cached}, "
          f"missing {tot_exp - tot_cached} ===")
    for sid, exp, cac, miss, rs in rows[1:]:
        if exp != cac:
            print(f"  {sid}: {cac}/{exp} cached  missing=[{miss}]  {rs}")


if __name__ == "__main__":
    main()
