#!/usr/bin/env python3
"""B0.3 QA -- parity of the rebuilt early ictal field against the accepted cache.

``results/topic5_ictal_recruitment/ictal_field_long_cache`` is an accepted Topic 5
artifact, but it anchors Epilepsiae windows on the *clinical* onset. This script
separates two questions that must not be conflated:

1. **Is the rebuild faithful?**  Compare my EEG-onset field against the cache's
   own trace re-sliced at the same EEG-onset window. High agreement here means
   channel order, montage, block pointer and normalization all match.
2. **Does the anchor choice matter?**  Compare my EEG-onset field against the
   cache *as published* (clinical anchor). Disagreement here is the size of the
   effect that topic5 caveat 9 warns about, not an error.

Writes ``support/early_field_parity.csv`` and prints a cohort summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
CACHE = MAIN_TREE / "results/topic5_ictal_recruitment/ictal_field_long_cache"
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
DEFAULT_DATA = Path("/data/hfosp_group_event_state_v0_2/agent_b")

WINDOW = (0.0, 10.0)


def _win_mean(trace, relt, a, b):
    m = (relt >= a) & (relt <= b)
    if not m.any():
        return None
    return np.nanmean(trace[:, m], axis=1)


def _rho(a, b):
    from scipy.stats import spearmanr

    if a is None or b is None:
        return float("nan")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args()

    rows = []
    mine_dir = args.data_root / "early_field"
    for jp in sorted(mine_dir.glob("*.json")):
        subject = jp.stem
        cj_path = args.cache / f"{subject}.json"
        if not cj_path.exists():
            rows.append({"subject": subject, "status": "not_in_cache"})
            continue
        mj = json.loads(jp.read_text())
        cj = json.loads(cj_path.read_text())
        mz = np.load(mine_dir / f"{subject}.npz")
        cz = np.load(args.cache / f"{subject}.npz")

        same_order = list(mj["channels"]) == list(cj["channels"])
        id2k = {v.get("seizure_id"): k for k, v in cj["seizure"].items()}

        for i, s in enumerate(mj["seizures"]):
            if s["status"] != "ok":
                continue
            k = id2k.get(s["seizure_id"])
            if k is None:
                rows.append({"subject": subject, "seizure_id": s["seizure_id"],
                             "status": "seizure_not_in_cache",
                             "channel_order_identical": same_order})
                continue
            rel = cj["seizure"][k].get("eeg_onset_rel")
            rel = 0.0 if rel is None else float(rel)
            hz, hr = cz[f"hfa_zt__{k}"], cz[f"hfa_relt__{k}"]
            cache_eeg = _win_mean(hz, hr, rel + WINDOW[0], rel + WINDOW[1])
            cache_pub = _win_mean(hz, hr, WINDOW[0], WINDOW[1])
            mine = mz[f"hfa_field_10s__{i:03d}"]
            rows.append({
                "subject": subject, "seizure_id": s["seizure_id"],
                "status": "ok" if same_order else "channel_order_differs",
                "channel_order_identical": same_order,
                "eeg_minus_clinical_onset_sec": round(rel, 3),
                "rho_vs_cache_same_anchor": round(_rho(cache_eeg, mine), 4),
                "rho_vs_cache_as_published": round(_rho(cache_pub, mine), 4),
                "max_z_mine": round(float(np.nanmax(mine)), 3),
            })

    keys = sorted({k for r in rows for k in r})
    p = args.out_root / "support/early_field_parity.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.rename(p)

    ok = [r for r in rows if r.get("status") == "ok"]
    same = np.array([r["rho_vs_cache_same_anchor"] for r in ok], float)
    pub = np.array([r["rho_vs_cache_as_published"] for r in ok], float)
    off = np.abs(np.array([r["eeg_minus_clinical_onset_sec"] for r in ok], float))
    print(f"wrote {p}")
    print(f"\ncomparable seizures: {len(ok)} over {len(set(r['subject'] for r in ok))} subjects")
    print(f"channel order identical everywhere: {all(r['channel_order_identical'] for r in ok)}")
    if len(ok):
        print(f"\n  rho vs cache, SAME anchor      : median {np.nanmedian(same):+.4f}  "
              f"min {np.nanmin(same):+.4f}   (faithfulness of the rebuild)")
        print(f"  rho vs cache, AS PUBLISHED     : median {np.nanmedian(pub):+.4f}  "
              f"min {np.nanmin(pub):+.4f}   (cost of the clinical anchor)")
        print(f"\n  |EEG - clinical onset| seconds : median {np.nanmedian(off):.2f}  "
              f"p90 {np.nanpercentile(off, 90):.2f}  max {np.nanmax(off):.2f}")
        big = off > 2.0
        if big.any():
            print(f"  where offset > 2 s (n={int(big.sum())}): "
                  f"same-anchor median {np.nanmedian(same[big]):+.4f} vs "
                  f"as-published median {np.nanmedian(pub[big]):+.4f}")


if __name__ == "__main__":
    main()
