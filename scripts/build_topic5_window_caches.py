"""Topic5 A-line §3.2 window sweep — slice the v2 trace cache into per-window standard caches.

The v2 cache (t0_feature_cache_v2_windows) stores the FULL broadband/HFA z traces (bb_zt/hfa_zt)
+ per-bin relative-onset time (bb_relt/hfa_relt). The alignment runner, however, reads a
pre-computed per-channel scalar `bb_auc__<idx>` / `hfa_auc__<idx>`. So for each of the 6 windows
(0-5/5-10/0-10/0-20 post + [-10,0] proximal-pre + [-120,-90] distal-pre negative control) we
slice the trace with src.topic5_t0_features.window_activation and write a per-window npz that LOOKS
like a standard t0 cache (channels + bb_auc__idx + hfa_auc__idx + bact__idx) + meta json with
eligible_idxs. Then `run_topic5_axis_alignment.py --cache-dir <window dir>` runs unchanged per window.

bact (baseline activity anchor) is window-independent -> copied verbatim. A seizure whose trace does
not cover a window (e.g. distal [-120,-90] needs --pre-feature-window 130) yields all-NaN activation
for that window and is dropped by the runner's finite>=6 gate (tracked, not silently forced).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_t0_features import AXIS_WINDOWS, window_activation

V2_DIR = Path("results/topic5_ictal_recruitment/t0_feature_cache_v2_windows")
OUT_ROOT = Path("results/topic5_ictal_recruitment/axis_alignment/window_caches")


def _slice_subject(npz_path: Path, meta_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    meta = json.load(open(meta_path))
    channels = data["channels"]
    idxs = meta.get("eligible_idxs", [])
    per_window = {w: {"channels": channels} for w in AXIS_WINDOWS}
    cover = {w: 0 for w in AXIS_WINDOWS}
    for idx in idxs:
        bz, br = f"bb_zt__{idx}", f"bb_relt__{idx}"
        hz, hr = f"hfa_zt__{idx}", f"hfa_relt__{idx}"
        ba = f"bact__{idx}"
        if bz not in data.files or hz not in data.files:
            continue
        bb_zt, bb_relt = data[bz], data[br]
        hfa_zt, hfa_relt = data[hz], data[hr]
        for w, (a, b) in AXIS_WINDOWS.items():
            bb_auc = window_activation(bb_zt, bb_relt, a, b)
            hfa_auc = window_activation(hfa_zt, hfa_relt, a, b)
            per_window[w][f"bb_auc__{idx}"] = bb_auc.astype(np.float32)
            per_window[w][f"hfa_auc__{idx}"] = hfa_auc.astype(np.float32)
            if ba in data.files:
                per_window[w][f"bact__{idx}"] = data[ba]
            if np.isfinite(bb_auc).sum() >= 6:
                cover[w] += 1
    return per_window, meta, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default=str(V2_DIR))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    v2 = Path(args.v2_dir)
    out_root = Path(args.out_root)
    subs = sorted(p.stem for p in v2.glob("*.npz"))
    if args.subjects:
        subs = [s for s in subs if s in set(args.subjects)]
    print(f"[window-caches] {len(subs)} v2 subjects -> {len(AXIS_WINDOWS)} windows", flush=True)
    for w in AXIS_WINDOWS:
        (out_root / w).mkdir(parents=True, exist_ok=True)
    cover_tot = {w: 0 for w in AXIS_WINDOWS}
    for sid in subs:
        npz_path = v2 / f"{sid}.npz"
        meta_path = v2 / f"{sid}.json"
        if not meta_path.exists():
            print(f"  {sid}: no meta json, skip", flush=True)
            continue
        per_window, meta, cover = _slice_subject(npz_path, meta_path)
        for w, arrs in per_window.items():
            np.savez(out_root / w / f"{sid}.npz", **arrs)
            wm = dict(meta)
            wm["window"] = {"name": w, "range_sec": list(AXIS_WINDOWS[w])}
            json.dump(wm, open(out_root / w / f"{sid}.json", "w"), indent=2, ensure_ascii=False, default=float)
            cover_tot[w] += cover[w]
        print(f"  {sid}: sliced {len(meta.get('eligible_idxs', []))} seizures x {len(AXIS_WINDOWS)} windows "
              f"| finite>=6 per window: {dict(cover)}", flush=True)
    print("\n=== window seizure coverage (finite>=6) totals ===")
    for w, (a, b) in AXIS_WINDOWS.items():
        print(f"  {w:22s} ({a:5.0f},{b:3.0f}): {cover_tot[w]} seizures")


if __name__ == "__main__":
    main()
