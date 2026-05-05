"""1024 Hz 上游偏差诊断：subject 635 record_0 第一个 200s chunk。

对比 Path A 与 legacy ground truth（cusignal 路径产物）：
- Per-channel event counts（pre-side 与 with side_thresh）
- Per-channel envelope median 与 whole-data envelope median
- 跨阈值的 pre-side 事件数（用于定位是阈值还是 side rejection 出问题）

输出：results/_diag_1024hz/635_chunk0_pathA_vs_legacy.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.preprocessing import load_epilepsiae_block  # noqa: E402
from src.utils.bqk_utils import (  # noqa: E402
    BQKDetector,
    find_high_enveTimes,
)


SUBJECT_DIR = Path("/mnt/epilepsia_data/inv2/pat_63502/adm_635102/rec_63500102")
RECORD_STEM = "63500102_0000"
LEGACY_GPU = REPO / "results/_legacy_2021_backup/inv2/pat_63502/adm_635102/rec_63500102/63500102_0000_gpu.npz"
OUT = REPO / "results/_diag_1024hz/635_chunk0_pathA_vs_legacy.json"

CHUNK_SEC = 200.0


def legacy_first_chunk_counts() -> tuple[int, dict[str, int]]:
    g = np.load(LEGACY_GPU, allow_pickle=True)
    chns = [str(c) for c in g["chns_names"]]
    whole = g["whole_dets"]
    counts = {}
    total = 0
    for ci, ch in enumerate(chns):
        evts = whole[ci]
        if isinstance(evts, list):
            n = sum(1 for e in evts if e[0] < CHUNK_SEC)
        else:
            n = 0
        counts[ch] = n
        total += n
    return total, counts


def run_path_a() -> dict:
    print("[diag] loading 200 s of subject 635 chunk 0 ...", flush=True)
    t0 = time.time()
    pre = load_epilepsiae_block(
        SUBJECT_DIR / f"{RECORD_STEM}.data",
        SUBJECT_DIR / f"{RECORD_STEM}.head",
        reference="car",
        notch_freqs=[50.0, 100.0, 150.0, 200.0, 250.0],
        notch_filter_kind="fir_legacy",
    )
    print(f"[diag] load+CAR+notch: {time.time()-t0:.1f}s, "
          f"shape={pre.data.shape}, sfreq={pre.sfreq}", flush=True)

    n_samp_chunk = int(round(CHUNK_SEC * pre.sfreq))
    chunk = pre.data[:, :n_samp_chunk]
    print(f"[diag] chunk shape: {chunk.shape}", flush=True)

    # Compute envelope using BQKDetector legacy_align path (FIR-201 forward fftconvolve, 9 sub-bands)
    import os
    use_gpu = os.environ.get("DIAG_USE_GPU", "0") == "1"
    det = BQKDetector(
        sfreq=pre.sfreq,
        freqband=(80.0, 250.0),
        subband_width=20.0,
        rel_thresh=2.0,
        abs_thresh=2.0,
        min_gap=20.0,
        min_last=50.0,
        max_last=200.0,
        side_thresh=2.0,
        n_jobs=1,
        legacy_align=True,
        use_gpu=use_gpu,
    )
    print(f"[diag] use_gpu={det.use_gpu}", flush=True)
    print("[diag] computing envelope (9 sub-bands, FIR-201 forward) ...", flush=True)
    t0 = time.time()
    env = det.compute_envelope(chunk)
    print(f"[diag] envelope: {time.time()-t0:.1f}s, shape={env.shape}", flush=True)

    # Median diagnostics
    whole_median = float(np.median(env))
    ch_medians = np.median(env, axis=-1)
    ch_medians_dict = {ch: float(m) for ch, m in zip(pre.ch_names, ch_medians)}

    # Pre-side detection: side_thresh=None disables side rejection
    print("[diag] running pre-side detection ...", flush=True)
    pre_side_events = find_high_enveTimes(
        raw_enve=env,
        chns_nums=env.shape[0],
        fs=pre.sfreq,
        rel_thresh=2.0,
        abs_thresh=2.0,
        min_gap=20.0,
        min_last=50.0,
        max_last=200.0,
        side_thresh=None,
        start_time=0.0,
        legacy_align=True,
    )
    pre_side_counts = {ch: len(ev) for ch, ev in zip(pre.ch_names, pre_side_events)}
    pre_side_total = int(sum(pre_side_counts.values()))

    # With side_thresh=2 + legacy_align (current Path A default for chunk 0)
    print("[diag] running with side_thresh=2 + legacy_align ...", flush=True)
    side_events = find_high_enveTimes(
        raw_enve=env,
        chns_nums=env.shape[0],
        fs=pre.sfreq,
        rel_thresh=2.0,
        abs_thresh=2.0,
        min_gap=20.0,
        min_last=50.0,
        max_last=200.0,
        side_thresh=2.0,
        start_time=0.0,
        legacy_align=True,
    )
    side_counts = {ch: len(ev) for ch, ev in zip(pre.ch_names, side_events)}
    side_total = int(sum(side_counts.values()))

    return {
        "sfreq": float(pre.sfreq),
        "n_channels": int(chunk.shape[0]),
        "n_samples": int(chunk.shape[1]),
        "ch_names": list(pre.ch_names),
        "whole_data_median": whole_median,
        "ch_medians": ch_medians_dict,
        "pre_side_total": pre_side_total,
        "pre_side_counts": pre_side_counts,
        "with_side_total": side_total,
        "with_side_counts": side_counts,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    legacy_total, legacy_counts = legacy_first_chunk_counts()
    print(f"\n[diag] legacy [0, {CHUNK_SEC}) s total events = {legacy_total}", flush=True)

    res = run_path_a()

    # Overlap analysis
    common = sorted(set(res["ch_names"]) & set(legacy_counts.keys()))
    rows = []
    for ch in common:
        rows.append({
            "ch": ch,
            "legacy": int(legacy_counts.get(ch, 0)),
            "path_a_pre_side": int(res["pre_side_counts"].get(ch, 0)),
            "path_a_with_side": int(res["with_side_counts"].get(ch, 0)),
            "ch_median": float(res["ch_medians"].get(ch, float("nan"))),
        })

    summary = {
        "subject": "635",
        "record": RECORD_STEM,
        "chunk_sec": CHUNK_SEC,
        "legacy_total": int(legacy_total),
        "path_a_pre_side_total": res["pre_side_total"],
        "path_a_with_side_total": res["with_side_total"],
        "n_channels_path_a": res["n_channels"],
        "n_channels_legacy": len(legacy_counts),
        "n_channels_common": len(common),
        "whole_data_median_path_a": res["whole_data_median"],
        "rows": rows,
    }

    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[diag] wrote: {OUT}\n", flush=True)

    # Console summary
    print("=" * 72)
    print(f"Legacy total                 = {legacy_total}")
    print(f"Path A pre-side  total       = {res['pre_side_total']}  (ratio {res['pre_side_total']/max(1,legacy_total):.3f})")
    print(f"Path A with-side total       = {res['with_side_total']}  (ratio {res['with_side_total']/max(1,legacy_total):.3f})")
    print(f"Path A whole_data_median     = {res['whole_data_median']:.4f}")
    print()
    print(f"{'channel':8s} {'legacy':>7s} {'A_pre':>7s} {'A_side':>7s} {'ch_med':>10s}")
    for r in sorted(rows, key=lambda x: -x["legacy"])[:20]:
        print(f"{r['ch']:8s} {r['legacy']:7d} {r['path_a_pre_side']:7d} {r['path_a_with_side']:7d} {r['ch_median']:10.4f}")


if __name__ == "__main__":
    main()
