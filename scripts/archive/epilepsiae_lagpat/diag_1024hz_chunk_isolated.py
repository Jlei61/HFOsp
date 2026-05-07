"""1024 Hz 第二阶段诊断：notch 在整段 vs 单 chunk 上的差异。

原假设：Path A 在整个 3600s 上做 CAR+notch 一次，再切 200s；
而 legacy 是逐 200s chunk 各自做 CAR+notch。
本脚本在同一 200s 数据上 raw → CAR → notch → 包络 → 检测 流程做对照，
量化两条路径在第一个 chunk 上的事件计数差。
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import firwin, fftconvolve, hilbert

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.bqk_utils import find_high_enveTimes  # noqa: E402

DATA = Path("/mnt/epilepsia_data/inv2/pat_63502/adm_635102/rec_63500102/63500102_0000.data")
HEAD = Path("/mnt/epilepsia_data/inv2/pat_63502/adm_635102/rec_63500102/63500102_0000.head")
LEGACY = REPO / "results/_legacy_2021_backup/inv2/pat_63502/adm_635102/rec_63500102/63500102_0000_gpu.npz"
OUT = REPO / "results/_diag_1024hz/635_chunk0_isolated_vs_full.json"

EEG_DROP = {
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "FZ", "CZ", "PZ", "F7", "F8", "T3", "T4", "T5", "T6", "T1", "T2",
    "EOG", "EOG1", "EOG2", "ECG", "EMG", "EMG1", "EMG2",
    "HP1", "HP2", "HP3", "PHO",
}

CHUNK_SEC = 200.0


def parse_head() -> dict:
    info = {}
    with open(HEAD) as fh:
        for line in fh:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v
    elec = info["elec_names"][1:-1].split(",")
    info["elec_names"] = [x.strip() for x in elec]
    info["sample_freq"] = float(info["sample_freq"])
    info["conversion_factor"] = float(info["conversion_factor"])
    info["num_channels"] = int(info["num_channels"])
    info["sample_bytes"] = int(info["sample_bytes"])
    return info


def fetch_chunk(head: dict, t_start: float, t_end: float) -> tuple[np.ndarray, list[str]]:
    fs = head["sample_freq"]
    sb = head["sample_bytes"]
    nc = head["num_channels"]
    bb = int(round(t_start * fs)) * sb * nc
    be = int(round(t_end * fs)) * sb * nc
    file_size = os.path.getsize(DATA)
    if be > file_size:
        be = file_size
    with open(DATA, "rb") as fh:
        fh.seek(bb)
        raw = fh.read(be - bb)
    n = (be - bb) // sb
    arr = np.array(struct.unpack(f"<{n}h", raw)).reshape(-1, nc).T
    arr = -1.0 * head["conversion_factor"] * arr.astype(np.float64)
    keep_idx = [i for i, n in enumerate(head["elec_names"])
                if n.strip().upper() not in EEG_DROP]
    return arr[keep_idx], [head["elec_names"][i] for i in keep_idx]


def car(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=0, keepdims=True)


def notch_legacy(data: np.ndarray, fs: float) -> np.ndarray:
    """Verbatim copy of epilepsiae_detectHFOs.py:79-83 in CPU."""
    nyq = fs / 2.0
    out = data
    for f in [50.0, 100.0, 150.0, 200.0, 250.0]:
        if (f + 1) >= nyq:
            continue
        b = firwin(801, [(f - 1) / nyq, (f + 1) / nyq], pass_zero=True)
        out = fftconvolve(out, b[None, :], mode="same", axes=-1)
    return out


def envelope_legacy(data: np.ndarray, fs: float) -> np.ndarray:
    """9-band envelope sum, FIR-201 forward fftconvolve. Matches GPU path."""
    nyq = fs / 2.0
    edges = list(np.arange(80.0, 250.0, 20.0)) + [250.0]
    bands = list(zip(edges[:-1], edges[1:]))
    env = np.zeros_like(data)
    for low, high in bands:
        b = firwin(201, [low / nyq, high / nyq], pass_zero=False)
        filt = fftconvolve(data, b[None, :], mode="same", axes=-1)
        analytic = hilbert(filt, axis=-1)
        env += np.abs(analytic)
    return env


def detect_pre_and_with_side(env: np.ndarray, fs: float, ch_names: list[str]) -> dict:
    pre = find_high_enveTimes(
        raw_enve=env, chns_nums=env.shape[0], fs=fs,
        rel_thresh=2.0, abs_thresh=2.0, min_gap=20.0, min_last=50.0, max_last=200.0,
        side_thresh=None, start_time=0.0, legacy_align=True,
    )
    side = find_high_enveTimes(
        raw_enve=env, chns_nums=env.shape[0], fs=fs,
        rel_thresh=2.0, abs_thresh=2.0, min_gap=20.0, min_last=50.0, max_last=200.0,
        side_thresh=2.0, start_time=0.0, legacy_align=True,
    )
    return {
        "pre_side_total": int(sum(len(e) for e in pre)),
        "with_side_total": int(sum(len(e) for e in side)),
        "pre_side_per_ch": {ch: len(e) for ch, e in zip(ch_names, pre)},
        "with_side_per_ch": {ch: len(e) for ch, e in zip(ch_names, side)},
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = parse_head()
    fs = head["sample_freq"]

    # === Path B: pure isolated chunk (legacy emulation) ===
    print("[diag] PATH B: legacy-emulated, chunk-isolated ...", flush=True)
    t0 = time.time()
    raw, ch_names = fetch_chunk(head, 0.0, CHUNK_SEC)
    print(f"  fetched: {raw.shape}, {len(ch_names)} ch, {time.time()-t0:.1f}s", flush=True)

    car_data = car(raw)
    notched = notch_legacy(car_data, fs)
    env_iso = envelope_legacy(notched, fs)
    iso = detect_pre_and_with_side(env_iso, fs, ch_names)
    iso["whole_data_median"] = float(np.median(env_iso))
    iso["ch_medians"] = {ch: float(m) for ch, m in zip(ch_names, np.median(env_iso, axis=-1))}
    print(f"  PATH B pre_side={iso['pre_side_total']}, with_side={iso['with_side_total']}", flush=True)

    # === Path A: load whole recording first, then notch, then chunk ===
    print("\n[diag] PATH A: notch-on-whole-then-chunk ...", flush=True)
    t0 = time.time()
    full_raw, full_ch = fetch_chunk(head, 0.0, head.get("duration_in_sec", 3600.0))  # whole 3600s
    print(f"  fetched: {full_raw.shape}, {time.time()-t0:.1f}s", flush=True)

    full_car = car(full_raw)
    full_notched = notch_legacy(full_car, fs)
    n_samp_chunk = int(round(CHUNK_SEC * fs))
    chunk_data = full_notched[:, :n_samp_chunk]
    env_full = envelope_legacy(chunk_data, fs)
    full = detect_pre_and_with_side(env_full, fs, ch_names)
    full["whole_data_median"] = float(np.median(env_full))
    full["ch_medians"] = {ch: float(m) for ch, m in zip(ch_names, np.median(env_full, axis=-1))}
    print(f"  PATH A pre_side={full['pre_side_total']}, with_side={full['with_side_total']}", flush=True)

    # Per-channel signed delta (PATH A − PATH B)
    delta = []
    for ch in ch_names:
        a_pre = full["pre_side_per_ch"][ch]
        b_pre = iso["pre_side_per_ch"][ch]
        delta.append({
            "ch": ch,
            "iso_pre": int(b_pre),
            "full_pre": int(a_pre),
            "iso_med": float(iso["ch_medians"][ch]),
            "full_med": float(full["ch_medians"][ch]),
        })

    # Legacy ground truth
    g = np.load(LEGACY, allow_pickle=True)
    legacy_total = sum(1 for ev in g["whole_dets"][ci] if ev[0] < CHUNK_SEC
                       for ci in [list(g["chns_names"]).index(ch_names[0])]) if ch_names[0] in list(g["chns_names"]) else None  # hacky
    chns_legacy = list(g["chns_names"])
    legacy_per_ch = {}
    for ci, ch in enumerate(chns_legacy):
        evts = g["whole_dets"][ci]
        legacy_per_ch[ch] = sum(1 for e in evts if e[0] < CHUNK_SEC) if isinstance(evts, list) else 0
    legacy_total = sum(legacy_per_ch.values())

    summary = {
        "chunk_sec": CHUNK_SEC,
        "fs": fs,
        "legacy_total": int(legacy_total),
        "isolated_pre_side": iso["pre_side_total"],
        "isolated_with_side": iso["with_side_total"],
        "isolated_whole_median": iso["whole_data_median"],
        "full_pre_side": full["pre_side_total"],
        "full_with_side": full["with_side_total"],
        "full_whole_median": full["whole_data_median"],
        "delta_per_ch": delta,
    }
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print("=" * 76)
    print(f"Legacy total ground-truth        = {legacy_total}")
    print(f"PATH B (chunk-isolated)  pre={iso['pre_side_total']}  side={iso['with_side_total']}  med={iso['whole_data_median']:.4f}")
    print(f"PATH A (notch-on-whole)  pre={full['pre_side_total']}  side={full['with_side_total']}  med={full['whole_data_median']:.4f}")
    print()
    print(f"{'ch':8s} {'legacy':>7s} {'isoP':>5s} {'fulP':>5s} {'iso_med':>9s} {'ful_med':>9s}")
    rows = sorted(delta, key=lambda x: -legacy_per_ch.get(x["ch"], 0))[:25]
    for r in rows:
        print(f"{r['ch']:8s} {legacy_per_ch.get(r['ch'],0):7d} {r['iso_pre']:5d} {r['full_pre']:5d} "
              f"{r['iso_med']:9.4f} {r['full_med']:9.4f}")


if __name__ == "__main__":
    main()
