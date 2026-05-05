"""1024 Hz 第五阶段诊断：cusignal 走 float32 是否会逼近 legacy 计数？

假设 2021 年旧版 cusignal 默认 float32（cuFFT 在 GPU 上 float32 比 float64 快很多）。
当代 cusignal 23.x 默认 float64。本脚本强制把 envelope 计算中的 fftconvolve 与
hilbert 都做成 float32 / complex64，看是否复刻 legacy。
"""
from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


DATA = Path("/mnt/epilepsia_data/inv2/pat_63502/adm_635102/rec_63500102/63500102_0000.data")
HEAD = Path("/mnt/epilepsia_data/inv2/pat_63502/adm_635102/rec_63500102/63500102_0000.head")
LEGACY = REPO / "results/_legacy_2021_backup/inv2/pat_63502/adm_635102/rec_63500102/63500102_0000_gpu.npz"

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


def fetch_chunk(head, t0, t1, dtype=np.float64):
    fs = head["sample_freq"]
    sb = head["sample_bytes"]
    nc = head["num_channels"]
    bb = int(round(t0 * fs)) * sb * nc
    be = int(round(t1 * fs)) * sb * nc
    file_size = os.path.getsize(DATA)
    be = min(be, file_size)
    with open(DATA, "rb") as fh:
        fh.seek(bb)
        raw = fh.read(be - bb)
    n = (be - bb) // sb
    arr = np.array(struct.unpack(f"<{n}h", raw)).reshape(-1, nc).T
    arr = -1.0 * head["conversion_factor"] * arr.astype(dtype)
    keep_idx = [i for i, n in enumerate(head["elec_names"])
                if n.strip().upper() not in EEG_DROP]
    return arr[keep_idx], [head["elec_names"][i] for i in keep_idx]


def run_gpu(dtype):
    """Verbatim emulation of epilepsiae_detectHFOs.py:226-235 in cuSignal."""
    import cupy as cp
    import cusignal

    print(f"\n=== dtype = {np.dtype(dtype).name} ===", flush=True)
    head = parse_head()
    fs = head["sample_freq"]
    raw, ch_names = fetch_chunk(head, 0.0, CHUNK_SEC, dtype=dtype)
    data_gpu = cp.asarray(raw)
    print(f"data_gpu.dtype = {data_gpu.dtype}, shape={data_gpu.shape}", flush=True)

    # CAR
    data_gpu = data_gpu - cp.mean(data_gpu, axis=0, keepdims=True)

    # Notch (5 freqs, FIR-801 forward fftconvolve verbatim)
    nyq = fs / 2.0
    for f in [50.0, 100.0, 150.0, 200.0, 250.0]:
        b = cusignal.firwin(801, [(f - 1) / nyq, (f + 1) / nyq], pass_zero=True)
        b = cp.asarray(b, dtype=dtype)
        data_gpu = cusignal.convolution.fftconvolve(data_gpu, b[None, :], mode="same")
        data_gpu = data_gpu.astype(dtype)
    print(f"after notch dtype = {data_gpu.dtype}", flush=True)

    # 9-band envelope sum (FIR-201 forward fftconvolve + hilbert)
    edges = list(np.arange(80.0, 250.0, 20.0)) + [250.0]
    bands = list(zip(edges[:-1], edges[1:]))
    env_sum = cp.zeros_like(data_gpu)
    for low, high in bands:
        b = cusignal.firwin(201, [low / nyq, high / nyq], pass_zero=False)
        b = cp.asarray(b, dtype=dtype)
        filt = cusignal.convolution.fftconvolve(data_gpu, b[None, :], mode="same")
        filt = filt.astype(dtype)
        analytic = cusignal.hilbert(filt, axis=-1)
        env_sum = env_sum + cp.abs(analytic).astype(dtype)
    print(f"envelope dtype = {env_sum.dtype}", flush=True)

    env_np = cp.asnumpy(env_sum).astype(np.float64)

    from src.utils.bqk_utils import find_high_enveTimes
    side = find_high_enveTimes(
        raw_enve=env_np, chns_nums=env_np.shape[0], fs=fs,
        rel_thresh=2.0, abs_thresh=2.0, min_gap=20.0, min_last=50.0, max_last=200.0,
        side_thresh=2.0, start_time=0.0, legacy_align=True,
    )
    pre = find_high_enveTimes(
        raw_enve=env_np, chns_nums=env_np.shape[0], fs=fs,
        rel_thresh=2.0, abs_thresh=2.0, min_gap=20.0, min_last=50.0, max_last=200.0,
        side_thresh=None, start_time=0.0, legacy_align=True,
    )

    print(f"  pre_side_total = {sum(len(e) for e in pre)}")
    print(f"  with_side_total = {sum(len(e) for e in side)}")
    print(f"  whole_data_median = {float(np.median(env_np)):.4f}")
    return ch_names, env_np, pre, side


def main():
    g = np.load(LEGACY, allow_pickle=True)
    chns = list(g["chns_names"])
    legacy_total = sum(sum(1 for ev in g["whole_dets"][ci] if ev[0] < CHUNK_SEC)
                       for ci in range(len(chns)) if isinstance(g["whole_dets"][ci], list))
    print(f"Legacy [0, {CHUNK_SEC}) s total = {legacy_total}")

    ch64, env64, pre64, side64 = run_gpu(np.float64)
    ch32, env32, pre32, side32 = run_gpu(np.float32)

    # Per-channel float32 vs float64
    legacy_per_ch = {}
    for ci, ch in enumerate(chns):
        evts = g["whole_dets"][ci]
        legacy_per_ch[ch] = sum(1 for e in evts if e[0] < CHUNK_SEC) if isinstance(evts, list) else 0

    print(f"\n{'ch':8s} {'leg':>5s} {'f64':>5s} {'f32':>5s}")
    for ch in sorted(legacy_per_ch.keys(), key=lambda c: -legacy_per_ch[c])[:25]:
        if ch not in ch64:
            continue
        i64 = ch64.index(ch)
        i32 = ch32.index(ch)
        print(f"{ch:8s} {legacy_per_ch[ch]:5d} {len(side64[i64]):5d} {len(side32[i32]):5d}")


if __name__ == "__main__":
    main()
