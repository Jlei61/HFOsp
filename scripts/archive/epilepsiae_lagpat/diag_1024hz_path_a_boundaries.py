"""1024 Hz 第四阶段诊断：Path A 在 TLA1 上 pre-side 与 with-side 的实际事件边界。

逐 event 显示 dur，pick/side 比值；并对比 legacy 同一通道的事件。
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.signal import firwin, fftconvolve, hilbert

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.bqk_utils import find_high_enveTimes  # noqa: E402

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


def fetch_chunk(head, t0, t1):
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
    arr = -1.0 * head["conversion_factor"] * arr.astype(np.float64)
    keep_idx = [i for i, n in enumerate(head["elec_names"])
                if n.strip().upper() not in EEG_DROP]
    return arr[keep_idx], [head["elec_names"][i] for i in keep_idx]


def car(d): return d - d.mean(axis=0, keepdims=True)


def notch(d, fs):
    nyq = fs / 2.0
    out = d
    for f in [50.0, 100.0, 150.0, 200.0, 250.0]:
        if f + 1 >= nyq:
            continue
        b = firwin(801, [(f - 1) / nyq, (f + 1) / nyq], pass_zero=True)
        out = fftconvolve(out, b[None, :], mode="same", axes=-1)
    return out


def envelope(d, fs):
    nyq = fs / 2.0
    edges = list(np.arange(80.0, 250.0, 20.0)) + [250.0]
    bands = list(zip(edges[:-1], edges[1:]))
    env = np.zeros_like(d)
    for low, high in bands:
        b = firwin(201, [low / nyq, high / nyq], pass_zero=False)
        f = fftconvolve(d, b[None, :], mode="same", axes=-1)
        env += np.abs(hilbert(f, axis=-1))
    return env


def main():
    head = parse_head()
    fs = head["sample_freq"]
    raw, ch_names = fetch_chunk(head, 0.0, CHUNK_SEC)
    env = envelope(notch(car(raw), fs), fs)
    ci = ch_names.index("TLA1")
    env_ch = env[ci]
    ch_med = float(np.median(env_ch))
    whole_med = float(np.median(env))
    print(f"TLA1 ch_median = {ch_med:.4f}, whole_median = {whole_med:.4f}")
    print(f"thresholds: rel = 2 * ch_med = {2*ch_med:.4f}, abs = 2 * whole_med = {2*whole_med:.4f}")

    # Pre-side detection on this channel only
    pre_evts = find_high_enveTimes(
        raw_enve=env, chns_nums=env.shape[0], fs=fs,
        rel_thresh=2.0, abs_thresh=2.0, min_gap=20.0, min_last=50.0, max_last=200.0,
        side_thresh=None, start_time=0.0, legacy_align=True,
    )[ci]
    side_evts = find_high_enveTimes(
        raw_enve=env, chns_nums=env.shape[0], fs=fs,
        rel_thresh=2.0, abs_thresh=2.0, min_gap=20.0, min_last=50.0, max_last=200.0,
        side_thresh=2.0, start_time=0.0, legacy_align=True,
    )[ci]

    g = np.load(LEGACY, allow_pickle=True)
    legacy_chns = list(g["chns_names"])
    legacy_evts_all = g["whole_dets"][legacy_chns.index("TLA1")]
    legacy_evts = [e for e in legacy_evts_all if e[0] < CHUNK_SEC]

    print(f"\nLegacy events on TLA1 in [0, 200): {len(legacy_evts)}")
    for e in legacy_evts:
        print(f"  legacy: t0={e[0]:.4f}, t1={e[1]:.4f}, dur={(e[1]-e[0])*1000:.1f}ms")
    print(f"\nPath A pre-side events on TLA1: {len(pre_evts)} (showing top 30 sorted by t0)")
    for ei, e in enumerate(sorted(pre_evts, key=lambda x: x[0])[:30]):
        t0, t1 = float(e[0]), float(e[1])
        # Check pick/side
        dur = t1 - t0
        i_pre_s = max(0, int((t0 - dur) * fs))
        i_pre_e = int(t0 * fs)
        i_post_s = int(t1 * fs)
        i_post_e = min(len(env_ch), int((t1 + dur) * fs))
        side_pre = env_ch[i_pre_s:i_pre_e]
        side_post = env_ch[i_post_s:i_post_e]
        side_mean = float(np.mean(np.concatenate([side_pre, side_post]))) if (len(side_pre)+len(side_post))>0 else float("nan")
        pick_mean = float(np.mean(env_ch[int(t0*fs):int(t1*fs)])) if int(t1*fs) > int(t0*fs) else float("nan")
        ratio = pick_mean / side_mean if side_mean > 0 else float("nan")
        # Match against legacy
        match = ""
        for le in legacy_evts:
            if abs(t0 - le[0]) < 0.05:
                match = f"~legacy {le[0]:.3f}"
                break
        print(f"  pre {ei+1:2d}: t0={t0:8.4f} t1={t1:8.4f} dur={dur*1000:5.1f}ms "
              f"pick={pick_mean:7.3f} side={side_mean:7.3f} r={ratio:5.2f}  {match}")
    print(f"\nPath A with-side events on TLA1: {len(side_evts)}")
    for e in side_evts:
        print(f"  with_side: t0={e[0]:.4f}, t1={e[1]:.4f}")


if __name__ == "__main__":
    main()
