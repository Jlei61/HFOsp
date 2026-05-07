"""1024 Hz 第三阶段诊断：取 legacy 已知事件，回到 Path A envelope 上看 pick/side 比值。

如果 legacy 在该事件 keep（说明 GPU pipeline 算出 pick_mean >= 2*side_mean），
但 Path A 在同一事件 reject（pick_mean < 2*side_mean），
说明 envelope 形态在 GPU 与 scipy CPU 间确实不同。
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


def fetch_chunk(head: dict, t_start: float, t_end: float):
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


def event_pick_side(env_ch, fs, t0, t1):
    dur = t1 - t0
    i_pre_s = max(0, int((t0 - dur) * fs))
    i_pre_e = int(t0 * fs)
    i_post_s = int(t1 * fs)
    i_post_e = min(len(env_ch), int((t1 + dur) * fs))
    side_pre = env_ch[i_pre_s:i_pre_e]
    side_post = env_ch[i_post_s:i_post_e]
    pick = env_ch[int(t0 * fs):int(t1 * fs)]
    side_mean = float(np.mean(np.concatenate([side_pre, side_post]))) if len(side_pre) + len(side_post) > 0 else float("nan")
    pick_mean = float(np.mean(pick)) if len(pick) > 0 else float("nan")
    return pick_mean, side_mean


def main():
    head = parse_head()
    fs = head["sample_freq"]

    print("[diag] loading 200 s ...", flush=True)
    raw, ch_names = fetch_chunk(head, 0.0, CHUNK_SEC)
    print("[diag] CAR + notch + envelope ...", flush=True)
    env = envelope(notch(car(raw), fs), fs)
    ch_index = {ch: i for i, ch in enumerate(ch_names)}

    g = np.load(LEGACY, allow_pickle=True)
    legacy_chns = list(g["chns_names"])

    targets = ["TLA1", "HL2", "HRA5", "TBC1", "TLC1", "HRC4", "HL10"]
    print()
    print(f"{'ch':6s} {'event #':>7s} {'t0':>9s} {'t1':>9s} {'dur':>6s} "
          f"{'pick_m':>9s} {'side_m':>9s} {'ratio':>6s} {'keep':>5s}")
    print("-" * 80)
    for ch in targets:
        if ch not in ch_index or ch not in legacy_chns:
            continue
        ci_path = ch_index[ch]
        ci_leg = legacy_chns.index(ch)
        evts = g["whole_dets"][ci_leg]
        if not isinstance(evts, list):
            continue
        evts_first200 = [e for e in evts if e[0] < CHUNK_SEC]
        env_ch = env[ci_path]
        for ei, ev in enumerate(evts_first200):
            t0, t1 = float(ev[0]), float(ev[1])
            p, s = event_pick_side(env_ch, fs, t0, t1)
            ratio = p / s if s > 0 else float("nan")
            keep = "YES" if ratio >= 2.0 else "NO"
            print(f"{ch:6s} {ei+1:7d} {t0:9.4f} {t1:9.4f} {(t1-t0)*1000:6.1f}ms "
                  f"{p:9.4f} {s:9.4f} {ratio:6.3f} {keep:>5s}")


if __name__ == "__main__":
    main()
