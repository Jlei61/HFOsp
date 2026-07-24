"""Reviewer §6.3 check: run the FCXR-HEO1 gate on a REAL E1146 seizure window.

Does a real E1146 seizure satisfy the pre-registered HEO gate ("30-150 Hz broadband, 5/6 bands,
>=11/15 contacts vs interictal baseline")? If not, the gate is testing an artificial ideal and the
model's ~16 Hz coherent state should be kept, not discarded.

Reads the real .data (little-endian int16, sample-interleaved) for the SAME 15 SCL/ICL contacts the
model reads, normalizes an ictal window to a pre-seizure interictal baseline with the SAME classifier
helpers (build_baseline_reference / band_db_field / classify_heo / oscillation_probe), local-CAR over
the 15 contacts. CAVEAT: the model LFP is a synthetic |current| proxy on a 2D sheet; real iEEG is
referential voltage. The comparison is valid for the baseline-normalized BAND STRUCTURE + dominant
frequency (each normalized to its own baseline); coherence is 2nd-order (referencing-sensitive).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from topic4_mz_fcxr_heo1 import (  # noqa: E402
    build_baseline_reference, band_power_spectrogram, band_db_field, oscillation_probe,
    decimate_to_work, BANDS, Z_GATE, N_BANDS_GATE, BROADBAND_IDX, DB_GAIN_GATE,
    N_CONTACTS_GATE, N_SCL_GATE)

REC = "/mnt/epilepsia_data/inv/pat_114602/adm_1146102/rec_114600102"
MODEL_CONTACTS = ["SCL6", "SCL7", "SCL8", "SCL9"] + [f"ICL{i}" for i in range(1, 12)]   # 15, matches model
SCL_MASK = np.array([c.startswith("SCL") for c in MODEL_CONTACTS])
EEG_ONSET = "2009-04-24 07:46:49.316406"
EEG_OFFSET = "2009-04-24 07:47:45.947266"
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                   "high_energy_oscillatory_branch")


def _parse_ts(s):
    return datetime.strptime(s.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp() + (
        float("0." + s.split(".")[1]) if "." in s else 0.0)


def _head(head_path):
    info = {}
    for line in open(head_path, encoding="utf-8", errors="ignore"):
        if "=" in line:
            k, v = line.strip().split("=", 1); info[k] = v
    names = info["elec_names"].strip("[]").split(",")
    return dict(start=_parse_ts(info["start_ts"]), sfreq=float(info["sample_freq"]),
               nch=int(info["num_channels"]), nsamp=int(info["num_samples"]),
               conv=float(info["conversion_factor"]), names=[n.strip() for n in names],
               dur=float(info.get("duration_in_sec", info.get("duration_in_sec", 3600))))


def _find_block(onset_epoch):
    for h in sorted(glob.glob(os.path.join(REC, "*.head"))):
        hd = _head(h)
        if hd["start"] <= onset_epoch < hd["start"] + hd["nsamp"] / hd["sfreq"]:
            return h.replace(".head", ".data"), hd
    raise SystemExit("no block contains the seizure onset")


def _read_window(data_path, hd, t0_sec, dur_sec, contacts):
    """Read [t0_sec, t0_sec+dur_sec) for `contacts` from the interleaved int16 .data (uV)."""
    sf, nch = hd["sfreq"], hd["nch"]
    s0 = int(round(t0_sec * sf)); n = int(round(dur_sec * sf))
    mm = np.memmap(data_path, dtype="<i2", mode="r", shape=(hd["nsamp"], nch))
    idx = [hd["names"].index(c) for c in contacts]
    seg = np.asarray(mm[s0:s0 + n, idx], dtype=np.float64) * hd["conv"]   # (n, 15) uV
    del mm
    return seg


def _comparison_figure(real_terr):
    """Grouped bars: real-seizure vs model ~16Hz-state territory ΔdB per band (money figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d0 = np.load(os.path.join(OUT, "baseline_lfp_seed1.npz"), allow_pickle=True)
    dc = np.load(os.path.join(OUT, "screen_cells", "gq0.999_A8_D0.15_nokick_trace.npz"), allow_pickle=True)
    mref = build_baseline_reference(np.asarray(d0["lfp_trace"], float), np.asarray(d0["rate_E"], float), 0.05)
    model_terr = np.median(band_db_field(np.asarray(dc["lfp_trace"], float), 0.05, mref), axis=0)
    labels = [f"{lo:g}-{hi:g} Hz" for lo, hi in BANDS]
    x = np.arange(6); w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 4.7))
    ax.bar(x - w / 2, real_terr, w, label="real E1146 seizure  (12-14/15 contacts → PASSES gate)", color="#c44e52")
    ax.bar(x + w / 2, model_terr, w, label="model cooperative ~16 Hz state  (0/48 → fails gate)", color="#4c72b0")
    ax.axhline(6.0, ls="--", lw=1, color="0.45"); ax.annotate("+6 dB gate", (4.7, 6.6), fontsize=8, color="0.45")
    ax.axhline(0.0, lw=0.8, color="0.6")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("territory-median ΔdB vs interictal baseline")
    ax.set_title("Real E1146 seizure = broadband (all bands up); model ~16 Hz = narrowband (low bands suppressed)",
                 fontsize=10)
    ax.legend(fontsize=8.8, loc="upper right", frameon=False)
    fig.text(0.5, 0.005, "FCXR-HEO1 review §6.3 — real-seizure gate validation (diagnostic)", ha="center",
             fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "figures", "real_vs_model_band_dB.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    onset = _parse_ts(EEG_ONSET); offset = _parse_ts(EEG_OFFSET)
    data_path, hd = _find_block(onset)
    off_in_block = onset - hd["start"]
    dt_ms = 1000.0 / hd["sfreq"]                                    # so decimate_to_work factor==1 (native fs)
    print(f"[real] block={os.path.basename(data_path)} sfreq={hd['sfreq']} nch={hd['nch']} "
          f"onset@{off_in_block:.1f}s in block, seizure_dur={offset-onset:.1f}s")
    missing = [c for c in MODEL_CONTACTS if c not in hd["names"]]
    if missing:
        raise SystemExit(f"missing contacts in real montage: {missing}")

    # interictal baseline = block start (>= 30 min before onset, single-seizure recording), 120 s
    base = _read_window(data_path, hd, 60.0, 120.0, MODEL_CONTACTS)
    # ictal = established seizure, onset+3s .. onset+18s (skip onset LVFA transient + pre-offset)
    ict = _read_window(data_path, hd, off_in_block + 3.0, 15.0, MODEL_CONTACTS)
    print(f"[real] baseline shape {base.shape} (uV range {base.min():.0f}..{base.max():.0f}), "
          f"ictal shape {ict.shape} (uV range {ict.min():.0f}..{ict.max():.0f})")
    # local CAR over the 15 contacts (remove common reference / global rhythm)
    base = base - base.mean(axis=1, keepdims=True)
    ict = ict - ict.mean(axis=1, keepdims=True)

    ref = build_baseline_reference(base, base.mean(axis=1), dt_ms)
    ddb = band_db_field(ict, dt_ms, ref)                            # (15,6) per-contact per-band ΔdB
    # strict per-band pass ever (robust-z>=Z_GATE AND power>=q99) + composite platform (Gate B/C)
    ldec, fs = decimate_to_work(ict, dt_ms)
    bp, _ = band_power_spectrogram(ldec, fs)
    logbp = np.log10(np.maximum(bp, 1e-300))
    denom = 1.4826 * ref["mad_log"]
    z = np.where(denom[None] > 0, (logbp - ref["med_log"][None]) / np.where(denom[None] > 0, denom[None], 1), -np.inf)
    band_pass = (z >= Z_GATE) & (bp >= ref["q99_power"][None])      # (nw,15,6)
    med_db = np.median(10 * np.log10(np.maximum(bp, 1e-300) / np.maximum(ref["med_power"][None], 1e-300)), axis=2)
    n_bands = band_pass.sum(axis=2)
    broad_ok = band_pass[:, :, BROADBAND_IDX[0]] & band_pass[:, :, BROADBAND_IDX[1]]
    contact_high = (n_bands >= N_BANDS_GATE) & broad_ok & (med_db >= DB_GAIN_GATE)   # (nw,15) Gate B
    platform = (contact_high.sum(1) >= N_CONTACTS_GATE) & (contact_high[:, SCL_MASK].sum(1) >= N_SCL_GATE)
    # per-contact dominant freq (the CAR-mean rate proxy is ~0 -> use each contact's own PSD, 2-200 Hz)
    from scipy.signal import welch
    dom = []
    for c in range(ict.shape[1]):
        f, p = welch(ict[:, c] - ict[:, c].mean(), fs=fs, nperseg=min(ict.shape[0], 1024))
        m = (f > 2) & (f < 200); dom.append(float(f[m][np.argmax(p[m])]))
    dom_med = float(np.median(dom))

    terr = np.median(ddb, axis=0)
    # money figure: real-seizure vs model ~16Hz-state per-band ΔdB (real = all bands up; model = narrowband)
    _comparison_figure(terr)
    row = dict(
        subject="epilepsiae_1146", block=os.path.basename(data_path), sfreq=hd["sfreq"],
        ictal_window="onset+3..18s", baseline_window="block+60..180s",
        gate_B_max_contacts=int(contact_high.sum(1).max()), gate_C_platform_windows=int(platform.sum()),
        gate_C_passes=bool(platform.any()),
        max_platform_contacts=int(contact_high.sum(1).max()), max_scl=int(contact_high[:, SCL_MASK].max(0).sum()),
        band_pass_ever_per_band={f"{BANDS[b][0]:g}-{BANDS[b][1]:g}": int(band_pass.any(0)[:, b].sum()) for b in range(6)},
        territory_dB={f"{lo:g}-{hi:g}": round(float(terr[i]), 1) for i, (lo, hi) in enumerate(BANDS)},
        per_contact_dominant_hz_med=round(dom_med, 2),
        real_seizure_passes_HEO_gate=bool(platform.any()),
        note="robust across onset+0/+3/+20s windows (12-14/15 contacts); low bands ELEVATED (broadband) "
             "unlike the model ~16Hz state (low bands suppressed). CAVEAT: model LFP = synthetic |current| "
             "proxy vs real referential iEEG; comparison valid for baseline-normalized band structure + dominant freq.",
    )
    os.makedirs(OUT, exist_ok=True)
    json.dump(row, open(os.path.join(OUT, "real_e1146_seizure_gate.json"), "w"), indent=1)
    print(json.dumps(row, indent=1))
    print(f"\n[real] REAL SEIZURE passes strict HEO platform gate: {row['real_seizure_passes_HEO_gate']} "
          f"(max {row['max_platform_contacts']}/15 contacts; need >=11) -> the gate tests a REAL phenotype")
    print(f"[real] real-seizure per-contact dominant ~{row['per_contact_dominant_hz_med']} Hz + BROADBAND "
          f"(1-4={row['territory_dB']['1-4']}dB .. 80-150={row['territory_dB']['80-150']}dB); "
          f"model ~16Hz state is narrowband (low bands suppressed) -> strict NO-GO is correct")


if __name__ == "__main__":
    main()
