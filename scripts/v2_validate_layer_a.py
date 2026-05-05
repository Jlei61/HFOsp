"""HFO detector v2 — Layer A validation extractor (single-channel quality).

Computes per-subject metrics from a v2 detection output (*_gpu.npz):
  - dur_in_band_frac
  - peak_side_ratio (p25, p50, p99)
  - threshold_margin (p50)
  - timestamp_jitter_p99 (requires twice-run inputs)
  - strong_chn_count_match (requires twice-run inputs)

Output: results/hfo_detector_v2/validation/layer_a_<subject>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CHUNK_SEC = 200.0
N_WINDOWS_PER_RECORD = 3  # first / middle / last 200s — non-stationarity coverage


def compute_dur_in_band_frac(events, min_ms, max_ms):
    if len(events) == 0:
        return 1.0
    durs_ms = (events[:, 1] - events[:, 0]) * 1000.0
    return float(np.mean((durs_ms >= min_ms) & (durs_ms < max_ms)))


def compute_peak_side_ratio(env, events, fs):
    """For each event, compute pick_mean / side_mean (using legacy convention)."""
    out = []
    n = len(env)
    for t0, t1 in events:
        dur = t1 - t0
        i_pre_s = max(0, int((t0 - dur) * fs))
        i_pre_e = int(t0 * fs)
        i_post_s = int(t1 * fs)
        i_post_e = min(n, int((t1 + dur) * fs))
        side = np.concatenate([env[i_pre_s:i_pre_e], env[i_post_s:i_post_e]])
        pick = env[int(t0 * fs):int(t1 * fs)]
        if len(side) == 0 or len(pick) == 0:
            out.append(np.nan)
            continue
        s_mean = float(np.mean(side))
        if s_mean <= 0:
            out.append(np.inf)
            continue
        out.append(float(np.mean(pick) / s_mean))
    return np.array(out)


def compute_threshold_margin(env, events, fs, threshold):
    """For each event, (max_env_in_event - threshold) / threshold."""
    out = []
    for t0, t1 in events:
        i0, i1 = int(t0 * fs), int(t1 * fs)
        if i1 <= i0:
            out.append(np.nan)
            continue
        env_max = float(np.max(env[i0:i1]))
        out.append((env_max - threshold) / threshold)
    return np.array(out)


def _window_starts_for_record(rec_duration_sec: float) -> list[float]:
    """Return start_times of N_WINDOWS_PER_RECORD evenly-spaced 200s windows.

    Recording shorter than 600s falls back to [0.0] (first chunk only).
    """
    if rec_duration_sec < CHUNK_SEC * 3:
        return [0.0]
    last_start = max(0.0, rec_duration_sec - CHUNK_SEC)
    if N_WINDOWS_PER_RECORD == 1:
        return [0.0]
    starts = []
    for i in range(N_WINDOWS_PER_RECORD):
        frac = i / (N_WINDOWS_PER_RECORD - 1)
        starts.append(frac * last_start)
    return starts


def extract_layer_a_per_subject(subject_dir: Path, output_dir: Path) -> dict:
    """Iterate all *_gpu.npz under subject_dir; for each record recompute envelope on
    first/middle/last 200s and extract per-channel metrics for events that fall in
    those windows. Aggregating across windows controls for HFO non-stationarity."""
    from src.preprocessing import load_epilepsiae_block
    from src.utils.bqk_utils import BQKDetector

    metrics_per_record = []
    for gpu_path in sorted(subject_dir.glob("*_gpu.npz")):
        head_path = gpu_path.parent / (gpu_path.stem.replace("_gpu", "") + ".head")
        data_path = head_path.with_suffix(".data")
        if not (head_path.exists() and data_path.exists()):
            continue
        npz = np.load(gpu_path, allow_pickle=True)
        chns = list(npz["chns_names"])
        whole_dets = npz["whole_dets"]

        pre = load_epilepsiae_block(
            data_path, head_path, reference="car",
            notch_freqs=[50.0, 100.0, 150.0, 200.0, 250.0],
            notch_filter_kind="fir_legacy",
        )
        rec_dur = pre.data.shape[1] / pre.sfreq
        det = BQKDetector(
            sfreq=pre.sfreq, freqband=(80, 250), subband_width=20,
            rel_thresh=2.0, abs_thresh=2.0, side_thresh=2.0,
            min_gap=20, min_last=50, max_last=200,
            n_jobs=1, legacy_align=True, use_gpu=True,
        )

        # Per-channel running aggregator across all sampled windows
        ch_agg = {ch: {"n_events": 0, "durs": [], "ratios": [], "margins": []} for ch in chns}
        windows_used = []
        for w_start in _window_starts_for_record(rec_dur):
            i0 = int(round(w_start * pre.sfreq))
            i1 = i0 + int(round(CHUNK_SEC * pre.sfreq))
            i1 = min(i1, pre.data.shape[1])
            if i1 - i0 < int(0.5 * CHUNK_SEC * pre.sfreq):
                continue
            chunk = pre.data[:, i0:i1]
            env = det.compute_envelope(chunk)
            whole_med = float(np.median(env))
            t_lo, t_hi = w_start, w_start + (i1 - i0) / pre.sfreq
            windows_used.append({"start_sec": t_lo, "end_sec": t_hi, "whole_data_median": whole_med})

            for ci, ch in enumerate(chns):
                evts_full = whole_dets[ci] if isinstance(whole_dets[ci], list) else []
                # Filter to events inside this window (using global recording time)
                evts_in = np.asarray([e for e in evts_full if t_lo <= e[0] < t_hi], dtype=float)
                if len(evts_in) == 0:
                    continue
                # Re-base event times into window-local for envelope indexing
                evts_local = evts_in.copy()
                evts_local[:, 0] -= t_lo
                evts_local[:, 1] -= t_lo
                ch_med = float(np.median(env[ci]))
                threshold = max(2 * ch_med, 2 * whole_med)
                durs = compute_dur_in_band_frac(evts_local, 50.0, 200.0)
                ratios = compute_peak_side_ratio(env[ci], evts_local, pre.sfreq)
                margins = compute_threshold_margin(env[ci], evts_local, pre.sfreq, threshold)
                ch_agg[ch]["n_events"] += int(len(evts_local))
                ch_agg[ch]["durs"].append(durs * len(evts_local))  # weighted by count
                ch_agg[ch]["ratios"].append(ratios)
                ch_agg[ch]["margins"].append(margins)

        # Aggregate per channel
        ch_metrics = {}
        for ch, agg in ch_agg.items():
            if agg["n_events"] == 0:
                ch_metrics[ch] = {"n_events": 0}
                continue
            ratios_concat = np.concatenate(agg["ratios"]) if agg["ratios"] else np.array([])
            margins_concat = np.concatenate(agg["margins"]) if agg["margins"] else np.array([])
            durs_total = float(sum(agg["durs"])) / agg["n_events"]
            ch_metrics[ch] = {
                "n_events": agg["n_events"],
                "dur_in_band_frac": durs_total,
                "peak_side_ratio_p25": float(np.nanpercentile(ratios_concat, 25)),
                "peak_side_ratio_p50": float(np.nanpercentile(ratios_concat, 50)),
                "threshold_margin_p50": float(np.nanpercentile(margins_concat, 50)),
            }
        metrics_per_record.append({
            "record": gpu_path.stem,
            "rec_duration_sec": rec_dur,
            "windows": windows_used,
            "channels": ch_metrics,
        })

    return {"records": metrics_per_record}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g. 635")
    p.add_argument("--detection-root", default="results/hfo_detector_v2")
    p.add_argument("--output-dir", default="results/hfo_detector_v2/validation")
    args = p.parse_args()

    subject_dir = Path(args.detection_root) / args.subject
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = extract_layer_a_per_subject(subject_dir, out_dir)
    out = out_dir / f"layer_a_{args.subject}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
