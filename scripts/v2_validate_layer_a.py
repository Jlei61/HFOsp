"""HFO detector v2 — Layer A validation extractor (single-channel quality).

Computes per-subject metrics from a v2 detection output (*_gpu.npz):
  - dur_in_band_frac
  - peak_side_ratio (p25, p50)
  - threshold_margin (p50)

Sampling strategy: 3 windows × 200s per record (first / middle / last) to
control HFO non-stationarity. Recordings shorter than 600s fall back to
[0.0] only.

Scope: this is *pipeline internal self-consistency*, NOT biological validity.
See `docs/archive/hfo_detector_v2/v2_validation_contract.md` Layer A row.

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
    """Per-event ratio of pick_mean to side_mean, where side = pre [t0-dur, t0]
    ∪ post [t1, t1+dur]. Returns NaN if either pre or post side is empty
    (event near recording edge), or if pick is empty, or if side_mean <= 0.

    Convention: BQKDetector legacy symmetric-context rule (matches the
    side-rejection used by the production detector).
    """
    out = []
    n = len(env)
    for t0, t1 in events:
        dur = t1 - t0
        i_pre_s = max(0, int((t0 - dur) * fs))
        i_pre_e = int(t0 * fs)
        i_post_s = int(t1 * fs)
        i_post_e = min(n, int((t1 + dur) * fs))
        pre = env[i_pre_s:i_pre_e]
        post = env[i_post_s:i_post_e]
        pick = env[int(t0 * fs):int(t1 * fs)]
        if len(pre) == 0 or len(post) == 0 or len(pick) == 0:
            out.append(np.nan)
            continue
        s_mean = float(np.mean(np.concatenate([pre, post])))
        if s_mean <= 0:
            out.append(np.nan)
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


def _find_first_record_dir(raw_root: Path, subject: str) -> Path | None:
    """Walk Epilepsiae raw root inv*/pat_<subject>*/adm_*/rec_* and return the
    first rec_* directory (sorted) for the given subject id. Returns None when
    no matching subject directory exists.
    """
    if not raw_root.exists():
        return None
    for inv_dir in raw_root.iterdir():
        if not inv_dir.is_dir() or not inv_dir.name.startswith("inv"):
            continue
        for pat_dir in inv_dir.iterdir():
            if not pat_dir.is_dir() or not pat_dir.name.startswith(f"pat_{subject}"):
                continue
            for adm_dir in sorted(pat_dir.iterdir()):
                if not adm_dir.is_dir():
                    continue
                rec_dirs = sorted(
                    d for d in adm_dir.iterdir()
                    if d.is_dir() and d.name.startswith("rec_")
                )
                if rec_dirs:
                    return rec_dirs[0]
    return None


def check_determinism(subject_dir: Path) -> dict:
    """Run v2 detection twice on first record's first chunk; verify bit-identical
    output. PASS condition: count_match=True AND max_timestamp_diff_sec=0.0.

    Used as Layer A's `timestamp_jitter_p99` and `strong_chn_count_match` PASS
    gate (validation contract). Determinism is a property of the detector code,
    not of any single recording — but we anchor on real .data so the test
    exercises the actual production path (not a synthetic input).
    """
    from src.preprocessing import load_epilepsiae_block
    from src.hfo_detector import HFODetector, HFODetectionConfig

    head_paths = sorted(subject_dir.glob("*.head"))
    if not head_paths:
        return {
            "error": f"no .head found in {subject_dir} — check raw Epilepsiae path",
            "deterministic": False,
        }
    head_path = head_paths[0]
    data_path = head_path.with_suffix('.data')
    if not data_path.exists():
        return {"error": f"missing .data for {head_path.name}", "deterministic": False}

    pre = load_epilepsiae_block(
        data_path, head_path, reference="car",
        notch_freqs=[50.0, 100.0, 150.0, 200.0, 250.0],
        notch_filter_kind="fir_legacy",
    )
    cfg = HFODetectionConfig(
        rel_thresh=2.0, abs_thresh=2.0, side_thresh=2.0,
        min_gap_ms=20, min_last_ms=50, max_last_ms=200,
        chunk_sec=200, chunk_overlap_sec=0,
        legacy_align=True, use_gpu=True, n_jobs=1,
    )
    det = HFODetector(cfg)
    r1 = det.detect(pre)
    r2 = det.detect(pre)

    count_match = bool(np.array_equal(r1.events_count, r2.events_count))
    max_t_diff = 0.0
    for ev1, ev2 in zip(r1.events_by_channel, r2.events_by_channel):
        if ev1.shape != ev2.shape:
            count_match = False
            continue
        if len(ev1) > 0:
            max_t_diff = max(max_t_diff, float(np.max(np.abs(ev1 - ev2))))
    return {
        "head_path": str(head_path.name),
        "n_channels": int(pre.data.shape[0]),
        "rec_seconds": float(pre.data.shape[1] / pre.sfreq),
        "count_match": count_match,
        "max_timestamp_diff_sec": max_t_diff,
        "deterministic": count_match and max_t_diff == 0.0,
    }


def extract_layer_a_per_subject(subject_dir: Path, output_dir: Path) -> dict:
    """Iterate v2 detection *_gpu.npz under subject_dir and recompute envelope+
    threshold metrics on first/middle/last 200s windows of each record.

    ⚠️ DETECTOR-PARAM COUPLING: The BQKDetector instantiation below MUST match
    the params used by the producing detection script (currently
    scripts/run_hfo_detection.py Epilepsiae path:
      freqband=(80,250), subband_width=20,
      rel_thresh=2.0, abs_thresh=2.0, side_thresh=2.0,
      min_gap=20, min_last=50, max_last=200,
      legacy_align=True, notch_filter_kind='fir_legacy',
      notch_freqs=[50,100,150,200,250].
    If the detection script changes any of these, this Layer A recompute
    silently drifts — propose persisting them in *_gpu.npz metadata before
    Phase 3 scales up. See v2_validation_contract.md for Layer A scope.
    """
    from src.preprocessing import load_epilepsiae_block
    from src.utils.bqk_utils import BQKDetector

    metrics_per_record = []
    skipped = []
    for gpu_path in sorted(subject_dir.glob("*_gpu.npz")):
        head_path = gpu_path.parent / (gpu_path.stem.replace("_gpu", "") + ".head")
        data_path = head_path.with_suffix(".data")
        if not (head_path.exists() and data_path.exists()):
            skipped.append({"record": gpu_path.name, "error": "raw .head/.data not adjacent"})
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
        ch_agg = {ch: {"n_events": 0, "in_band_counts": [], "ratios": [], "margins": []} for ch in chns}
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
                ch_agg[ch]["in_band_counts"].append(durs * len(evts_local))  # weighted by count
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
            durs_total = float(sum(agg["in_band_counts"])) / agg["n_events"]
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

    return {"records": metrics_per_record, "skipped": skipped}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g. 635")
    p.add_argument("--detection-root", default="results/hfo_detector_v2")
    p.add_argument("--output-dir", default="results/hfo_detector_v2/validation")
    p.add_argument(
        "--check-determinism", action="store_true",
        help="Run determinism / twice-run jitter check instead of metric extraction.",
    )
    p.add_argument(
        "--raw-root", type=Path, default=Path("/mnt/epilepsia_data"),
        help="Epilepsiae raw root with inv*/pat_*/adm_*/rec_*/*.{head,data} layout.",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check_determinism:
        rec_dir = _find_first_record_dir(args.raw_root, args.subject)
        if rec_dir is None:
            print(
                f"ERROR: no rec_* dir found under {args.raw_root} for subject {args.subject}",
                file=sys.stderr,
            )
            sys.exit(1)
        det_res = check_determinism(rec_dir)
        det_res["raw_record_dir"] = str(rec_dir)
        out = out_dir / f"layer_a_determinism_{args.subject}.json"
        out.write_text(json.dumps(det_res, indent=2))
        print(json.dumps(det_res, indent=2))
        print(f"wrote {out}")
        sys.exit(0 if det_res.get("deterministic") else 1)

    subject_dir = Path(args.detection_root) / args.subject
    if not subject_dir.exists():
        print(f"ERROR: subject_dir does not exist: {subject_dir}", file=sys.stderr)
        sys.exit(1)
    res = extract_layer_a_per_subject(subject_dir, out_dir)
    out = out_dir / f"layer_a_{args.subject}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    n_records = len(res.get("records", []))
    n_skipped = len(res.get("skipped", []))
    if n_records == 0 and n_skipped == 0:
        print(
            f"ERROR: no work done — no *_gpu.npz found under {subject_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    if n_records == 0 and n_skipped > 0:
        print(
            f"WARNING: all {n_skipped} record(s) under {subject_dir} were skipped; "
            "see 'skipped' field in JSON for reasons",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
