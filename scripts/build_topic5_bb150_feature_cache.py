#!/usr/bin/env python3
"""Build a lightweight Topic 5 1-150 Hz broadband activation sidecar cache.

This does not replace the legacy T0 cache's `bb_auc` (1-45 Hz).  It writes only
the fields needed by the axis-alignment runner:

  - channels
  - bb150_auc__<seizure_idx>

The sidecar avoids repeatedly rewriting the large v2 cache that contains full
HFA/BB traces.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

from scripts import build_topic5_t0_feature_cache as base  # noqa: E402
from src import topic5_ictal_recruitment as recruit  # noqa: E402
from src.topic5_ictal_recruitment import _spectrogram_on_hop  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window, resolve_baseline_window  # noqa: E402
from src.topic5_t0_features import activation_mean  # noqa: E402
from src.topic5_v2_band_scan import (  # noqa: E402
    band_bin_selection,
    line_noise_bin_mask,
    load_phase1_config,
)


BAND = (1.0, 150.0)


def _onset_window(times: np.ndarray, pre_sec: float) -> np.ndarray:
    rel = np.asarray(times, float) - float(pre_sec)
    return np.where((rel >= base.T0_SEC) & (rel <= base.T1_SEC))[0]


def _complete(npz_path: Path, json_path: Path, expected_idxs: list[int]) -> bool:
    if not npz_path.exists() or not json_path.exists() or not expected_idxs:
        return False
    meta = json.loads(json_path.read_text())
    if not meta.get("line_noise_masked_1_150", False):
        return False
    data = np.load(npz_path, allow_pickle=True)
    return all(f"bb150_auc__{idx}" in data.files for idx in expected_idxs)


def _load_existing(npz_path: Path) -> dict:
    if npz_path.exists():
        return dict(np.load(npz_path, allow_pickle=True))
    return {}


def _write(out_npz: Path, out_json: Path, arrays: dict, meta: dict) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def build_subject(ds_sid: str, *, source_cache: Path, out_dir: Path) -> dict:
    cfg = load_phase1_config()
    ln = cfg["line_noise"]
    spec_win = float(cfg["power"]["spectrogram_win_sec"])
    spec_hop = float(cfg["power"]["spectrogram_hop_sec"])
    source_meta_f = source_cache / f"{ds_sid}.json"
    if not source_meta_f.exists():
        return {"subject_id": ds_sid, "status": "missing_source_meta"}
    source_meta = json.loads(source_meta_f.read_text())
    expected_channels = [str(x) for x in source_meta.get("channels", [])]
    expected_idxs = [int(x) for x in source_meta.get("eligible_idxs", [])]
    out_npz = out_dir / f"{ds_sid}.npz"
    out_json = out_dir / f"{ds_sid}.json"

    if _complete(out_npz, out_json, expected_idxs):
        return {"subject_id": ds_sid, "status": "already_complete", "n_seizures": len(expected_idxs)}

    arrays = _load_existing(out_npz)
    if out_json.exists():
        old_meta = json.loads(out_json.read_text())
        if not old_meta.get("line_noise_masked_1_150", False):
            arrays = {k: v for k, v in arrays.items() if not str(k).startswith("bb150_auc__")}
    arrays["channels"] = np.array(expected_channels)
    cached = {
        int(k.split("__", 1)[1])
        for k in arrays
        if isinstance(k, str) and k.startswith("bb150_auc__")
    }
    meta = {
        "dataset": source_meta["dataset"],
        "subject": source_meta["subject"],
        "source_cache": str(source_cache),
        "band_broad_1_150": list(BAND),
        "hop_sec": base.HOP,
        "t_window": [base.T0_SEC, base.T1_SEC],
        "post_sec": base.POST_SEC,
        "pre_feature_sec": base.PRE_FEATURE_SEC,
        "channels": expected_channels,
        "eligible_idxs": expected_idxs,
        "feature": "bb150_auc_0_10s (mean baseline-robust-z 1-150Hz over [0,10]s)",
        "line_noise_masked_1_150": True,
        "note": "sidecar cache for Topic5 axis-alignment sensitivity; legacy bb_auc remains 1-45Hz",
    }

    dataset, sid = source_meta["dataset"], source_meta["subject"]
    ref = base.ICTAL_REFERENCE[dataset]
    inv_rows, _ = base._inventory_rows(dataset, sid)
    added = 0
    for idx in expected_idxs:
        if idx in cached:
            continue
        inv = inv_rows[idx] if idx < len(inv_rows) else {}
        try:
            sw = extract_seizure_window(
                f"{dataset}/{sid}",
                idx,
                pre_sec=base._pre_target(dataset, inv),
                post_sec=base.POST_SEC,
                reference=ref,
            )
            channels = [recruit.bipolar_alias_label(c) for c in sw.ch_names]
            if channels != expected_channels:
                print(f"  [{ds_sid} sz{idx}] channel order mismatch, skip", flush=True)
                continue
            freqs, times, sxx = _spectrogram_on_hop(sw.signal, sw.fs, spec_win, spec_hop)
            line_mask = line_noise_bin_mask(freqs, ln["harmonics_hz"], ln["halfwidth_hz"])
            band_mask, eff_frac, n_band_bins = band_bin_selection(
                freqs, BAND[0], BAND[1], line_mask, half_open=False
            )
            if not band_mask.any():
                raise ValueError(f"no FFT bins inside line-noise-masked band={BAND}")
            power = np.log(np.maximum(sxx[:, band_mask, :].sum(axis=1), 1e-30))
            eeg_rel = (
                (sw.eeg_onset_epoch - sw.clin_onset_epoch)
                if sw.eeg_onset_epoch is not None
                else None
            )
            bl = resolve_baseline_window(
                power.shape[1],
                hop_sec=spec_hop,
                pre_sec=sw.pre_sec,
                buffer_sec=base.GUARD_SEC,
                eeg_onset_rel_sec=eeg_rel,
                min_baseline_valid_sec=base.MIN_BASELINE_SEC,
            )
            z = recruit.baseline_robust_z(
                power,
                (bl.start_idx, bl.end_idx),
                hop_sec=spec_hop,
                min_baseline_valid_sec=base.MIN_BASELINE_SEC,
            )
            arrays[f"bb150_auc__{idx}"] = activation_mean(
                z, _onset_window(times, sw.pre_sec)
            ).astype(np.float32)
            meta.setdefault("qc", {})[str(idx)] = {
                "line_noise_masked": True,
                "n_band_bins": int(n_band_bins),
                "effective_band_fraction": float(eff_frac),
            }
            added += 1
            _write(out_npz, out_json, arrays, meta)
            print(f"  [{ds_sid} sz{idx}] cached bb150", flush=True)
        except Exception as e:
            print(f"  [{ds_sid} sz{idx}] bb150 skip {type(e).__name__}: {e}", flush=True)

    n_cached = sum(1 for idx in expected_idxs if f"bb150_auc__{idx}" in arrays)
    _write(out_npz, out_json, arrays, meta)
    status = "ok" if n_cached == len(expected_idxs) and n_cached > 0 else "partial"
    return {"subject_id": ds_sid, "status": status, "n_cached": n_cached,
            "n_expected": len(expected_idxs), "n_added": added}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--source-cache", default="results/topic5_ictal_recruitment/t0_feature_cache_v2_windows")
    ap.add_argument("--out-dir", default="results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150")
    ap.add_argument("--post-sec", type=float, default=20.0)
    ap.add_argument("--pre-feature-window", type=float, default=130.0)
    args = ap.parse_args()

    base.POST_SEC = float(args.post_sec)
    base.PRE_FEATURE_SEC = float(args.pre_feature_window)
    source_cache = Path(args.source_cache)
    out_dir = Path(args.out_dir)

    print(f"[bb150-sidecar] subjects={len(args.subjects)} band={BAND} out={out_dir}", flush=True)
    for ds_sid in args.subjects:
        print(f"[bb150-sidecar] {ds_sid} ...", flush=True)
        res = build_subject(ds_sid, source_cache=source_cache, out_dir=out_dir)
        print(f"  -> {res}", flush=True)


if __name__ == "__main__":
    main()
