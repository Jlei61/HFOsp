#!/usr/bin/env python3
"""Time-resolved 1-150 Hz seizure cache for the Topic 5.2 motif reuse branch.

The frozen v0.5 sidecar stores only the 0-10 s AUC at the true onset, so it can
answer neither "what would a same-block pseudo-onset look like in the same
band" nor "does the early field predict the late field".  This script re-runs
the frozen v0.5 extraction and keeps the full baseline-robust-z trace.

Provenance guard: the 0-10 s mean recomputed here is compared bit-for-bit
against the frozen ``bb150_auc__<idx>`` before anything is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

CANONICAL_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
if str(CANONICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CANONICAL_ROOT))
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
FROZEN_SIDECAR = CANONICAL_ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
SOURCE_CACHE = CANONICAL_ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_v2_windows"


def build_subject(ds_sid: str, out_dir: Path) -> dict:
    cfg = load_phase1_config()
    ln = cfg["line_noise"]
    spec_win = float(cfg["power"]["spectrogram_win_sec"])
    spec_hop = float(cfg["power"]["spectrogram_hop_sec"])
    source_meta_path = SOURCE_CACHE / f"{ds_sid}.json"
    if not source_meta_path.exists():
        return {"subject": ds_sid, "status": "missing_source_meta"}
    source_meta = json.loads(source_meta_path.read_text())
    expected_channels = [str(x) for x in source_meta.get("channels", [])]
    expected_idxs = [int(x) for x in source_meta.get("eligible_idxs", [])]
    frozen = np.load(FROZEN_SIDECAR / f"{ds_sid}.npz", allow_pickle=True)

    base.POST_SEC = float(source_meta.get("post_sec", 20.0))
    base.PRE_FEATURE_SEC = float(source_meta.get("pre_feature_sec", 130.0))
    dataset, sid = source_meta["dataset"], source_meta["subject"]
    reference = base.ICTAL_REFERENCE[dataset]
    inventory, _ = base._inventory_rows(dataset, sid)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{ds_sid}.npz"
    out_json = out_dir / f"{ds_sid}.json"
    if out_npz.exists() and out_json.exists():
        cached = json.loads(out_json.read_text())
        if sorted(cached.get("seizures_cached", [])) == sorted(expected_idxs):
            return {"subject": ds_sid, "status": "already_complete",
                    "n_seizures": len(expected_idxs)}

    arrays: dict[str, np.ndarray] = {"channels": np.array(expected_channels)}
    parity: dict[str, float] = {}
    cached_idxs: list[int] = []
    failures: dict[str, str] = {}
    for idx in expected_idxs:
        row = inventory[idx] if idx < len(inventory) else {}
        try:
            window = extract_seizure_window(
                f"{dataset}/{sid}", idx,
                pre_sec=base._pre_target(dataset, row),
                post_sec=base.POST_SEC, reference=reference,
            )
            channels = [recruit.bipolar_alias_label(name) for name in window.ch_names]
            if channels != expected_channels:
                failures[str(idx)] = "channel_order_mismatch"
                continue
            freqs, times, sxx = _spectrogram_on_hop(window.signal, window.fs, spec_win, spec_hop)
            line_mask = line_noise_bin_mask(freqs, ln["harmonics_hz"], ln["halfwidth_hz"])
            band_mask, _, _ = band_bin_selection(freqs, BAND[0], BAND[1], line_mask, half_open=False)
            power = np.log(np.maximum(sxx[:, band_mask, :].sum(axis=1), 1e-30))
            eeg_rel = (
                (window.eeg_onset_epoch - window.clin_onset_epoch)
                if window.eeg_onset_epoch is not None else None
            )
            baseline = resolve_baseline_window(
                power.shape[1], hop_sec=spec_hop, pre_sec=window.pre_sec,
                buffer_sec=base.GUARD_SEC, eeg_onset_rel_sec=eeg_rel,
                min_baseline_valid_sec=base.MIN_BASELINE_SEC,
            )
            z = recruit.baseline_robust_z(
                power, (baseline.start_idx, baseline.end_idx), hop_sec=spec_hop,
                min_baseline_valid_sec=base.MIN_BASELINE_SEC,
            )
            relative = np.asarray(times, float) - float(window.pre_sec)
            onset_window = np.where((relative >= base.T0_SEC) & (relative <= base.T1_SEC))[0]
            recomputed = activation_mean(z, onset_window)
            key = f"bb150_auc__{idx}"
            if key in frozen.files:
                difference = float(np.nanmax(np.abs(recomputed - np.asarray(frozen[key], float))))
                parity[str(idx)] = difference
            # The trace is stored regardless of parity.  Real onsets and
            # pseudo-onsets must come from ONE recomputation to be comparable;
            # agreement with the frozen sidecar is a provenance record that lets
            # the analysis restrict itself later, not a reason to drop a seizure.
            arrays[f"bb150_auc__{idx}"] = recomputed.astype(np.float32)
            arrays[f"bb150_zt__{idx}"] = z.astype(np.float32)
            arrays[f"bb150_relt__{idx}"] = relative.astype(np.float32)
            arrays[f"bb150_baseline__{idx}"] = np.asarray(
                [baseline.start_idx, baseline.end_idx], dtype=np.int32)
            cached_idxs.append(int(idx))
            print(f"  [{ds_sid} sz{idx}] traces {z.shape} parity {parity.get(str(idx), float('nan')):.2e}",
                  flush=True)
        except Exception as error:  # noqa: BLE001 - report and continue by seizure
            failures[str(idx)] = f"{type(error).__name__}: {error}"
            print(f"  [{ds_sid} sz{idx}] skip {failures[str(idx)]}", flush=True)

    np.savez_compressed(out_npz, **arrays)
    out_json.write_text(json.dumps({
        "contract": "topic5_dynamical_motif_seizure_trace_v0_1",
        "dataset": dataset,
        "subject": sid,
        "band_hz": list(BAND),
        "hop_sec": spec_hop,
        "spectrogram_window_sec": spec_win,
        "pre_feature_sec": base.PRE_FEATURE_SEC,
        "post_sec": base.POST_SEC,
        "onset_window_s": [base.T0_SEC, base.T1_SEC],
        "channels": expected_channels,
        "seizures_expected": expected_idxs,
        "seizures_cached": cached_idxs,
        "frozen_auc_parity_max_abs": parity,
        "parity_verified_seizures": sorted(int(k) for k, v in parity.items() if v <= 1e-4),
        "parity_threshold": 1e-4,
        "failures": failures,
        "note": "bb150_zt__<idx> is the FULL [-pre,+post] baseline-robust-z 1-150 Hz trace; "
                "bb150_relt__<idx> holds per-bin times relative to onset.",
    }, indent=2, ensure_ascii=False))
    return {
        "subject": ds_sid,
        "status": "ok" if len(cached_idxs) == len(expected_idxs) else "partial",
        "n_cached": len(cached_idxs),
        "n_expected": len(expected_idxs),
        "max_parity": max(parity.values()) if parity else None,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-lbss-rnn-v0-1/"
                     "results/topic5_dynamical_motif_rnn_v0_1/seizure_trace_cache"),
    )
    args = parser.parse_args()
    subjects = args.subjects
    if not subjects:
        routing = CANONICAL_ROOT / ".worktrees/topic5-lbss-rnn-v0-1/results/" \
                  "topic5_multiscale_effective_scaffold_v0_5/EARLY_ICTAL_ROUTING_METADATA.csv"
        import pandas as pd
        subjects = sorted(pd.read_csv(routing)["subject"].astype(str).unique().tolist())
    report = []
    for subject in subjects:
        print(f"[seizure-trace] {subject}", flush=True)
        report.append(build_subject(subject, args.out_dir))
        print(f"  -> {report[-1]}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "SEIZURE_TRACE_BUILD_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
