#!/usr/bin/env python3
"""Reapply the frozen oscillatory-state onset/gate to completed qI--M runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_global_recruited_oscillation import (  # noqa: E402
    classify_global_recruited_oscillation,
    contact_rhythm_metrics,
    detect_sustained_high_state_onset,
    recruitment_duty_metrics,
    state_rate_metrics,
)


ONSET_CONTRACT = {
    "version": "oscillatory_median_v1",
    "threshold_hz": 120.0,
    "block_ms": 20.0,
    "forward_window_ms": 300.0,
    "isolated_bursts_are_onsets": False,
    "oscillatory_troughs_are_allowed": True,
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def reclassify(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("status") != "SPATIAL_ZQIM_HYBRID_CANARY_COMPLETE":
        raise ValueError(f"not a spatial qI--M canary: {json_path}")
    npz_path = json_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as data:
        dt_ms = float(data["lfp_dt_ms"])
        rate = np.asarray(data["rate_E_hz"], float)
        onset = detect_sustained_high_state_onset(rate, dt_ms=dt_ms)
        payload["scientific_onset_ms"] = onset
        payload["scientific_onset_contract"] = dict(ONSET_CONTRACT)
        for key in ("state_rate", "global_recruitment", "contact_rhythm",
                    "classification", "metric_error"):
            payload.pop(key, None)
        if onset is None:
            payload["verdict"] = "NO_SUSTAINED_HIGH_STATE_WITHIN_CANARY"
        else:
            recruitment = {
                "time_ms": np.asarray(data["full_field_time_ms"], float),
                "active_neuron_fraction": np.asarray(
                    data["active_neuron_fraction_20ms"], float),
                "recruited_spatial_fraction": np.asarray(
                    data["recruited_spatial_fraction_1mm"], float),
            }
            try:
                rates = state_rate_metrics(rate, dt_ms=dt_ms, onset_ms=onset)
                rec_metrics = recruitment_duty_metrics(
                    recruitment, onset_ms=onset)
                rhythm = contact_rhythm_metrics(
                    np.asarray(data["lfp_trace"], float),
                    dt_ms=dt_ms, onset_ms=onset)
                classification = classify_global_recruited_oscillation(
                    onset_ms=onset, rates=rates,
                    recruitment=rec_metrics, rhythm=rhythm)
                payload["state_rate"] = rates
                payload["global_recruitment"] = rec_metrics
                payload["contact_rhythm"] = _json_safe(rhythm)
                payload["classification"] = classification
                payload["verdict"] = classification["status"]
            except ValueError as error:
                payload["verdict"] = "INCOMPLETE_POST_ONSET_WINDOW"
                payload["metric_error"] = str(error)
    atomic_write_json(_json_safe(payload), str(json_path))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="*.json")
    args = parser.parse_args()
    root = Path(args.input_dir)
    if not root.is_absolute():
        root = ROOT / root
    completed = 0
    verdicts = {}
    for path in sorted(root.rglob(args.pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "SPATIAL_ZQIM_HYBRID_CANARY_COMPLETE":
            continue
        updated = reclassify(path)
        completed += 1
        verdict = str(updated["verdict"])
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    print(json.dumps({
        "status": "SPATIAL_ZQIM_RECLASSIFICATION_COMPLETE",
        "n_runs": completed,
        "verdicts": verdicts,
        "scientific_onset_contract": ONSET_CONTRACT,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
