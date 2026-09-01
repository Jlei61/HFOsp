#!/usr/bin/env python3
"""Audit timing and the stored smoothed observation spectrum without SNN reruns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results/topic4_sef_hfo/data_driven_dual_core_ood"
DEFAULT_TARGET = ROOT / (
    "results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/"
    "shaft_aware_patient_training_target.npz"
)
ARMS = {
    "Node": "frozen_dualcore_node",
    "Node+EE": "frozen_dualcore_ee",
    "Node+EtoI": "frozen_dualcore_etoi",
    "Node+EE+EtoI": "frozen_dualcore_both",
}


def _quantiles(values: list[float], probabilities=(0.05, 0.5, 0.95)) -> dict:
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "quantiles": {str(p): None for p in probabilities}}
    return {
        "n": int(len(array)),
        "quantiles": {
            str(p): float(value)
            for p, value in zip(probabilities, np.quantile(array, probabilities))
        },
    }


def _onset_spans(onsets: np.ndarray, *, scale_to_ms: float) -> list[float]:
    spans = []
    for row in np.asarray(onsets, float):
        finite = np.isfinite(row)
        if int(finite.sum()) >= 3:
            spans.append(float(np.ptp(row[finite]) * scale_to_ms))
    return spans


def _event_spectrum(
    envelope: np.ndarray, *, dt_ms: float, onset_ms: float, samples: int = 128,
) -> tuple[float, float, float] | None:
    values = np.asarray(envelope, float)
    if values.ndim != 2:
        raise ValueError("contact envelope must be two-dimensional")
    if values.shape[0] == 15:
        values = values.T
    center = int(round(float(onset_ms) / float(dt_ms)))
    left = max(0, center - samples // 4)
    right = min(len(values), left + samples)
    left = max(0, right - samples)
    if right - left != samples:
        return None
    segment = values[left:right] - values[left:right].mean(axis=0, keepdims=True)
    frequency, power = welch(
        segment.T, fs=1000.0 / dt_ms, nperseg=samples, noverlap=0,
        axis=1, detrend=False,
    )
    power = np.mean(power, axis=0)
    carrier = (frequency >= 20.0) & (frequency <= 150.0)
    carrier_power = power[carrier]
    if not np.any(carrier) or float(carrier_power.sum()) <= 0.0:
        return None
    carrier_frequency = frequency[carrier]
    peak = float(carrier_frequency[np.argmax(carrier_power)])
    centroid = float(
        np.sum(carrier_frequency * carrier_power) / np.sum(carrier_power)
    )
    high = float(np.sum(power[(frequency >= 30.0) & (frequency <= 80.0)]))
    low = float(np.sum(power[(frequency >= 5.0) & (frequency < 30.0)]))
    return peak, centroid, high / max(low, np.finfo(float).tiny)


def _patient_timing(target_path: Path) -> dict:
    with np.load(target_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["patient_train_onsets"], float)
        labels = np.asarray(loaded["patient_train_old_labels"], int)
    return {
        f"mode_{mode}": _quantiles(
            _onset_spans(onsets[labels == mode], scale_to_ms=1000.0)
        )
        for mode in (0, 1)
    }


def _arm_audit(result_root: Path, candidate_id: str) -> dict:
    per_network = []
    for seed in range(2441, 2453):
        path = result_root / "pathway/workers" / f"{candidate_id}_seed_{seed}.npz"
        with np.load(path, allow_pickle=False) as loaded:
            onsets = np.asarray(loaded["onsets"], float)
            event_on = np.asarray(loaded["event_t_on_ms"], float)
            returned = np.asarray(loaded["event_returned"], bool)
            envelope = np.asarray(loaded["contact_envelope"], float)
            dt_ms = float(loaded["contact_envelope_dt_ms"])
        readable = np.sum(np.isfinite(onsets), axis=1) >= 3
        selected = returned & readable
        spans = _onset_spans(onsets[selected], scale_to_ms=1.0)
        spectra = [
            _event_spectrum(envelope, dt_ms=dt_ms, onset_ms=onset)
            for onset in event_on[selected]
        ]
        spectra = [row for row in spectra if row is not None]
        per_network.append({
            "seed": seed,
            "n_returned_readable": int(np.sum(selected)),
            "median_onset_span_ms": float(np.median(spans)) if spans else None,
            "median_peak_hz": float(np.median([row[0] for row in spectra])),
            "median_centroid_20_150_hz": float(np.median([row[1] for row in spectra])),
            "median_power_ratio_30_80_over_5_30": float(np.median([
                row[2] for row in spectra
            ])),
        })
    summary = {}
    for key in (
        "median_onset_span_ms", "median_peak_hz", "median_centroid_20_150_hz",
        "median_power_ratio_30_80_over_5_30",
    ):
        summary[key] = _quantiles(
            [row[key] for row in per_network], probabilities=(0.1, 0.5, 0.9)
        )
    return {"summary_across_network_medians": summary, "per_network": per_network}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--patient-target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    output = args.output or result_root / "temporal_carrier_audit.json"
    payload = {
        "status": "DUAL_CORE_TEMPORAL_CARRIER_AUDIT_COMPLETE",
        "no_new_simulation": True,
        "patient_absolute_onset_span_ms": _patient_timing(args.patient_target.resolve()),
        "model_pathway_arms": {
            arm: _arm_audit(result_root, candidate_id)
            for arm, candidate_id in ARMS.items()
        },
        "spectral_contract": {
            "readout": (
                "virtual-contact firing-density envelope without an additional bandpass, "
                "but after 2 ms binning and 5 ms Gaussian temporal smoothing"
            ),
            "not_lfp_or_seeg": True,
            "raw_spike_or_synaptic_current_carrier_evaluable": False,
            "sampling_ms": 2.0,
            "smoothing_ms": 5.0,
            "spatial_gaussian_footprint_sigma_mm": 0.25,
            "window_ms": 256.0,
            "window_alignment": "64 ms before to 192 ms after detector onset",
            "frequency_resolution_hz": 3.90625,
            "theoretical_gaussian_amplitude_retention": {
                "23.4375_hz": 0.7625579441684482,
                "60_hz": 0.16922454248244995,
                "80_hz": 0.04249905628536254
            },
            "interpretation": (
                "diagnostic of the stored smoothed observation envelope only; the 5 ms "
                "Gaussian strongly attenuates 60-80 Hz, so this artifact cannot establish "
                "or exclude an intrinsic spike-rate or synaptic-current carrier"
            ),
        },
        "claim_boundary": (
            "absolute recruitment timing and smoothed model-envelope spectrum only; "
            "raw model carrier, clinical waveform and HFO reproduction remain untested"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
