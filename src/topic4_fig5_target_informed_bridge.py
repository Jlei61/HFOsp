"""Target-informed early-energy bridge for the data-driven Z/M SNN.

This module contains only deterministic transforms and scores.  Clinical data
loading and SNN execution live in scripts so tests can audit the scientific
contract without touching either data source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr


SCHEMA_ID = "topic4_fig5_target_informed_zm_bridge_v1"


def lse(values: Sequence[float], tau: float = 0.25) -> float:
    """Smooth maximum with zero offset for equal inputs."""
    x = np.asarray(values, float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("values must be a finite non-empty vector")
    peak = float(np.max(x))
    return float(peak + tau * np.log(np.mean(np.exp((x - peak) / tau))))


def exact_contact_reorder(values, source_names, target_names):
    values = np.asarray(values)
    source = [str(v) for v in source_names]
    target = [str(v) for v in target_names]
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ValueError("contact names must be unique")
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise ValueError(f"exact contact mismatch: missing={missing}, extra={extra}")
    if values.shape[-1] != len(source):
        raise ValueError("last value axis must match source_names")
    index = {name: i for i, name in enumerate(source)}
    return values[..., [index[name] for name in target]]


def _band_mask(freqs, band_hz):
    lo, hi = map(float, band_hz)
    return (freqs >= lo) & (freqs <= hi)


def log_band_power(window, dt_ms, band_hz=(10.0, 150.0)):
    """Per-contact natural-log band power for a time-by-contact window."""
    x = np.asarray(window, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 8:
        raise ValueError("window must be time-by-contact with at least 8 samples")
    fs = 1000.0 / float(dt_ms)
    hi = min(float(band_hz[1]), np.nextafter(fs / 2.0, 0.0))
    lo = float(band_hz[0])
    if not hi > lo:
        raise ValueError("band is outside the model sampling range")
    freqs, psd = welch(
        x,
        fs=fs,
        axis=0,
        nperseg=x.shape[0],
        noverlap=0,
        detrend="linear",
        scaling="density",
    )
    use = _band_mask(freqs, (lo, hi))
    if not np.any(use):
        raise ValueError("no frequency bins in requested band")
    power = np.trapz(psd[use], freqs[use], axis=0)
    return np.log(np.maximum(power, np.finfo(float).tiny))


def spectral_centroid(window, dt_ms, band_hz=(10.0, 150.0)):
    """Per-contact spectral centroid in a fixed band."""
    x = np.asarray(window, float)
    if x.ndim == 1:
        x = x[:, None]
    fs = 1000.0 / float(dt_ms)
    hi = min(float(band_hz[1]), np.nextafter(fs / 2.0, 0.0))
    freqs, psd = welch(
        x,
        fs=fs,
        axis=0,
        nperseg=x.shape[0],
        noverlap=0,
        detrend="linear",
        scaling="density",
    )
    use = _band_mask(freqs, (float(band_hz[0]), hi))
    weighted = psd[use]
    denom = np.sum(weighted, axis=0)
    return np.divide(
        np.sum(freqs[use, None] * weighted, axis=0),
        denom,
        out=np.full(denom.shape, np.nan),
        where=denom > 0,
    )


def nonoverlap_log_power_windows(trace, dt_ms, window_ms=500.0,
                                 band_hz=(10.0, 150.0)):
    trace = np.asarray(trace, float)
    if trace.ndim != 2:
        raise ValueError("trace must be time-by-contact")
    n = int(round(float(window_ms) / float(dt_ms)))
    if n < 8:
        raise ValueError("spectral window is too short")
    count = trace.shape[0] // n
    if count < 2:
        raise ValueError("paired baseline needs at least two non-overlapping windows")
    return np.asarray([
        log_band_power(trace[i * n:(i + 1) * n], dt_ms, band_hz)
        for i in range(count)
    ])


def robust_z_against_reference(reference_log_power, values):
    """Contact-wise robust-z using only an independent reference trajectory."""
    ref = np.asarray(reference_log_power, float)
    x = np.asarray(values, float)
    if ref.ndim != 2 or x.shape[-1] != ref.shape[1]:
        raise ValueError("reference must be windows-by-contact and align with values")
    med = np.median(ref, axis=0)
    mad = 1.4826 * np.median(np.abs(ref - med), axis=0)
    if np.any(~np.isfinite(mad)) or np.any(mad <= 1e-12):
        raise ValueError("reference baseline has unresolved contact scale")
    return (x - med) / mad, {"median": med, "mad": mad}


def smooth_rate(rate_hz, dt_ms, window_ms=20.0):
    rate = np.asarray(rate_hz, float)
    n = max(1, int(round(float(window_ms) / float(dt_ms))))
    return np.convolve(rate, np.ones(n, float) / n, mode="same")


def _window_slice(values, dt_ms, start_ms, width_ms):
    start = int(round(float(start_ms) / float(dt_ms)))
    stop = int(round((float(start_ms) + float(width_ms)) / float(dt_ms)))
    if start < 0 or stop > len(values) or stop <= start:
        return None
    return np.asarray(values)[start:stop]


def _recruitment_duty(time_ms, f_e, f_sheet, start_ms, width_ms,
                      activity_threshold):
    time_ms = np.asarray(time_ms, float)
    use = (time_ms >= float(start_ms)) & (
        time_ms < float(start_ms) + float(width_ms))
    if not np.any(use):
        return None
    joint = (np.asarray(f_e)[use] >= float(activity_threshold)) & (
        np.asarray(f_sheet)[use] >= float(activity_threshold))
    return float(np.mean(joint))


@dataclass(frozen=True)
class ReadoutWindow:
    start_ms: float
    stop_ms: float
    joint_duty: float
    contact_centroid_base_hz: float
    contact_centroid_read_hz: float


def select_state_defined_readout(
    *,
    trace,
    dt_ms,
    full_field_time_ms,
    active_fraction,
    spatial_fraction,
    t_ictal_ms,
    baseline_trace,
    window_ms=500.0,
    step_ms=25.0,
    activity_threshold=0.5,
    duty_threshold=0.8,
    frequency_shift_hz=5.0,
    frequency_ratio=1.25,
    band_hz=(10.0, 150.0),
):
    """Choose the earliest model-qualified window; patient data are not inputs."""
    trace = np.asarray(trace, float)
    base_centroid = float(np.nanmedian(spectral_centroid(
        baseline_trace, dt_ms, band_hz)))
    latest = trace.shape[0] * float(dt_ms) - float(window_ms)
    if latest < float(t_ictal_ms):
        return None
    for start in np.arange(float(t_ictal_ms), latest + 1e-9, float(step_ms)):
        duty = _recruitment_duty(
            full_field_time_ms, active_fraction, spatial_fraction, start,
            window_ms, activity_threshold)
        if duty is None or duty < float(duty_threshold):
            continue
        window = _window_slice(trace, dt_ms, start, window_ms)
        read_centroid = float(np.nanmedian(spectral_centroid(window, dt_ms, band_hz)))
        if (read_centroid - base_centroid >= float(frequency_shift_hz)
                and read_centroid / max(base_centroid, 1e-12) >= float(frequency_ratio)):
            return ReadoutWindow(
                start_ms=float(start),
                stop_ms=float(start + window_ms),
                joint_duty=float(duty),
                contact_centroid_base_hz=base_centroid,
                contact_centroid_read_hz=read_centroid,
            )
    return None


def bootstrap_patient_summary(pre, early, *, draws=4096, seed=20260821):
    """Freeze seizure-level target vectors and uncertainty without event pooling."""
    pre = np.asarray(pre, float)
    early = np.asarray(early, float)
    if pre.shape != early.shape or pre.ndim != 2 or pre.shape[0] < 3:
        raise ValueError("pre and early must be matching seizure-by-contact matrices")
    if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(early)):
        raise ValueError("target matrices must be complete and finite")
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, pre.shape[0], size=(int(draws), pre.shape[0]))
    pre_boot = np.median(pre[index], axis=1)
    early_boot = np.median(early[index], axis=1)
    inc = early - pre
    inc_boot = np.median(inc[index], axis=1)
    def _summary(values, boot):
        return {
            "median": np.median(values, axis=0),
            "q025": np.quantile(boot, 0.025, axis=0),
            "q975": np.quantile(boot, 0.975, axis=0),
            "bootstrap_iqr": np.subtract(*np.quantile(boot, [0.75, 0.25], axis=0)),
        }
    return {
        "pre": _summary(pre, pre_boot),
        "early": _summary(early, early_boot),
        "increment": _summary(inc, inc_boot),
        "global_early_per_seizure": np.median(early, axis=1),
        "positive_fraction_per_seizure": np.mean(early > 0, axis=1),
        "contact_iqr_per_seizure": np.subtract(
            np.quantile(early, 0.75, axis=1), np.quantile(early, 0.25, axis=1)),
    }


def shaft_balanced_scaled_l1(model, target, scale, shaft_ids, tau=0.25):
    model = np.asarray(model, float)
    target = np.asarray(target, float)
    scale = np.asarray(scale, float)
    shafts = np.asarray(shaft_ids).astype(str)
    if not (model.shape == target.shape == scale.shape == shafts.shape):
        raise ValueError("contact vectors and shaft ids must align")
    errors = []
    for shaft in sorted(set(shafts.tolist())):
        use = shafts == shaft
        errors.append(float(np.mean(np.abs(model[use] - target[use])
                                    / np.maximum(scale[use], 1e-6))))
    return lse(errors, tau=tau), dict(zip(sorted(set(shafts.tolist())), errors))


def score_energy_field(model_pre, model_early, target: Mapping[str, object], shaft_ids):
    """Score a model readout against a frozen patient summary."""
    model_pre = np.asarray(model_pre, float)
    model_early = np.asarray(model_early, float)
    early_target = np.asarray(target["early"]["median"], float)
    early_scale = np.asarray(target["early"]["bootstrap_iqr"], float)
    inc_target = np.asarray(target["increment"]["median"], float)
    inc_scale = np.asarray(target["increment"]["bootstrap_iqr"], float)
    contact, by_shaft = shaft_balanced_scaled_l1(
        model_early, early_target, early_scale, shaft_ids)
    model_inc = model_early - model_pre
    model_inc = model_inc - np.median(model_inc)
    target_inc = inc_target - np.median(inc_target)
    increment = float(np.mean(np.abs(model_inc - target_inc)
                              / np.maximum(inc_scale, 1e-6)))
    return {
        "D_contact": contact,
        "D_contact_by_shaft": by_shaft,
        "D_increment": increment,
        "J_field": lse([contact, increment]),
        "model_increment_demeaned": model_inc,
        "target_increment_demeaned": target_inc,
        "early_spearman": float(spearmanr(model_early, early_target).statistic),
    }


def score_energy_burden(model_early, target: Mapping[str, object]):
    model = np.asarray(model_early, float)
    observed = np.asarray([
        np.median(model), np.mean(model > 0),
        np.quantile(model, 0.75) - np.quantile(model, 0.25),
    ])
    patient = [
        np.asarray(target["global_early_per_seizure"], float),
        np.asarray(target["positive_fraction_per_seizure"], float),
        np.asarray(target["contact_iqr_per_seizure"], float),
    ]
    centers = np.asarray([np.median(values) for values in patient])
    scales = np.asarray([
        max(np.quantile(values, 0.75) - np.quantile(values, 0.25), 1e-6)
        for values in patient
    ])
    components = np.abs(observed - centers) / scales
    return {
        "D_energy": float(np.mean(components)),
        "model": observed,
        "patient_median": centers,
        "patient_iqr": scales,
        "scaled_components": components,
    }


def rank_selection_candidates(records, minimum_eligible=2):
    """Rank candidate groups without pooling seeds as independent replicates."""
    grouped = {}
    for row in records:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    summary = []
    for candidate_id, rows in grouped.items():
        scores = np.asarray([
            row["J_bridge_without_time"] for row in rows
            if row.get("status") == "BRIDGE_EVALUABLE"
        ], float)
        parameters = dict(rows[0].get("parameters") or {})
        summary.append({
            "candidate_id": candidate_id,
            "parameters": parameters,
            "n_runs": len(rows),
            "n_eligible": int(len(scores)),
            "eligible_proportion": float(len(scores) / len(rows)),
            "median_J_bridge": (float(np.median(scores)) if len(scores) else None),
            "worst_J_bridge": (float(np.max(scores)) if len(scores) else None),
            "selection_eligible": bool(len(scores) >= int(minimum_eligible)),
        })
    summary.sort(key=lambda row: (
        not row["selection_eligible"],
        -row["eligible_proportion"],
        float("inf") if row["median_J_bridge"] is None else row["median_J_bridge"],
        float("inf") if row["worst_J_bridge"] is None else row["worst_J_bridge"],
        row["candidate_id"],
    ))
    return summary


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
