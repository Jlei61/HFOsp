"""Pure metrics and resource gates for the MZ-divisive lifecycle experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LifecycleThresholds:
    recruited_hz: float = 20.0
    recruited_sustain_ms: float = 250.0
    min_recruited_ms: float = 1000.0
    recovery_ms: float = 2000.0
    recovery_margin_hz: float = 5.0
    envelope_ms: float = 50.0
    merge_gap_ms: float = 100.0
    burst_min_peaks: int = 4
    burst_min_modulation: float = 0.30
    burst_band_hz: tuple[float, float] = (0.5, 20.0)
    peak_min_separation_ms: float = 50.0

    def validate(self) -> None:
        for name, value in (
            ("recruited_hz", self.recruited_hz),
            ("recruited_sustain_ms", self.recruited_sustain_ms),
            ("min_recruited_ms", self.min_recruited_ms),
            ("recovery_ms", self.recovery_ms),
            ("envelope_ms", self.envelope_ms),
            ("peak_min_separation_ms", self.peak_min_separation_ms),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        lo, hi = self.burst_band_hz
        if not (0.0 < lo < hi):
            raise ValueError(f"burst_band_hz must satisfy 0 < lo < hi, got {self.burst_band_hz}")
        if self.burst_min_peaks < 1:
            raise ValueError("burst_min_peaks must be >= 1")
        if not (0.0 <= self.burst_min_modulation <= 1.0):
            raise ValueError("burst_min_modulation must be in [0,1]")


@dataclass(frozen=True)
class StrictLifecycleThresholds:
    """Post-hoc lifecycle contract tied to a paired same-seed slow-off trace.

    The online classifier above is intentionally retained as a broad screen descriptor.  This
    stricter audit defines the recruited *macro-state* with a 250 ms envelope, then separately asks
    whether that state is rhythmic and whether the trace returns to the empirical slow-off regime.
    """

    recruited_hz: float = 20.0
    state_envelope_ms: float = 250.0
    min_recruited_ms: float = 1000.0
    recovery_ms: float = 2000.0
    reference_quantile: float = 0.99
    event_envelope_ms: float = 50.0
    event_hz: float = 20.0
    event_max_ms: float = 200.0
    burst_min_peaks: int = 4
    burst_min_modulation: float = 0.30
    burst_min_power_ratio: float = 0.10
    burst_band_hz: tuple[float, float] = (0.5, 20.0)
    peak_min_separation_ms: float = 50.0

    def validate(self) -> None:
        for name, value in (
            ("recruited_hz", self.recruited_hz),
            ("state_envelope_ms", self.state_envelope_ms),
            ("min_recruited_ms", self.min_recruited_ms),
            ("recovery_ms", self.recovery_ms),
            ("event_envelope_ms", self.event_envelope_ms),
            ("event_hz", self.event_hz),
            ("event_max_ms", self.event_max_ms),
            ("peak_min_separation_ms", self.peak_min_separation_ms),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if not (0.5 < self.reference_quantile < 1.0):
            raise ValueError("reference_quantile must be in (0.5, 1)")
        if not (0.0 <= self.burst_min_power_ratio <= 1.0):
            raise ValueError("burst_min_power_ratio must be in [0,1]")
        lo, hi = self.burst_band_hz
        if not (0.0 < lo < hi):
            raise ValueError("burst_band_hz must satisfy 0 < lo < hi")


def _moving_average(x, n):
    x = np.asarray(x, float)
    if n <= 1:
        return x.copy()
    n = int(n)
    left = n // 2
    right = n - 1 - left
    padded = np.pad(x, (left, right), mode="edge")
    csum = np.r_[0.0, np.cumsum(padded, dtype=float)]
    return (csum[n:] - csum[:-n]) / float(n)


def _rolling_valid(x, n):
    """Unpadded rolling mean used to derive an empirical reference band."""
    x = np.asarray(x, float)
    n = int(n)
    if n <= 1:
        return x.copy()
    if x.size < n:
        return np.array([float(x.mean())])
    csum = np.r_[0.0, np.cumsum(x, dtype=float)]
    return (csum[n:] - csum[:-n]) / float(n)


def _episodes(mask, max_gap_bins):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out = []
    start = prev = int(idx[0])
    for raw in idx[1:]:
        i = int(raw)
        if i - prev - 1 > max_gap_bins:
            out.append((start, prev + 1))
            start = i
        prev = i
    out.append((start, prev + 1))
    return out


def _resolved_peak_count(x, min_sep_bins):
    x = np.asarray(x, float)
    if x.size < 3:
        return 0
    p5, p95 = np.percentile(x, [5, 95])
    threshold = float(np.median(x) + 0.25 * (p95 - p5))
    candidates = np.flatnonzero(
        (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] >= threshold)
    ) + 1
    if candidates.size == 0:
        return 0
    selected = [int(candidates[0])]
    for raw in candidates[1:]:
        i = int(raw)
        if i - selected[-1] >= min_sep_bins:
            selected.append(i)
        elif x[i] > x[selected[-1]]:
            selected[-1] = i
    return len(selected)


def _spectral_peak(x, dt_ms, band):
    x = np.asarray(x, float)
    if x.size < 4:
        return None, 0.0
    y = x - float(x.mean())
    power = np.abs(np.fft.rfft(y)) ** 2
    freq = np.fft.rfftfreq(y.size, d=dt_ms * 1e-3)
    lo, hi = band
    mask = (freq >= lo) & (freq <= hi)
    if not mask.any() or float(power[mask].sum()) <= 0.0:
        return None, 0.0
    idx_local = int(np.argmax(power[mask]))
    freq_band = freq[mask]
    peak_hz = float(freq_band[idx_local])
    denom = float(power[(freq >= lo) & (freq <= min(40.0, freq.max()))].sum())
    ratio = float(power[mask][idx_local] / denom) if denom > 0.0 else 0.0
    return peak_hz, ratio


def tail_slope_per_s(x, dt_ms, tail_ms=3000.0):
    """Linear last-window slope with seconds as the time unit."""
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        return None
    n = min(x.size, max(2, int(round(float(tail_ms) / float(dt_ms)))))
    t_s = np.arange(n, dtype=float) * float(dt_ms) * 1e-3
    return float(np.polyfit(t_s, x[-n:], 1)[0])


def audit_lifecycle_against_reference(
    rate_hz,
    dt_ms,
    *,
    reference_rate_hz,
    reference_dt_ms,
    runaway_ms=None,
    slow_traces=None,
    thresholds: StrictLifecycleThresholds | None = None,
):
    """Strict post-hoc audit against a paired slow-off trace.

    A constant low-rate tail cannot count as return: the tail must be inside the empirical slow-off
    2 s band *and* contain at least one brief returning-event-like excursion.  Conversely, a 4 Hz
    clonic train made of 100 ms bursts is treated as a recruited macro-state rather than a sequence
    of unrelated short episodes.
    """
    th = thresholds or StrictLifecycleThresholds()
    th.validate()
    rate = np.asarray(rate_hz, float)
    ref = np.asarray(reference_rate_hz, float)
    if (
        rate.ndim != 1
        or ref.ndim != 1
        or rate.size == 0
        or ref.size == 0
        or not np.all(np.isfinite(rate))
        or not np.all(np.isfinite(ref))
    ):
        raise ValueError("rate and reference_rate_hz must be non-empty finite 1D arrays")
    if dt_ms <= 0.0 or reference_dt_ms <= 0.0:
        raise ValueError("dt_ms and reference_dt_ms must be > 0")

    recovery_bins = max(1, int(round(th.recovery_ms / dt_ms)))
    ref_recovery_bins = max(1, int(round(th.recovery_ms / reference_dt_ms)))
    reference_windows = _rolling_valid(ref, ref_recovery_bins)
    reference_return_upper_hz = float(np.quantile(reference_windows, th.reference_quantile))

    state_bins = max(1, int(round(th.state_envelope_ms / dt_ms)))
    state_env = _moving_average(rate, state_bins)
    min_state_bins = max(1, int(round(th.min_recruited_ms / dt_ms)))
    all_recruited_eps = _episodes(state_env >= th.recruited_hz, 0)
    recruited_eps = [
        (i0, i1)
        for i0, i1 in all_recruited_eps
        if i1 - i0 >= min_state_bins
    ]

    out = dict(
        strict_phenotype="runaway" if runaway_ms is not None else "no_recruited_macrostate",
        runaway_ms=runaway_ms,
        onset_ms=None,
        offset_ms=None,
        recruited_duration_ms=0.0,
        state_envelope_max_hz=float(state_env.max()),
        max_recruited_envelope_episode_ms=float(
            max(((i1 - i0) * dt_ms for i0, i1 in all_recruited_eps), default=0.0)
        ),
        returned_to_same_seed_slowoff=False,
        recovery_window_mean_hz=None,
        final_window_mean_hz=float(rate[-recovery_bins:].mean()),
        reference_return_upper_hz=reference_return_upper_hz,
        late_returning_event_count=0,
        rebound_macrostate_count=0,
        burst_peak_count=0,
        burst_modulation=0.0,
        burst_peak_hz=None,
        burst_peak_power_ratio=0.0,
        rhythmic_bursting=False,
        m_rise_before_rate_decay=None,
        m_rise_ms=None,
        rate_decay_ms=None,
        tail_slopes_per_s={},
        thresholds=asdict(th),
    )

    for name, values in (slow_traces or {}).items():
        arr = np.asarray(values, float)
        out["tail_slopes_per_s"][str(name)] = tail_slope_per_s(arr, dt_ms)

    if runaway_ms is not None or not recruited_eps:
        return out

    i0, i1 = recruited_eps[0]
    duration_ms = float((i1 - i0) * dt_ms)
    segment = rate[i0:i1]
    p5, p95 = np.percentile(segment, [5, 95])
    modulation = float((p95 - p5) / (abs(p95) + abs(p5) + 1e-12))
    peak_count = _resolved_peak_count(
        _moving_average(segment, max(1, int(round(th.event_envelope_ms / dt_ms)))),
        max(1, int(round(th.peak_min_separation_ms / dt_ms))),
    )
    peak_hz, peak_ratio = _spectral_peak(segment, dt_ms, th.burst_band_hz)
    rhythmic = bool(
        peak_count >= th.burst_min_peaks
        and modulation >= th.burst_min_modulation
        and peak_hz is not None
        and peak_ratio >= th.burst_min_power_ratio
    )
    out.update(
        strict_phenotype="bounded_recruited_bursting" if rhythmic else "bounded_recruited_nonrhythmic",
        onset_ms=float(i0 * dt_ms),
        recruited_duration_ms=duration_ms,
        burst_peak_count=int(peak_count),
        burst_modulation=modulation,
        burst_peak_hz=peak_hz,
        burst_peak_power_ratio=peak_ratio,
        rhythmic_bursting=rhythmic,
    )

    if i1 < rate.size:
        out["offset_ms"] = float(i1 * dt_ms)
        out["rebound_macrostate_count"] = int(len(recruited_eps) - 1)
        enough_recovery = rate.size - i1 >= recovery_bins
        if enough_recovery:
            recovery = rate[i1 : i1 + recovery_bins]
            out["recovery_window_mean_hz"] = float(recovery.mean())

            event_env = _moving_average(rate, max(1, int(round(th.event_envelope_ms / dt_ms))))
            brief_events = [
                (j0, j1)
                for j0, j1 in _episodes(event_env >= th.event_hz, 0)
                if j0 >= i1 + recovery_bins and (j1 - j0) * dt_ms <= th.event_max_ms
            ]
            out["late_returning_event_count"] = int(len(brief_events))
            returned = bool(
                float(recovery.mean()) <= reference_return_upper_hz
                and out["final_window_mean_hz"] <= reference_return_upper_hz
                and len(brief_events) >= 1
                and len(recruited_eps) == 1
            )
            out["returned_to_same_seed_slowoff"] = returned
            if returned:
                out["strict_phenotype"] = (
                    "terminate_bursting_strict" if rhythmic else "terminate_nonrhythmic_strict"
                )

        m_values = None if slow_traces is None else slow_traces.get("m_mean")
        if m_values is not None:
            m = np.asarray(m_values, float)
            if m.size == rate.size and i1 > i0 and float(m[i0:i1].max()) > float(m[i0]):
                m_target = float(m[i0] + 0.25 * (m[i0:i1].max() - m[i0]))
                m_hits = np.flatnonzero(m[i0:i1] >= m_target)
                peak_i = i0 + int(np.argmax(state_env[i0:i1]))
                decay_level = max(reference_return_upper_hz, 0.75 * float(state_env[peak_i]))
                decay_hits = np.flatnonzero(state_env[peak_i:i1] <= decay_level)
                if m_hits.size:
                    m_rise_i = i0 + int(m_hits[0])
                    out["m_rise_ms"] = float(m_rise_i * dt_ms)
                if decay_hits.size:
                    decay_i = peak_i + int(decay_hits[0])
                    out["rate_decay_ms"] = float(decay_i * dt_ms)
                if out["m_rise_ms"] is not None and out["rate_decay_ms"] is not None:
                    out["m_rise_before_rate_decay"] = bool(
                        out["m_rise_ms"] < out["rate_decay_ms"]
                    )
    return out


def analyze_lifecycle(
    rate_hz,
    dt_ms,
    *,
    baseline_rate_hz=None,
    runaway_ms=None,
    thresholds: LifecycleThresholds | None = None,
):
    """Classify one population-rate trace without treating rhythmic multi-episode activity as failure.

    The classification is a screen descriptor, not a bifurcation or seizure label. ``runaway_ms`` comes
    from the engine's fixed 120 Hz / 100 ms detector and has priority over trace shape.
    """
    th = thresholds or LifecycleThresholds()
    th.validate()
    rate = np.asarray(rate_hz, float)
    if rate.ndim != 1 or rate.size == 0 or not np.all(np.isfinite(rate)):
        raise ValueError("rate_hz must be a non-empty finite 1D array")
    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be > 0")
    if baseline_rate_hz is None:
        n0 = max(1, min(rate.size, int(round(500.0 / dt_ms))))
        baseline_rate_hz = float(np.median(rate[:n0]))
    baseline_rate_hz = float(baseline_rate_hz)
    base = dict(
        phenotype=None,
        onset_ms=None,
        offset_ms=None,
        recruited_duration_ms=0.0,
        returned_to_baseline=False,
        burst_peak_count=0,
        burst_modulation=0.0,
        burst_peak_hz=None,
        burst_peak_power_ratio=0.0,
        baseline_rate_hz=baseline_rate_hz,
        runaway_ms=runaway_ms,
        thresholds=asdict(th),
    )
    if runaway_ms is not None:
        base["phenotype"] = "runaway"
        return base

    env_bins = max(1, int(round(th.envelope_ms / dt_ms)))
    envelope = _moving_average(rate, env_bins)
    high = envelope >= th.recruited_hz
    gap_bins = max(0, int(round(th.merge_gap_ms / dt_ms)))
    sustain_bins = max(1, int(round(th.recruited_sustain_ms / dt_ms)))
    episodes = [(i0, i1) for i0, i1 in _episodes(high, gap_bins) if i1 - i0 >= sustain_bins]
    if not episodes:
        base["phenotype"] = "interictal_like"
        return base

    i0, i1 = episodes[0]
    duration_ms = (i1 - i0) * dt_ms
    base["onset_ms"] = float(i0 * dt_ms)
    base["recruited_duration_ms"] = float(duration_ms)
    segment = rate[i0:i1]
    p5, p95 = np.percentile(segment, [5, 95])
    modulation = float((p95 - p5) / (abs(p95) + abs(p5) + 1e-12))
    peak_count = _resolved_peak_count(
        segment, max(1, int(round(th.peak_min_separation_ms / dt_ms)))
    )
    peak_hz, peak_ratio = _spectral_peak(segment, dt_ms, th.burst_band_hz)
    base.update(
        burst_peak_count=int(peak_count),
        burst_modulation=modulation,
        burst_peak_hz=peak_hz,
        burst_peak_power_ratio=peak_ratio,
    )
    bursting = bool(
        duration_ms >= th.min_recruited_ms
        and peak_count >= th.burst_min_peaks
        and modulation >= th.burst_min_modulation
        and peak_hz is not None
    )

    recovery_bins = max(1, int(round(th.recovery_ms / dt_ms)))
    enough_tail = rate.size - i1 >= recovery_bins
    returned = False
    if enough_tail:
        tail = rate[i1 : i1 + recovery_bins]
        later_eps = [
            ep
            for ep in _episodes(envelope[i1:] >= th.recruited_hz, gap_bins)
            if ep[1] - ep[0] >= sustain_bins
        ]
        returned = bool(
            float(tail.mean()) <= baseline_rate_hz + th.recovery_margin_hz and not later_eps
        )
    base["returned_to_baseline"] = returned
    if returned:
        base["offset_ms"] = float(i1 * dt_ms)
        base["phenotype"] = "terminate_bursting" if bursting else "terminate_plateau"
    elif not enough_tail:
        base["phenotype"] = "bounded_bursting" if bursting else "bounded_plateau"
    else:
        base["offset_ms"] = float(i1 * dt_ms)
        base["phenotype"] = "fragment_or_fade"
    return base


def safe_worker_count(
    requested,
    n_cells,
    mem_available_gib,
    peak_worker_gib,
    *,
    reserve_gib=96.0,
    safety_factor=1.2,
    hard_cap=12,
    cpu_count=None,
    cpu_reserve=16,
):
    """Return a fail-closed launch count that preserves the registered memory/CPU reserve."""
    if peak_worker_gib <= 0.0 or safety_factor <= 0.0:
        raise ValueError("peak_worker_gib and safety_factor must be > 0")
    if requested < 1 or n_cells < 1:
        return 0
    memory_budget = float(mem_available_gib) - float(reserve_gib)
    by_memory = max(0, math.floor(memory_budget / (safety_factor * peak_worker_gib)))
    if cpu_count is None:
        by_cpu = hard_cap
    else:
        by_cpu = max(0, int(cpu_count) - int(cpu_reserve))
    return int(min(requested, n_cells, hard_cap, by_memory, by_cpu))
