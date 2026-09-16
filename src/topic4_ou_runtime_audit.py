"""Runtime evidence and stationarity statistics for the spatial OU drive.

The spatial OU field is declared in the round config, but a config entry is not
evidence that the field acted on every integration step.  ``OUAuditProxy`` wraps
an existing :class:`src.topic4_spatial_ou_drive.SpatialOUDrive` without touching
it: every ``step`` call is delegated verbatim, so the wrapped run is bit-identical
to the unwrapped one, while the proxy records how often the drive was queried,
at which times, and periodic snapshots of both the latent grid and the mapped
per-neuron field.

The measured statistics answer three separate questions:

* *runtime*  -- was the drive queried once per membrane step over the whole run?
* *stationarity* -- are mean, SD, temporal autocorrelation time and spatial
  correlation length the same before and after any state transition?
* *calibration* -- do the measured tau and correlation length match the declared
  ``tau_ms`` and ``ell_mm``?

Nothing here selects a working point; it only reports.
"""
from __future__ import annotations

import numpy as np


class OUAuditProxy:
    """Observation-only wrapper around a spatial OU drive.

    ``snapshot_interval_ms`` controls how often the latent grid and the mapped
    per-neuron field are copied out.  Snapshots are pure reads of the state the
    wrapped drive already computed, so the SNN sees exactly the same array
    object it would have seen without the proxy.
    """

    def __init__(self, drive, *, dt_ms, snapshot_interval_ms=1.0,
                 max_snapshots=8192):
        self.drive = drive
        self.dt_ms = float(dt_ms)
        self.snapshot_interval_ms = float(snapshot_interval_ms)
        self.max_snapshots = int(max_snapshots)
        self.n_step_calls = 0
        self.first_call_ms = None
        self.last_call_ms = None
        self._call_steps = []
        self._next_snapshot_ms = 0.0
        self.snapshot_times_ms = []
        self._grid_snapshots = []
        self._neuron_snapshots = []

    def step(self, time_ms):
        values = self.drive.step(time_ms)
        self.n_step_calls += 1
        if self.first_call_ms is None:
            self.first_call_ms = float(time_ms)
        self.last_call_ms = float(time_ms)
        self._call_steps.append(int(round(float(time_ms) / self.dt_ms)))
        if (float(time_ms) >= self._next_snapshot_ms
                and len(self.snapshot_times_ms) < self.max_snapshots):
            self.snapshot_times_ms.append(float(time_ms))
            self._grid_snapshots.append(np.array(self.drive._state, copy=True))
            self._neuron_snapshots.append(np.array(values, copy=True))
            self._next_snapshot_ms += self.snapshot_interval_ms
        return values

    # --- runtime evidence -------------------------------------------------
    def runtime_evidence(self, expected_steps):
        steps = np.asarray(self._call_steps, np.int64)
        gaps = np.diff(steps) if steps.size > 1 else np.zeros(0, np.int64)
        return {
            "n_step_calls": int(self.n_step_calls),
            "expected_membrane_steps": int(expected_steps),
            "called_every_membrane_step": bool(
                self.n_step_calls == int(expected_steps)),
            "first_call_ms": self.first_call_ms,
            "last_call_ms": self.last_call_ms,
            "call_step_gap_max": int(gaps.max()) if gaps.size else 0,
            "call_step_gap_min": int(gaps.min()) if gaps.size else 0,
            "monotonic_call_times": bool(np.all(gaps >= 0)) if gaps.size else True,
            "n_field_updates_recorded": int(
                len(self.drive.trace_arrays()["time_ms"])),
            "snapshot_interval_ms": self.snapshot_interval_ms,
            "n_snapshots": int(len(self.snapshot_times_ms)),
        }

    def snapshot_arrays(self):
        return {
            "ou_snapshot_time_ms": np.asarray(self.snapshot_times_ms, np.float32),
            "ou_grid_snapshots": np.asarray(self._grid_snapshots, np.float32),
        }

    def neuron_snapshots(self):
        return np.asarray(self._neuron_snapshots, float)


def temporal_autocorrelation_time_ms(grid_snapshots, snapshot_interval_ms,
                                     max_lag_ms=120.0):
    """Field-wide lag correlation, then the exponential time constant.

    ``tau`` is estimated by a least-squares fit of ``log rho`` against lag over
    the lags whose correlation is still above ``exp(-2)``, which keeps the fit
    inside the range where the OU process is not yet dominated by sampling
    noise.  The raw lag profile is returned so the fit can be inspected.
    """
    field = np.asarray(grid_snapshots, float)
    if field.ndim != 3 or len(field) < 4:
        raise ValueError("grid snapshots must be (time, nx, ny) with >=4 frames")
    flat = field.reshape(len(field), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    max_lag = int(round(float(max_lag_ms) / float(snapshot_interval_ms)))
    max_lag = max(1, min(max_lag, len(flat) - 2))
    lags_ms, rho = [], []
    for lag in range(0, max_lag + 1):
        a = flat[:len(flat) - lag]
        b = flat[lag:]
        denominator = norms[:len(flat) - lag] * norms[lag:]
        keep = denominator > 0
        if not np.any(keep):
            continue
        lags_ms.append(lag * float(snapshot_interval_ms))
        rho.append(float(np.mean(np.sum(a * b, axis=1)[keep] / denominator[keep])))
    lags_ms = np.asarray(lags_ms, float)
    rho = np.asarray(rho, float)
    usable = (lags_ms > 0) & (rho > np.exp(-2.0))
    if int(np.sum(usable)) >= 2:
        slope, _ = np.polyfit(lags_ms[usable], np.log(rho[usable]), 1)
        tau_hat = float(-1.0 / slope) if slope < 0 else float("nan")
    else:
        tau_hat = float("nan")
    return {
        "lag_ms": lags_ms.tolist(),
        "lag_correlation": rho.tolist(),
        "tau_hat_ms": tau_hat,
        "n_lags_used_in_fit": int(np.sum(usable)),
    }


def spatial_correlation_length_mm(grid_snapshots, grid_spacing_mm):
    """Radially averaged spatial autocorrelation and its 1/e crossing.

    The field is periodic on a square grid, so the autocorrelation is computed
    exactly by FFT rather than by binning pairwise distances.
    """
    field = np.asarray(grid_snapshots, float)
    if field.ndim != 3:
        raise ValueError("grid snapshots must be (time, nx, ny)")
    n = field.shape[1]
    if field.shape[2] != n:
        raise ValueError("grid snapshots must be square")
    accumulated = np.zeros((n, n), float)
    for frame in field:
        centred = frame - frame.mean()
        spectrum = np.fft.rfft2(centred)
        accumulated += np.fft.irfft2(spectrum * np.conj(spectrum), s=(n, n))
    accumulated /= len(field)
    if accumulated[0, 0] <= 0:
        raise ValueError("spatial autocorrelation has no power")
    correlation = accumulated / accumulated[0, 0]
    offsets = np.minimum(np.arange(n), n - np.arange(n)) * float(grid_spacing_mm)
    distance = np.hypot(offsets[:, None], offsets[None, :])
    edges = np.arange(0.0, distance.max() + float(grid_spacing_mm),
                      float(grid_spacing_mm))
    index = np.digitize(distance.ravel(), edges) - 1
    profile, centres = [], []
    for bin_index in range(len(edges) - 1):
        keep = index == bin_index
        if not np.any(keep):
            continue
        profile.append(float(correlation.ravel()[keep].mean()))
        centres.append(float(0.5 * (edges[bin_index] + edges[bin_index + 1])))
    profile = np.asarray(profile, float)
    centres = np.asarray(centres, float)
    crossing = float("nan")
    below = np.flatnonzero(profile < np.exp(-1.0))
    if below.size and below[0] > 0:
        i = int(below[0])
        x0, x1 = centres[i - 1], centres[i]
        y0, y1 = profile[i - 1], profile[i]
        if y0 != y1:
            crossing = float(x0 + (np.exp(-1.0) - y0) * (x1 - x0) / (y1 - y0))
    return {
        "distance_mm": centres.tolist(),
        "correlation": profile.tolist(),
        "correlation_length_mm_1_over_e": crossing,
    }


def stationarity_report(neuron_snapshots, snapshot_times_ms, split_time_ms):
    """Compare the drive's own statistics before and after a reference time.

    The contract is that the OU process is part of the environment and never
    changes across the transition, so every reported quantity must match to
    within sampling error.  A non-finite split time reports the whole run once.
    """
    values = np.asarray(neuron_snapshots, float)
    times = np.asarray(snapshot_times_ms, float)
    if values.ndim != 2 or len(values) != len(times):
        raise ValueError("neuron snapshots and times must align")

    def _block(mask):
        if not np.any(mask):
            return None
        block = values[mask]
        return {
            "n_snapshots": int(block.shape[0]),
            "mean_rate_per_ms": float(block.mean()),
            "sd_rate_per_ms": float(block.std()),
            "abs_max_rate_per_ms": float(np.abs(block).max()),
            "median_frame_sd_rate_per_ms": float(np.median(block.std(axis=1))),
        }

    whole = _block(np.ones(len(times), bool))
    if split_time_ms is None or not np.isfinite(float(split_time_ms)):
        return {"whole_run": whole, "before": None, "after": None,
                "split_time_ms": None}
    split = float(split_time_ms)
    before = _block(times < split)
    after = _block(times >= split)
    report = {"whole_run": whole, "before": before, "after": after,
              "split_time_ms": split}
    if before and after:
        report["sd_ratio_after_over_before"] = float(
            after["sd_rate_per_ms"] / max(before["sd_rate_per_ms"], 1e-20))
        report["mean_difference_rate_per_ms"] = float(
            after["mean_rate_per_ms"] - before["mean_rate_per_ms"])
    return report


class OUProtocolProxy(OUAuditProxy):
    """Audit proxy that can change the noise *realisation* or *amplitude*.

    Two causal controls need the environment altered in a specific, minimal way
    after the state has already changed:

    * ``reseed_at_ms`` swaps the innovation stream for a fresh one while keeping
      every declared statistic (amplitude, tau, correlation length) and the
      current field value. It answers "does the high state depend on the exact
      noise history that produced it, or only on there being some stationary
      noise of this kind?"
    * ``dip_start_ms`` / ``dip_duration_ms`` / ``dip_factor`` scale the field for
      a bounded interval and then restore it exactly. It answers "is the state
      autonomous, or supported by the ongoing noise?"

    Neither control introduces timing structure the network could follow: the
    reseed changes nothing measurable about the process, and the dip is a single
    step change in amplitude, not a pulse train or a periodic drive.
    """

    def __init__(self, drive, *, dt_ms, snapshot_interval_ms=1.0,
                 max_snapshots=8192, reseed_at_ms=None, reseed_seed=None,
                 dip_start_ms=None, dip_duration_ms=None, dip_factor=1.0):
        super().__init__(drive, dt_ms=dt_ms,
                         snapshot_interval_ms=snapshot_interval_ms,
                         max_snapshots=max_snapshots)
        if (reseed_at_ms is not None) != (reseed_seed is not None):
            raise ValueError("reseeding needs both a time and a seed")
        if dip_start_ms is not None and (dip_duration_ms is None
                                         or float(dip_duration_ms) <= 0.0):
            raise ValueError("a noise dip needs a positive duration")
        self.reseed_at_ms = None if reseed_at_ms is None else float(reseed_at_ms)
        self.reseed_seed = None if reseed_seed is None else int(reseed_seed)
        self.dip_start_ms = None if dip_start_ms is None else float(dip_start_ms)
        self.dip_duration_ms = (None if dip_duration_ms is None
                                else float(dip_duration_ms))
        self.dip_factor = float(dip_factor)
        self._reseeded_at = None
        self._dip_steps = 0

    def step(self, time_ms):
        if (self.reseed_at_ms is not None and self._reseeded_at is None
                and float(time_ms) >= self.reseed_at_ms):
            self.drive._rng = np.random.default_rng(self.reseed_seed)
            self._reseeded_at = float(time_ms)
        values = super().step(time_ms)
        if self.dip_start_ms is not None:
            inside = (self.dip_start_ms <= float(time_ms)
                      < self.dip_start_ms + self.dip_duration_ms)
            if inside:
                self._dip_steps += 1
                # Copy: the drive caches this array and reuses it between updates.
                return np.asarray(values, float) * self.dip_factor
        return values

    def protocol_evidence(self):
        return {
            "reseed_requested_ms": self.reseed_at_ms,
            "reseed_applied_ms": self._reseeded_at,
            "reseed_seed": self.reseed_seed,
            "dip_start_ms": self.dip_start_ms,
            "dip_duration_ms": self.dip_duration_ms,
            "dip_factor": self.dip_factor,
            "n_steps_inside_dip": int(self._dip_steps),
            "semantics": (
                "reseed keeps amplitude, tau and correlation length and only "
                "changes the innovation stream; the dip is one bounded step "
                "change in amplitude, not a pulse train or periodic drive"),
        }
