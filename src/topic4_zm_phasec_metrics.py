"""Pure Phase-C0 diagnostics for the Z/M sustained high-rate branch.

This module deliberately has no simulator, checkpoint, result-tree, or plotting
dependency.  It accepts either an in-memory boolean E-spike raster or a metrics
dictionary produced by an online observer.  The scientific question is narrow:

    asynchronous/irregular tonic activity
        versus
    a refractory-limited high-rate plateau.

The labels below are descriptive source-space identities.  They do not establish
an ictal observation match, entry, offset, recovery, or a complete lifecycle.

Units
-----
``dt_ms``, ``tau_ref_ms`` and bin widths are milliseconds; firing rates are Hz;
local threshold gain is Hz/mV; CV2, Fano factor, correlations, fractions, entropy
and posterior probabilities are dimensionless.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta


PHASEC_METRICS_VERSION = "zm_phasec_metrics_v1_2026-07-28"
HIERARCHICAL_STATS_VERSION = "zm_phasec_hierarchical_stats_v2_2026-07-29"
PAIR_NULL_STRATUM_NAMES = (
    "core_core",
    "core_surround",
    "surround_surround",
)
REFRACTORY_ISI_STRATUM_NAMES = ("core", "surround")


@dataclass(frozen=True)
class PhaseCThresholds:
    """Locked descriptive thresholds for the Phase-C0 branch identity.

    The two positive classes intentionally leave a wide gap.  Runs in that gap,
    runs mixing both neuron types, and runs missing any required metric remain
    unresolved rather than being forced into the nearest label.
    """

    ceiling_rate_fraction: float = 0.80
    ai_ceiling_fraction_max: float = 0.20
    ai_ref_lock_fraction_max: float = 0.25
    ai_isi_cv2_median_min: float = 0.70
    ai_fano20_median_min: float = 0.60
    ai_pairwise_excess_max: float = 0.10
    plateau_ceiling_fraction_min: float = 0.50
    plateau_ref_lock_fraction_min: float = 0.80
    plateau_isi_cv2_median_max: float = 0.10
    plateau_fano20_median_max: float = 0.15
    mixed_neuron_fraction_min: float = 0.20
    min_eligible_neurons: int = 20
    min_pairwise_neurons: int = 16
    min_seeds: int = 3


def refractory_ceiling_hz(tau_ref_ms, dt_ms):
    """Return the implemented discrete-time refractory ceiling.

    The engine rounds ``tau_ref_ms / dt_ms`` to an integer number of refractory
    steps.  Using the requested floating-point duration directly can therefore
    misstate the ceiling at a changed integration step.
    """
    tau_ref = float(tau_ref_ms)
    dt = float(dt_ms)
    if not np.isfinite(tau_ref) or tau_ref <= 0:
        raise ValueError("tau_ref_ms must be finite and positive")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_ms must be finite and positive")
    ref_steps = int(round(tau_ref / dt))
    if ref_steps < 1:
        raise ValueError("tau_ref_ms rounds to fewer than one integration step")
    return 1000.0 / (ref_steps * dt)


def _finite_summary(x) -> dict:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {
            "n": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "mean": None,
        }
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "mean": float(np.mean(a)),
    }


def _validate_raster(spikes, dt_ms):
    x = np.asarray(spikes)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("spikes must be a non-empty time x neuron array")
    if x.dtype != np.bool_:
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("spikes must be boolean or binary")
        x = x.astype(bool)
    dt = float(dt_ms)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_ms must be finite and positive")
    return x, dt


def _validate_mask(mask, n_neurons, name):
    if mask is None:
        return None
    m = np.asarray(mask, bool)
    if m.shape != (n_neurons,):
        raise ValueError(f"{name} must have shape ({n_neurons},)")
    return m


def firing_and_ceiling_metrics(
    spikes,
    dt_ms,
    tau_ref_ms,
    *,
    core_mask=None,
    ceiling_rate_fraction=0.80,
    active_rate_hz=5.0,
    window_ms=250.0,
    window_stride_ms=50.0,
):
    """Per-neuron firing rates and refractory-ceiling fractions.

    The exact implemented LIF refractory ceiling is
    ``1000/(round(tau_ref_ms/dt_ms)*dt_ms)`` Hz.  The reported
    ceiling fraction is the fraction of neurons whose time-average rate is at
    least ``ceiling_rate_fraction`` of that value.  ``active`` means at least one
    spike in the analysed window; both all-E and core-only fractions are emitted.
    """
    x, dt = _validate_raster(spikes, dt_ms)
    tau_ref = float(tau_ref_ms)
    frac = float(ceiling_rate_fraction)
    if not np.isfinite(tau_ref) or tau_ref <= 0:
        raise ValueError("tau_ref_ms must be finite and positive")
    if not 0 < frac <= 1:
        raise ValueError("ceiling_rate_fraction must lie in (0,1]")
    core = _validate_mask(core_mask, x.shape[1], "core_mask")
    duration_s = x.shape[0] * dt * 1e-3
    rates = x.sum(axis=0, dtype=np.int64) / duration_s
    active = rates >= float(active_rate_hz)
    max_rate = refractory_ceiling_hz(tau_ref, dt)
    at_ceiling = rates >= frac * max_rate

    win_steps = int(round(float(window_ms) / dt))
    stride_steps = int(round(float(window_stride_ms) / dt))
    if (win_steps < 1 or stride_steps < 1
            or not np.isclose(win_steps * dt, float(window_ms))
            or not np.isclose(stride_steps * dt, float(window_stride_ms))):
        raise ValueError("ceiling window/stride must be exactly representable at dt_ms")
    starts = np.arange(0, max(0, x.shape[0] - win_steps + 1), stride_steps, dtype=int)
    if starts.size == 0:
        raise ValueError("spike raster is shorter than the ceiling window")
    rolling = np.stack(
        [x[s:s + win_steps].sum(axis=0, dtype=np.int32) for s in starts],
        axis=0,
    ) / (float(window_ms) * 1e-3)
    rolling_ceiling = rolling >= frac * max_rate

    def _rho(mask):
        mask = np.asarray(mask, bool)
        if not np.any(mask):
            return None, None
        vals = rolling_ceiling[:, mask].mean(axis=1)
        return float(np.median(vals)), _finite_summary(vals)

    out = {
        "max_refractory_rate_hz": float(max_rate),
        "ceiling_rate_fraction": frac,
        "ceiling_rate_threshold_hz": float(frac * max_rate),
        "ceiling_fraction_all": float(np.mean(at_ceiling)),
        "ceiling_fraction_active": (
            float(np.mean(at_ceiling[active])) if np.any(active) else None
        ),
        "active_neuron_fraction": float(np.mean(active)),
        "rate_hz": _finite_summary(rates),
        "n_neurons": int(x.shape[1]),
        "n_active_neurons": int(active.sum()),
        "active_rate_hz": float(active_rate_hz),
        "ceiling_window_ms": float(window_ms),
        "ceiling_window_stride_ms": float(window_stride_ms),
        "rho80_all_median": _rho(np.ones(x.shape[1], bool))[0],
        "rho80_active_median": _rho(active)[0],
        "rho80_all_windows": _rho(np.ones(x.shape[1], bool))[1],
        "rho80_active_windows": _rho(active)[1],
    }
    if core is not None:
        active_core = active & core
        out.update(
            n_core=int(core.sum()),
            active_core_fraction=(
                float(np.mean(active[core])) if np.any(core) else None
            ),
            ceiling_fraction_core=(
                float(np.mean(at_ceiling[core])) if np.any(core) else None
            ),
            ceiling_fraction_active_core=(
                float(np.mean(at_ceiling[active_core]))
                if np.any(active_core) else None
            ),
            rate_hz_core=_finite_summary(rates[core]),
            rho80_core_median=_rho(core)[0],
            rho80_active_core_median=_rho(active_core)[0],
            rho80_core_windows=_rho(core)[1],
            rho80_active_core_windows=_rho(active_core)[1],
        )
    else:
        out.update(
            n_core=0,
            active_core_fraction=None,
            ceiling_fraction_core=None,
            ceiling_fraction_active_core=None,
            rate_hz_core=_finite_summary([]),
            rho80_core_median=None,
            rho80_active_core_median=None,
            rho80_core_windows=None,
            rho80_active_core_windows=None,
        )
    return out, rates


def isi_cv2_and_refractory_lock(
    spikes,
    dt_ms,
    tau_ref_ms,
    *,
    min_isis=5,
    ref_tolerance_ms=None,
):
    """Single-neuron local ISI irregularity and refractory locking.

    For consecutive ISIs ``a,b``, CV2 is ``2*|a-b|/(a+b)``.  It is robust to a
    slowly drifting rate and equals zero for a periodic spike train.  A neuron is
    refractory-locked when at least 80% of its ISIs lie within the explicit
    tolerance of ``tau_ref_ms``.
    """
    x, dt = _validate_raster(spikes, dt_ms)
    tau_ref = float(tau_ref_ms)
    if not np.isfinite(tau_ref) or tau_ref <= 0:
        raise ValueError("tau_ref_ms must be finite and positive")
    if int(min_isis) < 2:
        raise ValueError("min_isis must be at least 2")
    tol = (
        max(0.51 * dt, 0.05 * tau_ref)
        if ref_tolerance_ms is None
        else float(ref_tolerance_ms)
    )
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("ref_tolerance_ms must be finite and nonnegative")

    cv2 = np.full(x.shape[1], np.nan)
    lock_score = np.full(x.shape[1], np.nan)
    pooled_isi = []
    for j in range(x.shape[1]):
        times = np.flatnonzero(x[:, j]).astype(float) * dt
        isi = np.diff(times)
        if isi.size < int(min_isis):
            continue
        pooled_isi.append(isi)
        adjacent_sum = isi[1:] + isi[:-1]
        vals = np.divide(
            2.0 * np.abs(isi[1:] - isi[:-1]),
            adjacent_sum,
            out=np.full(isi.size - 1, np.nan),
            where=adjacent_sum > 0,
        )
        cv2[j] = float(np.nanmean(vals))
        lock_score[j] = float(np.mean(np.abs(isi - tau_ref) <= tol))

    eligible = np.isfinite(cv2) & np.isfinite(lock_score)
    pooled = np.concatenate(pooled_isi) if pooled_isi else np.asarray([], float)
    return {
        "isi_cv2": _finite_summary(cv2),
        "refractory_lock_score": _finite_summary(lock_score),
        "refractory_locked_fraction": (
            float(np.mean(lock_score[eligible] >= 0.80))
            if np.any(eligible) else None
        ),
        "n_isi_eligible": int(eligible.sum()),
        "min_isis": int(min_isis),
        "ref_tolerance_ms": float(tol),
        "refractory_isi_fraction": (
            float(np.mean(pooled <= tau_ref + 2.0 * dt))
            if pooled.size else None
        ),
        "n_pooled_isi": int(pooled.size),
    }, cv2, lock_score


def fano_metrics(spikes, dt_ms, *, bin_widths_ms=(5.0, 20.0, 100.0)):
    """Per-neuron spike-count Fano factors over several time scales."""
    x, dt = _validate_raster(spikes, dt_ms)
    out = {}
    arrays = {}
    for width in bin_widths_ms:
        width = float(width)
        bs = int(round(width / dt))
        if bs < 1 or not np.isclose(bs * dt, width):
            raise ValueError(f"bin width {width:g} ms is not representable at dt={dt:g} ms")
        nb = x.shape[0] // bs
        key = f"{width:g}ms"
        if nb < 2:
            out[key] = {"n": 0, "median": None, "q25": None, "q75": None, "mean": None}
            arrays[key] = np.full(x.shape[1], np.nan)
            continue
        counts = x[: nb * bs].reshape(nb, bs, x.shape[1]).sum(
            axis=1, dtype=np.uint16
        )
        mean = counts.mean(axis=0)
        var = counts.var(axis=0, ddof=1)
        fano = np.divide(
            var,
            mean,
            out=np.full(mean.shape, np.nan, dtype=float),
            where=mean > 0,
        )
        out[key] = _finite_summary(fano)
        arrays[key] = fano
    return {"fano_by_bin": out}, arrays


def _pairwise_corr_from_counts(counts):
    y = np.asarray(counts, float)
    if y.ndim != 2 or y.shape[0] < 3 or y.shape[1] < 2:
        return np.asarray([])
    sd = y.std(axis=0, ddof=1)
    valid = np.isfinite(sd) & (sd > 0)
    y = y[:, valid]
    if y.shape[1] < 2:
        return np.asarray([])
    z = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
    c = (z.T @ z) / (z.shape[0] - 1)
    return c[np.triu_indices(c.shape[0], k=1)]


def pairwise_spike_count_correlation(
    spikes,
    dt_ms,
    *,
    bin_ms=5.0,
    max_neurons=256,
    min_spikes=5,
    n_null=100,
    null_seed=0,
    fixed_panel=False,
):
    """Observed pairwise spike-count correlation and circular-shift null.

    Eligible neurons are selected deterministically at evenly spaced ranks; the
    null RNG is analysis-only and never touches the simulator stream.  Each null
    independently circularly shifts every selected neuron's count series,
    preserving its rate and autocorrelation while breaking zero-lag coupling.
    """
    x, dt = _validate_raster(spikes, dt_ms)
    bs = int(round(float(bin_ms) / dt))
    if bs < 1 or not np.isclose(bs * dt, float(bin_ms)):
        raise ValueError("bin_ms is not representable at dt_ms")
    nb = x.shape[0] // bs
    if nb < 4:
        return {
            "status": "insufficient_time_bins",
            "observed_median": None,
            "null_median": None,
            "null_q95": None,
            "excess_over_null": None,
            "n_neurons": 0,
            "n_pairs": 0,
        }
    totals = x[: nb * bs].sum(axis=0)
    if fixed_panel:
        eligible = np.arange(x.shape[1], dtype=int)
        if eligible.size > int(max_neurons):
            raise ValueError("fixed pairwise panel exceeds max_neurons")
    else:
        eligible = np.flatnonzero(totals >= int(min_spikes))
        if eligible.size > int(max_neurons):
            choose = np.linspace(0, eligible.size - 1, int(max_neurons)).astype(int)
            eligible = eligible[choose]
    if eligible.size < 2:
        return {
            "status": "insufficient_active_neurons",
            "observed_median": None,
            "null_median": None,
            "null_q95": None,
            "excess_over_null": None,
            "n_neurons": int(eligible.size),
            "n_pairs": 0,
        }
    counts = x[: nb * bs, eligible].reshape(nb, bs, eligible.size).sum(
        axis=1, dtype=np.uint16
    )
    variable = np.isfinite(counts).all(axis=0) & (counts.std(axis=0, ddof=1) > 0)
    counts = counts[:, variable]
    n_panel = int(eligible.size)
    n_valid = int(variable.sum())
    observed = _pairwise_corr_from_counts(counts)
    if not observed.size:
        return {
            "status": "degenerate_count_variance",
            "observed_median": None,
            "null_median": None,
            "null_q95": None,
            "excess_over_null": None,
            "n_neurons": n_valid,
            "n_panel_neurons": n_panel,
            "n_pairs": 0,
        }
    rng = np.random.default_rng(int(null_seed))
    null_medians = []
    for _ in range(int(n_null)):
        shifted = np.empty_like(counts)
        for j in range(counts.shape[1]):
            shift = int(rng.integers(1, counts.shape[0]))
            shifted[:, j] = np.roll(counts[:, j], shift)
        vals = _pairwise_corr_from_counts(shifted)
        if vals.size:
            null_medians.append(float(np.median(vals)))
    if not null_medians:
        raise RuntimeError("circular-shift null produced no valid pairwise correlations")
    obs = float(np.median(observed))
    null = np.asarray(null_medians)
    return {
        "status": "ok",
        "observed_median": obs,
        "observed_q25": float(np.percentile(observed, 25)),
        "observed_q75": float(np.percentile(observed, 75)),
        "null_median": float(np.median(null)),
        "null_q95": float(np.percentile(null, 95)),
        "null_q97_5": float(np.percentile(null, 97.5)),
        "excess_over_null": float(obs - np.median(null)),
        "n_neurons": n_valid,
        "n_panel_neurons": n_panel,
        "n_pairs": int(observed.size),
        "bin_ms": float(bin_ms),
        "n_null": int(n_null),
        "null_seed": int(null_seed),
    }


def activity_and_spatial_entropy(
    spikes,
    dt_ms,
    *,
    bin_ms=5.0,
    positions=None,
    L=None,
    n_grid=16,
):
    """Active-neuron fraction and normalized spatial spike entropy per bin."""
    x, dt = _validate_raster(spikes, dt_ms)
    bs = int(round(float(bin_ms) / dt))
    if bs < 1 or not np.isclose(bs * dt, float(bin_ms)):
        raise ValueError("bin_ms is not representable at dt_ms")
    nb = x.shape[0] // bs
    if nb < 1:
        raise ValueError("raster is shorter than one activity bin")
    counts = x[: nb * bs].reshape(nb, bs, x.shape[1]).sum(axis=1, dtype=np.uint16)
    active_fraction = (counts > 0).mean(axis=1)
    out = {
        "bin_ms": float(bin_ms),
        "active_fraction_by_bin": _finite_summary(active_fraction),
        "ever_active_fraction": float(np.mean(counts.sum(axis=0) > 0)),
    }
    if positions is None:
        out["spatial_status"] = "positions_unavailable"
        out["spatial_entropy"] = _finite_summary([])
        out["active_grid_fraction"] = _finite_summary([])
        return out
    pos = np.asarray(positions, float)
    if pos.shape != (x.shape[1], 2):
        raise ValueError(f"positions must have shape ({x.shape[1]},2)")
    if L is None or not np.isfinite(float(L)) or float(L) <= 0:
        raise ValueError("positive finite L is required with positions")
    n_grid = int(n_grid)
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")
    ix = np.clip((pos[:, 0] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((pos[:, 1] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    cell = iy * n_grid + ix
    ncell = n_grid * n_grid
    entropy = np.full(nb, np.nan)
    occupied = np.zeros(nb)
    for b in range(nb):
        c = np.bincount(cell, weights=counts[b], minlength=ncell)
        occupied[b] = np.mean(c > 0)
        total = c.sum()
        if total > 0:
            p = c[c > 0] / total
            entropy[b] = -float(np.sum(p * np.log(p))) / np.log(ncell)
    out.update(
        spatial_status="ok",
        spatial_entropy=_finite_summary(entropy),
        active_grid_fraction=_finite_summary(occupied),
        n_grid=n_grid,
    )
    return out


def spatial_active_area_from_rate_grid(
    rate_grid, neurons_per_cell, *, active_floor_hz
):
    """Fraction of anatomy-occupied spatial bins above a local rate floor."""
    rates = np.asarray(rate_grid, float)
    anatomy = np.asarray(neurons_per_cell)
    if rates.ndim != 3 or anatomy.shape != rates.shape[1:]:
        raise ValueError("rate grid and anatomy grid shapes do not align")
    occupied = np.isfinite(anatomy) & (anatomy > 0)
    if not np.any(occupied):
        raise ValueError("spatial grid contains no anatomy-occupied bins")
    if not np.isfinite(float(active_floor_hz)) or float(active_floor_hz) < 0:
        raise ValueError("active_floor_hz must be finite and non-negative")
    if not np.all(np.isfinite(rates[:, occupied])):
        raise ValueError("occupied rate-grid bins must be finite")
    return np.mean(
        rates[:, occupied] >= float(active_floor_hz), axis=1
    )


def phasec_metrics_from_raster(
    spikes,
    dt_ms,
    tau_ref_ms,
    *,
    core_mask=None,
    positions=None,
    L=None,
    thresholds=None,
    pairwise_null_seed=0,
    analysis_panel_ids=None,
    pairwise_panel_ids=None,
):
    """Build the complete spike-derived Phase-C0 metrics dictionary."""
    th = thresholds or PhaseCThresholds()
    x, dt = _validate_raster(spikes, dt_ms)
    firing, rates = firing_and_ceiling_metrics(
        x,
        dt,
        tau_ref_ms,
        core_mask=core_mask,
        ceiling_rate_fraction=th.ceiling_rate_fraction,
    )
    analysis_ids = (
        np.arange(x.shape[1], dtype=int)
        if analysis_panel_ids is None
        else np.asarray(analysis_panel_ids, int)
    )
    pair_ids = (
        analysis_ids
        if pairwise_panel_ids is None
        else np.asarray(pairwise_panel_ids, int)
    )
    for name, ids in (("analysis_panel_ids", analysis_ids),
                      ("pairwise_panel_ids", pair_ids)):
        if ids.ndim != 1 or ids.size == 0 or np.any(ids < 0) or np.any(ids >= x.shape[1]):
            raise ValueError(f"{name} must contain valid E-local indices")
        if np.unique(ids).size != ids.size:
            raise ValueError(f"{name} contains duplicate indices")
    isi, cv2, lock = isi_cv2_and_refractory_lock(
        x[:, analysis_ids], dt, tau_ref_ms
    )
    fano, fano_arrays = fano_metrics(x[:, analysis_ids], dt)
    corr = pairwise_spike_count_correlation(
        x[:, pair_ids],
        dt,
        bin_ms=5.0,
        min_spikes=0,
        max_neurons=max(256, int(pair_ids.size)),
        null_seed=pairwise_null_seed,
        fixed_panel=True,
    )
    spatial = activity_and_spatial_entropy(
        x, dt, bin_ms=5.0, positions=positions, L=L
    )

    ceiling = (
        rates[analysis_ids]
        >= th.ceiling_rate_fraction * refractory_ceiling_hz(tau_ref_ms, dt)
    )
    ref_like = ceiling & np.isfinite(cv2) & np.isfinite(lock) & (cv2 <= 0.10) & (lock >= 0.80)
    ai_like = (
        (~ceiling)
        & np.isfinite(cv2)
        & np.isfinite(lock)
        & (cv2 >= th.ai_isi_cv2_median_min)
        & (lock <= th.ai_ref_lock_fraction_max)
    )
    valid_type = np.isfinite(cv2) & np.isfinite(lock)
    neuron_types = {
        "n_type_eligible": int(valid_type.sum()),
        "ai_like_neuron_fraction": (
            float(np.mean(ai_like[valid_type])) if np.any(valid_type) else None
        ),
        "refractory_like_neuron_fraction": (
            float(np.mean(ref_like[valid_type])) if np.any(valid_type) else None
        ),
    }
    return {
        "phasec_metrics_version": PHASEC_METRICS_VERSION,
        "dt_ms": float(dt),
        "tau_ref_ms": float(tau_ref_ms),
        "thresholds": asdict(th),
        "firing": firing,
        "isi": isi,
        "fano": fano,
        "pairwise_5ms": corr,
        "spatial": spatial,
        "neuron_types": neuron_types,
        # Compact private-to-caller summaries only: no N-neuron vectors are retained.
        "analysis_counts": {
            "n_fano20_valid": int(np.isfinite(fano_arrays["20ms"]).sum()),
            "n_isi_valid": int(np.isfinite(cv2).sum()),
            "analysis_panel_n": int(analysis_ids.size),
            "pairwise_panel_n": int(pair_ids.size),
        },
    }


def phasec_bootstrap_units(
    spikes,
    dt_ms,
    tau_ref_ms,
    *,
    core_mask,
    analysis_panel_ids,
    pairwise_panel_ids,
    positions,
    L,
    block_ms=500.0,
    pairwise_bin_ms=5.0,
    pairwise_n_null=100,
    pairwise_null_seed=0,
    ceiling_window_ms=250.0,
    ceiling_stride_ms=50.0,
    active_area_window_ms=25.0,
    spatial_active_floor_hz=5.0,
    n_grid=16,
):
    """Compact block/neuron units retained for hierarchical bootstrap.

    These arrays are analysis sufficient statistics, not simulator state.  The
    fixed neuron panel is supplied by the immutable panel manifest.
    """
    x, dt = _validate_raster(spikes, dt_ms)
    core = _validate_mask(core_mask, x.shape[1], "core_mask")
    panel = np.asarray(analysis_panel_ids, int)
    if panel.ndim != 1 or panel.size == 0 or np.any(panel < 0) or np.any(panel >= x.shape[1]):
        raise ValueError("analysis_panel_ids are invalid")
    if np.unique(panel).size != panel.size:
        raise ValueError("analysis_panel_ids contain duplicates")
    pair_panel = np.asarray(pairwise_panel_ids, int)
    if (
        pair_panel.ndim != 1
        or pair_panel.size < 2
        or np.any(pair_panel < 0)
        or np.any(pair_panel >= x.shape[1])
    ):
        raise ValueError("pairwise_panel_ids are invalid")
    if np.unique(pair_panel).size != pair_panel.size:
        raise ValueError("pairwise_panel_ids contain duplicates")
    bs = int(round(float(block_ms) / dt))
    if bs < 1 or not np.isclose(bs * dt, float(block_ms)):
        raise ValueError("block_ms must be exactly representable at dt_ms")
    nb = x.shape[0] // bs
    if nb < 2:
        raise ValueError("at least two bootstrap blocks are required")
    duration_s = x.shape[0] * dt * 1e-3
    full_rates = x.sum(axis=0) / duration_s
    active_core = core & (full_rates >= 5.0)
    if not np.any(active_core):
        raise ValueError("no active core neurons for rho80 bootstrap")
    block = x[:nb * bs].reshape(nb, bs, x.shape[1])
    ceiling_bs = int(round(float(ceiling_window_ms) / dt))
    ceiling_stride = int(round(float(ceiling_stride_ms) / dt))
    if (
        ceiling_bs < 1
        or ceiling_stride < 1
        or not np.isclose(ceiling_bs * dt, float(ceiling_window_ms))
        or not np.isclose(ceiling_stride * dt, float(ceiling_stride_ms))
        or ceiling_bs > bs
    ):
        raise ValueError("ceiling window/stride is invalid at dt_ms")
    ceiling_starts = np.arange(
        0, bs - ceiling_bs + 1, ceiling_stride, dtype=int
    )
    rho_block_windows = np.empty((nb, ceiling_starts.size), np.float32)
    ceiling_hz = 0.8 * refractory_ceiling_hz(tau_ref_ms, dt)
    n_active_core = int(np.sum(active_core))
    for b in range(nb):
        for wi, start in enumerate(ceiling_starts):
            # Index the time axis before applying the boolean neuron mask.
            # Combining an integer, a slice, and an advanced boolean index in
            # one expression moves the advanced axis to the front, silently
            # producing (neuron, time) rather than (time, neuron).
            window = block[b, start:start + ceiling_bs][:, active_core]
            expected_shape = (ceiling_bs, n_active_core)
            if window.shape != expected_shape:
                raise RuntimeError(
                    "active-core ceiling window has wrong axis order: "
                    f"expected {expected_shape}, got {window.shape}"
                )
            rates = (
                window.sum(axis=0, dtype=np.int32)
                / (float(ceiling_window_ms) * 1e-3)
            )
            if rates.shape != (n_active_core,):
                raise RuntimeError(
                    "active-core ceiling rates must be one value per neuron: "
                    f"expected {(n_active_core,)}, got {rates.shape}"
                )
            rho_block_windows[b, wi] = float(np.mean(rates >= ceiling_hz))

    panel_spikes = x[:, panel]
    _isi, cv2, _lock = isi_cv2_and_refractory_lock(panel_spikes, dt, tau_ref_ms)
    ref_fraction = np.full(panel.size, np.nan)
    block_cv2 = np.full((nb, panel.size), np.nan, np.float32)
    block_ref_fraction = np.full((nb, panel.size), np.nan, np.float32)
    # f_ref is the pooled-ISI probability from spec §4.1.  A median of
    # per-neuron fractions gives every neuron equal weight and is not that
    # estimand.  Keep the legacy per-neuron array only as a descriptive
    # diagnostic, and save event-count sufficient statistics separately.
    # Assign each ISI exactly once to the 500-ms block containing its second
    # spike so that intervals crossing a block boundary are retained.
    ref_numerator = np.zeros(
        (nb, len(REFRACTORY_ISI_STRATUM_NAMES)), np.int64
    )
    ref_denominator = np.zeros_like(ref_numerator)
    panel_is_core = core[panel]
    ref_limit_ms = float(tau_ref_ms) + 2.0 * dt
    for j in range(panel.size):
        spike_steps = np.flatnonzero(panel_spikes[:, j])
        times = spike_steps.astype(float) * dt
        intervals = np.diff(times)
        if intervals.size >= 5:
            ref_fraction[j] = float(np.mean(intervals <= ref_limit_ms))
        if intervals.size:
            ending_blocks = spike_steps[1:] // bs
            retained = ending_blocks < nb
            stratum = 0 if panel_is_core[j] else 1
            np.add.at(
                ref_denominator[:, stratum], ending_blocks[retained], 1
            )
            np.add.at(
                ref_numerator[:, stratum],
                ending_blocks[retained],
                (intervals[retained] <= ref_limit_ms).astype(np.int64),
            )
        for b in range(nb):
            block_times = np.flatnonzero(block[b, :, panel[j]]).astype(float) * dt
            block_isi = np.diff(block_times)
            if block_isi.size < 5:
                continue
            adjacent = block_isi[1:] + block_isi[:-1]
            cv2_values = np.divide(
                2.0 * np.abs(block_isi[1:] - block_isi[:-1]),
                adjacent,
                out=np.full(block_isi.size - 1, np.nan),
                where=adjacent > 0,
            )
            block_cv2[b, j] = float(np.nanmean(cv2_values))
            block_ref_fraction[b, j] = float(
                np.mean(block_isi <= ref_limit_ms)
            )

    pair_bs = int(round(float(pairwise_bin_ms) / dt))
    if pair_bs < 1 or not np.isclose(pair_bs * dt, float(pairwise_bin_ms)):
        raise ValueError("pairwise_bin_ms must be exactly representable at dt_ms")
    pair_bins_per_block = bs // pair_bs
    if pair_bins_per_block < 3 or pair_bins_per_block * pair_bs != bs:
        raise ValueError("block_ms must contain an integer number of pairwise bins")
    pair_counts = block[:, :, pair_panel].reshape(
        nb, pair_bins_per_block, pair_bs, pair_panel.size
    ).sum(axis=2, dtype=np.uint16)
    tri = np.triu_indices(pair_panel.size, k=1)
    pair_is_core = core[pair_panel]
    pair_strata = (
        (~pair_is_core[tri[0]]).astype(np.int8)
        + (~pair_is_core[tri[1]]).astype(np.int8)
    )
    if set(np.unique(pair_strata).tolist()) != {0, 1, 2}:
        raise ValueError(
            "pairwise panel must contain core-core, core-surround, and "
            "surround-surround pairs"
        )
    pair_corr = np.full((nb, tri[0].size), np.nan, np.float32)
    pair_null_median = np.full(
        (nb, len(PAIR_NULL_STRATUM_NAMES), int(pairwise_n_null)),
        np.nan,
        np.float32,
    )
    rng = np.random.default_rng(int(pairwise_null_seed))
    for b in range(nb):
        counts = np.asarray(pair_counts[b], float)
        sd = counts.std(axis=0, ddof=1)
        valid = np.isfinite(sd) & (sd > 0)
        if valid.sum() >= 2:
            z = np.full_like(counts, np.nan, dtype=float)
            z[:, valid] = (
                counts[:, valid] - counts[:, valid].mean(axis=0)
            ) / counts[:, valid].std(axis=0, ddof=1)
            corr = (z.T @ z) / (z.shape[0] - 1)
            pair_corr[b] = corr[tri]
        for draw in range(int(pairwise_n_null)):
            shifted = np.empty_like(counts)
            for j in range(counts.shape[1]):
                shift = int(rng.integers(1, counts.shape[0]))
                shifted[:, j] = np.roll(counts[:, j], shift)
            shifted_sd = shifted.std(axis=0, ddof=1)
            shifted_valid = np.isfinite(shifted_sd) & (shifted_sd > 0)
            shifted_corr = np.full(
                (shifted.shape[1], shifted.shape[1]), np.nan, dtype=float
            )
            if shifted_valid.sum() >= 2:
                shifted_z = (
                    shifted[:, shifted_valid]
                    - shifted[:, shifted_valid].mean(axis=0)
                ) / shifted[:, shifted_valid].std(axis=0, ddof=1)
                shifted_corr[np.ix_(shifted_valid, shifted_valid)] = (
                    shifted_z.T @ shifted_z
                ) / (shifted_z.shape[0] - 1)
            shifted_values = shifted_corr[tri]
            for stratum in range(len(PAIR_NULL_STRATUM_NAMES)):
                values = shifted_values[pair_strata == stratum]
                values = values[np.isfinite(values)]
                if values.size:
                    pair_null_median[b, stratum, draw] = float(
                        np.median(values)
                    )

    pos = np.asarray(positions, float)
    if pos.shape != (x.shape[1], 2):
        raise ValueError("positions must align with E raster")
    ix = np.clip((pos[:, 0] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((pos[:, 1] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    cell = iy * n_grid + ix
    neurons_per_cell = np.bincount(cell, minlength=n_grid * n_grid)
    occupied_cells = neurons_per_cell > 0
    n_occupied_cells = int(np.sum(occupied_cells))
    if n_occupied_cells < 1:
        raise ValueError("spatial grid contains no anatomy-occupied E bins")
    active_grid_blocks = np.zeros(nb, float)
    for b in range(nb):
        neuron_active = block[b].any(axis=0)
        active_grid_blocks[b] = (
            np.unique(cell[neuron_active]).size / n_occupied_cells
        )
    area_bs = int(round(float(active_area_window_ms) / dt))
    if (
        area_bs < 1
        or not np.isclose(area_bs * dt, float(active_area_window_ms))
        or bs % area_bs
    ):
        raise ValueError("active_area_window_ms must divide block_ms exactly")
    area_per_block = bs // area_bs
    area_counts_neuron = block.reshape(
        nb, area_per_block, area_bs, x.shape[1]
    ).sum(axis=2, dtype=np.int32)
    active_area_windows = np.zeros((nb, area_per_block), np.float32)
    area_duration_s = float(active_area_window_ms) * 1e-3
    for b in range(nb):
        for wi in range(area_per_block):
            counts_grid = np.bincount(
                cell,
                weights=area_counts_neuron[b, wi],
                minlength=n_grid * n_grid,
            )
            rates_grid = np.divide(
                counts_grid,
                neurons_per_cell * area_duration_s,
                out=np.zeros_like(counts_grid, dtype=float),
                where=occupied_cells,
            )
            active_area_windows[b, wi] = (
                spatial_active_area_from_rate_grid(
                    rates_grid.reshape(1, n_grid, n_grid),
                    neurons_per_cell.reshape(n_grid, n_grid),
                    active_floor_hz=spatial_active_floor_hz,
                )[0]
            )
    return {
        "block_ms": np.asarray(float(block_ms)),
        "rho80_active_core_by_block_window": rho_block_windows,
        "ceiling_window_ms": np.asarray(float(ceiling_window_ms)),
        "ceiling_stride_ms": np.asarray(float(ceiling_stride_ms)),
        "analysis_panel_E_ids": np.asarray(panel, np.int32),
        "pairwise_panel_E_ids": np.asarray(pair_panel, np.int32),
        "isi_cv2_by_panel_neuron": np.asarray(cv2, np.float32),
        "block_isi_cv2_by_panel_neuron": block_cv2,
        "refractory_isi_fraction_by_panel_neuron": np.asarray(
            ref_fraction, np.float32
        ),
        "block_refractory_isi_fraction_by_panel_neuron": block_ref_fraction,
        "block_refractory_isi_numerator_by_stratum": ref_numerator,
        "block_refractory_isi_denominator_by_stratum": ref_denominator,
        "refractory_isi_stratum_names": np.asarray(
            REFRACTORY_ISI_STRATUM_NAMES
        ),
        "active_grid_fraction_by_block": np.asarray(
            active_grid_blocks, np.float32
        ),
        "active_area_window_ms": np.asarray(float(active_area_window_ms)),
        "spatial_grid_n": np.asarray(int(n_grid)),
        "spatial_grid_n_occupied_E": np.asarray(n_occupied_cells),
        "spatial_grid_all_E_bins_occupied": np.asarray(
            n_occupied_cells == n_grid * n_grid
        ),
        "spatial_active_floor_hz": np.asarray(
            float(spatial_active_floor_hz)
        ),
        "spatial_area_denominator": np.asarray("anatomy_occupied_E_grid_bins"),
        "active_area_fraction_by_block_window": np.asarray(
            active_area_windows, np.float32
        ),
        "pairwise_bin_ms": np.asarray(float(pairwise_bin_ms)),
        "pairwise_null_draws": np.asarray(int(pairwise_n_null)),
        "pair_corr_by_block_and_pair": pair_corr,
        "pair_null_median_by_block_and_draw": pair_null_median,
        "pair_null_stratum_names": np.asarray(PAIR_NULL_STRATUM_NAMES),
        "pairwise_null_seed": np.asarray(int(pairwise_null_seed)),
    }


def paired_local_gain(points, *, max_relative_spread=0.25):
    """Central threshold-gain curve and a small-signal linearity gate.

    Each point must contain ``delta_mV``, ``rate_vth_minus_hz`` and
    ``rate_vth_plus_hz``.  Lowering threshold is the positive-input direction:

    ``gain = (rate_minus - rate_plus) / (2*delta)`` [Hz/mV].

    Optional ``rate_baseline_hz`` adds a central-symmetry check.  At least two
    distinct deltas are required to call the response locally linear.
    """
    rows = []
    for p in points:
        try:
            delta = float(p["delta_mV"])
            rminus = float(p["rate_vth_minus_hz"])
            rplus = float(p["rate_vth_plus_hz"])
        except (KeyError, TypeError, ValueError):
            return {"status": "invalid_point", "linearity_pass": False, "rows": []}
        if not np.isfinite([delta, rminus, rplus]).all() or delta <= 0:
            return {"status": "invalid_point", "linearity_pass": False, "rows": []}
        gain = (rminus - rplus) / (2.0 * delta)
        row = {
            "delta_mV": delta,
            "rate_vth_minus_hz": rminus,
            "rate_vth_plus_hz": rplus,
            "gain_hz_per_mV": float(gain),
            "monotone": bool(rminus >= rplus),
        }
        if p.get("rate_baseline_hz") is not None:
            base = float(p["rate_baseline_hz"])
            if not np.isfinite(base):
                return {"status": "invalid_point", "linearity_pass": False, "rows": []}
            row["rate_baseline_hz"] = base
            row["central_symmetry_error_hz"] = float(abs(0.5 * (rminus + rplus) - base))
            row["baseline_bracketed"] = bool(rminus >= base >= rplus)
        rows.append(row)
    rows.sort(key=lambda r: r["delta_mV"])
    deltas = np.asarray([r["delta_mV"] for r in rows])
    if len(rows) < 2 or np.unique(deltas).size < 2:
        return {
            "status": "insufficient_deltas",
            "linearity_pass": False,
            "rows": rows,
        }
    gains = np.asarray([r["gain_hz_per_mV"] for r in rows])
    denom = max(abs(float(np.median(gains))), 1e-12)
    rel_spread = float((np.max(gains) - np.min(gains)) / denom)
    monotone = all(r["monotone"] for r in rows)
    bracketed = all(r.get("baseline_bracketed", True) for r in rows)
    passed = bool(
        np.all(np.isfinite(gains))
        and np.all(gains >= 0)
        and rel_spread <= float(max_relative_spread)
        and monotone
        and bracketed
    )
    return {
        "status": "ok" if passed else "nonlinear_or_nonmonotone",
        "linearity_pass": passed,
        "gain_hz_per_mV_median": float(np.median(gains)),
        "gain_relative_spread": rel_spread,
        "max_relative_spread": float(max_relative_spread),
        "rows": rows,
    }


def classify_phasec_seed(metrics, *, local_gain=None, require_local_gain=True, thresholds=None):
    """Fail-closed per-seed identity of the sustained branch."""
    th = thresholds or PhaseCThresholds()
    if not isinstance(metrics, dict):
        return {"klass": "no_evidence", "reasons": ["metrics_not_mapping"]}
    try:
        ceiling_value = metrics["firing"].get("rho80_active_core_median")
        if ceiling_value is None:
            ceiling_value = metrics["firing"].get("rho80_active_median")
        ceiling = float(ceiling_value)
        ref_lock = float(metrics["isi"]["refractory_isi_fraction"])
        cv2 = float(metrics["isi"]["isi_cv2"]["median"])
        fano20 = float(metrics["fano"]["fano_by_bin"]["20ms"]["median"])
        n_isi = int(metrics["isi"]["n_isi_eligible"])
        pair = metrics["pairwise_5ms"]
        pair_excess = float(pair["excess_over_null"])
        n_pair_neurons = int(pair["n_neurons"])
        ai_frac = float(metrics["neuron_types"]["ai_like_neuron_fraction"])
        ref_frac = float(metrics["neuron_types"]["refractory_like_neuron_fraction"])
    except (KeyError, TypeError, ValueError):
        return {"klass": "no_evidence", "reasons": ["missing_or_nonfinite_required_metric"]}
    required = np.asarray([ceiling, ref_lock, cv2, fano20, pair_excess, ai_frac, ref_frac])
    if not np.all(np.isfinite(required)):
        return {"klass": "no_evidence", "reasons": ["missing_or_nonfinite_required_metric"]}
    reasons = []
    if n_isi < th.min_eligible_neurons:
        reasons.append(f"n_isi_eligible<{th.min_eligible_neurons}")
    if pair.get("status") != "ok" or n_pair_neurons < th.min_pairwise_neurons:
        reasons.append("pairwise_correlation_ineligible")
    if require_local_gain and not (
        isinstance(local_gain, dict) and bool(local_gain.get("linearity_pass"))
    ):
        reasons.append("local_gain_missing_or_nonlinear")
    if reasons:
        return {"klass": "no_evidence", "reasons": reasons}

    mixed = (
        ai_frac >= th.mixed_neuron_fraction_min
        and ref_frac >= th.mixed_neuron_fraction_min
    )
    ai = (
        ceiling <= th.ai_ceiling_fraction_max
        and ref_lock <= th.ai_ref_lock_fraction_max
        and cv2 >= th.ai_isi_cv2_median_min
        and fano20 >= th.ai_fano20_median_min
        and pair_excess <= th.ai_pairwise_excess_max
    )
    gain_value = (
        float(local_gain.get("gain_relative_to_preentry"))
        if isinstance(local_gain, dict)
        and local_gain.get("gain_relative_to_preentry") is not None
        else None
    )
    plateau = (
        ceiling >= th.plateau_ceiling_fraction_min
        and (
            ref_lock >= th.plateau_ref_lock_fraction_min
            or (gain_value is not None and gain_value <= 0.20)
        )
    )
    if mixed:
        klass = "mixed_or_unresolved"
        why = ["both_ai_like_and_refractory_like_subpopulations"]
    elif ai and not plateau:
        klass = "balanced_asynchronous_tonic_candidate"
        why = []
    elif plateau and not ai:
        klass = "refractory_limited_plateau"
        why = []
    else:
        klass = "mixed_or_unresolved"
        why = ["metrics_fall_between_pre_registered_identity_regions"]
    return {
        "klass": klass,
        "reasons": why,
        "phasec_metrics_version": PHASEC_METRICS_VERSION,
        "claim_boundary": (
            "source-space branch identity only; not ictal observation matching "
            "or lifecycle evidence"
        ),
    }


def jeffreys_interval(k, n, *, cred=0.95):
    """Jeffreys beta-binomial posterior for a replicated binary outcome."""
    k, n = int(k), int(n)
    cred = float(cred)
    if n < 1 or k < 0 or k > n:
        raise ValueError("require n>=1 and 0<=k<=n")
    if not 0 < cred < 1:
        raise ValueError("cred must lie in (0,1)")
    a, b = k + 0.5, n - k + 0.5
    q = (1.0 - cred) / 2.0
    return {
        "k": k,
        "n": n,
        "median": float(beta.ppf(0.5, a, b)),
        "mean": float(a / (a + b)),
        "lo": float(beta.ppf(q, a, b)),
        "hi": float(beta.ppf(1.0 - q, a, b)),
        "cred": cred,
    }


def aggregate_phasec_taxonomy(seed_rows, *, min_seeds=3):
    """Replicate the identity across seeds without majority-vote overclaiming.

    All eligible seeds must agree.  A 2-versus-1 split is biological/model
    heterogeneity, not a positive replication.  Jeffreys posteriors are reported
    for transparency but do not override this fail-closed unanimity rule.
    """
    by_seed = {}
    duplicates = []
    for row in seed_rows:
        if not isinstance(row, dict) or "seed" not in row:
            continue
        seed = int(row["seed"])
        klass = row.get("klass")
        if seed in by_seed and by_seed[seed] != klass:
            duplicates.append(seed)
        by_seed[seed] = klass
    if duplicates:
        return {
            "verdict": "no_evidence",
            "reasons": ["within_seed_conflict"],
            "conflicting_seeds": sorted(set(duplicates)),
        }
    if len(by_seed) < int(min_seeds):
        return {
            "verdict": "no_evidence",
            "reasons": [f"n_seeds<{int(min_seeds)}"],
            "n_seeds": len(by_seed),
        }
    eligible_labels = {
        "balanced_asynchronous_tonic_candidate",
        "refractory_limited_plateau",
    }
    ineligible = {
        s: k for s, k in by_seed.items()
        if k not in eligible_labels
    }
    posteriors = {
        label: jeffreys_interval(sum(k == label for k in by_seed.values()), len(by_seed))
        for label in sorted(eligible_labels)
    }
    if ineligible:
        return {
            "verdict": "heterogeneous_or_unresolved",
            "seed_classes": by_seed,
            "ineligible_seeds": ineligible,
            "posteriors": posteriors,
        }
    labels = set(by_seed.values())
    if len(labels) != 1:
        return {
            "verdict": "heterogeneous_or_unresolved",
            "seed_classes": by_seed,
            "posteriors": posteriors,
        }
    label = next(iter(labels))
    return {
        "verdict": f"replicated_{label}",
        "seed_classes": by_seed,
        "posteriors": posteriors,
        "n_seeds": len(by_seed),
        "claim_boundary": (
            "replicated source-space identity only; not an ictal or lifecycle verdict"
        ),
    }
