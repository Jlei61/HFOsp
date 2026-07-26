"""Source-space carrier metrics, rest distance and the probabilistic carrier taxonomy
(spec rev3.1 §4.4 / §6.3, plan Task 7).

Everything here is a PURE function of saved arrays, so the taxonomy is unit-testable on synthetic
trajectories (plateau, runaway, pulse train, fixed global oscillator, stable/metastable carrier)
without running the SNN.

The rest distance is the multivariate, baseline-standardized

    d_rest(t) = || [r_core, r_surround, A_active, E_vSEEG, H_spatial](t) - mu_rest || / sigma_rest

of spec §4.4. A trough is not a reset: the trajectory only counts as having returned to the
interictal basin when d_rest stays under the locked threshold for the locked dwell time.
"""
from __future__ import annotations

import numpy as np

METRICS_VERSION = "zm_minimal_carrier_v1_2026-07-26"

BIN_MS = 25.0            # source-space rate bin (matches the existing Z/M lifecycle runner)
REST_KEYS = ("r_core", "r_surround", "A_active", "E_vSEEG", "H_spatial")


# ================================================================ source-space metrics
def _binned(x, bin_steps, reducer):
    n = x.shape[0] // bin_steps * bin_steps
    if n == 0:
        return np.zeros(0)
    return reducer(x[:n].reshape(-1, bin_steps, *x.shape[1:]))


def source_metrics(E_spk_bool, core_mask, posE, L, dt_ms, bin_ms=BIN_MS, n_grid=16,
                   lfp_trace=None, axis_coord=None, n_axial=24):
    """Per-bin source-space readout of one continuation.

    r_core / r_surround  Hz over the two low-V_th cores vs the rest of the sheet
    A_active             fraction of E cells that fired at least once in the bin (spatial extent)
    H_spatial            normalized spatial entropy of the per-bin activity map (1 = uniform,
                         ~0 = a single hotspot) -> distinguishes recruitment from one fixed focus
    n_grid_active        number of occupied lattice cells (extent, robust to rate)
    E_vSEEG              mean current-based virtual-SEEG amplitude over contacts (if lfp given)
    kymo                 axial activity map for wavefront / first-passage diagnostics
    """
    E = np.asarray(E_spk_bool)
    nsteps, NE = E.shape
    bs = max(1, int(round(bin_ms / dt_ms)))
    nb = nsteps // bs
    core = np.asarray(core_mask, bool)
    out = {}
    if nb == 0:
        return dict(n_bins=0, bin_ms=bin_ms)
    Eb = E[:nb * bs].reshape(nb, bs, NE)
    fired = Eb.any(axis=1)                                   # (nb, NE) distinct neurons per bin
    cnt = Eb.sum(axis=1)                                     # (nb, NE) spikes per bin
    dur_s = bs * dt_ms * 1e-3
    out["r_core"] = cnt[:, core].sum(axis=1) / max(1, core.sum()) / dur_s
    out["r_surround"] = cnt[:, ~core].sum(axis=1) / max(1, (~core).sum()) / dur_s
    out["r_all"] = cnt.sum(axis=1) / NE / dur_s
    out["A_active"] = fired.mean(axis=1)

    ix = np.clip((np.asarray(posE)[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((np.asarray(posE)[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    cell = iy * n_grid + ix
    ncell = n_grid * n_grid
    H = np.zeros(nb)
    nact = np.zeros(nb)
    for b in range(nb):
        c = np.bincount(cell, weights=cnt[b], minlength=ncell)
        tot = c.sum()
        nact[b] = float((c > 0).sum())
        if tot > 0:
            p = c[c > 0] / tot
            H[b] = float(-(p * np.log(p)).sum() / np.log(ncell))
    out["H_spatial"] = H
    out["n_grid_active"] = nact

    if lfp_trace is not None:
        lf = np.abs(np.asarray(lfp_trace, float))
        out["E_vSEEG"] = _binned(lf, bs, lambda a: a.mean(axis=(1, 2)))[:nb]
    else:
        out["E_vSEEG"] = np.zeros(nb)

    if axis_coord is not None:
        edges = np.linspace(np.min(axis_coord), np.max(axis_coord) + 1e-9, n_axial + 1)
        abin = np.clip(np.digitize(axis_coord, edges) - 1, 0, n_axial - 1)
        kymo = np.zeros((n_axial, nb))
        for b in range(nb):
            kymo[:, b] = np.bincount(abin, weights=cnt[b], minlength=n_axial)
        out["kymo_axial"] = kymo
    out["n_bins"] = nb
    out["bin_ms"] = bin_ms
    return out


# ================================================================ rest distance
def rest_reference(metrics, lo_bin=0, hi_bin=None):
    """mu/sigma of the REST_KEYS over a pre-event interictal window of the anchor."""
    hi = metrics["n_bins"] if hi_bin is None else hi_bin
    ref = {}
    for k in REST_KEYS:
        v = np.asarray(metrics[k], float)[lo_bin:hi]
        ref[k] = dict(mu=float(v.mean()), sd=float(v.std()) if v.size > 1 else 0.0)
    ref["n_bins"] = int(hi - lo_bin)
    return ref


def rest_distance(metrics, ref, floor_sd=None):
    """Standardized Euclidean distance from the interictal rest distribution (spec §4.4).

    `floor_sd` guards a degenerate baseline: if a coordinate never moves during rest, its sd is 0
    and any deviation would read as infinite. The floor is a fraction of |mu| (or an absolute floor
    when mu is 0 too), locked in Phase 0C.
    """
    n = metrics["n_bins"]
    d2 = np.zeros(n)
    for k in REST_KEYS:
        mu, sd = ref[k]["mu"], ref[k]["sd"]
        fl = (0.05 * abs(mu) if floor_sd is None else floor_sd * max(abs(mu), 1e-12))
        s = max(sd, fl, 1e-12)
        d2 += ((np.asarray(metrics[k], float)[:n] - mu) / s) ** 2
    return np.sqrt(d2 / len(REST_KEYS))


# ================================================================ survival / stationarity
def first_rest_return(d_rest, bin_ms, d_thresh, dwell_ms):
    """Index of the bin at which the trajectory has definitively returned to the rest basin: the
    START of the first run of >= dwell_ms consecutive bins with d_rest < d_thresh. None if never."""
    below = np.asarray(d_rest) < d_thresh
    need = max(1, int(round(dwell_ms / bin_ms)))
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= need:
            return i - need + 1
    return None


def survival(d_rest, bin_ms, d_thresh, dwell_ms, T_ms, runaway_ms=None, plateau_bin=None):
    """Lifetime of the active state and whether it survived the whole window.

    A runaway or a saturated plateau ends the carrier state too: neither is a bounded carrier.
    """
    idx = first_rest_return(d_rest, bin_ms, d_thresh, dwell_ms)
    ends = []
    if idx is not None:
        ends.append((idx * bin_ms, "rest_return"))
    if runaway_ms is not None:
        ends.append((float(runaway_ms), "runaway"))
    if plateau_bin is not None:
        ends.append((plateau_bin * bin_ms, "saturated_plateau"))
    if not ends:
        return dict(lifetime_ms=float(T_ms), survived=True, end_reason=None)
    t_end, why = min(ends)
    return dict(lifetime_ms=float(t_end), survived=False, end_reason=why)


def drift_stats(x, half=True):
    """Monotonic-drift check over the latter half (spec §6.3 'no systematic monotonic drift')."""
    v = np.asarray(x, float)
    v = v[len(v) // 2:] if half else v
    if v.size < 4:
        return dict(slope_per_s=float("nan"), rel_drift=float("nan"), cv=float("nan"))
    t = np.arange(v.size, dtype=float)
    slope = float(np.polyfit(t, v, 1)[0])
    m = float(np.mean(v))
    return dict(slope_per_s=slope, rel_drift=float(slope * v.size / m) if m else float("nan"),
                cv=float(np.std(v) / m) if m else float("nan"))


# ================================================================ taxonomy
#: P(p > threshold) inside this band means the replicas do not resolve that decision threshold
INDET_BAND = (0.25, 0.75)


def jeffreys_posterior(k, n, cred=0.95):
    """Beta-binomial posterior for P_carrier with the Jeffreys prior Beta(1/2,1/2) (spec §6.3)."""
    from scipy.stats import beta
    a, b = 0.5 + k, 0.5 + (n - k)
    lo, hi = beta.ppf([(1 - cred) / 2, 1 - (1 - cred) / 2], a, b)
    return dict(k=int(k), n=int(n), median=float(beta.ppf(0.5, a, b)),
                mean=float(a / (a + b)), lo=float(lo), hi=float(hi), cred=cred,
                mass_above={str(t): float(1.0 - beta.cdf(t, a, b)) for t in (0.3, 0.8)})


def classify_replicas(replicas, ied_lifetime_ms, cred=0.95):
    """The locked probabilistic classes of spec §6.3.

    `replicas` = list of dicts with keys survived / lifetime_ms / end_reason / drift / rest_returns.
    Threshold-edge posteriors return `probabilistically_indeterminate` rather than being forced.
    """
    n = len(replicas)
    if n == 0:
        return dict(klass="no_evidence", posterior=None, n=0)
    k = sum(bool(r["survived"]) for r in replicas)
    post = jeffreys_posterior(k, n, cred=cred)
    med = post["median"]
    life = np.array([float(r["lifetime_ms"]) for r in replicas])
    beats_ied = bool(np.median(life) > 2.0 * float(ied_lifetime_ms))
    resets = np.array([int(r.get("rest_returns", 0)) for r in replicas])
    repeated_reset = bool(np.median(resets) >= 2)
    runaway = sum(r.get("end_reason") == "runaway" for r in replicas)
    plateau = sum(r.get("end_reason") == "saturated_plateau" for r in replicas)

    if runaway >= max(1, n // 2 + 1):
        return dict(klass="runaway", posterior=post, k=k, n=n)
    if plateau >= max(1, n // 2 + 1):
        return dict(klass="saturated_plateau", posterior=post, k=k, n=n)
    # Threshold-edge uncertainty -> indeterminate, never forced up or down (spec §6.3). The rule is
    # posterior MASS, not the median's distance to the line: a decision threshold is unresolved when
    # the data leave P(p > thr) between INDET_BAND. With 3 replicas that is usually the honest
    # answer; with 10 it usually resolves. Pre-registered before any real fork was inspected.
    for thr in (0.3, 0.8):
        mass = float(post["mass_above"][str(thr)])
        if INDET_BAND[0] <= mass <= INDET_BAND[1]:
            return dict(klass="probabilistically_indeterminate", posterior=post, k=k, n=n,
                        unresolved_threshold=thr, mass_above=mass)
    if med > 0.8:
        return dict(klass="stable_carrier" if beats_ied else "transient_carrier_like",
                    posterior=post, k=k, n=n, beats_ied=beats_ied)
    if med > 0.3:
        if repeated_reset:
            return dict(klass="hfo_like_relaxation_train", posterior=post, k=k, n=n)
        return dict(klass="metastable_carrier" if beats_ied else "transient_carrier_like",
                    posterior=post, k=k, n=n, beats_ied=beats_ied)
    if repeated_reset:
        return dict(klass="hfo_like_relaxation_train", posterior=post, k=k, n=n)
    return dict(klass="transient_carrier_like", posterior=post, k=k, n=n, beats_ied=beats_ied)


POSITIVE_CLASSES = ("stable_carrier", "metastable_carrier")
#: partial order of spec §6.3: the smallest dynamic subsystem that still supports a carrier
SUBSYSTEM_OF_ARM = {
    "freeze_all": "carrier_fast_only",
    "freeze_zm": "carrier_fast_plus_sg",
    "freeze_zsg": "carrier_fast_plus_m",
    "freeze_z": "carrier_fast_plus_m_sg",
}


def smallest_positive_subsystem(arm_classes):
    """arm_classes: {arm: klass}. Returns the minimal positive subsystem label(s), or None.

    Ties are reported, not broken: if both one-variable subsystems pass, both are returned.
    """
    if arm_classes.get("freeze_all") in POSITIVE_CLASSES:
        return ["carrier_fast_only"]
    ones = [SUBSYSTEM_OF_ARM[a] for a in ("freeze_zm", "freeze_zsg")
            if arm_classes.get(a) in POSITIVE_CLASSES]
    if ones:
        return sorted(ones)
    if arm_classes.get("freeze_z") in POSITIVE_CLASSES:
        return ["carrier_fast_plus_m_sg"]
    return None
