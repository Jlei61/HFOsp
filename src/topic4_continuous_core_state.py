"""One continuous slow coordinate acting through fixed local I projections.

This is an external slow-state experiment, not endogenous bistability.  The
legacy master RNG and all E afferent draws remain unchanged across conditions.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.special import pdtr
from scipy.signal import lfilter


def ou_path(n_steps, dt_ms, tau_ms, seed):
    """Exact OU transition, stationary N(0,1), independent of simulator RNG."""
    if n_steps < 1 or dt_ms <= 0 or tau_ms <= 0:
        raise ValueError("positive length and time constants required")
    rng = np.random.default_rng(seed)
    alpha = np.exp(-dt_ms / tau_ms)
    initial = rng.standard_normal()
    eps = rng.standard_normal(n_steps) * np.sqrt(-np.expm1(-2*dt_ms/tau_ms))
    eps[0] = initial
    return lfilter([1.0], [1.0, -alpha], eps)


def poisson_quantile(u, mu):
    """Poisson inverse CDF for small afferent count means, with exact fallback."""
    u, mu = np.broadcast_arrays(np.asarray(u, float), np.asarray(mu, float))
    if np.any(mu < 0) or np.any(~np.isfinite(mu)) or np.any((u <= 0) | (u >= 1)):
        raise ValueError("invalid Poisson quantile input")
    if mu.max(initial=0) > 30:
        from scipy.stats import poisson
        return poisson.ppf(u, mu).astype(np.int64)
    prob = np.exp(-mu)
    cdf = prob.copy()
    result = np.zeros(u.shape, np.int64)
    active = u > cdf
    k = 0
    while np.any(active):
        k += 1
        prob *= mu / k
        cdf += prob
        result[active] = k
        active = u > cdf
        if k > 150:
            from scipy.stats import poisson
            return poisson.ppf(u, mu).astype(np.int64)
    return result


def coupled_counts(base, mu, gain, uniforms):
    """Randomized probability-integral transform of legacy Poisson counts.

    U=F_mu(K-1)+v*(F_mu(K)-F_mu(K-1)) is uniform when K~Poisson(mu).
    Transforming U with the new inverse CDF gives exact new Poisson marginals
    and monotone coupling.  Gain=1 is a literal identity (including tails).
    """
    base, mu, gain, uniforms = np.broadcast_arrays(base, mu, gain, uniforms)
    if np.any(gain <= 0) or np.any(~np.isfinite(gain)):
        raise ValueError("positive finite gain required")
    out = np.asarray(base, np.int64).copy()
    changed = (gain != 1) & (mu > 0)
    if not changed.any():
        return out
    k, m, g, v = (np.asarray(a)[changed] for a in (base, mu, gain, uniforms))
    lo = np.where(k > 0, pdtr(np.maximum(k-1, 0), m), 0.)
    hi = pdtr(k, m)
    u = np.clip(lo + v*(hi-lo), np.nextafter(0., 1.), np.nextafter(1., 0.))
    out[changed] = poisson_quantile(u, m*g)
    return out


def spatial_groups(substrate, centers):
    ne = substrate.n_e
    pos = np.asarray(substrate.net['pos'])
    distances = np.linalg.norm(pos[:, None] - np.asarray(centers)[None], axis=-1)
    nearest = distances.argmin(axis=1)
    h = np.asarray(substrate.h_e) > 0
    radius = float(distances[:ne][h].min(axis=1).max())
    groups = {}
    for j, name in enumerate(['coreA', 'coreB']):
        groups[name+'E'] = np.flatnonzero(h & (nearest[:ne] == j))
        groups[name+'I'] = ne + np.flatnonzero(
            (distances[ne:].min(axis=1) <= radius) & (nearest[ne:] == j))
    groups['surroundE'] = np.flatnonzero(~h)
    groups['allI'] = np.arange(ne, len(pos))
    a, b = groups['coreAI'], groups['coreBI']
    if min(len(a), len(b)) == 0:
        raise ValueError('no I cells in one of the fixed geometric cores')
    indices = np.r_[a, b]
    nmin = min(len(a), len(b))
    loading = np.r_[np.full(len(a), nmin/len(a)), np.full(len(b), -nmin/len(b))]
    assert abs(loading.sum()) < 1e-10 and np.max(abs(loading)) <= 1
    return groups, indices, loading, radius


class ContinuousIState:
    def __init__(self, indices, loading, z, *, amplitude, dt_ms, seed,
                 warmup_ms=1000., ramp_ms=200.):
        self.indices = np.asarray(indices, int)
        self.loading = np.asarray(loading, float)
        self.z = np.asarray(z, float)
        if (self.loading.shape != self.indices.shape or len(np.unique(self.indices)) != len(self.indices)
                or np.max(abs(self.loading)) > 1 or abs(self.loading.sum()) > 1e-9
                or not 0 <= amplitude < 1 or not np.all(np.isfinite(self.z))):
            raise ValueError('invalid fixed state projection')
        self.amplitude, self.dt_ms = float(amplitude), float(dt_ms)
        self.warmup_ms, self.ramp_ms = warmup_ms, ramp_ms
        self.rng = np.random.default_rng(seed)
        self.digest = hashlib.sha256()
        self.prefix_digest = None
        self.n_seen = 0
        self.maximum_relative_total_rate_error = 0.
        self.changed_counts = 0
        self.base_counts = np.zeros(len(indices), np.int64)
        self.actual_counts = np.zeros(len(indices), np.int64)
        self.expected_base = np.zeros(len(indices))
        self.expected_actual = np.zeros(len(indices))
        self.q = np.zeros(len(z))

    def apply(self, ext, nu_vec, step, dt):
        if step != self.n_seen or dt != self.dt_ms:
            raise ValueError('state replay must start at zero and advance contiguously')
        # Hash all unmodified external counts, not only the selected population.
        if ext.max(initial=0) > 65535:
            raise ValueError('external count exceeds digest representation')
        self.digest.update(ext.astype(np.uint16).tobytes())
        if step+1 == round(6000./dt):
            self.prefix_digest = self.digest.copy().hexdigest()
        v = self.rng.random(len(self.indices))  # fixed consumption independent of state
        ramp = np.clip((step*dt-self.warmup_ms)/self.ramp_ms, 0, 1)
        q = self.amplitude*np.tanh(self.z[step])*ramp
        self.q[step] = q
        gain = 1+q*self.loading
        mu = nu_vec[self.indices]*dt
        target = mu*gain
        error = abs(target.sum()-mu.sum()) / max(mu.sum(), 1e-30)
        self.maximum_relative_total_rate_error = max(self.maximum_relative_total_rate_error, error)
        if error > 1e-12:
            raise ValueError('I input rate budget not conserved')
        base = ext[self.indices].astype(np.int64)
        actual = coupled_counts(base, mu, gain, v)
        self.changed_counts += int(np.count_nonzero(actual != base))
        self.base_counts += base
        self.actual_counts += actual
        self.expected_base += mu
        self.expected_actual += target
        ext[self.indices] = actual
        self.n_seen += 1

    def audit(self):
        return dict(n_steps=self.n_seen, legacy_external_counts_sha256=self.digest.hexdigest(),
                    legacy_first_6s_sha256=self.prefix_digest,
                    maximum_relative_total_rate_error=self.maximum_relative_total_rate_error,
                    changed_cell_steps=self.changed_counts,
                    coupling='legacy Poisson randomized PIT; independent fixed-size uniform stream',
                    conservation='expected I external count sum per step, not realized stochastic counts')
