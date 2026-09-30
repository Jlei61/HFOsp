"""Local joint-moment transport candidate; not an accepted spatial rate model.

Retain voltage as well as four colored-current coordinates within each bin.
Gaussian bin transitions preserve their first two raw moments instead of
randomly depositing voltage at neighboring nodes. No sampled neurons.
"""
import math
import numpy as np
from numba import njit
from conditional_current_density import normal_integrals, voltage_grid, advance


@njit(cache=True)
def joint_advance(raw, pars, mu, ve, vi, active):
    """Exact linear Gaussian current/membrane step before threshold/reset."""
    left = np.empty((6, 6)); out = np.empty((6, 6))
    for j in range(6):
        left[0, j] = raw[0, j]; left[1, j] = raw[1, j]
        left[2, j] = pars[6]*raw[2, j]
        left[3, j] = pars[7]*raw[2, j]+pars[8]*raw[3, j]
        left[4, j] = pars[17]*raw[4, j]
        left[5, j] = pars[9]*raw[4, j]+pars[10]*raw[5, j]
    for i in range(6):
        out[i, 0] = left[i, 0]; out[i, 1] = left[i, 1]
        out[i, 2] = pars[6]*left[i, 2]
        out[i, 3] = pars[7]*left[i, 2]+pars[8]*left[i, 3]
        out[i, 4] = pars[17]*left[i, 4]
        out[i, 5] = pars[9]*left[i, 4]+pars[10]*left[i, 5]
    mass = raw[0, 0]
    out[2, 2] += mass*ve*pars[11]**2
    out[2, 3] += mass*ve*pars[11]*pars[12]; out[3, 2] = out[2, 3]
    out[3, 3] += mass*ve*(pars[12]**2+pars[13]**2)
    out[4, 4] += mass*vi*pars[14]**2
    out[4, 5] += mass*vi*pars[14]*pars[15]; out[5, 4] = out[4, 5]
    out[5, 5] += mass*vi*(pars[15]**2+pars[16]**2)
    left = out.copy()
    if active:
        a = pars[18]; d = 1-a
        for j in range(6):
            left[1, j] = a*out[1, j]+d*(mu*out[0, j]+out[3, j]-out[5, j])
        answer = left.copy()
        for i in range(6):
            answer[i, 1] = a*left[i, 1]+d*(mu*left[i, 0]+left[i, 3]-left[i, 5])
    else:
        for j in range(6):
            left[1, j] = pars[21]*out[0, j]
        answer = left.copy()
        for i in range(6):
            answer[i, 1] = pars[21]*left[i, 0]
    return answer


@njit(cache=True)
def truncate(raw, mean, direction, lo, hi):
    m0, m1, m2, _ = normal_integrals(lo, hi)
    result = raw*m0; mass = raw[0, 0]
    for i in range(5):
        result[0, i+1] += mass*direction[i]*m1
        result[i+1, 0] = result[0, i+1]
        for j in range(5):
            result[i+1, j+1] += mass*((mean[i]*direction[j]+direction[i]*mean[j])*m1+
                                             direction[i]*direction[j]*(m2-m0))
    return result


@njit(cache=True)
def reset_voltage(raw, reset):
    answer = raw.copy()
    for j in range(6):
        answer[1, j] = reset*raw[0, j]
        answer[j, 1] = reset*raw[j, 0]
    answer[1, 1] = reset*reset*raw[0, 0]
    return answer


@njit(cache=True)
def joint_partition(raw, edges, reset, output, spikes):
    mass = raw[0, 0]
    if mass < 1e-24:
        return mass, 0., 0.
    mean = raw[0, 1:]/mass
    covariance = raw[1:, 1:]/mass-np.outer(mean, mean)
    variance = covariance[0, 0]
    scale = max(abs(raw[1, 1]/mass), mean[0]**2, 1.)
    if variance < -1e-10*scale:
        raise ValueError('Nonpositive joint voltage covariance')
    threshold = edges[-1]
    if variance <= 1e-13*scale:
        if mean[0] >= threshold:
            spikes += reset_voltage(raw, reset)
        else:
            k = max(0, min(len(output)-1, np.searchsorted(edges, mean[0], side='right')-1))
            output[k] += raw
        return 0., mass if mean[0] < edges[1] else 0., min(variance/scale, 0.)
    sigma = math.sqrt(variance); direction = covariance[:, 0]/sigma
    high = (threshold-mean[0])/sigma
    if high < 10:
        spikes += reset_voltage(truncate(raw, mean, direction, max(-40., high), 40.), reset)
    k0 = max(0, np.searchsorted(edges, mean[0]-10*sigma, side='right')-1)
    k1 = min(len(output), np.searchsorted(edges, mean[0]+10*sigma, side='right'))
    for k in range(k0, k1):
        lo = max(-40., (edges[k]-mean[0])/sigma)
        hi = min(40., (edges[k+1]-mean[0])/sigma)
        if hi > lo:
            output[k] += truncate(raw, mean, direction, lo, hi)
    low = (edges[1]-mean[0])/sigma
    low_mass = mass*normal_integrals(-40., min(40., low))[0] if low > -10 else 0.
    return 0., low_mass, min(variance/scale, 0.)


def voltage_edges(theta, reset, nodes):
    return np.r_[-np.inf, voltage_grid(theta, reset, nodes), theta]


@njit(cache=True)
def simulate_joint(pars, wave, edges, dt, period, burn_steps, record_steps, bins,
                   baseline_mu=np.nan, baseline_ve=0., baseline_vi=0., phase_offset_ms=0.):
    n = len(edges)-1; nref = int(pars[19]); size = wave.shape[1]
    reset_bin = np.searchsorted(edges, pars[21], side='right')-1
    free = np.zeros((n, 6, 6)); initial = np.zeros(6); initial[0] = 1.; initial[1] = pars[21]
    free[reset_bin] = np.outer(initial, initial)
    refractory = np.zeros((nref, 6, 6)); head = 0
    expected = np.zeros((5, 5)); expected[0, 0] = 1.
    indices = np.array([0, 2, 3, 4, 5])
    rates = np.zeros(record_steps); counts = np.zeros(bins); exposure = np.zeros(bins)
    voltage = np.zeros(bins); ref_sum = np.zeros(bins)
    max_mass = 0.; max_current = 0.; removed = 0.; lower = 0.; negative = 0.
    for t in range(-burn_steps, record_steps):
        phase = (((t+1)*dt+phase_offset_ms)/period) % 1.; pos = phase*size
        lo = int(pos) % size; hi = (lo+1) % size; f = pos-math.floor(pos)
        mu = (1-f)*wave[0, lo]+f*wave[0, hi]
        ve = (1-f)*wave[1, lo]+f*wave[1, hi]
        vi = (1-f)*wave[2, lo]+f*wave[2, hi]
        if t < 0 and not math.isnan(baseline_mu):
            mu = baseline_mu; ve = baseline_ve; vi = baseline_vi
        # Release before the common current/membrane step, exactly matching
        # decrement-then-test refractory ordering of the native discrete LIF.
        free[reset_bin] += refractory[head]; refractory[head] = 0.
        for k in range(nref):
            if refractory[k, 0, 0] >= 1e-24:
                refractory[k] = joint_advance(refractory[k], pars, mu, ve, vi, False)
        new = np.zeros((n, 6, 6)); spikes = np.zeros((6, 6))
        for k in range(n):
            if free[k, 0, 0] >= 1e-24:
                raw = joint_advance(free[k], pars, mu, ve, vi, True)
                cut, under, cov = joint_partition(raw, edges, pars[21], new, spikes)
                removed += cut; lower += under; negative = min(negative, cov)
            else:
                removed += free[k, 0, 0]
        free = new; refractory[head] = spikes; head = (head+1) % nref
        expected = advance(expected, pars, ve, vi)
        if t >= 0:
            b = min(int(phase*bins), bins-1)
            counts[b] += spikes[0, 0]; exposure[b] += dt
            rates[t] = spikes[0, 0]/dt*1000
            for k in range(n):
                voltage[b] += free[k, 0, 1]
            for k in range(nref):
                voltage[b] += refractory[k, 0, 1]; ref_sum[b] += refractory[k, 0, 0]
        if t % 100 == 0:
            total = np.zeros((6, 6))
            for k in range(n): total += free[k]
            for k in range(nref): total += refractory[k]
            max_mass = max(max_mass, abs(total[0, 0]-1.))
            for i in range(5):
                for j in range(5):
                    max_current = max(max_current, abs(total[indices[i], indices[j]]-expected[i, j])/max(1., abs(expected[i, j])))
    return (counts/exposure*1000, voltage/(exposure/dt), ref_sum/(exposure/dt), rates,
            free, refractory, np.array([max_mass, max_current, removed, lower, negative]))
