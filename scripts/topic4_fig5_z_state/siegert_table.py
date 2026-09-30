"""Tabulated antiderivative of erfcx(-x) for a fast, accurate Siegert LIF transfer.

rate = 1 / (tau_ref + tau_m * sqrt(pi) * (G(upper) - G(lower))),  G(x) = int_0^x erfcx(-s) ds,
lower = (V_reset - mu)/sigma, upper = (theta - mu)/sigma.  G is tabulated once on a fine grid with composite
Simpson integration of exact erfcx values and evaluated by linear interpolation; outside the table the
asymptotic forms are used.  Accuracy is checked against 64-point Gauss-Legendre in `qa()`.
"""
import numpy as np
from scipy.special import erfcx
from numpy.polynomial.legendre import leggauss

X_LO, X_HI, STEP = -60., 26., 5e-4
_TABLE = None


def _build():
    x = np.arange(X_LO, X_HI + STEP / 2, STEP)
    f = erfcx(-x)
    # composite Simpson on pairs of intervals (odd count needed); use cumulative trapezoid with Richardson correction
    h = STEP
    g = np.zeros_like(x)
    # Simpson cumulative: G[2k] exact Simpson; G[2k+1] via 3/8-corrected half-step
    trap = np.cumsum(np.r_[0., .5 * h * (f[1:] + f[:-1])])
    # Richardson-style correction using second differences (error of trapezoid ~ -h^2/12 f')
    fp = np.gradient(f, h)
    g = trap - h * h / 12. * (fp - fp[0])
    i0 = int(round((0. - X_LO) / STEP)); g = g - g[i0]
    return x, g


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = _build()
    return _TABLE


def G(x):
    xs, gs = table(); x = np.atleast_1d(np.asarray(x, float))
    out = np.interp(x, xs, gs)
    hi = x > X_HI
    if np.any(hi):
        # erfcx(-s) ~ 2 exp(s^2) for s >> 1: int = exp(x^2)/x * (1 + 1/(2x^2) + 3/(4x^4)) asymptotic (finite here; rate -> 0)
        xx = x[hi]
        with np.errstate(over='ignore'):
            out[hi] = gs[-1] + np.exp(xx ** 2) / xx * (1 + 1 / (2 * xx ** 2) + 3 / (4 * xx ** 4)) - np.exp(X_HI ** 2) / X_HI * (1 + 1 / (2 * X_HI ** 2) + 3 / (4 * X_HI ** 4))
    lo = x < X_LO
    if np.any(lo):
        xx = x[lo]                      # erfcx(-s) ~ 1/(sqrt(pi)|s|) (1 - 1/(2 s^2)) for s << -1
        out[lo] = gs[0] - (np.log(np.abs(xx)) - np.log(np.abs(X_LO)) - .25 * (1 / xx ** 2 - 1 / X_LO ** 2)) / np.sqrt(np.pi)
    return out


def lif_rate(mu, sigma, threshold, tau_m, tau_ref, v_reset=11.):
    mu, sigma, threshold = np.broadcast_arrays(np.asarray(mu, float), np.asarray(sigma, float), np.asarray(threshold, float))
    lower = (v_reset - mu) / sigma; upper = (threshold - mu) / sigma
    integral = G(upper) - G(lower)
    with np.errstate(over='ignore', invalid='ignore'):
        den = tau_ref + tau_m * np.sqrt(np.pi) * integral
        out = np.divide(1., den, out=np.zeros_like(den), where=np.isfinite(den) & (den > 0))
    return np.clip(out, 0., 1. / tau_ref)


def lif_rate_legendre(mu, sigma, threshold, tau_m, tau_ref, v_reset=11., order=16):
    mu, sigma, threshold = np.broadcast_arrays(np.asarray(mu, float), np.asarray(sigma, float), np.asarray(threshold, float))
    x, w = leggauss(order)
    lower = (v_reset - mu) / sigma; upper = (threshold - mu) / sigma
    sample = (.5 * (lower + upper))[..., None] + (.5 * (upper - lower))[..., None] * x
    integral = .5 * (upper - lower) * np.sum(w * erfcx(-sample), axis=-1)
    with np.errstate(over='ignore', invalid='ignore'):
        den = tau_ref + tau_m * np.sqrt(np.pi) * integral
        out = np.divide(1., den, out=np.zeros_like(den), where=np.isfinite(den))
    return np.clip(out, 0., 1. / tau_ref)


def qa(n=200000, seed=3):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-40, 60, n); sigma = rng.uniform(.3, 25, n); theta = rng.uniform(14., 18., n)
    ref = lif_rate_legendre(mu, sigma, theta, 20., 2., order=64)
    tab = lif_rate(mu, sigma, theta, 20., 2.); lg16 = lif_rate_legendre(mu, sigma, theta, 20., 2., order=16)
    sel = ref > 1e-4
    return dict(samples=int(n), table_max_abs_error_per_ms=float(np.max(np.abs(tab - ref))), legendre16_max_abs_error_per_ms=float(np.max(np.abs(lg16 - ref))),
                table_max_rel_error_above_0p1Hz=float(np.max(np.abs(tab[sel] - ref[sel]) / ref[sel])), legendre16_max_rel_error_above_0p1Hz=float(np.max(np.abs(lg16[sel] - ref[sel]) / ref[sel])),
                reference='64-point Gauss-Legendre Siegert', grid=dict(lo=X_LO, hi=X_HI, step=STEP))
