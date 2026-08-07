"""Which small nudge grows, and into what: an empirical finite-time response operator.

A 40000-cell spiking network with resets, refractory counters and delay rings is a hybrid system;
writing an ordinary Jacobian for it is not clean, and the plan is explicit that the main figure
does not need one.  What it needs is the operational version of the same question: from a frozen
slow state, which spatial perturbation is amplified most over a finite window, and what shape does
the amplified response take.

The construction is a finite-difference operator, not a linearisation claim:

* pick K smooth spatial directions p_k over the sheet;
* from the same state, run +eps*p_k and -eps*p_k **with the same generator state**, so the noise is
  common to both and cancels to first order in the difference;
* coarse-grain each run to bins and take (X_+ - X_-) / (2 eps) as that direction's response;
* stack the responses into R_tau and solve R_tau = K_tau P for the operator.

The leading right singular vector of K_tau is the perturbation that grows most; the leading left
singular vector is what it grows into; the leading singular value is the gain.  Reporting all three
alongside the spectrum is what keeps "the network amplifies a two-lobed mode" from being read off a
response that has no dominant direction.

Paired differencing is the part that makes this measurable at all.  A single perturbed run differs
from an unperturbed one by the perturbation *and* by every spike the noise moved; at these
amplitudes the second is much larger than the first.
"""
from __future__ import annotations

import numpy as np

EPS_DEFAULT = 0.25       # mV on a threshold of 18 and a reset of 11: small against the 7 mV swing


def gaussian_basis(pos, n_side=4, sigma_mm=2.5, L=20.0):
    """K smooth spatial directions, one bump per centre, each normalised to unit norm."""
    pos = np.asarray(pos, float)
    centres = [(x, y)
               for x in np.linspace(L / (2 * n_side), L - L / (2 * n_side), n_side)
               for y in np.linspace(L / (2 * n_side), L - L / (2 * n_side), n_side)]
    P = np.empty((len(pos), len(centres)), float)
    for k, (cx, cy) in enumerate(centres):
        d2 = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
        v = np.exp(-d2 / (2.0 * sigma_mm ** 2))
        P[:, k] = v / max(np.linalg.norm(v), 1e-12)
    return P, np.asarray(centres, float)


def bin_response(spikes, pos, grid=16, L=20.0):
    """Coarse-grain a spike record to a grid of counts -- the operator's output coordinates."""
    spikes = np.asarray(spikes)
    pos = np.asarray(pos, float)
    ix = np.clip((pos[:, 0] / L * grid).astype(int), 0, grid - 1)
    iy = np.clip((pos[:, 1] / L * grid).astype(int), 0, grid - 1)
    flat = iy * grid + ix
    counts = spikes.sum(axis=0) if spikes.ndim == 2 else spikes
    return np.bincount(flat, weights=np.asarray(counts, float), minlength=grid * grid)


def response_operator(responses, basis_grid):
    """Solve R = K P for K, where P holds each direction expressed in the same output coordinates.

    ``responses`` is (n_bins, K); ``basis_grid`` is (n_bins, K) -- the same directions binned the
    same way, so the operator maps the sheet to itself rather than mixing two coordinate systems.
    """
    R = np.asarray(responses, float)
    P = np.asarray(basis_grid, float)
    if R.shape != P.shape:
        raise ValueError(f"responses {R.shape} and basis {P.shape} must share a shape")
    K = R @ np.linalg.pinv(P)
    u, s, vt = np.linalg.svd(K, full_matrices=False)
    total = float(np.sum(s ** 2))
    return dict(operator=K, singular=s,
                optimal_perturbation=vt[0], response_pattern=u[:, 0],
                gain=float(s[0] ** 2),
                leading_share=float(s[0] ** 2 / max(total, 1e-12)),
                spectrum=[float(v) for v in s])


def paired_difference(plus_counts, minus_counts, eps):
    """The first-order response, with the noise the two runs share removed by construction."""
    return (np.asarray(plus_counts, float) - np.asarray(minus_counts, float)) / (2.0 * float(eps))


def alignment(pattern, target):
    """How much of a response pattern lies along a reference field, sign-free."""
    a = np.asarray(pattern, float).ravel()
    b = np.asarray(target, float).ravel()
    if a.size != b.size:
        raise ValueError(f"pattern {a.size} and target {b.size} must have the same length")
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(a @ b) / den) if den > 0 else float("nan")
