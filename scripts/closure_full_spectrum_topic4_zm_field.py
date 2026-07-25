#!/usr/bin/env python
"""Closure computation for the reduced-field screen (review P0): the shipped run evaluated the transverse
Floquet exponent on only |mx|,|my| <= 4 (80 modes), but the n=32 torus carries ~511 unique non-DC modes.
This recomputes the FULL discrete spectrum so the verdict can be stated over every representable spatial
wavelength rather than a low/mid-wavenumber window.

Vectorised over modes: at each orbit step every mode shares the same base state (r0,mu0,S0) and therefore
the same F'(u0); modes differ ONLY through Wk (in the r-r entry) and Khat_sigmaS(k) (in the mu-r entry).
So one batched (nmodes,3,3) matrix product per step replaces ~511 separate integrations.

Also does, at every level (the shipped run did neither):
  * the pre-registered dt vs dt/2 sign check (spec §6.2),
  * persistence of the full lambda(kx,ky), k*, and k*'s angle vs the E->E long axis (spec §6.2).

Reads the locked operating point; writes floquet_full_spectrum.json. Does NOT touch field_screen_summary.json.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.topic4_zm_field_screen import (FieldParams, elliptical_exp_kernel, gaussian_kernel,  # noqa: E402
                                        resolve_w_frac, arm_beta, uniform_orbit, ARMS)
from src.topic4_zm_field_meanfield import psi_prime  # noqa: E402
from src.topic4_zm_field_verdict import TH  # noqa: E402

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_field_screen")


def unique_nondc_modes(n):
    """Every distinct non-DC mode of the n x n torus, deduped by the lambda(k)=lambda(-k) symmetry that a
    real, centrosymmetric kernel guarantees. Indices are signed (-n/2, n/2]."""
    idx = [(m + n // 2) % n - n // 2 for m in range(n)]
    seen, out = set(), []
    for mx in idx:
        for my in idx:
            if (mx % n, my % n) == (0, 0):
                continue
            key = tuple(sorted([(mx % n, my % n), ((-mx) % n, (-my) % n)]))
            if key in seen:
                continue
            seen.add(key)
            out.append((mx, my))
    return out


def spectrum(p: FieldParams, arm, orbit, dt, modes):
    """lambda_perp for EVERY mode in `modes`, via one batched monodromy integration."""
    n = p.n
    KE = np.fft.fft2(elliptical_exp_kernel(n, p.L, p.l_par, p.l_perp, p.theta_EE))
    KS = np.fft.fft2(gaussian_kernel(n, p.L, p.sigma_S))
    q = resolve_w_frac(p)
    w_rec, w_c = p.W0 * q, p.W0 * (1.0 - q)
    mx = np.array([m[0] % n for m in modes]); my = np.array([m[1] % n for m in modes])
    Wk = w_rec + w_c * KE[mx, my].real                      # (nmodes,)
    Kk = KS[mx, my].real
    beta = arm_beta(p, arm)
    one_d = arm in ("div_global", "dual_global")            # global pool has no spatial d.o.f. off DC
    c_S = 1.0 if arm == "dual_local" else (1.0 - p.eps_G) if arm == "dual_mixed" else 1.0
    nm = len(modes)
    M = np.broadcast_to(np.eye(1 if one_d else 3), (nm, 1 if one_d else 3, 1 if one_d else 3)).copy()
    for r0, mu0, S0 in orbit:
        D = 1.0 + p.alpha * S0
        u0 = p.I0 + p.W0 * r0 / D - beta * S0 - p.theta     # BASE state uses W0, never Wk
        Fp = 0.0 if u0 <= 0 else 0.5 / (0.5 + u0) ** 2
        a_rr = (-1.0 + Fp * Wk / D) / p.tau_a               # (nmodes,)
        if one_d:
            J = a_rr.reshape(nm, 1, 1)
        else:
            a_rS = Fp * (-p.alpha * p.W0 * r0 / D ** 2 - beta) * c_S / p.tau_a
            a_mr = Kk * psi_prime(r0, p.r50, p.n_psi) / p.tau_mu
            J = np.zeros((nm, 3, 3))
            J[:, 0, 0] = a_rr; J[:, 0, 2] = a_rS
            J[:, 1, 0] = a_mr; J[:, 1, 1] = -1.0 / p.tau_mu
            J[:, 2, 1] = p.S_max / p.tau_S; J[:, 2, 2] = -1.0 / p.tau_S
        step = np.eye(J.shape[1])[None, :, :] + dt * J
        M = np.einsum("kij,kjl->kil", step, M)
    rho = np.max(np.abs(np.linalg.eigvals(M)), axis=1)
    return np.log(np.maximum(rho, 1e-300)) / (len(orbit) * dt)


def main():
    lock = json.load(open(os.path.join(OUT, "phaseA_lock.json")))
    op, dt, n = lock["operating_point"], lock["dt"], lock["grid_n"]
    modes = unique_nondc_modes(n)
    print(f"[closure] n={n}: {len(modes)} unique non-DC modes (shipped run checked 80)")
    res = {}
    any_positive = False
    for I0 in lock["I0_levels"]:
        key = f"{I0:.4f}"
        p = FieldParams(W0=op["W0"], alpha=op["alpha"], beta=op["beta"], theta=op["theta"], I0=I0, n=n)
        orbit, per = uniform_orbit(p, dt)
        orbit2, _ = uniform_orbit(p, dt / 2)
        lvl = {}
        for arm in ARMS:
            lam = spectrum(p, arm, orbit, dt, modes)
            lam2 = spectrum(p, arm, orbit2, dt / 2, modes)
            i = int(np.argmax(lam))
            kx, ky = modes[i]
            ang = float(np.degrees(np.arctan2(ky, kx))) if (kx or ky) else float("nan")
            pos = int((lam > TH["lam_floor"]).sum())
            any_positive |= pos > 0
            lvl[arm] = dict(
                lam_max=float(lam.max()), lam_min=float(lam.min()),
                k_star=[int(kx), int(ky)], k_star_angle_deg_vs_EE_axis=ang,
                n_modes=len(modes),
                n_modes_above_floor=pos,
                n_modes_positive=int((lam > 0).sum()),
                lam_max_dt_half=float(lam2.max()),
                sign_stable_dt_half=bool(np.sign(lam.max()) == np.sign(lam2.max())),
                max_abs_drift_dt_half=float(np.max(np.abs(lam - lam2))))
        res[key] = lvl
        m = lvl["dual_local"]
        print(f"[closure I0={I0:.3f}] dual_local lam_max={m['lam_max']:+.5f} (k*={m['k_star']}, "
              f"{m['k_star_angle_deg_vs_EE_axis']:.0f}deg) modes>floor={m['n_modes_above_floor']}/{m['n_modes']} "
              f"dt/2 drift={m['max_abs_drift_dt_half']:.2e}")
    out = dict(
        note=("FULL discrete spectrum over every unique non-DC mode of the n=32 torus (the shipped run used "
              "m_max=4 = 80 modes). Includes the pre-registered dt vs dt/2 sign check and k* with its angle "
              "to the E->E long axis."),
        provenance=dict(
            git_sha=subprocess.check_output(["git", "-C", _ROOT, "rev-parse", "HEAD"], text=True).strip(),
            git_dirty=bool(subprocess.check_output(
                ["git", "-C", _ROOT, "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
            lam_floor=TH["lam_floor"], n_modes=len(modes), dt=dt, grid_n=n,
            lock_sha256=hashlib.sha256(open(os.path.join(OUT, "phaseA_lock.json"), "rb").read()).hexdigest(),
            generated_at=datetime.datetime.now().isoformat(timespec="seconds")),
        any_mode_above_floor=bool(any_positive), levels=res)
    p_out = os.path.join(OUT, "floquet_full_spectrum.json")
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"[closure] any mode above +lam_floor anywhere: {any_positive}")
    print(f"[closure] wrote {p_out}")


if __name__ == "__main__":
    main()
