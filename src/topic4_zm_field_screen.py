# src/topic4_zm_field_screen.py
"""Reduced 2-D S_L(x)+S_G rate field (Fix A dual pool) + anisotropic K_E + per-mode Floquet + streaming
metrics + 4 arms. Spec 2026-07-24 rev3. Reduced rate field only -- no SNN, no H, no E->E."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _offset_grid(n, L):
    """Periodic signed offsets (mm); offset (0,0) sits at index (0,0). DX varies along axis 0."""
    idx = (np.arange(n) + n // 2) % n - n // 2
    d = idx * (L / n)
    return np.meshgrid(d, d, indexing="ij")

def elliptical_exp_kernel(n, L, l_par, l_perp, theta):
    DX, DY = _offset_grid(n, L)
    u = DX * np.cos(theta) + DY * np.sin(theta)
    v = -DX * np.sin(theta) + DY * np.cos(theta)
    K = np.exp(-np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2))
    K[0, 0] = 0.0
    return K / K.sum()

def gaussian_kernel(n, L, sigma):
    DX, DY = _offset_grid(n, L)
    K = np.exp(-0.5 * (DX ** 2 + DY ** 2) / sigma ** 2)
    return K / K.sum()

def cell_mass_fraction(L, n, l_par=0.537, l_perp=0.269, theta=0.0, sub=64, span_cells=12):
    """q_cell = fraction of the CONTINUOUS K_E mass inside one lattice cell (fine quadrature). Sets the
    local/non-local recurrent split w_rec=W0*q_cell, w_c=W0*(1-q_cell) -- derived, never hand-set."""
    h = L / n
    xs = (np.arange(-span_cells * sub, span_cells * sub) + 0.5) * (h / sub)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u = X * np.cos(theta) + Y * np.sin(theta); v = -X * np.sin(theta) + Y * np.cos(theta)
    K = np.exp(-np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2))
    inside = (np.abs(X) <= h / 2) & (np.abs(Y) <= h / 2)
    return float(K[inside].sum() / K.sum())

def kernel_axis_and_ar(K, L):
    """Principal axis (rad) + aspect ratio of a kernel, via its mass-weighted spatial covariance."""
    n = K.shape[0]
    DX, DY = _offset_grid(n, L)
    w = K / K.sum()
    cxx = float((w * DX * DX).sum()); cyy = float((w * DY * DY).sum()); cxy = float((w * DX * DY).sum())
    C = np.array([[cxx, cxy], [cxy, cyy]])
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]
    return float(np.arctan2(v[1], v[0])), float(np.sqrt(max(vals) / max(min(vals), 1e-30)))
