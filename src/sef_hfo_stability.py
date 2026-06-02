# src/sef_hfo_stability.py
"""SEF-HFO Step-0a delayed linear stability. See 2026-06-02 amendment."""
import numpy as np
from dataclasses import replace, fields

def _f(x, beta): return 1.0 / (1.0 + np.exp(-beta * x))
def _fprime(x, beta):
    s = _f(x, beta); return beta * s * (1.0 - s)

_Z = np.linspace(-8.0, 8.0, 2001); _PZ = np.exp(-0.5 * _Z**2) / np.sqrt(2 * np.pi)

def F_eff(h, phi_bar, sigma_phi, beta):
    return np.trapz(_f(h - (phi_bar + sigma_phi * _Z), beta) * _PZ, _Z)

def gain(h, phi_bar, sigma_phi, beta):
    return np.trapz(_fprime(h - (phi_bar + sigma_phi * _Z), beta) * _PZ, _Z)
