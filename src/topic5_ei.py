"""Topic 5 B-line — EI-like early-participation index (pure, no I/O).

Bartolomei-style: high fast-activity AND early -> high index. Delay is in the DENOMINATOR
(a penalty), never multiplied (that would reward LATE contacts and flip the sign):

    EI_like_i = ER_i / (delay_i + tau)
      ER_i    = max(hfa_auc_i, 0)              # mean baseline-robust-z 60-100 Hz over [0,10]s
      delay_i = onset_i - min_j(onset_j)       # >=0; onset = first frame HFA-z crosses z_onset
      tau     = audit-locked (1 s)
    no onset detected -> EI_like = 0 (channel did not early-participate; never NaN-imputed).

Computed from the T0 cache's hfa_zt ([0,10]s HFA robust-z trace) + hfa_auc, so B-line is just
the A-line alignment with EI_like as the activation. NOTE: this is an HFA-only EI-LIKE proxy,
not the literal 60-100/4-40 energy ratio (the cache does not store the low-frequency band);
flagged as such, and B is a secondary/exploratory track.
"""
from __future__ import annotations

import numpy as np


def onset_delays(hfa_zt: np.ndarray, *, hop_sec: float = 0.1, z_onset: float = 2.0) -> np.ndarray:
    """Per-channel onset delay (seconds) relative to the earliest channel. hfa_zt = [n_ch, n_t].
    A channel whose HFA-z never crosses z_onset gets NaN (did not participate)."""
    z = np.asarray(hfa_zt, float)
    n_ch = z.shape[0]
    onset = np.full(n_ch, np.nan)
    for c in range(n_ch):
        cross = np.where(z[c] > z_onset)[0]
        if cross.size:
            onset[c] = cross[0] * hop_sec
    if np.isfinite(onset).any():
        return onset - np.nanmin(onset)
    return onset


def ei_like(hfa_zt: np.ndarray, hfa_auc: np.ndarray, *, hop_sec: float = 0.1,
            z_onset: float = 2.0, tau: float = 1.0) -> np.ndarray:
    """Per-channel EI-like = max(hfa_auc,0) / (onset_delay + tau); 0 where no onset (§ formula)."""
    er = np.maximum(np.asarray(hfa_auc, float), 0.0)
    delay = onset_delays(hfa_zt, hop_sec=hop_sec, z_onset=z_onset)
    return np.where(np.isfinite(delay), er / (delay + tau), 0.0)
