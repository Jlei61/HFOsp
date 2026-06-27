"""Task 4.5 mechanism diagnostics for the M2 ahead-of-front shunting gate.

PURE numpy functions — no SNN engine import, no runner. These probe whether a
feed-forward inhibitory gate fires AHEAD of the excitatory propagation front
(`front_lead_by_axis`) and whether shunting (divisive) inhibition clamps the
high-drive axial-front E cells below threshold where plain current-subtraction
inhibition would not (`clamp_check`).
"""

import numpy as np


def front_lead_by_axis(E_spk_bool, I_spk_bool, along_E, along_I, n_bins, dt):
    """Per-axial-bin onset lead of inhibition over the excitatory front.

    Bins cells into ``n_bins`` equal-width bins spanning the COMBINED along-axis
    range (min..max over both ``along_E`` and ``along_I``). For each bin the E
    (resp I) onset is the first time-index at which ANY E (resp I) cell whose
    along-coord lands in that bin spikes, converted to ms via ``* dt``.
    ``I_lead_ms = t_E_onset_ms - t_I_onset_ms`` (>0 means I fired ahead of the
    E front). Bins with no firing E (resp I) cell get ``np.nan`` onset.

    Parameters
    ----------
    E_spk_bool : (T, NE) bool array
    I_spk_bool : (T, NI) bool array
    along_E, along_I : per-cell along-axis coordinates (length NE, NI)
    n_bins : int
    dt : float (ms per time-index)

    Returns
    -------
    dict with lists of length n_bins:
        bin_along    : bin CENTERS
        t_E_onset_ms : E front onset per bin (ms, nan if no E firing)
        t_I_onset_ms : I onset per bin (ms, nan if no I firing)
        I_lead_ms    : t_E_onset_ms - t_I_onset_ms (nan if either nan)
    """
    E_spk_bool = np.asarray(E_spk_bool, dtype=bool)
    I_spk_bool = np.asarray(I_spk_bool, dtype=bool)
    along_E = np.asarray(along_E, dtype=float)
    along_I = np.asarray(along_I, dtype=float)

    combined = np.concatenate([along_E, along_I])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    # Equal-width bin edges over the combined along range.
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _bin_index(along):
        # np.digitize gives 1..n_bins for values within (edges[0], edges[-1]];
        # clamp so the right edge and any equal-lo values land in valid bins.
        idx = np.digitize(along, edges[1:-1], right=False)
        return np.clip(idx, 0, n_bins - 1)

    bin_E = _bin_index(along_E)
    bin_I = _bin_index(along_I)

    def _onset_ms(spk_bool, cell_bins, b):
        cells = np.where(cell_bins == b)[0]
        if cells.size == 0:
            return np.nan
        sub = spk_bool[:, cells]
        fired_t = np.where(np.any(sub, axis=1))[0]
        if fired_t.size == 0:
            return np.nan
        return float(fired_t[0]) * dt

    t_E_onset_ms = []
    t_I_onset_ms = []
    I_lead_ms = []
    for b in range(n_bins):
        tE = _onset_ms(E_spk_bool, bin_E, b)
        tI = _onset_ms(I_spk_bool, bin_I, b)
        t_E_onset_ms.append(tE)
        t_I_onset_ms.append(tI)
        if np.isnan(tE) or np.isnan(tI):
            I_lead_ms.append(np.nan)
        else:
            I_lead_ms.append(tE - tI)

    return {
        "bin_along": list(centers),
        "t_E_onset_ms": t_E_onset_ms,
        "t_I_onset_ms": t_I_onset_ms,
        "I_lead_ms": I_lead_ms,
    }


def clamp_check(I_E, I_I, along_E, axis_unit, e_gaba, g_gaba_scale, v_th):
    """Fraction of axial-front E cells that shunting inhibition gates but
    current-subtraction would not.

    Selects axial-FRONT E cells = those whose ``along_E`` is in the TOP QUARTILE
    (>= 75th percentile of ``along_E``; ahead of the centroid). For those cells:
        g_I = g_gaba_scale * max(I_I, 0)
        cur = I_E - I_I                      (current-subtraction target)
        sh  = (I_E + g_I * e_gaba) / (1 + g_I)   (shunting/divisive target)
    ``frac_axial_gated_by_shunt`` = mean over the front cells of
    ``(sh < v_th) & (cur >= v_th)`` — high-drive axial-front cells the shunting
    clamps below threshold while plain subtraction would still fire.

    ``axis_unit`` is accepted for signature completeness; the along projection is
    already supplied via ``along_E`` so no re-projection is performed.

    Returns
    -------
    dict:
        frac_axial_gated_by_shunt : float
        front_idx : indices of the selected axial-front cells
        cur, sh   : current-path and shunting targets for the selected cells
    """
    I_E = np.asarray(I_E, dtype=float)
    I_I = np.asarray(I_I, dtype=float)
    along_E = np.asarray(along_E, dtype=float)

    thresh = np.percentile(along_E, 75.0)
    front_idx = np.where(along_E >= thresh)[0]

    I_E_f = I_E[front_idx]
    I_I_f = I_I[front_idx]
    g_I = g_gaba_scale * np.maximum(I_I_f, 0.0)
    cur = I_E_f - I_I_f
    sh = (I_E_f + g_I * e_gaba) / (1.0 + g_I)

    if front_idx.size == 0:
        frac = float("nan")
    else:
        gated = (sh < v_th) & (cur >= v_th)
        frac = float(np.mean(gated))

    return {
        "frac_axial_gated_by_shunt": frac,
        "front_idx": front_idx,
        "cur": cur,
        "sh": sh,
    }
