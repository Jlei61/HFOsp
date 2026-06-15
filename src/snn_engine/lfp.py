"""
LFP proxy + virtual recording grid (Methods 4.2, Eq 9-11).

LFP at electrode i = spatially-weighted sum of |I_AMPA| + |I_GABA| over nearby
EXCITATORY neurons:

    g(j,t) = |I_E(j,t)| + |I_I(j,t)|          (j in E)            (Eq, Methods)
    f(r)   = r^{-1/2}                 for 0 < r < rx               (Eq 9)
             (1/sqrt(rx)) (rx/r)^2    for r >= rx
    LFP(i) = sum_j 1[d_ij<=Rr] f(d_ij) g(j) / sum_j 1[d_ij<=Rr] f(d_ij)   (Eq 11)

Sites sit on a regular grid (pitch `grid_spacing`, `grid_margin` from edges).
"""

from __future__ import annotations
import numpy as np


def make_grid(p):
    """Recording-site coordinates on an L x L sheet (Methods 4.2)."""
    lo = p.grid_margin
    hi = p.L - p.grid_margin
    if hi <= lo:                       # tiny sheet: single centered site
        c = np.array([[p.L / 2, p.L / 2]])
        return c
    n = int(np.floor((hi - lo) / p.grid_spacing)) + 1
    xs = lo + np.arange(n) * p.grid_spacing
    X, Y = np.meshgrid(xs, xs)
    return np.column_stack([X.ravel(), Y.ravel()])


def _shape_f(r, rx):
    f = np.empty_like(r)
    near = r < rx
    f[near] = r[near] ** (-0.5)
    far = ~near
    f[far] = (1.0 / np.sqrt(rx)) * (rx / r[far]) ** 2
    return f


class LFPRecorder:
    def __init__(self, p, pos, labels, sites=None):
        # sites=None -> default regular grid (back-compat); else custom contact coords
        # (e.g. a virtual-SEEG montage at real electrode spacing). (Increment-2/3 patch)
        self.sites = make_grid(p) if sites is None else np.asarray(sites, float)
        NE = int(np.sum(labels == 0))
        posE = pos[:NE]
        self.NE = NE
        # precompute, per site, the contributing E neurons and normalized weights
        self._idx = []        # indices into E
        self._w = []          # normalized shape weights
        for s in self.sites:
            d = np.linalg.norm(posE - s, axis=1)
            m = d <= p.Rr
            if not np.any(m):
                # fall back to nearest single neuron to avoid empty site
                j = np.argmin(d)
                self._idx.append(np.array([j]))
                self._w.append(np.array([1.0]))
                continue
            dd = np.maximum(d[m], 1e-4)            # avoid r=0 singularity
            w = _shape_f(dd, p.rx)
            w = w / w.sum()
            self._idx.append(np.where(m)[0])
            self._w.append(w)

    def sample(self, I_E, I_I):
        """Return one LFP value per site (uses E-neuron currents only)."""
        g = np.abs(I_E[:self.NE]) + np.abs(I_I[:self.NE])     # g(j,t), j in E
        out = np.empty(len(self.sites))
        for k in range(len(self.sites)):
            out[k] = np.dot(self._w[k], g[self._idx[k]])
        return out
