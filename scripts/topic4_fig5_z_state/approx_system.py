#!/usr/bin/env python3
"""Reduced (coarse-cell rate) system of the current Fig.5 network with target-side frozen Z and dynamic M.

State per coarse cell c (n = grid^2):
  synaptic filters (means over target neurons; exact for the mean):  g^A_E, c^A_E, g^G_E, c^G_E  (E targets)
                                                                      g^A_I, c^A_I, g^G_I, c^G_I  (I targets)
  rate histories h_E, h_I over the exact delay bins 1..D (0.1 ms) and the native pending-arrival schedule at start
  E units (cell, threshold chunk k=1..8): rate r_u, adaptation m_u, frozen z_u, z2_u, threshold theta_u
  I units (cell): rate r_c
Closure:
  r <- r + dt/tau_rate (Phi(mu, sigma_A^2, sigma_G^2) - r)   (first-order rate relaxation, tau_rate_E/I)
  mu_u = c^A_E[c] - z_u c^G_E[c] - eta_M m_u ;  mu_I = c^A_I - c^G_I
  variance: stationary (current rates) or kernel-filtered (exact double-exponential impulse kernel, delayed rates)
  Phi: 'mixed' = AMPA colored-noise shifted Siegert averaged over a Gaussian slow GABA current (variance from the
       exact discrete GABA kernel, z2-scaled on E targets); 'colored' = single colored-noise shift (older closure)
  M: dm_u/dt = -m_u/tau_M + r_u  (r in spikes/ms) ; frozen-M runs skip the update.
Units: rates in spikes/ms per neuron; currents in mV (asymptotic voltage), time in ms.
"""
import numpy as np
from scipy import sparse
from scipy.special import erfcx
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from common import *  # noqa: F401,F403
import siegert_table

APPROX = OUT / 'approx'
DEFAULT_VARIANT = dict(grid=20, phi='mixed', gh_nodes=15, variance='stationary', tau_rate_e_ms=5., tau_rate_i_ms=2.5,
                       joint_z=True, chunks=8, lif='table', version='v1')


class ReducedModel:
    def __init__(self, variant=None):
        v = dict(DEFAULT_VARIANT); v.update(variant or {}); self.v = v
        g = int(v['grid']); self.grid = g; folder = APPROX / f'coarse_{g}'; self.folder = folder
        m = np.load(folder / 'model.npz'); p = read(folder / 'prepared.json'); geo = np.load(folder / 'geometry.npz')
        self.n = n = g * g; self.K = K = int(m['threshold_nodes_e'].shape[1]); self.D = D = int(p['max_delay_steps'])
        self.dt = float(p['dt_ms']); self.count_e = m['count_e'].astype(float); self.count_i = m['count_i'].astype(float)
        self.theta_u = m['threshold_nodes_e'].reshape(-1); self.w_u = m['threshold_weights_e'].reshape(-1)
        self.theta_i = float(m['v_threshold_i_mv']); self.v_reset = float(m['v_reset_mv'])
        self.te, self.ti = float(m['tau_mem_e_ms']), float(m['tau_mem_i_ms'])
        self.tref_e, self.tref_i = float(m['tau_ref_e_ms']), float(m['tau_ref_i_ms'])
        self.ra, self.ta = float(p['tau_r_ampa_ms']), float(p['tau_d_ampa_ms']); self.rg, self.tg = float(p['tau_r_gaba_ms']), float(p['tau_d_gaba_ms'])
        self.je, self.ji = float(m['j_ext_e_mv']), float(m['j_ext_i_mv']); self.nu_sig = float(p['nu_ext_per_ms'])
        self.tau_M = 1000.; self.eta_M = .0005
        for k in ('v_ee', 'v_ei', 'v_ie', 'v_ii', 'w_ee', 'w_ei', 'w_ie', 'w_ii'):
            setattr(self, k, np.asarray(m[k], float))
        self.ops = {k: sparse.load_npz(folder / f'delay_{k}.npz').tocsr() for k in ('ee', 'ei', 'ie', 'ii')}
        self.vops = None
        if v['variance'] == 'filtered':
            self.vops = {k: sparse.load_npz(folder / f'vdelay_{k}.npz').tocsr() for k in ('ee', 'ei', 'ie', 'ii')}
        dt = self.dt
        self.arA, self.adA = np.exp(-dt / self.ra), np.exp(-dt / self.ta); self.arG, self.adG = np.exp(-dt / self.rg), np.exp(-dt / self.tg)
        self.BAE, self.BGE = dt * self.te / self.ra, dt * self.te / self.rg; self.BAI, self.BGI = dt * self.ti / self.ra, dt * self.ti / self.rg
        self.gaA = dt / (self.ra * (1 - np.exp(-dt / self.ra))); self.gaG = dt / (self.rg * (1 - np.exp(-dt / self.rg)))
        # exact discrete GABA kernel variance factor (per unit weight^2 per unit rate), as in the 2026-09-09 closure
        a, b = self.arG, self.adG
        self.w2cv_e = dt * (self.te / self.rg * (1 - b) / (a - b)) ** 2 * (a * a / (1 - a * a) + b * b / (1 - b * b) - 2 * a * b / (1 - a * b)) / self.te
        self.w2cv_i = dt * (self.ti / self.rg * (1 - b) / (a - b)) ** 2 * (a * a / (1 - a * a) + b * b / (1 - b * b) - 2 * a * b / (1 - a * b)) / self.ti
        self.gh_x, self.gh_w = hermgauss(int(v['gh_nodes'])); self.gh_w = self.gh_w / np.sqrt(np.pi)
        self.lg_x, self.lg_w = leggauss(16)
        self.cell_e = geo['cell_e']; self.cell_i = geo['cell_i']; self.vtheta_e = geo['vtheta_e']
        self.g15 = geo['g15']; self.g175 = geo['g175']
        # neuron -> (cell, chunk) membership consistent with grouped_threshold_support (sorted chunks)
        self.chunk_of = np.zeros(NE, int)
        for c in range(n):
            idx = np.flatnonzero(self.cell_e == c); order = idx[np.argsort(self.vtheta_e[idx], kind='stable')]
            for k, part in enumerate(np.array_split(order, min(K, len(order)))):
                self.chunk_of[part] = k
        self.unit_of = self.cell_e * K + self.chunk_of
        self.unit_count = np.bincount(self.unit_of, minlength=n * K).astype(float); assert np.all(self.unit_count > 0)
        # region weights (E neurons per cell in each region) for readouts
        self.region_w = {}
        for name, grp in (('175', self.g175), ('15', self.g15)):
            for r in range(3):
                self.region_w[f'{name}_{r}'] = np.bincount(self.cell_e[grp == r], minlength=n).astype(float)
        self.tau_rate = np.r_[np.full(n * K, float(v['tau_rate_e_ms'])), np.full(n, float(v['tau_rate_i_ms']))]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        n, K, D = self.n, self.K, self.D
        self.r_u = np.zeros(n * K); self.r_i = np.zeros(n); self.m_u = np.zeros(n * K)
        self.z_u = np.ones(n * K); self.z2_u = np.ones(n * K)
        self.gAE = np.zeros(n); self.cAE = np.zeros(n); self.gGE = np.zeros(n); self.cGE = np.zeros(n)
        self.gAI = np.zeros(n); self.cAI = np.zeros(n); self.gGI = np.zeros(n); self.cGI = np.zeros(n)
        self.hE = np.zeros((D, n)); self.hI = np.zeros((D, n))
        self.pending = None; self.pending_index = 0
        self.y = {k: np.zeros((3, n)) for k in ('ee', 'ei', 'ie', 'ii', 'exte', 'exti')}
        self.step_count = 0; self.freeze_m = False

    def cell_rate_e(self):
        return (self.r_u * self.w_u).reshape(self.n, self.K).sum(1)

    def unit_mean(self, values_e):
        return np.bincount(self.unit_of, weights=values_e, minlength=self.n * self.K) / self.unit_count

    def cell_mean(self, values, cells, n=None):
        n = self.n if n is None else n
        return np.bincount(cells, weights=values, minlength=n) / np.bincount(cells, minlength=n)

    def set_slow_fields(self, z_e, m_e):
        if self.v['joint_z']:
            self.z_u = self.unit_mean(z_e); self.z2_u = self.unit_mean(z_e ** 2); self.m_u = self.unit_mean(m_e)
        else:
            zc = self.cell_mean(z_e, self.cell_e); self.z_u = np.repeat(zc, self.K); self.z2_u = self.z_u ** 2
            self.m_u = np.repeat(self.cell_mean(m_e, self.cell_e), self.K)

    def set_z_field(self, z_e):
        if self.v['joint_z']:
            self.z_u = self.unit_mean(z_e); self.z2_u = self.unit_mean(z_e ** 2)
        else:
            self.z_u = np.repeat(self.cell_mean(z_e, self.cell_e), self.K); self.z2_u = self.z_u ** 2

    def project_native_state(self, state, e_counts_recent_0p1ms, i_rate_recent_cell):
        """Exact projection of the fast native state: filter means, pending delayed arrivals, slow fields, rates."""
        ne = NE; ce, ci = self.cell_e, self.cell_i
        self.gAE = self.cell_mean(state['s_E'][:ne], ce); self.cAE = self.cell_mean(state['I_E'][:ne], ce)
        self.gGE = self.cell_mean(state['s_I'][:ne], ce); self.cGE = self.cell_mean(state['I_I'][:ne], ce)
        self.gAI = self.cell_mean(state['s_E'][ne:], ci); self.cAI = self.cell_mean(state['I_E'][ne:], ci)
        self.gGI = self.cell_mean(state['s_I'][ne:], ci); self.cGI = self.cell_mean(state['I_I'][ne:], ci)
        M = state['ring_sE'].shape[0]; step = int(state['step'])
        pend = np.zeros((M, 4, self.n))
        for tau in range(M):
            slot = (step + tau) % M
            pend[tau, 0] = self.cell_mean(state['ring_sE'][slot, :ne], ce); pend[tau, 1] = self.cell_mean(state['ring_sE'][slot, ne:], ci)
            pend[tau, 2] = self.cell_mean(state['ring_sI'][slot, :ne], ce); pend[tau, 3] = self.cell_mean(state['ring_sI'][slot, ne:], ci)
        self.pending = pend; self.pending_index = 0
        self.set_slow_fields(state['slow']['z'][:ne], state['slow']['m'][:ne])
        re = np.asarray(e_counts_recent_0p1ms, float).sum(0) / self.count_e / (len(e_counts_recent_0p1ms) * self.dt)
        self.r_u = np.repeat(re, self.K); self.r_i = np.asarray(i_rate_recent_cell, float)
        self.hE[:] = 0.; self.hI[:] = 0.
        if self.vops is not None:
            # No native record of squared-weight arrivals: start the variance filters at their stationary values.
            xe = self.v_ee @ re + self.je ** 2 * self.nu_sig; xei = self.v_ei @ self.r_i
            xi_ = self.v_ie @ re + self.ji ** 2 * self.nu_sig; xii = self.v_ii @ self.r_i
            for k, x, (a, b) in (('ee', xe, (self.arA, self.adA)), ('ei', xei, (self.arG, self.adG)), ('ie', xi_, (self.arA, self.adA)), ('ii', xii, (self.arG, self.adG))):
                for j, c in enumerate((a * a, b * b, a * b)):
                    self.y[k][j] = c / (1 - c) * x
        self.step_count = 0

    # ------------------------------------------------------------------ transfer
    def _lif(self, mu, sigma, threshold, tm, tref):
        if self.v.get('lif', 'table') == 'table':
            return siegert_table.lif_rate(mu, sigma, threshold, tm, tref, self.v_reset)
        return siegert_table.lif_rate_legendre(mu, sigma, threshold, tm, tref, self.v_reset, order=16)

    def phi_e(self, mu_u, ex_c, inh_u):
        """E units: mu per unit, AMPA variance per cell (repeated), GABA variance per unit."""
        ex = np.repeat(ex_c, self.K)
        if self.v['phi'] == 'mixed':
            sigG = np.sqrt(np.maximum(inh_u * self.w2cv_e, 0.))
            shifted = mu_u - 2.065 / 2 * np.sqrt(np.maximum(ex * (self.ra + self.ta) / self.te, 1e-16))
            means = shifted[None, :] + np.sqrt(2) * sigG[None, :] * self.gh_x[:, None]
            p = self._lif(means, np.sqrt(np.maximum(ex, 1e-12))[None, :], self.theta_u[None, :], self.te, self.tref_e)
            return np.sum(self.gh_w[:, None] * p, axis=0)
        sig = np.sqrt(np.maximum(ex + inh_u, 1e-12))
        shift = 2.065 / 2 * np.sqrt(np.maximum((ex * (self.ra + self.ta) + inh_u * (self.rg + self.tg)) / self.te, 1e-16))
        return self._lif(mu_u - shift, sig, self.theta_u, self.te, self.tref_e)

    def phi_i(self, mu, ex, inh):
        if self.v['phi'] == 'mixed':
            sigG = np.sqrt(np.maximum(inh * self.w2cv_i, 0.))
            shifted = mu - 2.065 / 2 * np.sqrt(np.maximum(ex * (self.ra + self.ta) / self.ti, 1e-16))
            means = shifted[None, :] + np.sqrt(2) * sigG[None, :] * self.gh_x[:, None]
            p = self._lif(means, np.sqrt(np.maximum(ex, 1e-12))[None, :], self.theta_i, self.ti, self.tref_i)
            return np.sum(self.gh_w[:, None] * p, axis=0)
        sig = np.sqrt(np.maximum(ex + inh, 1e-12))
        shift = 2.065 / 2 * np.sqrt(np.maximum((ex * (self.ra + self.ta) + inh * (self.rg + self.tg)) / self.ti, 1e-16))
        return self._lif(mu - shift, sig, self.theta_i, self.ti, self.tref_i)

    # ------------------------------------------------------------------ one step
    def _filtered_second_moment(self, key, x, a, b):
        y = self.y[key]
        for j, c in enumerate((a * a, b * b, a * b)):
            y[j] = c * (x + y[j])
        norm = a * a / (1 - a * a) + b * b / (1 - b * b) - 2 * a * b / (1 - a * b)
        return (y[0] + y[1] - 2 * y[2]) / norm

    def step(self, nu_e_cell, nu_i):
        """nu_e_cell: expected external rate per E cell (per ms); nu_i: external rate for I cells (scalar, per ms)."""
        n, K = self.n, self.K
        re = self.cell_rate_e(); ri = self.r_i
        hE = self.hE.ravel(); hI = self.hI.ravel()
        dEE = self.ops['ee'] @ hE; dEI = self.ops['ei'] @ hI; dIE = self.ops['ie'] @ hE; dII = self.ops['ii'] @ hI
        pAE = pAI = pGE = pGI = 0.
        if self.pending is not None and self.pending_index < len(self.pending):
            p = self.pending[self.pending_index]; pAE, pAI, pGE, pGI = p[0], p[1], p[2], p[3]; self.pending_index += 1
        self.gAE = self.arA * self.gAE + self.BAE * (dEE + self.je * nu_e_cell) + pAE; self.cAE = self.gAE + (self.cAE - self.gAE) * self.adA
        self.gGE = self.arG * self.gGE + self.BGE * dEI + pGE; self.cGE = self.gGE + (self.cGE - self.gGE) * self.adG
        self.gAI = self.arA * self.gAI + self.BAI * (dIE + self.ji * nu_i) + pAI; self.cAI = self.gAI + (self.cAI - self.gAI) * self.adA
        self.gGI = self.arG * self.gGI + self.BGI * dII + pGI; self.cGI = self.gGI + (self.cGI - self.gGI) * self.adG
        if self.vops is None:
            xe = self.v_ee @ re + self.je ** 2 * nu_e_cell; xei = self.v_ei @ ri
            xi_ = self.v_ie @ re + self.ji ** 2 * nu_i; xii = self.v_ii @ ri
        else:
            xe = self._filtered_second_moment('ee', self.vops['ee'] @ hE + self.je ** 2 * nu_e_cell, self.arA, self.adA)
            xei = self._filtered_second_moment('ei', self.vops['ei'] @ hI, self.arG, self.adG)
            xi_ = self._filtered_second_moment('ie', self.vops['ie'] @ hE + self.ji ** 2 * nu_i, self.arA, self.adA)
            xii = self._filtered_second_moment('ii', self.vops['ii'] @ hI, self.arG, self.adG)
        ex_e = self.te * xe; inh_u = self.te * self.z2_u * np.repeat(xei, K)
        ex_i = self.ti * xi_; inh_i = self.ti * xii
        mu_u = np.repeat(self.cAE, K) - self.z_u * np.repeat(self.cGE, K) - self.eta_M * self.m_u
        mu_i = self.cAI - self.cGI
        phi_u = self.phi_e(mu_u, ex_e, inh_u); phi_i = self.phi_i(mu_i, ex_i, inh_i)
        self.hE[1:] = self.hE[:-1]; self.hE[0] = re; self.hI[1:] = self.hI[:-1]; self.hI[0] = ri
        self.r_u = self.r_u + self.dt / self.tau_rate[:n * K] * (phi_u - self.r_u)
        self.r_i = self.r_i + self.dt / self.tau_rate[n * K:] * (phi_i - self.r_i)
        if not self.freeze_m:
            self.m_u = self.m_u - self.dt / self.tau_M * self.m_u + self.dt * self.r_u
        self.step_count += 1
        return re, ri, mu_u, ex_e, inh_u

    # ------------------------------------------------------------------ persistence
    def state_dict(self):
        keys = ('r_u', 'r_i', 'm_u', 'z_u', 'z2_u', 'gAE', 'cAE', 'gGE', 'cGE', 'gAI', 'cAI', 'gGI', 'cGI', 'hE', 'hI')
        d = {k: np.array(getattr(self, k), copy=True) for k in keys}
        d['pending'] = None if self.pending is None else self.pending[self.pending_index:].copy()
        d['y'] = {k: v.copy() for k, v in self.y.items()}; d['step_count'] = self.step_count; d['freeze_m'] = self.freeze_m
        return d

    def load_state_dict(self, d):
        for k, v in d.items():
            if k == 'pending':
                self.pending = None if v is None else np.array(v, copy=True); self.pending_index = 0
            elif k == 'y':
                self.y = {kk: np.array(vv, copy=True) for kk, vv in v.items()}
            elif k in ('step_count', 'freeze_m'):
                setattr(self, k, v)
            else:
                setattr(self, k, np.array(v, copy=True))

    # ------------------------------------------------------------------ readout helpers
    def region_rate(self, re_cell, key):
        w = self.region_w[key]
        return float(np.average(re_cell, weights=w)) if w.sum() > 0 else float('nan')
