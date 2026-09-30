#!/usr/bin/env python3
"""Design 6.2: bounded isolated-LIF dynamic-response diagnostics (<=60 runs, 2048 cells, 2 s each).

Six local contexts (A/B/surround x E/I) defined from the replay's 8.8-9.3 s native statistics of the representative
20x20 cell holding the largest share of the region's neurons (fixed before any approximation result is read).
Each run: same membrane, threshold field, Z/M joint samples, AMPA/GABA filters as the engine; compound-Poisson input
matched to the pathway's first/second moments (independent-input assumption); 1 s baseline, 100 ms +5% or +2.5%
on the excitatory or inhibitory component, 900 ms return; sham = no perturbation.  Two input seeds
(9108601 / 9108602) with common future random numbers across the 5 arms of one context/seed.
The closure (Phi + first-order rate relaxation) is driven by the same input moments for comparison.
"""
import argparse
import numpy as np
from common import *  # noqa: F401,F403
from approx_system import ReducedModel, APPROX
import approx_checks as C

SEEDS_SMALL = [9108601, 9108602]
N_CELLS = 2048
BASE_MS, PERT_MS, TAIL_MS = 1000., 100., 900.
ARMS = [('sham', None, 0.), ('E5', 'E', .05), ('E2p5', 'E', .025), ('I5', 'I', .05), ('I2p5', 'I', .025)]
SMALL = APPROX / 'small_set'


def contexts(model):
    """Representative cell per region (largest occupancy) for E and I; fixed rule, not fit-dependent."""
    out = {}
    for pop, cells, grp in (('E', model.cell_e, model.g175), ('I', model.cell_i, None)):
        if grp is None:
            centers = np.asarray(read(SUBSTRATE / 'substrate.json')['centers_mm']); pos = np.load(APPROX / 'coarse_20/geometry.npz')['positions_i']
            d = np.linalg.norm(pos[:, None] - centers[None], axis=2); grp = np.full(len(pos), 2); grp[d[:, 0] < 1.75] = 0; grp[(d[:, 1] < 1.75) & (d[:, 1] < d[:, 0])] = 1
        for r, name in enumerate(('A', 'B', 'S')):
            counts = np.bincount(cells[grp == r], minlength=model.n)
            cell = int(np.argmax(counts)) if name != 'S' else int(np.argmax(counts * (np.bincount(cells[grp != 2], minlength=model.n) == 0)))
            out[f'{name}_{pop}'] = dict(cell=cell, population=pop, region=name, occupancy=int(counts[cell]))
    return out


def context_statistics(model, ctx, lo_ms=8800, hi_ms=9300):
    """Native local input statistics 8.8-9.3 s: per-neuron mean/var of I_E and I_I (5-ms samples), z, m, thresholds."""
    f = C.load_replay_fields(lo_ms=lo_ms, hi_ms=hi_ms)
    e_counts, i_reg = C.native_rates_0p1ms(lo_ms=lo_ms, hi_ms=hi_ms)
    c = ctx['cell']
    if ctx['population'] == 'E':
        idx = np.flatnonzero(model.cell_e == c)
        ie = f['ie'][:, idx].astype(float); ii = f['ii'][:, idx].astype(float); z = f['z'][:, idx]; m = f['m'][:, idx]
        theta = model.vtheta_e[idx]; tm, tref = model.te, model.tref_e
        rate_native = float(e_counts[:, c].sum() / len(idx) / (len(e_counts) * model.dt))
    else:
        idx = np.flatnonzero(model.cell_i == c)
        # I-cell currents are not in the per-cell record (E only); use the pathway moments from the model at native rates
        ie = ii = z = m = None; theta = np.full(len(idx), model.theta_i); tm, tref = model.ti, model.tref_i
        g = np.load(REF / 'geometry.npz'); rc = g['region_counts'].astype(float)
        from approx_run import i_region_of_cells
        reg = i_region_of_cells(model)[c]; rate_native = float(i_reg[:, reg].sum() / rc[3 + reg] / (len(i_reg) * 1.))
    re_cells = e_counts.astype(float).sum(0) / model.count_e / (len(e_counts) * model.dt)
    g = np.load(REF / 'geometry.npz'); rc = g['region_counts'].astype(float)
    from approx_run import i_region_of_cells
    ri_cells = (i_reg.astype(float).sum(0) / rc[3:] / (len(i_reg) * 1.))[i_region_of_cells(model)]
    # pathway input moments to this cell (independent-input, weights in mV): mean rate of weighted arrivals
    if ctx['population'] == 'E':
        wA = model.w_ee[c]; vA = model.v_ee[c]; wG = model.w_ei[c]; vG = model.v_ei[c]
    else:
        wA = model.w_ie[c]; vA = model.v_ie[c]; wG = model.w_ii[c]; vG = model.v_ii[c]
    lamA = float(wA @ re_cells); qA = float(vA @ re_cells); lamG = float(wG @ ri_cells); qG = float(vG @ ri_cells)
    stats = dict(cell=c, n_neurons=int(len(idx)), native_rate_per_ms=rate_native,
                 rate_based_ampa_mean_mv_per_ms=lamA, rate_based_ampa_sq_mv2_per_ms=qA, rate_based_gaba_mean_mv_per_ms=lamG, rate_based_gaba_sq_mv2_per_ms=qG,
                 ext_rate_per_ms=float(model.nu_sig), theta_mean=float(theta.mean()), theta_min=float(theta.min()))
    # kernel variance factors: Var[I] = q * factor for independent arrivals with second moment q (mV^2/ms)
    def kernel_factor(a, b, tr):
        return model.dt * (tm / tr * (1 - b) / (a - b)) ** 2 * (a * a / (1 - a * a) + b * b / (1 - b * b) - 2 * a * b / (1 - a * b))
    fA = kernel_factor(model.arA, model.adA, model.ra); fG = kernel_factor(model.arG, model.adG, model.rg)
    if ie is not None:
        # Real local input statistics (design 6.2): match the recorded current mean and temporal variance per pathway.
        mA = float(ie.mean()); vA = float(ie.var(0).mean()); mG = float(ii.mean()); vG = float(ii.var(0).mean())
        ext_mean = tm * model.gaA * model.je * model.nu_sig; ext_var = model.nu_sig * model.je ** 2 * fA
        lamA_c = max(mA - ext_mean, 1e-9) / (tm * model.gaA); qA_c = max(vA - ext_var, 1e-9) / fA
        lamG_c = mG / (tm * model.gaG); qG_c = vG / fG
        stats.update(ampa_mean_mv_per_ms=lamA_c, ampa_sq_mv2_per_ms=qA_c, gaba_mean_mv_per_ms=lamG_c, gaba_sq_mv2_per_ms=qG_c,
                     input_definition='native current mean and temporal variance (5-ms samples, 8.8-9.3 s)',
                     native_IE_mean=mA, native_IE_sd_time=float(np.sqrt(vA)), native_II_mean=mG, native_II_sd_time=float(np.sqrt(vG)),
                     native_z_mean=float(z.mean()), native_z_sd=float(z.std()), native_m_mean=float(m.mean()),
                     independent_input_equivalent_arrival_rate_per_ms=dict(ampa=lamA_c ** 2 / qA_c, gaba=lamG_c ** 2 / qG_c),
                     rate_based_current_sd=dict(ampa=float(np.sqrt(qA * fA)), gaba=float(np.sqrt(qG * fG))))
    else:
        stats.update(ampa_mean_mv_per_ms=lamA, ampa_sq_mv2_per_ms=qA, gaba_mean_mv_per_ms=lamG, gaba_sq_mv2_per_ms=qG,
                     input_definition='rate-based independent-input pathway moments (no per-neuron I-cell current record)')
    samples = dict(theta=theta, z=None if z is None else z[-1], m=None if m is None else m[-1], idx=idx)
    return stats, samples


class IsolatedLIF:
    """2048 independent LIF cells with the engine's synaptic filters, threshold field and frozen z / dynamic m."""
    def __init__(self, model, pop, theta, z, m, seed):
        self.dt = model.dt; self.N = N_CELLS; rng = np.random.default_rng(seed + 777)
        pick = rng.integers(0, len(theta), self.N)
        self.theta = theta[pick]; self.z = np.ones(self.N) if z is None else z[pick]; self.m = np.zeros(self.N) if m is None else m[pick].copy()
        self.tm = model.te if pop == 'E' else model.ti; self.tref_steps = int(round((model.tref_e if pop == 'E' else model.tref_i) / self.dt))
        self.arA, self.adA, self.arG, self.adG = model.arA, model.adA, model.arG, model.adG
        self.decay_V = np.exp(-self.dt / self.tm)
        self.V = np.full(self.N, model.v_reset); self.ref = np.zeros(self.N, int)
        self.sE = np.zeros(self.N); self.IE = np.zeros(self.N); self.sI = np.zeros(self.N); self.II = np.zeros(self.N)
        self.eta = model.eta_M; self.tau_M = model.tau_M; self.pop = pop; self.model = model
        self.ext_incr = (self.tm / model.ra) * (model.je if pop == 'E' else model.ji)
        self.scaleA = self.tm / model.ra; self.scaleG = self.tm / model.rg      # mV weight -> engine s jump

    def step(self, rng, lamA, qA, lamG, qG, nu_ext, dynamic_m=True):
        """Compound-Poisson input: per step, arrivals with total mean lam*dt and second moment q*dt (gamma-shaped weights)."""
        dt = self.dt
        self.sE *= self.arA; self.sI *= self.arG
        for lam, q, scale, target in ((lamA, qA, self.scaleA, self.sE), (lamG, qG, self.scaleG, self.sI)):
            if lam <= 0:
                continue
            wbar = q / lam                                     # mean weight of an arrival (mV) given (mean, second moment)
            rate = lam / wbar                                  # arrivals per ms
            k = rng.poisson(rate * dt, self.N)                 # number of arrivals per cell per step (independent)
            target += scale * wbar * k                         # each arrival carries the mean weight (second moment matched via wbar)
        ext = rng.poisson(nu_ext * dt, self.N); self.sE += ext * self.ext_incr
        self.IE = self.sE + (self.IE - self.sE) * self.adA; self.II = self.sI + (self.II - self.sI) * self.adG
        Inet = self.IE - self.z * self.II - self.eta * self.m
        self.ref -= 1; np.maximum(self.ref, 0, out=self.ref); free = self.ref == 0
        Vtmp = Inet + (self.V - Inet) * self.decay_V
        self.V = np.where(free, Vtmp, self.model.v_reset)
        spk = free & (self.V >= self.theta); self.V[spk] = self.model.v_reset; self.ref[spk] = self.tref_steps
        if dynamic_m:
            self.m -= dt / self.tau_M * self.m; self.m[spk] += 1.
        return int(spk.sum())


def closure_response(model, pop, stats, samples, arm, seed_unused=None):
    """Phi + rate relaxation driven by the same input moments (deterministic)."""
    steps = ms_to_step(BASE_MS + PERT_MS + TAIL_MS); tm = model.te if pop == 'E' else model.ti
    theta = samples['theta']; z = np.ones_like(theta) if samples['z'] is None else samples['z']; m = np.zeros_like(theta) if samples['m'] is None else samples['m'].copy()
    r = np.full(len(theta), stats['native_rate_per_ms']); tau = model.v['tau_rate_e_ms'] if pop == 'E' else model.v['tau_rate_i_ms']
    gA = cA = gG = cG = 0.; out = np.empty(steps // 10)
    for k in range(steps):
        fE = fI = 1.
        if BASE_MS <= k * model.dt < BASE_MS + PERT_MS:
            fE = 1. + arm[2] if arm[1] == 'E' else 1.; fI = 1. + arm[2] if arm[1] == 'I' else 1.
        lamA, qA, lamG, qG = stats['ampa_mean_mv_per_ms'] * fE, stats['ampa_sq_mv2_per_ms'] * fE, stats['gaba_mean_mv_per_ms'] * fI, stats['gaba_sq_mv2_per_ms'] * fI
        BA = model.dt * tm / model.ra; BG = model.dt * tm / model.rg
        gA = model.arA * gA + BA * (lamA + (model.je if pop == 'E' else model.ji) * model.nu_sig); cA = gA + (cA - gA) * model.adA
        gG = model.arG * gG + BG * lamG; cG = gG + (cG - gG) * model.adG
        ex = tm * (qA + (model.je if pop == 'E' else model.ji) ** 2 * model.nu_sig); inh = tm * (z ** 2) * qG
        mu = cA - z * cG - model.eta_M * m
        if pop == 'E':
            sigG = np.sqrt(np.maximum(inh * model.w2cv_e, 0)); shifted = mu - 2.065 / 2 * np.sqrt(max(ex * (model.ra + model.ta) / tm, 1e-16))
            means = shifted[None, :] + np.sqrt(2) * sigG[None, :] * model.gh_x[:, None]
            p = model._lif(means, np.sqrt(max(ex, 1e-12)), theta[None, :], tm, model.tref_e); phi = np.sum(model.gh_w[:, None] * p, axis=0)
        else:
            sigG = np.sqrt(np.maximum(inh * model.w2cv_i, 0)); shifted = mu - 2.065 / 2 * np.sqrt(max(ex * (model.ra + model.ta) / tm, 1e-16))
            means = shifted[None, :] + np.sqrt(2) * sigG[None, :] * model.gh_x[:, None]
            p = model._lif(means, np.sqrt(max(ex, 1e-12)), model.theta_i, tm, model.tref_i); phi = np.sum(model.gh_w[:, None] * p, axis=0)
        r = r + model.dt / tau * (phi - r)
        m = m - model.dt / model.tau_M * m + model.dt * r
        if (k + 1) % 10 == 0:
            out[(k + 1) // 10 - 1] = r.mean() * 1000.
    return out


def run_one(model, name, ctx, stats, samples, seed, arm):
    rng = np.random.default_rng(seed); pop = ctx['population']
    lif = IsolatedLIF(model, pop, samples['theta'], samples['z'], samples['m'], seed)
    steps = ms_to_step(BASE_MS + PERT_MS + TAIL_MS); counts = np.zeros(steps // 10)
    for k in range(steps):
        fE = fI = 1.
        if BASE_MS <= k * model.dt < BASE_MS + PERT_MS:
            fE = 1. + arm[2] if arm[1] == 'E' else 1.; fI = 1. + arm[2] if arm[1] == 'I' else 1.
        n = lif.step(rng, stats['ampa_mean_mv_per_ms'] * fE, stats['ampa_sq_mv2_per_ms'] * fE, stats['gaba_mean_mv_per_ms'] * fI, stats['gaba_sq_mv2_per_ms'] * fI, model.nu_sig)
        counts[k // 10] += n
    rate_hz = counts / N_CELLS / 1e-3
    return rate_hz


def summarize(rate_hz, sham_hz):
    """Baseline (0.5-1.0 s), peak response above sham in 1.0-1.1 s, peak latency, return (1.1-2.0 s) in 10-ms bins."""
    t = np.arange(len(rate_hz)) * 1e-3
    b = slice(500, 1000); p = slice(1000, 1100); q = slice(1100, 2000)
    d = rate_hz - sham_hz
    d10 = d[:2000].reshape(-1, 10).mean(1)                                  # 10-ms bins
    base = float(rate_hz[b].mean()); base_sham = float(sham_hz[b].mean())
    resp = d10[100:110]; peak_i = int(np.argmax(np.abs(resp)))
    return dict(baseline_hz=base, sham_baseline_hz=base_sham, peak_response_hz=float(resp[peak_i]), peak_latency_ms=float(peak_i * 10 + 5),
                mean_response_hz=float(resp.mean()), post_mean_hz=float(d10[110:200].mean()), post_first_100ms_hz=float(d10[110:120].mean()))


def main(max_runs=60):
    model = ReducedModel(dict(grid=20)); SMALL.mkdir(parents=True, exist_ok=True)
    ctxs = contexts(model); table = {}
    for name, ctx in ctxs.items():
        stats, samples = context_statistics(model, ctx); table[name] = dict(ctx, stats=stats)
    write(SMALL / 'contexts.json', dict(contexts=table, arms=ARMS, seeds=SEEDS_SMALL, cells=N_CELLS, windows_ms=dict(base=BASE_MS, pert=PERT_MS, tail=TAIL_MS),
          defined_before_approximation_results=True))
    results = {}; n_runs = 0
    for name, ctx in ctxs.items():
        stats, samples = context_statistics(model, ctx)
        for seed in SEEDS_SMALL:
            sham = None; per = {}
            for arm in ARMS:
                if n_runs >= max_runs:
                    break
                key = f'{name}_s{seed}_{arm[0]}'; path = SMALL / f'{key}.npz'
                if path.exists():
                    rate = np.load(path)['rate_hz']
                else:
                    rate = run_one(model, key, ctx, stats, samples, seed, arm); np.savez_compressed(path, rate_hz=rate)
                n_runs += 1
                closure = closure_response(model, ctx['population'], stats, samples, arm)
                np.savez_compressed(SMALL / f'{key}_closure.npz', rate_hz=closure)
                per[arm[0]] = dict(rate=rate, closure=closure)
            sham = per['sham']
            for arm in ARMS[1:]:
                if arm[0] not in per:
                    continue
                nat = summarize(per[arm[0]]['rate'], sham['rate']); clo = summarize(per[arm[0]]['closure'], sham['closure'])
                results[f'{name}_s{seed}_{arm[0]}'] = dict(context=name, seed=seed, arm=arm[0], native=nat, closure=clo)
            write(SMALL / 'results.json', dict(results=results, runs_completed=n_runs))
            print(name, seed, {a: round(results.get(f'{name}_s{seed}_{a}', {}).get('native', {}).get('peak_response_hz', float('nan')), 2) for a in ('E5', 'E2p5', 'I5', 'I2p5')}, flush=True)
    # linearity / usability flags per context & seed: half-dose normalised response within 20%
    flags = {}
    for name in ctxs:
        for seed in SEEDS_SMALL:
            for comp in ('E', 'I'):
                full = results.get(f'{name}_s{seed}_{comp}5'); half = results.get(f'{name}_s{seed}_{comp}2p5')
                if full and half:
                    a, b = full['native']['peak_response_hz'], half['native']['peak_response_hz']
                    ratio = (b * 2) / a if abs(a) > 1e-9 else float('nan')
                    flags[f'{name}_s{seed}_{comp}'] = dict(full=a, half_times_two=b * 2, normalised_ratio=ratio, linear_within_20pct=bool(abs(ratio - 1) <= .2) if np.isfinite(ratio) else False)
    write(SMALL / 'results.json', dict(results=results, linearity=flags, runs_completed=n_runs,
          scope='Isolated compound-Poisson LIF sets; independent-input assumption; not a test of network correlations. Closure coefficients unchanged (tau_rate 5/2.5 ms).'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--max-runs', type=int, default=60); a = ap.parse_args()
    main(a.max_runs)
