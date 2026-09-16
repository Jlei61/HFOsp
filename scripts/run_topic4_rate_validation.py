#!/usr/bin/env python3
"""Run old and synapse-matched rate closures against recorded exact SNN input."""
import argparse
import time
from validate_topic4_fixed_rate_base import OUT, read, write
import numpy as np
from scipy import sparse
from src.topic4_patient_zm_meanfield import load_patient_coarse_model, transfer_rates, lif_rate_gauss_legendre


def run(name, grid, closure, seed=9108001):
    started = time.time(); folder = OUT / f'coarse_{grid}'
    model = load_patient_coarse_model(folder / 'model.npz')
    cfg = read(folder / 'prepared.json'); dt = cfg['dt_ms']; n = model.n_cells
    with np.load(OUT / 'snn' / f'{name}.npz') as a:
        ext_e = a[f'external_rate_{grid}'].astype(float)
        ext_i = a['external_i_rate'].astype(float)
    steps = len(ext_e); delay = cfg['max_delay_steps']
    ops = {key: sparse.load_npz(folder / f'delay_{key}.npz') for key in ('ee', 'ei', 'ie', 'ii')}
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    ta, tg = model.tau_ampa_ms, model.tau_gaba_ms
    ra, rg = cfg['tau_r_ampa_ms'], cfg['tau_r_gaba_ms']
    re = np.zeros(n); ri = np.zeros(n)
    he = np.zeros((delay, n)); hi = np.zeros((delay, n))
    # Four recurrent channels and two external channels, initially empty as in SNN.
    gating = np.zeros((6, n)); current = np.zeros((6, n))
    rise = np.array([ra, rg, ra, rg, ra, ra])[:, None]
    decay = np.array([ta, tg, ta, tg, ta, ta])[:, None]
    membrane = np.array([te, te, ti, ti, te, ti])[:, None]
    traces = np.empty((steps, 2), np.float32)
    fields = np.empty((steps//20, 2, n), np.float32)
    currents = np.empty((steps//20, 2, n), np.float32)
    clipped = 0
    rng = np.random.default_rng(seed)
    empirical = closure == 'cascade_empirical'
    colored = closure in ('cascade_colored','cascade_colored_slow')
    tau_rate_e, tau_rate_i = te, ti
    if closure in ('cascade_colored','cascade_fast'):
        tau_rate_e = read(OUT/'colored_response_diagnostic.json')['best_tau_rate_ms']
        tau_rate_i = read(OUT/'colored_response_diagnostic_I.json')['best_tau_rate_ms']
    if empirical:
        geo=np.load(folder/'geometry.npz');flat_cells=[];flat_thresholds=[];flat_weights=[]
        for cell in range(n):
            thresholds, counts=np.unique(geo['vtheta'][geo['cell_e']==cell],return_counts=True)
            flat_cells.extend([cell]*len(thresholds));flat_thresholds.extend(thresholds)
            flat_weights.extend(counts/model.count_e[cell])
        flat_cells=np.asarray(flat_cells);flat_thresholds=np.asarray(flat_thresholds);flat_weights=np.asarray(flat_weights)
    for step in range(steps):
        drive = np.stack([ops['ee'] @ he.ravel(), ops['ei'] @ hi.ravel(),
                          ops['ie'] @ he.ravel(), ops['ii'] @ hi.ravel(),
                          model.j_ext_e_mv * ext_e[step],
                          np.full(n, model.j_ext_i_mv * ext_i[step])])
        if closure == 'legacy':
            mu_e = current[0] - current[1] + te*drive[4]
            mu_i = current[2] - current[3] + ti*drive[5]
            next_current = current + dt * (membrane*drive-current)/decay
        elif closure in ('cascade', 'cascade_empirical', 'mesoscopic', 'cascade_colored','cascade_colored_slow','cascade_fast'):
            if closure == 'mesoscopic':
                drive[4] = model.j_ext_e_mv*rng.poisson(model.count_e*ext_e[step]*dt)/model.count_e/dt
                drive[5] = model.j_ext_i_mv*rng.poisson(model.count_i*ext_i[step]*dt)/model.count_i/dt
            # Literal expectation of the engine's two sequential exponential updates.
            gating = gating*np.exp(-dt/rise) + dt*membrane/rise*drive
            current = gating + (current-gating)*np.exp(-dt/decay)
            mu_e = current[0]-current[1]+current[4]
            mu_i = current[2]-current[3]+current[5]
            next_current = current
        else:
            raise ValueError(closure)
        ve = te*(model.v_ee@re + model.v_ei@ri + model.j_ext_e_mv**2*ext_e[step])
        vi = ti*(model.v_ie@re + model.v_ii@ri + model.j_ext_i_mv**2*ext_i[step])
        if colored:
            # Development approximation only: variance-weighted correlation times.
            # This is not a validated mixed-receptor population closure.
            exc_e = te*(model.v_ee@re + model.j_ext_e_mv**2*ext_e[step])
            exc_i = ti*(model.v_ie@re + model.j_ext_i_mv**2*ext_i[step])
            corr_e = (exc_e*(ra+ta)+(ve-exc_e)*(rg+tg))/np.maximum(ve,1e-12)
            corr_i = (exc_i*(ra+ta)+(vi-exc_i)*(rg+tg))/np.maximum(vi,1e-12)
            mu_e = mu_e - 2.065/2*np.sqrt(np.maximum(ve*corr_e/te,0))
            mu_i = mu_i - 2.065/2*np.sqrt(np.maximum(vi*corr_i/ti,0))
        pe, pi = transfer_rates(model, mu_e, np.sqrt(np.maximum(ve, 1e-12)),
                               mu_i, np.sqrt(np.maximum(vi, 1e-12)))
        if empirical:
            rates=lif_rate_gauss_legendre(mu_e[flat_cells],np.sqrt(np.maximum(ve[flat_cells],1e-12)),
                tau_mem_ms=te,tau_ref_ms=model.tau_ref_e_ms,v_threshold_mv=flat_thresholds,v_reset_mv=model.v_reset_mv)
            pe=np.bincount(flat_cells,weights=rates*flat_weights,minlength=n)
        ne = re+dt/tau_rate_e*(pe-re); ni = ri+dt/tau_rate_i*(pi-ri)
        clipped += int(np.any(ne<0) or np.any(ni<0) or np.any(ne>1/model.tau_ref_e_ms) or np.any(ni>1/model.tau_ref_i_ms))
        he[1:] = he[:-1].copy(); hi[1:] = hi[:-1].copy()
        if closure == 'mesoscopic':
            he[0] = rng.poisson(model.count_e*re*dt)/model.count_e/dt
            hi[0] = rng.poisson(model.count_i*ri*dt)/model.count_i/dt
        else:
            he[0] = re; hi[0] = ri
        re, ri = ne, ni; current = next_current
        traces[step] = 1000*np.array([np.average(re, weights=model.count_e), np.average(ri, weights=model.count_i)])
        if step % 20 == 0:
            fields[step//20] = np.array([re, ri])*1000
            currents[step//20] = [current[0]+current[4], current[1]]
    dest = OUT / 'rate'; dest.mkdir(exist_ok=True)
    stem = f'{name}_grid{grid}_{closure}' + (f'_seed{seed}' if closure=='mesoscopic' else '')
    np.savez_compressed(dest / f'{stem}.npz', rates_hz=traces, field_rates_hz=fields,
                        currents=currents, dt_ms=dt, field_frame_ms=2.)
    write(dest / f'{stem}.json', {'status':'COMPLETE', 'snn_reference':name,'grid':grid,
        'closure':closure, 'mean_e_hz_after_500ms':float(traces[int(500/dt):,0].mean()),
        'duration_ms':steps*dt, 'clipped_steps':clipped, 'seconds':time.time()-started,
        'initialization':'zero rates and zero synaptic/history states; SNN cold-start analogue',
        'external_input':'actual SNN per-step afferent rates after global OU, spatial OU, clipping and pulse; includes I global drive',
        'remaining_approximation':'instantaneous Poisson variance, one rate per spatial population, threshold-averaged transfer and phenomenological rate relaxation; colored correction when specified below; not yet validated',
        'mesoscopic_noise': {'enabled':closure=='mesoscopic','seed':seed,
            'rule':'Poisson population counts at actual cell population size, propagated through same coarse delayed weights; no fitted noise amplitude',
            'boundary':'finite-size diagnostic; recurrent covariance and neuron refractory correlations are approximated, not a derived exact mesoscopic closure'},
        'empirical_threshold_integration':empirical,
        'colored_development_correction':colored,
        'rate_relaxation_ms':{'E':tau_rate_e,'I':tau_rate_i},
        'colored_limit':'variance-weighted mixed-receptor correlation is heuristic; asymptotic small synaptic/membrane ratio fails for GABA; must pass direct network validation before use',
        'Z_M':'off'})
    print(stem, float(traces[int(500/dt):,0].mean()), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--grid',type=int,default=10)
    p.add_argument('--closure',choices=['legacy','cascade','cascade_empirical','mesoscopic','cascade_colored','cascade_colored_slow','cascade_fast'],default='legacy')
    p.add_argument('--seed',type=int,default=9108001);a=p.parse_args()
    run(a.name,a.grid,a.closure,a.seed)
