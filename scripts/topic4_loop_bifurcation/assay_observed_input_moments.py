#!/usr/bin/env python3
"""Bounded local diagnosis of source-mean and white-input variance errors.

Prescribed measured moments are diagnostic inputs, never an autonomous closure.
The unchanged colored-current LIF assay holds the native mean M/Z/K/G fixed.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, time
import numpy as np
from campaign import ROOT, read, write, sha
from observe_source_aggregation import OUT as SOURCE, INITIAL
from coupled_density_exit import ADAPTED
from conditional_density_inputs import OPS
from audit_target_root_response import density_condition
from phase_lif_mc import run
import run_topic4_loop_zk_conditional as native

OUT = SOURCE / 'local_response_factorial'
LABELS = ['group_mean_white_variance', 'native_mean_white_variance',
          'group_mean_measured_variance', 'native_mean_measured_variance']


def main(exact_thresholds=False):
    global OUT
    if exact_thresholds:
        OUT = SOURCE / 'local_response_factorial_exact_thresholds'
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'contract.json').exists(), 'Preserve completed attempts.'
    assert read(SOURCE / 'observer_audit.json')['status'] == 'PASS'
    data = dict(np.load(SOURCE / 'input_analysis/input_reconstruction.npz'))
    obs = dict(np.load(SOURCE / 'cell_statistics.npz'))
    geo = dict(np.load(ADAPTED / 'geometry.npz'))
    params = read(OPS / 'prepared.json')['params']
    state = native.read_pickle(INITIAL)['engine']
    group = geo['cell_group']; region = geo['group_region'][group]
    display = geo['group_cell'][group]; E = np.arange(40000) < 32000
    rates = data['native_rate_Hz']; selected = {}
    edge = [r['cell'] for r in read(SOURCE / 'contract.json')['edge_cells']]
    # Selection uses only the already observed native rates, before fresh assay.
    pools = [(f'edge{c}', E & (display == c)) for c in edge]
    pools += [('coreA_control', E & (region == 0)),
              ('coreB_control', E & (region == 1)), ('I_control', ~E)]
    for label, mask in pools:
        ids = np.flatnonzero(mask)
        order = ids[np.argsort(rates[ids], kind='stable')]
        for q in [.25, .5, .75]:
            cell = int(order[round((len(order) - 1) * q)])
            selected.setdefault(cell, []).append(f'{label}: native rate quantile {q}; N={len(ids)}')
    cells = np.array(list(selected), dtype=int)
    M = obs['per_cell_mean_moments'][:, 6].mean(0)
    Z = state['slow']['z'].copy()
    K = np.r_[state['termination_mechanism']['sahp_g'], np.zeros(8000)]
    G = float(30 * obs['global_R_and_s'][:, 1].mean())
    assert np.all(K[~E] == 0) and np.all(Z[~E] == 1) and np.all(M[~E] == 0)
    means = data['observed_IE_II_mean']
    bias = data['source_group_minus_cell_mean_IE_II']
    white = data['independent_white_variance_IE_II_cell_group'][:, :, 1]
    measured = data['observed_variance_IE_II']
    theta_all=geo['threshold_mv'][group]
    theta_source=dict(kind='Density group-averaged thresholds, exact for the selected edge targets; not exact for every core target.')
    if exact_thresholds:
        from native_target_thresholds import load
        theta_all,theta_source=load()
    pars = []; rows = []; covariance_errors = []
    for cell in cells:
        pop = 'E' if E[cell] else 'I'
        g = float(K[cell] + (Z[cell] * G if E[cell] else 0))
        h = 1 + g; theta = float(theta_all[cell])
        _, unit = density_condition([0., 1., 1.], g, theta, pop, params)
        for c, label in enumerate(LABELS):
            mean = means[:, cell] + (bias[:, cell] if c in [0, 2] else 0.)
            variance = white[:, cell] if c < 2 else measured[:, cell]
            effective_var = variance * np.array([1., Z[cell]**2]) / h**2
            mu = (mean[0] - Z[cell] * mean[1] - .0005 * M[cell]
                  - 30 * K[cell] - (17.662847938268442 * Z[cell] * G if E[cell] else 0)) / h
            p, check = density_condition(np.r_[mu, effective_var / unit], g, theta, pop, params)
            covariance_errors.append(float(np.max(abs(np.array(check) - effective_var))))
            assert np.allclose(check, effective_var, rtol=1e-11, atol=1e-11)
            assert p[4] == p[5] == 0 and p[19] == round(params['tau_ref_' + pop] / .1)
            rows.append(dict(cell=int(cell), source_group=int(group[cell]), display_cell=int(display[cell]),
                             population=pop, region=int(region[cell]), selected_by=selected[int(cell)],
                             condition=label, mean_current_mV=float(mu), conductance=g,
                             threshold_mV=theta, native_rate_Hz=float(rates[cell]),
                             raw_IE_II_mean=mean.tolist(), raw_IE_II_variance=variance.tolist(),
                             held_native_M=float(M[cell]), held_Z=float(Z[cell]), held_K=float(K[cell])))
            pars.append(p)
    pars = np.array(pars)
    replicas = 1024; duration_ms = 16000; seed = 929471
    write(OUT / 'contract.json', dict(status='FROZEN_BEFORE_LOCAL_FACTORIAL', created_epoch=time.time(),
        question='At selected targets on the missed recruitment edge, how much do source averaging and excessive white-input variance independently change the local frozen-M response?',
        design='Four conditions per target: native observed means or their exact group-source bias, crossed with group-white predicted variance or measured total variance. Native thresholds/refractory/Z/K, measured mean M and G, original discrete synaptic filters and membrane. Same Gaussian streams and randomized count phase for all conditions; no fitted scale.',
        selection='Native rate 25/50/75 percentiles within each of the previously selected ten display-edge cells, plus coreA/coreB/I controls; selected before fresh local counts. Duplicate targets retained once.',
        limits='Measured moments prescribe data and are not an autonomous closure. Measured-variance conditions still omit native autocorrelation, E/I cross-covariance, non-Gaussian currents and M dynamics. Two seconds of native data is not stationary ground truth; target selection is development-only. Numerical replicas estimate Monte Carlo precision, not independent native seed uncertainty. This is a local response diagnosis, not a network causal experiment or new bifurcation.',
        threshold_source=theta_source, exact_native_thresholds=exact_thresholds,
        targets=len(cells), conditions=len(pars), replicas=replicas, duration_ms=duration_ms,
        burn_ms=1000, extra_burn_phase_ms=1000, seed=seed, device=1,
        source_sha256=sha(SOURCE / 'input_analysis/input_reconstruction.npz'),
        observer_sha256=sha(SOURCE / 'cell_statistics.npz'), producer_sha256=sha(__file__),
        formal_bifurcation_allowed=False, counts_as_autonomous_loop=False))
    np.savez_compressed(OUT / 'inputs.npz', cells=cells, pars=pars, held_global_G=G)
    write(OUT / 'implementation_qa.json', dict(status='PASS',
        prescribed_current_covariance_max_error=max(covariance_errors),
        physical_kernel='Existing phase_lif_mc with exact density_condition coefficients; no kernel edit.',
        current_variance_to_driver_conversion='Each channel divided by its exact discrete unit-driver stationary current variance, after conductance and Z factors.'))
    started = time.time(); chunks = []
    for lo in range(0, len(pars), 8):
        write(OUT / 'progress.json', dict(status='RUNNING_LOCAL_RESPONSE', pid=os.getpid(),
             completed=lo, total=len(pars), elapsed_s=time.time()-started))
        out = run(pars[lo:lo+8], replicas, duration_ms, 1000, seed, device=1,
                  phase_ms=1000, stream_mode=1)
        assert not out[:, :, 3].any()
        count = out[:, :, 2].astype('i4'); chunks.append(count)
        np.savez_compressed(OUT / f'counts_{lo:04d}.npz', counts=count)
    counts = np.concatenate(chunks); samples = counts / 16.
    for i, row in enumerate(rows):
        row.update(measured_rate_Hz=float(samples[i].mean()),
                   MC_SEM_Hz=float(samples[i].std(ddof=1)/np.sqrt(replicas)))
    effects = []
    for j, cell in enumerate(cells):
        a = samples[4*j:4*j+4]
        contrasts = {'source_mean_effect_white': a[0]-a[1],
                     'source_mean_effect_measured_variance': a[2]-a[3],
                     'variance_effect_group_mean': a[0]-a[2],
                     'variance_effect_native_mean': a[1]-a[3],
                     'interaction': a[0]-a[1]-a[2]+a[3]}
        effects.append(dict(cell=int(cell), native_rate_Hz=float(rates[cell]),
            rates_Hz=a.mean(1).tolist(), selected_by=selected[int(cell)],
            contrasts={name:dict(mean_Hz=float(x.mean()), paired_MC_SEM_Hz=float(x.std(ddof=1)/np.sqrt(replicas)))
                       for name,x in contrasts.items()}))
    np.savez_compressed(OUT / 'counts.npz', counts=counts, cells=cells)
    result = dict(status='COMPLETE_LOCAL_INPUT_FACTORIAL', rows=rows, effects=effects,
        elapsed_s=time.time()-started, native_correspondence_validated=False,
        autonomous_closure_repaired=False, formal_bifurcation_allowed=False)
    write(OUT / 'result.json', result)
    write(OUT / 'progress.json', dict(status=result['status'], elapsed_s=result['elapsed_s']))
    print('COMPLETE', len(cells), 'targets;', result['elapsed_s'], 'seconds', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--exact-thresholds',action='store_true')
    main(p.parse_args().exact_thresholds)
