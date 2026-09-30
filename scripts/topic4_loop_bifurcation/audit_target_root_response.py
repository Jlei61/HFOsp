#!/usr/bin/env python3
"""Fresh, bounded local DC tests at the actual-field K9 equilibrium.

No fitting. Independent count paths test the frozen response and its derivatives,
with both its training-current discretisation and the coupled density filters.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import time
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from campaign import ROOT, read, write, sha
from target_stationary_root import TargetStationary, OUT as POINT
import lif_mc

OUT = ROOT / 'target_root_response_audit'
CHANNELS = ['mean_mV', 'variance_E', 'variance_I', 'conductance']


def density_condition(physical, g, theta, pop, params):
    """Same effective moments, exact one-increment native-density filters."""
    p = lif_mc.condition(*[physical[0], theta, physical[1], physical[2], pop])
    tm = params['tau_m_' + pop]
    p[18] = np.exp(-.1 * (1 + g) / tm)
    checks = []
    for kind, var, indices in [('AMPA', physical[1], (6, 7, 8, 11, 12, 13)),
                                ('GABA', physical[2], (17, 9, 10, 14, 15, 16))]:
        tr, td = params['tau_r_' + kind], params['tau_d_' + kind]
        ar, ad = np.exp(-.1 / tr), np.exp(-.1 / td)
        area = .1 / (tr * (1 - ar))
        jump = np.sqrt(tm * var * .1) / (tr * area)
        a, b, d, c11, c21, c22 = indices
        p[a], p[b], p[d] = ar, (1 - ad) * ar, ad
        p[c11], p[c21], p[c22] = jump, (1 - ad) * jump, 0.
        A = np.array([[ar, 0], [(1 - ad) * ar, ad]])
        B = jump * np.array([1., 1 - ad])
        covariance = solve_discrete_lyapunov(A, np.outer(B, B))
        expected = jump**2 / (1 - ar**2) * (1 - ad) / (1 + ad) * (1 + ar * ad) / (1 - ar * ad)
        assert np.isclose(covariance[1, 1], expected, rtol=1e-10, atol=1e-12)
        checks.append(float(covariance[1, 1]))
    return p, checks


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'contract.json').exists(), 'Preserve every completed attempt.'
    assert read(POINT / 'result.json')['status'] == 'NUMERICAL_ROOT'
    started = time.time()
    e = TargetStationary()
    with np.load(POINT / 'root_candidate.npz') as z:
        r = z['source_rate_per_ms'].copy()
    residual, meta = e.evaluate(r)
    assert abs(residual).max() * 1000 < 1e-6
    physical, g = meta['physical'], meta['g']
    predicted, grad = e.model.phi(physical, g)
    rates = predicted * 1000
    cell_region = e.geo['group_region'][e.group]
    selected = {}
    # Fixed rate strata cover the partially recruited fringe as well as cores.
    for label, mask in [('coreA', e.E & (cell_region == 0)),
                        ('coreB', e.E & (cell_region == 1)),
                        ('surround', e.E & (cell_region == 2)), ('I', ~e.E)]:
        edges = [0., .1, 10., 100., 300., 450., 500.01] if label != 'I' else [0., 1., 50., 200., 400., 700., 1000.01]
        for lo, hi in zip(edges[:-1], edges[1:]):
            ids = np.flatnonzero(mask & (rates >= lo) & (rates < hi))
            if not len(ids):
                continue
            order = ids[np.argsort(rates[ids], kind='stable')]
            cell = int(order[len(order) // 2])
            selected.setdefault(cell, []).append(f'{label}: median of [{lo:g},{hi:g}) Hz; N={len(ids)}')
        cell = int(np.flatnonzero(mask)[np.argmax(abs(grad[mask, 0]))])
        selected.setdefault(cell, []).append(label + ': maximum absolute mean-current derivative')
    cells = np.array(list(selected), dtype=int)
    specifications, pars, qa = [], [], []
    for kernel in ['exact_colored', 'density_discrete']:
        for cell in cells:
            pop = 'E' if e.E[cell] else 'I'
            theta = e.model.s.theta[cell]
            base = np.r_[physical[cell], g[cell]]
            channels = range(4) if pop == 'E' else range(3)
            steps = [.02 * (theta - 11), .01 * base[1], .01 * base[2], .01 * (1 + base[3])]
            for channel in channels:
                if steps[channel] <= 0:
                    continue
                for factor in [1., .5]:
                    h = steps[channel] * factor
                    pair = []
                    for sign in [1, -1]:
                        q = base.copy()
                        q[channel] += sign * h
                        assert q[1:].min() >= 0
                        if kernel == 'exact_colored':
                            p = lif_mc.condition(q[0], theta, q[1], q[2], pop)
                            p[18] = np.exp(-.1 * (1 + q[3]) / e.p['tau_m_' + pop])
                        else:
                            p, check = density_condition(q[:3], q[3], theta, pop, e.p)
                            qa.append(check)
                        pair.append(len(pars)); pars.append(p)
                    specifications.append(dict(kernel=kernel, cell=int(cell), group=int(e.group[cell]),
                        population=pop, region=int(cell_region[cell]), selected_by=selected[int(cell)],
                        physical=base.tolist(), threshold_mV=float(theta), channel=CHANNELS[channel],
                        amplitude=float(h), amplitude_factor=factor, count_indices=pair,
                        prediction_Hz=float(rates[cell]), predicted_gain_Hz=float(grad[cell, channel] * 1000)))
    pars = np.array(pars)
    write(OUT / 'contract.json', dict(status='FROZEN_BEFORE_FRESH_TARGETS', created_epoch=time.time(),
        question='Are the static values and local DC derivatives accurate at actual-exit K9 target inputs, especially partially recruited cells that control spatial recruitment?',
        selection='One median physical target per nonempty prespecified rate stratum in coreA/coreB/surround/I, plus maximum absolute mean-current derivative in each. No target or network fit; selection precedes fresh Monte Carlo.',
        kernels=['Frozen surrogate training exact-colored-current discretisation', 'Same moments with the coupled density one-increment synaptic ordering; independent covariance algebra checked'],
        design=dict(physical_targets=len(cells), paired_tests=len(specifications), conditions=len(pars),
                    replicas=8192, record_ms=4000, burn_ms=1000, seed=928811, device=0,
                    amplitudes='mu .02*(threshold-11); variances1%; g1%*(1+g), each repeated at half amplitude'),
        gate='Report every channel. Estimable when paired gain abs/SEM>=10; error <=max(10%abs(measuredgain),2SEM,1e-7), unchanged from earlier DC rule. Nonestimable is not PASS. Half-amplitude difference uses paired covariance, max(10%fullgain,2SEM,1e-7). Static mean error <=max(2Hz,10%measured,3SEM).',
        limits='Independent local Gaussian-current counts, not native network seeds, adaptation feedback or dynamic susceptibility. M is fixed at its root value for the local response. Passing values alone cannot certify stability or native bifurcation.',
        model_sha256=sha(ROOT / 'conductance_static_v3/locked_model.pt'), root_sha256=sha(POINT / 'root_candidate.npz'),
        producer_sha256=sha(__file__), MC_source=str(lif_mc.__file__), MC_sha256=sha(lif_mc.__file__),
        formal_bifurcation_allowed=False, rows=specifications))
    np.savez_compressed(OUT / 'inputs.npz', pars=pars, cells=cells, physical=physical[cells], g=g[cells], gradient=grad[cells])
    write(OUT / 'implementation_qa.json', dict(status='PASS', native_density_covariance_checks=len(qa),
         kernel_change='None. The existing paired-count kernel receives exact coefficients for either stationary-current discretisation. Same stream for replica k across all conditions.'))
    counts = []
    for lo in range(0, len(pars), 8):
        write(OUT / 'progress.json', dict(status='MONTE_CARLO', pid=os.getpid(), completed=lo, total=len(pars), elapsed_s=time.time()-started))
        out = lif_mc.run(pars[lo:lo+8], 8192, 4000, 1000, 928811, device=0, batch=8)
        assert not out[:, :, 3].any()
        c = out[:, :, 2].astype('i4');counts.append(c)
        np.savez_compressed(OUT / f'counts_{lo:04d}.npz', counts=c)
    counts = np.concatenate(counts)
    gains = []
    for row in specifications:
        plus, minus = row['count_indices']
        value = (counts[plus].astype(float)-counts[minus]) / (8 * row['amplitude'])
        mean = float(value.mean());sem = float(value.std(ddof=1)/np.sqrt(len(value)))
        rate = (counts[plus]+counts[minus]) / 8.
        rate_sem = float(rate.std(ddof=1)/np.sqrt(len(rate)))
        tolerance = max(.1*abs(mean), 2*sem, 1e-7)
        estimable = abs(mean)/max(sem, 1e-15) >= 10.
        row.update(measured_gain_Hz=mean, SEM=sem, estimable=bool(estimable), tolerance=tolerance,
            derivative_pass=bool(abs(mean-row['predicted_gain_Hz']) <= tolerance) if estimable else None,
            mean_rate_Hz=float(rate.mean()), mean_rate_SEM=rate_sem,
            mean_rate_pass=bool(abs(rate.mean()-row['prediction_Hz']) <= max(2., .1*rate.mean(), 3*rate_sem)))
        gains.append(value)
    sensitivities = []
    for i, row in enumerate(specifications):
        if row['amplitude_factor'] == 1.:
            continue
        j = i-1
        assert all(row[k] == specifications[j][k] for k in ['kernel', 'cell', 'channel'])
        delta = gains[i]-gains[j]
        sem = float(delta.std(ddof=1)/np.sqrt(len(delta)))
        tolerance = max(.1*abs(specifications[j]['measured_gain_Hz']), 2*sem, 1e-7)
        sensitivities.append(dict(kernel=row['kernel'], cell=row['cell'], channel=row['channel'],
            difference=float(delta.mean()), paired_SEM=sem, tolerance=tolerance,
            passed=bool(abs(delta.mean()) <= tolerance)))
    summary = []
    for kernel in ['exact_colored', 'density_discrete']:
        for population in ['E', 'I']:
            a = [q for q in specifications if q['kernel']==kernel and q['population']==population and q['amplitude_factor']==1.]
            summary.append(dict(kernel=kernel, population=population, primary=len(a),
                estimable=sum(q['estimable'] for q in a), derivative_pass=sum(q['derivative_pass'] is True for q in a),
                mean_pass=sum(q['mean_rate_pass'] for q in a)))
    result = dict(status='COMPLETE_LOCAL_DC_AUDIT', rows=specifications, summary=summary,
         amplitude_sensitivity=sensitivities, elapsed_s=time.time()-started,
         dynamic_response_validated=False, formal_bifurcation_allowed=False)
    write(OUT / 'result.json', result)
    write(OUT / 'progress.json', dict(status=result['status'], summary=summary, elapsed_s=result['elapsed_s']))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
