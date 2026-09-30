"""Bounded voltage-grid refinement at the two original SN7 mode workpoints.

All three input channels and both original frequencies are retained. This is
an after-failure numerical diagnosis, not a replacement validation cohort.
"""
from common import OUT, read, write, np, log
from conditional_current_density import simulate, voltage_grid
from lif_mc import condition
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import os
import subprocess
import sys
import time

DEST = OUT / 'conditional_density_grid_diagnostic'


def register():
    path = DEST / 'contract.json'
    assert not path.exists()
    parent = read(OUT / 'conditional_density_linear_response_contract.json')
    cases = [c for c in parent['cases'] if
             c.get('source') == 'local_response_additional_mode_groups/result.json']
    assert len(cases) == 12 and {c['id'] for c in cases} == set(range(54, 66))
    DEST.mkdir(exist_ok=True)
    write(path, dict(
        created_local=datetime.now().astimezone().isoformat(),
        question='Does voltage-grid refinement resolve the local gain error at the original E/I critical-mode workpoints?',
        selection='All twelve conditions of the original two additional SN7 mode workpoints, including passing mean-channel controls and both variance channels. Selected after observing the 256-node failure; diagnostic, not a blind validation.',
        parent_contract=str(OUT / 'conditional_density_linear_response_contract.json'),
        response_source=parent['response_source'], response_sha256=parent['response_sha256'],
        cases=cases, grids=[512, 1024], workers=8, wave_samples=parent['wave_samples'],
        protocol='Reuse original 256-node results and all original clocks, time steps, amplitudes, durations and initial state. Change voltage-node count only. No interpolation coefficient, moment closure, threshold or physical parameter change.',
        interpretation=dict(
            convergence_max_normalized_change=.02,
            original_error_gate_unchanged=True,
            final_grid_definition='abs(g1024-g512)/abs(reference_DC); report every condition and both successive changes. Two levels do not prove the asymptotic limit.',
            stop='At most these 24 paired predictions. Converged errors above the original gate do not support more voltage refinement as a repair. Any unresolved numerical change remains unresolved, not closure acceptance.'),
        model_promoted=False,
        scope='Local numerical diagnosis only; no autonomous network, coefficient fitting or bifurcation launch.'))


def one(case_id, nodes):
    contract = read(DEST / 'contract.json')
    assert nodes in contract['grids']
    assert hashlib.sha256(Path(contract['response_source']).read_bytes()).hexdigest() == contract['response_sha256']
    case = next(c for c in contract['cases'] if c['id'] == case_id)
    q = case['workpoint']; ch = case['channel']; f = case['frequency_hz']
    dt = case['dt_ms']; burn = round(case['burn_ms'] / dt); steps = round(case['duration_ms'] / dt)
    period = 1000 / f if f else case['duration_ms']; n = contract['wave_samples']
    waveform = np.repeat(np.array([q['mu'], q['ve'], q['vi']])[:, None], n, axis=1)
    oscillation = np.sin(2 * np.pi * np.arange(n) / n) if f else np.ones(n)
    pars = condition(0, q['theta'], 1., 1., q['pop'], dt=dt)
    grid = voltage_grid(q['theta'], 11., nodes)
    prefix = DEST / f'case{case_id:03d}_grid{nodes}'
    assert not prefix.with_suffix('.json').exists()
    rates = []; evidence = []; started = time.monotonic()
    for sign in [1., -1.]:
        drive = waveform.copy(); drive[ch] += sign * case['absolute_amplitude'] * oscillation
        answer = simulate(pars, drive, grid, dt, period, burn, steps, 128,
                          q['mu'], q['ve'], q['vi'], case['burn_ms'])
        rates.append(answer[3]); evidence.append(answer[6])
    rates = np.array(rates); evidence = np.array(evidence)
    phase = 2 * np.pi * f / 1000 * ((np.arange(steps) + 1) * dt + case['burn_ms'])
    difference = rates[0] - rates[1]
    gain = (np.mean(difference * (np.sin(phase) + 1j * np.cos(phase))) /
            case['absolute_amplitude']) if f else complex(difference.mean() / (2 * case['absolute_amplitude']))
    ref = case['reference']; error = abs(gain - complex(*ref['measured'])) / max(abs(ref['dc_measured']), 1e-12)
    numeric = bool(evidence[:, 0].max() < 1e-8 and evidence[:, 1].max() < 1e-7 and
                   evidence[:, 2].max() < 1e-10 and evidence[:, 3].max() < 1e-8)
    result = dict(status='GRID_DIAGNOSTIC_CASE_COMPLETE', case_id=case_id, grid_nodes=nodes,
                  workpoint=q, channel=ch, frequency_hz=f, dt_ms=dt,
                  predicted=[gain.real, gain.imag], measured=ref['measured'],
                  dc_measured=ref['dc_measured'], normalized_error=float(error),
                  tolerance=ref['tol'], counted=ref['counted'],
                  passed=bool(error <= ref['tol']) if ref['counted'] else None,
                  numerical_pass=numeric, numerical=evidence.tolist(),
                  mean_rates_hz=rates.mean(axis=1).tolist(),
                  elapsed_seconds=time.monotonic()-started, model_promoted=False)
    np.savez_compressed(prefix.with_suffix('.npz'), rate_hz=rates, grid_mv=grid,
                        dt_ms=dt, frequency_hz=f, burn_ms=case['burn_ms'],
                        amplitude=case['absolute_amplitude'])
    write(prefix.with_suffix('.json'), result)
    log('GRID CASE', case_id, nodes, float(error), numeric)


def dispatch():
    contract = read(DEST / 'contract.json')
    status = dict(status='RUNNING', pid=os.getpid(), workers=contract['workers'],
                  conditions=24, finished=[])
    write(DEST / 'jobs.json', status)

    def worker(case, nodes):
        prefix = DEST / f'case{case["id"]:03d}_grid{nodes}'
        assert not prefix.with_suffix('.json').exists()
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--case', str(case['id']), '--grid', str(nodes)]
        with prefix.with_suffix('.log').open('w') as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            code = process.wait()
        return dict(case_id=case['id'], grid_nodes=nodes, pid=process.pid, exit_code=code)

    with ThreadPoolExecutor(max_workers=contract['workers']) as pool:
        futures = [pool.submit(worker, case, nodes) for nodes in contract['grids'] for case in contract['cases']]
        for future in as_completed(futures):
            status['finished'].append(future.result()); write(DEST / 'jobs.json', status)
            log('GRID PROGRESS', len(status['finished']), 24)
    status['status'] = 'COMPLETE' if all(r['exit_code'] == 0 for r in status['finished']) else 'EXECUTION_FAILURE'
    write(DEST / 'jobs.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--dispatch', action='store_true')
    parser.add_argument('--case', type=int)
    parser.add_argument('--grid', type=int)
    args = parser.parse_args()
    if args.register:
        register()
    elif args.dispatch:
        dispatch()
    else:
        one(args.case, args.grid)
