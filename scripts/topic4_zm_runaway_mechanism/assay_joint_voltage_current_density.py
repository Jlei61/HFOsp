"""Fixed local pilot for moment-preserving voltage/current transport."""
from common import OUT, np, read, write, log
from joint_voltage_current_density import simulate_joint, voltage_edges
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

DEST = OUT / 'joint_voltage_current_density'


def register():
    assert read(OUT / 'joint_voltage_current_density_implementation_check.json')['status'] == 'JOINT_MOMENT_IMPLEMENTATION_PASS'
    assert read(OUT / 'conditional_density_transport_accuracy/result.json')['status'] == 'ZERO_NOISE_TRANSPORT_DIAGNOSIS_COMPLETE'
    DEST.mkdir(exist_ok=True)
    path = DEST / 'contract.json'; assert not path.exists()
    parent = read(OUT / 'conditional_density_grid_diagnostic/contract.json')
    source = Path(__file__).with_name('joint_voltage_current_density.py')
    cases = [dict(kind='waveform', index=k) for k in [0, 3, 6, 9]]
    cases += [dict(kind='linear', original=c) for c in parent['cases']]
    for k, case in enumerate(cases): case['id'] = k
    write(path, dict(created_local=datetime.now().astimezone().isoformat(),
          question='Does preserving bin voltage moments repair artificial dispersion without losing strong-waveform accuracy or original critical-mode response?',
          change='Store joint first/second moments of voltage and four currents in each bin; exact Gaussian affine step and threshold/bin truncation. Preserve reset-current history. Replace positive point remapping only; no fit, parameter, input, geometry or Z/M change.',
          approximation='Joint Gaussian distribution within each voltage bin; not claimed exact for threshold-conditioned colored currents.',
          response_source=str(source), response_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
          grids=[128], cases=cases, workers=8,
          acceptance=dict(waveform_L2_max=.15, waveform_mean_relative_max=.1,
                          linear_original_per_case_gate=True, mass_error_max=1e-8,
                          current_moment_error_max=1e-7, dropped_mass_max=1e-10, lower_tail_sum_max=1e-8),
          selection='Four unchanged full strong waveforms and all twelve original additional SN7 E/I mode response conditions, including passing controls. All targets already observed. This is a repair pilot, not a new blind/full validation.',
          budget='Sixteen fixed local predictions on128bins. No network or bifurcation launch. Further grid or full-response validation must have a separately stated scope; no fitting failed targets.',
          limitations='Finite-step deterministic density map, not yet a compact continuous rate equation. Passing this pilot alone cannot promote the spatial model.'))


def one(case_id):
    c = read(DEST / 'contract.json'); case = c['cases'][case_id]
    assert hashlib.sha256(Path(c['response_source']).read_bytes()).hexdigest() == c['response_sha256']
    prefix = DEST / f'case{case_id:02d}'; assert not prefix.with_suffix('.json').exists()
    started = time.monotonic()
    if case['kind'] == 'waveform':
        source = np.load(OUT / 'factorial_waveform/prepared.npz')
        target = np.load(OUT / 'factorial_waveform/response.npz')
        index = case['index']; pars = source['pars'][index]; T = float(source['T_ms']); dt = .1
        edges = voltage_edges(pars[1], pars[21], 128)
        answer = simulate_joint(pars, source['wave'][index], edges, dt, T, round(5*T/dt), round(20*T/dt), 128)
        prediction = answer[0]; measured = target['measured_hz'][index]
        reference = read(OUT / 'factorial_waveform/result.json')['rows'][index]
        error = float(np.linalg.norm(prediction-measured)/max(np.linalg.norm(measured), np.sqrt(len(measured))))
        mean = float(np.average(prediction, weights=target['occupancy_ms']))
        bias = abs(mean-reference['MC_mean_hz'])/max(reference['MC_mean_hz'], 1.)
        result = dict(kind='waveform', source_index=index, waveform_L2=error, relative_mean_error=bias,
                      predicted_mean_hz=mean, reference_mean_hz=reference['MC_mean_hz'],
                      passed=bool(error <= .15 and bias <= .1))
        numerical = answer[6][None]
        np.savez_compressed(prefix.with_suffix('.npz'), predicted_hz=prediction, measured_hz=measured,
                            step_rate_hz=answer[3], final_free=answer[4], final_refractory=answer[5],
                            edges=edges, T_ms=T, dt_ms=dt, phase_centres=target['phase_centres'])
    else:
        original = case['original']; q = original['workpoint']; ch = original['channel']; f = original['frequency_hz']
        dt = original['dt_ms']; steps = round(original['duration_ms']/dt); burn = round(original['burn_ms']/dt)
        T = 1000/f if f else original['duration_ms']; size = 4096
        pars = condition(0., q['theta'], 1., 1., q['pop'], dt=dt)
        edges = voltage_edges(q['theta'], 11., 128)
        waveform = np.repeat(np.array([q['mu'], q['ve'], q['vi']])[:, None], size, axis=1)
        oscillator = np.sin(2*np.pi*np.arange(size)/size) if f else np.ones(size)
        rates = []; evidence = []; final = []
        for sign in [1., -1.]:
            wave = waveform.copy(); wave[ch] += sign*original['absolute_amplitude']*oscillator
            answer = simulate_joint(pars, wave, edges, dt, T, burn, steps, 128,
                                    q['mu'], q['ve'], q['vi'], original['burn_ms'])
            rates.append(answer[3]); evidence.append(answer[6]); final.append(answer[4])
        rates = np.array(rates); numerical = np.array(evidence)
        phase = 2*np.pi*f/1000*((np.arange(steps)+1)*dt+original['burn_ms'])
        difference = rates[0]-rates[1]
        gain = (np.mean(difference*(np.sin(phase)+1j*np.cos(phase)))/original['absolute_amplitude']) if f else complex(difference.mean()/(2*original['absolute_amplitude']))
        reference = original['reference']
        error = float(abs(gain-complex(*reference['measured']))/max(abs(reference['dc_measured']), 1e-12))
        result = dict(kind='linear', original_case_id=original['id'], workpoint=q,
                      channel=ch, frequency_hz=f, predicted=[gain.real, gain.imag],
                      normalized_error=error, tolerance=reference['tol'], counted=reference['counted'],
                      passed=bool(error <= reference['tol']) if reference['counted'] else None)
        np.savez_compressed(prefix.with_suffix('.npz'), rate_hz=rates, final_free=np.array(final),
                            edges=edges, dt_ms=dt, frequency_hz=f, burn_ms=original['burn_ms'],
                            amplitude=original['absolute_amplitude'])
    numeric_pass = bool(numerical[:, 0].max() < 1e-8 and numerical[:, 1].max() < 1e-7 and
                        numerical[:, 2].max() < 1e-10 and numerical[:, 3].max() < 1e-8)
    result.update(status='JOINT_DENSITY_CASE_COMPLETE', id=case_id, grid=128,
                  numerical_pass=numeric_pass, numerical=numerical.tolist(),
                  elapsed_seconds=time.monotonic()-started, model_promoted=False)
    write(prefix.with_suffix('.json'), result); log('JOINT CASE', case_id, result)


def dispatch():
    contract = read(DEST / 'contract.json')
    status = dict(status='RUNNING', pid=os.getpid(), expected=16, workers=8, finished=[])
    write(DEST / 'jobs.json', status)
    def worker(case):
        prefix = DEST / f'case{case["id"]:02d}'
        assert not prefix.with_suffix('.json').exists()
        with prefix.with_suffix('.log').open('w') as handle:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--case', str(case['id'])], stdout=handle, stderr=subprocess.STDOUT)
            code = process.wait()
        return dict(case_id=case['id'], pid=process.pid, exit_code=code)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(worker, case) for case in contract['cases']]):
            status['finished'].append(future.result()); write(DEST / 'jobs.json', status)
            log('JOINT PROGRESS', len(status['finished']), 16)
    status['status'] = 'COMPLETE' if all(r['exit_code'] == 0 for r in status['finished']) else 'EXECUTION_FAILURE'
    write(DEST / 'jobs.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true'); parser.add_argument('--dispatch', action='store_true')
    parser.add_argument('--case', type=int); args = parser.parse_args()
    if args.register: register()
    elif args.dispatch: dispatch()
    else: one(args.case)
